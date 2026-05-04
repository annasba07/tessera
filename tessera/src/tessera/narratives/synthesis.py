"""Cross-session synthesis over per-session narratives.

Takes a directory of v1 narrative JSON files and produces a single set of
cross-cutting observations the user can act on this week. The LLM gets each
session compacted to its high-signal fields plus deterministic aggregates,
keeping the whole input under ~150K tokens for one Sonnet call.

Validation rule: every cited ref MUST appear in the input set, else it's
dropped — see ``validator.py`` for the same fail-closed pattern at the
per-session layer.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_OBSERVATIONS = 10


SYNTHESIS_PROMPT = """You are reading {n_sessions} narrative summaries of one person's AI coding agent sessions. Your job: surface the cross-cutting patterns they would benefit from acting on this week.

{prior_context_block}

Each session below is already structured (tasks identified, friction moments labeled, recurring environmental issues called out, counterfactuals written). The per-session analysis is done. Your job is the *cross-session* layer — what shows up across sessions that they probably can't see from any single one.

## CRITICAL: Citation by ref token only

Every session below has a `ref` field — a short token like `S001`, `S047`, `S221`. When you cite a session as evidence, **cite by ref only**. Refs are exactly 4 characters: `S` followed by a 3-digit zero-padded number. Copy them character-by-character from the input.

DO NOT invent refs. DO NOT cite by `session_id`. If a ref doesn't appear in the input below, it will be dropped from your output. Refs in the input go from `S001` through `S{n_sessions_padded}` — anything outside that range is a fabrication.

Before you finalize, mentally scan each cited ref and confirm it appears verbatim in the input. The dropped-citation rate from prior runs was 50%+ — we are tightening this rule.

## What makes a great cross-session observation

- **Specific numbers**: "67 env issues across 57 atella sessions, 41 of them are stale Chrome locks" — not "you have lots of friction."
- **Specific evidence**: cite the actual ref tokens that support the claim. Minimum 3 supporting refs for a "high" confidence claim.
- **Specific fix**: a command to run, a config to add, a prompt template, a one-line habit change. Not "be more careful."
- **Surprising or aggregate-only**: a pattern that's invisible looking at any single session. If it's obvious from one session, the per-session narrative already covered it.

## Categories of observation to look for

- **Environmental**: stale lock files, expired auth, missing deps, MCP issues — recurring infra. Often a one-time fix per project.
- **Prompting**: prompts the user keeps writing that lead to wasted exchanges. Usually surfaced via `lesson_for_user` and `prompt_quality_signal`.
- **Project-specific**: a particular codebase or worktree where the agent reliably struggles in a specific way.
- **Tooling/workflow**: tool choices, sequencing patterns, MCP server combos that backfire.
- **Cross-agent**: differences between Claude / Codex / Gemini behavior on similar tasks.

## Output format

Return ONE JSON object. No markdown fence. No preamble:

```
{
  "headline": "<≤200 chars: the single biggest takeaway from the data>",
  "if_you_do_one_thing_this_week": "<≤300 chars: ONE concrete action grounded in specific evidence>",

  "observations": [
    {
      "title": "<short phrase>",
      "claim": "<1-2 sentences with specific numbers>",
      "evidence_refs": ["S012", "S047", ...],
      "supporting_count": <int = len(evidence_refs)>,
      "interpretation": "<2-4 sentences on why this is happening>",
      "next_action": "<concrete command, prompt, habit, or config — not 'be more careful'>",
      "confidence": "high | medium | low",
      "category": "environmental | prompting | project_specific | tooling | workflow | cross_agent"
    }
  ],

  "quick_wins": [
    {
      "fix": "<one-line action with command if applicable>",
      "affected_sessions": <int>,
      "evidence_refs": ["S001", ...]
    }
  ],

  "per_project": [
    {
      "project": "<project_label>",
      "session_count": <int>,
      "headline": "<≤200 chars: the dominant pattern in this project>",
      "biggest_friction": "<which friction type and why>"
    }
  ],

  "meta": {
    "notes": "<caveats, blind spots, anything you couldn't tell from the data>"
  }
}
```

## Hard rules

- Cite ONLY refs that appear in the INPUT below. Format: `S` + 3 digits (e.g., `S012`).
- Cap evidence at 8 refs per observation. If a pattern has more, pick the 8 most representative.
- Don't pad observations. If you only find 4 strong patterns, return 4. Better 4 sharp than 10 weak.
- Confidence calibration:
  - **high** = ≥5 supporting refs, clear pattern
  - **medium** = 3-4 supporting refs, or 5+ with some noise
  - **low** = a hunch worth flagging, weak evidence
- For `quick_wins`: only include items with a concrete one-time fix (a command, a config line, a one-paragraph prompt).
- For `per_project`: include only projects with ≥3 sessions in the input.
- No percentages unless the underlying count is ≥10.

## Aggregate stats (already computed — trust these, don't recount)

{aggregate_block}

## Sessions

{sessions_block}

Final reminder: cite refs (`S001`-`S{n_sessions_padded}`) verbatim from above. Anything else will be dropped. Return ONE JSON object.
"""


def _compact_session(narr: dict, ref: str) -> dict:
    """Reduce a narrative to its high-signal fields for the synthesis prompt.

    Aggressively tight — synthesis needs breadth across many sessions, not
    depth on any one. Heavier fields (lesson_for_agent, notable, full
    descriptions) are dropped here; a follow-up drilldown can re-load them.

    `session_id` is intentionally OMITTED — the LLM only sees `ref` and is
    instructed to cite by ref. The validator translates refs back to
    session_ids.
    """
    tasks_compact = []
    for t in narr.get("tasks") or []:
        tasks_compact.append(
            {
                "intent": (t.get("intent") or "")[:80],
                "type": t.get("task_type"),
                "difficulty": (t.get("task_difficulty") or {}).get("overall"),
                "outcome": t.get("outcome"),
            }
        )
    friction_compact = []
    for fm in narr.get("friction_moments") or []:
        friction_compact.append(
            {
                "type": fm.get("type"),
                "cat": fm.get("tool_category"),
                "cost_events": fm.get("cost_events"),
                "desc": (fm.get("description") or "")[:90],
            }
        )
    env_compact = [
        (ei.get("description") or "")[:140]
        for ei in (narr.get("recurring_environmental_issues") or [])
    ]
    return {
        "ref": ref,
        "agent": narr.get("agent"),
        "project": narr.get("project_label"),
        "date": (narr.get("started_at") or "")[:10],
        "model": narr.get("primary_model"),
        "active_min": narr.get("active_minutes"),
        "bursts": narr.get("bursts"),
        "events": narr.get("event_count"),
        "tool_calls": narr.get("tool_call_count"),
        "tool_err_rate": narr.get("tool_error_rate"),
        "user_corr": (narr.get("user_friction_signals") or {}).get(
            "explicit_corrections", 0
        ),
        "user_caught": (narr.get("user_caught_model_errors") or {}).get("count", 0),
        "verification": narr.get("verification_completeness"),
        "prompt_q": narr.get("prompt_quality_signal"),
        "topics": (narr.get("topics") or [])[:5],
        "ext_sys": narr.get("external_systems_touched") or [],
        "goal": (narr.get("goal") or "")[:120],
        "waste": narr.get("waste_signature"),
        "tasks": tasks_compact,
        "friction": friction_compact,
        "env_issues": env_compact,
        "counterfactual": (narr.get("counterfactual") or "")[:200],
        "lesson_user": (narr.get("lesson_for_user") or "")[:160],
    }


def _build_ref_map(narratives: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Assign each narrative a stable short ref. Returns (ref→session_id, session_id→ref)."""
    ref_to_id: dict[str, str] = {}
    id_to_ref: dict[str, str] = {}
    for i, n in enumerate(narratives, start=1):
        sid = n.get("session_id")
        if not sid:
            continue
        ref = f"S{i:03d}"
        ref_to_id[ref] = sid
        id_to_ref[sid] = ref
    return ref_to_id, id_to_ref


def _build_aggregate_block(narratives: list[dict]) -> str:
    """Compute deterministic aggregates the LLM can reference without recounting."""
    n = len(narratives)
    if not n:
        return "(no sessions)"
    agents = Counter(d.get("agent") for d in narratives)
    waste_sigs = Counter(d.get("waste_signature") for d in narratives)
    projects = Counter(d.get("project_label") for d in narratives)
    verif = Counter(d.get("verification_completeness") for d in narratives)
    pq = Counter(d.get("prompt_quality_signal") for d in narratives)
    task_types: Counter = Counter()
    outcomes: Counter = Counter()
    diffs: Counter = Counter()
    for d in narratives:
        for t in d.get("tasks") or []:
            task_types[t.get("task_type")] += 1
            outcomes[t.get("outcome")] += 1
            diffs[(t.get("task_difficulty") or {}).get("overall")] += 1
    total_friction = sum(len(d.get("friction_moments") or []) for d in narratives)
    total_env = sum(
        len(d.get("recurring_environmental_issues") or []) for d in narratives
    )
    total_user_caught = sum(
        (d.get("user_caught_model_errors") or {}).get("count", 0) for d in narratives
    )
    total_active = sum(d.get("active_minutes") or 0 for d in narratives)

    lines = [
        f"Total sessions: {n}",
        f"Agent split: {dict(agents)}",
        f"Total active minutes captured: {int(total_active):,}",
        f"Total friction moments: {total_friction}",
        f"Total recurring env issues: {total_env}",
        f"Total user-caught model errors: {total_user_caught}",
        f"Waste signature distribution: {dict(waste_sigs.most_common())}",
        f"Task type distribution: {dict(task_types.most_common())}",
        f"Task outcomes: {dict(outcomes.most_common())}",
        f"Task difficulty: {dict(diffs.most_common())}",
        f"Verification: {dict(verif.most_common())}",
        f"Prompt quality: {dict(pq.most_common())}",
        f"Top 15 projects by session count: {dict(projects.most_common(15))}",
    ]
    return "\n".join(lines)


def _build_sessions_block(narratives: list[dict], id_to_ref: dict[str, str]) -> str:
    compact = []
    for n in narratives:
        ref = id_to_ref.get(n.get("session_id") or "")
        if not ref:
            continue
        compact.append(_compact_session(n, ref))
    return "\n".join(json.dumps(c, ensure_ascii=False) for c in compact)


def build_synthesis_prompt(
    narratives: list[dict],
    prior_context: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the synthesis prompt. Returns (prompt, ref_to_session_id_map)."""
    ref_to_id, id_to_ref = _build_ref_map(narratives)
    aggregate = _build_aggregate_block(narratives)
    sessions = _build_sessions_block(narratives, id_to_ref)
    prior_block = _build_prior_context_block(prior_context)
    prompt = SYNTHESIS_PROMPT
    n = len(narratives)
    for token, value in (
        ("{n_sessions}", str(n)),
        ("{n_sessions_padded}", f"{n:03d}"),
        ("{aggregate_block}", aggregate),
        ("{sessions_block}", sessions),
        ("{prior_context_block}", prior_block),
    ):
        prompt = prompt.replace(token, value)
    return prompt, ref_to_id


def _build_prior_context_block(prior_context: str | None) -> str:
    if not prior_context:
        return ""
    return (
        "## Prior runs (with the user's ratings)\n\n"
        + prior_context.strip()
        + "\n\n"
        + "## How to use prior-run context\n"
        "- If a pattern you'd surface now was rated WRONG before, do NOT surface it again unless there's genuinely new, stronger evidence (more sessions, new project affected, etc.).\n"
        "- If a pattern was rated USEFUL and is still present, mark the new observation with `continues: \"<prior key>\"` and `trend: \"improving\" | \"worsening\" | \"stable\"`.\n"
        "- If a pattern was rated ALREADY KNOWN, only re-surface with a meaningful new twist.\n"
        "- Prioritize NET-NEW patterns that haven't been surfaced before.\n"
    )


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


async def _call_claude(prompt: str, model: str) -> str:
    collected = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model=model,
            allowed_tools=[],
            system_prompt="Return only valid JSON. No markdown fence, no preamble.",
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    collected += block.text
        elif isinstance(message, ResultMessage):
            break
    return collected


def _validate(parsed: dict, ref_to_id: dict[str, str]) -> dict:
    """Translate refs to session_ids; drop fabricated refs.

    Each observation/quick_win ends up with both `evidence_refs` (verified)
    and `evidence_sessions` (resolved real IDs). Fabricated refs are dropped
    and counted in meta.notes.
    """
    valid_refs = set(ref_to_id.keys())
    dropped_total = 0

    def _resolve(refs: list) -> tuple[list[str], list[str], int]:
        """Return (kept_refs, resolved_session_ids, dropped_count)."""
        if not isinstance(refs, list):
            return [], [], 0
        kept_refs: list[str] = []
        resolved: list[str] = []
        for r in refs:
            if isinstance(r, str) and r in valid_refs:
                kept_refs.append(r)
                resolved.append(ref_to_id[r])
        return kept_refs, resolved, len(refs) - len(kept_refs)

    obs_clean: list[dict] = []
    for obs in parsed.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        kept_refs, resolved, dropped = _resolve(obs.get("evidence_refs") or [])
        dropped_total += dropped
        obs["evidence_refs"] = kept_refs
        obs["evidence_sessions"] = resolved
        obs["supporting_count"] = len(kept_refs)
        if kept_refs:
            obs_clean.append(obs)
    parsed["observations"] = obs_clean

    qw_clean: list[dict] = []
    for qw in parsed.get("quick_wins") or []:
        if not isinstance(qw, dict):
            continue
        kept_refs, resolved, dropped = _resolve(qw.get("evidence_refs") or [])
        dropped_total += dropped
        qw["evidence_refs"] = kept_refs
        qw["evidence_sessions"] = resolved
        qw["affected_sessions"] = len(kept_refs)
        if kept_refs:
            qw_clean.append(qw)
    parsed["quick_wins"] = qw_clean

    if dropped_total:
        meta = parsed.setdefault("meta", {})
        existing = meta.get("notes") or ""
        meta["notes"] = (
            existing
            + (". " if existing and not existing.endswith(".") else "")
            + f"Dropped {dropped_total} fabricated ref{'s' if dropped_total != 1 else ''}."
        ).strip()
    meta = parsed.setdefault("meta", {})
    meta["fabricated_ref_count"] = dropped_total
    return parsed


def load_narratives(narratives_dir: Path) -> list[dict]:
    """Load all per-session narrative JSON files in a directory."""
    out: list[dict] = []
    for path in sorted(narratives_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("session_id"):
            continue
        out.append(data)
    return out


def synthesize(
    narratives: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    project_filter: str | None = None,
    prior_context: str | None = None,
) -> dict[str, Any]:
    """Run the cross-session synthesis pass.

    Args:
        narratives: list of per-session narrative dicts (output of v1 schema).
        model: LLM model.
        project_filter: if set, only include sessions whose project_label
            substring-matches this value (case-insensitive).
        prior_context: optional pre-formatted summary of prior synthesis runs +
            user ratings; produced by ``HistoryStore.summarize_for_prompt``.
    """
    if project_filter:
        needle = project_filter.lower()
        narratives = [
            n for n in narratives if needle in (n.get("project_label") or "").lower()
        ]
    if not narratives:
        return {
            "observations": [],
            "quick_wins": [],
            "per_project": [],
            "meta": {"notes": "No narratives matched filter.", "input_sessions": 0},
        }

    prompt, ref_to_id = build_synthesis_prompt(narratives, prior_context=prior_context)

    # Sonnet's input cap is ~200K tokens (~800K chars at the conservative
    # 4 chars/token heuristic). Anything close to that risks truncation
    # mid-prompt, which produces silently-degraded output. Hard-fail
    # instead so the user sees the cause and can drop --lookback-days.
    PROMPT_CHAR_LIMIT = 700_000
    if len(prompt) > PROMPT_CHAR_LIMIT:
        raise ValueError(
            f"synthesis prompt is {len(prompt):,} chars (~{len(prompt)//4:,} tokens), "
            f"exceeding the {PROMPT_CHAR_LIMIT:,}-char safety limit. "
            f"Pass --lookback-days N or --limit N to reduce the input set."
        )

    raw = asyncio.run(_call_claude(prompt, model))
    parsed = _extract_json(raw)
    validated = _validate(parsed, ref_to_id)
    meta = validated.setdefault("meta", {})
    meta["input_sessions"] = len(narratives)
    meta["model"] = model
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    meta["had_prior_context"] = bool(prior_context)
    if project_filter:
        meta["project_filter"] = project_filter
    return validated
