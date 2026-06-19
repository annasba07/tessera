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

from ..backends import LLMBackend, get_backend


DEFAULT_MODEL = "claude-sonnet-4-6"


SYNTHESIS_PROMPT = """You are reading {n_sessions} narrative summaries of one person's AI coding agent sessions. Your job: surface cross-cutting patterns through TWO lenses simultaneously:

  1. **OPERATIONAL** — what's broken in their tooling, environment, projects (the things a Friday-afternoon config edit would fix).
  2. **BEHAVIORAL** — what's broken or worth experimenting with in *how this person works* — their prompting habits, decision patterns, intervention timing, delegation balance, verification discipline, cognitive rhythm. (The things a one-week experiment in *how* they prompt or *when* they intervene would change.)

Both lenses matter. Operational gets concrete fixes. Behavioral gets the deeper insight nobody else can give them — patterns invisible from any single session.

{prior_context_block}

Per-session analysis is done. Your job is the cross-session layer — what shows up across sessions they can't see from any single one.

## CRITICAL: Citation by ref token only

Every session below has a `ref` field — a short token like `S001`, `S047`, `S221`. When you cite a session as evidence, **cite by ref only**. Refs are exactly 4 characters: `S` followed by a 3-digit zero-padded number. Copy them character-by-character from the input.

DO NOT invent refs. DO NOT cite by `session_id`. If a ref doesn't appear in the input below, it will be dropped. Refs in the input go from `S001` through `S{n_sessions_padded}` — anything outside that range is a fabrication.

Before you finalize, mentally scan each cited ref and confirm it appears verbatim in the input.

## What separates great observations from mediocre ones

**Mediocre**: "You had 67 env issues across 57 sessions, 41 of them were stale Chrome locks." (Operational, true, but the user already knows their tooling breaks.)

**Great operational**: "67 env issues, 41 are stale Chrome locks — but ALL 41 happened in `agent-2/` and `agent-3/` worktrees, never in `main`. The stale lock is a worktree-isolation problem, not a Chrome problem. Fix: pre-create per-worktree Chrome profile dirs in the worktree-init skill."

**Great behavioral**: "When you open with 'try X' (47 sessions), median dead-ends = 2.3 and time-to-first-verified-progress = 8.4 events. When you open with 'the goal is Y, constraints are Z' (34 sessions), median dead-ends = 0.6 and TTFP = 3.1. The goal-first style is 2.7× more efficient by both metrics. You used `try X` 31% of the time including 8 sessions where you knew the goal — try the goal-first phrasing on Mon-Wed next week and compare your own dashboards."

Behavioral observations should be:
- **Comparative** (this style vs that style, this domain vs that domain, this time-of-day vs another) — patterns reveal themselves only by comparison
- **Quantified** with the specific numerator AND denominator
- **Actionable as a 1-week experiment**, not "be more careful"
- **Surprising** — something the user wouldn't have guessed about themselves

## Outcome data — use this as ground truth, not just intent

Many sessions have an `outcome` field with what *actually shipped*: branch lifecycle, file churn in the next 14 days, and PR state (merged/open/closed, CI pass/fail, review approval). Outcome signals: `shipped_clean`, `shipped_with_followups`, `reverted`, `in_progress`, `abandoned`, `no_artifact`, `unavailable`.

This changes how you should think about patterns:

- **Friction without outcome is half the story.** A session with 200 events of friction that `shipped_clean` is very different from a session with 200 events of friction that was `reverted` or needed 5 fixup commits. Rank patterns by *value-delivered impact*, not just event count.
- **Look for shipping-quality patterns**: which prompting styles, intervention timings, or workflows correlate with `shipped_clean` vs. `shipped_with_followups` vs. `reverted`?
- **No-artifact sessions are not failures.** Many are exploration/debugging — judge them by whether the exploration was efficient, not by absence of a PR.
- **Calibration check**: claimed task `outcome="completed"` from the per-session narrative + `outcome_signal="reverted"` or `shipped_with_followups` reveals over-claiming. If a session said it succeeded but the code came back with fix commits a week later, that's signal — surface it.

When citing patterns that involve outcomes, prefer comparative claims grounded in real ship-quality: "in 8 sessions where you used X style, 6 shipped_clean and 2 shipped_with_followups; in 8 sessions with Y style, 2 shipped_clean, 4 shipped_with_followups, 2 reverted." That's much more useful than counting friction events alone.

## Behavioral patterns to actively look for

Read across `lesson_for_user`, `lesson_for_agent`, `prompt_q`, `decisions[].retro`, `dead_ends`, `user_friction`, `user_caught_count/examples`, `verification`, `bursts`, `events_per_min`, `subagents`, `tod` (time-of-day), `weekday`:

- **Prompting style signature**: vocabulary, openers, constraint-first vs action-first, vague-trying vs specified-goal. Which styles correlate with shorter paths / fewer dead-ends?
- **Intervention timing**: when does the user redirect the agent? (Inferred from `user_friction.explicit_corrections` vs total events.) Do early interventions correlate with better outcomes?
- **Delegation balance**: where do they over-delegate (high `subagents` but high `dead_ends`)? Where do they under-delegate (long `active_min` with no subagents on tasks subagents could handle)?
- **Verification discipline**: `claimed_only` vs `tested` vs `verified` rates per task type / project — where do they trust agent output without checking, and what's the cost?
- **Catch-rate vs miss-rate**: `user_caught_count` patterns. Are they vigilant on backend Python, blind on frontend React? Late-evening sessions vs morning?
- **Cognitive arc**: long-active-min sessions, single-burst vs fragmented (`bursts`), `events_per_min` across `tod` and `weekday`. When does work go best vs worst?
- **Recurring lessons**: `lesson_for_user` patterns that repeat verbatim across sessions = lessons they keep rediscovering and not internalizing.
- **Dead-end clusters**: shared root cause or shared cognitive pattern across `dead_ends`?
- **Domain-specific blind spots**: same person succeeds easily in domain A, struggles in domain B — what's the underlying skill/style mismatch?
- **Counterfactual recurrences**: across `counterfactual` fields, what could-have-been-better patterns repeat?

## Operational patterns to actively look for

- **Environmental**: stale lock files, expired auth, missing deps, MCP issues. Often a one-time per-project fix.
- **Project-specific failure modes**: a worktree or repo where the agent reliably struggles in a specific way.
- **Tooling/workflow**: tool choices, sequencing patterns, MCP server combos that backfire.
- **Cross-agent**: differences between Claude / Codex / Gemini on similar tasks.

## Output format

Return ONE JSON object. No markdown fence. No preamble:

```
{
  "headline": "<≤120 chars: ONE declarative sentence stating the single biggest takeaway. No semicolons. No estimation hedging ('roughly', 'approximately', 'an estimated 30-40%'). Plain English: 'X happened in N sessions; doing Y instead would prevent it.'>",
  "if_you_do_one_thing_this_week": "<≤200 chars: ONE concrete action — a command, a config edit, a literal sentence to add to CLAUDE.md, or a one-line habit change. MUST NOT restate the headline. The headline names the problem; this names the next move. If the headline is 'X breaks things in N sessions', this is 'add Y to file Z' — not 'fix X.'>",

  "observations": [
    {
      "title": "<short phrase>",
      "claim": "<1-2 sentences with specific numbers>",
      "evidence_refs": ["S012", "S047", ...],
      "supporting_count": <int = len(evidence_refs)>,
      "interpretation": "<2-4 sentences on why this is happening>",
      "next_action": "<concrete command, prompt, habit, or config — not 'be more careful'>",
      "confidence": "high | medium | low",
      "category": "environmental | project_specific | tooling | workflow | cross_agent"
    }
  ],

  "behavioral_patterns": [
    {
      "title": "<short phrase>",
      "pattern": "<1-3 sentences: the recurring USER habit/style/tendency, with comparative numbers (style A in N sessions → outcome X; style B in M sessions → outcome Y)>",
      "evidence_refs": ["S012", "S047", ...],
      "supporting_count": <int = len(evidence_refs)>,
      "interpretation": "<2-4 sentences: why this pattern exists, what underlying habit/blind spot drives it, what the cost is>",
      "experiment_to_try": "<a specific 1-week experiment with success metric: 'On Mon-Wed, prompt with X format. On Thu-Fri, prompt with Y format. Compare dead_ends per session in next dashboard.'>",
      "confidence": "high | medium | low",
      "dimension": "prompting_style | intervention_timing | delegation | verification | cognitive_arc | recurring_lesson | domain_blind_spot | counterfactual_pattern"
    }
  ],

  "quick_wins": [
    {
      "fix": "<one-line action with command if applicable>",
      "affected_sessions": <int>,
      "evidence_refs": ["S001", ...]
    }
  ],

  "skill_candidates": [
    {
      "kind": "new_skill | deepen_existing",
      "title": "<≤80 chars: name the skill, e.g. 'pulse-preflight: gworkspace auth canary'>",
      "trigger_pattern": "<what recurring agent behavior would invoke this skill — be specific>",
      "what_it_should_do": "<3-6 lines of actionable steps the skill should encode>",
      "affected_sessions": <int>,
      "evidence_refs": ["S001", ...],
      "existing_skill_hint": "<if kind=deepen_existing: which existing skill name does the data suggest is missing depth; null otherwise>",
      "confidence": "high | medium | low"
    }
  ],

  "per_project": [
    {
      "project": "<project_label>",
      "session_count": <int>,
      "headline": "<≤200 chars: dominant pattern in this project>",
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
- Cap evidence at 8 refs per observation/pattern. If more support exists, pick the 8 most representative.
- Don't pad. If you only find 4 strong operational observations and 3 behavioral patterns, return 4 and 3. Better sharp than padded. Quality > count, always.
- **Pattern targets** (these are floors, not ceilings — surface every distinct pattern the data supports):
  - `observations` (operational): aim for 6-12. Don't merge unrelated friction sources just to hit a number; don't pad either. Prefer fewer-but-sharper.
  - `behavioral_patterns`: **aim for 8-15 strong comparative patterns**. Each `dimension` value can support multiple distinct patterns (3 different `prompting_style` patterns is fine if they're genuinely distinct). Output budget caps at ~8K tokens; if you have more candidates than budget, drop the weakest evidence first rather than truncating fields. **Crucially: a complete short pattern is more valuable than a truncated detailed one.**
- Confidence:
  - **high** = ≥5 supporting refs, clear pattern, comparison-supported claim
  - **medium** = 3-4 supporting refs, or 5+ with noise/edge cases
  - **low** = hunch worth flagging, weak evidence (these are valuable — surface them with `low` confidence rather than dropping them)
- For `quick_wins`: only one-time fixes (a command, a config line). Behavioral changes go in `behavioral_patterns.experiment_to_try`, not here.
- For `skill_candidates`: surface a candidate when a recurring agent task (≥3 sessions) would benefit from being codified as a reusable agent skill rather than re-improvised every session.
  - **new_skill** = the work is being repeated freshly each time. The codified skill should encode the canonical steps, the failure modes to avoid, and any pre-flight checks. Example: "pulse synth requires 5-account auth check + parallel fan-out + named-track fallback for Granola" → a `pulse-preflight` skill.
  - **deepen_existing** = a skill name (or skill-like prompt template) appears in sessions but keeps missing the same failure mode. Set `existing_skill_hint` to the skill name. The "what_it_should_do" should be what's MISSING, not the whole skill. Example: existing `/pulse` skill is being used, but every run rediscovers the OAuth pre-flight gap — that's a deepen_existing for `pulse`.
  - Don't restate observations or behavioral_patterns as skill candidates. A skill candidate is when CODIFYING the fix is meaningfully better than just running the fix. If a one-line config change suffices, it's a quick_win. If a behavioral shift is needed, it's a behavioral_pattern. If a reusable workflow encoding would help, it's a skill_candidate.
  - Aim for 2-5. Often zero is the right answer — only surface if the codification value is clear.
- For `per_project`: only projects with ≥3 sessions.
- No percentages unless the underlying count is ≥10.
- Behavioral patterns MUST include a comparison ("style A vs style B", "domain X vs domain Y", "early in session vs late", "Codex vs Claude on this task type", "well-formed prompt vs vague prompt", "with-subagent vs without"). Comparison is what makes them insightful rather than descriptive. **A pattern without a comparison is a description, not a pattern — drop it or downgrade to `low` confidence.** "You work mostly at night (54K events vs 8.8K afternoon)" is descriptive ONLY because it has no behavioral comparison; "Night sessions show 2.3× the rate of context-loss waste vs. afternoon sessions" is a real pattern. The control matters: if you can't show that the same person/task in the contrasting condition behaves differently, you don't have a pattern.

- **MANDATORY METRICS: every behavioral pattern's `pattern` field must include at least TWO specific numeric values on the comparison — one per side.** Counts, percentages, ratios, event-counts, time-deltas, error rates — whatever is grounded in the input narratives. The numbers are how this differs from advice / horoscope writing. Examples of the bar:
  - GOOD: "When you open with 'try X' (47 sessions), median dead-ends=2.3 and time-to-first-progress=8.4 events. When you open with 'goal is Y, constraints are Z' (34 sessions), median dead-ends=0.6, TTFP=3.1. Goal-first is 2.7× more efficient."
  - GOOD: "Subagent-zero sessions (49/50, 98%) average 41 events to completion. The one delegated session (S018, 1/50, 2%) consumed 4,594 events across 219 subagents. The bimodal distribution wastes both modes' strengths."
  - BAD (drop or low): "User has a binary delegation style." (no numbers)
  - BAD (drop or low): "Slack vigilance is higher than Gmail vigilance." (no numbers)
  - BAD (drop or low): "Afternoon sessions tend to be longer." (one-sided, no comparison metrics)

- **MANDATORY MEASURABLE EXPERIMENT: every `experiment_to_try` must specify the exact metric to compare next week.** "Try X next week" is not measurable. "On Mon-Wed do X, log metric M; on Thu-Fri do Y, compare M in next dashboard" IS measurable. The point of the experiment field is that the next-week evaluator can read your dashboard and judge effect by reading M. If you can't name M, you don't have a real experiment — replace with one you can.

- **Framing reminder — agent-USE quality, not life advice.** You are helping someone improve how they USE coding agents (Claude Code, Codex, Gemini, Antigravity). Insights that translate to "edit a prompt", "add a skill", "intervene earlier", "verify differently", "delegate this kind of task" are gold. Insights that translate to "go to bed earlier" or "work less weekends" are not what this user paid for. Stay in the lane of: prompting style, tool selection, intervention timing, delegation patterns, verification discipline, error-recovery patterns, context provision, iteration vs one-shot strategy, agent/model choice per task. If a pattern doesn't map to one of those levers, it's noise.

## Aggregate stats (already computed — trust, don't recount)

{aggregate_block}

## Sessions

{sessions_block}

## SCHEMA LOCK — read this before writing the JSON

The output schema above is fixed. Do NOT rename fields. Do NOT invent new fields (no `corpus_size`, no `dominant_outcomes`, no `pattern_refs`, no `O01` / `B01` ref scheme). Do NOT migrate to a "v2" shape because prior runs feel familiar — every run uses this exact schema.

In particular:
- Findings live in `observations` (operational) and `behavioral_patterns` (behavioral). Not in `per_project`. Not in `meta.notes`. Not anywhere else.
- Evidence refs are `S001`-`S{n_sessions_padded}`, not `O05` or `B12`.
- `meta` may contain only: `notes` (string, optional caveats). The aggregate stats above are pre-computed; don't restate them in meta.
- Returning `observations: []` AND `behavioral_patterns: []` while writing about patterns in `notes` is a contradiction — patterns belong in the typed fields, not in a free-text note. If you genuinely found nothing, say so explicitly in notes; if you found something, put it in the right list.

Final reminder: cite refs (`S001`-`S{n_sessions_padded}`) verbatim from above. Return ONE JSON object with both `observations` (operational) and `behavioral_patterns` (behavioral) populated.
"""


def _compact_session(narr: dict, ref: str) -> dict:
    """Reduce a narrative to its high-signal fields for the synthesis prompt.

    Two simultaneous goals:
      1. Operational signal — what's failing, where, how many sessions affected
         (kept tight: friction descriptions truncated to ~120 chars, env
         issues to ~160).
      2. Behavioral signal — how *this person* prompts, decides, intervenes,
         delegates, verifies (kept verbatim: lessons, decision retrospectives,
         dead_ends, friction_signals, time/burst rhythm, delegation pattern).

    `session_id` is intentionally omitted — the LLM only sees `ref` and is
    instructed to cite by ref. The validator translates refs back to
    session_ids.
    """
    tasks_compact = []
    for t in narr.get("tasks") or []:
        diff = t.get("task_difficulty") or {}
        tasks_compact.append(
            {
                "intent": (t.get("intent") or "")[:140],
                "type": t.get("task_type"),
                "difficulty": diff.get("overall"),
                "spec": diff.get("specification"),
                "verify": diff.get("verification_ease"),
                "outcome": t.get("outcome"),
                "ttfvp": t.get("time_to_first_verified_progress"),
            }
        )
    friction_compact = []
    for fm in narr.get("friction_moments") or []:
        friction_compact.append(
            {
                "type": fm.get("type"),
                "cat": fm.get("tool_category"),
                "cost_events": fm.get("cost_events"),
                "desc": (fm.get("description") or "")[:120],
            }
        )
    env_compact = [
        (ei.get("description") or "")[:160]
        for ei in (narr.get("recurring_environmental_issues") or [])
    ]
    # Behavioral fields kept verbatim — these are what synthesis needs to
    # surface "how does this person actually work" patterns.
    decisions_compact = []
    for kd in narr.get("key_decisions") or []:
        decisions_compact.append(
            {
                "decision": (kd.get("decision") or "")[:140],
                "retro": (kd.get("retrospective") or "")[:200],
            }
        )
    dead_ends_compact = []
    for de in narr.get("dead_ends") or []:
        dead_ends_compact.append(
            {
                "what": (de.get("what") or de.get("description") or "")[:140],
                "why": (de.get("why_dead_end") or de.get("why") or "")[:200],
                "cost_events": de.get("cost_events"),
            }
        )
    ucme = narr.get("user_caught_model_errors") or {}
    user_caught_examples = [
        (e if isinstance(e, str) else (e.get("description") or ""))[:140]
        for e in (ucme.get("examples") or [])
    ][:3]
    # Outcome (if enriched). Compact form — LLM only needs the signal label
    # plus the fields that explain it, not full git logs.
    outcome_compact: dict | None = None
    outcome = narr.get("outcome") or {}
    if outcome.get("outcome_signal"):
        oc: dict = {"signal": outcome["outcome_signal"]}
        churn = outcome.get("files_churn") or {}
        if churn.get("commits_touching_files"):
            oc["churn"] = {
                "commits_in_14d": churn.get("commits_touching_files"),
                "fixup": churn.get("fixup_shape_commits"),
                "revert": churn.get("revert_commits"),
            }
        if outcome.get("prs"):
            oc["prs"] = [
                {
                    "n": p.get("number"),
                    "state": p.get("state"),
                    "ci": p.get("ci_status"),
                    "review": p.get("review_decision"),
                }
                for p in outcome["prs"][:3]
            ]
        b = outcome.get("branch") or {}
        if b.get("merged_into") or b.get("commits_after_session"):
            oc["branch"] = {
                "merged_into": b.get("merged_into"),
                "commits_after": b.get("commits_after_session"),
            }
        outcome_compact = oc

    return {
        # Identity
        "ref": ref,
        "agent": narr.get("agent"),
        "project": narr.get("project_label"),
        "date": (narr.get("started_at") or "")[:10],
        "weekday": narr.get("weekday"),
        "tod": narr.get("time_of_day_buckets"),
        "model": narr.get("primary_model"),
        # Outcome — what actually happened to the work after the session
        "outcome": outcome_compact,
        # Rhythm + delegation (behavioral)
        "active_min": narr.get("active_minutes"),
        "bursts": narr.get("bursts"),
        "primary_burst_min": narr.get("primary_burst_minutes"),
        "events": narr.get("event_count"),
        "events_per_min": narr.get("events_per_active_minute"),
        "tool_calls": narr.get("tool_call_count"),
        "tool_err_rate": narr.get("tool_error_rate"),
        "subagents": narr.get("subagent_count"),
        "subagent_types": narr.get("subagent_types_spawned"),
        # User behavior signals (verbatim)
        "user_friction": narr.get("user_friction_signals"),
        "user_caught_count": ucme.get("count"),
        "user_caught_examples": user_caught_examples,
        "verification": narr.get("verification_completeness"),
        "prompt_q": narr.get("prompt_quality_signal"),
        # Topical
        "topics": (narr.get("topics") or [])[:5],
        "ext_sys": narr.get("external_systems_touched") or [],
        "goal": (narr.get("goal") or "")[:200],
        "waste": narr.get("waste_signature"),
        # Operational structure
        "tasks": tasks_compact,
        "friction": friction_compact,
        "env_issues": env_compact,
        # Behavioral narrative — kept verbatim
        "decisions": decisions_compact,
        "dead_ends": dead_ends_compact,
        "counterfactual": narr.get("counterfactual"),
        "lesson_user": narr.get("lesson_for_user"),
        "lesson_agent": narr.get("lesson_for_agent"),
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
    """Compute deterministic aggregates the LLM can reference without recounting.

    Operational aggregates (counts of friction/env/etc.) plus behavioral
    aggregates (rhythm, intervention rate, delegation balance, time-of-day
    distribution) — both lenses get pre-computed numbers to ground claims.
    """
    n = len(narratives)
    if not n:
        return "(no sessions)"
    agents = Counter(d.get("agent") for d in narratives)
    waste_sigs = Counter(d.get("waste_signature") for d in narratives)
    projects = Counter(d.get("project_label") for d in narratives)
    verif = Counter(d.get("verification_completeness") for d in narratives)
    pq = Counter(d.get("prompt_quality_signal") for d in narratives)
    weekdays = Counter(d.get("weekday") for d in narratives if d.get("weekday"))
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

    # Outcome aggregates (only for enriched narratives)
    outcome_signals = Counter()
    sessions_enriched = 0
    fixup_total = 0
    revert_total = 0
    for d in narratives:
        oc = d.get("outcome") or {}
        sig = oc.get("outcome_signal")
        if sig:
            outcome_signals[sig] += 1
            sessions_enriched += 1
        churn = oc.get("files_churn") or {}
        fixup_total += churn.get("fixup_shape_commits") or 0
        revert_total += churn.get("revert_commits") or 0

    # Behavioral aggregates
    sessions_with_corrections = sum(
        1 for d in narratives
        if (d.get("user_friction_signals") or {}).get("explicit_corrections", 0) > 0
    )
    sessions_with_caught_errors = sum(
        1 for d in narratives
        if (d.get("user_caught_model_errors") or {}).get("count", 0) > 0
    )
    sessions_with_subagents = sum(
        1 for d in narratives if (d.get("subagent_count") or 0) > 0
    )
    total_subagents = sum(d.get("subagent_count") or 0 for d in narratives)
    sessions_with_dead_ends = sum(
        1 for d in narratives if d.get("dead_ends")
    )
    total_dead_ends = sum(len(d.get("dead_ends") or []) for d in narratives)
    # Time-of-day: aggregate the per-session bucket dicts
    tod_total: Counter = Counter()
    for d in narratives:
        for bucket, count in (d.get("time_of_day_buckets") or {}).items():
            tod_total[bucket] += count
    # Session-length distribution by active minutes
    length_bins = Counter()
    for d in narratives:
        m = d.get("active_minutes") or 0
        if m < 5: length_bins["<5min"] += 1
        elif m < 15: length_bins["5-15min"] += 1
        elif m < 45: length_bins["15-45min"] += 1
        elif m < 120: length_bins["45-120min"] += 1
        else: length_bins["120min+"] += 1
    # Burst-rhythm distribution: how many sessions are single-burst vs multi-burst
    burst_dist = Counter()
    for d in narratives:
        b = d.get("bursts") or 0
        if b == 1: burst_dist["1-burst"] += 1
        elif b <= 3: burst_dist["2-3-burst"] += 1
        else: burst_dist["4+-burst"] += 1

    lines = [
        "## Operational",
        f"Total sessions: {n}",
        f"Agent split: {dict(agents)}",
        f"Total active minutes captured: {int(total_active):,}",
        f"Total friction moments: {total_friction}  (avg {total_friction/n:.1f}/session)",
        f"Total recurring env issues: {total_env}  (avg {total_env/n:.1f}/session)",
        f"Total dead-ends recorded: {total_dead_ends}  (in {sessions_with_dead_ends}/{n} sessions)",
        f"Waste signature distribution: {dict(waste_sigs.most_common())}",
        f"Task type distribution: {dict(task_types.most_common())}",
        f"Task outcomes: {dict(outcomes.most_common())}",
        f"Task difficulty: {dict(diffs.most_common())}",
        f"Top 15 projects by session count: {dict(projects.most_common(15))}",
        "",
        "## Behavioral",
        f"User-caught model errors: {total_user_caught} total in {sessions_with_caught_errors}/{n} sessions ({sessions_with_caught_errors/n*100:.0f}%)",
        f"Sessions where user explicitly corrected the agent: {sessions_with_corrections}/{n} ({sessions_with_corrections/n*100:.0f}%)",
        f"Subagent delegation: {sessions_with_subagents}/{n} sessions used subagents ({sessions_with_subagents/n*100:.0f}%); {total_subagents} total spawn",
        f"Verification habit: {dict(verif.most_common())}",
        f"Prompt quality (per-session signal): {dict(pq.most_common())}",
        f"Session length distribution: {dict(length_bins.most_common())}",
        f"Burst rhythm: {dict(burst_dist.most_common())}  (1-burst = single sustained flow, 4+ = highly fragmented)",
        f"Time-of-day event distribution: {dict(tod_total.most_common())}",
        f"Weekday distribution: {dict(weekdays.most_common())}",
    ]
    if sessions_enriched:
        lines += [
            "",
            "## Outcomes (enriched from git/gh after the fact)",
            f"Sessions with outcome lookup: {sessions_enriched}/{n}",
            f"Outcome signal distribution: {dict(outcome_signals.most_common())}",
            f"Total fixup-shape commits in 14d after sessions: {fixup_total}",
            f"Total revert commits in 14d after sessions: {revert_total}",
            "(Signals: shipped_clean = merged via PR, no follow-ups; shipped_direct = committed straight to trunk without a PR (common for solo work — scripts, memory files, configs, infra); shipped_with_followups = shipped (PR or trunk) but needed fix/hotfix commits in 14d; reverted = explicit revert detected; in_progress = still landing commits or open PR; abandoned = closed unmerged or branch deleted; unshipped = touched files but no commits landed in window (drafts, scratched work); exploration = no files touched (chat, research, Q&A); non_repo = work in non-git directory.)",
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Recovery 0: Flash 3.5 High sometimes emits a complete valid JSON
        # object followed by prose commentary ("...this analysis covers..."
        # explaining what it did). raw_decode parses the first JSON value
        # and returns its end-index; everything after is silently dropped.
        if text.lstrip().startswith("{"):
            try:
                decoder = json.JSONDecoder()
                obj, _end = decoder.raw_decode(text.lstrip())
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        # Recovery 0.5: Sonnet 4.6 sometimes emits invalid JSON escapes
        # mid-string. Observed in consistency Run 3: 'domains matching the
        # client track.\{"kind":"deepen_existing"' — literal '\{' which
        # isn't a valid JSON escape. Strip any '\' that's followed by
        # something other than a valid escape char and retry. Safe for
        # tessera output because legitimate backslashes are rare in
        # narrative free-text and the cleanup preserves the character.
        try:
            cleaned = re.sub(r'\\([^"\\/bfnrtu])', r'\1', text)
            if cleaned != text:
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
                # If escape-cleanup didn't fully fix it, try raw_decode
                # on the cleaned version too.
                if cleaned.lstrip().startswith("{"):
                    try:
                        decoder = json.JSONDecoder()
                        obj, _end = decoder.raw_decode(cleaned.lstrip())
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        # Recovery 1: Sonnet occasionally collapses mid-output and restarts,
        # leaving partial corrupted JSON followed by a fresh complete one.
        # The schema's required top-level key is "headline" — find every
        # `{"headline"` boundary, try to parse a balanced object from each
        # (prefer the LAST one — the model's final attempt is usually
        # the complete one). Return the first that parses.
        anchors = []
        i = 0
        while True:
            j = text.find('{"headline"', i)
            if j < 0:
                break
            anchors.append(j)
            i = j + 1
        for start in reversed(anchors):  # last attempt first
            depth = 0
            in_str = False
            esc = False
            for k in range(start, len(text)):
                ch = text[k]
                if esc:
                    esc = False
                    continue
                if ch == "\\" and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : k + 1])
                        except json.JSONDecodeError:
                            break
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Dump the raw LLM output so the user can see what came back
        # (empty? truncated? markdown wrapped in unexpected way?) instead
        # of just a "Expecting value: line 1 column 1" trace with no body.
        import os, tempfile, time
        dump_path = Path(tempfile.gettempdir()) / f"tessera-synthesis-raw-{int(time.time())}.txt"
        try:
            dump_path.write_text(
                f"# Raw synthesis output that failed json.loads\n"
                f"# error: {exc}\n"
                f"# raw length: {len(raw)} chars\n"
                f"# stripped length: {len(text)} chars\n"
                f"#\n"
                f"{raw}",
                encoding="utf-8",
            )
            print(
                f"\nerror: model returned non-JSON output (length={len(raw)} chars). "
                f"Raw response saved to {dump_path} for inspection.",
                file=__import__("sys").stderr,
            )
        except OSError:
            pass
        raise


# Sentinel substrings the SDK returns as the entire assistant text when the
# underlying request fails at the HTTP/CLI layer. These look like content
# but really mean "the model never got to generate." Retry once, then bail.
_TRANSIENT_FAILURE_SIGNALS = (
    "Request timed out",
    "Connection reset",
    "Connection error",
    "Rate limit",
    "rate_limit_error",
    "overloaded_error",
    "Service Unavailable",
    "503 Service",
    # CLI-backend subprocess timeout (set in backends._CLI_TIMEOUT_SEC).
    # Observed: agy/Flash 3.5 occasionally chooses to spawn agentic search
    # tasks instead of producing JSON, then times out at 15 min. The other
    # 2-of-3 runs on the same input completed in 146s and 209s — a fresh
    # retry usually succeeds.
    "hung past",
)

# Sentinel substrings that mean the backend (agy in particular) decided
# to use tools and either timed out or returned nothing but exploration
# narration instead of the synthesis JSON. The consistency check on
# Flash High showed 2 of 3 runs failed this way. Detected separately so
# we can retry with a stronger no-tool-use reminder appended.
_AGENTIC_ROGUE_SIGNALS = (
    "I am waiting for",
    "background task",
    "background search",
    "I will check the files",
    "I will list the contents",
    "Error: timed out waiting for response",
)


async def _call_llm(
    backend: LLMBackend,
    prompt: str,
    model: str,
    *,
    max_retries: int = 2,
) -> str:
    """Call the configured backend with one-shot retry on transient failures.

    Backends return short error sentinels ("Request timed out", rate-limit
    messages) as the entire response when the underlying HTTP request
    fails. Without retry, every long-prompt synthesis would crash on the
    first hiccup. Retries with exponential backoff (3s, 9s).

    Also catches RuntimeError from CLI backends (non-zero exit) and treats
    it as a transient signal for retry — the CLI's stderr often contains
    the same transient strings.
    """
    import sys
    last_collected = ""
    for attempt in range(max_retries + 1):
        # On retry after a rogue-tool-use failure, append a stronger
        # no-exploration reminder. Agy's Flash 3.5 sometimes decides to
        # search the filesystem for "untruncated source data" instead of
        # working from the provided narratives.
        attempt_prompt = prompt
        if attempt > 0:
            attempt_prompt = (
                prompt
                + "\n\n## CRITICAL: do not use tools\n"
                "All session data needed is in this prompt above. Do NOT search the filesystem, "
                "do NOT spawn background tasks, do NOT try to find 'untruncated source' — the "
                "compact narratives below are the complete input. Produce the JSON directly "
                "from what's in this prompt."
            )
        try:
            collected = await backend.complete(attempt_prompt, model)
        except RuntimeError as exc:
            collected = str(exc)
        last_collected = collected
        stripped = collected.strip()
        is_short_failure = (
            len(stripped) < 200
            and any(sig.lower() in stripped.lower() for sig in _TRANSIENT_FAILURE_SIGNALS)
        )
        # Rogue agentic narration: the model never produced JSON, just
        # described what it was "waiting for." Length threshold higher
        # because these can run 500-5000 chars of narration.
        is_rogue_agentic = (
            "{" not in stripped[:200]  # didn't start with JSON
            and any(sig.lower() in stripped.lower() for sig in _AGENTIC_ROGUE_SIGNALS)
        )
        if not (is_short_failure or is_rogue_agentic):
            return collected
        if attempt < max_retries:
            failure_kind = "transient" if is_short_failure else "rogue tool-use"
            backoff_s = 3 * (1 + 2 * attempt)  # 3s, 9s
            print(
                f"  → LLM returned {failure_kind} failure ({stripped[:80]!r}) — "
                f"retrying in {backoff_s}s (attempt {attempt + 2}/{max_retries + 1})",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff_s)
    return last_collected


# Substrings/patterns that indicate the LLM compared two states. Loose
# enough to catch most legit comparisons; strict enough to flag pure
# descriptions like "you mostly work at night."
_COMPARATIVE_MARKERS = (
    " vs ", " vs.", " versus ",
    " compared to ", " compared with ", " relative to ",
    " more than ", " less than ",
    " whereas ", " whereas,",
    " in contrast", " on the other hand",
    " in sessions where", " in sessions with",
    " when you ", " when the ",
    " before ", " after ",
    " ratio ",
    " instead of ", " rather than ",
    " unique to ", " specific to ",
    " differs from ", " differ from ",
    " same pattern doesn't", " same pattern does not",
    "doesn't show", "don't show", "does not show", "do not show",
    " was added", " were added", " was applied", " were applied",
    " was enforced", " were enforced", " was enabled", " were enabled",
    " accepted in ", " rejected in ",       # "...accepted in second pass..."
    " second pass ", " first pass ",        # iteration-comparison framing
    # Implicit comparison framing — "X without Y" implies "vs X with Y";
    # "X independently re-reads" implies "vs X reusing parent's work".
    " without receiving ", " without the ", " without a ",
    " without explicit ", " without context", " without falsif",
    " without verifying ", " without checking",
    " independently re-read", " independently re-discover", " independently re-explore",
    " missing from", " absent from",
    "x vs", "× vs",
    "× more", "x more",
    "× fewer", "x fewer",
    "× faster", "x faster", "× slower", "x slower",
)
_COMPARATIVE_REGEX = re.compile(
    # "N out of M", "N/M"
    r"\b\d+\s*(out of|/)\s*\d+\b|"
    # "X% vs Y%", "X% compared"
    r"\b\d+(\.\d+)?\s*%\s*(vs|versus|compared)|"
    # "from N to M", "N → M"
    r"\bfrom\s+\d+(\.\d+)?\s+to\s+\d+(\.\d+)?\b|\d+\s*→\s*\d+|"
    # comparatives with arbitrary words between: "higher [...] than", "lower [...] than"
    r"\b(higher|lower|faster|slower|larger|smaller|more|fewer|better|worse)\b[^.\n]{0,80}\bthan\b|"
    # "Nx [adj]", "N× [adj]" — implicit comparative magnitudes
    r"\b\d+(\.\d+)?\s*[x×]\s*(more|fewer|faster|slower|higher|lower)\b|"
    # "when X was added/applied/used"
    r"\bwhen\b\s+\w+\s+(was|were)\s+(added|applied|used|enforced|enabled)",
    re.IGNORECASE,
)


def _has_comparative_grounding(text: str) -> bool:
    """Heuristic: does this pattern text actually compare two states?

    True if any comparative phrase or numerical-comparison pattern fires.
    Conservative — false negatives are OK (a bit of noise demotes), false
    positives are worse (would let descriptions through).
    """
    if not text:
        return False
    # Pad with a leading space so markers like " in sessions where" still
    # match when they appear at position 0.
    t = " " + text.lower()
    if any(m in t for m in _COMPARATIVE_MARKERS):
        return True
    if _COMPARATIVE_REGEX.search(text):
        return True
    return False


def _trim_headline(headline: str, max_chars: int = 200) -> str:
    """Enforce the headline constraint deterministically. The prompt asks
    for ≤120 chars + no semicolons, but Sonnet sometimes ignores both.
    Trim at the first sentence boundary (or first semicolon) when the
    headline runs long, falling back to a hard char cut as a last resort.
    """
    if not headline:
        return headline
    h = headline.strip()
    if len(h) <= max_chars and ";" not in h:
        return h
    # Cut at first sentence-ending punctuation followed by whitespace, or
    # at first semicolon (LLMs love using semicolons to staple two
    # thoughts together; we want only the first thought).
    sentence_end = re.search(r"[.;!?](?:\s|$)", h)
    if sentence_end:
        return h[: sentence_end.start() + 1].strip()
    # Hard fallback — truncate at last space before max_chars
    if len(h) > max_chars:
        cut = h[:max_chars].rsplit(" ", 1)[0]
        return cut + "…"
    return h


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

    # Field-name normalization — the model frequently uses sensible
    # synonyms ('description' for 'claim', 'fix' for 'next_action',
    # 'intervention' for 'experiment_to_try'). Map them back to the
    # canonical schema names so downstream renderers don't need to know.
    OBS_SYNONYMS = {
        "description": "claim",
        "summary": "claim",
        "why": "interpretation",
        "explanation": "interpretation",
        "fix": "next_action",
        "action": "next_action",
        "recommendation": "next_action",
        "type": "category",
    }
    BP_SYNONYMS = {
        "description": "pattern",
        "claim": "pattern",
        "summary": "pattern",
        "why": "interpretation",
        "explanation": "interpretation",
        "intervention": "experiment_to_try",
        "experiment": "experiment_to_try",
        "recommendation": "experiment_to_try",
        "pattern_type": "dimension",
        "type": "dimension",
    }

    def _normalize_keys(item: dict, synonyms: dict[str, str]) -> dict:
        for src, dest in synonyms.items():
            if src in item and dest not in item:
                item[dest] = item.pop(src)
        return item

    obs_clean: list[dict] = []
    for obs in parsed.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        obs = _normalize_keys(obs, OBS_SYNONYMS)
        kept_refs, resolved, dropped = _resolve(obs.get("evidence_refs") or [])
        dropped_total += dropped
        obs["evidence_refs"] = kept_refs
        obs["evidence_sessions"] = resolved
        obs["supporting_count"] = len(kept_refs)
        if kept_refs:
            obs_clean.append(obs)
    parsed["observations"] = obs_clean

    bp_clean: list[dict] = []
    bp_demoted = 0
    for bp in parsed.get("behavioral_patterns") or []:
        if not isinstance(bp, dict):
            continue
        bp = _normalize_keys(bp, BP_SYNONYMS)
        kept_refs, resolved, dropped = _resolve(bp.get("evidence_refs") or [])
        dropped_total += dropped
        bp["evidence_refs"] = kept_refs
        bp["evidence_sessions"] = resolved
        bp["supporting_count"] = len(kept_refs)
        if not kept_refs:
            continue

        # Comparative-grounding check: a behavioral pattern is only valuable
        # if it contrasts two states. The persona-review caught one weakness
        # ('night-heavy work correlates with fragmentation') that was just
        # a description because it had no afternoon control. Demote to low
        # confidence + flag when the pattern field shows no comparison.
        if not _has_comparative_grounding(bp.get("pattern") or ""):
            if bp.get("confidence") in ("high", "medium"):
                bp_demoted += 1
            bp["confidence"] = "low"
            bp["non_comparative"] = True
        bp_clean.append(bp)
    parsed["behavioral_patterns"] = bp_clean
    # Always surface the demotion count, even when zero — makes the heuristic
    # claim auditable on every run instead of "trust me, nothing got demoted."
    meta = parsed.setdefault("meta", {})
    meta["behavioral_patterns_demoted_non_comparative"] = bp_demoted
    meta["behavioral_patterns_total"] = len(bp_clean)

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

    # skill_candidates: same ref-validator pass; drop fabricated refs and
    # any candidate left with zero supporting sessions. kind must be one
    # of the two allowed values; default to new_skill if missing.
    sc_clean: list[dict] = []
    for sc in parsed.get("skill_candidates") or []:
        if not isinstance(sc, dict):
            continue
        kept_refs, resolved, dropped = _resolve(sc.get("evidence_refs") or [])
        dropped_total += dropped
        sc["evidence_refs"] = kept_refs
        sc["evidence_sessions"] = resolved
        sc["affected_sessions"] = len(kept_refs)
        kind = (sc.get("kind") or "new_skill").lower()
        sc["kind"] = "deepen_existing" if kind == "deepen_existing" else "new_skill"
        if sc["kind"] != "deepen_existing":
            sc["existing_skill_hint"] = None
        if kept_refs:
            sc_clean.append(sc)
    parsed["skill_candidates"] = sc_clean

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

    # Post-process headline: trim long / semicolon-stapled headlines into
    # one declarative sentence. Persona-review caught one that was 78
    # words across two semicolons.
    if parsed.get("headline"):
        original = parsed["headline"]
        trimmed = _trim_headline(original, max_chars=200)
        if trimmed != original:
            parsed["headline"] = trimmed
            meta["headline_trimmed"] = True

    # Overlap detection: if `if_you_do_one_thing_this_week` shares >40%
    # of its tokens with the headline, the LLM is restating instead of
    # extending. Flag for the dashboard to render with a warning.
    h_text = (parsed.get("headline") or "").lower()
    one_thing = (parsed.get("if_you_do_one_thing_this_week") or "").lower()
    if h_text and one_thing:
        h_tokens = set(re.findall(r"\b\w{4,}\b", h_text))  # words ≥4 chars
        ot_tokens = set(re.findall(r"\b\w{4,}\b", one_thing))
        if h_tokens and ot_tokens:
            overlap = len(h_tokens & ot_tokens) / len(h_tokens | ot_tokens)
            if overlap > 0.4:
                meta["headline_one_thing_overlap"] = round(overlap, 2)

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
    backend: LLMBackend | None = None,
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
            "skill_candidates": [],
            "per_project": [],
            "meta": {"notes": "No narratives matched filter.", "input_sessions": 0},
        }

    prompt, ref_to_id = build_synthesis_prompt(narratives, prior_context=prior_context)

    # Sonnet's input cap is ~200K tokens (~800K chars at the conservative
    # 4 chars/token heuristic). Anything close to that risks truncation
    # mid-prompt, which produces silently-degraded output. Hard-fail
    # instead so the user sees the cause and can drop --lookback-days.
    # Was 700K — too conservative. Sonnet 4.6 handles ~200K input tokens
    # natively (~800K chars), with 1M context available via long-context mode.
    # 900K leaves headroom for output without forcing the prompt to truncate.
    PROMPT_CHAR_LIMIT = 900_000
    if len(prompt) > PROMPT_CHAR_LIMIT:
        raise ValueError(
            f"synthesis prompt is {len(prompt):,} chars (~{len(prompt)//4:,} tokens), "
            f"exceeding the {PROMPT_CHAR_LIMIT:,}-char safety limit. "
            f"Pass --lookback-days N or --limit N to reduce the input set."
        )

    raw = asyncio.run(_call_llm(backend or get_backend(), prompt, model))
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
