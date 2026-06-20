"""Multi-lens behavioral synthesis.

Why this exists: the single-pass synthesis prompt asks for 8-15 behavioral
patterns across 9 dimensions. Even with sharpened metric requirements,
asking one LLM call to be excellent at prompting + delegation + verification
+ intervention + recovery + cognition simultaneously dilutes depth.

This module runs ONE focused synthesis call per "lens" (dimension), each
with a prompt that says "look only at THIS dimension, find 3-5 quantified
comparative patterns, every experiment must be measurable." Then a merge
pass deduplicates overlapping patterns across lenses and returns the
top-N ranked by confidence × supporting refs.

Cost: ~6-8 lens calls + 1 merge call. On Flash 3.5 High that's ~$0.25.
Wall clock with parallel lens dispatch: ~5 min (vs 3 min single-pass).
Worth it for the depth on weekly review; not worth it for ad-hoc runs.

Lens-to-backend mapping (from the 4-backend bake-off):
- Codex was sharpest on COMPARATIVE BEHAVIORAL framing → delegation,
  intervention
- Claude was best on META-PATTERN detection + quantified observations →
  verification, error-recovery
- Flash High had highest consistency + lowest cost → prompting,
  tool-selection, cognitive-arc, merge

To start we'll use one backend for all lenses (the user's configured
default). The per-lens backend dispatch is a follow-up — needs care
because the bake-off was N=1 per backend per lens.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from .backends import LLMBackend, get_backend
from .narratives.synthesis import (
    _build_aggregate_block,
    _build_prior_context_block,
    _build_ref_map,
    _build_sessions_block,
    _extract_json,
    _validate,
)


@dataclass(frozen=True)
class Lens:
    """A single behavioral dimension to investigate in isolation.

    The `focus` string is injected into the lens-specific synthesis
    prompt. The `signals` string lists narrative fields the model
    should ground its claims in for this dimension."""
    key: str
    name: str
    focus: str
    signals: str


# Ordered roughly by signal-density: prompting + delegation + verification
# tend to have the most observable evidence in a typical week's narratives.
LENSES: tuple[Lens, ...] = (
    Lens(
        key="prompting",
        name="Prompting style",
        focus=(
            "How the user phrases prompts to coding agents. Compare openers "
            "(goal-first vs try-first), constraint specification (early vs "
            "late), vocabulary precision (file paths and tool names vs vague "
            "references), and examples-in-prompt vs not. The question to "
            "answer: 'Which phrasing patterns shorten paths and reduce "
            "dead-ends, and by how much?'"
        ),
        signals=(
            "user message content (extracted from narrative tasks), "
            "prompt_quality_signal, lesson_for_user mentioning prompt "
            "phrasing, first-event vs later-event prompts"
        ),
    ),
    Lens(
        key="delegation",
        name="Delegation & subagents",
        focus=(
            "When does the user delegate to subagents vs work inline? Compare "
            "tasks where delegation paid off (lower event count for "
            "comparable work) vs tasks where it backfired (high subagent "
            "count + high dead-ends). Also: WHY zero-subagent on multi-source "
            "work that could parallelize."
        ),
        signals=(
            "subagent_count, subagent_event_count, subagent_types_spawned, "
            "event_count, active_minutes, task type"
        ),
    ),
    Lens(
        key="verification",
        name="Verification discipline",
        focus=(
            "What does the user verify vs trust blindly? Compare verification "
            "intensity across data sources (Slack vs Gmail vs Drive), task "
            "types (info retrieval vs content generation), and outcomes "
            "(verified-and-correct vs verified-and-caught-error vs "
            "unverified-and-broken). The question: 'Which task types are at "
            "highest risk of silent failure due to under-verification?'"
        ),
        signals=(
            "verification_completeness, user_caught_model_errors, "
            "user_friction_signals, tool_error_rate"
        ),
    ),
    Lens(
        key="intervention_timing",
        name="Intervention timing",
        focus=(
            "When does the user step in to correct the agent vs let it run? "
            "Compare early interventions (corrections in first 20% of events) "
            "vs late ones (last 20%) vs none. Tie to outcome quality. Look "
            "for the 'mega-session pull-back' pattern where delegation leads "
            "to a tight loop of manual corrections."
        ),
        signals=(
            "user_friction_signals, user_caught_model_errors, "
            "user_turn_count, event_count, ratio of user-events to "
            "tool-events"
        ),
    ),
    Lens(
        key="error_recovery",
        name="Error recovery patterns",
        focus=(
            "After a dead-end or tool failure, what does the user/agent do? "
            "Compare 'pivot to alternative source' vs 'retry same approach' "
            "vs 'abort'. Quantify the event overhead of each strategy. Look "
            "for permission-wall and credential-failure cascades."
        ),
        signals=(
            "dead_ends, recurring_environmental_issues, "
            "tool_error_classes, counterfactual"
        ),
    ),
    Lens(
        key="tool_selection",
        name="Tool & model selection",
        focus=(
            "Which tool or model is used for which task? Compare ToolSearch "
            "vs direct invocation (event overhead), Claude vs Codex vs Gemini "
            "vs Antigravity on similar task types, MCP combos that work vs "
            "backfire. Identify 'wrong tool for the job' patterns."
        ),
        signals=(
            "tool_call_distribution, tool_category_distribution, "
            "models_used, primary_model, external_systems_touched"
        ),
    ),
    Lens(
        key="cognitive_arc",
        name="Cognitive arc & timing",
        focus=(
            "How does the user's effectiveness vary across time-of-day, "
            "weekday, session length, and burst pattern? Compare focused "
            "single-burst sessions vs fragmented ones. ONLY surface patterns "
            "that map to an actionable change (schedule shift, session-length "
            "limit, mid-session break). Skip pure correlations."
        ),
        signals=(
            "time_of_day_buckets, weekday, bursts, primary_burst_minutes, "
            "longest_idle_minutes, active_minutes, events_per_active_minute"
        ),
    ),
)


def _build_lens_prompt(
    lens: Lens,
    narratives: list[dict],
    *,
    aggregate_block: str,
    sessions_block: str,
    prior_context: str | None,
) -> str:
    """Build a focused synthesis prompt for a single lens.

    The prompt is much shorter than the main synthesis prompt because
    it only asks for behavioral_patterns within one dimension. No
    operational observations, no quick wins, no skill candidates —
    those come from the standard synthesis pass."""
    prior_block = _build_prior_context_block(prior_context)
    n_sessions = len(narratives)
    n_pad = max(3, len(str(n_sessions)))
    return f"""## CRITICAL: This is an analysis task with a strict response format. Do NOT use tools. Do NOT spawn scripts. Do NOT search the filesystem for the data — the complete narrative set is already in this prompt below. Read the data from this prompt, then output ONE JSON object. If you find yourself wanting to write a Python script, you have misread the task — re-read this paragraph.

You are reading {n_sessions} narrative summaries of one person's AI coding agent sessions, with ONE focused goal: find 3-5 strong behavioral patterns in ONE dimension only.

## Your dimension: {lens.name}

{lens.focus}

## Signals to read for this dimension

{lens.signals}

Ignore patterns outside this dimension. If you notice a great pattern about, say, verification while you're supposed to be looking at delegation, NOTE IT IN `meta.cross_dimension_observations` but do not put it in the patterns list. A focused list of 3-5 is the deliverable.

## Quality bar (mandatory)

Every pattern must satisfy ALL of these:

1. **Two numbers on both sides of a comparison.** "Pattern A (47 sessions, median X=2.3) vs Pattern B (34 sessions, median X=0.6) — A is 3.8× more efficient." Patterns without quantification are descriptive, not insights.

2. **A measurable experiment.** Not "try X next week." Specify the exact metric to compare: "On Mon-Wed do X, log metric M; on Thu-Fri do Y, compare M in next dashboard." If you can't name M, the experiment doesn't belong.

3. **Maps to an agent-USE lever.** prompting / delegation / verification / intervention / recovery / tool-selection / cognitive-arc. If it maps to "life advice" (sleep more, work less) instead of "agent use" (prompt this way, delegate this kind of task), drop it.

4. **Cited evidence.** Every pattern cites 3-8 refs from the input below. Refs are `S001`-`S{str(n_sessions).zfill(n_pad)}` verbatim.

{prior_block}

## Aggregate stats (already computed — trust, don't recount)

{aggregate_block}

## Sessions

{sessions_block}

## Output format — ONE JSON object, no preamble, no fence

{{
  "lens": "{lens.key}",
  "behavioral_patterns": [
    {{
      "title": "<short noun-phrase>",
      "pattern": "<the comparative quantified claim — must include 2 numbers on the comparison>",
      "evidence_refs": ["S001", "S004", ...],
      "interpretation": "<2-3 sentences: why this pattern exists, what underlying habit drives it, what the cost is>",
      "experiment_to_try": "<measurable experiment: 'On Mon-Wed do X, log metric M; on Thu-Fri do Y, compare M.'>",
      "confidence": "high | medium | low",
      "dimension": "{lens.key}"
    }}
  ],
  "meta": {{
    "cross_dimension_observations": "<optional: patterns you noticed outside {lens.key} that future runs should investigate>",
    "notes": "<caveats specific to this lens>"
  }}
}}

Confidence:
- **high** = ≥5 supporting refs, two-sided quantified comparison, replicable
- **medium** = 3-4 refs OR 5+ with noise
- **low** = hunch worth flagging, weak evidence (still valuable)

Aim for 3-5 patterns. Better to return 3 sharp ones than 8 padded.

Final reminder: cite refs (`S001`-`S{str(n_sessions).zfill(n_pad)}`) verbatim. Every pattern needs two numbers on the comparison AND a measurable experiment, or it doesn't ship."""


async def _run_lens(
    lens: Lens,
    narratives: list[dict],
    backend: LLMBackend,
    *,
    model: str,
    aggregate_block: str,
    sessions_block: str,
    prior_context: str | None,
) -> dict:
    """Run one lens-specific synthesis call. Returns the parsed JSON
    (or an empty result on failure — multi-lens never fails entirely
    because one bad lens shouldn't kill the run)."""
    prompt = _build_lens_prompt(
        lens, narratives,
        aggregate_block=aggregate_block,
        sessions_block=sessions_block,
        prior_context=prior_context,
    )
    try:
        raw = await backend.complete(prompt, model)
        parsed = _extract_json(raw)
        parsed["_lens"] = lens.key
        return parsed
    except Exception as exc:
        return {
            "lens": lens.key,
            "_lens": lens.key,
            "behavioral_patterns": [],
            "_error": f"{type(exc).__name__}: {exc}",
        }


def _build_merge_prompt(
    lens_results: list[dict],
    *,
    target_count: int,
) -> str:
    """Build the merge prompt that dedupes and ranks across lenses."""
    blocks: list[str] = []
    for r in lens_results:
        lens_key = r.get("_lens") or r.get("lens") or "?"
        bps = r.get("behavioral_patterns") or []
        if not bps:
            continue
        blocks.append(f"### Lens: {lens_key} ({len(bps)} patterns)\n")
        for i, bp in enumerate(bps, 1):
            title = bp.get("title", "(untitled)")
            pattern = bp.get("pattern", "")
            conf = bp.get("confidence", "?")
            refs = ", ".join(bp.get("evidence_refs") or [])
            blocks.append(f"- {lens_key}-{i} [{conf}] {title}")
            blocks.append(f"    pattern: {pattern}")
            if refs:
                blocks.append(f"    refs: {refs}")
            blocks.append("")
    candidate_block = "\n".join(blocks)
    return f"""You are merging {len(lens_results)} parallel single-lens analyses into ONE ranked list of {target_count} behavioral patterns about how this person uses AI coding agents.

## Merge rules

1. **Dedupe.** Two patterns are duplicates if they make the same comparative claim about the same behavioral lever, even if the lens labels differ. Pick the version with MORE specific numbers or MORE supporting refs and drop the other. List the dropped one's lens-id in `meta.deduped_pairs`.

2. **Keep ALL high-confidence non-duplicate patterns.** Even if total exceeds {target_count}, surface every high-confidence one — the user can trim. Then fill remaining slots with medium → low.

3. **Don't write new patterns.** Only re-rank and re-filter what the lenses produced. If you think a pattern is missing, note it in `meta.notes` rather than fabricating one.

4. **Preserve the source.** Each kept pattern gets a `source_lens` field set to the lens-id it came from (e.g. `prompting-2`).

## Candidate patterns from each lens

{candidate_block}

## Output — ONE JSON object, no preamble

{{
  "behavioral_patterns": [
    {{
      "source_lens": "prompting-2",
      "title": "<from the lens result>",
      "pattern": "<from the lens result>",
      "evidence_refs": ["S001", ...],
      "interpretation": "<from the lens result, preserved>",
      "experiment_to_try": "<from the lens result, preserved>",
      "confidence": "high | medium | low",
      "dimension": "<the lens key>"
    }}
  ],
  "meta": {{
    "lens_counts": {{"prompting": 4, "delegation": 3, ...}},
    "deduped_pairs": [["prompting-1", "delegation-2 (kept)"]],
    "notes": "<caveats from the merge>"
  }}
}}

Target: {target_count} patterns. Quality > count — if only 8 unique strong patterns exist across all lenses, return 8."""


async def _run_merge(
    lens_results: list[dict],
    backend: LLMBackend,
    *,
    model: str,
    target_count: int = 12,
) -> dict:
    """Run the merge synthesis call. On failure, return a naive
    concatenation so multi-lens still produces something usable."""
    prompt = _build_merge_prompt(lens_results, target_count=target_count)
    try:
        raw = await backend.complete(prompt, model)
        return _extract_json(raw)
    except Exception as exc:
        # Fallback: flatten everything and tag with source_lens, no ranking.
        flat: list[dict] = []
        for r in lens_results:
            lens_key = r.get("_lens") or r.get("lens") or "?"
            for i, bp in enumerate(r.get("behavioral_patterns") or [], 1):
                merged = {**bp, "source_lens": f"{lens_key}-{i}", "dimension": lens_key}
                flat.append(merged)
        return {
            "behavioral_patterns": flat,
            "meta": {
                "notes": f"merge failed ({type(exc).__name__}: {exc}); "
                         f"returning naive concatenation",
            },
        }


async def multilens_synthesize_async(
    narratives: list[dict],
    *,
    backend: LLMBackend | None = None,
    model: str = "",
    prior_context: str | None = None,
    lenses: tuple[Lens, ...] = LENSES,
    target_count: int = 12,
    parallelism: int = 1,
) -> dict[str, Any]:
    """Async entry. Runs each lens with bounded parallelism, then merges.

    Parallelism default = 1 (sequential) after observing that parallel
    agy subprocesses don't just share auth — they each independently
    decide to spawn data-analysis scripts when the lens prompt feels
    'investigative.' Sequential calls give the model less excuse to
    treat the task as agentic + share token state cleanly. Wall-clock
    impact: 7 × ~2 min = ~14 min instead of ~5 min for the lens phase.
    Worth the reliability."""
    backend = backend or get_backend()
    effective_model = model or backend.default_model

    ref_to_id, id_to_ref = _build_ref_map(narratives)
    aggregate_block = _build_aggregate_block(narratives)
    sessions_block = _build_sessions_block(narratives, id_to_ref)

    sem = asyncio.Semaphore(parallelism)

    async def _bounded(lens: Lens) -> dict:
        async with sem:
            return await _run_lens(
                lens, narratives, backend,
                model=effective_model,
                aggregate_block=aggregate_block,
                sessions_block=sessions_block,
                prior_context=prior_context,
            )

    lens_results = await asyncio.gather(*(_bounded(l) for l in lenses))

    merged = await _run_merge(
        lens_results, backend, model=effective_model, target_count=target_count,
    )

    # Validate the merged patterns: drop fabricated refs same way the
    # single-pass synthesis does. _validate also enforces other field
    # cleanup (severity, evidence_sessions resolution, etc.).
    cleaned = _validate(merged, ref_to_id)

    cleaned.setdefault("meta", {})
    cleaned["meta"]["multilens"] = {
        "lenses_run": [l.key for l in lenses],
        "per_lens_bp_counts": {
            l.key: len((lens_results[i].get("behavioral_patterns") or []))
            for i, l in enumerate(lenses)
        },
        "errors": [
            {"lens": r.get("_lens"), "error": r.get("_error")}
            for r in lens_results if r.get("_error")
        ],
    }
    return cleaned


def multilens_synthesize(
    narratives: list[dict],
    *,
    backend: LLMBackend | None = None,
    model: str = "",
    prior_context: str | None = None,
    target_count: int = 12,
) -> dict[str, Any]:
    """Synchronous public entry point. Run all lenses in parallel,
    merge, return one synthesis dict shaped like the regular synthesis
    output (so the dashboard / render code Just Works)."""
    return asyncio.run(
        multilens_synthesize_async(
            narratives,
            backend=backend,
            model=model,
            prior_context=prior_context,
            target_count=target_count,
        )
    )
