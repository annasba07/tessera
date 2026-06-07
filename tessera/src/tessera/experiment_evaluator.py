"""LLM-driven evaluator for self-experiments — the brain of the closed loop.

Given an active experiment (a behavioral pattern the user rated `useful`
last week) and the narratives that happened since the user committed to it,
ask Claude:

    1. Did the user actually try the experiment? (adherence)
    2. What's the effect on the metric the experiment specified? (effect)
    3. What's the evidence?

The output feeds `evaluate_pending()` which transitions the experiment to
`graduated` / `not_tried` / `inconclusive` accordingly.

This module is intentionally separate from experiments.py so the core
data model stays LLM-free and tests don't need to mock the SDK.

Cost: ~$0.10 per experiment per evaluation pass (one Sonnet call with a
small structured prompt). Weekly cost: a few cents per active experiment.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .backends import LLMBackend, get_backend
from .experiments import Experiment


DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_POST_NARRATIVES = 25  # cap how many post-baseline narratives we feed in
MAX_NARRATIVE_CHARS = 1200  # per-narrative compaction budget


_EVAL_PROMPT_TEMPLATE = """You are evaluating whether a user followed through on an experiment they committed to one week ago, and what the effect was.

## The experiment

Title: {title}
Dimension: {dimension}
Started: {started_at}

Experiment text (what the user committed to trying):
{experiment_text}

## Sessions since the experiment started ({n_sessions} sessions)

Each session is one entry below. Look at the user's prompts, the agent's behavior, friction moments, lessons, and outcomes.

{sessions_block}

## Your task

Decide two things, grounded in specific session evidence:

### 1. ADHERENCE — did the user actually try the experiment?

- **full**: clear evidence in 3+ sessions that the user applied the experimented-with behavior (specific prompt phrasing, deliberate intervention, verification step the experiment recommended, etc.)
- **partial**: 1-2 sessions with adherence; others not following the experiment
- **none**: no evidence in any session that the experiment was tried

### 2. EFFECT — what changed because of (or independent of) the experiment?

- **positive**: the metric the experiment proposed to move actually moved in the desired direction (fewer dead-ends, faster ship, lower verification gaps, etc.)
- **neutral**: no clear change in the metric
- **negative**: metric got worse (rare but possible — sometimes the proposed habit was wrong)

If adherence is `none`, set effect to `unknown` — you can't measure an effect from something untried.

## Output format

Return ONE JSON object. No markdown fence, no preamble:

```
{{
  "adherence": "full | partial | none",
  "effect": "positive | neutral | negative | unknown",
  "adherence_evidence": "<1-2 sentences citing specific session refs and what you saw>",
  "effect_evidence": "<1-2 sentences with the actual metric change or 'no measurable change'>",
  "recommendation": "<one sentence: graduate / keep running / drop / try variant>"
}}
```

Be honest. If you can't tell, say so. The whole point of the system is to know when advice works and when it doesn't.
"""


def _compact_narrative_for_eval(narr: dict, idx: int) -> str:
    """Pull the fields most likely to surface adherence + effect signal."""
    sid = (narr.get("session_id") or "")[:48]
    project = narr.get("project_label") or "?"
    started = (narr.get("started_at") or "")[:10]
    goal = (narr.get("goal") or "")[:200]
    lesson = (narr.get("lesson_for_user") or "")[:240]
    counter = (narr.get("counterfactual") or "")[:240]
    waste = narr.get("waste_signature") or "—"
    dead_ends = len(narr.get("dead_ends") or [])
    user_caught = (narr.get("user_caught_model_errors") or {}).get("count", 0)
    user_corr = (narr.get("user_friction_signals") or {}).get("explicit_corrections", 0)
    out_sig = (narr.get("outcome") or {}).get("outcome_signal", "—")
    friction_first = ""
    fm = narr.get("friction_moments") or []
    if fm:
        friction_first = (fm[0].get("description") or "")[:160]

    lines = [
        f"--- ref S{idx:03d} · {sid} · {started} · {project} · waste={waste} · outcome={out_sig} ---",
        f"goal: {goal}",
    ]
    if friction_first:
        lines.append(f"first_friction: {friction_first}")
    if dead_ends or user_caught or user_corr:
        lines.append(
            f"stats: dead_ends={dead_ends} user_caught={user_caught} explicit_corrections={user_corr}"
        )
    if lesson:
        lines.append(f"lesson_for_user: {lesson}")
    if counter:
        lines.append(f"counterfactual: {counter}")
    text = "\n".join(lines)
    return text[:MAX_NARRATIVE_CHARS]


def build_evaluator_prompt(exp: Experiment, post_narratives: list[dict]) -> str:
    """Compose the eval prompt for one experiment + N post-baseline narratives."""
    # Cap input size — most experiments have a clear adherence signal within
    # the first dozen relevant sessions; more just inflates cost.
    capped = post_narratives[:MAX_POST_NARRATIVES]
    blocks = [_compact_narrative_for_eval(n, i + 1) for i, n in enumerate(capped)]
    return _EVAL_PROMPT_TEMPLATE.format(
        title=exp.title,
        dimension=exp.dimension or "general",
        started_at=exp.started_at[:10],
        experiment_text=exp.experiment_text or "(no experiment text)",
        n_sessions=len(capped),
        sessions_block="\n\n".join(blocks),
    )


async def _call_llm_once(backend: LLMBackend, prompt: str, model: str) -> str:
    return await backend.complete(prompt, model)


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def evaluate_one(
    exp: Experiment,
    post_narratives: list[dict],
    model: str = DEFAULT_MODEL,
    backend: LLMBackend | None = None,
) -> dict[str, Any]:
    """Synchronous public entry: evaluate one experiment, return a dict
    suitable for evaluate_pending() to consume.

    On any failure (timeout, malformed JSON, CLI exit code), returns
    adherence/effect of 'unknown' so the experiment stays active and
    gets retried next week.
    """
    if not post_narratives:
        return {
            "method": "llm",
            "adherence": "unknown",
            "effect": "unknown",
            "adherence_evidence": "no post-baseline narratives to evaluate",
            "effect_evidence": "",
            "recommendation": "keep running — no data yet",
        }
    prompt = build_evaluator_prompt(exp, post_narratives)
    try:
        raw = asyncio.run(_call_llm_once(backend or get_backend(), prompt, model))
        parsed = _extract_json(raw)
    except Exception as exc:
        return {
            "method": "llm",
            "adherence": "unknown",
            "effect": "unknown",
            "adherence_evidence": f"evaluator error: {type(exc).__name__}: {exc}",
            "effect_evidence": "",
            "recommendation": "keep running — eval failed",
        }
    parsed.setdefault("method", "llm")
    parsed.setdefault("adherence", "unknown")
    parsed.setdefault("effect", "unknown")
    return parsed


def make_callable(model: str = DEFAULT_MODEL, backend: LLMBackend | None = None):
    """Return a callable suitable for `evaluate_pending(llm_evaluator=...)`.

    Bound to a chosen model + backend so the caller can A/B different
    evaluators (e.g. Sonnet vs Codex's gpt-5 vs Gemini 2.5) without
    modifying experiments.py.
    """
    def _wrapped(exp: Experiment, post_narratives: list[dict]) -> dict:
        return evaluate_one(exp, post_narratives, model=model, backend=backend)
    return _wrapped
