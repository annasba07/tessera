"""LLM-driven narrative extraction for one session.

Given the deterministic metadata + a compressed event stream, asks Sonnet
to produce the LLM block of the v1 schema. Returns the raw parsed dict;
validation lives in ``validator.py`` so this module stays focused on
prompt + parse.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..backends import LLMBackend, get_backend


DEFAULT_MODEL = "claude-sonnet-4-6"


PROMPT_TEMPLATE = """You are analyzing ONE AI coding agent session. Read the event stream below and return a structured narrative about what happened.

Your job is to understand the *story* — what the user was trying to do, how the agent approached it, where it got stuck, and what would have made this session faster or more successful.

## Event stream format

Each line is one event in chronological order:
    [seq offset (kind)] action

- `seq` is the 0-based event index (cite this in `event_range`/`event_index` fields)
- `offset` is time since first event (s/m/h/d)
- `kind` is `s` for subagent events, omitted for top-level
- Actions:
    U: <text>     — user message
    A: <text>     — assistant message
    T: <tool>(<input preview>)   — tool call
    R: <tool> ok | ERR:<class> [<preview>]   — tool result
    !: <text>     — assistant reasoning

You may also see lines like `... [N successful tool results collapsed: seq A–B] ...` — treat the seq range A–B as a successful tool run with no friction.

## Session metadata (deterministic facts — trust these)

{metadata_block}

## Event stream

{stream}

## What to produce

Return ONE JSON object. No markdown fence. No preamble. The shape is:

```
{
  "goal": "<≤200 chars: one-line overarching goal of this session>",
  "topics": ["<3-7 short free-form tags>"],
  "external_systems_touched": ["<from controlled vocab>"],
  "verification_completeness": "<from controlled vocab>",
  "prompt_quality_signal": "<from controlled vocab>",

  "tasks": [
    {
      "task_id": "t1",
      "intent": "<≤200 chars: specific>",
      "event_range": [start_seq, end_seq],
      "task_type": "<from controlled vocab>",
      "task_difficulty": {
        "scope": "<single_file|few_files|cross_module|system_wide>",
        "specification": "<clear|partial|exploratory>",
        "external_dependencies": "<none|one|multiple>",
        "verification_ease": "<automated|partial|requires_human>",
        "overall": "<trivial|moderate|hard|very_hard>"
      },
      "outcome": "<from controlled vocab>",
      "outcome_evidence_event": <seq>,
      "time_to_first_verified_progress": <seq or -1>
    }
  ],

  "friction_moments": [
    {
      "event_range": [start_seq, end_seq],
      "task_id": "<one of the task_ids above>",
      "type": "<from controlled vocab>",
      "tool_category": "<browser|shell|read|write|delegation|web|other>",
      "description": "<≤200 chars>",
      "key_quote": "<verbatim short quote from one event in event_range>",
      "cost_events": <int = end_seq - start_seq + 1>,
      "cost_active_minutes": <float, your best estimate from offset deltas>
    }
  ],
  "waste_signature": "<dominant friction type, or 'none' if session was clean>",

  "key_decisions": [
    {
      "event_index": <seq>,
      "decision": "<what was chosen>",
      "retrospective": "<hindsight evaluation grounded in what happened later>"
    }
  ],

  "dead_ends": [
    {
      "event_range": [start, end],
      "approach": "<what was tried>",
      "abandoned_at": <seq>,
      "lesson": "<what was learned>"
    }
  ],

  "recurring_environmental_issues": [
    {
      "description": "<root cause + the one-time fix>",
      "occurrences": [<seq>, <seq>, ...]
    }
  ],

  "user_caught_model_errors": {
    "count": <int>,
    "examples": [
      {"event_index": <seq>, "what_user_caught": "<≤200 chars>"}
    ]
  },

  "counterfactual": "<30-300 chars: ONE specific thing the user could have done differently — must reference specific events. NOT generic advice.>",
  "lesson_for_agent": "<≤200 chars: what Claude should do on similar future tasks>",
  "lesson_for_user": "<≤200 chars: ONE concrete setup change. Not 'be more careful'. Use empty string if nothing concrete.>",
  "notable": "<optional, ≤250 chars: surprising observation that doesn't fit elsewhere. Empty string if nothing.>"
}
```

## Controlled vocabularies (use EXACTLY these strings)

- `task_type`: feature_implementation | bug_fix | refactor | exploration | code_review | debugging | planning | infrastructure | data_analysis | content_writing | research | other
- `outcome`: completed | partially_completed | abandoned | stuck_at_end | user_interrupted | blocked | unclear
- `friction_moments[].type` AND `waste_signature`: blind_retry | browser_spiral | permission_wall | wrong_abstraction | exploration_drift | over_delegation | verify_gap | tool_error_loop | model_misfire | context_loss | scope_creep | none
- `tool_category`: browser | shell | read | write | delegation | web | other
- `external_systems_touched`: github | supabase | linear | browser | gcp | aws | slack | figma | openai_api | local_only
- `verification_completeness`: none | claimed_only | lightly_verified | thoroughly_verified
- `prompt_quality_signal`: underspecified | well_formed | overspecified

## task_difficulty rubric (anchored examples)

- **trivial**: 1 file, clear spec, no external deps, automated verification (rename, format, fix typo)
- **moderate**: 1-3 files, clear spec, some context reading, ~10-30min for a competent dev (add CLI flag, update test, fix known bug)
- **hard**: ≥4 files or cross-module, ambiguous spec, may involve external systems, 1-3 hours (implement feature backend+frontend, debug intermittent failure, migrate schema)
- **very_hard**: large scope, exploratory spec, deep external dependencies, often spans sessions (new module from scratch, performance regression of unknown cause, architectural refactor)

Score the 4 dimensions independently, then assign `overall` consistent with them.

## STRICT RULES

- Every cited `seq` must be a valid event index in [0, {event_count}).
- Every `key_quote` must be a verbatim substring of an event message in its `event_range`.
- `friction_moments[].cost_events` MUST equal `event_range[1] - event_range[0] + 1`.
- `friction_moments[].task_id` MUST reference one of the `task_id`s you defined.
- Caps: tasks ≤ 5, friction_moments ≤ 6, key_decisions ≤ 5, dead_ends ≤ 3, recurring_environmental_issues ≤ 3, user_caught_model_errors.examples ≤ 3.
- `counterfactual` must be 30-300 chars and reference specific events.
- If the session is clean and productive, return `waste_signature: "none"` and empty `friction_moments`, `key_decisions`, `dead_ends`, `recurring_environmental_issues`. DO NOT invent friction.
- Cover the full event range with `tasks[]` — task event_ranges should be non-overlapping and span [0, last_seq] (with possible gaps for idle stretches if needed).

Return ONE JSON object. No markdown, no preamble.
"""


def _format_metadata_block(metadata: dict) -> str:
    """Render deterministic metadata as a clean, scannable block for the LLM."""
    fields = [
        ("session_id", metadata.get("session_id")),
        ("agent", metadata.get("agent")),
        ("project_label", metadata.get("project_label")),
        ("project_path", metadata.get("project_path")),
        ("git_branch_last", metadata.get("git_branch_last")),
        ("primary_model", metadata.get("primary_model")),
        ("models_used", ", ".join(metadata.get("models_used") or [])),
        ("started_at", metadata.get("started_at")),
        ("ended_at", metadata.get("ended_at")),
        ("event_count", metadata.get("event_count")),
        ("tool_call_count", metadata.get("tool_call_count")),
        ("user_turn_count", metadata.get("user_turn_count")),
        ("subagent_count", metadata.get("subagent_count")),
        ("active_minutes", metadata.get("active_minutes")),
        ("primary_burst_minutes", metadata.get("primary_burst_minutes")),
        ("bursts", metadata.get("bursts")),
        ("longest_idle_minutes", metadata.get("longest_idle_minutes")),
        ("tool_error_rate", metadata.get("tool_error_rate")),
        ("tool_error_classes", metadata.get("tool_error_classes")),
        ("tool_category_distribution", metadata.get("tool_category_distribution")),
        ("most_repeated_tool_call", metadata.get("most_repeated_tool_call")),
        ("unique_files_touched", metadata.get("unique_files_touched")),
        ("directory_concentration", metadata.get("directory_concentration")),
        ("tests_invoked", metadata.get("tests_invoked")),
        ("git_status_invoked", metadata.get("git_status_invoked")),
        ("git_diff_invoked", metadata.get("git_diff_invoked")),
        ("pr_referenced", metadata.get("pr_referenced")),
        ("user_friction_signals", metadata.get("user_friction_signals")),
        ("last_event_type", metadata.get("last_event_type")),
        ("final_tool_status", metadata.get("final_tool_status")),
        ("error_concentration_tail", metadata.get("error_concentration_tail")),
        ("events_since_last_user_message", metadata.get("events_since_last_user_message")),
    ]
    lines = []
    for name, value in fields:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"- {name}: {value}")
    return "\n".join(lines)


def build_prompt(metadata: dict, stream: str) -> str:
    """Assemble the full per-session prompt."""
    metadata_block = _format_metadata_block(metadata)
    event_count = metadata.get("event_count") or 0
    prompt = PROMPT_TEMPLATE
    for token, value in (
        ("{metadata_block}", metadata_block),
        ("{stream}", stream),
        ("{event_count}", str(event_count)),
    ):
        prompt = prompt.replace(token, value)
    return prompt


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


_TRANSIENT_FAILURE_SIGNALS = (
    "Request timed out", "Connection reset", "Connection error",
    "Rate limit", "rate_limit_error", "overloaded_error",
    "Service Unavailable", "503 Service",
)


async def _call_llm(
    backend: LLMBackend,
    prompt: str,
    model: str,
    *,
    max_retries: int = 2,
) -> str:
    """Backend-agnostic retry on transient failures. The transient signals
    list was tuned for the Claude SDK but the same strings appear in
    Codex/Gemini CLI stderr — keep one list, retry uniformly."""
    import asyncio, sys
    last_collected = ""
    for attempt in range(max_retries + 1):
        try:
            collected = await backend.complete(prompt, model)
        except RuntimeError as exc:
            # CLI backend non-zero exit. Treat as a transient signal and retry.
            collected = str(exc)
        last_collected = collected
        stripped = collected.strip()
        is_short_failure = (
            len(stripped) < 200
            and any(sig.lower() in stripped.lower() for sig in _TRANSIENT_FAILURE_SIGNALS)
        )
        if not is_short_failure:
            return collected
        if attempt < max_retries:
            backoff_s = 3 * (1 + 2 * attempt)
            print(
                f"  → per-session LLM call returned transient failure ({stripped[:80]!r}) "
                f"— retrying in {backoff_s}s",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff_s)
    return last_collected


async def extract_narrative(
    metadata: dict,
    stream: str,
    model: str = DEFAULT_MODEL,
    backend: LLMBackend | None = None,
) -> dict[str, Any]:
    """Run the LLM and return the parsed narrative dict (pre-validation).

    Args:
        backend: which LLM backend to call. None → uses
            ``backends.get_backend()`` which respects $TESSERA_BACKEND.

    Raises:
        json.JSONDecodeError if the model returns non-JSON.
    """
    prompt = build_prompt(metadata, stream)
    raw = await _call_llm(backend or get_backend(), prompt, model)
    return _extract_json(raw)
