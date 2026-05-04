"""Validate the LLM-extracted narrative per the v1 schema rules.

Fails closed at the field level — drops invalid fields, keeps the rest of
the session. Computes ``narrative_quality`` based on drop rate.

See docs/schema/v1.md for the full rule list.
"""

from __future__ import annotations

import re
from typing import Any


# ---------- Controlled vocabularies -----------------------------------------

TASK_TYPES = {
    "feature_implementation", "bug_fix", "refactor", "exploration",
    "code_review", "debugging", "planning", "infrastructure",
    "data_analysis", "content_writing", "research", "other",
}
OUTCOMES = {
    "completed", "partially_completed", "abandoned", "stuck_at_end",
    "user_interrupted", "blocked", "unclear",
}
WASTE_TYPES = {
    "blind_retry", "browser_spiral", "permission_wall", "wrong_abstraction",
    "exploration_drift", "over_delegation", "verify_gap", "tool_error_loop",
    "model_misfire", "context_loss", "scope_creep", "none",
}
TOOL_CATEGORIES = {
    "browser", "shell", "read", "write", "delegation", "web", "other",
}
EXTERNAL_SYSTEMS = {
    "github", "supabase", "linear", "browser", "gcp", "aws", "slack",
    "figma", "openai_api", "local_only",
}
VERIFICATION = {"none", "claimed_only", "lightly_verified", "thoroughly_verified"}
PROMPT_QUALITY = {"underspecified", "well_formed", "overspecified"}
DIFFICULTY_SCOPE = {"single_file", "few_files", "cross_module", "system_wide"}
DIFFICULTY_SPEC = {"clear", "partial", "exploratory"}
DIFFICULTY_DEPS = {"none", "one", "multiple"}
DIFFICULTY_VERIFY = {"automated", "partial", "requires_human"}
DIFFICULTY_OVERALL = {"trivial", "moderate", "hard", "very_hard"}

# List length caps from rule 12
LIST_CAPS = {
    "tasks": 5,
    "friction_moments": 6,
    "key_decisions": 5,
    "dead_ends": 3,
    "recurring_environmental_issues": 3,
}
EXAMPLES_CAP = 3  # user_caught_model_errors.examples


# ---------- Helpers ---------------------------------------------------------


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _is_valid_index(value, event_count: int) -> bool:
    return isinstance(value, int) and 0 <= value < event_count


def _is_valid_range(value, event_count: int) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    a, b = value
    return (
        isinstance(a, int)
        and isinstance(b, int)
        and 0 <= a <= b < event_count
    )


def _enum_or(value, vocab: set[str], fallback: str) -> tuple[str, bool]:
    """Returns (cleaned_value, was_replaced)."""
    if isinstance(value, str) and value in vocab:
        return value, False
    return fallback, True


def _quote_in_events(quote: str, event_range: list[int], events: list[dict]) -> bool:
    """Substring check (normalized whitespace) against any event's text fields
    in the given range."""
    if not quote:
        return False
    needle = _normalize_ws(quote)
    if not needle:
        return False
    a, b = event_range
    for e in events[a : b + 1]:
        for key in ("message_text", "tool_input_preview", "tool_output_preview"):
            haystack = _normalize_ws(e.get(key) or "")
            if haystack and needle in haystack:
                return True
    return False


# ---------- Per-field validators --------------------------------------------


def _validate_task_difficulty(td: Any) -> tuple[dict | None, int]:
    """Returns (cleaned_difficulty, drops). None if unsalvageable."""
    if not isinstance(td, dict):
        return None, 1
    drops = 0
    cleaned = {}
    for field, vocab in (
        ("scope", DIFFICULTY_SCOPE),
        ("specification", DIFFICULTY_SPEC),
        ("external_dependencies", DIFFICULTY_DEPS),
        ("verification_ease", DIFFICULTY_VERIFY),
        ("overall", DIFFICULTY_OVERALL),
    ):
        value, replaced = _enum_or(td.get(field), vocab, "")
        if replaced:
            drops += 1
        if value:
            cleaned[field] = value
    return cleaned if cleaned else None, drops


def _validate_task(task: Any, event_count: int, llm_field_total: list) -> tuple[dict | None, int]:
    """Returns (cleaned_task, drops). None if event_range invalid (whole task dropped)."""
    if not isinstance(task, dict):
        return None, 1
    drops = 0
    cleaned: dict[str, Any] = {}

    # event_range — required
    er = task.get("event_range")
    if not _is_valid_range(er, event_count):
        return None, 1
    cleaned["event_range"] = er

    # task_id — required
    tid = task.get("task_id")
    if not isinstance(tid, str) or not tid:
        return None, 1
    cleaned["task_id"] = tid

    # intent
    intent = task.get("intent")
    if isinstance(intent, str) and intent.strip():
        cleaned["intent"] = intent.strip()[:200]
    else:
        drops += 1

    # task_type
    task_type, replaced = _enum_or(task.get("task_type"), TASK_TYPES, "other")
    if replaced:
        drops += 1
    cleaned["task_type"] = task_type
    llm_field_total.append("task_type")

    # task_difficulty
    td, td_drops = _validate_task_difficulty(task.get("task_difficulty"))
    drops += td_drops
    if td:
        cleaned["task_difficulty"] = td
    llm_field_total.append("task_difficulty")

    # outcome
    outcome, replaced = _enum_or(task.get("outcome"), OUTCOMES, "unclear")
    if replaced:
        drops += 1
    cleaned["outcome"] = outcome
    llm_field_total.append("outcome")

    # outcome_evidence_event — if invalid, downgrade outcome to unclear
    oee = task.get("outcome_evidence_event")
    if _is_valid_index(oee, event_count):
        cleaned["outcome_evidence_event"] = oee
    else:
        cleaned["outcome"] = "unclear"
        drops += 1

    # time_to_first_verified_progress
    ttfvp = task.get("time_to_first_verified_progress")
    if isinstance(ttfvp, int) and (ttfvp == -1 or _is_valid_index(ttfvp, event_count)):
        cleaned["time_to_first_verified_progress"] = ttfvp
    else:
        drops += 1

    return cleaned, drops


def _validate_friction(
    fm: Any,
    event_count: int,
    valid_task_ids: set[str],
    events: list[dict],
    llm_field_total: list,
) -> tuple[dict | None, int]:
    if not isinstance(fm, dict):
        return None, 1
    drops = 0
    cleaned: dict[str, Any] = {}

    er = fm.get("event_range")
    if not _is_valid_range(er, event_count):
        return None, 1
    cleaned["event_range"] = er

    # task_id reference must be valid; otherwise blank
    tid = fm.get("task_id")
    if isinstance(tid, str) and tid in valid_task_ids:
        cleaned["task_id"] = tid
    else:
        drops += 1

    ftype, replaced = _enum_or(fm.get("type"), WASTE_TYPES, "none")
    if replaced:
        drops += 1
    cleaned["type"] = ftype
    llm_field_total.append("friction_type")

    tcat, replaced = _enum_or(fm.get("tool_category"), TOOL_CATEGORIES, "other")
    if replaced:
        drops += 1
    cleaned["tool_category"] = tcat
    llm_field_total.append("friction_tool_category")

    desc = fm.get("description")
    if isinstance(desc, str) and desc.strip():
        cleaned["description"] = desc.strip()[:200]
    else:
        drops += 1

    # Recompute cost_events deterministically
    cleaned["cost_events"] = er[1] - er[0] + 1

    cost_min = fm.get("cost_active_minutes")
    if isinstance(cost_min, (int, float)) and cost_min >= 0:
        cleaned["cost_active_minutes"] = round(float(cost_min), 1)
    else:
        drops += 1

    # key_quote: must substring-match an event in the range
    quote = fm.get("key_quote")
    if isinstance(quote, str) and _quote_in_events(quote, er, events):
        cleaned["key_quote"] = quote.strip()[:200]
    elif isinstance(quote, str) and quote.strip():
        drops += 1
    # silently omit if missing entirely

    return cleaned, drops


def _validate_key_decision(kd: Any, event_count: int) -> tuple[dict | None, int]:
    if not isinstance(kd, dict):
        return None, 1
    ei = kd.get("event_index")
    if not _is_valid_index(ei, event_count):
        return None, 1
    cleaned = {"event_index": ei}
    drops = 0
    decision = kd.get("decision")
    if isinstance(decision, str) and decision.strip():
        cleaned["decision"] = decision.strip()[:200]
    else:
        drops += 1
    retro = kd.get("retrospective")
    if isinstance(retro, str) and retro.strip():
        cleaned["retrospective"] = retro.strip()[:300]
    else:
        drops += 1
    return cleaned, drops


def _validate_dead_end(de: Any, event_count: int) -> tuple[dict | None, int]:
    if not isinstance(de, dict):
        return None, 1
    er = de.get("event_range")
    if not _is_valid_range(er, event_count):
        return None, 1
    cleaned: dict[str, Any] = {"event_range": er}
    drops = 0
    approach = de.get("approach")
    if isinstance(approach, str) and approach.strip():
        cleaned["approach"] = approach.strip()[:200]
    else:
        drops += 1
    abandoned_at = de.get("abandoned_at")
    if _is_valid_index(abandoned_at, event_count):
        cleaned["abandoned_at"] = abandoned_at
    else:
        drops += 1
    lesson = de.get("lesson")
    if isinstance(lesson, str) and lesson.strip():
        cleaned["lesson"] = lesson.strip()[:300]
    else:
        drops += 1
    return cleaned, drops


def _validate_environmental_issue(ei: Any, event_count: int) -> tuple[dict | None, int]:
    if not isinstance(ei, dict):
        return None, 1
    occ = ei.get("occurrences")
    if not isinstance(occ, list):
        return None, 1
    valid_occ = [i for i in occ if _is_valid_index(i, event_count)]
    if len(valid_occ) < 2:
        return None, 1  # by definition needs ≥2 occurrences
    cleaned = {"occurrences": valid_occ}
    drops = 0
    desc = ei.get("description")
    if isinstance(desc, str) and desc.strip():
        cleaned["description"] = desc.strip()[:300]
    else:
        drops += 1
    return cleaned, drops


def _validate_user_caught(value: Any, event_count: int) -> tuple[dict, int]:
    cleaned: dict[str, Any] = {"count": 0, "examples": []}
    drops = 0
    if not isinstance(value, dict):
        return cleaned, 1
    count = value.get("count")
    if isinstance(count, int) and count >= 0:
        cleaned["count"] = count
    else:
        drops += 1
    examples = value.get("examples")
    if isinstance(examples, list):
        for ex in examples[:EXAMPLES_CAP]:
            if not isinstance(ex, dict):
                continue
            ei = ex.get("event_index")
            what = ex.get("what_user_caught")
            if _is_valid_index(ei, event_count) and isinstance(what, str) and what.strip():
                cleaned["examples"].append(
                    {"event_index": ei, "what_user_caught": what.strip()[:200]}
                )
    return cleaned, drops


# ---------- Top-level validator --------------------------------------------


def validate_narrative(narrative: dict, events: list[dict]) -> dict:
    """Validate the LLM output against schema v1 rules.

    Args:
        narrative: parsed dict from extractor.extract_narrative.
        events: the chronological events for the session, used for
            event_index bounds and key_quote substring checks.

    Returns:
        Cleaned narrative dict with `narrative_quality` set.
    """
    event_count = len(events)
    cleaned: dict[str, Any] = {}
    llm_field_total: list = []  # tracks fields evaluated, for quality scoring
    llm_drops = 0

    # ---- Session-level scalars ----
    goal = narrative.get("goal")
    if isinstance(goal, str) and goal.strip():
        cleaned["goal"] = goal.strip()[:200]
    else:
        llm_drops += 1
    llm_field_total.append("goal")

    topics = narrative.get("topics")
    if isinstance(topics, list):
        valid_topics = [
            t.strip()[:30] for t in topics if isinstance(t, str) and t.strip()
        ]
        if 3 <= len(valid_topics) <= 7:
            cleaned["topics"] = valid_topics
        elif len(valid_topics) > 7:
            cleaned["topics"] = valid_topics[:7]
        elif valid_topics:
            cleaned["topics"] = valid_topics
        else:
            llm_drops += 1
    else:
        llm_drops += 1
    llm_field_total.append("topics")

    ext_systems = narrative.get("external_systems_touched")
    if isinstance(ext_systems, list):
        valid_ext = [s for s in ext_systems if isinstance(s, str) and s in EXTERNAL_SYSTEMS]
        cleaned["external_systems_touched"] = valid_ext
    else:
        llm_drops += 1
    llm_field_total.append("external_systems_touched")

    vc, replaced = _enum_or(narrative.get("verification_completeness"), VERIFICATION, "none")
    if replaced:
        llm_drops += 1
    cleaned["verification_completeness"] = vc
    llm_field_total.append("verification_completeness")

    pq, replaced = _enum_or(narrative.get("prompt_quality_signal"), PROMPT_QUALITY, "well_formed")
    if replaced:
        llm_drops += 1
    cleaned["prompt_quality_signal"] = pq
    llm_field_total.append("prompt_quality_signal")

    # ---- Tasks ----
    raw_tasks = narrative.get("tasks") or []
    valid_tasks: list[dict] = []
    valid_task_ids: set[str] = set()
    for t in raw_tasks[: LIST_CAPS["tasks"]]:
        cleaned_task, drops = _validate_task(t, event_count, llm_field_total)
        llm_drops += drops
        if cleaned_task:
            valid_tasks.append(cleaned_task)
            valid_task_ids.add(cleaned_task["task_id"])
    cleaned["tasks"] = valid_tasks
    llm_field_total.append("tasks_present")
    if not valid_tasks:
        llm_drops += 1

    # ---- Friction moments ----
    raw_fms = narrative.get("friction_moments") or []
    valid_fms: list[dict] = []
    for fm in raw_fms[: LIST_CAPS["friction_moments"]]:
        cleaned_fm, drops = _validate_friction(
            fm, event_count, valid_task_ids, events, llm_field_total
        )
        llm_drops += drops
        if cleaned_fm:
            valid_fms.append(cleaned_fm)
    cleaned["friction_moments"] = valid_fms

    ws, replaced = _enum_or(narrative.get("waste_signature"), WASTE_TYPES, "none")
    if replaced:
        llm_drops += 1
    cleaned["waste_signature"] = ws
    llm_field_total.append("waste_signature")

    # ---- Key decisions ----
    raw_kds = narrative.get("key_decisions") or []
    valid_kds: list[dict] = []
    for kd in raw_kds[: LIST_CAPS["key_decisions"]]:
        cleaned_kd, drops = _validate_key_decision(kd, event_count)
        llm_drops += drops
        if cleaned_kd:
            valid_kds.append(cleaned_kd)
    cleaned["key_decisions"] = valid_kds

    # ---- Dead ends ----
    raw_des = narrative.get("dead_ends") or []
    valid_des: list[dict] = []
    for de in raw_des[: LIST_CAPS["dead_ends"]]:
        cleaned_de, drops = _validate_dead_end(de, event_count)
        llm_drops += drops
        if cleaned_de:
            valid_des.append(cleaned_de)
    cleaned["dead_ends"] = valid_des

    # ---- Recurring environmental issues ----
    raw_eis = narrative.get("recurring_environmental_issues") or []
    valid_eis: list[dict] = []
    for ei in raw_eis[: LIST_CAPS["recurring_environmental_issues"]]:
        cleaned_ei, drops = _validate_environmental_issue(ei, event_count)
        llm_drops += drops
        if cleaned_ei:
            valid_eis.append(cleaned_ei)
    cleaned["recurring_environmental_issues"] = valid_eis

    # ---- User caught model errors ----
    uc, drops = _validate_user_caught(narrative.get("user_caught_model_errors"), event_count)
    cleaned["user_caught_model_errors"] = uc
    llm_drops += drops
    llm_field_total.append("user_caught_model_errors")

    # ---- Lessons ----
    cf = narrative.get("counterfactual")
    if isinstance(cf, str) and 30 <= len(cf.strip()) <= 300:
        cleaned["counterfactual"] = cf.strip()
    else:
        llm_drops += 1
    llm_field_total.append("counterfactual")

    la = narrative.get("lesson_for_agent")
    if isinstance(la, str) and la.strip():
        cleaned["lesson_for_agent"] = la.strip()[:200]
    else:
        llm_drops += 1
    llm_field_total.append("lesson_for_agent")

    lu = narrative.get("lesson_for_user")
    if isinstance(lu, str):
        cleaned["lesson_for_user"] = lu.strip()[:200]
    llm_field_total.append("lesson_for_user")

    notable = narrative.get("notable")
    if isinstance(notable, str):
        cleaned["notable"] = notable.strip()[:250]

    # ---- Quality scoring ----
    total = max(len(llm_field_total), 1)
    drop_rate = llm_drops / total
    if drop_rate > 0.5:
        quality = "low"
    elif drop_rate > 0.1:
        quality = "medium"
    else:
        quality = "high"
    cleaned["narrative_quality"] = quality

    return cleaned
