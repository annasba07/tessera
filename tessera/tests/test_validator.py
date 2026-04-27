"""Tests for narrative validator — should drop bad fields, not whole sessions."""

from __future__ import annotations

from tessera.narratives.validator import (
    TASK_TYPES,
    WASTE_TYPES,
    validate_narrative,
)


def _make_events(n: int) -> list[dict]:
    """Minimal event list of length n; only the count matters for bound checks."""
    return [{"event_type": "message", "message_text": f"event {i}"} for i in range(n)]


def test_validate_drops_invalid_event_range():
    events = _make_events(50)
    narrative = {
        "goal": "Test session",
        "topics": ["a", "b", "c"],
        "tasks": [
            {
                "task_id": "t1",
                "intent": "Good task",
                "event_range": [0, 49],
                "task_type": "feature_implementation",
                "outcome": "completed",
                "outcome_evidence_event": 49,
            }
        ],
        "friction_moments": [
            {
                "event_range": [10, 200],  # invalid — out of bounds
                "task_id": "t1",
                "type": "blind_retry",
                "tool_category": "shell",
                "description": "bad range",
                "cost_active_minutes": 1.0,
            },
            {
                "event_range": [5, 15],  # valid
                "task_id": "t1",
                "type": "blind_retry",
                "tool_category": "shell",
                "description": "valid friction",
                "cost_active_minutes": 1.0,
            },
        ],
        "counterfactual": "An event-grounded counterfactual that is at least thirty characters.",
    }
    cleaned = validate_narrative(narrative, events)
    assert len(cleaned["friction_moments"]) == 1
    assert cleaned["friction_moments"][0]["event_range"] == [5, 15]


def test_validate_replaces_bad_enum_with_other():
    events = _make_events(20)
    narrative = {
        "goal": "g",
        "tasks": [
            {
                "task_id": "t1",
                "intent": "i",
                "event_range": [0, 19],
                "task_type": "totally_made_up_type",  # invalid vocab
                "outcome": "completed",
                "outcome_evidence_event": 19,
            }
        ],
        "counterfactual": "An event-grounded counterfactual that is at least thirty characters long.",
    }
    cleaned = validate_narrative(narrative, events)
    assert cleaned["tasks"][0]["task_type"] == "other"


def test_validate_recomputes_cost_events():
    events = _make_events(30)
    narrative = {
        "goal": "g",
        "tasks": [
            {
                "task_id": "t1",
                "intent": "i",
                "event_range": [0, 29],
                "task_type": "exploration",
                "outcome": "completed",
                "outcome_evidence_event": 29,
            }
        ],
        "friction_moments": [
            {
                "event_range": [5, 12],
                "task_id": "t1",
                "type": "blind_retry",
                "tool_category": "shell",
                "description": "f",
                "cost_events": 999,  # wrong; should be 8
                "cost_active_minutes": 1.0,
            }
        ],
        "counterfactual": "An event-grounded counterfactual that is at least thirty characters long.",
    }
    cleaned = validate_narrative(narrative, events)
    assert cleaned["friction_moments"][0]["cost_events"] == 8


def test_validate_drops_friction_with_unknown_task_id():
    events = _make_events(30)
    narrative = {
        "goal": "g",
        "tasks": [
            {
                "task_id": "t1",
                "intent": "i",
                "event_range": [0, 29],
                "task_type": "exploration",
                "outcome": "completed",
                "outcome_evidence_event": 29,
            }
        ],
        "friction_moments": [
            {
                "event_range": [5, 12],
                "task_id": "t99",  # references nonexistent task
                "type": "blind_retry",
                "tool_category": "shell",
                "description": "f",
                "cost_active_minutes": 1.0,
            }
        ],
        "counterfactual": "An event-grounded counterfactual that is at least thirty characters long.",
    }
    cleaned = validate_narrative(narrative, events)
    # friction_moment is kept (range valid) but task_id is blanked
    assert "task_id" not in cleaned["friction_moments"][0]


def test_validate_caps_lists():
    events = _make_events(100)
    narrative = {
        "goal": "g",
        "tasks": [
            {
                "task_id": f"t{i}",
                "intent": "i",
                "event_range": [i, i + 5],
                "task_type": "exploration",
                "outcome": "completed",
                "outcome_evidence_event": i,
            }
            for i in range(10)  # 10 tasks; cap is 5
        ],
        "counterfactual": "An event-grounded counterfactual that is at least thirty characters long.",
    }
    cleaned = validate_narrative(narrative, events)
    assert len(cleaned["tasks"]) == 5


def test_validate_blanks_short_counterfactual():
    events = _make_events(20)
    narrative = {
        "goal": "g",
        "counterfactual": "too short",  # <30 chars
    }
    cleaned = validate_narrative(narrative, events)
    assert "counterfactual" not in cleaned


def test_validate_sets_quality_label():
    events = _make_events(20)
    # Mostly-valid narrative
    good = {
        "goal": "good goal",
        "topics": ["a", "b", "c"],
        "tasks": [
            {
                "task_id": "t1",
                "intent": "i",
                "event_range": [0, 19],
                "task_type": "exploration",
                "task_difficulty": {
                    "scope": "single_file",
                    "specification": "clear",
                    "external_dependencies": "none",
                    "verification_ease": "automated",
                    "overall": "trivial",
                },
                "outcome": "completed",
                "outcome_evidence_event": 19,
                "time_to_first_verified_progress": 5,
            }
        ],
        "waste_signature": "none",
        "external_systems_touched": [],
        "verification_completeness": "thoroughly_verified",
        "prompt_quality_signal": "well_formed",
        "user_caught_model_errors": {"count": 0, "examples": []},
        "counterfactual": "An event-grounded counterfactual that is at least thirty characters long.",
        "lesson_for_agent": "be more careful",
        "lesson_for_user": "share more context",
    }
    cleaned = validate_narrative(good, events)
    assert cleaned["narrative_quality"] in ("high", "medium")


def test_controlled_vocabularies_are_non_empty():
    """Sanity check that vocab constants are populated — guards against a refactor that empties them."""
    assert len(TASK_TYPES) >= 10
    assert "feature_implementation" in TASK_TYPES
    assert "tool_error_loop" in WASTE_TYPES
    assert "none" in WASTE_TYPES
