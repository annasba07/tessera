"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_events():
    """A small synthetic session covering all event types we extract."""
    base_ts = "2026-04-25T10:00:00+00:00"
    return [
        {
            "agent": "claude",
            "session_id": "test-session-1",
            "trace_kind": "top_level",
            "timestamp": base_ts,
            "role": "user",
            "event_type": "message",
            "message_text": "Add a CLI flag for --verbose to the build command",
            "cwd": "/Users/dev/proj/myapp",
            "git_branch": "feature/verbose",
            "model": "claude-sonnet-4-6",
        },
        {
            "agent": "claude",
            "session_id": "test-session-1",
            "trace_kind": "top_level",
            "timestamp": "2026-04-25T10:00:05+00:00",
            "role": "assistant",
            "event_type": "reasoning",
            "message_text": "Let me find the build command first.",
            "cwd": "/Users/dev/proj/myapp",
        },
        {
            "agent": "claude",
            "session_id": "test-session-1",
            "trace_kind": "top_level",
            "timestamp": "2026-04-25T10:00:10+00:00",
            "role": "assistant",
            "event_type": "tool_call",
            "tool_name": "Grep",
            "tool_input_preview": '{"pattern": "build", "path": "src"}',
            "tool_call_id": "t1",
        },
        {
            "agent": "claude",
            "session_id": "test-session-1",
            "trace_kind": "top_level",
            "timestamp": "2026-04-25T10:00:11+00:00",
            "role": "tool",
            "event_type": "tool_result",
            "tool_name": "Grep",
            "tool_call_id": "t1",
            "tool_status": "success",
        },
        {
            "agent": "claude",
            "session_id": "test-session-1",
            "trace_kind": "top_level",
            "timestamp": "2026-04-25T10:00:30+00:00",
            "role": "assistant",
            "event_type": "tool_call",
            "tool_name": "Bash",
            "tool_input_preview": '{"command": "pytest tests/"}',
            "tool_call_id": "t2",
        },
        {
            "agent": "claude",
            "session_id": "test-session-1",
            "trace_kind": "top_level",
            "timestamp": "2026-04-25T10:00:35+00:00",
            "role": "tool",
            "event_type": "tool_result",
            "tool_name": "Bash",
            "tool_call_id": "t2",
            "tool_status": "error",
            "error_class": "tool_error",
            "tool_output_preview": "ImportError: ...",
        },
        {
            "agent": "claude",
            "session_id": "test-session-1",
            "trace_kind": "top_level",
            "timestamp": "2026-04-25T10:00:50+00:00",
            "role": "user",
            "event_type": "message",
            "message_text": "no, that's wrong, try the other path",
        },
    ]


@pytest.fixture
def sample_narrative():
    """A representative validated narrative payload (post-LLM)."""
    return {
        "session_id": "claude:test-1",
        "agent": "claude",
        "project_label": "proj/myapp",
        "started_at": "2026-04-25T10:00:00Z",
        "active_minutes": 12.5,
        "event_count": 50,
        "primary_model": "claude-sonnet-4-6",
        "tasks": [
            {
                "task_id": "t1",
                "intent": "Add --verbose flag",
                "event_range": [0, 49],
                "task_type": "feature_implementation",
                "task_difficulty": {"overall": "trivial"},
                "outcome": "completed",
                "outcome_evidence_event": 49,
            }
        ],
        "friction_moments": [
            {
                "event_range": [10, 18],
                "type": "tool_error_loop",
                "tool_category": "shell",
                "description": "pytest failed 3 times",
                "key_quote": "ImportError",
                "cost_events": 9,
                "cost_active_minutes": 2.5,
            }
        ],
        "waste_signature": "tool_error_loop",
        "counterfactual": "Should have run pip install before pytest — would save ~3 minutes.",
        "lesson_for_user": "Add a make setup target.",
        "narrative_quality": "high",
    }
