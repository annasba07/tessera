"""Tests for deterministic metadata extraction."""

from __future__ import annotations

from tessera.narratives.deterministic import (
    _project_label,
    _tool_category,
    extract_deterministic,
)


def test_project_label_takes_last_two_segments():
    assert _project_label("/Users/x/Software-Projects/myapp") == "Software-Projects/myapp"
    assert _project_label("/single") == "single"
    assert _project_label("") == "(no project)"
    assert _project_label(None) == "(no project)"


def test_tool_category_classifies_by_substring():
    assert _tool_category("Bash") == "shell"
    assert _tool_category("exec_command") == "shell"
    assert _tool_category("mcp__chrome-devtools__take_screenshot") == "browser"
    assert _tool_category("Read") == "read"
    assert _tool_category("Edit") == "write"
    assert _tool_category("Task") == "delegation"
    assert _tool_category("WebSearch") == "web"
    assert _tool_category("Unknown") == "other"


def test_extract_deterministic_returns_basic_shape(sample_events):
    result = extract_deterministic("claude:test-session-1", sample_events)
    assert result["session_id"] == "claude:test-session-1"
    assert result["agent"] == "claude"
    assert result["project_label"] == "proj/myapp"
    assert result["event_count"] == len(sample_events)
    assert result["tool_call_count"] == 2
    assert result["user_turn_count"] == 2
    assert result["primary_model"] == "claude-sonnet-4-6"
    assert "events_content_hash" in result
    assert result["events_content_hash"].startswith("sha256:")


def test_extract_deterministic_counts_errors(sample_events):
    result = extract_deterministic("claude:test-session-1", sample_events)
    assert result["tool_error_classes"]["tool_error"] == 1
    assert result["tool_error_rate"] == 0.5  # 1 error / 2 tool calls


def test_extract_deterministic_detects_user_corrections(sample_events):
    result = extract_deterministic("claude:test-session-1", sample_events)
    # "no, that's wrong" matches both "no, " and "wrong"
    assert result["user_friction_signals"]["explicit_corrections"] >= 1


def test_extract_deterministic_handles_empty_events():
    result = extract_deterministic("claude:nothing", [])
    assert result["event_count"] == 0
    assert result["session_id"] == "claude:nothing"


def test_extract_deterministic_aggregates_tool_categories(sample_events):
    result = extract_deterministic("claude:test-session-1", sample_events)
    cats = result["tool_category_distribution"]
    # 1 Grep (read) + 1 Bash (shell)
    assert cats.get("read") == 1
    assert cats.get("shell") == 1
