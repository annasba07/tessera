"""Tests for cross-session synthesis — particularly ref-token translation."""

from __future__ import annotations

from tessera.narratives.synthesis import (
    _build_ref_map,
    _validate,
    build_synthesis_prompt,
)


def _make_narratives(n: int) -> list[dict]:
    return [
        {
            "session_id": f"claude:session-{i:03d}",
            "agent": "claude",
            "project_label": "test/project",
            "started_at": f"2026-04-{20 + i:02d}T10:00:00Z",
            "active_minutes": 10.0,
            "event_count": 50,
            "primary_model": "claude-sonnet-4-6",
            "tasks": [],
            "friction_moments": [],
            "recurring_environmental_issues": [],
            "user_caught_model_errors": {"count": 0},
            "topics": ["test"],
            "external_systems_touched": [],
        }
        for i in range(n)
    ]


def test_build_ref_map_assigns_sequential_short_tokens():
    narratives = _make_narratives(5)
    ref_to_id, id_to_ref = _build_ref_map(narratives)
    assert ref_to_id["S001"] == "claude:session-000"
    assert ref_to_id["S005"] == "claude:session-004"
    assert id_to_ref["claude:session-002"] == "S003"
    assert len(ref_to_id) == 5


def test_validator_translates_refs_and_drops_fabrications():
    ref_to_id = {"S001": "claude:real-1", "S002": "claude:real-2"}
    parsed = {
        "observations": [
            {
                "title": "Real obs",
                "claim": "c",
                "evidence_refs": ["S001", "S002", "S999"],  # S999 fabricated
                "next_action": "fix it",
            },
            {
                "title": "All-fake obs",
                "claim": "c",
                "evidence_refs": ["S888", "S777"],
                "next_action": "x",
            },
        ],
        "quick_wins": [
            {"fix": "do thing", "evidence_refs": ["S001", "S404"]},
        ],
    }
    cleaned = _validate(parsed, ref_to_id)
    # First obs: 2 valid + 1 dropped; resolved to real session_ids
    assert len(cleaned["observations"]) == 1
    obs = cleaned["observations"][0]
    assert obs["evidence_refs"] == ["S001", "S002"]
    assert obs["evidence_sessions"] == ["claude:real-1", "claude:real-2"]
    assert obs["supporting_count"] == 2
    # Second obs: all refs fake → dropped entirely
    # Quick win: 1 valid + 1 dropped → kept
    assert len(cleaned["quick_wins"]) == 1
    assert cleaned["quick_wins"][0]["evidence_refs"] == ["S001"]
    # Fabrication count surfaced in meta
    assert cleaned["meta"]["fabricated_ref_count"] == 4  # S999 + S888 + S777 + S404


def test_validator_passes_clean_input_without_drops():
    ref_to_id = {"S001": "claude:real-1"}
    parsed = {
        "observations": [
            {
                "title": "Clean",
                "claim": "c",
                "evidence_refs": ["S001"],
                "next_action": "x",
            }
        ],
        "quick_wins": [],
    }
    cleaned = _validate(parsed, ref_to_id)
    assert cleaned["meta"]["fabricated_ref_count"] == 0
    assert cleaned["observations"][0]["supporting_count"] == 1


def test_build_synthesis_prompt_includes_ref_tokens_in_input():
    narratives = _make_narratives(3)
    prompt, ref_to_id = build_synthesis_prompt(narratives)
    # Refs appear in the embedded session JSON
    assert '"ref": "S001"' in prompt
    assert '"ref": "S003"' in prompt
    # Real session IDs deliberately omitted from the LLM's input
    assert "claude:session-000" not in prompt
    # Citation rules section present
    assert "Cite sessions by ref" in prompt or "ref token" in prompt.lower()
