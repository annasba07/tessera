"""Tests for dashboard helpers — particularly the inline-JSON safety escaping."""

from __future__ import annotations

import json

from tessera.narratives.dashboard import (
    _safe_inline_json,
    _short_session,
    render_dashboard,
)


def test_safe_inline_json_escapes_script_close_tag():
    """A session description containing </script> would otherwise close the inline tag."""
    payload = {"text": "Some text </script><script>alert(1)</script>"}
    out = _safe_inline_json(payload)
    assert "</script>" not in out
    assert "<\\/script>" in out
    # Still valid JSON
    parsed = json.loads(out.replace("<\\/", "</"))
    assert parsed["text"] == payload["text"]


def test_safe_inline_json_escapes_line_separators():
    """U+2028 / U+2029 are valid in JSON but break inline JS."""
    ls = chr(0x2028)
    ps = chr(0x2029)
    payload = {"text": f"line1{ls}line2{ps}line3"}
    out = _safe_inline_json(payload)
    assert ls not in out
    assert ps not in out
    assert "\\u2028" in out or "\\u2029" in out


def test_short_session_truncates_sensibly():
    long_uuid = "claude:76ef401c-faa7-4e0c-ac02-14508cdd8c4d"
    assert _short_session(long_uuid).startswith("claude:")
    assert "…" in _short_session(long_uuid)
    short = "claude:abc"
    assert _short_session(short) == "claude:abc"  # no truncation when short
    assert _short_session("") == "?"
    assert _short_session(None) == "?"


def test_render_dashboard_produces_self_contained_html():
    """End-to-end: minimal synthesis + narratives → valid HTML doc."""
    synthesis = {
        "headline": "Test headline",
        "if_you_do_one_thing_this_week": "Do this one thing.",
        "observations": [],
        "quick_wins": [],
        "per_project": [],
        "meta": {"input_sessions": 0, "model": "test-model"},
    }
    html = render_dashboard(synthesis, [])
    assert html.startswith("<!doctype html>")
    assert "<title>Tessera</title>" in html
    assert "Test headline" in html
    assert "Do this one thing." in html
    # Has both views' DOM
    assert 'data-view="findings"' in html
    assert 'data-view="explore"' in html
    # JS is inlined
    assert "window.__AR__" in html


def test_render_dashboard_with_observations_shows_evidence_and_chips():
    synthesis = {
        "headline": "h",
        "observations": [
            {
                "title": "Obs 1",
                "claim": "c",
                "evidence_refs": ["S001"],
                "evidence_sessions": ["claude:abc-123"],
                "supporting_count": 1,
                "confidence": "high",
                "category": "environmental",
                "next_action": "fix it",
            }
        ],
        "quick_wins": [],
        "per_project": [],
        "meta": {"model": "m"},
    }
    narratives = [
        {
            "session_id": "claude:abc-123",
            "agent": "claude",
            "project_label": "p",
            "started_at": "2026-04-25T10:00:00Z",
            "event_count": 10,
            "active_minutes": 5.0,
            "friction_moments": [{"cost_active_minutes": 2.5}],
        }
    ]
    html = render_dashboard(synthesis, narratives)
    assert "S001" in html
    assert "claude:abc-123" in html
    assert "high" in html
    assert "environmental" in html
    assert "fix it" in html
