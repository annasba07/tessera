"""Recovery scenarios for the synthesis JSON extractor.

Lock in the failure modes we've actually observed in the wild:
- Sonnet collapsing mid-output and emitting two JSON objects
- Flash 3.5 High emitting valid JSON followed by prose commentary
- Markdown fence wrapping
"""
import json

import pytest

from tessera.narratives.synthesis import _extract_json


def test_clean_json_parses():
    raw = '{"headline":"x","observations":[]}'
    assert _extract_json(raw) == {"headline": "x", "observations": []}


def test_markdown_fence_stripped():
    raw = '```json\n{"headline":"x"}\n```'
    assert _extract_json(raw) == {"headline": "x"}


def test_trailing_prose_after_valid_json_recovered():
    """Flash 3.5 High observed in consistency-check Run 2: emits a
    complete valid JSON object, then adds a commentary paragraph after
    the closing brace. json.loads raises 'Extra data' but the JSON itself
    is fine — raw_decode pulls it out cleanly."""
    raw = (
        '{"headline":"OAuth blocked 22 sessions","observations":[]}'
        "\n\n"
        "This analysis covers the 7-day window from June 1-7, 2026, "
        "meaning we cannot analyze long-term seasonal patterns. "
        "S028 and S029 are the only Codex engineering sessions."
    )
    obj = _extract_json(raw)
    assert obj["headline"] == "OAuth blocked 22 sessions"
    assert obj["observations"] == []


def test_doubled_object_picks_last_via_anchor_recovery():
    """Sonnet observed in earlier run: emits one object, restarts mid-
    string, emits a second complete object. The valid one is the LAST.
    Recovery iterates anchors in reverse (last attempt first)."""
    raw = (
        '{"headline":"truncated mid-string and the model fell into pulse-'
        '{"headline":"OAuth blocked 22 sessions","observations":[{"title":"x"}]}'
    )
    obj = _extract_json(raw)
    assert obj["headline"] == "OAuth blocked 22 sessions"
    assert len(obj["observations"]) == 1


def test_invalid_backslash_escape_recovered():
    """Sonnet 4.6 observed in consistency-check Run 3: emits invalid
    JSON escapes mid-string like '...domains.\\{' where the bare \\{
    isn't a valid JSON escape. Recovery strips the offending backslash
    and re-parses. The valid escapes (\\n, \\\", \\\\) must survive."""
    raw = (
        '{"headline":"OAuth blocked 22 sessions",'
        '"observations":[{"title":"track.\\{kind issue","why":"line1\\nline2"}]}'
    )
    obj = _extract_json(raw)
    assert obj["headline"] == "OAuth blocked 22 sessions"
    # The invalid \\{ should have been recovered to {
    assert "{kind" in obj["observations"][0]["title"]
    # The valid \\n escape should survive intact (interpreted as newline)
    assert "line1\nline2" == obj["observations"][0]["why"]


def test_completely_malformed_raises_with_dump():
    """If we can't recover anything, raise the original JSONDecodeError
    so the synthesizer catches it (and dumps the raw text for the user)."""
    with pytest.raises(json.JSONDecodeError):
        _extract_json("this is not JSON at all")


def test_empty_string_raises():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("")
