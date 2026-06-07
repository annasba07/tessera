"""Tests for the narrative cache."""

from __future__ import annotations

import pytest

from tessera.narratives.cache import NarrativeCache


def test_cache_round_trip(tmp_path):
    cache = NarrativeCache(tmp_path)
    sid = "claude:abc-123"
    payload = {"session_id": sid, "agent": "claude", "narrative_quality": "high"}
    key = NarrativeCache.make_key("sha256:hash1", 1, "claude-sonnet-4-6")
    cache.save(sid, payload, key)
    loaded = cache.load(sid, key)
    assert loaded is not None
    assert loaded["session_id"] == sid
    assert loaded["agent"] == "claude"


def test_cache_key_mismatch_returns_none(tmp_path):
    cache = NarrativeCache(tmp_path)
    sid = "claude:abc-123"
    cache.save(sid, {"session_id": sid}, NarrativeCache.make_key("h1", 1, "model"))
    assert cache.load(sid, NarrativeCache.make_key("h2", 1, "model")) is None
    assert cache.load(sid, NarrativeCache.make_key("h1", 2, "model")) is None
    assert cache.load(sid, NarrativeCache.make_key("h1", 1, "other")) is None


def test_cache_key_backwards_compat_for_claude():
    """The default backend='claude' must produce the legacy key format
    (just the model id, no 'claude:' prefix) so 700+ on-disk narratives
    written by tessera < 0.6.0 stay valid after upgrade. If this test
    fails, the next `tessera run` will re-narrate every cached session
    at significant cost — DO NOT change the format without a migration."""
    legacy = "sha256:hash|v1|claude-sonnet-4-6"
    assert NarrativeCache.make_key("sha256:hash", 1, "claude-sonnet-4-6") == legacy
    assert (
        NarrativeCache.make_key("sha256:hash", 1, "claude-sonnet-4-6", backend="claude")
        == legacy
    )


def test_cache_key_namespaces_non_claude_backends():
    """Codex and Gemini default to empty model id — without backend
    namespacing they'd collide on the same session. Confirm they don't."""
    codex_empty = NarrativeCache.make_key("h", 1, "", backend="codex")
    gemini_empty = NarrativeCache.make_key("h", 1, "", backend="gemini")
    assert codex_empty != gemini_empty
    assert "codex:" in codex_empty
    assert "gemini:" in gemini_empty


def test_cache_missing_session_returns_none(tmp_path):
    cache = NarrativeCache(tmp_path)
    assert cache.load("claude:never-cached", "any-key") is None


def test_cache_clear(tmp_path):
    cache = NarrativeCache(tmp_path)
    sid = "claude:gone"
    cache.save(sid, {"session_id": sid}, NarrativeCache.make_key("h", 1, "m"))
    assert cache.clear(sid) is True
    assert cache.clear(sid) is False  # second clear returns False


def test_cache_list_cached_returns_session_ids(tmp_path):
    cache = NarrativeCache(tmp_path)
    cache.save("claude:a", {"session_id": "claude:a"}, NarrativeCache.make_key("h", 1, "m"))
    cache.save("codex:b", {"session_id": "codex:b"}, NarrativeCache.make_key("h", 1, "m"))
    cached = sorted(cache.list_cached())
    assert cached == ["claude:a", "codex:b"]


def test_cache_filename_safe_for_session_id_with_colon(tmp_path):
    cache = NarrativeCache(tmp_path)
    sid = "claude:has:multiple:colons"
    cache.save(sid, {"session_id": sid}, NarrativeCache.make_key("h", 1, "m"))
    # File should exist and not contain raw colons (filesystem-unfriendly on some platforms)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    # Round-trips correctly
    loaded = cache.load(sid, NarrativeCache.make_key("h", 1, "m"))
    assert loaded["session_id"] == sid
