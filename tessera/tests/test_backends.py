"""Unit tests for the LLM backend registry.

The actual completion calls require external CLIs (`claude`, `codex`,
`gemini`) — those are smoke-tested manually. Here we verify the registry
plumbing, name resolution, and default-model selection logic that the
rest of tessera depends on.
"""
from __future__ import annotations

import os

import pytest

from tessera.backends import (
    ClaudeSDKBackend,
    CodexCLIBackend,
    GeminiCLIBackend,
    LLMBackend,
    default_model_for,
    get_backend,
    list_backends,
)


def test_list_backends_returns_all_three_in_display_order():
    assert list_backends() == ["claude", "codex", "gemini"]


def test_get_backend_explicit_name():
    assert isinstance(get_backend("claude"), ClaudeSDKBackend)
    assert isinstance(get_backend("codex"), CodexCLIBackend)
    assert isinstance(get_backend("gemini"), GeminiCLIBackend)


def test_get_backend_case_insensitive():
    assert isinstance(get_backend("CLAUDE"), ClaudeSDKBackend)
    assert isinstance(get_backend("Codex"), CodexCLIBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("ollama")


def test_get_backend_defaults_to_claude_when_no_args():
    """No name, no env var → claude (the historical default)."""
    # Ensure env var isn't set during the assertion
    prior = os.environ.pop("TESSERA_BACKEND", None)
    try:
        assert isinstance(get_backend(), ClaudeSDKBackend)
        assert isinstance(get_backend(None), ClaudeSDKBackend)
    finally:
        if prior is not None:
            os.environ["TESSERA_BACKEND"] = prior


def test_get_backend_reads_env_var(monkeypatch):
    monkeypatch.setenv("TESSERA_BACKEND", "codex")
    assert isinstance(get_backend(), CodexCLIBackend)


def test_get_backend_explicit_name_overrides_env(monkeypatch):
    """A specific --backend on the CLI beats the env var."""
    monkeypatch.setenv("TESSERA_BACKEND", "codex")
    assert isinstance(get_backend("gemini"), GeminiCLIBackend)


def test_default_model_for_known_backends():
    assert default_model_for("claude") == "claude-sonnet-4-6"
    # codex and gemini default to empty — the CLI picks its session
    # default. This matters because changing it would silently break
    # ChatGPT-subscription auth on codex.
    assert default_model_for("codex") == ""
    assert default_model_for("gemini") == ""


def test_default_model_for_unknown_falls_back_to_claude():
    """Unknown backends shouldn't crash callers — fall back gracefully."""
    assert default_model_for("ollama") == "claude-sonnet-4-6"
    assert default_model_for(None) == "claude-sonnet-4-6"


def test_backend_subclass_required_fields():
    """Every registered backend must declare name + default_model + cli_binary
    so doctor / cache-key / model-resolution code doesn't crash on AttributeError."""
    for name in list_backends():
        b = get_backend(name)
        assert isinstance(b, LLMBackend)
        assert b.name == name
        # default_model can be empty string but must exist
        assert isinstance(b.default_model, str)
        # cli_binary is the doctor check key
        assert isinstance(b.cli_binary, str) and b.cli_binary


def test_backend_available_falls_back_to_path_check(monkeypatch):
    """available() returns False when the backend's CLI is missing.
    Used by tessera doctor to surface "install one of these" hints."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert get_backend("codex").available() is False
    assert get_backend("gemini").available() is False
    assert get_backend("claude").available() is False
