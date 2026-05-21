"""Tests for the pluggable normalizer registry.

Onboarding feature: users should be able to drop a single Python file
into ~/.config/tessera/normalizers/ and have it auto-registered alongside
the built-ins. These tests cover the registration, discovery, and
graceful-failure paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera import normalizers


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts with a fresh registry — initialize() is idempotent
    in production but tests need a clean slate."""
    normalizers._REGISTRY.clear()
    normalizers._INITIALIZED = False
    yield
    normalizers._REGISTRY.clear()
    normalizers._INITIALIZED = False


def test_register_adds_to_registry(tmp_path):
    def fake(raw_root, writer):
        pass

    normalizers.register(
        name="fake", default_source=tmp_path, normalize=fake, description="t",
    )
    assert normalizers.get("fake").name == "fake"
    assert normalizers.get("fake").default_source == tmp_path
    assert normalizers.get("fake").normalize is fake


def test_register_rejects_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        normalizers.register(name="", default_source=tmp_path, normalize=lambda r, w: None)
    with pytest.raises(ValueError):
        normalizers.register(
            name="bad", default_source=tmp_path, normalize="not callable"  # type: ignore[arg-type]
        )


def test_get_all_orders_builtins_first(tmp_path):
    # Register one user-defined and one builtin (manually as builtin)
    normalizers.register(name="zzz_user", default_source=tmp_path, normalize=lambda r, w: None)
    normalizers.register(
        name="claude", default_source=tmp_path, normalize=lambda r, w: None, source="builtin",
    )
    names = [n.name for n in normalizers.get_all()]
    assert names == ["claude", "zzz_user"]


def test_initialize_loads_builtins():
    warnings = normalizers.initialize()
    assert warnings == []  # nothing user-defined to fail
    names = {n.name for n in normalizers.get_all()}
    assert {"claude", "codex", "gemini"} <= names


def test_initialize_is_idempotent():
    normalizers.initialize()
    count_after_first = len(normalizers.get_all())
    normalizers.initialize()
    normalizers.initialize()
    assert len(normalizers.get_all()) == count_after_first


def test_user_dir_loads_valid_normalizer(tmp_path):
    norm_dir = tmp_path / "normalizers"
    norm_dir.mkdir()
    (norm_dir / "mock_agent.py").write_text(
        "from pathlib import Path\n"
        "from tessera.normalizers import register\n"
        "def _normalize(raw_root, writer):\n"
        "    pass\n"
        "register(name='mock_agent', default_source=Path('/tmp/mock'),\n"
        "         normalize=_normalize, description='mock')\n"
    )
    warnings = normalizers._load_user_dir(norm_dir)
    assert warnings == []
    assert normalizers.get("mock_agent") is not None
    assert normalizers.get("mock_agent").description == "mock"


def test_user_dir_surfaces_broken_normalizer_warning(tmp_path):
    norm_dir = tmp_path / "normalizers"
    norm_dir.mkdir()
    (norm_dir / "broken.py").write_text("raise RuntimeError('intentional')\n")
    warnings = normalizers._load_user_dir(norm_dir)
    assert len(warnings) == 1
    assert "broken.py" in warnings[0]
    assert "RuntimeError" in warnings[0]
    # The broken file shouldn't have polluted the registry
    assert normalizers.get("broken") is None


def test_user_dir_skips_underscore_files(tmp_path):
    norm_dir = tmp_path / "normalizers"
    norm_dir.mkdir()
    (norm_dir / "_helpers.py").write_text("raise RuntimeError('should not load')\n")
    warnings = normalizers._load_user_dir(norm_dir)
    assert warnings == []


def test_user_dir_silent_when_dir_missing(tmp_path):
    """No warnings, no errors when the user hasn't created the dir yet."""
    missing = tmp_path / "does-not-exist"
    warnings = normalizers._load_user_dir(missing)
    assert warnings == []
