"""Tests for HistoryStore — focus on slug validation (path-traversal defense)."""

from __future__ import annotations

import pytest

from tessera.history import HistoryStore, _validated_slug


def test_validated_slug_accepts_iso_timestamp_slugs():
    assert _validated_slug("2026-04-28T10-30-00-000000_00-00") == \
        "2026-04-28T10-30-00-000000_00-00"


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "..",
        "foo/bar",
        "foo\\bar",
        "foo.bar",  # _safe_slug strips dots, so a dot here is suspicious
        "",
        " ",
        "a" * 200,  # too long
    ],
)
def test_validated_slug_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        _validated_slug(bad)


def test_save_ratings_rejects_path_traversal(tmp_path):
    """save_ratings must not write outside data_dir, even if asked to."""
    store = HistoryStore(data_dir=tmp_path)
    with pytest.raises(ValueError):
        store.save_ratings("../escaped", [{"index": 0, "rating": "useful"}])
    # Confirm no file landed outside the data dir
    assert not (tmp_path.parent / "escaped.json").exists()


def test_load_run_rejects_path_traversal(tmp_path):
    store = HistoryStore(data_dir=tmp_path)
    with pytest.raises(ValueError):
        store.load_run("../../../../etc/hosts")
