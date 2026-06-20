"""Tests for inherited-rating display on the dashboard.

Two inheritance sources:
  1. localStorage entries from OTHER dashboard files (handled in JS;
     not unit-tested here — UI integration test territory)
  2. Prior-history ratings under ~/.config/tessera/history/ratings/

This test file covers source #2: when a fresh dashboard is rendered
with a history store that has rated patterns, the prior_ratings_by_key
embedding is populated and the JS receives it via D.prior_ratings.
"""
from __future__ import annotations

import json
from pathlib import Path

from tessera.history import HistoryStore
from tessera.narratives.dashboard import (
    _collect_prior_ratings_by_key,
    render_dashboard,
)


def _seed_history(tmp_path: Path, slug: str, observations: list[dict],
                  ratings: list[dict]) -> None:
    """Write a fake history run with paired ratings to tmp."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{slug}.json").write_text(json.dumps({
        "observations": observations,
        "behavioral_patterns": [],
    }))
    ratings_dir = tmp_path / "ratings"
    ratings_dir.mkdir(parents=True, exist_ok=True)
    (ratings_dir / f"{slug}.json").write_text(json.dumps({
        "slug": slug,
        "rated_at": "2026-06-07T15:00:00+00:00",
        "ratings": ratings,
    }))
    index_path = tmp_path / "history.json"
    existing: list[dict] = []
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text())
        except Exception:
            existing = []
    existing.insert(0, {
        "slug": slug,
        "timestamp": "2026-06-07T15:00:00+00:00",
        "lookback_days": 7,
    })
    index_path.write_text(json.dumps(existing))


def test_collect_prior_ratings_indexes_by_key(tmp_path):
    """The simplest case: one prior run with one rating; the rating
    should appear under its observation_key in the returned dict."""
    _seed_history(
        tmp_path,
        slug="2026-06-07T15-00-00-000000_00-00",
        observations=[{"title": "Auth fix never applied", "claim": "x"}],
        ratings=[{
            "index": 0,
            "title": "Auth fix never applied",
            "key": "abc123def4",
            "rating": "useful",
        }],
    )
    store = HistoryStore(tmp_path)
    result = _collect_prior_ratings_by_key(store)
    assert "abc123def4" in result
    rated = result["abc123def4"][0]
    assert rated["rating"] == "useful"
    assert rated["source"] == "history"
    assert rated["date"].startswith("2026-06-07")


def test_collect_prior_ratings_most_recent_first(tmp_path):
    """If the same pattern was rated in two runs, the more recent
    rating should be the head of the list (so JS picks it as
    'current truth'). Older runs are still accessible for an
    'audit history' view, just not the default."""
    _seed_history(
        tmp_path,
        slug="2026-05-01T10-00-00-000000_00-00",
        observations=[{"title": "Same pattern"}],
        ratings=[{"index": 0, "title": "Same", "key": "samekey0001",
                  "rating": "useful"}],
    )
    _seed_history(
        tmp_path,
        slug="2026-06-07T15-00-00-000000_00-00",
        observations=[{"title": "Same pattern"}],
        ratings=[{"index": 0, "title": "Same", "key": "samekey0001",
                  "rating": "wrong"}],
    )
    store = HistoryStore(tmp_path)
    result = _collect_prior_ratings_by_key(store)
    assert len(result["samekey0001"]) == 2
    # Most recent first → wrong is the current truth
    assert result["samekey0001"][0]["rating"] == "wrong"
    assert result["samekey0001"][1]["rating"] == "useful"


def test_collect_prior_ratings_empty_when_no_history(tmp_path):
    store = HistoryStore(tmp_path)
    result = _collect_prior_ratings_by_key(store)
    assert result == {}


def test_render_dashboard_embeds_prior_ratings():
    """The prior_ratings_by_key arg should land in the embedded JSON
    that the JS reads via window.__SYNTHESIS_DATA__.prior_ratings."""
    synthesis = {
        "headline": "Test",
        "observations": [],
        "behavioral_patterns": [],
        "quick_wins": [],
        "per_project": [],
        "meta": {"run_slug": "test-run-001"},
    }
    narratives: list[dict] = []
    prior = {
        "samplekey01": [{
            "slug": "2026-06-07T15-00-00-000000_00-00",
            "date": "2026-06-07",
            "rating": "useful",
            "source": "history",
            "title": "Sample pattern",
        }],
    }
    html = render_dashboard(synthesis, narratives, prior_ratings_by_key=prior)
    # The embedded SYNTHESIS_DATA constant must include our prior ratings
    # under the prior_ratings field for the JS to pick it up.
    assert "samplekey01" in html
    assert "prior_ratings" in html


def test_render_dashboard_without_prior_ratings_no_breakage():
    """The kwarg is optional; absence should produce an empty
    prior_ratings object, not crash or leave a `null` that JS
    would mishandle."""
    synthesis = {
        "headline": "Test",
        "observations": [],
        "behavioral_patterns": [],
        "quick_wins": [],
        "per_project": [],
        "meta": {"run_slug": "test-run-002"},
    }
    html = render_dashboard(synthesis, [])
    assert "prior_ratings" in html
    # An empty {} is what the JS expects when no priors exist.
    assert '"prior_ratings": {}' in html
