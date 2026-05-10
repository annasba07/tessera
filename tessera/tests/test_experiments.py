"""Tests for experiments — registration from ratings, status transitions, and prompt summary."""

from __future__ import annotations

import pytest

from tessera.experiments import (
    Experiment,
    ExperimentStore,
    _pattern_key,
    register_from_ratings,
    summarize_for_prompt,
    evaluate_pending,
)


def _bp(title: str, pattern: str = "x" * 50, dim: str = "prompting_style") -> dict:
    return {
        "title": title,
        "pattern": pattern,
        "experiment_to_try": f"For one week, {title.lower()}",
        "dimension": dim,
    }


def test_register_useful_creates_experiment(tmp_path):
    store = ExperimentStore(data_dir=tmp_path)
    bps = [_bp("Goal-first prompts"), _bp("Sample-first artifacts")]
    ratings = [
        {"index": 0, "key": _pattern_key(bps[0]), "rating": "useful"},
        {"index": 1, "key": _pattern_key(bps[1]), "rating": "wrong"},
    ]
    registered = register_from_ratings(ratings, bps, "run-001", store)
    assert len(registered) == 1
    assert registered[0].title == "Goal-first prompts"
    # Wrong-rated pattern doesn't become an experiment
    assert all(e.title != "Sample-first artifacts" for e in store.list("active"))


def test_re_rating_useful_does_not_double_register(tmp_path):
    store = ExperimentStore(data_dir=tmp_path)
    bp = _bp("Goal-first prompts")
    ratings = [{"index": 0, "key": _pattern_key(bp), "rating": "useful"}]
    register_from_ratings(ratings, [bp], "run-001", store)
    register_from_ratings(ratings, [bp], "run-002", store)
    assert len(store.list("active")) == 1


def test_register_resolves_by_key_when_index_drifts(tmp_path):
    """A user rates BP at index 2 in run A; in run B the same BP is at index 5.
    Registration should match by key, not index."""
    store = ExperimentStore(data_dir=tmp_path)
    bps = [_bp(f"pattern-{i}") for i in range(6)]
    target = bps[5]
    # Rating's index is 2 (wrong) but key matches bps[5]
    ratings = [{"index": 2, "key": _pattern_key(target), "rating": "useful"}]
    registered = register_from_ratings(ratings, bps, "run-001", store)
    assert len(registered) == 1
    assert registered[0].title == "pattern-5"


def test_summarize_for_prompt_lists_active(tmp_path):
    store = ExperimentStore(data_dir=tmp_path)
    register_from_ratings(
        [{"index": 0, "key": _pattern_key(_bp("Foo")), "rating": "useful"}],
        [_bp("Foo")],
        "run-001",
        store,
    )
    text = summarize_for_prompt(store)
    assert "Active experiments" in text
    assert "Foo" in text
    assert "experiment:" in text


def test_evaluate_transitions_to_not_tried_on_zero_adherence(tmp_path):
    store = ExperimentStore(data_dir=tmp_path)
    bp = _bp("Foo")
    register_from_ratings(
        [{"index": 0, "key": _pattern_key(bp), "rating": "useful"}],
        [bp],
        "run-001",
        store,
    )
    # Fake one post-baseline narrative + an evaluator that says "no adherence"
    narratives = [{"started_at": "2099-01-01T00:00:00+00:00", "session_id": "x"}]
    evaluate_pending(
        narratives,
        store,
        llm_evaluator=lambda exp, post: {"adherence": "none", "effect": "unknown"},
    )
    assert not store.list("active")
    assert len(store.list("not_tried")) == 1


def test_evaluate_graduates_on_partial_adherence_positive_effect(tmp_path):
    store = ExperimentStore(data_dir=tmp_path)
    bp = _bp("Foo")
    register_from_ratings(
        [{"index": 0, "key": _pattern_key(bp), "rating": "useful"}],
        [bp],
        "run-001",
        store,
    )
    narratives = [{"started_at": "2099-01-01T00:00:00+00:00", "session_id": "x"}]
    evaluate_pending(
        narratives,
        store,
        llm_evaluator=lambda exp, post: {"adherence": "partial", "effect": "positive"},
    )
    assert not store.list("active")
    assert len(store.list("graduated")) == 1


def test_evaluate_marks_inconclusive_after_two_neutral_evals(tmp_path):
    store = ExperimentStore(data_dir=tmp_path)
    bp = _bp("Foo")
    register_from_ratings(
        [{"index": 0, "key": _pattern_key(bp), "rating": "useful"}],
        [bp],
        "run-001",
        store,
    )
    narratives = [{"started_at": "2099-01-01T00:00:00+00:00", "session_id": "x"}]
    # First neutral eval keeps it active
    evaluate_pending(
        narratives,
        store,
        llm_evaluator=lambda exp, post: {"adherence": "partial", "effect": "neutral"},
    )
    assert len(store.list("active")) == 1
    # Second neutral eval transitions to inconclusive
    evaluate_pending(
        narratives,
        store,
        llm_evaluator=lambda exp, post: {"adherence": "partial", "effect": "neutral"},
    )
    assert not store.list("active")
    assert len(store.list("inconclusive")) == 1


def test_evaluate_skips_when_no_post_baseline_narratives(tmp_path):
    store = ExperimentStore(data_dir=tmp_path)
    bp = _bp("Foo")
    register_from_ratings(
        [{"index": 0, "key": _pattern_key(bp), "rating": "useful"}],
        [bp],
        "run-001",
        store,
    )
    # All narratives are BEFORE the baseline cutoff (start of time)
    narratives = [{"started_at": "1999-01-01T00:00:00+00:00", "session_id": "x"}]
    summary = evaluate_pending(narratives, store)
    assert summary["evaluated"] == 0
    assert len(summary["skipped_no_post_baseline_data"]) == 1
    assert len(store.list("active")) == 1
