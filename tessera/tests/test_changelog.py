"""Tests for changelog — cross-run diff classification."""

from __future__ import annotations

import pytest

from tessera.changelog import compare_runs


def _obs(title: str, claim: str = "x" * 50, count: int = 5) -> dict:
    return {"title": title, "claim": claim, "supporting_count": count}


def _bp(title: str, pattern: str = "y" * 50, count: int = 5) -> dict:
    return {"title": title, "pattern": pattern, "supporting_count": count}


def test_no_prior_run_returns_empty_diff():
    cl = compare_runs({"observations": [_obs("Foo")]}, prior=None)
    assert cl["compared"] is False
    assert all(len(v) == 0 for v in cl["observations"].values())


def test_new_observation_in_current_only():
    cl = compare_runs(
        {"observations": [_obs("Foo")]},
        prior={"observations": []},
    )
    assert cl["compared"] is True
    assert len(cl["observations"]["new"]) == 1
    assert cl["observations"]["new"][0]["title"] == "Foo"
    assert cl["summary"]["obs_new"] == 1


def test_resolved_observation_in_prior_only():
    cl = compare_runs(
        {"observations": []},
        prior={"observations": [_obs("Foo", count=4)]},
    )
    assert len(cl["observations"]["resolved"]) == 1
    assert cl["observations"]["resolved"][0]["previous_count"] == 4


def test_escalating_when_count_grows_past_noise():
    # Same observation by key, count went from 5 → 9 (delta +4 > noise of 1)
    cl = compare_runs(
        {"observations": [_obs("Foo", count=9)]},
        prior={"observations": [_obs("Foo", count=5)]},
    )
    assert len(cl["observations"]["escalating"]) == 1
    item = cl["observations"]["escalating"][0]
    assert item["delta"] == 4
    assert item["previous_count"] == 5 and item["current_count"] == 9


def test_improving_when_count_drops_past_noise():
    cl = compare_runs(
        {"observations": [_obs("Foo", count=3)]},
        prior={"observations": [_obs("Foo", count=8)]},
    )
    assert len(cl["observations"]["improving"]) == 1
    assert cl["observations"]["improving"][0]["delta"] == -5


def test_continuing_when_count_within_noise():
    # Delta of +1 should be 'continuing', not 'escalating'
    cl = compare_runs(
        {"observations": [_obs("Foo", count=6)]},
        prior={"observations": [_obs("Foo", count=5)]},
    )
    assert len(cl["observations"]["continuing"]) == 1
    assert len(cl["observations"]["escalating"]) == 0


def test_regressed_when_in_current_and_older_but_not_prior():
    cl = compare_runs(
        current={"observations": [_obs("Foo", count=4)]},
        prior={"observations": []},
        older=[{"observations": [_obs("Foo", count=3)]}],
    )
    assert len(cl["observations"]["regressed"]) == 1
    assert cl["observations"]["regressed"][0]["title"] == "Foo"


def test_behavioral_patterns_diffed_independently():
    cl = compare_runs(
        current={
            "observations": [],
            "behavioral_patterns": [_bp("Goal-first prompts", count=10)],
        },
        prior={
            "observations": [],
            "behavioral_patterns": [_bp("Goal-first prompts", count=4)],
        },
    )
    assert len(cl["behavioral_patterns"]["escalating"]) == 1
    assert cl["summary"]["bp_escalating"] == 1


def test_compared_against_metadata_present():
    cl = compare_runs(
        {"observations": []},
        prior={
            "observations": [],
            "meta": {"run_slug": "r1", "generated_at": "2026-01-01T00:00:00Z"},
        },
    )
    assert cl["compared_against_slug"] == "r1"
    assert cl["compared_against_date"] == "2026-01-01T00:00:00Z"
