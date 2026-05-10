"""Tests for the comparative-grounding validator that demotes
behavioral_patterns lacking an X-vs-Y comparison.
"""

from __future__ import annotations

import pytest

from tessera.narratives.synthesis import _has_comparative_grounding, _validate


class TestHasComparativeGrounding:
    def test_descriptive_only_returns_false(self):
        # The exact persona-review B7 case
        assert not _has_comparative_grounding(
            "You work mostly at night — 54K events occur in the evening "
            "and only 8.8K in the afternoon."
        )

    def test_explicit_vs_phrase_returns_true(self):
        assert _has_comparative_grounding(
            "Night sessions show 2.3x context-loss waste vs. afternoon sessions."
        )

    def test_x_out_of_y_returns_true(self):
        assert _has_comparative_grounding(
            "In 6 out of 8 sessions where you used the goal-first format, "
            "outcome was shipped_clean."
        )

    def test_in_sessions_where_returns_true(self):
        assert _has_comparative_grounding(
            "In sessions where you applied the goal-first phrasing, dead-end "
            "rate dropped to 0.6 per investigation."
        )

    def test_arrow_transition_returns_true(self):
        assert _has_comparative_grounding(
            "Verification rate moved from 30% to 65% after the change."
        )

    def test_compared_to_returns_true(self):
        assert _has_comparative_grounding(
            "Codex sessions average 3.1 file reads compared to Claude's 1.4 "
            "for the same animation tasks."
        )

    def test_empty_returns_false(self):
        assert not _has_comparative_grounding("")
        assert not _has_comparative_grounding(None)


class TestValidatorDemotesNonComparative:
    def _ref_map(self):
        return {f"S{i:03d}": f"sess-{i}" for i in range(1, 11)}

    def test_high_confidence_pattern_without_comparison_gets_demoted(self):
        parsed = {
            "behavioral_patterns": [
                {
                    "title": "Night-heavy work",
                    "pattern": "You mostly work at night — 54K events vs 8.8K... wait this has vs.",
                    "confidence": "high",
                    "evidence_refs": ["S001", "S002", "S003"],
                }
            ]
        }
        # That sample DOES have "vs" — let me confirm it doesn't get demoted
        result = _validate(parsed, self._ref_map())
        assert result["behavioral_patterns"][0]["confidence"] == "high"

    def test_truly_descriptive_pattern_gets_demoted_to_low(self):
        parsed = {
            "behavioral_patterns": [
                {
                    "title": "Frequent night sessions",
                    "pattern": "The user logs many sessions during evening hours.",
                    "confidence": "high",
                    "evidence_refs": ["S001", "S002", "S003"],
                }
            ]
        }
        result = _validate(parsed, self._ref_map())
        bp = result["behavioral_patterns"][0]
        assert bp["confidence"] == "low"
        assert bp["non_comparative"] is True
        assert result["meta"]["behavioral_patterns_demoted_non_comparative"] == 1

    def test_comparative_pattern_keeps_confidence(self):
        parsed = {
            "behavioral_patterns": [
                {
                    "title": "Codex over-reads",
                    "pattern": "In Codex animation sessions agents read 3-8 files compared to Claude's 1-2 — adds 30-90s overhead with near-zero quality gain.",
                    "confidence": "medium",
                    "evidence_refs": ["S001", "S002", "S003"],
                }
            ]
        }
        result = _validate(parsed, self._ref_map())
        bp = result["behavioral_patterns"][0]
        assert bp["confidence"] == "medium"
        assert "non_comparative" not in bp

    def test_observations_not_affected(self):
        # Observations are operational, not behavioral — they don't need
        # comparative grounding. The check should only run on behavioral_patterns.
        parsed = {
            "observations": [
                {
                    "title": "Stale chrome lock blocks browser MCP",
                    "claim": "11 sessions hit SingletonLock conflicts.",
                    "confidence": "high",
                    "evidence_refs": ["S001", "S002", "S003"],
                }
            ]
        }
        result = _validate(parsed, self._ref_map())
        assert result["observations"][0]["confidence"] == "high"
