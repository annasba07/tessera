"""Tests for multi-lens synthesis.

Doesn't hit real LLMs — uses stub backends so the structural plumbing
(parallelism, merge fallback, ref validation) gets exercised without
LLM cost or non-determinism."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from tessera.backends import LLMBackend
from tessera.multilens import LENSES, multilens_synthesize, multilens_synthesize_async


@dataclass
class _StubBackend(LLMBackend):
    """Returns canned JSON; tracks calls so tests can assert dispatch."""
    name: str = "stub"
    default_model: str = "stub-1"
    cli_binary: str = ""
    responses: list[str] | None = None
    received_prompts: list[str] | None = None

    def __post_init__(self):
        if self.responses is None:
            self.responses = []
        if self.received_prompts is None:
            self.received_prompts = []

    async def complete(self, prompt, model, *, system_prompt=None):
        self.received_prompts.append(prompt)
        if not self.responses:
            return '{"behavioral_patterns": [], "meta": {}}'
        return self.responses.pop(0)


def _make_narrative(sid: str, project: str = "kaizen/pulse") -> dict:
    return {
        "session_id": sid,
        "project_label": project,
        "agent": "claude",
        "started_at": "2026-06-19T10:00:00+00:00",
        "ended_at": "2026-06-19T10:15:00+00:00",
        "event_count": 30,
        "active_minutes": 15,
        "tool_call_count": 5,
        "subagent_count": 0,
        "verification_completeness": "claimed_only",
        "lesson_for_user": "remember to re-auth",
    }


def test_lenses_have_required_fields():
    for lens in LENSES:
        assert lens.key
        assert lens.name
        assert len(lens.focus) > 50, f"lens {lens.key} focus too short"
        assert len(lens.signals) > 20, f"lens {lens.key} signals too short"


def test_lens_keys_are_unique():
    keys = [l.key for l in LENSES]
    assert len(keys) == len(set(keys))


def test_dispatches_one_call_per_lens_plus_merge(monkeypatch):
    """N lenses + 1 merge = N+1 backend calls. Validates we're not
    silently doubling up or skipping lenses."""
    narratives = [_make_narrative(f"claude:{i}") for i in range(5)]
    # 7 lenses → 7 lens responses, then 1 merge response = 8 total
    lens_response = json.dumps({
        "behavioral_patterns": [{
            "title": "Stub pattern",
            "pattern": "47 vs 34 example",
            "evidence_refs": ["S001"],
            "confidence": "medium",
            "dimension": "prompting",
        }],
        "meta": {},
    })
    merge_response = json.dumps({
        "behavioral_patterns": [{
            "title": "Merged",
            "pattern": "Stub merged claim with 5 vs 7",
            "evidence_refs": ["S001"],
            "confidence": "high",
            "source_lens": "prompting-1",
            "dimension": "prompting",
        }],
        "meta": {"lens_counts": {}},
    })
    backend = _StubBackend(
        responses=[lens_response] * len(LENSES) + [merge_response],
    )

    result = multilens_synthesize(narratives, backend=backend)

    assert len(backend.received_prompts) == len(LENSES) + 1
    # First N are lens prompts; last is the merge prompt
    for i, lens in enumerate(LENSES):
        assert lens.name in backend.received_prompts[i]
    assert "merge" in backend.received_prompts[-1].lower()


def test_merge_fallback_when_merge_call_raises():
    """If the merge LLM call fails, we should still get a usable result
    (naive concatenation of lens outputs). Multi-lens must not lose
    everything because one call dies."""
    narratives = [_make_narrative(f"claude:{i}") for i in range(3)]
    lens_response = json.dumps({
        "behavioral_patterns": [{
            "title": "Lens-only pattern",
            "pattern": "30 vs 40",
            "evidence_refs": ["S001", "S002"],
            "confidence": "medium",
            "dimension": "prompting",
        }],
        "meta": {},
    })
    # Lens responses are fine; merge response is garbage → fallback
    backend = _StubBackend(
        responses=[lens_response] * len(LENSES) + ["NOT JSON AT ALL"],
    )
    result = multilens_synthesize(narratives, backend=backend)
    # Fallback flattens all lens outputs — should have at least one BP
    # per lens that ran
    assert len(result["behavioral_patterns"]) >= len(LENSES)
    assert "merge failed" in result["meta"]["notes"].lower()


def test_lens_failure_doesnt_kill_run():
    """If one lens response is malformed, others still surface. The
    multilens scaffold should be partition-tolerant."""
    narratives = [_make_narrative(f"claude:{i}") for i in range(3)]
    good = json.dumps({
        "behavioral_patterns": [{
            "title": "Healthy lens result",
            "pattern": "10 vs 20",
            "evidence_refs": ["S001"],
            "confidence": "medium",
            "dimension": "prompting",
        }],
        "meta": {},
    })
    # Half good, half malformed
    responses = []
    for i, _ in enumerate(LENSES):
        responses.append(good if i % 2 == 0 else "NOT JSON")
    merge = json.dumps({
        "behavioral_patterns": [{
            "title": "Merged from healthy lenses",
            "pattern": "10 vs 20 — surfaced from successful lenses",
            "evidence_refs": ["S001"],
            "confidence": "medium",
            "dimension": "prompting",
        }],
        "meta": {},
    })
    backend = _StubBackend(responses=responses + [merge])
    result = multilens_synthesize(narratives, backend=backend)
    # multilens metadata should record which lenses errored
    assert "multilens" in result["meta"]
    errors = result["meta"]["multilens"]["errors"]
    failed_lens_count = len(errors)
    assert failed_lens_count > 0, "expected some lenses to fail in this setup"


def test_aggregate_stats_block_present_in_lens_prompts():
    """Every lens prompt must include the aggregate stats so the model
    grounds its claims in numbers rather than vibes."""
    narratives = [_make_narrative(f"claude:{i}") for i in range(5)]
    backend = _StubBackend(
        responses=[json.dumps({"behavioral_patterns": [], "meta": {}})]
        * (len(LENSES) + 1),
    )
    multilens_synthesize(narratives, backend=backend)
    # Check at least one lens prompt has aggregate stats
    for prompt in backend.received_prompts[: len(LENSES)]:
        # The aggregate block contains text like "## Aggregate stats"
        assert "Aggregate stats" in prompt or "aggregate" in prompt.lower()
