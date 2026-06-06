"""Tests for the append-only audit logbook."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from tessera.logbook import Logbook


def test_append_writes_one_line(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    eid = lb.append("note", text="hello")
    lines = (tmp_path / "lb.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    ev = json.loads(lines[0])
    assert ev["event"] == "note"
    assert ev["text"] == "hello"
    assert ev["event_id"] == eid
    assert ev["schema_version"] == 1
    assert "ts" in ev


def test_append_only_grows(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    lb.append("note", text="a")
    lb.append("note", text="b")
    lb.append("note", text="c")
    lines = (tmp_path / "lb.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    # Order preserved
    assert [json.loads(line)["text"] for line in lines] == ["a", "b", "c"]


def test_iter_events_filters_by_type(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    lb.append("note", text="a")
    lb.append("run.started", run_slug="r1")
    lb.append("note", text="b")
    lb.append("run.started", run_slug="r2")
    runs = list(lb.iter_events(event_type="run.started"))
    assert [e["run_slug"] for e in runs] == ["r1", "r2"]


def test_iter_events_filters_by_since(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    # Patch the timestamps directly to test the since filter
    lb.append("note", text="early")
    lb.append("note", text="late")
    raw = (tmp_path / "lb.jsonl").read_text().strip().splitlines()
    e1 = json.loads(raw[0])
    e2 = json.loads(raw[1])
    e1["ts"] = "2026-01-01T00:00:00+00:00"
    e2["ts"] = "2026-06-01T00:00:00+00:00"
    (tmp_path / "lb.jsonl").write_text(
        json.dumps(e1) + "\n" + json.dumps(e2) + "\n"
    )
    events = list(lb.iter_events(since="2026-05-01"))
    assert len(events) == 1
    assert events[0]["text"] == "late"


def test_log_run_lifecycle(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    lb.log_run_started(run_slug="r1", lookback_days=7, model="sonnet")
    lb.log_run_completed(
        run_slug="r1", narratives_processed=50,
        observations_count=8, behavioral_patterns_count=12,
        fabricated_refs=0,
    )
    events = list(lb.iter_events())
    assert events[0]["event"] == "run.started"
    assert events[0]["model"] == "sonnet"
    assert events[1]["event"] == "run.completed"
    assert events[1]["observations_count"] == 8


def test_log_rating_routes_useful_to_accepted(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    lb.log_rating(run_slug="r1", key="k1", title="t1", rating="useful")
    lb.log_rating(run_slug="r1", key="k2", title="t2", rating="wrong")
    events = list(lb.iter_events())
    assert events[0]["event"] == "recommendation.accepted"
    assert events[1]["event"] == "recommendation.declined"
    assert events[1]["rating"] == "wrong"


def test_log_experiment_transitions(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    lb.log_experiment_registered(exp_id="e1", key="k1", title="Try X", dimension="prompting")
    lb.log_experiment_evaluated(
        exp_id="e1", title="Try X",
        adherence="partial", effect="positive",
        adherence_evidence="seen in 4 sessions",
        effect_evidence="dead_ends dropped 30%",
        recommendation="graduate",
    )
    lb.log_experiment_transition(new_status="graduated", exp_id="e1", title="Try X")
    events = list(lb.iter_events())
    assert [e["event"] for e in events] == [
        "experiment.registered",
        "experiment.evaluated",
        "experiment.graduated",
    ]


def test_event_id_is_stable(tmp_path):
    """Two events with the same payload + ts should NOT have the same id
    because event_id includes the payload — but the function is deterministic
    given a fixed input."""
    lb = Logbook(tmp_path / "lb.jsonl")
    eid1 = lb.append("note", text="a")
    eid2 = lb.append("note", text="a")
    # Different ts means different ids
    assert eid1 != eid2
    # But both 12 chars
    assert len(eid1) == 12 and len(eid2) == 12


def test_iter_events_skips_malformed_lines(tmp_path):
    lb = Logbook(tmp_path / "lb.jsonl")
    lb.append("note", text="good")
    # Corrupt the file with a bad line
    with open(tmp_path / "lb.jsonl", "a") as f:
        f.write("not json\n")
        f.write('{"event": "note", "text": "also good"}\n')
    events = list(lb.iter_events())
    assert len(events) == 2  # malformed line skipped
    assert events[0]["text"] == "good"
    assert events[1]["text"] == "also good"
