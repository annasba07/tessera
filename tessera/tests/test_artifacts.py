"""Tests for artifact-tracker.

Covers the loop: a fresh tracker reports every existing artifact as
created; editing a file flips it to modified; unchanged files emit
nothing on rescan; project paths discovered from narratives drive
where to look for CLAUDE.md files.
"""
from pathlib import Path

from tessera.artifacts import (
    KIND_CLAUDEMD_GLOBAL,
    KIND_CLAUDEMD_PROJECT,
    KIND_SKILL,
    ArtifactTracker,
    discover_project_paths_from_narrative_cache,
    summarize_recent_events,
)


def _make_skill(skills_root: Path, name: str, body: str = "stub") -> Path:
    sd = skills_root / name
    sd.mkdir(parents=True, exist_ok=True)
    f = sd / "SKILL.md"
    f.write_text(f"---\nname: {name}\ndescription: {body}\n---\n\n# {name}\n\n{body}\n")
    return f


def test_first_scan_reports_all_skills_as_created(tmp_path):
    skills = tmp_path / "skills"
    _make_skill(skills, "pulse-auth-preflight")
    _make_skill(skills, "deep-investigation")
    tracker = ArtifactTracker(state_path=tmp_path / "state.json")
    events = tracker.scan(skills_root=skills, global_claudemd=tmp_path / "nope.md")
    assert len(events) == 2
    assert all(e.event == "artifact.created" for e in events)
    assert all(e.kind == KIND_SKILL for e in events)
    # Title hint comes from the first non-frontmatter heading.
    titles = {e.title_hint for e in events}
    assert "pulse-auth-preflight" in titles


def test_second_scan_reports_nothing_when_unchanged(tmp_path):
    skills = tmp_path / "skills"
    _make_skill(skills, "demo-capture")
    tracker = ArtifactTracker(state_path=tmp_path / "state.json")
    first = tracker.scan(skills_root=skills, global_claudemd=tmp_path / "nope.md")
    assert len(first) == 1
    second = tracker.scan(skills_root=skills, global_claudemd=tmp_path / "nope.md")
    assert second == []


def test_modifying_existing_skill_emits_modified_event(tmp_path):
    skills = tmp_path / "skills"
    skill_path = _make_skill(skills, "pulse-fallback-pivot", body="v1 stub")
    tracker = ArtifactTracker(state_path=tmp_path / "state.json")
    tracker.scan(skills_root=skills, global_claudemd=tmp_path / "nope.md")
    # User edits the skill — different body, different hash.
    skill_path.write_text(skill_path.read_text() + "\nUpdated step here.\n")
    events = tracker.scan(skills_root=skills, global_claudemd=tmp_path / "nope.md")
    assert len(events) == 1
    assert events[0].event == "artifact.modified"
    assert events[0].prev_hash is not None
    assert events[0].prev_hash != events[0].hash


def test_project_claudemd_tracked_when_project_path_supplied(tmp_path):
    skills = tmp_path / "skills"
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "CLAUDE.md").write_text("# myproj\nProject-specific rules.\n")
    tracker = ArtifactTracker(state_path=tmp_path / "state.json")
    events = tracker.scan(
        project_paths=[proj],
        skills_root=skills,
        global_claudemd=tmp_path / "nope.md",
    )
    assert len(events) == 1
    assert events[0].kind == KIND_CLAUDEMD_PROJECT
    assert events[0].path.endswith("/myproj/CLAUDE.md")


def test_project_path_dedup(tmp_path):
    """Two narratives in the same project resolve to one project_path each;
    the tracker must not double-count a single CLAUDE.md."""
    proj = tmp_path / "shared"
    proj.mkdir()
    (proj / "CLAUDE.md").write_text("shared project\n")
    tracker = ArtifactTracker(state_path=tmp_path / "state.json")
    events = tracker.scan(
        project_paths=[proj, proj, proj],
        skills_root=tmp_path / "skills",
        global_claudemd=tmp_path / "nope.md",
    )
    assert len(events) == 1


def test_global_claudemd_detection(tmp_path):
    g = tmp_path / "global-CLAUDE.md"
    g.write_text("global rules\n")
    tracker = ArtifactTracker(state_path=tmp_path / "state.json")
    events = tracker.scan(
        skills_root=tmp_path / "skills",
        global_claudemd=g,
    )
    assert len(events) == 1
    assert events[0].kind == KIND_CLAUDEMD_GLOBAL


def test_state_persists_across_tracker_instances(tmp_path):
    """A fresh ArtifactTracker reading the same state file must not
    re-emit events for artifacts seen by a prior instance."""
    state = tmp_path / "state.json"
    skills = tmp_path / "skills"
    _make_skill(skills, "first-skill")
    t1 = ArtifactTracker(state_path=state)
    e1 = t1.scan(skills_root=skills, global_claudemd=tmp_path / "nope.md")
    assert len(e1) == 1
    t2 = ArtifactTracker(state_path=state)
    e2 = t2.scan(skills_root=skills, global_claudemd=tmp_path / "nope.md")
    assert e2 == []


def test_discover_project_paths_reads_narrative_cache(tmp_path):
    cache = tmp_path / "narratives"
    cache.mkdir()
    import json as _json
    (cache / "a.json").write_text(_json.dumps({"session_id": "x", "project_path": "/Users/kaizen/pulse"}))
    (cache / "b.json").write_text(_json.dumps({"session_id": "y", "project_path": "/Users/kaizen/pulse"}))
    (cache / "c.json").write_text(_json.dumps({"session_id": "z", "project_path": "/Users/kaizen/Software-Projects/foo"}))
    paths = discover_project_paths_from_narrative_cache(cache)
    assert len(paths) == 2
    str_paths = {str(p) for p in paths}
    assert "/Users/kaizen/pulse" in str_paths
    assert "/Users/kaizen/Software-Projects/foo" in str_paths


def test_summarize_recent_events_format():
    from tessera.artifacts import ArtifactEvent
    events = [
        ArtifactEvent(
            event="artifact.created",
            kind=KIND_SKILL,
            path="/Users/x/.claude/skills/foo/SKILL.md",
            hash="abc",
            prev_hash=None,
            title_hint="foo: a thing",
        ),
        ArtifactEvent(
            event="artifact.modified",
            kind=KIND_CLAUDEMD_PROJECT,
            path="/Users/x/pulse/CLAUDE.md",
            hash="def",
            prev_hash="aaa",
            title_hint="Pulse rules",
        ),
    ]
    text = summarize_recent_events(events)
    assert "+ [skill]" in text
    assert "~ [claudemd_project]" in text
    assert "foo: a thing" in text
    assert "Pulse rules" in text


def test_summarize_empty_returns_empty_string():
    assert summarize_recent_events([]) == ""
