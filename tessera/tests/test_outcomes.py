"""Tests for outcome enrichment — the early-no-artifact-fallthrough fix
and the new signal taxonomy (exploration / non_repo / unshipped /
shipped_direct).
"""

from __future__ import annotations

import subprocess

import pytest

from tessera.narratives.outcomes import (
    SIGNAL_EXPLORATION,
    SIGNAL_NON_REPO,
    SIGNAL_SHIPPED_DIRECT,
    SIGNAL_UNAVAILABLE,
    SIGNAL_UNSHIPPED,
    enrich_narrative_with_outcome,
    extract_pr_numbers,
)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(path):
    """Initialize a tiny git repo with one commit on main."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "test")
    (path / "README.md").write_text("seed\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-q", "-m", "init")


def test_unavailable_when_project_path_missing():
    out = enrich_narrative_with_outcome(
        {"session_id": "x", "project_path": "/no/such/path/xyz", "ended_at": "2026-01-01T00:00:00+00:00"},
        use_gh=False,
    )
    assert out["outcome_signal"] == SIGNAL_UNAVAILABLE


def test_non_repo_when_project_path_isnt_git(tmp_path):
    (tmp_path / "scratch").mkdir()
    out = enrich_narrative_with_outcome(
        {
            "session_id": "x",
            "project_path": str(tmp_path / "scratch"),
            "ended_at": "2026-01-01T00:00:00+00:00",
        },
        use_gh=False,
    )
    assert out["outcome_signal"] == SIGNAL_NON_REPO


def test_exploration_when_no_files_touched(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    out = enrich_narrative_with_outcome(
        {
            "session_id": "x",
            "project_path": str(repo),
            "ended_at": "2026-01-01T00:00:00+00:00",
            "top_files_touched": [],
        },
        use_gh=False,
    )
    assert out["outcome_signal"] == SIGNAL_EXPLORATION


def test_shipped_direct_when_trunk_commits_touch_session_files(tmp_path):
    """The ship signal we recover: solo work that lands on main without a PR."""
    from datetime import datetime, timezone, timedelta

    repo = tmp_path / "repo"
    _make_repo(repo)
    # Session "ended" 1 hour before the commit lands — well within the
    # [-1d, +14d] window the trunk lookup uses.
    session_end = datetime.now(timezone.utc) - timedelta(hours=1)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "build.sh").write_text("echo hi\n")
    _git(repo, "add", "scripts/build.sh")
    _git(repo, "commit", "-q", "-m", "feat: add build script")

    out = enrich_narrative_with_outcome(
        {
            "session_id": "x",
            "project_path": str(repo),
            "git_branch_last": "main",
            "ended_at": session_end.isoformat(),
            "top_files_touched": [{"path": str(repo / "scripts" / "build.sh"), "count": 5}],
        },
        use_gh=False,
    )
    assert out["outcome_signal"] == SIGNAL_SHIPPED_DIRECT
    assert out.get("trunk_commits", {}).get("trunk_commits_in_window") == 1


def test_unshipped_when_files_touched_but_no_commits(tmp_path):
    """Touched files but the work didn't land in 14d — drafts, abandoned scratch."""
    repo = tmp_path / "repo"
    _make_repo(repo)
    out = enrich_narrative_with_outcome(
        {
            "session_id": "x",
            "project_path": str(repo),
            "git_branch_last": "main",
            "ended_at": "2026-01-01T00:00:00+00:00",
            "top_files_touched": [{"path": str(repo / "fictional-file.py"), "count": 3}],
        },
        use_gh=False,
    )
    assert out["outcome_signal"] == SIGNAL_UNSHIPPED


def test_extract_pr_numbers_from_text_fields():
    n = {
        "goal": "Ship PR #244 (show_text feature)",
        "lesson_for_user": "see #295 for the fix; also #1 (too small to be a real PR)",
        "notable": "follow-up in #5001",
    }
    refs = extract_pr_numbers(n)
    # #1 dropped (below 10), others kept in order of first appearance
    assert refs == [244, 295, 5001]
    assert all(r >= 10 for r in refs)
