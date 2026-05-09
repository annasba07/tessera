"""Outcome enrichment: join per-session narratives to git/GitHub ground truth.

For each narrative we want to know what *actually* happened to the work after
the session ended — did it ship, get reverted, accumulate fixup commits, pass
CI? Friction without outcome is half the story; "this took 200 events" is
useful, "this took 200 events AND the PR needed 4 fixup commits within a
week" is much more useful.

Signals we capture (best-effort, all optional — gracefully skip when absent):

  * Branch lifecycle — did `git_branch_last` get merged, deleted, or is it
    still open with commits after the session ended?
  * File churn — for each file the session touched, count subsequent commits
    in the 14d window after session end. Distinguish ordinary follow-ups
    from `fix:` / `revert:` / `hotfix:` subjects.
  * Revert detection — any commit subject matching `^Revert\b` or `revert:`
    that touches the same files in 14d.
  * PR state — if a PR number can be extracted from the narrative text
    fields, look it up via `gh pr view` and capture state, mergedAt, CI
    rollup, and review decision. `gh` is opt-in: graceful no-op if the
    binary is missing or unauthenticated.

Output goes inline on the narrative as `outcome` + `outcome_signal`. Cached
on the narrative, refreshed only with `--force`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
WINDOW_DAYS = 14
GH_TIMEOUT_S = 30
GIT_TIMEOUT_S = 20

# Subjects that signal "this commit was cleaning up after the original work."
# Two layers — the strict revert pattern, and a softer fixup heuristic.
_REVERT_RE = re.compile(r"^revert\b|^Revert\b", re.MULTILINE)
_FIXUP_RE = re.compile(
    r"\b(fix|fixup|hotfix|patch|cleanup|followup|follow-up|amend)\b", re.IGNORECASE
)
# Inline PR refs like "#1234" or "PR #1234" anywhere in narrative text.
_PR_NUM_RE = re.compile(r"(?:^|\W)#(\d{1,5})\b")
# Outcome signal taxonomy — one per narrative.
SIGNAL_SHIPPED_CLEAN = "shipped_clean"
SIGNAL_SHIPPED_WITH_FOLLOWUPS = "shipped_with_followups"
SIGNAL_REVERTED = "reverted"
SIGNAL_ABANDONED = "abandoned"
SIGNAL_IN_PROGRESS = "in_progress"
SIGNAL_NO_ARTIFACT = "no_artifact"
SIGNAL_UNAVAILABLE = "unavailable"


def _run_git(repo: Path, *args: str, timeout: int = GIT_TIMEOUT_S) -> tuple[int, str]:
    """Run git in `repo`. Return (returncode, stdout). Stderr swallowed."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return 1, ""


def _run_gh(*args: str, timeout: int = GH_TIMEOUT_S) -> tuple[int, str]:
    """Run gh CLI. Return (returncode, stdout). Empty if gh missing/unauth'd."""
    if not shutil.which("gh"):
        return 1, ""
    try:
        result = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return 1, ""


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def extract_pr_numbers(narrative: dict) -> list[int]:
    """Best-effort extraction of PR numbers from narrative text fields.

    Looks at goal, lesson_for_user, lesson_for_agent, notable, counterfactual.
    Dedupes; preserves order of first appearance. Returns at most 5 numbers
    (more than that is usually noise from unrelated issue refs).
    """
    text_fields = ("goal", "lesson_for_user", "lesson_for_agent", "notable", "counterfactual")
    seen: set[int] = set()
    out: list[int] = []
    for field in text_fields:
        v = narrative.get(field)
        if not isinstance(v, str):
            continue
        for m in _PR_NUM_RE.finditer(v):
            try:
                num = int(m.group(1))
            except ValueError:
                continue
            # Drop tiny "ref" numbers like "#1" — almost always not a PR
            if num < 10 or num in seen:
                continue
            seen.add(num)
            out.append(num)
            if len(out) >= 5:
                return out
    return out


def lookup_repo_remote(project_path: Path) -> tuple[str | None, str | None]:
    """Return (owner, name) for the repo's `origin` remote, or (None, None).

    Parses both `git@github.com:owner/name(.git)?` and `https://github.com/...`
    forms. We restrict to GitHub since that's where `gh` works.
    """
    rc, url = _run_git(project_path, "remote", "get-url", "origin")
    if rc != 0 or not url.strip():
        return (None, None)
    url = url.strip()
    # SSH form
    m = re.match(r"git@github\.com:([^/]+)/([^/.]+)(?:\.git)?$", url)
    if m:
        return (m.group(1), m.group(2))
    # HTTPS form
    m = re.match(r"https?://github\.com/([^/]+)/([^/.]+?)(?:\.git)?/?$", url)
    if m:
        return (m.group(1), m.group(2))
    return (None, None)


def lookup_branch_outcome(
    project_path: Path, branch: str | None, session_end: datetime | None
) -> dict[str, Any]:
    """Inspect what happened to `branch` after the session ended.

    Returns a dict with branch_state and commits_after_session — empty dict
    if the path isn't a git repo or the branch is missing. We treat 'main',
    'master', and detached HEAD as 'long-lived' (no abandon signal).
    """
    if not branch or branch in ("HEAD", "main", "master") or not project_path.is_dir():
        return {}
    rc, _ = _run_git(project_path, "rev-parse", "--git-dir")
    if rc != 0:
        return {}

    out: dict[str, Any] = {"branch": branch}

    # Does the branch still exist locally?
    rc, _ = _run_git(project_path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    out["branch_exists_local"] = rc == 0
    rc, _ = _run_git(
        project_path, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"
    )
    out["branch_exists_remote"] = rc == 0

    # Commits on this branch after session_end
    if session_end and (out["branch_exists_local"] or out["branch_exists_remote"]):
        ref = (
            f"refs/heads/{branch}"
            if out["branch_exists_local"]
            else f"refs/remotes/origin/{branch}"
        )
        rc, log = _run_git(
            project_path,
            "log",
            ref,
            f"--since={session_end.isoformat()}",
            "--format=%H|%ct|%s",
        )
        commits = []
        if rc == 0:
            for line in log.strip().splitlines():
                parts = line.split("|", 2)
                if len(parts) == 3:
                    commits.append({"sha": parts[0], "ts": int(parts[1]), "subject": parts[2]})
        out["commits_after_session"] = len(commits)
        if commits:
            out["commits_after_subjects"] = [c["subject"][:140] for c in commits[:6]]

    # Was branch merged into main/master?
    for trunk in ("main", "master"):
        rc_t, _ = _run_git(
            project_path, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{trunk}"
        )
        if rc_t != 0:
            continue
        # `branch --merged` returns merged branches into the given commit.
        rc, merged_into = _run_git(
            project_path,
            "branch",
            "-a",
            "--merged",
            f"origin/{trunk}",
            "--format=%(refname:short)",
        )
        if rc == 0 and any(
            line.strip() in {branch, f"origin/{branch}"}
            for line in merged_into.splitlines()
        ):
            out["merged_into"] = trunk
            break

    return out


def lookup_files_churn(
    project_path: Path,
    files: list[str],
    session_end: datetime | None,
    window_days: int = WINDOW_DAYS,
) -> dict[str, Any]:
    """For each file the session touched, count subsequent commits in window.

    Distinguishes 'fixup-shape' commits (subject matches fix/revert/hotfix)
    from ordinary follow-ups. Returns aggregates, not per-file detail.
    """
    if not files or not session_end or not project_path.is_dir():
        return {}
    rc, _ = _run_git(project_path, "rev-parse", "--git-dir")
    if rc != 0:
        return {}

    since = session_end.isoformat()
    until = (session_end + timedelta(days=window_days)).isoformat()

    # Cap files at 25 — beyond that the git invocation gets slow and noisy
    files_to_check = files[:25]
    rc, log = _run_git(
        project_path,
        "log",
        f"--since={since}",
        f"--until={until}",
        "--format=%H|%s",
        "--",
        *files_to_check,
    )
    if rc != 0:
        return {}
    total = 0
    fixup = 0
    revert = 0
    seen_shas: set[str] = set()
    for line in log.strip().splitlines():
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        sha, subject = parts
        if sha in seen_shas:
            continue
        seen_shas.add(sha)
        total += 1
        if _REVERT_RE.search(subject):
            revert += 1
        elif _FIXUP_RE.search(subject):
            fixup += 1
    return {
        "files_checked": len(files_to_check),
        "window_days": window_days,
        "commits_touching_files": total,
        "fixup_shape_commits": fixup,
        "revert_commits": revert,
    }


def lookup_pr_outcome(owner: str, name: str, pr_number: int) -> dict[str, Any]:
    """Hit `gh pr view` to capture state. Empty dict on any failure."""
    rc, payload = _run_gh(
        "pr",
        "view",
        str(pr_number),
        "--repo",
        f"{owner}/{name}",
        "--json",
        "state,mergedAt,mergeCommit,reviewDecision,statusCheckRollup,additions,deletions,changedFiles,closedAt,isDraft",
    )
    if rc != 0 or not payload.strip():
        return {}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    # Distill the CI rollup to a pass/fail/pending bool — full detail isn't
    # useful for synthesis-time pattern matching.
    ci_status: str | None = None
    rollup = data.get("statusCheckRollup") or []
    if rollup:
        results = {(c.get("conclusion") or c.get("state") or "").upper() for c in rollup}
        if results & {"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED"}:
            ci_status = "failed"
        elif results & {"PENDING", "IN_PROGRESS", "QUEUED"}:
            ci_status = "pending"
        elif results & {"SUCCESS"}:
            ci_status = "passed"

    return {
        "number": pr_number,
        "state": data.get("state"),
        "merged_at": data.get("mergedAt"),
        "closed_at": data.get("closedAt"),
        "is_draft": data.get("isDraft"),
        "merge_commit": (data.get("mergeCommit") or {}).get("oid"),
        "review_decision": data.get("reviewDecision"),
        "ci_status": ci_status,
        "additions": data.get("additions"),
        "deletions": data.get("deletions"),
        "changed_files": data.get("changedFiles"),
    }


def _summarize_signal(outcome: dict) -> str:
    """Roll the captured dict into a single signal label.

    Priority order: explicit revert > failed/changes-requested PR (worst-case
    wins when there's a mix) > merged PR + clean churn > merged PR + fixup
    churn > open PR / commits-still-coming > abandoned > no_artifact.

    The PR aggregation is worst-case-wins: if any referenced PR is failing,
    treat the whole session as 'in_progress' even if a sibling PR merged —
    a partial ship-with-failures is more honest than a 'shipped_clean'
    that hides the open work.
    """
    churn = outcome.get("files_churn") or {}
    if churn.get("revert_commits", 0) > 0:
        return SIGNAL_REVERTED

    pr_outcomes = [p for p in (outcome.get("prs") or []) if p.get("state")]
    if pr_outcomes:
        states = {p["state"] for p in pr_outcomes}
        ci_failed = any(p.get("ci_status") == "failed" for p in pr_outcomes)
        changes_requested = any(
            p.get("review_decision") == "CHANGES_REQUESTED" for p in pr_outcomes
        )
        # Worst-case: any open PR (especially failing) outweighs a merged sibling.
        if "OPEN" in states or ci_failed or changes_requested:
            # If at least one merged AND one is still failing, that's a partial
            # ship — call it in_progress with a noted churn signal.
            return SIGNAL_IN_PROGRESS
        if states == {"MERGED"}:
            if churn.get("fixup_shape_commits", 0) > 0:
                return SIGNAL_SHIPPED_WITH_FOLLOWUPS
            return SIGNAL_SHIPPED_CLEAN
        if "CLOSED" in states:
            return SIGNAL_ABANDONED

    branch = outcome.get("branch") or {}
    if branch.get("merged_into"):
        if churn.get("fixup_shape_commits", 0) > 0:
            return SIGNAL_SHIPPED_WITH_FOLLOWUPS
        return SIGNAL_SHIPPED_CLEAN
    if (
        branch.get("branch")
        and not branch.get("branch_exists_local")
        and not branch.get("branch_exists_remote")
    ):
        # Branch deleted without a merge signal — likely abandoned (or merged
        # and cleaned up before our window catches it; rare).
        return SIGNAL_ABANDONED
    if branch.get("commits_after_session", 0) > 0:
        return SIGNAL_IN_PROGRESS

    return SIGNAL_NO_ARTIFACT


def enrich_narrative_with_outcome(
    narrative: dict, *, use_gh: bool = True
) -> dict[str, Any]:
    """Compute the outcome dict for one narrative. Pure function — caller
    decides whether to write it back to the file."""
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "looked_up_at": datetime.now(timezone.utc).isoformat(),
        "window_days": WINDOW_DAYS,
    }

    project_path_str = narrative.get("project_path")
    if not project_path_str:
        out["outcome_signal"] = SIGNAL_UNAVAILABLE
        out["reason"] = "no project_path on narrative"
        return out
    project_path = Path(project_path_str).expanduser()
    if not project_path.is_dir():
        out["outcome_signal"] = SIGNAL_UNAVAILABLE
        out["reason"] = "project_path no longer exists locally"
        out["project_path"] = project_path_str
        return out

    session_end = _parse_iso(narrative.get("ended_at"))

    # Branch lifecycle
    branch_info = lookup_branch_outcome(
        project_path, narrative.get("git_branch_last"), session_end
    )
    if branch_info:
        out["branch"] = branch_info

    # File churn over `top_files_touched` (already prioritized in narrative)
    files = narrative.get("top_files_touched") or []
    # `top_files_touched` is sometimes [{path, count}, ...] and sometimes [str]
    file_paths = [
        f["path"] if isinstance(f, dict) else f
        for f in files
        if (isinstance(f, dict) and f.get("path")) or isinstance(f, str)
    ]
    if file_paths and session_end:
        churn = lookup_files_churn(project_path, file_paths, session_end)
        if churn:
            out["files_churn"] = churn

    # PR lookups (capped at 3 PRs to keep runtime sane)
    if use_gh:
        owner, name = lookup_repo_remote(project_path)
        if owner and name:
            pr_numbers = extract_pr_numbers(narrative)
            pr_outcomes = []
            for pr_num in pr_numbers[:3]:
                pr_data = lookup_pr_outcome(owner, name, pr_num)
                if pr_data:
                    pr_outcomes.append(pr_data)
            if pr_outcomes:
                out["prs"] = pr_outcomes
            elif pr_numbers:
                # We extracted refs but couldn't resolve any — note it
                out["pr_refs_unresolved"] = pr_numbers[:3]

    out["outcome_signal"] = _summarize_signal(out)
    return out


def enrich_directory(
    narratives_dir: Path,
    *,
    use_gh: bool = True,
    force: bool = False,
    max_age_days: int = 7,
) -> dict[str, Any]:
    """Walk a directory of narratives, enrich each with outcome data inline.

    Idempotent: skips narratives whose `outcome.looked_up_at` is fresher
    than `max_age_days`, unless `force=True`. Returns a summary dict.
    """
    fresh_cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    summary = {
        "total": 0,
        "enriched": 0,
        "skipped_fresh": 0,
        "skipped_no_path": 0,
        "signal_counts": {},
    }
    signal_counts: dict[str, int] = {}

    for path in sorted(narratives_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("session_id"):
            continue
        summary["total"] += 1

        existing = data.get("outcome") or {}
        existing_ts = _parse_iso(existing.get("looked_up_at"))
        if not force and existing_ts and existing_ts > fresh_cutoff:
            summary["skipped_fresh"] += 1
            sig = existing.get("outcome_signal") or SIGNAL_UNAVAILABLE
            signal_counts[sig] = signal_counts.get(sig, 0) + 1
            continue

        outcome = enrich_narrative_with_outcome(data, use_gh=use_gh)
        data["outcome"] = outcome
        sig = outcome.get("outcome_signal") or SIGNAL_UNAVAILABLE
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

        try:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary["enriched"] += 1
        except OSError:
            continue

    summary["signal_counts"] = dict(
        sorted(signal_counts.items(), key=lambda kv: -kv[1])
    )
    return summary
