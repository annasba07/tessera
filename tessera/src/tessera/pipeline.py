"""Orchestrate the snapshot + normalize steps for a single `tessera` run.

Rather than copy gigabytes of traces into a staging dir, we point the
existing normalizer at a temp directory of symlinks — it's structurally the
same from the normalizer's point of view and there are no writes into the
real agent directories.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _normalize_script as _norm


DEFAULT_CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
DEFAULT_CODEX_SESSIONS = Path.home() / ".codex" / "sessions"
DEFAULT_GEMINI_TMP = Path.home() / ".gemini" / "tmp"
DEFAULT_GEMINI_PROJECTS_JSON = Path.home() / ".gemini" / "projects.json"


@dataclass
class NormalizeResult:
    output_dir: Path
    summary_path: Path
    sessions_path: Path
    events_path: Path


def _build_symlink_root(
    claude_projects: Path | None,
    codex_sessions: Path | None,
    gemini_tmp: Path | None,
    gemini_projects_json: Path | None,
) -> Path:
    """Make a temp dir shaped like `raw_root` with symlinks into the real stores."""
    tmp = Path(tempfile.mkdtemp(prefix="tessera-"))
    if claude_projects and claude_projects.exists():
        (tmp / "claude").mkdir()
        os.symlink(claude_projects.resolve(), tmp / "claude" / "projects")
    if codex_sessions and codex_sessions.exists():
        (tmp / "codex").mkdir()
        os.symlink(codex_sessions.resolve(), tmp / "codex" / "sessions")
    if gemini_tmp and gemini_tmp.exists():
        (tmp / "gemini").mkdir()
        os.symlink(gemini_tmp.resolve(), tmp / "gemini" / "tmp")
        # projects.json is optional; only link if it exists
        if gemini_projects_json and gemini_projects_json.exists():
            os.symlink(gemini_projects_json.resolve(), tmp / "gemini" / "projects.json")
    return tmp


def normalize_live_traces(
    output_dir: Path,
    claude_projects: Path | None = None,
    codex_sessions: Path | None = None,
    gemini_tmp: Path | None = None,
    gemini_projects_json: Path | None = None,
    max_text_chars: int = 2000,
) -> NormalizeResult:
    """Read live traces and write normalized events.jsonl + sessions.jsonl.

    Args:
        output_dir: Where normalized outputs are written.
        claude_projects: Claude Code projects dir. Defaults to ~/.claude/projects.
        codex_sessions: Codex sessions dir. Defaults to ~/.codex/sessions.
        gemini_tmp: Gemini CLI tmp dir. Defaults to ~/.gemini/tmp.
        gemini_projects_json: Gemini hash→path map. Defaults to ~/.gemini/projects.json.
        max_text_chars: Truncation limit for message + tool preview fields.

    Returns:
        NormalizeResult with paths to the normalized artifacts.
    """
    claude_projects = claude_projects or DEFAULT_CLAUDE_PROJECTS
    codex_sessions = codex_sessions or DEFAULT_CODEX_SESSIONS
    gemini_tmp = gemini_tmp or DEFAULT_GEMINI_TMP
    gemini_projects_json = gemini_projects_json or DEFAULT_GEMINI_PROJECTS_JSON

    if (
        not claude_projects.exists()
        and not codex_sessions.exists()
        and not gemini_tmp.exists()
    ):
        raise FileNotFoundError(
            "No traces found. Looked in "
            f"{claude_projects}, {codex_sessions}, and {gemini_tmp}."
        )

    symlink_root = _build_symlink_root(
        claude_projects, codex_sessions, gemini_tmp, gemini_projects_json
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = _norm.EventWriter(output_dir, max_text_chars)
    try:
        if (symlink_root / "codex").exists():
            _norm.normalize_codex(symlink_root, writer)
        if (symlink_root / "claude").exists():
            _norm.normalize_claude(symlink_root, writer)
        if (symlink_root / "gemini").exists():
            _norm.normalize_gemini(symlink_root, writer)
    finally:
        writer.close()
        shutil.rmtree(symlink_root, ignore_errors=True)

    return NormalizeResult(
        output_dir=output_dir,
        summary_path=output_dir / "summary.json",
        sessions_path=output_dir / "sessions.jsonl",
        events_path=output_dir / "events.jsonl",
    )
