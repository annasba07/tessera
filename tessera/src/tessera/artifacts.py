"""Detect agent-skill + project-CLAUDE.md files the user creates or
edits between Tessera runs, so the closed loop knows what action the
user actually took in response to past recommendations.

Why this exists: the synthesis surfaces patterns, the user (sometimes)
acts on them by writing new SKILL.md or CLAUDE.md files, then the next
weekly evaluator tries to judge whether things improved. Without this
module, step 2 ("user took an action") is invisible to Tessera — the
evaluator has to infer adherence purely from session behavior, which
is noisy. With this module, the synthesis prompt for next week can
literally include "you built skill `pulse-auth-preflight` on 2026-06-19"
so the model can credit/discredit experiments based on what you did.

Privacy: we hash file content (SHA-256), never store it. A short title
snippet (first ~80 chars of the first non-frontmatter line) is kept
for human-readable log lines. Hashes are 16-char prefixes — collision
risk is negligible for a single user's small artifact set.

What we watch (all optional — missing files just produce no events):

  ~/.claude/skills/<name>/SKILL.md       (global Claude Code skills)
  ~/.claude/CLAUDE.md                    (global Claude Code instructions)
  <project_path>/CLAUDE.md               (project-level instructions —
                                          project_path comes from narratives)

Future targets (not implemented yet):

  ~/.codex/skills/                       (Codex skills if/when format stable)
  ~/.gemini/skills/                      (Gemini skills if/when format stable)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_STATE_PATH = Path.home() / ".config" / "tessera" / "artifact-tracker.json"
SKILLS_ROOT = Path.home() / ".claude" / "skills"
GLOBAL_CLAUDEMD = Path.home() / ".claude" / "CLAUDE.md"


# Event kinds emitted to the logbook. Distinct from `artifact.created`
# vs `artifact.modified` (those are EVENT NAMES) — these are the KIND
# field so the dashboard / logbook reader can filter by what the user
# actually built.
KIND_SKILL = "skill"
KIND_CLAUDEMD_GLOBAL = "claudemd_global"
KIND_CLAUDEMD_PROJECT = "claudemd_project"


@dataclass
class ArtifactEvent:
    """An artifact change worth logging."""
    event: str         # "artifact.created" | "artifact.modified"
    kind: str          # KIND_SKILL | KIND_CLAUDEMD_GLOBAL | KIND_CLAUDEMD_PROJECT
    path: str          # absolute filesystem path
    hash: str          # 16-char sha256 prefix of current content
    prev_hash: str | None  # previous hash if modified, None if created
    title_hint: str    # first non-frontmatter line (≤80 chars), for log display

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "kind": self.kind,
            "path": self.path,
            "hash": self.hash,
            "prev_hash": self.prev_hash,
            "title_hint": self.title_hint,
        }


def _hash_file(path: Path) -> str:
    """16-char prefix of SHA-256. Sufficient to detect change without
    storing the full digest (which would still be a privacy issue if
    the file content were sensitive and the hash space narrow)."""
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:16]


def _title_hint(path: Path) -> str:
    """Pull a short human-readable label from the file.

    For markdown with YAML frontmatter (SKILL.md format), skip the
    frontmatter and grab the first heading or non-blank line. Cap at
    80 chars."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Strip YAML frontmatter if present.
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5:]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Prefer a heading line stripped of leading hashes.
        line = re.sub(r"^#+\s*", "", line)
        return line[:80]
    return ""


class ArtifactTracker:
    """Persistent tracker mapping artifact path → last-seen hash.

    State is serialized as JSON for `grep`/`jq` debuggability. Concurrent
    writes are not protected (Tessera invocations don't overlap)."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or DEFAULT_STATE_PATH
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"version": 1, "artifacts": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "artifacts": {}}
        data.setdefault("artifacts", {})
        return data

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _check_file(self, path: Path, kind: str) -> ArtifactEvent | None:
        """Compare current hash against stored hash. Returns an event if
        the file is new or modified; None if unchanged. Updates state
        in either case (so last_seen advances)."""
        path_str = str(path)
        try:
            current_hash = _hash_file(path)
        except OSError:
            return None
        now = datetime.now(timezone.utc).isoformat()
        prev = self.state["artifacts"].get(path_str)
        if prev is None:
            self.state["artifacts"][path_str] = {
                "hash": current_hash,
                "kind": kind,
                "first_seen": now,
                "last_seen": now,
            }
            return ArtifactEvent(
                event="artifact.created",
                kind=kind,
                path=path_str,
                hash=current_hash,
                prev_hash=None,
                title_hint=_title_hint(path),
            )
        if prev["hash"] != current_hash:
            prev_hash = prev["hash"]
            prev["hash"] = current_hash
            prev["last_seen"] = now
            return ArtifactEvent(
                event="artifact.modified",
                kind=kind,
                path=path_str,
                hash=current_hash,
                prev_hash=prev_hash,
                title_hint=_title_hint(path),
            )
        prev["last_seen"] = now
        return None

    def scan(
        self,
        project_paths: Iterable[Path] | None = None,
        *,
        skills_root: Path = SKILLS_ROOT,
        global_claudemd: Path = GLOBAL_CLAUDEMD,
    ) -> list[ArtifactEvent]:
        """Scan all watch targets and return the list of new + modified
        events. Also persists state so subsequent calls only report
        further changes."""
        events: list[ArtifactEvent] = []

        # 1. Global Claude Code skills: one dir per skill, each with SKILL.md
        if skills_root.exists():
            for skill_md in sorted(skills_root.glob("*/SKILL.md")):
                ev = self._check_file(skill_md, KIND_SKILL)
                if ev:
                    events.append(ev)

        # 2. Global CLAUDE.md
        if global_claudemd.exists():
            ev = self._check_file(global_claudemd, KIND_CLAUDEMD_GLOBAL)
            if ev:
                events.append(ev)

        # 3. Project-level CLAUDE.md — one per unique project_path from
        # the user's narratives. Dedup since two sessions in the same
        # project both report the same project_path.
        seen_proj_paths: set[Path] = set()
        for proj in project_paths or ():
            try:
                resolved = Path(proj).expanduser().resolve()
            except (RuntimeError, OSError):
                continue
            if resolved in seen_proj_paths:
                continue
            seen_proj_paths.add(resolved)
            candidate = resolved / "CLAUDE.md"
            if candidate.exists():
                ev = self._check_file(candidate, KIND_CLAUDEMD_PROJECT)
                if ev:
                    events.append(ev)

        self._save_state()
        return events


def discover_project_paths_from_narrative_cache(cache_dir: Path) -> set[Path]:
    """Pull unique project_path values from cached per-session narratives.

    The narrator records a `project_path` field for every session it
    extracts. Reading the cache (rather than the live narratives dir)
    lets us see EVERY project the user has worked in across all runs,
    not just the current one's sessions."""
    out: set[Path] = set()
    if not cache_dir.exists():
        return out
    for p in cache_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        proj = data.get("project_path")
        if not proj:
            continue
        try:
            out.add(Path(proj).expanduser())
        except (RuntimeError, OSError):
            continue
    return out


def summarize_recent_events(events: list[ArtifactEvent]) -> str:
    """Human-readable summary, used by the synthesis prompt context.

    Format: one line per event, prefixed by emoji or symbol indicating
    create vs modify, with kind + title_hint + path. Empty string if
    no events (caller can skip the section)."""
    if not events:
        return ""
    lines: list[str] = []
    for ev in events:
        sym = "+" if ev.event == "artifact.created" else "~"
        title = ev.title_hint or "(no title)"
        lines.append(f"  {sym} [{ev.kind}] {title}  ({ev.path})")
    return "\n".join(lines)
