"""Session-state tracker for the coach.

Hooks are invoked as fresh Python processes, so we persist a rolling window
of the last ~50 events per session to disk. One file per session_id:

    ~/.cache/tessera-live/sessions/<session_id>.json

Rule evaluation reads the window, appends the new event, trims, and writes
back. Cooldowns (which rules are currently suppressed) live in the same file
so they survive across hook invocations.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tessera-live" / "sessions"
MAX_EVENTS_PER_SESSION = 80
SESSION_TTL_DAYS = 7


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_input(value) -> str:
    """Stable short hash of a tool input for retry-without-change detection."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    return hashlib.sha1(text[:4000].encode("utf-8", errors="replace")).hexdigest()[:10]


@dataclass
class Event:
    """One PostToolUse / PreToolUse record in a session's rolling window."""

    seq: int
    timestamp: str
    tool_name: str
    tool_input_hash: str
    is_error: bool = False
    error_class: str | None = None
    # Extracted from tool_input for edit-tracking rules.
    target_path: str | None = None


@dataclass
class SessionState:
    session_id: str
    started_at: str
    cwd: str | None = None
    project: str | None = None
    event_seq: int = 0
    events: list[Event] = field(default_factory=list)
    # rule_key -> event_seq until which the rule is suppressed
    suppressed_until: dict[str, int] = field(default_factory=dict)
    # Paths that have been Read in this session; used by edit_without_read rule
    read_files: list[str] = field(default_factory=list)
    # Log of rule firings so `tessera rate` can surface them later.
    fired: list[dict] = field(default_factory=list)

    def append_event(self, event: Event) -> None:
        self.events.append(event)
        if len(self.events) > MAX_EVENTS_PER_SESSION:
            self.events = self.events[-MAX_EVENTS_PER_SESSION:]

    def is_suppressed(self, rule_key: str) -> bool:
        threshold = self.suppressed_until.get(rule_key, 0)
        return self.event_seq < threshold

    def suppress(self, rule_key: str, for_events: int) -> None:
        self.suppressed_until[rule_key] = self.event_seq + for_events

    def log_fire(self, rule_key: str, message: str) -> None:
        self.fired.append(
            {
                "rule_key": rule_key,
                "message": message,
                "event_seq": self.event_seq,
                "fired_at": _now_iso(),
            }
        )


def state_path(session_id: str, cache_dir: Path | None = None) -> Path:
    base = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    safe = session_id.replace("/", "_").replace(":", "_")
    return base / f"{safe}.json"


def load_or_create(
    session_id: str, cwd: str | None = None, cache_dir: Path | None = None
) -> SessionState:
    path = state_path(session_id, cache_dir)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt session file — start fresh rather than crash the hook.
            raw = None
        if raw:
            events = [Event(**e) for e in raw.get("events", []) if isinstance(e, dict)]
            return SessionState(
                session_id=raw.get("session_id", session_id),
                started_at=raw.get("started_at", _now_iso()),
                cwd=raw.get("cwd") or cwd,
                project=raw.get("project"),
                event_seq=int(raw.get("event_seq", len(events))),
                events=events,
                suppressed_until={
                    k: int(v) for k, v in (raw.get("suppressed_until") or {}).items()
                },
                read_files=list(raw.get("read_files") or []),
                fired=list(raw.get("fired") or []),
            )
    project = None
    if cwd:
        project = cwd.rstrip("/").rsplit("/", 1)[-1] or None
    return SessionState(
        session_id=session_id,
        started_at=_now_iso(),
        cwd=cwd,
        project=project,
    )


def save(state: SessionState, cache_dir: Path | None = None) -> None:
    path = state_path(state.session_id, cache_dir)
    payload = asdict(state)
    # Write atomically so a half-written file can't break the next hook call.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def record_event(
    state: SessionState,
    tool_name: str,
    tool_input,
    *,
    is_error: bool = False,
    error_class: str | None = None,
    target_path: str | None = None,
    timestamp: str | None = None,
) -> Event:
    state.event_seq += 1
    event = Event(
        seq=state.event_seq,
        timestamp=timestamp or _now_iso(),
        tool_name=tool_name,
        tool_input_hash=_hash_input(tool_input),
        is_error=bool(is_error),
        error_class=error_class,
        target_path=target_path,
    )
    state.append_event(event)
    return event


def prune_stale_sessions(cache_dir: Path | None = None, ttl_days: int = SESSION_TTL_DAYS) -> int:
    """Delete session files older than ttl_days. Returns count deleted.

    Called opportunistically on hook invocation — keeps the cache from
    growing forever on long-lived machines.
    """
    base = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    if not base.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - (ttl_days * 86400)
    deleted = 0
    for path in base.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            continue
    return deleted
