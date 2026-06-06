"""Append-only audit log of the self-improving loop.

The closed loop has a lot of moving parts:
  - weekly runs producing observations + behavioral patterns
  - ratings accepting/declining recommendations
  - experiments registered, evaluated, graduating or dying
  - effects measured against the user's baseline

Without an explicit logbook, the only way to answer "what did tessera
recommend three weeks ago, did I act on it, and did it work?" is to
piece together history files, experiments dirs, and ratings JSON by
hand. That defeats the audit purpose of the loop.

This module provides one append-only JSONL stream where every loop
decision lands. Each line is one event. Designed so you can:

    # See everything tessera surfaced about prompting style in May
    grep '"event": "insight.surfaced"' ~/.config/tessera/logbook.jsonl \\
        | jq 'select(.dimension == "prompting_style" and (.run_date | startswith("2026-05")))'

    # Compute acceptance rate
    jq -s 'group_by(.event) | map({event: .[0].event, count: length})' \\
        ~/.config/tessera/logbook.jsonl

    # Or use `tessera logbook` for human-readable summaries

Append-only — never edit or delete an entry. If something needs
correction, write a follow-up entry with a `corrects: <event_id>` field.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal


SCHEMA_VERSION = 1
DEFAULT_LOGBOOK_PATH = Path.home() / ".config" / "tessera" / "logbook.jsonl"


# Event types in the audit log. Add new ones in a backwards-compatible way
# (existing consumers should ignore unknown event types, not crash).
EventType = Literal[
    "run.started",
    "run.completed",
    "insight.surfaced",          # one observation OR behavioral pattern from a synthesis
    "recommendation.accepted",   # user rated `useful` → experiment registered
    "recommendation.declined",   # user rated `wrong` | `known` | `skip`
    "experiment.registered",
    "experiment.evaluated",
    "experiment.graduated",
    "experiment.marked_not_tried",
    "experiment.marked_inconclusive",
    "note",                      # free-form human-added or system note
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_event_id(event: dict) -> str:
    """A short deterministic id for an event so corrections can reference it.

    Built from event_type + ts + a hash of the payload. Not cryptographically
    meaningful — just enough to be unique within a logbook.
    """
    seed = (event.get("event", "") + "|" + event.get("ts", "") + "|" + json.dumps(event, sort_keys=True))[:512]
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


@dataclass
class Logbook:
    """Append-only JSONL writer. Cheap to construct; safe to share across
    threads (each write is one os.write() call after open-append-close)."""

    path: Path = field(default_factory=lambda: DEFAULT_LOGBOOK_PATH)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch so consumers can rely on the file existing
        if not self.path.exists():
            self.path.touch()

    def append(self, event_type: str, **fields: Any) -> str:
        """Write one event. Returns the assigned event_id."""
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "ts": _now_iso(),
            "event": event_type,
            **fields,
        }
        event["event_id"] = _stable_event_id(event)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        # Open in append mode each time so concurrent processes don't
        # truncate each other's writes. Atomic for lines under PIPE_BUF
        # (4KB on macOS); our events are well under that.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
        return event["event_id"]

    def iter_events(
        self,
        event_type: str | None = None,
        since: str | None = None,
    ) -> Iterable[dict]:
        """Stream events, optionally filtered by type or since-timestamp.

        `since` is an ISO 8601 string; events with ts >= since are yielded.
        """
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event_type and ev.get("event") != event_type:
                    continue
                if since and (ev.get("ts") or "") < since:
                    continue
                yield ev

    # ----- convenience helpers for the call sites -----

    def log_run_started(self, run_slug: str, lookback_days: int, model: str) -> str:
        return self.append(
            "run.started",
            run_slug=run_slug,
            lookback_days=lookback_days,
            model=model,
        )

    def log_run_completed(
        self,
        run_slug: str,
        narratives_processed: int,
        observations_count: int,
        behavioral_patterns_count: int,
        fabricated_refs: int,
    ) -> str:
        return self.append(
            "run.completed",
            run_slug=run_slug,
            narratives_processed=narratives_processed,
            observations_count=observations_count,
            behavioral_patterns_count=behavioral_patterns_count,
            fabricated_refs=fabricated_refs,
        )

    def log_insight(
        self,
        run_slug: str,
        kind: Literal["observation", "behavioral_pattern"],
        key: str,
        title: str,
        confidence: str | None,
        supporting_count: int,
        category_or_dimension: str | None,
        non_comparative: bool = False,
    ) -> str:
        return self.append(
            "insight.surfaced",
            run_slug=run_slug,
            kind=kind,
            key=key,
            title=title,
            confidence=confidence,
            supporting_count=supporting_count,
            category_or_dimension=category_or_dimension,
            non_comparative=non_comparative,
        )

    def log_rating(
        self,
        run_slug: str,
        key: str,
        title: str,
        rating: Literal["useful", "wrong", "known", "skip"],
    ) -> str:
        event = "recommendation.accepted" if rating == "useful" else "recommendation.declined"
        return self.append(
            event,
            run_slug=run_slug,
            key=key,
            title=title,
            rating=rating,
        )

    def log_experiment_registered(
        self, exp_id: str, key: str, title: str, dimension: str | None
    ) -> str:
        return self.append(
            "experiment.registered",
            experiment_id=exp_id,
            pattern_key=key,
            title=title,
            dimension=dimension,
        )

    def log_experiment_evaluated(
        self,
        exp_id: str,
        title: str,
        adherence: str,
        effect: str,
        adherence_evidence: str = "",
        effect_evidence: str = "",
        recommendation: str = "",
        method: str = "llm",
        post_baseline_session_count: int = 0,
    ) -> str:
        return self.append(
            "experiment.evaluated",
            experiment_id=exp_id,
            title=title,
            adherence=adherence,
            effect=effect,
            adherence_evidence=adherence_evidence,
            effect_evidence=effect_evidence,
            recommendation=recommendation,
            method=method,
            post_baseline_session_count=post_baseline_session_count,
        )

    def log_experiment_transition(
        self,
        new_status: Literal["graduated", "not_tried", "inconclusive"],
        exp_id: str,
        title: str,
    ) -> str:
        event = {
            "graduated": "experiment.graduated",
            "not_tried": "experiment.marked_not_tried",
            "inconclusive": "experiment.marked_inconclusive",
        }[new_status]
        return self.append(event, experiment_id=exp_id, title=title)

    def log_note(self, text: str, **context: Any) -> str:
        return self.append("note", text=text, **context)


# Module-level convenience instance. Most callers use this directly.
_default: Logbook | None = None


def default() -> Logbook:
    global _default
    if _default is None:
        path = os.environ.get("TESSERA_LOGBOOK")
        _default = Logbook(Path(path)) if path else Logbook()
    return _default
