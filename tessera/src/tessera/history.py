"""Local history store for past runs + user ratings.

Layout under the data dir (default ~/.config/tessera/history):

    history.json                 # index of all runs, newest first
    runs/<iso>.json              # full observations payload from one run
    ratings/<iso>.json           # user's ratings for that run's observations

Ratings are keyed by observation index within the run file. Each rating is
one of: `useful`, `wrong`, `known`, `skip`. Skipped observations carry no
downstream signal; the other three feed the next run's prompt.

Files are plain JSON so the user can inspect or hand-edit them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


Rating = Literal["useful", "wrong", "known", "skip"]


DEFAULT_DATA_DIR = Path.home() / ".config" / "tessera" / "history"
DEFAULT_KEEP_RUNS = 12
DEFAULT_RECENT_FOR_PROMPT = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(iso_ts: str) -> str:
    """Return a filename-safe slug of an ISO timestamp."""
    return iso_ts.replace(":", "-").replace("+", "_").replace(".", "-")


def _observation_key(observation: dict) -> str:
    """Stable short hash of a single observation, for cross-run identity."""
    seed = (observation.get("title", "") + "|" + (observation.get("claim", "") or ""))[:400]
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


@dataclass
class RunRecord:
    """One entry in the history index."""

    timestamp: str
    slug: str
    observation_count: int
    lookback_days: int | None
    model: str | None
    sessions_reviewed: int | None
    rated: bool = False

    @classmethod
    def from_observations(cls, timestamp: str, observations: dict) -> "RunRecord":
        meta = observations.get("meta", {})
        return cls(
            timestamp=timestamp,
            slug=_safe_slug(timestamp),
            observation_count=len(observations.get("observations", []) or []),
            lookback_days=meta.get("lookback_days"),
            model=meta.get("model"),
            sessions_reviewed=meta.get("sessions_reviewed"),
        )


@dataclass
class HistoryStore:
    """Append-only store of past runs + ratings under a single data dir."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        (self.data_dir / "runs").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "ratings").mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.data_dir / "history.json"

    def _read_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _write_index(self, entries: list[dict]) -> None:
        self.index_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add_run(
        self, observations: dict, keep: int = DEFAULT_KEEP_RUNS
    ) -> RunRecord:
        """Persist one run and return its RunRecord."""
        timestamp = _now_iso()
        record = RunRecord.from_observations(timestamp, observations)
        run_path = self.data_dir / "runs" / f"{record.slug}.json"
        run_path.write_text(
            json.dumps(observations, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        entries = self._read_index()
        entries.insert(
            0,
            {
                "timestamp": record.timestamp,
                "slug": record.slug,
                "observation_count": record.observation_count,
                "lookback_days": record.lookback_days,
                "model": record.model,
                "sessions_reviewed": record.sessions_reviewed,
                "rated": False,
            },
        )
        # Prune old entries + corresponding files
        if len(entries) > keep:
            for dropped in entries[keep:]:
                for kind in ("runs", "ratings"):
                    p = self.data_dir / kind / f"{dropped['slug']}.json"
                    if p.exists():
                        p.unlink()
            entries = entries[:keep]
        self._write_index(entries)
        return record

    def load_run(self, slug: str) -> dict | None:
        run_path = self.data_dir / "runs" / f"{slug}.json"
        if not run_path.exists():
            return None
        try:
            return json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def latest_unrated(self) -> RunRecord | None:
        for entry in self._read_index():
            if not entry.get("rated"):
                return RunRecord(**{k: v for k, v in entry.items() if k in RunRecord.__dataclass_fields__})
        return None

    def latest(self) -> RunRecord | None:
        entries = self._read_index()
        if not entries:
            return None
        entry = entries[0]
        return RunRecord(**{k: v for k, v in entry.items() if k in RunRecord.__dataclass_fields__})

    def save_ratings(self, slug: str, ratings: list[dict]) -> None:
        """Persist ratings for one run. `ratings` is a list of dicts with
        keys: index, title, key, rating."""
        ratings_path = self.data_dir / "ratings" / f"{slug}.json"
        payload = {
            "slug": slug,
            "rated_at": _now_iso(),
            "ratings": ratings,
        }
        ratings_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Mark the index entry as rated
        entries = self._read_index()
        for entry in entries:
            if entry.get("slug") == slug:
                entry["rated"] = True
                break
        self._write_index(entries)

    def load_ratings(self, slug: str) -> list[dict]:
        ratings_path = self.data_dir / "ratings" / f"{slug}.json"
        if not ratings_path.exists():
            return []
        try:
            payload = json.loads(ratings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return payload.get("ratings") or []

    def summarize_for_prompt(
        self, n: int = DEFAULT_RECENT_FOR_PROMPT
    ) -> tuple[str, dict[str, Rating]]:
        """Build a prompt-ready summary of recent runs + the user's ratings.

        Returns (text_block, rating_by_key) where rating_by_key maps the short
        observation hash → its rating. The model uses the text block for
        context; the rating map is useful for downstream summaries.
        """
        entries = self._read_index()[:n]
        if not entries:
            return "", {}

        lines: list[str] = []
        now = datetime.now(timezone.utc)
        rating_by_key: dict[str, Rating] = {}

        for entry in entries:
            slug = entry.get("slug")
            if not slug:
                continue
            run_payload = self.load_run(slug)
            if not run_payload:
                continue
            ratings_by_index = {r["index"]: r for r in self.load_ratings(slug)}

            try:
                run_ts = datetime.fromisoformat(entry["timestamp"])
                days_ago = max(0, (now - run_ts).days)
                when = f"{days_ago} day{'s' if days_ago != 1 else ''} ago"
            except (KeyError, ValueError):
                when = "prior run"

            lookback = entry.get("lookback_days")
            header = f"RUN {entry.get('timestamp', '')[:10]} ({when}, {lookback}d window):"
            lines.append(header)

            for idx, obs in enumerate(run_payload.get("observations", []) or []):
                title = obs.get("title") or f"(untitled §{idx + 1})"
                key = _observation_key(obs)
                rating_row = ratings_by_index.get(idx) or {}
                rating = rating_row.get("rating") or "unrated"
                if rating and rating != "unrated":
                    rating_by_key[key] = rating
                marker = {
                    "useful": "✓ USEFUL",
                    "wrong": "✗ WRONG (don't repeat)",
                    "known": "≈ ALREADY KNOWN",
                    "skip": "· skipped",
                    "unrated": "· unrated",
                }.get(rating, "· unrated")
                lines.append(f"  §{idx + 1} [{key}] {title} — {marker}")
                # Include cited session_ids so the model can reuse them when
                # flagging a continuation. Synthesis observations use
                # `evidence_sessions`; older observations used `evidence`.
                evidence = (
                    obs.get("evidence_sessions")
                    or obs.get("evidence")
                    or []
                )
                if evidence:
                    shown = ", ".join(evidence[:4])
                    extra = f" (+{len(evidence) - 4} more)" if len(evidence) > 4 else ""
                    lines.append(f"       evidence last time: {shown}{extra}")
                claim = (obs.get("claim") or "").strip()
                if claim:
                    snippet = claim if len(claim) <= 200 else claim[:197].rstrip() + "…"
                    lines.append(f"       claim: {snippet}")
            lines.append("")

        text_block = "\n".join(lines).rstrip()
        return text_block, rating_by_key
