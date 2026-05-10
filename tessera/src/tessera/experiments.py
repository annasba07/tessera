"""Self-experiment tracking — closing the loop on behavioral_patterns.

When the user rates a behavioral_pattern as `useful` in the dashboard, the
pattern's `experiment_to_try` becomes a thing the user committed to. Until
now, that commitment vanished into the next-run prompt's prior-context block
without any structured tracking.

This module:
  1. Registers an experiment record at rating time (synchronous, cheap).
  2. Lists active experiments (for inclusion in the next synthesis prompt
     and for dashboard display).
  3. Evaluates active experiments against the latest narratives — did the
     user actually try the experiment, and what happened to the metrics
     it specified? (One LLM call per evaluation pass, batched across
     all active experiments.)

Storage layout (under ~/.config/tessera/experiments/):

    index.json                         # cross-experiment list, newest first
    active/<exp-slug>.json             # active experiments
    graduated/<exp-slug>.json          # confirmed-working experiments
    inconclusive/<exp-slug>.json       # ran the eval window, no clear effect
    not_tried/<exp-slug>.json          # zero adherence — gentler resurface

Lifecycle:
    rated `useful`  ──▶  active
    eval shows clear positive effect  ──▶  graduated
    eval shows zero adherence         ──▶  not_tried
    eval shows no/mixed effect after  ──▶  inconclusive
        2+ evaluation passes
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


SCHEMA_VERSION = 1
DEFAULT_DATA_DIR = Path.home() / ".config" / "tessera" / "experiments"
EVALS_BEFORE_INCONCLUSIVE = 2

ExperimentStatus = Literal["active", "graduated", "inconclusive", "not_tried"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug from arbitrary text. Used for experiment IDs."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", text.lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] or "untitled"


def _pattern_key(pattern: dict) -> str:
    """Same key scheme used elsewhere — sha1 of title + claim/pattern field.

    Lets us match a behavioral_pattern across runs even if its evidence_refs
    drift week-to-week.
    """
    seed = (
        (pattern.get("title") or "")
        + "|"
        + (pattern.get("pattern") or pattern.get("claim") or "")
    )[:400]
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


@dataclass
class Experiment:
    """One experiment the user committed to by rating a pattern `useful`."""

    id: str
    pattern_key: str
    title: str
    experiment_text: str
    dimension: str | None
    started_at: str
    started_run_slug: str | None
    baseline_period_end: str  # narratives before this are "baseline"
    status: ExperimentStatus = "active"
    evaluations: list[dict] = field(default_factory=list)

    @classmethod
    def from_pattern(
        cls,
        pattern: dict,
        run_slug: str | None,
    ) -> "Experiment":
        title = pattern.get("title") or "untitled-pattern"
        now = _now_iso()
        return cls(
            id=_slugify(title) + "-" + _pattern_key(pattern),
            pattern_key=_pattern_key(pattern),
            title=title,
            experiment_text=(
                pattern.get("experiment_to_try")
                or pattern.get("intervention")
                or pattern.get("next_action")
                or ""
            ),
            dimension=pattern.get("dimension"),
            started_at=now,
            started_run_slug=run_slug,
            baseline_period_end=now,
            status="active",
        )


@dataclass
class ExperimentStore:
    """Append-only store of experiments under one data dir."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        for sub in ("active", "graduated", "inconclusive", "not_tried"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.json"

    def _bucket_path(self, status: ExperimentStatus, exp_id: str) -> Path:
        return self.data_dir / status / f"{exp_id}.json"

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

    def get(self, exp_id: str) -> Experiment | None:
        for status in ("active", "graduated", "inconclusive", "not_tried"):
            p = self._bucket_path(status, exp_id)  # type: ignore[arg-type]
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    return Experiment(**data)
                except (OSError, json.JSONDecodeError, TypeError):
                    return None
        return None

    def list(self, status: ExperimentStatus | None = None) -> list[Experiment]:
        out = []
        statuses = (status,) if status else ("active", "graduated", "inconclusive", "not_tried")
        for s in statuses:
            for p in sorted((self.data_dir / s).glob("*.json")):  # type: ignore[arg-type]
                try:
                    out.append(Experiment(**json.loads(p.read_text(encoding="utf-8"))))
                except (OSError, json.JSONDecodeError, TypeError):
                    continue
        return out

    def upsert(self, exp: Experiment) -> None:
        """Write the experiment to its current-status bucket. If it lives in
        a different bucket already, move it (delete the old file).
        """
        for s in ("active", "graduated", "inconclusive", "not_tried"):
            p = self._bucket_path(s, exp.id)  # type: ignore[arg-type]
            if p.exists() and s != exp.status:
                p.unlink()
        self._bucket_path(exp.status, exp.id).write_text(
            json.dumps(asdict(exp), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Index: keep newest-first, dedupe by id
        entries = [e for e in self._read_index() if e.get("id") != exp.id]
        entries.insert(
            0,
            {
                "id": exp.id,
                "title": exp.title,
                "pattern_key": exp.pattern_key,
                "dimension": exp.dimension,
                "started_at": exp.started_at,
                "status": exp.status,
                "evaluation_count": len(exp.evaluations),
            },
        )
        self._write_index(entries)


def register_from_ratings(
    ratings: list[dict],
    behavioral_patterns: list[dict],
    run_slug: str | None,
    store: ExperimentStore | None = None,
) -> list[Experiment]:
    """For each `useful` rating on a behavioral_pattern, register an experiment.

    `ratings` is the rating list saved by HistoryStore.save_ratings (each
    entry has index, title, key, rating). `behavioral_patterns` is the list
    of patterns from the same synthesis run (ordered to match the rating's
    index). Returns the experiments that were newly registered.
    """
    if store is None:
        store = ExperimentStore()
    registered: list[Experiment] = []
    by_index = {i: bp for i, bp in enumerate(behavioral_patterns)}
    by_key = {_pattern_key(bp): bp for bp in behavioral_patterns}

    for r in ratings:
        if r.get("rating") != "useful":
            continue
        # Resolve the rated pattern: first by key (more stable across runs),
        # then by index as fallback.
        pattern = by_key.get(r.get("key") or "")
        if pattern is None and isinstance(r.get("index"), int):
            pattern = by_index.get(r["index"])
        if pattern is None:
            continue
        exp = Experiment.from_pattern(pattern, run_slug)
        # Don't re-register an experiment that's already active or graduated
        # for the same pattern — this lets the user re-rate without churning.
        existing = store.get(exp.id)
        if existing and existing.status in ("active", "graduated"):
            continue
        store.upsert(exp)
        registered.append(exp)
    return registered


def summarize_for_prompt(
    store: ExperimentStore | None = None,
    *,
    include_statuses: tuple[str, ...] = ("active", "graduated"),
    max_chars: int = 4000,
) -> str:
    """Compact summary of experiments to drop into the synthesis prompt as
    prior-context. Lets the synthesis avoid re-surfacing patterns the user
    is already actively running an experiment on.
    """
    if store is None:
        store = ExperimentStore()
    lines: list[str] = []
    for status in include_statuses:
        bucket = store.list(status=status)  # type: ignore[arg-type]
        if not bucket:
            continue
        lines.append(f"\n## {status.capitalize()} experiments ({len(bucket)})")
        for exp in bucket:
            line = (
                f"- [{exp.dimension or 'general'}] {exp.title}"
                f"  (started {exp.started_at[:10]}, {len(exp.evaluations)} evals)"
            )
            lines.append(line)
            # Include the experiment text only for active ones
            if status == "active":
                lines.append(f"  experiment: {exp.experiment_text[:280]}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 30] + "\n... (truncated)"
    return text


def evaluate_pending(
    narratives: list[dict],
    store: ExperimentStore | None = None,
    *,
    llm_evaluator=None,
) -> dict[str, Any]:
    """For every active experiment, look at narratives newer than the
    experiment's baseline_period_end and ask: was the experiment tried?
    What happened to the relevant metric?

    `llm_evaluator` is a callable(experiment, post_baseline_narratives)
    that returns a dict with the eval result. Caller wires it (so this
    module stays LLM-free for tests).

    Returns a summary dict; mutates each experiment's evaluations list in
    place and persists.
    """
    if store is None:
        store = ExperimentStore()
    summary: dict[str, Any] = {
        "evaluated": 0,
        "graduated": [],
        "marked_not_tried": [],
        "marked_inconclusive": [],
        "still_active": [],
        "skipped_no_post_baseline_data": [],
    }
    for exp in store.list(status="active"):
        cutoff = exp.baseline_period_end
        post = [
            n for n in narratives
            if (n.get("started_at") or "") > cutoff
        ]
        if not post:
            summary["skipped_no_post_baseline_data"].append(exp.id)
            continue

        eval_result: dict[str, Any]
        if llm_evaluator:
            eval_result = llm_evaluator(exp, post) or {}
        else:
            # Default heuristic eval (no LLM): just count post-baseline
            # sessions in the same dimension/project. Crude but offline-safe.
            eval_result = {
                "method": "heuristic_count",
                "post_baseline_sessions": len(post),
                "adherence": "unknown",
                "effect": "unknown",
            }
        eval_result["evaluated_at"] = _now_iso()
        eval_result["post_baseline_session_count"] = len(post)
        exp.evaluations.append(eval_result)
        summary["evaluated"] += 1

        # Status transition based on the eval result fields the LLM should
        # populate: adherence ∈ {none, partial, full, unknown},
        # effect ∈ {positive, neutral, negative, unknown}
        adherence = eval_result.get("adherence", "unknown")
        effect = eval_result.get("effect", "unknown")
        if adherence == "none":
            exp.status = "not_tried"
            summary["marked_not_tried"].append(exp.id)
        elif effect == "positive" and adherence in ("partial", "full"):
            exp.status = "graduated"
            summary["graduated"].append(exp.id)
        elif (
            len(exp.evaluations) >= EVALS_BEFORE_INCONCLUSIVE
            and effect in ("neutral", "negative", "unknown")
        ):
            exp.status = "inconclusive"
            summary["marked_inconclusive"].append(exp.id)
        else:
            summary["still_active"].append(exp.id)

        store.upsert(exp)
    return summary
