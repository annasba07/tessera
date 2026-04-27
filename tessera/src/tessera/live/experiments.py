"""Proposed-experiment writer.

When a coach rule fires for the first time on a given (rule_key, project)
pair and there's no prior rated observation about it, we write a pending
experiment proposal to disk. The next ``tessera run`` reads these and
includes them in the prompt so the model can turn them into a concrete
test the user can try next week.

Storage: ~/.config/tessera/experiments/<id>.json — one file per
experiment so appends are atomic and hand-editing is safe.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_EXPERIMENTS_DIR = Path.home() / ".config" / "tessera" / "experiments"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Experiment:
    id: str
    created_at: str
    trigger_rule: str
    project: str | None
    session_id: str
    context: str
    status: str = "pending"


def load_all(directory: Path | None = None) -> list[Experiment]:
    base = Path(directory) if directory else DEFAULT_EXPERIMENTS_DIR
    if not base.exists():
        return []
    out: list[Experiment] = []
    for path in sorted(base.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            out.append(Experiment(**data))
        except TypeError:
            continue
    return out


def has_pending_for(
    trigger_rule: str,
    project: str | None,
    directory: Path | None = None,
) -> bool:
    for e in load_all(directory):
        if (
            e.status == "pending"
            and e.trigger_rule == trigger_rule
            and e.project == project
        ):
            return True
    return False


def add(
    trigger_rule: str,
    project: str | None,
    session_id: str,
    context: str,
    directory: Path | None = None,
) -> Experiment:
    base = Path(directory) if directory else DEFAULT_EXPERIMENTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(
        id=str(uuid.uuid4()),
        created_at=_now_iso(),
        trigger_rule=trigger_rule,
        project=project,
        session_id=session_id,
        context=context,
    )
    path = base / f"{experiment.id}.json"
    path.write_text(json.dumps(asdict(experiment), indent=2, ensure_ascii=False), encoding="utf-8")
    return experiment


def add_if_novel(
    trigger_rule: str,
    project: str | None,
    session_id: str,
    context: str,
    directory: Path | None = None,
) -> Experiment | None:
    """Create an experiment only if no pending one already exists for this
    (rule, project) combination. Returns the new Experiment, or None if one
    already existed.
    """
    if has_pending_for(trigger_rule, project, directory):
        return None
    return add(trigger_rule, project, session_id, context, directory)
