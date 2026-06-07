"""Per-session narrative pipeline.

Load events → compute deterministic metadata → check cache →
compress events → LLM extract → validate → save to cache → return.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache import NarrativeCache
from .compressor import compress_events
from .deterministic import extract_deterministic
from ..backends import LLMBackend
from .extractor import DEFAULT_MODEL, extract_narrative as llm_extract
from .validator import validate_narrative


SCHEMA_VERSION = 1
DEFAULT_MAX_STREAM_CHARS = 180_000

logger = logging.getLogger(__name__)


@dataclass
class NarrativeResult:
    session_id: str
    payload: dict
    from_cache: bool
    error: str | None = None
    timing: dict[str, float] = field(default_factory=dict)


def load_session_events(events_path: Path, session_id: str) -> list[dict]:
    """Read normalized events.jsonl and filter to one session, chronological."""
    if ":" not in session_id:
        raise ValueError(f"session_id must be '<agent>:<uuid>', got {session_id!r}")
    agent, sid = session_id.split(":", 1)
    events: list[dict] = []
    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("agent") == agent and e.get("session_id") == sid:
                events.append(e)
    _sort_events(events)
    return events


def load_all_sessions_events(
    events_path: Path,
    wanted_ids: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Single-pass load of events.jsonl into a {session_id: [events]} dict.

    Args:
        events_path: path to events.jsonl
        wanted_ids: if provided, only collect events for these `<agent>:<uuid>`
            keys. None means load everything.
    """
    bucket: dict[str, list[dict]] = {}
    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            agent = e.get("agent")
            sid = e.get("session_id")
            if not agent or not sid:
                continue
            key = f"{agent}:{sid}"
            if wanted_ids is not None and key not in wanted_ids:
                continue
            bucket.setdefault(key, []).append(e)
    for events in bucket.values():
        _sort_events(events)
    return bucket


def _sort_events(events: list[dict]) -> None:
    events.sort(
        key=lambda e: (
            e.get("timestamp") or "",
            0 if e.get("trace_kind") == "top_level" else 1,
        )
    )


async def extract_session_narrative(
    session_id: str,
    events: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    backend: LLMBackend | None = None,
    cache: NarrativeCache | None = None,
    force: bool = False,
    skip_llm: bool = False,
    max_stream_chars: int = DEFAULT_MAX_STREAM_CHARS,
) -> NarrativeResult:
    """Run the per-session pipeline. Returns NarrativeResult with payload.

    Args:
        session_id: `<agent>:<uuid>`.
        events: chronological events for this session.
        model: LLM model for narrative extraction.
        cache: optional cache; if provided, used for read+write.
        force: bypass cache read (still writes).
        skip_llm: deterministic-only mode — returns metadata without narrative.
        max_stream_chars: budget for the compressed event stream.
    """
    timing: dict[str, float] = {}

    t0 = _now()
    metadata = extract_deterministic(session_id, events)
    timing["deterministic_sec"] = _elapsed(t0)

    if metadata.get("event_count", 0) == 0:
        return NarrativeResult(
            session_id=session_id,
            payload={**metadata, "_skip_reason": "no_events"},
            from_cache=False,
            error="no_events",
            timing=timing,
        )

    backend_name = backend.name if backend else "claude"
    cache_key = NarrativeCache.make_key(
        metadata["events_content_hash"], SCHEMA_VERSION, model, backend=backend_name
    )

    if cache and not force and not skip_llm:
        cached = cache.load(session_id, cache_key)
        if cached is not None:
            return NarrativeResult(
                session_id=session_id,
                payload=cached,
                from_cache=True,
                timing=timing,
            )

    if skip_llm:
        payload = {
            **metadata,
            "narrative_extracted_at": None,
            "narrative_model": None,
            "schema_version": SCHEMA_VERSION,
            "narrative_quality": "skipped",
        }
        return NarrativeResult(
            session_id=session_id,
            payload=payload,
            from_cache=False,
            timing=timing,
        )

    t0 = _now()
    stream = compress_events(events, max_chars=max_stream_chars)
    timing["compress_sec"] = _elapsed(t0)

    t0 = _now()
    try:
        raw_narrative = await llm_extract(metadata, stream, model=model, backend=backend)
    except json.JSONDecodeError as exc:
        return NarrativeResult(
            session_id=session_id,
            payload={**metadata, "_skip_reason": "llm_json_decode_error"},
            from_cache=False,
            error=f"llm_json_decode: {exc}",
            timing=timing,
        )
    except Exception as exc:  # SDK / network / unexpected
        logger.exception("LLM call failed for %s", session_id)
        return NarrativeResult(
            session_id=session_id,
            payload={**metadata, "_skip_reason": "llm_call_failed"},
            from_cache=False,
            error=f"llm_call_failed: {exc}",
            timing=timing,
        )
    timing["llm_sec"] = _elapsed(t0)

    t0 = _now()
    cleaned_narrative = validate_narrative(raw_narrative, events)
    timing["validate_sec"] = _elapsed(t0)

    payload = {
        **metadata,
        **cleaned_narrative,
        "narrative_extracted_at": datetime.now(timezone.utc).isoformat(),
        "narrative_model": model,
        "schema_version": SCHEMA_VERSION,
        "stream_chars": len(stream),
    }

    if cache:
        try:
            cache.save(session_id, payload, cache_key)
        except OSError as exc:
            logger.warning("Cache save failed for %s: %s", session_id, exc)

    return NarrativeResult(
        session_id=session_id,
        payload=payload,
        from_cache=False,
        timing=timing,
    )


async def extract_many(
    session_ids_and_events: list[tuple[str, list[dict]]],
    *,
    model: str = DEFAULT_MODEL,
    backend: LLMBackend | None = None,
    cache: NarrativeCache | None = None,
    force: bool = False,
    skip_llm: bool = False,
    concurrency: int = 5,
    progress_cb=None,
) -> list[NarrativeResult]:
    """Extract narratives for many sessions concurrently."""
    sem = asyncio.Semaphore(concurrency)
    completed = [0]
    total = len(session_ids_and_events)

    async def _one(sid: str, events: list[dict]) -> NarrativeResult:
        async with sem:
            result = await extract_session_narrative(
                sid,
                events,
                model=model,
                backend=backend,
                cache=cache,
                force=force,
                skip_llm=skip_llm,
            )
            completed[0] += 1
            if progress_cb:
                progress_cb(completed[0], total, result)
            return result

    return await asyncio.gather(
        *(_one(sid, events) for sid, events in session_ids_and_events)
    )


# ---------- helpers ---------------------------------------------------------


def _now() -> float:
    import time

    return time.monotonic()


def _elapsed(t0: float) -> float:
    import time

    return round(time.monotonic() - t0, 2)
