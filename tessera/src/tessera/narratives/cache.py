"""On-disk cache for per-session narratives.

Cache key: events_content_hash + schema_version + model_id. Bumping any of
the three invalidates the entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tessera" / "narratives"


def _safe_filename(session_id: str) -> str:
    """Convert `<agent>:<uuid>` to a safe filename (no path separators)."""
    return session_id.replace(":", "__").replace("/", "_") + ".json"


@dataclass
class CacheEntry:
    payload: dict
    cache_key: str

    @property
    def is_hit(self) -> bool:
        return bool(self.payload)


class NarrativeCache:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.cache_dir / _safe_filename(session_id)

    @staticmethod
    def make_key(
        events_content_hash: str,
        schema_version: int,
        model: str,
        backend: str = "claude",
    ) -> str:
        """Cache key tuple. For backend='claude' we keep the pre-multi-backend
        format (just the model id) so existing on-disk entries from earlier
        tessera versions stay valid — the claude model names ('claude-sonnet-4-6')
        already carry provider context. For other backends, namespace by
        backend so an empty model id on codex doesn't collide with an empty
        model id on gemini."""
        if backend == "claude":
            return f"{events_content_hash}|v{schema_version}|{model}"
        return f"{events_content_hash}|v{schema_version}|{backend}:{model}"

    def load(self, session_id: str, expected_key: str) -> dict | None:
        """Return cached payload if its cache_key matches; else None."""
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.pop("_cache_key", None) != expected_key:
            return None
        return payload

    def save(self, session_id: str, payload: dict, cache_key: str) -> None:
        """Atomic write of the narrative payload + cache key marker."""
        path = self._path(session_id)
        body = dict(payload)
        body["_cache_key"] = cache_key
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(body, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def clear(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_cached(self) -> list[str]:
        """Return list of cached session_ids."""
        result = []
        for path in self.cache_dir.glob("*.json"):
            stem = path.stem
            if "__" in stem:
                result.append(stem.replace("__", ":", 1))
        return result
