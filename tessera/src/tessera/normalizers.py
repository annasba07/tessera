"""Pluggable normalizer registry — built-in agents + user-defined.

Adding a new agent CLI to tessera is one Python file:

    # ~/.config/tessera/normalizers/aider.py
    from tessera.normalizers import register
    from pathlib import Path

    def normalize_aider(raw_root, writer):
        # walk raw_root looking for Aider's trace files (e.g. .aider/history)
        # for each session, emit events via writer.write_event(...)
        # then write_session(...) once per session
        ...

    register(
        name="aider",
        default_source=Path.home() / ".aider",
        normalize=normalize_aider,
        description="Aider AI pair-programming CLI",
    )

Any `.py` file in `~/.config/tessera/normalizers/` is imported at startup;
calling `register(...)` adds the normalizer to the pipeline. The downstream
narrative + synthesis + dashboard layers are agent-agnostic — they consume
the shared event schema (see docs/schema/v1.md).

This module is intentionally small. The heavy lifting (event writer,
schema, etc.) stays in `_normalize_script.py`. This file is just the
registry + the discovery loader.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# Where user-defined normalizers live. Override via TESSERA_NORMALIZERS_DIR
# environment variable if needed.
DEFAULT_USER_NORMALIZERS_DIR = Path.home() / ".config" / "tessera" / "normalizers"


@dataclass
class Normalizer:
    """One agent CLI's trace-to-event normalizer."""

    name: str
    default_source: Path
    normalize: Callable[[Path, Any], None]
    """Callable: (raw_root, writer) -> None. Must write events + sessions
    via the writer; see _normalize_script.EventWriter for the interface."""
    description: str = ""
    source: str = "builtin"  # "builtin" | "user-dir" | "entry-point"


_REGISTRY: dict[str, Normalizer] = {}


def register(
    name: str,
    default_source: Path,
    normalize: Callable[[Path, Any], None],
    description: str = "",
    source: str = "user-dir",
) -> None:
    """Register a normalizer. Called from builtin loaders and user files."""
    if not name or not isinstance(name, str):
        raise ValueError(f"normalizer name must be a non-empty string, got {name!r}")
    if not callable(normalize):
        raise ValueError(f"normalize must be callable, got {type(normalize)}")
    _REGISTRY[name] = Normalizer(
        name=name,
        default_source=Path(default_source).expanduser(),
        normalize=normalize,
        description=description,
        source=source,
    )


def get_all() -> list[Normalizer]:
    """Return all registered normalizers, builtins-first."""
    builtin_order = ("claude", "codex", "gemini")
    builtins = [_REGISTRY[n] for n in builtin_order if n in _REGISTRY]
    extras = sorted(
        (n for k, n in _REGISTRY.items() if k not in builtin_order),
        key=lambda n: n.name,
    )
    return builtins + extras


def get(name: str) -> Normalizer | None:
    return _REGISTRY.get(name)


def _register_builtins() -> None:
    """Wire the 3 built-in normalizers (claude/codex/gemini)."""
    # Lazy import to avoid circular deps; _normalize_script imports this
    # module if a user dir normalizer needs the EventWriter.
    from . import _normalize_script as ns

    register(
        name="claude",
        default_source=Path.home() / ".claude" / "projects",
        normalize=ns.normalize_claude,
        description="Anthropic Claude Code (~/.claude/projects/*.jsonl)",
        source="builtin",
    )
    register(
        name="codex",
        default_source=Path.home() / ".codex" / "sessions",
        normalize=ns.normalize_codex,
        description="OpenAI Codex CLI (~/.codex/sessions/**/rollout-*.jsonl)",
        source="builtin",
    )
    register(
        name="gemini",
        default_source=Path.home() / ".gemini" / "tmp",
        normalize=ns.normalize_gemini,
        description="Google Gemini CLI (~/.gemini/tmp/**)",
        source="builtin",
    )


def _load_user_dir(dir_path: Path | None = None) -> list[str]:
    """Import every .py file in the user normalizers dir.

    Returns a list of warnings (e.g., 'failed to load aider.py: <error>')
    that the caller can surface. Skips silently when the dir doesn't exist.
    """
    dir_path = (dir_path or DEFAULT_USER_NORMALIZERS_DIR).expanduser()
    warnings: list[str] = []
    if not dir_path.is_dir():
        return warnings
    for py in sorted(dir_path.glob("*.py")):
        if py.name.startswith("_"):
            continue
        module_name = f"tessera._user_normalizers.{py.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py)
            if not spec or not spec.loader:
                warnings.append(f"could not load {py.name}: no module spec")
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
        except Exception as exc:
            warnings.append(
                f"failed to import {py.name}: {type(exc).__name__}: {exc}"
                + "\n    " + traceback.format_exc().splitlines()[-1]
            )
    return warnings


def _load_entry_points() -> list[str]:
    """Load normalizers registered via the 'tessera.normalizers' entry-point
    group (for installable plugins like `pip install tessera-aider`).
    """
    warnings: list[str] = []
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return warnings
    try:
        eps = entry_points(group="tessera.normalizers")
    except TypeError:
        # Older Python: entry_points() returns a dict, not a callable-with-group
        eps = entry_points().get("tessera.normalizers", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            ep.load()  # The plugin should call register() at import time.
        except Exception as exc:
            warnings.append(
                f"failed to load entry-point plugin {ep.name!r}: {type(exc).__name__}: {exc}"
            )
    return warnings


_INITIALIZED = False


def initialize() -> list[str]:
    """Idempotent: register builtins + load user dir + load entry-points.

    Returns a list of non-fatal warnings (failed user-file imports etc.) that
    the caller may print. Safe to call multiple times; subsequent calls are
    no-ops.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return []
    _register_builtins()
    warnings = _load_user_dir()
    warnings.extend(_load_entry_points())
    _INITIALIZED = True
    return warnings
