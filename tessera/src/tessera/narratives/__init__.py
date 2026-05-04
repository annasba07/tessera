"""Per-session narrative extraction.

The unit of analysis is one session. For each qualifying session we produce
one JSON artifact combining deterministic event-stream metadata with an
LLM-extracted narrative. See docs/schema/v1.md for the full schema.
"""

from .cache import DEFAULT_CACHE_DIR, NarrativeCache
from .dashboard import render_dashboard, write_dashboard
from .pipeline import (
    extract_many,
    extract_session_narrative,
    load_all_sessions_events,
    load_session_events,
)
from .render import render_synthesis_markdown, render_synthesis_text

SCHEMA_VERSION = 1

__all__ = [
    "DEFAULT_CACHE_DIR",
    "NarrativeCache",
    "SCHEMA_VERSION",
    "extract_many",
    "extract_session_narrative",
    "load_all_sessions_events",
    "load_session_events",
    "render_dashboard",
    "render_synthesis_markdown",
    "render_synthesis_text",
    "write_dashboard",
]
