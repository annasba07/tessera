"""Compress a session's event stream into a compact narrative for the LLM.

Per-line format:
    [seq offset (kind)] action

Where action is one of:
    U: <user msg>          — user message
    A: <assistant msg>     — assistant message
    T: <tool>(<input>)     — tool call
    R: <tool> <status> [<preview>]   — tool result
    !: <reasoning>         — assistant reasoning

`kind` is `s` for subagent events, omitted for top-level.
`offset` is time since first event (s/m/h/d).

The seq numbers are 0-based indices into the events list — the LLM cites
these as `event_range`, and the validator enforces they're in [0, event_count).
"""

from __future__ import annotations

from datetime import datetime


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _short(s, n: int) -> str:
    if not s:
        return ""
    text = str(s).replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _format_offset(seconds: float) -> str:
    if seconds < 60:
        return f"+{seconds:.0f}s"
    if seconds < 3600:
        return f"+{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"+{seconds / 3600:.1f}h"
    return f"+{seconds / 86400:.1f}d"


def _line_for(event: dict, seq: int, first_ts: datetime | None) -> str | None:
    kind = "s" if event.get("trace_kind") == "subagent" else ""
    ts = _parse_ts(event.get("timestamp"))
    offset = ""
    if ts and first_ts:
        offset = _format_offset((ts - first_ts).total_seconds())
    prefix = f"[{seq} {offset}{' s' if kind else ''}]"

    et = event.get("event_type")
    if et == "message":
        role = event.get("role")
        text = _short(event.get("message_text"), 400)
        if role == "user":
            return f"{prefix} U: {text}" if text else None
        if role == "assistant":
            return f"{prefix} A: {text}" if text else None
        return None
    if et == "tool_call":
        tool = event.get("tool_name") or "?"
        inp = _short(event.get("tool_input_preview"), 160)
        return f"{prefix} T: {tool}({inp})"
    if et == "tool_result":
        tool = event.get("tool_name") or "?"
        status = event.get("tool_status") or ""
        ec = event.get("error_class")
        if ec:
            tag = f"ERR:{ec}"
        elif status == "success":
            tag = "ok"
        elif status == "error":
            tag = "ERR"
        else:
            tag = status or "?"
        # Show preview only for errors — successes are noise
        if ec or status == "error":
            preview = _short(event.get("tool_output_preview"), 200)
            if preview:
                return f"{prefix} R: {tool} {tag} · {preview}"
        return f"{prefix} R: {tool} {tag}"
    if et == "reasoning":
        text = _short(event.get("message_text"), 250)
        return f"{prefix} !: {text}" if text else None
    return None


def _is_collapsible_success(line: str) -> bool:
    """Detect a successful tool result line we can fold into a run-length count."""
    if " R: " not in line:
        return False
    suffix = line.split(" R: ", 1)[1]
    return " ok" in suffix[:40] and "ERR" not in suffix


def compress_events(events: list[dict], max_chars: int = 180_000) -> str:
    """Compress an event stream to a compact narrative.

    Args:
        events: chronological events for one session.
        max_chars: target output budget. If exceeded, collapses runs of
            successful tool results, then truncates the middle.

    Returns:
        Multi-line string. Each line is one event keyed by its 0-based seq.
    """
    if not events:
        return ""

    first_ts = next(
        (_parse_ts(e.get("timestamp")) for e in events if e.get("timestamp")),
        None,
    )
    lines: list[str] = []
    for i, event in enumerate(events):
        line = _line_for(event, i, first_ts)
        if line:
            lines.append(line)

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text

    # Pass 1: collapse runs of ≥3 successful tool results
    compressed: list[str] = []
    i = 0
    while i < len(lines):
        if _is_collapsible_success(lines[i]):
            j = i
            while j < len(lines) and _is_collapsible_success(lines[j]):
                j += 1
            run_len = j - i
            if run_len >= 3:
                start_seq = lines[i].split(" ", 1)[0].strip("[]").split()[0]
                end_seq = lines[j - 1].split(" ", 1)[0].strip("[]").split()[0]
                compressed.append(
                    f"... [{run_len} successful tool results collapsed: seq {start_seq}–{end_seq}] ..."
                )
                i = j
                continue
        compressed.append(lines[i])
        i += 1
    text = "\n".join(compressed)
    if len(text) <= max_chars:
        return text

    # Pass 2: truncate middle, preserve head + tail
    keep = max_chars // 2 - 200
    return (
        text[:keep]
        + f"\n\n... [middle truncated, original {len(text):,} chars] ...\n\n"
        + text[-keep:]
    )
