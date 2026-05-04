"""Deterministic metadata extraction from a session's event stream.

Produces every non-LLM field in docs/schema/v1.md. Pure function over a
pre-filtered list of events for a single (agent, session_id).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any


# ---------- Constants -------------------------------------------------------

CORRECTION_PHRASES = (
    "no, ",
    "no,",
    "wrong",
    "stop",
    "actually",
    "that's not",
    "thats not",
    "incorrect",
    "not what",
)

REPHRASE_OVERLAP_THRESHOLD = 0.5  # >50% word overlap within 10 turns
REPHRASE_LOOKBACK_TURNS = 10

ACTIVE_GAP_CAP_SEC = 5 * 60  # 5 minutes
BURST_GAP_THRESHOLD_SEC = 5 * 60

TEST_RUNNERS_RE = re.compile(
    r"\b(pytest|jest|vitest|cargo\s+test|go\s+test|npm\s+test|yarn\s+test|pnpm\s+test|"
    r"ruff(\s+check)?|mypy|tsc|eslint|prettier(\s+--check)?|black(\s+--check)?|"
    r"rspec|phpunit|cypress|playwright\s+test|deno\s+test)\b",
    re.IGNORECASE,
)
GIT_STATUS_RE = re.compile(r"\bgit\s+status\b", re.IGNORECASE)
GIT_DIFF_RE = re.compile(r"\bgit\s+diff\b", re.IGNORECASE)
PR_REF_RE = re.compile(r"\b(gh\s+pr|pull[_\-]?request)\b", re.IGNORECASE)

FILE_PATH_KEYS = ("file_path", "path", "filename", "target_path", "filepath")
SHELL_TOOL_KEYS = (
    "bash",
    "shell",
    "exec_command",
    "shell_command",
    "run_shell_command",
    "execute_command",
)
BROWSER_TOOL_KEYS = ("browser", "chrome", "playwright", "puppeteer")
READ_TOOL_KEYS = ("read", "glob", "grep", "ls", "cat", "find", "list_directory")
WRITE_TOOL_KEYS = ("write", "edit", "apply", "create", "patch", "multi_edit")
DELEGATION_TOOL_KEYS = ("task", "subagent", "delegate", "spawn", "agent")
WEB_TOOL_KEYS = ("web_fetch", "web_search", "websearch", "webfetch", "fetch", "http", "curl")


def _tool_category(name: str) -> str:
    lower = (name or "").lower()
    if any(k in lower for k in BROWSER_TOOL_KEYS):
        return "browser"
    if any(k in lower for k in DELEGATION_TOOL_KEYS):
        return "delegation"
    if any(k in lower for k in WRITE_TOOL_KEYS):
        return "write"
    if any(k in lower for k in READ_TOOL_KEYS):
        return "read"
    if any(k in lower for k in SHELL_TOOL_KEYS):
        return "shell"
    if any(k in lower for k in WEB_TOOL_KEYS):
        return "web"
    return "other"


def _project_label(cwd: str | None) -> str:
    if not cwd:
        return "(no project)"
    parts = [p for p in cwd.rstrip("/").split("/") if p]
    if not parts:
        return cwd
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _time_of_day_bucket(dt: datetime) -> str:
    h = dt.hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


def _file_paths_from_input(tool_input: Any) -> list[str]:
    """Extract file paths from a tool_input dict or JSON string preview."""
    if tool_input is None:
        return []
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(tool_input, dict):
        return []
    paths = []
    for key in FILE_PATH_KEYS:
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            paths.append(v)
    # MultiEdit-style edits[]
    edits = tool_input.get("edits")
    if isinstance(edits, list):
        for e in edits:
            if isinstance(e, dict):
                for key in FILE_PATH_KEYS:
                    v = e.get(key)
                    if isinstance(v, str) and v:
                        paths.append(v)
    return paths


def _bash_command_text(tool_input: Any) -> str | None:
    if tool_input is None:
        return None
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return tool_input
    if not isinstance(tool_input, dict):
        return None
    for key in ("command", "cmd", "shell_command", "code"):
        v = tool_input.get(key)
        if isinstance(v, str):
            return v
    return None


def _input_hash(tool_name: str, tool_input: Any) -> str:
    raw = json.dumps(tool_input, sort_keys=True, ensure_ascii=False, default=str) if tool_input is not None else ""
    return hashlib.sha1(f"{tool_name}\x00{raw}".encode("utf-8")).hexdigest()[:10]


def _word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _word_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def _events_content_hash(events: list[dict]) -> str:
    """Stable hash for cache invalidation. Uses the durable identifying fields
    of each event so cosmetic changes (added previews) don't bust the cache."""
    h = hashlib.sha256()
    for e in events:
        keyed = {
            "ts": e.get("timestamp"),
            "type": e.get("event_type"),
            "role": e.get("role"),
            "tool": e.get("tool_name"),
            "tool_call_id": e.get("tool_call_id"),
            "status": e.get("tool_status"),
            "ec": e.get("error_class"),
            "trace_kind": e.get("trace_kind"),
            "len": e.get("text_length") or e.get("input_length") or 0,
        }
        h.update(json.dumps(keyed, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"


# ---------- Main entry point ------------------------------------------------


def extract_deterministic(session_id: str, events: list[dict]) -> dict:
    """Compute every deterministic field for a session.

    Args:
        session_id: `<agent>:<uuid>` form.
        events: pre-filtered list of events for this session, chronological.

    Returns:
        Dict matching the deterministic block of docs/schema/v1.md.
    """
    if ":" in session_id:
        agent, _ = session_id.split(":", 1)
    else:
        agent = events[0].get("agent") if events else "unknown"

    if not events:
        return {
            "session_id": session_id,
            "agent": agent,
            "event_count": 0,
            "events_content_hash": _events_content_hash([]),
        }

    # ---- Identity ----
    cwd = next((e.get("cwd") for e in events if e.get("cwd")), None)
    project_path = cwd
    project_label = _project_label(cwd)
    git_branch_last = None
    for e in reversed(events):
        if e.get("git_branch"):
            git_branch_last = e["git_branch"]
            break

    models_seen: Counter = Counter()
    for e in events:
        m = e.get("model")
        if m:
            models_seen[m] += 1
    models_used = sorted(models_seen.keys())
    primary_model = models_seen.most_common(1)[0][0] if models_seen else None

    # ---- Time ----
    timestamps = [_parse_ts(e.get("timestamp")) for e in events]
    timestamps = [t for t in timestamps if t is not None]
    started_at = timestamps[0].isoformat() if timestamps else None
    ended_at = timestamps[-1].isoformat() if timestamps else None
    wall_clock_minutes = 0.0
    active_minutes = 0.0
    longest_idle_minutes = 0.0
    bursts = 0
    primary_burst_minutes = 0.0
    if len(timestamps) >= 2:
        wall_clock_minutes = round(
            (timestamps[-1] - timestamps[0]).total_seconds() / 60, 1
        )
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        active_seconds = sum(min(g, ACTIVE_GAP_CAP_SEC) for g in gaps)
        active_minutes = round(active_seconds / 60, 2)
        longest_idle_minutes = round(max(gaps) / 60, 1) if gaps else 0.0
        # bursts: count windows separated by >5min idle
        bursts = 1
        burst_lengths_sec: list[float] = []
        burst_start = timestamps[0]
        prev = timestamps[0]
        for ts in timestamps[1:]:
            if (ts - prev).total_seconds() > BURST_GAP_THRESHOLD_SEC:
                burst_lengths_sec.append((prev - burst_start).total_seconds())
                bursts += 1
                burst_start = ts
            prev = ts
        burst_lengths_sec.append((prev - burst_start).total_seconds())
        primary_burst_minutes = round(max(burst_lengths_sec) / 60, 1)
    elif len(timestamps) == 1:
        bursts = 1

    events_per_active_minute = (
        round(len(events) / active_minutes, 1) if active_minutes > 0 else 0.0
    )

    time_of_day_buckets: Counter = Counter()
    for ts in timestamps:
        time_of_day_buckets[_time_of_day_bucket(ts)] += 1
    weekday = timestamps[0].strftime("%A") if timestamps else None

    # ---- Volume + Tool / Scope ----
    event_count = len(events)
    tool_call_count = 0
    user_turn_count = 0
    subagent_event_count = 0
    subagent_trace_ids: set[str] = set()
    subagent_types: Counter = Counter()
    tool_call_distribution: Counter = Counter()
    tool_category_distribution: Counter = Counter()
    tool_error_count = 0
    tool_error_classes: Counter = Counter()
    tool_input_hash_counter: Counter = Counter()
    tool_input_hash_to_pair: dict[str, tuple[str, Any]] = {}
    file_paths_touched: Counter = Counter()
    dir_counter: Counter = Counter()
    tests_invoked = 0
    git_status_invoked = 0
    git_diff_invoked = 0
    pr_referenced = False

    assistant_chars = 0
    user_chars = 0
    reasoning_chars = 0
    user_messages_text: list[str] = []
    user_messages_idx: list[int] = []

    last_user_msg_idx = -1

    for i, e in enumerate(events):
        et = e.get("event_type")
        if e.get("trace_kind") == "subagent":
            subagent_event_count += 1
            tid = e.get("trace_id")
            if tid:
                subagent_trace_ids.add(tid)
            stype = e.get("subagent_type") or e.get("agent_subtype")
            if stype:
                subagent_types[stype] += 1

        if et == "message":
            role = e.get("role")
            text = e.get("message_text") or ""
            if role == "user":
                user_turn_count += 1
                user_chars += len(text)
                user_messages_text.append(text)
                user_messages_idx.append(i)
                last_user_msg_idx = i
            elif role == "assistant":
                assistant_chars += len(text)
        elif et == "reasoning":
            reasoning_chars += len(e.get("message_text") or "")
        elif et == "tool_call":
            tool_call_count += 1
            tname = e.get("tool_name") or "(unknown)"
            tool_call_distribution[tname] += 1
            tool_category_distribution[_tool_category(tname)] += 1
            tinput = e.get("tool_input_preview") or e.get("tool_input")
            ihash = _input_hash(tname, tinput)
            tool_input_hash_counter[ihash] += 1
            tool_input_hash_to_pair.setdefault(ihash, (tname, tinput))
            # File paths
            for p in _file_paths_from_input(tinput):
                file_paths_touched[p] += 1
                directory = "/".join(p.split("/")[:-1]) or "/"
                dir_counter[directory] += 1
            # Verification proxies — inspect bash command text
            cmd = _bash_command_text(tinput)
            if cmd:
                if TEST_RUNNERS_RE.search(cmd):
                    tests_invoked += 1
                if GIT_STATUS_RE.search(cmd):
                    git_status_invoked += 1
                if GIT_DIFF_RE.search(cmd):
                    git_diff_invoked += 1
                if PR_REF_RE.search(cmd):
                    pr_referenced = True
            # Also catch PR ref in any string field of input
            elif isinstance(tinput, str) and PR_REF_RE.search(tinput):
                pr_referenced = True

        if e.get("error_class") or e.get("tool_status") == "error":
            tool_error_count += 1
            ec = e.get("error_class")
            if ec:
                tool_error_classes[ec] += 1

    subagent_count = len(subagent_trace_ids)
    tool_call_distribution_top10 = dict(tool_call_distribution.most_common(10))
    tool_category_distribution_dict = dict(tool_category_distribution)
    tool_error_rate = (
        round(tool_error_count / tool_call_count, 4) if tool_call_count > 0 else 0.0
    )
    most_repeated_input_hash, most_repeated_count = (
        tool_input_hash_counter.most_common(1)[0]
        if tool_input_hash_counter
        else (None, 0)
    )
    most_repeated_tool_call = None
    if most_repeated_input_hash and most_repeated_count >= 2:
        tname, tinput = tool_input_hash_to_pair[most_repeated_input_hash]
        # Truncate input preview
        ipreview = tinput
        if isinstance(ipreview, str) and len(ipreview) > 120:
            ipreview = ipreview[:117] + "..."
        elif isinstance(ipreview, dict):
            ipreview = json.dumps(ipreview, ensure_ascii=False)[:120]
        most_repeated_tool_call = {
            "tool": tname,
            "input_hash": most_repeated_input_hash,
            "count": most_repeated_count,
            "input_preview": ipreview,
        }

    unique_files_touched = len(file_paths_touched)
    top_files_touched = [
        {"path": p, "count": c} for p, c in file_paths_touched.most_common(10)
    ]
    total_file_ops = sum(dir_counter.values())
    directory_concentration = (
        round(dir_counter.most_common(1)[0][1] / total_file_ops, 3)
        if total_file_ops > 0
        else 0.0
    )

    assistant_to_user_text_ratio = (
        round(assistant_chars / user_chars, 2) if user_chars > 0 else 0.0
    )

    # ---- Behavioral signals ----
    explicit_corrections = 0
    rephrased_retries = 0
    user_word_sets = [_word_set(t) for t in user_messages_text]
    for i, text in enumerate(user_messages_text):
        lowered = text.lower().strip()
        if any(p in lowered for p in CORRECTION_PHRASES):
            explicit_corrections += 1
        if i > 0:
            current = user_word_sets[i]
            lookback_start = max(0, i - REPHRASE_LOOKBACK_TURNS)
            for j in range(lookback_start, i):
                if _word_overlap(current, user_word_sets[j]) > REPHRASE_OVERLAP_THRESHOLD:
                    rephrased_retries += 1
                    break

    # ---- Tail / outcome proxies ----
    last_event = events[-1]
    last_event_type = last_event.get("event_type")
    final_tool_status = None
    for e in reversed(events):
        if e.get("event_type") == "tool_result":
            final_tool_status = e.get("tool_status")
            break

    head_size = max(1, event_count // 10)
    tail_size = max(1, event_count // 10)
    head_errors = sum(
        1
        for e in events[:head_size]
        if e.get("error_class") or e.get("tool_status") == "error"
    )
    tail_errors = sum(
        1
        for e in events[-tail_size:]
        if e.get("error_class") or e.get("tool_status") == "error"
    )
    error_concentration_tail = (
        round(tail_errors / head_errors, 2)
        if head_errors > 0
        else (float(tail_errors) if tail_errors > 0 else 0.0)
    )
    events_since_last_user_message = (
        event_count - 1 - last_user_msg_idx if last_user_msg_idx >= 0 else event_count
    )

    return {
        "session_id": session_id,
        "agent": agent,
        "project_path": project_path,
        "project_label": project_label,
        "git_branch_last": git_branch_last,
        "models_used": models_used,
        "primary_model": primary_model,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_clock_minutes": wall_clock_minutes,
        "active_minutes": active_minutes,
        "primary_burst_minutes": primary_burst_minutes,
        "longest_idle_minutes": longest_idle_minutes,
        "bursts": bursts,
        "events_per_active_minute": events_per_active_minute,
        "time_of_day_buckets": dict(time_of_day_buckets),
        "weekday": weekday,
        "event_count": event_count,
        "tool_call_count": tool_call_count,
        "user_turn_count": user_turn_count,
        "subagent_count": subagent_count,
        "subagent_event_count": subagent_event_count,
        "subagent_types_spawned": dict(subagent_types),
        "tool_call_distribution": tool_call_distribution_top10,
        "tool_category_distribution": tool_category_distribution_dict,
        "tool_error_rate": tool_error_rate,
        "tool_error_classes": dict(tool_error_classes),
        "most_repeated_tool_call": most_repeated_tool_call,
        "unique_files_touched": unique_files_touched,
        "top_files_touched": top_files_touched,
        "directory_concentration": directory_concentration,
        "tests_invoked": tests_invoked,
        "git_status_invoked": git_status_invoked,
        "git_diff_invoked": git_diff_invoked,
        "pr_referenced": pr_referenced,
        "assistant_chars": assistant_chars,
        "user_chars": user_chars,
        "reasoning_chars": reasoning_chars,
        "assistant_to_user_text_ratio": assistant_to_user_text_ratio,
        "user_friction_signals": {
            "explicit_corrections": explicit_corrections,
            "rephrased_retries": rephrased_retries,
        },
        "last_event_type": last_event_type,
        "final_tool_status": final_tool_status,
        "error_concentration_tail": error_concentration_tail,
        "events_since_last_user_message": events_since_last_user_message,
        "events_content_hash": _events_content_hash(events),
    }
