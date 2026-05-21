#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TOOL_RESULT_EXIT_RE = re.compile(r"^Exit code:?\s*(-?\d+)\b", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                yield line_number, {"_decode_error": True, "_raw": line}


def truncate_text(value, max_chars: int):
    if value is None:
        return None, 0, False
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    original_length = len(value)
    if original_length <= max_chars:
        return value, original_length, False
    return value[:max_chars], original_length, True


def json_preview(value, max_chars: int):
    if value is None:
        return None, 0, False
    if isinstance(value, str):
        return truncate_text(value, max_chars)
    return truncate_text(json.dumps(value, ensure_ascii=False, sort_keys=True), max_chars)


def join_text_parts(parts):
    texts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if "text" in part and isinstance(part["text"], str):
            texts.append(part["text"])
        elif "thinking" in part and isinstance(part["thinking"], str):
            texts.append(part["thinking"])
        elif "content" in part and isinstance(part["content"], str):
            texts.append(part["content"])
    return "\n\n".join(texts).strip() or None


def parse_tool_output_blob(raw_output):
    if raw_output is None:
        return None, None

    exit_code = None
    output_text = None

    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            metadata = parsed.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("exit_code"), int):
                exit_code = metadata["exit_code"]
            output_text = parsed.get("output") if isinstance(parsed.get("output"), str) else raw_output
        else:
            output_text = raw_output
            match = TOOL_RESULT_EXIT_RE.search(raw_output)
            if match:
                exit_code = int(match.group(1))
    elif isinstance(raw_output, dict):
        metadata = raw_output.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("exit_code"), int):
            exit_code = metadata["exit_code"]
        output_text = raw_output.get("output")
    else:
        output_text = str(raw_output)

    return exit_code, output_text


def tool_status_from_exit_code(exit_code, fallback_status=None):
    if isinstance(exit_code, int):
        return "success" if exit_code == 0 else "error"
    if fallback_status in {"completed", "success"}:
        return "success"
    if fallback_status in {"failed", "error"}:
        return "error"
    return "unknown"


def codex_error_class(raw_text, exit_code):
    if isinstance(exit_code, int) and exit_code != 0:
        return "nonzero_exit"
    if not isinstance(raw_text, str):
        return None
    lowered = raw_text.lower()
    if "permission denied" in lowered:
        return "permission_denied"
    if "requires approval" in lowered:
        return "approval_required"
    return None


def claude_error_class(text, is_error):
    if not is_error:
        return None
    lowered = (text or "").lower()
    if "requires approval" in lowered:
        return "approval_required"
    if "sibling tool call errored" in lowered:
        return "sibling_tool_error"
    if "permission denied" in lowered:
        return "permission_denied"
    return "tool_error"


class EventWriter:
    def __init__(self, output_dir: Path, max_text_chars: int):
        ensure_dir(output_dir)
        self.events_path = output_dir / "events.jsonl"
        self.sessions_path = output_dir / "sessions.jsonl"
        self.summary_path = output_dir / "summary.json"
        self.max_text_chars = max_text_chars
        self.events_handle = self.events_path.open("w", encoding="utf-8")
        self.sessions_handle = self.sessions_path.open("w", encoding="utf-8")
        self.summary = {
            "generated_at": utc_now_iso(),
            "max_text_chars": max_text_chars,
            "events_total": 0,
            "sessions_total": 0,
            "events_by_agent": Counter(),
            "events_by_type": Counter(),
            "tool_calls_by_agent": Counter(),
            "tool_results_by_agent": Counter(),
            "errors_by_agent": Counter(),
            "error_class_counts": Counter(),
            "trace_kind_counts": Counter(),
        }

    def write_event(self, event):
        self.events_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.summary["events_total"] += 1
        self.summary["events_by_agent"][event["agent"]] += 1
        self.summary["events_by_type"][event["event_type"]] += 1
        self.summary["trace_kind_counts"][event["trace_kind"]] += 1
        if event["event_type"] == "tool_call":
            self.summary["tool_calls_by_agent"][event["agent"]] += 1
        if event["event_type"] == "tool_result":
            self.summary["tool_results_by_agent"][event["agent"]] += 1
        if event.get("error_class"):
            self.summary["errors_by_agent"][event["agent"]] += 1
            self.summary["error_class_counts"][event["error_class"]] += 1

    def write_session(self, session_row):
        self.sessions_handle.write(json.dumps(session_row, ensure_ascii=False) + "\n")
        self.summary["sessions_total"] += 1

    def close(self):
        self.events_handle.close()
        self.sessions_handle.close()
        summary_json = {
            "generated_at": self.summary["generated_at"],
            "max_text_chars": self.summary["max_text_chars"],
            "events_total": self.summary["events_total"],
            "sessions_total": self.summary["sessions_total"],
            "events_by_agent": dict(self.summary["events_by_agent"]),
            "events_by_type": dict(self.summary["events_by_type"]),
            "tool_calls_by_agent": dict(self.summary["tool_calls_by_agent"]),
            "tool_results_by_agent": dict(self.summary["tool_results_by_agent"]),
            "errors_by_agent": dict(self.summary["errors_by_agent"]),
            "error_class_counts": dict(self.summary["error_class_counts"]),
            "trace_kind_counts": dict(self.summary["trace_kind_counts"]),
        }
        self.summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


def make_base_event(
    *,
    agent,
    trace_id,
    session_id,
    trace_kind,
    timestamp,
    role,
    event_type,
    raw_path,
    raw_line,
    event_index,
    project=None,
    cwd=None,
    model=None,
    git_branch=None,
    tool_name=None,
    tool_call_id=None,
    tool_status=None,
    error_class=None,
    message_text=None,
    tool_input_preview=None,
    tool_output_preview=None,
    text_length=None,
    input_length=None,
    output_length=None,
    text_truncated=False,
    input_truncated=False,
    output_truncated=False,
    metadata=None,
):
    return {
        "agent": agent,
        "trace_id": trace_id,
        "session_id": session_id,
        "trace_kind": trace_kind,
        "timestamp": timestamp,
        "role": role,
        "event_type": event_type,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "tool_status": tool_status,
        "error_class": error_class,
        "project": project,
        "cwd": cwd,
        "model": model,
        "git_branch": git_branch,
        "message_text": message_text,
        "tool_input_preview": tool_input_preview,
        "tool_output_preview": tool_output_preview,
        "text_length": text_length,
        "input_length": input_length,
        "output_length": output_length,
        "text_truncated": text_truncated,
        "input_truncated": input_truncated,
        "output_truncated": output_truncated,
        "raw_path": raw_path,
        "raw_line": raw_line,
        "event_index": event_index,
        "metadata": metadata or {},
    }


def normalize_codex(raw_root: Path, writer: EventWriter):
    sessions_root = raw_root / "codex" / "sessions"
    files = sorted(sessions_root.glob("**/*.jsonl"))

    for file_path in files:
        rel_path = file_path.relative_to(raw_root)
        trace_id = file_path.stem
        session_id = trace_id
        cwd = None
        git_branch = None
        model = None
        event_index = 0
        tool_name_by_call_id = {}

        session_counts = Counter()
        session_errors = 0
        first_ts = None
        last_ts = None

        for line_number, item in iter_jsonl(file_path):
            if item.get("_decode_error"):
                continue

            timestamp = item.get("timestamp")
            if timestamp:
                if first_ts is None or timestamp < first_ts:
                    first_ts = timestamp
                if last_ts is None or timestamp > last_ts:
                    last_ts = timestamp

            item_type = item.get("type")
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}

            if item_type == "session_meta":
                session_id = payload.get("id") or session_id
                cwd = payload.get("cwd") or cwd
                git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
                git_branch = git.get("branch") or git_branch
                continue

            if item_type == "turn_context":
                cwd = payload.get("cwd") or cwd
                model = payload.get("model") or model
                continue

            if item_type != "response_item":
                continue

            payload_type = payload.get("type")
            event = None

            if payload_type == "message":
                text = join_text_parts(payload.get("content") or [])
                message_text, text_length, text_truncated = truncate_text(text, writer.max_text_chars)
                event = make_base_event(
                    agent="codex",
                    trace_id=trace_id,
                    session_id=session_id,
                    trace_kind="session",
                    timestamp=timestamp,
                    role=payload.get("role"),
                    event_type="message",
                    raw_path=str(rel_path),
                    raw_line=line_number,
                    event_index=event_index,
                    project=cwd,
                    cwd=cwd,
                    model=model,
                    git_branch=git_branch,
                    message_text=message_text,
                    text_length=text_length,
                    text_truncated=text_truncated,
                    metadata={"source_type": payload_type},
                )
            elif payload_type == "reasoning":
                text = join_text_parts(payload.get("summary") or [])
                message_text, text_length, text_truncated = truncate_text(text, writer.max_text_chars)
                event = make_base_event(
                    agent="codex",
                    trace_id=trace_id,
                    session_id=session_id,
                    trace_kind="session",
                    timestamp=timestamp,
                    role="assistant",
                    event_type="reasoning",
                    raw_path=str(rel_path),
                    raw_line=line_number,
                    event_index=event_index,
                    project=cwd,
                    cwd=cwd,
                    model=model,
                    git_branch=git_branch,
                    message_text=message_text,
                    text_length=text_length,
                    text_truncated=text_truncated,
                    metadata={"source_type": payload_type},
                )
            elif payload_type in {"function_call", "custom_tool_call"}:
                tool_name = payload.get("name")
                tool_call_id = payload.get("call_id")
                if tool_call_id and tool_name:
                    tool_name_by_call_id[tool_call_id] = tool_name
                tool_input_preview, input_length, input_truncated = json_preview(
                    payload.get("arguments") if payload_type == "function_call" else payload.get("input"),
                    writer.max_text_chars,
                )
                event = make_base_event(
                    agent="codex",
                    trace_id=trace_id,
                    session_id=session_id,
                    trace_kind="session",
                    timestamp=timestamp,
                    role="assistant",
                    event_type="tool_call",
                    raw_path=str(rel_path),
                    raw_line=line_number,
                    event_index=event_index,
                    project=cwd,
                    cwd=cwd,
                    model=model,
                    git_branch=git_branch,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_status=tool_status_from_exit_code(None, payload.get("status")),
                    tool_input_preview=tool_input_preview,
                    input_length=input_length,
                    input_truncated=input_truncated,
                    metadata={"source_type": payload_type},
                )
            elif payload_type in {"function_call_output", "custom_tool_call_output"}:
                raw_output = payload.get("output")
                tool_call_id = payload.get("call_id")
                exit_code, output_text = parse_tool_output_blob(raw_output)
                tool_output_preview, output_length, output_truncated = truncate_text(output_text, writer.max_text_chars)
                error_class = codex_error_class(output_text, exit_code)
                event = make_base_event(
                    agent="codex",
                    trace_id=trace_id,
                    session_id=session_id,
                    trace_kind="session",
                    timestamp=timestamp,
                    role="tool",
                    event_type="tool_result",
                    raw_path=str(rel_path),
                    raw_line=line_number,
                    event_index=event_index,
                    project=cwd,
                    cwd=cwd,
                    model=model,
                    git_branch=git_branch,
                    tool_name=tool_name_by_call_id.get(tool_call_id),
                    tool_call_id=tool_call_id,
                    tool_status=tool_status_from_exit_code(exit_code),
                    error_class=error_class,
                    tool_output_preview=tool_output_preview,
                    output_length=output_length,
                    output_truncated=output_truncated,
                    metadata={"source_type": payload_type, "exit_code": exit_code},
                )
            elif payload_type == "web_search_call":
                action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
                tool_input_preview, input_length, input_truncated = json_preview(action, writer.max_text_chars)
                event = make_base_event(
                    agent="codex",
                    trace_id=trace_id,
                    session_id=session_id,
                    trace_kind="session",
                    timestamp=timestamp,
                    role="assistant",
                    event_type="tool_call",
                    raw_path=str(rel_path),
                    raw_line=line_number,
                    event_index=event_index,
                    project=cwd,
                    cwd=cwd,
                    model=model,
                    git_branch=git_branch,
                    tool_name="web_search",
                    tool_status=tool_status_from_exit_code(None, payload.get("status")),
                    tool_input_preview=tool_input_preview,
                    input_length=input_length,
                    input_truncated=input_truncated,
                    metadata={"source_type": payload_type},
                )

            if event is None:
                continue

            writer.write_event(event)
            session_counts[event["event_type"]] += 1
            if event.get("error_class"):
                session_errors += 1
            event_index += 1

        writer.write_session(
            {
                "agent": "codex",
                "trace_id": trace_id,
                "session_id": session_id,
                "trace_kind": "session",
                "raw_path": str(rel_path),
                "project": cwd,
                "cwd": cwd,
                "model": model,
                "git_branch": git_branch,
                "start_timestamp": first_ts,
                "end_timestamp": last_ts,
                "event_count": sum(session_counts.values()),
                "message_count": session_counts["message"],
                "reasoning_count": session_counts["reasoning"],
                "tool_call_count": session_counts["tool_call"],
                "tool_result_count": session_counts["tool_result"],
                "error_count": session_errors,
            }
        )


def normalize_claude(raw_root: Path, writer: EventWriter):
    projects_root = raw_root / "claude" / "projects"
    files = sorted(projects_root.glob("**/*.jsonl"))

    for file_path in files:
        rel_path = file_path.relative_to(raw_root)
        rel_from_projects = file_path.relative_to(projects_root)
        trace_kind = "subagent" if "subagents" in rel_from_projects.parts else "top_level"
        project_key = rel_from_projects.parts[0] if rel_from_projects.parts else None

        if trace_kind == "subagent":
            session_id = rel_from_projects.parts[1]
            trace_id = f"{session_id}:{file_path.stem}"
        else:
            session_id = file_path.stem
            trace_id = session_id

        cwd = None
        git_branch = None
        model = None
        event_index = 0
        tool_name_by_call_id = {}

        session_counts = Counter()
        session_errors = 0
        first_ts = None
        last_ts = None

        for line_number, item in iter_jsonl(file_path):
            if item.get("_decode_error"):
                continue

            timestamp = item.get("timestamp")
            if timestamp:
                if first_ts is None or timestamp < first_ts:
                    first_ts = timestamp
                if last_ts is None or timestamp > last_ts:
                    last_ts = timestamp

            cwd = item.get("cwd") or cwd
            git_branch = item.get("gitBranch") or git_branch
            message = item.get("message") if isinstance(item.get("message"), dict) else {}
            model = message.get("model") or model

            item_type = item.get("type")
            role = message.get("role") or item_type
            content = message.get("content")

            if item_type not in {"user", "assistant"}:
                continue

            if isinstance(content, str):
                message_text, text_length, text_truncated = truncate_text(content, writer.max_text_chars)
                event = make_base_event(
                    agent="claude",
                    trace_id=trace_id,
                    session_id=item.get("sessionId") or session_id,
                    trace_kind=trace_kind,
                    timestamp=timestamp,
                    role=role,
                    event_type="message",
                    raw_path=str(rel_path),
                    raw_line=line_number,
                    event_index=event_index,
                    project=project_key,
                    cwd=cwd,
                    model=model,
                    git_branch=git_branch,
                    message_text=message_text,
                    text_length=text_length,
                    text_truncated=text_truncated,
                    metadata={"source_type": item_type, "uuid": item.get("uuid")},
                )
                writer.write_event(event)
                session_counts["message"] += 1
                event_index += 1
                continue

            if not isinstance(content, list):
                continue

            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    continue

                event = None
                part_type = part.get("type")

                if part_type == "text":
                    message_text, text_length, text_truncated = truncate_text(part.get("text"), writer.max_text_chars)
                    event = make_base_event(
                        agent="claude",
                        trace_id=trace_id,
                        session_id=item.get("sessionId") or session_id,
                        trace_kind=trace_kind,
                        timestamp=timestamp,
                        role=role,
                        event_type="message",
                        raw_path=str(rel_path),
                        raw_line=line_number,
                        event_index=event_index,
                        project=project_key,
                        cwd=cwd,
                        model=model,
                        git_branch=git_branch,
                        message_text=message_text,
                        text_length=text_length,
                        text_truncated=text_truncated,
                        metadata={"source_type": item_type, "part_index": part_index, "uuid": item.get("uuid")},
                    )
                elif part_type == "thinking":
                    message_text, text_length, text_truncated = truncate_text(part.get("thinking"), writer.max_text_chars)
                    event = make_base_event(
                        agent="claude",
                        trace_id=trace_id,
                        session_id=item.get("sessionId") or session_id,
                        trace_kind=trace_kind,
                        timestamp=timestamp,
                        role="assistant",
                        event_type="reasoning",
                        raw_path=str(rel_path),
                        raw_line=line_number,
                        event_index=event_index,
                        project=project_key,
                        cwd=cwd,
                        model=model,
                        git_branch=git_branch,
                        message_text=message_text,
                        text_length=text_length,
                        text_truncated=text_truncated,
                        metadata={"source_type": item_type, "part_index": part_index, "uuid": item.get("uuid")},
                    )
                elif part_type == "tool_use":
                    tool_name = part.get("name")
                    tool_call_id = part.get("id")
                    if tool_call_id and tool_name:
                        tool_name_by_call_id[tool_call_id] = tool_name
                    tool_input_preview, input_length, input_truncated = json_preview(part.get("input"), writer.max_text_chars)
                    event = make_base_event(
                        agent="claude",
                        trace_id=trace_id,
                        session_id=item.get("sessionId") or session_id,
                        trace_kind=trace_kind,
                        timestamp=timestamp,
                        role="assistant",
                        event_type="tool_call",
                        raw_path=str(rel_path),
                        raw_line=line_number,
                        event_index=event_index,
                        project=project_key,
                        cwd=cwd,
                        model=model,
                        git_branch=git_branch,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_status="unknown",
                        tool_input_preview=tool_input_preview,
                        input_length=input_length,
                        input_truncated=input_truncated,
                        metadata={"source_type": item_type, "part_index": part_index, "uuid": item.get("uuid")},
                    )
                elif part_type == "tool_result":
                    output_text = part.get("content")
                    tool_call_id = part.get("tool_use_id")
                    tool_output_preview, output_length, output_truncated = truncate_text(output_text, writer.max_text_chars)
                    error_class = claude_error_class(output_text, bool(part.get("is_error")))
                    event = make_base_event(
                        agent="claude",
                        trace_id=trace_id,
                        session_id=item.get("sessionId") or session_id,
                        trace_kind=trace_kind,
                        timestamp=timestamp,
                        role="tool",
                        event_type="tool_result",
                        raw_path=str(rel_path),
                        raw_line=line_number,
                        event_index=event_index,
                        project=project_key,
                        cwd=cwd,
                        model=model,
                        git_branch=git_branch,
                        tool_name=tool_name_by_call_id.get(tool_call_id),
                        tool_call_id=tool_call_id,
                        tool_status="error" if part.get("is_error") else "success",
                        error_class=error_class,
                        tool_output_preview=tool_output_preview,
                        output_length=output_length,
                        output_truncated=output_truncated,
                        metadata={"source_type": item_type, "part_index": part_index, "uuid": item.get("uuid")},
                    )

                if event is None:
                    continue

                writer.write_event(event)
                session_counts[event["event_type"]] += 1
                if event.get("error_class"):
                    session_errors += 1
                event_index += 1

        writer.write_session(
            {
                "agent": "claude",
                "trace_id": trace_id,
                "session_id": session_id,
                "trace_kind": trace_kind,
                "raw_path": str(rel_path),
                "project": project_key,
                "cwd": cwd,
                "model": model,
                "git_branch": git_branch,
                "start_timestamp": first_ts,
                "end_timestamp": last_ts,
                "event_count": sum(session_counts.values()),
                "message_count": session_counts["message"],
                "reasoning_count": session_counts["reasoning"],
                "tool_call_count": session_counts["tool_call"],
                "tool_result_count": session_counts["tool_result"],
                "error_count": session_errors,
            }
        )


def _load_gemini_project_map(gemini_root: Path) -> dict[str, str]:
    """Return {dirname_or_hash: real_path} from ~/.gemini/projects.json.

    Gemini CLI stores chat dirs under ~/.gemini/tmp/<name_or_hash>/chats/. A
    `projects.json` at the root maps real paths to optional human-readable
    aliases. For each known path we index both the alias (if set) and the
    SHA-256 hex of the path, so either dirname shape resolves back.
    """
    import hashlib

    projects_file = gemini_root / "projects.json"
    if not projects_file.exists():
        return {}
    try:
        data = json.loads(projects_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    mapping: dict[str, str] = {}
    for real_path, alias in (data.get("projects") or {}).items():
        if alias:
            mapping[alias] = real_path
        digest = hashlib.sha256(real_path.encode("utf-8")).hexdigest()
        mapping[digest] = real_path
    return mapping


def _gemini_error_class(text, tool_status):
    lowered = (text or "").lower()
    if tool_status == "error":
        if "permission" in lowered and "denied" in lowered:
            return "permission_denied"
        if "approval" in lowered and "required" in lowered:
            return "approval_required"
        return "tool_error"
    return None


def normalize_gemini(raw_root: Path, writer: EventWriter):
    gemini_root = raw_root / "gemini"
    tmp_root = gemini_root / "tmp"
    if not tmp_root.exists():
        return

    project_map = _load_gemini_project_map(gemini_root)
    files = sorted(tmp_root.glob("*/chats/session-*.json"))

    for file_path in files:
        rel_path = file_path.relative_to(raw_root)
        project_dirname = file_path.parent.parent.name
        cwd = project_map.get(project_dirname)
        project_key = cwd or project_dirname

        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        session_id = payload.get("sessionId") or file_path.stem
        trace_id = session_id
        model = None
        messages = payload.get("messages") or []

        event_index = 0
        session_counts = Counter()
        session_errors = 0
        first_ts = payload.get("startTime")
        last_ts = payload.get("lastUpdated") or first_ts

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            timestamp = msg.get("timestamp")
            if timestamp:
                if first_ts is None or timestamp < first_ts:
                    first_ts = timestamp
                if last_ts is None or timestamp > last_ts:
                    last_ts = timestamp

            # Map Gemini's message types to the cross-agent role vocabulary.
            # `info` and `error` are system-emitted notices from the CLI itself.
            if msg_type == "user":
                role = "user"
            elif msg_type == "gemini":
                role = "assistant"
                model = msg.get("model") or model
            elif msg_type in {"info", "error"}:
                role = "system"
            else:
                continue

            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                text, text_length, text_truncated = truncate_text(content, writer.max_text_chars)
                error_class = "tool_error" if msg_type == "error" else None
                event = make_base_event(
                    agent="gemini",
                    trace_id=trace_id,
                    session_id=session_id,
                    trace_kind="session",
                    timestamp=timestamp,
                    role=role,
                    event_type="message",
                    raw_path=str(rel_path),
                    raw_line=event_index,
                    event_index=event_index,
                    project=project_key,
                    cwd=cwd,
                    model=model,
                    message_text=text,
                    text_length=text_length,
                    text_truncated=text_truncated,
                    error_class=error_class,
                )
                writer.write_event(event)
                session_counts["message"] += 1
                event_index += 1
                if error_class:
                    session_errors += 1

            # Gemini reasoning lives in a structured `thoughts` list.
            thoughts = msg.get("thoughts")
            if isinstance(thoughts, list) and thoughts:
                thought_text = "\n\n".join(
                    f"{t.get('subject', '')}\n{t.get('description', '')}".strip()
                    for t in thoughts if isinstance(t, dict)
                ).strip()
                if thought_text:
                    text, text_length, text_truncated = truncate_text(thought_text, writer.max_text_chars)
                    writer.write_event(
                        make_base_event(
                            agent="gemini",
                            trace_id=trace_id,
                            session_id=session_id,
                            trace_kind="session",
                            timestamp=timestamp,
                            role="assistant",
                            event_type="reasoning",
                            raw_path=str(rel_path),
                            raw_line=event_index,
                            event_index=event_index,
                            project=project_key,
                            cwd=cwd,
                            model=model,
                            message_text=text,
                            text_length=text_length,
                            text_truncated=text_truncated,
                        )
                    )
                    session_counts["reasoning"] += 1
                    event_index += 1

            # Tool calls on gemini messages: emit a tool_call + tool_result pair.
            tool_calls = msg.get("toolCalls") or []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tool_name = tc.get("name") or "(unknown)"
                tool_call_id = tc.get("id")
                status_raw = tc.get("status")
                tool_status = (
                    "success" if status_raw in {"success", "succeeded", "completed"}
                    else "error" if status_raw in {"error", "failed"}
                    else "unknown"
                )
                tool_ts = tc.get("timestamp") or timestamp
                args = tc.get("args")
                args_preview, input_length, input_truncated = json_preview(args, writer.max_text_chars)

                writer.write_event(
                    make_base_event(
                        agent="gemini",
                        trace_id=trace_id,
                        session_id=session_id,
                        trace_kind="session",
                        timestamp=tool_ts,
                        role="assistant",
                        event_type="tool_call",
                        raw_path=str(rel_path),
                        raw_line=event_index,
                        event_index=event_index,
                        project=project_key,
                        cwd=cwd,
                        model=model,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_input_preview=args_preview,
                        input_length=input_length,
                        input_truncated=input_truncated,
                    )
                )
                session_counts["tool_call"] += 1
                event_index += 1

                # Tool result: Gemini nests the actual response inside a list
                # like [{ functionResponse: { response: { output: "..." } } }].
                result = tc.get("result")
                result_text = None
                if isinstance(result, list):
                    chunks = []
                    for r in result:
                        if not isinstance(r, dict):
                            continue
                        fr = r.get("functionResponse") or {}
                        resp = fr.get("response") or {}
                        out = resp.get("output") or resp.get("error") or resp.get("content")
                        if isinstance(out, str):
                            chunks.append(out)
                        elif out is not None:
                            chunks.append(json.dumps(out, ensure_ascii=False))
                    if chunks:
                        result_text = "\n\n".join(chunks)
                elif isinstance(result, (str, dict, list)):
                    result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

                output_preview, output_length, output_truncated = json_preview(result_text, writer.max_text_chars)
                error_class = _gemini_error_class(result_text, tool_status)
                writer.write_event(
                    make_base_event(
                        agent="gemini",
                        trace_id=trace_id,
                        session_id=session_id,
                        trace_kind="session",
                        timestamp=tool_ts,
                        role="tool",
                        event_type="tool_result",
                        raw_path=str(rel_path),
                        raw_line=event_index,
                        event_index=event_index,
                        project=project_key,
                        cwd=cwd,
                        model=model,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_status=tool_status,
                        error_class=error_class,
                        tool_output_preview=output_preview,
                        output_length=output_length,
                        output_truncated=output_truncated,
                    )
                )
                session_counts["tool_result"] += 1
                event_index += 1
                if error_class:
                    session_errors += 1

        writer.write_session(
            {
                "agent": "gemini",
                "trace_id": trace_id,
                "session_id": session_id,
                "trace_kind": "session",
                "raw_path": str(rel_path),
                "project": project_key,
                "cwd": cwd,
                "model": model,
                "git_branch": None,
                "start_timestamp": first_ts,
                "end_timestamp": last_ts,
                "event_count": sum(session_counts.values()),
                "message_count": session_counts["message"],
                "reasoning_count": session_counts["reasoning"],
                "tool_call_count": session_counts["tool_call"],
                "tool_result_count": session_counts["tool_result"],
                "error_count": session_errors,
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Normalize copied Codex and Claude trace data into a shared event schema.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Location of the copied raw trace snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "normalized",
        help="Directory for normalized outputs.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=2000,
        help="Maximum characters to keep for message and tool preview fields.",
    )
    args = parser.parse_args()

    from . import normalizers as _norm

    warnings = _norm.initialize()
    for w in warnings:
        print(f"warning: {w}", file=__import__("sys").stderr)

    writer = EventWriter(args.output_dir, args.max_text_chars)
    try:
        for n in _norm.get_all():
            try:
                n.normalize(args.raw_root, writer)
            except Exception as exc:
                # One broken normalizer shouldn't kill the run — surface
                # the failure and continue with the rest. User-defined
                # normalizers are most likely to misbehave here.
                print(
                    f"warning: normalizer {n.name!r} ({n.source}) raised "
                    f"{type(exc).__name__}: {exc}",
                    file=__import__("sys").stderr,
                )
    finally:
        writer.close()

    summary = json.loads(writer.summary_path.read_text(encoding="utf-8"))
    print("Normalized Trace Events")
    print(f"Generated: {summary['generated_at']}")
    print(f"Output dir: {args.output_dir}")
    print(f"- Sessions indexed: {summary['sessions_total']}")
    print(f"- Events written: {summary['events_total']}")
    print(f"- Events by agent: {summary['events_by_agent']}")
    print(f"- Events by type: {summary['events_by_type']}")
    print(f"- Error classes: {summary['error_class_counts']}")


if __name__ == "__main__":
    main()
