"""Claude Code hook entry point.

Invoked by Claude Code as a child process on PostToolUse / PreToolUse /
SessionStart events. Reads the hook event JSON from stdin, updates the
relevant session's rolling state, runs the rules, and writes a nudge to
stdout so it's injected back into Claude's context as a system note.

Design constraints:

- MUST be silent on the happy path — blank stdout when no rule fires.
- MUST NEVER crash the user's session. Any exception becomes a logged
  warning and an exit code of 0.
- MUST be fast (<50ms typical). No LLM calls. No network. Pure local rule
  evaluation over at most 80 in-memory events.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from . import experiments
from . import rating_lookup
from . import rules
from . import state


# Error-class markers in tool responses. Claude Code hook payloads don't
# standardize an error_class field, so we sniff common strings.
_ERROR_STRINGS = {
    "permission_denied": ("permission denied", "requires approval"),
    "approval_required": ("requires approval", "approval required"),
    "tool_error": ("error:", "failed", "exception"),
}


def _classify_error(tool_response) -> tuple[bool, str | None]:
    """Return (is_error, error_class) by inspecting the tool response payload.

    Claude Code hook payloads are loosely-typed. We look for an explicit
    `is_error` flag, then fall back to string sniffing on the response body.
    """
    if tool_response is None:
        return False, None
    if isinstance(tool_response, dict):
        explicit = tool_response.get("is_error")
        if isinstance(explicit, bool):
            if not explicit:
                return False, None
            text = str(tool_response.get("content") or tool_response.get("output") or "")
            for klass, needles in _ERROR_STRINGS.items():
                if any(n in text.lower() for n in needles):
                    return True, klass
            return True, "tool_error"
        # No explicit flag — sniff the response text.
        text = str(tool_response.get("content") or tool_response.get("output") or "")
    elif isinstance(tool_response, str):
        text = tool_response
    else:
        return False, None
    lower = text.lower()
    for klass, needles in _ERROR_STRINGS.items():
        if any(n in lower for n in needles):
            return True, klass
    return False, None


def _target_path(tool_name: str, tool_input) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "path", "filename", "target_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _build_nudge_output(nudge: rules.Nudge) -> str:
    """Claude Code injects hook stdout into the session context as a note.

    We prepend a stable marker so downstream tools (and the user) can
    recognise coach notes unambiguously.
    """
    return f"[tessera-live · rule:{nudge.rule_key}] {nudge.message.strip()}\n"


def _read_hook_input() -> dict:
    raw = sys.stdin.read()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _coach_disabled() -> bool:
    flag = os.environ.get("TESSERA_LIVE", "").strip().lower()
    return flag in {"0", "off", "disabled", "false", "no"}


def main() -> int:
    try:
        payload = _read_hook_input()
    except Exception:
        return 0

    if _coach_disabled():
        return 0

    event_name = payload.get("hook_event_name")
    session_id = payload.get("session_id") or "unknown"
    cwd = payload.get("cwd") or os.getcwd()

    # Opportunistic cleanup — fire-and-forget, errors swallowed.
    try:
        state.prune_stale_sessions()
    except Exception:
        pass

    try:
        session_state = state.load_or_create(session_id, cwd=cwd)
    except Exception:
        return 0

    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input")
    tool_response = payload.get("tool_response")

    # Track read-before-edit context (lightweight — just path list).
    if event_name == "PostToolUse" and tool_name and rules.is_read_tool(tool_name):
        target = _target_path(tool_name, tool_input)
        if target and target not in session_state.read_files:
            session_state.read_files.append(target)
            # Keep list bounded so we don't grow it unbounded
            if len(session_state.read_files) > 200:
                session_state.read_files = session_state.read_files[-200:]

    # Only PostToolUse events feed the error-class rules.
    if event_name == "PostToolUse" and tool_name:
        is_error, error_class = _classify_error(tool_response)
        state.record_event(
            session_state,
            tool_name=tool_name,
            tool_input=tool_input,
            is_error=is_error,
            error_class=error_class,
            target_path=_target_path(tool_name, tool_input),
        )
    elif event_name == "PreToolUse" and tool_name:
        # Lightweight tracking — don't count pre-use events toward rule windows.
        pass

    nudge = rules.evaluate(session_state)

    # Cross-session check — does the user's past ratings tell us to suppress
    # this nudge, or to attach evidence from a prior observation?
    signal: rating_lookup.RuleSignal | None = None
    if nudge:
        try:
            signal = rating_lookup.get_rule_signal(
                nudge.rule_key, session_state.project
            )
        except Exception:
            signal = None

    if nudge and signal and signal.suppress:
        # User rated this pattern as wrong in the recent past — suppress
        # the nudge but still record that the rule fired so the weekly
        # retro can see the suppressed hit.
        session_state.log_fire(
            nudge.rule_key, f"[suppressed by prior rating] {nudge.message}"
        )
        session_state.suppress(nudge.rule_key, nudge.suppress_until_events)
        nudge = None

    if nudge:
        # Attach any useful/known evidence line from recent rated runs.
        if signal:
            evidence = signal.evidence_line()
            if evidence:
                nudge.message = f"{nudge.message}\n\n{evidence}"
        session_state.log_fire(nudge.rule_key, nudge.message)
        session_state.suppress(nudge.rule_key, nudge.suppress_until_events)
        # Create a pending experiment if this is the first time we've
        # seen this (rule, project) pair — feeds the next weekly observation run.
        try:
            experiments.add_if_novel(
                trigger_rule=nudge.rule_key,
                project=session_state.project,
                session_id=session_id,
                context=nudge.message[:400],
            )
        except Exception:
            pass

    try:
        state.save(session_state)
    except Exception:
        pass

    if nudge:
        sys.stdout.write(_build_nudge_output(nudge))
        sys.stdout.flush()

    return 0


def cli() -> int:
    """Console-script entry point wired in pyproject.toml.

    Wraps ``main()`` with a top-level try/except so any unexpected failure
    in the coach is swallowed and never bubbles up to the user's session.
    """
    try:
        return main()
    except Exception:
        # Write a short diagnostic to a log file but don't print anything —
        # we never want to leak a traceback into Claude's context.
        try:
            log_dir = Path.home() / ".cache" / "tessera-live"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "hook_errors.log").open("a", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(cli())
