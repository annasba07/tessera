"""Fast rules for detecting known waste patterns mid-session.

Each rule is a pure function ``(SessionState) -> Nudge | None``. Rules run
without any LLM call; they're regex-style checks over the event window.
When a rule fires the coach sets a cooldown so the same rule doesn't
re-fire on every subsequent hook invocation in the same session.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .state import Event, SessionState


@dataclass
class Nudge:
    rule_key: str
    message: str
    # After firing, don't re-fire this rule until state.event_seq advances this much.
    suppress_until_events: int = 20


# ---------------------------------------------------------------------------
# Tool-category helpers
# ---------------------------------------------------------------------------

# Using small substring checks so we match across different CLI vocabularies
# (Claude's "Bash" vs Codex's "exec_command" vs Gemini's "run_shell_command").

_BROWSER_SUBSTRINGS = ("browser", "chrome", "playwright", "puppeteer")
_EDIT_SUBSTRINGS = ("edit", "write", "apply_patch", "create_file", "multi_edit")
_READ_SUBSTRINGS = ("read", "grep", "glob", "ls", "list_directory", "view")
_SHELL_SUBSTRINGS = ("bash", "shell", "exec_command", "run_shell", "shell_command")
_DELEGATION_SUBSTRINGS = ("task", "subagent", "delegate", "dispatch_agent", "spawn")
_VERIFY_COMMAND_SUBSTRINGS = (
    "test", "pytest", "jest", "vitest", "mocha", "cargo test", "go test",
    "npm test", "yarn test", "pnpm test",
    "build", "compile", "cargo build", "npm run build",
    "lint", "tsc", "mypy", "ruff", "eslint", "cargo check",
    "commit", "git commit", "git push",
)


def _has_any(name: str, substrings: tuple[str, ...]) -> bool:
    lower = (name or "").lower()
    return any(s in lower for s in substrings)


def is_browser_tool(name: str) -> bool:
    return _has_any(name, _BROWSER_SUBSTRINGS)


def is_edit_tool(name: str) -> bool:
    return _has_any(name, _EDIT_SUBSTRINGS)


def is_read_tool(name: str) -> bool:
    return _has_any(name, _READ_SUBSTRINGS)


def is_shell_tool(name: str) -> bool:
    return _has_any(name, _SHELL_SUBSTRINGS)


def is_delegation_tool(name: str) -> bool:
    return _has_any(name, _DELEGATION_SUBSTRINGS)


def _looks_like_verification(event: Event) -> bool:
    """Best-effort: did this event run a test / build / lint / commit?

    We only have the tool name and an input hash here, so we conservatively
    assume a shell call is a verification only if the input_hash is
    different from the known-verify commands. This is weaker than what
    `digest.py` does with the full command text; the trade-off is speed —
    hooks can't afford to re-parse the tool input.
    """
    return is_shell_tool(event.tool_name)
    # Note: we can't see command text from hashes alone. The hook will also
    # stash the first 120 chars of shell commands in state (future work); for
    # now, all shell calls count as possible verification to avoid false
    # positives on edit_without_verify.


# ---------------------------------------------------------------------------
# Rule 1 — browser_spiral
# ---------------------------------------------------------------------------


def rule_browser_spiral(state: SessionState) -> Nudge | None:
    if state.is_suppressed("browser_spiral"):
        return None
    recent = state.events[-10:]
    errors = [e for e in recent if is_browser_tool(e.tool_name) and e.is_error]
    if len(errors) < 3:
        return None
    tools = sorted({e.tool_name for e in errors})
    return Nudge(
        rule_key="browser_spiral",
        message=(
            f"Coach note: {len(errors)} browser-tool errors in the last 10 calls "
            f"({', '.join(tools)}). This looks like a browser spiral — a pattern "
            "that historically burns the session without reaching a verified state. "
            "Consider: stop the current approach, fall back to Read + a direct "
            "API/curl check to confirm state, and only return to browser tools "
            "after diagnosing why the current selectors are failing."
        ),
    )


# ---------------------------------------------------------------------------
# Rule 2 — retry_without_change
# ---------------------------------------------------------------------------


def rule_retry_without_change(state: SessionState) -> Nudge | None:
    if state.is_suppressed("retry_without_change"):
        return None
    recent = state.events[-20:]
    # Group by (tool_name, input_hash); count failures only.
    seen = Counter(
        (e.tool_name, e.tool_input_hash)
        for e in recent
        if e.is_error and e.tool_input_hash
    )
    if not seen:
        return None
    (tool, _), count = seen.most_common(1)[0]
    if count < 3:
        return None
    return Nudge(
        rule_key="retry_without_change",
        message=(
            f"Coach note: the same {tool} call has failed {count} times in the "
            "last 20 events with the exact same input. Retrying without changing "
            "the inputs or running a diagnostic is the classic blind-retry "
            "pattern. Before the next attempt, run a small read-only probe "
            "that would surface the real cause of the failure."
        ),
    )


# ---------------------------------------------------------------------------
# Rule 3 — permission_wall_repeat
# ---------------------------------------------------------------------------


_PERMISSION_ERROR_CLASSES = {"permission_denied", "approval_required"}


def rule_permission_wall_repeat(state: SessionState) -> Nudge | None:
    if state.is_suppressed("permission_wall_repeat"):
        return None
    recent = state.events[-30:]
    perm_errors = [
        e for e in recent
        if e.error_class in _PERMISSION_ERROR_CLASSES
    ]
    if len(perm_errors) < 2:
        return None
    # Only nudge if the same tool keeps hitting the wall.
    tool_counts = Counter(e.tool_name for e in perm_errors)
    (tool, count) = tool_counts.most_common(1)[0]
    if count < 2:
        return None
    return Nudge(
        rule_key="permission_wall_repeat",
        message=(
            f"Coach note: {tool} has been blocked by permission "
            f"{count} times in this session. Each permission denial pauses the "
            "workflow and the same wall is being retried rather than "
            "pre-authorized. Consider adding the needed allow-pattern to the "
            "project's permissions config (or to .claude/settings.local.json "
            "for Claude Code) before the next attempt."
        ),
    )


# ---------------------------------------------------------------------------
# Rule 4 — runaway (high call rate in a short window)
# ---------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def rule_runaway(state: SessionState) -> Nudge | None:
    if state.is_suppressed("runaway"):
        return None
    if len(state.events) < 30:
        return None
    window = state.events[-30:]
    t_first = _parse_ts(window[0].timestamp)
    t_last = _parse_ts(window[-1].timestamp)
    if not t_first or not t_last:
        return None
    span_seconds = (t_last - t_first).total_seconds()
    if span_seconds <= 0 or span_seconds > 60:
        return None
    rate = 30 / span_seconds  # calls per second in this window
    return Nudge(
        rule_key="runaway",
        message=(
            f"Coach note: 30 tool calls in the last {span_seconds:.0f}s "
            f"(≈{rate * 60:.0f}/min). This cadence usually means a tight "
            "retry loop or runaway script, not purposeful work. Stop and "
            "read the last tool result before firing the next batch — "
            "confirm you're making progress, not cycling on the same error."
        ),
    )


# ---------------------------------------------------------------------------
# Rule 5 — edit_without_verify
# ---------------------------------------------------------------------------


def rule_edit_without_verify(state: SessionState) -> Nudge | None:
    if state.is_suppressed("edit_without_verify"):
        return None
    edits = [e for e in state.events if is_edit_tool(e.tool_name)]
    if len(edits) < 5:
        return None
    verifies = [e for e in state.events if _looks_like_verification(e)]
    if verifies:
        return None
    return Nudge(
        rule_key="edit_without_verify",
        message=(
            f"Coach note: {len(edits)} edits in this session with no shell / "
            "test / build / lint call between them. The pattern that "
            "historically ships broken code is 'edit-edit-edit-done' without "
            "a check. Consider running the project's test or build command "
            "now before more edits — catching an error on edit #5 is much "
            "cheaper than catching it on edit #15."
        ),
        suppress_until_events=30,
    )


# ---------------------------------------------------------------------------
# Rule 6 — delegation_sprawl
# ---------------------------------------------------------------------------


def rule_delegation_sprawl(state: SessionState) -> Nudge | None:
    if state.is_suppressed("delegation_sprawl"):
        return None
    delegations = [e for e in state.events if is_delegation_tool(e.tool_name)]
    if len(delegations) < 4:
        return None
    return Nudge(
        rule_key="delegation_sprawl",
        message=(
            f"Coach note: {len(delegations)} subagents spawned in this "
            "session. Coordination cost tends to eat the leverage after "
            "~3 concurrent subagents. Consider consolidating the remaining "
            "work into one subagent with a broader brief, or switch to "
            "sequential execution for the next steps."
        ),
        suppress_until_events=30,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ALL_RULES: list[Callable[[SessionState], Nudge | None]] = [
    rule_browser_spiral,
    rule_retry_without_change,
    rule_permission_wall_repeat,
    rule_runaway,
    rule_edit_without_verify,
    rule_delegation_sprawl,
]


def evaluate(state: SessionState) -> Nudge | None:
    """Run all rules and return the first nudge, or None if silent."""
    for rule in ALL_RULES:
        try:
            nudge = rule(state)
        except Exception:
            # A buggy rule must never break the user's session.
            continue
        if nudge:
            return nudge
    return None
