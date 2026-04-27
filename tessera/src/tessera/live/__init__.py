"""In-session coach for Claude Code.

The coach watches Claude Code sessions via PostToolUse / PreToolUse /
SessionStart hooks and nudges Claude (not the user) when the session looks
like it's sliding into a known waste pattern — things like browser spirals,
blind retries, permission-wall loops, runaway tool-call bursts, edits
without verification, or delegation sprawl.

The nudge is injected into Claude's context as a short system note. Claude
then decides whether to mention it to the user, silently course-correct, or
ignore if the context genuinely differs.

All state lives under ~/.cache/tessera-live/ (session state, cooldowns) and
~/.config/tessera/experiments/ (proposed experiments the coach creates
when a rule fires for the first time).
"""

__all__ = []
