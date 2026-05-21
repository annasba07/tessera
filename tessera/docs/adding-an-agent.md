# Adding your own coding-agent CLI

Tessera ships with built-in normalizers for **Claude Code**, **OpenAI Codex**, and **Google Gemini CLI**. Adding a fourth (Aider, Cursor, Continue.dev, Cline, Roo Code — whatever you use) is one Python file. The narrative + synthesis + dashboard layers are agent-agnostic — they consume a shared event schema and don't care which CLI produced the events.

## The short version

1. Find where your agent writes its session traces.
2. Write a `normalize_<agent>(raw_root, writer)` function that walks those traces and emits standard events.
3. Drop the file at `~/.config/tessera/normalizers/<agent>.py` — it's auto-loaded on every run.
4. `tessera doctor` confirms it's registered. `tessera run` picks it up alongside the built-ins.

No fork, no PR, no rebuild. If you do want to share your normalizer back — a PR adding `normalize_<agent>()` to `_normalize_script.py` is the long-term home.

## The event schema

Your normalizer's job is to read one agent's trace format and emit two things via the provided `writer`:

- **Events** — every message, reasoning chunk, tool call, and tool result, with timestamps and short text previews.
- **Sessions** — one summary per session with start/end times, project path, model name, event counts.

The schema is intentionally minimal. Six event types cover everything tessera needs:

| `event_type` | When | Notable fields |
|---|---|---|
| `message` | user turn or assistant text | `role`, `text_preview`, `text_chars` |
| `reasoning` | assistant thinking block | `text_preview`, `text_chars` |
| `tool_call` | agent invokes a tool | `tool_name`, `tool_input_preview` |
| `tool_result` | tool returns | `tool_name`, `tool_output_preview`, `error_class` (if failed) |
| `summary` | model-emitted condensation | `text_preview` |
| `system` | session-level metadata events | `text_preview` |

A full session schema spec lives in [docs/schema/v1.md](schema/v1.md).

## Walkthrough: adding Aider

Aider writes a chat history file (`.aider.chat.history.md`) and an input history file per project. The chat history is a markdown log of user/assistant turns and shell commands. We'll write a minimal normalizer.

### 1. Stub the file

Create `~/.config/tessera/normalizers/aider.py`:

```python
"""Aider normalizer for tessera."""
from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path

from tessera.normalizers import register


AIDER_HEADER = re.compile(r"^# (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def normalize_aider(raw_root: Path, writer) -> None:
    """Walk every .aider.chat.history.md under ~/.aider and per project dirs."""
    history_files = list(raw_root.rglob(".aider.chat.history.md"))
    history_files.extend(Path.cwd().rglob(".aider.chat.history.md"))
    for hist in dict.fromkeys(history_files):  # dedupe, preserve order
        _normalize_one_history(hist, writer)


def _normalize_one_history(hist: Path, writer) -> None:
    """Each markdown history file is one project's chat log over time."""
    try:
        text = hist.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    project_path = hist.parent
    project_label = project_path.name
    # Aider uses '> ' for user turns and '#### ' for assistant intent.
    # This is a deliberately tiny parser — see the codex/claude normalizers
    # for richer examples (event grouping, tool call extraction, etc.).
    session_id = f"aider:{hist.stat().st_mtime_ns:x}"
    event_index = 0
    session_started_at = None
    session_ended_at = None
    counts = {"message": 0, "reasoning": 0, "tool_call": 0, "tool_result": 0}

    for line in text.splitlines():
        header = AIDER_HEADER.match(line)
        if header:
            ts = header.group("ts")
            try:
                dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            session_started_at = session_started_at or dt.isoformat()
            session_ended_at = dt.isoformat()
            continue
        if line.startswith("> "):
            writer.write_event({
                "session_id": session_id, "agent": "aider",
                "event_index": event_index, "event_type": "message",
                "ts": session_ended_at or "1970-01-01T00:00:00+00:00",
                "role": "user", "text_preview": line[2:][:2000],
                "text_chars": len(line) - 2,
            })
            counts["message"] += 1
            event_index += 1
        elif line.startswith("#### "):
            writer.write_event({
                "session_id": session_id, "agent": "aider",
                "event_index": event_index, "event_type": "message",
                "ts": session_ended_at or "1970-01-01T00:00:00+00:00",
                "role": "assistant", "text_preview": line[5:][:2000],
                "text_chars": len(line) - 5,
            })
            counts["message"] += 1
            event_index += 1

    if session_started_at:
        writer.write_session({
            "session_id": session_id, "agent": "aider",
            "project_path": str(project_path), "project_label": project_label,
            "started_at": session_started_at, "ended_at": session_ended_at,
            "event_count": sum(counts.values()),
            "message_count": counts["message"], "reasoning_count": 0,
            "tool_call_count": 0, "tool_result_count": 0, "error_count": 0,
        })


register(
    name="aider",
    default_source=Path.home() / ".aider",
    normalize=normalize_aider,
    description="Aider AI pair-programming CLI",
)
```

### 2. Verify

```bash
$ tessera doctor
...
Registered agent normalizers (4):
  ✓ claude     ~/.claude/projects     ~3088 raw trace files
  ✓ codex      ~/.codex/sessions      ~427 raw trace files
  ✓ gemini     ~/.gemini/tmp          ~901 raw trace files
  ✓ aider      [user-dir]   ~/.aider  ~12 raw trace files
      Aider AI pair-programming CLI
```

If the import fails, doctor surfaces the error.

### 3. Run

```bash
tessera run --lookback-days 30 --min-events 10
```

Your Aider sessions now appear in narratives, synthesis, dashboard — same treatment as the built-ins.

## How the writer works

The `writer` argument is an `EventWriter` instance from `tessera._normalize_script`. The methods you'll use:

```python
writer.write_event(event: dict) -> None
writer.write_session(session: dict) -> None
```

Both expect dicts matching the schema in `docs/schema/v1.md`. Required fields:

**Event** (minimum viable):
- `session_id`: stable string identifier
- `agent`: short string ("aider", "cline", etc.) — same string for every event in this session
- `event_index`: integer, increments within a session starting at 0
- `event_type`: one of message / reasoning / tool_call / tool_result / summary / system
- `ts`: ISO 8601 timestamp string

**Session** (minimum viable):
- `session_id`: same as on the events
- `agent`: same as on the events
- `project_path`: filesystem path the session ran in (used for outcome enrichment via git)
- `started_at`, `ended_at`: ISO 8601
- `event_count`: total events in this session

Optional but useful: `project_label` (short name), `models_used` (list of strings), `git_branch_last` (string).

The deterministic stats layer (file-touch counts, tool distributions, burst detection) derives everything else from the events themselves.

## Pluggable discovery — three loading mechanisms

1. **Built-in** — registered automatically (`claude`, `codex`, `gemini`).
2. **User directory** — any `.py` file in `~/.config/tessera/normalizers/` is imported at startup. Override via `TESSERA_NORMALIZERS_DIR` env var.
3. **Entry-point plugin** — declare in your package's `pyproject.toml`:

   ```toml
   [project.entry-points."tessera.normalizers"]
   aider = "tessera_aider.normalizer:register_aider"
   ```

   Then `pip install tessera-aider` ships your normalizer to anyone.

All three sources merge into one registry. `tessera doctor` shows which is which.

## Common gotchas

- **Don't share `session_id` across runs.** A stable identifier (UUID, file hash, mtime_ns) is fine; just don't reuse the same one for different sessions.
- **`ts` must be ISO 8601 with timezone.** `datetime.now(timezone.utc).isoformat()` works. Naïve timestamps will break the time-of-day bucketing.
- **Don't `print()` from inside the normalizer.** Anything you print becomes stderr noise during `tessera run`. Raise an exception if you hit a fatal issue — the pipeline catches per-normalizer exceptions and continues with the rest.
- **Be defensive with file reads.** Trace files can be partial, corrupted, or unicode-mixed. `errors="replace"` is your friend.
- **Test with `tessera narrate --dry-run`.** That skips the LLM call and only runs the deterministic extraction — fast, free, surfaces parser bugs quickly.

## Want to contribute it back?

If you write a normalizer that works well, PR it into `src/tessera/_normalize_script.py` as a `normalize_<agent>()` function and tessera's contributors will adopt it as a built-in. See [CONTRIBUTING.md](../CONTRIBUTING.md).
