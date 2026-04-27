# Contributing to tessera

Thanks for your interest. The most useful contributions are listed at the bottom — start there if you're not sure.

## Dev setup

```bash
git clone https://github.com/annasba07/tessera
cd tessera
uv venv && source .venv/bin/activate   # or python -m venv
uv pip install -e ".[dev]"             # installs with pytest
```

Verify install:

```bash
tessera --version
pytest -q
```

You also need an authenticated `claude` CLI (any v2+) on PATH for the LLM-touching code paths to work. The unit tests don't hit the LLM — they use fixtures.

## Testing

```bash
pytest                                 # full test suite
pytest tests/test_validator.py -v      # one module
pytest -k "not llm"                    # skip anything that calls Claude
```

Tests live in `tests/`. Pure-function modules (deterministic.py, validator.py, dashboard rendering, cache round-trip) are easy to add coverage for. The LLM extractor and synthesis are harder — their tests use recorded fixtures rather than live calls.

## Project layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module-level flow. The high-level shape:

```
src/tessera/
├── _normalize_script.py       # turn agent-specific traces into a common event schema
├── pipeline.py                # symlink + invoke normalizer
├── history.py                 # ratings store
├── narratives/
│   ├── deterministic.py       # extract metadata that never needs an LLM
│   ├── compressor.py          # event stream → compact text for the LLM
│   ├── extractor.py           # per-session LLM call → narrative JSON
│   ├── validator.py           # post-process: drop bad fields, enforce vocab
│   ├── cache.py               # content-hashed narrative cache
│   ├── pipeline.py            # orchestrate per-session: load → meta → compress → extract → validate → cache
│   ├── synthesis.py           # cross-session LLM call + ref-citation validator
│   ├── render.py              # text + markdown renderers for synthesis
│   ├── dashboard.py           # interactive HTML dashboard
│   └── eval.py                # quality metrics
├── coach/                     # in-session hook (deployed via tessera-live plugin)
└── cli.py                     # all `tessera <command>` entry points
```

## Conventions

- Python 3.11+ idioms (use `str | None`, dict / list builtins as types, dataclasses).
- No third-party deps unless absolutely necessary. Current deps: `claude-agent-sdk`. Test-only deps go in `[project.optional-dependencies].dev`.
- Public functions get a one-line docstring. Avoid multi-paragraph docstrings.
- Comments are only for **why**, not **what**. The code says what.
- Prefer dropping invalid LLM-output fields to retrying. Validators fail closed at field level, not at session level.
- Hardcoded constants (vocabularies, thresholds, prompts) live near where they're used, not in a central config.

## Adding a new agent CLI

This is the highest-leverage contribution. The pipeline is agent-agnostic from the normalized event schema onward, so adding a new CLI is one parser.

1. Pick the agent (e.g. `aider`, `cursor`, `cline`).
2. Find where it stores session traces on disk.
3. Read its on-disk format. Each "session" should map to a list of chronologically-ordered events.
4. Add a `normalize_<agent>(symlink_root, writer)` function in `src/tessera/_normalize_script.py` that:
   - Iterates the agent's session files
   - Emits one normalized event per atomic happening (user message, assistant message, tool call, tool result, reasoning)
   - Each event must include: `agent` (the new name), `session_id`, `trace_kind` (`top_level` or `subagent`), `timestamp` (ISO-8601), `event_type` (`message` / `tool_call` / `tool_result` / `reasoning`), `cwd` (for project labeling)
   - For `tool_call`: `tool_name`, `tool_input_preview`, `input_length`
   - For `tool_result`: `tool_call_id`, `tool_status` (`success` / `error` / `unknown`), `error_class` (if error), `tool_output_preview`
   - For `message` / `reasoning`: `role`, `message_text`, `text_length`
5. Add a CLI flag `--<agent>-sessions DIR` that overrides the default discovery path.
6. Wire it in `pipeline.normalize_live_traces`.
7. Add a fixture-backed test under `tests/normalize/test_<agent>.py` with a small synthetic session.

Once your normalizer emits the standard schema, the entire downstream pipeline (deterministic metadata, narrative extraction, synthesis, dashboard) works on it without any further changes.

## Most-useful contributions

| Contribution | Why |
|---|---|
| Parser for a new agent CLI (aider, cursor, cline, …) | Ships value to a new audience for one parser function. |
| Better friction detectors for the coach | `coach/rules.py` has 6 mechanical rules. Adding empirical detectors derived from synthesis output is the v2 plan — see `docs/schema/v1.md`. |
| Test coverage on `narratives/validator.py` | Most schema enforcement happens here. Edge cases welcome. |
| Dashboard polish (filtering UX, week-over-week visual, deep linking) | The dashboard surface is where users actually live. |
| Privacy review of what ends up in synthesis output | We strip nothing from the LLM input today. Fixtures suggesting redactions are welcome. |
| README / examples for a specific agent | "Here's what the dashboard looks like for a Codex-heavy user" is more actionable than abstract docs. |

## Filing issues

When reporting a bug, please include:

- `tessera --version`
- The command you ran
- Relevant stderr (the `[1/4]`–`[4/4]` lines + any error)
- If it's a synthesis-quality issue, the run slug from `~/.config/tessera/history/` (you can paste the synthesis.json — it's local-only data unless you choose to share it)

## License

By contributing, you agree your contributions are licensed under MIT.
