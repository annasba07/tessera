# Architecture

This is a module-level walkthrough of how `tessera run` produces its output. For the per-session schema spec see [schema/v1.md](schema/v1.md). For the in-session coach see the [tessera-live README](../../tessera-live/).

## End-to-end flow

```
                    ┌──────────────────────────────────────────────────┐
USER               │  tessera run --lookback-days 30            │
                    └────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
   [1] NORMALIZE          ──────────────────────────────────────
                          src/tessera/pipeline.py
                          src/tessera/_normalize_script.py

                          • Build a temp dir of symlinks into
                            ~/.claude/projects/, ~/.codex/sessions/,
                            ~/.gemini/tmp/ (no copies of trace data)
                          • Walk each agent's on-disk format and emit
                            chronological events into a single
                            events.jsonl in a standard schema
                          • Output: events.jsonl + sessions.jsonl

                                         │
                                         ▼
   [2] FILTER             ──────────────────────────────────────
                          src/tessera/cli.py:_qualifying_session_ids

                          • Walk sessions.jsonl
                          • Apply: lookback-days, min-events, slash-cmd
                            exclusion, dedupe-by-(agent,session_id)-keep-largest
                          • Return list of session_ids + filter breakdown
                            (in_window, dropped_short, dropped_slash, ...)
                          • Filter context is later stamped onto
                            synthesis.meta for the dashboard masthead

                                         │
                                         ▼
   [3] PER-SESSION        ──────────────────────────────────────
       NARRATIVES          src/tessera/narratives/

                          For each qualifying session, run pipeline.py:

                          ┌── deterministic.py ──────────────────┐
                          │ Extract ~30 facts from the events:   │
                          │ identity, time, volume, tool/scope,  │
                          │ verification proxies, conversation   │
                          │ shape, behavioral signals, tail/     │
                          │ outcome proxies. Pure function.      │
                          └──────────────────────────────────────┘
                                         │
                          ┌── compressor.py ─────────────────────┐
                          │ Compress full event stream to        │
                          │ ~30-50K tokens of one-line-per-event │
                          │ narrative format the LLM can parse.  │
                          │ Drops noisy success runs.            │
                          └──────────────────────────────────────┘
                                         │
                          ┌── cache.py (check) ──────────────────┐
                          │ Key = sha256(events) +               │
                          │       schema_version + model         │
                          │ If hit, skip LLM call entirely       │
                          └──────────────────────────────────────┘
                                         │ miss
                          ┌── extractor.py ──────────────────────┐
                          │ One Sonnet 4.6 call via              │
                          │ claude-agent-sdk (uses local CLI     │
                          │ auth, no API key). Returns the v1    │
                          │ narrative JSON: goal, tasks,         │
                          │ friction_moments, key_decisions,     │
                          │ dead_ends, recurring_env_issues,     │
                          │ counterfactual, lessons.             │
                          └──────────────────────────────────────┘
                                         │
                          ┌── validator.py ──────────────────────┐
                          │ Drop fields that fail spec rules:    │
                          │ event_range bounds, key_quote        │
                          │ substring match, controlled-vocab    │
                          │ membership, list-length caps. Sets   │
                          │ narrative_quality (high/medium/low)  │
                          │ based on drop rate.                  │
                          └──────────────────────────────────────┘
                                         │
                          ┌── cache.py (write) ──────────────────┐
                          │ Persist to ~/.cache/tessera/   │
                          │ narratives/<session_id>.json          │
                          └──────────────────────────────────────┘

                          Concurrency: default 10 sessions in flight.

                                         │
                                         ▼
   [4] CROSS-SESSION      ──────────────────────────────────────
       SYNTHESIS           src/tessera/narratives/synthesis.py

                          • Compact each narrative to its high-signal
                            fields (~2KB per session)
                          • Assign each session a 4-char ref token
                            (S001, S002, …)
                          • Optional: load prior runs' synthesis +
                            ratings via history.py for prior context
                          • One Sonnet 4.6 call producing
                            observations + quick_wins + per_project
                          • Validator translates refs → real
                            session_ids; drops fabricated refs
                          • Output: synthesis dict (in memory)

                                         │
                                         ▼
   [5] PERSIST + RENDER   ──────────────────────────────────────
                          src/tessera/cli.py:_run_command
                          src/tessera/narratives/dashboard.py
                          src/tessera/narratives/render.py
                          src/tessera/history.py

                          • Compute timeline (new / continuing /
                            improving / stable / worsening / resolved)
                            from this run vs prior history
                          • Save to history (gets a slug)
                          • Stamp slug + timeline + filter_context
                            into synthesis.meta
                          • Write synthesis.json (raw)
                          • Write synthesis.md (shareable)
                          • Write synthesis.html (dashboard) — embeds
                            all narratives compacted, plus the
                            synthesis, into one self-contained file
                          • Print the text-rendered synthesis to stdout
```

## Modules at a glance

### Boundary modules

| Module | Job |
|---|---|
| `cli.py` | All entry points (`run`, `narrate`, `synthesize`, `rate`, `rate-import`, `eval`, `dashboard`). Argument parsing, orchestration, output writing. |
| `_normalize_script.py` | Cross-agent parsers (`normalize_claude`, `normalize_codex`, `normalize_gemini`). Emits the common event schema. **The only place that knows about each agent's on-disk format.** |
| `pipeline.py` | Builds a symlink tree of the user's trace dirs and invokes the normalizer. |

### Per-session pipeline (`narratives/`)

| Module | Job |
|---|---|
| `deterministic.py` | Computes the deterministic block of the v1 schema (timing, file scope, tool counts, behavioral signals). Pure function. |
| `compressor.py` | Turns the event stream into a compact one-line-per-event format the LLM consumes. |
| `extractor.py` | Builds the per-session prompt and calls the LLM. Returns parsed JSON, no validation. |
| `validator.py` | Enforces the v1 schema. Drops invalid fields (not whole sessions). |
| `cache.py` | Content-hash cache for narratives; bumping `schema_version` invalidates everything. |
| `pipeline.py` | Orchestrates the per-session flow above. Concurrency control. |

### Cross-session pipeline

| Module | Job |
|---|---|
| `synthesis.py` | Compacts narratives, builds the cross-session prompt with ref tokens, validates citations. |
| `history.py` | Stores runs + ratings; provides prior-run context for the next synthesis prompt. |

### Output

| Module | Job |
|---|---|
| `render.py` | Text + markdown renderers for the synthesis output. |
| `dashboard.py` | The single-file interactive HTML dashboard. Findings + Explore views, inline rating, week-over-week panel, cost chips. |
| `eval.py` | Quality metrics for a narratives directory and/or synthesis output. |

### In-session coach (sibling)

| Module | Job |
|---|---|
| `coach/rules.py` | Six mechanical waste-pattern rules over a rolling event window. |
| `coach/state.py` | Per-session rolling state (events, suppression cooldowns, fired log). |
| `coach/rating_lookup.py` | Reads rated synthesis observations from `history.py` to suppress / enrich nudges per `(rule, project)`. |
| `coach/experiments.py` | Writes pending-experiment files when a rule fires for a new (rule, project) pair. |
| `coach/hook.py` | Claude Code PostToolUse hook entry point. Silent on the happy path; nudges Claude when a rule fires. |

## Key design decisions

**Two-stage LLM pipeline (per-session → synthesis)**: pure histograms can't surface narrative patterns. Per-session narratives turn each session into a compact story; synthesis finds patterns across stories. This is the change between v0.1 (digest-based, surfaced 0 observations on real data) and v0.2+ (narrative-based, dozens of evidence-grounded observations).

**Ref-token citations (S001, S002, …)**: long UUIDs in evidence lists triggered ~50% LLM fabrication. Short ref tokens dropped fabrication to 0% in testing. The validator translates refs back to real session_ids deterministically.

**Validator fails closed at field level, not session level**: an LLM that produces 90% good output on a session shouldn't lose all of that output because of one bad citation. Each invalid field is dropped independently; the session's `narrative_quality` reflects the drop rate.

**Content-hash cache**: re-running on unchanged sessions is free. Bumping `schema_version` invalidates everything globally. Cache lives at `~/.cache/tessera/narratives/`.

**Symlink-based normalize**: never copies user trace data; always reads the originals via symlink into a temp dir that's cleaned up at the end.

**Local-only**: no network calls outside the LLM SDK (which uses your local Claude CLI auth). No telemetry. No data leaves the machine.

## Versioning

| Version | Significance |
|---|---|
| Schema version | Bumping invalidates all narrative caches globally. Reserved for breaking changes to the per-session schema. |
| Package version | `pyproject.toml`. Follows semver loosely. |
| Plugin version | `.claude-plugin/plugin.json`. Tracks user-visible changes to the slash command and the coach. |
