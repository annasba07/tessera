# tessera-live

A Claude Code plugin that watches your sessions for known waste patterns and nudges **Claude** (not you) when one appears. The nudge is a short system note Claude can mention to you, silently course-correct around, or ignore if the context genuinely differs.

Silent on the happy path. Fires at most 1–2 times in an average session.

Sibling tool to [`tessera`](../tessera/). The two close a feedback loop: the dashboard's weekly ratings tune which coach nudges fire (and how loudly) in real time during your next session.

## What it watches for

Six fast rules, each derived from real patterns observed in the upstream `tessera` synthesis:

| Rule | Fires when |
|---|---|
| `browser_spiral` | ≥3 browser tool errors in the last 10 calls |
| `retry_without_change` | Same tool + same input has failed ≥3 times in the last 20 events |
| `permission_wall_repeat` | Same tool has been blocked by permission ≥2 times |
| `runaway` | 30 tool calls in under 60 seconds (tight retry loop) |
| `edit_without_verify` | ≥5 Edit/Write ops with no shell call between them |
| `delegation_sprawl` | ≥4 subagents spawned in one session |

Every rule has a cooldown so it doesn't re-fire constantly. First time a rule fires for a given `(rule, project)` pair, the coach also writes a **pending experiment** under `~/.config/tessera/experiments/` that the next weekly `tessera run` will pick up.

## Install

Requires [`tessera`](../tessera/) installed first (provides the `tessera-live-hook` CLI the plugin shells out to):

```bash
pip install tessera
# or
uv tool install tessera
```

Then install this plugin in Claude Code:

```
/plugin add <path-to-this-directory>
```

The plugin declares a PostToolUse hook that invokes `tessera-live-hook` on every tool call. The hook is constrained to <5 seconds and fails silently, so a bug in the coach can never break your session.

## Disable

Three ways, pick whichever is easiest for the situation:

```bash
# This session only, via env var
TESSERA_LIVE=off claude

# For this directory only, via .env or shell profile
export TESSERA_LIVE=off

# Permanently, uninstall the plugin
/plugin remove tessera-live
```

## What the nudge looks like

Claude sees a system-context note like this when a rule fires:

```
[tessera-live · rule:browser_spiral] Coach note: 3 browser-tool errors in
the last 10 calls (browser_snapshot, browser_evaluate). This looks like a
browser spiral — a pattern that historically burns the session without
reaching a verified state. Consider: stop the current approach, fall back
to Read + a direct API/curl check to confirm state, and only return to
browser tools after diagnosing why the current selectors are failing.

Prior signal: you rated this pattern USEFUL on 2026-04-19 — key [b0841a663c].
Suggested next: pkill -f chrome-headless-shell && rm -f ~/.config/google-chrome/SingletonLock
```

The first half is the rule; the second half (after the blank line) is the **rating-driven enrichment** — pulled from your most recent `tessera` synthesis if you've rated a similar pattern useful. That second half includes the concrete `next_action` you previously confirmed worked, so Claude isn't just nudged — it has your validated playbook.

In practice:

- On clear cases that match a prior `useful` rating, Claude usually mentions the pattern and applies the suggested fix.
- On ambiguous cases, Claude may quietly adjust without interrupting.
- If the context really doesn't match (e.g. browser errors are expected because you're debugging the browser layer itself), Claude ignores the nudge.

## How the rules learn over time

The weekly ritual is to open the `tessera` dashboard (`synthesis.html`) and rate observations inline with the `[u]/[w]/[k]/[s]` buttons. Click `SAVE`, paste the resulting `tessera rate-import` command in your terminal — done. The coach reads those ratings on every hook fire:

- Observation rated `wrong` (in the last 3 runs) → coach **suppresses** the rule for that project for the rest of the session.
- Observation rated `useful` → coach **enriches** the nudge with the observation's `next_action` and short-key as evidence.
- Observation rated `known` → coach still nudges but with softer language.

So each rating round tunes the coach's signal-to-noise ratio. A pattern you've explicitly marked as a false positive stops nagging you; a pattern you confirmed as real comes with the fix you already validated.

## Scope

- Pure-local rule evaluation per hook call. No LLM call in the happy path, no network.
- Uses Claude Code PostToolUse hooks only. Doesn't intercept or modify tool behavior.
- Session state lives at `~/.cache/tessera-live/sessions/<id>.json` (pruned after 7 days).
- Pending experiments at `~/.config/tessera/experiments/*.json` (readable / editable).
- Rating history at `~/.config/tessera/history/` (shared with `tessera`).

## License

MIT.
