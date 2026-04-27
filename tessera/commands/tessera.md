---
description: Reflect on your last 30 days of AI coding agent sessions — produces a dashboard you can open in any browser
disable-model-invocation: false
---

You are running the `/tessera` command. Your job is to invoke the `tessera` CLI on the user's behalf and surface the resulting dashboard.

## Steps

1. **Check the CLI is installed.** Run:

   ```
   tessera --version
   ```

   If that fails with "command not found" or similar, instruct the user to install it first:

   ```
   pip install tessera
   # or
   uv tool install tessera
   ```

   Stop until they confirm it's installed.

2. **Run the reflection.** Default lookback is 30 days; if `$ARGUMENTS` contains a number, use it as `--lookback-days`. If `$ARGUMENTS` contains `all-time` or `all`, pass `--all-time` instead.

   ```
   tessera run --format text --lookback-days $ARGUMENTS
   ```

   Expect stderr progress lines `[1/4]`–`[4/4]` covering normalize, session selection, per-session narrative extraction (cached after the first run), and the final cross-session synthesis. The first run on a user's traces takes ~30–50 minutes wall clock; subsequent runs in the same window mostly hit cache and finish in 5–10 minutes (only the synthesis is uncached).

3. **Surface the dashboard path.** The CLI writes three files in the current directory by default:
   - `synthesis.html` — interactive dashboard with two views (Findings + Explore)
   - `synthesis.md` — shareable markdown
   - `synthesis.json` — raw output
   - plus per-session narratives in `./narratives/`

   Tell the user: *"Your dashboard is ready: open `./synthesis.html` in any browser. Click `Findings` for the headline + observations, `Explore` to browse all sessions, env issues, and lessons. Rate observations inline with the `[u]/[w]/[k]/[s]` buttons — when you're done, click the floating `SAVE` button and paste the command into your terminal."*

4. **Brief text summary.** From the stdout (which is the text-rendered synthesis), extract and present:
   - The headline
   - The "if you do one thing this week"
   - A bulleted list of observation titles with confidence and supporting count
   - The quick-wins list

   Keep it short — the dashboard has the full content. Do **not** re-interpret or summarize the observations themselves; they're already grounded in specific sessions.

5. **Mention the in-session coach** *only if the user hasn't installed it yet*. If you don't know, mention briefly: *"For real-time nudges during sessions, install the `tessera-live` plugin — it watches for known waste patterns and informs Claude mid-session, using your dashboard ratings to tune signal."*

## Error handling

- **Exit code 2** (`--min-sessions` not met): the user doesn't have enough recent sessions in the lookback window. Suggest running with `/tessera 60` or `/tessera all-time`.
- **Exit code 3** (model returned non-JSON): rare and usually transient. Suggest a retry.
- **`FileNotFoundError: No traces found`**: the user has neither `~/.claude/projects/`, `~/.codex/sessions/`, nor `~/.gemini/tmp/` on this machine. Tell them so.
- **Cost concern**: a fresh full historical extraction on a heavy user can be ~$15-30 in token usage routed through the user's `claude` CLI auth. The cache makes weekly re-runs ~$1-3.
