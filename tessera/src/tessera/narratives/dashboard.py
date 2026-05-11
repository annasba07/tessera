"""Self-contained HTML dashboard for a synthesis run.

Produces a single ``synthesis.html`` that opens in any browser with no
server. Embeds the synthesis output plus a compact view of each
per-session narrative so users can drill from an evidence ref into the
session it came from.

Aesthetic: paper-colored, serif headers, generous whitespace, no dark
mode, no generic-AI gradients. Designed to feel like a printed memo more
than a SaaS dashboard.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def _compact_narrative_for_drilldown(n: dict) -> dict:
    """Reduce a per-session narrative to the fields we actually surface
    in the drilldown panel. Keeps the embedded JSON small."""
    tasks = []
    for t in n.get("tasks") or []:
        tasks.append(
            {
                "id": t.get("task_id"),
                "intent": (t.get("intent") or "")[:240],
                "type": t.get("task_type"),
                "outcome": t.get("outcome"),
                "difficulty": (t.get("task_difficulty") or {}).get("overall"),
            }
        )
    friction = []
    for fm in n.get("friction_moments") or []:
        friction.append(
            {
                "type": fm.get("type"),
                "tool_cat": fm.get("tool_category"),
                "cost_events": fm.get("cost_events"),
                "desc": (fm.get("description") or "")[:240],
                "quote": (fm.get("key_quote") or "")[:200],
            }
        )
    env = []
    for ei in n.get("recurring_environmental_issues") or []:
        env.append(
            {
                "desc": (ei.get("description") or "")[:240],
                "occurrences": len(ei.get("occurrences") or []),
            }
        )
    return {
        "session_id": n.get("session_id"),
        "agent": n.get("agent"),
        "project": n.get("project_label"),
        "date": (n.get("started_at") or "")[:10],
        "primary_model": n.get("primary_model"),
        "events": n.get("event_count"),
        "tool_calls": n.get("tool_call_count"),
        "active_min": n.get("active_minutes"),
        "bursts": n.get("bursts"),
        "tool_err_rate": n.get("tool_error_rate"),
        "user_caught": (n.get("user_caught_model_errors") or {}).get("count", 0),
        "verification": n.get("verification_completeness"),
        "goal": (n.get("goal") or "")[:300],
        "waste_signature": n.get("waste_signature"),
        "tasks": tasks,
        "friction": friction,
        "env_issues": env,
        "counterfactual": (n.get("counterfactual") or "")[:400],
        "notable": (n.get("notable") or "")[:300],
        "lesson_user": (n.get("lesson_for_user") or "")[:240],
        "topics": (n.get("topics") or [])[:6],
    }


# ---------- HTML template ---------------------------------------------------

CSS = r"""
:root {
  --paper: #faf9f4;
  --ink: #1c1a17;
  --ink-soft: #4a463f;
  --ink-mute: #7c7568;
  --line: #e7e3d9;
  --line-strong: #d6d0c2;
  --warm: #7d3a00;
  --warm-soft: #b8794a;
  --leaf: #4a6a3a;
  --rust: #8a3a1a;
  --slate: #3a4a6a;
  --hint: #f3efdf;
  --serif: "Source Serif 4", "Source Serif Pro", Georgia, serif;
  --sans: "Inter", -apple-system, "Helvetica Neue", sans-serif;
  --mono: "IBM Plex Mono", "Menlo", monospace;
}

* { box-sizing: border-box; }

html { -webkit-font-smoothing: antialiased; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.55;
}

.frame {
  max-width: 980px;
  margin: 0 auto;
  padding: 56px 40px 96px;
}

header.masthead {
  border-bottom: 1px solid var(--line-strong);
  padding-bottom: 18px;
  margin-bottom: 36px;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}

.masthead .brand {
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-soft);
}

.masthead .meta {
  font-family: var(--sans);
  font-size: 12px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
  text-align: right;
  line-height: 1.5;
}
.masthead .meta[hidden] { display: none; }
.masthead .meta .filter-context {
  font-size: 11px;
  color: var(--warm-soft);
  letter-spacing: 0.06em;
}

h1.headline {
  font-family: var(--serif);
  font-weight: 600;
  font-size: 32px;
  line-height: 1.2;
  letter-spacing: -0.005em;
  margin: 0 0 24px;
  color: var(--ink);
}

.callout {
  border-left: 3px solid var(--warm);
  background: linear-gradient(180deg, var(--hint), transparent 60%);
  padding: 18px 24px 18px 22px;
  margin: 0 0 56px;
  border-radius: 2px;
}
.callout .label {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--warm);
  margin-bottom: 8px;
}
.callout .body {
  font-family: var(--serif);
  font-size: 18px;
  line-height: 1.5;
  color: var(--ink);
}

.aggregate-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: 14px 0;
  margin: 0 0 56px;
}
.stat {
  padding: 0 18px;
  border-right: 1px solid var(--line);
}
.stat:last-child { border-right: 0; }
.stat .num {
  font-family: var(--serif);
  font-weight: 600;
  font-size: 22px;
  color: var(--ink);
  line-height: 1.1;
}
.stat .lab {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 4px;
}

section.block { margin-bottom: 56px; }

h2.section-title {
  font-family: var(--sans);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--ink-mute);
  margin: 0 0 22px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--line);
}
h2.section-title .count {
  color: var(--ink);
  margin-left: 6px;
}

article.observation {
  margin: 0 0 36px;
  padding-bottom: 36px;
  border-bottom: 1px solid var(--line);
}
article.observation:last-of-type { border-bottom: 0; }

.obs-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 6px;
  flex-wrap: wrap;
}
.obs-num {
  font-family: var(--serif);
  font-size: 20px;
  color: var(--ink-mute);
  font-weight: 400;
}
.obs-title {
  font-family: var(--serif);
  font-weight: 600;
  font-size: 21px;
  color: var(--ink);
  flex: 1 1 auto;
  line-height: 1.3;
}

.chips {
  display: inline-flex;
  gap: 6px;
  flex-wrap: wrap;
  font-family: var(--sans);
}
.chip {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid var(--line-strong);
  color: var(--ink-soft);
  background: transparent;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
.chip.conf-high { color: #fff; background: var(--warm); border-color: var(--warm); }
.chip.conf-medium { color: var(--warm); border-color: var(--warm-soft); }
.chip.conf-low { color: var(--ink-mute); }

.chip.cat-environmental { background: #f1ede0; border-color: #e2dcc6; color: var(--leaf); }
.chip.cat-tooling { background: #ebf0e9; border-color: #d9e1d6; color: var(--leaf); }
.chip.cat-workflow { background: #ecedf3; border-color: #d8dae5; color: var(--slate); }
.chip.cat-prompting { background: #f4ecea; border-color: #e6d4cf; color: var(--rust); }
.chip.cat-project_specific { background: #f1ecdc; border-color: #e3d9bf; color: var(--ink-soft); }
.chip.cat-cross_agent { background: #ecf2f3; border-color: #d6dee0; color: var(--slate); }
.chip.cat-behavioral { background: #efe7d8; border-color: #e0d3b4; color: var(--warm); font-style: italic; }
.chip.warn-noncomp { background: transparent; border-color: var(--rust); color: var(--rust); font-style: italic; }

/* Three-tier visual hierarchy by supporting_count.
 * Default (≥6 sessions): full strength.
 * thin-evidence (3-5):     subtle muting — present but lower-weight.
 * weak-evidence (<3):      strong muting — visible but clearly anecdotal. */
.observation.thin-evidence {
  opacity: 0.88;
}
.observation.thin-evidence .obs-title {
  font-size: 17px;
  color: var(--ink);
}
.observation.thin-evidence .obs-claim {
  color: var(--ink-soft);
}
.observation.weak-evidence {
  opacity: 0.65;
}
.observation.weak-evidence .obs-title {
  font-size: 15px;
  color: var(--ink-soft);
}
.observation.weak-evidence .obs-claim {
  font-size: 13px;
  color: var(--ink-soft);
}
.observation.weak-evidence .obs-num {
  color: var(--ink-mute);
  font-weight: 400;
}
/* Patterns that lack a comparative grounding get an extra subtle border */
.observation.behavioral.non-comparative {
  border-left-color: var(--ink-mute);
}

.section-sub { font-size: 13px; color: var(--ink-mute); font-weight: 400; font-style: italic; margin-left: 6px; }
.observation.behavioral { border-left: 3px solid var(--warm-soft); padding-left: 16px; margin-left: -19px; background: linear-gradient(to right, rgba(184,121,74,0.04), transparent 200px); }

.obs-claim {
  font-size: 17px;
  color: var(--ink);
  margin: 8px 0 18px;
}

.obs-section-label {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--ink-mute);
  margin: 16px 0 8px;
}

.evidence-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}
.ref-chip {
  font-family: var(--mono);
  font-size: 12px;
  padding: 4px 9px;
  border-radius: 3px;
  background: #fff;
  border: 1px solid var(--line-strong);
  color: var(--ink-soft);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
  user-select: none;
  display: inline-flex;
  gap: 6px;
  align-items: center;
}
.ref-chip:hover {
  background: var(--hint);
  border-color: var(--warm-soft);
  color: var(--ink);
}
.ref-chip .sid {
  color: var(--ink-mute);
  font-size: 11px;
}
.ref-chip.active {
  background: var(--warm);
  color: #fff;
  border-color: var(--warm);
}
.ref-chip.active .sid { color: rgba(255,255,255,0.8); }

.obs-prose {
  font-size: 16px;
  color: var(--ink);
  margin: 6px 0 12px;
}
.obs-prose .lead {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  color: var(--ink-mute);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-right: 8px;
}

.next-action {
  background: #fff;
  border: 1px solid var(--line-strong);
  border-left: 3px solid var(--warm);
  padding: 14px 16px;
  margin-top: 12px;
  font-size: 15px;
  position: relative;
}
.next-action .lead {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  color: var(--warm);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-bottom: 6px;
  display: block;
}
.next-action .copy {
  position: absolute;
  top: 8px;
  right: 8px;
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.1em;
  color: var(--ink-mute);
  border: 1px solid var(--line-strong);
  background: var(--paper);
  border-radius: 3px;
  padding: 3px 7px;
  cursor: pointer;
}
.next-action .copy:hover { color: var(--warm); border-color: var(--warm-soft); }
.next-action code {
  font-family: var(--mono);
  font-size: 13px;
  background: var(--hint);
  padding: 1px 4px;
  border-radius: 2px;
}

.drilldown {
  background: #fff;
  border: 1px solid var(--line-strong);
  border-radius: 4px;
  padding: 18px 22px;
  margin: 14px 0 4px;
  font-family: var(--serif);
  font-size: 15px;
  display: none;
}
.drilldown.open { display: block; }
.drilldown .dd-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 14px;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
}
.drilldown .dd-sid {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--ink-mute);
}
.drilldown .dd-meta {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
}
.drilldown .dd-goal {
  font-style: italic;
  color: var(--ink);
  margin: 0 0 14px;
}
.drilldown h4 {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-mute);
  margin: 14px 0 6px;
}
.drilldown .friction-item, .drilldown .env-item, .drilldown .task-item {
  font-size: 14px;
  margin: 4px 0;
  color: var(--ink);
}
.drilldown .friction-item .ftype {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--rust);
  margin-right: 6px;
}
.drilldown blockquote.quote {
  border-left: 2px solid var(--line-strong);
  padding: 4px 0 4px 12px;
  margin: 4px 0 8px;
  font-style: italic;
  color: var(--ink-soft);
  font-size: 14px;
}
.drilldown .stats {
  display: flex;
  gap: 18px;
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
  flex-wrap: wrap;
}

.quick-wins ol {
  list-style: none;
  padding: 0;
  margin: 0;
}
.quick-wins li {
  padding: 14px 0;
  border-bottom: 1px solid var(--line);
  display: flex;
  gap: 14px;
  align-items: flex-start;
}
.quick-wins li:last-child { border-bottom: 0; }
.quick-wins .qw-count {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  background: var(--hint);
  color: var(--warm);
  padding: 4px 9px;
  border-radius: 3px;
  white-space: nowrap;
  letter-spacing: 0.04em;
}
.quick-wins .qw-text {
  font-family: var(--serif);
  font-size: 16px;
  color: var(--ink);
  flex: 1;
}
.quick-wins .qw-text code {
  font-family: var(--mono);
  font-size: 13px;
  background: var(--hint);
  padding: 1px 4px;
  border-radius: 2px;
}
.quick-wins .qw-copy {
  font-family: var(--sans);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-mute);
  border: 1px solid var(--line-strong);
  background: transparent;
  border-radius: 3px;
  padding: 4px 8px;
  cursor: pointer;
}
.quick-wins .qw-copy:hover { color: var(--warm); border-color: var(--warm-soft); }

.per-project {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}
.proj-card {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 3px;
  padding: 16px 18px;
}
.proj-card .proj-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
  gap: 8px;
}
.proj-card .proj-name {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
  word-break: break-all;
}
.proj-card .proj-count {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.06em;
  white-space: nowrap;
}
.proj-card .proj-headline {
  font-size: 14px;
  color: var(--ink);
  margin: 4px 0 10px;
  line-height: 1.45;
}
.proj-card .proj-fric {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 4px;
}
.proj-card .proj-fric-body {
  font-size: 13px;
  color: var(--ink-soft);
  line-height: 1.45;
}

.notes {
  background: var(--hint);
  border-radius: 3px;
  padding: 16px 20px;
  font-size: 14px;
  color: var(--ink-soft);
  font-style: italic;
}
.notes .lead {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  color: var(--ink-mute);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  display: block;
  margin-bottom: 6px;
  font-style: normal;
}

footer.colophon {
  margin-top: 60px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}

/* Week-over-week timeline panel */
.timeline {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  align-items: center;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 12px 18px;
  margin-bottom: 32px;
  font-family: var(--sans);
  font-size: 12px;
  color: var(--ink-soft);
  letter-spacing: 0.04em;
}
.timeline .tl-label {
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 10px;
  color: var(--ink-mute);
  font-weight: 600;
  margin-right: 4px;
}
.timeline .tl-bullet {
  font-variant-numeric: tabular-nums;
}
.timeline .tl-bullet strong {
  color: var(--ink);
  font-size: 14px;
  margin-right: 4px;
}
.timeline .tl-new strong { color: var(--warm); }
.timeline .tl-worsening strong { color: var(--rust); }
.timeline .tl-improving strong { color: var(--leaf); }
.timeline .tl-resolved strong { color: var(--ink-mute); text-decoration: line-through; }
.timeline .tl-prior {
  margin-left: auto;
  color: var(--ink-mute);
  font-size: 11px;
}

/* Inline rating buttons on each observation */
.rate-row {
  display: flex;
  gap: 6px;
  margin-top: 14px;
  align-items: center;
  font-family: var(--sans);
}
.rate-row .rate-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--ink-mute);
  margin-right: 8px;
  font-weight: 600;
}
.rate-row button {
  font-family: var(--sans);
  font-size: 11px;
  letter-spacing: 0.04em;
  padding: 4px 10px;
  border: 1px solid var(--line-strong);
  background: var(--paper);
  color: var(--ink-soft);
  border-radius: 3px;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.rate-row button:hover { background: var(--hint); }
.rate-row button.active.useful  { background: var(--leaf); color: #fff; border-color: var(--leaf); }
.rate-row button.active.wrong   { background: var(--rust); color: #fff; border-color: var(--rust); }
.rate-row button.active.known   { background: var(--warm-soft); color: #fff; border-color: var(--warm-soft); }
.rate-row button.active.skip    { background: var(--ink-mute); color: #fff; border-color: var(--ink-mute); }

/* Floating "save ratings" button (only visible when ≥1 rating made) */
.rate-sync {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: var(--warm);
  color: #fff;
  border: 0;
  border-radius: 4px;
  padding: 12px 18px;
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.04em;
  cursor: pointer;
  box-shadow: 0 6px 20px rgba(0,0,0,0.12);
  z-index: 50;
}
.rate-sync[hidden] { display: none; }
.rate-sync .count { font-weight: 700; margin-right: 4px; }

/* Modal shown by clicking the floating save button */
.rate-modal {
  position: fixed;
  inset: 0;
  background: rgba(28, 26, 23, 0.5);
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.rate-modal[hidden] { display: none; }
.rate-modal-inner {
  background: var(--paper);
  border-radius: 5px;
  padding: 28px 32px;
  max-width: 720px;
  width: 100%;
  box-shadow: 0 24px 64px rgba(0,0,0,0.25);
}
.rate-modal h3 {
  font-family: var(--serif);
  font-size: 20px;
  margin: 0 0 8px;
  color: var(--ink);
}
.rate-modal p {
  font-family: var(--sans);
  font-size: 13px;
  color: var(--ink-soft);
  margin: 0 0 14px;
}
.rate-modal pre {
  background: #fff;
  border: 1px solid var(--line-strong);
  border-radius: 3px;
  padding: 14px 16px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--ink);
  overflow-x: auto;
  white-space: pre-wrap;
  margin: 0 0 14px;
}
.rate-modal .actions { display: flex; gap: 10px; justify-content: flex-end; }
.rate-modal .actions button {
  font-family: var(--sans);
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 8px 16px;
  border: 1px solid var(--line-strong);
  background: var(--paper);
  color: var(--ink-soft);
  border-radius: 3px;
  cursor: pointer;
}
.rate-modal .actions button.primary {
  background: var(--warm);
  color: #fff;
  border-color: var(--warm);
}

/* Cost chip on observations */
.chip.cost {
  background: var(--paper);
  border: 1px dashed var(--warm-soft);
  color: var(--warm);
  font-variant-numeric: tabular-nums;
}

/* Top-level view switcher (Findings / Explore) */
nav.primary-nav {
  display: flex;
  gap: 4px;
  align-items: center;
}
nav.primary-nav button {
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.06em;
  background: none;
  border: 1px solid transparent;
  color: var(--ink-mute);
  cursor: pointer;
  padding: 6px 14px;
  border-radius: 3px;
  transition: color 0.12s, border-color 0.12s, background 0.12s;
}
nav.primary-nav button:hover { color: var(--ink-soft); }
nav.primary-nav button.active {
  color: var(--ink);
  background: #fff;
  border-color: var(--line-strong);
  font-weight: 600;
}
nav.primary-nav button .count {
  font-weight: 400;
  color: var(--ink-mute);
  margin-left: 4px;
  font-size: 11px;
}
nav.primary-nav button.active .count { color: var(--warm); }

/* Top-level views — only one shown at a time */
section.view { display: none; }
section.view.active { display: block; }

.cross-page-link {
  font-family: var(--sans);
  font-size: 12px;
  color: var(--warm);
  text-decoration: none;
  border-bottom: 1px dotted var(--warm-soft);
  letter-spacing: 0.04em;
  margin-left: 8px;
}
.cross-page-link:hover { color: var(--ink); border-bottom-color: var(--ink); }

@media (max-width: 720px) {
  .frame { padding: 32px 20px 64px; }
  h1.headline { font-size: 24px; }
  .callout .body { font-size: 16px; }
}
"""

JS = r"""
(function() {
  const D = window.__SYNTHESIS_DATA__ || {};
  const narrativesById = (D.narratives || []).reduce(function(acc, n) {
    if (n && n.session_id) acc[n.session_id] = n;
    return acc;
  }, {});

  function fmtNum(n, suffix) {
    if (n === undefined || n === null) return '';
    return (Math.round(n * 10) / 10) + (suffix || '');
  }

  function buildDrilldownHTML(n) {
    if (!n) return '<p style="color:#7c7568">Session details not available.</p>';
    const tasks = (n.tasks || []).map(function(t) {
      return '<div class="task-item"><strong>' + esc(t.id || '') + '</strong> · ' +
        esc(t.intent || '') + ' <em style="color:#7c7568">(' + esc(t.type || '') +
        ' · ' + esc(t.difficulty || '') + ' · ' + esc(t.outcome || '') + ')</em></div>';
    }).join('');
    const friction = (n.friction || []).map(function(f) {
      var quote = f.quote ? '<blockquote class="quote">' + esc(f.quote) + '</blockquote>' : '';
      return '<div class="friction-item"><span class="ftype">' + esc(f.type || '?') + '</span>' +
        '<em style="color:#7c7568">(' + esc(f.tool_cat || '?') + ' · ' + (f.cost_events || 0) + 'ev)</em><br>' +
        esc(f.desc || '') + quote + '</div>';
    }).join('');
    const env = (n.env_issues || []).map(function(e) {
      return '<div class="env-item">• ' + esc(e.desc || '') +
        ' <em style="color:#7c7568">(' + (e.occurrences || 0) + ' occurrences)</em></div>';
    }).join('');
    const counterfactual = n.counterfactual ? '<h4>Counterfactual</h4><div>' + esc(n.counterfactual) + '</div>' : '';
    const lessonUser = n.lesson_user ? '<h4>Lesson for user</h4><div>' + esc(n.lesson_user) + '</div>' : '';
    const notable = n.notable ? '<h4>Notable</h4><div><em>' + esc(n.notable) + '</em></div>' : '';
    const goal = n.goal ? '<p class="dd-goal">' + esc(n.goal) + '</p>' : '';
    const topics = (n.topics || []).length
      ? '<div style="margin-top:6px;font-size:12px;color:#7c7568">Topics: ' +
        (n.topics || []).map(function(t){ return esc(t); }).join(' · ') + '</div>'
      : '';

    return [
      '<div class="dd-head">',
      '  <span class="dd-sid">' + esc(n.session_id || '') + '</span>',
      '  <span class="dd-meta">' + esc(n.agent || '') + ' · ' + esc(n.project || '') +
            ' · ' + esc(n.date || '') + ' · ' + esc(n.primary_model || '') + '</span>',
      '</div>',
      goal,
      '<div class="stats">',
      '  <span>events: ' + (n.events || 0) + '</span>',
      '  <span>tools: ' + (n.tool_calls || 0) + '</span>',
      '  <span>active: ' + fmtNum(n.active_min, ' min') + '</span>',
      '  <span>bursts: ' + (n.bursts || 0) + '</span>',
      '  <span>tool err rate: ' + fmtNum((n.tool_err_rate || 0) * 100, '%') + '</span>',
      '  <span>user-caught errors: ' + (n.user_caught || 0) + '</span>',
      '  <span>verified: ' + esc(n.verification || '?') + '</span>',
      '  <span>waste: ' + esc(n.waste_signature || 'none') + '</span>',
      '</div>',
      topics,
      tasks ? '<h4>Tasks</h4>' + tasks : '',
      friction ? '<h4>Friction moments</h4>' + friction : '',
      env ? '<h4>Recurring environmental issues</h4>' + env : '',
      counterfactual,
      lessonUser,
      notable
    ].join('\n');
  }

  function esc(s) {
    s = String(s == null ? '' : s);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // Wire ref chips
  document.querySelectorAll('[data-obs-block]').forEach(function(block) {
    const dd = block.querySelector('.drilldown');
    block.querySelectorAll('.ref-chip').forEach(function(chip) {
      chip.addEventListener('click', function() {
        const sid = chip.getAttribute('data-sid');
        const n = narrativesById[sid];
        const wasActive = chip.classList.contains('active');
        block.querySelectorAll('.ref-chip.active').forEach(function(c) { c.classList.remove('active'); });
        if (wasActive) {
          dd.classList.remove('open');
          dd.innerHTML = '';
          return;
        }
        chip.classList.add('active');
        dd.innerHTML = buildDrilldownHTML(n);
        dd.classList.add('open');
      });
    });
  });

  // Copy buttons
  document.querySelectorAll('[data-copy]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      const text = btn.getAttribute('data-copy');
      const orig = btn.textContent;
      navigator.clipboard && navigator.clipboard.writeText(text).then(function() {
        btn.textContent = 'COPIED';
        setTimeout(function() { btn.textContent = orig; }, 1200);
      });
    });
  });
})();
"""


# ---------- Server-side render helpers --------------------------------------


def _esc(s) -> str:
    return escape(str(s if s is not None else ""))


def _safe_inline_json(obj) -> str:
    """Serialize JSON for embedding inside an HTML <script> tag.

    Defends against:
    - </script> in any string value prematurely closing the tag
    - U+2028 / U+2029 line separators which JS treats as newlines and which
      will cause a SyntaxError inside an inline script
    """
    s = json.dumps(obj, ensure_ascii=False)
    s = s.replace("</", "<\\/")
    s = s.replace(chr(0x2028), "\\u2028").replace(chr(0x2029), "\\u2029")
    return s


def _short_session(sid: str, n: int = 14) -> str:
    if not sid:
        return "?"
    if ":" not in sid:
        return sid[:n] + "…" if len(sid) > n else sid
    agent, rest = sid.split(":", 1)
    return f"{agent}:{rest[:n]}…" if len(rest) > n else sid


def _render_observation(
    obs: dict,
    idx: int,
    *,
    cost_minutes: float | None = None,
    obs_key: str | None = None,
) -> str:
    title = _esc(obs.get("title") or "Observation")
    claim = _esc(obs.get("claim") or "")
    interpretation = _esc(obs.get("interpretation") or "")
    next_action = obs.get("next_action") or ""
    confidence = obs.get("confidence") or "medium"
    category = obs.get("category") or ""
    supporting = obs.get("supporting_count") or 0
    trend = obs.get("trend") or ""
    continues = obs.get("continues") or ""

    chips = []
    if confidence:
        chips.append(f'<span class="chip conf-{_esc(confidence)}">{_esc(confidence)}</span>')
    if category:
        chips.append(f'<span class="chip cat-{_esc(category)}">{_esc(category)}</span>')
    if supporting:
        chips.append(f'<span class="chip">{supporting} sessions</span>')
    if cost_minutes and cost_minutes > 0:
        if cost_minutes >= 60:
            cost_label = f"~{cost_minutes / 60:.1f}h friction"
        else:
            cost_label = f"~{int(round(cost_minutes))}m friction"
        chips.append(f'<span class="chip cost">{_esc(cost_label)}</span>')
    if trend and trend != "new":
        chips.append(f'<span class="chip">trend: {_esc(trend)}</span>')
    if continues:
        chips.append(f'<span class="chip">continues {_esc(continues)}</span>')
    chips_html = '<span class="chips">' + "".join(chips) + "</span>" if chips else ""

    refs = obs.get("evidence_refs") or []
    sids = obs.get("evidence_sessions") or []
    ref_chips = []
    for ref, sid in zip(refs, sids):
        ref_chips.append(
            f'<span class="ref-chip" data-sid="{_esc(sid)}">'
            f'<span>{_esc(ref)}</span><span class="sid">{_esc(_short_session(sid))}</span></span>'
        )
    refs_html = (
        f'<div class="obs-section-label">Evidence ({supporting} sessions — click to drill in)</div>'
        f'<div class="evidence-row">{"".join(ref_chips)}</div>'
        if ref_chips
        else ""
    )

    interp_html = (
        f'<div class="obs-prose"><span class="lead">Why</span>{interpretation}</div>'
        if interpretation
        else ""
    )

    next_html = ""
    if next_action:
        copy_attr = _esc(next_action.replace('"', "&quot;"))
        next_html = (
            f'<div class="next-action">'
            f'<span class="lead">Next</span>'
            f'<button class="copy" data-copy="{copy_attr}">COPY</button>'
            f"{_esc(next_action)}"
            f"</div>"
        )

    rate_row = ""
    if obs_key:
        rate_row = (
            f'<div class="rate-row" data-rate-row data-obs-index="{idx - 1}" '
            f'data-obs-key="{_esc(obs_key)}" data-obs-title="{_esc(obs.get("title") or "")}">'
            '<span class="rate-label">Was this useful?</span>'
            '<button data-rate="useful">[u]seful</button>'
            '<button data-rate="wrong">[w]rong</button>'
            '<button data-rate="known">[k]nown</button>'
            '<button data-rate="skip">[s]kip</button>'
            "</div>"
        )

    # Three tiers — the persona-review caught that strong (11-session) and
    # mid (4-session) items rendered identically.
    if supporting < 3:
        evidence_class = " weak-evidence"
    elif supporting < 6:
        evidence_class = " thin-evidence"
    else:
        evidence_class = ""
    return (
        f'<article class="observation{evidence_class}" data-obs-block>'
        f'<div class="obs-header">'
        f'<span class="obs-num">§{idx}</span>'
        f'<span class="obs-title">{title}</span>'
        f"{chips_html}"
        f"</div>"
        f'<p class="obs-claim">{claim}</p>'
        f"{refs_html}"
        f'<div class="drilldown"></div>'
        f"{interp_html}"
        f"{next_html}"
        f"{rate_row}"
        f"</article>"
    )


def _render_behavioral_pattern(
    bp: dict,
    idx: int,
    *,
    bp_key: str | None = None,
) -> str:
    """Render one behavioral_pattern entry — same structural shape as an
    observation but with the behavioral schema fields (pattern, dimension,
    experiment_to_try) instead of (claim, category, next_action)."""
    title = _esc(bp.get("title") or "Pattern")
    pattern = _esc(bp.get("pattern") or "")
    interpretation = _esc(bp.get("interpretation") or "")
    experiment = bp.get("experiment_to_try") or ""
    confidence = bp.get("confidence") or "medium"
    dimension = bp.get("dimension") or ""
    supporting = bp.get("supporting_count") or 0

    chips = []
    if confidence:
        chips.append(f'<span class="chip conf-{_esc(confidence)}">{_esc(confidence)}</span>')
    if dimension:
        chips.append(f'<span class="chip cat-behavioral">{_esc(dimension)}</span>')
    if supporting:
        chips.append(f'<span class="chip">{supporting} sessions</span>')
    if bp.get("non_comparative"):
        chips.append(
            '<span class="chip warn-noncomp" title="Pattern lacks a comparative grounding (X vs Y) — treat as descriptive, not predictive.">no comparison</span>'
        )
    chips_html = '<span class="chips">' + "".join(chips) + "</span>" if chips else ""

    refs = bp.get("evidence_refs") or []
    sids = bp.get("evidence_sessions") or []
    ref_chips = []
    for ref, sid in zip(refs, sids):
        ref_chips.append(
            f'<span class="ref-chip" data-sid="{_esc(sid)}">'
            f'<span>{_esc(ref)}</span><span class="sid">{_esc(_short_session(sid))}</span></span>'
        )
    refs_html = (
        f'<div class="obs-section-label">Evidence ({supporting} sessions — click to drill in)</div>'
        f'<div class="evidence-row">{"".join(ref_chips)}</div>'
        if ref_chips
        else ""
    )

    interp_html = (
        f'<div class="obs-prose"><span class="lead">Why</span>{interpretation}</div>'
        if interpretation
        else ""
    )

    experiment_html = ""
    if experiment:
        copy_attr = _esc(experiment.replace('"', "&quot;"))
        experiment_html = (
            f'<div class="next-action">'
            f'<span class="lead">Try</span>'
            f'<button class="copy" data-copy="{copy_attr}">COPY</button>'
            f"{_esc(experiment)}"
            f"</div>"
        )

    rate_row = ""
    if bp_key:
        rate_row = (
            f'<div class="rate-row" data-rate-row data-obs-index="{idx - 1}" '
            f'data-obs-key="{_esc(bp_key)}" data-obs-title="{_esc(bp.get("title") or "")}" '
            f'data-section="behavioral_patterns">'
            '<span class="rate-label">Resonates?</span>'
            '<button data-rate="useful">[u]seful</button>'
            '<button data-rate="wrong">[w]rong</button>'
            '<button data-rate="known">[k]nown</button>'
            '<button data-rate="skip">[s]kip</button>'
            "</div>"
        )

    if supporting < 3:
        evidence_class = " weak-evidence"
    elif supporting < 6:
        evidence_class = " thin-evidence"
    else:
        evidence_class = ""
    non_comp_class = " non-comparative" if bp.get("non_comparative") else ""
    return (
        f'<article class="observation behavioral{evidence_class}{non_comp_class}" data-obs-block>'
        f'<div class="obs-header">'
        f'<span class="obs-num">§{idx}</span>'
        f'<span class="obs-title">{title}</span>'
        f"{chips_html}"
        f"</div>"
        f'<p class="obs-claim">{pattern}</p>'
        f"{refs_html}"
        f'<div class="drilldown"></div>'
        f"{interp_html}"
        f"{experiment_html}"
        f"{rate_row}"
        f"</article>"
    )


def _render_quick_win(qw: dict) -> str:
    fix = qw.get("fix") or ""
    affected = qw.get("affected_sessions") or 0
    copy_attr = _esc(fix.replace('"', "&quot;"))
    return (
        f"<li>"
        f'<span class="qw-count">{affected} sessions</span>'
        f'<span class="qw-text">{_esc(fix)}</span>'
        f'<button class="qw-copy" data-copy="{copy_attr}">COPY</button>'
        f"</li>"
    )


def _render_project_card(pp: dict) -> str:
    return (
        f'<div class="proj-card">'
        f'<div class="proj-head">'
        f'<span class="proj-name">{_esc(pp.get("project") or "?")}</span>'
        f'<span class="proj-count">{_esc(pp.get("session_count") or "?")} sessions</span>'
        f"</div>"
        f'<div class="proj-headline">{_esc(pp.get("headline") or "")}</div>'
        f'<div class="proj-fric">Biggest friction</div>'
        f'<div class="proj-fric-body">{_esc(pp.get("biggest_friction") or "")}</div>'
        f"</div>"
    )


def _observation_key(observation: dict) -> str:
    """Same hash function used in history.py — safe to import is fine but
    we keep this local to avoid pulling history into dashboard."""
    import hashlib

    seed = (observation.get("title", "") + "|" + (observation.get("claim", "") or ""))[:400]
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


def _observation_cost_minutes(obs: dict, narratives_by_id: dict[str, dict]) -> float:
    """Sum friction-moment minutes across this observation's supporting sessions.

    Loose attribution: any friction in a supporting session counts. This
    over-credits but is the most defensible aggregate without per-friction
    type matching (which is fuzzy).
    """
    total = 0.0
    for sid in obs.get("evidence_sessions") or []:
        n = narratives_by_id.get(sid)
        if not n:
            continue
        for fm in n.get("friction_moments") or []:
            cm = fm.get("cost_active_minutes")
            if isinstance(cm, (int, float)):
                total += cm
    return round(total, 1)


def _render_timeline(timeline: dict) -> str:
    if not timeline:
        return ""
    bits = []
    n = timeline.get("new") or 0
    cont = timeline.get("continuing") or 0
    impr = timeline.get("improving") or 0
    stable = timeline.get("stable") or 0
    wors = timeline.get("worsening") or 0
    resolved = timeline.get("resolved_count") or 0
    if n:
        bits.append(f'<span class="tl-bullet tl-new"><strong>{n}</strong>new</span>')
    if cont:
        bits.append(f'<span class="tl-bullet"><strong>{cont}</strong>continuing</span>')
    if wors:
        bits.append(f'<span class="tl-bullet tl-worsening"><strong>{wors}</strong>worsening</span>')
    if impr:
        bits.append(f'<span class="tl-bullet tl-improving"><strong>{impr}</strong>improving</span>')
    if stable:
        bits.append(f'<span class="tl-bullet"><strong>{stable}</strong>stable</span>')
    if resolved:
        bits.append(f'<span class="tl-bullet tl-resolved"><strong>{resolved}</strong>resolved since last</span>')
    if not bits:
        return ""
    prior_ts = (timeline.get("prior_run_timestamp") or "")[:10]
    prior_html = (
        f'<span class="tl-prior">vs run from {_esc(prior_ts)}</span>' if prior_ts else ""
    )
    return (
        '<div class="timeline">'
        '<span class="tl-label">Week-over-week</span>'
        + " ".join(bits)
        + prior_html
        + "</div>"
    )


def _render_changelog_pulse(synthesis: dict) -> str:
    """Compact above-the-fold 'pulse' summary — what changed this week.

    The detailed Since-last-run section can sit further down; this block is
    designed to be the first thing the eye lands on when the dashboard
    opens, so the user sees the diff before the wall of patterns. Per
    persona-review feedback: the dashboard should be diff-driven, not
    content-driven.

    Three render modes:
      1. No prior run            → friendly hint, no big numbers
      2. First diff (all `new`) → "baseline established" framing
      3. Real diff               → 3 big numbers + top 3 deltas
    """
    try:
        from ..changelog import changelog_for_current
        from ..history import HistoryStore
    except ImportError:
        return ""
    try:
        cl = changelog_for_current(synthesis, HistoryStore())
    except Exception:
        return ""

    if not cl.get("compared"):
        return (
            '<section class="pulse pulse-baseline">'
            '<div class="pulse-label">Since last run</div>'
            '<div class="pulse-msg">First run in history — this becomes the baseline. '
            'Next run will show what changed.</div>'
            '</section>'
        )

    s = cl.get("summary", {})
    against = (cl.get("compared_against_date") or "")[:10]
    obs = cl["observations"]
    bp = cl["behavioral_patterns"]

    # First-real-diff mode: prior run exists but everything is `new` —
    # likely a schema-change run. Don't crow about "29 new" as if it were
    # signal; explain the situation honestly.
    no_continuity = (
        s.get("obs_continuing", 0) == 0
        and s.get("bp_continuing", 0) == 0
        and s.get("obs_escalating", 0) == 0
        and s.get("bp_escalating", 0) == 0
        and s.get("obs_improving", 0) == 0
        and s.get("bp_improving", 0) == 0
        and s.get("obs_resolved", 0) == 0
        and s.get("bp_resolved", 0) == 0
        and s.get("obs_regressed", 0) == 0
        and s.get("bp_regressed", 0) == 0
    )
    if no_continuity:
        new_total = s.get("obs_new", 0) + s.get("bp_new", 0)
        return (
            '<section class="pulse pulse-baseline">'
            '<div class="pulse-label">Since last run</div>'
            f'<div class="pulse-msg">No patterns from <span class="pulse-date">{_esc(against)}</span> match this run\'s shape — likely a schema or prompt change between runs. '
            f'These {new_total} patterns become the new baseline; future diffs will surface real escalations and improvements.</div>'
            '</section>'
        )

    # Top-3 deltas: prefer regressed (most actionable), then escalating,
    # then net-new highlights, then resolved closeable items.
    def _to_card(item: dict, kind: str, arrow: str, color_class: str) -> str:
        title = _esc(item.get("title") or "")
        meta = ""
        if "delta" in item:
            d = item["delta"]
            meta = f'{item.get("previous_count","?")} → {item.get("current_count","?")} ({"+" if d > 0 else ""}{d})'
        elif "current_count" in item:
            meta = f'{item.get("current_count")} sessions'
        elif "previous_count" in item:
            meta = f'was {item.get("previous_count")} sessions'
        return (
            f'<div class="pulse-card {color_class}">'
            f'<div class="pulse-card-head"><span class="pulse-arrow">{arrow}</span><span class="pulse-kind">{kind}</span></div>'
            f'<div class="pulse-card-title">{title}</div>'
            f'<div class="pulse-card-meta">{_esc(meta)}</div>'
            f'</div>'
        )

    candidates: list[tuple[str, dict, str, str, str]] = []
    for item in obs.get("regressed", []) + bp.get("regressed", []):
        candidates.append(("regressed", item, "↻", "regressed", "regressed"))
    # Escalating: bigger delta first
    escalating = sorted(
        obs.get("escalating", []) + bp.get("escalating", []),
        key=lambda x: -abs(x.get("delta", 0)),
    )
    for item in escalating:
        candidates.append(("escalating", item, "▲", "escalating", "escalating"))
    # Net-new: pick by current_count (highest evidence first)
    net_new = sorted(
        obs.get("new", []) + bp.get("new", []),
        key=lambda x: -(x.get("current_count") or 0),
    )
    for item in net_new[:3]:
        candidates.append(("new", item, "+", "new", "new"))
    # Improving + Resolved fill remaining slots
    for item in obs.get("improving", []) + bp.get("improving", []):
        candidates.append(("improving", item, "▼", "improving", "improving"))
    for item in obs.get("resolved", []) + bp.get("resolved", []):
        candidates.append(("resolved", item, "✓", "resolved", "resolved"))

    cards_html = "".join(
        _to_card(item, kind, arrow, color)
        for (kind, item, arrow, color, _) in candidates[:3]
    )

    # Three big numbers: net new (gross of resolved), escalating, regressed.
    new_n = s.get("obs_new", 0) + s.get("bp_new", 0)
    esc_n = s.get("obs_escalating", 0) + s.get("bp_escalating", 0)
    reg_n = s.get("obs_regressed", 0) + s.get("bp_regressed", 0)
    res_n = s.get("obs_resolved", 0) + s.get("bp_resolved", 0)

    big_numbers = (
        '<div class="pulse-numbers">'
        f'<div class="pulse-stat"><div class="pulse-num">{esc_n}</div><div class="pulse-num-lab">escalating</div></div>'
        f'<div class="pulse-stat"><div class="pulse-num pulse-num-warn">{reg_n}</div><div class="pulse-num-lab">regressed</div></div>'
        f'<div class="pulse-stat"><div class="pulse-num">{new_n}</div><div class="pulse-num-lab">new</div></div>'
        f'<div class="pulse-stat pulse-stat-good"><div class="pulse-num">{res_n}</div><div class="pulse-num-lab">resolved</div></div>'
        '</div>'
    )

    return (
        '<section class="pulse">'
        f'<div class="pulse-label">Since last run <span class="pulse-date">({_esc(against)})</span></div>'
        f'{big_numbers}'
        f'<div class="pulse-deltas">{cards_html}</div>'
        '<a class="pulse-jump" href="#changelog-detail">See full changelog ↓</a>'
        '</section>'
    )


def _render_changelog_block(synthesis: dict) -> str:
    """Render the 'Since last run' diff at the top of Findings.

    Empty when there's no prior run in history (first-time users).
    Pulls from the live HistoryStore so the diff stays accurate even if
    the user re-renders the dashboard later without re-synthesizing.
    """
    try:
        from ..changelog import changelog_for_current
        from ..history import HistoryStore
    except ImportError:
        return ""

    try:
        cl = changelog_for_current(synthesis, HistoryStore())
    except Exception:
        return ""
    if not cl.get("compared"):
        return ""

    s = cl.get("summary", {})
    # Don't render if literally nothing changed — avoid Monday-morning noise.
    interesting_keys = (
        "obs_new", "obs_escalating", "obs_improving", "obs_resolved", "obs_regressed",
        "bp_new", "bp_escalating", "bp_improving", "bp_resolved", "bp_regressed",
    )
    if all(s.get(k, 0) == 0 for k in interesting_keys):
        return ""

    against = (cl.get("compared_against_date") or "")[:10]

    def _bullet_list(items: list[dict], show_delta: bool = False) -> str:
        if not items:
            return ""
        out = []
        for item in items:
            title = _esc(item.get("title") or "")
            cat = item.get("category") or ""
            cat_chip = f'<span class="cl-cat">{_esc(cat)}</span>' if cat else ""
            count_label = ""
            if show_delta and "delta" in item:
                d = item["delta"]
                arrow = "▲" if d > 0 else ("▼" if d < 0 else "·")
                count_label = (
                    f'<span class="cl-delta">'
                    f'{arrow} {item["previous_count"]}→{item["current_count"]} '
                    f'({"+" if d > 0 else ""}{d})'
                    f'</span>'
                )
            elif "current_count" in item and item["current_count"]:
                count_label = f'<span class="cl-delta">{item["current_count"]}s</span>'
            elif "previous_count" in item:
                count_label = f'<span class="cl-delta">was {item["previous_count"]}s</span>'
            out.append(f'<li>{title} {cat_chip} {count_label}</li>')
        return "<ul>" + "".join(out) + "</ul>"

    def _bucket(label: str, color_class: str, obs_items: list, bp_items: list, show_delta: bool = False) -> str:
        if not (obs_items or bp_items):
            return ""
        body = ""
        if obs_items:
            body += '<div class="cl-subhead">operational</div>' + _bullet_list(obs_items, show_delta)
        if bp_items:
            body += '<div class="cl-subhead">behavioral</div>' + _bullet_list(bp_items, show_delta)
        return (
            f'<div class="cl-bucket cl-{color_class}">'
            f'<div class="cl-bucket-label">{label}</div>'
            f'{body}'
            f'</div>'
        )

    obs = cl["observations"]
    bp = cl["behavioral_patterns"]

    buckets_html = (
        _bucket("New since last run", "new", obs["new"], bp["new"])
        + _bucket("Escalating", "escalating", obs["escalating"], bp["escalating"], show_delta=True)
        + _bucket("Regressed (was resolved, now back)", "regressed", obs["regressed"], bp["regressed"])
        + _bucket("Improving", "improving", obs["improving"], bp["improving"], show_delta=True)
        + _bucket("Resolved (gone since last run — close the loop?)", "resolved", obs["resolved"], bp["resolved"])
    )

    headline_parts = []
    if s.get("obs_new") or s.get("bp_new"):
        headline_parts.append(f'{s.get("obs_new",0) + s.get("bp_new",0)} new')
    if s.get("obs_escalating") or s.get("bp_escalating"):
        headline_parts.append(f'{s.get("obs_escalating",0) + s.get("bp_escalating",0)} escalating')
    if s.get("obs_resolved") or s.get("bp_resolved"):
        headline_parts.append(f'{s.get("obs_resolved",0) + s.get("bp_resolved",0)} resolved')
    if s.get("obs_regressed") or s.get("bp_regressed"):
        headline_parts.append(f'{s.get("obs_regressed",0) + s.get("bp_regressed",0)} regressed')
    headline_str = " · ".join(headline_parts) if headline_parts else "no major changes"

    return (
        '<section class="block changelog-block" id="changelog-detail">'
        f'<h2 class="section-title">Since last run <span class="count">({_esc(against)})</span> '
        f'<span class="section-sub">— {_esc(headline_str)}</span></h2>'
        f'{buckets_html}'
        '</section>'
    )


def _render_experiments_block() -> str:
    """Render active + recently-graduated self-experiments. Empty string
    if the experiments store is unset or empty (silent first-time users).
    """
    try:
        from ..experiments import ExperimentStore
    except ImportError:
        return ""
    store = ExperimentStore()
    active = store.list("active")
    graduated = store.list("graduated")
    not_tried = store.list("not_tried")
    if not (active or graduated or not_tried):
        return ""

    def _exp_card(exp, status_label: str, status_class: str) -> str:
        last_eval = exp.evaluations[-1] if exp.evaluations else None
        eval_html = ""
        if last_eval:
            adherence = last_eval.get("adherence", "?")
            effect = last_eval.get("effect", "?")
            eval_html = (
                f'<div class="exp-eval">'
                f'<span class="exp-meta-pill">adherence: {_esc(adherence)}</span>'
                f'<span class="exp-meta-pill">effect: {_esc(effect)}</span>'
                f'<span class="exp-meta">evaluated {_esc(last_eval.get("evaluated_at","")[:10])}</span>'
                f'</div>'
            )
        return (
            f'<article class="experiment exp-{status_class}">'
            f'<div class="exp-head">'
            f'<span class="exp-status">{status_label}</span>'
            f'<span class="exp-title">{_esc(exp.title)}</span>'
            f'<span class="exp-dim">{_esc(exp.dimension or "")}</span>'
            f'</div>'
            f'<p class="exp-text">{_esc(exp.experiment_text)}</p>'
            f'<div class="exp-meta">started {_esc(exp.started_at[:10])} · {len(exp.evaluations)} eval(s)</div>'
            f'{eval_html}'
            f'</article>'
        )

    cards = []
    for e in graduated:
        cards.append(_exp_card(e, "graduated", "graduated"))
    for e in active:
        cards.append(_exp_card(e, "active", "active"))
    for e in not_tried:
        cards.append(_exp_card(e, "not tried", "not_tried"))

    total = len(active) + len(graduated) + len(not_tried)
    return (
        '<section class="block experiments-block">'
        f'<h2 class="section-title">Active experiments <span class="count">({total})</span> '
        '<span class="section-sub">— things you committed to by rating a behavioral pattern [useful]</span></h2>'
        f'{"".join(cards)}'
        '</section>'
    )


def _render_findings_section(synthesis: dict, narratives: list[dict]) -> str:
    """The Findings view — what was previously the synthesis page."""
    headline = synthesis.get("headline") or "Cross-session synthesis"
    one_thing = synthesis.get("if_you_do_one_thing_this_week") or ""
    obs_list = synthesis.get("observations") or []
    bp_list = synthesis.get("behavioral_patterns") or []
    qw_list = synthesis.get("quick_wins") or []
    pp_list = synthesis.get("per_project") or []
    meta = synthesis.get("meta", {})

    # Above-the-fold "pulse" — three numbers + top 3 deltas, before
    # anything else. Persona-review: dashboard should be diff-driven.
    pulse_html = _render_changelog_pulse(synthesis)

    # Active self-experiments — pulled from the live experiments store, not
    # from the synthesis output. This makes the section reflect the user's
    # current commitments even if the synthesis is days old.
    experiments_html = _render_experiments_block()

    # Detailed changelog — sits below experiments, anchored for the
    # "See full changelog ↓" jump in the pulse block.
    changelog_html = _render_changelog_block(synthesis)

    total_sessions = len(narratives) or meta.get("input_sessions") or 0
    total_active = sum(n.get("active_minutes") or 0 for n in narratives)
    narratives_by_id = {n["session_id"]: n for n in narratives if n.get("session_id")}

    obs_html_parts = []
    for i, o in enumerate(obs_list, start=1):
        cost = _observation_cost_minutes(o, narratives_by_id)
        key = _observation_key(o)
        obs_html_parts.append(
            _render_observation(o, i, cost_minutes=cost, obs_key=key)
        )
    obs_html = "".join(obs_html_parts)
    bp_html_parts = []
    for i, bp in enumerate(bp_list, start=1):
        key = _observation_key(bp)
        bp_html_parts.append(_render_behavioral_pattern(bp, i, bp_key=key))
    bp_html = "".join(bp_html_parts)
    qw_html = "".join(_render_quick_win(q) for q in qw_list)
    pp_html = "".join(_render_project_card(p) for p in pp_list)
    timeline_html = _render_timeline(meta.get("timeline") or {})

    notes = meta.get("notes") or ""
    notes_html = (
        f'<section class="block"><div class="notes"><span class="lead">Notes</span>{_esc(notes)}</div></section>'
        if notes
        else ""
    )

    callout_html = ""
    if one_thing:
        callout_html = (
            '<div class="callout">'
            '<div class="label">If you do one thing this week</div>'
            f'<div class="body">{_esc(one_thing)}</div>'
            "</div>"
        )

    aggregate_html = (
        '<div class="aggregate-strip">'
        f'<div class="stat"><div class="num">{total_sessions}</div><div class="lab">Sessions</div></div>'
        f'<div class="stat"><div class="num">{len(obs_list)}</div><div class="lab">Observations</div></div>'
        f'<div class="stat"><div class="num">{len(bp_list)}</div><div class="lab">Patterns</div></div>'
        f'<div class="stat"><div class="num">{len(qw_list)}</div><div class="lab">Quick wins</div></div>'
        f'<div class="stat"><div class="num">{len(pp_list)}</div><div class="lab">Projects</div></div>'
        f'<div class="stat"><div class="num">{int(total_active):,}</div><div class="lab">Active min</div></div>'
        f'<div class="stat"><div class="num">{meta.get("fabricated_ref_count", 0)}</div><div class="lab">Fabrications</div></div>'
        "</div>"
    )

    obs_block = (
        '<section class="block">'
        f'<h2 class="section-title">Observations <span class="count">({len(obs_list)})</span> '
        '<span class="section-sub">— operational fixes</span></h2>'
        f"{obs_html}"
        "</section>"
        if obs_html
        else ""
    )
    bp_block = (
        '<section class="block">'
        f'<h2 class="section-title">Behavioral patterns <span class="count">({len(bp_list)})</span> '
        '<span class="section-sub">— how you work, comparative & experiments to try</span></h2>'
        f"{bp_html}"
        "</section>"
        if bp_html
        else ""
    )
    qw_block = (
        '<section class="block quick-wins">'
        f'<h2 class="section-title">Quick wins <span class="count">({len(qw_list)})</span></h2>'
        f"<ol>{qw_html}</ol>"
        "</section>"
        if qw_html
        else ""
    )
    pp_block = (
        '<section class="block">'
        f'<h2 class="section-title">Per project <span class="count">({len(pp_list)})</span></h2>'
        f'<div class="per-project">{pp_html}</div>'
        "</section>"
        if pp_html
        else ""
    )

    return f"""
<section class="view" data-view="findings">
  {pulse_html}
  <h1 class="headline">{_esc(headline)}</h1>
  {callout_html}
  {experiments_html}
  {changelog_html}
  {timeline_html}
  {aggregate_html}
  {obs_block}
  {bp_block}
  {qw_block}
  {pp_block}
  {notes_html}
</section>
"""


def _render_explore_section(synthesis: dict, narratives: list[dict]) -> str:
    """The Explore view — sessions browser + aggregated views."""
    n_sessions = len(narratives)
    n_env = sum(
        len(n.get("recurring_environmental_issues") or []) for n in narratives
    )
    n_lessons = sum(
        1 for n in narratives if (n.get("lesson_for_user") or "").strip()
    ) + sum(1 for n in narratives if (n.get("counterfactual") or "").strip())
    n_projects = len(
        {n.get("project_label") for n in narratives if n.get("project_label")}
    )

    sub_nav = (
        '<nav class="tabs">'
        f'<button data-tab="sessions">Sessions <span class="count">({n_sessions})</span></button>'
        f'<button data-tab="env">Env issues <span class="count">({n_env})</span></button>'
        f'<button data-tab="lessons">Lessons <span class="count">({n_lessons})</span></button>'
        f'<button data-tab="projects">Projects <span class="count">({n_projects})</span></button>'
        "</nav>"
    )

    return f"""
<section class="view" data-view="explore">
  {sub_nav}
  {_render_sessions_tab(narratives)}
  {_render_env_tab(narratives)}
  {_render_lessons_tab(narratives)}
  {_render_projects_tab(narratives)}
</section>
"""


def render_dashboard(
    synthesis: dict,
    narratives: list[dict],
    all_narratives: list[dict] | None = None,
) -> str:
    """Build the unified self-contained tessera dashboard.

    One HTML file with two top-level views:
      - Findings — the synthesis output, scoped to ``narratives``
      - Explore  — sessions browser + aggregated env issues / lessons / projects,
        scoped to ``all_narratives`` if provided (the user's full cached
        history), else ``narratives``.

    The Findings view always reflects the current synthesis run. The Explore
    view shows everything the user has ever extracted, so they can browse
    older sessions even when their weekly synthesis is just the last 30 days.
    """
    if all_narratives is None:
        all_narratives = narratives

    meta = synthesis.get("meta", {})
    fc = meta.get("filter_context") or {}
    in_synthesis = len(narratives)
    in_explore = len(all_narratives)

    total_active_synth = sum(n.get("active_minutes") or 0 for n in narratives)
    by_agent_synth: dict[str, int] = {}
    for n in narratives:
        a = n.get("agent") or "unknown"
        by_agent_synth[a] = by_agent_synth.get(a, 0) + 1
    agent_str_synth = " · ".join(
        f"{c} {a}" for a, c in sorted(by_agent_synth.items(), key=lambda kv: -kv[1])
    ) or "—"

    # Filter-context line for the Findings masthead
    fc_bits: list[str] = []
    if fc.get("lookback_days"):
        fc_bits.append(f"{fc['lookback_days']}d window")
    elif fc.get("lookback_days") is None and fc:
        fc_bits.append("all-time")
    if fc.get("min_events"):
        fc_bits.append(f"min ≥{fc['min_events']} events")
    drops = []
    if fc.get("dropped_short"):
        drops.append(f"{fc['dropped_short']} short")
    if fc.get("dropped_slash"):
        drops.append(f"{fc['dropped_slash']} slash-cmd")
    if drops:
        fc_bits.append("excluded: " + " + ".join(drops))
    fc_line = " · ".join(fc_bits) if fc_bits else ""

    fab = meta.get("fabricated_ref_count")
    fab_html = (
        f' · {fab} fabricated ref{"s" if fab != 1 else ""} dropped' if fab else ""
    )
    findings_meta = " · ".join(
        bit
        for bit in [
            f"{in_synthesis} sessions",
            f"{int(total_active_synth):,} active min" if total_active_synth else "",
            agent_str_synth,
            (meta.get("generated_at") or "")[:10],
            meta.get("model") or "",
        ]
        if bit
    )
    findings_meta_html = (
        _esc(findings_meta) + fab_html
        + (f'<br><span class="filter-context">{_esc(fc_line)}</span>' if fc_line else "")
    )

    # Explore meta line — uses all_narratives counts
    by_agent_all: dict[str, int] = {}
    total_active_all = 0
    for n in all_narratives:
        a = n.get("agent") or "unknown"
        by_agent_all[a] = by_agent_all.get(a, 0) + 1
        total_active_all += n.get("active_minutes") or 0
    agent_str_all = " · ".join(
        f"{c} {a}" for a, c in sorted(by_agent_all.items(), key=lambda kv: -kv[1])
    ) or "—"
    explore_scope = (
        f"showing all {in_explore} cached sessions"
        if in_explore > in_synthesis
        else f"showing {in_explore} sessions"
    )
    explore_meta = " · ".join(
        bit
        for bit in [
            explore_scope,
            f"{int(total_active_all):,} active min",
            agent_str_all,
        ]
        if bit
    )
    explore_meta_html = _esc(explore_meta)

    # Embed all narratives so Explore can render them; Findings drilldowns
    # also use this same data (any session referenced in observations is in
    # the synthesis set, which is a subset of all_narratives).
    embed = {
        "sessions": [_expanded_session_for_explore(n) for n in all_narratives],
        "run_slug": meta.get("run_slug") or "latest",
    }

    primary_nav = (
        '<nav class="primary-nav">'
        '<button data-view="findings">Findings</button>'
        f'<button data-view="explore">Explore <span class="count">({in_explore})</span></button>'
        "</nav>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tessera</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}{EXPLORE_CSS}</style>
</head>
<body>
<div class="frame">
  <header class="masthead">
    <div class="brand">Tessera</div>
    {primary_nav}
    <div class="meta meta-findings">{findings_meta_html}</div>
    <div class="meta meta-explore" hidden>{explore_meta_html}</div>
  </header>

  {_render_findings_section(synthesis, narratives)}
  {_render_explore_section(synthesis, all_narratives)}

  <footer class="colophon">
    <span>tessera · cross-session synthesis</span>
    <span>schema v1 · {_esc(meta.get("model") or "")}</span>
  </footer>
</div>

<button id="rate-sync" class="rate-sync" hidden>
  <span class="count">0</span> ratings ready · <strong style="margin-left:6px">SAVE</strong>
</button>

<div id="rate-modal" class="rate-modal" hidden>
  <div class="rate-modal-inner">
    <h3>Save your ratings to history</h3>
    <p>Copy the command below and paste it in your terminal. Your ratings get applied to the latest synthesis run, and the next <code>tessera run</code> will use them as prior context.</p>
    <pre></pre>
    <div class="actions">
      <button data-close>Close</button>
      <button class="primary" data-copy-cmd>Copy command</button>
    </div>
  </div>
</div>

<script>window.__AR__ = {_safe_inline_json(embed)};</script>
<script>{COMBINED_JS}</script>
</body>
</html>
"""


def write_dashboard(
    synthesis_path: Path,
    narratives_dir: Path,
    output_path: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Read synthesis + narratives from disk, write the unified dashboard HTML.

    One self-contained file with two top-level views — Findings (synthesis
    output) and Explore (sessions browser + aggregated env issues / lessons /
    projects). Switch between them via the in-page nav or via URL hash
    (``#findings``, ``#explore``, ``#explore/sessions``, ``#session=<id>``).

    If ``cache_dir`` is provided, Explore will show all cached narratives
    (the user's full history), not just the sessions in the current synthesis
    window. Findings always reflects the current synthesis run.
    """
    synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
    narratives: list[dict] = []
    for p in sorted(narratives_dir.glob("*.json")):
        try:
            narratives.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue

    all_narratives: list[dict] | None = None
    if cache_dir and cache_dir.exists():
        merged: dict[str, dict] = {n["session_id"]: n for n in narratives if n.get("session_id")}
        for p in sorted(cache_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sid = payload.get("session_id")
            if sid and sid not in merged:
                merged[sid] = payload
        all_narratives = list(merged.values())

    target = output_path or synthesis_path.with_suffix(".html")
    target.write_text(
        render_dashboard(synthesis, narratives, all_narratives=all_narratives),
        encoding="utf-8",
    )
    return target


# ============================================================================
# Explore page — browse all the rich data
# ============================================================================

EXPLORE_CSS = r"""
nav.tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--line-strong);
  margin-bottom: 32px;
  font-family: var(--sans);
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 600;
}
nav.tabs button {
  background: none;
  border: 0;
  padding: 14px 18px;
  color: var(--ink-mute);
  cursor: pointer;
  font: inherit;
  letter-spacing: inherit;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
nav.tabs button:hover { color: var(--ink-soft); }
nav.tabs button.active {
  color: var(--ink);
  border-bottom-color: var(--warm);
}
nav.tabs button .count {
  color: var(--ink-mute);
  font-weight: 400;
  margin-left: 4px;
  font-size: 11px;
}
nav.tabs button.active .count { color: var(--warm); }

.tab-panel { display: none; }
.tab-panel.active { display: block; }

.toolbar {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
  align-items: center;
}
.toolbar input[type="search"], .toolbar select {
  font-family: var(--sans);
  font-size: 13px;
  padding: 6px 10px;
  border: 1px solid var(--line-strong);
  background: #fff;
  color: var(--ink);
  border-radius: 3px;
  min-width: 180px;
}
.toolbar input[type="search"]:focus, .toolbar select:focus {
  outline: none;
  border-color: var(--warm-soft);
}
.toolbar .summary {
  font-family: var(--sans);
  font-size: 12px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
  margin-left: auto;
}

table.sessions {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--sans);
  font-size: 13px;
  background: #fff;
  border: 1px solid var(--line);
}
table.sessions thead {
  background: var(--hint);
  border-bottom: 1px solid var(--line-strong);
}
table.sessions th {
  text-align: left;
  padding: 10px 12px;
  font-weight: 600;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-soft);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
table.sessions th:hover { color: var(--warm); }
table.sessions th .arrow { color: var(--ink-mute); margin-left: 4px; font-size: 10px; }
table.sessions td {
  padding: 8px 12px;
  border-top: 1px solid var(--line);
  vertical-align: top;
  color: var(--ink);
}
table.sessions tr.row { cursor: pointer; }
table.sessions tr.row:hover td { background: var(--hint); }
table.sessions tr.row.open td { background: var(--hint); }
table.sessions td.sid { font-family: var(--mono); font-size: 11px; color: var(--ink-mute); }
table.sessions td.proj { font-family: var(--mono); font-size: 11px; color: var(--ink-soft); word-break: break-all; max-width: 220px; }
table.sessions td.num { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink-soft); }
table.sessions td .pill {
  display: inline-block;
  font-size: 10px;
  padding: 1px 6px;
  background: var(--hint);
  color: var(--warm);
  border-radius: 8px;
  letter-spacing: 0.04em;
}
table.sessions td .agent-pill {
  display: inline-block;
  font-size: 10px;
  padding: 1px 6px;
  background: #fff;
  border: 1px solid var(--line-strong);
  color: var(--ink-soft);
  border-radius: 8px;
  letter-spacing: 0.04em;
}

/* Above-the-fold pulse — first thing the eye hits on Monday morning. */
.pulse {
  margin: 0 0 32px 0;
  padding: 22px 28px 18px 28px;
  background: linear-gradient(180deg, #fbf6e8 0%, #faf9f4 100%);
  border: 1px solid var(--line-strong);
  border-radius: 8px;
}
.pulse.pulse-baseline {
  background: var(--paper);
  border-style: dashed;
}
.pulse-label {
  font-family: var(--mono);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-mute);
  margin-bottom: 12px;
}
.pulse-date { color: var(--ink-soft); }
.pulse-msg {
  font-family: var(--serif);
  font-size: 15px;
  color: var(--ink-soft);
  line-height: 1.5;
  font-style: italic;
}
.pulse-numbers {
  display: flex;
  gap: 36px;
  margin-bottom: 18px;
  flex-wrap: wrap;
}
.pulse-stat { text-align: left; }
.pulse-num {
  font-family: var(--serif);
  font-size: 36px;
  font-weight: 600;
  color: var(--ink);
  line-height: 1;
}
.pulse-num.pulse-num-warn { color: var(--rust); }
.pulse-stat-good .pulse-num { color: var(--leaf); }
.pulse-num-lab {
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-mute);
  margin-top: 4px;
}
.pulse-deltas {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}
.pulse-card {
  padding: 10px 14px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 4px;
  border-left: 3px solid var(--line-strong);
}
.pulse-card.regressed { border-left-color: var(--rust); }
.pulse-card.escalating { border-left-color: var(--rust); }
.pulse-card.new { border-left-color: var(--warm); }
.pulse-card.improving { border-left-color: var(--leaf); }
.pulse-card.resolved { border-left-color: var(--leaf); opacity: 0.85; }
.pulse-card-head {
  display: flex; align-items: center; gap: 6px;
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-mute);
  margin-bottom: 4px;
}
.pulse-arrow {
  font-size: 14px;
  color: var(--ink-soft);
}
.pulse-card.regressed .pulse-arrow,
.pulse-card.escalating .pulse-arrow { color: var(--rust); }
.pulse-card.new .pulse-arrow { color: var(--warm); }
.pulse-card.improving .pulse-arrow,
.pulse-card.resolved .pulse-arrow { color: var(--leaf); }
.pulse-card-title {
  font-family: var(--serif);
  font-size: 14px;
  color: var(--ink);
  line-height: 1.3;
  margin-bottom: 4px;
}
.pulse-card-meta {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--ink-mute);
}
.pulse-jump {
  display: inline-block;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--ink-mute);
  text-decoration: none;
  border-bottom: 1px dotted var(--line-strong);
  padding-bottom: 1px;
}
.pulse-jump:hover { color: var(--warm); border-bottom-color: var(--warm); }

.changelog-block { margin-top: 24px; }
.changelog-block .cl-bucket {
  margin: 12px 0;
  padding: 10px 16px;
  border-left: 3px solid var(--line-strong);
  background: #fbfaf3;
  border-radius: 0 4px 4px 0;
}
.changelog-block .cl-bucket.cl-new        { border-left-color: var(--warm); background: #faf3e8; }
.changelog-block .cl-bucket.cl-escalating { border-left-color: var(--rust); background: #faecea; }
.changelog-block .cl-bucket.cl-regressed  { border-left-color: var(--rust); background: #faecea; }
.changelog-block .cl-bucket.cl-improving  { border-left-color: var(--leaf); background: #f1f6ee; }
.changelog-block .cl-bucket.cl-resolved   { border-left-color: var(--leaf); background: #f1f6ee; opacity: 0.95; }
.changelog-block .cl-bucket-label {
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  color: var(--ink-soft); letter-spacing: 0.04em; text-transform: uppercase;
  margin-bottom: 4px;
}
.changelog-block .cl-subhead {
  font-family: var(--mono); font-size: 10px; color: var(--ink-mute);
  margin-top: 6px; letter-spacing: 0.04em;
}
.changelog-block ul {
  list-style: none; padding: 0; margin: 4px 0;
}
.changelog-block li {
  padding: 2px 0;
  font-size: 14px;
  color: var(--ink);
}
.changelog-block .cl-cat {
  font-family: var(--mono); font-size: 10px; color: var(--ink-mute);
  margin-left: 6px;
}
.changelog-block .cl-delta {
  font-family: var(--mono); font-size: 10px; color: var(--warm);
  margin-left: 6px;
}

.experiments-block .experiment {
  margin: 14px 0;
  padding: 14px 18px;
  background: #fbf6e8;
  border-left: 3px solid var(--warm-soft);
  border-radius: 0 6px 6px 0;
}
.experiments-block .experiment.exp-graduated { border-left-color: var(--leaf); background: #f1f6ee; }
.experiments-block .experiment.exp-not_tried { border-left-color: var(--ink-mute); background: #f5f3ec; opacity: 0.85; }
.experiments-block .exp-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.experiments-block .exp-status {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  padding: 1px 7px; background: var(--warm); color: #fff; border-radius: 8px;
  letter-spacing: 0.06em;
}
.experiments-block .experiment.exp-graduated .exp-status { background: var(--leaf); }
.experiments-block .experiment.exp-not_tried .exp-status { background: var(--ink-mute); }
.experiments-block .exp-title { font-family: var(--serif); font-size: 16px; color: var(--ink); font-weight: 600; }
.experiments-block .exp-dim { font-family: var(--mono); font-size: 10px; color: var(--ink-mute); margin-left: auto; }
.experiments-block .exp-text { color: var(--ink-soft); font-size: 14px; margin: 8px 0; }
.experiments-block .exp-meta { font-family: var(--mono); font-size: 11px; color: var(--ink-mute); }
.experiments-block .exp-eval { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
.experiments-block .exp-meta-pill {
  font-family: var(--mono); font-size: 10px; padding: 1px 7px;
  border: 1px solid var(--line-strong); border-radius: 8px; color: var(--ink-soft);
}

.outcome-pill {
  display: inline-block;
  font-size: 10px;
  padding: 1px 7px;
  border-radius: 8px;
  letter-spacing: 0.04em;
  font-weight: 500;
  white-space: nowrap;
}
.outcome-pill.out-shipped_clean        { background: #e2eee0; color: #2d4a26; border: 1px solid #c6dac1; }
.outcome-pill.out-shipped_direct       { background: #ddebd9; color: #2f4f29; border: 1px solid #c0d6ba; }
.outcome-pill.out-shipped_with_followups { background: #f4ecd8; color: #6b4f1d; border: 1px solid #e0d3a8; }
.outcome-pill.out-reverted             { background: #f4dad5; color: #7a2a1a; border: 1px solid #e6b6ab; }
.outcome-pill.out-in_progress          { background: #e6ecef; color: #34495a; border: 1px solid #cad6dd; }
.outcome-pill.out-abandoned            { background: #ece4d6; color: #6b5a3a; border: 1px solid #d8c8a8; }
.outcome-pill.out-exploration          { background: #ecedf3; color: #3a4a6a; border: 1px solid #d8dae5; }
.outcome-pill.out-non_repo             { background: #f1ede0; color: #7c7568; border: 1px solid #e2dcc6; }
.outcome-pill.out-unshipped            { background: #f4ede2; color: #6b5a3a; border: 1px solid #e2d3b5; }
.outcome-pill.out-no_artifact          { background: #f1ede0; color: #7c7568; border: 1px solid #e2dcc6; }
.outcome-pill.out-unavailable          { background: transparent; color: #b3ad9f; border: 1px dashed #d6d0c2; }

tr.detail-row { display: none; }
tr.detail-row.open { display: table-row; }
tr.detail-row td {
  background: var(--paper);
  padding: 18px 22px;
  border-top: 0;
}
.session-detail h3 {
  font-family: var(--serif);
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 8px;
  color: var(--ink);
}
.session-detail .sd-meta {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
  margin-bottom: 12px;
}
.session-detail .sd-goal {
  font-family: var(--serif);
  font-size: 16px;
  color: var(--ink);
  margin: 0 0 16px;
  font-style: italic;
}
.session-detail .grid-2 {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 28px;
}
@media (max-width: 720px) {
  .session-detail .grid-2 { grid-template-columns: 1fr; }
}
.session-detail h4 {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-mute);
  margin: 18px 0 8px;
}
.session-detail .item {
  font-family: var(--serif);
  font-size: 14px;
  color: var(--ink);
  margin: 6px 0;
}
.session-detail .item .small {
  color: var(--ink-mute);
  font-size: 12px;
  margin-left: 4px;
  font-family: var(--sans);
}
.session-detail .quote {
  border-left: 2px solid var(--line-strong);
  padding: 4px 10px;
  margin: 4px 0 8px;
  font-style: italic;
  color: var(--ink-soft);
  font-size: 13px;
}
.session-detail .stats-card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 14px 16px;
  font-family: var(--sans);
  font-size: 12px;
  color: var(--ink-soft);
}
.session-detail .stats-card .row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px dashed var(--line);
}
.session-detail .stats-card .row:last-child { border-bottom: 0; }
.session-detail .stats-card .row .lab {
  color: var(--ink-mute);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 10px;
}
.session-detail .stats-card .row .val {
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

.env-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
}
.env-card {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 3px;
  padding: 14px 18px;
}
.env-card .env-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.env-card .env-text {
  font-family: var(--serif);
  font-size: 15px;
  color: var(--ink);
  line-height: 1.45;
}
.env-card .env-foot {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  margin-top: 8px;
}
.env-card .env-foot a {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--warm);
  text-decoration: none;
  border-bottom: 1px dotted var(--warm-soft);
  margin-right: 6px;
}

.lesson-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 18px;
}
@media (max-width: 980px) {
  .lesson-grid { grid-template-columns: 1fr; }
}
.lesson-col h3 {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-mute);
  margin: 0 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
}
.lesson-card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 12px 14px;
  margin-bottom: 12px;
}
.lesson-card .l-text {
  font-family: var(--serif);
  font-size: 14px;
  color: var(--ink);
  line-height: 1.45;
}
.lesson-card .l-foot {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--ink-mute);
  margin-top: 6px;
  word-break: break-all;
}

.proj-deep {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 3px;
  padding: 18px 22px;
  margin-bottom: 18px;
}
.proj-deep .pd-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.proj-deep .pd-name {
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
}
.proj-deep .pd-meta {
  font-family: var(--sans);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
}
.proj-deep .pd-headline {
  font-family: var(--serif);
  font-size: 15px;
  margin: 4px 0 14px;
  color: var(--ink);
}
.proj-deep h4 {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-mute);
  margin: 10px 0 6px;
}
.proj-deep .session-line {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--ink-soft);
  padding: 4px 0;
  border-top: 1px dashed var(--line);
  display: flex;
  justify-content: space-between;
  gap: 8px;
}
.proj-deep .session-line .when { color: var(--ink-mute); font-size: 11px; }

.cross-page-link {
  font-family: var(--sans);
  font-size: 12px;
  color: var(--warm);
  text-decoration: none;
  border-bottom: 1px dotted var(--warm-soft);
  letter-spacing: 0.04em;
}
.cross-page-link:hover { color: var(--ink); border-bottom-color: var(--ink); }
"""

EXPLORE_JS = r"""
(function() {
  const D = window.__EXPLORE_DATA__ || {};
  const sessions = D.sessions || [];
  const sessionsById = sessions.reduce(function(acc, s) {
    acc[s.session_id] = s;
    return acc;
  }, {});

  function esc(s) {
    s = String(s == null ? '' : s);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function fmt(n, suffix) {
    if (n === undefined || n === null) return '';
    return (Math.round(n * 10) / 10) + (suffix || '');
  }

  // --- Tabs ---
  const tabs = document.querySelectorAll('nav.tabs button');
  const panels = document.querySelectorAll('.tab-panel');
  function showTab(name) {
    tabs.forEach(function(b) { b.classList.toggle('active', b.dataset.tab === name); });
    panels.forEach(function(p) { p.classList.toggle('active', p.dataset.panel === name); });
    if (window.history && history.replaceState) history.replaceState(null, '', '#' + name);
  }
  tabs.forEach(function(b) {
    b.addEventListener('click', function() { showTab(b.dataset.tab); });
  });
  // Hash deep-link (initial + on subsequent changes)
  function applyHash() {
    const hash = location.hash || '#sessions';
    if (hash.startsWith('#session=')) return; // handled separately below
    const name = hash.slice(1);
    if (document.querySelector('nav.tabs button[data-tab="' + name + '"]')) showTab(name);
    else showTab('sessions');
  }
  applyHash();
  window.addEventListener('hashchange', applyHash);

  // --- Sessions table ---
  const tbody = document.querySelector('table.sessions tbody');
  const filterAgent = document.getElementById('filter-agent');
  const filterProject = document.getElementById('filter-project');
  const filterWaste = document.getElementById('filter-waste');
  const search = document.getElementById('filter-search');
  const summary = document.getElementById('sessions-summary');

  let sortKey = 'date';
  let sortDir = -1; // -1 desc, 1 asc

  function shortSid(s) {
    if (!s) return '?';
    const i = s.indexOf(':');
    if (i < 0) return s;
    return s.slice(0, i+1) + s.slice(i+1, i+9) + '…';
  }

  function rowMatches(s) {
    if (filterAgent.value && s.agent !== filterAgent.value) return false;
    if (filterProject.value && s.project !== filterProject.value) return false;
    if (filterWaste.value && s.waste_signature !== filterWaste.value) return false;
    const q = search.value.trim().toLowerCase();
    if (q) {
      const hay = (s.session_id + ' ' + (s.project||'') + ' ' + (s.goal||'') + ' ' +
                   (s.waste_signature||'') + ' ' + (s.topics||[]).join(' ')).toLowerCase();
      if (hay.indexOf(q) < 0) return false;
    }
    return true;
  }

  function sortRows(rows) {
    return rows.slice().sort(function(a, b) {
      let av = a[sortKey], bv = b[sortKey];
      if (av == null) av = '';
      if (bv == null) bv = '';
      if (typeof av === 'string') av = av.toLowerCase();
      if (typeof bv === 'string') bv = bv.toLowerCase();
      if (av < bv) return -1 * sortDir;
      if (av > bv) return 1 * sortDir;
      return 0;
    });
  }

  function renderSessions() {
    const filtered = sessions.filter(rowMatches);
    const sorted = sortRows(filtered);
    let html = '';
    sorted.forEach(function(s) {
      html += '<tr class="row" data-sid="' + esc(s.session_id) + '">' +
        '<td class="sid">' + esc(shortSid(s.session_id)) + '</td>' +
        '<td><span class="agent-pill">' + esc(s.agent) + '</span></td>' +
        '<td class="proj">' + esc(s.project || '') + '</td>' +
        '<td>' + esc(s.date || '') + '</td>' +
        '<td class="num">' + (s.events || 0) + '</td>' +
        '<td class="num">' + fmt(s.active_min) + '</td>' +
        '<td class="num">' + (s.bursts || 0) + '</td>' +
        '<td>' + (s.waste_signature && s.waste_signature !== 'none'
          ? '<span class="pill">' + esc(s.waste_signature) + '</span>' : '') + '</td>' +
        '<td class="num">' + (s.friction_count || 0) + '</td>' +
        '<td class="num">' + (s.user_caught || 0) + '</td>' +
        '<td>' + (s.outcome_signal
          ? '<span class="outcome-pill out-' + esc(s.outcome_signal) + '">' +
            esc(s.outcome_signal.replace(/_/g, ' ')) + '</span>'
          : '') + '</td>' +
        '</tr>' +
        '<tr class="detail-row" data-sid="' + esc(s.session_id) + '">' +
        '<td colspan="11"></td></tr>';
    });
    tbody.innerHTML = html;
    summary.textContent = filtered.length + ' / ' + sessions.length + ' sessions';
    // Wire row clicks
    tbody.querySelectorAll('tr.row').forEach(function(row) {
      row.addEventListener('click', function() { toggleDetail(row); });
    });
  }

  function buildDetail(s) {
    const tasks = (s.tasks || []).map(function(t) {
      return '<div class="item"><strong>' + esc(t.id || '') + '</strong> · ' +
        esc(t.intent || '') + ' <span class="small">(' + esc(t.type || '?') + ' · ' +
        esc(t.difficulty || '?') + ' · ' + esc(t.outcome || '?') + ')</span></div>';
    }).join('');
    const friction = (s.friction || []).map(function(f) {
      const quote = f.quote ? '<div class="quote">' + esc(f.quote) + '</div>' : '';
      return '<div class="item"><strong>' + esc(f.type || '?') + '</strong> ' +
        '<span class="small">(' + esc(f.tool_cat || '?') + ' · ' + (f.cost_events || 0) + 'ev · ' +
        fmt(f.cost_active_minutes, 'min') + ')</span><br>' +
        esc(f.desc || '') + quote + '</div>';
    }).join('');
    const decisions = (s.key_decisions || []).map(function(d) {
      return '<div class="item"><strong>ev' + (d.event_index||0) + '</strong> · ' + esc(d.decision || '') +
        '<div class="quote">' + esc(d.retrospective || '') + '</div></div>';
    }).join('');
    const deadends = (s.dead_ends || []).map(function(d) {
      return '<div class="item">' + esc(d.approach || '') +
        '<div class="quote">Lesson: ' + esc(d.lesson || '') + '</div></div>';
    }).join('');
    const env = (s.env_issues || []).map(function(e) {
      return '<div class="item">• ' + esc(e.desc || '') +
        ' <span class="small">(' + (e.occurrences || 0) + 'x)</span></div>';
    }).join('');
    const ucme = (s.user_caught_examples || []).map(function(e) {
      return '<div class="item">ev' + (e.event_index||0) + ': ' + esc(e.what_user_caught || '') + '</div>';
    }).join('');
    const counterfactual = s.counterfactual ? '<h4>Counterfactual</h4><div class="item">' + esc(s.counterfactual) + '</div>' : '';
    const lessonUser = s.lesson_user ? '<h4>Lesson for user</h4><div class="item">' + esc(s.lesson_user) + '</div>' : '';
    const lessonAgent = s.lesson_agent ? '<h4>Lesson for agent</h4><div class="item">' + esc(s.lesson_agent) + '</div>' : '';
    const notable = s.notable ? '<h4>Notable</h4><div class="item"><em>' + esc(s.notable) + '</em></div>' : '';
    const topics = (s.topics || []).length
      ? '<h4>Topics</h4><div class="item">' + (s.topics || []).map(esc).join(' · ') + '</div>'
      : '';

    const stats = [
      ['session_id', s.session_id],
      ['agent', s.agent],
      ['project', s.project],
      ['date', s.date],
      ['weekday', s.weekday],
      ['model', s.primary_model],
      ['events', s.events],
      ['tool_calls', s.tool_calls],
      ['user_turns', s.user_turn_count],
      ['subagents', s.subagent_count],
      ['active_min', fmt(s.active_min)],
      ['wall_clock_min', fmt(s.wall_clock_min)],
      ['bursts', s.bursts],
      ['primary_burst_min', fmt(s.primary_burst_min)],
      ['tool_err_rate', fmt((s.tool_err_rate||0)*100, '%')],
      ['unique_files', s.unique_files],
      ['tests_invoked', s.tests_invoked],
      ['user_corrections', s.user_corrections],
      ['user_caught_errors', s.user_caught],
      ['verification', s.verification],
      ['waste_signature', s.waste_signature],
      ['narrative_quality', s.narrative_quality],
    ];
    const statsHtml = stats.map(function(kv) {
      return '<div class="row"><span class="lab">' + esc(kv[0]) + '</span>' +
        '<span class="val">' + esc(kv[1] == null ? '—' : kv[1]) + '</span></div>';
    }).join('');

    // Outcome block — what happened to the work after the session
    let outcomeHtml = '';
    if (s.outcome) {
      const o = s.outcome;
      const lines = ['<div class="item"><strong>signal:</strong> <span class="outcome-pill out-' +
        esc(o.signal) + '">' + esc(o.signal.replace(/_/g, ' ')) + '</span></div>'];
      if (o.prs && o.prs.length) {
        const prHtml = o.prs.map(function(p) {
          return '<div class="item"><strong>PR #' + p.n + '</strong> ' +
            '<span class="small">state=' + esc(p.state || '?') + ' · ci=' + esc(p.ci || '?') +
            ' · review=' + esc(p.review || '?') + '</span></div>';
        }).join('');
        lines.push(prHtml);
      }
      if (o.churn) {
        lines.push('<div class="item"><strong>14-day churn</strong> ' +
          '<span class="small">' + (o.churn.commits_in_14d || 0) + ' commits touching files (' +
          (o.churn.fixup || 0) + ' fixup-shape, ' + (o.churn.revert || 0) + ' revert)</span></div>');
      }
      if (o.branch) {
        lines.push('<div class="item"><strong>branch</strong> ' +
          '<span class="small">' + (o.branch.merged_into ? 'merged → ' + esc(o.branch.merged_into) : '') +
          (o.branch.commits_after ? ' · ' + o.branch.commits_after + ' commits after session' : '') +
          '</span></div>');
      }
      outcomeHtml = '<h4>Outcome</h4>' + lines.join('');
    }

    return '<div class="session-detail">' +
      '<h3>' + esc(s.goal || 'Session') + '</h3>' +
      '<div class="sd-meta">' + esc(s.session_id) + '</div>' +
      '<div class="grid-2">' +
      '<div>' +
      (tasks ? '<h4>Tasks</h4>' + tasks : '') +
      (friction ? '<h4>Friction moments</h4>' + friction : '') +
      (decisions ? '<h4>Key decisions</h4>' + decisions : '') +
      (deadends ? '<h4>Dead ends</h4>' + deadends : '') +
      (env ? '<h4>Recurring environmental issues</h4>' + env : '') +
      (ucme ? '<h4>User-caught model errors</h4>' + ucme : '') +
      outcomeHtml +
      counterfactual + lessonUser + lessonAgent + notable + topics +
      '</div>' +
      '<div><div class="stats-card">' + statsHtml + '</div></div>' +
      '</div>' +
      '</div>';
  }

  function toggleDetail(row) {
    const sid = row.dataset.sid;
    const detail = tbody.querySelector('tr.detail-row[data-sid="' + CSS.escape(sid) + '"]');
    if (!detail) return;
    const wasOpen = row.classList.contains('open');
    // Close any other open
    tbody.querySelectorAll('tr.row.open').forEach(function(r) { r.classList.remove('open'); });
    tbody.querySelectorAll('tr.detail-row.open').forEach(function(r) {
      r.classList.remove('open');
      r.querySelector('td').innerHTML = '';
    });
    if (!wasOpen) {
      row.classList.add('open');
      detail.classList.add('open');
      const cell = detail.querySelector('td');
      cell.innerHTML = buildDetail(sessionsById[sid]);
    }
  }

  // Wire sort headers
  document.querySelectorAll('table.sessions th[data-sort]').forEach(function(th) {
    th.addEventListener('click', function() {
      const k = th.dataset.sort;
      if (sortKey === k) sortDir = -sortDir;
      else { sortKey = k; sortDir = (k === 'date' || k === 'events' || k === 'active_min' ? -1 : 1); }
      // Update arrows
      document.querySelectorAll('table.sessions th .arrow').forEach(function(a) { a.textContent = ''; });
      th.querySelector('.arrow').textContent = sortDir > 0 ? '▲' : '▼';
      renderSessions();
    });
  });

  // Wire filters
  [filterAgent, filterProject, filterWaste, search].forEach(function(el) {
    if (el) el.addEventListener('input', renderSessions);
  });

  // Initial render
  renderSessions();

  // --- Deep-link to a specific session ---
  if (location.hash.startsWith('#session=')) {
    const sid = decodeURIComponent(location.hash.slice('#session='.length));
    showTab('sessions');
    setTimeout(function() {
      const row = tbody.querySelector('tr.row[data-sid="' + CSS.escape(sid) + '"]');
      if (row) {
        toggleDetail(row);
        row.scrollIntoView({behavior: 'smooth', block: 'center'});
      }
    }, 50);
  }
})();
"""


COMBINED_JS = r"""
(function() {
  const D = window.__AR__ || {};
  const sessions = D.sessions || [];
  const sessionsById = sessions.reduce(function(acc, s) {
    if (s && s.session_id) acc[s.session_id] = s;
    return acc;
  }, {});

  function esc(s) {
    s = String(s == null ? '' : s);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function fmt(n, suffix) {
    if (n === undefined || n === null) return '';
    return (Math.round(n * 10) / 10) + (suffix || '');
  }
  function shortSid(s) {
    if (!s) return '?';
    const i = s.indexOf(':');
    if (i < 0) return s;
    return s.slice(0, i+1) + s.slice(i+1, i+9) + '…';
  }

  // ---------- Top-level view switcher ----------
  const primaryButtons = document.querySelectorAll('nav.primary-nav button');
  const views = document.querySelectorAll('section.view');
  function showView(name) {
    primaryButtons.forEach(function(b) { b.classList.toggle('active', b.dataset.view === name); });
    views.forEach(function(v) { v.classList.toggle('active', v.dataset.view === name); });
    // Swap masthead meta line: each view has its own count/filter context
    const findingsMeta = document.querySelector('.meta-findings');
    const exploreMeta = document.querySelector('.meta-explore');
    if (findingsMeta) findingsMeta.hidden = (name !== 'findings');
    if (exploreMeta) exploreMeta.hidden = (name !== 'explore');
  }
  primaryButtons.forEach(function(b) {
    b.addEventListener('click', function() {
      showView(b.dataset.view);
      // When entering Explore, ensure a sub-tab is active
      if (b.dataset.view === 'explore') {
        const activeSub = document.querySelector('nav.tabs button.active');
        if (!activeSub) showSubTab('sessions');
      }
      writeHash(b.dataset.view);
    });
  });

  // ---------- Sub-tab switcher (inside Explore) ----------
  const subButtons = document.querySelectorAll('nav.tabs button');
  const subPanels = document.querySelectorAll('.tab-panel');
  function showSubTab(name) {
    subButtons.forEach(function(b) { b.classList.toggle('active', b.dataset.tab === name); });
    subPanels.forEach(function(p) { p.classList.toggle('active', p.dataset.panel === name); });
  }
  subButtons.forEach(function(b) {
    b.addEventListener('click', function() {
      showSubTab(b.dataset.tab);
      writeHash('explore/' + b.dataset.tab);
    });
  });

  // ---------- Hash routing ----------
  // #findings (default) | #explore[/sub] | #session=<id> (explore + sessions + open detail)
  function writeHash(h) {
    if (window.history && history.replaceState) history.replaceState(null, '', '#' + h);
  }
  function applyHash() {
    const raw = (location.hash || '#findings').slice(1);
    if (raw.startsWith('session=')) {
      const sid = decodeURIComponent(raw.slice('session='.length));
      showView('explore');
      showSubTab('sessions');
      setTimeout(function() {
        const row = document.querySelector('table.sessions tr.row[data-sid="' + CSS.escape(sid) + '"]');
        if (row) {
          if (!row.classList.contains('open')) row.click();
          row.scrollIntoView({behavior: 'smooth', block: 'center'});
        }
      }, 60);
      return;
    }
    if (raw === 'findings' || raw === '') {
      showView('findings');
      return;
    }
    if (raw === 'explore') {
      showView('explore');
      // keep current sub-tab if one is active, else default to sessions
      const activeSub = document.querySelector('nav.tabs button.active');
      if (!activeSub) showSubTab('sessions');
      return;
    }
    if (raw.startsWith('explore/')) {
      const sub = raw.slice('explore/'.length);
      showView('explore');
      if (document.querySelector('nav.tabs button[data-tab="' + sub + '"]')) showSubTab(sub);
      else showSubTab('sessions');
      return;
    }
    // Bare sub-tab name (legacy convenience): #sessions, #env, #lessons, #projects
    if (document.querySelector('nav.tabs button[data-tab="' + raw + '"]')) {
      showView('explore');
      showSubTab(raw);
      return;
    }
    showView('findings');
  }
  applyHash();
  window.addEventListener('hashchange', applyHash);

  // ---------- Findings: ref-chip drilldown (inline panel under observation) ----------
  function buildCompactSessionHTML(s) {
    if (!s) return '<p style="color:#7c7568">Session details not available.</p>';
    const tasks = (s.tasks || []).map(function(t) {
      return '<div class="task-item"><strong>' + esc(t.id || '') + '</strong> · ' +
        esc(t.intent || '') + ' <em style="color:#7c7568">(' + esc(t.type || '') +
        ' · ' + esc(t.difficulty || '') + ' · ' + esc(t.outcome || '') + ')</em></div>';
    }).join('');
    const friction = (s.friction || []).slice(0, 4).map(function(f) {
      const quote = f.quote ? '<blockquote class="quote">' + esc(f.quote) + '</blockquote>' : '';
      return '<div class="friction-item"><span class="ftype">' + esc(f.type || '?') + '</span>' +
        '<em style="color:#7c7568">(' + esc(f.tool_cat || '?') + ' · ' + (f.cost_events || 0) + 'ev)</em><br>' +
        esc(f.desc || '') + quote + '</div>';
    }).join('');
    const env = (s.env_issues || []).map(function(e) {
      return '<div class="env-item">• ' + esc(e.desc || '') +
        ' <em style="color:#7c7568">(' + (e.occurrences || 0) + ' occurrences)</em></div>';
    }).join('');
    const counterfactual = s.counterfactual ? '<h4>Counterfactual</h4><div>' + esc(s.counterfactual) + '</div>' : '';
    const lessonUser = s.lesson_user ? '<h4>Lesson for user</h4><div>' + esc(s.lesson_user) + '</div>' : '';
    const notable = s.notable ? '<h4>Notable</h4><div><em>' + esc(s.notable) + '</em></div>' : '';
    const goal = s.goal ? '<p class="dd-goal">' + esc(s.goal) + '</p>' : '';
    const fullLink = '<div style="margin-top:10px;font-size:11px"><a href="#session=' + encodeURIComponent(s.session_id) +
      '" class="session-link">view full session in Explore →</a></div>';

    return [
      '<div class="dd-head">',
      '  <span class="dd-sid">' + esc(s.session_id || '') + '</span>',
      '  <span class="dd-meta">' + esc(s.agent || '') + ' · ' + esc(s.project || '') +
            ' · ' + esc(s.date || '') + ' · ' + esc(s.primary_model || '') + '</span>',
      '</div>',
      goal,
      '<div class="stats">',
      '  <span>events: ' + (s.events || 0) + '</span>',
      '  <span>tools: ' + (s.tool_calls || 0) + '</span>',
      '  <span>active: ' + fmt(s.active_min, ' min') + '</span>',
      '  <span>bursts: ' + (s.bursts || 0) + '</span>',
      '  <span>tool err: ' + fmt((s.tool_err_rate || 0) * 100, '%') + '</span>',
      '  <span>user-caught: ' + (s.user_caught || 0) + '</span>',
      '  <span>verified: ' + esc(s.verification || '?') + '</span>',
      '</div>',
      tasks ? '<h4>Tasks</h4>' + tasks : '',
      friction ? '<h4>Top friction moments</h4>' + friction : '',
      env ? '<h4>Recurring environmental issues</h4>' + env : '',
      counterfactual,
      lessonUser,
      notable,
      fullLink,
    ].join('\n');
  }

  document.querySelectorAll('[data-obs-block]').forEach(function(block) {
    const dd = block.querySelector('.drilldown');
    block.querySelectorAll('.ref-chip').forEach(function(chip) {
      chip.addEventListener('click', function() {
        const sid = chip.getAttribute('data-sid');
        const wasActive = chip.classList.contains('active');
        block.querySelectorAll('.ref-chip.active').forEach(function(c) { c.classList.remove('active'); });
        if (wasActive) {
          dd.classList.remove('open');
          dd.innerHTML = '';
          return;
        }
        chip.classList.add('active');
        dd.innerHTML = buildCompactSessionHTML(sessionsById[sid]);
        dd.classList.add('open');
      });
    });
  });

  // ---------- Copy buttons (Findings: next-action + quick-wins) ----------
  document.querySelectorAll('[data-copy]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      const text = btn.getAttribute('data-copy');
      const orig = btn.textContent;
      navigator.clipboard && navigator.clipboard.writeText(text).then(function() {
        btn.textContent = 'COPIED';
        setTimeout(function() { btn.textContent = orig; }, 1200);
      });
    });
  });

  // ---------- Explore: sessions table sort + filter + detail ----------
  const tbody = document.querySelector('table.sessions tbody');
  if (!tbody) return;
  const filterAgent = document.getElementById('filter-agent');
  const filterProject = document.getElementById('filter-project');
  const filterWaste = document.getElementById('filter-waste');
  const search = document.getElementById('filter-search');
  const summary = document.getElementById('sessions-summary');

  let sortKey = 'date';
  let sortDir = -1;

  function rowMatches(s) {
    if (filterAgent.value && s.agent !== filterAgent.value) return false;
    if (filterProject.value && s.project !== filterProject.value) return false;
    if (filterWaste.value && s.waste_signature !== filterWaste.value) return false;
    const q = search.value.trim().toLowerCase();
    if (q) {
      const hay = (s.session_id + ' ' + (s.project||'') + ' ' + (s.goal||'') + ' ' +
                   (s.waste_signature||'') + ' ' + (s.topics||[]).join(' ')).toLowerCase();
      if (hay.indexOf(q) < 0) return false;
    }
    return true;
  }
  function sortRows(rows) {
    return rows.slice().sort(function(a, b) {
      let av = a[sortKey], bv = b[sortKey];
      if (av == null) av = '';
      if (bv == null) bv = '';
      if (typeof av === 'string') av = av.toLowerCase();
      if (typeof bv === 'string') bv = bv.toLowerCase();
      if (av < bv) return -1 * sortDir;
      if (av > bv) return 1 * sortDir;
      return 0;
    });
  }
  function renderSessions() {
    const filtered = sessions.filter(rowMatches);
    const sorted = sortRows(filtered);
    let html = '';
    sorted.forEach(function(s) {
      html += '<tr class="row" data-sid="' + esc(s.session_id) + '">' +
        '<td class="sid">' + esc(shortSid(s.session_id)) + '</td>' +
        '<td><span class="agent-pill">' + esc(s.agent) + '</span></td>' +
        '<td class="proj">' + esc(s.project || '') + '</td>' +
        '<td>' + esc(s.date || '') + '</td>' +
        '<td class="num">' + (s.events || 0) + '</td>' +
        '<td class="num">' + fmt(s.active_min) + '</td>' +
        '<td class="num">' + (s.bursts || 0) + '</td>' +
        '<td>' + (s.waste_signature && s.waste_signature !== 'none'
          ? '<span class="pill">' + esc(s.waste_signature) + '</span>' : '') + '</td>' +
        '<td class="num">' + (s.friction_count || 0) + '</td>' +
        '<td class="num">' + (s.user_caught || 0) + '</td>' +
        '<td>' + (s.outcome_signal
          ? '<span class="outcome-pill out-' + esc(s.outcome_signal) + '">' +
            esc(s.outcome_signal.replace(/_/g, ' ')) + '</span>'
          : '') + '</td>' +
        '</tr>' +
        '<tr class="detail-row" data-sid="' + esc(s.session_id) + '">' +
        '<td colspan="11"></td></tr>';
    });
    tbody.innerHTML = html;
    summary.textContent = filtered.length + ' / ' + sessions.length + ' sessions';
    tbody.querySelectorAll('tr.row').forEach(function(row) {
      row.addEventListener('click', function() { toggleDetail(row); });
    });
  }
  function buildFullDetail(s) {
    const tasks = (s.tasks || []).map(function(t) {
      return '<div class="item"><strong>' + esc(t.id || '') + '</strong> · ' +
        esc(t.intent || '') + ' <span class="small">(' + esc(t.type || '?') + ' · ' +
        esc(t.difficulty || '?') + ' · ' + esc(t.outcome || '?') + ')</span></div>';
    }).join('');
    const friction = (s.friction || []).map(function(f) {
      const quote = f.quote ? '<div class="quote">' + esc(f.quote) + '</div>' : '';
      return '<div class="item"><strong>' + esc(f.type || '?') + '</strong> ' +
        '<span class="small">(' + esc(f.tool_cat || '?') + ' · ' + (f.cost_events || 0) + 'ev · ' +
        fmt(f.cost_active_minutes, 'min') + ')</span><br>' +
        esc(f.desc || '') + quote + '</div>';
    }).join('');
    const decisions = (s.key_decisions || []).map(function(d) {
      return '<div class="item"><strong>ev' + (d.event_index||0) + '</strong> · ' + esc(d.decision || '') +
        '<div class="quote">' + esc(d.retrospective || '') + '</div></div>';
    }).join('');
    const deadends = (s.dead_ends || []).map(function(d) {
      return '<div class="item">' + esc(d.approach || '') +
        '<div class="quote">Lesson: ' + esc(d.lesson || '') + '</div></div>';
    }).join('');
    const env = (s.env_issues || []).map(function(e) {
      return '<div class="item">• ' + esc(e.desc || '') +
        ' <span class="small">(' + (e.occurrences || 0) + 'x)</span></div>';
    }).join('');
    const ucme = (s.user_caught_examples || []).map(function(e) {
      return '<div class="item">ev' + (e.event_index||0) + ': ' + esc(e.what_user_caught || '') + '</div>';
    }).join('');
    const counterfactual = s.counterfactual ? '<h4>Counterfactual</h4><div class="item">' + esc(s.counterfactual) + '</div>' : '';
    const lessonUser = s.lesson_user ? '<h4>Lesson for user</h4><div class="item">' + esc(s.lesson_user) + '</div>' : '';
    const lessonAgent = s.lesson_agent ? '<h4>Lesson for agent</h4><div class="item">' + esc(s.lesson_agent) + '</div>' : '';
    const notable = s.notable ? '<h4>Notable</h4><div class="item"><em>' + esc(s.notable) + '</em></div>' : '';
    const topics = (s.topics || []).length
      ? '<h4>Topics</h4><div class="item">' + (s.topics || []).map(esc).join(' · ') + '</div>'
      : '';

    const stats = [
      ['session_id', s.session_id], ['agent', s.agent], ['project', s.project],
      ['date', s.date], ['weekday', s.weekday], ['model', s.primary_model],
      ['events', s.events], ['tool_calls', s.tool_calls], ['user_turns', s.user_turn_count],
      ['subagents', s.subagent_count], ['active_min', fmt(s.active_min)],
      ['wall_clock_min', fmt(s.wall_clock_min)], ['bursts', s.bursts],
      ['primary_burst_min', fmt(s.primary_burst_min)],
      ['tool_err_rate', fmt((s.tool_err_rate||0)*100, '%')],
      ['unique_files', s.unique_files], ['tests_invoked', s.tests_invoked],
      ['user_corrections', s.user_corrections], ['user_caught_errors', s.user_caught],
      ['verification', s.verification], ['waste_signature', s.waste_signature],
      ['narrative_quality', s.narrative_quality],
    ];
    const statsHtml = stats.map(function(kv) {
      return '<div class="row"><span class="lab">' + esc(kv[0]) + '</span>' +
        '<span class="val">' + esc(kv[1] == null ? '—' : kv[1]) + '</span></div>';
    }).join('');

    return '<div class="session-detail">' +
      '<h3>' + esc(s.goal || 'Session') + '</h3>' +
      '<div class="sd-meta">' + esc(s.session_id) + '</div>' +
      '<div class="grid-2"><div>' +
      (tasks ? '<h4>Tasks</h4>' + tasks : '') +
      (friction ? '<h4>Friction moments</h4>' + friction : '') +
      (decisions ? '<h4>Key decisions</h4>' + decisions : '') +
      (deadends ? '<h4>Dead ends</h4>' + deadends : '') +
      (env ? '<h4>Recurring environmental issues</h4>' + env : '') +
      (ucme ? '<h4>User-caught model errors</h4>' + ucme : '') +
      counterfactual + lessonUser + lessonAgent + notable + topics +
      '</div><div><div class="stats-card">' + statsHtml + '</div></div></div></div>';
  }
  function toggleDetail(row) {
    const sid = row.dataset.sid;
    const detail = tbody.querySelector('tr.detail-row[data-sid="' + CSS.escape(sid) + '"]');
    if (!detail) return;
    const wasOpen = row.classList.contains('open');
    tbody.querySelectorAll('tr.row.open').forEach(function(r) { r.classList.remove('open'); });
    tbody.querySelectorAll('tr.detail-row.open').forEach(function(r) {
      r.classList.remove('open');
      r.querySelector('td').innerHTML = '';
    });
    if (!wasOpen) {
      row.classList.add('open');
      detail.classList.add('open');
      detail.querySelector('td').innerHTML = buildFullDetail(sessionsById[sid]);
    }
  }
  document.querySelectorAll('table.sessions th[data-sort]').forEach(function(th) {
    th.addEventListener('click', function() {
      const k = th.dataset.sort;
      if (sortKey === k) sortDir = -sortDir;
      else { sortKey = k; sortDir = (k === 'date' || k === 'events' || k === 'active_min' ? -1 : 1); }
      document.querySelectorAll('table.sessions th .arrow').forEach(function(a) { a.textContent = ''; });
      th.querySelector('.arrow').textContent = sortDir > 0 ? '▲' : '▼';
      renderSessions();
    });
  });
  [filterAgent, filterProject, filterWaste, search].forEach(function(el) {
    if (el) el.addEventListener('input', renderSessions);
  });
  renderSessions();

  // Re-trigger hash-based session detail open after table rendered
  if (location.hash.startsWith('#session=')) applyHash();

  // ---------- Inline rating ----------
  const RUN_SLUG = (D.run_slug) || 'latest';
  const STORAGE_KEY = 'tessera-ratings::' + RUN_SLUG;

  function loadRatings() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    } catch (e) { return {}; }
  }
  function saveRatings(state) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) {}
  }
  function refreshSync() {
    const state = loadRatings();
    const n = Object.keys(state).length;
    const sync = document.getElementById('rate-sync');
    if (!sync) return;
    sync.hidden = (n === 0);
    sync.querySelector('.count').textContent = n;
  }

  // Wire each rate row
  document.querySelectorAll('[data-rate-row]').forEach(function(row) {
    const idx = parseInt(row.dataset.obsIndex, 10);
    const key = row.dataset.obsKey;
    const title = row.dataset.obsTitle;
    const state = loadRatings();
    const existing = state[key];
    if (existing) {
      const btn = row.querySelector('button[data-rate="' + existing.rating + '"]');
      if (btn) btn.classList.add('active', existing.rating);
    }
    row.querySelectorAll('button[data-rate]').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const rating = btn.dataset.rate;
        const state = loadRatings();
        // Toggle off if already this rating
        if (state[key] && state[key].rating === rating) {
          delete state[key];
          row.querySelectorAll('button.active').forEach(function(b) {
            b.classList.remove('active', 'useful', 'wrong', 'known', 'skip');
          });
        } else {
          state[key] = { index: idx, key: key, title: title, rating: rating };
          row.querySelectorAll('button.active').forEach(function(b) {
            b.classList.remove('active', 'useful', 'wrong', 'known', 'skip');
          });
          btn.classList.add('active', rating);
        }
        saveRatings(state);
        refreshSync();
      });
    });
  });

  // Floating sync button + modal
  const sync = document.getElementById('rate-sync');
  const modal = document.getElementById('rate-modal');
  if (sync && modal) {
    sync.addEventListener('click', function() {
      const state = loadRatings();
      const ratings = Object.values(state);
      const payload = JSON.stringify({ slug: RUN_SLUG, ratings: ratings }, null, 2);
      const cmd = "tessera rate-import <<'EOF'\n" + payload + "\nEOF";
      modal.querySelector('pre').textContent = cmd;
      modal.hidden = false;
    });
    modal.querySelector('[data-close]').addEventListener('click', function() {
      modal.hidden = true;
    });
    modal.querySelector('[data-copy-cmd]').addEventListener('click', function() {
      const text = modal.querySelector('pre').textContent;
      navigator.clipboard && navigator.clipboard.writeText(text).then(function() {
        const btn = modal.querySelector('[data-copy-cmd]');
        const orig = btn.textContent;
        btn.textContent = 'COPIED — paste in terminal';
        setTimeout(function() { btn.textContent = orig; }, 1800);
      });
    });
    modal.addEventListener('click', function(e) {
      if (e.target === modal) modal.hidden = true;
    });
  }
  refreshSync();
})();
"""


def _expanded_session_for_explore(n: dict) -> dict:
    """Richer than the synthesis drilldown — surfaces all per-session fields
    needed for the full session detail panel + sessions table sort/filter.
    """
    tasks = []
    for t in n.get("tasks") or []:
        tasks.append(
            {
                "id": t.get("task_id"),
                "intent": (t.get("intent") or "")[:300],
                "type": t.get("task_type"),
                "outcome": t.get("outcome"),
                "difficulty": (t.get("task_difficulty") or {}).get("overall"),
            }
        )
    friction = []
    for fm in n.get("friction_moments") or []:
        friction.append(
            {
                "type": fm.get("type"),
                "tool_cat": fm.get("tool_category"),
                "cost_events": fm.get("cost_events"),
                "cost_active_minutes": fm.get("cost_active_minutes"),
                "desc": (fm.get("description") or "")[:300],
                "quote": (fm.get("key_quote") or "")[:240],
            }
        )
    key_decisions = []
    for kd in n.get("key_decisions") or []:
        key_decisions.append(
            {
                "event_index": kd.get("event_index"),
                "decision": (kd.get("decision") or "")[:240],
                "retrospective": (kd.get("retrospective") or "")[:300],
            }
        )
    dead_ends = []
    for de in n.get("dead_ends") or []:
        dead_ends.append(
            {
                "approach": (de.get("approach") or "")[:240],
                "lesson": (de.get("lesson") or "")[:300],
            }
        )
    env = []
    for ei in n.get("recurring_environmental_issues") or []:
        env.append(
            {
                "desc": (ei.get("description") or "")[:300],
                "occurrences": len(ei.get("occurrences") or []),
            }
        )
    ucme = n.get("user_caught_model_errors") or {}
    ucme_examples = []
    for ex in (ucme.get("examples") or [])[:3]:
        ucme_examples.append(
            {
                "event_index": ex.get("event_index"),
                "what_user_caught": (ex.get("what_user_caught") or "")[:240],
            }
        )
    return {
        "session_id": n.get("session_id"),
        "agent": n.get("agent"),
        "project": n.get("project_label"),
        "date": (n.get("started_at") or "")[:10],
        "weekday": n.get("weekday"),
        "primary_model": n.get("primary_model"),
        "events": n.get("event_count"),
        "tool_calls": n.get("tool_call_count"),
        "user_turn_count": n.get("user_turn_count"),
        "subagent_count": n.get("subagent_count"),
        "active_min": n.get("active_minutes"),
        "wall_clock_min": n.get("wall_clock_minutes"),
        "bursts": n.get("bursts"),
        "primary_burst_min": n.get("primary_burst_minutes"),
        "tool_err_rate": n.get("tool_error_rate"),
        "user_corrections": (n.get("user_friction_signals") or {}).get(
            "explicit_corrections"
        ),
        "user_caught": ucme.get("count", 0),
        "user_caught_examples": ucme_examples,
        "verification": n.get("verification_completeness"),
        "unique_files": n.get("unique_files_touched"),
        "tests_invoked": n.get("tests_invoked"),
        "narrative_quality": n.get("narrative_quality"),
        "goal": (n.get("goal") or "")[:300],
        "waste_signature": n.get("waste_signature"),
        "topics": (n.get("topics") or [])[:6],
        "tasks": tasks,
        "friction": friction,
        "friction_count": len(friction),
        "key_decisions": key_decisions,
        "dead_ends": dead_ends,
        "env_issues": env,
        "counterfactual": (n.get("counterfactual") or "")[:400],
        "notable": (n.get("notable") or "")[:300],
        "lesson_user": (n.get("lesson_for_user") or "")[:240],
        "lesson_agent": (n.get("lesson_for_agent") or "")[:240],
        "outcome_signal": (n.get("outcome") or {}).get("outcome_signal"),
        "outcome": _compact_outcome_for_session(n.get("outcome") or {}),
    }


def _compact_outcome_for_session(outcome: dict) -> dict | None:
    """Subset of the outcome dict suitable for embedding in the per-session
    JSON payload that ships to the browser. Drops verbose churn detail."""
    if not outcome.get("outcome_signal"):
        return None
    out: dict = {"signal": outcome["outcome_signal"]}
    churn = outcome.get("files_churn") or {}
    if churn.get("commits_touching_files"):
        out["churn"] = {
            "commits_in_14d": churn.get("commits_touching_files"),
            "fixup": churn.get("fixup_shape_commits", 0),
            "revert": churn.get("revert_commits", 0),
        }
    if outcome.get("prs"):
        out["prs"] = [
            {
                "n": p.get("number"),
                "state": p.get("state"),
                "ci": p.get("ci_status"),
                "review": p.get("review_decision"),
            }
            for p in outcome["prs"][:3]
        ]
    branch = outcome.get("branch") or {}
    if branch.get("merged_into") or branch.get("commits_after_session"):
        out["branch"] = {
            "merged_into": branch.get("merged_into"),
            "commits_after": branch.get("commits_after_session"),
        }
    trunk = outcome.get("trunk_commits") or {}
    if trunk.get("trunk_commits_in_window"):
        out["trunk"] = {
            "ref": trunk.get("ref"),
            "commits_in_window": trunk.get("trunk_commits_in_window"),
            "subjects": trunk.get("trunk_commit_subjects", [])[:3],
        }
    return out


def _aggregate_env_issues(narratives: list[dict]) -> list[dict]:
    """Produce a flat list of all env issues across all sessions, sorted by
    occurrence count then by session count. Each item includes the source
    session ids so users can drill back."""
    items = []
    for n in narratives:
        sid = n.get("session_id")
        proj = n.get("project_label")
        for ei in n.get("recurring_environmental_issues") or []:
            items.append(
                {
                    "session_id": sid,
                    "project": proj,
                    "desc": ei.get("description") or "",
                    "occurrences": len(ei.get("occurrences") or []),
                }
            )
    items.sort(key=lambda x: (-x["occurrences"], x["project"] or ""))
    return items


def _aggregate_lessons(narratives: list[dict]) -> dict:
    counterfactuals = []
    for n in narratives:
        cf = (n.get("counterfactual") or "").strip()
        if cf:
            counterfactuals.append(
                {
                    "text": cf,
                    "session_id": n.get("session_id"),
                    "project": n.get("project_label"),
                }
            )
    lessons_user = []
    for n in narratives:
        lu = (n.get("lesson_for_user") or "").strip()
        if lu:
            lessons_user.append(
                {
                    "text": lu,
                    "session_id": n.get("session_id"),
                    "project": n.get("project_label"),
                }
            )
    lessons_agent = []
    for n in narratives:
        la = (n.get("lesson_for_agent") or "").strip()
        if la:
            lessons_agent.append(
                {
                    "text": la,
                    "session_id": n.get("session_id"),
                    "project": n.get("project_label"),
                }
            )
    return {
        "counterfactuals": counterfactuals,
        "lessons_user": lessons_user,
        "lessons_agent": lessons_agent,
    }


def _per_project_deep(narratives: list[dict]) -> list[dict]:
    """Group sessions by project_label, return rich per-project info for
    the Projects tab. Sorted by session count desc."""
    from collections import defaultdict, Counter

    by_proj: dict[str, list[dict]] = defaultdict(list)
    for n in narratives:
        by_proj[n.get("project_label") or "(unknown)"].append(n)
    out = []
    for proj, items in by_proj.items():
        waste_counter = Counter(i.get("waste_signature") for i in items)
        agent_counter = Counter(i.get("agent") for i in items)
        total_active = sum(i.get("active_minutes") or 0 for i in items)
        total_events = sum(i.get("event_count") or 0 for i in items)
        env_total = sum(
            len(i.get("recurring_environmental_issues") or []) for i in items
        )
        ucme_total = sum(
            (i.get("user_caught_model_errors") or {}).get("count", 0) for i in items
        )
        sessions_summary = [
            {
                "session_id": i.get("session_id"),
                "date": (i.get("started_at") or "")[:10],
                "events": i.get("event_count"),
                "active_min": i.get("active_minutes"),
                "waste": i.get("waste_signature"),
                "goal": (i.get("goal") or "")[:120],
            }
            for i in sorted(
                items, key=lambda x: x.get("started_at") or "", reverse=True
            )
        ]
        out.append(
            {
                "project": proj,
                "session_count": len(items),
                "agents": dict(agent_counter),
                "waste_distribution": dict(waste_counter),
                "active_minutes": round(total_active, 1),
                "events": total_events,
                "env_total": env_total,
                "user_caught_total": ucme_total,
                "sessions": sessions_summary,
            }
        )
    out.sort(key=lambda x: -x["session_count"])
    return out


def _render_sessions_tab(narratives: list[dict]) -> str:
    # filter dropdown options
    agents = sorted({n.get("agent") for n in narratives if n.get("agent")})
    projects = sorted(
        {n.get("project_label") for n in narratives if n.get("project_label")}
    )
    wastes = sorted(
        {
            n.get("waste_signature")
            for n in narratives
            if n.get("waste_signature")
        }
    )

    def _opt(values: list[str], label: str) -> str:
        opts = '<option value="">all ' + label + "</option>"
        for v in values:
            opts += f'<option value="{_esc(v)}">{_esc(v)}</option>'
        return opts

    th = (
        '<th data-sort="session_id">Session</th>'
        '<th data-sort="agent">Agent</th>'
        '<th data-sort="project">Project</th>'
        '<th data-sort="date">Date <span class="arrow">▼</span></th>'
        '<th data-sort="events">Events</th>'
        '<th data-sort="active_min">Active min</th>'
        '<th data-sort="bursts">Bursts</th>'
        '<th data-sort="waste_signature">Waste</th>'
        '<th data-sort="friction_count">Friction</th>'
        '<th data-sort="user_caught">User caught</th>'
        '<th data-sort="outcome_signal">Outcome</th>'
    )

    return f"""
<section class="tab-panel" data-panel="sessions">
  <div class="toolbar">
    <input type="search" id="filter-search" placeholder="search session id, project, goal, topic…">
    <select id="filter-agent">{_opt(list(agents), "agents")}</select>
    <select id="filter-project">{_opt(list(projects), "projects")}</select>
    <select id="filter-waste">{_opt(list(wastes), "waste signatures")}</select>
    <span class="summary" id="sessions-summary"></span>
  </div>
  <table class="sessions">
    <thead><tr>{th}</tr></thead>
    <tbody></tbody>
  </table>
</section>
"""


def _render_env_tab(narratives: list[dict]) -> str:
    env_items = _aggregate_env_issues(narratives)
    if not env_items:
        return (
            '<section class="tab-panel" data-panel="env">'
            '<p style="color:var(--ink-mute)">No recurring environmental issues found.</p>'
            "</section>"
        )
    cards = []
    for item in env_items:
        cards.append(
            '<div class="env-card">'
            f'<div class="env-head"><span>{_esc(item["project"] or "?")}</span>'
            f'<span>{item["occurrences"]}× in session</span></div>'
            f'<div class="env-text">{_esc(item["desc"])}</div>'
            '<div class="env-foot">From: '
            f'<a href="#session={_esc(item["session_id"])}" class="session-link">{_esc(item["session_id"])}</a>'
            "</div></div>"
        )
    return (
        '<section class="tab-panel" data-panel="env">'
        '<div class="toolbar"><span class="summary">'
        f'{len(env_items)} env issues across {len({i["session_id"] for i in env_items})} sessions'
        "</span></div>"
        '<div class="env-grid">' + "".join(cards) + "</div>"
        "</section>"
    )


def _render_lessons_tab(narratives: list[dict]) -> str:
    L = _aggregate_lessons(narratives)

    def col(title: str, items: list[dict]) -> str:
        cards = []
        for it in items:
            cards.append(
                '<div class="lesson-card">'
                f'<div class="l-text">{_esc(it["text"])}</div>'
                '<div class="l-foot">'
                f'<a href="#session={_esc(it["session_id"])}" class="session-link">{_esc(it["session_id"])}</a>'
                f' · {_esc(it["project"] or "?")}'
                "</div></div>"
            )
        return (
            '<div class="lesson-col">'
            f"<h3>{_esc(title)} ({len(items)})</h3>"
            + "".join(cards)
            + "</div>"
        )

    return (
        '<section class="tab-panel" data-panel="lessons">'
        '<div class="lesson-grid">'
        + col("Counterfactuals", L["counterfactuals"])
        + col("Lessons for user", L["lessons_user"])
        + col("Lessons for agent", L["lessons_agent"])
        + "</div></section>"
    )


def _render_projects_tab(narratives: list[dict]) -> str:
    pps = _per_project_deep(narratives)
    if not pps:
        return (
            '<section class="tab-panel" data-panel="projects">'
            '<p style="color:var(--ink-mute)">No projects.</p></section>'
        )
    cards = []
    for pp in pps:
        sessions_html = "".join(
            (
                '<div class="session-line">'
                f'<a href="#session={_esc(s["session_id"])}" class="session-link">{_esc(s["session_id"])}</a>'
                f"<span>{_esc(s['goal'])}</span>"
                f'<span class="when">{_esc(s["date"])} · {s["events"] or 0}ev · {_esc(s["waste"] or "—")}</span>'
                "</div>"
            )
            for s in pp["sessions"]
        )
        meta_bits = []
        if pp["agents"]:
            meta_bits.append(
                " · ".join(f"{c} {a}" for a, c in pp["agents"].items())
            )
        meta_bits.append(f"{int(pp['active_minutes']):,} active min")
        meta_bits.append(f"{pp['env_total']} env issues")
        meta_bits.append(f"{pp['user_caught_total']} user-caught errors")
        cards.append(
            '<div class="proj-deep">'
            '<div class="pd-head">'
            f'<span class="pd-name">{_esc(pp["project"])}</span>'
            f'<span class="pd-meta">{_esc(" · ".join(meta_bits))}</span>'
            "</div>"
            f'<div class="pd-headline">{pp["session_count"]} sessions in this project</div>'
            "<h4>Sessions (newest first)</h4>"
            f"{sessions_html}"
            "</div>"
        )
    return (
        '<section class="tab-panel" data-panel="projects">'
        + "".join(cards)
        + "</section>"
    )


def render_explore(synthesis: dict, narratives: list[dict]) -> str:
    """Build the Explore page — browse all the rich data."""
    meta = synthesis.get("meta", {})
    n_sessions = len(narratives)
    n_env = sum(
        len(n.get("recurring_environmental_issues") or []) for n in narratives
    )
    n_lessons = sum(
        1 for n in narratives if (n.get("lesson_for_user") or "").strip()
    ) + sum(1 for n in narratives if (n.get("counterfactual") or "").strip())
    n_projects = len({n.get("project_label") for n in narratives if n.get("project_label")})

    embed = {
        "sessions": [_expanded_session_for_explore(n) for n in narratives],
    }

    masthead_meta = " · ".join(
        bit
        for bit in [
            f"{n_sessions} sessions",
            (meta.get("generated_at") or "")[:10],
            meta.get("model") or "",
        ]
        if bit
    )

    nav_html = (
        '<nav class="tabs">'
        f'<button data-tab="sessions">Sessions <span class="count">({n_sessions})</span></button>'
        f'<button data-tab="env">Env issues <span class="count">({n_env})</span></button>'
        f'<button data-tab="lessons">Lessons <span class="count">({n_lessons})</span></button>'
        f'<button data-tab="projects">Projects <span class="count">({n_projects})</span></button>'
        "</nav>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tessera — Explore</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}{EXPLORE_CSS}</style>
</head>
<body>
<div class="frame">
  <header class="masthead">
    <div class="brand">Tessera · Explore</div>
    <div class="meta">{_esc(masthead_meta)} · <a href="synthesis.html" class="cross-page-link">view synthesis →</a></div>
  </header>

  {nav_html}

  {_render_sessions_tab(narratives)}
  {_render_env_tab(narratives)}
  {_render_lessons_tab(narratives)}
  {_render_projects_tab(narratives)}

  <footer class="colophon">
    <span>tessera · explore</span>
    <span>schema v1 · {_esc(meta.get("model") or "")}</span>
  </footer>
</div>

<script>window.__EXPLORE_DATA__ = {_safe_inline_json(embed)};</script>
<script>{EXPLORE_JS}</script>
</body>
</html>
"""
