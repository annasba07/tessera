"""Renderers for the synthesis output.

`render_synthesis_text` produces a terminal-friendly summary with section
headers and underlines. `render_synthesis_markdown` produces a shareable
doc-style writeup suitable for committing to a repo or pasting into Slack.
"""

from __future__ import annotations


def _short_session(sid: str, n: int = 12) -> str:
    """Trim a session_id to its leading agent + uuid prefix."""
    if not sid:
        return "?"
    if ":" not in sid:
        return sid[:n] + "…" if len(sid) > n else sid
    agent, rest = sid.split(":", 1)
    return f"{agent}:{rest[:n]}…" if len(rest) > n else sid


# ---------- Text -----------------------------------------------------------


def render_synthesis_text(syn: dict) -> str:
    lines: list[str] = []
    meta = syn.get("meta", {})

    lines.append("")
    lines.append("AGENT REFLECT — CROSS-SESSION SYNTHESIS")
    lines.append("=" * 70)

    parts = []
    if meta.get("input_sessions"):
        parts.append(f"{meta['input_sessions']} sessions")
    if meta.get("project_filter"):
        parts.append(f"project filter: {meta['project_filter']}")
    if meta.get("model"):
        parts.append(meta["model"])
    if meta.get("had_prior_context"):
        parts.append("with prior-run context")
    if parts:
        lines.append(" · ".join(parts))
    lines.append("")

    headline = syn.get("headline")
    if headline:
        lines.append(f"HEADLINE: {headline}")
        lines.append("")

    one_thing = syn.get("if_you_do_one_thing_this_week")
    if one_thing:
        lines.append(f"IF YOU DO ONE THING: {one_thing}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    obs_list = syn.get("observations") or []
    if obs_list:
        lines.append(f"OBSERVATIONS ({len(obs_list)})")
        lines.append("")
        for i, obs in enumerate(obs_list, start=1):
            tags = []
            if obs.get("confidence"):
                tags.append(obs["confidence"])
            if obs.get("category"):
                tags.append(obs["category"])
            if obs.get("trend") and obs["trend"] != "new":
                tags.append(f"trend: {obs['trend']}")
            if obs.get("continues"):
                tags.append(f"continues {obs['continues']}")
            tag_str = f"  [{' · '.join(tags)}]" if tags else ""
            lines.append(f"§{i}  {obs.get('title', 'Observation')}{tag_str}")
            lines.append("")
            lines.append(f"  {obs.get('claim', '')}")
            evidence_refs = obs.get("evidence_refs") or []
            evidence_sids = obs.get("evidence_sessions") or []
            n_supp = obs.get("supporting_count") or len(evidence_refs)
            if evidence_refs:
                lines.append("")
                lines.append(f"  Evidence ({n_supp} sessions):")
                shown = list(zip(evidence_refs, evidence_sids))[:6]
                for ref, sid in shown:
                    lines.append(f"    - {ref}  {_short_session(sid)}")
                if len(evidence_refs) > 6:
                    lines.append(f"    … and {len(evidence_refs) - 6} more")
            if obs.get("interpretation"):
                lines.append("")
                lines.append(f"  Why: {obs['interpretation']}")
            if obs.get("next_action"):
                lines.append("")
                lines.append(f"  Next: {obs['next_action']}")
            lines.append("")
            lines.append("-" * 70)
            lines.append("")

    qw_list = syn.get("quick_wins") or []
    if qw_list:
        lines.append("")
        lines.append("QUICK WINS (one-time fixes)")
        lines.append("")
        for qw in qw_list:
            count = qw.get("affected_sessions", "?")
            lines.append(f"  • [{count} sessions] {qw.get('fix', '')}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    sc_list = syn.get("skill_candidates") or []
    if sc_list:
        lines.append("")
        lines.append("SKILL CANDIDATES (recurring work worth codifying)")
        lines.append("")
        for sc in sc_list:
            count = sc.get("affected_sessions", "?")
            kind = sc.get("kind", "new_skill")
            tag = "NEW" if kind == "new_skill" else f"DEEPEN [{sc.get('existing_skill_hint','?')}]"
            lines.append(f"  • [{tag} · {count} sessions] {sc.get('title','')}")
            if sc.get("trigger_pattern"):
                lines.append(f"      trigger: {sc['trigger_pattern']}")
            if sc.get("what_it_should_do"):
                lines.append(f"      should: {sc['what_it_should_do']}")
            lines.append("")
        lines.append("-" * 70)
        lines.append("")

    pp_list = syn.get("per_project") or []
    if pp_list:
        lines.append("")
        lines.append("PER-PROJECT")
        lines.append("")
        for pp in pp_list:
            lines.append(
                f"  {pp.get('project', '?')} ({pp.get('session_count', '?')} sessions)"
            )
            if pp.get("headline"):
                lines.append(f"    headline: {pp['headline']}")
            if pp.get("biggest_friction"):
                lines.append(f"    biggest friction: {pp['biggest_friction']}")
            lines.append("")

    if meta.get("notes"):
        lines.append("")
        lines.append(f"Notes: {meta['notes']}")

    if meta.get("fabricated_ref_count"):
        lines.append(
            f"(Validator dropped {meta['fabricated_ref_count']} fabricated ref"
            f"{'s' if meta['fabricated_ref_count'] != 1 else ''}.)"
        )

    return "\n".join(lines)


# ---------- Markdown -------------------------------------------------------


def render_synthesis_markdown(syn: dict) -> str:
    parts: list[str] = []
    meta = syn.get("meta", {})

    parts.append("# Agent reflect — cross-session synthesis")
    parts.append("")
    sub = []
    if meta.get("input_sessions"):
        sub.append(f"**{meta['input_sessions']} sessions**")
    if meta.get("project_filter"):
        sub.append(f"project filter: `{meta['project_filter']}`")
    if meta.get("model"):
        sub.append(f"model: `{meta['model']}`")
    if meta.get("generated_at"):
        sub.append(f"generated: {meta['generated_at'][:19]}Z")
    if sub:
        parts.append(" · ".join(sub))
        parts.append("")

    if syn.get("headline"):
        parts.append("## Headline")
        parts.append("")
        parts.append(f"> {syn['headline']}")
        parts.append("")

    if syn.get("if_you_do_one_thing_this_week"):
        parts.append("## If you do one thing this week")
        parts.append("")
        parts.append(syn["if_you_do_one_thing_this_week"])
        parts.append("")

    obs_list = syn.get("observations") or []
    if obs_list:
        parts.append(f"## Observations ({len(obs_list)})")
        parts.append("")
        for i, obs in enumerate(obs_list, start=1):
            tags = []
            if obs.get("confidence"):
                tags.append(obs["confidence"])
            if obs.get("category"):
                tags.append(obs["category"])
            if obs.get("trend") and obs["trend"] != "new":
                tags.append(f"trend: {obs['trend']}")
            tag_str = f" — *{' · '.join(tags)}*" if tags else ""
            parts.append(f"### §{i} {obs.get('title', 'Observation')}{tag_str}")
            parts.append("")
            parts.append(obs.get("claim", ""))
            parts.append("")
            evidence_refs = obs.get("evidence_refs") or []
            evidence_sids = obs.get("evidence_sessions") or []
            n_supp = obs.get("supporting_count") or len(evidence_refs)
            if evidence_refs:
                parts.append(f"**Evidence** ({n_supp} sessions):")
                parts.append("")
                shown = list(zip(evidence_refs, evidence_sids))[:8]
                for ref, sid in shown:
                    parts.append(f"- `{ref}` — `{sid}`")
                if len(evidence_refs) > 8:
                    parts.append(f"- *…and {len(evidence_refs) - 8} more*")
                parts.append("")
            if obs.get("interpretation"):
                parts.append(f"**Why**: {obs['interpretation']}")
                parts.append("")
            if obs.get("next_action"):
                parts.append(f"**Next**: {obs['next_action']}")
                parts.append("")

    qw_list = syn.get("quick_wins") or []
    if qw_list:
        parts.append("## Quick wins")
        parts.append("")
        parts.append("One-time fixes that compound across future sessions.")
        parts.append("")
        for qw in qw_list:
            count = qw.get("affected_sessions", "?")
            parts.append(f"- **[{count} sessions]** {qw.get('fix', '')}")
        parts.append("")

    sc_list = syn.get("skill_candidates") or []
    if sc_list:
        parts.append("## Skill candidates")
        parts.append("")
        parts.append(
            "Recurring agent work the data suggests would be better as a "
            "codified skill — either a new one or a deeper version of one "
            "you already have."
        )
        parts.append("")
        for sc in sc_list:
            count = sc.get("affected_sessions", "?")
            kind = sc.get("kind", "new_skill")
            tag = "NEW" if kind == "new_skill" else f"DEEPEN `{sc.get('existing_skill_hint','?')}`"
            parts.append(f"### [{tag} · {count} sessions] {sc.get('title','')}")
            parts.append("")
            if sc.get("trigger_pattern"):
                parts.append(f"**Trigger**: {sc['trigger_pattern']}")
                parts.append("")
            if sc.get("what_it_should_do"):
                parts.append(f"**Should do**: {sc['what_it_should_do']}")
                parts.append("")

    pp_list = syn.get("per_project") or []
    if pp_list:
        parts.append("## Per-project")
        parts.append("")
        for pp in pp_list:
            parts.append(
                f"### {pp.get('project', '?')} — {pp.get('session_count', '?')} sessions"
            )
            parts.append("")
            if pp.get("headline"):
                parts.append(pp["headline"])
                parts.append("")
            if pp.get("biggest_friction"):
                parts.append(f"**Biggest friction**: {pp['biggest_friction']}")
                parts.append("")

    if meta.get("notes"):
        parts.append("---")
        parts.append("")
        parts.append("## Notes")
        parts.append("")
        parts.append(meta["notes"])
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"
