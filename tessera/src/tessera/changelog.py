"""Cross-run changelog — what changed since last week.

The synthesis output already captures *this run's* picture. The persona
review surfaced a real gap: every Monday you'd see roughly the same
observations in roughly the same order, with no answer to "what's
different from last week?" That makes the dashboard a one-time read
instead of a recurring ritual.

This module computes a deterministic diff between the current synthesis
run and the most recent prior run in history, classifying each pattern
into one of:

  * `new`         — wasn't in the prior run at all
  * `escalating`  — present in both, evidence count went UP
  * `continuing`  — present in both, evidence count roughly stable
  * `improving`   — present in both, evidence count went DOWN
  * `resolved`    — was in prior run, missing from current
  * `regressed`   — appears in current AND in some run >1 ago, but not
                    the immediately-prior run (a previously-resolved
                    pattern coming back)

Cross-run identity uses the same `_observation_key` (sha1 of title +
claim/pattern) that the rating + experiment systems already key on, so
a pattern carries the same identity across weeks even if its
evidence_refs change.

The output is a pure-data dict; renderers (dashboard, CLI) consume it.
"""

from __future__ import annotations

from typing import Any

from .history import HistoryStore, _observation_key, _now_iso


# A "stable" change in evidence count — anything within this band counts as
# "continuing", not improving/escalating. Avoids reporting +1 / -1 noise.
EVIDENCE_DELTA_NOISE = 1


def _index_by_key(items: list[dict]) -> dict[str, dict]:
    """Map observation_key → item, dropping items without a key."""
    out: dict[str, dict] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = _observation_key(item)
        if key:
            out[key] = item
    return out


def _classify_delta(prev_count: int, curr_count: int) -> str:
    """Return 'escalating' | 'continuing' | 'improving' based on count delta."""
    delta = curr_count - prev_count
    if delta > EVIDENCE_DELTA_NOISE:
        return "escalating"
    if delta < -EVIDENCE_DELTA_NOISE:
        return "improving"
    return "continuing"


def _diff_section(
    current_items: list[dict],
    prior_items: list[dict],
    older_items: list[dict] | None = None,
) -> dict[str, list[dict]]:
    """Diff one section (observations or behavioral_patterns) between runs.

    `older_items` is the union of items from runs older than `prior` —
    used to detect `regressed` (was-resolved, now-back) patterns.
    """
    current_by_key = _index_by_key(current_items)
    prior_by_key = _index_by_key(prior_items)
    older_by_key = _index_by_key(older_items or [])

    buckets: dict[str, list[dict]] = {
        "new": [],
        "escalating": [],
        "continuing": [],
        "improving": [],
        "resolved": [],
        "regressed": [],
    }

    for key, item in current_by_key.items():
        title = item.get("title") or ""
        curr_count = int(item.get("supporting_count") or 0)
        category = item.get("category") or item.get("dimension") or ""
        if key in prior_by_key:
            prev_count = int(prior_by_key[key].get("supporting_count") or 0)
            classification = _classify_delta(prev_count, curr_count)
            entry = {
                "key": key,
                "title": title,
                "category": category,
                "previous_count": prev_count,
                "current_count": curr_count,
                "delta": curr_count - prev_count,
            }
            buckets[classification].append(entry)
        elif key in older_by_key:
            # Was around in some older run, then resolved in the prior run,
            # now back. That's a regression worth flagging.
            buckets["regressed"].append(
                {
                    "key": key,
                    "title": title,
                    "category": category,
                    "current_count": curr_count,
                    "prior_run_resolved": True,
                }
            )
        else:
            buckets["new"].append(
                {
                    "key": key,
                    "title": title,
                    "category": category,
                    "current_count": curr_count,
                }
            )

    # Resolved: in prior, gone from current
    for key, item in prior_by_key.items():
        if key in current_by_key:
            continue
        buckets["resolved"].append(
            {
                "key": key,
                "title": item.get("title") or "",
                "category": item.get("category") or item.get("dimension") or "",
                "previous_count": int(item.get("supporting_count") or 0),
            }
        )

    return buckets


def compare_runs(
    current: dict,
    prior: dict | None,
    older: list[dict] | None = None,
) -> dict[str, Any]:
    """Diff `current` synthesis against the most recent `prior` synthesis.

    `older` is an optional list of further-back runs used to detect
    regressions (a pattern that was resolved one run ago but appears now
    AND was present in some run further back).

    Returns a dict shaped for both dashboard rendering and CLI text output.
    """
    if not prior:
        return {
            "compared": False,
            "reason": "no prior run in history yet — first run establishes the baseline.",
            "compared_against_slug": None,
            "compared_against_date": None,
            "observations": {b: [] for b in ("new", "escalating", "continuing", "improving", "resolved", "regressed")},
            "behavioral_patterns": {b: [] for b in ("new", "escalating", "continuing", "improving", "resolved", "regressed")},
            "summary": {},
        }

    older_obs: list[dict] = []
    older_bp: list[dict] = []
    for r in older or []:
        older_obs.extend(r.get("observations") or [])
        older_bp.extend(r.get("behavioral_patterns") or [])

    obs_diff = _diff_section(
        current.get("observations") or [],
        prior.get("observations") or [],
        older_obs,
    )
    bp_diff = _diff_section(
        current.get("behavioral_patterns") or [],
        prior.get("behavioral_patterns") or [],
        older_bp,
    )

    summary = {
        "obs_new": len(obs_diff["new"]),
        "obs_escalating": len(obs_diff["escalating"]),
        "obs_improving": len(obs_diff["improving"]),
        "obs_resolved": len(obs_diff["resolved"]),
        "obs_regressed": len(obs_diff["regressed"]),
        "obs_continuing": len(obs_diff["continuing"]),
        "bp_new": len(bp_diff["new"]),
        "bp_escalating": len(bp_diff["escalating"]),
        "bp_improving": len(bp_diff["improving"]),
        "bp_resolved": len(bp_diff["resolved"]),
        "bp_regressed": len(bp_diff["regressed"]),
        "bp_continuing": len(bp_diff["continuing"]),
    }

    prior_meta = prior.get("meta", {}) if isinstance(prior, dict) else {}
    return {
        "compared": True,
        "compared_against_slug": prior_meta.get("run_slug") or prior_meta.get("generated_at"),
        "compared_against_date": prior_meta.get("generated_at"),
        "observations": obs_diff,
        "behavioral_patterns": bp_diff,
        "summary": summary,
        "computed_at": _now_iso(),
    }


def changelog_for_current(
    current_synthesis: dict,
    history: HistoryStore | None = None,
    *,
    older_runs_lookback: int = 3,
) -> dict[str, Any]:
    """Convenience: pull the most-recent prior run + N older runs from
    history and compute the changelog against the current synthesis.

    Skips the prior run if its slug matches the current run's slug
    (otherwise re-running synthesize on the same window would diff
    a run against itself and report everything as 'continuing').
    """
    if history is None:
        history = HistoryStore()
    entries = history._read_index()
    if not entries:
        return compare_runs(current_synthesis, None)

    current_slug = (current_synthesis.get("meta") or {}).get("run_slug")
    # Find the most recent prior run that's NOT the current one
    prior_payload: dict | None = None
    older_payloads: list[dict] = []
    for entry in entries:
        slug = entry.get("slug")
        if not slug or slug == current_slug:
            continue
        try:
            payload = history.load_run(slug)
        except ValueError:
            continue
        if payload is None:
            continue
        if prior_payload is None:
            prior_payload = payload
        elif len(older_payloads) < older_runs_lookback:
            older_payloads.append(payload)
        else:
            break
    return compare_runs(current_synthesis, prior_payload, older_payloads)


def render_changelog_text(changelog: dict) -> str:
    """Plain-text rendering for `tessera changelog` CLI output."""
    if not changelog.get("compared"):
        return f"\nNo prior run to compare against.\n  ({changelog.get('reason', '')})\n"

    lines = []
    against = changelog.get("compared_against_date", "?")
    lines.append("")
    lines.append(f"SINCE LAST RUN ({against[:10]})")
    lines.append("=" * 60)

    s = changelog["summary"]
    lines.append("")
    lines.append("Operational observations:")
    lines.append(
        f"  new {s['obs_new']:>2}  ·  escalating {s['obs_escalating']:>2}  ·  "
        f"continuing {s['obs_continuing']:>2}  ·  improving {s['obs_improving']:>2}  ·  "
        f"resolved {s['obs_resolved']:>2}  ·  regressed {s['obs_regressed']:>2}"
    )
    lines.append("Behavioral patterns:")
    lines.append(
        f"  new {s['bp_new']:>2}  ·  escalating {s['bp_escalating']:>2}  ·  "
        f"continuing {s['bp_continuing']:>2}  ·  improving {s['bp_improving']:>2}  ·  "
        f"resolved {s['bp_resolved']:>2}  ·  regressed {s['bp_regressed']:>2}"
    )

    def _section(label: str, items: list[dict], show_delta: bool = False) -> None:
        if not items:
            return
        lines.append("")
        lines.append(f"{label} ({len(items)})")
        lines.append("-" * 60)
        for item in items:
            title = item.get("title") or "?"
            cat = item.get("category") or ""
            cat_chip = f" [{cat}]" if cat else ""
            if show_delta and "delta" in item:
                d = item["delta"]
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
                count_info = (
                    f"  {arrow} {item['previous_count']} → {item['current_count']}  ({'+' if d > 0 else ''}{d})"
                )
            elif "current_count" in item:
                count_info = f"  ({item['current_count']} sessions)"
            elif "previous_count" in item:
                count_info = f"  (was {item['previous_count']} sessions)"
            else:
                count_info = ""
            lines.append(f"  · {title}{cat_chip}{count_info}")

    for sec_name, items_dict in (
        ("OPERATIONAL", changelog["observations"]),
        ("BEHAVIORAL", changelog["behavioral_patterns"]),
    ):
        lines.append("")
        lines.append("=" * 60)
        lines.append(sec_name)
        _section("NEW", items_dict["new"])
        _section("ESCALATING", items_dict["escalating"], show_delta=True)
        _section("REGRESSED (was resolved, now back)", items_dict["regressed"])
        _section("IMPROVING", items_dict["improving"], show_delta=True)
        _section("RESOLVED (gone since last run)", items_dict["resolved"])
        # `continuing` is intentionally not printed — the user would just see
        # "everything's the same" noise. They can find continuing patterns in
        # the main dashboard sections.

    return "\n".join(lines)
