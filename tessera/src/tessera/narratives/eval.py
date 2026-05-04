"""Quality metrics for a narratives directory + a synthesis output.

These are the quantitative signals we'd want to track across releases (e.g.
"did v0.3 increase fabrication rate? did it produce more low-quality
narratives?"). Designed to be cheap to compute, cheap to compare across runs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


# Rough Sonnet 4.6 pricing — input/output dollars per million tokens.
# Update if pricing changes; only used as a back-of-envelope cost estimate.
SONNET_PRICE_INPUT_PER_MTOK = 3.0
SONNET_PRICE_OUTPUT_PER_MTOK = 15.0


def _est_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * SONNET_PRICE_INPUT_PER_MTOK
        + output_tokens / 1_000_000 * SONNET_PRICE_OUTPUT_PER_MTOK
    )


def evaluate_narratives(narratives_dir: Path) -> dict[str, Any]:
    """Quality metrics for a directory of per-session narrative JSON files."""
    narratives: list[dict] = []
    for path in sorted(narratives_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("session_id"):
            narratives.append(data)
    n = len(narratives)
    if n == 0:
        return {"narratives_dir": str(narratives_dir), "session_count": 0}

    quality = Counter(d.get("narrative_quality") for d in narratives)
    agents = Counter(d.get("agent") for d in narratives)
    waste_sigs = Counter(d.get("waste_signature") for d in narratives)
    verif = Counter(d.get("verification_completeness") for d in narratives)
    pq = Counter(d.get("prompt_quality_signal") for d in narratives)

    task_count = sum(len(d.get("tasks") or []) for d in narratives)
    friction_count = sum(len(d.get("friction_moments") or []) for d in narratives)
    decision_count = sum(len(d.get("key_decisions") or []) for d in narratives)
    dead_end_count = sum(len(d.get("dead_ends") or []) for d in narratives)
    env_count = sum(
        len(d.get("recurring_environmental_issues") or []) for d in narratives
    )
    ucme_count = sum(
        (d.get("user_caught_model_errors") or {}).get("count", 0) for d in narratives
    )

    sessions_with_zero_friction = sum(
        1 for d in narratives if not (d.get("friction_moments") or [])
    )
    sessions_with_env_issues = sum(
        1 for d in narratives if (d.get("recurring_environmental_issues") or [])
    )

    total_events = sum(d.get("event_count") or 0 for d in narratives)
    total_active_min = sum(d.get("active_minutes") or 0 for d in narratives)
    total_stream_chars = sum(d.get("stream_chars") or 0 for d in narratives)
    # Sonnet-ish heuristic: input ≈ stream_chars/4, output ≈ 3K tokens/session
    est_input_tok = total_stream_chars // 4
    est_output_tok = n * 3000
    est_cost = _est_cost(est_input_tok, est_output_tok)

    return {
        "narratives_dir": str(narratives_dir),
        "session_count": n,
        "agents": dict(agents),
        "narrative_quality": dict(quality),
        "narrative_quality_share": {
            k: round(v / n, 3) for k, v in quality.items()
        },
        "totals": {
            "events": total_events,
            "active_minutes": round(total_active_min, 1),
            "tasks": task_count,
            "friction_moments": friction_count,
            "key_decisions": decision_count,
            "dead_ends": dead_end_count,
            "recurring_env_issues": env_count,
            "user_caught_model_errors": ucme_count,
        },
        "averages_per_session": {
            "tasks": round(task_count / n, 2),
            "friction_moments": round(friction_count / n, 2),
            "key_decisions": round(decision_count / n, 2),
            "env_issues": round(env_count / n, 2),
            "events": round(total_events / n, 0),
        },
        "coverage": {
            "sessions_with_zero_friction": sessions_with_zero_friction,
            "share_zero_friction": round(sessions_with_zero_friction / n, 3),
            "sessions_with_env_issues": sessions_with_env_issues,
            "share_with_env_issues": round(sessions_with_env_issues / n, 3),
        },
        "distributions": {
            "waste_signature": dict(waste_sigs),
            "verification_completeness": dict(verif),
            "prompt_quality_signal": dict(pq),
        },
        "cost_estimate": {
            "est_input_tokens": est_input_tok,
            "est_output_tokens": est_output_tok,
            "est_cost_usd": round(est_cost, 2),
            "pricing_basis": f"sonnet-4-6 @ ${SONNET_PRICE_INPUT_PER_MTOK}/M in, ${SONNET_PRICE_OUTPUT_PER_MTOK}/M out",
        },
    }


def evaluate_synthesis(synthesis_path: Path) -> dict[str, Any]:
    """Quality metrics for a single synthesis JSON output."""
    try:
        syn = json.loads(synthesis_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"synthesis_path": str(synthesis_path), "error": "could_not_load"}

    obs_list = syn.get("observations") or []
    qw_list = syn.get("quick_wins") or []
    pp_list = syn.get("per_project") or []
    meta = syn.get("meta", {})

    confidence = Counter(o.get("confidence") for o in obs_list)
    category = Counter(o.get("category") for o in obs_list)
    supporting_counts = [o.get("supporting_count") or 0 for o in obs_list]

    total_evidence_refs = sum(supporting_counts)
    fabricated = meta.get("fabricated_ref_count")
    fabrication_rate = None
    if fabricated is not None and (fabricated + total_evidence_refs) > 0:
        fabrication_rate = round(
            fabricated / (fabricated + total_evidence_refs), 3
        )

    return {
        "synthesis_path": str(synthesis_path),
        "input_sessions": meta.get("input_sessions"),
        "model": meta.get("model"),
        "had_prior_context": meta.get("had_prior_context"),
        "headline_present": bool(syn.get("headline")),
        "if_one_thing_present": bool(syn.get("if_you_do_one_thing_this_week")),
        "observation_count": len(obs_list),
        "quick_win_count": len(qw_list),
        "per_project_count": len(pp_list),
        "confidence_distribution": dict(confidence),
        "category_distribution": dict(category),
        "supporting_count": {
            "min": min(supporting_counts) if supporting_counts else 0,
            "max": max(supporting_counts) if supporting_counts else 0,
            "mean": round(sum(supporting_counts) / max(len(supporting_counts), 1), 1),
            "total_refs_cited": total_evidence_refs,
        },
        "fabrication": {
            "fabricated_ref_count": fabricated,
            "fabrication_rate": fabrication_rate,
        },
    }


def render_eval_text(narrative_eval: dict, synthesis_eval: dict | None = None) -> str:
    lines = []
    lines.append("")
    lines.append("TESSERA — EVAL")
    lines.append("=" * 60)
    lines.append(f"Narratives dir: {narrative_eval.get('narratives_dir')}")
    lines.append(f"Sessions: {narrative_eval.get('session_count')}")
    if narrative_eval.get("agents"):
        lines.append(f"  by agent: {narrative_eval['agents']}")
    lines.append("")
    lines.append("Quality:")
    nq = narrative_eval.get("narrative_quality", {})
    for k in ("high", "medium", "low", "skipped"):
        if nq.get(k):
            share = narrative_eval.get("narrative_quality_share", {}).get(k, 0)
            lines.append(f"  {k}: {nq[k]} ({share*100:.0f}%)")
    lines.append("")
    lines.append("Coverage:")
    cov = narrative_eval.get("coverage", {})
    if cov:
        lines.append(
            f"  sessions with friction: "
            f"{narrative_eval['session_count'] - cov.get('sessions_with_zero_friction', 0)}/"
            f"{narrative_eval['session_count']}"
        )
        lines.append(
            f"  sessions with env issues: "
            f"{cov.get('sessions_with_env_issues', 0)}/"
            f"{narrative_eval['session_count']} "
            f"({cov.get('share_with_env_issues', 0)*100:.0f}%)"
        )
    lines.append("")
    lines.append("Per-session averages:")
    avg = narrative_eval.get("averages_per_session", {})
    for k, v in avg.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Totals:")
    tot = narrative_eval.get("totals", {})
    for k, v in tot.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    cost = narrative_eval.get("cost_estimate", {})
    if cost:
        lines.append(
            f"Cost estimate: ~${cost.get('est_cost_usd')} "
            f"({cost.get('est_input_tokens'):,} in + {cost.get('est_output_tokens'):,} out tokens)"
        )

    if synthesis_eval:
        lines.append("")
        lines.append("-" * 60)
        lines.append("SYNTHESIS")
        lines.append("")
        lines.append(f"Synthesis path: {synthesis_eval.get('synthesis_path')}")
        lines.append(f"Input sessions: {synthesis_eval.get('input_sessions')}")
        lines.append(f"Observations: {synthesis_eval.get('observation_count')}")
        lines.append(f"Quick wins: {synthesis_eval.get('quick_win_count')}")
        lines.append(f"Per-project entries: {synthesis_eval.get('per_project_count')}")
        lines.append(f"Confidence: {synthesis_eval.get('confidence_distribution')}")
        lines.append(f"Category: {synthesis_eval.get('category_distribution')}")
        sc = synthesis_eval.get("supporting_count", {})
        if sc:
            lines.append(
                f"Supporting refs per observation: "
                f"min={sc.get('min')} max={sc.get('max')} mean={sc.get('mean')} "
                f"(total cited: {sc.get('total_refs_cited')})"
            )
        fab = synthesis_eval.get("fabrication", {})
        if fab.get("fabricated_ref_count") is not None:
            rate = fab.get("fabrication_rate")
            rate_str = f" ({rate*100:.1f}%)" if rate is not None else ""
            lines.append(
                f"Fabrication: {fab['fabricated_ref_count']} dropped{rate_str}"
            )

    return "\n".join(lines)
