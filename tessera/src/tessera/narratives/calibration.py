"""Calibration audit — verify a synthesis's quantified claims against
the narratives it was generated from.

Why: the bake-off audit (4 backends × 29 sessions) showed that every
synthesizer produces at least one false quantified claim per run.
Gemini undercounted by 4 ("18 sessions" vs ground truth 22). Codex
fabricated a comparative pattern ("intervene in visual but not Pulse",
falsified — both 100% intervention rate). Claude varied run-to-run
on the same input (22 in v2, 17 in v3).

The audit catches these inline so the user doesn't have to trust the
synthesizer blindly. Pure Python — no LLM calls. Runs in milliseconds.

Approach: pattern-match quantified claims in synthesis fields, compute
the corresponding ground truth from narrative content, classify each
claim as PASS / FAIL / UNDETERMINED with a short explanation.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# Patterns that indicate auth/OAuth friction. The synthesizers all
# converge on this category for OAuth-related claims, so it's the most
# valuable to audit. The keyword set comes from real narratives —
# expand as new failure modes appear.
_OAUTH_KEYWORDS = (
    "invalid_grant", "invalid grant",
    "oauth", "token expir", "expired token",
    "re-auth", "reauth", "re-authent",
    "pulse-workspace-cli", "workspace-cli add",
    "permission wall", "permission_wall",
    "token has been expired or revoked",
)


# A bare number, surrounded by word boundaries.
_NUM = re.compile(r"\b(\d+)\b")
# Tries to anchor the "X sessions" pattern that's most common in headlines.
_NUM_SESSIONS = re.compile(r"\b(\d+)\s+(?:of\s+\d+\s+)?sessions?\b", re.IGNORECASE)


def _narrative_text(narr: dict) -> str:
    """All extracted-by-LLM and free-text fields concatenated, lowercased.
    We match against this for keyword presence."""
    return json.dumps(narr).lower()


def _has_oauth_friction(narr: dict) -> bool:
    txt = _narrative_text(narr)
    return any(k in txt for k in _OAUTH_KEYWORDS)


def _project_label(narr: dict) -> str:
    return (narr.get("project_label") or "").lower()


def _has_subagent(narr: dict) -> bool:
    return (narr.get("subagent_count") or 0) > 0


def _verification_level(narr: dict) -> str:
    return (narr.get("verification_completeness") or "").lower()


def compute_ground_truth(narratives: list[dict]) -> dict[str, Any]:
    """Deterministic counts every claim is graded against."""
    n = len(narratives)
    oauth_all = sum(1 for x in narratives if _has_oauth_friction(x))
    pulse_total = sum(1 for x in narratives if "pulse" in _project_label(x))
    oauth_pulse = sum(
        1 for x in narratives
        if "pulse" in _project_label(x) and _has_oauth_friction(x)
    )
    verif_strong = sum(
        1 for x in narratives
        if _verification_level(x) not in ("claimed_only", "none", "")
    )
    subagent_sessions = sum(1 for x in narratives if _has_subagent(x))
    return {
        "total_sessions": n,
        "sessions_with_oauth_friction": oauth_all,
        "pulse_sessions_total": pulse_total,
        "pulse_sessions_with_oauth": oauth_pulse,
        "sessions_with_subagent_use": subagent_sessions,
        "sessions_with_stronger_verification": verif_strong,
        "verification_distribution": dict(
            Counter(_verification_level(x) or "(missing)" for x in narratives)
        ),
        "project_distribution": dict(
            Counter(_project_label(x) or "(missing)" for x in narratives)
        ),
    }


def _likely_denominator(text: str) -> str:
    """Heuristic: does this claim's scope sound pulse-only or all-sessions?
    Used to pick the right ground-truth row to compare against."""
    t = text.lower()
    # "pulse" in the headline + "sessions" usually means pulse-only.
    if "pulse session" in t or "pulse-only" in t:
        return "pulse_sessions_with_oauth"
    return "sessions_with_oauth_friction"


def _audit_oauth_claim(text: str, truth: dict) -> dict | None:
    """If `text` makes a quantified OAuth/auth claim, grade it.
    Returns None if no such claim found."""
    tlow = text.lower()
    # Must mention auth/OAuth-ish topic to qualify as an OAuth claim.
    if not any(k in tlow for k in ("oauth", "auth", "permission wall", "token", "workspace", "expir")):
        return None

    # Prefer "N sessions" anchored matches; fall back to first bare number.
    m = _NUM_SESSIONS.search(text)
    if m:
        claim = int(m.group(1))
    else:
        m = _NUM.search(text)
        if not m:
            return None
        claim = int(m.group(1))

    denom_key = _likely_denominator(text)
    ground = truth.get(denom_key, 0)
    delta = claim - ground
    return {
        "topic": "oauth",
        "claim": claim,
        "denominator": denom_key,
        "ground_truth": ground,
        "delta": delta,
        "verdict": "PASS" if delta == 0 else "FAIL",
        "explanation": (
            f"claim says {claim}; narratives show {ground} sessions "
            f"({denom_key.replace('_', ' ')}); off by {delta:+d}"
        ),
    }


def _audit_subagent_claim(text: str, truth: dict) -> dict | None:
    """Match claims like 'zero subagent delegation across 29 sessions'."""
    tlow = text.lower()
    if "subagent" not in tlow and "delegation" not in tlow and "delegat" not in tlow:
        return None
    # Look for a number adjacent to 'subagent' or for "zero"
    if "zero" in tlow or "0 of" in tlow or "no subagent" in tlow:
        claim = 0
    else:
        m = _NUM.search(text)
        if not m:
            return None
        claim = int(m.group(1))
    ground = truth["sessions_with_subagent_use"]
    delta = claim - ground
    return {
        "topic": "subagent_delegation",
        "claim": claim,
        "denominator": "sessions_with_subagent_use",
        "ground_truth": ground,
        "delta": delta,
        "verdict": "PASS" if delta == 0 else "FAIL",
        "explanation": (
            f"claim says {claim}; narratives show {ground}; off by {delta:+d}"
        ),
    }


def _audit_verification_claim(text: str, truth: dict) -> dict | None:
    """Match claims about how many sessions had verification.
    Pattern: 'verified in X of Y sessions' or 'X sessions verified'."""
    tlow = text.lower()
    if "verif" not in tlow:
        return None
    m = re.search(r"\b(\d+)\s+(?:of\s+\d+\s+)?sessions?\b", text, re.IGNORECASE)
    if not m:
        m = _NUM.search(text)
    if not m:
        return None
    claim = int(m.group(1))
    ground = truth["sessions_with_stronger_verification"]
    delta = claim - ground
    return {
        "topic": "verification",
        "claim": claim,
        "denominator": "sessions_with_stronger_verification",
        "ground_truth": ground,
        "delta": delta,
        "verdict": "PASS" if delta == 0 else "FAIL",
        "explanation": (
            f"claim says {claim}; narratives show {ground} sessions "
            f"with stronger-than-claimed verification; off by {delta:+d}"
        ),
    }


def _extract_claim_texts(synthesis: dict) -> list[tuple[str, str]]:
    """Pull every (source_label, text) pair that could contain a
    quantified claim. Source labels track where each claim came from
    so the audit output points the user to the right section."""
    out: list[tuple[str, str]] = []
    if hl := synthesis.get("headline"):
        out.append(("headline", hl))
    if one := synthesis.get("if_you_do_one_thing_this_week"):
        out.append(("if_you_do_one_thing", one))
    for i, o in enumerate(synthesis.get("observations", []), 1):
        t = " ".join(filter(None, [o.get("title"), o.get("evidence_excerpt"), o.get("why")]))
        if t.strip():
            out.append((f"observations[{i}]", t))
    for i, b in enumerate(synthesis.get("behavioral_patterns", []), 1):
        t = " ".join(filter(None, [b.get("title"), b.get("evidence_excerpt"), b.get("why")]))
        if t.strip():
            out.append((f"behavioral_patterns[{i}]", t))
    return out


def calibrate(synthesis: dict, narratives: list[dict]) -> dict:
    """Run the full calibration audit.

    Returns a dict with:
      - `ground_truth`: deterministic counts computed from narratives
      - `findings`: per-claim PASS / FAIL / UNDETERMINED records
      - `summary`: aggregate counts so dashboard can show at-a-glance
    """
    truth = compute_ground_truth(narratives)
    findings: list[dict] = []
    auditors = (_audit_oauth_claim, _audit_subagent_claim, _audit_verification_claim)
    for source, text in _extract_claim_texts(synthesis):
        for audit_fn in auditors:
            result = audit_fn(text, truth)
            if result is None:
                continue
            findings.append({
                "source": source,
                "text_excerpt": text[:160],
                **result,
            })
    passes = sum(1 for f in findings if f["verdict"] == "PASS")
    fails = sum(1 for f in findings if f["verdict"] == "FAIL")
    return {
        "ground_truth": truth,
        "findings": findings,
        "summary": {
            "total_quantified_claims_checked": len(findings),
            "passed": passes,
            "failed": fails,
            "pass_rate": (passes / len(findings)) if findings else None,
        },
    }


def render_calibration_text(report: dict) -> str:
    """Pretty-print the audit report. Mirrors render_eval_text style."""
    out: list[str] = []
    out.append("CALIBRATION AUDIT")
    out.append("=" * 70)
    g = report["ground_truth"]
    out.append(
        f"Ground truth from {g['total_sessions']} narratives:"
    )
    out.append(
        f"  OAuth friction       : {g['sessions_with_oauth_friction']} sessions "
        f"(pulse-only: {g['pulse_sessions_with_oauth']})"
    )
    out.append(
        f"  Subagent delegation  : {g['sessions_with_subagent_use']} sessions"
    )
    out.append(
        f"  Stronger verification: {g['sessions_with_stronger_verification']} sessions"
    )
    summary = report["summary"]
    if summary["total_quantified_claims_checked"] == 0:
        out.append("")
        out.append("No quantified OAuth/subagent/verification claims found in synthesis.")
        return "\n".join(out)
    out.append("")
    pr = summary["pass_rate"]
    out.append(
        f"Quantified claims checked: {summary['total_quantified_claims_checked']}  "
        f"({summary['passed']} PASS · {summary['failed']} FAIL"
        + (f" · {pr:.0%} pass rate" if pr is not None else "")
        + ")"
    )
    out.append("")
    for f in report["findings"]:
        marker = "✓" if f["verdict"] == "PASS" else "✗"
        out.append(f"  {marker} [{f['source']}] {f['topic']}")
        out.append(f"      {f['explanation']}")
        out.append(f"      from: {f['text_excerpt']}")
    return "\n".join(out)
