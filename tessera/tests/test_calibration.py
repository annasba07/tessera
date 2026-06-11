"""Calibration audit tests.

Lock in the behavior that catches the failures the bake-off revealed:
gemini's "18 vs 22" undercount, codex's hallucinated comparative pattern,
claude's run-to-run variance (22 v2 vs 17 v3).
"""
from tessera.narratives.calibration import (
    calibrate,
    compute_ground_truth,
    render_calibration_text,
)


def _make_narrative(
    *,
    session_id="claude:abc",
    project="kaizen/pulse",
    has_oauth=False,
    has_subagent=False,
    verification="claimed_only",
):
    """Build a minimal narrative with the fields the calibrator inspects."""
    n = {
        "session_id": session_id,
        "project_label": project,
        "subagent_count": 1 if has_subagent else 0,
        "verification_completeness": verification,
        "tasks": [],
    }
    if has_oauth:
        # The keyword has to appear *somewhere* in the JSON-serialized
        # narrative for the keyword detector to fire. Stuff it into a
        # plausible-looking field.
        n["recurring_environmental_issues"] = [
            {"issue": "invalid_grant on gworkspace fetch — needs pulse-workspace-cli add"}
        ]
    return n


def test_ground_truth_counts_oauth_friction_correctly():
    narratives = (
        [_make_narrative(has_oauth=True) for _ in range(22)]
        + [_make_narrative(has_oauth=False) for _ in range(7)]
    )
    g = compute_ground_truth(narratives)
    assert g["total_sessions"] == 29
    assert g["sessions_with_oauth_friction"] == 22


def test_ground_truth_separates_pulse_from_all_sessions():
    """A claim like '17 pulse sessions' must be graded against the
    pulse-only subset, not all sessions."""
    pulse_oauth = [
        _make_narrative(project="kaizen/pulse", has_oauth=True) for _ in range(17)
    ]
    other_oauth = [
        _make_narrative(project="kaizen/atella", has_oauth=True) for _ in range(5)
    ]
    g = compute_ground_truth(pulse_oauth + other_oauth)
    assert g["sessions_with_oauth_friction"] == 22
    assert g["pulse_sessions_with_oauth"] == 17


def test_catches_gemini_undercount():
    """The exact failure from the bake-off: gemini said '18 sessions'
    when narratives showed 22. Must flag as FAIL with delta -4."""
    narratives = [
        _make_narrative(has_oauth=True) for _ in range(22)
    ] + [_make_narrative() for _ in range(7)]
    synthesis = {
        "headline": "18 sessions hit permission walls due to expired Google Workspace tokens",
    }
    report = calibrate(synthesis, narratives)
    oauth_findings = [f for f in report["findings"] if f["topic"] == "oauth"]
    assert len(oauth_findings) == 1
    f = oauth_findings[0]
    assert f["claim"] == 18
    assert f["ground_truth"] == 22
    assert f["delta"] == -4
    assert f["verdict"] == "FAIL"
    assert f["source"] == "headline"


def test_passes_codex_correct_count():
    """The exact match from the bake-off: codex said '22 sessions',
    truth was 22. Must flag as PASS."""
    narratives = [_make_narrative(has_oauth=True) for _ in range(22)] + [
        _make_narrative() for _ in range(7)
    ]
    synthesis = {
        "headline": "Google Workspace auth blocked 22 sessions and turned source-complete Pulse work into partial work."
    }
    report = calibrate(synthesis, narratives)
    f = next(x for x in report["findings"] if x["topic"] == "oauth")
    assert f["claim"] == 22
    assert f["ground_truth"] == 22
    assert f["verdict"] == "PASS"


def test_pulse_only_claim_uses_pulse_denominator():
    """Claude v3's headline: '17 pulse sessions blocked'. The audit
    must grade this against pulse_sessions_with_oauth, not against
    sessions_with_oauth_friction. Otherwise we'd FAIL a correct claim."""
    pulse_oauth = [
        _make_narrative(project="kaizen/pulse", has_oauth=True) for _ in range(17)
    ]
    extras = [
        _make_narrative(project="kaizen/atella", has_oauth=True) for _ in range(5)
    ]
    report = calibrate(
        {"headline": "OAuth token expiry blocks 17 pulse sessions and a weekly re-auth sweep would prevent all of them."},
        pulse_oauth + extras,
    )
    f = next(x for x in report["findings"] if x["topic"] == "oauth")
    assert f["claim"] == 17
    assert f["denominator"] == "pulse_sessions_with_oauth"
    assert f["ground_truth"] == 17
    assert f["verdict"] == "PASS"


def test_subagent_zero_claim_passes_when_truth_is_zero():
    """Claude's 'zero subagent delegation across 29 sessions' — supported."""
    narratives = [_make_narrative(has_subagent=False) for _ in range(29)]
    synthesis = {
        "behavioral_patterns": [
            {"title": "Zero subagent delegation across all 29 sessions despite multi-source work"}
        ]
    }
    report = calibrate(synthesis, narratives)
    f = next(x for x in report["findings"] if x["topic"] == "subagent_delegation")
    assert f["claim"] == 0
    assert f["ground_truth"] == 0
    assert f["verdict"] == "PASS"


def test_verification_undercount_flagged():
    """Claude said 'verified in 1 of 29 sessions' but narratives showed 5
    sessions with stronger-than-claimed verification. Must flag."""
    narratives = (
        [_make_narrative(verification="lightly_verified") for _ in range(4)]
        + [_make_narrative(verification="thoroughly_verified") for _ in range(1)]
        + [_make_narrative(verification="claimed_only") for _ in range(24)]
    )
    synthesis = {
        "observations": [
            {"title": "Research output verified in 1 of 29 sessions"}
        ]
    }
    report = calibrate(synthesis, narratives)
    f = next(x for x in report["findings"] if x["topic"] == "verification")
    assert f["claim"] == 1
    assert f["ground_truth"] == 5
    assert f["delta"] == -4
    assert f["verdict"] == "FAIL"


def test_empty_synthesis_returns_no_findings():
    narratives = [_make_narrative() for _ in range(5)]
    report = calibrate({}, narratives)
    assert report["findings"] == []
    assert report["summary"]["total_quantified_claims_checked"] == 0


def test_renderer_marks_pass_and_fail_clearly():
    narratives = [_make_narrative(has_oauth=True) for _ in range(22)] + [
        _make_narrative() for _ in range(7)
    ]
    synthesis = {
        "headline": "18 sessions hit permission walls",
        "observations": [{"title": "OAuth blocked 22 sessions"}],
    }
    report = calibrate(synthesis, narratives)
    out = render_calibration_text(report)
    assert "CALIBRATION AUDIT" in out
    assert "✗" in out  # the 18 failure
    assert "✓" in out  # the 22 pass
    assert "Ground truth" in out


def test_summary_pass_rate_computation():
    narratives = [_make_narrative(has_oauth=True) for _ in range(22)] + [
        _make_narrative() for _ in range(7)
    ]
    synthesis = {
        "headline": "18 sessions hit permission walls",  # FAIL
        "observations": [
            {"title": "OAuth blocked 22 sessions"},      # PASS
            {"title": "Permission walls in 22 sessions"}, # PASS
        ],
    }
    report = calibrate(synthesis, narratives)
    s = report["summary"]
    assert s["total_quantified_claims_checked"] == 3
    assert s["passed"] == 2
    assert s["failed"] == 1
    assert abs(s["pass_rate"] - 2 / 3) < 1e-6
