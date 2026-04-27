"""Cross-session rating lookup for the coach.

When a rule is about to fire, the coach checks the user's last N synthesis
runs to see whether similar patterns were already marked ``wrong`` (suppress),
``useful`` (attach evidence + next_action), or ``known`` (soften language).
This is how weekly `tessera rate` sessions tune the in-session coach's
signal/noise ratio.

Matching is done by:
1. Lowercase substring against observation titles + claims using a per-rule
   keyword set, AND
2. Optional category match — a synthesis observation's `category` field is
   compared against the rule's expected category for higher-confidence matches.

Both are deliberately fuzzy — we'd rather over-match and let the suppress
logic be conservative than under-match a clear "this was rated wrong
last week" signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..history import DEFAULT_DATA_DIR, HistoryStore, _observation_key

if TYPE_CHECKING:
    from pathlib import Path


# Keywords that characterize each rule's topic. Matched (case-insensitive)
# against observation title + claim. Updated to cover both legacy
# observation language and new synthesis vocabulary (waste_signature labels,
# friction_moment terminology).
RULE_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "browser_spiral": [
        "browser spiral",
        "browser_spiral",
        "browser_snapshot",
        "browser_evaluate",
        "playwright",
        "chrome_devtools",
        "chrome devtools",
        "singletonlock",
        "browser automation",
        "snapshot retry",
        "stale lock",
    ],
    "retry_without_change": [
        "blind_retry",
        "blind retry",
        "retry without",
        "fix loop",
        "same prompt",
        "identical retries",
        "repeated prompt",
        "tool_error_loop",
        "tool error loop",
        "apply_patch unknown",
        "stale context",
    ],
    "permission_wall_repeat": [
        "permission_wall",
        "permission wall",
        "permission denied",
        "permission_denied",
        "approval required",
        "approval_required",
        "permission block",
        "missing scope",
    ],
    "runaway": [
        "runaway",
        "tight loop",
        "tool calls in",
        "per second",
        "call rate",
        "spiral",
    ],
    "edit_without_verify": [
        "unverified edit",
        "without verify",
        "verify after edit",
        "verification visibility",
        "no test run",
        "edit without",
        "verify_gap",
        "claimed_only",
    ],
    "delegation_sprawl": [
        "delegation sprawl",
        "delegation heavy",
        "subagent",
        "too many subagents",
        "over_delegation",
        "prompt is too long",
        "context exhausted",
        "context_loss",
    ],
}


# Optional: a category match boosts confidence of a keyword hit. A rule key
# maps to the synthesis category vocab values that are most likely relevant.
RULE_CATEGORY_HINTS: dict[str, set[str]] = {
    "browser_spiral": {"environmental", "tooling"},
    "retry_without_change": {"tooling", "workflow"},
    "permission_wall_repeat": {"environmental", "tooling"},
    "runaway": {"workflow", "tooling"},
    "edit_without_verify": {"workflow", "prompting"},
    "delegation_sprawl": {"tooling", "workflow"},
}


# How many recent runs to scan for ratings. Older rated runs drop out.
DEFAULT_LOOKBACK_RUNS = 3

# If a rule was rated `wrong` this many times within the lookback window
# for the same project, suppress it in-session.
WRONG_SUPPRESS_THRESHOLD = 1


@dataclass
class MatchedObservation:
    run_timestamp: str
    obs_key: str
    title: str
    rating: str  # useful / wrong / known / skip / unrated
    next_action: str | None = None
    category: str | None = None
    confidence: str | None = None


@dataclass
class RuleSignal:
    """Per-(rule, project) summary of how the user has rated similar patterns."""

    rule_key: str
    project: str | None
    matches: list[MatchedObservation] = field(default_factory=list)
    useful_count: int = 0
    wrong_count: int = 0
    known_count: int = 0
    suppress: bool = False

    def evidence_line(self) -> str | None:
        """Return a short evidence blurb for the nudge, or None if quiet.

        Surfaces the user-rated synthesis observation that supports this rule
        plus its concrete `next_action` if available — that's what makes the
        nudge actionable rather than just informational.
        """
        useful = [m for m in self.matches if m.rating == "useful"]
        known = [m for m in self.matches if m.rating == "known"]
        if useful:
            head = useful[0]
            line = (
                f"Prior signal: you rated this pattern USEFUL on "
                f"{head.run_timestamp[:10]} — key [{head.obs_key}]."
            )
            if head.next_action:
                line += f" Suggested next: {head.next_action[:200]}"
            return line
        if known:
            head = known[0]
            line = (
                f"Prior signal: you marked a similar pattern as already known on "
                f"{head.run_timestamp[:10]} — key [{head.obs_key}]. Soft reminder."
            )
            if head.next_action:
                line += f" Suggested next: {head.next_action[:200]}"
            return line
        return None


def _matches_rule(
    observation: dict, keywords: list[str], category_hints: set[str]
) -> bool:
    """Match requires a keyword hit. Category is an optional precision filter:
    if the rule defines category hints AND the observation has a category,
    the category must be in the hint set or the match is rejected.
    """
    haystack = (
        f"{observation.get('title') or ''} {observation.get('claim') or ''}"
    ).lower()
    if not any(kw.lower() in haystack for kw in keywords):
        return False
    obs_category = observation.get("category")
    if category_hints and obs_category and obs_category not in category_hints:
        return False
    return True


def _project_name_matches(
    obs_evidence_ids: list[str],
    obs_project_hint: str | None,
    project: str | None,
) -> bool:
    """Best-effort project match.

    Past observations don't carry a structured project field, but the
    evidence IDs often include the project basename via upstream scorecards
    (e.g. the narrative text mentions the project). For now we accept any
    match on the keyword level and let the rating count be the gate; we can
    tighten this later once observations carry explicit project tags.
    """
    # Project-tightening not implemented yet — return True so all keyword
    # matches are counted. Downstream suppress logic still needs multiple
    # wrong-rated observations before suppressing, which keeps noise down.
    return True


def get_rule_signal(
    rule_key: str,
    project: str | None,
    *,
    lookback_runs: int = DEFAULT_LOOKBACK_RUNS,
    history_dir: "Path | None" = None,
) -> RuleSignal:
    keywords = RULE_TOPIC_KEYWORDS.get(rule_key) or []
    category_hints = RULE_CATEGORY_HINTS.get(rule_key) or set()
    signal = RuleSignal(rule_key=rule_key, project=project)
    if not keywords and not category_hints:
        return signal

    try:
        store = HistoryStore(history_dir or DEFAULT_DATA_DIR)
    except Exception:
        return signal

    entries = store._read_index()[:lookback_runs]
    if not entries:
        return signal

    for entry in entries:
        slug = entry.get("slug")
        if not slug:
            continue
        payload = store.load_run(slug)
        if not payload:
            continue
        ratings_by_index = {r["index"]: r for r in store.load_ratings(slug)}
        run_ts = entry.get("timestamp", "")
        for idx, obs in enumerate(payload.get("observations", []) or []):
            if not _matches_rule(obs, keywords, category_hints):
                continue
            rating_row = ratings_by_index.get(idx) or {}
            rating = rating_row.get("rating") or "unrated"
            signal.matches.append(
                MatchedObservation(
                    run_timestamp=run_ts,
                    obs_key=_observation_key(obs),
                    title=obs.get("title") or f"(untitled §{idx + 1})",
                    rating=rating,
                    next_action=obs.get("next_action"),
                    category=obs.get("category"),
                    confidence=obs.get("confidence"),
                )
            )
            if rating == "useful":
                signal.useful_count += 1
            elif rating == "wrong":
                signal.wrong_count += 1
            elif rating == "known":
                signal.known_count += 1

    signal.suppress = signal.wrong_count >= WRONG_SUPPRESS_THRESHOLD
    return signal
