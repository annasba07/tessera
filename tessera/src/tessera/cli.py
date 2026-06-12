"""Command-line entry point for tessera."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__
from .backends import default_model_for, get_backend, list_backends
from .history import DEFAULT_DATA_DIR, HistoryStore, _observation_key
from .narratives import DEFAULT_CACHE_DIR, NarrativeCache
from .narratives.extractor import DEFAULT_MODEL as DEFAULT_NARRATIVE_MODEL
from .narratives.pipeline import extract_many, load_session_events
from .pipeline import (
    DEFAULT_CLAUDE_PROJECTS,
    DEFAULT_CODEX_SESSIONS,
    DEFAULT_GEMINI_PROJECTS_JSON,
    DEFAULT_GEMINI_TMP,
    normalize_live_traces,
)


def _resolve_backend_and_model(args: argparse.Namespace):
    """Resolve (backend_instance, model_id) from CLI args.

    Falls back via ``backends.get_backend`` to antigravity if installed
    else claude. Lets --model override the backend's default; otherwise
    uses the backend's default model so `--backend codex` doesn't try to
    send claude-sonnet-4-6 to OpenAI.

    Backends with empty default_model (codex, gemini) mean "let the CLI
    pick its session default" — we pass empty string through so the
    backend knows to omit the --model flag.
    """
    explicit = getattr(args, "backend", None)
    backend = get_backend(explicit)  # honors --backend → env → installed-default
    requested_model = getattr(args, "model", None)
    # The arg default is the Claude historical default. If the resolved
    # backend isn't Claude AND the model wasn't explicitly overridden,
    # swap to the backend's own default (which may be empty).
    if backend.name != "claude" and requested_model == DEFAULT_NARRATIVE_MODEL:
        model = backend.default_model
    else:
        model = requested_model or backend.default_model
    return backend, model


RATING_PROMPTS = {
    "u": "useful",
    "w": "wrong",
    "k": "known",
    "s": "skip",
    "": "skip",
}


def _run_command(args: argparse.Namespace) -> int:
    """Full retrospective: normalize → narrate (per-session) → synthesize → save → render."""
    from .narratives.pipeline import load_all_sessions_events
    from .narratives.synthesis import load_narratives, synthesize as run_synthesis

    backend, args.model = _resolve_backend_and_model(args)
    from .narratives.render import render_synthesis_markdown, render_synthesis_text

    # --all-time / lookback_days=0 → no time filter
    if args.all_time or (args.lookback_days is not None and args.lookback_days <= 0):
        args.lookback_days = None

    narratives_dir = Path(args.narratives_dir).expanduser().resolve()
    narratives_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = NarrativeCache(
        Path(args.cache_dir).expanduser() if args.cache_dir else DEFAULT_CACHE_DIR
    )

    with tempfile.TemporaryDirectory(prefix="tessera-run-") as work_dir:
        norm_dir = Path(work_dir)
        print(
            f"[1/4] Normalizing live traces (Claude: {args.claude_projects}, "
            f"Codex: {args.codex_sessions}, Gemini: {args.gemini_tmp})...",
            file=sys.stderr,
        )
        try:
            normalize_live_traces(
                norm_dir,
                claude_projects=Path(args.claude_projects).expanduser(),
                codex_sessions=Path(args.codex_sessions).expanduser(),
                gemini_tmp=Path(args.gemini_tmp).expanduser(),
                gemini_projects_json=Path(args.gemini_projects_json).expanduser(),
                max_text_chars=args.max_text_chars,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        sessions_path = norm_dir / "sessions.jsonl"
        events_path = norm_dir / "events.jsonl"

        window_label = (
            f"{args.lookback_days}d window" if args.lookback_days else "all-time"
        )
        print(
            f"[2/4] Selecting sessions ({window_label}, min_events={args.min_events})...",
            file=sys.stderr,
        )
        qualifying, filter_breakdown = _qualifying_session_ids(
            sessions_path,
            min_events=args.min_events,
            lookback_days=args.lookback_days,
            max_age_days=None,
        )
        if args.limit and args.limit > 0:
            qualifying = qualifying[: args.limit]
        target_ids = [q["session_id"] for q in qualifying]
        print(
            f"       → {len(target_ids)} qualified "
            f"(window total: {filter_breakdown['in_window']}, "
            f"dropped {filter_breakdown['dropped_short']} short + "
            f"{filter_breakdown['dropped_slash']} slash-cmd).",
            file=sys.stderr,
        )
        if len(target_ids) < args.min_sessions:
            print(
                f"error: need at least --min-sessions={args.min_sessions} sessions; "
                f"only {len(target_ids)} qualified. Try a wider --lookback-days or lower --min-events.",
                file=sys.stderr,
            )
            return 2

        # First-run pre-flight: when there's no history yet, show the
        # user what they're about to spend and let them bail. After the
        # first run, narratives cache and re-runs are cheap, so we skip
        # the prompt.
        history_dir = Path(args.history_dir).expanduser()
        is_first_run = (
            not (history_dir / "history.json").exists()
            and not args.no_prompt
            and not args.no_history
            and sys.stdin.isatty()
        )
        if is_first_run:
            est_cost = len(target_ids) * 0.05 + 1.50
            est_minutes = max(5, len(target_ids) // args.concurrency)
            print(file=sys.stderr)
            print(
                f"  First run — about to extract narratives from {len(target_ids)} sessions.",
                file=sys.stderr,
            )
            print(
                f"    Estimated cost: ~${est_cost:.0f}  (cached re-runs are ~$1-2)",
                file=sys.stderr,
            )
            print(
                f"    Estimated wall clock: ~{est_minutes} min",
                file=sys.stderr,
            )
            print(
                f"    Token usage routes through your `claude` CLI auth (no separate billing).",
                file=sys.stderr,
            )
            if len(target_ids) > 150:
                print(
                    f"    Tip: heavy first run. Consider --limit 100 to bound cost; "
                    f"narratives cache so a second run with --limit 0 only pays the delta.",
                    file=sys.stderr,
                )
            try:
                reply = input("  Continue? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(" → cancelled.", file=sys.stderr)
                return 130
            if reply and reply not in ("y", "yes"):
                print("  Cancelled. (Pass --no-prompt to skip this check in scripts.)", file=sys.stderr)
                return 0
            print(file=sys.stderr)

        print(
            f"[3/4] Extracting per-session narratives via {backend.name}/{args.model} "
            f"(concurrency={args.concurrency})...",
            file=sys.stderr,
        )
        wanted = set(target_ids)
        grouped = load_all_sessions_events(events_path, wanted_ids=wanted)
        inputs = [(sid, grouped.get(sid, [])) for sid in target_ids]

        cached_count = new_count = error_count = 0

        def _progress(done: int, total: int, result):
            nonlocal cached_count, new_count, error_count
            if result.from_cache:
                cached_count += 1
                tag = "cache"
            elif result.error:
                error_count += 1
                tag = "error"
            else:
                new_count += 1
                tag = "new "
            print(
                f"  {done:>4}/{total} [{tag}] {result.session_id}",
                file=sys.stderr,
            )

        results = asyncio.run(
            extract_many(
                inputs,
                model=args.model,
                backend=backend,
                cache=cache,
                force=args.force,
                concurrency=args.concurrency,
                progress_cb=_progress,
            )
        )

        # Persist all narrative payloads (one file per session)
        for r in results:
            if r.error:
                continue
            out_path = narratives_dir / (r.session_id.replace(":", "__") + ".json")
            out_path.write_text(
                json.dumps(r.payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        print(
            f"       → {new_count} new, {cached_count} from cache, {error_count} errors. "
            f"Outputs in {narratives_dir}",
            file=sys.stderr,
        )

    # Load narratives back and synthesize
    narratives = load_narratives(narratives_dir)
    if len(narratives) < args.min_sessions:
        print(
            f"error: only {len(narratives)} narratives available for synthesis; "
            f"need ≥ {args.min_sessions}.",
            file=sys.stderr,
        )
        return 2

    prior_context: str | None = None
    if not args.no_history:
        store = HistoryStore(Path(args.history_dir).expanduser())
        prior_text, _ = store.summarize_for_prompt(n=args.prior_runs)
        if prior_text:
            prior_context = prior_text
            print(
                f"       → including {args.prior_runs} prior runs as context.",
                file=sys.stderr,
            )
    # Feed active experiments into prior context too — so synthesis doesn't
    # re-surface a pattern the user is already actively experimenting on.
    from .experiments import ExperimentStore, summarize_for_prompt as exp_summary

    exp_text = exp_summary(ExperimentStore())
    if exp_text:
        prior_context = (prior_context or "") + "\n\n## Active self-experiments\n" + exp_text
        print("       → including active experiments as context.", file=sys.stderr)

    print(f"[4/4] Cross-session synthesis via {backend.name}/{args.model}...", file=sys.stderr)
    t0 = time.monotonic()
    try:
        synthesis = run_synthesis(
            narratives,
            model=args.model,
            backend=backend,
            prior_context=prior_context,
        )
    except json.JSONDecodeError as exc:
        print(f"error: model returned non-JSON output: {exc}", file=sys.stderr)
        return 3
    print(f"       → done in {time.monotonic() - t0:.1f}s", file=sys.stderr)

    # Inject filter context so the dashboard masthead can show what was excluded
    synthesis.setdefault("meta", {})["filter_context"] = {
        "lookback_days": args.lookback_days,
        "min_events": args.min_events,
        **filter_breakdown,
        "narratives_in_synthesis": len(narratives),
    }

    # Compute week-over-week timeline + persist to history before rendering,
    # so the dashboard has both the timeline panel and the run slug it needs
    # for inline rating.
    if not args.no_history:
        try:
            store = HistoryStore(Path(args.history_dir).expanduser())
            synthesis["meta"]["timeline"] = _compute_timeline(synthesis, store)
            record = store.add_run(synthesis)
            synthesis["meta"]["run_slug"] = record.slug
            print(
                f"       → saved to history ({record.slug}).",
                file=sys.stderr,
            )
            # Audit log: run completed + every insight surfaced
            try:
                from .logbook import default as _logbook_default
                _lb = _logbook_default()
                _lb.log_run_completed(
                    run_slug=record.slug,
                    narratives_processed=len(narratives),
                    observations_count=len(synthesis.get("observations") or []),
                    behavioral_patterns_count=len(synthesis.get("behavioral_patterns") or []),
                    fabricated_refs=synthesis.get("meta", {}).get("fabricated_ref_count", 0),
                )
                from .history import _observation_key
                for o in synthesis.get("observations") or []:
                    _lb.log_insight(
                        run_slug=record.slug, kind="observation",
                        key=_observation_key(o), title=o.get("title", ""),
                        confidence=o.get("confidence"),
                        supporting_count=o.get("supporting_count", 0),
                        category_or_dimension=o.get("category"),
                    )
                for bp in synthesis.get("behavioral_patterns") or []:
                    _lb.log_insight(
                        run_slug=record.slug, kind="behavioral_pattern",
                        key=_observation_key(bp), title=bp.get("title", ""),
                        confidence=bp.get("confidence"),
                        supporting_count=bp.get("supporting_count", 0),
                        category_or_dimension=bp.get("dimension"),
                        non_comparative=bool(bp.get("non_comparative")),
                    )
            except Exception:
                pass
        except Exception as exc:
            print(f"warning: history save failed: {exc}", file=sys.stderr)

    output_path.write_text(
        json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"→ wrote {output_path}", file=sys.stderr)

    md_path = output_path.with_suffix(".md")
    md_path.write_text(render_synthesis_markdown(synthesis), encoding="utf-8")
    print(f"→ wrote {md_path}", file=sys.stderr)

    # Calibration audit — deterministic check on quantified claims. Catches
    # the failure modes the bake-off revealed (Gemini's 18-vs-22 undercount,
    # Codex's hallucinated comparatives, Claude's run-to-run variance).
    # No LLM call; runs in milliseconds.
    try:
        from .narratives.calibration import calibrate, render_calibration_text
        cal_report = calibrate(synthesis, narratives)
        cal_path = output_path.with_name(output_path.stem + "-calibration.json")
        cal_path.write_text(json.dumps(cal_report, indent=2), encoding="utf-8")
        summary = cal_report["summary"]
        n_checked = summary["total_quantified_claims_checked"]
        if n_checked:
            fails = summary["failed"]
            tag = f"{summary['passed']}/{n_checked} claims passed"
            if fails:
                tag += f" — {fails} quantified claim(s) flagged"
            print(f"→ calibration: {tag} (see {cal_path.name})", file=sys.stderr)
            if fails:
                # Surface each failure so the user sees it without opening a file.
                for f in cal_report["findings"]:
                    if f["verdict"] == "FAIL":
                        print(
                            f"    ✗ [{f['source']}] {f['explanation']}",
                            file=sys.stderr,
                        )
    except Exception as exc:
        print(f"  (calibration audit skipped: {exc})", file=sys.stderr)

    from .narratives.dashboard import render_dashboard

    # Load all cached narratives so Explore can show the user's full history,
    # not just the sessions in the current synthesis window.
    all_narratives_dict = {n["session_id"]: n for n in narratives}
    for cached_session_id in cache.list_cached():
        if cached_session_id in all_narratives_dict:
            continue
        path = cache.cache_dir / (cached_session_id.replace(":", "__") + ".json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("session_id"):
            all_narratives_dict[payload["session_id"]] = payload
    all_narratives = list(all_narratives_dict.values())
    if len(all_narratives) > len(narratives):
        print(
            f"       → Explore will show all {len(all_narratives)} cached sessions "
            f"({len(narratives)} in this synthesis window).",
            file=sys.stderr,
        )

    html_path = output_path.with_suffix(".html")
    html_path.write_text(
        render_dashboard(synthesis, narratives, all_narratives=all_narratives),
        encoding="utf-8",
    )
    print(
        f"→ wrote {html_path}  (open in any browser; rate inline)",
        file=sys.stderr,
    )

    if args.format == "json":
        print(json.dumps(synthesis, indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(render_synthesis_markdown(synthesis))
    else:
        print(render_synthesis_text(synthesis))
    return 0


def _rate_command(args: argparse.Namespace) -> int:
    store = HistoryStore(Path(args.history_dir).expanduser())
    record = store.latest_unrated() if not args.all else store.latest()
    if not record:
        if args.all:
            print("No history yet. Run `tessera run` first.", file=sys.stderr)
        else:
            print(
                "Nothing to rate — the most recent run is already rated. "
                "Use `--all` to rate the latest run again.",
                file=sys.stderr,
            )
        return 0

    payload = store.load_run(record.slug)
    if not payload:
        print(f"Could not load run {record.slug}. Skipping.", file=sys.stderr)
        return 1

    observations = payload.get("observations") or []
    if not observations:
        print(f"Run {record.slug} has no observations to rate.", file=sys.stderr)
        store.save_ratings(record.slug, [])
        return 0

    print("")
    print(f"Rating run from {record.timestamp[:10]} ({len(observations)} observations)")
    print("For each observation: [u]seful  [w]rong  [k]new already  [s]kip (default)")
    print("")

    ratings: list[dict] = []
    for idx, obs in enumerate(observations):
        title = obs.get("title") or f"§{idx + 1}"
        tags = []
        if obs.get("confidence"):
            tags.append(obs["confidence"])
        if obs.get("category"):
            tags.append(obs["category"])
        if obs.get("supporting_count"):
            tags.append(f"{obs['supporting_count']} sessions")
        tag_str = f"  [{' · '.join(tags)}]" if tags else ""
        print(f"§{idx + 1}  {title}{tag_str}")
        claim = (obs.get("claim") or "").strip()
        if claim:
            wrapped = claim if len(claim) < 300 else claim[:297].rstrip() + "…"
            print(f"     {wrapped}")
        next_action = (obs.get("next_action") or "").strip()
        if next_action:
            wrapped_na = next_action if len(next_action) < 200 else next_action[:197].rstrip() + "…"
            print(f"     → {wrapped_na}")
        try:
            answer = input("     u/w/k/s > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n(aborted, nothing saved)", file=sys.stderr)
            return 1
        rating = RATING_PROMPTS.get(answer, "skip")
        ratings.append(
            {
                "index": idx,
                "title": title,
                "key": _observation_key(obs),
                "rating": rating,
            }
        )
        print("")

    store.save_ratings(record.slug, ratings)
    summary = Counter(r["rating"] for r in ratings)
    print(f"→ saved {len(ratings)} ratings  ({dict(summary)})")
    return 0


SLASH_COMMAND_PREFIXES = ("/tessera", "/plugin", "/compact", "/clear", "/help", "/login", "/model", "/init")


def _compute_timeline(synthesis: dict, store: "HistoryStore") -> dict:
    """Compute week-over-week observation flow.

    Returns counts of:
      - new: observations with no `continues` reference (didn't exist last run)
      - continuing: observations with `continues` (carried over)
      - improving / stable / worsening: trend split of continuing
      - resolved: prior-run observation keys that no current observation continues
    """
    obs_list = synthesis.get("observations") or []
    new_count = 0
    continuing = 0
    improving = stable = worsening = 0
    continues_keys: set[str] = set()
    for o in obs_list:
        cont = o.get("continues")
        if cont:
            continuing += 1
            continues_keys.add(cont)
            trend = o.get("trend")
            if trend == "improving":
                improving += 1
            elif trend == "worsening":
                worsening += 1
            elif trend == "stable":
                stable += 1
        else:
            new_count += 1

    # Resolved: prior obs keys not referenced by any current `continues`.
    resolved: list[dict] = []
    prior = store.latest()
    if prior:
        prior_payload = store.load_run(prior.slug)
        if prior_payload:
            for prior_obs in prior_payload.get("observations") or []:
                key = _observation_key(prior_obs)
                if key not in continues_keys:
                    resolved.append(
                        {
                            "key": key,
                            "title": prior_obs.get("title") or "(untitled)",
                            "category": prior_obs.get("category"),
                        }
                    )

    return {
        "new": new_count,
        "continuing": continuing,
        "improving": improving,
        "stable": stable,
        "worsening": worsening,
        "resolved_count": len(resolved),
        "resolved": resolved[:8],  # cap for prompt/display size
        "prior_run_timestamp": prior.timestamp if prior else None,
    }


def _qualifying_session_ids(
    sessions_path: Path,
    *,
    min_events: int,
    lookback_days: int | None,
    max_age_days: int | None,
) -> tuple[list[dict], dict]:
    """Walk sessions.jsonl and return (qualifying, filter_breakdown).

    Filter breakdown:
        in_window: total sessions whose start falls in the lookback/max_age window
        dropped_short: in-window sessions with event_count < min_events
        dropped_slash: in-window sessions whose first prompt was a slash-command artifact
        dropped_undated: sessions without parseable timestamps
    """
    cutoff_lookback = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
        if lookback_days
        else None
    )
    cutoff_max_age = (
        datetime.now(timezone.utc) - timedelta(days=max_age_days)
        if max_age_days
        else None
    )

    by_id: dict[str, dict] = {}
    with sessions_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if s.get("trace_kind") == "subagent":
                continue
            agent = s.get("agent")
            sid = s.get("session_id")
            if not agent or not sid:
                continue
            key = f"{agent}:{sid}"
            event_count = s.get("event_count") or 0
            start = s.get("start_timestamp") or s.get("end_timestamp") or ""
            existing = by_id.get(key)
            if existing and existing["event_count"] >= event_count:
                continue
            by_id[key] = {
                "session_id": key,
                "start": start,
                "event_count": event_count,
                "first_prompt": s.get("first_prompt"),
            }

    qualifying: list[dict] = []
    in_window = 0
    dropped_short = 0
    dropped_slash = 0
    dropped_undated = 0
    for entry in by_id.values():
        try:
            start_dt = datetime.fromisoformat(entry["start"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            dropped_undated += 1
            continue
        if cutoff_lookback and start_dt < cutoff_lookback:
            continue
        if cutoff_max_age and start_dt < cutoff_max_age:
            continue
        in_window += 1
        if entry["event_count"] < min_events:
            dropped_short += 1
            continue
        fp = (entry.get("first_prompt") or "").strip().lower()
        if fp.startswith(SLASH_COMMAND_PREFIXES):
            dropped_slash += 1
            continue
        qualifying.append(entry)

    qualifying.sort(key=lambda e: e["start"], reverse=True)
    breakdown = {
        "in_window": in_window,
        "dropped_short": dropped_short,
        "dropped_slash": dropped_slash,
        "dropped_undated": dropped_undated,
        "total_raw_top_level_sessions": len(by_id),
    }
    return qualifying, breakdown


def _narrate_command(args: argparse.Namespace) -> int:
    from .narratives import (
        NarrativeCache,
        extract_many,
        load_all_sessions_events,
        load_session_events,
    )

    from .narratives import DEFAULT_CACHE_DIR

    backend, args.model = _resolve_backend_and_model(args)
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else DEFAULT_CACHE_DIR
    )
    cache = NarrativeCache(cache_dir)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tessera-narrate-") as work_dir:
        norm_dir = Path(work_dir)
        print(
            f"[1/4] Normalizing live traces (Claude: {args.claude_projects}, "
            f"Codex: {args.codex_sessions}, Gemini: {args.gemini_tmp})...",
            file=sys.stderr,
        )
        try:
            normalize_live_traces(
                norm_dir,
                claude_projects=Path(args.claude_projects).expanduser(),
                codex_sessions=Path(args.codex_sessions).expanduser(),
                gemini_tmp=Path(args.gemini_tmp).expanduser(),
                gemini_projects_json=Path(args.gemini_projects_json).expanduser(),
                max_text_chars=args.max_text_chars,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        sessions_path = norm_dir / "sessions.jsonl"
        events_path = norm_dir / "events.jsonl"

        # ---- Resolve which sessions to process ----
        if args.session_id:
            target_ids = [args.session_id]
            print(f"[2/4] Single-session mode: {args.session_id}", file=sys.stderr)
        else:
            print(
                f"[2/4] Selecting qualifying sessions "
                f"(min_events={args.min_events}, "
                f"lookback_days={args.lookback_days}, "
                f"max_age_days={args.max_age_days})...",
                file=sys.stderr,
            )
            qualifying, _filter_breakdown = _qualifying_session_ids(
                sessions_path,
                min_events=args.min_events,
                lookback_days=args.lookback_days,
                max_age_days=args.max_age_days,
            )
            if args.limit and args.limit > 0:
                qualifying = qualifying[: args.limit]
            target_ids = [q["session_id"] for q in qualifying]
            print(f"       → {len(target_ids)} sessions qualified.", file=sys.stderr)

        if not target_ids:
            print("No sessions to process. Exiting.", file=sys.stderr)
            return 0

        # ---- Load events ----
        t0 = time.monotonic()
        if args.session_id:
            events = load_session_events(events_path, args.session_id)
            inputs = [(args.session_id, events)]
        else:
            print(f"[3/4] Loading events for {len(target_ids)} sessions...", file=sys.stderr)
            wanted = set(target_ids)
            grouped = load_all_sessions_events(events_path, wanted_ids=wanted)
            inputs = [(sid, grouped.get(sid, [])) for sid in target_ids]
        print(
            f"       → events loaded in {time.monotonic() - t0:.1f}s "
            f"(total events: {sum(len(e) for _, e in inputs):,})",
            file=sys.stderr,
        )

        # ---- Extract narratives ----
        mode_label = "deterministic only" if args.dry_run else f"narratives via {args.model}"
        print(
            f"[4/4] Extracting {mode_label} (concurrency={args.concurrency})...",
            file=sys.stderr,
        )

        results = []
        cached_count = 0
        error_count = 0
        new_count = 0

        def _progress(done: int, total: int, result):
            nonlocal cached_count, error_count, new_count
            if result.from_cache:
                cached_count += 1
                tag = "[cache]"
            elif result.error:
                error_count += 1
                tag = "[error]"
            else:
                new_count += 1
                tag = "[new]"
            timing_parts = [f"{k.replace('_sec','')}={v}s" for k, v in result.timing.items()]
            timing_str = " ".join(timing_parts) if timing_parts else ""
            print(
                f"  {done:>4}/{total} {tag} {result.session_id}  {timing_str}"
                f"{' err=' + result.error if result.error else ''}",
                file=sys.stderr,
            )

        results = asyncio.run(
            extract_many(
                inputs,
                model=args.model,
                backend=backend,
                cache=cache,
                force=args.force,
                skip_llm=args.dry_run,
                concurrency=args.concurrency,
                progress_cb=_progress,
            )
        )

        # ---- Write outputs ----
        for r in results:
            out_path = output_dir / (r.session_id.replace(":", "__") + ".json")
            out_path.write_text(
                json.dumps(r.payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print("", file=sys.stderr)
        print(
            f"Done. {len(results)} sessions: {new_count} new, {cached_count} from cache, {error_count} errors.",
            file=sys.stderr,
        )
        print(f"Outputs in {output_dir}", file=sys.stderr)
        print(f"Cache in {cache_dir}", file=sys.stderr)

    return 0


def _rate_import_command(args: argparse.Namespace) -> int:
    """Import ratings produced by the dashboard's inline rate buttons.

    Reads JSON from stdin or a file. Schema:
        {"slug": "<run-slug>", "ratings": [{"index": 0, "key": "abc", "rating": "useful"}, ...]}

    Use slug "latest" to apply to the most recent run.
    """
    if args.input and args.input != "-":
        try:
            raw = Path(args.input).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        raw = sys.stdin.read()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 1

    slug = payload.get("slug")
    ratings = payload.get("ratings") or []
    if not slug or not isinstance(ratings, list):
        print("error: payload must have 'slug' and 'ratings' fields", file=sys.stderr)
        return 1

    store = HistoryStore(Path(args.history_dir).expanduser())
    if slug == "latest":
        latest = store.latest()
        if not latest:
            print("error: no runs in history yet", file=sys.stderr)
            return 1
        slug = latest.slug

    try:
        if not store.load_run(slug):
            print(f"error: run not found in history: {slug}", file=sys.stderr)
            return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cleaned: list[dict] = []
    valid = {"useful", "wrong", "known", "skip"}
    for r in ratings:
        if not isinstance(r, dict):
            continue
        if r.get("rating") not in valid:
            continue
        if not isinstance(r.get("index"), int):
            continue
        cleaned.append(
            {
                "index": r["index"],
                "title": r.get("title") or "",
                "key": r.get("key") or "",
                "rating": r["rating"],
            }
        )

    store.save_ratings(slug, cleaned)
    summary = Counter(c["rating"] for c in cleaned)

    # Audit log: every rating event (accepted = recommendation.accepted,
    # other = recommendation.declined)
    try:
        from .logbook import default as _logbook_default
        _lb = _logbook_default()
        for r in cleaned:
            _lb.log_rating(
                run_slug=slug,
                key=r.get("key") or "",
                title=r.get("title") or "",
                rating=r["rating"],
            )
    except Exception:
        pass

    # Self-experiment registration: any `useful` rating that resolves to a
    # behavioral_pattern in the rated run becomes an active experiment.
    # Lookup is by key (stable across runs) — works even if the user rates
    # in the dashboard without ever touching observation indices.
    from .experiments import ExperimentStore, register_from_ratings

    run_payload = store.load_run(slug) or {}
    behavioral_patterns = run_payload.get("behavioral_patterns") or []
    registered = []
    if behavioral_patterns:
        registered = register_from_ratings(
            cleaned, behavioral_patterns, slug, ExperimentStore()
        )

    msg = f"→ imported {len(cleaned)} ratings into run {slug}  ({dict(summary)})"
    if registered:
        msg += f"\n→ registered {len(registered)} new experiment(s) from useful behavioral_patterns:"
        for exp in registered:
            msg += f"\n    · {exp.title}"
    print(msg, file=sys.stderr)
    return 0


def _changelog_command(args: argparse.Namespace) -> int:
    from .changelog import changelog_for_current, render_changelog_text
    from .history import HistoryStore

    synth_path = Path(args.synthesis).expanduser()
    if not synth_path.exists():
        print(f"error: synthesis file not found: {synth_path}", file=sys.stderr)
        return 1
    try:
        synthesis = json.loads(synth_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: invalid synthesis JSON: {exc}", file=sys.stderr)
        return 1

    history = HistoryStore(Path(args.history_dir).expanduser())
    cl = changelog_for_current(synthesis, history)

    if args.format == "json":
        print(json.dumps(cl, indent=2, ensure_ascii=False))
    else:
        print(render_changelog_text(cl))
    return 0


def _experiments_command(args: argparse.Namespace) -> int:
    """List or operate on tracked experiments."""
    from .experiments import ExperimentStore

    store = (
        ExperimentStore(data_dir=Path(args.data_dir).expanduser())
        if args.data_dir
        else ExperimentStore()
    )

    if args.action == "list":
        statuses = ("active", "graduated", "inconclusive", "not_tried")
        any_shown = False
        for status in statuses:
            bucket = store.list(status=status)  # type: ignore[arg-type]
            if not bucket:
                continue
            any_shown = True
            print(f"\n{status.upper()} ({len(bucket)})", file=sys.stderr)
            for exp in bucket:
                print(
                    f"  · [{exp.dimension or '—':<20}] {exp.title}"
                    f"  (started {exp.started_at[:10]}, {len(exp.evaluations)} eval{'s' if len(exp.evaluations)!=1 else ''})",
                    file=sys.stderr,
                )
        if not any_shown:
            print(
                "no experiments tracked yet — rate a behavioral_pattern as "
                "[useful] in the dashboard to register one.",
                file=sys.stderr,
            )
        return 0

    if args.action == "show":
        if not args.id:
            print("error: --id required for `show`", file=sys.stderr)
            return 1
        exp = store.get(args.id)
        if not exp:
            print(f"error: no experiment with id {args.id!r}", file=sys.stderr)
            return 1
        from dataclasses import asdict

        print(json.dumps(asdict(exp), indent=2, ensure_ascii=False))
        return 0

    print(f"error: unknown action {args.action!r}", file=sys.stderr)
    return 1


def _dashboard_command(args: argparse.Namespace) -> int:
    """Render an HTML dashboard from an existing synthesis + narratives.

    Useful for re-rendering after a dashboard.py edit without re-running
    the LLM. Also pulls in any narratives cached at ``--cache-dir`` so the
    Explore tab shows the user's full history.
    """
    from .narratives.dashboard import write_dashboard
    from .narratives import DEFAULT_CACHE_DIR

    synthesis_path = Path(args.synthesis).expanduser()
    narratives_dir = Path(args.narratives_dir).expanduser()
    if not synthesis_path.exists():
        print(f"error: synthesis file not found: {synthesis_path}", file=sys.stderr)
        return 1
    if not narratives_dir.exists():
        print(f"error: narratives dir not found: {narratives_dir}", file=sys.stderr)
        return 1
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else synthesis_path.with_suffix(".html")
    )
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else DEFAULT_CACHE_DIR
    )
    target = write_dashboard(
        synthesis_path,
        narratives_dir,
        output_path,
        cache_dir=cache_dir if cache_dir.exists() else None,
    )
    print(f"→ wrote {target}  (open in any browser)", file=sys.stderr)
    return 0


def _enrich_outcomes_command(args: argparse.Namespace) -> int:
    from .narratives.outcomes import enrich_directory

    narratives_dir = Path(args.narratives_dir).expanduser()
    if not narratives_dir.exists():
        print(f"error: narratives dir not found: {narratives_dir}", file=sys.stderr)
        return 1

    print(
        f"Enriching narratives in {narratives_dir} "
        f"(gh={'off' if args.no_gh else 'on'}, force={args.force})...",
        file=sys.stderr,
    )
    summary = enrich_directory(
        narratives_dir,
        use_gh=not args.no_gh,
        force=args.force,
        max_age_days=args.max_age_days,
    )
    print(
        f"  {summary['enriched']} enriched, {summary['skipped_fresh']} skipped (fresh) "
        f"of {summary['total']} total.",
        file=sys.stderr,
    )
    if summary.get("signal_counts"):
        print("  Outcome signal distribution:", file=sys.stderr)
        for sig, count in summary["signal_counts"].items():
            print(f"    {sig}: {count}", file=sys.stderr)
    return 0


def _logbook_command(args: argparse.Namespace) -> int:
    """Read the append-only loop audit log.

    Three modes:
      --tail N        last N entries (default 20)
      --since DATE    entries since YYYY-MM-DD
      --event TYPE    filter by event type (run.started, insight.surfaced, ...)
      --json          raw jsonl output (no formatting, for piping into jq)
    """
    from .logbook import default as _logbook_default
    from collections import Counter

    lb = _logbook_default()

    events = list(lb.iter_events(
        event_type=args.event,
        since=args.since,
    ))
    if args.tail and args.tail > 0:
        events = events[-args.tail:]

    if args.json:
        for ev in events:
            print(json.dumps(ev, ensure_ascii=False))
        return 0

    if args.summary:
        # Group by event type + show counts
        counts = Counter(ev.get("event", "?") for ev in events)
        print(f"\nLogbook at {lb.path}")
        print(f"  Total entries: {len(events)}")
        print("  By event type:")
        for ev_type, count in counts.most_common():
            print(f"    {count:>5}  {ev_type}")
        # Acceptance rate
        accepted = counts.get("recommendation.accepted", 0)
        declined = counts.get("recommendation.declined", 0)
        if accepted + declined > 0:
            print(f"  Acceptance rate: {accepted}/{accepted + declined} ratings = {accepted/(accepted+declined)*100:.0f}%")
        # Graduation rate
        evald = counts.get("experiment.evaluated", 0)
        grad = counts.get("experiment.graduated", 0)
        nope = counts.get("experiment.marked_not_tried", 0)
        if evald > 0:
            print(f"  Evaluation outcomes: {grad} graduated, {nope} not-tried, {counts.get('experiment.marked_inconclusive', 0)} inconclusive (of {evald} total evals)")
        return 0

    # Default: human-readable timeline
    for ev in events:
        ts = (ev.get("ts") or "")[:19].replace("T", " ")
        et = ev.get("event", "?")
        if et == "run.started":
            print(f"  {ts}  run.started     {ev.get('run_slug', '?')[:16]}  lookback={ev.get('lookback_days')}d")
        elif et == "run.completed":
            print(f"  {ts}  run.completed   {ev.get('run_slug', '?')[:16]}  {ev.get('narratives_processed')} sessions → {ev.get('observations_count')} obs + {ev.get('behavioral_patterns_count')} bp")
        elif et == "insight.surfaced":
            tag = "bp" if ev.get("kind") == "behavioral_pattern" else "obs"
            nc = " · nc" if ev.get("non_comparative") else ""
            print(f"  {ts}  insight.{tag:<3}    [{ev.get('confidence') or '?':<6}·{(ev.get('category_or_dimension') or '?'):<22}·{ev.get('supporting_count', 0):>2}s]{nc}  {(ev.get('title') or '')[:80]}")
        elif et == "recommendation.accepted":
            print(f"  {ts}  ✓ accepted       {(ev.get('title') or '')[:80]}")
        elif et == "recommendation.declined":
            print(f"  {ts}  ✗ declined({ev.get('rating')})  {(ev.get('title') or '')[:80]}")
        elif et == "experiment.registered":
            print(f"  {ts}  exp.registered  [{ev.get('dimension') or '?':<22}]  {(ev.get('title') or '')[:80]}")
        elif et == "experiment.evaluated":
            adh = ev.get("adherence", "?")
            eff = ev.get("effect", "?")
            print(f"  {ts}  exp.evaluated   adherence={adh:<7} effect={eff:<8}  {(ev.get('title') or '')[:60]}")
        elif et in ("experiment.graduated", "experiment.marked_not_tried", "experiment.marked_inconclusive"):
            label = et.replace("experiment.", "")
            print(f"  {ts}  → {label:<18} {(ev.get('title') or '')[:70]}")
        else:
            print(f"  {ts}  {et:<22}  {json.dumps({k:v for k,v in ev.items() if k not in ('ts','event','schema_version','event_id')}, ensure_ascii=False)[:160]}")
    if not events:
        print("\n  (no logbook entries match — start by running `tessera run` or `tessera weekly`)\n")
    return 0


def _weekly_command(args: argparse.Namespace) -> int:
    """The closed loop, one command:

      1. tessera run --lookback-days 7 (narrate + synth + render)
      2. tessera evaluate-experiments (LLM-judges each active experiment
         against this week's narratives; transitions status)
      3. open the dashboard so the user can see results + accept up to N
         new experiments for next week

    Designed to be run from launchd / cron weekly. Single command means
    nothing for the user to remember; the loop is self-driving.
    """
    import subprocess
    from .history import HistoryStore
    from .experiments import ExperimentStore, evaluate_pending

    # 1. Run the analysis pass
    print("\n=== Tessera weekly · step 1/3: analysis ===\n", file=sys.stderr)
    run_args = argparse.Namespace(
        lookback_days=args.lookback_days,
        all_time=False,
        max_age_days=None,
        min_events=args.min_events,
        min_sessions=args.min_sessions,
        limit=args.limit,
        model=args.model,
        backend=getattr(args, "backend", "claude"),
        concurrency=args.concurrency,
        force=False,
        output=str(Path(args.output_dir).expanduser() / "synthesis.json"),
        narratives_dir=str(Path(args.output_dir).expanduser() / "narratives"),
        cache_dir=None,
        format="text",
        history_dir=args.history_dir,
        prior_runs=3,
        no_history=False,
        no_prompt=True,  # cron-friendly
        claude_projects=str(DEFAULT_CLAUDE_PROJECTS),
        codex_sessions=str(DEFAULT_CODEX_SESSIONS),
        gemini_tmp=str(DEFAULT_GEMINI_TMP),
        gemini_projects_json=str(DEFAULT_GEMINI_PROJECTS_JSON),
        max_text_chars=2000,
        project=None,
    )
    Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
    rc = _run_command(run_args)
    if rc != 0:
        print(f"weekly: analysis stage failed (exit {rc}) — aborting evaluation + open", file=sys.stderr)
        return rc

    # 2. Evaluate active experiments against this week's narratives
    print("\n=== Tessera weekly · step 2/3: experiment evaluation ===\n", file=sys.stderr)
    exp_store = ExperimentStore()
    active = exp_store.list("active")
    if not active:
        print("  No active experiments to evaluate.", file=sys.stderr)
    else:
        print(f"  Evaluating {len(active)} active experiment(s)...", file=sys.stderr)
        # Load this run's narratives
        from .narratives.synthesis import load_narratives
        narratives_dir = Path(args.output_dir).expanduser() / "narratives"
        narratives = load_narratives(narratives_dir)
        if args.skip_eval:
            print("  --skip-eval set; using offline heuristic (no LLM call).", file=sys.stderr)
            summary = evaluate_pending(narratives, exp_store)
        else:
            from .experiment_evaluator import make_callable
            eval_backend, eval_model = _resolve_backend_and_model(args)
            print(f"  Evaluator: {eval_backend.name}/{eval_model}", file=sys.stderr)
            summary = evaluate_pending(
                narratives,
                exp_store,
                llm_evaluator=make_callable(model=eval_model, backend=eval_backend),
            )
        print(f"  Evaluated: {summary['evaluated']}", file=sys.stderr)
        if summary.get("graduated"):
            print(f"  ✓ Graduated: {len(summary['graduated'])}", file=sys.stderr)
            for eid in summary["graduated"]:
                exp = exp_store.get(eid)
                if exp:
                    print(f"      · {exp.title}", file=sys.stderr)
        if summary.get("marked_not_tried"):
            print(f"  ✗ Marked not tried: {len(summary['marked_not_tried'])}", file=sys.stderr)
            for eid in summary["marked_not_tried"]:
                exp = exp_store.get(eid)
                if exp:
                    print(f"      · {exp.title}", file=sys.stderr)
        if summary.get("marked_inconclusive"):
            print(f"  · Inconclusive after 2+ neutral evals: {len(summary['marked_inconclusive'])}", file=sys.stderr)
        if summary.get("still_active"):
            print(f"  · Still active: {len(summary['still_active'])}", file=sys.stderr)
        if summary.get("skipped_no_post_baseline_data"):
            print(
                f"  · Skipped (no post-baseline narratives yet): {len(summary['skipped_no_post_baseline_data'])}",
                file=sys.stderr,
            )

    # 3. Re-render the dashboard so it reflects the evaluator's verdicts,
    #    then open it for review + accept-experiments-by-rating.
    print("\n=== Tessera weekly · step 3/3: dashboard ===\n", file=sys.stderr)
    output_dir = Path(args.output_dir).expanduser()
    synthesis_path = output_dir / "synthesis.json"
    narratives_dir = output_dir / "narratives"
    dashboard = output_dir / "synthesis.html"
    if synthesis_path.exists() and narratives_dir.exists():
        try:
            from .narratives.dashboard import write_dashboard
            write_dashboard(
                synthesis_path,
                narratives_dir,
                dashboard,
                cache_dir=DEFAULT_CACHE_DIR if DEFAULT_CACHE_DIR.exists() else None,
            )
            print("  Re-rendered dashboard with latest experiment verdicts.", file=sys.stderr)
        except Exception as exc:
            print(f"  Dashboard re-render failed ({exc}); using step-1 render.", file=sys.stderr)
    if dashboard.exists() and not args.no_open:
        try:
            subprocess.run(["open", str(dashboard)], check=False, timeout=5)
            print(f"  Opened {dashboard}", file=sys.stderr)
        except Exception as exc:
            print(f"  Could not auto-open ({exc}); open manually: {dashboard}", file=sys.stderr)
    else:
        print(f"  Dashboard at: {dashboard}", file=sys.stderr)

    print(
        "\n  Click [useful] on behavioral patterns you commit to trying this week."
        "\n  Click SAVE → paste the one-liner in terminal to register experiments."
        "\n  Next week's `tessera weekly` will evaluate whether they worked.",
        file=sys.stderr,
    )
    return 0


def _doctor_command(args: argparse.Namespace) -> int:
    """Pre-flight diagnostic: shows what tessera can see + estimates a run.

    Goal: a first-time user runs `tessera doctor` and sees, before spending
    a dollar, exactly which agent traces are visible, how many sessions
    qualify, what the LLM cost looks like, and which CLIs are missing.
    """
    import shutil
    import subprocess
    from . import normalizers as _norm
    from .history import HistoryStore

    warnings = _norm.initialize()
    print("\nTessera diagnostic")
    print("=" * 60)

    # 1. LLM backends — at least one must be available
    auto_default = get_backend().name
    print(
        f"\nLLM backends (auto-default: {auto_default}; "
        f"override via --backend {{claude,codex,gemini,antigravity}}):"
    )
    backend_status = {}
    for bname in list_backends():
        b = get_backend(bname)
        bin_path = shutil.which(b.cli_binary) if b.cli_binary else None
        if not bin_path:
            print(f"  - {bname:<7} `{b.cli_binary}` CLI not on PATH (default model: {b.default_model})")
            backend_status[bname] = False
            continue
        try:
            ver = subprocess.run(
                [b.cli_binary, "--version"], capture_output=True, text=True, timeout=5
            )
            ver_str = (ver.stdout or ver.stderr).strip().splitlines()[0] if ver.returncode == 0 else "(version check failed)"
        except Exception:
            ver_str = "(version check failed)"
        print(f"  ✓ {bname:<7} {bin_path}  ({ver_str}, default model: {b.default_model})")
        backend_status[bname] = True
    claude_ok = backend_status.get("claude", False)
    if not any(backend_status.values()):
        print("\n  ✗ No LLM backend available — install at least one CLI before running tessera.")
        print("    claude: https://docs.claude.com/en/docs/claude-code/setup")
        print("    codex:  https://github.com/openai/codex")
        print("    gemini: https://github.com/google-gemini/gemini-cli")

    # 2. Optional: gh CLI for richer outcome enrichment
    print("\nOutcome enrichment (gh CLI, optional):")
    gh_path = shutil.which("gh")
    if not gh_path:
        print("  - gh CLI not installed → PR outcome data will be skipped (git-only mode still works).")
    else:
        try:
            gh_auth = subprocess.run(
                ["gh", "auth", "status"], capture_output=True, text=True, timeout=5
            )
            authed = gh_auth.returncode == 0
        except Exception:
            authed = False
        print(f"  {'✓' if authed else '⚠'} {gh_path}  ({'authenticated' if authed else 'not authenticated — `gh auth login`'})")

    # 3. Registered normalizers + how many sessions each sees
    print(f"\nRegistered agent normalizers ({len(_norm.get_all())}):")
    total_sessions = 0
    for n in _norm.get_all():
        src_label = f"[{n.source}]" if n.source != "builtin" else ""
        exists = n.default_source.exists()
        marker = "✓" if exists else "—"
        sessions_estimate = ""
        if exists:
            try:
                # Cheap upper-bound count by file pattern per known agent
                if n.name == "claude":
                    cnt = sum(1 for _ in n.default_source.rglob("*.jsonl"))
                elif n.name == "codex":
                    cnt = sum(1 for _ in n.default_source.rglob("rollout-*.jsonl"))
                elif n.name == "gemini":
                    cnt = sum(1 for p in n.default_source.rglob("*") if p.is_file())
                else:
                    cnt = sum(1 for p in n.default_source.rglob("*") if p.is_file())
                sessions_estimate = f"  ~{cnt} raw trace files"
                total_sessions += cnt
            except Exception:
                sessions_estimate = "  (count failed)"
        print(f"  {marker} {n.name:<10} {src_label:<12} {n.default_source}{sessions_estimate}")
        if n.description:
            print(f"      {n.description}")

    # 4. Loader warnings (failed user-dir / entry-point imports)
    if warnings:
        print("\n⚠  Normalizer loader warnings:")
        for w in warnings:
            print(f"  {w}")

    # 5. History store + cache state
    print("\nLocal state:")
    history = HistoryStore()
    runs = history._read_index()
    print(f"  History: {len(runs)} prior run(s) at {history.data_dir}")
    cache_dir = Path.home() / ".cache" / "tessera" / "narratives"
    cached_narratives = sum(1 for _ in cache_dir.glob("*.json")) if cache_dir.exists() else 0
    print(f"  Narrative cache: {cached_narratives} cached at {cache_dir}")

    # 6. Cost estimate
    print("\nRough cost estimate for `tessera run --lookback-days 30 --min-events 10`:")
    print("  (this is an upper-bound — most sessions hit the cache after the first run)")
    # Heuristic: assume ~30-40% of raw trace files qualify after min_events filter,
    # then ~$0.05/session for narrative + ~$1 for synthesis
    qualifying_est = max(1, int(total_sessions * 0.35))
    cold_cost = qualifying_est * 0.05 + 1.50
    cached_cost = 0.50 + 1.50
    print(f"  Cold run (all sessions fresh):     ~${cold_cost:.0f}  ({qualifying_est} sessions × ~$0.05 + ~$1.50 synth)")
    print(f"  Cached run (most narratives hit):  ~${cached_cost:.0f}")
    print(f"  Wall clock: ~{max(5, qualifying_est // 10)} min cold, ~5 min cached")

    print()
    if not claude_ok:
        print("⚠  Fix the missing `claude` CLI before running anything else.")
        return 1
    print("Looks good — run `tessera run --lookback-days 30 --min-events 10`")
    print("(add `--limit 100` to bound cost on a heavy corpus)")
    return 0


def _eval_command(args: argparse.Namespace) -> int:
    from .narratives.calibration import calibrate, render_calibration_text
    from .narratives.eval import (
        evaluate_narratives,
        evaluate_synthesis,
        render_eval_text,
    )
    from .narratives.synthesis import load_narratives

    narratives_dir = Path(args.narratives_dir).expanduser()
    if not narratives_dir.exists():
        print(f"error: narratives dir not found: {narratives_dir}", file=sys.stderr)
        return 1

    narrative_eval = evaluate_narratives(narratives_dir)

    synthesis_eval = None
    calibration_report = None
    if args.synthesis:
        synthesis_path = Path(args.synthesis).expanduser()
        if not synthesis_path.exists():
            print(f"error: synthesis file not found: {synthesis_path}", file=sys.stderr)
            return 1
        synthesis_eval = evaluate_synthesis(synthesis_path)
        # Calibration: grade quantified claims against narratives.
        # Deterministic — no LLM call.
        try:
            synthesis_data = json.loads(synthesis_path.read_text(encoding="utf-8"))
            narratives = load_narratives(narratives_dir)
            calibration_report = calibrate(synthesis_data, narratives)
        except Exception as exc:
            print(f"  (calibration audit skipped: {exc})", file=sys.stderr)

    if args.format == "json":
        result = {"narratives": narrative_eval}
        if synthesis_eval:
            result["synthesis"] = synthesis_eval
        if calibration_report:
            result["calibration"] = calibration_report
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render_eval_text(narrative_eval, synthesis_eval))
        if calibration_report:
            print()
            print(render_calibration_text(calibration_report))
    return 0


def _synthesize_command(args: argparse.Namespace) -> int:
    from .narratives.render import render_synthesis_markdown, render_synthesis_text
    from .narratives.synthesis import load_narratives, synthesize

    backend, args.model = _resolve_backend_and_model(args)
    narratives_dir = Path(args.narratives_dir).expanduser()
    if not narratives_dir.exists():
        print(f"error: narratives dir not found: {narratives_dir}", file=sys.stderr)
        return 1
    print(f"[1/2] Loading narratives from {narratives_dir}...", file=sys.stderr)
    narratives = load_narratives(narratives_dir)
    print(f"       → {len(narratives)} narratives loaded.", file=sys.stderr)
    if args.project:
        print(f"       → filtering on project substring: {args.project!r}", file=sys.stderr)

    prior_context: str | None = None
    if not args.no_history:
        store = HistoryStore(Path(args.history_dir).expanduser())
        prior_text, _ = store.summarize_for_prompt(n=args.prior_runs)
        if prior_text:
            prior_context = prior_text
            print(
                f"       → including {args.prior_runs} prior runs as context.",
                file=sys.stderr,
            )
    # Feed active experiments into prior context too — so synthesis doesn't
    # re-surface a pattern the user is already actively experimenting on.
    from .experiments import ExperimentStore, summarize_for_prompt as exp_summary

    exp_text = exp_summary(ExperimentStore())
    if exp_text:
        prior_context = (prior_context or "") + "\n\n## Active self-experiments\n" + exp_text
        print("       → including active experiments as context.", file=sys.stderr)

    print(f"[2/2] Asking {backend.name}/{args.model} for cross-session synthesis...", file=sys.stderr)
    t0 = time.monotonic()
    result = synthesize(
        narratives,
        model=args.model,
        backend=backend,
        project_filter=args.project,
        prior_context=prior_context,
    )
    print(f"       → done in {time.monotonic() - t0:.1f}s", file=sys.stderr)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"→ wrote {output_path}", file=sys.stderr)

    md_path = output_path.with_suffix(".md")
    md_path.write_text(render_synthesis_markdown(result), encoding="utf-8")
    print(f"→ wrote {md_path}", file=sys.stderr)

    from .narratives.dashboard import render_dashboard

    html_path = output_path.with_suffix(".html")
    html_path.write_text(render_dashboard(result, narratives), encoding="utf-8")
    print(f"→ wrote {html_path}  (open in any browser)", file=sys.stderr)

    if not args.no_history:
        store = HistoryStore(Path(args.history_dir).expanduser())
        record = store.add_run(result)
        print(
            f"→ saved to history ({record.slug}). Rate with `tessera rate`.",
            file=sys.stderr,
        )

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(render_synthesis_markdown(result))
    else:
        print(render_synthesis_text(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tessera",
        description="Turn your local Claude Code, Codex, and Gemini CLI traces into evidence-grounded observations via an LLM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    run = sub.add_parser(
        "run",
        help="Full pipeline: normalize → narrate per-session → synthesize cross-session → save and render.",
    )
    run.add_argument("--lookback-days", type=int, default=30,
                     help="Window in days. Use 0 for all-time (alias of --all-time).")
    run.add_argument("--all-time", action="store_true",
                     help="Synthesize across all sessions ever extracted, ignoring --lookback-days.")
    run.add_argument("--min-events", type=int, default=20,
                     help="Skip sessions with fewer than this many events.")
    run.add_argument("--limit", type=int, default=0,
                     help="Process at most N sessions (newest first). 0 = no limit.")
    run.add_argument("--min-sessions", type=int, default=6,
                     help="Abort if fewer than N sessions qualify.")
    run.add_argument("--backend", default=None, choices=list_backends(),
                     help="LLM backend: claude (Anthropic Claude SDK), codex (OpenAI Codex CLI), "
                          "gemini (Google Gemini CLI). Default: claude.")
    run.add_argument("--model", default=DEFAULT_NARRATIVE_MODEL,
                     help="Model id used for both narrate and synthesize stages. "
                          "Default picks the backend's recommended model.")
    run.add_argument("--concurrency", type=int, default=10,
                     help="Per-session narrative extraction concurrency.")
    run.add_argument("--force", action="store_true",
                     help="Bypass the narrative cache (still writes).")
    run.add_argument(
        "--output",
        default="./synthesis.json",
        help="Where to write the synthesis JSON. The .md sibling is also written.",
    )
    run.add_argument(
        "--narratives-dir",
        default="./narratives",
        help="Where per-session narrative JSON files are saved.",
    )
    run.add_argument(
        "--cache-dir",
        default=None,
        help=f"Override narrative cache dir (default {DEFAULT_CACHE_DIR}).",
    )
    run.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    run.add_argument("--claude-projects", default=str(DEFAULT_CLAUDE_PROJECTS))
    run.add_argument("--codex-sessions", default=str(DEFAULT_CODEX_SESSIONS))
    run.add_argument("--gemini-tmp", default=str(DEFAULT_GEMINI_TMP))
    run.add_argument(
        "--gemini-projects-json",
        default=str(DEFAULT_GEMINI_PROJECTS_JSON),
        help="Gemini CLI's projects.json that maps path hashes to real paths.",
    )
    run.add_argument("--max-text-chars", type=int, default=2000,
                     help="Truncation limit for normalized message + tool preview text.")
    run.add_argument(
        "--history-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Where to persist past runs + ratings.",
    )
    run.add_argument("--prior-runs", type=int, default=3,
                     help="How many prior runs to feed back into the synthesis prompt.")
    run.add_argument("--no-history", action="store_true",
                     help="Skip prior-run context and don't save to history.")
    run.add_argument("--no-prompt", action="store_true",
                     help="Skip the first-run cost-estimate confirmation prompt "
                     "(useful for cron/scripts).")
    run.set_defaults(func=_run_command)

    # ---- rate ----
    rate = sub.add_parser(
        "rate",
        help="Rate the most recent run's observations (feeds into the next run).",
    )
    rate.add_argument(
        "--history-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Where past runs + ratings are stored.",
    )
    rate.add_argument(
        "--all",
        action="store_true",
        help="Rate the most recent run even if it's already been rated.",
    )
    rate.set_defaults(func=_rate_command)

    # ---- rate-import ----
    rate_import = sub.add_parser(
        "rate-import",
        help="Import ratings from the dashboard's inline rate buttons (JSON via stdin or file).",
    )
    rate_import.add_argument(
        "--input", default="-",
        help="Path to JSON file, or - for stdin (default: stdin).",
    )
    rate_import.add_argument(
        "--history-dir", default=str(DEFAULT_DATA_DIR),
        help="Where past runs + ratings are stored.",
    )
    rate_import.set_defaults(func=_rate_import_command)

    # ---- narrate ----
    narrate = sub.add_parser(
        "narrate",
        help="Per-session narrative extraction (deterministic metadata + LLM story).",
    )
    narrate.add_argument(
        "--session-id",
        default=None,
        help="Process only this session ('<agent>:<uuid>'). Skips filtering.",
    )
    narrate.add_argument("--lookback-days", type=int, default=None,
                        help="Only process sessions started within this window. Default: no cap.")
    narrate.add_argument("--max-age-days", type=int, default=None,
                        help="Equivalent to --lookback-days; preferred name when doing historical backfills.")
    narrate.add_argument("--min-events", type=int, default=20,
                        help="Skip sessions with fewer than this many events.")
    narrate.add_argument("--limit", type=int, default=0,
                        help="Process at most N sessions (newest first). 0 = no limit.")
    narrate.add_argument("--backend", default=None, choices=list_backends(),
                        help="LLM backend (claude | codex | gemini). Default: claude.")
    narrate.add_argument("--model", default=DEFAULT_NARRATIVE_MODEL,
                        help="LLM model for narrative extraction. "
                             "Default picks the backend's recommended model.")
    narrate.add_argument("--concurrency", type=int, default=5,
                        help="Number of sessions to extract in parallel.")
    narrate.add_argument("--force", action="store_true",
                        help="Bypass cache reads (still writes).")
    narrate.add_argument("--dry-run", action="store_true",
                        help="Deterministic only — skip the LLM call.")
    narrate.add_argument(
        "--output-dir",
        default="./narratives",
        help="Where to write per-session JSON outputs.",
    )
    narrate.add_argument(
        "--cache-dir",
        default=None,
        help="Override the default cache dir (~/.cache/tessera/narratives).",
    )
    narrate.add_argument("--claude-projects", default=str(DEFAULT_CLAUDE_PROJECTS))
    narrate.add_argument("--codex-sessions", default=str(DEFAULT_CODEX_SESSIONS))
    narrate.add_argument("--gemini-tmp", default=str(DEFAULT_GEMINI_TMP))
    narrate.add_argument("--gemini-projects-json", default=str(DEFAULT_GEMINI_PROJECTS_JSON))
    narrate.add_argument("--max-text-chars", type=int, default=2000)
    narrate.set_defaults(func=_narrate_command)

    # ---- synthesize ----
    synthesize = sub.add_parser(
        "synthesize",
        help="Cross-session synthesis from existing per-session narratives.",
    )
    synthesize.add_argument(
        "--narratives-dir",
        default="./narratives",
        help="Directory of per-session narrative JSON files.",
    )
    synthesize.add_argument("--backend", default=None, choices=list_backends(),
                            help="LLM backend (claude | codex | gemini). Default: claude.")
    synthesize.add_argument("--model", default=DEFAULT_NARRATIVE_MODEL,
                            help="Default picks the backend's recommended model.")
    synthesize.add_argument(
        "--project",
        default=None,
        help="Substring filter on project_label. Default: include all projects.",
    )
    synthesize.add_argument(
        "--output",
        default="./synthesis.json",
        help="Where to write the synthesis JSON. The .md sibling is also written.",
    )
    synthesize.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    synthesize.add_argument(
        "--history-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Where past runs + ratings are stored.",
    )
    synthesize.add_argument("--prior-runs", type=int, default=3,
                           help="How many prior runs to feed back as context.")
    synthesize.add_argument("--no-history", action="store_true",
                           help="Skip prior-run context and don't save to history.")
    synthesize.set_defaults(func=_synthesize_command)

    # ---- eval ----
    eval_parser = sub.add_parser(
        "eval",
        help="Quality metrics for a narratives directory and/or synthesis output.",
    )
    eval_parser.add_argument(
        "--narratives-dir",
        default="./narratives",
        help="Directory of per-session narrative JSON files.",
    )
    eval_parser.add_argument(
        "--synthesis",
        default=None,
        help="Optional synthesis.json to also evaluate.",
    )
    eval_parser.add_argument("--format", choices=["text", "json"], default="text")
    eval_parser.set_defaults(func=_eval_command)

    # ---- doctor ----
    doctor = sub.add_parser(
        "doctor",
        help="Diagnose tessera setup — checks LLM CLI, agent traces, "
        "registered normalizers (built-in + user-added), prior runs, "
        "and gives a cost estimate for the next run. Run this before "
        "`tessera run` on a new machine.",
    )
    doctor.set_defaults(func=_doctor_command)

    # ---- logbook (append-only audit) ----
    log = sub.add_parser(
        "logbook",
        help="View the append-only audit log of the self-improving loop "
        "(runs, insights surfaced, recommendations accepted/declined, "
        "experiments registered/evaluated/graduated). "
        "Stored at ~/.config/tessera/logbook.jsonl; override via "
        "TESSERA_LOGBOOK env var.",
    )
    log.add_argument("--tail", type=int, default=20,
                    help="Show last N entries (default 20).")
    log.add_argument("--since", default=None,
                    help="Show entries since YYYY-MM-DD (ISO 8601).")
    log.add_argument("--event", default=None,
                    help="Filter by event type (e.g. insight.surfaced, "
                    "recommendation.accepted, experiment.evaluated).")
    log.add_argument("--summary", action="store_true",
                    help="Show aggregate counts + acceptance/graduation rates.")
    log.add_argument("--json", action="store_true",
                    help="Raw JSONL output for piping into jq.")
    log.set_defaults(func=_logbook_command)

    # ---- weekly (the closed loop) ----
    weekly = sub.add_parser(
        "weekly",
        help="The closed-loop weekly heartbeat: run analysis → evaluate "
        "active experiments against this week's data → open dashboard. "
        "Designed for launchd / cron. After review, rate behavioral "
        "patterns [useful] to commit experiments for the following week.",
    )
    weekly.add_argument("--lookback-days", type=int, default=7,
                        help="Window to analyze (default: 7 = last week).")
    weekly.add_argument("--min-events", type=int, default=10,
                        help="Min normalized events for a session to qualify.")
    weekly.add_argument("--min-sessions", type=int, default=5,
                        help="Abort if fewer than this many sessions qualify.")
    weekly.add_argument("--limit", type=int, default=0,
                        help="Cap at N most-recent qualifying sessions (0 = no cap).")
    weekly.add_argument("--backend", default=None, choices=list_backends(),
                        help="LLM backend for all three stages (narrate + synth + eval). "
                             "Options: claude | codex | gemini. Default: claude.")
    weekly.add_argument("--model", default=DEFAULT_NARRATIVE_MODEL,
                        help="LLM model for narration + synthesis + experiment eval. "
                             "Default picks the backend's recommended model.")
    weekly.add_argument("--concurrency", type=int, default=3,
                        help="Parallel per-session narrative extractions. "
                             "Default 3 instead of 10 because CLI backends "
                             "(antigravity, codex, gemini) share a single "
                             "auth token; 10 parallel subprocess each trying "
                             "to refresh OAuth at once can cascade into "
                             "browser popups. 3 keeps the wall clock reasonable "
                             "while serializing token reads.")
    weekly.add_argument("--output-dir", default="~/tessera-weekly",
                        help="Where the run's synthesis + narratives + dashboard land.")
    weekly.add_argument("--history-dir", default=str(DEFAULT_DATA_DIR),
                        help="History store location.")
    weekly.add_argument("--skip-eval", action="store_true",
                        help="Skip the LLM evaluator pass (use offline heuristic instead).")
    weekly.add_argument("--no-open", action="store_true",
                        help="Don't auto-open the dashboard (useful for headless / cron).")
    weekly.set_defaults(func=_weekly_command)

    # ---- changelog ----
    changelog = sub.add_parser(
        "changelog",
        help="Show what changed since the last run — new, escalating, "
        "improving, resolved, regressed observations and behavioral patterns. "
        "Pure-data diff, no LLM call.",
    )
    changelog.add_argument(
        "--synthesis",
        default="./synthesis.json",
        help="Current synthesis to diff against history.",
    )
    changelog.add_argument(
        "--history-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"History store. Default {DEFAULT_DATA_DIR}.",
    )
    changelog.add_argument("--format", choices=["text", "json"], default="text")
    changelog.set_defaults(func=_changelog_command)

    # ---- experiments ----
    exp = sub.add_parser(
        "experiments",
        help="List or inspect self-experiments (auto-registered when you rate "
        "a behavioral_pattern as [useful] in the dashboard).",
    )
    exp.add_argument(
        "action",
        nargs="?",
        choices=["list", "show"],
        default="list",
        help="What to do. Default: list.",
    )
    exp.add_argument("--id", default=None, help="Experiment id (required for `show`).")
    exp.add_argument(
        "--data-dir",
        default=None,
        help="Override the experiments data dir. Default ~/.config/tessera/experiments.",
    )
    exp.set_defaults(func=_experiments_command)

    # ---- enrich-outcomes ----
    enrich = sub.add_parser(
        "enrich-outcomes",
        help="For each narrative, look up what actually happened to the work "
        "(branch lifecycle, file churn, PR state via git + gh CLI). Idempotent; "
        "writes outcome data inline. No LLM calls — fast and free.",
    )
    enrich.add_argument(
        "--narratives-dir",
        default="./narratives",
        help="Directory of per-session narrative JSON files.",
    )
    enrich.add_argument(
        "--no-gh",
        action="store_true",
        help="Skip GitHub PR lookups (git-only mode). Use if you don't have gh "
        "installed/authenticated, or want a fully offline run.",
    )
    enrich.add_argument(
        "--force",
        action="store_true",
        help="Re-lookup even narratives that already have a recent outcome.",
    )
    enrich.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Skip narratives whose outcome was looked up more recently than "
        "this many days ago (unless --force). Default 7.",
    )
    enrich.set_defaults(func=_enrich_outcomes_command)

    # ---- dashboard ----
    dash = sub.add_parser(
        "dashboard",
        help="Render the static HTML dashboard from an existing synthesis + narratives.",
    )
    dash.add_argument(
        "--synthesis",
        default="./synthesis.json",
        help="Path to synthesis JSON.",
    )
    dash.add_argument(
        "--narratives-dir",
        default="./narratives",
        help="Directory of per-session narrative JSON files.",
    )
    dash.add_argument(
        "--output",
        default=None,
        help="Where to write the HTML. Default: synthesis.html alongside the JSON.",
    )
    dash.add_argument(
        "--cache-dir",
        default=None,
        help=f"Narrative cache to source the all-time Explore view from. Default {DEFAULT_CACHE_DIR}.",
    )
    dash.set_defaults(func=_dashboard_command)

    args = parser.parse_args(argv)
    # Map --max-age-days into --lookback-days for the narrate flow (they're aliases).
    # Only relevant when the subcommand actually has both — newer subcommands
    # (e.g. enrich-outcomes) reuse --max-age-days for unrelated meaning.
    if (
        hasattr(args, "max_age_days")
        and hasattr(args, "lookback_days")
        and args.max_age_days
        and not args.lookback_days
    ):
        args.lookback_days = args.max_age_days
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
