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

        print(
            f"[3/4] Extracting per-session narratives via {args.model} "
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

    print(f"[4/4] Cross-session synthesis via {args.model}...", file=sys.stderr)
    t0 = time.monotonic()
    try:
        synthesis = run_synthesis(
            narratives,
            model=args.model,
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
        except Exception as exc:
            print(f"warning: history save failed: {exc}", file=sys.stderr)

    output_path.write_text(
        json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"→ wrote {output_path}", file=sys.stderr)

    md_path = output_path.with_suffix(".md")
    md_path.write_text(render_synthesis_markdown(synthesis), encoding="utf-8")
    print(f"→ wrote {md_path}", file=sys.stderr)

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
    print(
        f"→ imported {len(cleaned)} ratings into run {slug}  ({dict(summary)})",
        file=sys.stderr,
    )
    return 0


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


def _eval_command(args: argparse.Namespace) -> int:
    from .narratives.eval import (
        evaluate_narratives,
        evaluate_synthesis,
        render_eval_text,
    )

    narratives_dir = Path(args.narratives_dir).expanduser()
    if not narratives_dir.exists():
        print(f"error: narratives dir not found: {narratives_dir}", file=sys.stderr)
        return 1

    narrative_eval = evaluate_narratives(narratives_dir)

    synthesis_eval = None
    if args.synthesis:
        synthesis_path = Path(args.synthesis).expanduser()
        if not synthesis_path.exists():
            print(f"error: synthesis file not found: {synthesis_path}", file=sys.stderr)
            return 1
        synthesis_eval = evaluate_synthesis(synthesis_path)

    if args.format == "json":
        result = {"narratives": narrative_eval}
        if synthesis_eval:
            result["synthesis"] = synthesis_eval
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render_eval_text(narrative_eval, synthesis_eval))
    return 0


def _synthesize_command(args: argparse.Namespace) -> int:
    from .narratives.render import render_synthesis_markdown, render_synthesis_text
    from .narratives.synthesis import load_narratives, synthesize

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

    print(f"[2/2] Asking {args.model} for cross-session synthesis...", file=sys.stderr)
    t0 = time.monotonic()
    result = synthesize(
        narratives,
        model=args.model,
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
    run.add_argument("--model", default=DEFAULT_NARRATIVE_MODEL,
                     help="Claude model used for both narrate and synthesize stages.")
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
    narrate.add_argument("--model", default=DEFAULT_NARRATIVE_MODEL,
                        help="LLM model for narrative extraction.")
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
    synthesize.add_argument("--model", default=DEFAULT_NARRATIVE_MODEL)
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
    # Map --max-age-days into --lookback-days for the narrate flow (they're aliases)
    if hasattr(args, "max_age_days") and args.max_age_days and not args.lookback_days:
        args.lookback_days = args.max_age_days
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
