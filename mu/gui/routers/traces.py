"""Trace Analyzer router — list/read raw trace runs.

Globs ``$MUCLI_HOME/trace/*.jsonl`` (mirrors the sessions router's
``_session_dirs`` glob) and serves parsed runs + derived series + the
canvas-ready snapshot, plus a streaming raw endpoint for export. All heavy
work is delegated to ``mu/trace/parser.py`` / ``snapshot.py`` so the GUI and
the ``mucli trace`` CLI share one code path.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from mu.trace import (
    build_series,
    build_session_view,
    build_summary,
    build_trace_snapshot,
    find_trace_path,
    list_trace_runs,
    load_session_runs,
    parse_trace,
)

router = APIRouter()

# Round-44 F7/F8: response bounds. A single run of a long session can hold
# tens of thousands of iterations; the combined session endpoint merges
# EVERY run. The analyzer UI windows its charts (xCount) and the detail
# drill-downs only need the newest window, so the default responses are
# bounded + downsampled; the raw JSONL stream remains the unbounded export
# path, and `full=true` restores the complete payload deliberately.
_MAX_ITERATIONS = 500
_MAX_EVENTS = 2000
_DEFAULT_LIST_LIMIT = 100
# Per-category event caps for the detail endpoints (tools/nudges/compactions/
# requests/context_artifacts). Each list is capped independently; the newest
# entries win because the parser appends chronologically.
_EVENT_CATEGORIES = (
    "nudges",
    "compactions",
    "requests",
    "context_artifacts",
)


def _bounded_run_payload(run: Any, series: Any, snapshot: Any, summary: Any, path: str) -> Dict[str, Any]:
    """Slice a parsed run to the bounded response shape (F7)."""
    iters = run.iters
    total_iterations = len(iters)
    if total_iterations > _MAX_ITERATIONS:
        iters = iters[-_MAX_ITERATIONS:]
    payload = {
        "run_id": run.run_id,
        "header": run.header,
        "iters": iters,
        "total_iterations": total_iterations,
        "iterations_window": len(iters),
        # Round-45 F10: window metadata. The sliced iters keep their ABSOLUTE
        # iter ids, and per-iter series points also carry iter ids — the UI
        # windows series by those ids (_view/_iterBounds), so charts stay
        # aligned. window_start names the first iter in the detail window so
        # clients can reason about coverage without guessing.
        "iterations_window_start": (iters[0].get("iter") if iters else None),
        "tools": run.tools[-_MAX_EVENTS:],
        "total_tools": len(run.tools or []),
        "turn_end": run.turn_end,
        "series": series,
        "snapshot": snapshot,
        "summary": summary,
        "path": path,
    }
    for category in _EVENT_CATEGORIES:
        events = getattr(run, category) or []
        payload[category] = events[-_MAX_EVENTS:]
        payload[f"total_{category}"] = len(events)
    return payload


def _bounded_session_view(view: Dict[str, Any]) -> Dict[str, Any]:
    """Slice a merged session view to the bounded response shape (F7)."""
    iters = view.get("iters") or []
    total_iterations = len(iters)
    bounded = dict(view)
    if total_iterations > _MAX_ITERATIONS:
        bounded["iters"] = iters[-_MAX_ITERATIONS:]
    bounded["total_iterations"] = total_iterations
    bounded["iterations_window"] = len(bounded["iters"])
    # Round-45 F10: same window metadata as the single-run payload. Series
    # points carry absolute iter ids (UI windows by them), so only the
    # detail-window boundary needs naming explicitly.
    bounded["iterations_window_start"] = (
        bounded["iters"][0].get("iter") if bounded["iters"] else None
    )
    tools = view.get("tools") or []
    if len(tools) > _MAX_EVENTS:
        bounded["tools"] = tools[-_MAX_EVENTS:]
    bounded["total_tools"] = len(tools)
    for category in _EVENT_CATEGORIES:
        events = view.get(category) or []
        if len(events) > _MAX_EVENTS:
            bounded[category] = events[-_MAX_EVENTS:]
        bounded[f"total_{category}"] = len(events)
    return bounded


def _find_trace(run_id: str) -> str:
    """Resolve a run_id to its file path, or 404. Thin HTTP wrapper over
    :func:`mu.trace.find_trace_path`."""
    path = find_trace_path(run_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"trace run not found: {run_id}")
    return path


# The trace parser + snapshot builder are CPU-bound and can take seconds on a
# huge run (a 1.1M-token trace has been seen in the wild). These handlers are
# ``async def``, so running that work inline would block the single event loop
# for the whole parse — freezing SSE, /api/sessions, and prompt-answer routes
# (the "GUI freezes, can't load traces / navigate" symptom). Every heavy call
# is offloaded to the default thread executor via ``asyncio.to_thread`` so the
# loop stays free to serve everything else while a big trace parses on a
# worker thread. The ``/raw`` streaming endpoint already runs its sync
# generator in a threadpool (Starlette), so it needs no change.


@router.get("")
async def list_traces(
    session: str | None = None,
    limit: int = _DEFAULT_LIST_LIMIT,
) -> List[Dict[str, Any]]:
    """List trace runs (newest first) with header metadata + iter count.

    Optional ``?session=`` narrows the list to one session's runs — the Trace
    Analyzer is session-scoped, so when opened from the chat it passes the
    current session and ignores every other session's runs.

    Round-44 F8: ``limit`` (default 100, 0 = all) bounds the response so a
    long-lived installation with thousands of runs transfers only the
    newest list-card rows. Each row is already cheap (header + iter count
    only, no full parse), but the transfer itself is now bounded.
    """
    runs = await asyncio.to_thread(list_trace_runs)
    if session:
        runs = [r for r in runs if r.get("session") == session]
    if limit and limit > 0:
        runs = runs[:limit]  # newest first — slice the newest page
    return runs


@router.get("/session/{session_name}")
async def get_session_trace(
    session_name: str,
    cols: int = 128,
    limit: int = 5,
    full: bool = False,
) -> Dict[str, Any]:
    """Combined multi-run view for one session — merged series/summary/
    snapshot with per-run bounds.

    Round-44 F7: the session endpoint previously loaded and merged EVERY
    run in the session — a long-lived session accumulates hundreds of runs,
    each costing a full JSONL parse, before any chart could render.
    ``limit`` (default 5) takes the NEWEST runs for the merged view;
    ``limit=0`` restores "all runs"; ``full=true`` additionally skips the
    per-category event/iteration caps on the merged payload.
    """
    runs = await asyncio.to_thread(load_session_runs, session_name)
    if not runs:
        raise HTTPException(
            status_code=404, detail=f"no trace runs for session: {session_name}"
        )
    if limit and limit > 0:
        runs = runs[-limit:]  # newest N runs, chronological order preserved
    view = await asyncio.to_thread(build_session_view, runs, cols)
    if full:
        return view
    return _bounded_session_view(view)


@router.get("/{run_id}")
async def get_trace(
    run_id: str,
    cols: int = 128,
    full: bool = False,
) -> Dict[str, Any]:
    """Parsed run + derived series + snapshot + overview summary.

    Round-44 F7: bounded by default — the newest 500 iterations and 2000
    events per category (tools/nudges/compactions/requests/artifacts) with
    ``total_*`` counts so nothing is silently lost. ``full=true`` returns
    the complete run (export tooling); ``/{run_id}/raw`` remains the
    streaming export path.
    """
    path = _find_trace(run_id)
    # Round-46 F4: parse-level retention. parse_trace now caps per-category
    # event lists at parse time (default = the same 2000 this router slices
    # to below), so a multi-GB trace no longer materializes ~4x its size in
    # RAM before the bounded payload is built. `full=true` parses with
    # max_events=None for the complete export payload.
    run = await asyncio.to_thread(parse_trace, path, max_events=None if full else _MAX_EVENTS)
    series = await asyncio.to_thread(build_series, run)
    snapshot = await asyncio.to_thread(build_trace_snapshot, run, cols)
    summary = await asyncio.to_thread(build_summary, run, series)
    if full:
        return {
            "run_id": run.run_id,
            "header": run.header,
            "iters": run.iters,
            "tools": run.tools,
            "nudges": run.nudges,
            "compactions": run.compactions,
            "requests": run.requests,
            "context_artifacts": run.context_artifacts,
            "turn_end": run.turn_end,
            "series": series,
            "snapshot": snapshot,
            "summary": summary,
            "path": path,
        }
    return _bounded_run_payload(run, series, snapshot, summary, path)


@router.get("/{run_id}/raw")
async def get_trace_raw(run_id: str):
    """Stream the raw JSONL (for large runs / export)."""
    path = _find_trace(run_id)

    def gen():
        with open(path, encoding="utf-8") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.get("/{run_id}/summary")
async def get_trace_summary(run_id: str) -> Dict[str, Any]:
    """Just the overview cards — cheap for the run picker's hover detail."""
    path = _find_trace(run_id)
    run = await asyncio.to_thread(parse_trace, path)
    series = await asyncio.to_thread(build_series, run)
    return await asyncio.to_thread(build_summary, run, series)


@router.delete("/{run_id}")
async def delete_trace(run_id: str) -> Dict[str, Any]:
    """Delete one trace run's JSONL file. 404 if the run_id is unknown."""
    path = _find_trace(run_id)
    await asyncio.to_thread(os.remove, path)
    return {"ok": True}
