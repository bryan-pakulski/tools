"""Ingest + analyze a per-run trace JSONL (the read side of the emitter).

``parse_trace(path)`` streams a trace file into a typed :class:`TraceRun`.
``build_series(run)`` derives the dashboard series from it — context-growth,
tokenizer drift, compaction/nudge/tool/subagent timelines, tool histograms,
redundant-read events, nudge efficacy, memory counts, token breakdown.

Everything here is a pure function over the parsed run, so it is unit-testable
and shared by the GUI router (``mu/gui/routers/traces.py``) and the
``mucli trace analyze`` CLI. Robust to truncated/blank lines and missing
fields — a malformed line is skipped, never raised.
"""

from __future__ import annotations

import glob
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .emitter import trace_dir


# Round-46 F5: per-file cache for the list-view iter count. Keyed by absolute
# path; validated by (size, mtime_ns) so an unchanged file is a pure dict hit
# and an actively-appending run only rescans its newly appended tail.
_SCAN_CACHE: Dict[str, Dict[str, Any]] = {}
_SCAN_CACHE_LOCK = threading.Lock()
_SCAN_CACHE_CAP = 512

# Round-46 F4: default per-category retention for parse_trace. Matches the
# GUI's per-category event caps (mu/gui/routers/traces.py _MAX_EVENTS) so a
# parsed run is bounded end-to-end without changing dashboard fidelity.
_DEFAULT_EVENT_CAP = 2000


# ----------------------------------------------------------- parsing


@dataclass
class TraceRun:
    """One parsed trace file."""

    path: str
    run_id: str = ""
    header: Dict[str, Any] = field(default_factory=dict)
    iters: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    nudges: List[Dict[str, Any]] = field(default_factory=list)
    compactions: List[Dict[str, Any]] = field(default_factory=list)
    requests: List[Dict[str, Any]] = field(default_factory=list)
    context_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    context_collapses: List[Dict[str, Any]] = field(default_factory=list)
    turn_end: Optional[Dict[str, Any]] = None
    run_end: Optional[Dict[str, Any]] = None
    bytes: int = 0

    @property
    def iter_count(self) -> int:
        return len(self.iters)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


_PARSE_CATEGORY_CAPS: Dict[str, Optional[int]] = {
    "iters": None,          # never dropped — snapshot grid + window math need all
    "tools": _DEFAULT_EVENT_CAP,
    "nudges": _DEFAULT_EVENT_CAP,
    "compactions": _DEFAULT_EVENT_CAP,
    "requests": _DEFAULT_EVENT_CAP,
    "context_artifacts": _DEFAULT_EVENT_CAP,
    "context_collapses": _DEFAULT_EVENT_CAP,
}


def parse_trace(path: str, *, max_events: Optional[int] = _DEFAULT_EVENT_CAP) -> "TraceRun":
    """Stream a trace JSONL into a :class:`TraceRun`. Never raises on bad data.

    Round-46 F4: the per-category event lists used to retain every record, so
    a multi-GB trace was fully materialized as Python objects (~4x file size
    in RAM) just to render a dashboard that only shows the newest window.
    ``max_events`` (default 2000) caps tools / nudges / compactions /
    requests / context_artifacts to the NEWEST records via a drop-from-front
    list; iterations stay complete (bounded separately by the GUI payload)
    because the snapshot grid and series math need every iter id. Pass
    ``max_events=None`` for the full-fidelity export path.
    """
    run = TraceRun(path=path)
    if max_events is not None:
        caps = dict(_PARSE_CATEGORY_CAPS)
        for key, cap in caps.items():
            if key != "iters":
                caps[key] = max_events
    else:
        caps = {key: None for key in _PARSE_CATEGORY_CAPS}
    lists = {key: _BoundedList(cap) for key, cap in caps.items()}
    try:
        run.bytes = os.path.getsize(path)
    except OSError:
        run.bytes = 0
    for obj in _iter_jsonl(path):
        t = obj.get("type")
        if t == "run_start":
            run.header = obj
            run.run_id = obj.get("run_id", "") or run.run_id
        elif t == "iter":
            lists["iters"].append(obj)
            if not run.run_id:
                run.run_id = obj.get("run_id", "")
        elif t == "tool":
            lists["tools"].append(obj)
        elif t == "nudge":
            lists["nudges"].append(obj)
        elif t == "compaction":
            lists["compactions"].append(obj)
        elif t == "request":
            lists["requests"].append(obj)
        elif t == "context_artifact":
            lists["context_artifacts"].append(obj)
        elif t == "context_collapse":
            lists["context_collapses"].append(obj)
        elif t == "turn_end":
            run.turn_end = obj
            if not run.run_id:
                run.run_id = obj.get("run_id", "")
        elif t == "run_end":
            run.run_end = obj
            if not run.run_id:
                run.run_id = obj.get("run_id", "")
    run.iters = lists["iters"]
    run.tools = lists["tools"]
    run.nudges = lists["nudges"]
    run.context_collapses = lists["context_collapses"]
    run.compactions = lists["compactions"]
    run.requests = lists["requests"]
    run.context_artifacts = lists["context_artifacts"]
    return run


# ----------------------------------------------------------- discovery / resolve


def _trace_files() -> List[str]:
    """All trace JSONL files, newest first (by mtime via sorted basename)."""
    return sorted(glob.glob(os.path.join(trace_dir(), "*.jsonl")), reverse=True)


def _read_header(path: str) -> Dict[str, Any]:
    """Read only the first JSON line (run_start) for the list view."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return json.loads(line)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {}


class _BoundedList(list):
    """A list that keeps only the NEWEST ``cap`` entries (F4).

    The parser appends chronologically, so dropping from the front preserves
    exactly the ``[-cap:]`` slice the GUI's bounded payloads want — without
    ever materializing the full list first. ``cap=None`` behaves like list.
    """

    __slots__ = ("cap",)

    def __init__(self, cap: Optional[int] = None) -> None:
        super().__init__()
        self.cap = cap

    def append(self, item: Any) -> None:
        super().append(item)
        cap = self.cap
        if cap is not None:
            excess = len(self) - cap
            if excess > 0:
                del self[0:excess]


def _iter_count_fast(path: str) -> int:
    """Count ``"type":"iter"`` lines without a full parse — cheap for listing.

    Round-46 F5: this used to re-read EVERY byte of EVERY trace on each
    ``list_trace_runs()`` call, so every GUI poll rescanned the whole trace
    corpus (one active multi-GB run = a multi-GB scan per poll). Results are
    now cached per file and refreshed incrementally: unchanged files are a
    dict hit, and an actively-appending run only has its newly appended tail
    rescanned (append-only JSONL). A tail scan is only trusted when the
    previous scan ended on a line boundary; otherwise the file is rescanned
    in full once, which re-establishes the clean boundary.
    """
    try:
        st = os.stat(path)
    except OSError:
        return 0
    size, mtime_ns = st.st_size, st.st_mtime_ns
    with _SCAN_CACHE_LOCK:
        entry = _SCAN_CACHE.get(path)
    if entry and entry["size"] == size and entry["mtime_ns"] == mtime_ns:
        return entry["count"]
    count = 0
    if entry and entry["clean"] and size > entry["size"]:
        # Incremental: count iter lines in the appended tail only.
        base = entry["count"]
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                fh.seek(entry["size"])
                for line in fh:
                    if '"type": "iter"' in line or '"type":"iter"' in line:
                        count += 1
        except OSError:
            return entry["count"]
        count += base
        clean = True
    else:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            return 0
        count = sum(
            1
            for line in raw.splitlines()
            if '"type": "iter"' in line or '"type":"iter"' in line
        )
        clean = raw.endswith("\n") or not raw
    with _SCAN_CACHE_LOCK:
        if len(_SCAN_CACHE) >= _SCAN_CACHE_CAP and path not in _SCAN_CACHE:
            _SCAN_CACHE.pop(next(iter(_SCAN_CACHE)), None)
        _SCAN_CACHE[path] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "count": count,
            "clean": clean,
        }
    return count


def list_trace_runs() -> List[Dict[str, Any]]:
    """List every trace run (newest first) with header metadata + iter count.

    Single source of truth shared by the GUI router's ``GET /api/traces`` and
    the agent-facing ``list_traces`` tool. Reads only the first line per file
    (the ``run_start`` header) plus a cheap iter-line count — no full parse —
    so a directory of large traces stays cheap to list.
    """
    out: List[Dict[str, Any]] = []
    for path in _trace_files():
        header = _read_header(path)
        if not header:
            continue
        out.append(
            {
                "run_id": header.get("run_id", ""),
                "session": header.get("session", ""),
                "model": header.get("model", ""),
                "provider": header.get("provider", ""),
                "mode": header.get("mode", ""),
                "context_limit": header.get("context_limit", 0),
                "max_iterations": header.get("max_iterations", 0),
                "iters": _iter_count_fast(path),
                "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
                "file": os.path.basename(path),
            }
        )
    return out


def find_trace_path(run_id: str) -> Optional[str]:
    """Resolve a run_id to its trace file path, or ``None`` if not found.

    Matches the run_id anywhere in the filename (traces are named
    ``<session>_run_<id>.jsonl``), then falls back to an exact filename match
    under the trace dir. Callers that need a raise-on-miss (the GUI router)
    wrap this; the agent tools return an error envelope on ``None``.
    """
    if run_id == "latest":
        files = glob.glob(os.path.join(trace_dir(), "*.jsonl"))
        if files:
            return max(files, key=os.path.getmtime)
        return None
    for path in _trace_files():
        if run_id and run_id in os.path.basename(path):
            return path
    target = os.path.join(trace_dir(), run_id)
    if os.path.exists(target):
        return target
    return None


def load_session_runs(session_name: str) -> List["TraceRun"]:
    """Parse every trace run for one session, oldest-first (chronological).

    A session spans multiple runs (each agent-loop invocation = one run).
    Order is by file mtime ascending — ``run_start`` carries no timestamp, and
    the run-id hex is a uuid (not monotonic), so mtime is the only chronological
    proxy. Matches on the header ``session`` field (accurate even when session
    names are substrings of one another), not the filename prefix.
    """
    selected: List[tuple] = []
    for path in glob.glob(os.path.join(trace_dir(), "*.jsonl")):
        header = _read_header(path)
        if header.get("session") == session_name:
            try:
                selected.append((os.path.getmtime(path), path))
            except OSError:
                continue
    selected.sort(key=lambda x: x[0])  # oldest first
    return [parse_trace(p) for _, p in selected]


def combine_runs(runs: List["TraceRun"]) -> "TraceRun":
    """Merge a chronologically-ordered list of runs into one TraceRun with
    globally-numbered iterations, so :func:`build_series` /
    :func:`build_summary` / :func:`build_trace_snapshot` produce a combined
    *session* view. Each record keeps its original ``run_id`` so the UI can
    draw run boundaries.

    Iterations are renumbered by *order* (not by adding an offset to the local
    iter value), so runs whose iters start at 1 or have gaps still lay out
    contiguously. Tools / nudges / compactions are remapped to the global iter
    via a per-run {local: global} map built from that run's iters; an event
    whose iter isn't in the map falls back to the run's last global iter.

    Round-46 F6: the merge CONSUMES the source runs (event dicts are remapped
    in place and their lists emptied) so peak memory is the merged run alone,
    not sources + merged.
    """
    merged = TraceRun(path="")
    if not runs:
        return merged
    merged.run_id = runs[0].run_id
    merged.header = dict(runs[0].header)
    merged.turn_end = runs[-1].turn_end
    merged.run_end = next((r.run_end for r in runs if r.run_end), None)
    global_iter = 0
    for run in runs:
        iter_map: Dict[Any, int] = {}
        for i in run.iters:
            local = i.get("iter")
            iter_map[local] = global_iter
            i["iter"] = global_iter
            merged.iters.append(i)
            global_iter += 1
        # Fall-back global iter for events whose local iter isn't recorded as
        # an iteration (defensive — shouldn't normally happen).
        fallback = global_iter - 1 if run.iters else global_iter
        # Round-46 F6: remap each source event IN PLACE (mutating the source
        # run's event dicts and moving them into the merged run) instead of
        # shallow-copying everything. The source runs are consumed by the
        # merge — build_session_view passes ownership — so peak memory for a
        # multi-run session drops from (sources + merged) to just merged.
        # Callers that still need their runs intact pass copies.
        for t in run.tools:
            t["iter"] = iter_map.get(t.get("iter"), fallback)
            merged.tools.append(t)
        run.tools = []
        for n in run.nudges:
            local = n.get("iteration", n.get("iter"))
            if local in iter_map:
                gi = iter_map[local]
                n["iteration"] = gi
                if "iter" in n:
                    n["iter"] = gi
            merged.nudges.append(n)
        run.nudges = []
        for c in run.compactions:
            c["iter"] = iter_map.get(c.get("iter"), fallback)
            merged.compactions.append(c)
        run.compactions = []
        for req in run.requests:
            req["iter"] = iter_map.get(req.get("iter"), fallback)
            merged.requests.append(req)
        run.requests = []
        for artifact in run.context_artifacts:
            artifact["iter"] = iter_map.get(artifact.get("iter"), fallback)
            merged.context_artifacts.append(artifact)
        run.context_artifacts = []
    return merged


def build_session_view(
    runs: List["TraceRun"], cols: int = 128
) -> Dict[str, Any]:
    """Combined multi-run view for one session: merged series + summary +
    snapshot + per-run bounds, in the same shape as the single-run endpoint so
    the frontend reuses one render path.

    ``run_bounds`` marks each run's global [start_iter, end_iter] so the UI can
    draw run-boundary dividers; token/cost totals are summed across all runs'
    ``turn_end`` records (the merged run's ``turn_end`` is only the last run's,
    so ``build_summary`` alone would undercount).
    """
    from .snapshot import build_trace_snapshot

    merged = combine_runs(runs)
    series = build_series(merged)
    summary = build_summary(merged, series)
    snapshot = build_trace_snapshot(merged, cols=cols)

    # Token / cost totals: sum every run's turn_end (build_summary only sees the
    # merged run's = the last run's turn_end).
    total_in = sum(_num((r.turn_end or {}).get("total_in")) for r in runs)
    total_out = sum(_num((r.turn_end or {}).get("total_out")) for r in runs)
    total_cost = sum(_num((r.turn_end or {}).get("total_cost")) for r in runs)
    summary["total_in"] = int(total_in)
    summary["total_out"] = int(total_out)
    summary["total_cost"] = round(total_cost, 6)
    # Session status: completed only if every run completed. Round-51 T2:
    # prefer the terminal run_end record when present; legacy traces fall
    # back to turn_end, and only fully headless traces stay 'running'.
    statuses = [
        (r.run_end or r.turn_end or {}).get("status") for r in runs
    ]
    summary["status"] = "completed" if all(s == "completed" for s in statuses) else (
        statuses[-1] if statuses and statuses[-1] else "running"
    )

    run_bounds: List[Dict[str, Any]] = []
    gi = 0
    for run in runs:
        n = len(run.iters)
        run_bounds.append(
            {
                "run_id": run.run_id,
                "start_iter": gi,
                "end_iter": gi + n - 1,
                "iters": n,
                "model": run.header.get("model", ""),
                "mode": run.header.get("mode", ""),
                "status": (run.run_end or run.turn_end or {}).get("status", "running"),
            }
        )
        gi += n

    return {
        "run_id": "session:" + (runs[0].header.get("session", "") if runs else ""),
        "header": merged.header,
        "iters": merged.iters,
        "tools": merged.tools,
        "nudges": merged.nudges,
        "compactions": merged.compactions,
        "requests": merged.requests,
        "context_artifacts": merged.context_artifacts,
        "turn_end": merged.turn_end,
        "series": series,
        "snapshot": snapshot,
        "summary": summary,
        "run_bounds": run_bounds,
        "n_runs": len(runs),
        "path": None,
    }


# ----------------------------------------------------------- series derivation


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _iter_of(v: Any) -> int:
    """Coerce an iteration field to int, treating only missing/None as -1.

    ``int(v or -1)`` would wrongly map a legitimate ``0`` to ``-1`` (0 is
    falsy), so handle None/missing explicitly.
    """
    if v is None:
        return -1
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def build_series(run: TraceRun) -> Dict[str, Any]:
    """Derive the dashboard-ready series from a parsed run.

    Returns a dict of named series, each a list of per-iteration points (or
    event lists for timelines). All pure over ``run`` — no I/O.
    """
    iters = run.iters
    n = len(iters)
    xs = [int(i.get("iter", k)) for k, i in enumerate(iters)]

    # --- context growth (total_est vs prompt_tokens_actual) + per-layer ---
    context = []
    layers_stacked: Dict[str, List[float]] = {
        "l0": [], "l1a": [], "l1c": [], "l1b": [], "l2": [], "l3": [], "l4b": [], "l5": []
    }
    drift: List[Dict[str, Any]] = []
    for i in iters:
        ctx = i.get("context", {}) or {}
        context.append(
            {
                "iter": i.get("iter"),
                "total_est": _num(ctx.get("total_est")),
                "actual": _num(ctx.get("prompt_tokens_actual")),
                # Drift-corrected real-prompt estimate — the representative
                # real fill (Ollama's prompt_tokens_actual is the cached
                # delta, near-zero in a warm loop). The GUI "Context growth"
                # chart plots this as the real line so a looming overflow is
                # visible instead of hidden behind a near-zero actual.
                "real_est": _num(ctx.get("prompt_tokens_real_est")),
                "drift_ratio": _num(ctx.get("drift_ratio")),
                "drift_pct": _num(ctx.get("drift_pct")),
                # Representative real fill for the chart's solid line: the
                # drift-corrected estimate, but never below the provider's own
                # reported count (frontier providers report the true full
                # prompt; Ollama reports the cached delta, which real_est
                # dominates).
                "real": max(_num(ctx.get("prompt_tokens_actual")),
                            _num(ctx.get("prompt_tokens_real_est"))),
            }
        )
        for key in layers_stacked:
            layers_stacked[key].append(_num(ctx.get(key)))
        drift.append(
            {
                "iter": i.get("iter"),
                "drift_pct": _num(ctx.get("drift_pct")),
                # Whether drift_pct is a real full-prompt comparison or was
                # zeroed because the provider reported only a cached delta
                # (Ollama warm cache). Summary stats exclude unreliable 0.0
                # readings so median/peak drift reflect real estimate error.
                "reliable": bool(ctx.get("drift_pct_reliable")),
                "actual": _num(ctx.get("prompt_tokens_actual")),
                "total_est": _num(ctx.get("total_est")),
            }
        )

    # --- model-directed context artifact lifecycle ---
    context_artifacts = sorted(run.context_artifacts, key=lambda a: (_iter_of(a.get("iter")), str(a.get("artifact_id", ""))))
    artifact_counts = {}
    for artifact in context_artifacts:
        state = str(artifact.get("state") or "unknown")
        artifact_counts[state] = artifact_counts.get(state, 0) + 1

    # --- exact request composition + spike attribution -----------------
    # New traces carry privacy-preserving per-part token counts.  Older
    # traces only have byte totals; keep them useful with a clearly marked
    # bytes/4 approximation instead of hiding the panel entirely.
    attribution_keys = (
        "system", "user", "assistant", "tool_calls", "tool_results",
        "files_images", "other", "tool_schemas",
    )
    iter_actual = {
        _iter_of(item.get("iter")): _num((item.get("tokens") or {}).get("in"))
        for item in iters
    }
    context_attribution = []
    previous = {key: 0.0 for key in attribution_keys}
    previous_total = 0.0
    for req in sorted(run.requests, key=lambda item: _iter_of(item.get("iter"))):
        raw_components = req.get("component_tokens") or {}
        approximate = not bool(raw_components)
        components = {key: _num(raw_components.get(key)) for key in attribution_keys}
        if approximate:
            components["system"] = _num(req.get("system_prompt_bytes")) / 4.0
            components["tool_schemas"] = _num(req.get("tool_schema_bytes")) / 4.0
            for msg in req.get("messages") or []:
                role = str(msg.get("role") or "")
                bucket = role if role in {"user", "assistant"} else "other"
                components[bucket] += _num(msg.get("bytes")) / 4.0
        total = _num(req.get("token_estimate"), sum(components.values()))
        deltas = {key: components[key] - previous[key] for key in attribution_keys}
        growth_source = max(deltas, key=deltas.get) if deltas else "other"
        largest_item = {"label": "", "tokens": 0.0}
        for msg in req.get("messages") or []:
            parts = msg.get("part_details") or msg.get("parts") or []
            if not isinstance(parts, list):
                continue
            for part in parts:
                part_tokens = _num(part.get("tokens"))
                if part_tokens > largest_item["tokens"]:
                    label = f"msg {msg.get('index', '?')} {msg.get('role', '')}/{part.get('type', '')}"
                    if part.get("tool_name"):
                        label += f" ({part.get('tool_name')})"
                    largest_item = {"label": label, "tokens": part_tokens}
        iteration = _iter_of(req.get("iter"))
        request_delta = total - previous_total if context_attribution else 0.0
        point = {
            "iter": iteration,
            **{key: round(value, 2) for key, value in components.items()},
            "total": round(total, 2),
            # The first request is the baseline, not a growth spike.
            "delta": round(request_delta, 2),
            "growth_source": growth_source,
            "growth_tokens": round(deltas.get(growth_source, 0.0), 2),
            "provider_input": iter_actual.get(iteration, 0.0),
            "provider_gap": round(iter_actual.get(iteration, 0.0) - total, 2),
            "largest_item": largest_item,
            "approximate": approximate,
        }
        context_attribution.append(point)
        previous = components
        previous_total = total
    top_context_spikes = sorted(
        context_attribution,
        key=lambda point: point.get("delta", 0),
        reverse=True,
    )[:12]

    # --- token breakdown per iter ---
    tokens = []
    for i in iters:
        tk = i.get("tokens", {}) or {}
        tokens.append(
            {
                "iter": i.get("iter"),
                "in": _num(tk.get("in")),
                "out": _num(tk.get("out")),
                "cached": _num(tk.get("cached")),
                "reasoning": _num(tk.get("reasoning")),
                "cost_delta": _num(tk.get("cost_delta")),
            }
        )

    # --- per-iteration wall time (provider-call latency) ---
    latency = [
        {"iter": i.get("iter"), "wall_ms": _num(i.get("wall_ms"))}
        for i in iters
    ]

    # --- tool histogram + per-tool latency series ---
    tool_hist: Dict[str, Dict[str, Any]] = {}
    tool_latency: Dict[str, List[Dict[str, Any]]] = {}
    # Per-iteration efficiency accumulation (spec #12): raw vs injected
    # tool-output tokens, omitted/stored/retrieval counts.
    eff_by_iter: Dict[int, Dict[str, Any]] = {}
    retrieval_tool_names = {
        "recall", "result_range", "result_head", "result_tail",
        "result_search", "result_diagnostics", "result_json_path",
        "compare_results",
    }
    for tr in run.tools:
        name = str(tr.get("name") or "unknown")
        h = tool_hist.setdefault(
            name,
            {"name": name, "count": 0, "ok": 0, "error": 0, "latency_sum": 0.0,
             "cache_hits": 0, "result_bytes_sum": 0, "error_codes": {},
             "stored": 0, "omitted": 0, "raw_tokens_sum": 0,
             "injected_tokens_sum": 0},
        )
        h["count"] += 1
        if tr.get("ok"):
            h["ok"] += 1
        else:
            h["error"] += 1
            code = str(tr.get("error_code") or "unknown")
            h["error_codes"][code] = h["error_codes"].get(code, 0) + 1
        h["latency_sum"] += _num(tr.get("latency_ms"))
        if tr.get("cache_hit"):
            h["cache_hits"] += 1
        h["result_bytes_sum"] += _num(tr.get("result_bytes"))
        if tr.get("stored"):
            h["stored"] += 1
        if tr.get("omitted"):
            h["omitted"] += 1
        h["raw_tokens_sum"] += _num(tr.get("raw_tokens"))
        h["injected_tokens_sum"] += _num(tr.get("injected_tokens"))
        tool_latency.setdefault(name, []).append(
            {
                "iter": tr.get("iter"),
                "latency_ms": _num(tr.get("latency_ms")),
                "ok": bool(tr.get("ok")),
                "cache_hit": bool(tr.get("cache_hit")),
                "omitted": bool(tr.get("omitted")),
                "stored": bool(tr.get("stored")),
                "raw_tokens": _num(tr.get("raw_tokens")),
                "injected_tokens": _num(tr.get("injected_tokens")),
            }
        )
        # Efficiency accumulation per iteration.
        _ei = _iter_of(tr.get("iter"))
        e = eff_by_iter.setdefault(
            _ei,
            {"iter": tr.get("iter"), "raw_tokens": 0, "injected_tokens": 0,
             "tool_calls": 0, "omitted": 0, "stored": 0, "retrievals": 0,
             "cache_hits": 0},
        )
        e["raw_tokens"] += _num(tr.get("raw_tokens"))
        e["injected_tokens"] += _num(tr.get("injected_tokens"))
        e["tool_calls"] += 1
        if tr.get("omitted"):
            e["omitted"] += 1
        if tr.get("stored"):
            e["stored"] += 1
        if tr.get("cache_hit"):
            e["cache_hits"] += 1
        if name in retrieval_tool_names:
            e["retrievals"] += 1
    for h in tool_hist.values():
        c = max(1, h["count"])
        h["avg_latency_ms"] = round(h["latency_sum"] / c, 2)
        h["cache_hit_rate"] = round(h["cache_hits"] / c, 3)
        h["avg_result_bytes"] = int(h["result_bytes_sum"] / c)
        h["stored_rate"] = round(h["stored"] / c, 3)
        h["omitted_rate"] = round(h["omitted"] / c, 3)
        h["avg_raw_tokens"] = int(h["raw_tokens_sum"] / c)
        h["avg_injected_tokens"] = int(h["injected_tokens_sum"] / c)
        raw = h["raw_tokens_sum"]
        h["tokens_saved"] = max(0, raw - h["injected_tokens_sum"])
        h["compression_ratio"] = (
            round((raw - h["injected_tokens_sum"]) / raw, 3) if raw > 0 else 0.0
        )
    for name, series in tool_latency.items():
        tool_hist[name]["latency_series"] = series
    efficiency_series = [
        eff_by_iter[k] for k in sorted(eff_by_iter.keys())
    ]

    # --- compaction timeline ---
    compaction_timeline = []
    for c in run.compactions:
        compaction_timeline.append(
            {
                "iter": c.get("iter"),
                "kind": c.get("kind"),
                "tokens_before": _num(c.get("tokens_before")),
                "tokens_after": _num(c.get("tokens_after")),
                "tokens_saved": _num(c.get("tokens_saved")),
                "summarizer": c.get("summarizer"),
                "keep_recent": c.get("keep_recent"),
                "budget": c.get("budget"),
                "anchor_delta": _num(c.get("anchor_delta")),
            }
        )

    # --- nudge timeline + efficacy ---
    nudge_timeline = [
        {
            "iter": nd.get("iteration"),
            "kind": nd.get("kind"),
            "extra": {k: v for k, v in nd.items()
                      if k not in {"type", "run_id", "kind", "iteration"}},
        }
        for nd in run.nudges
    ]
    nudge_efficacy = _nudge_efficacy(run, k=3)

    # --- redundant reads (same path re-read with no intervening write) ---
    redundant_reads = _redundant_reads(run)

    # --- subagent timeline (per-iter snapshot deltas) ---
    subagent_timeline = []
    for i in iters:
        sa = i.get("subagents", {}) or {}
        subagent_timeline.append(
            {
                "iter": i.get("iter"),
                "active": int(sa.get("active", 0) or 0),
                "stuck": int(sa.get("stuck", 0) or 0),
                "stall": int(sa.get("stall", 0) or 0),
                "children": sa.get("children", []) or [],
            }
        )

    # --- memory counts per iter ---
    memory_series = []
    for i in iters:
        mem = i.get("memory", {}) or {}
        memory_series.append(
            {
                "iter": i.get("iter"),
                "task_memory_count": int(mem.get("task_memory_count", 0) or 0),
                "scratchpad_count": int(mem.get("scratchpad_count", 0) or 0),
                "by_status": mem.get("by_status", {}) or {},
            }
        )

    return {
        "n": n,
        "xs": xs,
        "context": context,
        "layers_stacked": layers_stacked,
        "drift": drift,
        "tokens": tokens,
        "latency": latency,
        "tool_histogram": list(tool_hist.values()),
        "efficiency": efficiency_series,
        "compaction_timeline": compaction_timeline,
        "nudge_timeline": nudge_timeline,
        "nudge_efficacy": nudge_efficacy,
        "redundant_reads": redundant_reads,
        "subagent_timeline": subagent_timeline,
        "memory_series": memory_series,
        "context_artifacts": context_artifacts,
        "context_artifact_counts": artifact_counts,
        "context_attribution": context_attribution,
        "top_context_spikes": top_context_spikes,
        # Perf (trace-viewer optimization): the UI never reads series.requests
        # raw messages — it charts the aggregated context_attribution points
        # instead. Each request record carries its FULL messages array (up to
        # ~130KB per record on long sessions), which made series.requests the
        # single biggest payload item (~2MB on a 6-run session). Shed the
        # heavy fields here; the complete records stay available via the
        # top-level `requests` category and `full=true`.
        "requests": [
            _light_request(req)
            for req in sorted(run.requests, key=lambda req: _iter_of(req.get("iter")))
        ],
    }


# ----------------------------------------------------------- efficacy / reads


def _light_request(req: Any) -> Dict[str, Any]:
    """Copy a request record without its verbose nested arrays.

    Perf (trace-viewer optimization): `messages` (and its nested
    `part_details`) are the bulk of each request record — up to ~130KB per
    request on long sessions. They are only consumed for the full detail
    drill-down (`full=true`) and the approximate-attribution fallback, both
    of which re-read the complete `run.requests` list. The light form keeps
    every scalar the UI summarizes (token_estimate, component totals,
    messages_hash, tool_names, ...) so charts and tables render identically.
    """
    light = {key: value for key, value in req.items() if key not in ("messages",)}
    msgs = req.get("messages")
    if isinstance(msgs, list) and msgs:
        light["messages_count"] = len(msgs)
    return light


_WRITE_TOOLS = {
    "write_file", "apply_diff", "search_and_replace_file", "bash",
    "bash_background", "save_memory", "update_memory_status", "todo_write",
    "todo_set_status", "todo_delete", "todo_clear",
}
_READ_TOOLS = {
    "read_file", "get_chunk", "list_dir", "search_for_string",
    "search_references", "retrieve_relevant_context", "get_workspace_details",
}


def _nudge_efficacy(run: TraceRun, k: int = 3) -> List[Dict[str, Any]]:
    """For each nudge, did a materially different action follow within k iters?

    "Materially different" = a write tool, or a tool call whose (name, arg_fp)
    differs from the tool call immediately preceding the nudge's iteration.
    Falls back to ``broke=False`` when there's not enough surrounding data.
    """
    # Index tools by iter.
    by_iter: Dict[int, List[Dict[str, Any]]] = {}
    for tr in run.tools:
        by_iter.setdefault(_iter_of(tr.get("iter")), []).append(tr)
    out = []
    for nd in run.nudges:
        it = _iter_of(nd.get("iteration"))
        # The tool calls in the nudged iteration itself.
        pre = by_iter.get(it, [])
        pre_fps = {f"{t.get('name')}:{t.get('arg_fp')}" for t in pre}
        broke = False
        how = None
        for j in range(it + 1, it + 1 + k):
            calls = by_iter.get(j, [])
            if not calls:
                continue
            for c in calls:
                if str(c.get("name", "")) in _WRITE_TOOLS:
                    broke = True
                    how = "write"
                    break
                fp = f"{c.get('name')}:{c.get('arg_fp')}"
                if fp not in pre_fps:
                    broke = True
                    how = "novel_call"
                    break
            if broke:
                break
        out.append({"iter": it, "kind": nd.get("kind"), "broke": broke, "how": how})
    return out


def _redundant_reads(run: TraceRun) -> List[Dict[str, Any]]:
    """Flag a read of a path that was already read with no intervening write.

    Quantifies the context-gathering stall the recoverage nudge reacts to, and
    lets the dashboard correlate re-reads with compaction events (re-reading
    *caused by* state loss vs aimlessness).
    """
    last_read: Dict[str, int] = {}  # path -> iter of last read
    out = []
    # Walk tools in emit order; emit order is iter-major, input-order within.
    for tr in run.tools:
        name = str(tr.get("name", ""))
        path = str(tr.get("path", "") or "")
        it = _iter_of(tr.get("iter"))
        if name in _WRITE_TOOLS:
            # A write invalidates the "already read" state for that path.
            last_read.pop(path, None)
            continue
        if name in _READ_TOOLS and path:
            prev = last_read.get(path)
            if prev is not None and prev != it:
                out.append(
                    {
                        "iter": it,
                        "path": path,
                        "tool": name,
                        "prev_iter": prev,
                        "gap": it - prev,
                    }
                )
            last_read[path] = it
    return out


# ----------------------------------------------------------- overview summary


def _build_efficiency_summary(
    run: TraceRun, series: Dict[str, Any]
) -> Dict[str, Any]:
    """Aggregate the per-iteration efficiency series into run-wide totals and
    merge in the cache counters the session stamped on ``turn_end.efficiency``
    (spec #12). Tolerates older traces that carry none of these fields."""
    eff_series = series.get("efficiency") or []
    total_raw = sum(_num(e.get("raw_tokens")) for e in eff_series)
    total_injected = sum(_num(e.get("injected_tokens")) for e in eff_series)
    total_saved = max(0, total_raw - total_injected)
    omitted = sum(1 for e in eff_series if e.get("omitted"))
    stored = sum(_num(e.get("stored")) for e in eff_series)
    retrievals = sum(_num(e.get("retrievals")) for e in eff_series)
    tool_calls = sum(_num(e.get("tool_calls")) for e in eff_series)
    peak_raw = max((_num(e.get("raw_tokens")) for e in eff_series), default=0)

    out: Dict[str, Any] = {
        "raw_tokens": int(total_raw),
        "injected_tokens": int(total_injected),
        "tokens_saved": int(total_saved),
        "compression_ratio": (
            round(total_saved / total_raw, 3) if total_raw > 0 else 0.0
        ),
        "omitted_results": int(omitted),
        "stored_results": int(stored),
        "retrieval_calls": int(retrievals),
        "tool_calls": int(tool_calls),
        "retrieval_rate": (
            round(retrievals / tool_calls, 3) if tool_calls > 0 else 0.0
        ),
        "peak_raw_tokens": int(peak_raw),
    }

    # Merge cache counters from turn_end.efficiency (written by
    # collect_efficiency_metrics at turn end). Present on traces recorded
    # after the spec-#12 wiring; absent on older traces.
    te_eff = ((run.turn_end or {}).get("efficiency") or {}) if run.turn_end else {}
    cache = te_eff.get("cache") or {}
    if cache:
        out["cache"] = {
            "evictions": int(cache.get("evictions", 0)),
            "invalidations": int(cache.get("invalidations", 0)),
            "disk_hits": int(cache.get("disk_hits", 0)),
            "dup_bytes_avoided": int(cache.get("dup_bytes_avoided", 0)),
            "locator_hits": int(cache.get("locator_hits", 0)),
        }
    if te_eff.get("tool_output_share") is not None:
        out["tool_output_share"] = float(te_eff.get("tool_output_share") or 0.0)
    return out


def build_summary(run: TraceRun, series: Dict[str, Any]) -> Dict[str, Any]:
    """Overview cards: totals, peaks, counts by type — the at-a-glance read."""
    # Drift stats use only reliable readings — Ollama warm-cache iters
    # report a near-zero cached delta, so their drift_pct is zeroed (not a
    # real 0% estimate error). Including those 0.0s would drag the median
    # down and hide real cl100k undercount. Fall back to all readings only
    # when none are reliable (e.g. a fully warm Ollama loop).
    reliable_drift = [d for d in series["drift"] if d.get("reliable")]
    drift_src = reliable_drift if reliable_drift else series["drift"]
    drift_pts = [d["drift_pct"] for d in drift_src]
    peak_ctx = max(
        (c["actual"] for c in series["context"]), default=0.0
    )
    peak_est = max(
        (c["total_est"] for c in series["context"]), default=0.0
    )
    wall_pts = [w["wall_ms"] for w in series["latency"]]
    total_wall = sum(wall_pts) if wall_pts else 0.0
    peak_wall = max(wall_pts, default=0.0)
    mean_wall = total_wall / max(1, len(wall_pts))
    # Median is more robust than mean for drift_pct — the (actual−est)/actual
    # formula blows up when prompt_tokens_actual is small, so the mean is dragged
    # by a few extreme outliers (e.g. −1985%). Median reflects the typical iter.
    sorted_drift = sorted(drift_pts)
    median_drift = (
        sorted_drift[len(sorted_drift) // 2] if sorted_drift else 0.0
    )
    comp_by_kind: Dict[str, int] = {}
    mechanical = 0
    for c in series["compaction_timeline"]:
        comp_by_kind[c["kind"] or "unknown"] = comp_by_kind.get(c["kind"] or "unknown", 0) + 1
        if c.get("summarizer") == "mechanical":
            mechanical += 1
    nudge_by_kind: Dict[str, int] = {}
    for nd in series["nudge_timeline"]:
        nudge_by_kind[nd["kind"] or "unknown"] = nudge_by_kind.get(nd["kind"] or "unknown", 0) + 1
    nudges_broken = sum(1 for e in series["nudge_efficacy"] if e["broke"])

    total_in = 0.0
    total_out = 0.0
    total_cost = 0.0
    if run.turn_end:
        total_in = _num(run.turn_end.get("total_in"))
        total_out = _num(run.turn_end.get("total_out"))
        total_cost = _num(run.turn_end.get("total_cost"))
    else:
        for t in series["tokens"]:
            total_in += t["in"]
            total_out += t["out"]
            total_cost += t["cost_delta"]

    subagent_iters = sum(1 for s in series["subagent_timeline"] if s["active"])

    return {
        "run_id": run.run_id,
        "session": run.header.get("session", ""),
        "model": run.header.get("model", ""),
        "provider": run.header.get("provider", ""),
        "mode": run.header.get("mode", ""),
        "context_limit": int(run.header.get("context_limit", 0) or 0),
        "max_iterations": int(run.header.get("max_iterations", 0) or 0),
        "iters": run.iter_count,
        "total_in": int(total_in),
        "total_out": int(total_out),
        "total_cost": round(total_cost, 6),
        "compaction_count": len(series["compaction_timeline"]),
        "compaction_by_kind": comp_by_kind,
        "mechanical_fallback_count": mechanical,
        "nudge_count": len(series["nudge_timeline"]),
        "nudge_by_kind": nudge_by_kind,
        "nudges_broken": nudges_broken,
        "subagent_iters": subagent_iters,
        "peak_context": int(peak_ctx),
        "peak_estimated": int(peak_est),
        "peak_drift_abs": round(max((abs(d) for d in drift_pts), default=0.0), 2),
        "mean_drift": round(
            sum(drift_pts) / max(1, len(drift_pts)), 2
        ) if drift_pts else 0.0,
        "median_drift": round(median_drift, 2),
        "total_wall_ms": int(total_wall),
        "peak_wall_ms": int(peak_wall),
        "mean_wall_ms": int(mean_wall),
        "tool_calls": len(run.tools),
        "request_count": len(run.requests),
        "peak_request_estimate": int(max(
            (point.get("total", 0) for point in series.get("context_attribution", [])),
            default=0,
        )),
        "peak_request_delta": int(max(
            (point.get("delta", 0) for point in series.get("context_attribution", [])),
            default=0,
        )),
        "context_artifact_counts": series.get("context_artifact_counts", {}),
        "redundant_reads": len(series["redundant_reads"]),
        "status": (run.run_end or run.turn_end or {}).get("status", "running"),
        "bytes": run.bytes,
        # Tool-output efficiency (spec #12). Run-wide aggregates from the
        # per-iteration efficiency series, merged with the cache counters
        # the session stamped on turn_end.efficiency (evictions /
        # invalidations / disk_hits / locator_hits / dup_bytes_avoided) when
        # present.
        "efficiency": _build_efficiency_summary(run, series),
        # Suspects digest (UX polish): which harness suspects actually
        # fired, so a debugging agent gets the "what to look at next"
        # pointers without diffing the full summary by hand.
        "suspects": _build_suspects(run, series, mechanical, nudges_broken),
    }


def _build_suspects(run: TraceRun, series: Dict[str, Any], mechanical: int, nudges_broken: int) -> list:
    """Ranked list of harness suspects observed in this run.

    Each suspect names the signal, its severity, and the iterations to
    drill into next (trace_series / trace_iteration). Empty list = clean
    run. This is the 'where do I look' layer between the totals above
    and the per-iteration drill-down.
    """
    suspects: list = []
    context_limit = int(run.header.get("context_limit", 0) or 0)

    # 1. Tokenizer drift — only reliable readings count.
    reliable = [d for d in series["drift"] if d.get("reliable")]
    drift_pts = [(d.get("iter"), d["drift_pct"]) for d in reliable]
    if drift_pts:
        worst_iter, worst_pct = max(drift_pts, key=lambda p: abs(p[1]))
        if abs(worst_pct) >= 15.0:
            suspects.append({
                "suspect": "tokenizer_drift",
                "severity": "high" if abs(worst_pct) >= 40.0 else "medium",
                "detail": f"drift {worst_pct:.1f}% at iter {worst_iter} (reliable reading)",
                "next": f"trace_series drift; trace_iteration iter={worst_iter}",
            })

    # 2. Emergency/mechanical compactions — lossy summarizer.
    if mechanical:
        iters = [c.get("iter") for c in series["compaction_timeline"]
                 if c.get("summarizer") == "mechanical"]
        suspects.append({
            "suspect": "mechanical_compaction",
            "severity": "high",
            "detail": f"{mechanical} compaction(s) fell back to the lossy mechanical summarizer",
            "next": f"trace_series compaction_timeline; iters {sorted(set(i for i in iters if i is not None))}",
        })

    # 3. Nudges that failed to break a loop — stuck-loop signal.
    # Round-46 F9: this used to condition on nudges_broken (the HEALTHY
    # case), so successful interventions were reported as suspects and the
    # actual stuck-loop signal — a fired-but-unbroken nudge — was invisible.
    # failed = fired - broken; each failed nudge's `broke` flag is False.
    nudge_efficacy = series.get("nudge_efficacy") or []
    nudge_failed = len(nudge_efficacy) - nudges_broken
    if nudge_failed > 0:
        iters = [e.get("iter") for e in nudge_efficacy if not e["broke"]]
        suspects.append({
            "suspect": "nudge_failed_to_break",
            "severity": "medium",
            "detail": f"{nudge_failed} nudge(s) fired but did NOT break the loop",
            "next": f"trace_iteration for iters {sorted(set(i for i in iters if i is not None))}",
        })

    # 4. Context pressure — peak within 10% of the window.
    peak_ctx = max((c["actual"] for c in series["context"]), default=0.0)
    if context_limit and peak_ctx >= 0.9 * context_limit:
        peak_iter = max(
            series["context"], key=lambda c: c["actual"], default={}
        ).get("iter")
        suspects.append({
            "suspect": "context_pressure",
            "severity": "medium",
            "detail": f"peak context {int(peak_ctx)} of {context_limit} tokens",
            "next": "trace_series context" + (f"; trace_iteration iter={peak_iter}" if peak_iter else ""),
        })

    # 5. Redundant reads — wasted turns.
    redundant = series["redundant_reads"]
    if redundant:
        iters = sorted({r.get("iter") for r in redundant if r.get("iter") is not None})
        suspects.append({
            "suspect": "redundant_reads",
            "severity": "low",
            "detail": f"{len(redundant)} redundant read(s)",
            "next": f"trace_series redundant_reads; iters {iters[:10]}",
        })

    # 6. Subagent stalls.
    stuck = [s for s in series["subagent_timeline"] if s.get("stuck") or s.get("stall")]
    if stuck:
        iters = sorted({s.get("iter") for s in stuck if s.get("iter") is not None})
        suspects.append({
            "suspect": "subagent_stall",
            "severity": "medium",
            "detail": f"{len(stuck)} subagent stuck/stall reading(s)",
            "next": f"trace_series subagent_timeline; iters {iters[:10]}",
        })

    # Rank by severity.
    order = {"high": 0, "medium": 1, "low": 2}
    suspects.sort(key=lambda s: order.get(s["severity"], 3))
    return suspects


__all__ = [
    "TraceRun",
    "parse_trace",
    "build_series",
    "build_summary",
    "list_trace_runs",
    "find_trace_path",
    "load_session_runs",
    "combine_runs",
    "build_session_view",
]
