"""Agent-facing trace-reading tools.

Four read-only `@tool` handlers that expose the per-run JSONL traces under
``$MUCLI_HOME/trace/`` to the agent loop and spawned subagents, so an agent can
inspect a past or in-flight run and debug it (drift, compaction fallbacks,
nudges, redundant reads, subagent stalls, per-iteration latency).

Thin wrappers over ``mu/trace/parser.py`` (``parse_trace`` / ``build_series`` /
``build_summary`` / ``find_trace_path`` / ``list_trace_runs``). No writes; never
raises on bad data (the parser is defensive). Traces are global under
``$MUCLI_HOME`` — not session-scoped — so these work even when no session is
attached (``server_policy="allowed"``), which lets a spawned debug subagent
read a run without the parent's session.

Mirrors the session-tool envelope pattern: handlers return ``json.dumps(...)``
strings with ``result_mode="raw"``.
"""

import json
from typing import Any, Dict

from mu.tools import tool


# Series whose top-level elements carry an ``iter`` field, so an ``iter``
# filter can narrow them to a single iteration. ``layers_stacked`` (a dict of
# aligned arrays) and ``tool_histogram`` (per-tool aggregates) are intentionally
# excluded — iter filtering does not apply to them.
_ITER_FILTERABLE = {
    "context", "drift", "tokens", "latency", "memory_series",
    "subagent_timeline", "compaction_timeline", "nudge_timeline",
    "nudge_efficacy", "redundant_reads",
}

_VALID_SERIES = [
    "context", "layers_stacked", "drift", "tokens", "latency",
    "tool_histogram", "compaction_timeline", "nudge_timeline",
    "nudge_efficacy", "redundant_reads", "subagent_timeline", "memory_series",
]


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


def _scrub_json_text(text: str) -> str:
    """Redact known secret patterns from a serialized trace payload.

    Trace records embed raw tool-result and assistant previews, so a
    credential that reached a preview must not leak back out through the
    trace tools.
    """
    try:
        from mu.security.secret_paths import redact_secrets

        scrubbed, _ = redact_secrets(text)
        return scrubbed
    except Exception:
        return text


def _resolve(run_id: str):
    """Resolve run_id to a parsed TraceRun, or (None, error_str).

    ``latest`` resolves to the newest recorded run — skips the
    list_traces round-trip when debugging the most recent run.
    """
    from mu.trace import find_trace_path, list_trace_runs, parse_trace

    run_id = str(run_id or "").strip()
    if not run_id:
        return None, _err("run_id is required")
    if run_id.lower() == "latest":
        runs = list_trace_runs()  # newest-first
        if not runs:
            return None, _err("no traces recorded under $MUCLI_HOME/trace/")
        run_id = runs[0]["run_id"]
    path = find_trace_path(run_id)
    if path is None:
        return None, _err("trace run not found: " + run_id)
    return parse_trace(path), None


@tool(
    name="list_traces",
    description=(
        "List recorded agent-run traces under $MUCLI_HOME/trace/ (newest "
        "first), each with run_id, session, model, provider, mode, iter "
        "count, and bytes. Optional `session` substring narrows the list. "
        "Read-only — use this to discover which runs exist before pulling a "
        "summary or drilling into one with trace_summary / trace_series / "
        "trace_iteration."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session": {
                "type": "string",
                "description": (
                    "Optional case-insensitive substring filter on the run's "
                    "session name. Omit to list all runs."
                ),
            },
        },
        "required": [],
    },
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
    server_policy="allowed",
    result_mode="raw",
)
def list_traces(args: Dict[str, Any], context) -> str:
    from mu.trace import list_trace_runs

    runs = list_trace_runs()
    session = str(args.get("session") or "").strip().lower()
    if session:
        runs = [r for r in runs if session in str(r.get("session", "")).lower()]
    return _scrub_json_text(json.dumps({"runs": runs, "count": len(runs)}))


@tool(
    name="trace_summary",
    description=(
        "Get the at-a-glance overview of one run: iteration count, status, "
        "tokens in/out, cost, wall time (total/peak/mean), peak context, "
        "tokenizer drift (mean/median/peak |drift|), compaction count + "
        "mechanical fallbacks, nudge count + how many broke the loop, "
        "redundant reads, tool-call count, and subagent iteration count. "
        "The first call when debugging a run — it tells you which harness "
        "suspect (drift, emergency compaction, stuck loop, redundant reads) "
        "actually fired and how expensive the run was. Use trace_series / "
        "trace_iteration to drill in."
    ),
    parameters={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": (
                    "Run id (or a unique substring of the trace filename). "
                    "The special value 'latest' resolves to the newest run."
                ),
            },
        },
        "required": ["run_id"],
    },
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
    server_policy="allowed",
    result_mode="raw",
)
def trace_summary(args: Dict[str, Any], context) -> str:
    from mu.trace import build_series, build_summary

    run, err = _resolve(args.get("run_id"))
    if run is None:
        return err
    series = build_series(run)
    return _scrub_json_text(json.dumps(build_summary(run, series), default=str))


@tool(
    name="trace_series",
    description=(
        "Get a derived series for one run. With no `series` argument, returns "
        "the full series dict (all of them). With a `series` name, returns "
        "just that one. Optional `iter` narrows per-iteration and event series "
        "to that iteration (applies to: context, drift, tokens, latency, "
        "memory_series, subagent_timeline, compaction_timeline, "
        "nudge_timeline, nudge_efficacy, redundant_reads). Valid series: "
        "context, layers_stacked, drift, tokens, latency, tool_histogram, "
        "compaction_timeline, nudge_timeline, nudge_efficacy, "
        "redundant_reads, subagent_timeline, memory_series. Use "
        "trace_iteration for the full per-iteration drill-down (iter record + "
        "tools + nudges + compactions)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Run id (or a unique substring of the trace filename).",
            },
            "series": {
                "type": "string",
                "description": "Optional series name to return (see valid list above).",
                "enum": _VALID_SERIES,
            },
            "iter": {
                "type": "integer",
                "description": (
                    "Optional iteration number; narrows per-iter and event "
                    "series to that iteration."
                ),
            },
        },
        "required": ["run_id"],
    },
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
    server_policy="allowed",
    result_mode="raw",
)
def trace_series(args: Dict[str, Any], context) -> str:
    from mu.trace import build_series

    run, err = _resolve(args.get("run_id"))
    if run is None:
        return err
    series = build_series(run)
    name = str(args.get("series") or "").strip()
    iter_val = args.get("iter")
    if iter_val is not None:
        try:
            iter_val = int(iter_val)
        except (TypeError, ValueError):
            iter_val = None

    if not name:
        payload: Any = series
    elif name not in series:
        return json.dumps({
            "error": "unknown series: " + name,
            "valid": _VALID_SERIES,
        })
    else:
        value = series[name]
        if iter_val is not None and name in _ITER_FILTERABLE:
            value = [el for el in value
                     if isinstance(el, dict) and el.get("iter") == iter_val]
        payload = value

    return _scrub_json_text(json.dumps(payload, default=str))


@tool(
    name="trace_iteration",
    description=(
        "Drill into one iteration of a run: the raw iteration record (context "
        "layers, total_est vs actual prompt tokens, drift, tokens in/out/"
        "cached/reasoning, wall_ms, memory counts + by_status, subagent "
        "active/stuck/stall, assistant preview), the tool calls fired that "
        "iteration (name, path, ok, error_code, latency_ms, cache_hit, "
        "result_bytes, preview), and any nudges or compactions at that "
        "iteration. The 'what actually happened at iteration N' view — most "
        "useful for pinning a stuck loop, a latency spike, or a compaction "
        "event."
    ),
    parameters={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Run id (or a unique substring of the trace filename).",
            },
            "iter": {
                "type": "integer",
                "description": "Iteration number to inspect.",
            },
        },
        "required": ["run_id", "iter"],
    },
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
    server_policy="allowed",
    result_mode="raw",
)
def trace_iteration(args: Dict[str, Any], context) -> str:
    run, err = _resolve(args.get("run_id"))
    if run is None:
        return err
    try:
        iter_val = int(args.get("iter"))
    except (TypeError, ValueError):
        return _err("iter is required and must be an integer")

    iter_record = next((i for i in run.iters if i.get("iter") == iter_val), None)
    if iter_record is None:
        # Anomaly-aware hint (UX polish): point the caller at the
        # iterations that actually had events (tools/nudges/compactions),
        # not just the full iteration list.
        event_iters = sorted({
            t.get("iter") for t in run.tools if t.get("iter") is not None
        } | {
            c.get("iter") for c in run.compactions if c.get("iter") is not None
        })
        return json.dumps({
            "error": "iteration not found: " + str(iter_val),
            "run_id": run.run_id,
            "iters": [i.get("iter") for i in run.iters],
            "iters_with_events": event_iters,
            "hint": "use trace_summary(run_id).suspects to find anomalous iters, or pick from iters_with_events",
        })

    _tool_fields = (
        "name", "path", "ok", "error_code", "latency_ms",
        "cache_hit", "result_bytes", "preview",
    )
    tools = [
        {k: t.get(k) for k in _tool_fields if k in t}
        for t in run.tools
        if t.get("iter") == iter_val
    ]
    nudges = [
        {k: v for k, v in n.items() if k != "type"}
        for n in run.nudges
        if n.get("iteration") == iter_val or n.get("iter") == iter_val
    ]
    compactions = [
        {k: v for k, v in c.items() if k != "type"}
        for c in run.compactions
        if c.get("iter") == iter_val
    ]
    return _scrub_json_text(json.dumps({
        "run_id": run.run_id,
        "iter": iter_record,
        "tools": tools,
        "nudges": nudges,
        "compactions": compactions,
    }, default=str))