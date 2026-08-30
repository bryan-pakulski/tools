"""Deep retrospective evidence for Job Trace Analyzer.

This layer enriches the stable aggregate model from :mod:`mu.jobs.analysis`
without changing job execution semantics. It makes an important distinction:
being *in* a failure state for 37 minutes is not the same as spending 37
minutes actively failing. Failure states are stopped/resident until retry.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# Round-42 F6: hard cap on events materialized for the detail layer
# (mirrors mu/jobs/analysis.py _MAX_EVENTS).
_MAX_EVENTS = 20_000

from .models import JobEvent
from .service import JobService


_ACTIVE = {"preparing", "running", "recovering"}
_VERIFY = {"verifying"}
_WAITING = {"queued", "needs_human"}
_STOPPED = {"failed", "environment_error", "timed_out", "budget_exceeded", "conflicted"}
_REVIEW = {"ready_for_review"}
_TERMINAL = {"cancelled", "merged"}

_STATE_EXPLANATION = {
    "queued": "Waiting for a controller worker lease; no agent execution is expected in this state.",
    "preparing": "Worker is actively resolving repository/base state and preparing the isolated worktree.",
    "running": "The MuCLI agent runtime is actively executing this engineering attempt.",
    "recovering": "Controller/worker recovery is actively reconciling a previously interrupted execution.",
    "needs_human": "Execution is paused until a durable human response is supplied.",
    "verifying": "Deterministic verification is actively running against the isolated worktree.",
    "ready_for_review": "Implementation is stopped and waiting for review; this is not agent execution time.",
    "environment_error": "Execution is stopped after an environment/worktree failure and remains resident here until retry or cancellation.",
    "failed": "Execution is stopped after an implementation/runtime failure and remains resident here until retry or cancellation.",
    "timed_out": "Execution is stopped because a runtime limit was reached.",
    "budget_exceeded": "Execution is stopped because a configured budget was reached.",
    "conflicted": "Execution is stopped on a conflict and requires resolution before work can continue.",
    "cancelled": "Terminal cancelled state; no execution occurs after entry.",
    "merged": "Terminal merged state; no execution occurs after entry.",
}

_ACTIVITY_EVENT_TYPES = {
    "worker_process_started", "worker_process_exited", "worker_process_terminated",
    "worktree_preflight_started", "repository_inspected", "job_base_resolved",
    "worktree_inventory", "worktree_add_started", "worktree_ready",
    "agent_message", "tool_call_ui", "tool_result_ui", "runtime_status", "runtime_info",
    "runtime_error", "verification_pending", "verification_evidence_created",
    "verification_failed", "verification_worker_error", "checkpoint_created",
}


def _all_events(service: JobService, job_id: str) -> List[JobEvent]:
    """Round-42 F6: bounded event snapshot (mirrors analysis.py cap). The
    detail layer re-scanned the FULL event history on top of the raw
    analysis — a long job paid the unbounded load twice. The newest
    _MAX_EVENTS cover every interval detail (intervals are built from the
    same tail window)."""
    values: List[JobEvent] = []
    after = 0
    total_seen = 0
    while True:
        batch = service.events(job_id, after_id=after, limit=5000)
        if not batch:
            break
        values.extend(batch)
        total_seen += len(batch)
        after = batch[-1].id
        if len(batch) < 5000 or total_seen >= _MAX_EVENTS:
            break
    if total_seen > _MAX_EVENTS:
        values = values[-_MAX_EVENTS:]
    return values


def _classification(status: str) -> str:
    if status in _ACTIVE:
        return "active"
    if status in _VERIFY:
        return "verification"
    if status in _WAITING:
        return "waiting"
    if status in _STOPPED:
        return "stopped"
    if status in _REVIEW:
        return "review"
    if status in _TERMINAL:
        return "terminal"
    return "other"


def _bounded_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Round-42 F6: cap every preview payload — string fields to 400
    chars, nested structures to a 2000-char JSON dump. Interval previews
    previously embedded FULL payloads (agent messages, tool results,
    trace blobs) for up to 250 events per interval, so the enriched
    response for a long job serialized megabytes to the GUI."""
    bounded: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            bounded[key] = value[:400]
        elif isinstance(value, (int, float, bool)) or value is None:
            bounded[key] = value
        else:
            try:
                bounded[key] = json.dumps(value, ensure_ascii=False, default=str)[:2000]
            except (TypeError, ValueError):
                bounded[key] = str(value)[:2000]
    return bounded


def _event_preview(event: JobEvent) -> Dict[str, Any]:
    payload = event.payload if isinstance(event.payload, dict) else {}
    if event.event_type == "status_changed":
        left = event.from_status.value if event.from_status else "new"
        right = event.to_status.value if event.to_status else ""
        summary = f"{left} → {right}" + (f" · {event.reason}" if event.reason else "")
    elif event.event_type == "tool_call_ui":
        summary = str(payload.get("tool_name") or "tool call")
    elif event.event_type == "agent_message":
        summary = str(payload.get("text") or "")[:420]
    else:
        summary = str(
            event.reason
            or payload.get("error")
            or payload.get("message")
            or payload.get("summary")
            or payload.get("text")
            or payload.get("stage")
            or payload.get("phase")
            or ""
        )[:420]
    return {
        "id": event.id,
        "event_type": event.event_type,
        "created_at": event.created_at,
        "summary": summary,
        "reason": event.reason,
        "from_status": event.from_status.value if event.from_status else None,
        "to_status": event.to_status.value if event.to_status else None,
        # Round-42 F6: bounded payload — full payloads belonged in the
        # detail drill-down, not 250-per-interval previews.
        "payload": _bounded_payload(payload),
    }


def _interval_detail(interval: Dict[str, Any], events: List[JobEvent]) -> Dict[str, Any]:
    start = float(interval.get("started_at") or 0.0)
    finish = float(interval.get("finished_at") or start)
    status = str(interval.get("status") or "")
    classification = _classification(status)

    entry = None
    exit_event = None
    inside: List[JobEvent] = []
    for event in events:
        when = float(event.created_at)
        if event.event_type == "status_changed" and event.to_status and event.to_status.value == status:
            if abs(when - start) < 0.001 or (when <= start and (entry is None or when > float(entry.created_at))):
                entry = event
        if event.event_type == "status_changed" and event.from_status and event.from_status.value == status:
            if abs(when - finish) < 0.001 or (when >= finish and (exit_event is None or when < float(exit_event.created_at))):
                exit_event = event
        if start <= when < finish and not (
            event.event_type == "status_changed" and event.to_status and event.to_status.value == status and abs(when - start) < 0.001
        ):
            inside.append(event)

    activity = [event for event in inside if event.event_type in _ACTIVITY_EVENT_TYPES]
    worker_activity = [event for event in activity if event.event_type.startswith("worker_")]
    agent_activity = [event for event in activity if event.event_type in {"agent_message", "tool_call_ui", "tool_result_ui"}]
    duration = max(0.0, finish - start)

    if classification == "stopped":
        if activity:
            interpretation = (
                f"Stopped-state residence for {duration:.1f}s. The failure occurred at/near state entry; "
                f"{len(activity)} diagnostic/cleanup event(s) were recorded while stopped, but this interval is not agent execution time."
            )
        else:
            interpretation = (
                f"Stopped-state residence for {duration:.1f}s. The failure occurred at state entry and no worker/agent activity "
                "was recorded before the next retry/cancel transition."
            )
    elif classification == "waiting":
        interpretation = (
            f"Passive waiting residence for {duration:.1f}s; no agent execution is attributed to this interval."
        )
    elif classification == "review":
        interpretation = (
            f"Review residence for {duration:.1f}s; implementation had stopped and was waiting for a reviewer decision."
        )
    elif classification == "terminal":
        interpretation = f"Terminal residence for {duration:.1f}s; no further execution occurs."
    elif classification == "verification":
        interpretation = f"Deterministic verification activity for {duration:.1f}s."
    elif classification == "active":
        interpretation = f"Active controller/agent execution for {duration:.1f}s."
    else:
        interpretation = f"Lifecycle residence for {duration:.1f}s."

    return {
        **interval,
        "classification": classification,
        "explanation": _STATE_EXPLANATION.get(status, "Lifecycle state."),
        "interpretation": interpretation,
        "active_execution": classification == "active",
        "passive_residence": classification in {"waiting", "stopped", "review", "terminal"},
        "entry_event": _event_preview(entry) if entry else None,
        "exit_event": _event_preview(exit_event) if exit_event else None,
        "events": [_event_preview(event) for event in inside[-250:]],
        "event_count": len(inside),
        "activity_event_count": len(activity),
        "worker_event_count": len(worker_activity),
        "agent_event_count": len(agent_activity),
    }


def _runtime_trace(job_session_name: str) -> Dict[str, Any]:
    session_name = str(job_session_name or "").strip()
    if not session_name:
        return {
            "available": False,
            "session_name": "",
            "trace_url": "",
            "reason": "The job never reached a durable agent session.",
        }
    try:
        from mu.trace import build_session_view, load_session_runs

        runs = load_session_runs(session_name)
        if not runs:
            return {
                "available": False,
                "session_name": session_name,
                "trace_url": f"/trace?session={session_name}",
                "reason": "No harness trace was recorded for this job session. Older durable-job attempts ran with trace recording disabled.",
            }
        view = build_session_view(runs, 128)
        summary = dict(view.get("summary") or {})
        series = dict(view.get("series") or {})
        run_bounds = list(view.get("run_bounds") or [])
        keep_summary = {
            key: summary.get(key)
            for key in (
                "iters", "status", "total_in", "total_out", "total_cost", "total_wall_ms",
                "peak_context", "peak_request_estimate", "peak_request_delta", "compaction_count",
                "mechanical_fallback_count", "nudge_count", "nudges_broken", "redundant_reads",
                "tool_calls", "subagent_iters",
            )
            if key in summary
        }
        return {
            "available": True,
            "session_name": session_name,
            "trace_url": f"/trace?session={session_name}",
            "run_count": len(runs),
            "run_bounds": run_bounds,
            "summary": keep_summary,
            "top_context_spikes": list(series.get("top_context_spikes") or [])[:20],
            "reason": "Full MuCLI harness trace is available for iteration-level drill-down.",
        }
    except Exception as exc:
        return {
            "available": False,
            "session_name": session_name,
            "trace_url": f"/trace?session={session_name}",
            "reason": f"Harness trace could not be parsed: {exc}",
        }


def enrich_job_analysis(service: JobService, job_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    events = _all_events(service, job_id)
    intervals = list(analysis.get("phase_intervals") or [])
    detailed = [_interval_detail(interval, events) for interval in intervals]
    analysis["phase_intervals"] = detailed

    summary = analysis.setdefault("summary", {})
    classification_seconds: Dict[str, float] = {}
    stopped_by_status: Dict[str, float] = {}
    for interval in detailed:
        classification = str(interval.get("classification") or "other")
        seconds = float(interval.get("duration_seconds") or 0.0)
        classification_seconds[classification] = classification_seconds.get(classification, 0.0) + seconds
        if classification == "stopped":
            status = str(interval.get("status") or "stopped")
            stopped_by_status[status] = stopped_by_status.get(status, 0.0) + seconds
    summary["classification_seconds"] = classification_seconds
    summary["stopped_seconds"] = classification_seconds.get("stopped", 0.0)
    summary["review_wait_seconds"] = classification_seconds.get("review", 0.0)
    summary["passive_seconds"] = sum(
        classification_seconds.get(name, 0.0)
        for name in ("waiting", "stopped", "review", "terminal")
    )
    summary["stopped_by_status"] = stopped_by_status

    job = service.get(job_id)
    analysis["runtime_trace"] = _runtime_trace(job.session_name)
    return analysis


__all__ = ["enrich_job_analysis"]
