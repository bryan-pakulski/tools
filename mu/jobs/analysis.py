"""Retrospective performance analysis for durable engineering jobs.

The analyzer is deliberately built from durable job evidence rather than GUI
state. GUI, TUI and mobile can therefore inspect the same historical job after
browser/daemon restarts or archival.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

# Round-42 F6: hard cap on events materialized for analysis. Long-running
# jobs (200k+ events) previously loaded EVERYTHING — GUI analysis
# requests blocked and RSS ballooned. The newest window is enough for
# every consumer; full-history totals use COUNT queries instead.
_MAX_EVENTS = 20_000

from .models import Job, JobEvent, JobStatus
from .receipt import JobReceiptBuilder
from .service import JobService
from .verification import VerificationStore


ANALYSIS_SCHEMA_VERSION = 1

_FAILURE_STATES = {
    "failed", "environment_error", "timed_out", "budget_exceeded", "conflicted",
}
_ACTIVE_STATES = {"preparing", "running", "recovering"}
_WAITING_STATES = {"queued", "needs_human"}

_EVENT_CATEGORY = {
    "job_created": "lifecycle",
    "status_changed": "lifecycle",
    "attempt_started": "attempt",
    "attempt_finished": "attempt",
    "worker_lease_acquired": "worker",
    "worker_lease_released": "worker",
    "worker_process_started": "worker",
    "worker_process_exited": "worker",
    "worker_process_terminated": "worker",
    "worker_spawn_failed": "worker",
    "worker_rejected": "worker",
    "worktree_preflight_started": "git",
    "repository_inspected": "git",
    "job_base_resolved": "git",
    "worktree_inventory": "git",
    "worktree_add_started": "git",
    "worktree_prepare_failed": "git",
    "worktree_ready": "git",
    "worktree_removed": "git",
    "checkpoint_created": "git",
    "checkpoint_failed": "git",
    "verification_pending": "verification",
    "verification_evidence_created": "verification",
    "verification_failed": "verification",
    "verification_worker_error": "verification",
    "verification_lease_expired": "verification",
    "verification_contract_updated": "verification",
    "human_response": "human",
    "interaction_response": "human",
    "interaction_response_consumed": "human",
    "review_feedback": "human",
    "agent_message": "agent",
    "tool_call_ui": "tool",
    "tool_result_ui": "tool",
    "runtime_error": "runtime",
    "runtime_info": "runtime",
    "runtime_status": "runtime",
}


def _all_events(service: JobService, job_id: str) -> List[JobEvent]:
    """Round-42 F6: bounded event snapshot. The unbounded variant loaded
    EVERY event of a long-running job into memory before any analysis —
    a 200k-event job allocated all of it up front (GUI analysis requests
    blocked the event loop, analyzer RSS ballooned). The newest
    _MAX_EVENTS are enough for every consumer: phase intervals, attempt
    scoping, tool/model counters, gates/failures, and the timeline all
    slice from the tail."""
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
        # Keep the NEWEST window; totals that need the full count come
        # from a lightweight COUNT query (timeline_total_events), not
        # from materialized events.
        values = values[-_MAX_EVENTS:]
    return values


def _event_total(service: JobService, job_id: str) -> int:
    """Round-42 F6: exact full-history event count without materializing
    the events — keeps timeline_total_events truthful when the analysis
    window is capped."""
    store = service.store
    conn = store._connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()
    return int(row["n"]) if row is not None else 0


def _category(event: JobEvent) -> str:
    return _EVENT_CATEGORY.get(event.event_type, "system")


def _severity(event: JobEvent) -> str:
    if event.event_type in {
        "runtime_error", "worker_spawn_failed", "worker_rejected",
        "worktree_prepare_failed", "checkpoint_failed", "verification_worker_error",
    }:
        return "error"
    if event.event_type == "status_changed" and event.to_status and event.to_status.value in _FAILURE_STATES:
        return "error"
    if event.event_type in {"verification_failed", "verification_lease_expired"}:
        return "warning"
    if event.event_type == "status_changed" and event.to_status and event.to_status.value == "needs_human":
        return "warning"
    return "info"


def _event_summary(event: JobEvent) -> str:
    payload = event.payload if isinstance(event.payload, dict) else {}
    if event.event_type == "status_changed":
        left = event.from_status.value if event.from_status else "new"
        right = event.to_status.value if event.to_status else ""
        return f"{left} → {right}" + (f" · {event.reason}" if event.reason else "")
    if event.event_type == "tool_call_ui":
        return str(payload.get("tool_name") or "tool call")
    if event.event_type == "agent_message":
        return str(payload.get("text") or "")[:600]
    if event.event_type == "checkpoint_created":
        return f"{payload.get('label') or 'checkpoint'} · {str(payload.get('sha') or '')[:12]}"
    if event.reason:
        return str(event.reason)
    for key in ("error", "message", "summary", "text", "stage", "phase"):
        if payload.get(key):
            return str(payload[key])[:600]
    return ""


def _numeric_tokens(attempts: Iterable[Any]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for attempt in attempts:
        metadata = attempt.metadata if isinstance(attempt.metadata, dict) else {}
        result = metadata.get("agent_result") if isinstance(metadata.get("agent_result"), dict) else {}
        tokens = result.get("tokens") if isinstance(result.get("tokens"), dict) else {}
        for key, value in tokens.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[str(key)] += float(value)
    return dict(totals)


def _phase_analysis(job: Job, events: List[JobEvent], end_at: float) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    transitions = [event for event in events if event.event_type == "status_changed" and event.to_status]
    current = JobStatus.QUEUED.value
    started = float(job.created_at or (transitions[0].created_at if transitions else end_at))
    intervals: List[Dict[str, Any]] = []

    for event in transitions:
        when = max(started, float(event.created_at))
        duration = max(0.0, when - started)
        if duration or not intervals:
            intervals.append({
                "status": current,
                "started_at": started,
                "finished_at": when,
                "duration_seconds": duration,
                "transition_event_id": event.id,
                "reason": event.reason,
            })
        current = event.to_status.value
        started = when

    if end_at >= started:
        intervals.append({
            "status": current,
            "started_at": started,
            "finished_at": end_at,
            "duration_seconds": max(0.0, end_at - started),
            "transition_event_id": None,
            "reason": "",
        })

    grouped: Dict[str, Dict[str, Any]] = {}
    total = sum(float(item["duration_seconds"]) for item in intervals)
    for item in intervals:
        status = str(item["status"])
        row = grouped.setdefault(status, {"status": status, "seconds": 0.0, "occurrences": 0})
        row["seconds"] += float(item["duration_seconds"])
        row["occurrences"] += 1
    breakdown = sorted(grouped.values(), key=lambda item: item["seconds"], reverse=True)
    for item in breakdown:
        item["percent"] = (item["seconds"] / total * 100.0) if total > 0 else 0.0
    return intervals, breakdown


def _attempt_rows(attempts: List[Any], events: List[JobEvent]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attempt in attempts:
        # Round-35 F5: an unfinished (running/interrupted) attempt has no
        # finished_at — the old finish=started_at collapsed its event scope
        # to a single timestamp, reporting zero tool calls/messages/duration
        # for live execution. Scope it to the analysis snapshot instead,
        # capped by the next attempt's start when one exists.
        start = float(attempt.started_at or 0.0)
        finish = float(attempt.finished_at or 0.0)
        if finish <= start:
            finish = float("inf")
        scoped = [event for event in events if start <= float(event.created_at) <= finish]
        if not attempt.finished_at:
            next_start = next(
                (
                    float(other.started_at)
                    for other in attempts
                    if other is not attempt
                    and other.started_at
                    and float(other.started_at) > start
                ),
                None,
            )
            if next_start is not None:
                finish = next_start
                scoped = [event for event in events if start <= float(event.created_at) <= finish]
        metadata = attempt.metadata if isinstance(attempt.metadata, dict) else {}
        result = metadata.get("agent_result") if isinstance(metadata.get("agent_result"), dict) else {}
        tokens = result.get("tokens") if isinstance(result.get("tokens"), dict) else {}
        rows.append({
            "id": attempt.id,
            "number": attempt.number,
            "status": attempt.status,
            "started_at": attempt.started_at,
            "finished_at": attempt.finished_at,
            "duration_seconds": max(0.0, finish - start),
            "cost_usd": float(attempt.cost_usd or 0.0),
            "tokens": {str(k): float(v) for k, v in tokens.items() if isinstance(v, (int, float))},
            "tool_calls": sum(1 for event in scoped if event.event_type == "tool_call_ui"),
            "agent_messages": sum(1 for event in scoped if event.event_type == "agent_message"),
            "runtime_errors": sum(1 for event in scoped if _severity(event) == "error"),
            "agent_status": str(metadata.get("agent_status") or ""),
            "checkpoint": str(metadata.get("checkpoint") or ""),
            "error": attempt.error,
        })
    return rows


def build_job_analysis(service: JobService, job_id: str, *, timeline_limit: int = 5000) -> Dict[str, Any]:
    job = service.get(job_id)
    events = _all_events(service, job_id)
    attempts = service.attempts(job_id)
    verification_store = VerificationStore(service.store)
    verifications = verification_store.list(job_id, limit=500)
    receipt = JobReceiptBuilder(service).build(job_id)

    now = time.time()
    end_at = float(job.completed_at or (now if job.status in {
        JobStatus.PREPARING, JobStatus.RUNNING, JobStatus.VERIFYING,
        JobStatus.RECOVERING, JobStatus.NEEDS_HUMAN, JobStatus.QUEUED,
    } else job.updated_at or now))
    elapsed = max(0.0, end_at - float(job.created_at or end_at))
    intervals, phase_breakdown = _phase_analysis(job, events, end_at)
    phase_seconds = defaultdict(float)
    for item in intervals:
        phase_seconds[str(item["status"])] += float(item["duration_seconds"])

    attempt_rows = _attempt_rows(attempts, events)
    token_totals = _numeric_tokens(attempts)
    tools = Counter(
        str(event.payload.get("tool_name") or "unknown")
        for event in events
        if event.event_type == "tool_call_ui"
    )
    models = Counter(
        str(event.payload.get("model") or job.execution.get("model") or "unknown")
        for event in events
        if event.event_type == "agent_message"
    )
    tool_total = sum(tools.values())
    tool_rows = [
        {"name": name, "count": count, "share": (count / tool_total if tool_total else 0.0)}
        for name, count in tools.most_common()
    ]

    verification_rows = []
    for run in reversed(verifications):
        verification_rows.append({
            "id": run.id,
            "status": run.status,
            "passed": bool(run.passed),
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "duration_seconds": float(run.duration_ms or 0) / 1000.0,
            "checks": len(run.checks),
            "checks_passed": sum(1 for check in run.checks if check.passed),
            "changed_files": len(run.changed_files),
            "additions": int(run.additions or 0),
            "deletions": int(run.deletions or 0),
            "dirty": bool(run.dirty),
        })

    gates = []
    failures = []
    for event in events:
        if event.event_type == "status_changed" and event.to_status == JobStatus.NEEDS_HUMAN:
            gates.append({
                "event_id": event.id,
                "created_at": event.created_at,
                "reason": event.reason,
                "payload": event.payload,
            })
        if _severity(event) == "error":
            failures.append({
                "event_id": event.id,
                "created_at": event.created_at,
                "event_type": event.event_type,
                "summary": _event_summary(event),
                "payload": event.payload,
            })

    timeline_all = [{
        "id": event.id,
        "created_at": event.created_at,
        "elapsed_seconds": max(0.0, float(event.created_at) - float(job.created_at or event.created_at)),
        "event_type": event.event_type,
        "category": _category(event),
        "severity": _severity(event),
        "from_status": event.from_status.value if event.from_status else None,
        "to_status": event.to_status.value if event.to_status else None,
        "title": event.event_type.replace("_", " ").title(),
        "summary": _event_summary(event),
        "reason": event.reason,
        "payload": event.payload,
    } for event in events]
    bounded = max(100, min(int(timeline_limit), 20000))
    timeline = timeline_all[-bounded:]

    latest_verification = verifications[0] if verifications else None
    git = receipt.get("git") or {}
    changed_files = len(git.get("changed_files") or [])
    additions = int(git.get("additions") or 0)
    deletions = int(git.get("deletions") or 0)
    verification_passes = sum(1 for run in verifications if run.passed)
    verification_failures = sum(1 for run in verifications if not run.passed)
    human_responses = sum(1 for event in events if event.event_type in {"human_response", "interaction_response"})
    recoveries = sum(
        1 for event in events
        if event.event_type == "status_changed" and event.to_status == JobStatus.RECOVERING
    )

    active_seconds = sum(phase_seconds[s] for s in _ACTIVE_STATES)
    waiting_seconds = sum(phase_seconds[s] for s in _WAITING_STATES)
    verification_seconds = phase_seconds["verifying"]
    retries = max(0, len(attempts) - 1)
    first_pass = None if not verifications else bool(len(verifications) == 1 and latest_verification and latest_verification.passed)

    cumulative_cost = []
    running_cost = 0.0
    for row in attempt_rows:
        running_cost += float(row["cost_usd"])
        cumulative_cost.append({
            "attempt": row["number"],
            "created_at": row["finished_at"] or row["started_at"],
            "cost_usd": row["cost_usd"],
            "cumulative_cost_usd": running_cost,
        })

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "generated_at": now,
        "job": {
            **job.to_dict(),
            "archived": bool((job.metadata or {}).get("archived_at")),
            "archived_at": (job.metadata or {}).get("archived_at"),
        },
        "summary": {
            "elapsed_seconds": elapsed,
            "active_seconds": active_seconds,
            "waiting_seconds": waiting_seconds,
            "verification_seconds": verification_seconds,
            "queue_seconds": phase_seconds["queued"],
            "attempts": len(attempts),
            "retries": retries,
            "cost_usd": float(job.cost_usd or 0.0),
            "tokens": token_totals,
            "tool_calls": tool_total,
            "unique_tools": len(tools),
            "agent_messages": sum(1 for event in events if event.event_type == "agent_message"),
            "human_gates": len(gates),
            "human_responses": human_responses,
            "verification_runs": len(verifications),
            "verification_passes": verification_passes,
            "verification_failures": verification_failures,
            "verification_pass_rate": (verification_passes / len(verifications)) if verifications else None,
            "first_pass_verification": first_pass,
            "failures": len(failures),
            "recoveries": recoveries,
            "checkpoints": sum(1 for event in events if event.event_type == "checkpoint_created"),
            "changed_files": changed_files,
            "additions": additions,
            "deletions": deletions,
            "cost_per_attempt": (float(job.cost_usd or 0.0) / len(attempts)) if attempts else None,
            "cost_per_changed_file": (float(job.cost_usd or 0.0) / changed_files) if changed_files else None,
            "tool_calls_per_active_minute": (tool_total / (active_seconds / 60.0)) if active_seconds > 0 else None,
            "active_ratio": (active_seconds / elapsed) if elapsed > 0 else None,
            "waiting_ratio": (waiting_seconds / elapsed) if elapsed > 0 else None,
        },
        "phase_intervals": intervals,
        "phase_breakdown": phase_breakdown,
        "attempts": attempt_rows,
        "tools": tool_rows,
        "models": [{"name": name, "messages": count} for name, count in models.most_common()],
        "verifications": verification_rows,
        "human_gates": gates,
        "failures": failures[-100:],
        "timeline": timeline,
        # Round-42 F6: exact full-history total via COUNT — the materialized
        # window is capped, so len() would understate long jobs.
        "timeline_total_events": _event_total(service, job_id),
        "timeline_window_events": len(timeline_all),
        "series": {
            "cumulative_cost": cumulative_cost,
            "attempt_duration": [
                {"attempt": row["number"], "seconds": row["duration_seconds"], "status": row["status"]}
                for row in attempt_rows
            ],
            "attempt_tool_calls": [
                {"attempt": row["number"], "count": row["tool_calls"], "status": row["status"]}
                for row in attempt_rows
            ],
            "verification_duration": [
                {"index": index + 1, "seconds": row["duration_seconds"], "passed": row["passed"]}
                for index, row in enumerate(verification_rows)
            ],
        },
        "git": git,
        "execution": dict(job.execution or {}),
    }


def compare_job_analyses(primary: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    p = primary.get("summary") or {}
    c = comparison.get("summary") or {}
    specs = [
        ("elapsed_seconds", "Wall time", "seconds", True),
        ("active_seconds", "Active time", "seconds", True),
        ("waiting_seconds", "Waiting time", "seconds", True),
        ("attempts", "Attempts", "count", True),
        ("retries", "Retries", "count", True),
        ("cost_usd", "Cost", "usd", True),
        ("tool_calls", "Tool calls", "count", True),
        ("human_gates", "Human gates", "count", True),
        ("verification_failures", "Verification failures", "count", True),
        ("failures", "Runtime failures", "count", True),
        ("changed_files", "Changed files", "count", None),
    ]
    rows = []
    for key, label, unit, lower_better in specs:
        pv = p.get(key)
        cv = c.get(key)
        delta = None
        if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
            delta = float(pv) - float(cv)
        rows.append({
            "key": key,
            "label": label,
            "unit": unit,
            "primary": pv,
            "comparison": cv,
            "delta": delta,
            "lower_is_better": lower_better,
        })
    return {
        "primary_job_id": (primary.get("job") or {}).get("id"),
        "comparison_job_id": (comparison.get("job") or {}).get("id"),
        "metrics": rows,
    }
