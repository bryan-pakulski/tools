"""Subprocess entry point for one durable engineering-job attempt.

Each worker owns its process CWD, Session runtime, lease heartbeat and managed
Git worktree. The background controller can therefore run several jobs in
parallel without sharing Session process-global state.
"""

from __future__ import annotations

import argparse
import os
import threading
from types import SimpleNamespace
from typing import Optional

from .models import AttentionReason, JobStatus
from .receipt import JobReceiptBuilder
from .runner import JobRunOutcome, SessionJobRunner
from .service import JobService, JobStateError, get_default_job_service
from .worktree import JobWorktreeManager, WorktreeError


DEFAULT_JOB_SYSTEM = (
    "You are an autonomous software engineering agent working on a durable MuCLI job. "
    "Make focused, production-quality changes inside the attached job workspace. "
    "Use tools to inspect and validate the repository. Never claim merge readiness; "
    "the controller owns verification and review state."
)


def _base_args(job):
    execution = dict(job.execution or {})
    return SimpleNamespace(
        session=SessionJobRunner.session_name(job),
        provider=str(execution.get("provider") or ""),
        model=str(execution.get("model") or ""),
        provider_prevalidated=True,
        session_type=str(execution.get("session_type") or "workspace"),
        workspace=[job.worktree] if job.worktree else [],
        yolo=bool(execution.get("auto_approve_writes", False)),
        gui=False,
        trace=False,
        debug=bool(execution.get("debug", False)),
        system=str(execution.get("system_instruction") or DEFAULT_JOB_SYSTEM),
        system_file=None,
        mode_prompt=None,
    )


def _heartbeat(
    service: JobService,
    job_id: str,
    worker_id: str,
    ttl: int,
    stop: threading.Event,
    lease_lost: Optional[threading.Event] = None,
) -> None:
    interval = max(3.0, float(ttl) / 3.0)
    while not stop.wait(interval):
        try:
            if not service.heartbeat(job_id, worker_id, ttl_seconds=ttl):
                # Lease gone (expired and taken over / released / job
                # transitioned): signal the main worker to stop.
                if lease_lost is not None:
                    lease_lost.set()
                return
        except Exception:
            if lease_lost is not None:
                lease_lost.set()
            return


def _checkpoint(manager: JobWorktreeManager, service: JobService, job_id: str, label: str) -> str | None:
    try:
        return manager.checkpoint(service.get(job_id), label=label)
    except Exception as exc:
        payload = {"label": label}
        if isinstance(exc, WorktreeError):
            payload.update(exc.to_dict())
        service.store.append_event(
            job_id,
            "checkpoint_failed",
            reason=str(exc),
            payload=payload,
        )
        return None


def _refresh_receipt(service: JobService, job_id: str) -> None:
    try:
        JobReceiptBuilder(service).write(job_id)
    except Exception as exc:
        service.store.append_event(
            job_id,
            "work_receipt_failed",
            reason=str(exc),
        )


def _add_cost(service: JobService, job_id: str, amount: float) -> float:
    current = service.get(job_id)
    total = float(current.cost_usd or 0.0) + max(0.0, float(amount or 0.0))
    service.store.update_runtime_fields(job_id, cost_usd=total)
    return total


def _apply_outcome(
    service: JobService,
    manager: JobWorktreeManager,
    job_id: str,
    attempt_id: str,
    attempt_number: int,
    outcome: JobRunOutcome,
    *,
    worker_id: str = "",
) -> int:
    """Apply a run outcome under the worker's lease.

    Round-39 F1: every branch finishes the attempt AND transitions the
    job through store.finish_attempt_owned() — a single lease-conditioned
    transaction. Round-40 F1: job-cost accumulation happens INSIDE that
    transaction (cost_add) — a superseded worker can no longer add its
    cost to the replacement's job before failing the ownership check.
    Round-40 F4: on lease loss every branch returns 5 IMMEDIATELY — no
    receipt refresh (the job may have been deleted by the new owner).
    """
    current = service.get(job_id)
    common_metadata = {
        "agent_status": outcome.status,
        "agent_result": dict(outcome.result or {}),
    }
    # Checkpoints are git operations OUTSIDE the lease transaction; if
    # ownership is lost below, the checkpoint remains as harmless evidence
    # on the managed branch (it never mutates the job row).
    if current.status == JobStatus.CANCELLED:
        checkpoint = _checkpoint(manager, service, job_id, f"attempt-{attempt_number}-cancelled")
        owned = service.store.finish_attempt_owned(
            attempt_id,
            worker_id=worker_id,
            status="cancelled",
            cost_usd=outcome.cost_usd,
            cost_add=outcome.cost_usd,
            metadata={**common_metadata, "cancelled_while_running": True, "checkpoint": checkpoint},
        )
        if not owned:
            return 5
        _refresh_receipt(service, job_id)
        return 0

    if outcome.kind == "needs_human":
        checkpoint = _checkpoint(manager, service, job_id, f"attempt-{attempt_number}-needs-human")
        attention_meta = {
            **common_metadata,
            "attention": outcome.attention_payload,
            # Round-16 F20: persist the reason so crash recovery can
            # idempotently re-enter NEEDS_HUMAN without re-running the
            # attempt (the lease-recovery path replays from this row).
            "attention_reason": str(
                getattr(outcome.attention_reason, "value", None)
                or outcome.attention_reason or ""
            ),
            "attention_detail": outcome.attention_detail,
            "checkpoint": checkpoint,
        }
        owned = service.store.finish_attempt_owned(
            attempt_id,
            worker_id=worker_id,
            status="needs_human",
            cost_usd=outcome.cost_usd,
            cost_add=outcome.cost_usd,
            metadata=attention_meta,
            target_status=JobStatus.NEEDS_HUMAN,
            transition_reason="human attention required",
            transition_payload={**outcome.attention_payload, "checkpoint": checkpoint},
            attention_reason=outcome.attention_reason,
            attention_detail=outcome.attention_detail,
        )
        if not owned:
            return 5
        _refresh_receipt(service, job_id)
        return 20

    if outcome.kind == "completed":
        checkpoint = _checkpoint(manager, service, job_id, f"attempt-{attempt_number}-implementation")
        owned = service.store.finish_attempt_owned(
            attempt_id,
            worker_id=worker_id,
            status="completed",
            cost_usd=outcome.cost_usd,
            cost_add=outcome.cost_usd,
            metadata={**common_metadata, "checkpoint": checkpoint},
            target_status=JobStatus.VERIFYING,
            transition_reason="implementation attempt completed",
            transition_payload={"checkpoint": checkpoint, "attempt_id": attempt_id},
        )
        if not owned:
            return 5
        service.store.append_event(
            job_id,
            "verification_pending",
            reason="deterministic verifier scheduled",
            payload={"attempt_id": attempt_id, "checkpoint": checkpoint},
        )
        _refresh_receipt(service, job_id)
        return 0

    checkpoint = _checkpoint(manager, service, job_id, f"attempt-{attempt_number}-failed")
    target = JobStatus.ENVIRONMENT_ERROR if outcome.status == "environment_error" else JobStatus.FAILED
    owned = service.store.finish_attempt_owned(
        attempt_id,
        worker_id=worker_id,
        status="failed",
        error=outcome.error,
        cost_usd=outcome.cost_usd,
        cost_add=outcome.cost_usd,
        metadata={**common_metadata, "checkpoint": checkpoint},
        target_status=target,
        transition_reason="job attempt failed",
        transition_payload={
            "error": outcome.error,
            "agent_status": outcome.status,
            "checkpoint": checkpoint,
        },
    )
    if not owned:
        return 5
    _refresh_receipt(service, job_id)
    return 1


def run_job(job_id: str, worker_id: str, *, lease_ttl_seconds: int = 45) -> int:
    service = get_default_job_service()
    manager = JobWorktreeManager(service)
    heartbeat_stop = threading.Event()
    lease_lost = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat,
        args=(service, job_id, worker_id, lease_ttl_seconds, heartbeat_stop, lease_lost),
        name=f"mucli-job-worker-heartbeat-{job_id[:8]}",
        daemon=True,
    )
    attempt = None
    try:
        job = service.get(job_id)
        if job.worker_id != worker_id:
            service.store.append_event(
                job_id,
                "worker_rejected",
                reason="worker does not own job lease",
                payload={"expected": job.worker_id, "received": worker_id},
            )
            return 2
        heartbeat_thread.start()

        if job.status == JobStatus.QUEUED:
            job = service.transition(job_id, JobStatus.PREPARING, reason="worker preparing isolated workspace")
        elif job.status != JobStatus.RECOVERING:
            return 0

        execution = dict(job.execution or {})
        if str(execution.get("session_type") or "workspace") == "workspace":
            try:
                manager.prepare(job)
            except WorktreeError as exc:
                diagnostics = exc.to_dict()
                diagnostics.update(
                    {
                        "repository": job.repository,
                        "requested_base_branch": job.base_branch,
                        "requested_base_sha": job.base_sha,
                        "worker_process_id": os.getpid(),
                    }
                )
                service.transition(
                    job_id,
                    JobStatus.ENVIRONMENT_ERROR,
                    reason="could not prepare isolated Git worktree",
                    payload=diagnostics,
                )
                _refresh_receipt(service, job_id)
                return 3
            job = service.get(job_id)

        attempt = service.start_attempt(
            job_id,
            worker_id=worker_id,
            session_name=SessionJobRunner.session_name(job),
            metadata={
                "process_id": os.getpid(),
                "branch": job.branch,
                "worktree": job.worktree,
                "base_sha": job.base_sha,
            },
        )
        job = service.transition(job_id, JobStatus.RUNNING, reason="isolated worker attempt started")

        import mucli as mucli_entry

        runner = SessionJobRunner(
            service,
            build_session_fn=mucli_entry.build_session,
            base_args=_base_args(job),
        )
        outcome = runner.run(job, attempt)
        if lease_lost.is_set():
            # Heartbeat failed mid-run: a replacement worker may own the
            # job. Do NOT checkpoint/finish/transition — that would race
            # the replacement and corrupt its state.
            service.store.append_event(
                job_id,
                "worker_lease_lost_midrun",
                reason="heartbeat lost; worker abandoning outcome",
                payload={"worker_id": worker_id, "attempt_id": attempt.id},
            )
            return 5
        if not service.store.assert_lease(job_id, worker_id):
            # Final transactional gate: the heartbeat check above can race
            # an expiry + takeover between the check and _apply_outcome.
            service.store.append_event(
                job_id,
                "worker_lease_lost_midrun",
                reason="lease lost before outcome application",
                payload={"worker_id": worker_id, "attempt_id": attempt.id},
            )
            return 5
        # Round-39 F1: worker_id threaded through — _apply_outcome now
        # enforces ownership transactionally; the assert_lease gate above
        # remains as a fast pre-check.
        return _apply_outcome(
            service, manager, job_id, attempt.id, attempt.number, outcome,
            worker_id=worker_id,
        )

    except JobStateError:
        return 0
    except BaseException as exc:
        # Round-36 F3: the crash path is fenced exactly like the success
        # path — a worker that has lost its lease (heartbeat reported loss,
        # takeover acquired it, or it lapsed) must NOT mark the attempt
        # failed or transition a job that now belongs to a replacement.
        # Without this, a shutdown-time exception after lease loss flipped
        # the replacement worker's job to FAILED. Detached diagnostics
        # (events only) are still allowed.
        still_owned = False
        try:
            still_owned = service.store.assert_lease(job_id, worker_id)
        except Exception:
            still_owned = False
        # Round-40 F2: the ONLY attempt write is the atomic
        # finish_attempt_owned(target_status=FAILED) below. The previous
        # duplicate unconditioned finish already committed 'failed' before
        # the atomic call — a later TIMED_OUT transition could no longer
        # roll it back, defeating the guarantee, and every ordinary crash
        # emitted a duplicate attempt_finished event.
        if attempt is not None and still_owned:
            try:
                # Round-39/40 F2: finish the attempt AND transition to
                # FAILED in the same lease-conditioned transaction. A
                # runtime deadline may already have flipped the job to
                # TIMED_OUT (TIMED_OUT→FAILED is rejected) — the
                # JobStateError rolls the whole transaction back, so a
                # timed-out job never gets a misleading 'failed' attempt.
                service.store.finish_attempt_owned(
                    attempt.id,
                    worker_id=worker_id,
                    status="failed",
                    error=str(exc),
                    target_status=JobStatus.FAILED,
                    transition_reason="isolated worker crashed",
                    transition_payload={
                        "error": str(exc),
                        "process_id": os.getpid(),
                        "exception_type": type(exc).__name__,
                    },
                )
            except Exception:
                pass
        elif attempt is not None:
            try:
                service.store.append_event(
                    job_id,
                    "worker_crash_result_discarded",
                    reason="lease lost; crash outcome not applied to superseded job",
                    payload={"attempt_id": attempt.id, "error": str(exc)[:500]},
                )
            except Exception:
                pass
        # Round-40 F4: receipt refresh is best-effort in both branches —
        # a job deleted by the new owner must not turn the clean lease-loss
        # exit into a worker crash.
        try:
            _refresh_receipt(service, job_id)
        except Exception:
            pass
        return 4
    finally:
        heartbeat_stop.set()
        if heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=1.0)
        try:
            service.release(job_id, worker_id, reason="isolated worker exited")
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one MuCLI durable engineering job")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--lease-ttl", type=int, default=45)
    args = parser.parse_args()
    return run_job(args.job_id, args.worker_id, lease_ttl_seconds=args.lease_ttl)


if __name__ == "__main__":
    raise SystemExit(main())
