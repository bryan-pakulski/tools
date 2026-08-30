"""Subprocess verifier for durable engineering jobs."""

from __future__ import annotations

import argparse
import os
import threading
from typing import Optional

from .models import AttentionReason, JobStatus
from .receipt import JobReceiptBuilder
from .review_branch import materialize_review_branch
from .service import JobStateError, get_default_job_service
from .verification import DeterministicVerifier, VerificationRun
from .worktree import WorktreeError


def _heartbeat(
    service,
    job_id: str,
    worker_id: str,
    ttl: int,
    stop: threading.Event,
    lease_lost: "Optional[threading.Event]" = None,
) -> None:
    interval = max(3.0, float(ttl) / 3.0)
    while not stop.wait(interval):
        try:
            if not service.heartbeat(job_id, worker_id, ttl_seconds=ttl):
                # Lease gone: signal the verifier to stop before it can
                # apply results it no longer owns.
                if lease_lost is not None:
                    lease_lost.set()
                return
        except Exception:
            if lease_lost is not None:
                lease_lost.set()
            return


def verification_feedback(run: VerificationRun) -> dict:
    failed = []
    for check in run.checks:
        if check.passed:
            continue
        failed.append({
            "command": check.command,
            "return_code": check.return_code,
            "timed_out": check.timed_out,
            "error": check.error,
            "stdout": check.stdout[-6000:],
            "stderr": check.stderr[-6000:],
        })
    return {
        "verification_id": run.id,
        "status": run.status,
        "verified_head_sha": run.head_sha,
        "manifest_path": run.manifest_path,
        "summary": run.summary,
        "failed_checks": failed,
        "dirty_status": run.dirty_status[-6000:] if run.dirty else "",
    }


def _refresh_receipt(service, job_id: str) -> None:
    try:
        JobReceiptBuilder(service).write(job_id)
    except Exception as exc:
        service.store.append_event(
            job_id,
            "work_receipt_failed",
            reason=str(exc),
        )


def _materialize_for_review(service, job_id: str, feedback: dict) -> tuple[bool, dict]:
    """Convert the verified execution worktree into a normal review branch.

    Verification success is necessary but review state is only entered once the
    branch is confirmed and the temporary worktree is safely retired.  This
    keeps READY_FOR_REVIEW usable from ordinary Git tooling instead of requiring
    callers to navigate MuCLI's managed worktree directory.
    """

    try:
        info = materialize_review_branch(
            service,
            job_id,
            expected_head_sha=str(feedback.get("verified_head_sha") or "") or None,
        )
        return True, {**feedback, "review_branch": info.to_dict()}
    except WorktreeError as exc:
        payload = {**feedback, "review_branch_error": exc.to_dict()}
        service.store.append_event(
            job_id,
            "review_branch_materialization_failed",
            reason=str(exc),
            payload=payload,
        )
        service.require_human(
            job_id,
            AttentionReason.ENVIRONMENT_FAILURE,
            "Verification passed, but MuCLI could not safely move the completed work into branch-only review state.",
            payload=payload,
        )
        return False, payload
    except Exception as exc:
        payload = {**feedback, "review_branch_error": {"error": str(exc)}}
        service.store.append_event(
            job_id,
            "review_branch_materialization_failed",
            reason=str(exc),
            payload=payload,
        )
        service.require_human(
            job_id,
            AttentionReason.ENVIRONMENT_FAILURE,
            "Verification passed, but MuCLI could not safely move the completed work into branch-only review state.",
            payload=payload,
        )
        return False, payload


def apply_verification_result(service, job_id: str, run: VerificationRun) -> int:
    """Apply deterministic readiness policy to persisted verification evidence."""
    feedback = verification_feedback(run)
    current = service.get(job_id)
    if current.status == JobStatus.CANCELLED:
        _refresh_receipt(service, job_id)
        return 0
    if current.status != JobStatus.VERIFYING:
        _refresh_receipt(service, job_id)
        return 0

    if run.status == "missing_contract":
        service.require_human(
            job_id,
            AttentionReason.VERIFICATION_REQUIRED,
            "No validation commands are configured, so MuCLI cannot automatically verify this job.",
            payload=feedback,
        )
        _refresh_receipt(service, job_id)
        return 21

    if run.passed:
        materialized, review_feedback = _materialize_for_review(service, job_id, feedback)
        if not materialized:
            _refresh_receipt(service, job_id)
            return 23
        service.transition(
            job_id,
            JobStatus.READY_FOR_REVIEW,
            reason="deterministic verification passed; review branch materialized",
            payload=review_feedback,
        )
        _refresh_receipt(service, job_id)
        return 0

    attempts = service.attempts(job_id)
    latest_attempt = max((attempt.number for attempt in attempts), default=0)
    if latest_attempt <= int(current.max_retries):
        service.store.append_event(
            job_id,
            "verification_failed",
            reason="automatic repair attempt scheduled",
            payload=feedback,
        )
        service.transition(
            job_id,
            JobStatus.QUEUED,
            reason="verification failed; requeueing implementation",
            payload={
                "verification_id": run.id,
                "attempt": latest_attempt,
                "max_retries": current.max_retries,
            },
        )
        _refresh_receipt(service, job_id)
        return 10

    service.store.append_event(
        job_id,
        "verification_failed",
        reason="automatic retry budget exhausted",
        payload=feedback,
    )
    service.require_human(
        job_id,
        AttentionReason.TEST_FAILURE,
        "Verification failed and the automatic implementation retry budget is exhausted.",
        payload=feedback,
    )
    _refresh_receipt(service, job_id)
    return 22


def run_verification(job_id: str, worker_id: str, *, lease_ttl_seconds: int = 45) -> int:
    service = get_default_job_service()
    heartbeat_stop = threading.Event()
    lease_lost = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat,
        args=(service, job_id, worker_id, lease_ttl_seconds, heartbeat_stop, lease_lost),
        name=f"mucli-job-verify-heartbeat-{job_id[:8]}",
        daemon=True,
    )
    try:
        job = service.get(job_id)
        if job.worker_id != worker_id:
            service.store.append_event(
                job_id,
                "verification_worker_rejected",
                reason="worker does not own job lease",
                payload={"expected": job.worker_id, "received": worker_id},
            )
            return 2
        if job.status != JobStatus.VERIFYING:
            return 0
        heartbeat_thread.start()
        run = DeterministicVerifier(service).verify(job)
        if lease_lost.is_set():
            # Lease expired mid-verification: a retried verifier will take
            # over. Do not apply results this worker no longer owns.
            service.store.append_event(
                job_id,
                "verification_lease_lost_midrun",
                reason="heartbeat lost; verifier abandoning result",
                payload={"worker_id": worker_id},
            )
            return 6
        if not service.store.assert_lease(job_id, worker_id):
            # Final transactional gate before apply_verification_result:
            # closes the check-then-act window against takeover.
            service.store.append_event(
                job_id,
                "verification_lease_lost_midrun",
                reason="lease lost before result application",
                payload={"worker_id": worker_id},
            )
            return 6
        return apply_verification_result(service, job_id, run)

    except JobStateError:
        return 0
    except Exception as exc:
        try:
            service.store.append_event(
                job_id,
                "verification_worker_error",
                reason=str(exc),
                payload={"process_id": os.getpid()},
            )
            _refresh_receipt(service, job_id)
        except Exception:
            pass
        # Leave VERIFYING in place. Once this lease releases/expires the
        # controller starts a fresh deterministic verifier, not implementation.
        return 4
    finally:
        heartbeat_stop.set()
        if heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=1.0)
        try:
            service.release(job_id, worker_id, reason="verification worker exited")
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one MuCLI durable engineering job")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--lease-ttl", type=int, default=45)
    args = parser.parse_args()
    return run_verification(args.job_id, args.worker_id, lease_ttl_seconds=args.lease_ttl)


if __name__ == "__main__":
    raise SystemExit(main())
