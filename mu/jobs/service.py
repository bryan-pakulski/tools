"""Control-plane-neutral service for durable engineering jobs."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Iterable, List, Optional

from .models import AttentionReason, Job, JobAttempt, JobEvent, JobSpec, JobStatus, can_transition, coerce_status
from .store import JobStore

logger = logging.getLogger("mucli")


class JobStateError(RuntimeError):
    pass


class JobService:
    def __init__(self, store: Optional[JobStore] = None):
        self.store = store or JobStore()

    def create(self, spec: JobSpec, *, job_id: Optional[str] = None) -> Job:
        return self.store.create_job(spec, job_id=job_id)

    def create_from_payload(self, payload: Dict[str, Any]) -> Job:
        value = dict(payload or {})
        repository = str(value.get("repository") or value.get("repo") or "").strip()
        execution = dict(value.get("execution") or {})
        metadata = dict(value.get("metadata") or {})
        supplied_base_branch = str(value.get("base_branch") or "").strip()
        supplied_base_sha = str(value.get("base_sha") or "").strip()
        base_branch = supplied_base_branch or "main"
        base_sha = supplied_base_sha

        # GUI/mobile normally submit a workspace but no base. Resolve that
        # ambiguity synchronously, before the job is queued, so a bad path or
        # non-Git directory is reported immediately instead of producing an
        # opaque child-worker environment failure several seconds later.
        # Snapshot the current committed HEAD: that is the actual code state
        # the user was looking at when delegating the ticket.
        session_type = str(execution.get("session_type") or "workspace").strip().lower()
        if repository and session_type == "workspace" and not supplied_base_branch and not supplied_base_sha:
            try:
                from .repository import RepositoryRegistry

                info = RepositoryRegistry.inspect(repository)
            except Exception as exc:
                raise ValueError(f"Repository preflight failed for {repository}: {exc}") from exc
            base_branch = str(info.get("current_branch") or info.get("default_branch") or "main")
            base_sha = str(info.get("head_sha") or "")
            if not base_sha:
                raise ValueError(
                    f"Repository preflight failed for {repository}: could not resolve current HEAD commit"
                )
            metadata["submission_repository_preflight"] = {
                "canonical_path": info.get("canonical_path", ""),
                "git_common_dir": info.get("git_common_dir", ""),
                "origin_url": info.get("origin_url", ""),
                "current_branch": info.get("current_branch", ""),
                "default_branch": info.get("default_branch", ""),
                "remote_default_ref": info.get("remote_default_ref", ""),
                "head_sha": base_sha,
                "clean": bool(info.get("clean", False)),
            }

        return self.create(JobSpec(
            title=str(value.get("title") or ""),
            description=str(value.get("description") or ""),
            repository=repository,
            base_branch=base_branch,
            base_sha=base_sha,
            acceptance_criteria=list(value.get("acceptance_criteria") or []),
            validation_commands=list(value.get("validation_commands") or []),
            max_cost_usd=value.get("max_cost_usd"),
            max_runtime_seconds=value.get("max_runtime_seconds"),
            max_iterations=value.get("max_iterations"),
            max_retries=int(value.get("max_retries", 2)),
            max_subagents=value.get("max_subagents"),
            environment=dict(value.get("environment") or {}),
            execution=execution,
            metadata=metadata,
        ))

    def get(self, job_id: str) -> Job:
        return self.store.get_job(job_id)

    def list(self, *, statuses: Optional[Iterable[JobStatus | str]] = None, limit: int = 200) -> List[Job]:
        return self.store.list_jobs(statuses=statuses, limit=limit)

    def list_unarchived(self, *, limit: int = 200) -> List[Job]:
        """Jobs not archived (round-49 F9/F10): SQL-side filter so the board
        never materializes the full archived-id set per render."""
        return self.store.list_unarchived_jobs(limit=limit)

    def events(self, job_id: str, *, after_id: int = 0, limit: int = 500) -> List[JobEvent]:
        self.get(job_id)
        return self.store.list_events(job_id, after_id=after_id, limit=limit)

    def attempts(self, job_id: str) -> List[JobAttempt]:
        self.get(job_id)
        return self.store.list_attempts(job_id)

    def transition(
        self,
        job_id: str,
        target: JobStatus | str,
        *,
        reason: str = "",
        payload: Optional[Dict[str, Any]] = None,
        attention_reason: AttentionReason | str = AttentionReason.NONE,
        attention_detail: str = "",
        expected_version: Optional[int] = None,
    ) -> Job:
        current = self.get(job_id)
        target_status = coerce_status(target)
        if current.status == target_status:
            return current
        if not can_transition(current.status, target_status):
            raise JobStateError(
                f"cannot transition job {job_id} from {current.status.value} to {target_status.value}"
            )
        if target_status == JobStatus.NEEDS_HUMAN:
            attention = attention_reason if isinstance(attention_reason, AttentionReason) else AttentionReason(str(attention_reason or ""))
            if attention == AttentionReason.NONE:
                raise JobStateError("needs_human requires an attention_reason")
        else:
            attention_reason = AttentionReason.NONE
            attention_detail = ""
        return self.store.transition(
            job_id,
            target_status,
            reason=reason,
            payload=payload,
            attention_reason=attention_reason,
            attention_detail=attention_detail,
            expected_version=expected_version,
        )

    def cancel(self, job_id: str, *, reason: str = "cancelled by user") -> Job:
        current = self.get(job_id)
        if current.status == JobStatus.CANCELLED:
            return current
        return self.transition(job_id, JobStatus.CANCELLED, reason=reason)

    def require_human(
        self,
        job_id: str,
        reason: AttentionReason | str,
        detail: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Job:
        return self.transition(
            job_id,
            JobStatus.NEEDS_HUMAN,
            reason="human attention required",
            payload=payload,
            attention_reason=reason,
            attention_detail=detail,
        )

    def resume(self, job_id: str, *, detail: str = "") -> Job:
        current = self.get(job_id)
        if current.status != JobStatus.NEEDS_HUMAN:
            raise JobStateError("only a needs_human job can be resumed")
        self.store.append_event(job_id, "human_response", payload={"detail": str(detail or "")})
        return self.transition(job_id, JobStatus.QUEUED, reason="human response received; requeued")

    def retry(self, job_id: str, *, reason: str = "retry requested") -> Job:
        current = self.get(job_id)
        if current.status not in {
            JobStatus.FAILED, JobStatus.TIMED_OUT, JobStatus.BUDGET_EXCEEDED,
            JobStatus.ENVIRONMENT_ERROR, JobStatus.NEEDS_HUMAN,
        }:
            raise JobStateError(f"job {job_id} is not retryable from {current.status.value}")
        return self.transition(job_id, JobStatus.QUEUED, reason=reason)

    def acquire(self, job_id: str, worker_id: str, *, ttl_seconds: int = 60) -> bool:
        if not worker_id:
            raise ValueError("worker_id is required")
        return self.store.acquire_lease(job_id, worker_id, ttl_seconds=ttl_seconds)

    def heartbeat(self, job_id: str, worker_id: str, *, ttl_seconds: int = 60) -> bool:
        return self.store.heartbeat(job_id, worker_id, ttl_seconds=ttl_seconds)

    def release(self, job_id: str, worker_id: str, *, reason: str = "") -> bool:
        return self.store.release_lease(job_id, worker_id, reason=reason)

    def start_attempt(self, job_id: str, *, worker_id: str = "", session_name: str = "", metadata: Optional[Dict[str, Any]] = None) -> JobAttempt:
        return self.store.start_attempt(job_id, worker_id=worker_id, session_name=session_name, metadata=metadata)

    def finish_attempt(self, attempt_id: str, *, status: str, error: str = "", cost_usd: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> JobAttempt:
        return self.store.finish_attempt(attempt_id, status=status, error=error, cost_usd=cost_usd, metadata=metadata)

    def _resume_after_finished_attempt(
        self, job: Job, attempt: JobAttempt, *, expired_worker: str
    ) -> Job:
        """Round-16 F20: idempotent replay of the post-finish steps the
        worker would have taken for an attempt that committed as
        completed/needs_human before the crash. Safe to call repeatedly —
        the store transitions validate against the live row."""
        metadata = attempt.metadata or {}
        checkpoint = metadata.get("checkpoint")
        if attempt.status == "completed":
            return self.transition(
                job.id,
                JobStatus.VERIFYING,
                reason="implementation attempt completed (recovered after crash)",
                payload={"checkpoint": checkpoint, "attempt_id": attempt.id},
            )
        attention = metadata.get("attention") or {}
        raw_reason = str(metadata.get("attention_reason") or "")
        try:
            reason = AttentionReason(raw_reason)
        except ValueError:
            reason = AttentionReason.NONE
        if reason in (AttentionReason.NONE, ""):
            # Older attempts (pre-round-16) never persisted the reason;
            # NEEDS_HUMAN requires one, so fall back to a generic value
            # instead of dropping the job into RECOVERING (re-run).
            reason = AttentionReason.QUESTION
        return self.require_human(
            job.id,
            reason,
            str(metadata.get("attention_detail") or
                "recovered from crash after needs-human attempt finished"),
            payload={**attention, "checkpoint": checkpoint},
        )

    def recover_expired_leases(self) -> List[Job]:
        recovered: List[Job] = []
        for job in self.store.expired_leases():
            worker = job.worker_id
            # Round-36 F1: CLAIM the expired lease BEFORE acting on the
            # stale snapshot. release_expired_lease is a CAS — it succeeds
            # only when the lease is STILL this worker's AND still expired
            # at write time. Acting first (transitioning to RECOVERING,
            # replaying a finished attempt) and releasing afterwards let a
            # worker that renewed between the snapshot and the release
            # keep running while its job was yanked to RECOVERING — two
            # owners, duplicated side effects. Now a renewed lease means
            # this recovery pass skips the job entirely.
            if not self.store.release_expired_lease(
                job.id, worker, reason="expired lease claimed for recovery"
            ):
                continue
            if job.status in {JobStatus.PREPARING, JobStatus.RUNNING}:
                # Round-16 F20: the worker commits the attempt result FIRST
                # and the job transition SECOND — a crash between the two
                # leaves a finished attempt attached to a RUNNING job. The
                # old unconditional recovery sent that job to RECOVERING and
                # re-ran work that had already completed. If the newest
                # attempt for this job is terminal, finish what the worker
                # started instead (idempotent replay of the post-finish
                # steps) and only recover when the attempt truly died
                # mid-run.
                attempts = self.store.list_attempts(job.id)
                newest = max(
                    attempts,
                    key=lambda a: (a.number, a.started_at or 0.0),
                    default=None,
                )
                if newest is not None and newest.status in {
                    "completed", "needs_human",
                }:
                    try:
                        recovered.append(
                            self._resume_after_finished_attempt(
                                job, newest, expired_worker=worker
                            )
                        )
                    except Exception:
                        logger.exception(
                            "Idempotent recovery failed for job %s "
                            "(attempt %s, status %s); falling back to "
                            "RECOVERING",
                            job.id, newest.id, newest.status,
                        )
                        recovered.append(self.transition(
                            job.id,
                            JobStatus.RECOVERING,
                            reason="implementation worker lease expired",
                            payload={"worker_id": worker},
                        ))
                    continue
                try:
                    recovered.append(self.transition(
                        job.id,
                        JobStatus.RECOVERING,
                        reason="implementation worker lease expired",
                        payload={"worker_id": worker},
                    ))
                except JobStateError:
                    pass
            elif job.status == JobStatus.VERIFYING:
                self.store.append_event(
                    job.id,
                    "verification_lease_expired",
                    reason="verification worker lease expired; verifier will retry",
                    payload={"worker_id": worker},
                )
                recovered.append(self.get(job.id))
        return recovered


_default_lock = threading.Lock()
_default_service: Optional[JobService] = None


def get_default_job_service() -> JobService:
    global _default_service
    with _default_lock:
        if _default_service is None:
            _default_service = JobService()
        return _default_service
