"""Shared job-board projection used by GUI, TUI and mobile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import Job, JobStatus
from .service import JobService


BOARD_ORDER = ("needs_you", "running", "queued", "ready", "failed", "done")

RUNNING_STATUSES = {
    JobStatus.PREPARING,
    JobStatus.RUNNING,
    JobStatus.VERIFYING,
    JobStatus.RECOVERING,
}
FAILED_STATUSES = {
    JobStatus.FAILED,
    JobStatus.TIMED_OUT,
    JobStatus.BUDGET_EXCEEDED,
    JobStatus.ENVIRONMENT_ERROR,
}
DONE_STATUSES = {JobStatus.MERGED, JobStatus.CANCELLED}


@dataclass(frozen=True)
class JobBoard:
    needs_you: List[Job]
    running: List[Job]
    queued: List[Job]
    ready: List[Job]
    failed: List[Job]
    done: List[Job]

    @property
    def counts(self) -> Dict[str, int]:
        return {name: len(getattr(self, name)) for name in BOARD_ORDER}

    def to_dict(self) -> dict:
        return {
            "counts": self.counts,
            "sections": {
                name: [job.to_dict() for job in getattr(self, name)]
                for name in BOARD_ORDER
            },
        }


def bucket_for(job: Job) -> str:
    if job.status == JobStatus.NEEDS_HUMAN or job.status == JobStatus.CONFLICTED:
        return "needs_you"
    if job.status in RUNNING_STATUSES:
        return "running"
    if job.status == JobStatus.QUEUED:
        return "queued"
    if job.status == JobStatus.READY_FOR_REVIEW:
        return "ready"
    if job.status in FAILED_STATUSES:
        return "failed"
    if job.status in DONE_STATUSES:
        return "done"
    # Unknown future non-terminal states should demand attention rather than
    # disappear from every control plane.
    return "needs_you"


def build_job_board(service: JobService, *, limit: int = 1000) -> JobBoard:
    # Archive is a management concern rather than a runtime state. Keep the
    # operational board quiet while retaining archived jobs in durable history.
    # Round-49 F9/F10: the board previously loaded ALL archived ids (a set
    # that grows forever) then materialized up to 1000 jobs and filtered in
    # Python — O(archived + 1000) work per GUI poll, and the silent 1000-job
    # cap could hide older ACTIVE jobs while section counts described only
    # the truncated window. The cap now applies per section (each section
    # keeps its newest `limit` entries — active work never disappears), and
    # archived filtering runs in the service query (NOT EXISTS) instead of
    # materializing the archived-id set in Python.
    sections = {name: [] for name in BOARD_ORDER}
    for job in service.list_unarchived(limit=limit):
        sections[bucket_for(job)].append(job)
    return JobBoard(**sections)
