"""Durable engineering-job domain model and explicit lifecycle state machine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    NEEDS_HUMAN = "needs_human"
    VERIFYING = "verifying"
    READY_FOR_REVIEW = "ready_for_review"
    RECOVERING = "recovering"
    CONFLICTED = "conflicted"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BUDGET_EXCEEDED = "budget_exceeded"
    ENVIRONMENT_ERROR = "environment_error"
    CANCELLED = "cancelled"
    MERGED = "merged"


class AttentionReason(str, Enum):
    NONE = ""
    QUESTION = "question"
    APPROVAL_REQUIRED = "approval_required"
    AMBIGUOUS_REQUIREMENT = "ambiguous_requirement"
    SECRET_REQUIRED = "secret_required"
    MERGE_CONFLICT = "merge_conflict"
    BUDGET_EXCEEDED = "budget_exceeded"
    ENVIRONMENT_FAILURE = "environment_failure"
    VERIFICATION_REQUIRED = "verification_required"
    TEST_FAILURE = "test_failure"
    UNSAFE_ACTION = "unsafe_action"
    WORKER_LOST = "worker_lost"
    PROVIDER_ERROR = "provider_error"


TERMINAL_STATUSES = frozenset({JobStatus.CANCELLED, JobStatus.MERGED})

ALLOWED_TRANSITIONS = {
    JobStatus.QUEUED: {JobStatus.PREPARING, JobStatus.CANCELLED},
    # Round-38 F3: every worker-active state can hit the whole-job runtime
    # deadline — a hung provider call is equally possible while preparing
    # the worktree or verifying as while the agent runs.
    JobStatus.PREPARING: {
        JobStatus.RUNNING, JobStatus.NEEDS_HUMAN, JobStatus.RECOVERING,
        JobStatus.ENVIRONMENT_ERROR, JobStatus.FAILED, JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
    },
    JobStatus.RUNNING: {
        JobStatus.NEEDS_HUMAN, JobStatus.VERIFYING, JobStatus.RECOVERING,
        JobStatus.FAILED, JobStatus.TIMED_OUT, JobStatus.BUDGET_EXCEEDED,
        JobStatus.ENVIRONMENT_ERROR, JobStatus.CANCELLED,
    },
    JobStatus.NEEDS_HUMAN: {
        JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.FAILED, JobStatus.CANCELLED,
    },
    JobStatus.VERIFYING: {
        JobStatus.READY_FOR_REVIEW, JobStatus.QUEUED, JobStatus.RUNNING,
        JobStatus.NEEDS_HUMAN, JobStatus.RECOVERING, JobStatus.CONFLICTED,
        JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT,
    },
    # Reviewer feedback is a first-class loop. READY_FOR_REVIEW keeps the same
    # durable job + branch, but its execution worktree has already been retired.
    # Requeueing creates a fresh temporary worktree on that branch if more agent
    # work is required.
    JobStatus.READY_FOR_REVIEW: {
        JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CONFLICTED,
        JobStatus.MERGED, JobStatus.CANCELLED,
    },
    JobStatus.RECOVERING: {
        JobStatus.RUNNING, JobStatus.NEEDS_HUMAN, JobStatus.FAILED, JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
    },
    JobStatus.CONFLICTED: {
        JobStatus.RUNNING, JobStatus.NEEDS_HUMAN, JobStatus.FAILED, JobStatus.CANCELLED,
    },
    JobStatus.FAILED: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.TIMED_OUT: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.BUDGET_EXCEEDED: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.ENVIRONMENT_ERROR: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.CANCELLED: set(),
    JobStatus.MERGED: set(),
}


def coerce_status(value: JobStatus | str) -> JobStatus:
    return value if isinstance(value, JobStatus) else JobStatus(str(value))


def can_transition(current: JobStatus | str, target: JobStatus | str) -> bool:
    return coerce_status(target) in ALLOWED_TRANSITIONS[coerce_status(current)]


def normalize_execution(value: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normalize the reproducible agent execution policy stored with a job."""
    raw = dict(value or {})
    execution = {
        "provider": str(raw.get("provider") or "").strip(),
        "model": str(raw.get("model") or "").strip(),
        "agent_mode": str(raw.get("agent_mode") or "default").strip() or "default",
        "session_type": str(raw.get("session_type") or "workspace").strip().lower() or "workspace",
        "auto_approve_writes": bool(raw.get("auto_approve_writes", False)),
    }
    if execution["session_type"] not in {"chat", "workspace", "container"}:
        raise ValueError("execution.session_type must be chat, workspace, or container")
    for key, item in raw.items():
        if key not in execution:
            execution[str(key)] = item
    return execution


@dataclass(frozen=True)
class JobSpec:
    title: str
    description: str = ""
    repository: str = ""
    base_branch: str = "main"
    base_sha: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    validation_commands: List[str] = field(default_factory=list)
    max_cost_usd: Optional[float] = None
    max_runtime_seconds: Optional[int] = None
    max_iterations: Optional[int] = None
    max_retries: int = 2
    max_subagents: Optional[int] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "JobSpec":
        title = str(self.title or "").strip()
        if not title:
            raise ValueError("job title is required")
        # Round-35 F8: finiteness + practical upper bounds. NaN passes
        # `<= 0` comparisons and later defeats every budget check; absurd
        # integers create effectively unbounded runs / resource fan-out.
        max_cost_usd = (
            float(self.max_cost_usd) if self.max_cost_usd is not None else None
        )
        max_runtime_seconds = (
            int(self.max_runtime_seconds)
            if self.max_runtime_seconds is not None
            else None
        )
        max_iterations = (
            int(self.max_iterations) if self.max_iterations is not None else None
        )
        max_retries = int(self.max_retries)
        max_subagents = (
            int(self.max_subagents) if self.max_subagents is not None else None
        )
        if max_cost_usd is not None and not math.isfinite(max_cost_usd):
            raise ValueError("max_cost_usd must be finite")
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        if max_cost_usd is not None and max_cost_usd > 1_000_000.0:
            raise ValueError("max_cost_usd exceeds the supported maximum (1,000,000)")
        if max_runtime_seconds is not None:
            if max_runtime_seconds <= 0:
                raise ValueError("max_runtime_seconds must be positive")
            if max_runtime_seconds > 7 * 24 * 3600:
                raise ValueError("max_runtime_seconds exceeds the supported maximum (7 days)")
        if max_iterations is not None:
            if max_iterations <= 0:
                raise ValueError("max_iterations must be positive")
            if max_iterations > 100_000:
                raise ValueError("max_iterations exceeds the supported maximum (100,000)")
        if not (0 <= max_retries <= 100):
            raise ValueError("max_retries must be between 0 and 100")
        if max_subagents is not None:
            if max_subagents < 0:
                raise ValueError("max_subagents cannot be negative")
            if max_subagents > 10_000:
                raise ValueError("max_subagents exceeds the supported maximum (10,000)")
        return JobSpec(
            title=title,
            description=str(self.description or "").strip(),
            repository=str(self.repository or "").strip(),
            base_branch=str(self.base_branch or "main").strip() or "main",
            base_sha=str(self.base_sha or "").strip(),
            acceptance_criteria=[str(v).strip() for v in self.acceptance_criteria if str(v).strip()],
            validation_commands=[str(v).strip() for v in self.validation_commands if str(v).strip()],
            max_cost_usd=float(self.max_cost_usd) if self.max_cost_usd is not None else None,
            max_runtime_seconds=int(self.max_runtime_seconds) if self.max_runtime_seconds is not None else None,
            max_iterations=int(self.max_iterations) if self.max_iterations is not None else None,
            max_retries=int(self.max_retries),
            max_subagents=int(self.max_subagents) if self.max_subagents is not None else None,
            environment=dict(self.environment or {}),
            execution=normalize_execution(self.execution),
            metadata=dict(self.metadata or {}),
        )


@dataclass
class Job:
    id: str
    title: str
    description: str
    repository: str
    base_branch: str
    base_sha: str
    acceptance_criteria: List[str]
    validation_commands: List[str]
    status: JobStatus
    attention_reason: AttentionReason = AttentionReason.NONE
    attention_detail: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    max_cost_usd: Optional[float] = None
    max_runtime_seconds: Optional[int] = None
    max_iterations: Optional[int] = None
    max_retries: int = 2
    max_subagents: Optional[int] = None
    cost_usd: float = 0.0
    branch: str = ""
    worktree: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_name: str = ""
    worker_id: str = ""
    lease_expires_at: Optional[float] = None
    heartbeat_at: Optional[float] = None
    version: int = 1

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def needs_attention(self) -> bool:
        return self.status == JobStatus.NEEDS_HUMAN or self.attention_reason != AttentionReason.NONE

    @property
    def review_branch(self) -> str:
        """Normal Git branch that is authoritative once execution is finished."""
        value = str((self.metadata or {}).get("review_branch") or "").strip()
        if value:
            return value
        if self.status == JobStatus.READY_FOR_REVIEW:
            return str(self.branch or "").strip()
        return ""

    @property
    def review_head_sha(self) -> str:
        return str((self.metadata or {}).get("review_head_sha") or "").strip()

    @property
    def review_artifact(self) -> str:
        """Expose review semantics without making callers infer worktree state."""
        if self.review_branch:
            return "branch"
        if self.worktree:
            return "worktree"
        return "none"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["attention_reason"] = self.attention_reason.value
        value["terminal"] = self.terminal
        value["needs_attention"] = self.needs_attention
        value["review_artifact"] = self.review_artifact
        value["review_branch"] = self.review_branch
        value["review_head_sha"] = self.review_head_sha
        return value


@dataclass
class JobEvent:
    id: int
    job_id: str
    event_type: str
    from_status: Optional[JobStatus]
    to_status: Optional[JobStatus]
    reason: str
    payload: Dict[str, Any]
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["from_status"] = self.from_status.value if self.from_status else None
        value["to_status"] = self.to_status.value if self.to_status else None
        return value


@dataclass
class JobAttempt:
    id: str
    job_id: str
    number: int
    status: str
    session_name: str = ""
    worker_id: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    error: str = ""
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def statuses(values: Iterable[JobStatus | str]) -> List[str]:
    return [coerce_status(value).value for value in values]
