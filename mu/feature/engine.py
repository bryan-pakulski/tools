from __future__ import annotations

import json
import os
import re
import time
from utils.config import HISTORY_DIR
from dataclasses import asdict, dataclass, field
from typing import Any

STATUS_PENDING = "pending"
STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_BLOCKED = "blocked"
STATUS_COMPLETED = "completed"
STATUS_ARCHIVED = "archived"
STATUS_REVIEW_PENDING = "pending"


VALID_TASK_STATUSES = {
    STATUS_PENDING,
    STATUS_NOT_STARTED,
    STATUS_IN_PROGRESS,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_ARCHIVED,
}

ALLOWED_TASK_TRANSITIONS = {
    STATUS_PENDING: {STATUS_IN_PROGRESS, STATUS_BLOCKED, STATUS_COMPLETED},
    STATUS_NOT_STARTED: {STATUS_IN_PROGRESS, STATUS_BLOCKED, STATUS_COMPLETED},
    STATUS_IN_PROGRESS: {STATUS_BLOCKED, STATUS_COMPLETED, STATUS_NOT_STARTED},
    STATUS_BLOCKED: {STATUS_IN_PROGRESS, STATUS_NOT_STARTED},
    STATUS_COMPLETED: {STATUS_ARCHIVED, STATUS_IN_PROGRESS, STATUS_NOT_STARTED},
    STATUS_ARCHIVED: set(),
}


def normalize_task_status(status: str | None) -> str:
    raw = str(status or "").strip().lower()
    if raw in {"todo", "queued"}:
        return STATUS_NOT_STARTED
    if raw in VALID_TASK_STATUSES:
        return raw
    return STATUS_NOT_STARTED


@dataclass
class FeatureTask:
    id: int
    title: str
    phase_id: int | None = None
    objectives: list[str] = field(default_factory=list)
    action_points: list[str] = field(default_factory=list)
    exit_criteria: list[str] = field(default_factory=list)
    verified_exit_criteria: list[str] = field(default_factory=list)
    status: str = STATUS_NOT_STARTED
    notes: str = ""
    blocked_reason: str = ""

    @property
    def number(self) -> int:
        return self.id


@dataclass
class FeaturePhase:
    id: int
    title: str
    goal: str = ""
    order: int = 0
    status: str = STATUS_PENDING
    task_ids: list[int] = field(default_factory=list)


@dataclass
class FeatureEvent:
    id: str
    kind: str
    entity: str
    entity_id: int | str
    payload: dict[str, Any] = field(default_factory=dict)
    actor: str = "system"
    created_at: float = field(default_factory=time.time)


@dataclass
class DiffProposal:
    id: str
    review_id: str
    task_id: int
    issue_id: str
    diff: str
    status: str = "pending"
    decision_reason: str = ""
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    # Set when the approved diff is actually applied via review-mode
    # apply_diff; a proposal may only be applied once.
    applied_at: float | None = None
    applied_file: str = ""


@dataclass
class TaskReviewRecord:
    id: str
    task_id: int
    summary: str
    limitations: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class FeaturePlan:
    feature_id: str
    feature_name: str
    feature_request: str
    directory: str  # Workspace root
    metadata_path: str = ""
    approved: bool = False
    review_status: str = STATUS_REVIEW_PENDING
    review_notes: str = ""
    tasks: list[FeatureTask] = field(default_factory=list)
    phases_meta: list[FeaturePhase] = field(default_factory=list)
    event_log: list[FeatureEvent] = field(default_factory=list)
    review_records: list[TaskReviewRecord] = field(default_factory=list)
    diff_proposals: list[DiffProposal] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def next_incomplete_task(self) -> FeatureTask | None:
        for task in self.tasks:
            if normalize_task_status(task.status) not in {STATUS_COMPLETED, STATUS_ARCHIVED}:
                return task
        return None

    def next_incomplete_phase(self) -> FeatureTask | None:
        # Alias for backward compatibility
        next_task = self.next_incomplete_task()
        if not next_task:
            return None
        return next_task

    def tasks_completed(self) -> bool:
        return bool(self.tasks) and all(
            normalize_task_status(task.status) in {STATUS_COMPLETED, STATUS_ARCHIVED}
            for task in self.tasks
        )

    def phases_completed(self) -> bool:
        # Alias for backward compatibility
        return self.tasks_completed()

    def overall_status(self) -> str:
        if self.review_status == STATUS_COMPLETED:
            return STATUS_COMPLETED
        if self.tasks_completed():
            return STATUS_IN_PROGRESS
        if any(
            normalize_task_status(task.status)
            in {STATUS_IN_PROGRESS, STATUS_BLOCKED, STATUS_COMPLETED, STATUS_ARCHIVED}
            for task in self.tasks
        ):
            return STATUS_IN_PROGRESS
        return STATUS_NOT_STARTED

    def add_event(
        self,
        *,
        kind: str,
        entity: str,
        entity_id: int | str,
        payload: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> None:
        event_id = f"{int(time.time() * 1000)}-{len(self.event_log) + 1}"
        self.event_log.append(
            FeatureEvent(
                id=event_id,
                kind=kind,
                entity=entity,
                entity_id=entity_id,
                payload=payload or {},
                actor=actor,
            )
        )


def create_feature_shell(
    *,
    feature_name: str,
    feature_request: str,
    folder_context=None,
    session_id: str | None = None,
    feature_id: str | None = None,
    metadata_path: str | None = None,
) -> FeaturePlan:
    workspace_root = _workspace_root(folder_context)
    slug = _slugify(feature_id or feature_name)
    directory = _feature_directory(workspace_root, slug)
    plan = FeaturePlan(
        feature_id=slug,
        feature_name=feature_name.strip() or slug,
        feature_request=feature_request.strip() or feature_name.strip() or slug,
        directory=directory,
        metadata_path=_resolve_metadata_path(
            directory=directory,
            metadata_path=metadata_path,
            session_id=session_id,
            feature_id=slug,
        ),
        tasks=[],
        phases_meta=[],
    )
    plan.add_event(
        kind="feature_created",
        entity="feature",
        entity_id=plan.feature_id,
        payload={"feature_name": plan.feature_name},
    )
    return save_feature_plan(session_id or "", plan)


def is_valid_task_transition(from_status: str, to_status: str) -> bool:
    from_normalized = normalize_task_status(from_status)
    to_normalized = normalize_task_status(to_status)
    return to_normalized in ALLOWED_TASK_TRANSITIONS.get(from_normalized, set())


def transition_task_status(
    plan: FeaturePlan,
    *,
    task_id: int,
    to_status: str,
    notes: str | None = None,
    blocked_reason: str | None = None,
    verified_exit_criteria: list[str] | None = None,
    actor: str = "system",
) -> FeatureTask:
    target_status = normalize_task_status(to_status)
    for task in plan.tasks:
        if task.id != task_id:
            continue
        current_status = normalize_task_status(task.status)
        if not is_valid_task_transition(current_status, target_status):
            raise ValueError(
                f"Invalid task status transition: {current_status} -> {target_status}"
            )
        task.status = target_status
        if notes is not None:
            task.notes = notes
        if blocked_reason is not None:
            task.blocked_reason = blocked_reason
        if verified_exit_criteria is not None:
            task.verified_exit_criteria = [
                str(item).strip()
                for item in verified_exit_criteria
                if str(item).strip()
            ]
        plan.add_event(
            kind="status_transition",
            entity="task",
            entity_id=task.id,
            payload={
                "from_status": current_status,
                "to_status": target_status,
                "blocked_reason": blocked_reason or "",
            },
            actor=actor,
        )
        recalculate_phase_statuses(plan)
        return task
    raise ValueError(f"Task {task_id} not found")


def _tasks_for_phase(plan: FeaturePlan, phase_id: int) -> list[FeatureTask]:
    return [task for task in plan.tasks if task.phase_id == phase_id]


def recalculate_phase_statuses(plan: FeaturePlan) -> None:
    for phase in plan.phases_meta:
        phase_tasks = _tasks_for_phase(plan, phase.id)
        if not phase_tasks:
            phase.status = STATUS_PENDING
            continue
        normalized = [normalize_task_status(task.status) for task in phase_tasks]
        if all(status in {STATUS_COMPLETED, STATUS_ARCHIVED} for status in normalized):
            phase.status = STATUS_COMPLETED
        elif any(status == STATUS_BLOCKED for status in normalized):
            phase.status = STATUS_BLOCKED
        elif any(status == STATUS_IN_PROGRESS for status in normalized):
            phase.status = STATUS_IN_PROGRESS
        else:
            phase.status = STATUS_PENDING


def next_pending_phase(plan: FeaturePlan) -> FeaturePhase | None:
    recalculate_phase_statuses(plan)
    ordered = sorted(plan.phases_meta, key=lambda phase: phase.order or phase.id)
    for phase in ordered:
        if normalize_task_status(phase.status) not in {STATUS_COMPLETED, STATUS_ARCHIVED}:
            return phase
    return None


def next_actionable_task(plan: FeaturePlan, phase_id: int | None = None) -> FeatureTask | None:
    if phase_id is None:
        phase = next_pending_phase(plan)
        if not phase:
            return None
        phase_id = phase.id
    phase_tasks = sorted(
        _tasks_for_phase(plan, phase_id),
        key=lambda task: task.id,
    )
    for task in phase_tasks:
        if normalize_task_status(task.status) == STATUS_IN_PROGRESS:
            return task
    for task in phase_tasks:
        if normalize_task_status(task.status) in {STATUS_PENDING, STATUS_NOT_STARTED}:
            return task
    return None


def feature_execution_snapshot(plan: FeaturePlan) -> dict[str, Any]:
    recalculate_phase_statuses(plan)
    phase = next_pending_phase(plan)
    task = next_actionable_task(plan, phase.id) if phase else None
    blocked_tasks = [
        asdict(item)
        for item in plan.tasks
        if normalize_task_status(item.status) == STATUS_BLOCKED
    ]
    return {
        "feature_id": plan.feature_id,
        "next_phase": None if phase is None else asdict(phase),
        "next_task": None if task is None else asdict(task),
        "blocked_tasks": blocked_tasks,
        "all_phases_completed": phase is None and bool(plan.phases_meta),
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "feature"


def _workspace_root(folder_context) -> str:
    if (
        folder_context
        and getattr(folder_context, "folders", None)
        and folder_context.folders
    ):
        return os.path.abspath(folder_context.folders[0])
    return os.getcwd()


def save_feature_plan(session_id: str, plan: FeaturePlan) -> FeaturePlan:
    full_path = _resolve_metadata_path(
        directory=plan.directory,
        metadata_path=plan.metadata_path,
        session_id=session_id,
        feature_id=plan.feature_id,
    )
    plan.metadata_path = full_path
    plan.updated_at = time.time()
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(plan), handle, indent=2)
    return plan


def _feature_directory(workspace_root: str, feature_id: str) -> str:
    return workspace_root


def _resolve_metadata_path(
    *,
    directory: str | None,
    metadata_path: str | None,
    session_id: str | None = None,
    feature_id: str | None = None,
) -> str:
    if metadata_path:
        if os.path.isabs(metadata_path):
            return metadata_path
        if session_id:
            return os.path.join(HISTORY_DIR, "sessions", session_id, "features", metadata_path)
        if directory:
            return os.path.join(directory, metadata_path)
        return os.path.abspath(metadata_path)
    if session_id:
        return os.path.join(
            HISTORY_DIR,
            "sessions",
            session_id,
            "features",
            f"{_slugify(feature_id or 'feature')}.json",
        )
    return os.path.join(str(directory or os.getcwd()), "feature_plan.json")


def _task_counts_from_status(status: str) -> dict[str, int]:
    normalized = normalize_task_status(status)
    if normalized == STATUS_COMPLETED:
        return {"not_started": 0, "in_progress": 0, "completed": 1}
    if normalized == STATUS_IN_PROGRESS:
        return {"not_started": 0, "in_progress": 1, "completed": 0}
    if normalized == STATUS_BLOCKED:
        return {"not_started": 0, "in_progress": 1, "completed": 0}
    return {"not_started": 1, "in_progress": 0, "completed": 0}


def create_feature_plan(
    feature_name: str,
    feature_request: str,
    tasks_data: list[dict[str, Any]] | None = None,
    phases: list[dict[str, Any]] | None = None,
    folder_context=None,
    session_id: str | None = None,
    feature_id: str | None = None,
    metadata_path: str | None = None,
) -> FeaturePlan:
    workspace_root = _workspace_root(folder_context)
    slug = _slugify(feature_id or feature_name)
    directory = _feature_directory(workspace_root, slug)
    items = tasks_data if tasks_data is not None else phases or []

    tasks = []
    for idx, t in enumerate(items, start=1):
        if not isinstance(t, dict):
            raise ValueError(
                "Invalid tasks_data entry at index "
                f"{idx}: expected object, got {type(t).__name__} ({t!r})"
            )
        tasks.append(
            FeatureTask(
                id=idx,
                title=str(t.get("title") or f"Task {idx}").strip(),
                objectives=[str(o).strip() for o in t.get("objectives", [])],
                action_points=[str(a).strip() for a in t.get("action_points", [])],
                exit_criteria=[str(e).strip() for e in t.get("exit_criteria", [])],
                status=normalize_task_status(t.get("status")),
                notes=str(t.get("notes", "") or ""),
            )
        )

    phases_meta = [
        FeaturePhase(
            id=task.id,
            title=task.title,
            goal=task.objectives[0] if task.objectives else task.title,
            order=index,
            status=normalize_task_status(task.status),
            task_ids=[task.id],
        )
        for index, task in enumerate(tasks, start=1)
    ]

    plan = FeaturePlan(
        feature_id=slug,
        feature_name=feature_name.strip() or slug,
        feature_request=feature_request.strip() or feature_name.strip() or slug,
        directory=directory,
        metadata_path=_resolve_metadata_path(
            directory=directory,
            metadata_path=metadata_path,
            session_id=session_id,
            feature_id=slug,
        ),
        tasks=tasks,
        phases_meta=phases_meta,
    )
    return save_feature_plan(session_id or "", plan)


def create_feature_phases(
    metadata_path: str,
    phases_data: list[dict[str, Any]],
    *,
    replace_existing: bool = True,
    actor: str = "system",
) -> FeaturePlan:
    plan = load_feature_plan(metadata_path)
    if replace_existing:
        plan.phases_meta = []
        for task in plan.tasks:
            task.phase_id = None
    for raw in phases_data:
        phase_id = int(raw.get("id") or (len(plan.phases_meta) + 1))
        phase = FeaturePhase(
            id=phase_id,
            title=str(raw.get("title") or f"Phase {phase_id}").strip(),
            goal=str(raw.get("goal") or "").strip(),
            order=int(raw.get("order") or phase_id),
            status=normalize_task_status(raw.get("status")),
            task_ids=[int(tid) for tid in raw.get("task_ids", [])],
        )
        plan.phases_meta.append(phase)
        plan.add_event(
            kind="phase_created",
            entity="phase",
            entity_id=phase.id,
            payload={"title": phase.title, "goal": phase.goal},
            actor=actor,
        )
    plan.phases_meta = sorted(plan.phases_meta, key=lambda p: p.order)
    recalculate_phase_statuses(plan)
    return save_feature_plan("", plan)


def create_feature_task(
    metadata_path: str,
    task_data: dict[str, Any],
    *,
    actor: str = "system",
) -> tuple[FeaturePlan, FeatureTask]:
    plan = load_feature_plan(metadata_path)
    next_id = max((task.id for task in plan.tasks), default=0) + 1
    phase_id = task_data.get("phase_id")
    if phase_id is not None:
        phase_id = int(phase_id)
    task = FeatureTask(
        id=next_id,
        phase_id=phase_id,
        title=str(task_data.get("title") or f"Task {next_id}").strip(),
        objectives=[str(o).strip() for o in task_data.get("objectives", [])],
        action_points=[str(a).strip() for a in task_data.get("action_points", [])],
        exit_criteria=[str(e).strip() for e in task_data.get("exit_criteria", [])],
        status=normalize_task_status(task_data.get("status")),
        notes=str(task_data.get("notes", "") or ""),
    )
    plan.tasks.append(task)
    if phase_id is not None:
        for phase in plan.phases_meta:
            if phase.id == phase_id:
                if task.id not in phase.task_ids:
                    phase.task_ids.append(task.id)
                break
    plan.add_event(
        kind="task_created",
        entity="task",
        entity_id=task.id,
        payload={"title": task.title, "phase_id": phase_id},
        actor=actor,
    )
    recalculate_phase_statuses(plan)
    return save_feature_plan("", plan), task


def load_feature_plan(path_or_session_id: str, metadata_id: str | None = None) -> FeaturePlan:
    if metadata_id is not None:
        full_path = _resolve_metadata_path(
            directory=None,
            metadata_path=metadata_id,
            session_id=path_or_session_id,
        )
    else:
        full_path = (
            os.path.join(path_or_session_id, "feature_plan.json")
            if os.path.isdir(path_or_session_id)
            else path_or_session_id
        )
    with open(full_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    data.pop("overall_status", None)
    data.pop("phases_completed", None)
    data.pop("phase_count", None)
    data.pop("phases", None)
    data.pop("next_phase", None)
    tasks_data = data.pop("tasks", [])
    tasks = [FeatureTask(**t) for t in tasks_data]

    phases_data = data.pop("phases_meta", [])
    phases_meta = [FeaturePhase(**p) for p in phases_data]
    if not phases_meta:
        phases_meta = [
            FeaturePhase(
                id=task.id,
                title=task.title,
                goal=task.objectives[0] if task.objectives else task.title,
                order=idx,
                status=normalize_task_status(task.status),
                task_ids=[task.id],
            )
            for idx, task in enumerate(tasks, start=1)
        ]

    events_data = data.pop("event_log", [])
    event_log = [FeatureEvent(**evt) for evt in events_data]
    reviews_data = data.pop("review_records", [])
    review_records = [TaskReviewRecord(**item) for item in reviews_data]
    proposals_data = data.pop("diff_proposals", [])
    diff_proposals = [DiffProposal(**item) for item in proposals_data]

    return FeaturePlan(
        tasks=tasks,
        phases_meta=phases_meta,
        event_log=event_log,
        review_records=review_records,
        diff_proposals=diff_proposals,
        **data,
    )


def refresh_and_persist_feature_plan(
    path_or_session_id: str,
    metadata_path: str | None = None,
) -> FeaturePlan:
    return load_feature_plan(path_or_session_id, metadata_path)


def summarize_feature_plan(plan: FeaturePlan) -> dict[str, Any]:
    recalculate_phase_statuses(plan)
    summary = asdict(plan)
    summary["overall_status"] = plan.overall_status()
    summary["tasks_completed"] = plan.tasks_completed()
    summary["task_count"] = len(plan.tasks)
    summary["phase_count"] = len(plan.phases_meta) if plan.phases_meta else len(plan.tasks)
    summary["event_count"] = len(plan.event_log)
    summary["review_count"] = len(plan.review_records)
    summary["diff_proposal_count"] = len(plan.diff_proposals)

    next_task = plan.next_incomplete_task()
    summary["next_task"] = asdict(next_task) if next_task else None

    phases = []
    for task in plan.tasks:
        phases.append(
            {
                **asdict(task),
                "number": task.id,
                "task_counts": _task_counts_from_status(task.status),
            }
        )
    summary["phases"] = phases
    summary["active_tasks"] = [
        item for item in phases if normalize_task_status(item.get("status")) != STATUS_ARCHIVED
    ]
    summary["review_summaries"] = [
        {
            "task_id": record.task_id,
            "review_id": record.id,
            "summary": record.summary,
            "issue_count": len(record.issues),
            "categories": sorted(
                {
                    str(issue.get("category", "")).strip().lower()
                    for issue in record.issues
                    if str(issue.get("category", "")).strip()
                }
            ),
        }
        for record in plan.review_records
    ]
    phase = next_pending_phase(plan)
    summary["next_phase"] = asdict(phase) if phase else None
    summary["phases_completed"] = summary["tasks_completed"]
    summary["workflow"] = {
        "task_status_model": [
            STATUS_PENDING,
            STATUS_IN_PROGRESS,
            STATUS_BLOCKED,
            STATUS_COMPLETED,
            STATUS_ARCHIVED,
        ]
    }
    summary["execution"] = feature_execution_snapshot(plan)

    return summary


def update_feature_plan_metadata(
    path_or_session_id: str | None = None,
    *,
    session_id: str | None = None,
    approved: bool | None = None,
    review_status: str | None = None,
    review_notes: str | None = None,
    metadata_path: str | None = None,
) -> FeaturePlan:
    locator = path_or_session_id or session_id
    if not locator:
        raise ValueError("session_id or directory is required")
    resolved_metadata = metadata_path or (
        os.path.join(locator, "feature_plan.json")
        if os.path.isdir(locator)
        else None
    )
    if not resolved_metadata:
        raise ValueError("metadata_path is required")
    plan = (
        load_feature_plan(locator, resolved_metadata)
        if session_id
        else load_feature_plan(resolved_metadata)
    )
    if approved is not None:
        plan.approved = approved
    if review_status is not None:
        plan.review_status = review_status
    if review_notes is not None:
        plan.review_notes = review_notes
    return save_feature_plan(session_id or "", plan)


def update_task_status(
    metadata_path: str,
    task_id: int,
    status: str,
    notes: str | None = None,
    verified_exit_criteria: list[str] | None = None,
) -> FeaturePlan:
    plan = load_feature_plan(metadata_path)
    transition_task_status(
        plan,
        task_id=task_id,
        to_status=status,
        notes=notes,
        verified_exit_criteria=verified_exit_criteria,
    )
    return save_feature_plan("", plan)


def update_task_content(
    metadata_path: str,
    task_id: int,
    title: str | None = None,
    objectives: list[str] | None = None,
    action_points: list[str] | None = None,
    exit_criteria: list[str] | None = None,
    notes: str | None = None,
) -> FeaturePlan:
    plan = load_feature_plan(metadata_path)
    for task in plan.tasks:
        if task.id == task_id:
            if title is not None:
                task.title = title
            if objectives is not None:
                task.objectives = objectives
            if action_points is not None:
                task.action_points = action_points
            if exit_criteria is not None:
                # Round-30 F1: an empty list would strip all exit
                # criteria, after which the completion gate's
                # `missing = [c for c in expected ...]` is vacuously
                # empty and the task can be marked completed with zero
                # evidence. The create path requires non-empty
                # criteria; the update path must uphold the same
                # invariant. Empty list = refuse.
                cleaned = [
                    str(item).strip()
                    for item in (exit_criteria or [])
                    if str(item).strip()
                ]
                if not cleaned:
                    raise ValueError(
                        "exit_criteria cannot be emptied — a task must "
                        "always carry at least one verifiable exit "
                        "criterion"
                    )
                task.exit_criteria = cleaned
            if notes is not None:
                task.notes = notes
            break
    return save_feature_plan("", plan)


def create_task_review_record(
    metadata_path: str,
    *,
    task_id: int,
    summary: str,
    limitations: list[str] | None = None,
    issues: list[dict[str, Any]] | None = None,
    actor: str = "agent",
) -> tuple[FeaturePlan, TaskReviewRecord]:
    plan = load_feature_plan(metadata_path)
    task = next((item for item in plan.tasks if item.id == task_id), None)
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    review = TaskReviewRecord(
        id=f"review-{int(time.time() * 1000)}-{task_id}",
        task_id=task_id,
        summary=summary.strip(),
        limitations=[str(item).strip() for item in (limitations or []) if str(item).strip()],
        issues=[
            {
                "id": str(issue.get("id") or f"issue-{idx}"),
                "title": str(issue.get("title") or "").strip(),
                "category": str(issue.get("category") or "risk").strip().lower(),
                "details": str(issue.get("details") or "").strip(),
            }
            for idx, issue in enumerate((issues or []), start=1)
        ],
    )
    plan.review_records.append(review)
    plan.add_event(
        kind="task_review_created",
        entity="task",
        entity_id=task_id,
        payload={"review_id": review.id, "issue_count": len(review.issues)},
        actor=actor,
    )
    return save_feature_plan("", plan), review


def review_all_completed_tasks(
    metadata_path: str,
    *,
    actor: str = "agent",
) -> tuple[FeaturePlan, list[TaskReviewRecord]]:
    plan = load_feature_plan(metadata_path)
    reviewed_task_ids = {item.task_id for item in plan.review_records}
    created: list[TaskReviewRecord] = []
    for task in plan.tasks:
        if normalize_task_status(task.status) not in {STATUS_COMPLETED, STATUS_ARCHIVED}:
            continue
        if task.id in reviewed_task_ids:
            continue
        review = TaskReviewRecord(
            id=f"review-{int(time.time() * 1000)}-{task.id}",
            task_id=task.id,
            summary=f"Review completed for task '{task.title}'.",
            limitations=[],
            issues=[],
        )
        plan.review_records.append(review)
        created.append(review)
        plan.add_event(
            kind="task_review_created",
            entity="task",
            entity_id=task.id,
            payload={"review_id": review.id, "issue_count": 0, "auto_generated": True},
            actor=actor,
        )
    return save_feature_plan("", plan), created


def create_diff_proposal(
    metadata_path: str,
    *,
    review_id: str,
    issue_id: str,
    diff: str,
    actor: str = "agent",
) -> tuple[FeaturePlan, DiffProposal]:
    plan = load_feature_plan(metadata_path)
    review = next((item for item in plan.review_records if item.id == review_id), None)
    if review is None:
        raise ValueError(f"Review {review_id} not found")
    proposal = DiffProposal(
        id=f"proposal-{int(time.time() * 1000)}-{review.task_id}",
        review_id=review.id,
        task_id=review.task_id,
        issue_id=issue_id.strip(),
        diff=diff,
    )
    plan.diff_proposals.append(proposal)
    plan.add_event(
        kind="diff_proposed",
        entity="task",
        entity_id=review.task_id,
        payload={"proposal_id": proposal.id, "review_id": review.id, "issue_id": issue_id},
        actor=actor,
    )
    return save_feature_plan("", plan), proposal


def decide_diff_proposal(
    metadata_path: str,
    *,
    proposal_id: str,
    decision: str,
    reason: str = "",
    actor: str = "user",
) -> tuple[FeaturePlan, DiffProposal]:
    plan = load_feature_plan(metadata_path)
    proposal = next((item for item in plan.diff_proposals if item.id == proposal_id), None)
    if proposal is None:
        raise ValueError(f"Proposal {proposal_id} not found")
    normalized = decision.strip().lower()
    if normalized not in {"approved", "denied"}:
        raise ValueError("decision must be 'approved' or 'denied'")
    proposal.status = normalized
    proposal.decision_reason = reason.strip()
    proposal.decided_at = time.time()
    plan.add_event(
        kind="diff_decision",
        entity="task",
        entity_id=proposal.task_id,
        payload={
            "proposal_id": proposal.id,
            "decision": proposal.status,
            "reason": proposal.decision_reason,
        },
        actor=actor,
    )
    return save_feature_plan("", plan), proposal


def mark_proposal_applied(
    metadata_path: str,
    *,
    proposal_id: str,
    applied_file: str,
    actor: str = "agent",
) -> tuple[FeaturePlan, DiffProposal]:
    """Mark an approved proposal as applied (single-use enforcement)."""
    plan = load_feature_plan(metadata_path)
    proposal = next((item for item in plan.diff_proposals if item.id == proposal_id), None)
    if proposal is None:
        raise ValueError(f"Proposal {proposal_id} not found")
    if proposal.applied_at is not None:
        raise ValueError(f"Proposal {proposal_id} was already applied")
    proposal.applied_at = time.time()
    proposal.applied_file = applied_file
    plan.add_event(
        kind="diff_applied",
        entity="task",
        entity_id=proposal.task_id,
        payload={"proposal_id": proposal.id, "file": applied_file},
        actor=actor,
    )
    return save_feature_plan("", plan), proposal


def task_archive_ready(plan: FeaturePlan, task_id: int) -> bool:
    if not any(review.task_id == task_id for review in plan.review_records):
        return False
    proposals = [item for item in plan.diff_proposals if item.task_id == task_id]
    if not proposals:
        return True
    return all(item.status == "approved" for item in proposals)


def archive_task(metadata_path: str, *, task_id: int, actor: str = "user") -> FeaturePlan:
    plan = load_feature_plan(metadata_path)
    if not task_archive_ready(plan, task_id):
        raise ValueError(f"Task {task_id} is not archive-ready")
    transition_task_status(plan, task_id=task_id, to_status=STATUS_ARCHIVED, actor=actor)
    plan.add_event(
        kind="task_archived",
        entity="task",
        entity_id=task_id,
        payload={"archived": True},
        actor=actor,
    )
    return save_feature_plan("", plan)


def build_phase_execution_prompt(plan: FeaturePlan, task: FeatureTask) -> str:
    return (
        f"Continue implementing feature '{plan.feature_name}'. "
        f"Work on task {task.id}: '{task.title}' only. "
        f"Objectives: {', '.join(task.objectives)}. "
        f"Exit Criteria: {', '.join(task.exit_criteria)}. "
        "Perform one bounded step for this task, then verify and record progress before taking the next step. "
        "Use save_scratchpad for short-lived notes/plans and save_memory for durable findings/decisions during this loop. "
        "Update the task status to 'completed' only when all exit criteria are met and verified. "
        "If you are blocked, explain why in your response."
    )


def build_review_prompt(plan: FeaturePlan) -> str:
    return (
        f"Review the completed feature '{plan.feature_name}'. "
        "Use memory and scratchpad context to confirm all decisions and validations are consistent with prior steps. "
        "Verify all tasks and their exit criteria were genuinely met. "
        "Run phase-4 review sequence explicitly: call review_all_completed_tasks, then review_completed_tasks for each completed task with categorized issues (bug/risk/enhancement), then propose_task_diff for any fixes, then decide_task_diff after user decision, and finally archive_task for archive-ready tasks. "
        "If everything passes review, set review_status to 'completed'. "
        "Otherwise, update the failing task status and explain the failure."
    )
