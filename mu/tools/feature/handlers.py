"""Feature-mode `@tool` handlers.

Each `@tool` wrapper delegates to a private `_handle_<tool>` body
defined in this module. Helpers `_resolve_feature_state` and
`_resolve_feature_metadata_path` are shared across the handlers.
"""

import json
import os
import re
import time
from dataclasses import asdict
from typing import Any, Dict

from mu.feature.engine import (
    STATUS_ARCHIVED,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    STATUS_NOT_STARTED,
    STATUS_PENDING,
    ALLOWED_TASK_TRANSITIONS,
    _workspace_root,
    archive_task as archive_feature_task,
    create_diff_proposal,
    create_feature_phases,
    create_feature_plan,
    create_feature_shell,
    create_feature_task as engine_create_feature_task,
    create_task_review_record,
    decide_diff_proposal,
    feature_execution_snapshot,
    load_feature_plan,
    next_pending_phase,
    normalize_task_status,
    refresh_and_persist_feature_plan,
    review_all_completed_tasks as create_reviews_for_completed_tasks,
    save_feature_plan,
    summarize_feature_plan,
    transition_task_status,
    update_feature_plan_metadata,
    update_task_content,
    update_task_status as engine_update_task_status,
)
from mu.tools import tool
from mu.tools.descriptors import ToolExecutionContext


# ----------------------------------------------------------------- helpers


def _handle_raise_blocker(args, folder_context, ui, variables) -> str:
    payload = {
        "kind": "feature_blocker",
        "summary": str(args.get("summary", "")).strip(),
        "details": str(args.get("details", "")).strip(),
        "requested_input": str(args.get("requested_input", "")).strip(),
        "questions": [
            str(item).strip() for item in args.get("questions", []) if str(item).strip()
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _resolve_feature_state(session, requested_feature_id: str | None = None):
    feature_state = None
    if requested_feature_id:
        feature_state = session.session_manager.get_feature(requested_feature_id)
    if not feature_state:
        feature_state = session.session_manager.get_feature_state()
    if not isinstance(feature_state, dict):
        return feature_state

    feature_id = str(feature_state.get("feature_id", "") or "").strip()
    directory = str(feature_state.get("directory", "") or "").strip()
    metadata_path = str(feature_state.get("metadata_path", "") or "").strip()

    candidates = [metadata_path]
    if feature_id and hasattr(session.session_manager, "get_feature_metadata_path"):
        try:
            candidates.append(
                str(session.session_manager.get_feature_metadata_path(feature_id) or "").strip()
            )
        except TypeError:
            pass
    if directory and hasattr(session.session_manager, "get_feature_metadata_index"):
        metadata_index = session.session_manager.get_feature_metadata_index() or {}
        if isinstance(metadata_index, dict):
            candidates.append(str(metadata_index.get(directory, "") or "").strip())
    if directory:
        candidates.append(os.path.join(directory, "feature_plan.json"))

    resolved = next((path for path in candidates if path and os.path.exists(path)), "")
    if resolved and resolved != metadata_path:
        feature_state["metadata_path"] = resolved
        if feature_id:
            session.session_manager.upsert_feature(feature_state)
        if session.session_manager.get_feature_state():
            session.session_manager.set_feature_state(feature_state)
        session.session_manager.save_history_turn()
    return feature_state


def _resolve_feature_metadata_path(
    session,
    context: ToolExecutionContext,
    *,
    feature_id: str | None = None,
    directory: str | None = None,
) -> str:
    feature_state = _resolve_feature_state(session, feature_id)
    candidates: list[str] = []
    if isinstance(feature_state, dict):
        candidates.append(str(feature_state.get("metadata_path", "") or "").strip())
        if not directory:
            directory = str(feature_state.get("directory", "") or "").strip()
    if feature_id and hasattr(session.session_manager, "get_feature_metadata_path"):
        try:
            candidates.append(
                str(session.session_manager.get_feature_metadata_path(feature_id) or "").strip()
            )
        except TypeError:
            pass
    if directory and hasattr(session.session_manager, "get_feature_metadata_index"):
        metadata_index = session.session_manager.get_feature_metadata_index() or {}
        if isinstance(metadata_index, dict):
            candidates.append(str(metadata_index.get(directory, "") or "").strip())
    if directory:
        candidates.append(os.path.join(directory, "feature_plan.json"))
    folder_index = getattr(context.folder_context, "feature_metadata_index", {}) or {}
    if directory and isinstance(folder_index, dict):
        candidates.append(str(folder_index.get(directory, "") or "").strip())
    return next((path for path in candidates if path and os.path.exists(path)), "")


def _handle_create_feature(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."

    # Duplicate-feature guard: refuse if a non-archived feature already exists.
    registry = getattr(session.session_manager, "feature_registry", None) or {}
    existing_active = [
        (fid, rec.get("feature_name", fid))
        for fid, rec in registry.items()
        if isinstance(rec, dict) and not rec.get("archived")
    ]
    if existing_active:
        active_list = ", ".join(f"{name} (id={fid})" for fid, name in existing_active)
        return json.dumps({
            "ok": False,
            "error": (
                f"An active feature already exists: {active_list}. "
                "Use the existing feature instead of creating a duplicate. "
                "If you want to start fresh, archive or delete the existing feature first."
            ),
        })

    feature_name = str(args.get("feature_name", "")).strip()
    feature_request = str(args.get("feature_request", "")).strip()
    feature_id = str(args.get("feature_id", "")).strip() or None
    design_plan = str(args.get("design_plan", "")).strip()

    if not feature_name:
        return "Error: feature_name is required."
    if not feature_request:
        return "Error: feature_request is required."

    requested_feature_id = feature_id or re.sub(
        r"[^a-zA-Z0-9]+", "_", feature_name.lower()
    ).strip("_")
    feature_id = session.session_manager.allocate_feature_id(requested_feature_id)
    metadata_path = session.session_manager.get_feature_metadata_path(feature_id)
    plan = create_feature_shell(
        feature_name=feature_name,
        feature_request=feature_request,
        folder_context=context.folder_context,
        feature_id=feature_id,
        metadata_path=metadata_path,
    )
    if design_plan:
        plan.review_notes = design_plan
        plan = update_feature_plan_metadata(
            path_or_session_id=plan.metadata_path,
            review_notes=design_plan,
            metadata_path=plan.metadata_path,
        )

    summary = summarize_feature_plan(plan)
    feature_record = {
        "type": "feature",
        "status": "draft",
        "feature_id": plan.feature_id,
        "feature_name": plan.feature_name,
        "directory": plan.directory,
        "metadata_path": plan.metadata_path,
        "feature_plan": summary,
        "blocker": None,
        "updated_at": time.time(),
    }
    session.session_manager.upsert_feature(feature_record)
    session.session_manager.activate_feature(plan.feature_id)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "feature_id": plan.feature_id,
            "metadata_path": plan.metadata_path,
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_create_phases(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    phases = args.get("phases", [])
    if not isinstance(phases, list) or not phases:
        return "Error: phases array is required."
    feature_id = str(args.get("feature_id", "")).strip() or None
    feature_state = _resolve_feature_state(session, feature_id)
    if not feature_state:
        return "Error: No active feature in session. Call create_feature first."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."

    # Approval gate: refuse to create phases on an unapproved feature.
    plan = load_feature_plan(metadata_path)
    if not getattr(plan, "approved", False):
        return json.dumps({
            "ok": False,
            "error": (
                "Feature plan has not been approved yet. "
                "Call approve_feature_task or use the /feature approve flow before "
                "creating phases or tasks. The plan must be approved by the user first."
            ),
        })

    plan = create_feature_phases(
        metadata_path,
        phases,
        replace_existing=bool(args.get("replace_existing", True)),
        actor="agent",
    )
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "feature_id": plan.feature_id,
            "phase_count": len(plan.phases_meta),
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_create_task(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    feature_id = str(args.get("feature_id", "")).strip() or None
    feature_state = _resolve_feature_state(session, feature_id)
    if not feature_state:
        return "Error: No active feature in session. Call create_feature first."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."

    # Approval gate: refuse to create tasks on an unapproved feature.
    plan_check = load_feature_plan(metadata_path)
    if not getattr(plan_check, "approved", False):
        return json.dumps({
            "ok": False,
            "error": (
                "Feature plan has not been approved yet. "
                "Call approve_feature_task or use the /feature approve flow before "
                "creating phases or tasks. The plan must be approved by the user first."
            ),
        })

    title = str(args.get("title", "")).strip()
    exit_criteria = args.get("exit_criteria", [])
    if not title:
        return "Error: title is required."
    if not isinstance(exit_criteria, list) or not exit_criteria:
        return "Error: exit_criteria must be a non-empty array."
    task_data = {
        "phase_id": args.get("phase_id"),
        "title": title,
        "objectives": [str(args.get("overview", "")).strip()] if args.get("overview") else [],
        "action_points": [str(item).strip() for item in args.get("design", [])],
        "exit_criteria": [str(item).strip() for item in exit_criteria],
        "notes": str(args.get("notes", "") or ""),
    }
    plan, task = engine_create_feature_task(metadata_path, task_data, actor="agent")
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "feature_id": plan.feature_id,
            "task_id": task.id,
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_get_execution_state(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    feature_id = str(args.get("feature_id", "")).strip() or None
    feature_state = _resolve_feature_state(session, feature_id)
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = _resolve_feature_metadata_path(
        session,
        context,
        feature_id=feature_id,
        directory=str(feature_state.get("directory", "") or "").strip(),
    )
    if not metadata_path:
        return "Error: Feature metadata not found."
    plan = load_feature_plan(metadata_path)
    snapshot = feature_execution_snapshot(plan)
    return json.dumps({"ok": True, "execution": snapshot}, indent=2, sort_keys=True)


def _handle_block_task(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    task_id = args.get("task_id")
    reason = str(args.get("reason", "")).strip()
    if task_id is None:
        return "Error: task_id is required."
    if not reason:
        return "Error: reason is required."
    feature_id = str(args.get("feature_id", "")).strip() or None
    feature_state = _resolve_feature_state(session, feature_id)
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."
    plan = load_feature_plan(metadata_path)
    transition_task_status(
        plan,
        task_id=int(task_id),
        to_status="blocked",
        notes=str(args.get("requested_input", "") or ""),
        blocked_reason=reason,
        actor="agent",
    )
    plan = save_feature_plan("", plan)
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "task_id": int(task_id),
            "status": "blocked",
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_resume_task(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    task_id = args.get("task_id")
    if task_id is None:
        return "Error: task_id is required."
    feature_id = str(args.get("feature_id", "")).strip() or None
    feature_state = _resolve_feature_state(session, feature_id)
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."
    plan = load_feature_plan(metadata_path)
    transition_task_status(
        plan,
        task_id=int(task_id),
        to_status="in_progress",
        notes=str(args.get("notes", "") or ""),
        actor="agent",
    )
    plan = save_feature_plan("", plan)
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "task_id": int(task_id),
            "status": "in_progress",
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_review_completed_tasks(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    feature_state = _resolve_feature_state(
        session, str(args.get("feature_id", "")).strip() or None
    )
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."
    plan, review = create_task_review_record(
        metadata_path,
        task_id=int(args.get("task_id")),
        summary=str(args.get("summary", "")),
        limitations=args.get("limitations", []),
        issues=args.get("issues", []),
        actor="agent",
    )
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "review": asdict(review),
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_review_all_completed_tasks(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    feature_state = _resolve_feature_state(
        session, str(args.get("feature_id", "")).strip() or None
    )
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."
    plan, created = create_reviews_for_completed_tasks(metadata_path, actor="agent")
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "created_review_count": len(created),
            "reviews": [asdict(item) for item in created],
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_propose_task_diff(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    feature_state = _resolve_feature_state(
        session, str(args.get("feature_id", "")).strip() or None
    )
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."
    plan, proposal = create_diff_proposal(
        metadata_path,
        review_id=str(args.get("review_id", "")),
        issue_id=str(args.get("issue_id", "")),
        diff=str(args.get("diff", "")),
        actor="agent",
    )
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {"ok": True, "proposal": asdict(proposal), "plan": summary},
        indent=2,
        sort_keys=True,
    )


def _handle_decide_task_diff(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    feature_state = _resolve_feature_state(
        session, str(args.get("feature_id", "")).strip() or None
    )
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."
    plan, proposal = decide_diff_proposal(
        metadata_path,
        proposal_id=str(args.get("proposal_id", "")),
        decision=str(args.get("decision", "")),
        reason=str(args.get("reason", "")),
        actor="user",
    )
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {"ok": True, "proposal": asdict(proposal), "plan": summary},
        indent=2,
        sort_keys=True,
    )


def _handle_archive_task(args: dict, context: ToolExecutionContext) -> str:
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."
    feature_state = _resolve_feature_state(
        session, str(args.get("feature_id", "")).strip() or None
    )
    if not feature_state:
        return "Error: No active feature in session."
    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."
    plan = archive_feature_task(metadata_path, task_id=int(args.get("task_id")), actor="user")
    summary = summarize_feature_plan(plan)
    feature_state["feature_plan"] = summary
    feature_state["updated_at"] = time.time()
    session.session_manager.upsert_feature(feature_state)
    session.session_manager.save_history_turn()
    return json.dumps(
        {
            "ok": True,
            "task_id": int(args.get("task_id")),
            "status": "archived",
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_create_feature_task(args: dict, context: ToolExecutionContext) -> str:
    """Creates a structured feature implementation plan."""
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."

    # Duplicate-feature guard: refuse if a non-archived feature already exists.
    registry = getattr(session.session_manager, "feature_registry", None) or {}
    existing_active = [
        (fid, rec.get("feature_name", fid))
        for fid, rec in registry.items()
        if isinstance(rec, dict) and not rec.get("archived")
    ]
    if existing_active:
        active_list = ", ".join(f"{name} (id={fid})" for fid, name in existing_active)
        return json.dumps({
            "ok": False,
            "error": (
                f"An active feature already exists: {active_list}. "
                "Use the existing feature instead of creating a duplicate. "
                "If you want to start fresh, archive or delete the existing feature first."
            ),
        })

    feature_name = args.get("feature_name", "").strip()
    feature_request = args.get("feature_request", "").strip()
    feature_id = args.get("feature_id", "").strip() or None
    tasks_data = args.get("tasks", [])

    if not feature_name:
        return "Error: feature_name is required."
    if not tasks_data:
        return "Error: tasks array is required."

    if isinstance(tasks_data, str):
        raw_tasks = tasks_data.strip()
        try:
            tasks_data = json.loads(raw_tasks)
        except json.JSONDecodeError as exc:
            return (
                "Error: tasks must be a JSON array of task objects. "
                f"Received an invalid JSON string ({exc.msg} at pos {exc.pos})."
            )

    if not isinstance(tasks_data, list):
        return (
            "Error: tasks must be an array of task objects, "
            f"got {type(tasks_data).__name__}."
        )

    first_invalid = next(
        (
            (idx, item)
            for idx, item in enumerate(tasks_data, start=1)
            if not isinstance(item, dict)
        ),
        None,
    )
    if first_invalid:
        idx, item = first_invalid
        return (
            "Error: tasks must be an array of objects. "
            f"Task #{idx} is {type(item).__name__}: {item!r}"
        )

    # Get or create feature record
    existing_feature = session.session_manager.get_feature(feature_id)
    if existing_feature:
        metadata_path = existing_feature.get("metadata_path", "")
        directory = existing_feature.get("directory", "")
    else:
        # Create new feature
        directory = _workspace_root(context.folder_context)
        requested_feature_id = feature_id or re.sub(
            r"[^a-zA-Z0-9]+", "_", feature_name.lower()
        ).strip("_")
        feature_id = session.session_manager.allocate_feature_id(requested_feature_id)
        metadata_path = session.session_manager.get_feature_metadata_path(feature_id)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Create the feature plan
    plan = create_feature_plan(
        feature_name=feature_name,
        feature_request=feature_request,
        tasks_data=tasks_data,
        folder_context=context.folder_context,
        feature_id=feature_id,
        metadata_path=metadata_path,
    )

    # Update session state
    summary = summarize_feature_plan(plan)
    feature_record = {
        "type": "feature",
        "status": "draft",
        "feature_id": plan.feature_id,
        "feature_name": plan.feature_name,
        "directory": directory or plan.directory,
        "metadata_path": plan.metadata_path,
        "feature_plan": summary,
        "blocker": None,
        "updated_at": time.time(),
    }
    session.session_manager.upsert_feature(feature_record)
    session.session_manager.activate_feature(plan.feature_id)
    session.session_manager.save_history_turn()

    if context.ui:
        context.ui.show_info(
            f"Created feature plan: {plan.feature_id} with {len(plan.tasks)} tasks"
        )

    return json.dumps(
        {
            "ok": True,
            "feature_id": plan.feature_id,
            "task_count": len(plan.tasks),
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_update_feature_task(args: dict, context: ToolExecutionContext) -> str:
    """Updates task content before approval."""
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."

    task_id = args.get("task_id")
    if task_id is None:
        return "Error: task_id is required."

    feature_state = session.session_manager.get_feature_state()
    if not feature_state:
        return "Error: No active feature in session."

    metadata_path = feature_state.get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."

    plan = update_task_content(
        metadata_path,
        task_id,
        title=args.get("title"),
        objectives=args.get("objectives"),
        action_points=args.get("action_points"),
        exit_criteria=args.get("exit_criteria"),
        notes=args.get("notes"),
    )

    summary = summarize_feature_plan(plan)
    session.session_manager.set_feature_state(
        {
            "feature_plan": summary,
            **feature_state,
        }
    )

    return json.dumps(
        {"ok": True, "task_id": task_id, "plan": summary}, indent=2, sort_keys=True
    )


def _handle_approve_feature_task(args: dict, context: ToolExecutionContext) -> str:
    """Approves or rejects the feature plan."""
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."

    approved = args.get("approved", True)

    feature_state = session.session_manager.get_feature_state() or {}
    directory = str(
        args.get("directory")
        or feature_state.get("directory", "")
        or ""
    ).strip()
    metadata_path = str(feature_state.get("metadata_path", "") or "").strip()
    if not metadata_path:
        metadata_path = str(
            getattr(context.folder_context, "feature_metadata_index", {}).get(
                directory, ""
            )
            or ""
        ).strip()
    if not metadata_path and directory:
        metadata_path = os.path.join(directory, "feature_plan.json")
    if not metadata_path or not os.path.exists(metadata_path):
        return "Error: Feature metadata not found."

    plan = update_feature_plan_metadata(
        directory or feature_state.get("directory", ""),
        approved=approved,
        review_status=args.get("review_status"),
        review_notes=args.get("review_notes"),
        metadata_path=metadata_path,
    )

    summary = summarize_feature_plan(plan)
    status = "approved" if approved else "rejected"

    if context.ui:
        context.ui.show_info(f"Feature plan {status}: {plan.feature_id}")

    # Update in-memory feature state so status reflects approval/review
    feature_state = session.session_manager.get_feature_state() or {}
    updated_feature = {
        **feature_state,
        "directory": directory or feature_state.get("directory", ""),
        "metadata_path": metadata_path,
        "feature_plan": summary,
    }
    session.session_manager.set_feature_state(updated_feature)

    return json.dumps(
        {
            "ok": True,
            "approved": approved,
            "feature_id": plan.feature_id,
            "plan": summary,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_get_current_task(args: dict, context: ToolExecutionContext) -> str:
    """Gets the current active task."""
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."

    feature_state = _resolve_feature_state(session)
    if not feature_state:
        return json.dumps(
            {"error": "No active feature in session.", "task": None}, indent=2
        )

    metadata_path = _resolve_feature_metadata_path(
        session,
        context,
        feature_id=str(feature_state.get("feature_id", "") or "").strip() or None,
        directory=str(feature_state.get("directory", "") or "").strip(),
    )
    if not metadata_path:
        return json.dumps(
            {"error": "Feature metadata not found.", "task": None}, indent=2
        )

    plan = load_feature_plan(metadata_path)
    next_task = plan.next_incomplete_task()

    if next_task:
        return json.dumps(
            {"task": asdict(next_task), "feature_id": plan.feature_id},
            indent=2,
            sort_keys=True,
        )
    else:
        return json.dumps(
            {
                "task": None,
                "message": "All tasks completed.",
                "feature_id": plan.feature_id,
            },
            indent=2,
        )


def _handle_get_tasks(args: dict, context: ToolExecutionContext) -> str:
    """Gets all tasks in the feature plan."""
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."

    feature_state = _resolve_feature_state(session)
    if not feature_state:
        return json.dumps(
            {"error": "No active feature in session.", "tasks": []}, indent=2
        )

    metadata_path = _resolve_feature_metadata_path(
        session,
        context,
        feature_id=str(feature_state.get("feature_id", "") or "").strip() or None,
        directory=str(feature_state.get("directory", "") or "").strip(),
    )
    if not metadata_path:
        return json.dumps(
            {"error": "Feature metadata not found.", "tasks": []}, indent=2
        )

    plan = load_feature_plan(metadata_path)
    tasks = [asdict(t) for t in plan.tasks]

    return json.dumps(
        {
            "tasks": tasks,
            "feature_id": plan.feature_id,
            "feature_name": plan.feature_name,
        },
        indent=2,
        sort_keys=True,
    )


def _handle_update_task_status(args: dict, context: ToolExecutionContext) -> str:
    """Updates task status during execution."""
    session = context.session
    if not session:
        return "Error: This tool requires an active session context."

    task_id = args.get("task_id")
    status = args.get("status")
    notes = args.get("notes")
    verified_exit_criteria = args.get("verified_exit_criteria", [])

    if task_id is None:
        return "Error: task_id is required."
    if not status:
        return "Error: status is required."

    valid_statuses = [
        "pending",
        "not_started",
        "in_progress",
        "blocked",
        "completed",
        "archived",
    ]
    if status not in valid_statuses:
        return f"Error: status must be one of {valid_statuses}."

    feature_state = _resolve_feature_state(session) or {}
    directory = str(
        args.get("directory")
        or feature_state.get("directory", "")
        or ""
    ).strip()
    metadata_path = _resolve_feature_metadata_path(
        session,
        context,
        feature_id=str(feature_state.get("feature_id", "") or "").strip() or None,
        directory=directory,
    )
    if not metadata_path:
        return "Error: Feature metadata not found."

    # Approval gate: refuse to update tasks on an unapproved feature.
    approval_check = load_feature_plan(metadata_path)
    if not getattr(approval_check, "approved", False):
        return json.dumps({
            "ok": False,
            "error": (
                "Feature plan has not been approved yet. "
                "Call approve_feature_task or use the /feature approve flow before "
                "updating task status. The plan must be approved by the user first."
            ),
        })

    if verified_exit_criteria is not None and not isinstance(verified_exit_criteria, list):
        return "Error: verified_exit_criteria must be an array when provided."

    if status == "completed":
        plan_snapshot = load_feature_plan(metadata_path)
        target_task = next(
            (item for item in plan_snapshot.tasks if item.id == int(task_id)),
            None,
        )
        if target_task is None:
            return f"Error: Task {task_id} not found."
        expected = [str(item).strip() for item in target_task.exit_criteria if str(item).strip()]
        # Round-30 F1: an empty expected list would make the missing
        # check vacuously pass — completing a task with no verifiable
        # criteria is refused (mirrors the engine-side guard).
        if not expected:
            return (
                "Error: Task has no exit criteria to verify — it cannot "
                "be marked completed on a vacuous basis. Add exit "
                "criteria first."
            )
        already_verified = {
            str(item).strip()
            for item in getattr(target_task, "verified_exit_criteria", []) or []
            if str(item).strip()
        }
        provided = {
            str(item).strip() for item in (verified_exit_criteria or []) if str(item).strip()
        }
        effective_verified = already_verified | provided
        missing = [criterion for criterion in expected if criterion not in effective_verified]
        if missing:
            return (
                "Error: Cannot mark task completed until all exit criteria are verified. "
                f"Missing: {missing}"
            )
    else:
        plan_snapshot = load_feature_plan(metadata_path)
        target_task = next(
            (item for item in plan_snapshot.tasks if item.id == int(task_id)),
            None,
        )
        if target_task is None:
            return f"Error: Task {task_id} not found."
        already_verified = {
            str(item).strip()
            for item in getattr(target_task, "verified_exit_criteria", []) or []
            if str(item).strip()
        }
        provided = {
            str(item).strip() for item in (verified_exit_criteria or []) if str(item).strip()
        }
        effective_verified = already_verified | provided

    plan = engine_update_task_status(
        metadata_path,
        task_id,
        status,
        notes,
        verified_exit_criteria=sorted(effective_verified),
    )
    summary = summarize_feature_plan(plan)

    # Update session state
    updated_feature = {
        **feature_state,
        "directory": directory or feature_state.get("directory", summary.get("directory")),
        "metadata_path": metadata_path,
        "feature_plan": summary,
    }
    session.session_manager.set_feature_state(updated_feature)

    if context.ui:
        context.ui.show_info(f"Task {task_id} status updated to '{status}'")

    return json.dumps(
        {"ok": True, "task_id": task_id, "status": status, "plan": summary},
        indent=2,
        sort_keys=True,
    )



# ----------------------------------------------------------------- @tool wrappers


# ---------------------------------------------------------------- planning


@tool(
    name="create_feature",
    description=(
        "Creates (or upserts) a feature shell from a confirmed design "
        "plan. Stage 1 of feature mode planning."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_name": {"type": "string"},
            "feature_request": {"type": "string"},
            "feature_id": {"type": "string"},
            "design_plan": {"type": "string"},
        },
        "required": ["feature_name", "feature_request"],
    },
    requires_approval=False,
    phase="feature",
)
def create_feature(args: Dict[str, Any], context) -> str:
    return _handle_create_feature(args, context)


@tool(
    name="create_phases",
    description=(
        "Creates or replaces phases/epics for an active feature. Stage 2 "
        "of feature mode planning."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "replace_existing": {"type": "boolean", "default": True},
            "phases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "goal": {"type": "string"},
                        "order": {"type": "integer"},
                        "status": {"type": "string"},
                    },
                    "required": ["title", "goal"],
                },
            },
        },
        "required": ["phases"],
    },
    requires_approval=False,
    phase="feature",
)
def create_phases(args: Dict[str, Any], context) -> str:
    return _handle_create_phases(args, context)


@tool(
    name="create_task",
    description=(
        "Creates a single task/ticket for an active feature phase. "
        "Stage 3 of feature mode planning."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "phase_id": {"type": "integer"},
            "title": {"type": "string"},
            "overview": {"type": "string"},
            "design": {"type": "array", "items": {"type": "string"}},
            "exit_criteria": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["title", "exit_criteria"],
    },
    requires_approval=False,
    phase="feature",
)
def create_task(args: Dict[str, Any], context) -> str:
    return _handle_create_task(args, context)


@tool(
    name="get_execution_state",
    description=(
        "Returns the phase/task execution cursor, including blocked "
        "tasks and next actionable work item."
    ),
    parameters={
        "type": "object",
        "properties": {"feature_id": {"type": "string"}},
    },
    requires_approval=False,
    phase="feature",
)
def get_execution_state(args: Dict[str, Any], context) -> str:
    return _handle_get_execution_state(args, context)


# ---------------------------------------------------------------- status


@tool(
    name="block_task",
    description=(
        "Moves a task to blocked with an explicit reason and optional "
        "user input request."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "task_id": {"type": "integer"},
            "reason": {"type": "string"},
            "requested_input": {"type": "string"},
        },
        "required": ["task_id", "reason"],
    },
    requires_approval=False,
    phase="feature",
)
def block_task(args: Dict[str, Any], context) -> str:
    return _handle_block_task(args, context)


@tool(
    name="resume_task",
    description=(
        "Moves a blocked task back to in_progress after required user "
        "input has been provided."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "task_id": {"type": "integer"},
            "notes": {"type": "string"},
        },
        "required": ["task_id"],
    },
    requires_approval=False,
    phase="feature",
)
def resume_task(args: Dict[str, Any], context) -> str:
    return _handle_resume_task(args, context)


@tool(
    name="archive_task",
    description=(
        "Archives an archive-ready task after review and diff decisions "
        "are complete."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "task_id": {"type": "integer"},
        },
        "required": ["task_id"],
    },
    requires_approval=False,
    phase="feature",
)
def archive_task(args: Dict[str, Any], context) -> str:
    return _handle_archive_task(args, context)


@tool(
    name="update_task_status",
    description=(
        "Updates the status of a specific task. Provide "
        "verified_exit_criteria incrementally as criteria are met; set "
        "status='completed' only after all task exit_criteria are verified."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "status": {
                "type": "string",
                "enum": [
                    "pending",
                    "not_started",
                    "in_progress",
                    "blocked",
                    "completed",
                    "archived",
                ],
            },
            "notes": {"type": "string"},
            "verified_exit_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exit criteria already verified for this task. Update "
                    "incrementally as work progresses; must include every "
                    "task exit criterion before completion."
                ),
            },
            "directory": {"type": "string"},
        },
        "required": ["task_id", "status"],
    },
    requires_approval=False,
    phase="feature",
)
def update_task_status(args: Dict[str, Any], context) -> str:
    return _handle_update_task_status(args, context)


# ---------------------------------------------------------------- review


@tool(
    name="review_completed_tasks",
    description=(
        "Creates structured review records for completed tasks with "
        "categorized issues (bug/risk/enhancement)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "task_id": {"type": "integer"},
            "summary": {"type": "string"},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["bug", "risk", "enhancement"],
                        },
                        "details": {"type": "string"},
                    },
                    "required": ["title", "category"],
                },
            },
        },
        "required": ["task_id", "summary"],
    },
    requires_approval=False,
    phase="feature",
)
def review_completed_tasks(args: Dict[str, Any], context) -> str:
    return _handle_review_completed_tasks(args, context)


@tool(
    name="review_all_completed_tasks",
    description=(
        "Auto-creates baseline review records for every completed task "
        "that does not yet have one."
    ),
    parameters={
        "type": "object",
        "properties": {"feature_id": {"type": "string"}},
    },
    requires_approval=False,
    phase="feature",
)
def review_all_completed_tasks(args: Dict[str, Any], context) -> str:
    return _handle_review_all_completed_tasks(args, context)


@tool(
    name="propose_task_diff",
    description=(
        "Creates a diff proposal for a review issue, requiring later "
        "user decision."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "review_id": {"type": "string"},
            "issue_id": {"type": "string"},
            "diff": {"type": "string"},
        },
        "required": ["review_id", "issue_id", "diff"],
    },
    requires_approval=False,
    phase="feature",
)
def propose_task_diff(args: Dict[str, Any], context) -> str:
    return _handle_propose_task_diff(args, context)


@tool(
    name="decide_task_diff",
    description="Stores user decision (approved/denied) for a proposed task diff.",
    parameters={
        "type": "object",
        "properties": {
            "feature_id": {"type": "string"},
            "proposal_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["approved", "denied"]},
            "reason": {"type": "string"},
        },
        "required": ["proposal_id", "decision"],
    },
    requires_approval=False,
    phase="feature",
)
def decide_task_diff(args: Dict[str, Any], context) -> str:
    return _handle_decide_task_diff(args, context)


# ---------------------------------------------------------------- feature-task engine


@tool(
    name="create_feature_task",
    description=(
        "Creates a structured feature implementation plan consisting of "
        "one or more tasks. Each task must include explicit "
        "exit_criteria. Stores metadata internally."
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_name": {
                "type": "string",
                "description": "Short feature name.",
            },
            "feature_request": {
                "type": "string",
                "description": "Full description of the feature request.",
            },
            "feature_id": {
                "type": "string",
                "description": "Optional stable identifier.",
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "objectives": {"type": "array", "items": {"type": "string"}},
                        "action_points": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "exit_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "title",
                        "objectives",
                        "action_points",
                        "exit_criteria",
                    ],
                },
            },
        },
        "required": ["feature_name", "feature_request", "tasks"],
    },
    requires_approval=False,
    phase="feature",
)
def create_feature_task(args: Dict[str, Any], context) -> str:
    return _handle_create_feature_task(args, context)


@tool(
    name="update_feature_task",
    description="Modifies the details of a task before approval.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "title": {"type": "string"},
            "objectives": {"type": "array", "items": {"type": "string"}},
            "action_points": {"type": "array", "items": {"type": "string"}},
            "exit_criteria": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["task_id"],
    },
    requires_approval=True,
    phase="feature",
)
def update_feature_task(args: Dict[str, Any], context) -> str:
    return _handle_update_feature_task(args, context)


@tool(
    name="approve_feature_task",
    description="Approves the feature plan, allowing implementation to begin.",
    parameters={
        "type": "object",
        "properties": {
            "approved": {"type": "boolean", "default": True},
        },
    },
    requires_approval=True,
    phase="feature",
)
def approve_feature_task(args: Dict[str, Any], context) -> str:
    return _handle_approve_feature_task(args, context)


@tool(
    name="get_current_task",
    description="Retrieves the currently active task in the feature plan.",
    parameters={"type": "object", "properties": {}},
    requires_approval=False,
    phase="feature",
    execution_kind="read",
    preview_policy="none",
    result_mode="structured+collated",
)
def get_current_task(args: Dict[str, Any], context) -> str:
    return _handle_get_current_task(args, context)


@tool(
    name="get_tasks",
    description=(
        "Retrieves all tasks in the feature plan (previous, current, "
        "and upcoming)."
    ),
    parameters={"type": "object", "properties": {}},
    requires_approval=False,
    phase="feature",
    execution_kind="read",
    preview_policy="none",
    result_mode="structured+collated",
)
def get_tasks(args: Dict[str, Any], context) -> str:
    return _handle_get_tasks(args, context)


# ---------------------------------------------------------------- blocker


@tool(
    name="raise_blocker",
    description=(
        "Raises a structured blocker when the feature loop needs user "
        "input or an external decision before it can safely continue."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Short blocker summary shown to the user.",
            },
            "details": {
                "type": "string",
                "description": (
                    "Longer explanation of what is blocked and what has "
                    "already been tried."
                ),
            },
            "requested_input": {
                "type": "string",
                "description": (
                    "Describe the exact information or decision needed from the user."
                ),
            },
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional focused questions for the user to answer.",
            },
        },
        "required": ["summary", "requested_input"],
    },
    requires_approval=False,
    phase="feature",
    execution_kind="control",
    preview_policy="none",
    result_mode="structured",
    server_policy="session_only",
    summary_builder="blocker_summary",
)
def raise_blocker(args: Dict[str, Any], context) -> str:
    """`_handle_raise_blocker` uses a 4-arg signature; adapt to (args, context)."""
    return _handle_raise_blocker(
        args, context.folder_context, context.ui, context.variables
    )
