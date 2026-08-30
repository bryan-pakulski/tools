"""Feature-mode CLI helpers extracted from the mucli entry module.

Pure session/feature-plan manipulation shared by the CLI and GUI surfaces.
The mucli module re-exports every public name so existing
`from mucli import X` call sites keep working unchanged.
"""

from __future__ import annotations

import json
import os
import re
import time

from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.text import Text
from rich.table import Table
from rich import box

from mu.feature.engine import (
    load_feature_plan,
    refresh_and_persist_feature_plan,
    save_feature_plan,
    summarize_feature_plan,
)
from mu.session.session import derive_feature_state_status
from mu.tools._dispatcher import execute_tool
from utils.helpers import safe_markup
from utils.logger import logger

console = Console()


def _slugify_feature_id(value):
    return (
        re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        or "feature"
    )


def _default_feature_directory(session, feature_name):
    workspace_root = (
        os.path.abspath(session.folder_context.folders[0])
        if session.folder_context.folders
        else os.getcwd()
    )
    return os.path.join(
        workspace_root,
        "documentation",
        f"feature_req_{_slugify_feature_id(feature_name)}",
    )


def refresh_feature_record(session, feature_id=None):
    record = session.session_manager.get_feature(feature_id)
    if not isinstance(record, dict):
        return None

    metadata_path = str(record.get("metadata_path", "") or "").strip()
    directory = str(record.get("directory", "") or "").strip()
    if not (metadata_path and directory and os.path.exists(metadata_path)):
        return record

    try:
        plan = refresh_and_persist_feature_plan(
            session.session_manager.current_session_name,
            metadata_path=metadata_path,
        )
    except (FileNotFoundError, OSError, ValueError):
        return record

    summary = summarize_feature_plan(plan)
    updated = {
        **record,
        "feature_id": summary["feature_id"],
        "feature_name": summary["feature_name"],
        "directory": summary["directory"],
        "metadata_path": summary.get("metadata_path"),
        "feature_plan": summary,
        "next_phase": summary.get("next_phase"),
        "status": derive_feature_state_status(summary),
        "updated_at": record.get("updated_at"),
    }
    session.session_manager.upsert_feature(updated)
    if session.session_manager.active_feature_id == updated["feature_id"]:
        session.session_manager.set_feature_state(updated, session.folder_context)
    else:
        session.session_manager.save_history(session.folder_context)
        session.sync_runtime_state()
    return session.session_manager.get_feature(updated["feature_id"])


def get_current_feature_task_label(session):
    feature_state = session.session_manager.get_feature_state()
    if not isinstance(feature_state, dict):
        return None

    feature_plan = feature_state.get("feature_plan")
    if not isinstance(feature_plan, dict):
        return None

    next_task = feature_plan.get("next_task") or feature_plan.get("next_phase")
    if isinstance(next_task, dict):
        title = str(next_task.get("title", "") or "").strip()
        return title or None
    return None


def get_feature_prompt_context(session):
    feature_state = session.session_manager.get_feature_state()
    if not isinstance(feature_state, dict):
        return None

    plan = feature_state.get("feature_plan")
    if not isinstance(plan, dict):
        return None

    tasks = plan.get("phases", [])
    overall_total = max(1, len(tasks))
    overall_done = sum(1 for task in tasks if task.get("status") == "completed")
    all_completed = bool(tasks) and (
        bool(plan.get("phases_completed"))
        or bool(plan.get("tasks_completed"))
        or overall_done >= len(tasks)
        or str(feature_state.get("status", "")).strip().lower() == "completed"
    )

    next_task = plan.get("next_task") or plan.get("next_phase")
    active_task = None
    if isinstance(next_task, dict):
        next_number = next_task.get("number") or next_task.get("id")
        active_task = next(
            (task for task in tasks if task.get("number") == next_number),
            None,
        )
    if active_task is None and tasks:
        active_task = next(
            (task for task in tasks if task.get("status") != "completed"),
            tasks[0],
        )

    phase_done = 0
    phase_total = 1
    task_title = "n/a"
    if all_completed:
        phase_done = 1
        phase_total = 1
        task_title = "completed"
    elif isinstance(active_task, dict):
        task_title = str(active_task.get("title", "") or "").strip() or "n/a"
        counts = active_task.get("task_counts", {}) or {}
        phase_done = int(counts.get("completed", 0) or 0)
        phase_total = int(sum(int(v or 0) for v in counts.values()) or 0)
        if phase_total <= 0:
            phase_total = 1
            if active_task.get("status") == "completed":
                phase_done = 1

    return {
        "status": str(feature_state.get("status", "unknown") or "unknown"),
        "task": task_title,
        "phase_done": phase_done,
        "phase_total": phase_total,
        "overall_done": overall_done,
        "overall_total": overall_total,
    }


def build_feature_markdown(feature, *, include_phases=True):
    if not isinstance(feature, dict):
        return "## Feature\n\nNo feature is currently selected."

    plan = (
        feature.get("feature_plan")
        if isinstance(feature.get("feature_plan"), dict)
        else {}
    )
    feature_name = (
        feature.get("feature_name")
        or plan.get("feature_name")
        or feature.get("feature_id", "feature")
    )
    lines = [
        f"# Feature: {feature_name}",
        "",
        f"- **ID:** `{feature.get('feature_id', 'unknown')}`",
        f"- **Status:** `{feature.get('status', 'unknown')}`",
        f"- **Directory:** `{feature.get('directory', 'n/a')}`",
        f"- **Metadata:** `{feature.get('metadata_path', 'n/a')}`",
        f"- **Approved:** `{plan.get('approved', False)}`",
        f"- **Review:** `{plan.get('review_status', 'pending')}`",
        "",
    ]

    request = str(plan.get("feature_request", "") or "").strip()
    if request:
        lines.extend(["## Request", "", request, ""])

    tasks = plan.get("phases", [])
    completed = sum(1 for task in tasks if task.get("status") == "completed")
    total = len(tasks)
    started_at = float(feature.get("started_at", 0) or 0)
    elapsed = max(0, int(time.time() - started_at)) if started_at else 0
    token_total = int(feature.get("token_total", 0) or 0)
    start_tokens = int(feature.get("start_tokens", 0) or 0)
    token_delta = max(0, token_total - start_tokens)
    next_task = plan.get("next_phase")
    if not isinstance(next_task, dict):
        next_task = plan.get("next_task")

    def _fmt_elapsed(seconds):
        minutes, secs = divmod(max(0, int(seconds or 0)), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        return f"{minutes}m {secs}s"

    def _fmt_delta(tokens):
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}k"
        return str(tokens)

    lines.extend(
        [
            "## Progress Snapshot",
            "",
            f"- **Completed:** {completed}/{total}",
            f"- **Elapsed:** {_fmt_elapsed(elapsed)}",
            f"- **Token delta:** ↓ {_fmt_delta(token_delta)} tokens",
            "",
        ]
    )

    if isinstance(next_task, dict):
        lines.extend(
            [
                "### Active Work",
                "",
                f"*Implementing {next_task.get('title', '')}… ({_fmt_elapsed(elapsed)} · ↓ {_fmt_delta(token_delta)} tokens)*",
                "",
            ]
        )

    if include_phases:
        lines.extend(["### Task Checklist", ""])
        if tasks:
            for task in tasks:
                counts = task.get("task_counts", {})
                icon = {
                    "completed": "✔",
                    "in_progress": "◼",
                    "not_started": "◻",
                }.get(task.get("status", "not_started"), "◻")
                lines.append(
                    f"- {icon} **{task.get('title', '')}** "
                    f"`{task.get('status', 'unknown')}` "
                    f"(done: {counts.get('completed', 0)}, in-progress: {counts.get('in_progress', 0)}, remaining: {counts.get('not_started', 0)})"
                )
        else:
            lines.append("- No tasks defined yet.")
        lines.append("")

    blocker = feature.get("blocker")
    if isinstance(blocker, dict) and any(blocker.values()):
        lines.extend(
            [
                "## Blocker",
                "",
                f"- **Summary:** {blocker.get('summary', '')}",
                f"- **Requested input:** {blocker.get('requested_input', '')}",
                "",
            ]
        )

    return "\n".join(lines).strip()


def _feature_three_option_prompt(question, options, *, allow_prompt):
    choices = options[:3]
    if len(choices) != 3:
        raise ValueError("feature prompt requires exactly three options")
    if not allow_prompt:
        return choices[0][0]
    console.print(f"[bold cyan]{safe_markup(question)}[/bold cyan]")
    for idx, (_, label) in enumerate(choices, start=1):
        console.print(f"  {idx}. {label}", markup=False)
    selected = IntPrompt.ask("Select option", choices=[1, 2, 3], default=1)
    return choices[selected - 1][0]


def _log_feature_cli_event(session, *, kind, payload):
    feature_state = session.session_manager.get_feature_state()
    if not isinstance(feature_state, dict):
        return
    metadata_path = str(feature_state.get("metadata_path", "") or "").strip()
    if not metadata_path or not os.path.exists(metadata_path):
        return
    try:
        plan = load_feature_plan(metadata_path)
    except (FileNotFoundError, OSError, ValueError):
        return
    plan.add_event(
        kind=kind,
        entity="cli",
        entity_id=str(feature_state.get("feature_id", "unknown") or "unknown"),
        payload=payload,
        actor="cli",
    )
    save_feature_plan("", plan)
    refresh_feature_record(session, None)


def _feature_prompt_with_logging(
    session,
    *,
    question,
    options,
    allow_prompt,
    prompt_id,
    context=None,
):
    selected = _feature_three_option_prompt(question, options, allow_prompt=allow_prompt)
    _log_feature_cli_event(
        session,
        kind="cli_prompt_selected",
        payload={
            "prompt_id": prompt_id,
            "question": question,
            "selected": selected,
            "options": [option[0] for option in options[:3]],
            "context": context or {},
        },
    )
    return selected


def _feature_confirm_deny_edit_loop(
    session,
    *,
    label,
    value,
    allow_prompt,
    context=None,
):
    current_value = str(value or "").strip()
    while True:
        choice = _feature_prompt_with_logging(
            session,
            question=f"Confirm {label}: {current_value}",
            options=[
                ("confirm", "Confirm (Recommended): proceed"),
                ("edit", "Edit: change and re-confirm"),
                ("deny", "Deny: cancel command"),
            ],
            allow_prompt=allow_prompt,
            prompt_id=f"confirm_{label}",
            context={"label": label, **(context or {})},
        )
        if choice == "confirm":
            return {"decision": "confirm", "value": current_value}
        if choice == "deny":
            return {"decision": "deny", "value": current_value}
        current_value = Prompt.ask(f"Edit {label}", default=current_value).strip()


def _monitor_compact_line(snapshot):
    execution = snapshot.get("execution", {}) if isinstance(snapshot, dict) else {}
    next_phase = (execution.get("next_phase") or {}) if isinstance(execution, dict) else {}
    next_task = (execution.get("next_task") or {}) if isinstance(execution, dict) else {}
    blocked = execution.get("blocked_tasks", []) if isinstance(execution, dict) else []
    blockers = ", ".join(str(item.get("title", "")).strip() for item in blocked if isinstance(item, dict) and str(item.get("title", "")).strip())
    completion = "done" if execution.get("all_phases_completed") else "in_progress"
    return (
        f"phase={next_phase.get('title') or '-'} | "
        f"task={next_task.get('title') or '-'} | "
        f"blockers={len(blocked)}{f' ({blockers})' if blockers else ''} | "
        f"completion={completion}"
    )


def _execute_feature_tool(session, tool_name, args):
    raw = execute_tool(
        tool_name,
        args,
        session.folder_context,
        session.ui,
        session.variables,
        session=session,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": raw}
