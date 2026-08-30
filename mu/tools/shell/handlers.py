"""Shell + background-task `@tool` handlers.

The synchronous `bash` runs in-process; the bg family routes through
`BackgroundTaskRegistry` on the session (or a process-local fallback
so the tools still work in session-less unit tests).
"""

import json
import os
import subprocess
from typing import Any, Dict

from mu.tools import tool
from mu.tools._bounds import (
    check_bounds as _check_bounds,
    default_working_directory as _default_working_directory,
)
from mu.tools.capabilities import normalize_session_type, session_type_from_context
from mu.tools._scrub import scrub_and_annotate as _scrub_and_annotate


def _scrub_json_payload(payload: Any) -> Any:
    """Recursively scrub every string in a background-task payload before
    serialization (codex round-8 F4): bash_status/logs/kill/list return
    captured stdout/stderr that may contain provider keys, tokens, or
    private keys — same redaction contract as synchronous bash output."""
    from mu.security.secret_paths import redact_secrets

    if isinstance(payload, str):
        scrubbed, _ = redact_secrets(payload)
        return scrubbed
    if isinstance(payload, list):
        return [_scrub_json_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {k: _scrub_json_payload(v) for k, v in payload.items()}
    return payload


from utils.logger import logger  # noqa: E402


# ---------------------------------------------------------------- bg registry resolver


_STANDALONE_BG_REGISTRY = None


def _bg_registry(context):
    """Resolve the session's `BackgroundTaskRegistry`. Falls back to a
    process-global one if no Session is bound."""
    session = getattr(context, "session", None) if context is not None else None
    if session is not None and hasattr(session, "background_tasks"):
        return session.background_tasks
    global _STANDALONE_BG_REGISTRY
    if _STANDALONE_BG_REGISTRY is None:
        from mu.tools.shell.background import BackgroundTaskRegistry

        _STANDALONE_BG_REGISTRY = BackgroundTaskRegistry()
    return _STANDALONE_BG_REGISTRY


# ---------------------------------------------------------------- bash (synchronous)


def bash_command(
    command: str,
    folder_context,
    *,
    cwd: str | None = None,
    timeout_seconds: int = 120,
    max_output_chars: int = 12000,
    session_type: str = "workspace",
) -> str:
    """Execute a raw shell command in the active runtime.

    Container sessions use the Docker filesystem itself as the boundary; an
    attached workspace is not required and ``cwd`` may be any non-secret path
    inside the container.
    """
    command = str(command or "").strip()
    if not command:
        return "Error: command is required."

    if (
        normalize_session_type(session_type) != "container"
        and (not folder_context or not folder_context.folders)
    ):
        return "Error: No workspace attached."

    workdir = str(
        cwd or _default_working_directory(folder_context, session_type)
    ).strip()
    if not os.path.isdir(os.path.expanduser(workdir)):
        return f"Error: Working directory does not exist: '{workdir}'"
    if not _check_bounds(workdir, folder_context, session_type=session_type):
        logger.warning(f"bash_command: Access denied or path ignored: {workdir}")
        return f"Error: Access denied or path ignored. '{workdir}'"

    timeout_seconds = max(1, int(timeout_seconds or 120))
    max_output_chars = max(512, int(max_output_chars or 12000))

    try:
        process = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired as exc:
        partial = f"{exc.stdout or ''}\n{exc.stderr or ''}".strip()
        if len(partial) > max_output_chars:
            partial = partial[:max_output_chars]
        # Scrub the timeout path too (codex round-8 F5): a command that
        # prints a secret then blocks must not leak it via partial output.
        return _scrub_and_annotate(
            (
                f"Error: Command timed out after {timeout_seconds} seconds.\n"
                f"{partial}"
            ).strip()
        )
    except Exception as exc:
        logger.error(f"bash_command: Error executing command {command!r}: {exc}")
        return f"Error executing bash command: {exc}"

    chunks = []
    if process.stdout:
        chunks.append(f"STDOUT:\n{process.stdout.rstrip()}")
    if process.stderr:
        chunks.append(f"STDERR:\n{process.stderr.rstrip()}")
    if not chunks:
        chunks.append("Command executed with no output.")
    chunks.append(f"Exit code: {process.returncode}")
    output = "\n\n".join(chunks)

    if len(output) > max_output_chars:
        output = output[:max_output_chars] + "\n\n...[TRUNCATED]..."
    return _scrub_and_annotate(output)


@tool(
    name="bash",
    description=(
        "Executes a raw bash command in the active runtime and returns "
        "combined STDOUT/STDERR. Container sessions may use any non-secret "
        "working directory inside the container."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Optional working directory. Workspace sessions require "
                    "an attached path; container sessions may use any non-secret "
                    "directory inside the container."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": (
                    "Maximum seconds before terminating the command "
                    "(default 120)."
                ),
                "default": 120,
            },
            "max_output_chars": {
                "type": "integer",
                "description": (
                    "Maximum combined output length to return (default 12000)."
                ),
                "default": 12000,
            },
        },
        "required": ["command"],
    },
    requires_approval=True,
    execution_kind="mutate",
    preview_policy="optional",
)
def _bash_tool(args: Dict[str, Any], context) -> str:
    return bash_command(
        args.get("command", ""),
        context.folder_context,
        cwd=args.get("cwd"),
        timeout_seconds=args.get("timeout_seconds", 120),
        max_output_chars=args.get("max_output_chars", 12000),
        session_type=session_type_from_context(context),
    )


# ---------------------------------------------------------------- background bash


@tool(
    name="bash_background",
    description=(
        "Start a long-running bash command in the background and return a "
        "task_id you can poll with `bash_status` or read with `bash_logs`. "
        "Use this for test watchers, dev servers, builds, or anything that "
        "would block the synchronous `bash` tool for too long."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run in the background.",
            },
            "name": {
                "type": "string",
                "description": (
                    "Optional short label for the task (shown in /stats and logs)."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory. Must exist.",
            },
        },
        "required": ["command"],
    },
    requires_approval=True,
    execution_kind="mutate",
    preview_policy="optional",
)
def bash_background(args: Dict[str, Any], context) -> str:
    from mu.tools.shell.background import summarize_task

    registry = _bg_registry(context)
    command = str(args.get("command", "") or "").strip()
    if not command:
        return json.dumps({"error": "command is required"})
    session_type = session_type_from_context(context)
    raw_cwd = str(args.get("cwd", "") or "").strip()
    cwd = raw_cwd or _default_working_directory(context.folder_context, session_type)
    if not os.path.isdir(os.path.expanduser(cwd)):
        return json.dumps({"error": f"Working directory does not exist: {cwd}"})
    if not _check_bounds(cwd, context.folder_context, session_type=session_type):
        return json.dumps({"error": f"Access denied for working directory: {cwd}"})
    try:
        task = registry.start(
            command,
            name=str(args.get("name", "") or "") or None,
            cwd=cwd,
        )
    except (ValueError, RuntimeError) as e:
        return json.dumps({"error": str(e)})
    summary = summarize_task(task, tail_lines=0)
    summary["message"] = f"Background task started: {task.task_id}"
    return json.dumps(summary, indent=2)


@tool(
    name="bash_status",
    description=(
        "Poll a background task's status (running/completed/failed/killed) "
        "and the tail of its stdout/stderr."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by `bash_background`.",
            },
            "tail_lines": {
                "type": "integer",
                "description": "Lines of stdout/stderr to include (default 20).",
                "default": 20,
            },
        },
        "required": ["task_id"],
    },
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
)
def bash_status(args: Dict[str, Any], context) -> str:
    from mu.tools.shell.background import summarize_task

    registry = _bg_registry(context)
    task_id = str(args.get("task_id", "") or "").strip()
    task = registry.get(task_id)
    if task is None:
        return json.dumps({"error": f"no such task: {task_id}"})
    tail_lines = max(0, int(args.get("tail_lines", 20) or 20))
    return json.dumps(_scrub_json_payload(summarize_task(task, tail_lines=tail_lines)), indent=2)


@tool(
    name="bash_logs",
    description=(
        "Read the tail of stdout / stderr from a background task. "
        "Stream selector: 'stdout', 'stderr', or 'both'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by `bash_background`.",
            },
            "stream": {
                "type": "string",
                "description": (
                    "Which stream to read: 'stdout', 'stderr', or 'both'."
                ),
                "default": "both",
            },
            "lines": {
                "type": "integer",
                "description": "Number of trailing lines to return (default 200).",
                "default": 200,
            },
        },
        "required": ["task_id"],
    },
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
)
def bash_logs(args: Dict[str, Any], context) -> str:
    from mu.tools.shell.background import tail as _tail

    registry = _bg_registry(context)
    task_id = str(args.get("task_id", "") or "").strip()
    task = registry.get(task_id)
    if task is None:
        return json.dumps({"error": f"no such task: {task_id}"})
    stream = str(args.get("stream", "both") or "both").lower()
    lines = max(1, int(args.get("lines", 200) or 200))
    payload: dict = {"task_id": task_id, "status": task.status()}
    if stream in ("stdout", "both"):
        payload["stdout"] = _tail(task.stdout_buf, lines)
    if stream in ("stderr", "both"):
        payload["stderr"] = _tail(task.stderr_buf, lines)
    return json.dumps(_scrub_json_payload(payload), indent=2)


@tool(
    name="bash_kill",
    description=(
        "Terminate a background task. SIGTERM with a short grace, then SIGKILL."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id to kill.",
            }
        },
        "required": ["task_id"],
    },
    requires_approval=True,
    execution_kind="mutate",
    preview_policy="optional",
)
def bash_kill(args: Dict[str, Any], context) -> str:
    from mu.tools.shell.background import summarize_task

    registry = _bg_registry(context)
    task_id = str(args.get("task_id", "") or "").strip()
    task = registry.kill(task_id)
    if task is None:
        return json.dumps({"error": f"no such task: {task_id}"})
    return json.dumps(_scrub_json_payload(summarize_task(task, tail_lines=5)), indent=2)


@tool(
    name="bash_list",
    description="List every background task in the session — running or completed.",
    parameters={"type": "object", "properties": {}},
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
)
def bash_list(args: Dict[str, Any], context) -> str:
    from mu.tools.shell.background import summarize_task

    registry = _bg_registry(context)
    tasks = [summarize_task(t, tail_lines=3) for t in registry.list()]
    return json.dumps(_scrub_json_payload({"tasks": tasks, "count": len(tasks)}), indent=2)
