"""Slash command for workspace management.

Replaces the older `/folder`, `/file`, `/clearfiles`, and `/workspace`
commands under one grouped surface:

    /workspace                        — show attached folders + staged files
    /workspace clear                  — drop all folders AND staged files
    /workspace folder <path...>       — attach one or more folders
    /workspace folder remove <path>   — detach a folder
    /workspace folder clear           — drop all folders
    /workspace file [<path>]          — list staged files, or stage <path>
    /workspace file clear             — drop all staged files
"""

from __future__ import annotations

import os
import shlex
from typing import Any, List

from . import CommandResult, command


def _emit(session: Any, body: str, allow_prompt: bool) -> None:
    ui = getattr(session, "ui", None)
    if ui is not None and hasattr(ui, "show_info") and allow_prompt:
        ui.show_info(body)


def _emit_error(session: Any, body: str, allow_prompt: bool) -> None:
    ui = getattr(session, "ui", None)
    if ui is not None and hasattr(ui, "show_error") and allow_prompt:
        ui.show_error(body)


def _refresh_hud(session: Any) -> None:
    """Best-effort: refresh the memory HUD if the session exposes the helper."""
    try:
        from mucli import refresh_memory_hud

        refresh_memory_hud(session, getattr(session, "ui", None))
    except ImportError:
        pass


def _show(session: Any, allow_prompt: bool) -> CommandResult:
    folders = list(session.folder_context.folders)
    staged = list(session.staged_files)

    if allow_prompt:
        ui = getattr(session, "ui", None)
        console = getattr(ui, "console", None) if ui is not None else None
        if console is not None:
            console.print("\n[bold cyan]Workspace folders:[/bold cyan]")
            if folders:
                console.print(session.folder_context.get_tree_map(), markup=False)
            else:
                console.print("[dim](none)[/dim]")
            console.print("\n[bold cyan]Staged files:[/bold cyan]")
            if staged:
                for entry in staged:
                    if isinstance(entry, dict):
                        console.print(
                            f"  • {entry.get('path') or entry.get('name') or entry}",
                            markup=False,
                        )
                    else:
                        console.print(f"  • {entry}", markup=False)
            else:
                console.print("[dim](none)[/dim]")

    return CommandResult(
        ok=True,
        message=f"{len(folders)} folder(s), {len(staged)} staged file(s).",
        data={"folders": folders, "staged_files": staged},
    )


def _split_paths(raw: str) -> List[str]:
    try:
        paths = shlex.split(raw)
    except ValueError:
        paths = [raw.strip("'\"")]
    return [p.strip("'\"") for p in paths if p]


def _add_folders(session: Any, raw: str, allow_prompt: bool) -> CommandResult:
    paths = _split_paths(raw)
    added: List[str] = []
    invalid: List[str] = []
    for path in paths:
        if session.folder_context.add_folder(path):
            added.append(path)
            _emit(session, f"Added folder: {path}", allow_prompt)
            # No os.chdir here either: the process cwd is global and the GUI
            # runs sessions concurrently in one process. Tool handlers pick
            # the workspace root per-call via default_working_directory().
        else:
            invalid.append(path)
            _emit_error(session, f"Folder not found or invalid: {path}", allow_prompt)
    session.session_manager.save_history(session.folder_context)
    if added:
        _emit(
            session,
            "Files cached as initial context. Changes will be provided as diffs.",
            allow_prompt,
        )
    _refresh_hud(session)
    return CommandResult(
        ok=not invalid,
        message="Workspace folders updated.",
        data={"added": added, "invalid": invalid},
    )


def _remove_folder(session: Any, path: str, allow_prompt: bool) -> CommandResult:
    target = path.strip().strip("'\"")
    if not target:
        msg = "Usage: /workspace folder remove <path>"
        _emit_error(session, msg, allow_prompt)
        return CommandResult(ok=False, message=msg)
    if session.folder_context.remove_folder(target):
        session.session_manager.save_history(session.folder_context)
        _refresh_hud(session)
        msg = f"Removed folder from context: {target}"
        _emit(session, msg, allow_prompt)
        return CommandResult(ok=True, message=msg)
    msg = f"Folder not found in context: {target}"
    _emit_error(session, msg, allow_prompt)
    return CommandResult(ok=False, message=msg)


def _clear_folders(session: Any, allow_prompt: bool) -> CommandResult:
    session.folder_context.folders.clear()
    session.folder_context.workspace_file_tree = None
    session.session_manager.save_history(session.folder_context)
    _refresh_hud(session)
    msg = "Workspace folders cleared."
    _emit(session, msg, allow_prompt)
    return CommandResult(ok=True, message=msg)


def _folder_cmd(session: Any, rest: str, allow_prompt: bool) -> CommandResult:
    raw = rest.strip()
    if not raw:
        msg = "Usage: /workspace folder <path> | remove <path> | clear"
        _emit_error(session, msg, allow_prompt)
        return CommandResult(ok=False, message=msg)
    head, _, tail = raw.partition(" ")
    head_lower = head.lower()
    if head_lower == "clear":
        return _clear_folders(session, allow_prompt)
    if head_lower == "remove":
        return _remove_folder(session, tail, allow_prompt)
    return _add_folders(session, raw, allow_prompt)


def _list_staged(session: Any, allow_prompt: bool) -> CommandResult:
    staged = list(session.staged_files)
    if not staged:
        msg = "No files staged. Usage: /workspace file <path> | clear"
    else:
        lines = ["Staged files:"]
        for entry in staged:
            if isinstance(entry, dict):
                lines.append(f"  • {entry.get('path') or entry.get('name') or entry}")
            else:
                lines.append(f"  • {entry}")
        msg = "\n".join(lines)
    _emit(session, msg, allow_prompt)
    return CommandResult(ok=True, message=msg, data={"staged_files": staged})


def _stage_file(session: Any, path: str, allow_prompt: bool) -> CommandResult:
    target = path.strip().strip("'\"")
    if not target:
        msg = "Usage: /workspace file <path>"
        _emit_error(session, msg, allow_prompt)
        return CommandResult(ok=False, message=msg)
    session.add_file(target)
    msg = f"Staged file: {target}"
    _emit(session, msg, allow_prompt)
    return CommandResult(
        ok=True, message=msg, data={"staged_files": list(session.staged_files)}
    )


def _clear_staged(session: Any, allow_prompt: bool) -> CommandResult:
    session.clear_files()
    msg = "Staged files cleared."
    _emit(session, msg, allow_prompt)
    return CommandResult(
        ok=True, message=msg, data={"staged_files": list(session.staged_files)}
    )


def _file_cmd(session: Any, rest: str, allow_prompt: bool) -> CommandResult:
    raw = rest.strip()
    if not raw:
        return _list_staged(session, allow_prompt)
    head, _, _tail = raw.partition(" ")
    if head.lower() == "clear":
        return _clear_staged(session, allow_prompt)
    return _stage_file(session, raw, allow_prompt)


def _clear_everything(session: Any, allow_prompt: bool) -> CommandResult:
    _clear_folders(session, allow_prompt)
    _clear_staged(session, allow_prompt)
    msg = "Workspace folders and staged files cleared."
    return CommandResult(ok=True, message=msg)


@command(
    "/workspace",
    help=(
        "Workspace: show, clear; folder <add|remove|clear>; file <stage|clear>."
    ),
)
def workspace_cmd(session: Any, args: str, *, allow_prompt: bool = True) -> CommandResult:
    raw = (args or "").strip()
    if not raw:
        return _show(session, allow_prompt)

    head, _, rest = raw.partition(" ")
    sub = head.lower()
    rest = rest.strip()

    if sub == "clear":
        return _clear_everything(session, allow_prompt)
    if sub == "folder":
        return _folder_cmd(session, rest, allow_prompt)
    if sub == "file":
        return _file_cmd(session, rest, allow_prompt)

    return CommandResult(
        ok=False,
        message=(
            f"Unknown subcommand {sub!r}. "
            "Usage: /workspace [clear|folder <args>|file <args>]"
        ),
    )
