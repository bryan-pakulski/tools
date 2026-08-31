"""TUI thread picker and durable coordination audit command."""

from __future__ import annotations

import shlex
from typing import Any

from mu.threads.coordinator import ThreadCoordinator
from mu.threads.model import ensure_thread_meta

from . import CommandResult, command


def _coordinator(session):
    session.sync_runtime_state()
    coordinator = getattr(session, "thread_coordinator", None)
    if coordinator is None:
        saved_meta = getattr(session.session_manager, "thread_meta", None)
        meta = ensure_thread_meta(
            session.session_manager.current_session_name,
            saved_meta.to_dict() if hasattr(saved_meta, "to_dict") else None,
        )
        coordinator = ThreadCoordinator(meta.group_id)
        coordinator.register_thread(meta, session.session_manager.current_session_name)
    return coordinator


def _render_roster(session, coordinator) -> list[dict]:
    roster = coordinator.list_threads()
    ui = getattr(session, "ui", None)
    if ui is not None:
        lines = ["Threads in this workspace/chat/container:"]
        current = session.thread_meta.thread_id
        for index, item in enumerate(roster, 1):
            marker = "*" if item["thread_id"] == current else " "
            unread = f" · {item['unread_count']} unread" if item.get("unread_count") else ""
            claims = (
                f" · owns {', '.join(item.get('claimed_paths') or [])}"
                if item.get("claimed_paths") else ""
            )
            lines.append(
                f" {marker} {index}. {item['title']} [{item['status']}] "
                f"({item['session_name']}){unread}{claims}"
            )
        ui.show_info("\n".join(lines))
    return roster


def _switch(session, name: str, allow_prompt: bool) -> CommandResult:
    from .session import _load_session

    return _load_session(session, name, allow_prompt)


def _delete(session, name: str, allow_prompt: bool) -> CommandResult:
    """Delete one thread: its session dir plus its coordination rows."""
    name = str(name or "").strip()
    coordinator = _coordinator(session)
    roster = {item["session_name"]: item for item in coordinator.list_threads()}
    current_name = session.session_manager.current_session_name
    if not name and allow_prompt and getattr(session, "ui", None) is not None:
        others = [
            item
            for item in coordinator.list_threads()
            if item["session_name"] != current_name
        ]
        if not others:
            return CommandResult(ok=False, message="No other thread to delete.")
        choices = [
            str(index) for index in range(1, len(others) + 1)
        ] + ["q"]
        choice = session.ui.prompt_choices(
            "Delete which thread? (number or [q]uit)", choices=choices, default="q"
        )
        if not choice or not str(choice).isdigit():
            return CommandResult(ok=True, message="Delete cancelled.")
        item = others[int(choice) - 1]
        name = item["session_name"]
    if not name:
        return CommandResult(ok=False, message="Usage: /thread delete <session>")
    if name == current_name:
        return CommandResult(
            ok=False, message="Cannot delete the thread you are on — switch first."
        )
    item = roster.get(name)
    if item is None:
        return CommandResult(ok=False, message=f"No thread named '{name}' here.")
    ui = getattr(session, "ui", None)
    if allow_prompt and ui is not None:
        answer = session.ui.prompt_choices(
            f"Permanently delete thread '{name}' and its history?",
            choices=["y", "n"],
            default="n",
        )
        if str(answer).lower() not in {"y", "yes"}:
            return CommandResult(ok=True, message="Delete cancelled.")
    try:
        coordinator.delete_thread(item["thread_id"], force=True)
    except Exception as exc:  # ThreadCoordinatorError or sqlite issues
        return CommandResult(ok=False, message=f"Delete failed: {exc}")
    session.session_manager.delete_session(name)
    return CommandResult(
        ok=True,
        message=f"Deleted thread '{name}'.",
        data={"deleted_thread_id": item["thread_id"]},
    )


def _new(session, title: str, allow_prompt: bool) -> CommandResult:
    title = str(title or "").strip()
    if not title and allow_prompt and getattr(session, "ui", None) is not None:
        title = str(session.ui.prompt("Thread title", default="New thread") or "").strip()
    title = title or "New thread"
    result = session.session_manager.create_thread(title=title)
    loaded = _switch(session, result["session_name"], allow_prompt)
    loaded.data.update(result)
    return loaded


def _activity(session, coordinator) -> CommandResult:
    events = coordinator.activity(limit=100)
    ui = getattr(session, "ui", None)
    if ui is not None:
        lines = ["Coordination audit (oldest → newest):"]
        for event in events:
            actor = event.get("actor_title") or event.get("actor_thread_id") or "system"
            target = event.get("target_title") or event.get("target_thread_id") or "group"
            content = str((event.get("payload") or {}).get("content") or "")
            suffix = f" · {content}" if content else ""
            lines.append(f" #{event['event_id']} {actor} → {target}: {event['kind']}{suffix}")
        ui.show_info("\n".join(lines))
    return CommandResult(ok=True, message=f"{len(events)} coordination event(s).", data={"events": events})


@command(
    "/thread",
    help="Open the thread picker, or use list, new [title], switch <session>, delete <session>, activity.",
)
def thread_cmd(session: Any, args: str, *, allow_prompt: bool = True) -> CommandResult:
    parts = shlex.split(args or "")
    coordinator = _coordinator(session)
    if parts:
        sub = parts[0].lower()
        rest = " ".join(parts[1:])
        if sub == "list":
            roster = _render_roster(session, coordinator)
            return CommandResult(ok=True, message=f"{len(roster)} thread(s).", data={"threads": roster})
        if sub == "new":
            return _new(session, rest, allow_prompt)
        if sub in {"switch", "load"}:
            if not rest:
                return CommandResult(ok=False, message="Usage: /thread switch <session>")
            return _switch(session, rest, allow_prompt)
        if sub in {"delete", "rm"}:
            return _delete(session, rest, allow_prompt)
        if sub in {"activity", "audit"}:
            return _activity(session, coordinator)
        return CommandResult(ok=False, message="Usage: /thread [list|new [title]|switch <session>|delete <session>|activity]")

    roster = _render_roster(session, coordinator)
    if not allow_prompt or getattr(session, "ui", None) is None:
        return CommandResult(ok=True, message=f"{len(roster)} thread(s).", data={"threads": roster})
    choices = [str(index) for index in range(1, len(roster) + 1)] + ["n", "a", "d", "q"]
    choice = session.ui.prompt_choices(
        "Select a thread number, [n]ew, [a]ctivity, [d]elete, or [q]uit",
        choices=choices,
        default="q",
    )
    if choice == "n":
        return _new(session, "", allow_prompt)
    if choice == "a":
        return _activity(session, coordinator)
    if choice == "d":
        return _delete(session, "", allow_prompt)
    if choice and str(choice).isdigit():
        item = roster[int(choice) - 1]
        if item["thread_id"] == session.thread_meta.thread_id:
            return CommandResult(ok=True, message="Already on that thread.")
        return _switch(session, item["session_name"], allow_prompt)
    return CommandResult(ok=True, message="Thread picker closed.")
