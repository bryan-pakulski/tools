"""Thread-group roster, creation, deletion, and coordination-audit endpoints."""

from __future__ import annotations

import fcntl
import glob
import os
import shutil
from functools import partial
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

import utils.config as _config
from mu.gui.async_utils import run_sync_responsive
from mu.session.manager import SessionManager
from mu.threads.coordinator import ThreadCoordinator, ThreadCoordinatorError
from mu.threads.model import ensure_thread_meta


router = APIRouter()


def _session_data(name: str) -> dict[str, Any]:
    try:
        clean = SessionManager._validate_session_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = SessionManager().read_session_data(clean)
    if not isinstance(data, dict):
        raise HTTPException(status_code=404, detail=f"Session '{clean}' not found.")
    return data


def _coordinator_for(name: str) -> tuple[ThreadCoordinator, Any]:
    data = _session_data(name)
    meta = ensure_thread_meta(name, data.get("thread_meta"))
    coordinator = ThreadCoordinator(meta.group_id)
    sessions_pattern = os.path.join(
        _config.HISTORY_DIR, "sessions", "*", "session.json"
    )
    for path in glob.glob(sessions_pattern):
        peer_name = os.path.basename(os.path.dirname(path))
        try:
            peer_data = SessionManager().read_session_data(peer_name)
            if not isinstance(peer_data, dict):
                continue
            peer_meta = ensure_thread_meta(peer_name, peer_data.get("thread_meta"))
            if peer_meta.group_id == meta.group_id:
                coordinator.register_thread(peer_meta, peer_name)
        except Exception:
            continue
    coordinator.register_thread(meta, name)
    _prune_orphan_threads(coordinator)
    return coordinator, meta


def _session_dir_exists(session_name: str) -> bool:
    try:
        clean = SessionManager._validate_session_name(session_name)
    except ValueError:
        return False
    return os.path.isdir(
        os.path.join(str(_config.HISTORY_DIR), "sessions", clean)
    )


def _prune_orphan_threads(coordinator: ThreadCoordinator) -> list[str]:
    """Drop coordination rows whose session dir is gone (lazy orphan prune).

    A thread's session dir can disappear outside the threads API (sessions
    list delete on web/mobile). Those rows would otherwise surface as ghost
    threads forever. Pruning is lazy: only rows for the group currently being
    listed are removed, only when the session no longer exists on disk, and
    the current/last remaining thread is never pruned.
    """
    pruned: list[str] = []
    try:
        roster = coordinator.list_threads()
    except Exception:
        return pruned
    for item in roster:
        session_name = str(item.get("session_name") or "")
        if not session_name or _session_dir_exists(session_name):
            continue
        # Keep the last remaining row: the group must always resolve.
        if len(roster) - len(pruned) <= 1:
            break
        try:
            coordinator.delete_thread(
                str(item.get("thread_id") or ""), force=True
            )
            pruned.append(session_name)
        except ThreadCoordinatorError:
            continue
    return pruned


# Retained name for the delete endpoint helper below.


@router.get("")
async def list_threads(request: Request, session_name: str = Query(default="")):
    name = session_name.strip() or request.app.state.current_session_name
    if not name:
        raise HTTPException(status_code=412, detail="No session selected.")
    coordinator, meta = await run_sync_responsive(_coordinator_for, name)
    return {
        "session_name": name,
        "thread_group_id": meta.group_id,
        "current_thread_id": meta.thread_id,
        "threads": coordinator.list_threads(),
    }


@router.get("/activity")
async def thread_activity(
    request: Request,
    session_name: str = Query(default=""),
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
):
    name = session_name.strip() or request.app.state.current_session_name
    if not name:
        raise HTTPException(status_code=412, detail="No session selected.")
    coordinator, meta = await run_sync_responsive(_coordinator_for, name)
    events = await run_sync_responsive(
        partial(coordinator.activity, after_id=after_id, limit=limit)
    )
    return {
        "thread_group_id": meta.group_id,
        "events": events,
        "last_event_id": events[-1]["event_id"] if events else after_id,
    }


@router.post("")
async def create_thread(request: Request, payload: dict[str, Any]):
    parent_name = str(
        payload.get("parent_session_name")
        or payload.get("session_name")
        or request.app.state.current_session_name
        or ""
    ).strip()
    title = str(payload.get("title") or "New thread").strip()
    requested_name = str(payload.get("name") or "").strip() or None
    activate = bool(payload.get("activate", True))
    if not parent_name:
        raise HTTPException(status_code=412, detail="No parent session selected.")
    try:
        manager = SessionManager(session_name=parent_name)
        result = await run_sync_responsive(
            partial(
                manager.create_thread,
                title=title,
                session_name=requested_name,
                parent_session_name=parent_name,
            )
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    child_name = result["session_name"]
    parent_data = _session_data(parent_name)
    session_type = str(
        (parent_data.get("variables") or {}).get("session_type") or "workspace"
    ).lower()
    if session_type == "container":
        container_name = str(
            (parent_data.get("container_config") or {}).get("container_name") or ""
        )
        if container_name:
            try:
                await run_sync_responsive(
                    partial(
                        request.app.state.container_supervisor.attach_session,
                        container_name,
                        child_name,
                        supervisor_url=(
                            f"http://host.docker.internal:{request.app.state.port}"
                        ),
                    )
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Thread saved but container attachment failed: {exc}",
                ) from exc
    if activate:
        provider = str((parent_data.get("provider_config") or {}).get("provider") or "")
        model = str((parent_data.get("provider_config") or {}).get("model") or "")
        await run_sync_responsive(
            partial(
                request.app.state.load_session,
                name=child_name,
                provider=provider or None,
                model=model or None,
            )
        )
    await request.app.state.bus.publish(
        {
            "kind": "thread_created",
            "session_name": parent_name,
            "thread_group_id": result["thread_meta"]["group_id"],
            "thread": {**result["thread_meta"], "session_name": child_name},
        }
    )
    return {"ok": True, "active": activate, **result}


@router.delete("/{thread_id}")
async def delete_thread(request: Request, thread_id: str, session_name: str = Query(default="")):
    """Delete a thread: coordination rows + the underlying thread session dir.

    The group always keeps at least one thread, so deleting a singleton is
    rejected with 409.  Deleting the currently-loaded thread requires the
    client to switch away first (400).
    """
    name = session_name.strip() or request.app.state.current_session_name or ""
    state = request.app.state

    coordinator, meta = await run_sync_responsive(_coordinator_for, name) if name else (None, None)
    if coordinator is None:
        raise HTTPException(status_code=412, detail="No session selected.")

    target = coordinator.get_thread(thread_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found in this group.")
    target_name = str(target.get("session_name") or "")
    if not target_name:
        raise HTTPException(status_code=500, detail="Thread has no session name.")

    if target_name in state.sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Thread session {target_name!r} is loaded — switch away from it first.",
        )
    if state.session_busy_for(target_name).is_set():
        raise HTTPException(
            status_code=409,
            detail=f"Thread {target_name!r} is running — wait for its turn to finish.",
        )
    if target_name == state.current_session_name:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the thread you are currently viewing.",
        )

    session_dir = os.path.join(_config.HISTORY_DIR, "sessions", target_name)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail=f"Session '{target_name}' not found.")

    def _delete() -> None:
        from mu.session.manager import SessionManager as _SM

        lock_path = _SM.session_lock_path(target_name)
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                state.container_supervisor.detach(target_name, stop_if_idle=True)
                shutil.rmtree(session_dir)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    try:
        await run_sync_responsive(_delete)
    except ThreadCoordinatorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete thread session: {exc}") from exc

    try:
        await run_sync_responsive(partial(coordinator.delete_thread, thread_id))
    except ThreadCoordinatorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await state.bus.publish(
        {
            "kind": "thread_deleted",
            "session_name": name,
            "thread_group_id": meta.group_id,
            "thread_id": thread_id,
            "deleted_session_name": target_name,
        }
    )
    return {
        "ok": True,
        "thread_id": thread_id,
        "deleted_session_name": target_name,
    }
