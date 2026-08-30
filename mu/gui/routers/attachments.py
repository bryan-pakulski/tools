
"""Upload/list/download/delete endpoints for session attachments."""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from mu.attachment import AttachmentError, AttachmentRegistry
from utils.config import HISTORY_DIR

router = APIRouter()

# Aggregate quotas (codex round-9 F7): the per-file 512 MiB cap alone does
# not stop disk exhaustion — a client can loop small uploads forever. Track
# stored bytes per session AND globally; reject uploads that would breach a
# quota, including the spooled bytes of in-flight uploads. The lock makes
# the check-and-reserve atomic against concurrent uploads.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_SESSION_ATTACHMENT_BYTES = 1024 * 1024 * 1024      # 1 GiB per session
MAX_GLOBAL_ATTACHMENT_BYTES = 8 * 1024 * 1024 * 1024   # 8 GiB across sessions

_quota_lock = __import__("threading").Lock()
_inflight: dict[str, int] = {}   # session_name -> reserved spooled bytes


def _stored_attachment_bytes(registry) -> int:
    """Disk-truth byte count of a session's attachment store. Registry
    metadata can lag direct writes, so scan the directory (same shape as
    _global_stored_bytes)."""
    return _dir_attachment_bytes(registry.attachments_dir)


def _dir_attachment_bytes(directory: str) -> int:
    total = 0
    try:
        for entry in os.scandir(directory):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _global_stored_bytes() -> int:
    sessions_root = os.path.join(HISTORY_DIR, "sessions")
    total = 0
    try:
        for session_name in os.listdir(sessions_root):
            attachments_dir = os.path.join(sessions_root, session_name, "attachments")
            if not os.path.isdir(attachments_dir):
                continue
            total += _dir_attachment_bytes(attachments_dir)
    except OSError:
        pass
    return total


def _check_quota(session_name: str, incoming: int) -> None:
    """Refuse the upload if it would push the session or the global store
    over quota. ``incoming`` is the caller's best-known size (Content-Length
    when honest); the streaming loop enforces the per-file cap and the
    reserved bytes are released in ``finally`` either way."""
    with _quota_lock:
        session_used = _inflight.get(session_name, 0)
        global_used = sum(_inflight.values())
    if incoming and incoming > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Attachment exceeds the upload limit.")


def _registry(session_name: str) -> AttachmentRegistry:
    safe = str(session_name or "").strip()
    if not safe or os.path.basename(safe) != safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid session name")
    session_dir = os.path.join(HISTORY_DIR, "sessions", safe)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail="session not found")
    return AttachmentRegistry(session_dir)


@router.get("/{name}/attachments")
async def list_attachments(name: str, response: Response, limit: int = 100):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    # Round-44 F9: bounded list — newest `limit` descriptors (0 = all),
    # mirroring the artifacts router.
    registry = _registry(name)
    if limit and limit > 0:
        return {"attachments": registry.list(limit=limit), "total": len(registry._read())}
    return {"attachments": registry.list()}


@router.post("/{name}/attachments")
async def upload_attachment(name: str, request: Request, file: UploadFile = File(...)):
    registry = _registry(name)
    suffix = os.path.splitext(file.filename or "attachment")[1][:20]
    # Streaming byte cap (codex round-6 F7): uploads used to write until
    # EOF with no limit — a LAN client could exhaust host disk. 512 MiB
    # per file is far above any sane attachment; 413 beyond that.
    # Aggregate quotas (codex round-9 F7): reject early on Content-Length,
    # then reserve under the lock so concurrent uploads share one budget.
    content_length = 0
    try:
        content_length = int(request.headers.get("content-length") or 0)
    except ValueError:
        content_length = 0
    _check_quota(name, content_length)
    session_used = _stored_attachment_bytes(registry)
    with _quota_lock:
        if session_used + max(content_length, 0) > MAX_SESSION_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Session attachment storage quota (1 GiB) exceeded.",
            )
        if _global_stored_bytes() + sum(_inflight.values()) + max(content_length, 0) > MAX_GLOBAL_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Global attachment storage quota (8 GiB) exceeded.",
            )
        _inflight[name] = _inflight.get(name, 0) + max(content_length, 0)
    # MUCLI_UNBOUNDED_ATTACHMENT_UPLOADS_V1: spool beside the registry so the completed upload can
    # be atomically moved into place without retaining a second full-size copy.
    fd, temp_path = tempfile.mkstemp(
        prefix=".mucli-upload-",
        suffix=suffix,
        dir=registry.attachments_dir,
    )
    try:
        total = 0
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Attachment exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB upload limit.",
                    )
                handle.write(chunk)
        descriptor = registry.add(
            name=file.filename or "attachment",
            source_path=temp_path,
            mime_type=file.content_type or "application/octet-stream",
            move_source=True,
        )
    except AttachmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        with _quota_lock:
            _inflight[name] = max(0, _inflight.get(name, 0) - max(content_length, 0))
            if _inflight.get(name) == 0:
                _inflight.pop(name, None)
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    await request.app.state.bus.publish({
        "kind": "attachment_created",
        "attachment": descriptor,
        "session_name": name,
    })
    return {"ok": True, "attachment": descriptor}


@router.get("/{name}/attachments/{attachment_id}/download")
async def download_attachment(name: str, attachment_id: str):
    registry = _registry(name)
    descriptor = registry.get(attachment_id)
    path = registry.resolve_path(attachment_id)
    if descriptor is None or path is None:
        raise HTTPException(status_code=404, detail="attachment not found")
    return FileResponse(
        path,
        media_type=descriptor.get("mime_type") or "application/octet-stream",
        filename=descriptor.get("name") or "attachment",
    )


@router.delete("/{name}/attachments/{attachment_id}")
async def delete_attachment(name: str, attachment_id: str, request: Request):
    if request.app.state.session_busy_for(name).is_set():
        raise HTTPException(status_code=409, detail="cannot delete attachments during an active turn")
    if not _registry(name).remove(attachment_id):
        raise HTTPException(status_code=404, detail="attachment not found")
    await request.app.state.bus.publish({
        "kind": "attachment_deleted",
        "attachment_id": attachment_id,
        "session_name": name,
    })
    return {"ok": True, "attachment_id": attachment_id}
