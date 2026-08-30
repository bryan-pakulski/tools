"""Host-to-worker transport for container-backed Mode OS APIs."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from mu.tools.capabilities import normalize_session_type


MODE_API_PREFIXES = (
    "/api/feature",
    "/api/research",
    "/api/security",
    "/api/debug",
    "/api/loop",
    "/api/teacher",
    "/api/memory",
)


def _is_mode_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in MODE_API_PREFIXES)


async def proxy_container_mode_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Route mode reads and controls to the process that owns their state.

    Workspace sessions continue through the host routers unchanged. Container
    sessions execute those same routers inside the worker, where `/workspace`
    reports, course files, security proof artifacts, and live in-memory state
    are actually available.
    """
    if getattr(request.app.state, "is_container_worker", False):
        return await call_next(request)
    path = request.url.path
    if not _is_mode_path(path):
        return await call_next(request)

    requested_name = str(request.query_params.get("session_name") or "").strip()
    session = request.app.state.session_by_name(requested_name or None)
    if session is None or normalize_session_type(
        session.variables.get("session_type")
    ) != "container":
        return await call_next(request)

    session_name = str(session.session_manager.current_session_name or "").strip()
    query = [
        (key, value)
        for key, value in request.query_params.multi_items()
        if key != "session_name"
    ]
    query.append(("session_name", session_name))
    # Round-27 F3: enforce the body limit BEFORE materializing it —
    # await request.body() buffers the whole payload in memory, so a
    # huge loopback request would be fully read only to be rejected by
    # the supervisor's 2 MiB check afterwards.
    _MAX_PROXY_BODY = 2 * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > _MAX_PROXY_BODY:
                return JSONResponse(status_code=413, content={"detail": "payload too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "invalid content-length"})
    body = await request.body()
    if len(body) > _MAX_PROXY_BODY:
        return JSONResponse(status_code=413, content={"detail": "payload too large"})
    provider = getattr(session, "provider", None)
    try:
        result = await asyncio.to_thread(
            request.app.state.container_supervisor.proxy_mode_api,
            session_name,
            path,
            method=request.method,
            query=query,
            body=body,
            content_type=request.headers.get("content-type", "application/json"),
            provider=str(getattr(provider, "name", "") or ""),
            model=str(getattr(provider, "model_name", "") or ""),
            agent_mode=str(session.variables.get("agent_mode", "default") or "default"),
            system_instruction=str(
                getattr(session, "system_instruction", "You are a helpful assistant.")
            ),
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"container mode workspace unavailable: {exc}",
                "session_name": session_name,
            },
        )

    response = Response(
        content=result.get("content", b""),
        status_code=int(result.get("status_code") or 500),
    )
    response.headers["Content-Type"] = str(
        result.get("content_type") or "application/json"
    )
    response.headers["X-MuCLI-Execution-Boundary"] = "container"
    if request.method != "GET" and response.status_code < 400:
        # The worker writes through the mounted session directory. Refresh the
        # host mirror after a successful control action so a later host-side
        # `/mode`, settings save, or unload cannot overwrite the newer feature,
        # loop, security, or teacher state with a stale snapshot.
        try:
            session.session_manager._load_session(session_name)
            session.sync_runtime_state()
        except Exception:
            pass
    return response


__all__ = ["MODE_API_PREFIXES", "proxy_container_mode_request"]
