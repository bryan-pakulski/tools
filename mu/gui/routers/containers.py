"""Container worker callback and lifecycle endpoints."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import tempfile
import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from mu.artifact import ArtifactRegistry
from mu.container.builder import default_dockerfile
from mu.container.docker_cli import ContainerRuntimeError
from mu.container.network import DEFAULT_EGRESS_ALLOW
from mu.container.stats import ContainerStatsCollector  # MUCLI_CONTAINER_MONITOR_V1
from mu.container.shell_qol import CWD_MARKER_PREFIX, CWD_MARKER_SUFFIX, CwdMarkerFilter
import utils.config as _config

router = APIRouter()


def _require_local_client(connection, *, allow_private_network: bool = True) -> None:
    """Restrict host-container controls to loopback or an explicitly exposed LAN.

    The mobile application reaches the GUI over a private address, so lifecycle
    APIs may be used from RFC1918/ULA clients when the operator has bound MuCLI
    to the LAN. Interactive shell access remains loopback-only.
    """
    client = getattr(connection, "client", None)
    host = str(getattr(client, "host", "") or "")
    if not host or host in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return
    if allow_private_network:
        try:
            if ipaddress.ip_address(host).is_private:
                return
        except ValueError:
            pass
    scope = "localhost" if not allow_private_network else "localhost or a private network"
    raise HTTPException(
        status_code=403,
        detail=f"Container management access is restricted to {scope}.",
    )


def _environment_jobs(request: Request) -> dict[str, dict[str, Any]]:
    jobs = getattr(request.app.state, "container_environment_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.container_environment_jobs = jobs
    return jobs


def _set_environment_job(
    request: Request, job_id: str, *, state: str, stage: str, message: str,
    detail: str | None = None, container: dict[str, Any] | None = None,
) -> None:
    jobs = _environment_jobs(request)
    current = jobs.setdefault(job_id, {"logs": [], "next_log_seq": 1})
    current.update({
        "job_id": job_id, "state": state, "stage": stage,
        "message": message, "updated_at": time.time(),
    })
    if detail is not None:
        current["detail"] = detail
    if container is not None:
        current["container"] = container


def _append_environment_log(request: Request, job_id: str, stream: str, text: str) -> None:
    value = str(text or "").rstrip("\r\n")
    if not value:
        return
    jobs = _environment_jobs(request)
    current = jobs.setdefault(job_id, {"logs": [], "next_log_seq": 1})
    seq = int(current.get("next_log_seq") or 1)
    current.setdefault("logs", []).append({
        "seq": seq, "stream": stream, "text": value[:12000], "at": time.time(),
    })
    if len(current["logs"]) > 1200:
        del current["logs"][:-1200]
    current["next_log_seq"] = seq + 1
    current["updated_at"] = time.time()


async def _run_environment_creation(request: Request, job_id: str, payload: dict[str, Any]) -> None:
    def progress(stage: str, message: str) -> None:
        _set_environment_job(
            request, job_id, state="running", stage=stage, message=message
        )

    def output(stream: str, text: str) -> None:
        _append_environment_log(request, job_id, stream, text)

    try:
        operation = str(payload.get("operation") or "create")
        method = (
            request.app.state.container_supervisor.reconfigure_environment
            if operation == "reconfigure"
            else request.app.state.container_supervisor.create_environment
        )
        call_kwargs = {
            "dockerfile": payload.get("dockerfile"),
            "template_name": str(payload.get("template_name") or "") or None,
            "mounts": [item for item in (payload.get("mounts") or []) if isinstance(item, dict)],
            "gpu_request": payload.get("gpu_request"),  # MUCLI_CONTAINER_HARDWARE_V1
            "devices": [item for item in (payload.get("devices") or []) if isinstance(item, dict)],
            "egress_allow": payload.get("egress_allow"),
            "egress_deny": payload.get("egress_deny"),
            "supervisor_url": f"http://host.docker.internal:{request.app.state.port}",
            "start": bool(payload.get("start", True)),
            "progress": progress,
            "output": output,
        }
        if operation == "reconfigure":
            ref = await asyncio.to_thread(
                method,
                str(payload.get("name") or ""),
                **call_kwargs,
            )
        else:
            ref = await asyncio.to_thread(
                method,
                container_name=str(payload.get("name") or ""),
                **call_kwargs,
            )
        _set_environment_job(
            request, job_id, state="ready", stage="ready",
            message=(
                "Container environment was updated."
                if str(payload.get("operation") or "create") == "reconfigure"
                else "Container environment is ready."
            ),
            container=ref.to_dict(include_secret=False),
        )
    except Exception as exc:
        detail = str(exc)
        output("stderr", detail)
        _set_environment_job(
            request, job_id, state="error", stage="failed",
            message="Container environment creation failed.", detail=detail,
        )
    finally:
        tasks = getattr(request.app.state, "container_environment_tasks", {})
        tasks.pop(job_id, None)


@router.get("/containers", response_class=HTMLResponse)
async def container_manager_page(request: Request):
    _require_local_client(request)
    return request.app.state.templates.TemplateResponse(
        request, "containers.html", {}
    )


@router.get("/api/containers")
async def list_managed_containers(request: Request):
    _require_local_client(request)
    supervisor = request.app.state.container_supervisor
    refs = await asyncio.to_thread(supervisor.list_environments)
    templates = supervisor.template_registry.list_templates()
    return {
        "containers": [ref.to_dict(include_secret=False) for ref in refs],
        "templates": [item.to_dict() for item in templates],
    }


@router.get("/api/containers/stats")
async def managed_container_stats(request: Request):
    """Batch resource telemetry for web and mobile container monitors."""
    _require_local_client(request)
    supervisor = request.app.state.container_supervisor
    collector = getattr(request.app.state, "container_stats_collector", None)
    if collector is None:
        collector = ContainerStatsCollector(supervisor.runner)
        request.app.state.container_stats_collector = collector
    refs = await asyncio.to_thread(supervisor.list_environments)
    return await asyncio.to_thread(collector.collect, refs)


@router.get("/api/containers/{name}/configuration")
async def get_managed_container_configuration(name: str, request: Request):
    _require_local_client(request)
    try:
        config = await asyncio.to_thread(
            request.app.state.container_supervisor.configuration, name
        )
    except (ContainerRuntimeError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"name": name, **config}


@router.put("/api/containers/{name}")
async def update_managed_container(name: str, request: Request, payload: dict[str, Any]):
    _require_local_client(request)
    supervisor = request.app.state.container_supervisor
    ref = supervisor.resolve(name)
    if ref is None:
        raise HTTPException(status_code=404, detail="managed container not found")
    busy = [
        session_name
        for session_name in ref.attached_sessions
        if request.app.state.session_busy_for(session_name).is_set()
    ]
    if busy:
        raise HTTPException(
            status_code=409,
            detail="Container has active session turns: " + ", ".join(busy),
        )
    job_id = uuid.uuid4().hex
    job_payload = dict(payload)
    job_payload.update({"name": ref.name, "operation": "reconfigure"})
    _set_environment_job(
        request,
        job_id,
        state="queued",
        stage="queued",
        message="Container environment update queued.",
    )
    tasks = getattr(request.app.state, "container_environment_tasks", None)
    if tasks is None:
        tasks = {}
        request.app.state.container_environment_tasks = tasks
    task = asyncio.create_task(_run_environment_creation(request, job_id, job_payload))
    tasks[job_id] = task
    return JSONResponse(status_code=202, content={"ok": True, "job_id": job_id})


@router.post("/api/containers")
async def create_managed_container(request: Request, payload: dict[str, Any]):
    _require_local_client(request)
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="container name is required")
    job_id = uuid.uuid4().hex
    _set_environment_job(
        request, job_id, state="queued", stage="queued",
        message="Container environment creation queued.",
    )
    tasks = getattr(request.app.state, "container_environment_tasks", None)
    if tasks is None:
        tasks = {}
        request.app.state.container_environment_tasks = tasks
    task = asyncio.create_task(_run_environment_creation(request, job_id, dict(payload)))
    tasks[job_id] = task
    return JSONResponse(status_code=202, content={"ok": True, "job_id": job_id})


@router.get("/api/containers/jobs/{job_id}")
async def get_container_job(job_id: str, request: Request, after: int = 0):
    _require_local_client(request)
    value = _environment_jobs(request).get(job_id)
    if not isinstance(value, dict):
        raise HTTPException(status_code=404, detail="container job not found")
    payload = dict(value)
    payload["logs"] = [
        dict(item) for item in (value.get("logs") or [])
        if int(item.get("seq") or 0) > int(after or 0)
    ]
    return payload


def _persist_session_container_binding(
    request: Request, session_name: str, config: dict[str, Any] | None
) -> None:
    path = os.path.join(_config.HISTORY_DIR, "sessions", session_name, "session.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Session {session_name!r} not found")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read session: {exc}") from exc
    variables = dict(data.get("variables") or {})
    if config is None:
        variables["session_type"] = "workspace"
        data.pop("container_config", None)
    else:
        variables.update(
            {
                "session_type": "container",
                "yolo": True,
                "strict_mode": False,
                "plan_mode": False,
                "lazy_tools_enabled": True,
                "security_allow_secret_paths": False,
            }
        )
        data["container_config"] = dict(config)
    data["variables"] = variables
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    session = request.app.state.sessions.get(session_name)
    if session is not None:
        session.variables.update(variables)
        session.session_manager.container_config = dict(config or {})
        if config is None:
            session.container_ref = None
        else:
            session.container_ref = request.app.state.container_supervisor.resolve(
                str(config.get("container_name") or "")
            )
        session.session_manager.save_history(session.folder_context)
        session.sync_runtime_state()


@router.post("/api/containers/{name}/actions/{action}")
async def manage_container(
    name: str, action: str, request: Request, payload: dict[str, Any] | None = None
):
    _require_local_client(request)
    supervisor = request.app.state.container_supervisor
    current_ref = supervisor.resolve(name)
    if current_ref is not None and action in {"stop", "restart", "attach", "detach"}:
        busy = [
            session_name for session_name in current_ref.attached_sessions
            if request.app.state.session_busy_for(session_name).is_set()
        ]
        if busy:
            raise HTTPException(
                status_code=409,
                detail="Container has active session turns: " + ", ".join(busy),
            )
    try:
        if action == "start":
            ref = await asyncio.to_thread(supervisor.start, name)
        elif action == "stop":
            ref = await asyncio.to_thread(supervisor.stop, name)
        elif action == "restart":
            ref = await asyncio.to_thread(supervisor.restart, name)
        elif action == "attach":
            session_name = str((payload or {}).get("session_name") or "").strip()
            if not session_name:
                raise HTTPException(status_code=400, detail="session_name is required")
            ref = await asyncio.to_thread(supervisor.attach_session, name, session_name)
            _persist_session_container_binding(
                request, session_name, supervisor.configuration(ref.name)
            )
        elif action == "detach":
            session_name = str((payload or {}).get("session_name") or "").strip()
            if not session_name:
                raise HTTPException(status_code=400, detail="session_name is required")
            ref = await asyncio.to_thread(
                supervisor.detach_session, name, session_name, stop_if_idle=False
            )
            _persist_session_container_binding(request, session_name, None)
        else:
            raise HTTPException(status_code=404, detail="unknown container action")
    except (ContainerRuntimeError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "container": ref.to_dict(include_secret=False)}


@router.delete("/api/containers/{name}")
async def delete_managed_container(name: str, request: Request, force: bool = False):
    _require_local_client(request)
    supervisor = request.app.state.container_supervisor
    ref = supervisor.resolve(name)
    if ref is not None:
        busy = [
            session_name for session_name in ref.attached_sessions
            if request.app.state.session_busy_for(session_name).is_set()
        ]
        if busy:
            raise HTTPException(status_code=409, detail="Container has active session turns: " + ", ".join(busy))
    try:
        removed = await asyncio.to_thread(supervisor.remove, name, force=force)
    except (ContainerRuntimeError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="managed container not found")
    return {"ok": True}


@router.post("/api/containers/{name}/snapshot")
async def snapshot_managed_container(name: str, request: Request, payload: dict[str, Any]):
    _require_local_client(request)
    template_name = str(payload.get("template_name") or "").strip()
    if not template_name:
        raise HTTPException(status_code=400, detail="template_name is required")
    try:
        item = await asyncio.to_thread(
            request.app.state.container_supervisor.snapshot,
            name, template_name, description=str(payload.get("description") or ""),
        )
    except (ContainerRuntimeError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "template": item.to_dict()}


@router.delete("/api/container-templates/{name}")
async def delete_container_template(name: str, request: Request):
    _require_local_client(request)
    removed = await asyncio.to_thread(
        request.app.state.container_supervisor.remove_template, name
    )
    if not removed:
        raise HTTPException(status_code=404, detail="container template not found")
    return {"ok": True}


async def _shell_completion(
    docker: str,
    container_name: str,
    *,
    cwd: str,
    line: str,
    cursor: int,
    request_id: str,
) -> dict[str, Any]:
    """Run bounded bash completion inside the container without executing input."""
    from mu.container.shell_qol import build_completion_response, completion_target

    line = str(line or "")[:8192]
    target = completion_target(line, cursor)
    script = r"""
prefix=${MUCLI_SHELL_PREFIX-}
while IFS= read -r item; do
    if [ -d "$item" ]; then
        printf '%s/\n' "${item%/}"
    else
        printf '%s\n' "$item"
    fi
done < <(compgen -cdfa -- "$prefix" 2>/dev/null | LC_ALL=C sort -u | head -200)
"""

    async def run(workdir: str) -> tuple[int, str]:
        command = [docker, "exec", "-i"]
        if workdir:
            command += ["-w", workdir]
        command += [
            "-e", f"MUCLI_SHELL_PREFIX={target.prefix}",
            container_name,
            "/bin/bash", "--noprofile", "--norc", "-c", script,
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return 124, ""
        return int(process.returncode or 0), stdout.decode("utf-8", errors="replace")

    code, stdout = await run(cwd)
    if code != 0 and cwd:
        code, stdout = await run("")
    candidates = stdout.splitlines() if code == 0 else []
    return build_completion_response(
        line=line,
        cursor=cursor,
        candidates=candidates,
        request_id=request_id,
    )


@router.websocket("/api/containers/{name}/shell")
async def managed_container_shell(websocket: WebSocket, name: str):
    # MUCLI_SHELL_QOL_V1
    # Interactive shell is LOOPBACK-ONLY (codex round-6 F6): an
    # unauthenticated private-network client must not get a bash inside
    # the container. Empty host = uvicorn without client info (in-process
    # tests / unix socket) — keep allowed; anything remote is rejected
    # before accept().
    client = getattr(websocket, "client", None)
    host = str(getattr(client, "host", "") or "")
    if host and host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        await websocket.close(code=1008, reason="Container shell is restricted to localhost")
        return
    await websocket.accept()
    supervisor = websocket.app.state.container_supervisor
    try:
        ref = await asyncio.to_thread(supervisor.start, name)
        docker = supervisor.runner.require("docker")
        process = await asyncio.create_subprocess_exec(
            docker,
            "exec",
            "-i",
            "-e",
            "TERM=dumb",
            ref.name,
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-i",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if process.stdin is not None:
            initialization = (
                "export PS1='' PS2=''; "
                "PROMPT_COMMAND='printf \"\\036MUCLI_CWD:%s\\037\\n\" \"$PWD\"'\n"
            )
            process.stdin.write(initialization.encode("utf-8"))
            await process.stdin.drain()
    except Exception as exc:
        await websocket.send_text(f"Unable to open shell: {exc}\n")
        await websocket.close(code=1011)
        return

    cwd_filter = CwdMarkerFilter()

    async def pump_output() -> None:
        assert process.stdout is not None
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                break
            visible = cwd_filter.feed(chunk.decode("utf-8", errors="replace"))
            if visible:
                await websocket.send_text(visible)
        tail = cwd_filter.flush()
        if tail:
            await websocket.send_text(tail)

    output_task = asyncio.create_task(pump_output())
    try:
        while process.returncode is None:
            data = await websocket.receive_text()
            control: dict[str, Any] | None = None
            if data.startswith("{"):
                try:
                    parsed = json.loads(data)
                    control = parsed if isinstance(parsed, dict) else None
                except (TypeError, ValueError):
                    control = None

            if control and control.get("type") == "shell_complete":
                response = await _shell_completion(
                    docker,
                    ref.name,
                    cwd=cwd_filter.cwd,
                    line=str(control.get("line") or ""),
                    cursor=int(control.get("cursor") or 0),
                    request_id=str(control.get("request_id") or ""),
                )
                await websocket.send_text(json.dumps(response))
                continue

            shell_input = (
                str(control.get("data") or "")
                if control and control.get("type") == "shell_input"
                else data
            )
            if process.stdin is None:
                break
            process.stdin.write(shell_input.encode("utf-8"))
            await process.stdin.drain()
    except WebSocketDisconnect:
        pass
    finally:
        if process.returncode is None:
            process.terminate()
        await process.wait()
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)



@router.get("/api/container-defaults")
async def container_defaults(request: Request = None):
    """Editable defaults and detected host hardware shared by all creation flows."""
    hardware = None
    if request is not None:
        hardware = await asyncio.to_thread(
            request.app.state.container_supervisor.hardware_capabilities
        )
    return {
        "dockerfile": default_dockerfile(),
        "egress_allow": list(DEFAULT_EGRESS_ALLOW),
        "egress_deny": [],
        "hardware": hardware,
    }


@router.post("/api/container-worker/events")
async def worker_event(
    request: Request,
    payload: dict[str, Any],
    x_mucli_worker_token: str | None = Header(default=None),
):
    supervisor = request.app.state.container_supervisor
    container_name = str(payload.get("container_name") or "")
    if not supervisor.validate_token(
        container_name, x_mucli_worker_token or ""
    ):
        raise HTTPException(status_code=401, detail="invalid worker token")
    event = dict(payload)
    event.pop("container_name", None)
    session_name = str(event.get("session_name") or "")
    if session_name and event.get("kind") == "context_snapshot":
        try:
            from ..memory_snapshot import ingest_context_timeline_point

            mirrored = ingest_context_timeline_point(
                request.app.state.session_by_name(session_name),
                event.get("timeline_point"),
            )
            if mirrored:
                event["timeline_point"] = mirrored
        except Exception:
            # Observability mirroring is best-effort; never reject a worker
            # callback or interrupt the underlying model turn.
            pass
    if session_name:
        busy = request.app.state.session_busy_for(session_name)
        if event.get("kind") in {
            "assistant_start", "assistant_delta", "thinking_delta", "tool_call",
            "tool_result", "status_start", "status_update",
        }:
            busy.set()
        elif event.get("kind") in {"turn_complete", "error"}:
            busy.clear()
    await request.app.state.bus.publish(event)
    return {"ok": True}


@router.post("/api/container-worker/artifacts")
async def worker_artifact(
    request: Request,
    session_name: str,
    container_name: str,
    name: str,
    mime_type: str = "application/octet-stream",
    kind: str = "file",
    display: str = "download",
    title: str | None = None,
    height: int = 480,
    timeline_turn_id: str | None = None,
    timeline_history_index: int = -1,
    timeline_part_index: int = -1,
    x_mucli_worker_token: str | None = Header(default=None),
):
    """Persist a worker artifact directly into the host session registry.

    The host registry is authoritative. This control-plane upload remains
    reliable even if an older or manually recreated worker is missing the
    nested session bind mount under ``/root/.mucli``.
    """
    supervisor = request.app.state.container_supervisor
    if not supervisor.validate_token(
        container_name, x_mucli_worker_token or ""
    ):
        raise HTTPException(status_code=401, detail="invalid worker token")
    ref = supervisor.container_for_session(session_name)
    if ref is None or ref.name != container_name:
        raise HTTPException(
            status_code=403, detail="session is not attached to this worker"
        )

    safe_session = str(session_name or "").strip()
    if not safe_session or os.path.basename(safe_session) != safe_session:
        raise HTTPException(status_code=400, detail="invalid session name")
    session_dir = os.path.join(_config.HISTORY_DIR, "sessions", safe_session)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail="session not found")

    registry = ArtifactRegistry(session_dir)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > registry.max_bytes:
                raise HTTPException(status_code=413, detail="artifact exceeds maximum size")
        except ValueError:
            pass

    fd, incoming_path = tempfile.mkstemp(prefix="artifact-upload-", dir=session_dir)
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            async for chunk in request.stream():
                total += len(chunk)
                if total > registry.max_bytes:
                    raise HTTPException(status_code=413, detail="artifact exceeds maximum size")
                handle.write(chunk)
        artifact = registry.add(
            name=name,
            source_path=incoming_path,
            mime_type=mime_type,
            kind=kind,
            display=display,
            title=title,
            height=height,
            timeline_turn_id=timeline_turn_id,
            timeline_history_index=timeline_history_index,
            timeline_part_index=timeline_part_index,
        )
    finally:
        try:
            os.unlink(incoming_path)
        except FileNotFoundError:
            pass

    await request.app.state.bus.publish(
        {
            "kind": "artifact_created",
            "artifact": artifact,
            "session_name": safe_session,
        }
    )
    return {"ok": True, "artifact": artifact}


@router.post("/api/container-worker/prompt")
async def worker_prompt(
    request: Request,
    payload: dict[str, Any],
    x_mucli_worker_token: str | None = Header(default=None),
):
    supervisor = request.app.state.container_supervisor
    container_name = str(payload.get("container_name") or "")
    session_name = str(payload.get("session_name") or "")
    if not supervisor.validate_token(
        container_name, x_mucli_worker_token or ""
    ):
        raise HTTPException(status_code=401, detail="invalid worker token")
    ref = supervisor.container_for_session(session_name)
    if ref is None or ref.name != container_name:
        raise HTTPException(
            status_code=403, detail="session is not attached to this worker"
        )

    prompt = dict(payload.get("prompt") or {})
    prompt["session_name"] = session_name
    timeout = max(1.0, min(float(payload.get("timeout") or 600.0), 3600.0))
    prompt_id, event = request.app.state.prompts.open(prompt)
    await request.app.state.bus.publish(
        {"kind": "prompt", "id": prompt_id, "prompt": prompt, "session_name": session_name}
    )
    answered = await asyncio.to_thread(event.wait, timeout)
    if not answered:
        request.app.state.prompts.cancel(prompt_id)
        await request.app.state.bus.publish(
            {"kind": "prompt_cancelled", "id": prompt_id, "session_name": session_name}
        )
    answer = request.app.state.prompts.take(prompt_id)
    await request.app.state.bus.publish(
        {"kind": "prompt_resolved", "id": prompt_id, "session_name": session_name}
    )
    return {"ok": True, "answer": answer or {"cancelled": True}}


@router.get("/api/sessions/{name}/container")
async def container_status(name: str, request: Request):
    ref = request.app.state.container_supervisor.container_for_session(name)
    if ref is None:
        raise HTTPException(status_code=404, detail="session has no container")
    return ref.to_dict(include_secret=False)


@router.post("/api/sessions/{name}/container/mount")
async def add_mount(name: str, request: Request, payload: dict[str, Any]):
    if request.app.state.session_busy_for(name).is_set():
        raise HTTPException(status_code=409, detail="interrupt the active turn before changing mounts")
    try:
        ref = request.app.state.container_supervisor.add_mount(
            name,
            str(payload.get("host_path") or ""),
            str(payload.get("container_path") or ""),
            str(payload.get("mode") or "rw"),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ref.to_dict(include_secret=False)


@router.post("/api/sessions/{name}/container/stop")
async def stop_container(name: str, request: Request):
    ref = request.app.state.container_supervisor.detach(name, stop_if_idle=True)
    return {"ok": True, "container": ref.to_dict(include_secret=False) if ref else None}
