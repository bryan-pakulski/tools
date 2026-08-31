"""FastAPI factory for the GUI server and shared background control daemon."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.types import Scope

from .bus import EventBus
from .container_mode_proxy import proxy_container_mode_request
from .deps import require_session
from .live_observability import register_live_observability_hooks
from .prompts import PromptStore
from .routers import (
    artifacts as artifacts_router,
    attachments as attachments_router,
    audio as audio_router,
    chat,
    containers as containers_router,
    debug as debug_router,
    feature as feature_router,
    files as files_router,
    inspector,
    jobs as jobs_router,
    loop as loop_router,
    memory as memory_router,
    memories as memories_router,
    modes,
    prompts as prompts_router,
    providers as providers_router,
    research as research_router,
    security as security_router,
    sessions,
    skills as skills_router,
    system_prompts as system_prompts_router,
    teacher as teacher_router,
    traces as traces_router,
    threads as threads_router,
)
from .watcher import SessionWatcher
from .web_ui import WebUI
from mu.container import ContainerSupervisor
from mu.jobs import get_default_job_service
from mu.jobs.controller import JobController
from mu.threads.scheduler import ThreadWakeScheduler


def _register_memory_snapshot_hook() -> None:
    """Backward-compatible name for extensions importing the old hook."""
    register_live_observability_hooks()


def _register_subagent_snapshot_hook() -> None:
    """Backward-compatible name for extensions importing the old hook."""
    register_live_observability_hooks()

GUI_ROOT = Path(__file__).parent
TEMPLATES_DIR = GUI_ROOT / "templates"
STATIC_DIR = GUI_ROOT / "static"
__all__ = ["create_app", "require_session"]


class _NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response


def session_by_name(app: FastAPI, name: Optional[str]):
    sessions: Dict[str, Any] = app.state.sessions
    if name:
        return sessions.get(name)
    current = app.state.current_session_name
    return sessions.get(current) if current else None


def session_lock_for(app: FastAPI, name: Optional[str]) -> threading.Lock:
    target = name or app.state.current_session_name
    if target is None:
        return app.state._fallback_lock
    return app.state.session_locks.setdefault(target, threading.Lock())


def session_busy_for(app: FastAPI, name: Optional[str]) -> threading.Event:
    target = name or app.state.current_session_name
    if target is None:
        return app.state._fallback_busy
    return app.state.session_busy.setdefault(target, threading.Event())


def web_ui_for(app: FastAPI, name: Optional[str]) -> Optional[WebUI]:
    target = name or app.state.current_session_name
    return app.state.web_uis.get(target) if target else None


def create_app(*, args: Any, build_session_fn: Callable, port: int = 30311) -> FastAPI:
    app = FastAPI(title="mucli", version="1.0", docs_url=None, redoc_url=None)
    app.middleware("http")(proxy_container_mode_request)
    bus = EventBus()
    prompts = PromptStore()

    app.state.sessions = {}
    app.state.session_locks = {}
    app.state.session_busy = {}
    app.state.web_uis = {}
    app.state.current_session_name = None
    app.state._fallback_lock = threading.Lock()
    app.state._fallback_busy = threading.Event()
    app.state.session_registry_lock = threading.RLock()
    app.state.container_creation_status = {}
    app.state.container_creation_lock = threading.Lock()
    app.state.container_creation_tasks = {}
    app.state.container_environment_jobs = {}
    app.state.container_environment_tasks = {}

    app.state.bus = bus
    app.state.prompts = prompts
    app.state.port = port
    app.state.args = args
    app.state.build_session_fn = build_session_fn
    app.state.load_session = lambda **kw: _load_session(app, **kw)
    app.state.unload_session = lambda **kw: _unload_session(app, **kw)
    app.state.watcher = SessionWatcher(app)
    app.state.container_supervisor = ContainerSupervisor()
    app.state.job_service = get_default_job_service()
    # Milestone 2: each worker is a separate Python process with a managed Git
    # worktree, so five jobs can execute without sharing Session CWD/runtime.
    app.state.job_controller = JobController(
        app.state.job_service,
        max_workers=5,
    )

    app.state.session_by_name = lambda name=None: session_by_name(app, name)
    app.state.session_lock_for = lambda name=None: session_lock_for(app, name)
    app.state.session_busy_for = lambda name=None: session_busy_for(app, name)
    app.state.web_ui_for = lambda name=None: web_ui_for(app, name)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.auto_reload = True
    app.state.templates = templates
    app.mount("/static", _NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(artifacts_router.router, prefix="/api/sessions", tags=["artifacts"])
    app.include_router(attachments_router.router, prefix="/api/sessions", tags=["attachments"])
    app.include_router(containers_router.router, tags=["containers"])
    app.include_router(providers_router.router, prefix="/api/providers", tags=["providers"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(threads_router.router, prefix="/api/threads", tags=["threads"])
    app.include_router(modes.router, prefix="/api/modes", tags=["modes"])
    app.include_router(prompts_router.router, prefix="/api/prompts", tags=["prompts"])
    app.include_router(system_prompts_router.router, prefix="/api/system-prompts", tags=["system-prompts"])
    app.include_router(inspector.router, prefix="/api", tags=["inspector"])
    app.include_router(jobs_router.router, prefix="/api/jobs", tags=["jobs"])
    app.include_router(teacher_router.router, prefix="/api/teacher", tags=["teacher"])
    app.include_router(feature_router.router, prefix="/api/feature", tags=["feature"])
    app.include_router(research_router.router, prefix="/api/research", tags=["research"])
    app.include_router(security_router.router, prefix="/api/security", tags=["security"])
    app.include_router(loop_router.router, prefix="/api/loop", tags=["loop"])
    app.include_router(debug_router.router, prefix="/api/debug", tags=["debug"])
    app.include_router(memory_router.router, prefix="/api/memory", tags=["memory"])
    app.include_router(memories_router.router, prefix="/api/v1", tags=["memory-ledger"])
    app.include_router(files_router.router, prefix="/api/files", tags=["files"])
    app.include_router(skills_router.router, prefix="/api/skills", tags=["skills"])
    app.include_router(audio_router.router, prefix="/api/audio", tags=["audio"])
    traces_router._install_trace_gzip(app)
    app.include_router(traces_router.router, prefix="/api/traces", tags=["traces"])
    app.include_router(chat.events_router, tags=["events"])

    register_live_observability_hooks()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        session = session_by_name(app, None)
        manager = session.session_manager if session else None
        return templates.TemplateResponse(request, "index.html", {
            "session_name": manager.current_session_name if manager else "",
            "agent_mode": session.variables.get("agent_mode", "default") if session else "default",
            "session_type": session.variables.get("session_type", "workspace") if session else "workspace",
            "provider": session.provider.name if session and session.provider else "",
            "model": session.provider.model_name if session and session.provider else "",
            "session_active": session is not None,
        })

    @app.get("/trace", response_class=HTMLResponse)
    async def trace_analyzer(request: Request):
        return templates.TemplateResponse(request, "trace.html", {
            "session_name": "", "agent_mode": "default", "provider": "",
            "model": "", "session_active": False,
        })

    @app.get("/work", response_class=HTMLResponse)
    async def engineering_work(request: Request):
        requested = str(request.query_params.get("session") or "").strip()
        session_name = requested or app.state.current_session_name or ""
        return templates.TemplateResponse(request, "work.html", {
            "session_name": session_name,
            "agent_mode": "default",
            "provider": "",
            "model": "",
            "session_active": bool(session_name),
        })

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "session_active": app.state.current_session_name is not None,
            "loaded_sessions": list(app.state.sessions.keys()),
            "durable_jobs": len(app.state.job_service.list(limit=1000)),
            "job_controller": app.state.job_controller.snapshot(),
        }

    @app.on_event("startup")
    async def _bind_loop():
        bus.bind_loop(asyncio.get_running_loop())
        app.state.thread_wake_scheduler = ThreadWakeScheduler(
            lambda coordinator, wake: _run_thread_wake(app, coordinator, wake)
        )
        app.state.thread_wake_scheduler.start()
        app.state.job_controller.start()
        app.state.watcher.start()

    @app.on_event("shutdown")
    async def _stop_services():
        scheduler = getattr(app.state, "thread_wake_scheduler", None)
        if scheduler is not None:
            scheduler.stop(wait=False)
        app.state.watcher.stop()
        # Active worker processes are intentionally not killed here. Their own
        # heartbeat/lease lets them survive daemon/browser restarts.
        app.state.job_controller.stop(wait=False)
        tasks = list(app.state.container_creation_tasks.values())
        tasks += list(app.state.container_environment_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        app.state.container_supervisor.shutdown()

    return app


def _load_session(
    app: FastAPI,
    *,
    name: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    focus: bool = True,
):
    with app.state.session_registry_lock:
        if name in app.state.sessions:
            if focus:
                app.state.current_session_name = name
            return app.state.sessions[name]
        web_ui = WebUI(app.state.bus, app.state.prompts, session_name=name)
        app.state.web_uis[name] = web_ui
        args = copy.copy(app.state.args)
        args.session = name
        if provider is not None:
            args.provider = provider
        if model is not None:
            args.model = model
        session = app.state.build_session_fn(args, web_ui, allow_prompt=False)
        session.ui = web_ui
        session.session_manager.ui = web_ui
        app.state.sessions[name] = session
        app.state.session_locks.setdefault(name, threading.Lock())
        app.state.session_busy.setdefault(name, threading.Event())
        if focus:
            app.state.current_session_name = name
        return session


def _unload_session(app: FastAPI, *, name: Optional[str] = None) -> bool:
    target = name or app.state.current_session_name
    if not target or target not in app.state.sessions:
        return False
    session = app.state.sessions.pop(target)
    app.state.session_locks.pop(target, None)
    app.state.session_busy.pop(target, None)
    app.state.web_uis.pop(target, None)
    try:
        session.session_manager.save_history(session.folder_context)
    except Exception:
        pass
    if app.state.current_session_name == target:
        remaining = list(app.state.sessions.keys())
        app.state.current_session_name = remaining[-1] if remaining else None
    return True


_THREAD_WAKE_PROMPT = (
    "A peer thread sent coordination updates. Inspect LAYER 3C, respond or "
    "acknowledge each incoming message, and then continue your existing task "
    "where appropriate."
)


def _run_thread_wake(app: FastAPI, coordinator, wake: dict) -> bool:
    """Run one coalesced idle-peer wake without changing GUI focus."""

    target = coordinator.get_thread(str(wake.get("target_thread_id") or ""))
    if not target:
        return True
    name = str(target.get("session_name") or "")
    if not name:
        return True
    session = _load_session(app, name=name, focus=False)
    busy = app.state.session_busy_for(name)
    if busy.is_set():
        return False
    app.state.bus.publish_threadsafe(
        {
            "kind": "thread_wake_started",
            "session_name": name,
            "thread_group_id": coordinator.group_id,
            "thread_id": target["thread_id"],
        }
    )
    session_type = str(
        session.variables.get("session_type", "workspace") or "workspace"
    ).lower()
    try:
        if session_type == "container":
            busy.set()
            try:
                app.state.container_supervisor.send_sync(
                    name,
                    _THREAD_WAKE_PROMPT,
                    provider=session.provider.name,
                    model=session.provider.model_name,
                    agent_mode=str(session.variables.get("agent_mode", "default")),
                    system_instruction=session.system_instruction,
                    timeout=None,
                    origin="thread_wake",
                )
                session.session_manager._load_session(name)
                session.sync_runtime_state()
            finally:
                busy.clear()
        else:
            chat._run_send(
                session,
                _THREAD_WAKE_PROMPT,
                lock=app.state.session_lock_for(name),
                busy=busy,
                session_name=name,
                origin="thread_wake",
            )
        coordinator.record_event(
            "thread_wake_completed",
            actor_thread_id=target["thread_id"],
            message_id=str(wake.get("message_id") or ""),
        )
        app.state.bus.publish_threadsafe(
            {
                "kind": "history_refresh",
                "session_name": name,
                "thread_group_id": coordinator.group_id,
            }
        )
        return True
    except Exception as exc:
        coordinator.record_event(
            "thread_wake_failed",
            actor_thread_id=target["thread_id"],
            message_id=str(wake.get("message_id") or ""),
            payload={"error": str(exc)[:1000]},
        )
        return False
