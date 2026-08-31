"""Container entrypoint: runs the ordinary MuCLI agent loop inside Docker."""
from __future__ import annotations

import argparse
import contextvars
import ctypes
import json
import logging
import os
import socket
import threading
import time
import traceback
import uuid
from contextlib import AbstractContextManager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mu.container.ref import WORKER_PROTOCOL_VERSION
from mu.session.manager import RevisionConflict
from mu.ui.base import BaseUI


logger = logging.getLogger("mucli.container.worker")

_DELTAS_FLUSH_MS = 50.0
_DELTAS_MAX_CHARS = 8192


class _Status(AbstractContextManager):
    def __init__(self, ui: "WorkerBridgeUI", text: str):
        self.ui = ui
        self.text = text

    def __enter__(self):
        self.ui.publish({"kind": "status_start", "text": self.text})
        return self

    def __exit__(self, *_args):
        self.ui.publish_event({"kind": "status_end", "text": self.text})
        return False

    def update(self, message: str) -> None:
        self.text = str(message)
        self.ui.publish_event({"kind": "status_update", "text": self.text})


class WorkerBridgeUI(BaseUI):
    """BaseUI implementation that forwards events to the host supervisor."""

    def __init__(self, session_name: str):
        self.session_name = session_name
        self.container_name = os.getenv("MUCLI_CONTAINER_NAME", "")
        self.supervisor_url = os.getenv("MUCLI_SUPERVISOR_URL", "").rstrip("/")
        self.token = os.getenv("MUCLI_WORKER_TOKEN", "")
        self.variables: dict[str, Any] = {}
        self.turn_id: str | None = None
        # Worker → supervisor traffic is a control-plane callback on the
        # internal bridge. It must never inherit HTTP(S)_PROXY/SOCKS settings
        # intended for provider egress.
        self._client = httpx.Client(timeout=10.0, trust_env=False)
        # Dedup flag: when True, assistant text was already streamed via
        # assistant_delta events, so the post-stream render_message call
        # from loop_body must be suppressed to avoid a duplicate bubble.
        self._streamed_any_text = False
        self._delta_lock = threading.Lock()
        self._assistant_buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._last_delta_flush = 0.0

    def _publish_raw(self, event: dict[str, Any]) -> None:
        if not self.supervisor_url:
            return
        payload = {
            **event,
            "session_name": self.session_name,
            "container_name": self.container_name,
            "session_type": "container",
        }
        active_mode = str(self.variables.get("agent_mode") or "").strip()
        if active_mode:
            payload.setdefault("agent_mode", active_mode)
        try:
            self._client.post(
                f"{self.supervisor_url}/api/container-worker/events",
                json=payload,
                headers={"X-MuCLI-Worker-Token": self.token},
            ).raise_for_status()
        except Exception:
            # A temporary GUI disconnect must not abort an agent turn.  Session
            # history remains authoritative and will be recovered on reconnect.
            pass

    def _flush_deltas(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._delta_lock:
            size = sum(map(len, self._assistant_buffer)) + sum(
                map(len, self._thinking_buffer)
            )
            if not (
                force
                or size >= _DELTAS_MAX_CHARS
                or (now - self._last_delta_flush) * 1000.0 >= _DELTAS_FLUSH_MS
            ):
                return
            assistant = "".join(self._assistant_buffer)
            thinking = "".join(self._thinking_buffer)
            self._assistant_buffer.clear()
            self._thinking_buffer.clear()
            self._last_delta_flush = now
        if thinking:
            self.publish(
                {"kind": "thinking_delta", "turn_id": self.turn_id, "text": thinking}
            )
        if assistant:
            self.publish(
                {"kind": "assistant_delta", "turn_id": self.turn_id, "text": assistant}
            )

    def publish_event(self, event: dict[str, Any]) -> None:
        """Publish an ordered boundary event to the host event bus."""
        self._flush_deltas(force=True)
        self._publish_raw(event)

    def publish(self, event: dict[str, Any]) -> None:
        """Compatibility alias used by artifact and extension handlers."""
        self.publish_event(event)

    def _publish(self, event: dict[str, Any]) -> None:
        """Compatibility with integrations built against WebUI's old hook."""
        self.publish_event(event)

    def publish_artifact(
        self,
        *,
        name: str,
        source_path: str | None = None,
        content: str | bytes | None = None,
        mime_type: str = "application/octet-stream",
        kind: str = "file",
        display: str = "download",
        title: str | None = None,
        height: int | None = None,
        timeline_turn_id: str | None = None,
        timeline_history_index: int | None = None,
        timeline_part_index: int | None = None,
    ) -> dict[str, Any]:
        if not self.supervisor_url:
            raise RuntimeError("container supervisor URL is unavailable")
        if (source_path is None) == (content is None):
            raise RuntimeError("provide exactly one of source_path or content")
        requested_kind = str(kind or "file").strip().lower()
        params = {
            "session_name": self.session_name,
            "container_name": self.container_name,
            "name": str(name or ""),
            "mime_type": str(mime_type or "application/octet-stream"),
            "kind": requested_kind,
            "display": str(display or "download"),
        }
        if title:
            params["title"] = str(title)
        if height is not None:
            params["height"] = str(int(height))
        if timeline_turn_id:
            params["timeline_turn_id"] = str(timeline_turn_id)
        if (
            timeline_history_index is not None
            and int(timeline_history_index) >= 0
        ):
            params["timeline_history_index"] = str(int(timeline_history_index))
        if timeline_part_index is not None and int(timeline_part_index) >= 0:
            params["timeline_part_index"] = str(int(timeline_part_index))
        headers = {"X-MuCLI-Worker-Token": self.token}
        if source_path is not None:
            with open(source_path, "rb") as handle:
                chunks = iter(lambda: handle.read(1024 * 1024), b"")
                response = self._client.post(
                    f"{self.supervisor_url}/api/container-worker/artifacts",
                    params=params,
                    content=chunks,
                    headers=headers,
                    timeout=None,
                )
        else:
            payload = (
                content.encode("utf-8")
                if isinstance(content, str)
                else bytes(content or b"")
            )
            response = self._client.post(
                f"{self.supervisor_url}/api/container-worker/artifacts",
                params=params,
                content=payload,
                headers=headers,
                timeout=None,
            )
        response.raise_for_status()
        data = response.json()
        artifact = data.get("artifact") if isinstance(data, dict) else None
        if not isinstance(artifact, dict) or not artifact.get("artifact_id"):
            raise RuntimeError("container supervisor returned an invalid artifact descriptor")
        if requested_kind == "visualization" and (
            artifact.get("kind") != "visualization"
            or artifact.get("display") != "inline"
            or not artifact.get("view_url")
        ):
            raise RuntimeError(
                "container supervisor did not preserve visualization metadata; "
                "reload the container session to upgrade its worker bridge"
            )
        return artifact

    def list_artifacts(self) -> list[dict[str, Any]]:
        if not self.supervisor_url:
            return []
        response = self._client.get(
            f"{self.supervisor_url}/api/sessions/{self.session_name}/artifacts",
            params={"_ts": __import__("time").time_ns()},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        value = data.get("artifacts") if isinstance(data, dict) else []
        return [dict(item) for item in value if isinstance(item, dict)]

    def render_message(self, role, content, model_name=None):
        text = str(content or "")
        if role == "assistant":
            # Suppress the duplicate assistant bubble: loop_body calls
            # render_message("assistant", full_text) right after the streaming
            # loop delivers the same text token-by-token. If we already
            # streamed it, skip. _streamed_any_text resets on next stream.
            if self._streamed_any_text:
                self._streamed_any_text = False
                return
            turn_id = uuid.uuid4().hex[:12]
            # Route through publish (the compatibility alias) so integrations
            # that override publish observe assistant events too.
            self.publish({"kind": "assistant_start", "turn_id": turn_id})
            self.publish({"kind": "assistant_delta", "turn_id": turn_id, "text": text})
            self.publish({"kind": "assistant_end", "turn_id": turn_id})
        elif role == "user":
            self.publish({"kind": "user_message", "text": text})
        else:
            self.publish({"kind": "info", "text": text, "role": role, "model": model_name})

    def get_input(self, *_args, **_kwargs):
        return ""

    def show_error(self, message):
        self.publish_event({"kind": "error", "text": str(message)})

    def show_info(self, message):
        self.publish_event({"kind": "info", "text": str(message)})

    def show_status(self, message):
        return _Status(self, str(message))

    def show_tool_result(self, result_str):
        self.publish_event({"kind": "tool_result", "text": str(result_str)})

    def stream_assistant_delta(self, text: str):
        if not text:
            return
        if self.turn_id is None:
            # New turn — reset the dedup flag from the previous turn.
            self._streamed_any_text = False
            self.turn_id = uuid.uuid4().hex[:12]
            self.publish({"kind": "assistant_start", "turn_id": self.turn_id})
        self._streamed_any_text = True
        with self._delta_lock:
            self._assistant_buffer.append(text)
        # Force the flush so each delta is observable immediately — callers
        # that override publish see every delta, preserving the event contract.
        self._flush_deltas(force=True)

    def stream_thinking_delta(self, text: str):
        if not text:
            return
        if self.turn_id is None:
            self._publish_raw({"kind": "thinking_delta", "turn_id": None, "text": text})
            return
        with self._delta_lock:
            self._thinking_buffer.append(text)
        self._flush_deltas()

    def stream_tool_call(self, tool_name: str):
        self.publish_event({"kind": "tool_call", "turn_id": self.turn_id, "tool_name": tool_name})

    def stream_assistant_end(self):
        if self.turn_id is not None:
            self.publish_event({"kind": "assistant_end", "turn_id": self.turn_id})
            self.turn_id = None

    def set_variables(self, variables_dict):
        self.variables = dict(variables_dict or {})

    # Container sessions auto-approve normal modifying tools by design.
    # Claim overrides are the exception: peer work can be overwritten, so
    # this policy must reach the human supervisor even in a YOLO container.
    def request_tool_approval(self, *_args, **kwargs):
        if kwargs.get("approval_policy") == "always_human":
            return self._ask_prompt(
                {
                    "shape": "tool_approval",
                    "tool_name": kwargs.get("tool_name"),
                    "tool_args": kwargs.get("tool_args")
                    or kwargs.get("display_args")
                    or {},
                    "approval_policy": "always_human",
                }
            )
        return {"approved": True, "remember": True}

    def _ask_prompt(self, prompt: dict[str, Any], timeout: float = 600.0) -> Any:
        if not self.supervisor_url:
            return {"cancelled": True}
        payload = {
            "container_name": self.container_name,
            "session_name": self.session_name,
            "prompt": prompt,
            "timeout": timeout,
        }
        try:
            response = self._client.post(
                f"{self.supervisor_url}/api/container-worker/prompt",
                json=payload,
                headers={"X-MuCLI-Worker-Token": self.token},
                timeout=timeout + 15.0,
            )
            response.raise_for_status()
            value = response.json()
            return value.get("answer") if isinstance(value, dict) else {"cancelled": True}
        except Exception:
            return {"cancelled": True}

    def prompt(self, message, default=None):
        result = self._ask_prompt({
            "shape": "input",
            "message": str(message),
            "default": "" if default is None else str(default),
        })
        if isinstance(result, dict) and not result.get("cancelled"):
            return result.get("value", default)
        return default

    def confirm(self, message, default=True):
        result = self._ask_prompt({
            "shape": "confirm",
            "message": str(message),
            "default": bool(default),
        })
        if isinstance(result, dict) and not result.get("cancelled") and "value" in result:
            return bool(result["value"])
        return bool(default)

    def prompt_choices(self, message, choices, default=None):
        result = self._ask_prompt({
            "shape": "choices",
            "message": str(message),
            "choices": list(choices),
            "default": default,
        })
        if isinstance(result, dict) and not result.get("cancelled"):
            return result.get("value", default)
        return default

    def ask_user_choice(self, question, options, *, multi_select=False, description="", allow_other=False):
        result = self._ask_prompt({
            "shape": "choice",
            "question": str(question),
            "options": list(options),
            "multi_select": bool(multi_select),
            "description": str(description or ""),
            "allow_other": bool(allow_other),
        })
        if isinstance(result, dict) and not result.get("cancelled"):
            return {
                "selected": list(result.get("selected") or []),
                "other_text": str(result.get("other_text") or ""),
                "cancelled": False,
            }
        return {"selected": [], "other_text": "", "cancelled": True}

    def run_quiz(self, questions):
        result = self._ask_prompt({"shape": "quiz", "questions": list(questions or [])})
        if isinstance(result, dict) and not result.get("cancelled"):
            return dict(result.get("answers") or {})
        return {}

    def show_diff(self, filename, original_content, new_content):
        self.publish(
            {
                "kind": "diff",
                "filename": str(filename),
                "original": str(original_content or ""),
                "new": str(new_content or ""),
            }
        )


class SendRequest(BaseModel):
    session_name: str
    text: str
    provider: str
    model: str
    agent_mode: str = "default"
    system_instruction: str = "You are a helpful assistant."
    origin: str = "user"


class RuntimeRequest(BaseModel):
    session_name: str
    provider: str
    model: str
    agent_mode: str = "default"
    system_instruction: str = "You are a helpful assistant."

    def as_send_request(self) -> SendRequest:
        values = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        return SendRequest(text="__runtime_sync__", **values)


class InterruptRequest(BaseModel):
    session_name: str


app = FastAPI(title="MuCLI container worker", docs_url=None, redoc_url=None)
_sessions: dict[str, Any] = {}
_locks: dict[str, threading.Lock] = {}
_busy: dict[str, threading.Event] = {}
_threads: dict[str, int] = {}
# Round-18 F28: _sessions/_locks/_busy were mutated from concurrent HTTP
# handlers with no synchronization — two simultaneous first requests for
# the same session could both build a session and one would silently
# overwrite the other's cached state (and its per-session lock/busy event),
# while the sync/async endpoints could both pass the busy check and run
# overlapping turns on one history. One registry lock guards the
# check-build-insert; turn claims are then atomic (Event set inside the
# same critical section before any thread starts).
_registry_lock = threading.RLock()
_request_session_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mucli_worker_request_session", default=""
)


def _session_by_name(name: str | None = None):
    return _sessions.get(str(name or _request_session_name.get() or ""))


def _session_lock_for(name: str | None = None):
    resolved = str(name or _request_session_name.get() or "")
    return _locks.setdefault(resolved, threading.Lock())


app.state.session_by_name = _session_by_name
app.state.session_lock_for = _session_lock_for
app.state.is_container_worker = True


_MODE_API_PREFIXES = (
    "/api/feature",
    "/api/research",
    "/api/security",
    "/api/debug",
    "/api/loop",
    "/api/teacher",
    "/api/memory",
)


@app.middleware("http")
async def _authorize_mode_api(request: Request, call_next):
    """Authenticate and scope the GUI mode routers mounted in the worker."""
    if not any(
        request.url.path == prefix or request.url.path.startswith(prefix + "/")
        for prefix in _MODE_API_PREFIXES
    ):
        return await call_next(request)
    try:
        _authorize(request.headers.get("X-MuCLI-Worker-Token"))
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    session_name = str(request.query_params.get("session_name") or "").strip()
    if not session_name or session_name not in _sessions:
        return JSONResponse(
            status_code=412,
            content={"detail": "container session runtime is not loaded"},
        )
    token = _request_session_name.set(session_name)
    try:
        return await call_next(request)
    finally:
        _request_session_name.reset(token)


# The exact same domain routers power host workspaces and container workers.
# The host exposes them publicly; the worker copy is reachable only through
# the authenticated supervisor bridge above.
from mu.gui.routers import (  # noqa: E402  (app/state must exist first)
    debug as debug_router,
    feature as feature_router,
    loop as loop_router,
    memory as memory_router,
    research as research_router,
    security as security_router,
    teacher as teacher_router,
)

app.include_router(feature_router.router, prefix="/api/feature")
app.include_router(research_router.router, prefix="/api/research")
app.include_router(security_router.router, prefix="/api/security")
app.include_router(debug_router.router, prefix="/api/debug")
app.include_router(loop_router.router, prefix="/api/loop")
app.include_router(teacher_router.router, prefix="/api/teacher")
app.include_router(memory_router.router, prefix="/api/memory")


def _authorize(token: str | None) -> None:
    expected = os.getenv("MUCLI_WORKER_TOKEN", "")
    if not expected or not token or not __import__("hmac").compare_digest(expected, token):
        raise HTTPException(status_code=401, detail="invalid worker token")


def _proxy_readiness() -> tuple[bool, str]:
    raw = os.getenv("MUCLI_PROXY_URL", "").strip()
    if not raw:
        return True, "disabled"
    from urllib.parse import urlparse

    parsed = urlparse(raw)
    host = parsed.hostname or ""
    port = int(parsed.port or 3128)
    if not host:
        return False, "proxy URL has no host"
    try:
        with socket.create_connection((host, port), timeout=0.75):
            return True, f"{host}:{port}"
    except OSError as exc:
        return False, f"{host}:{port}: {exc}"


def _build_session(request: SendRequest):
    # Round-18 F28: the check-then-build-then-insert sequence ran without
    # any lock — two concurrent first requests for the same session both
    # built a session and the second insert silently replaced the first
    # (including its lock/busy event). The whole check/build/publish is
    # serialized under _registry_lock now. Build happens INSIDE the lock:
    # build_session is expensive but idempotent-per-name and the race it
    # prevents (duplicate cached sessions) is worse than holding a lock
    # across it.
    with _registry_lock:
        existing = _sessions.get(request.session_name)
        if existing is not None:
            _sync_request_context(existing, request)
            return existing
        from mu.gui.live_observability import register_live_observability_hooks
        from mucli import build_session

        register_live_observability_hooks()

        ui = WorkerBridgeUI(request.session_name)
        try:
            configured_workspaces = json.loads(os.getenv("MUCLI_WORKSPACES", "[\"/workspace\"]"))
        except (TypeError, ValueError):
            configured_workspaces = ["/workspace"]
        workspaces = [
            str(path) for path in configured_workspaces
            if isinstance(path, str) and os.path.isdir(path)
        ] or ["/workspace"]
        args = argparse.Namespace(
            session=request.session_name,
            provider=request.provider,
            model=request.model,
            provider_prevalidated=True,
            session_type="container",
            system=request.system_instruction,
            debug=False,
            workspace=workspaces,
            yolo=True,
            system_file=None,
            mode_prompt=[],
        )
        session = build_session(args, ui, allow_prompt=False)
        _sync_request_context(session, request)
        # Round-18 F29: CAS against the revision we loaded — the host GUI may
        # have written this session document while the worker was starting;
        # a plain save would silently clobber it (LWW).
        try:
            session.session_manager.save_history(
                session.folder_context,
                expected_revision=int(
                    getattr(session.session_manager, "revision", 0) or 0
                ),
            )
        except RevisionConflict:
            logger.warning(
                "session %s changed on disk during worker init; host copy wins",
                request.session_name,
            )
        session.sync_runtime_state()
        # Round-18 F28: still inside _registry_lock — the dicts are published
        # atomically as a group so no reader ever observes a session without
        # its lock/busy event.
        _sessions[request.session_name] = session
        _locks[request.session_name] = threading.Lock()
        _busy[request.session_name] = threading.Event()
        return session


def _sync_request_context(session: Any, request: SendRequest) -> None:
    """Apply host-controlled runtime state to a cached worker session.

    The host sends the selected strategy on every turn.  Without this seam a
    worker created in default mode kept running the default harness forever,
    even though web/mobile showed Feature, Security, or another mode as active.
    """
    mode = str(request.agent_mode or "default").strip().lower() or "default"
    session.variables["session_type"] = "container"
    session.variables["agent_mode"] = mode
    session.variables["yolo"] = True
    session.variables["strict_mode"] = False
    session.variables["plan_mode"] = False
    session.variables["lazy_tools_enabled"] = False
    session.variables["security_allow_secret_paths"] = False
    session.disabled_tools = []
    if getattr(session, "provider", None) is not None:
        session.provider.model_name = request.model
    session.system_instruction = request.system_instruction
    if getattr(session, "ui", None) is not None:
        session.ui.set_variables(session.variables)


def _run_turn(session, request: SendRequest) -> None:
    name = request.session_name
    busy = _busy[name]
    ui = session.ui
    # Round-18 F28: busy is claimed atomically by the endpoint before the
    # thread starts; setting it here again was a second, racy write.
    _threads[name] = threading.current_thread().ident or 0
    try:
        with _locks[name]:
            result = session.send_message(request.text, origin=request.origin)
            # Round-18 F29: CAS the turn save — a host GUI write between
            # our load and here must not be silently clobbered.
            try:
                session.session_manager.save_history(
                    session.folder_context,
                    expected_revision=int(
                        getattr(session.session_manager, "revision", 0) or 0
                    ),
                )
            except RevisionConflict:
                logger.warning(
                    "session %s: host wrote during worker turn; host copy wins",
                    name,
                )
        ui.publish(
            {
                "kind": "turn_complete",
                "result": {
                    "ok": bool(isinstance(result, dict) and result.get("ok", True)),
                    "status": result.get("status") if isinstance(result, dict) else None,
                    "error": result.get("error") if isinstance(result, dict) else None,
                },
            }
        )
    except KeyboardInterrupt:
        ui.publish({"kind": "turn_complete", "result": {"ok": False, "status": "interrupted"}})
    except Exception as exc:
        ui.publish({"kind": "error", "text": f"container turn failed: {exc}"})
        ui.publish({"kind": "turn_complete", "result": {"ok": False, "status": "error", "error": str(exc)}})
    finally:
        busy.clear()
        _threads.pop(name, None)


@app.get("/health")
def health(x_mucli_worker_token: str | None = Header(default=None)):
    _authorize(x_mucli_worker_token)
    proxy_ready, proxy_detail = _proxy_readiness()
    if not proxy_ready:
        raise HTTPException(status_code=503, detail=f"egress proxy unavailable: {proxy_detail}")
    return {
        "ok": True,
        "worker_protocol": WORKER_PROTOCOL_VERSION,
        "container_name": os.getenv("MUCLI_CONTAINER_NAME", ""),
        "sessions": sorted(_sessions),
        "busy": sorted(name for name, event in _busy.items() if event.is_set()),
        "proxy": proxy_detail,
    }


@app.post("/runtime/sync")
def sync_runtime(
    request: RuntimeRequest,
    x_mucli_worker_token: str | None = Header(default=None),
):
    """Load a session and synchronize the host-selected strategy/runtime."""
    _authorize(x_mucli_worker_token)
    try:
        session = _build_session(request.as_send_request())
    except Exception as exc:
        logger.exception("failed to synchronize container session %s", request.session_name)
        raise HTTPException(
            status_code=500,
            detail=f"worker session synchronization failed: {type(exc).__name__}: {exc}",
        ) from exc
    return {
        "ok": True,
        "session_name": request.session_name,
        "agent_mode": session.variables.get("agent_mode", "default"),
        "session_type": "container",
    }


def _assistant_text_since(session, start_index: int) -> str:
    history = getattr(session.session_manager, "history", []) or []
    for message in reversed(history[start_index:]):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        chunks = [
            str(part.get("text") or "")
            for part in (message.get("parts") or [])
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        ]
        if chunks:
            return "".join(chunks)
    return ""


@app.post("/send-sync")
def send_sync(request: SendRequest, x_mucli_worker_token: str | None = Header(default=None)):
    _authorize(x_mucli_worker_token)
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        session = _build_session(request)
    except Exception as exc:
        logger.exception("failed to initialise container session %s", request.session_name)
        raise HTTPException(
            status_code=500,
            detail=f"worker session initialisation failed: {type(exc).__name__}: {exc}",
        ) from exc
    # Round-18 F28: claim the turn atomically — check and set under the
    # registry lock so two concurrent sync requests cannot both pass the
    # busy gate and run overlapping turns on one history.
    with _registry_lock:
        if _busy[request.session_name].is_set():
            raise HTTPException(status_code=409, detail="session already has a turn in flight")
        _busy[request.session_name].set()
    name = request.session_name
    busy = _busy[name]
    _threads[name] = threading.current_thread().ident or 0
    start_index = len(session.session_manager.history)
    try:
        with _locks[name]:
            result = session.send_message(request.text, origin=request.origin)
            # Round-18 F29: CAS the turn save — a host GUI write between
            # our load and here must not be silently clobbered.
            try:
                session.session_manager.save_history(
                    session.folder_context,
                    expected_revision=int(
                        getattr(session.session_manager, "revision", 0) or 0
                    ),
                )
            except RevisionConflict:
                logger.warning(
                    "session %s: host wrote during worker turn; host copy wins",
                    name,
                )
        return jsonable_encoder({
            "ok": bool(not isinstance(result, dict) or result.get("status") != "error"),
            "session_name": name,
            "assistant_text": _assistant_text_since(session, start_index),
            "result": result if isinstance(result, dict) else {"status": "complete"},
        })
    except KeyboardInterrupt:
        return {"ok": False, "session_name": name, "assistant_text": "", "result": {"status": "interrupted"}}
    except Exception as exc:
        logger.exception("container turn failed for %s", request.session_name)
        detail = f"worker turn failed: {type(exc).__name__}: {exc}"
        logger.debug("worker traceback:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        busy.clear()
        _threads.pop(name, None)


@app.post("/send")
def send(request: SendRequest, x_mucli_worker_token: str | None = Header(default=None)):
    _authorize(x_mucli_worker_token)
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        session = _build_session(request)
    except Exception as exc:
        logger.exception("failed to initialise container session %s", request.session_name)
        raise HTTPException(
            status_code=500,
            detail=f"worker session initialisation failed: {type(exc).__name__}: {exc}",
        ) from exc
    # Round-18 F28: claim the turn atomically BEFORE starting the thread —
    # the old shape set busy inside _run_turn, so two requests could both
    # pass the gate and both spawn threads before either had set the flag.
    with _registry_lock:
        if _busy[request.session_name].is_set():
            raise HTTPException(status_code=409, detail="session already has a turn in flight")
        _busy[request.session_name].set()
    thread = threading.Thread(target=_run_turn, args=(session, request), daemon=True)
    thread.start()
    return {"accepted": True, "session_name": request.session_name}


@app.post("/interrupt")
def interrupt(request: InterruptRequest, x_mucli_worker_token: str | None = Header(default=None)):
    _authorize(x_mucli_worker_token)
    thread_id = _threads.get(request.session_name)
    if not thread_id:
        return {"ok": False, "detail": "No turn in flight."}
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(KeyboardInterrupt)
    )
    return {"ok": result == 1}


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("MUCLI_WORKER_PORT", "30312")),
        log_level=os.getenv("MUCLI_WORKER_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
