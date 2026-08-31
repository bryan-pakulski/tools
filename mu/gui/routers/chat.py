"""Chat send + SSE event stream.

Multi-session: each chat send names the target session (default: the
currently focused one). Lock and busy event are per-session so two
sessions can run turns in parallel without blocking each other.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
from utils.logger import logger
from utils.revision import js_safe_revision

router = APIRouter()
events_router = APIRouter()

_agent_threads: Dict[str, int] = {}


def _resolve_session(request: Request, name: Optional[str]):
    """Resolve a session by name or fall back to the focused one.

    Returns the Session object or raises 412.
    """
    session = request.app.state.session_by_name(name)
    if session is None:
        raise HTTPException(
            status_code=412,
            detail=(
                f"Session {name!r} is not loaded."
                if name
                else "No session loaded. Load or create a session first."
            ),
        )
    return session


def _run_send(
    session,
    text: str,
    *,
    lock: threading.Lock,
    busy: threading.Event,
    session_name: str = "",
    origin: str = "user",
):
    _agent_threads[session_name] = threading.current_thread().ident
    busy.set()
    try:
        with lock:
            try:
                result = session.send_message(text, origin=origin)
            except KeyboardInterrupt:
                result = {"status": "interrupted", "error": "User interrupted execution."}
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
            try:
                # Phase-6 r21 F5: CAS against this manager's own
                # revision. A concurrent surface write that landed after
                # the final turn save must stay authoritative — plain
                # save_history here would clobber it with stale
                # in-memory state once the turn CAS is disarmed.
                session.session_manager.save_history_if_current(
                    session.folder_context
                )
            except Exception:
                # Defensive: best-effort path must not break the caller.
                logger.debug("Suppressed exception", exc_info=True)
            return result
    finally:
        busy.clear()
        _agent_threads.pop(session_name, None)


def _artifact_snapshot(session) -> dict[str, dict[str, Any]]:
    """Return persisted artifacts keyed by id without affecting a turn."""
    registry = getattr(session, "artifact_registry", None)
    if registry is None:
        try:
            session.sync_runtime_state()
            registry = getattr(session, "artifact_registry", None)
        except Exception:
            registry = None
    if registry is None:
        return {}
    try:
        return {
            str(item.get("artifact_id")): dict(item)
            for item in registry.list()
            if isinstance(item, dict) and item.get("artifact_id")
        }
    except Exception:
        return {}


async def _replay_new_artifacts(bus, session, session_name: str, before: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay artifacts persisted by a worker even if its callback was missed."""
    after = _artifact_snapshot(session)
    created = [item for artifact_id, item in after.items() if artifact_id not in before]
    created.sort(key=lambda item: float(item.get("created_at", 0) or 0))
    for artifact in created:
        await bus.publish({
            "kind": "artifact_created",
            "artifact": artifact,
            "session_name": session_name,
        })
    return created


@router.get("/commands")
async def list_commands_endpoint():
    from mu.commands import list_commands

    specs = list_commands()
    return {
        "commands": [
            {"names": list(s.names), "help": s.help}
            for s in specs
        ]
    }


@router.get("/completions")
async def completions_endpoint(request: Request, kind: str = ""):
    """Return dynamic completion lists for subcommand arguments.

    Query param ``kind`` selects which list to return:
      sessions, features, tools, models, modes, variables, skills, docs
    """
    if kind == "sessions":
        import glob as _glob
        import os

        from utils.config import HISTORY_DIR

        sessions = []
        pattern = os.path.join(HISTORY_DIR, "sessions", "*", "session.json")
        for path in _glob.glob(pattern):
            sessions.append(os.path.basename(os.path.dirname(path)))
        return {"items": sorted(set(sessions))}

    if kind == "features":
        import glob as _glob
        import os

        from utils.config import HISTORY_DIR

        ids: set = set()
        for path in _glob.glob(
            os.path.join(HISTORY_DIR, "sessions", "*", "features", "*.json")
        ):
            try:
                import json as _json

                with open(path, "r", encoding="utf-8") as fh:
                    fid = str(_json.load(fh).get("feature_id", "")).strip()
                    if fid:
                        ids.add(fid)
            except Exception:
                continue
        return {"items": sorted(ids)}

    if kind == "tools":
        try:
            from mu.tools.descriptors import TOOLS

            names = sorted({t.name for t in TOOLS if getattr(t, "name", "")})
        except Exception:
            names = []
        return {"items": names}

    if kind == "models":
        try:
            from utils.config import KNOWN_MODELS

            return {"items": list(KNOWN_MODELS)}
        except Exception:
            return {"items": []}

    if kind == "modes":
        try:
            from utils.config import AGENT_MODE_METADATA

            return {"items": sorted(AGENT_MODE_METADATA.keys())}
        except Exception:
            return {"items": ["default"]}

    if kind == "variables":
        session = request.app.state.session_by_name()
        if session is None:
            return {"items": []}
        return {"items": sorted(session.variables.keys())}

    if kind == "skills":
        try:
            from mu.skills import discover_skills

            names = sorted({s.name for s in discover_skills([])})
        except Exception:
            names = []
        return {"items": names}

    if kind == "docs":
        try:
            from mu.commands.docs import list_doc_names

            return {"items": list_doc_names()}
        except Exception:
            return {"items": []}

    if kind == "memory_targets":
        try:
            from mu.commands.memory import LIST_TARGETS

            return {"items": list(LIST_TARGETS)}
        except Exception:
            return {"items": ["all", "task", "scratchpad",
                              "L1A", "L1A", "L1B", "L2", "L3", "L4", "L4B", "L5"]}

    if kind == "layer_ids":
        try:
            from mu.commands.variables import LAYER_BUDGET_VARS

            return {"items": list(LAYER_BUDGET_VARS.keys())}
        except Exception:
            return {"items": ["L1A", "L1B", "L2", "L3", "L4", "L4B"]}

    return {"items": []}


def _resolve_attachments(session, raw_ids: Any) -> list[dict[str, Any]]:
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, list):
        raise HTTPException(status_code=400, detail="attachment_ids must be an array")
    values = []
    seen = set()
    registry = getattr(session, "attachment_registry", None)
    if registry is None:
        session.sync_runtime_state()
        registry = getattr(session, "attachment_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="attachment registry unavailable")
    for raw_id in raw_ids[:20]:
        attachment_id = str(raw_id or "").strip()
        if not attachment_id or attachment_id in seen:
            continue
        descriptor = registry.get(attachment_id)
        if descriptor is None or registry.resolve_path(attachment_id) is None:
            raise HTTPException(status_code=404, detail=f"attachment not found: {attachment_id}")
        seen.add(attachment_id)
        values.append(dict(descriptor))
    return values


def _attachment_notice(attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return ""
    lines = ["[Attachments selected for this message; retrieve via attachment tools:]"]
    for item in attachments:
        lines.append(
            f"- id={item.get('attachment_id')} name={item.get('name')} mime={item.get('mime_type')}"
        )
    return "\n".join(lines)


@router.post("/send")
async def send_message(request: Request, payload: Dict[str, Any]):
    session_name = (payload.get("session_name") or "").strip() or None
    session = _resolve_session(request, session_name)
    name = session.session_manager.current_session_name

    busy = request.app.state.session_busy_for(name)
    text = str(payload.get("text") or "").strip()
    attachments = _resolve_attachments(session, payload.get("attachment_ids"))
    if not text and not attachments:
        raise HTTPException(status_code=400, detail="text or attachment_ids is required")
    if not text:
        text = "Please review the attached document(s)."
    if text.startswith("/") and attachments:
        raise HTTPException(status_code=400, detail="attachments cannot be sent with slash commands")
    session_type = str(session.variables.get("session_type", "workspace") or "workspace").lower()

    # Commands are deliberately permitted while the model works: they are
    # operational controls (for example /status or /interrupt), whereas a
    # second natural-language turn would race the active agent loop.
    if busy.is_set() and not text.startswith("/"):
        raise HTTPException(
            status_code=409,
            detail=f"Session {name!r} already has a turn in flight.",
        )

    if attachments and session_type != "container":
        session.stage_attachment_ids(
            [item["attachment_id"] for item in attachments]
        )

    bus = request.app.state.bus
    # Echo the user's message to the per-session stream so the browser
    # can render it immediately without waiting for the agent loop.
    await bus.publish(
        {"kind": "user_message", "text": text, "attachments": attachments, "session_name": name}
    )

    if text.startswith("/"):
        from mucli import handle_command

        lock = request.app.state.session_lock_for(name)

        def _run_cmd():
            with lock:
                return handle_command(session, text, allow_prompt=False)

        result = await asyncio.to_thread(_run_cmd)
        await bus.publish(
            {"kind": "command_result", "result": result, "session_name": name}
        )
        return {"accepted": True, "kind": "command", "session_name": name}

    if session_type == "container":
        busy.set()

        async def _drive_container() -> None:
            artifacts_before = _artifact_snapshot(session)
            try:
                response = await asyncio.to_thread(
                    request.app.state.container_supervisor.send_sync,
                    name,
                    text + (("\n\n" + _attachment_notice(attachments)) if attachments else ""),
                    provider=session.provider.name,
                    model=session.provider.model_name,
                    agent_mode=str(session.variables.get("agent_mode", "default")),
                    system_instruction=session.system_instruction,
                    timeout=None,
                )
                result = response.get("result") if isinstance(response, dict) else None
                # The worker writes through the mounted session directory.
                # Refresh the host mirror before notifying panels/history.
                try:
                    session.session_manager._load_session(name)
                    session.sync_runtime_state()
                except Exception:
                    # Defensive: best-effort path must not break the caller.
                    logger.debug("Suppressed exception", exc_info=True)
                new_artifacts = await _replay_new_artifacts(
                    bus, session, name, artifacts_before
                )
                # Clear the authoritative host busy flag before terminal events
                # reach reconnecting clients; otherwise a status poll can race
                # the event and put mobile back into "thinking".
                busy.clear()
                if isinstance(result, dict) and result.get("status") == "error":
                    await bus.publish(
                        {
                            "kind": "error",
                            "text": str(result.get("error") or "Container turn failed."),
                            "session_name": name,
                        }
                    )
                await bus.publish(
                    {
                        "kind": "turn_complete",
                        "result": {
                            **_summarize_result(result),
                            "artifacts": new_artifacts,
                        },
                        "session_name": name,
                    }
                )
                await bus.publish(
                    {"kind": "history_refresh", "session_name": name}
                )
            except Exception as exc:
                # A tool may have published an artifact before a later provider
                # or transport failure. Surface it even on failed turns.
                try:
                    session.session_manager._load_session(name)
                    session.sync_runtime_state()
                except Exception:
                    # Defensive: best-effort path must not break the caller.
                    logger.debug("Suppressed exception", exc_info=True)
                new_artifacts = await _replay_new_artifacts(
                    bus, session, name, artifacts_before
                )
                error_text = f"container send failed: {exc}"
                busy.clear()
                await bus.publish(
                    {
                        "kind": "error",
                        "text": error_text,
                        "session_name": name,
                    }
                )
                # Error events normally clear clients immediately. A terminal
                # event also protects reconnecting web/mobile clients from
                # remaining indefinitely busy after a worker failure.
                await bus.publish(
                    {
                        "kind": "turn_complete",
                        "result": {
                            "ok": False,
                            "status": "error",
                            "error": error_text,
                            "artifacts": new_artifacts,
                        },
                        "session_name": name,
                    }
                )
                await bus.publish({"kind": "history_refresh", "session_name": name})
            finally:
                busy.clear()

        asyncio.create_task(_drive_container())
        return {
            "accepted": True,
            "kind": "container",
            "session_name": name,
            # Round-31 F35: JS-safe If-Match token for optimistic concurrency.
            "revision": js_safe_revision(getattr(session.session_manager, "revision", 0) or 0),
        }

    lock = request.app.state.session_lock_for(name)

    def _run():
        return _run_send(session, text, lock=lock, busy=busy, session_name=name)

    async def _drive():
        try:
            result = await asyncio.to_thread(_run)
            # Round-31 F35: publish the post-turn revision (save_history
            # inside the turn already bumped it) so clients capture a
            # fresh If-Match token without an extra state poll.
            await bus.publish(
                {
                    "kind": "turn_complete",
                    "result": _summarize_result(result),
                    "session_name": name,
                    "revision": js_safe_revision(
                        getattr(session.session_manager, "revision", 0) or 0
                    ),
                }
            )
        except Exception as exc:
            await bus.publish(
                {"kind": "error", "text": f"send failed: {exc}", "session_name": name}
            )

    asyncio.create_task(_drive())
    return {
        "accepted": True,
        "kind": "chat",
        "session_name": name,
        # Round-31 F35: JS-safe If-Match token for optimistic concurrency.
        "revision": js_safe_revision(getattr(session.session_manager, "revision", 0) or 0),
    }


@router.post("/interrupt")
async def interrupt(request: Request, payload: Optional[Dict[str, Any]] = None):
    session_name = None
    if payload:
        session_name = (payload.get("session_name") or "").strip() or None
    session = _resolve_session(request, session_name)
    name = session.session_manager.current_session_name

    session_type = str(session.variables.get("session_type", "workspace") or "workspace").lower()
    if session_type == "container":
        try:
            result = await asyncio.to_thread(
                request.app.state.container_supervisor.interrupt, name
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    tid = _agent_threads.get(name)
    if tid is None:
        return {"ok": False, "detail": "No turn in flight for this session."}

    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt)
    )
    return {"ok": res == 1}


@router.get("/history/search")
async def search_history(
    request: Request,
    query: str = "",
    role: Optional[str] = None,
    tool_name: Optional[str] = None,
    max_results: int = 20,
    session_name: Optional[str] = None,
):
    """Search conversation history for matching messages.

    Read-only endpoint that calls session.search_history() and returns
    JSON results. Accepts query, role, tool_name, max_results params.
    """
    query = (query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    session = _resolve_session(request, session_name)
    sm = session.session_manager
    results = sm.search_history(
        query=query,
        role=role or None,
        tool_name=tool_name or None,
        max_results=max_results,
    )
    return results


def _summarize_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"ok": False}
    return {
        "ok": result.get("ok", False),
        "status": result.get("status"),
        "tokens": result.get("session_totals") or result.get("tokens"),
        "error": result.get("error"),
    }


@events_router.get("/api/events")
async def stream_events(request: Request, session: str | None = None):
    """SSE event stream.

    With ?session=NAME the stream is server-side filtered to that
    session's events (and session-agnostic events) — a tab viewing one
    session no longer receives other sessions' assistant text, tool
    results, or approval prompts (codex round-6 F4). Without the
    parameter behavior is unchanged (loopback single-user default).

    Round-47 F2: every event carries a monotonic ``seq``; the stream
    sends it as the SSE ``id:`` field, and a reconnecting browser's
    ``Last-Event-ID`` header replays the missed events from the bus's
    bounded ring instead of silently skipping them.
    """
    bus = request.app.state.bus
    last_id = None
    raw_last = request.headers.get("last-event-id")
    if raw_last:
        try:
            last_id = int(raw_last)
        except ValueError:
            last_id = None
    thread_group_id = None
    if session:
        try:
            from mu.session.manager import SessionManager
            from mu.threads.model import ensure_thread_meta

            _data = SessionManager().read_session_data(session)
            if isinstance(_data, dict):
                thread_group_id = ensure_thread_meta(
                    session, _data.get("thread_meta")
                ).group_id
        except Exception:
            thread_group_id = None
    queue = bus.subscribe(
        session_name=session,
        thread_group_id=thread_group_id,
        last_event_id=last_id,
    )

    async def generator():
        try:
            busy_names = [
                n for n, evt in request.app.state.session_busy.items()
                if evt.is_set()
            ]
            yield {"event": "message", "data": json.dumps({
                "kind": "hello",
                "busy": busy_names,
            })}
            for pending in request.app.state.prompts.pending():
                if session and pending.get("session_name") not in (None, session):
                    continue
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "kind": "prompt",
                            "id": pending["id"],
                            "prompt": pending,
                            "session_name": pending.get("session_name"),
                        }
                    ),
                }
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                # Round-47 F2 + Round-48 F1: seq rides as the SSE id so
                # EventSource reconnects replay from Last-Event-ID. The
                # queue holds the SAME dict the replay ring and other tabs
                # reference — never pop from it; serialize a copy.
                seq = event.get("seq")
                payload = {k: v for k, v in event.items() if k != "seq"}
                yield {
                    "event": "message",
                    "id": str(seq) if seq is not None else None,
                    "data": json.dumps(payload),
                }
        finally:
            bus.unsubscribe(queue)

    return EventSourceResponse(generator())


@router.get("/sessions/{name}/cache/{key}")
async def recall_cached_tool_result(request: Request, name: str, key: str):
    """Recall a full tool result from the session's ToolResultCache by key.

    The GUI fires this when the user clicks a tool_result trace event that
    has a cache_key — the L5 history only carries a compact ref, so the popup
    fetches the full content on demand via this endpoint instead of keeping
    it in the provider context. Falls back to the durable ResultStore on a
    memory-miss (the cache's recall() handles that transparently).
    """
    session = _resolve_session(request, name)
    cache = getattr(session, "tool_result_cache", None)
    if cache is None:
        raise HTTPException(status_code=404, detail="No tool result cache on this session.")
    entry = cache.recall(key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No cached result for key {key!r}.")
    return {
        "ok": True,
        "cache_key": key,
        "tool_name": entry.get("tool_name", ""),
        "result": entry.get("result"),
        "from_disk": bool(entry.get("from_disk", False)),
    }
