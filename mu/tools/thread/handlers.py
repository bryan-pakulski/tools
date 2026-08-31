"""Agent-visible operations for collaborating with workspace peer threads."""

from __future__ import annotations

from typing import Any

from mu.threads.coordinator import ThreadCoordinatorError
from mu.tools import tool


def _runtime(context):
    session = getattr(context, "session", None)
    coordinator = getattr(session, "thread_coordinator", None)
    meta = getattr(session, "thread_meta", None)
    if session is None or coordinator is None or meta is None:
        raise ThreadCoordinatorError("thread coordination is unavailable")
    return session, coordinator, meta


def _target(coordinator, value: Any) -> str:
    query = str(value or "").strip()
    if not query:
        raise ThreadCoordinatorError("target_thread_id is required")
    exact = coordinator.get_thread(query) or coordinator.get_thread_by_session(query)
    if exact is not None:
        return exact["thread_id"]
    matches = [
        item for item in coordinator.list_threads()
        if str(item.get("title") or "").casefold() == query.casefold()
    ]
    if len(matches) == 1:
        return matches[0]["thread_id"]
    raise ThreadCoordinatorError("target thread was not found or was ambiguous")


def _ok(message: str, data: Any) -> dict[str, Any]:
    return {"ok": True, "message": message, "data": data}


def _error(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": "thread_coordination_error",
        "message": str(exc),
        "data": {},
    }


@tool(
    name="list_threads",
    description=(
        "List all peer agent threads in this thread group, including live "
        "status, current goal, unread count, and claimed paths."
    ),
    parameters={"type": "object", "properties": {}},
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def list_threads(_args, context):
    try:
        _session, coordinator, meta = _runtime(context)
        return _ok(
            "Thread roster loaded.",
            {"current_thread_id": meta.thread_id, "threads": coordinator.list_threads()},
        )
    except Exception as exc:
        return _error(exc)


@tool(
    name="get_thread_activity",
    description=(
        "Read the durable, secret-scrubbed inter-thread audit timeline. Use "
        "after_id to incrementally inspect messages, claims, conflicts, and status."
    ),
    parameters={
        "type": "object",
        "properties": {
            "after_id": {"type": "integer", "default": 0},
            "limit": {"type": "integer", "default": 100},
        },
    },
    requires_approval=False,
    execution_kind="read",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def get_thread_activity(args, context):
    try:
        _session, coordinator, _meta = _runtime(context)
        events = coordinator.activity(
            after_id=int(args.get("after_id", 0) or 0),
            limit=int(args.get("limit", 100) or 100),
        )
        return _ok("Thread activity loaded.", {"events": events})
    except Exception as exc:
        return _error(exc)


@tool(
    name="send_thread_message",
    description=(
        "Send a durable message to a peer thread. Idle peers are automatically "
        "woken; busy peers receive it in live L3 coordination context."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_thread_id": {
                "type": "string",
                "description": "Peer thread ID, exact session name, or unique title.",
            },
            "content": {"type": "string"},
            "reply_to": {"type": "string"},
            "related_paths": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["target_thread_id", "content"],
    },
    requires_approval=False,
    execution_kind="coordination",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def send_thread_message(args, context):
    try:
        _session, coordinator, meta = _runtime(context)
        target = _target(coordinator, args.get("target_thread_id"))
        message = coordinator.send_message(
            meta.thread_id,
            target,
            args.get("content", ""),
            reply_to=str(args.get("reply_to") or ""),
            related_paths=args.get("related_paths") or [],
        )
        return _ok("Message sent; the peer will be woken if idle.", message)
    except Exception as exc:
        return _error(exc)


@tool(
    name="acknowledge_thread_message",
    description="Acknowledge and resolve an incoming peer message when no reply is needed.",
    parameters={
        "type": "object",
        "properties": {"message_id": {"type": "string"}},
        "required": ["message_id"],
    },
    requires_approval=False,
    execution_kind="coordination",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def acknowledge_thread_message(args, context):
    try:
        _session, coordinator, meta = _runtime(context)
        acknowledged = coordinator.acknowledge_message(
            meta.thread_id, str(args.get("message_id") or "")
        )
        if not acknowledged:
            raise ThreadCoordinatorError("message is not an open message for this thread")
        return _ok("Message acknowledged.", {"message_id": args.get("message_id")})
    except Exception as exc:
        return _error(exc)


@tool(
    name="wait_for_thread_reply",
    description=(
        "Wait for a direct reply to a previously sent peer message. Prefer "
        "continuing independent work when possible."
    ),
    parameters={
        "type": "object",
        "properties": {
            "message_id": {"type": "string"},
            "timeout_seconds": {"type": "number", "default": 120},
        },
        "required": ["message_id"],
    },
    requires_approval=False,
    execution_kind="coordination",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def wait_for_thread_reply(args, context):
    try:
        session, coordinator, meta = _runtime(context)
        coordinator.set_status(
            meta.thread_id,
            "waiting_peer",
            runtime_id=getattr(session, "_thread_runtime_id", ""),
        )
        try:
            reply = coordinator.wait_for_reply(
                str(args.get("message_id") or ""),
                timeout=float(args.get("timeout_seconds", 120) or 120),
            )
        finally:
            coordinator.set_status(
                meta.thread_id,
                "running",
                runtime_id=getattr(session, "_thread_runtime_id", ""),
            )
        if reply is None:
            return _ok("No reply arrived before the timeout.", {"reply": None})
        return _ok("Peer replied.", {"reply": reply})
    except Exception as exc:
        return _error(exc)


@tool(
    name="claim_thread_paths",
    description=(
        "Explicitly reserve files or directories before a coordinated edit. "
        "Native file writes also acquire turn-scoped claims automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        },
        "required": ["paths"],
    },
    requires_approval=False,
    execution_kind="coordination",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def claim_thread_paths(args, context):
    try:
        session, coordinator, meta = _runtime(context)
        result = coordinator.claim_paths(
            meta.thread_id,
            args.get("paths") or [],
            turn_id=str(getattr(session, "_thread_turn_id", "") or ""),
            note=str(args.get("note") or ""),
            explicit=True,
            ttl=3600,
        )
        if not result["ok"]:
            return {
                "ok": False,
                "error_code": "thread_path_conflict",
                "message": "One or more paths are owned by a peer. Coordinate before editing.",
                "data": result,
            }
        return _ok("Paths claimed.", result)
    except Exception as exc:
        return _error(exc)


@tool(
    name="release_thread_paths",
    description="Release this thread's active path claims when a peer may proceed.",
    parameters={
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}},
        },
    },
    requires_approval=False,
    execution_kind="coordination",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def release_thread_paths(args, context):
    try:
        _session, coordinator, meta = _runtime(context)
        count = coordinator.release_paths(meta.thread_id, args.get("paths") or None)
        return _ok("Path claims released.", {"released": count})
    except Exception as exc:
        return _error(exc)


@tool(
    name="handoff_thread_paths",
    description="Transfer this thread's active path claims to a peer after coordination.",
    parameters={
        "type": "object",
        "properties": {
            "target_thread_id": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        },
        "required": ["target_thread_id", "paths"],
    },
    requires_approval=False,
    execution_kind="coordination",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def handoff_thread_paths(args, context):
    try:
        _session, coordinator, meta = _runtime(context)
        target = _target(coordinator, args.get("target_thread_id"))
        count = coordinator.handoff_paths(
            meta.thread_id,
            target,
            args.get("paths") or [],
            note=str(args.get("note") or ""),
        )
        return _ok("Path claims handed off.", {"transferred": count, "target": target})
    except Exception as exc:
        return _error(exc)


@tool(
    name="request_thread_claim_override",
    description=(
        "Ask the human to override an unresolved peer path claim. This is an "
        "emergency action: it always requires explicit human approval, even "
        "in YOLO or container mode, and cannot be self-approved."
    ),
    parameters={
        "type": "object",
        "properties": {
            "conflict_id": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["conflict_id", "rationale"],
    },
    requires_approval=True,
    approval_policy="always_human",
    execution_kind="mutate",
    preview_policy="none",
    server_policy="session_only",
    group="thread",
)
def request_thread_claim_override(args, context):
    try:
        _session, coordinator, meta = _runtime(context)
        result = coordinator.override_conflict(
            str(args.get("conflict_id") or ""),
            meta.thread_id,
            str(args.get("rationale") or ""),
        )
        return _ok("Human-approved claim override applied.", result)
    except Exception as exc:
        return _error(exc)

