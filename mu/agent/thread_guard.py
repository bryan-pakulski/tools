"""Turn-scoped ownership gate for native workspace file mutations."""

from __future__ import annotations

import os
from typing import Iterable

from mu.tools._envelope import _build_tool_envelope

from .hooks import HookContext, HookRegistry, HookResult, HookSpec, default_registry


_NATIVE_WRITE_TOOLS = frozenset(
    {"write_file", "apply_diff", "search_and_replace_file"}
)


def _write_paths(tool_name: str, args: dict) -> Iterable[str]:
    if tool_name in _NATIVE_WRITE_TOOLS:
        value = args.get("filename") or args.get("file")
        if value:
            yield str(value)
        return
    if tool_name == "batch_job":
        for command in args.get("commands") or []:
            if not isinstance(command, dict):
                continue
            yield from _write_paths(
                str(command.get("tool_name") or ""),
                command.get("tool_args") or {},
            )


def _absolute_for_session(session, path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.realpath(expanded)
    folders = list(getattr(getattr(session, "folder_context", None), "folders", []) or [])
    base = folders[0] if folders else os.getcwd()
    return os.path.realpath(os.path.join(base, expanded))


def _guard(ctx: HookContext):
    paths = list(_write_paths(str(ctx.tool_name or ""), ctx.tool_args or {}))
    if not paths:
        return None
    session = ctx.session
    coordinator = getattr(session, "thread_coordinator", None)
    meta = getattr(session, "thread_meta", None)
    if meta is None:
        # Lightweight legacy/test facades with no thread identity keep their
        # established behavior.
        return None
    if coordinator is None:
        if not bool(getattr(session, "_thread_coordination_required", False)):
            return None
        payload = _build_tool_envelope(
            tool_name=str(ctx.tool_name or "thread_write_guard"),
            ok=False,
            error_code="thread_coordination_unavailable",
            message=(
                "Workspace writes are blocked because the thread coordination "
                "journal is unavailable; retry after restoring its storage."
            ),
            data={"error": str(getattr(session, "_thread_coordination_error", ""))},
            retryable=True,
        )
        return HookResult(
            action="short_circuit",
            payload=payload,
            data={"reason": "thread_coordination_unavailable"},
        )
    absolute = [_absolute_for_session(session, path) for path in paths]
    result = coordinator.claim_paths(
        meta.thread_id,
        absolute,
        turn_id=str(getattr(session, "_thread_turn_id", "") or ""),
        note=f"automatic claim for {ctx.tool_name}",
        ttl=3600,
    )
    if result.get("ok"):
        ctx.metadata["thread_claims"] = result.get("claims", [])
        return None

    for conflict in result.get("conflicts", []):
        if not conflict.get("new", True):
            continue
        try:
            coordinator.send_message(
                meta.thread_id,
                conflict["owner_thread_id"],
                (
                    "Path ownership conflict: I need to edit "
                    f"{conflict['path']}. Please reply with your intended changes "
                    "and release or hand off the claim when safe."
                ),
                related_paths=[conflict["path"]],
                kind="path_conflict",
            )
        except Exception:
            # The conflict remains durable and visible even if notification
            # delivery itself fails.
            pass
    payload = _build_tool_envelope(
        tool_name=str(ctx.tool_name or "thread_write_guard"),
        ok=False,
        error_code="thread_path_conflict",
        message=(
            "A peer thread owns one or more target paths. Coordinate using "
            "send_thread_message, wait_for_thread_reply, and claim handoff. "
            "Only request_thread_claim_override may bypass it, and that "
            "always requires human approval."
        ),
        data=result,
        retryable=False,
    )
    return HookResult(
        action="short_circuit",
        payload=payload,
        data={"reason": "thread_path_conflict"},
    )


def install(registry: HookRegistry | None = None) -> None:
    reg = registry or default_registry
    reg.remove("thread_path_claim_guard")
    reg.add(
        HookSpec(
            name="thread_path_claim_guard",
            point="pre_tool",
            # Plan mode rejects all mutations first; ownership is relevant
            # only to executable writes.
            priority=15,
            handler=_guard,
        )
    )


install()


__all__ = ["install"]
