"""Per-session usage tracker — tool counts, latencies, skill invocations.

Two hooks:

  * **pre_tool**  — stamps `metadata["_usage_start_ts"]` so the post
    hook can compute elapsed time. Also prints a visible banner when
    `invoke_skill` fires so the user can see exactly when a skill is
    being applied (per the user request: "highlight it in the text
    to show that it is being used"). The same banner is printed for
    trigger-regex auto-expansion from `mu/session/session.py`; the
    shared renderer lives in `mu.skills.announce_skill`.

  * **post_tool** — increments the per-tool counters on
    `session.tool_stats`, records last-used timestamp, and (for
    `invoke_skill`) bumps the per-skill counter so `/stats` can show
    which skills the model has been pulling on.

All updates are defensive: any exception in the tracker must not break
the agent loop. Worst case, /stats is stale for a turn.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

from .hooks import HookContext, default_registry


logger = logging.getLogger("mucli")

_ARG_PREVIEW_LIMIT = 80

# Round-20 F9: parallel tool execution (mu.agent.parallel.execute_calls)
# runs post_tool hooks concurrently on the SAME session object. The
# stats mutation below is a read-modify-write chain (setdefault,
# `bucket["count"] += 1`, `total_ms +=`) — without a lock, concurrent
# hooks lose increments. A single module-level lock is sufficient: the
# critical section is microseconds, and the lock deliberately does NOT
# live on the session because `tool_stats` is persisted with the
# session document (a Lock is not picklable).
_stats_lock = threading.Lock()


def _arg_preview(args: Any) -> str:
    """Compact one-line representation of tool args, for the
    last-used-with display in /stats."""
    if not args:
        return ""
    try:
        text = json.dumps(args, default=str)
    except Exception:
        text = str(args)
    text = text.replace("\n", " ")
    if len(text) > _ARG_PREVIEW_LIMIT:
        text = text[: _ARG_PREVIEW_LIMIT - 1] + "…"
    return text


def _ensure_stats(session: Any) -> Optional[Dict[str, Any]]:
    stats = getattr(session, "tool_stats", None)
    if not isinstance(stats, dict):
        # Older sessions deserialized before this field existed —
        # initialize on demand so /stats doesn't crash.
        session.tool_stats = {
            "session_started_at": time.time(),
            "first_call_at": None,
            "last_call_at": None,
            "tools": {},
            "skills": {},
            "approvals": {"approved": 0, "denied": 0},
            "errors": {},
        }
        stats = session.tool_stats
    return stats


def _skill_banner(ui: Any, skill_name: str) -> None:
    """Print a visible banner for an `invoke_skill` activation.

    Delegates to the shared `mu.skills.announce_skill` so the
    `invoke_skill` tool call and trigger-regex auto-expansion share one
    consistent highlight. `via` is left as None here so the invoke_skill
    banner text stays exactly as before ("🎯 SKILL ACTIVE: <name>")."""
    from mu.skills import announce_skill

    announce_skill(ui, skill_name)


@default_registry.register("pre_tool", name="usage_tracker_pre", priority=50)
def usage_tracker_pre(ctx: HookContext) -> Optional[object]:
    """Stamp a monotonic start time + (for invoke_skill) print a banner."""
    try:
        ctx.metadata["_usage_start_ts"] = time.monotonic()
        if ctx.tool_name == "invoke_skill":
            args = ctx.tool_args or {}
            skill_name = str(args.get("name") or "").strip()
            # Round-20 F9: the banner write is a UI print, not a stats
            # mutation — keep it OUTSIDE the lock (concurrent banners
            # interleaving on the render queue are harmless, and
            # announce_skill may do blocking I/O).
            _skill_banner(getattr(ctx.session, "ui", None), skill_name)
    except Exception:  # pragma: no cover — defensive
        logger.debug("usage_tracker_pre hook failed", exc_info=True)
    return None


@default_registry.register("post_tool", name="usage_tracker_post", priority=200)
def usage_tracker_post(ctx: HookContext) -> Optional[object]:
    """Update `session.tool_stats` with the result of this tool call."""
    try:
        session = ctx.session
        if session is None:
            return None
        tool_name = str(ctx.tool_name or "")
        if not tool_name:
            return None

        elapsed_ms: float = 0.0
        start_ts = ctx.metadata.get("_usage_start_ts")
        if isinstance(start_ts, (int, float)):
            elapsed_ms = max(0.0, (time.monotonic() - float(start_ts)) * 1000.0)

        # Was this call a success? Tool envelope `ok=False` counts as failure.
        result = ctx.tool_result
        ok = True
        error_code: Optional[str] = None
        if isinstance(result, dict):
            ok = bool(result.get("ok", True))
            if not ok:
                error_code = str(result.get("error_code") or "unknown")

        args_preview = _arg_preview(ctx.tool_args)
        now = time.time()
        with _stats_lock:
            # Round-20 F9: entire read-modify-write of session.tool_stats
            # is atomic — parallel tool execution runs this hook
            # concurrently for the same session.
            stats = _ensure_stats(session)
            if stats is None:
                return None

            if stats["first_call_at"] is None:
                stats["first_call_at"] = now
            stats["last_call_at"] = now

            tools = stats.setdefault("tools", {})
            bucket = tools.setdefault(
                tool_name,
                {
                    "count": 0,
                    "success": 0,
                    "failed": 0,
                    "total_ms": 0.0,
                    "last_used_at": None,
                    "last_args": "",
                },
            )
            bucket["count"] += 1
            bucket["total_ms"] += elapsed_ms
            bucket["last_used_at"] = now
            bucket["last_args"] = args_preview
            if ok:
                bucket["success"] += 1
            else:
                bucket["failed"] += 1
                if error_code:
                    err_bucket = stats.setdefault("errors", {})
                    err_bucket[error_code] = err_bucket.get(error_code, 0) + 1

            if tool_name == "invoke_skill":
                args = ctx.tool_args or {}
                skill_name = str(args.get("name") or "").strip()
                if skill_name:
                    skills = stats.setdefault("skills", {})
                    sk_bucket = skills.setdefault(
                        skill_name, {"invocations": 0, "last_used_at": None}
                    )
                    sk_bucket["invocations"] += 1
                    sk_bucket["last_used_at"] = now
    except Exception:  # pragma: no cover — defensive
        logger.debug("usage_tracker_post hook failed", exc_info=True)
    return None


__all__ = [
    "usage_tracker_pre",
    "usage_tracker_post",
]
