"""Transport-neutral live observability hooks for GUI-capable sessions.

Normal workspace sessions use :class:`mu.gui.web_ui.WebUI`; container
sessions use :class:`mu.container.worker.WorkerBridgeUI`.  Both expose the
small ``publish_event`` protocol below, so context and sub-agent telemetry is
captured at the provider boundary regardless of where the agent loop runs.
"""

from __future__ import annotations

import logging
from typing import Any

from mu.agent.hooks import HookContext, default_registry

from .memory_snapshot import (
    LIVE_RESOLUTION,
    build_memory_snapshot,
    record_context_snapshot,
)

_logger = logging.getLogger(__name__)

_MEMORY_HOOK_NAME = "gui_memory_snapshot"
_SUBAGENT_HOOK_NAME = "gui_subagent_snapshot"


def _publisher(ui: Any):
    publish = getattr(ui, "publish_event", None)
    return publish if callable(publish) else None


def register_live_observability_hooks() -> None:
    """Install idempotent provider-boundary hooks in the current process."""

    if not any(
        spec.name == _MEMORY_HOOK_NAME
        for spec in default_registry.list("pre_provider_call")
    ):

        def _context_snapshot(ctx: HookContext):
            ui = getattr(ctx.session, "ui", None)
            publish = _publisher(ui)
            if publish is None:
                return None
            try:
                # Round-47 F11: the request estimate reuses the preflight
                # manifest when available (loop_body stashes
                # _request_estimate_manifest) instead of re-walking every
                # message; only unseen callers estimate directly.
                est = getattr(ctx.session, "_request_estimate_manifest", None)
                request_tokens = (
                    int(est["total"])
                    if est
                    else (
                        estimate_tokens(ctx.system_prompt or "")
                        + _estimate_messages_tokens(ctx.messages or [])
                        + _estimate_tools_tokens(ctx.tools or [])
                    )
                )
                ctx.session._memory_map_request_token_estimate = int(request_tokens)
                # F11: materialize layer texts ONCE here — both
                # build_memory_snapshot (grid fingerprints) and
                # record_context_snapshot (64-slice timeline) previously
                # walked every layer independently per provider call
                # (2x O(history) passes).
                from mu.gui.memory_snapshot import _LAYER_ORDER, _layer_text
                ctx.session._memory_layer_texts = {
                    lid: _layer_text(ctx.session, lid) for lid in _LAYER_ORDER
                }
                try:
                    snapshot = build_memory_snapshot(
                        ctx.session,
                        cols=LIVE_RESOLUTION,
                        rows=LIVE_RESOLUTION,
                        request_token_estimate=request_tokens,
                    )
                    timeline_point = record_context_snapshot(ctx.session, snapshot)
                finally:
                    # Round-48 F14: try/finally — the r47 shape cleared the
                    # stash only on the success path, leaking stale texts
                    # after any exception between set and clear.
                    ctx.session._memory_layer_texts = None
            except Exception as exc:  # observability must never break a turn
                _logger.warning("context snapshot hook failed: %s", exc)
                return None
            publish(
                {
                    "kind": "context_snapshot",
                    **snapshot,
                    "timeline_point": timeline_point,
                }
            )
            return None

        default_registry.register(
            "pre_provider_call", name=_MEMORY_HOOK_NAME
        )(_context_snapshot)

    if not any(
        spec.name == _SUBAGENT_HOOK_NAME
        for spec in default_registry.list("pre_provider_call")
    ):

        def _subagent_snapshot(ctx: HookContext):
            ui = getattr(ctx.session, "ui", None)
            publish = _publisher(ui)
            if publish is None:
                return None
            registry = getattr(ctx.session, "_subagent_registry", None)
            if registry is None:
                return None
            try:
                children = registry.snapshot_active()
            except Exception as exc:  # observability must never break a turn
                _logger.warning("subagent snapshot hook failed: %s", exc)
                return None
            publish(
                {
                    "kind": "subagent_snapshot",
                    "children": children,
                    "active": sum(
                        1 for child in children if child.get("status") == "running"
                    ),
                    "stuck": sum(1 for child in children if child.get("stuck")),
                    "stall": sum(1 for child in children if child.get("stall")),
                    "batch_id": registry.active_batch_id(),
                }
            )
            return None

        default_registry.register(
            "pre_provider_call", name=_SUBAGENT_HOOK_NAME
        )(_subagent_snapshot)


__all__ = ["register_live_observability_hooks"]
