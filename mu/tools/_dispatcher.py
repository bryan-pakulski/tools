"""Tool dispatcher.

`dispatch(tool_name, args, ...)` is the canonical entry point for
executing a tool. It performs descriptor lookup, argument-shape
validation, path-emptiness gating, feature-mode review-gate enforcement
for `apply_diff`, then invokes the registered handler and wraps the
result in the canonical envelope.

The function returns a JSON-encoded envelope string. `mu.tools.execute(...)`
parses that back into a dict for callers that want structured access;
`execute_tool` re-exports `dispatch` as the legacy callable signature.

`TOOL_HANDLERS` is the bridge dict the `@tool` decorator mirrors into
at decoration time, kept for back-compat with test fixtures that mutate
it directly. The canonical handler registry lives in `mu/tools/__init__.py`.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Callable, Dict, Optional

from utils.logger import logger

from ._envelope import _build_tool_envelope, _envelope_from_handler_result


# =================================================================== handler registry


TOOL_HANDLERS: dict[str, Callable[[dict, Any], str]] = {}

# These tools register bibliography entries in CitationManager.  The manager
# is process-global for backwards compatibility, but its contents are
# snapshotted on the owning SessionManager so unloaded research sessions can
# resume with their evidence intact.
_RESEARCH_SOURCE_TOOLS = {
    "url_grounding",
    "web_search",
    "arxiv_search",
    "doi_resolve",
    "reddit_search",
    "stackoverflow_search",
    "hackernews_search",
    "read_document",
    "assess_source",
}


def _path_arg_error(key: str) -> str:
    return (
        f"Error: The '{key}' argument is empty. "
        "You must provide a valid filesystem path for the active runtime."
    )


def _apply_diff_review_gate(
    args: Dict[str, Any], session: Any, tool_name: str
) -> Optional[Dict[str, Any]]:
    """If feature-mode review is active, refuse `apply_diff` unless the
    call carries an approved `proposal_id`. Returns a failure envelope
    when the call is blocked, or None to let it through.

    The check is scoped to `apply_diff` only — every other tool returns
    None immediately."""
    if tool_name != "apply_diff" or session is None:
        return None
    from mu.tools.capabilities import normalize_session_type

    variables = getattr(session, "variables", None) or {}
    if normalize_session_type(variables.get("session_type")) == "container":
        # Container mode intentionally removes workflow-level write gates.
        # The Docker boundary and non-bypassable secret guard remain active.
        return None
    if not hasattr(session, "session_manager"):
        return None

    feature_state = session.session_manager.get_feature_state() or {}
    if not isinstance(feature_state, dict):
        return None
    feature_plan = feature_state.get("feature_plan") or {}
    if not isinstance(feature_plan, dict):
        return None

    in_review_mode = bool(feature_plan.get("tasks_completed")) and (
        str(feature_plan.get("review_status", "")).strip().lower() != "completed"
    )
    if not in_review_mode:
        return None

    proposal_id = str(args.get("proposal_id", "") or "").strip()
    if not proposal_id:
        return _build_tool_envelope(
            tool_name=tool_name,
            ok=False,
            error_code="invalid_args",
            message=(
                "Error: apply_diff in review mode requires proposal_id "
                "for an approved diff proposal."
            ),
        )

    metadata_path = str(feature_state.get("metadata_path", "") or "").strip()
    if not metadata_path or not os.path.exists(metadata_path):
        return _build_tool_envelope(
            tool_name=tool_name,
            ok=False,
            error_code="not_found",
            message="Error: Feature metadata not found for review-mode apply_diff.",
        )

    from mu.feature.engine import load_feature_plan

    plan = load_feature_plan(metadata_path)
    proposal = next(
        (item for item in plan.diff_proposals if item.id == proposal_id), None
    )
    if proposal is None or proposal.status != "approved":
        return _build_tool_envelope(
            tool_name=tool_name,
            ok=False,
            error_code="access_denied",
            message=(
                "Error: apply_diff blocked in review mode. "
                "proposal_id must reference an approved diff proposal."
            ),
        )
    if getattr(proposal, "applied_at", None) is not None:
        return _build_tool_envelope(
            tool_name=tool_name,
            ok=False,
            error_code="access_denied",
            message=(
                "Error: apply_diff blocked in review mode. "
                f"proposal {proposal_id} has already been applied and is single-use."
            ),
        )
    # The approved proposal's diff is the ONLY diff this proposal_id
    # authorizes - reusing the id with different content must fail.
    submitted_diff = str(args.get("diff", "") or "")
    if submitted_diff != proposal.diff:
        return _build_tool_envelope(
            tool_name=tool_name,
            ok=False,
            error_code="access_denied",
            message=(
                "Error: apply_diff blocked in review mode. "
                f"the submitted diff does not match approved proposal {proposal_id}."
            ),
        )
    return None


def _resolve_handler(descriptor) -> Optional[Any]:
    """Resolve the handler for a tool, checking the bridge dict first
    and falling back to the canonical `mu.tools._HANDLERS` registry."""
    handler = TOOL_HANDLERS.get(descriptor.handler_key)
    if handler is not None:
        return handler

    from . import _HANDLERS, _ensure_legacy_loaded

    _ensure_legacy_loaded()
    return _HANDLERS.get(descriptor.handler_key)


def dispatch(
    tool_name: str,
    args: Dict[str, Any],
    folder_context,
    ui: Any = None,
    variables: Optional[Dict[str, Any]] = None,
    *,
    invocation_source: str = "session",
    session: Any = None,
) -> str:
    """Canonical tool dispatcher.

    Resolves the tool descriptor, validates arguments, enforces the
    feature-mode `apply_diff` review gate, invokes the handler, and
    returns a JSON-encoded envelope string.
    """
    from mu.tools.descriptors import get_tool_descriptor, build_tool_context

    descriptor = get_tool_descriptor(tool_name)
    if not descriptor:
        return json.dumps(
            _build_tool_envelope(
                tool_name=tool_name,
                ok=False,
                error_code="not_found",
                message=f"Unknown tool: {tool_name}",
            )
        )

    # Capability filtering is enforced again at dispatch time so a stale model
    # tool call or direct programmatic invocation cannot bypass the provider
    # schema filter used by the agent loop.
    from mu.tools.capabilities import is_tool_allowed, normalize_session_type

    runtime_variables = variables or getattr(session, "variables", {}) or {}
    session_type = normalize_session_type(runtime_variables.get("session_type"))
    if not is_tool_allowed(tool_name, session_type):
        return json.dumps(
            _build_tool_envelope(
                tool_name=tool_name,
                ok=False,
                error_code="session_type_forbidden",
                message=(
                    f"Tool {tool_name!r} is unavailable in {session_type!r} "
                    "session type."
                ),
                telemetry={"session_type": session_type},
            )
        )

    if not isinstance(args, dict):
        return json.dumps(
            _build_tool_envelope(
                tool_name=tool_name,
                ok=False,
                error_code="invalid_args",
                message=(
                    f"Error: Tool '{tool_name}' arguments must be an "
                    f"object/dict, got {type(args).__name__}. Please re-issue "
                    "the tool call with JSON object arguments."
                ),
            )
        )

    # Path-shaped argument keys that must not be empty when provided.
    for key in ("filename", "file", "path"):
        if key in args and (not args[key] or str(args[key]).strip() == ""):
            return json.dumps(
                _build_tool_envelope(
                    tool_name=tool_name,
                    ok=False,
                    error_code="invalid_args",
                    message=_path_arg_error(key),
                )
            )

    gate = _apply_diff_review_gate(args, session, tool_name)
    if gate is not None:
        return json.dumps(gate)

    handler = _resolve_handler(descriptor)
    if not handler:
        return json.dumps(
            _build_tool_envelope(
                tool_name=tool_name,
                ok=False,
                error_code="not_found",
                message=f"Error: No handler registered for tool '{tool_name}'.",
            )
        )

    context = build_tool_context(
        folder_context,
        ui,
        variables,
        invocation_source=invocation_source,
        session=session,
    )
    try:
        if tool_name in _RESEARCH_SOURCE_TOOLS and session is not None:
            manager = getattr(session, "session_manager", None)
            if (
                manager is not None
                and getattr(manager, "current_session_name", "")
                and hasattr(manager, "restore_research_sources")
            ):
                manager.restore_research_sources()
        raw_result = handler(args, context)
        envelope = _envelope_from_handler_result(tool_name, raw_result)
        # Review-mode single-use: after a SUCCESSFUL review-gated apply,
        # mark the proposal applied so its id cannot be reused.
        if tool_name == "apply_diff" and envelope.get("ok") and session is not None:
            try:
                manager = getattr(session, "session_manager", None)
                feature_state = manager.get_feature_state() if manager else None
                plan_meta = str(
                    (feature_state or {}).get("metadata_path", "") or ""
                ).strip()
                proposal_id = str(args.get("proposal_id", "") or "").strip()
                if plan_meta and proposal_id:
                    from mu.feature.engine import mark_proposal_applied

                    mark_proposal_applied(
                        plan_meta,
                        proposal_id=proposal_id,
                        applied_file=str(args.get("filename", "") or ""),
                    )
            except Exception as exc:
                logger.warning(
                    "apply_diff: could not mark proposal %s applied: %s",
                    args.get("proposal_id"), exc,
                )
        if tool_name in _RESEARCH_SOURCE_TOOLS and session is not None:
            manager = getattr(session, "session_manager", None)
            if manager is not None and hasattr(manager, "snapshot_research_sources"):
                manager.snapshot_research_sources()
                manager.save_history(getattr(session, "folder_context", None))
        if "execution_source" not in envelope.get("telemetry", {}):
            envelope.setdefault("telemetry", {})["execution_source"] = (
                invocation_source
            )
        return json.dumps(envelope, indent=2, sort_keys=True)
    except Exception as exc:
        hint = ""
        if isinstance(exc, AttributeError) and (
            "'str' object has no attribute 'get'" in str(exc)
            or "object has no attribute 'get'" in str(exc)
        ):
            hint = (
                " Hint: the tool received a string where a dict was expected — "
                "the model likely passed a JSON-encoded string instead of a parsed object."
            )
        logger.error(
            "Tool %r failed with %s: %s\n%s",
            tool_name,
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        return json.dumps(
            _build_tool_envelope(
                tool_name=tool_name,
                ok=False,
                error_code="execution_failed",
                message=(
                    f"Error: Tool '{tool_name}' failed with {type(exc).__name__}: {exc}."
                    f"{hint} Please fix arguments and retry."
                ),
                telemetry={"execution_source": invocation_source},
            ),
            indent=2,
        )


def execute_tool(
    tool_name: str,
    args: dict,
    folder_context,
    ui=None,
    variables: dict = None,
    *,
    invocation_source: str = "session",
    session: Any = None,
) -> str:
    """Legacy-compatible alias for `dispatch`."""
    return dispatch(
        tool_name,
        args,
        folder_context,
        ui,
        variables,
        invocation_source=invocation_source,
        session=session,
    )


__all__ = ["TOOL_HANDLERS", "dispatch", "execute_tool"]
