"""Tool registry for the agent loop.

The registry exposes three primary operations:

    @tool(name=..., description=..., parameters=..., ...)        # register a handler
    def my_handler(args, context): ...

    execute(name, args, context) -> envelope dict                # invoke a tool
    list_tools(*, disabled=set()) -> list[ToolDefinition]        # enumerate tools
    get(name) -> ToolDescriptor | None                           # introspect

Every tool registers via the `@tool` decorator in its handler module.
The decorator stores the descriptor in `_REGISTRY` and the handler in
`_HANDLERS`, and mirrors both into `mu.tools.descriptors.TOOLS` /
`TOOL_DESCRIPTORS` and `mu.tools._dispatcher.TOOL_HANDLERS` so list-
style and dict-style consumers both see registrations.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from providers.base import ToolDefinition

from ._context import ToolExecutionContext, build_tool_context  # re-exports


def _import_registry():
    """Late-import `mu.tools.descriptors` to avoid a circular import on first
    decoration (descriptors imports nothing from this module, but the
    `@tool` decorator runs at import time of every `<group>/handlers.py`)."""
    from mu.tools import descriptors as _desc  # noqa: WPS433 — intentional late import
    return _desc


def _import_dispatcher():
    """Late-import `mu.tools._dispatcher` for the `TOOL_HANDLERS` mirror."""
    from mu.tools import _dispatcher as _disp  # noqa: WPS433
    return _disp


_REGISTRY: Dict[str, "ToolDescriptor"] = {}
_HANDLERS: Dict[str, Callable[..., Any]] = {}
_LEGACY_LOADED = False


def _ensure_legacy_loaded() -> None:
    """Defensive resync — copy any descriptor/handler that registered
    via `@tool` before this module finished its first import."""
    global _LEGACY_LOADED
    if _LEGACY_LOADED:
        return
    desc = _import_registry()
    disp = _import_dispatcher()
    for name, descriptor in getattr(desc, "TOOL_DESCRIPTORS", {}).items():
        _REGISTRY.setdefault(name, descriptor)
    for name, handler in getattr(disp, "TOOL_HANDLERS", {}).items():
        _HANDLERS.setdefault(name, handler)
    _LEGACY_LOADED = True


# ---------------------------------------------------------------- registration


def tool(
    *,
    name: str,
    description: str,
    parameters: Dict[str, Any],
    requires_approval: bool = True,
    execution_kind: str = "io",
    preview_policy: str = "default",
    server_policy: str = "default",
    result_mode: str = "json",
    error_mode: str = "text_error",
    summary_builder: Optional[str] = None,
    phase: str = "core",
    group: str = "",
    approval_policy: str = "default",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a handler as a tool in the registry.

    The handler signature is `(args: dict, context: ToolExecutionContext)`
    and may return either a string, a dict with an envelope shape, or a
    dict carrying a `success`/`error` pair. `execute()` wraps the return
    value through the same envelope builder the legacy harness uses.
    """

    desc = _import_registry()
    disp = _import_dispatcher()
    build_descriptor = getattr(desc, "_build_descriptor")
    default_server_policy = getattr(desc, "_default_server_policy")
    default_result_mode = getattr(desc, "_default_result_mode")

    # Resolve "default" sentinels against the descriptor-module fallbacks
    # so every tool gets a complete descriptor shape unless explicitly
    # overridden.
    resolved_server_policy = (
        default_server_policy(name) if server_policy == "default" else server_policy
    )
    resolved_result_mode = (
        default_result_mode(name) if result_mode == "default" else result_mode
    )
    resolved_preview_policy = (
        ("optional" if requires_approval else "none")
        if preview_policy == "default"
        else preview_policy
    )
    # Default `execution_kind`: requires_approval → mutate, otherwise read.
    # The "io" sentinel maps to the same default.
    resolved_execution_kind = (
        ("mutate" if requires_approval else "read")
        if execution_kind in ("io", "default")
        else execution_kind
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            requires_approval=requires_approval,
        )
        descriptor = build_descriptor(
            definition,
            execution_kind=resolved_execution_kind,
            preview_policy=resolved_preview_policy,
            server_policy=resolved_server_policy,
            result_mode=resolved_result_mode,
            handler_key=name,
            error_mode=error_mode,
            summary_builder=summary_builder,
            phase=phase,
            group=group,
            approval_policy=approval_policy,
        )
        _REGISTRY[name] = descriptor
        _HANDLERS[name] = func
        # Mirror into the descriptor/dispatcher tables so callers that
        # introspect those maps (TOOL_HANDLERS-mutating tests, UI/system
        # prompt iterating TOOLS) see new registrations.
        desc.TOOL_DESCRIPTORS[name] = descriptor
        disp.TOOL_HANDLERS[name] = func
        existing = {t.name for t in desc.TOOLS}
        if name not in existing:
            desc.TOOLS.append(definition)
        return func

    return decorator


# ---------------------------------------------------------------- introspection


def get(name: str) -> Optional["ToolDescriptor"]:
    _ensure_legacy_loaded()
    return _REGISTRY.get(name)


def list_tools(*, disabled: Optional[Iterable[str]] = None) -> List[ToolDefinition]:
    _ensure_legacy_loaded()
    skip: Set[str] = set(disabled or ())
    return [
        descriptor.definition
        for name, descriptor in _REGISTRY.items()
        if name not in skip
    ]


def list_descriptors() -> List["ToolDescriptor"]:
    _ensure_legacy_loaded()
    return list(_REGISTRY.values())


# ----------------------------------------------------------------- execution


def execute(name: str, args: Dict[str, Any], context: ToolExecutionContext) -> Dict[str, Any]:
    """Run a tool and return a normalized envelope dict.

    Calls the canonical `_dispatcher.dispatch(...)` directly. The legacy
    `mu.tools._dispatcher.execute_tool` is now a thin shim over the same function,
    so callers get identical behavior regardless of which entry point
    they use.

    `dispatch` returns a JSON-encoded envelope string (the wire format
    the agent loop inlines into message history). We parse it back into
    a dict here because the agent loop wants structured access.
    """

    import json as _json

    _ensure_legacy_loaded()
    from ._dispatcher import dispatch

    raw = dispatch(
        name,
        args,
        context.folder_context,
        context.ui,
        context.variables,
        invocation_source=context.invocation_source,
        session=context.session,
    )
    if isinstance(raw, dict):
        return raw
    try:
        parsed = _json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError):
        pass
    # Fall back to wrapping the raw text in an envelope so callers always
    # get a structured result.
    return {
        "ok": True,
        "error_code": None,
        "message": str(raw or ""),
        "data": {},
        "artifacts": [],
        "telemetry": {"tool_name": name},
    }


def execute_raw(name: str, args: Dict[str, Any], context: ToolExecutionContext) -> str:
    """Run a tool and return the raw JSON-string envelope (legacy contract)."""
    _ensure_legacy_loaded()
    from ._dispatcher import dispatch

    return dispatch(
        name,
        args,
        context.folder_context,
        context.ui,
        context.variables,
        invocation_source=context.invocation_source,
        session=context.session,
    )


def _load_builtin_tools() -> None:
    """Import built-in `@tool`-decorated modules so their handlers register.

    Done at the bottom of `__init__.py` so the `tool` decorator and
    `_REGISTRY` are defined first. The import is wrapped in try/except so
    a single bad tool module doesn't kill the whole registry.
    """
    try:
        from . import task  # noqa: F401 — registers todo_write/set_status/list
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load task tool package: %s", exc
        )
    try:
        from . import thread as _thread_tools  # noqa: F401
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load thread tool package: %s", exc
        )
    try:
        from . import agent as _agent_tools  # noqa: F401 — registers spawn_agent
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load agent tool package: %s", exc
        )
    try:
        from . import memory as _memory_tools  # noqa: F401 — registers save/search/list_{memory,scratchpad} + clear_scratchpad + recall
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load memory tool package: %s", exc
        )
    try:
        from . import workspace as _workspace_tools  # noqa: F401 — registers read_file, search_*, get_chunk, list_dir, retrieve_relevant_context, get_workspace_details
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load workspace tool package: %s", exc
        )
    try:
        from . import file as _file_tools  # noqa: F401 — registers write_file, apply_diff, search_and_replace_file
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load file tool package: %s", exc
        )
    try:
        from . import shell as _shell_tools  # noqa: F401 — registers bash + bash_{background,status,logs,kill,list}
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load shell tool package: %s", exc
        )
    try:
        from . import research as _research_tools  # noqa: F401 — registers web/arxiv/reddit/SO/HN searches + url_grounding + read_document + doi_resolve
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load research tool package: %s", exc
        )
    try:
        from . import skill as _skill_tools  # noqa: F401 — registers invoke_skill
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load skill tool package: %s", exc
        )
    try:
        from . import feature as _feature_tools  # noqa: F401 — registers create_feature, create_phases, create_task, get_execution_state, block_task, resume_task, archive_task, review_*, propose/decide_task_diff, create/update/approve_feature_task, get_current_task, get_tasks, update_task_status, raise_blocker
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load feature tool package: %s", exc
        )
    try:
        from . import security as _security_tools  # noqa: F401 — registers create_security_report + 8 audit-engine tools
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load security tool package: %s", exc
        )
    try:
        from . import batch as _batch_tools  # noqa: F401 — registers batch_job + flush
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load batch tool package: %s", exc
        )
    try:
        from . import teacher as _teacher_tools  # noqa: F401 — registers create_course + 15 teacher-engine tools
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load teacher tool package: %s", exc
        )
    try:
        from . import prompt as _prompt_tools  # noqa: F401 — registers ask_user_choice
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load prompt tool package: %s", exc
        )
    try:
        from . import attachment as _attachment_tools  # noqa: F401 — registers list/read/search attachments
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load attachment tool package: %s", exc
        )
    try:
        from . import artifact as _artifact_tools  # noqa: F401 — registers upload_artifact/list_artifacts
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load artifact tool package: %s", exc
        )
    try:
        from . import session as _session_tools  # noqa: F401 — registers search_history
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load session tool package: %s", exc
        )
    try:
        from . import trace as _trace_tools  # noqa: F401 — registers list_traces/trace_summary/trace_series/trace_iteration
    except Exception as exc:  # pragma: no cover — defensive
        import logging
        logging.getLogger("mucli").warning(
            "mu.tools: failed to load trace tool package: %s", exc
        )


_load_builtin_tools()


__all__ = [
    "ToolExecutionContext",
    "build_tool_context",
    "execute",
    "execute_raw",
    "get",
    "list_descriptors",
    "list_tools",
    "tool",
]
