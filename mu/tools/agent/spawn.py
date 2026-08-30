"""Async delegation onto a persistent specialist pool.

``spawn_agent`` keeps the existing task-id contract, but the underlying child
Session is retained after a delegation completes and reused for compatible
future work. This amortises repository discovery while keeping parent history
isolated.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict

from mu.memory.stores import ACTIVE
from mu.tools import tool

logger = logging.getLogger("mucli")
MAX_SUBAGENT_DEPTH = 2
_DEFAULT_MAX_ITERATIONS = 60
_MIN_MAX_ITERATIONS = 30
_HANDOFF_MAX_ENTRIES = 8
_HANDOFF_MINE_LIMIT = 6


def _model_installed(model: str, installed: list) -> bool:
    if not model:
        return False
    if model in installed:
        return True
    base = model.split(":", 1)[0]
    return any(str(m).split(":", 1)[0] == base for m in installed)


def _infer_specialist_key(task: str, explicit: str = "") -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", str(explicit or "").strip().lower()).strip("-")
    if value:
        return value[:48]
    text = str(task or "").lower()
    routes = (
        ("tests", ("test", "pytest", "failing suite", "regression", "ci failure")),
        ("review", ("review", "audit", "security", "vulnerability", "threat")),
        ("research", ("research", "paper", "docs", "documentation", "compare", "benchmark")),
        ("repo", ("investigate", "trace", "explain", "find where", "root cause", "understand")),
        ("implementation", ("implement", "fix", "refactor", "edit", "change", "add ", "remove", "migrate")),
    )
    for key, needles in routes:
        if any(needle in text for needle in needles):
            return key
    return "general"


def _short_task_title(task: str, explicit: str = "", specialist: str = "general") -> str:
    """Return a complete, compact action label without UI truncation.

    Models are encouraged to provide a 2–5 word ``title``. The fallback
    extracts the leading action phrase and removes repository/object scope,
    producing labels such as ``Security audit`` or ``Code quality review``.
    """
    supplied = re.sub(r"\s+", " ", str(explicit or "")).strip(" .:-")
    if supplied and len(supplied) <= 48 and len(supplied.split()) <= 5:
        return supplied

    text = re.sub(r"\s+", " ", str(task or "")).strip()
    first = re.split(r"[.!?](?:\s|$)|\b(?:focus|scope|review|inspect|check):\s*", text, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .:-")
    scoped = re.split(r"\s+(?:of|for|across|within)\s+(?:the\s+)?", first, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if 2 <= len(scoped.split()) <= 5 and len(scoped) <= 48:
        return scoped

    lowered = text.lower()
    routes = (
        ("Security audit", ("security", "vulnerability", "threat")),
        ("Infrastructure audit", ("k8s", "kubernetes", "infrastructure", "deployment")),
        ("Code quality review", ("code quality", "lint", "maintainability")),
        ("Test review", ("test", "pytest", "regression", "ci ")),
        ("Documentation review", ("docs", "documentation", "readme")),
        ("Root-cause analysis", ("root cause", "investigate", "trace why")),
        ("Implementation", ("implement", "refactor", "fix", "migrate")),
    )
    for label, needles in routes:
        if any(needle in lowered for needle in needles):
            return label
    specialist_label = re.sub(r"[_-]+", " ", str(specialist or "general")).strip()
    return f"{specialist_label.title()} task"


_SPECIALIST_SYSTEM_TEMPLATE = """ROLE: persistent SUB-AGENT specialist \"{specialist}\" (depth={depth}).
Sub-agent task directive: run the newest delegation to completion and return one self-contained result.
You retain your own history and durable memory across related delegations in this parent session.
The latest user message is the current delegation; execute it with available tools and return one concise self-contained result.
Reuse retained repository knowledge instead of rediscovering it. Supersede stale findings when evidence changes.
Use send_subagent_finding only for a material intermediate discovery the parent may need before completion; routine status belongs in the UI.
Do not interact with the end user. {depth_rule}
"""


def _build_system_prompt(task: str = "", remaining_depth: int = 0, max_iterations: int = 60, parent_findings: str = "", *, specialist: str = "general", depth: int | None = None) -> str:
    """Stable specialist persona. Mutable task/context deliberately stay out."""
    resolved_depth = int(depth if depth is not None else max(1, MAX_SUBAGENT_DEPTH - int(remaining_depth or 0)))
    remaining = max(0, MAX_SUBAGENT_DEPTH - resolved_depth)
    depth_rule = "Do not spawn further sub-agents (depth cap reached)." if remaining <= 0 else f"You may spawn up to {remaining} further sub-agent level(s)."
    return _SPECIALIST_SYSTEM_TEMPLATE.format(specialist=specialist, depth=resolved_depth, depth_rule=depth_rule)


def _mine_parent_memory(parent) -> list:
    mem = getattr(getattr(parent, "session_manager", None), "task_memory", None)
    if mem is None:
        return []
    priority = {"decision": 0, "goal": 1, "finding": 2}
    entries = [e for e in mem.entries if e.kind in priority and e.status in ("active", "done")]
    entries.sort(key=lambda e: (priority.get(e.kind, 9), 0 if e.status == "active" else 1, -e.updated_at))
    return entries[:_HANDOFF_MINE_LIMIT]


def _build_handoff(parent) -> list:
    inherited = list(getattr(parent, "_subagent_handoff", []) or [])
    mined = _mine_parent_memory(parent)
    out, seen = [], set()
    def add(content, kind, tags):
        content = str(content or "").strip()
        if not content or content in seen or len(out) >= _HANDOFF_MAX_ENTRIES:
            return
        seen.add(content)
        out.append({"content": content, "kind": kind or "finding", "tags": list(tags or [])})
    for item in inherited:
        add(item.get("content"), item.get("kind"), item.get("tags"))
    for entry in mined:
        add(entry.content, entry.kind, entry.tags)
    return out


def _seed_handoff(child_sm, handoff: list) -> None:
    mem = getattr(child_sm, "task_memory", None)
    if mem is None:
        return
    for item in handoff:
        try:
            mem.save(content=item["content"], tags=item["tags"], source="parent_handoff", kind=item["kind"], status=ACTIVE)
        except Exception:
            pass


def _delegation_prompt(task: str, explicit_context: str) -> str:
    if explicit_context:
        return f"DELEGATION:\n{task}\n\nParent-provided context for this delegation only:\n{explicit_context}"
    return f"DELEGATION:\n{task}"


def _envelope(*, ok: bool, message: str, error_code=None, data=None) -> Dict[str, Any]:
    return {"ok": ok, "error_code": error_code, "message": message, "data": data or {}, "artifacts": [], "telemetry": {"tool_name": "spawn_agent"}}


def _tool_profile(parent, requested_tools, remaining_depth: int) -> list[str]:
    from mu.tools.descriptors import TOOLS
    all_names = {t.name for t in TOOLS}
    if requested_tools:
        allowed = {str(x) for x in requested_tools}
        allowed.update({"flush", "load_tools", "send_subagent_finding"})
        disabled = sorted(all_names - allowed)
    else:
        disabled = []
    if remaining_depth <= 0 and "spawn_agent" not in disabled:
        disabled.append("spawn_agent")
    return sorted(disabled)


@tool(
    name="spawn_agent",
    description=("Delegate focused work asynchronously to a persistent specialist. Compatible specialists are reused across delegations, retaining their private repository context. Returns a task_id. Use await_subagent to block, poll_subagent only for occasional status checks, and kill_subagent to cancel."),
    parameters={"type": "object", "properties": {
        "task": {"type": "string", "description": "Focused delegation."},
        "title": {"type": "string", "description": "Short 2–5 word action label for the UI, such as 'Security audit' or 'API test review'. Never repeat the full task."},
        "specialist": {"type": "string", "description": "Optional stable specialist key; otherwise inferred from the task."},
        "tools": {"type": "array", "items": {"type": "string"}, "description": "Optional execution-tool whitelist."},
        "max_iterations": {"type": "integer", "description": "Delegation iteration cap."},
        "model": {"type": "string", "description": "Optional child model override."},
        "context": {"type": "string", "description": "Terse delegation-specific context; not added to the persistent persona."}
    }, "required": ["task"]},
    requires_approval=True,
    execution_kind="io",
    result_mode="json",
)
def spawn_agent(args: Dict[str, Any], context) -> Dict[str, Any]:
    task = str(args.get("task") or "").strip()
    if not task:
        return _envelope(ok=False, error_code="invalid_args", message="spawn_agent requires non-empty 'task'.")
    parent = getattr(context, "session", None)
    if parent is None:
        return _envelope(ok=False, error_code="no_session", message="spawn_agent requires a parent session.")
    if (getattr(parent, "variables", None) or {}).get("plan_mode"):
        return _envelope(ok=False, error_code="plan_mode_blocked", message="spawn_agent is blocked while plan_mode is active.")
    current_depth = int(getattr(parent, "_subagent_depth", 0) or 0)
    if current_depth >= MAX_SUBAGENT_DEPTH:
        return _envelope(ok=False, error_code="depth_exceeded", message=f"spawn_agent depth limit reached ({MAX_SUBAGENT_DEPTH}).")

    from mu.agent.lifecycle import SubagentLifecycleManager
    from mu.session.session import Session, SessionManager
    from mu.ui.subagent import SubagentUI

    parent.variables["session_role"] = "parent"
    registry = parent._subagent_registry
    registry.bind_parent(parent)

    max_iterations = max(_MIN_MAX_ITERATIONS, int(args.get("max_iterations") or _DEFAULT_MAX_ITERATIONS))
    child_depth = current_depth + 1
    remaining_depth = MAX_SUBAGENT_DEPTH - child_depth
    specialist_key = _infer_specialist_key(task, args.get("specialist") or "")
    title = _short_task_title(task, args.get("title") or "", specialist_key)
    explicit_context = str(args.get("context") or "").strip()
    handoff = _build_handoff(parent)

    try:
        installed = list(parent.provider.get_available_models() or [])
    except Exception:
        installed = []
    parent_model = str(getattr(parent.provider, "model_name", "") or "")
    cfg_model = str(parent.variables.get("subagent_model") or "").strip()
    arg_model = str(args.get("model") or "").strip()
    def pick(candidate):
        return candidate if candidate and _model_installed(candidate, installed) else ""
    if arg_model and not _model_installed(arg_model, installed):
        logger.warning(
            "spawn_agent: requested model '%s' is not installed; falling back to parent model '%s'.",
            arg_model, parent_model or "default",
        )
    if cfg_model and not _model_installed(cfg_model, installed):
        logger.warning(
            "spawn_agent: configured subagent_model '%s' is not installed; falling back to parent model '%s'.",
            cfg_model, parent_model or "default",
        )
    resolved_model = pick(cfg_model) or pick(arg_model) or parent_model
    provider_key = str(getattr(parent.provider, "name", "") or f"{type(parent.provider).__module__}.{type(parent.provider).__name__}")
    disabled = _tool_profile(parent, args.get("tools"), remaining_depth)

    worker = registry.acquire_specialist(specialist_key, depth=child_depth, model=resolved_model, provider_key=provider_key, disabled_tools=disabled)
    reused = worker is not None

    # Build/refresh UI before Session creation/reuse.
    root_ui = parent.ui
    while isinstance(root_ui, SubagentUI):
        root_ui = root_ui._parent
    publish_event = getattr(root_ui, "publish_event", None)
    if callable(publish_event):
        registry._publish = publish_event
    elif root_ui is not None and hasattr(root_ui, "_publish"):
        # Compatibility for third-party UIs that implemented the original
        # private WebUI hook before ``publish_event`` became public.
        registry._publish = lambda ev: root_ui._publish(ev)
    tracker_agent_id = None
    try:
        tracker_agent_id = registry.tracker.open(depth=child_depth, task=task)
    except Exception:
        pass
    child_ui = SubagentUI(parent.ui, depth=child_depth, tracker=registry.tracker if tracker_agent_id else None, agent_id=tracker_agent_id) if parent.ui is not None else None

    if worker is None:
        child_provider = parent.provider.clone_for_child()
        child_provider.model_name = resolved_model
        child_sm = SessionManager(session_name="__subagent__")
        child_sm.save_history = lambda *a, **kw: None
        child = Session(provider=child_provider, thinking=parent.thinking, system_instruction=_build_system_prompt(specialist=specialist_key, depth=child_depth), session_manager=child_sm, ui=child_ui, debug=getattr(parent, "debug", False))
        _seed_handoff(child_sm, handoff)
        worker = registry.register_specialist(child, specialist_key=specialist_key, depth=child_depth, model=resolved_model, provider_key=provider_key, disabled_tools=disabled)
    else:
        child = worker.child
        child.ui = child_ui
        child.thinking = parent.thinking
        _seed_handoff(child.session_manager, handoff)

    # Refresh mutable runtime wiring while preserving worker cognition/history.
    child.folder_context = parent.folder_context
    child.session_manager.folder_context = parent.folder_context
    child.disabled_tools = disabled
    child.variables["yolo"] = True
    child.variables["max_iterations"] = max_iterations
    child.variables["agent_mode"] = "default"
    child.variables["collation_enabled"] = False
    child.variables["session_role"] = "child"
    child.variables["subagent_depth"] = child_depth
    child.variables["compact_history"] = True
    child.variables["auto_compaction_enabled"] = True
    child.variables["lazy_tools_enabled"] = True
    child.variables["active_tool_phases"] = ["core"]
    child._loaded_tool_phases = []
    child._subagent_depth = child_depth
    child._subagent_handoff = handoff
    child._subagent_cancelled = False
    child._subagent_kill_reason = None
    child._subagent_wrap_up_injected = False
    session_type = str(parent.variables.get("session_type", "workspace") or "workspace").lower()
    child.variables["session_type"] = session_type
    if session_type == "container":
        child.variables["strict_mode"] = False
        child.variables["plan_mode"] = False
        child.variables["security_allow_secret_paths"] = False
        # Important: container execution access does not imply exposing every schema.
        child.variables["lazy_tools_enabled"] = True

    lifecycle = SubagentLifecycleManager(thresholds={
        "stuck_threshold": int(parent.variables.get("subagent_stuck_threshold", 3) or 3),
        "stall_threshold": int(parent.variables.get("subagent_stall_threshold", 5) or 5),
        "max_runtime_seconds": int(parent.variables.get("subagent_max_runtime_seconds", 300) or 300),
        "enabled": bool(parent.variables.get("subagent_lifecycle_enabled", True)),
    })
    child._subagent_lifecycle = lifecycle
    parent_history = list(getattr(parent.session_manager, "history", []) or [])
    parent_history_index = len(parent_history) - 1
    parent_turn_index = next(
        (
            index
            for index in range(len(parent_history) - 1, -1, -1)
            if parent_history[index].get("role") == "user"
            and parent_history[index].get("timeline_id")
        ),
        -1,
    )
    parent_turn_id = (
        str(parent_history[parent_turn_index].get("timeline_id") or "")
        if parent_turn_index >= 0
        else ""
    )
    record = registry.register(child, task=task, title=title, depth=child_depth, lifecycle=lifecycle, tracker_agent_id=tracker_agent_id, model=resolved_model, specialist_key=specialist_key, worker_id=worker.worker_id, reused_specialist=reused, max_iterations=max_iterations, parent_history_index=parent_history_index, parent_turn_index=parent_turn_index, parent_turn_id=parent_turn_id)
    child.variables["subagent_parent_task_id"] = record.task_id
    child._parent_registry = registry
    if root_ui is not None:
        try:
            root_ui.show_info(
                f"Spawning subagent '{title}' (task {record.task_id}): {task}"
            )
        except Exception:
            pass
    registry.launch(record, _delegation_prompt(task, explicit_context))

    return _envelope(ok=True, message=f"Dispatched {specialist_key} specialist ({'reused' if reused else 'new'} worker {worker.worker_id}); task_id={record.task_id}.", data={
        "task_id": record.task_id, "status": "running", "depth": child_depth, "task": task,
        "title": title,
        "specialist": specialist_key, "worker_id": worker.worker_id, "reused_specialist": reused,
        "batch_id": record.batch_id, "state_path": registry.snapshot(record.task_id).get("state_path"),
        "result_path": registry.snapshot(record.task_id).get("result_path"),
    })


__all__ = ["MAX_SUBAGENT_DEPTH", "spawn_agent", "_infer_specialist_key", "_build_system_prompt"]
