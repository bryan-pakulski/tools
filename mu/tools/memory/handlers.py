"""Session memory + scratchpad `@tool` handlers.

The handlers route through `context.session.{task_memory,turn_scratchpad}`.
A fallback in-process store keeps unit tests that pass `session=None`
working without a full session scaffold.

Lifecycle tools (update_memory_status, supersede_memory, retire_memory,
reactivate_memory, archive_memory) mutate the status field on MemoryEntry
so the agent can distinguish active work from completed/superseded/
archived entries. All are session-scoped and plan-mode blocked (same as
save_memory).
"""

import json
from typing import Any, Dict, List, Optional

from mu.tools import tool


# ---------------------------------------------------------------- stores


def _task_memory(context):
    session = getattr(context, "session", None)
    if session is not None and hasattr(session, "task_memory"):
        return session.task_memory
    return _fallback_task_memory()


def _durable_memory(context):
    session = getattr(context, "session", None)
    if (
        session is None
        or not hasattr(session, "get_durable_memory_service")
        or not bool(getattr(session, "variables", {}).get("durable_memory_enabled", True))
    ):
        return None
    try:
        return session.get_durable_memory_service()
    except Exception:
        return None


def _sync_durable_status(context, entry, status: str, reason: str = "") -> None:
    """Best-effort lifecycle mirror; working-memory curation never blocks."""

    durable_id = str(getattr(entry, "durable_id", "") or "")
    service = _durable_memory(context)
    if service is None or not durable_id or durable_id == "rejected":
        return
    try:
        if status == "archived":
            service.ledger.action(durable_id, "archive", actor="model", reason=reason)
        elif status == "active":
            service.ledger.action(durable_id, "restore", actor="model", reason=reason)
        elif status == "stale":
            service.ledger.action(
                durable_id, "mark_needs_review", actor="model", reason=reason
            )
        elif status == "superseded":
            service.ledger.revise(
                durable_id,
                {"lifecycle": "superseded"},
                actor="model",
                reason=reason or "working memory superseded",
            )
    except Exception:
        pass


def _scratchpad(context):
    session = getattr(context, "session", None)
    if session is not None and hasattr(session, "turn_scratchpad"):
        return session.turn_scratchpad
    return _fallback_scratchpad()


_FALLBACK_TASK_MEMORY = None
_FALLBACK_SCRATCHPAD = None


def _fallback_task_memory():
    """Process-local TaskMemoryStore for session-less contexts.

    Only used by unit tests that build a `ToolExecutionContext` directly
    without a Session; the real REPL always has `context.session` set.
    """
    global _FALLBACK_TASK_MEMORY
    if _FALLBACK_TASK_MEMORY is None:
        from mu.memory.stores import TaskMemoryStore

        _FALLBACK_TASK_MEMORY = TaskMemoryStore()
    return _FALLBACK_TASK_MEMORY


def _fallback_scratchpad():
    global _FALLBACK_SCRATCHPAD
    if _FALLBACK_SCRATCHPAD is None:
        from mu.memory.stores import ScratchpadStore

        _FALLBACK_SCRATCHPAD = ScratchpadStore()
    return _FALLBACK_SCRATCHPAD


def _int_arg(args: Dict[str, Any], key: str, default: int) -> int:
    raw = args.get(key, default)
    try:
        value = int(raw or default)
    except (TypeError, ValueError):
        return default
    return value


# ---------------------------------------------------------------- task memory


@tool(
    name="save_memory",
    description=(
        "Saves a concise, reusable fact to working memory and automatically "
        "promotes eligible non-secret content into the scoped cross-session "
        "Memory Ledger. This is model-controlled and never requires approval. "
        "Choose repository scope for project facts and personal scope only "
        "for genuine user-wide preferences. Use supersedes_id when replacing "
        "an earlier durable memory instead of creating conflicting siblings."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The concise fact, decision, or reminder to store.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to help later retrieval.",
            },
            "source": {
                "type": "string",
                "description": "Optional note about where this memory came from.",
            },
            "kind": {
                "type": "string",
                "enum": [
                    "constraint", "decision", "preference", "convention",
                    "finding", "procedure", "lesson", "handoff",
                    "observation", "goal",
                ],
                "description": (
                    "Classification of this memory. Defaults to 'observation'. "
                    "Use 'decision' for architectural choices, 'finding' for "
                    "verified facts, 'goal' for active work targets."
                ),
                "default": "observation",
            },
            "scope": {
                "type": "string",
                "enum": ["auto", "personal", "workspace", "repository", "branch", "feature"],
                "description": (
                    "Durable scope. auto chooses repository, then workspace, "
                    "then personal. Do not use personal for repository facts."
                ),
                "default": "auto",
            },
            "verification": {
                "type": "string",
                "enum": ["unverified", "source_backed", "tool_verified", "user_confirmed"],
                "default": "unverified",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence from 0.0 to 1.0.",
                "default": 0.7,
            },
            "pinned": {
                "type": "boolean",
                "description": "Guarantee recall when scope matches, within the pin budget.",
                "default": False,
            },
            "egress_policy": {
                "type": "string",
                "enum": ["never", "local_only", "any"],
                "description": "Which model providers may receive this memory during recall.",
                "default": "any",
            },
            "supersedes_id": {
                "type": "string",
                "description": "Optional durable UUID that this new memory supersedes.",
            },
            "status": {
                "type": "string",
                "enum": ["active", "done", "superseded", "archived", "stale"],
                "description": (
                    "Lifecycle state of this entry. Defaults to 'active'. "
                    "Use 'done' when work is complete, 'superseded' when a "
                    "newer entry replaces this one, 'archived' to remove "
                    "from search/summary but retain audit trail."
                ),
                "default": "active",
            },
        },
        "required": ["content"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def save_memory(args: Dict[str, Any], context) -> str:
    kind = str(args.get("kind", "observation") or "observation").strip()
    status = str(args.get("status", "active") or "active").strip()
    content = str(args.get("content", "") or "")
    session = getattr(context, "session", None)
    durable_item = None
    durable_created = False
    durable_error = ""
    service = _durable_memory(context)
    if service is not None and kind != "goal":
        from mu.memory.service import MemoryRejectedError

        try:
            durable_item, durable_created = service.remember(
                session,
                content,
                kind=kind,
                scope=str(
                    args.get("scope")
                    or getattr(session, "variables", {}).get(
                        "durable_memory_default_scope", "auto"
                    )
                ),
                tags=args.get("tags", []),
                source_refs=[
                    {
                        "type": "model_tool",
                        "tool": "save_memory",
                        "source": str(args.get("source", "") or ""),
                    }
                ],
                actor="model",
                trust_origin="model",
                verification=str(args.get("verification", "unverified") or "unverified"),
                confidence=float(args.get("confidence", 0.7) or 0.7),
                egress_policy=str(args.get("egress_policy", "any") or "any"),
                pinned=bool(args.get("pinned", False)),
                lifecycle=(
                    "archived"
                    if status == "archived"
                    else "needs_review"
                    if status in {"stale", "superseded"}
                    else "active"
                ),
                supersedes_id=str(args.get("supersedes_id", "") or ""),
                reason="model save_memory tool",
            )
        except MemoryRejectedError as exc:
            return f"Memory not stored: {exc}."
        except Exception as exc:  # durable failure must not break working memory
            durable_error = str(exc)
    entry = _task_memory(context).save(
        content,
        tags=args.get("tags", []),
        source=args.get("source", ""),
        kind=kind,
        status=status,
    )
    if durable_item is not None:
        entry.durable_id = durable_item.id
        if session is not None:
            writes = getattr(session, "_turn_durable_writes", None)
            if writes is None:
                writes = []
                session._turn_durable_writes = writes
            for index, item in enumerate(writes):
                if getattr(item, "id", "") == durable_item.id:
                    writes[index] = durable_item
                    break
            else:
                writes.append(durable_item)
        verb = "created" if durable_created else "reinforced"
        return (
            f"Saved working memory #{entry.id}; durable memory {verb} "
            f"{durable_item.id} [{durable_item.scope_type}/{durable_item.kind}] "
            f"with tags={entry.tags}. No approval required; visible in Memory Center."
        )
    suffix = f" Durable ledger unavailable: {durable_error}." if durable_error else ""
    return (
        f"Saved memory #{entry.id} [kind={entry.kind}, status={entry.status}] "
        f"with tags={entry.tags}.{suffix}"
    )


@tool(
    name="search_memory",
    description=(
        "Searches the in-task memory store for previously saved facts. "
        "By default returns ACTIVE + STALE entries — a search hit on a "
        "STALE (decayed) entry reactivates it to ACTIVE, so retrieving "
        "relevant-but-forgotten knowledge brings it back to the working "
        "set automatically. Pass a status filter or include_all=True to "
        "see done/superseded/archived entries. Use kind to filter by entry "
        "classification (decision/finding/observation/goal)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms to match against memory content, tags, and sources.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memory entries to return.",
                "default": 5,
            },
            "status": {
                "type": "string",
                "enum": ["active", "done", "superseded", "archived", "stale"],
                "description": (
                    "Filter by lifecycle status. If omitted, defaults to "
                    "active + stale. Pass 'done' or 'superseded' to see "
                    "historical entries of that type."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["decision", "finding", "observation", "goal"],
                "description": "Filter by entry classification.",
            },
            "tags_exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude entries that have any of these tags.",
            },
            "include_all": {
                "type": "boolean",
                "description": (
                    "If True, return all entries regardless of status "
                    "(overrides status filter). Use for full audit/debugging."
                ),
                "default": False,
            },
        },
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def search_memory(args: Dict[str, Any], context) -> str:
    store = _task_memory(context)
    status = args.get("status")
    kind = args.get("kind")
    tags_exclude = args.get("tags_exclude")
    include_all = bool(args.get("include_all", False))
    entries = store.search(
        args.get("query", ""),
        limit=_int_arg(args, "limit", 5),
        status_filter=status,
        kind_filter=kind,
        tags_exclude=tags_exclude,
        include_all=include_all,
    )
    parts = [store.format_results(entries)]
    service = _durable_memory(context)
    session = getattr(context, "session", None)
    if service is not None:
        try:
            seen = {
                str(getattr(entry, "durable_id", "") or "")
                for entry in entries
                if getattr(entry, "durable_id", "")
            }
            durable = [
                item
                for item in service.search_for_session(
                    session,
                    str(args.get("query", "") or ""),
                    limit=_int_arg(args, "limit", 5),
                )
                if item.id not in seen
            ]
            if durable:
                parts.append(
                    "Cross-session Memory Ledger:\n"
                    + "\n".join(
                        f"[D:{item.id}] [{item.scope_type}/{item.lifecycle}] "
                        f"kind={item.kind} :: {item.statement}"
                        for item in durable
                    )
                )
        except Exception:
            pass
    return "\n\n".join(parts)


@tool(
    name="list_memory",
    description=(
        "Lists the most recent in-task memory entries. Pass status to "
        "filter by lifecycle state (active/done/superseded/archived/stale). "
        "If omitted, lists all statuses."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of memory entries to return.",
                "default": 10,
            },
            "status": {
                "type": "string",
                "enum": ["active", "done", "superseded", "archived", "stale"],
                "description": (
                    "Filter by lifecycle status. If omitted, lists all statuses."
                ),
            },
        },
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def list_memory(args: Dict[str, Any], context) -> str:
    store = _task_memory(context)
    status = args.get("status")
    entries = store.list_entries(
        limit=_int_arg(args, "limit", 10),
        status_filter=status,
    )
    parts = [store.format_results(entries)]
    service = _durable_memory(context)
    session = getattr(context, "session", None)
    if service is not None:
        try:
            durable = service.list_for_session(
                session,
                lifecycle=(
                    status
                    if status in {"active", "archived", "superseded"}
                    else None
                ),
                limit=_int_arg(args, "limit", 10),
            )
            if durable:
                parts.append(
                    "Cross-session Memory Ledger:\n"
                    + "\n".join(
                        f"[D:{item.id}] [{item.scope_type}/{item.lifecycle}] "
                        f"kind={item.kind} :: {item.statement}"
                        for item in durable
                    )
                )
        except Exception:
            pass
    return "\n\n".join(parts)


@tool(
    name="manage_durable_memory",
    description=(
        "Curates a visible cross-session Memory Ledger record without asking "
        "the user for approval. Use archive for knowledge that should stop "
        "being recalled, needs_review for stale or uncertain knowledge, and "
        "pin sparingly for always-relevant scoped knowledge. Permanent Forget "
        "is intentionally a user-facing privacy control, not a model action."
    ),
    parameters={
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Durable UUID or unique compact prefix from search_memory.",
            },
            "action": {
                "type": "string",
                "enum": ["pin", "unpin", "archive", "restore", "needs_review"],
            },
            "reason": {
                "type": "string",
                "description": "Short audit reason for the lifecycle change.",
            },
        },
        "required": ["memory_id", "action"],
    },
)
def manage_durable_memory(args: Dict[str, Any], context) -> str:
    service = _durable_memory(context)
    session = getattr(context, "session", None)
    if service is None or session is None:
        return "Durable Memory Ledger is unavailable; working memory was unchanged."
    memory_ref = str(args.get("memory_id", "") or "").strip()
    action = str(args.get("action", "") or "").strip().lower()
    ledger_action = "mark_needs_review" if action == "needs_review" else action
    try:
        item = service.get_for_session(session, memory_ref)
        if item is None:
            return f"Durable memory {memory_ref!r} was not found in active scopes."
        item = service.ledger.action(
            item.id,
            ledger_action,
            actor="model",
            reason=str(args.get("reason", "") or f"model {action}"),
        )
    except Exception as exc:
        return f"Durable memory management failed: {exc}."
    return (
        f"Durable memory {item.id} is now {item.lifecycle}"
        f"{' and pinned' if item.pinned else ''}. The change is visible in Memory Center."
    )


# ---------------------------------------------------------------- lifecycle tools


@tool(
    name="update_memory_status",
    description=(
        "Update the lifecycle status of a memory entry. Valid statuses: "
        "active, done, superseded, archived, stale. Use 'done' when the "
        "work described is complete, 'superseded' when a newer entry "
        "replaces it, 'archived' to remove from search/summary but retain "
        "audit trail."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entry_id": {
                "type": "integer",
                "description": "The numeric id of the memory entry to update.",
            },
            "status": {
                "type": "string",
                "enum": ["active", "done", "superseded", "archived", "stale"],
                "description": "The new lifecycle status.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "Optional reason for the transition. If status='superseded' "
                    "and reason contains a numeric entry ID, that ID is set as "
                    "superseded_by."
                ),
            },
        },
        "required": ["entry_id", "status"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def update_memory_status(args: Dict[str, Any], context) -> str:
    store = _task_memory(context)
    entry_id = _int_arg(args, "entry_id", 0)
    if entry_id <= 0:
        return "Error: entry_id must be a positive integer."

    status = str(args.get("status", "") or "").strip()
    from mu.memory.stores import ALLOWED_STATUSES

    if status not in ALLOWED_STATUSES:
        return f"Error: Invalid status {status!r}. Valid: {', '.join(sorted(ALLOWED_STATUSES))}."

    entry = store.get_entry(entry_id)
    if entry is None:
        return f"Error: No memory entry with id #{entry_id}."

    old_status = entry.status
    reason = str(args.get("reason", "") or "").strip()

    # If superseded and reason contains an entry ID reference, set superseded_by
    if status == "superseded" and reason:
        import re

        id_match = re.search(r"#(\d+)", reason)
        if id_match:
            new_id = int(id_match.group(1))
            if store.get_entry(new_id) is not None:
                result = store.supersede(entry_id, new_id)
                if result is not None:
                    _sync_durable_status(context, entry, "superseded", reason)
                    return (
                        f"Memory #{entry_id} status: {old_status} → superseded "
                        f"(superseded_by=#{new_id})."
                    )
        # Fall through to regular update if supersede didn't work

    updated = store.update_status(entry_id, status)
    if updated is None:
        return f"Error: Could not update memory #{entry_id}."

    _sync_durable_status(context, updated, status, reason)

    return f"Memory #{entry_id} status: {old_status} → {status}."


@tool(
    name="supersede_memory",
    description=(
        "Mark an old memory entry as superseded by a newer one. Sets "
        "old.status='superseded', old.superseded_by=new_id, and "
        "new.supersedes=old_id. Both entries must exist. This is a "
        "singly-linked list, not a tree."
    ),
    parameters={
        "type": "object",
        "properties": {
            "old_id": {
                "type": "integer",
                "description": "The id of the entry being superseded.",
            },
            "new_id": {
                "type": "integer",
                "description": "The id of the entry that replaces it.",
            },
        },
        "required": ["old_id", "new_id"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def supersede_memory(args: Dict[str, Any], context) -> str:
    store = _task_memory(context)
    old_id = _int_arg(args, "old_id", 0)
    new_id = _int_arg(args, "new_id", 0)

    if old_id <= 0 or new_id <= 0:
        return "Error: old_id and new_id must be positive integers."

    old_entry = store.get_entry(old_id)
    new_entry = store.get_entry(new_id)
    if old_entry is None:
        return f"Error: No memory entry with id #{old_id}."
    if new_entry is None:
        return f"Error: No memory entry with id #{new_id}."

    result = store.supersede(old_id, new_id)
    if result is None:
        return f"Error: Could not supersede memory #{old_id}."

    old, new, old_status, new_status = result
    old_durable = str(getattr(old, "durable_id", "") or "")
    new_durable = str(getattr(new, "durable_id", "") or "")
    service = _durable_memory(context)
    if service is not None and old_durable and new_durable:
        try:
            old_item = service.ledger.get(old_durable)
            new_item = service.ledger.get(new_durable)
            if old_item is not None and new_item is not None:
                service.ledger.revise(
                    old_durable,
                    {
                        "lifecycle": "superseded",
                        "relations": [
                            *old_item.relations,
                            {"type": "superseded_by", "target_id": new_durable},
                        ],
                    },
                    actor="model",
                    reason=f"superseded by working memory #{new_id}",
                )
                service.ledger.revise(
                    new_durable,
                    {
                        "relations": [
                            *new_item.relations,
                            {"type": "supersedes", "target_id": old_durable},
                        ]
                    },
                    actor="model",
                    reason=f"supersedes working memory #{old_id}",
                )
        except Exception:
            pass
    return (
        f"Memory #{old_id} [{old_status} → superseded] superseded by #{new_id}. "
        f"Memory #{new_id} now supersedes #{old_id}."
    )


@tool(
    name="retire_memory",
    description=(
        "Mark a memory entry as done — the work it describes is complete. "
        "Entry stays searchable but deprioritized in search and summary. "
        "Shorthand for update_memory_status(entry_id, 'done')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entry_id": {
                "type": "integer",
                "description": "The id of the memory entry to retire.",
            },
        },
        "required": ["entry_id"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def retire_memory(args: Dict[str, Any], context) -> str:
    store = _task_memory(context)
    entry_id = _int_arg(args, "entry_id", 0)
    if entry_id <= 0:
        return "Error: entry_id must be a positive integer."

    entry = store.get_entry(entry_id)
    if entry is None:
        return f"Error: No memory entry with id #{entry_id}."

    old_status = entry.status
    updated = store.update_status(entry_id, "done")
    if updated is None:
        return f"Error: Could not retire memory #{entry_id}."

    return f"Memory #{entry_id} retired: {old_status} → done."


@tool(
    name="reactivate_memory",
    description=(
        "Set a memory entry's status back to 'active'. Clears "
        "superseded_by if set. Use when revisiting completed or "
        "superseded work that is now relevant again."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entry_id": {
                "type": "integer",
                "description": "The id of the memory entry to reactivate.",
            },
        },
        "required": ["entry_id"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def reactivate_memory(args: Dict[str, Any], context) -> str:
    store = _task_memory(context)
    entry_id = _int_arg(args, "entry_id", 0)
    if entry_id <= 0:
        return "Error: entry_id must be a positive integer."

    entry = store.get_entry(entry_id)
    if entry is None:
        return f"Error: No memory entry with id #{entry_id}."

    old_status = entry.status
    # Clear superseded_by when reactivating
    if entry.superseded_by is not None:
        entry.superseded_by = None

    updated = store.update_status(entry_id, "active")
    if updated is None:
        return f"Error: Could not reactivate memory #{entry_id}."

    _sync_durable_status(context, updated, "active", "model reactivated working memory")

    return f"Memory #{entry_id} reactivated: {old_status} → active."


@tool(
    name="archive_memory",
    description=(
        "Archive a memory entry — removes it from search results (unless "
        "include_all=True) and from the system-prompt summary, but retains "
        "it in the store for audit trail. Use for old project context or "
        "superseded decisions with no replacement."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entry_id": {
                "type": "integer",
                "description": "The id of the memory entry to archive.",
            },
        },
        "required": ["entry_id"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def archive_memory(args: Dict[str, Any], context) -> str:
    store = _task_memory(context)
    entry_id = _int_arg(args, "entry_id", 0)
    if entry_id <= 0:
        return "Error: entry_id must be a positive integer."

    entry = store.get_entry(entry_id)
    if entry is None:
        return f"Error: No memory entry with id #{entry_id}."

    old_status = entry.status
    updated = store.update_status(entry_id, "archived")
    if updated is None:
        return f"Error: Could not archive memory #{entry_id}."

    _sync_durable_status(context, updated, "archived", "model archived working memory")

    return f"Memory #{entry_id} archived: {old_status} → archived."


# ------------------------------------------------------------ retire_thread


@tool(
    name="retire_thread",
    description=(
        "Explicitly drop an investigation or work thread you have abandoned — "
        "the 'I'm done carrying this' lever for self-managed context. Archives "
        "every ACTIVE task-memory entry whose content, tags, or source "
        "matches the given topic (so they stop appearing in the default "
        "active-only search and the system-prompt summary), optionally "
        "removes matching scratchpad notes, and writes a single archived "
        "audit entry recording the drop with your reason. Use when the "
        "user's ask has shifted, a hypothesis was disproved and you're "
        "moving on, or a sub-thread is simply no longer relevant."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "Keyword or substring identifying the thread to drop. "
                    "Matched case-insensitively against memory content, tags, "
                    "and source, and against scratchpad note content."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Why the thread is being dropped (audit trail).",
            },
            "clear_scratchpad": {
                "type": "boolean",
                "description": (
                    "If True (default), also remove non-todo scratchpad notes "
                    "whose content contains the topic. Todo ledger entries "
                    "are never touched here — prune those with todo_delete "
                    "or todo_clear."
                ),
                "default": True,
            },
        },
        "required": ["topic"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="json",
)
def retire_thread(args: Dict[str, Any], context) -> Dict[str, Any]:
    topic = str(args.get("topic", "") or "").strip()
    if not topic:
        return {
            "ok": False,
            "error_code": "invalid_args",
            "message": "retire_thread requires non-empty 'topic'.",
            "data": {},
            "artifacts": [],
            "telemetry": {"tool_name": "retire_thread"},
        }
    reason = str(args.get("reason", "") or "").strip()
    clear_scratch = bool(args.get("clear_scratchpad", True))
    topic_l = topic.lower()

    store = _task_memory(context)
    # Active entries matching the topic → archived. Keep audit trail.
    archived_ids: list[int] = []
    for entry in list(store.entries):
        if entry.status != "active":
            continue
        haystack = " ".join([
            entry.content or "",
            " ".join(entry.tags or []),
            entry.source or "",
        ]).lower()
        if topic_l in haystack:
            if store.update_status(entry.id, "archived") is not None:
                archived_ids.append(entry.id)

    # Audit entry recording the drop.
    audit_text = f"Thread retired: {topic}"
    if reason:
        audit_text += f" — {reason}"
    store.save(
        audit_text,
        tags=["retired", "abandoned"],
        source="retire_thread",
        kind="observation",
        status="archived",
    )

    # Optionally drop matching scratchpad notes (never todos).
    scratch_removed = 0
    if clear_scratch:
        sp = _scratchpad(context)
        kept: list = []
        for e in sp.entries:
            if "todo" in (e.tags or []):
                kept.append(e)
                continue
            if topic_l in (e.content or "").lower():
                scratch_removed += 1
                continue
            kept.append(e)
        sp.entries = kept
        sp._next_id = (max(e.id for e in kept) + 1) if kept else 1

    return {
        "ok": True,
        "error_code": None,
        "message": (
            f"Retired thread '{topic}': archived {len(archived_ids)} active "
            f"memory entry/entries, removed {scratch_removed} scratchpad note(s)."
        ),
        "data": {
            "topic": topic,
            "archived_memory_ids": archived_ids,
            "scratchpad_removed": scratch_removed,
            "audit_recorded": True,
        },
        "artifacts": [],
        "telemetry": {"tool_name": "retire_thread"},
    }


# ---------------------------------------------------------------- scratchpad


@tool(
    name="save_scratchpad",
    description=(
        "Saves a temporary note in the current turn scratchpad. Use this "
        "for short-lived plans or observations that do not need durable memory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The temporary note to store for the current turn.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to help later retrieval during this turn.",
            },
            "source": {
                "type": "string",
                "description": "Optional source note for the scratchpad entry.",
            },
        },
        "required": ["content"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def save_scratchpad(args: Dict[str, Any], context) -> str:
    entry = _scratchpad(context).save(
        args.get("content", ""),
        tags=args.get("tags", []),
        source=args.get("source", ""),
    )
    return f"Saved scratchpad note #{entry.id} with tags={entry.tags}."


@tool(
    name="search_scratchpad",
    description="Searches turn-local scratchpad notes saved during the current task loop.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms to match against scratchpad content, tags, and sources.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of scratchpad entries to return.",
                "default": 5,
            },
        },
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def search_scratchpad(args: Dict[str, Any], context) -> str:
    store = _scratchpad(context)
    entries = store.search(args.get("query", ""), limit=_int_arg(args, "limit", 5))
    return store.format_results(entries)


@tool(
    name="list_scratchpad",
    description="Lists the most recent turn-local scratchpad entries.",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of scratchpad entries to return.",
                "default": 10,
            }
        },
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def list_scratchpad(args: Dict[str, Any], context) -> str:
    store = _scratchpad(context)
    entries = store.list_entries(limit=_int_arg(args, "limit", 10))
    return store.format_results(entries)


@tool(
    name="clear_scratchpad",
    description="Clears the current turn scratchpad.",
    parameters={"type": "object", "properties": {}},
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def clear_scratchpad(args: Dict[str, Any], context) -> str:
    _scratchpad(context).clear()
    return "Turn scratchpad cleared."


# ---------------------------------------------------------------- tool result cache


@tool(
    name="recall",
    description=(
        "Recall a previously-cached tool result by its cache key. "
        "When the L4 compression summary shows [cache:KEY], call this tool "
        "with the key to fetch the full original result — no need to re-read "
        "files or re-run searches."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key": {
                "type": "string",
                "description": "The cache key from a [cache:KEY] tag in the compressed summary.",
            },
        },
        "required": ["cache_key"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def recall(args: Dict[str, Any], context) -> str:
    """Fetch a cached tool result by its cache key.

    The L4 compression system stores full tool results in a sidecar cache
    before compressing them into short summary lines. This tool retrieves
    the original full result — avoiding re-reading files or re-running searches.
    """
    import json as _json

    session = getattr(context, "session", None)
    if session is None or not hasattr(session, "tool_result_cache"):
        return "Error: No tool result cache available on this session."
    key = args.get("cache_key", "")
    if not key:
        return "Error: cache_key argument is required."
    result = session.tool_result_cache.recall(key)
    if result is None:
        return (
            f"Cache key '{key}' not found or evicted. "
            "The result may have been dropped due to LRU eviction. "
            "Re-run the original tool call if needed."
        )
    return _json.dumps(result, default=str, indent=2)

# ---------------------------------------------------------------- bounded retrieval (spec #6)
#
# A family of read-only ops over the durable ResultStore / ToolResultCache.
# Each takes a `cache_key` (the stored_ref embedded in a compact observation)
# and returns a bounded slice of the stored raw — so the model can pull exact
# details back into context without re-running the original tool. Results are
# themselves subject to the observation transform if they exceed the inline
# budget.


def _cache_for(context):
    session = getattr(context, "session", None)
    if session is None or not hasattr(session, "tool_result_cache"):
        return None, "Error: No tool result cache available on this session."
    return session.tool_result_cache, None


def _require_key(args):
    key = args.get("cache_key", "")
    if not key:
        return None, "Error: cache_key argument is required."
    return key, None


@tool(
    name="result_range",
    description=(
        "Retrieve a line range [start_line, end_line] from a stored tool result "
        "(cache_key from a stored_ref). 1-indexed, inclusive. Use instead of "
        "re-running read_file/get_chunk when the full result is already stored."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key": {"type": "string", "description": "Stored result key."},
            "start_line": {"type": "integer", "description": "1-indexed start line (inclusive)."},
            "end_line": {"type": "integer", "description": "1-indexed end line (inclusive)."},
        },
        "required": ["cache_key", "start_line", "end_line"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def result_range(args: Dict[str, Any], context) -> str:
    cache, err = _cache_for(context)
    if err:
        return err
    key, err = _require_key(args)
    if err:
        return err
    start = int(args.get("start_line", 1))
    end = int(args.get("end_line", start))
    out = cache.line_range(key, start, end)
    if out is None:
        return f"Cache key '{key}' not found or evicted."
    return out


@tool(
    name="result_head",
    description=(
        "Retrieve the first N lines of a stored tool result (cache_key from a "
        "stored_ref). Default 20 lines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key": {"type": "string", "description": "Stored result key."},
            "lines": {"type": "integer", "description": "Number of lines (default 20)."},
        },
        "required": ["cache_key"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def result_head(args: Dict[str, Any], context) -> str:
    cache, err = _cache_for(context)
    if err:
        return err
    key, err = _require_key(args)
    if err:
        return err
    n = int(args.get("lines", 20))
    out = cache.head(key, n)
    if out is None:
        return f"Cache key '{key}' not found or evicted."
    return out


@tool(
    name="result_tail",
    description=(
        "Retrieve the last N lines of a stored tool result (cache_key from a "
        "stored_ref). Default 20 lines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key": {"type": "string", "description": "Stored result key."},
            "lines": {"type": "integer", "description": "Number of lines (default 20)."},
        },
        "required": ["cache_key"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def result_tail(args: Dict[str, Any], context) -> str:
    cache, err = _cache_for(context)
    if err:
        return err
    key, err = _require_key(args)
    if err:
        return err
    n = int(args.get("lines", 20))
    out = cache.tail(key, n)
    if out is None:
        return f"Cache key '{key}' not found or evicted."
    return out


@tool(
    name="result_search",
    description=(
        "Search a stored tool result (cache_key) for a query string; returns "
        "grouped matches with line numbers. Use instead of re-running a search "
        "tool when the raw output is already stored."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key": {"type": "string", "description": "Stored result key."},
            "query": {"type": "string", "description": "Substring or pattern to find."},
            "max_matches": {"type": "integer", "description": "Cap on matches (default 20)."},
        },
        "required": ["cache_key", "query"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def result_search(args: Dict[str, Any], context) -> str:
    cache, err = _cache_for(context)
    if err:
        return err
    key, err = _require_key(args)
    if err:
        return err
    query = args.get("query", "")
    if not query:
        return "Error: query argument is required."
    max_m = int(args.get("max_matches", 20))
    out = cache.search(key, query, max_matches=max_m)
    if out is None:
        return f"Cache key '{key}' not found or evicted."
    return out


@tool(
    name="result_diagnostics",
    description=(
        "Extract unique error/warning/traceback lines from a stored tool result "
        "(cache_key). Deduped, noise dropped. Use to inspect why a stored bash/"
        "build/test result failed without pulling the full log back into context."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key": {"type": "string", "description": "Stored result key."},
            "max_lines": {"type": "integer", "description": "Cap on diagnostic lines (default 12)."},
        },
        "required": ["cache_key"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def result_diagnostics(args: Dict[str, Any], context) -> str:
    cache, err = _cache_for(context)
    if err:
        return err
    key, err = _require_key(args)
    if err:
        return err
    max_l = int(args.get("max_lines", 12))
    out = cache.diagnostics(key, max_lines=max_l)
    if out is None:
        return f"Cache key '{key}' not found or evicted."
    if not out:
        return "No diagnostic lines found."
    return out


@tool(
    name="result_json_path",
    description=(
        "Extract a value from a stored JSON tool result (cache_key) via a JSON "
        "pointer (e.g. /data/matches/0/file). Returns the targeted slice only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key": {"type": "string", "description": "Stored result key."},
            "pointer": {"type": "string", "description": "JSON pointer path (e.g. /data/0/file)."},
        },
        "required": ["cache_key", "pointer"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def result_json_path(args: Dict[str, Any], context) -> str:
    cache, err = _cache_for(context)
    if err:
        return err
    key, err = _require_key(args)
    if err:
        return err
    pointer = args.get("pointer", "")
    if not pointer:
        return "Error: pointer argument is required."
    out = cache.json_path(key, pointer)
    if out is None:
        return f"Cache key '{key}' not found, evicted, or pointer not present."
    return out


@tool(
    name="compare_results",
    description=(
        "Produce a unified diff between two stored tool results (cache_key_a, "
        "cache_key_b). Use to compare two reads of the same file or two search "
        "runs without re-running either tool."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cache_key_a": {"type": "string", "description": "First stored result key."},
            "cache_key_b": {"type": "string", "description": "Second stored result key."},
        },
        "required": ["cache_key_a", "cache_key_b"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
)
def compare_results(args: Dict[str, Any], context) -> str:
    cache, err = _cache_for(context)
    if err:
        return err
    a = args.get("cache_key_a", "")
    b = args.get("cache_key_b", "")
    if not a or not b:
        return "Error: both cache_key_a and cache_key_b are required."
    out = cache.compare(a, b)
    if out is None:
        return f"One or both cache keys ('{a}', '{b}') not found or evicted."
    if not out:
        return "Results are identical."
    return out


# ---------------------------------------------------------------- phased tool exposure (spec #9)


@tool(
    name="load_tools",
    description=(
        "Activate a specialist tool phase (e.g. 'research', 'security', "
        "'feature', 'teacher') so its tools appear in your schema. Use when "
        "lazy_tools_enabled is on and you need a specialist tool that isn't "
        "in the core phase. Core tools are always available."
    ),
    parameters={
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "description": "The tool phase to activate (e.g. research, security).",
            },
        },
        "required": ["phase"],
    },
    requires_approval=False,
    execution_kind="memory",
    preview_policy="none",
    server_policy="session_only",
    result_mode="raw",
    phase="core",
)
def load_tools(args: Dict[str, Any], context) -> str:
    session = getattr(context, "session", None)
    phase = str(args.get("phase", "")).strip().lower()
    if not phase:
        return "Error: phase argument is required."
    if session is None:
        return "Error: no session available."
    loaded = set(getattr(session, "_loaded_tool_phases", []) or [])
    loaded.add(phase)
    session._loaded_tool_phases = sorted(loaded)
    from mu.tools.descriptors import resolve_active_tool_phases

    effective_phases = resolve_active_tool_phases(
        getattr(session, "variables", None),
        session._loaded_tool_phases,
    )
    session._active_tool_phases = tuple(effective_phases)
    # Count how many tools are now newly exposed in this phase.
    try:
        from mu.tools.descriptors import TOOL_DESCRIPTORS

        count = sum(
            1 for d in TOOL_DESCRIPTORS.values() if d.phase == phase
        )
    except Exception:  # noqa: BLE001
        count = 0
    return (
        f"Activated tool phase '{phase}'. {count} tool(s) tagged with that "
        f"phase will appear in the next request's schema. Active phases: "
        f"{', '.join(effective_phases)}."
    )
