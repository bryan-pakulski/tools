"""Tool-execution glue between the session loop and the dispatcher.

Three functions, all taking the live `session` as their first argument
so they can read/mutate per-turn state (hook abort flag, feature-state
syncs, the `_loop_blocker_raised` watchdog signal, etc.):

  * `execute_tool_with_memory(session, name, args)` — fires the
    `pre_tool` / `post_tool` hooks around `execute_tool`, honors
    short-circuit (plan-mode, secret-guard, custom hooks) and abort
    return values, and runs the feature-mode "no writes outside docs"
    check.

  * `build_structured_tool_result(session, name, args, raw_result)` —
    wraps a raw tool result string in the structured envelope the
    history stores (summary, args, raw, error_code, modified_files,
    telemetry). Per-tool data-extraction branches handle the cases
    where the model wants typed access (read_file → char_count, etc.).

  * `sync_feature_state_for_tool(session, name, args, raw, structured)`
    — when the just-executed tool was a feature-mode mutator or
    `raise_blocker`, write its result back into the session's feature
    state so the next turn sees the updated plan / blocker.

Tests covering these paths live in `tests/test_mu_agent_session_integration.py`
(pre/post_tool hooks, plan-mode block, abort flag), `tests/test_session.py`
(structured-result shape), and `tests/test_loop_blocker_halts_watchdog.py`
(the raise_blocker → `_loop_blocker_raised` interlock).
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------- hook-fire dispatch


def execute_tool_with_memory(
    session: Any,
    tool_name: str,
    tool_args: dict,
    *,
    invocation_source: str = "session",
) -> Any:
    """Fire pre_tool/post_tool hooks around the dispatcher.

    Returns whatever the tool produced (string or envelope dict),
    unless a `pre_tool` hook short-circuited (in which case the hook's
    payload is returned) or fired abort (synthetic
    `error_code=hook_aborted` envelope returned + the iteration loop
    sees `session._hook_abort_requested == True` next time around).
    """
    # Local imports to dodge cold-import overhead — these modules are
    # not always loaded when tools_glue itself is imported.
    from mu.tools._dispatcher import execute_tool
    from mu.session.helpers import _hook_abort_envelope
    from mu.agent.hooks import HookContext, default_registry

    pre_ctx = HookContext(
        point="pre_tool",
        session=session,
        variables=session.variables,
        tool_name=tool_name,
        tool_args=tool_args,
    )
    _, short, abort = default_registry.fire_with_signals("pre_tool", pre_ctx)
    if short is not None:
        return short.payload
    if abort is not None:
        session._record_hook_abort("pre_tool", abort)
        return _hook_abort_envelope(tool_name, session._hook_abort_reason)

    feature_violation = session._feature_doc_tool_violation(tool_name, tool_args)
    if feature_violation:
        return f"Error: {feature_violation}"

    # --- Pre-write snapshot for workspace diff tracking ---
    # Write tools (write_file, apply_diff, search_and_replace_file) modify
    # files.  We snapshot the file's CURRENT content BEFORE the tool runs
    # so that folder_context.get_context_diff_xml() can later produce a
    # real diff (original vs modified).  Without this, lazy-loading would
    # read the already-modified content as the "original", producing no diff.
    if tool_name in {"write_file", "apply_diff", "search_and_replace_file"}:
        _filename = tool_args.get("filename", "")
        if _filename and session.folder_context:
            try:
                import os as _os
                _full = _os.path.abspath(_filename)
                _fc = session.folder_context
                # Only snapshot if not already tracked (first write this turn)
                if _full not in _fc.initial_snapshots and _fc._is_text_file(_full):
                    try:
                        _content = _fc._load_file_content(_full)
                    except Exception:
                        _content = None
                    _fc.initial_snapshots[_full] = _content
            except Exception:
                pass
    # --- End pre-write snapshot ---

    # Memory and scratchpad tools used to short-circuit here; they now
    # route through the normal dispatcher to the `@tool`-registered
    # handlers in `mu/tools/memory/handlers.py`, which resolve the
    # stores from `context.session`.
    result = execute_tool(
        tool_name,
        tool_args,
        session.folder_context,
        session.ui,
        session.variables,
        invocation_source=invocation_source,
        session=session,
    )

    post_ctx = HookContext(
        point="post_tool",
        session=session,
        variables=session.variables,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=result,
        metadata=pre_ctx.metadata,
    )
    _, _, abort = default_registry.fire_with_signals("post_tool", post_ctx)
    if abort is not None:
        session._record_hook_abort("post_tool", abort)
    return result


# ---------------------------------------------------------------- structured result


_FEATURE_MODE_TOOL_NAMES = frozenset(
    {
        "create_feature",
        "create_phases",
        "create_task",
        "get_execution_state",
        "block_task",
        "resume_task",
        "review_completed_tasks",
        "review_all_completed_tasks",
        "propose_task_diff",
        "decide_task_diff",
        "archive_task",
        "create_feature_task",
        "update_feature_task",
        "approve_feature_task",
        "get_current_task",
        "get_tasks",
        "update_task_status",
        "raise_blocker",
    }
)


_MEMORY_TOOL_NAMES = frozenset(
    {
        "save_memory",
        "search_memory",
        "list_memory",
        "save_scratchpad",
        "search_scratchpad",
        "list_scratchpad",
        "clear_scratchpad",
        "flush",
    }
)


def build_structured_tool_result(
    session: Any,
    tool_name: str,
    tool_args: dict,
    raw_result: Any,
    *,
    execution_source: str = "session",
    cache_key: Optional[str] = None,
) -> dict:
    """Wrap a raw tool result in the structured envelope the history
    stores (summary, args, raw, error_code, modified_files, telemetry).

    Per-tool branches add typed `data` fields when callers want
    structured access (`read_file` → `char_count`, `list_dir` → parsed
    tree, etc.). For un-recognized tools, `data` is left empty and the
    raw text-preview lives in `summary`.

    Spec #1/#2/#10 — budget-thresholded observation: when the raw result
    exceeds its inline token budget, the full ``raw`` is dropped from the
    in-context envelope and replaced by a compact observation (excerpt,
    diagnostics, counts) plus ``stored_ref`` pointing at the durable
    ResultStore. Small results stay verbatim. ``cache_key`` (the durable
    store key) must be supplied so the observation can embed the
    reference; when it is None the observation degrades to keeping the
    raw inline (no store backing)."""
    from mu.tools._envelope import infer_tool_error_code
    from mu.session.helpers import _shorten_tool_args
    from mu.session.messages import clip_preview

    envelope, unwrapped_raw = session._unwrap_tool_envelope(raw_result)
    raw_text = str(unwrapped_raw)
    error_code = (
        envelope.get("error_code")
        if isinstance(envelope, dict)
        else infer_tool_error_code(tool_name, raw_text)
    )
    is_error = error_code is not None
    structured = {
        "tool_name": tool_name,
        "ok": (
            bool(envelope.get("ok"))
            if isinstance(envelope, dict)
            else error_code is None
        ),
        "summary": clip_preview(raw_text, 220),
        "args": _shorten_tool_args(tool_args),
        "raw": raw_text,
        "error_code": error_code,
        "error": (
            None
            if error_code is None
            else {
                "code": error_code,
                "message": clip_preview(raw_text, 220),
            }
        ),
        "data": {},
        "modified_files": [],
        "artifacts": [],
        "telemetry": {
            "execution_source": execution_source,
            "delivery_mode": "structured",
            "raw_char_count": len(raw_text),
            "raw_line_count": len(raw_text.splitlines()),
        },
    }
    if isinstance(envelope, dict):
        # Context-optimisation: no longer double-store the full envelope copy
        # inside telemetry. It duplicated ok/error/data/message that ride on
        # the structured envelope itself, and no consumer ever read it back
        # (loop_body + trace path re-derive the envelope from raw_result).
        envelope_artifacts = envelope.get("artifacts")
        if isinstance(envelope_artifacts, list):
            structured["artifacts"] = [
                dict(item) for item in envelope_artifacts if isinstance(item, dict)
            ]
        envelope_data = envelope.get("data")
        if isinstance(envelope_data, dict):
            structured["data"] = dict(envelope_data)

    if tool_name == "read_file":
        structured["data"] = {
            "filename": tool_args.get("filename", ""),
            "char_count": len(raw_text),
            "line_count": len(raw_text.splitlines()),
            "preview": clip_preview(raw_text, 240),
        }
    elif tool_name == "get_chunk":
        structured["data"] = {
            "file": tool_args.get("file", ""),
            "start_line": tool_args.get("start_line"),
            "end_line": tool_args.get("end_line"),
            "line_count": len(raw_text.splitlines()),
            "preview": clip_preview(raw_text, 240),
        }
    elif tool_name == "search_for_string":
        structured["data"] = {
            "query": tool_args.get("string", ""),
            **session._parse_search_results(raw_text),
        }
    elif tool_name == "list_dir":
        structured["data"] = session._parse_list_dir(
            raw_text, tool_args.get("path", "")
        )
    elif tool_name == "get_workspace_details":
        structured["data"] = session._parse_workspace_details(raw_text)
    elif tool_name in {"write_file", "apply_diff", "search_and_replace_file"}:
        filename = tool_args.get("filename", "")
        structured["data"] = {
            "filename": filename,
            "changed_file": filename,
        }
        if filename:
            structured["modified_files"] = [filename]
    elif tool_name in {"upload_artifact", "list_artifacts"}:
        if isinstance(envelope, dict):
            descriptor = envelope.get("artifact")
            if isinstance(descriptor, dict):
                structured["data"]["artifact"] = dict(descriptor)
    elif tool_name in _FEATURE_MODE_TOOL_NAMES:
        structured["data"] = session._parse_json_result(raw_text)
    elif tool_name in _MEMORY_TOOL_NAMES:
        structured["data"] = {"preview": clip_preview(raw_text, 220)}

    # ---- Budget-thresholded observation (spec #1/#2/#10) -----------------
    _apply_observation_transform(session, structured, raw_text, cache_key, is_error)
    return structured


def _apply_observation_transform(
    session: Any,
    structured: dict,
    raw_text: str,
    cache_key: Optional[str],
    is_error: bool,
) -> None:
    """Drop the full ``raw`` from the in-context envelope when it exceeds the
    inline token budget, replacing it with a compact observation + stored_ref.
    Small results stay verbatim. Best-effort: any failure leaves ``raw`` in
    place (the safe, pre-change behaviour)."""
    try:
        from mu.tools._observe import build_observation, resolve_inline_budget, RETRIEVABLE_VIA
        from utils.token_estimator import estimate_tokens

        variables = getattr(session, "variables", {}) or {}
        tool_name = structured["tool_name"]
        raw_tokens = int(estimate_tokens(raw_text) or 0)
        budget = resolve_inline_budget(tool_name, is_error, variables)
        structured["telemetry"]["raw_token_count"] = raw_tokens
        structured["telemetry"]["inline_budget"] = budget
        if raw_tokens <= budget:
            structured["data"]["omitted"] = False
            structured["telemetry"]["injected_token_count"] = raw_tokens
            return
        if not cache_key:
            # No store backing → keep raw inline (can't offer a stored_ref).
            structured["data"]["omitted"] = False
            structured["telemetry"]["injected_token_count"] = raw_tokens
            return
        obs, note = build_observation(
            tool_name, None, raw_text, structured["data"],
            budget_tokens=budget, is_error=is_error,
        )
        structured["data"] = obs
        structured["data"]["stored_ref"] = cache_key
        structured["data"]["retrievable_via"] = RETRIEVABLE_VIA
        structured["data"]["omission_note"] = note
        structured["raw"] = None  # full raw NOT in context (spec #1/#11)
        structured["telemetry"]["delivery_mode"] = "observed"
        # Injected token estimate: the observation dict (no raw) + summary.
        injected = int(estimate_tokens(str(structured["data"])) or 0) + int(
            estimate_tokens(structured.get("summary") or "") or 0
        )
        structured["telemetry"]["injected_token_count"] = injected
        structured["telemetry"]["compression_ratio"] = round(
            (raw_tokens - injected) / max(1, raw_tokens), 3
        )
    except Exception:  # noqa: BLE001 — never break the loop over a budget bug
        structured.setdefault("data", {})["omitted"] = False


# ---------------------------------------------------------------- feature-state sync


def sync_feature_state_for_tool(
    session: Any,
    tool_name: str,
    tool_args: dict,
    raw_result: Any,
    structured_result: Any,
) -> None:
    """When the just-executed tool was a feature-mode mutator or
    `raise_blocker`, write its result back into the session's feature
    state so the next turn sees the updated plan / blocker.

    Mutates `session._loop_blocker_raised` when `raise_blocker` fires
    so the loop-mode watchdog knows the pause was intentional and
    skips its "continue!" prod that would otherwise burn iterations
    re-raising the same blocker."""
    if tool_name in {
        "create_feature",
        "create_phases",
        "create_task",
        "get_execution_state",
        "block_task",
        "resume_task",
        "review_completed_tasks",
        "review_all_completed_tasks",
        "propose_task_diff",
        "decide_task_diff",
        "archive_task",
        "create_feature_task",
        "get_tasks",
        "get_current_task",
        "approve_feature_task",
        "update_feature_task",
        "update_task_status",
    }:
        data = {}
        if isinstance(structured_result, dict):
            data = structured_result.get("data", {}) or {}
            if isinstance(data.get("plan"), dict):
                data = data["plan"]
        if not isinstance(data, dict) or "feature_id" not in data:
            data = session._parse_json_result(raw_result)
            if isinstance(data.get("plan"), dict):
                data = data["plan"]
        if isinstance(data, dict) and data.get("feature_id"):
            is_plan_summary = any(
                key in data
                for key in (
                    "metadata_path",
                    "directory",
                    "review_status",
                    "phases",
                    "tasks",
                    "next_task",
                    "next_phase",
                )
            )
            if is_plan_summary:
                session._set_feature_state(feature_plan=data)
            elif tool_name in {"get_current_task", "get_tasks"}:
                metadata_path = str(
                    (session.session_manager.get_feature_state() or {}).get(
                        "metadata_path", ""
                    )
                    or ""
                ).strip()
                if metadata_path:
                    session._refresh_feature_state(metadata_path)
        return

    if tool_name == "raise_blocker":
        data = {}
        if isinstance(structured_result, dict):
            data = structured_result.get("data", {}) or {}
        if not isinstance(data, dict) or not data.get("kind"):
            data = session._parse_json_result(raw_result)
        if isinstance(data, dict):
            session._set_feature_state(status="awaiting_input", blocker=data)
        # Signal the loop-mode watchdog that this pause is intentional
        # — without this it would re-prompt the model with LOOP WATCHDOG
        # every iteration, forcing repeated re-raises until budget is
        # exhausted.
        session._loop_blocker_raised = True
        return


__all__ = [
    "execute_tool_with_memory",
    "build_structured_tool_result",
    "sync_feature_state_for_tool",
]
