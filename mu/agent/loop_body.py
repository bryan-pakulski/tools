"""Body of the agentic turn loop.

`run_turn(session, text)` is the function that drives one user turn
end-to-end: prepares the user message, assembles the system prompt
through every hierarchical layer, runs the agentic loop until the
model emits a final response or hits `max_iterations`, dispatches
tool calls (including parallel batches), and returns a structured
turn-response dict.

`Session.send_message` is a 3-line forwarder into `run_turn` here;
the body uses `session.<attr>` for every state access.

The control flow:

  1. Reset per-turn state (paused_execution_text, hook abort flag,
     loop blocker flag).
  2. Build the new user message (apply feature/loop mode prompt
     transforms; attach staged files).
  3. Compose the system prompt: agentic harness + mode-specific text
     + workspace context (retrieval-first when available) + L1–L5
     hierarchical layers.
  4. Roll history under the compaction budget.
  5. Loop: pre-turn hook abort check → provider stream → dispatch
     tool calls (serial or parallel) → post-process structured
     results → repeat until no tool calls OR hit max_iterations OR
     hook aborts OR user interrupts.
  6. Collect the turn response and return.

Hooks fire from this body via `session._execute_tool_with_memory`
(pre_tool/post_tool) and `session._provider_generate_with_retry`
(pre_provider_call/post_provider_call/on_stop). The body itself
doesn't fire hooks directly — see `mu/agent/hooks.py` for the points.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import traceback
import uuid
from collections import deque
from typing import Any

from mu.agent.approval import ApprovalPlan, build_approval_prompt, collect_approval_plans
from mu.agent.retry import is_context_overflow_error, parse_overflow_token_counts
from mu.feature.engine import refresh_and_persist_feature_plan, summarize_feature_plan
from mu.tools._dispatcher import execute_tool
from mu.tools._envelope import infer_tool_error_code
from mu.tools.capabilities import (
    filter_tools_for_session_type,
    normalize_session_type,
    tools_enabled_without_workspace,
)
from mu.trace.emitter import emit_nudge, emit_tool
from mu.tools.descriptors import (
    COLLATED_TOOLS,
    TOOLS,
    filter_tools_by_phase,
    resolve_active_tool_phases,
)
from providers.base import FileReference, ImageData, Message, MessagePart
from utils.config import (
    NUDGE_EMPTY_RESPONSE,
    NUDGE_EMPTY_RESPONSE_CHILD,
    SESSION_TYPE_PROMPTS,
    calculate_cost,
)
from utils.helpers import display_image_in_terminal, get_safe_mime_type
from utils.logger import logger
from utils.runtime_metrics import build_live_status_line
from utils.token_estimator import estimate_tokens


# Shared symbols extracted to `mu/session/helpers.py` so this top-level
# import is safe: helpers.py has no Session/SessionManager dependencies
# and so doesn't trigger the loop_body ↔ session.py cycle.
from mu.session.helpers import (
    _HookAbort,
    _hook_abort_envelope,
    _sanitize_for_log,
    _shorten_tool_args,
)


from mu.agent.context_guard import (  # noqa: F401
    _MAX_OVERFLOW_RECOVERIES_PER_TURN,
    _estimate_messages_tokens,
    _estimate_tools_tokens,
    _preflight_context_check,
    _resolve_real_context_window,
    _overflow_drift_ratio,
    _calibrate_drift_from_response,
    _aggressive_compact_for_overflow,
    _generate_with_overflow_recovery,
    _maybe_nudge_context_pressure,
)
from mu.agent.teacher_watcher import (
    _render_learner_profile_block,
    _run_teacher_watcher_assistant,
    _run_teacher_watcher_user,
)

def _empty_flush_message(session) -> str:
    """Self-diagnosing empty-flush message (increment 12).

    An empty flush used to return bare "No data in collation buffer to
    flush.", which models read as silent data loss — children (collation
    disabled, inline delivery) and single-tool-call batches then re-derived
    evidence they already held (observed: 10 redundant reads, ~8 wasted
    iterations in subagent run_db7ad867a85b). The message now states which
    delivery state the session is in, so the model acts on results it
    already has instead of re-gathering.
    """
    if not session.variables.get("collation_enabled", True):
        return (
            "Collation is disabled in this session — "
            "read-only results are delivered inline as "
            "they execute. Nothing was dropped; act on "
            "the results already in the conversation. "
            "No `flush` calls are needed here."
        )
    return (
        "No data in collation buffer to flush. "
        "Read-only results deliver inline unless a "
        "tool result explicitly said \"Stored ... in "
        "collation buffer\" with an artifact_id — "
        "only those are pending here. Nothing was "
        "dropped."
    )


def _filter_state_capsule_duplicates(payload: str, state_capsule: str) -> str:
    """Drop payload lines already duplicated by the L2 state capsule.

    The deterministic state capsule (LAYER 2) projects task-memory decisions
    and scratchpad todos into "Durable decisions and findings" / "Open work
    ledger". The LAYER 3 working-memory / scratchpad snapshots re-render the
    same stores, so long sessions carried every decision and todo twice per
    prompt. Frame prefixes ("- #id", "- [kind] (src):") are the layer's own
    rendering, not substance - the normalized core is what the capsule
    would duplicate.

    Two containment orders are checked per line core:
    1. full containment - the core appears anywhere in the capsule
       (short entries render whole in both layers);
    2. truncated-capsule prefix - capsule entries are char-capped and end
       with "...", so a long payload core can never be a substring of the
       truncated capsule text; when the capsule entry (minus its ellipsis,
       >=32 normalized chars) is a PREFIX of the payload core, the line is
       the same store entry and is dropped. Live evidence: increment-11/12
       memory entries (~700 chars) rendered fully in L3 while the capsule
       carried their ~220-char prefixes - full containment silently no-op'd.

    >=24-char cores only, so framing noise ("ok", "- #1 [active]") never
    matches. Capsule lines are frame-stripped identically before both checks.
    """
    if not payload or not state_capsule:
        return payload

    def _norm(text: str) -> str:
        return " ".join(text.split()).lower()

    frame_re = r"^\-\s*(#\d+\s*)?(\[[^\]]+\]\s*)*(\([^)]*\)\s*:\s*)?"

    # Capsule side: per-line normalized cores for both containment orders.
    cap_full = " ".join(state_capsule.split()).lower()
    cap_prefixes: list[str] = []
    for cap_line in state_capsule.splitlines():
        cap_core = re.sub(frame_re, "", _norm(cap_line)).strip()
        if cap_core.endswith("..."):
            cap_core = cap_core[:-3].strip()
        if len(cap_core) >= 32:
            cap_prefixes.append(cap_core)

    kept: list[str] = []
    for line in payload.splitlines():
        norm = _norm(line)
        core = re.sub(frame_re, "", norm).strip()
        if len(core) >= 24 and core in cap_full:
            continue
        if len(core) >= 32 and any(core.startswith(p) for p in cap_prefixes):
            continue
        kept.append(line)
    return "\n".join(kept)


def _filter_goal_echo_entries(summary: str, goal_texts: list[str]) -> str:
    """Drop memory-snapshot entries that restate an already-rendered goal.

    The LAYER 3 active-goal block renders ``session_goal`` / ``loop_goal``
    verbatim every prompt (goal-persistence policy keeps them pinned in L3),
    so working-memory entries that only restate the same sentence are pure
    duplication in long sessions. Entry-level (not line-level) matching is
    robust to multi-line entry content. Two drop rules, both requiring the
    goal text to be present inside the entry:

    * dominance: the whitespace-normalized goal makes up >=50% of the
      entry (floor 24 chars so short goals never nuke whole entries);
    * persistence framing: entries written by the goal-persistence hatch
      ("Locked session goal:" / "Locked loop goal:" prefix) carry no other
      substance.

    Returns the summary unchanged when no goals are set or nothing matches.
    """
    goal_norms = [" ".join(str(g or "").split()) for g in goal_texts]
    goal_norms = [g for g in goal_norms if g]
    if not goal_norms or not summary:
        return summary
    kept: list[str] = []
    # Snapshot entries render one "- #id ..." block each; entry-level
    # matching survives multi-line entry content.
    for entry in re.split(r"(?=\n- #)", summary):
        norm = " ".join(entry.split())
        low = norm.lower()
        dominated = any(
            g in norm and len(g) >= max(24, 0.5 * len(norm))
            for g in goal_norms
        )
        framed = ("locked session goal:" in low or "locked loop goal:" in low) and any(
            g in norm for g in goal_norms
        )
        if dominated or framed:
            continue
        kept.append(entry)
    return "\n".join(kept)


def run_turn(session, text, *, origin="user"):
    logger.info(f"Sending message: {text[:100]}...")
    session.paused_execution_text = None
    session._loop_blocker_raised = False  # fresh turn — last turn's pause doesn't apply
    session._hook_abort_requested = False
    session._hook_abort_reason = None
    # Fix #13: reset the consolidation guard each turn so a max-iterations
    # consolidation can fire on subsequent turns too.
    session._consolidation_done = False
    # Fix #12: reset per-turn re-coverage stall tracking.
    session._recoverage_seen_paths = set()
    session._recoverage_stall_iters = 0
    session._recoverage_last_nudge_iter = -10_000
    # Reset the reactive-overflow-recovery counter so each turn gets its own
    # compact-and-retry budget (circuit breaker is per-turn, capped at
    # _MAX_OVERFLOW_RECOVERIES_PER_TURN recoveries).
    session._overflow_recoveries_this_turn = 0
    session.sync_runtime_state()
    # Round-51 T7: trim oversized restored history BEFORE the first request.
    # The traced 517-iter run resumed with L5 at 300k (another at 795k,
    # above the 580k configured limit) because restore hydrated saved
    # history unbounded — a resumed session's first request could already
    # exceed the provider window. Fold overflow into the rolling summary
    # via the existing compaction path instead of silently truncating; log
    # loudly when the trim fires.
    try:
        from mu.session.budgets import (
            drift_corrected_context_limit,
            resolve_keep_recent,
            resolve_response_reserve,
        )
        _sm = session.session_manager
        _pre_len = len(getattr(_sm, "history", []) or [])
        if _pre_len:
            _tokens = _sm.estimate_runtime_history_tokens()
            _limit = drift_corrected_context_limit(session)
            _reserve = resolve_response_reserve(session)
            _budget = max(1024, _limit - _reserve)
            if _tokens > _budget:
                _sm._pending_compaction_kind = "restore_trim"
                _sm._pending_compaction_iter = 0
                _sm.roll_history_summary_to_token_budget(
                    _budget,
                    keep_recent=max(4, int(resolve_keep_recent(session))),
                    provider=session.provider,
                )
                _post = _sm.estimate_runtime_history_tokens()
                logger.warning(
                    "Restore trim: restored history estimated %d tokens "
                    "(limit %d, reserve %d). Compacted to %d tokens, "
                    "%d -> %d messages.",
                    _tokens, _limit, _reserve, _post, _pre_len,
                    len(getattr(_sm, "history", []) or []),
                )
    except Exception:
        logger.debug("restore trim skipped", exc_info=True)
    # Staleness decay (self-management): advance the memory turn counter
    # once per turn, then demote ACTIVE task-memory entries not hit in the
    # last `memory_stale_after_turns` turns to STALE. This keeps the active
    # set honest — "active" means recently mattered, not ever saved — so
    # search_memory (active-only by default) and L3 injection stay
    # high-signal instead of accumulating noise. Reversible: a search hit or
    # re-save promotes a STALE entry back to ACTIVE. 0 disables. Decay only
    # applies to task_memory (the durable store); the ephemeral scratchpad
    # is wiped per turn and never decays. See context_status for the
    # stale_memory_count signal that tells the agent what to retire.
    try:
        stale_after = int(session.variables.get("memory_stale_after_turns", 12) or 0)
    except (TypeError, ValueError):
        stale_after = 12
    if stale_after > 0 and hasattr(session, "task_memory") and session.task_memory is not None:
        session.task_memory.advance_turn()
        _demoted = session.task_memory.apply_staleness_decay(stale_after)
        if _demoted:
            logger.info(
                "memory staleness decay: %d active entries demoted to stale", _demoted
            )
    # Compute the active agent mode once — reused by scratchpad handling
    # below and by the mode-specific prompt builders that follow.
    active_mode = str(session.variables.get("agent_mode", "default")).lower()
    active_tool_phases = resolve_active_tool_phases(
        session.variables,
        getattr(session, "_loaded_tool_phases", None),
    )
    # Runtime receipt for diagnostics/UI integrations. This is derived state,
    # not persisted configuration: mode-required registries remain active even
    # when a saved session has only the historical ["core"] default.
    session._active_tool_phases = tuple(active_tool_phases)
    if session.variables.get("scratchpad_enabled", True):
        session.turn_scratchpad.max_entries = max(
            1,
            int(
                session.variables.get(
                    "scratchpad_max_entries", session.turn_scratchpad.max_entries
                )
            ),
        )
        # Scratchpad persistence (R12, FM-12): in loop/feature modes the
        # scratchpad defaults to persisting across turns so cross-turn
        # plans survive. In default/teacher modes it is cleared at turn
        # start unless the user explicitly set the persist flag. An
        # explicit True always persists; loop/feature always persists
        # (mode wins). Default mode with entries present emits a clear
        # notice so the human knows the plan was wiped.
        _explicit_persist = bool(
            session.variables.get("scratchpad_persist_across_turns", False)
        )
        _should_persist = _explicit_persist or active_mode in ("loop", "feature")
        if not _should_persist:
            _had_entries = len(session.turn_scratchpad.entries) > 0
            # Carve out the `todo`-tagged ledger — it is the agent's persistent
            # self-managed task plan and must survive the turn-start wipe so
            # the agent can reconcile/prune it across turns (the Claude-Code
            # "clean up the stale task list" move). Only ephemeral notes are
            # cleared; todos persist for the session.
            _removed = session.turn_scratchpad.clear_excluding({"todo"})
            if _had_entries:
                _notice = (
                    "Turn scratchpad auto-cleared at turn start "
                    "(scratchpad_persist_across_turns is off in default mode); "
                    "todo ledger retained."
                )
                logger.info(_notice)
                if session.ui and _removed:
                    session.ui.show_info(_notice)

    # Dual-registry contract (documented, not a bug): staged_files carries
    # THIS-turn provider-native payloads (image_input bytes, file_refs —
    # cleared after the turn), while staged_attachments carries durable
    # descriptors that rehydrate as a text notice pointing the model at
    # read_attachment/download_attachment tools. A file added via add_file
    # intentionally lands in both: the model sees the content now AND
    # retains a durable handle after compaction.
    parts = list(session.staged_files) + list(getattr(session, "staged_attachments", []) or [])
    effective_text = text
    if text and active_mode == "feature":
        effective_text = session._build_feature_mode_prompt(text)
    elif text and active_mode == "loop":
        effective_text = session._build_loop_mode_prompt(text)
    if active_mode == "loop":
        session._ensure_loop_goal_persistence()
    # Auto-pin session_goal if unset and text is a substantial user
    # message (not a slash command, not empty, >10 chars).
    # This ensures L3 always has a goal anchor from the first meaningful
    # user message without requiring /goal to be set manually.
    # Pin BEFORE persistence so the first meaningful request gets its
    # task-memory audit entry (codex review finding #6).
    if origin == "user" and not str(session.variables.get("session_goal", "") or "").strip():
        raw_text = str(text or "").strip()
        if raw_text and len(raw_text) > 10 and not raw_text.startswith("/"):
            session.variables["session_goal"] = raw_text
    # Mode-agnostic: every turn, mirror the pinned session_goal into
    # task_memory so it survives history compaction.
    if origin == "user":
        session._ensure_session_goal_persistence()
    if effective_text:
        parts.append({"type": "text", "text": effective_text})

    new_user_message = {
        "role": "user",
        "parts": parts,
        "timeline_id": "turn-" + uuid.uuid4().hex,
        "origin": origin,
        "synthetic": origin != "user",
    }
    if getattr(session, "_thread_turn_id", None):
        new_user_message["timeline_id"] = session._thread_turn_id

    # Teacher watcher: classify the user's message as a learner reply
    # against the active lesson's most recent check, BEFORE the agent
    # runs. The watcher records learner_response/learner_question turns
    # into the engine so the transcript reflects what the user actually
    # said — not what the agent later narrates about it.
    if origin == "user" and active_mode == "teacher" and text:
        _run_teacher_watcher_user(session, text)

    if origin == "user" and text and session.ui and session.variables.get("verbose", False):
        session.ui.render_message("user", text)

    workspace_context = ""
    session_type = normalize_session_type(
        session.variables.get("session_type", "workspace")
    )
    expose_tools = bool(
        session.agentic
        and (
            session.folder_context.folders
            or tools_enabled_without_workspace(session_type)
        )
    )

    if session.folder_context.folders or tools_enabled_without_workspace(session_type):
        # L4B auto-retrieval removed — model uses retrieve_relevant_context
        # tool on demand instead of pre-injected snippets.
        if session.agentic:
            active_tools = [t for t in TOOLS if t.name not in session.disabled_tools]
            active_tools = filter_tools_for_session_type(active_tools, session_type)
            # Spec #9: phased exposure — when lazy_tools_enabled, exclude
            # specialist-phase tools not in the active phase set (+ any the
            # model loaded via load_tools). Default off → no filtering.
            if session.variables.get("lazy_tools_enabled", False):
                active_tools = filter_tools_by_phase(active_tools, active_tool_phases)

            agent_mode = str(session.variables.get("agent_mode", "default")).lower()
            # Prompt resolution priority (highest first):
            #   1. runtime session-variable override (set via /set)
            #   2. file override under $MUCLI_HOME/prompts/ (PromptLibrary)
            #   3. hardcoded fallback in utils/config.py
            # The `or` falls through empty-string overrides to the library,
            # which itself falls through file→hardcoded.
            from mu.prompts import get_base as _get_base_prompt
            from mu.prompts import get_mode as _get_mode_prompt

            default_mode_instruction = _get_mode_prompt(agent_mode)
            mode_instruction = str(
                session.variables.get(
                    f"agentic_mode_prompt_{agent_mode}",
                )
                or default_mode_instruction
            )
            agentic_system_base = str(
                session.variables.get("agentic_system_base_override")
                or _get_base_prompt()
            )

            type_instruction = SESSION_TYPE_PROMPTS.get(session_type, "")
            # Providers generate the machine-readable tool declaration. The
            # human-readable capability boundary remains in L0 so the model
            # understands where it is executing and why tools differ.
            workspace_context = (
                f"{agentic_system_base}\n\n"
                f"### SESSION TYPE: {session_type.upper()}\n{type_instruction}\n\n"
                f"### CURRENT STRATEGY MODE: {agent_mode.upper()}\n{mode_instruction}\n\n"
                "### ACTIVE TOOL REGISTRIES\n"
                + (
                    f"{', '.join(active_tool_phases)}. "
                    f"Provider schema exposes {len(active_tools)} tools; "
                    "strategy-mode registries are activated automatically."
                    if session.variables.get("lazy_tools_enabled", False)
                    else (
                        f"all registered phases. Provider schema exposes "
                        f"{len(active_tools)} tools because lazy tool exposure is disabled."
                    )
                )
            )
        else:
            logger.debug(
                f"Using agent_mode={session.variables.get('agent_mode', 'default')}"
            )
            # Workspace file tree (L1C) removed — agent retrieves files on
            # demand via list_dir/read_file/search_for_string instead.

    base_system_prompt = session.system_instruction
    if active_mode == "feature":
        base_system_prompt += (
            "\n\nFEATURE MODE SYSTEM PROMPT\n"
            "You are in Feature Plan Engine mode. "
            "Use the staged feature-task engine for this request. Start with create_feature, then create_phases, then create_task for each ticket. "
            "Do not create alternate planning documents and do not begin code implementation until the user has reviewed and approved the plan. "
            "Every task must include explicit EXIT CRITERIA and tasks can be marked completed only after all exit criteria are verified. "
            "Continuously update verified_exit_criteria via update_task_status as each criterion is met so progress remains explicit. "
            "Step through one task at a time until completion; never work multiple tasks simultaneously. "
            "Use get_execution_state to choose the next actionable phase/task, use block_task if external input is required, and resume_task when user unblock context arrives. "
            "Use review_all_completed_tasks/review_completed_tasks/propose_task_diff/decide_task_diff/archive_task for review-and-archive flow after implementation completes. "
            "gather read-only context first, use save_scratchpad for temporary phase notes, call flush only after a tool result reported \"Stored ... in collation buffer\", and call raise_blocker when blocked on user input. In subagent sessions read-only results always deliver inline — never call flush there. "
            "You must use save_memory for durable facts/decisions and reuse search_memory/list_memory before re-deriving context in long loops. "
            "You must use save_scratchpad/list_scratchpad within each turn to track in-flight plans as context grows. "
            "Do not stall on status-only updates: unless blocked or awaiting explicit approval/decision, continue implementation autonomously until all phases and tasks are completed."
        )
    elif active_mode == "loop":
        loop_goal = str(session.variables.get("loop_goal", "") or "").strip()
        base_system_prompt += (
            "\n\nLOOP MODE SYSTEM PROMPT\n"
            "You are executing a long-horizon autonomous loop. "
            "Work continuously in increments (plan -> execute -> verify -> continue) until stopped by the user. "
            "Maintain a visible task list via `todo_write` and `todo_set_status` so the user can see your plan at any time. "
            "Exactly one todo should be in_progress at a time. "
            "At each increment, provide a concise timeline update: attempted action, outcome, evidence, and next step. "
            "Use save_memory for durable findings and save_scratchpad for short-lived planning. "
            "For focused side-quests that would clutter loop context (deep research, isolated refactors), delegate via `spawn_agent` with a tight tools whitelist."
        )
        if loop_goal:
            base_system_prompt += f"\nLocked loop goal: {loop_goal}"
    elif active_mode == "teacher":
        profile_block = _render_learner_profile_block(session)
        if profile_block:
            base_system_prompt += profile_block
    if workspace_context:
        base_system_prompt += f"\n\n{workspace_context}"
    from mu.session.budgets import (
        resolve_keep_recent,
        resolve_tool_result_floor,
        resolve_tool_cache_bounds,
    )

    session.session_manager._tool_result_floor = resolve_tool_result_floor(session)
    # Fix #10: grow the tool-result sidecar cache for long-horizon modes so
    # more on-disk reads stay recallable / auto-recallable by locator instead
    # of being evicted under the small default cap. Only raises the bounds.
    try:
        _tc_entries, _tc_bytes = resolve_tool_cache_bounds(session)
        session.tool_result_cache.max_entries = _tc_entries
        session.tool_result_cache.max_bytes = _tc_bytes
    except Exception:
        # Defensive: best-effort path must not break the caller.
        logger.debug("Suppressed exception", exc_info=True)
    # Reset per-turn efficiency accumulators (spec #12) so each turn's
    # metrics reflect that turn only.
    try:
        from mu.session.efficiency_metrics import reset_per_turn_accumulators

        reset_per_turn_accumulators(session)
    except Exception:  # noqa: BLE001
        pass
    # Attach the durable result store (spec #1/#11): full raw tool results
    # are written through to disk so they survive LRU eviction and session
    # restarts, and are retrievable via recall()/result_* ops. One store per
    # run, keyed by the trace run_id so results co-locate with their trace.
    try:
        from mu.session.result_store import ResultStore
        from mu.trace.emitter import get_emitter, new_run_id

        if (
            session.variables.get("result_store_enabled", True)
            and getattr(session, "result_store", None) is None
        ):
            _run_id = new_run_id()
            try:
                _tr = get_emitter(session)
                if _tr is not None and getattr(_tr, "run_id", None):
                    _run_id = _tr.run_id
            except Exception:  # noqa: BLE001
                pass
            session.result_store = ResultStore(
                _run_id,
                max_bytes=int(
                    session.variables.get("result_store_max_bytes", 16 * 1024 * 1024)
                    or 16 * 1024 * 1024
                ),
                gc_age_days=int(
                    session.variables.get("result_store_gc_age_days", 7) or 7
                ),
            )
            session.tool_result_cache.set_store(session.result_store)
            # Sync content-hash freshness flag from config (spec #7/#8).
            session.tool_result_cache.content_hash_enabled = bool(
                session.variables.get("cache_content_hash_enabled", True)
            )
            # Best-effort GC of stale run dirs once per session.
            try:
                session.result_store.gc()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    # Run tracer: emit the run_start header once, and tag this turn-start
    # compaction so the trace can record it (drained at the post-response seam).
    try:
        from mu.trace.emitter import get_emitter

        _tr = get_emitter(session)
        if _tr is not None and not getattr(session, "_trace_started", False):
            session._trace_started = True
            # Round-51 T1: emit BOTH the user-facing configured limit and the
            # drift-corrected effective limit the preflight guard actually
            # enforces. Divergence between the two was previously invisible
            # (displayed 580k while the guard resolved a different ceiling),
            # so runs could overflow the displayed limit with zero
            # compaction events. Warn when they diverge >5%.
            from mu.session.budgets import drift_corrected_context_limit

            _configured_limit = int(
                getattr(session, "_resolved_context_limit", 0)
                or session.variables.get("context_token_limit", 0)
                or 0
            )
            try:
                _effective_limit = int(drift_corrected_context_limit(session))
            except Exception:
                _effective_limit = 0
            # Round-51 T1: limit_source classifies which resolution path
            # produced the effective limit so traces are self-describing.
            # Precedence: a valid provider window below the configured user
            # limit means the window drives the effective limit
            # (provider_window); otherwise a difference between the
            # configured and effective limits comes from drift/safety
            # correction; when they agree the user limit stands.
            _prov_window = 0
            try:
                _prov_raw = session.provider.effective_context_window(
                    session.provider.model_name
                )
                if _prov_raw and int(_prov_raw) > 0:
                    _prov_window = int(_prov_raw)
            except Exception:
                _prov_window = 0
            if _effective_limit <= 0:
                _limit_source = "user"
            elif (
                _prov_window > 0
                and _configured_limit > 0
                and _configured_limit > _prov_window
                and _effective_limit <= _configured_limit
            ):
                _limit_source = "provider_window"
            elif (
                _configured_limit > 0
                and _effective_limit != _configured_limit
            ):
                _limit_source = "drift_corrected"
            elif (
                _configured_limit <= 0
                and _prov_window > 0
                and _effective_limit != _prov_window
            ):
                _limit_source = "provider_window"
            else:
                _limit_source = "user"
            session._effective_guard_limit = _effective_limit
            _tr.run_start(
                {
                    "session": _tr.session_name,
                    "model": getattr(session.provider, "model_name", ""),
                    "provider": type(session.provider).__name__,
                    "mode": active_mode,
                    "context_limit": _configured_limit,
                    "effective_limit": _effective_limit,
                    "limit_source": _limit_source,
                    "max_iterations": int(
                        session.variables.get("max_iterations", 50) or 50
                    ),
                }
            )
    except Exception:
        # Defensive: best-effort path must not break the caller.
        logger.debug("Suppressed exception", exc_info=True)
    try:
        session.session_manager._pending_compaction_kind = "turn_start"
        session.session_manager._pending_compaction_iter = 0
    except Exception:
        # Defensive: best-effort path must not break the caller.
        logger.debug("Suppressed exception", exc_info=True)
    # Bridge the optional compact_focus variable (Claude Code `/compact
    # <focus>` style) so the LLM summarizer emphasizes it when set.
    session.session_manager._compact_focus = (
        session.variables.get("compact_focus") or ""
    )
    # Normal history cleanup is model-directed via the `compact` tool. Keep
    # proactive automatic compaction as an explicit deployment opt-in; the
    # preflight/overflow paths below still enforce the provider's hard ceiling.
    _turn_start_rolled = False
    if session.variables.get("auto_compaction_enabled", False):
        _turn_start_rolled = session.session_manager.roll_history_summary_to_token_budget(
            session._compaction_token_budget(),
            keep_recent=resolve_keep_recent(session),
            provider=session.provider,
        )
    # This is the turn's proactive compaction pass. If it actually compacted,
    # mark the turn so the per-iteration auto-compaction hook does not fire
    # again this turn (Claude Code fires autocompact once per turn; mid-turn
    # overshoot is handled by the emergency preflight + reactive overflow
    # backstops). Reset in `_collect_turn_response` at turn end.
    session._compacted_this_turn = bool(_turn_start_rolled)
    # Record history length as the compaction watermark.  The
    # auto-compaction hook will allow another compaction pass only when
    # history has grown beyond this point, preventing redundant
    # re-compaction while still permitting re-compaction when long
    # turns with many tool calls push history past the threshold.
    # Reset to 0 in `_collect_turn_response` when the turn finishes.
    session._compaction_watermark = len(session.session_manager.history)
    session._pending_user_text = effective_text or text or ""
    # Cross-session recall runs once per user turn, not once per provider
    # iteration. The resulting block and receipt are reused by retries and
    # tool-call iterations so browsing/looping cannot inflate recall counts.
    session._turn_durable_recall_block = ""
    session._last_durable_recall_receipt = None
    session._turn_durable_writes = []
    if (
        effective_text
        and session.variables.get("durable_memory_enabled", True)
        and not str(text or "").lstrip().startswith("/")
    ):
        try:
            _memory_service = session.get_durable_memory_service()
            _memory_receipt = _memory_service.recall(
                session,
                effective_text,
                limit=max(
                    1,
                    int(session.variables.get("durable_memory_max_items", 6) or 6),
                ),
                budget_tokens=max(
                    64,
                    int(
                        session.variables.get("durable_memory_token_budget", 1200)
                        or 1200
                    ),
                ),
            )
            session._last_durable_recall_receipt = _memory_receipt
            session._turn_durable_recall_block = _memory_service.render_recall(
                _memory_receipt
            )
            if (
                _memory_receipt.included
                and session.ui
                and session.variables.get("durable_memory_show_receipts", True)
            ):
                session.ui.show_info(
                    f"Memory · recalled {len(_memory_receipt.included)} · "
                    f"{_memory_receipt.token_count} tokens · "
                    f"receipt {_memory_receipt.id.split('-')[0]}"
                )
        except Exception:
            logger.debug("durable memory recall failed", exc_info=True)
    # Resumption briefings: queued by /teach load, /feature load, and
    # session-switch paths. Drained here so the agent's next provider
    # call sees them once, then the queue clears.
    resumption_block = session._drain_resumption_briefings()
    if resumption_block:
        base_system_prompt = f"{base_system_prompt}\n\n{resumption_block}"
    # Cache the disk-backed L1 (workspace files) and L1B (skills) layers
    # once per turn. They're expensive to rebuild (read files / walk the
    # skills tree) and stable within a turn, so reusing the cached text
    # every iteration lets us rebuild L2 (conversation summary) and L3
    # (active goal) fresh each iteration without the disk cost — closing
    # the frozen-at-turn-start gap that starved the model of mid-turn
    # progress updates. See _inject_hierarchical_context(cached_*).
    session._turn_skills_block = session._build_skills_block(announce=True)
    session._turn_context_files_block = session._build_context_files_block()
    # The pre-injection persona text (system_instruction + mode prompt +
    # workspace_context + resumption block). Reused as the base for every
    # per-iteration system-prompt rebuild.
    base_persona_prompt = base_system_prompt
    base_system_prompt = session._inject_hierarchical_context(
        base_persona_prompt,
        cached_skills=session._turn_skills_block,
    )
    # Publish the pre-injection persona base so the emergency re-inject in
    # the context guard can rebuild the layered hierarchy from it instead
    # of stacking refreshed layers on the already-injected prompt.
    session._system_prompt_base = base_persona_prompt

    recent_history = session._prepare_runtime_history()
    messages = session._build_messages_from_history(recent_history, new_user_message)

    initial_history_len = len(session.session_manager.history)
    session.session_manager.history.append(new_user_message)
    turn_start_index = len(session.session_manager.history) - 1
    # True once any tool call in this turn has reached dispatch. Used by the
    # 4xx rollback path: after a tool side effect exists, history rollback
    # would orphan irreversibly-executed actions and invite duplication.
    turn_had_tool_execution = False
    # Store on session so send_message's finally block can call _cleanup_protected
    # after the turn ends (unprotect the turn prompt if it's not otherwise worthy).
    session._current_turn_start_index = turn_start_index
    # Mirror onto the session manager so the compaction paths can compute
    # the per-turn tool-result floor (R3, FM-8).
    session.session_manager._active_turn_start_index = turn_start_index
    # Reset the per-turn oversized-message summary cache (R4, FM-2) so a
    # new turn doesn't reuse a prior turn's chunk summaries.
    session._oversized_message_summaries = {}
    # Mark important user messages as protected from compaction.
    # The turn's starting prompt is always protected during this turn.
    session.session_manager._maybe_protect(turn_start_index, "user", effective_text, is_turn_prompt=True)
    session.session_manager.save_history_turn()
    session.staged_files = []
    session.staged_attachments = []

    # Clamp persisted values: a corrupted/non-integer max_iterations must
    # not raise TypeError at the loop condition on every turn.
    try:
        max_iterations = max(1, int(session.variables.get("max_iterations", 50)))
    except (TypeError, ValueError):
        max_iterations = 50
    iteration = 0
    active_tools = [t for t in TOOLS if t.name not in session.disabled_tools]
    active_tools = filter_tools_for_session_type(active_tools, session_type)
    provider_tools = active_tools if expose_tools else None
    # Spec #9: phased exposure (see the earlier filter site for details).
    if session.variables.get("lazy_tools_enabled", False):
        active_tools = filter_tools_by_phase(active_tools, active_tool_phases)
    provider_tools = active_tools if expose_tools else None

    total_in = 0
    total_out = 0
    total_cost = 0.0

    logger.info(f"Starting agentic loop (max_iterations={max_iterations})")
    provider_bad_request_retried = False
    # Round-46 F8: loop-detection fingerprint histories are bounded deques —
    # the detectors only ever inspect the last repeat_threshold entries
    # (consecutive) or max_period*min_repeats entries (periodic), so an
    # unbounded turn-long list retained 100k fingerprints for no benefit.
    # maxlen covers both detectors' maximum lookback. Computed here because
    # the threshold is defined further down (variable read must follow it).
    loop_detection_repeat_threshold = max(
        2,
        int(session.variables.get("loop_detection_repeat_threshold", 5) or 5),
    )
    _loop_hist_maxlen = loop_detection_repeat_threshold + 12
    exact_tool_sequence_history: deque = deque(maxlen=_loop_hist_maxlen)
    pattern_tool_sequence_history: deque = deque(maxlen=_loop_hist_maxlen)
    # Round-46 F1: history-checkpoint throttling. save_history_turn serializes
    # and atomically rewrites the COMPLETE session document; doing that after
    # EVERY tool iteration makes a 100k-iteration turn Θ(n²) in cumulative
    # JSON+disk. The per-iteration tool-result save is now a bounded
    # checkpoint: at most one full save per 2s wall clock OR every 10
    # iterations (whichever first), with turn end / compactions / loop nudges
    # still saving unconditionally. Staleness bound: the on-disk session can
    # lag the live history by up to 2s / 10 iterations — the GUI live
    # transcript refreshes from the next checkpoint (SSE-driven reloads are
    # idempotent), and a crash loses at most that window of tool results.
    _save_checkpoint_min_interval = 2.0
    _save_checkpoint_every_iters = 10
    _last_history_checkpoint = 0.0
    _iters_since_checkpoint = 0
    # Dedicated lane for retryable-failure storms (R8, FM-6). Each
    # iteration that hits a retryable failure appends a synthetic
    # `retryable~<error_code>` fingerprint here; `is_repeated_tool_sequence`
    # over this list catches storms that span iterations with different
    # tool args (which evade both the per-iteration escalation threshold
    # and the normal pattern lane, since non-search tools like read_file
    # produce per-arg-distinct fingerprints).
    retryable_tool_sequence_history: deque = deque(maxlen=_loop_hist_maxlen)
    loop_detection_enabled = bool(
        session.variables.get("loop_detection_enabled", True)
    )
    loop_detection_repeat_threshold = max(
        2,
        int(session.variables.get("loop_detection_repeat_threshold", 5) or 5),
    )
    # Fix #12: context-gathering stall detection. Tracks paths read across
    # the whole turn and counts consecutive iterations that re-cover
    # already-read paths without making a concrete change — the diffuse
    # "going around getting context, progress halts" stall that doesn't
    # form a clean repeated/periodic tool sequence. When the count hits the
    # threshold, inject a re-orient nudge telling the agent to stop
    # gathering and act. State lives on the session so it survives across
    # the turn's iterations.
    from mu.agent.loop_detection import (
        extract_read_paths,
        is_concrete_change_iter,
    )

    recoverage_threshold = max(
        0, int(session.variables.get("recoverage_stall_threshold", 4) or 0)
    )
    if getattr(session, "_recoverage_seen_paths", None) is None:
        session._recoverage_seen_paths = set()
    if getattr(session, "_recoverage_stall_iters", None) is None:
        session._recoverage_stall_iters = 0
    if getattr(session, "_recoverage_last_nudge_iter", None) is None:
        session._recoverage_last_nudge_iter = -10_000

    while iteration < max_iterations:
        iteration += 1
        session._trace_current_iter = iteration  # run tracer: compaction-kind iter
        _trace_iter_start = time.monotonic()  # run tracer: per-iter wall clock
        logger.debug(f"Agentic loop iteration {iteration}/{max_iterations}")
        # Periodic L2 progress checkpoint (Fix #9). On long turns that never
        # hit the compaction budget, conversation_summary stays frozen while
        # the model racks up real progress in L5 — it then keeps re-reading
        # files it already explored (the long-horizon stall). Every N
        # iterations, fold recent history into the structured L2 summary
        # WITHOUT compacting (anchor doesn't advance) so the next iteration's
        # system prompt reflects current Progress/State/Open-items. Cadence:
        # `progress_checkpoint_every` (0 disables); when unset, loop/feature
        # modes default to 12, default/chat to 0. Fires before the per-
        # iteration system-prompt rebuild so the refreshed L2 is used this
        # iteration.
        try:
            _ckpt_every = int(
                session.variables.get("progress_checkpoint_every", 0) or 0
            )
            if _ckpt_every <= 0:
                # Mode-aware default: only long-horizon modes auto-enable.
                _ckpt_mode = str(
                    session.variables.get("agent_mode", "default") or "default"
                ).lower()
                if _ckpt_mode in ("loop", "feature"):
                    _ckpt_every = 12
            if (
                _ckpt_every > 0
                and iteration > 1
                and (iteration % _ckpt_every) == 0
            ):
                _ckpt_ok = session.session_manager.force_progress_checkpoint(
                    session.provider
                )
                if _ckpt_ok:
                    logger.debug(
                        f"L2 progress checkpoint refreshed at iteration "
                        f"{iteration}/{max_iterations}"
                    )
        except Exception as _ckpt_exc:
            logger.debug(f"progress checkpoint skipped: {_ckpt_exc}")
        # Subagent wrap-up reminder. Two strategies:
        #  * Adaptive (preferred): when a SubagentLifecycleManager is attached
        #    (an async-orchestrated child), drive consolidation from its
        #    signals — inject "WRAP UP NOW" once the child stalls (no novel
        #    output for `subagent_stall_threshold` calls), with a hard
        #    iteration-based safety net (last 2 iters) so a child that never
        #    stalls still consolidates before the cap.
        #  * Legacy fallback: no lifecycle manager -> use the hardcoded
        #    `_subagent_wrap_up_iter` (max_iterations - 3) cutoff.
        # Injected once per run via `_subagent_wrap_up_injected`.
        if not getattr(session, "_subagent_wrap_up_injected", False):
            _lc = getattr(session, "_subagent_lifecycle", None)
            _should_wrap = False
            if _lc is not None:
                try:
                    _lc_snap = _lc.snapshot()
                except Exception:
                    _lc_snap = {}
                if bool(_lc_snap.get("stall", False)):
                    _should_wrap = True
                    logger.info(
                        f"Subagent wrap-up: stalled signal at iteration "
                        f"{iteration}/{max_iterations}"
                    )
                elif iteration >= max(1, max_iterations - 2):
                    _should_wrap = True
            else:
                wrap_up_iter = getattr(session, "_subagent_wrap_up_iter", None)
                if wrap_up_iter and iteration >= wrap_up_iter:
                    _should_wrap = True
            if _should_wrap:
                session._subagent_wrap_up_injected = True
                session.session_manager.history.append({
                    "role": "system",
                    "parts": [{"type": "text", "text": (
                        "WRAP UP NOW: You are at iteration "
                        f"{iteration}/{max_iterations}. Stop calling tools "
                        "immediately. Consolidate all findings into a single "
                        "final summary message. Do not call any more tools."
                    )}],
                })
                logger.info(
                    f"Subagent wrap-up reminder injected at iteration {iteration}/{max_iterations}"
                )
        # Honor a hook abort raised in the previous iteration. The
        # in-flight provider call / tool dispatch from that iteration
        # has already finished and stored its results in history; we
        # exit cleanly here with status="hook_aborted".
        if session._hook_abort_requested:
            logger.info(
                f"Agentic loop exiting on hook abort: {session._hook_abort_reason}"
            )
            if session.session_manager.get_feature_state():
                session._set_feature_state(status="hook_aborted")
            return session._collect_turn_response(
                initial_history_len,
                status="hook_aborted",
                total_in=total_in,
                total_out=total_out,
                total_cost=total_cost,
                error=(
                    session._hook_abort_reason
                    or "A lifecycle hook requested abort."
                ),
            )
        # Honor a cooperative sub-agent kill (parent called kill_subagent,
        # or the runtime watchdog fired). Same shape as the hook-abort path:
        # the previous iteration's work is already in history, so exit
        # cleanly here with status="killed". The registry's thread wrapper
        # captures partial findings from history after this returns.
        if getattr(session, "_subagent_cancelled", False):
            _kill_reason = getattr(session, "_subagent_kill_reason", None) or "killed"
            logger.info(f"Agentic loop exiting on sub-agent kill: {_kill_reason}")
            return session._collect_turn_response(
                initial_history_len,
                status="killed",
                total_in=total_in,
                total_out=total_out,
                total_cost=total_cost,
                error=_kill_reason,
            )
        current_tool_name = None
        current_tool_args = None
        iteration_tool_exact_fingerprints: list[str] = []
        iteration_tool_pattern: list[str] = []
        _max_retryable_count_this_iter = 0
        # Distinct retryable error_codes seen this iteration, collected so
        # the cross-reference block below can append synthetic
        # `retryable~<code>` fingerprints to the retryable loop-detection
        # lane (R8). A set is enough — one marker per distinct code per
        # iteration keeps the lane sensitive to storms without inflating
        # a single transient blip into a false positive.
        _iteration_retryable_codes: set[str] = set()

        try:
            # Rebuild the layered system prompt fresh each iteration so L2
            # (conversation summary) and L3 (active goal / feature_state /
            # scratchpad) reflect mid-turn updates — auto-compaction can
            # rewrite the summary via the pre_provider_call hook, and tools
            # can update feature_state / the scratchpad between iterations.
            # L1B is reused from the per-turn cache (no disk reads).
            # The memory + scratchpad snapshots are appended below as before.
            # L2 state capsule is the authoritative structured projection;
            # the L3 snapshots below are deduped against it so stores don't
            # render twice per prompt.
            try:
                from mu.session.state_capsule import build_state_capsule

                state_capsule_text = build_state_capsule(
                    session,
                    max_chars=int(
                        session.variables.get(
                            "conversation_summary_char_limit", 24000
                        )
                        * 0.7
                    ),
                    include_goal=False,
                )
            except Exception:
                state_capsule_text = ""
            dynamic_system_prompt = session._inject_hierarchical_context(
                base_persona_prompt,
                cached_skills=session._turn_skills_block,
            )
            _durable_recall = str(
                getattr(session, "_turn_durable_recall_block", "") or ""
            ).strip()
            if _durable_recall:
                dynamic_system_prompt += (
                    "\n\nLAYER 2M — Durable cross-session recall:\n"
                    f"{_durable_recall}"
                )
            if session.variables.get("memory_enabled", True):
                active_mode_for_mem = str(
                    session.variables.get("agent_mode", "default")
                ).lower()
                # Mode-aware memory floor (R12): loop/feature modes do more
                # multi-step work and benefit from a larger in-task memory
                # cap. Raise the effective max_entries floor in those modes
                # without lowering a user's explicit higher value.
                _mem_cap = int(
                    session.variables.get(
                        "memory_max_entries", session.task_memory.max_entries
                    )
                )
                if active_mode_for_mem in ("loop", "feature"):
                    _mem_cap = max(_mem_cap, 128)
                session.task_memory.max_entries = max(1, _mem_cap)
                memory_summary = session.task_memory.render_summary(
                    limit=int(session.variables.get("memory_summary_limit", 8)),
                    query=effective_text,
                )
                if memory_summary:
                    memory_summary = _filter_goal_echo_entries(
                        memory_summary,
                        [
                            str(session.variables.get(k, "") or "")
                            for k in ("session_goal", "loop_goal")
                        ],
                    )
                    memory_summary = _filter_state_capsule_duplicates(
                        memory_summary, state_capsule_text
                    )
                if memory_summary:
                    dynamic_system_prompt += (
                        "\n\nLAYER 3 — Persisted working memory snapshot:\n"
                        f"{memory_summary}"
                    )
                # Surface eviction notices to the model (R12, FM-11) so it
                # knows a memory it relied on is gone rather than silently
                # re-deriving it. Drained once per turn.
                _mem_evictions = session.task_memory.drain_eviction_log()
                if _mem_evictions:
                    dynamic_system_prompt += (
                        "\n\nLAYER 3 — Memory eviction notices (these entries "
                        "were removed to make room; do not assume they still "
                        "exist):\n- " + "\n- ".join(_mem_evictions)
                    )
            if session.variables.get("scratchpad_enabled", True):
                scratchpad_summary = session.turn_scratchpad.render_summary(limit=8)
                if scratchpad_summary:
                    # Same goal-echo dedup as the memory snapshot: loop-mode
                    # notes that only restate the pinned goal duplicate the
                    # L3 active-goal block.
                    scratchpad_summary = _filter_goal_echo_entries(
                        scratchpad_summary,
                        [
                            str(session.variables.get(k, "") or "")
                            for k in ("session_goal", "loop_goal")
                        ],
                    )
                    scratchpad_summary = _filter_state_capsule_duplicates(
                        scratchpad_summary, state_capsule_text
                    )
                if scratchpad_summary:
                    dynamic_system_prompt += (
                        "\n\nLAYER 3 — Turn scratchpad snapshot:\n"
                        f"{scratchpad_summary}"
                    )
                _scratch_evictions = session.turn_scratchpad.drain_eviction_log()
                if _scratch_evictions:
                    dynamic_system_prompt += (
                        "\n\nLAYER 3 — Scratchpad eviction notices:\n- "
                        + "\n- ".join(_scratch_evictions)
                    )

            # MUCLI_SUBAGENT_DURABLE_RESULTS_V1: parent context. Running delegations refresh every
            # provider iteration; terminal results inject once and enter memory.
            try:
                _subagent_context = session._subagent_registry.context_block(session)
            except Exception:
                _subagent_context = ""
            if _subagent_context:
                dynamic_system_prompt += (
                    "\n\nLAYER 3B — Authoritative delegated work:\n"
                    + _subagent_context
                )

            # L3C is rebuilt for every provider iteration. It gives the
            # agent a live peer roster, path ownership, and newly delivered
            # inter-thread messages without granting those messages system
            # authority.
            try:
                _thread_coordinator = getattr(session, "thread_coordinator", None)
                if _thread_coordinator is not None:
                    _thread_coordinator.heartbeat(
                        session.thread_meta.thread_id,
                        session._thread_runtime_id,
                    )
                    _thread_context = _thread_coordinator.context_block(
                        session.thread_meta.thread_id
                    )
                else:
                    _thread_context = ""
            except Exception:
                _thread_context = ""
                logger.debug("peer thread context refresh failed", exc_info=True)
            if _thread_context:
                dynamic_system_prompt += (
                    "\n\nLAYER 3C — Peer thread coordination:\n"
                    + _thread_context
                )

            dynamic_system_prompt, messages = _preflight_context_check(
                session, dynamic_system_prompt, messages,
                turn_start_index=turn_start_index,
                tools=(provider_tools),
            )
            # Capture the estimate at the same seam as the provider request.
            # The trace event is written after the response is archived, so
            # deriving its estimate from live history there would incorrectly
            # include the newly generated assistant message.
            # Round-46 F2: reuse the preflight estimate manifest instead of
            # re-tokenizing the full prompt + messages + tool schemas a third
            # time per provider call. Fallback covers preflight-less callers.
            _est = getattr(session, "_request_estimate_manifest", None)
            request_token_estimate = (
                _est["total"]
                if _est
                else (
                    estimate_tokens(dynamic_system_prompt)
                    + _estimate_messages_tokens(messages)
                    + _estimate_tools_tokens(provider_tools)
                )
            )
            try:
                from mu.trace.emitter import build_request_record, get_emitter
                _request_emitter = get_emitter(session)
                if _request_emitter is not None:
                    _request_emitter.request(build_request_record(
                        iteration=iteration, system_prompt=dynamic_system_prompt,
                        messages=messages,
                        tools=provider_tools,
                        token_estimate=request_token_estimate,
                        estimate_manifest=_est,
                        # Round-51 T6: keep only the first iteration full —
                        # later requests emit bounded summaries (per-message
                        # totals, no part_details). Configurable: a positive
                        # `trace_request_full_iters` keeps that many full.
                        summarize=iteration
                        > max(1, int(
                            session.variables.get("trace_request_full_iters", 1)
                            or 1
                        )),
                    ))
            except Exception:
                # Defensive: best-effort path must not break the caller.
                logger.debug("Suppressed exception", exc_info=True)

            if session.ui and hasattr(session.ui, "build_live_status"):
                status_msg = session.ui.build_live_status(
                    session,
                    session.provider.model_name,
                    iteration,
                    max_iterations,
                )
            else:
                status_msg = (
                    f"Generating ({session.provider.model_name}) it {iteration}/{max_iterations}"
                    f" | {build_live_status_line(session)}"
                )
            if session.ui:
                with session.ui.show_status(status_msg):
                    response = _generate_with_overflow_recovery(
                        session,
                        messages=messages,
                        system_prompt=dynamic_system_prompt,
                        thinking=session.thinking,
                        tools=provider_tools,
                        turn_start_index=turn_start_index,
                    )
            else:
                response = _generate_with_overflow_recovery(
                    session,
                    messages=messages,
                    system_prompt=dynamic_system_prompt,
                    thinking=session.thinking,
                    tools=active_tools if expose_tools else None,
                    turn_start_index=turn_start_index,
                )

            logger.debug(
                f"Provider response received. Tokens: In {response.input_tokens}, Out {response.output_tokens}"
            )

            ai_parts_archive = []
            has_tool_call = False
            has_text = False

            for part in response.parts:
                if part.type == "text" and part.text:
                    has_text = True
                    if session.ui:
                        session.ui.render_message(
                            "assistant", part.text, session.provider.model_name
                        )
                    logger.debug(f"Assistant text: {part.text[:200]}...")
                    ai_parts_archive.append({"type": "text", "text": part.text})

                elif (
                    part.type in {"thinking", "reasoning", "thought"}
                    and part.text
                ):
                    # Live thinking is already streamed to the UI. Keep a bounded
                    # durable copy in normal history so reload can restore it as
                    # a collapsed trace instead of silently dropping it.
                    ai_parts_archive.append(
                        {"type": "thinking", "text": str(part.text)[:8000]}
                    )

                elif part.type == "image_inline" and part.inline_data:
                    display_image_in_terminal(session.session_manager.current_session_name, part.inline_data, save=True)
                    ai_parts_archive.append(
                        {
                            "type": "text",
                            "text": "[Image Generated and Saved locally]",
                        }
                    )

                elif part.type == "tool_call":
                    has_tool_call = True
                    # Round-51 T5: the model responded to a nudge with real
                    # tool progress — reset the ineffective-nudge counter.
                    if getattr(session, "_watchdog_ineffective_count", 0):
                        session._watchdog_ineffective_count = 0
                    ai_parts_archive.append(
                        {
                            "type": "tool_call",
                            "tool_name": part.tool_name,
                            "tool_args": part.tool_args,
                            "thought_signature": part.thought_signature,
                        }
                    )
                    if session.ui and active_mode != "loop":
                        # Always emit; RichUI silences this when verbose=False.
                        # SubagentUI consumes the message to drive its
                        # per-tool progress tracker, so this MUST keep firing
                        # regardless of how the parent terminal renders it.
                        session.ui.show_info(
                            f"🔨 Running tool: {part.tool_name}({_shorten_tool_args(part.tool_args)})"
                        )
                    logger.info(
                        f"Tool call: {part.tool_name} with args {part.tool_args}"
                    )

            if ai_parts_archive:
                session.session_manager.history.append(
                    {
                        "role": "assistant",
                        "parts": ai_parts_archive,
                    }
                )

                # Teacher watcher: classify the assistant's just-emitted
                # text into engine events (explanation/check/wrap-up).
                # The hard cap (1 explanation + 1 check per message) is
                # enforced inside the classifier's contract and the
                # apply_assistant_classification function.
                if active_mode == "teacher":
                    assistant_text = "\n".join(
                        p["text"]
                        for p in ai_parts_archive
                        if p.get("type") == "text" and p.get("text")
                    ).strip()
                    if assistant_text:
                        _run_teacher_watcher_assistant(session, assistant_text)

            session.session_manager.token_counts["input"] += response.input_tokens
            session.session_manager.token_counts["output"] += response.output_tokens
            session.session_manager.token_counts["total"] += response.total_tokens
            session.session_manager.token_counts["cached"] = (
                session.session_manager.token_counts.get("cached", 0)
                + getattr(response, "cached_tokens", 0)
            )
            session.session_manager.token_counts["reasoning"] = (
                session.session_manager.token_counts.get("reasoning", 0)
                + getattr(response, "reasoning_tokens", 0)
            )

            # Cold-cache drift calibration. For Ollama, response.input_tokens
            # is the streamed prompt_eval_count — normally only the non-cached
            # delta (near-zero in a warm loop). But on a cold cache (first
            # call of a session, or after the prompt changed substantially)
            # it reflects (close to) the FULL prompt, so real/cl100k drift is
            # learnable from it. See budgets.update_observed_drift /
            # effective_drift_ratio. Factored into _calibrate_drift_from_response
            # so the warm-vs-cold gate is unit-testable.
            _calibrate_drift_from_response(session, response)

            # Model-directed context-pressure nudge (increment 11): default
            # mode has no proactive compaction, so nothing told the model WHEN
            # to run the `compact` tool. When the assembled request crosses
            # `context_pressure_nudge_pct` (default 80%) of the effective
            # limit, inject ONE synthetic compact nudge per threshold
            # crossing — hysteresis via the summary anchor. Turn-final
            # responses only (no tool-call parts — an in-flight tool batch
            # continues and gets the nudge after it settles). Zero cost
            # under the threshold: reads the manifest the preflight guard
            # already stashed this iteration. Parts-based gate is
            # scope-safe — `tool_calls` may not be bound at this seam in
            # the streaming path.
            if not any(
                getattr(p, "type", None) == "tool_call"
                for p in (getattr(response, "parts", None) or [])
            ):
                _maybe_nudge_context_pressure(
                    session,
                    limit=int(getattr(session, "_last_effective_limit", 0) or 0),
                    manifest=getattr(session, "_request_estimate_manifest", None),
                )

            total_in += response.input_tokens
            total_out += response.output_tokens

            est_cost = calculate_cost(
                session.provider.model_name,
                response.input_tokens,
                response.output_tokens,
            )
            cost_str = ""
            if est_cost is not None:
                total_cost += est_cost
                session.session_manager.token_counts["total_cost"] += est_cost
                cost_str = (
                    f"| Est. Cost: ${est_cost:.5f} (Total: ${total_cost:.5f})"
                )

            if session.ui:
                # Canonical prompt-size line. The drift-corrected cl100k
                # estimate (real_est) is the one number that's a meaningful
                # prompt size on EVERY provider: for frontier providers
                # effective_drift_ratio == 1.0 so it collapses to the cl100k
                # estimate (which agrees with response.input_tokens, the
                # full prompt they report); for Ollama, response.input_tokens
                # is the streamed prompt_eval_count — the non-cached prompt
                # DELTA, near-zero in a warm loop and NOT the prompt size.
                # So: lead with the provider's real In when it's a reliable
                # full-prompt signal (frontier), and lead with ~Real est
                # (labeling In as a cached delta) when it isn't (Ollama).
                # This keeps the TUI headline consistent with the trace's
                # `prompt_tokens_real_est` and the GUI Memory Map fill.
                try:
                    from mu.session.budgets import effective_drift_ratio as _eff_drift
                    _cl100k_est = int(getattr(session, "_last_prompt_cl100k_est", 0) or 0)
                    _drift = float(_eff_drift(session))
                    _real_est = int(_cl100k_est * _drift) if _cl100k_est else 0
                except Exception:  # noqa: BLE001
                    _drift = 1.0
                    _real_est = 0
                if _drift > 1.01:
                    _lead = (
                        f"~Real est {_real_est}" if _real_est else "~Real est n/a"
                    )
                    _in_part = f" | In (cached δ) {response.input_tokens}"
                else:
                    _lead = f"In {response.input_tokens}"
                    _in_part = ""
                session.ui.show_info(
                    f"Tokens: {_lead}{_in_part} | Out {response.output_tokens} | "
                    f"Total {response.total_tokens}{cost_str}"
                )

            # Run tracer: emit the per-iteration record at the post-response seam
            # (response.input_tokens is the real prompt size; context-layer
            # estimates are the harness's cl100k_base estimate — drift between
            # them is the headline long-horizon diagnostic). Compaction events
            # drained from session_manager._compaction_log are emitted here too.
            # cl100k_base estimate of the assembled prompt, captured from the
            # trace record when available. Used for the subagent context bar:
            # for cache-enabled providers (Ollama) response.input_tokens is only
            # the non-cached prompt delta (near-zero in a long loop), so the
            # cl100k total_est is the more representative fill signal.
            _ctx_est = 0
            try:
                from mu.trace.emitter import (
                    build_iter_record,
                    drain_compactions,
                    get_emitter,
                )

                _tr = get_emitter(session)
                if _tr is not None:
                    _comps = drain_compactions(session)
                    for _c in _comps:
                        _tr.compaction(_c)
                    _rec = build_iter_record(
                        session,
                        iteration=iteration,
                        max_iter=max_iterations,
                        response=response,
                        total_in=total_in,
                        total_out=total_out,
                        total_cost=total_cost,
                        has_text=has_text,
                        has_tool_call=has_tool_call,
                        iter_start=_trace_iter_start,
                        cost_delta=float(est_cost or 0.0),
                        compaction=_comps[-1] if _comps else None,
                        request_token_estimate=request_token_estimate,
                    )
                    _tr.iter_record(_rec)
                    # Round-51 T3: loud record when L5 collapses between
                    # consecutive iterations with no compaction record in
                    # between. The observed it200 578k→17k drop left the
                    # trace unable to explain where ~561k tokens went.
                    # Detection is best-effort telemetry: never break the
                    # loop, and skip when a compaction legitimately explains
                    # the drop (its own record carries the numbers).
                    try:
                        _now_l5 = int(
                            _rec.get("context", {}).get("l5", 0) or 0
                        )
                        _prev_l5 = int(
                            getattr(session, "_trace_prev_iter_l5", 0) or 0
                        )
                        _prev_iter = getattr(
                            session, "_trace_prev_iter_num", -1
                        )
                        try:
                            _prev_iter = int(_prev_iter)
                        except (TypeError, ValueError):
                            _prev_iter = -1
                        if (
                            _prev_iter >= 0
                            and iteration == _prev_iter + 1
                            and _prev_l5 > 50000
                            and not _comps
                            and _now_l5 < 0.5 * _prev_l5
                        ):
                            _tr.emit(
                                {
                                    "type": "context_collapse",
                                    "run_id": _tr.run_id,
                                    "iter": iteration,
                                    "from_l5": _prev_l5,
                                    "to_l5": _now_l5,
                                    "drop_pct": round(
                                        (_prev_l5 - _now_l5) / float(_prev_l5) * 100.0,
                                        1,
                                    )
                                    if _prev_l5 > 0
                                    else 0.0,
                                    "probable_cause": "unknown",
                                    "last_compaction_iter": getattr(
                                        session,
                                        "_trace_last_compaction_iter",
                                        None,
                                    ),
                                    "hint": "silently_reset",
                                }
                            )
                            logger.warning(
                                "Context collapse: L5 fell %d -> %d tokens "
                                "between iterations %d and %d with no "
                                "compaction record.",
                                _prev_l5, _now_l5, _prev_iter, iteration,
                            )
                        session._trace_prev_iter_l5 = _now_l5
                        session._trace_prev_iter_num = iteration
                    except Exception:  # noqa: BLE001 — telemetry must not break the loop
                        logger.debug("Suppressed exception", exc_info=True)
                    try:
                        _ctx_est = int(_rec.get("context", {}).get("total_est", 0) or 0)
                    except Exception:  # noqa: BLE001
                        _ctx_est = 0
            except Exception:
                # Defensive: best-effort path must not break the caller.
                logger.debug("Suppressed exception", exc_info=True)

            # Live subagent context-usage: a child session reports its
            # per-iteration context fill / iter / tokens to the parent's
            # registry so the GUI status panel can render a live context
            # bar. Single writer = the child thread; the registry reads
            # under its own lock. No-op for top-level (parent) sessions and
            # for CLI/TUI runs (no registry publish callback attached).
            try:
                _parent_reg = getattr(session, "_parent_registry", None)
                if _parent_reg is not None and str(
                    session.variables.get("session_role", "") or ""
                ).lower() == "child":
                    _tid = session.variables.get("subagent_parent_task_id")
                    if _tid:
                        from mu.session.budgets import (
                            effective_drift_ratio as _eff_drift,
                        )

                        try:
                            _climit = int(_resolve_real_context_window(session) or 0)
                        except Exception:  # noqa: BLE001
                            _climit = 0
                        _actual = int(getattr(response, "input_tokens", 0) or 0)
                        # Drift-corrected real prompt estimate. Ollama's
                        # input_tokens is the cached delta (tiny in a warm
                        # loop), so total_est * effective_drift is the
                        # representative real fill; for frontier providers
                        # effective_drift is 1.0 and this collapses to the
                        # cl100k estimate (which agrees with input_tokens).
                        try:
                            _real_est = int(_ctx_est * _eff_drift(session))
                        except Exception:  # noqa: BLE001
                            _real_est = _ctx_est
                        _fill = max(_actual, _real_est)
                        _ctx_pct = (
                            round(_fill / _climit * 100, 1)
                            if _climit > 0 and _fill > 0
                            else 0.0
                        )
                        _parent_reg.update_child_live(
                            _tid,
                            context_pct=_ctx_pct,
                            iter=iteration,
                            max_iter=max_iterations,
                            tokens_in=_fill,
                        )
            except Exception:
                # Defensive: best-effort path must not break the caller.
                logger.debug("Suppressed exception", exc_info=True)

            if not has_tool_call:
                # Round-51 T5: track watchdog effectiveness — an iteration
                # with no tool calls after a nudge counts as ineffective.
                if getattr(session, "_watchdog_nudge_count", 0):
                    session._watchdog_ineffective_count = int(
                        getattr(session, "_watchdog_ineffective_count", 0) or 0
                    ) + 1
                if not has_text:
                    logger.warning("Assistant provided empty response. Nudging.")

                    # Role-aware nudge: sub-agents are told (LAYER 3B) not to
                    # interact with the user and to return findings to the
                    # parent. The default nudge's "final answer to the user"
                    # phrasing contradicts that, producing confused reasoning
                    # ("the user is telling me…"). Use the child variant for
                    # child sessions so the prompt stays self-consistent.
                    _role = str(
                        session.variables.get("session_role", "") or ""
                    ).lower()
                    _nudge_text = (
                        NUDGE_EMPTY_RESPONSE_CHILD
                        if _role == "child"
                        else NUDGE_EMPTY_RESPONSE
                    )
                    # Round-14 F10: tag synthetic nudges so
                    # compact_completed_turn can locate the REAL turn
                    # boundary — a nudge stored as a plain user message
                    # made the compactor treat the suffix after the nudge
                    # as the whole turn, preserving stale tool metadata.
                    nudge_msg = {
                        "role": "user",
                        "parts": [
                            {"type": "text", "text": _nudge_text}
                        ],
                        "synthetic": True,
                    }
                    session.session_manager.history.append(nudge_msg)
                    emit_nudge(session, "empty_response", iteration, role=_role)
                    messages = session._build_messages_from_history(
                        session._prepare_runtime_history(),
                        {"role": "system", "parts": []},
                    )[:-1]
                    continue

                if active_mode == "loop" and iteration < max_iterations:
                    if session._loop_blocker_raised:
                        # The agent already raised a blocker this
                        # turn — pausing intentionally. Don't poke
                        # it; let the loop finalize so the user can
                        # respond. Without this gate the watchdog
                        # would re-prompt every iteration, burning
                        # tokens while the model repeats the
                        # blocker message.
                        logger.info(
                            "Loop mode: blocker raised; skipping watchdog continue."
                        )
                        if session.ui:
                            session.ui.show_info(
                                "Loop paused — blocker raised. "
                                "Provide the requested input, set a new loop goal, or /mode default."
                            )
                        # Fall through to the normal finalize path
                        # below (no `continue`).
                    else:
                        # Round-51 T5: watchdog escalation ladder. In the
                        # observed 517-iter run the watchdog nudged 15x with
                        # accelerating intervals and zero behavioral change.
                        # Track consecutive ineffective nudges (a nudge is
                        # ineffective when the assistant stopped without
                        # tool calls again), escalate to a plan-recap demand
                        # after 3, pause for user input after 5, and hard-
                        # cap total nudges per run at 8.
                        _wd_ineffective = int(
                            getattr(
                                session, "_watchdog_ineffective_count", 0
                            )
                            or 0
                        )
                        _wd_total = int(
                            getattr(session, "_watchdog_nudge_count", 0) or 0
                        )
                        if _wd_total >= 8:
                            # Hard cap: further watchdog triggers are no-ops.
                            logger.info(
                                "Loop mode watchdog: cap (%d) reached; "
                                "falling through to finalize.",
                                _wd_total,
                            )
                            if session.ui:
                                session.ui.show_info(
                                    "Loop watchdog nudge cap reached — falling "
                                    "through to finalize. Set a new loop goal "
                                    "or /mode default to continue."
                                )
                        else:
                            _watchdog_text = "LOOP WATCHDOG: Continue autonomous loop execution now. Re-plan the next increment, execute concrete actions with tools, verify outcomes, and proceed without waiting for user confirmation. Only pause if blocked, and if blocked call raise_blocker with exact unblock requirements."
                            if _wd_ineffective >= 5:
                                # Ladder rung 2: pause for user input via the
                                # existing raise_blocker mechanism instead of
                                # nudging again.
                                _watchdog_text = (
                                    "LOOP WATCHDOG ESCALATION: The continue-nudge has been ignored "
                                    f"{_wd_ineffective}x. Pause and ask for help: stop autonomous "
                                    "retrying, summarize exact progress and the concrete obstacle, "
                                    "and call raise_blocker with precise unblock requirements."
                                )
                                emit_nudge(
                                    session,
                                    "watchdog_pause",
                                    iteration,
                                    ineffective=_wd_ineffective,
                                )
                                session._watchdog_paused = True
                            elif _wd_ineffective >= 3:
                                # Ladder rung 1: structured re-plan demand.
                                _watchdog_text = (
                                    "LOOP WATCHDOG (plan recap required): Repeated continue-nudges "
                                    "have not produced progress. STOP executing. First output a "
                                    "structured re-plan: (1) original goal, (2) what you have "
                                    "actually completed so far, (3) the exact obstacle, (4) the next "
                                    "single concrete action. Then execute that one action with tools. "
                                    "If you cannot name a concrete action, call raise_blocker."
                                )
                                emit_nudge(
                                    session,
                                    "watchdog_escalated",
                                    iteration,
                                    ineffective=_wd_ineffective,
                                )
                                session._watchdog_escalated = True
                            watchdog_msg = {
                                "role": "user",
                                "parts": [
                                    {
                                        "type": "text",
                                        "text": _watchdog_text,
                                    }
                                ],
                                "synthetic": True,
                            }
                            session.session_manager.history.append(watchdog_msg)
                            emit_nudge(session, "loop_watchdog", iteration)
                            session._watchdog_nudge_count = _wd_total + 1
                            messages = session._build_messages_from_history(
                                session._prepare_runtime_history(),
                                {"role": "system", "parts": []},
                            )[:-1]
                            continue

                if session.ui:
                    session.ui.show_info(
                        f"Final session tokens: In {total_in} | Out {total_out} | Total {total_in + total_out} | Total Est. Cost: ${total_cost:.5f}"
                    )

                logger.info("Agentic loop finished (no tool calls).")

                if (
                    str(session.variables.get("agent_mode", "default")).lower()
                    == "feature"
                    and session.session_manager.get_feature_state()
                ):
                    session._set_feature_state()

                if session.variables.get("compact_history", False):
                    # Compaction must run regardless of UI presence —
                    # headless/API sessions with compact_history=True were
                    # silently skipping it under the old `if session.ui`.
                    if session.ui:
                        session.ui.show_info(
                            "[dim]Compacting turn history (removing tool metadata)...[/dim]"
                        )
                    session.session_manager.compact_completed_turn()
                    # Fold whatever the just-finished turn displaced into the
                    # rolling summary so compacted-away content survives in
                    # L2 — otherwise the summary never learns about turns
                    # that were dropped from history without a budget-driven
                    # compaction pass.
                    try:
                        session.session_manager.roll_history_summary(
                            keep_recent=resolve_keep_recent(session),
                            provider=session.provider,
                        )
                    except Exception as exc:
                        logger.debug("Post-turn summary roll skipped: %s", exc)
                    logger.debug("History compacted.")

                session.session_manager.save_history_turn(session.folder_context)
                # If a hook aborted during this final iteration, surface
                # that as the turn status — the abort fired, the user
                # should see why the loop stopped.
                final_status = (
                    "hook_aborted"
                    if session._hook_abort_requested
                    else "completed"
                )
                final_error = (
                    session._hook_abort_reason
                    if session._hook_abort_requested
                    else None
                )
                return session._collect_turn_response(
                    initial_history_len,
                    status=final_status,
                    total_in=total_in,
                    total_out=total_out,
                    total_cost=total_cost,
                    error=final_error,
                )

            strict_mode = session.variables.get("strict_mode", False)
            tool_result_parts = []
            tool_calls = [p for p in response.parts if p.type == "tool_call"]

            approval_plans = collect_approval_plans(
                tool_calls,
                session.folder_context,
                strict_mode=strict_mode,
                yolo=session.variables.get("yolo", False),
            )

            # Show bulk diffs if multiple
            if len(approval_plans) > 1:
                if session.ui:
                    session.ui.show_info(
                        f"\n[bold yellow]Turn contains {len(approval_plans)} modifications requiring approval.[/bold yellow]"
                    )
                for approval_plan in approval_plans.values():
                    for modification in approval_plan.modifications:
                        if modification.can_render_diff:
                            if session.ui:
                                session.ui.show_diff(
                                    modification.filename,
                                    modification.original_content,
                                    modification.modified_content,
                                )

            # --- PHASE 1: approval + decision (serial, in input order) ----
            # Walk every tool call once. For each, record either an
            # `early_result` (denied / preview-failed / etc.) OR mark it
            # `pending` so the parallel execution phase below will run it.
            pending_executions: list[int] = []  # indices to execute
            early_results: dict[int, Any] = {}  # i -> pre-resolved result string

            for i, part in enumerate(tool_calls):
                current_tool_name = part.tool_name
                current_tool_args = part.tool_args
                if session._track_tool_for_loop_detection(
                    part.tool_name, part.tool_args
                ):
                    iteration_tool_exact_fingerprints.append(
                        session._tool_call_fingerprint(part.tool_name, part.tool_args)
                    )
                    iteration_tool_pattern.append(
                        session._tool_call_fingerprint(
                            part.tool_name, part.tool_args, pattern_only=True
                        )
                    )
                approval_plan = approval_plans.get(i)
                needs_approval = approval_plan is not None
                if needs_approval:
                    result = None
                    auto_approved = bool(
                        session.variables.get("yolo", False)
                        and approval_plan.can_approve
                        and approval_plan.approval_policy != "always_human"
                    )

                    if approval_plan.preview_error and session.ui:
                        for modification in approval_plan.modifications:
                            if modification.preview_error:
                                session.ui.show_error(
                                    f"Cannot show diff for {modification.filename}: {modification.preview_error}"
                                )
                                logger.error(
                                    f"Diff error for {modification.filename}: {modification.preview_error}"
                                )
                                break

                    if (
                        approval_plan.error_code == "preview_failed"
                        and approval_plan.preview_error
                    ):
                        if session.ui:
                            session.ui.show_info(
                                f"  [yellow]Auto-retrying malformed patch for {part.tool_name}...[/yellow]"
                            )
                        result = (
                            "Error: Malformed patch detected. Please ensure your diff is correctly "
                            f"formatted. Check hunk headers and context.\n{approval_plan.preview_error}"
                        )
                        logger.warning(
                            f"Malformed patch detected for {part.tool_name}: {approval_plan.preview_error}"
                        )

                    # Show diffs if not already shown in bulk pre-calculation
                    if result is None and not auto_approved and len(approval_plans) <= 1:
                        for modification in approval_plan.modifications:
                            if modification.can_render_diff:
                                if session.ui:
                                    session.ui.show_diff(
                                        modification.filename,
                                        modification.original_content,
                                        modification.modified_content,
                                    )

                    # Shorten args for display
                    display_args = _shorten_tool_args(part.tool_args)

                    # Add count info to prompt if multiple
                    count_info = (
                        f" ({i + 1}/{len(tool_calls)})"
                        if len(tool_calls) > 1
                        else ""
                    )

                    if result is None and not auto_approved:
                        choice, reason = session._request_tool_approval(
                            approval_plan=approval_plan,
                            display_args=display_args,
                            count_info=count_info,
                        )
                        if choice == "n":
                            result = "User denied this tool call."
                            logger.info(
                                f"Tool call {part.tool_name} denied by user."
                            )
                        elif choice == "e":
                            result = f"User denied this tool call. Reason: {reason}"
                            logger.info(
                                f"Tool call {part.tool_name} denied by user with explanation: {reason}"
                            )
                        else:
                            auto_approved = True  # user said yes — defer to exec phase

                    if result is not None:
                        early_results[i] = result
                    elif auto_approved:
                        pending_executions.append(i)
                else:
                    # No approval needed — defer to exec phase.
                    pending_executions.append(i)

            # --- PHASE 2: execute pending calls (parallel for safe tools, serial for others) ---
            exec_results: dict[int, Any] = {}
            # Fix #10: auto-recall tracker. Maps tool-call index → cache_key for
            # reads that short-circuited to a cached result (same path, file
            # unchanged since the cached read). Post-processing annotates the
            # rendered result so the model knows it was served from cache and
            # doesn't re-burn tokens re-reading the same file — the core
            # context-gathering stall on long tasks.
            _auto_recall_hits: dict[int, str] = {}
            # Per-tool execution start times (monotonic) keyed by tool-call
            # index — consumed in the post-processing loop to emit a trace
            # `tool` record with real latency. Covers both the parallel
            # (lambda) and serial execution paths through this funnel.
            _tool_start_times: dict[int, float] = {}

            def _auto_recall_or_execute(part_idx: int, part) -> Any:
                _tool_start_times[part_idx] = time.monotonic()
                # Range-memo dedup (spec #7): if this exact range of an
                # unchanged file was already supplied this session, emit a
                # compact marker instead of re-injecting the content. The
                # marker offers the cache_key for recall, so the model can
                # still pull the content back if it was compacted away.
                if session.variables.get("read_dedup_enabled", True):
                    try:
                        rr = session.tool_result_cache.lookup_read_range(
                            part.tool_name, part.tool_args
                        )
                    except Exception:
                        rr = None
                    if rr is not None:
                        ck = str(rr.get("cache_key", "") or "")
                        rng = rr.get("range")
                        _auto_recall_hits[part_idx] = ck
                        if session.ui:
                            try:
                                session.ui.show_info(
                                    f"  [Dedup: {part.tool_name} range {rng} "
                                    f"already supplied — cache_key={ck}]"
                                )
                            except Exception:
                                # Defensive: best-effort path must not break the caller.
                                logger.debug("Suppressed exception", exc_info=True)
                        return (
                            f"[dedup: {part.tool_name} — file unchanged; this "
                            f"exact range was already read earlier in this "
                            f"conversation and its full content is in the "
                            f"message history above — use what you already "
                            f"have instead of re-reading. If a verbatim "
                            f"re-fetch is truly required, call "
                            f"recall({ck}) or result_range/result_search "
                            f"(when available in this session).]"
                        )
                hit = None
                try:
                    hit = session.tool_result_cache.lookup_by_locator(
                        part.tool_name, part.tool_args
                    )
                except Exception:
                    hit = None
                if hit is not None:
                    _auto_recall_hits[part_idx] = str(hit.get("cache_key", "") or "")
                    if session.ui:
                        try:
                            session.ui.show_info(
                                f"  [Auto-recall: {part.tool_name} served from "
                                f"cache {hit.get('cache_key', '')} — file unchanged]"
                            )
                        except Exception:
                            # Defensive: best-effort path must not break the caller.
                            logger.debug("Suppressed exception", exc_info=True)
                    return hit["result"]
                return session._execute_tool_with_memory(
                    part.tool_name, part.tool_args
                )

            if pending_executions:
                from mu.agent.parallel import (
                    PARALLEL_SAFE_TOOLS,
                    ToolCall as _ParTC,
                    execute_calls as _exec_calls,
                )

                parallel_indices: list[int] = []
                serial_indices: list[int] = []
                for i in pending_executions:
                    part = tool_calls[i]
                    # `flush` must be serial (it reads the collation
                    # buffer, which is populated by the post-processing
                    # phase below for each preceding call).
                    if part.tool_name == "flush":
                        serial_indices.append(i)
                    elif part.tool_name in PARALLEL_SAFE_TOOLS:
                        parallel_indices.append(i)
                    else:
                        serial_indices.append(i)

                # Order-preserving execution: a mixed batch like
                # [write_file, read_file] must NOT be reordered so all
                # "parallel-safe" calls run before serial mutations — the
                # model's call order encodes read-after-write dependencies.
                # Any serial call positioned BEFORE a parallel-safe call
                # forces the whole batch serial (barrier semantics); a
                # serial call after all parallel calls keeps the fast path.
                if (
                    parallel_indices
                    and serial_indices
                    and min(parallel_indices) > min(serial_indices)
                ):
                    parallel_indices, serial_indices = [], list(pending_executions)
                # One call has nothing to parallelize. Keeping it on the
                # agent thread avoids an event-loop/default-executor round
                # trip and preserves straightforward hook/UI ordering.
                if len(parallel_indices) == 1:
                    serial_indices.extend(parallel_indices)
                    serial_indices.sort()
                    parallel_indices = []

                # Parallel batch — preserves input-order results.
                if parallel_indices:
                    par_calls = [
                        _ParTC(
                            tool_name=tool_calls[i].tool_name,
                            tool_args=tool_calls[i].tool_args or {},
                            tool_call_id=str(i),
                            thought_signature=tool_calls[i].thought_signature,
                        )
                        for i in parallel_indices
                    ]
                    max_concurrency = max(
                        1,
                        int(
                            session.variables.get("parallel_tool_concurrency", 4) or 4
                        ),
                    )

                    # Sub-agent progress is now rendered on demand from the
                    # long-lived tracker owned by `session._subagent_registry`
                    # (populated by async spawn_agent). We deliberately do NOT
                    # run a continuous rich.Live here: a Live spanning the async
                    # gap would corrupt the console while the parent loop keeps
                    # writing tool results. Instead, when sub-agent spawns are
                    # in this batch, we render a one-shot static snapshot so the
                    # user sees the dispatch (single spawns included — the old
                    # `spawn_count >= 2` gate is gone).
                    spawn_count = sum(
                        1
                        for i in parallel_indices
                        if tool_calls[i].tool_name == "spawn_agent"
                    )
                    if spawn_count >= 1 and session.ui is not None:
                        try:
                            _tracker = session._subagent_registry.tracker
                            if _tracker.has_active():
                                session.ui.show_info(_tracker.render_panel())
                        except Exception:  # noqa: BLE001
                            pass

                    if len(par_calls) > 1 and session.ui:
                        session.ui.show_info(
                            f"⚡ Dispatching {len(par_calls)} tool call(s) in "
                            f"parallel (max_concurrency={max_concurrency})."
                        )

                    try:
                        par_results = _exec_calls(
                            par_calls,
                            lambda tc: _auto_recall_or_execute(
                                int(tc.tool_call_id), tc
                            ),
                            max_concurrency=max_concurrency,
                        )
                    finally:
                        # Keep the legacy transient attribute cleared for any
                        # external readers; the authoritative tracker lives on
                        # the registry now.
                        session._subagent_progress = None
                    for idx, pr in zip(parallel_indices, par_results):
                        if pr.error is not None:
                            logger.warning(
                                "Parallel tool %s raised %s",
                                tool_calls[idx].tool_name,
                                pr.error,
                            )
                            exec_results[idx] = f"Error: {pr.error}"
                        else:
                            exec_results[idx] = pr.result

                # Serial calls (executed in their original input order).
                for idx in serial_indices:
                    part = tool_calls[idx]
                    if part.tool_name == "flush":
                        # Flush is finalised inside the post-processing
                        # phase below so it can read the freshly written
                        # collation buffer. Placeholder result here.
                        exec_results[idx] = None
                        continue
                    try:
                        exec_results[idx] = _auto_recall_or_execute(idx, part)
                    except Exception as serial_exc:
                        # A serial tool raising must not abort the remaining
                        # serial calls or leave the tool call without a
                        # result (codex review finding #4) — that produced
                        # an invalid conversation sequence and lost the
                        # outcomes of earlier tools. Convert to an error
                        # result so Phase 3 archives one result per call.
                        logger.warning(
                            "Serial tool %s raised: %s", part.tool_name, serial_exc
                        )
                        exec_results[idx] = f"Error: {serial_exc}"

            # --- PHASE 3: post-processing (serial, in input order) -----------
            for i, part in enumerate(tool_calls):
                current_tool_name = part.tool_name
                current_tool_args = part.tool_args
                if i in early_results:
                    result = early_results[i]
                else:
                    result = exec_results.get(i)

                source_result = result
                raw_result = source_result
                logger.debug(
                    f"Tool result ({part.tool_name}): {_sanitize_for_log(raw_result)}"
                )
                # Surface retryable failures to the live UI with the
                # registered hint. The model already sees the structured
                # envelope in its next turn; this is for the human.
                # Returns retryable-failure count for this (tool, error_code)
                # pair so we can inject a corrective message when the model
                # keeps hitting the same retryable error with different args
                # (which evades pattern-based loop detection).
                _retryable_count = session._announce_retryable_failure(
                    part.tool_name, raw_result
                )
                if _retryable_count > _max_retryable_count_this_iter:
                    _max_retryable_count_this_iter = _retryable_count
                # Collect the error_code for the retryable loop-detection
                # lane (R8). The announcer stashes the latest code on the
                # session; we capture it here per failed call so multiple
                # distinct codes in one iteration each contribute a marker.
                if _retryable_count > 0:
                    _code = getattr(session, "_last_retryable_error_code", "")
                    if _code:
                        _iteration_retryable_codes.add(str(_code))
                # --- Collation Logic ---
                is_flush = part.tool_name == "flush"
                is_discard_deferred = part.tool_name == "discard_deferred_context"
                should_collate = (
                    part.tool_name in COLLATED_TOOLS
                    and session.variables.get("collation_enabled", True)
                    and len(tool_calls) > 1
                )

                if is_flush:
                    requested_ids = part.tool_args.get("artifact_ids") if isinstance(part.tool_args, dict) else None
                    collated_pairs = session.collation_buffer.flush_selected(requested_ids)
                    collated_data = [body for _, body in collated_pairs]
                    if not collated_data:
                        # Sub-increment 12a: self-diagnosing message so the
                        # model doesn't read "No data" as data loss (see
                        # _empty_flush_message — collation-disabled children
                        # and single-call batches were re-deriving evidence).
                        raw_result = _empty_flush_message(session)
                    else:
                        raw_result = "--- Flushed Context ---\n" + "\n\n".join(
                            collated_data
                        )
                    if session.ui:
                        session.ui.show_info(
                            f"  [Flushed {len(collated_data)} items from buffer]"
                        )
                    try:
                        from mu.trace.emitter import emit_context_artifact
                        for artifact_id, body in collated_pairs:
                            emit_context_artifact(session, iteration=iteration,
                                artifact_id=artifact_id, state="delivered",
                                bytes=len(body.encode("utf-8", errors="replace")),
                                reason="model_flush")
                    except Exception:
                        # Defensive: best-effort path must not break the caller.
                        logger.debug("Suppressed exception", exc_info=True)
                elif is_discard_deferred:
                    ids = part.tool_args.get("artifact_ids", []) if isinstance(part.tool_args, dict) else []
                    removed = session.collation_buffer.discard(ids if isinstance(ids, list) else [])
                    raw_result = f"Discarded {len(removed)} deferred context artifact(s): {', '.join(removed)}"
                    try:
                        from mu.trace.emitter import emit_context_artifact
                        for artifact_id in removed:
                            emit_context_artifact(session, iteration=iteration,
                                artifact_id=artifact_id, state="discarded",
                                reason=str(part.tool_args.get("reason", "model_cleanup")))
                    except Exception:
                        # Defensive: best-effort path must not break the caller.
                        logger.debug("Suppressed exception", exc_info=True)
                elif should_collate:
                    # Don't collate if there was an error
                    collation_cache_key = None
                    if raw_result and not str(raw_result).startswith("Error"):
                        # Cache remains a fast recall path, but collation is
                        # lossless: deferred evidence is retained until the
                        # model delivers or explicitly discards it. force=True
                        # caches even tools not in _CACHEABLE_TOOLS (e.g.
                        # web_search, read_document) so every collated
                        # read-only result has a recall() path. See R11
                        # (FM-4).
                        try:
                            collation_cache_key = session.tool_result_cache.store(
                                call_id=getattr(part, "tool_call_id", ""),
                                tool_name=part.tool_name,
                                result=source_result,
                                force=True,
                            )
                        except Exception:
                            collation_cache_key = None
                        session.collation_buffer.add(
                            part.tool_name, part.tool_args, raw_result
                        )
                        count = len(session.collation_buffer.entries)
                        # Round-46 F7: O(1) — cached at add() time, no full re-manifest.
                        artifact = session.collation_buffer.last_manifest_entry()
                        cache_hint = (
                            f" [cache:{collation_cache_key}]"
                            if collation_cache_key
                            else ""
                        )
                        raw_result = (
                            f"Stored '{part.tool_name}' result in collation buffer. "
                            f"artifact_id={artifact['id']} bytes={artifact['bytes']}. "
                            f"{count} item(s) currently pending. Call `flush` with artifact_ids "
                            "for selected evidence, omit IDs for all, or `discard_deferred_context` "
                            "when deliberately cleaning up."
                            f"{cache_hint}"
                        )
                        try:
                            from mu.trace.emitter import emit_context_artifact
                            emit_context_artifact(session, iteration=iteration,
                                artifact_id=artifact["id"], state="deferred",
                                tool_name=part.tool_name,
                                path=str(part.tool_args.get("filename") or part.tool_args.get("path") or "") if isinstance(part.tool_args, dict) else "",
                                bytes=artifact["bytes"], reason="collation")
                        except Exception:
                            # Defensive: best-effort path must not break the caller.
                            logger.debug("Suppressed exception", exc_info=True)
                    if session.ui and active_mode != "loop":
                        session.ui.show_info(f"  [Collated: {part.tool_name}]")
                    else:
                        # If it's an error, don't collate it, let the model see the error immediately
                        if session.ui:
                            session.ui.show_tool_result(
                                session._render_tool_result(raw_result)
                            )
                else:
                    if session.ui and active_mode != "loop":
                        session.ui.show_tool_result(
                            session._render_tool_result(raw_result)
                        )

                if session.ui and hasattr(session.ui, "emit_tool_trace"):
                    session.ui.emit_tool_trace(
                        part.tool_name,
                        part.tool_args,
                        source_result,
                        raw_result,
                    )

                # --- End Collation Logic ---
                # Feed this tool call to the child's lifecycle manager (only
                # set on async-orchestrated sub-agent sessions). Uses the full
                # args + result so the manager can detect stuck (same tool +
                # same args repeated) and stall (no novel output) signals.
                # No-op for the parent and for non-child sessions.
                _lifecycle = getattr(session, "_subagent_lifecycle", None)
                if _lifecycle is not None:
                    try:
                        _lifecycle.record_tool_call(
                            part.tool_name,
                            part.tool_args,
                            source_result,
                            cache_hit=bool(_auto_recall_hits.get(i)),
                        )
                    except Exception:
                        logger.debug(
                            "lifecycle record_tool_call failed", exc_info=True
                        )
                # Store full result in tool result cache BEFORE building the
                # structured envelope so the compact observation (spec #2/#10)
                # can embed the stored_ref cache_key. Cache the RAW
                # source_result (original tool output), not the structured
                # `result` envelope — when collation transforms the result,
                # `result` contains metadata like {"collated": True, ...} which
                # is useless to recall(). source_result has the actual file
                # content / search output the model needs to recover. The
                # store also write-throughs to the durable ResultStore (spec #11).
                #
                # If the collation branch already cached this result (and
                # stamped the key into the placeholder), reuse that key so
                # the part's cache_key matches what the model saw — and skip
                # the second store (idempotent, but wasteful).
                cache_key = None
                try:
                    if should_collate and collation_cache_key:
                        cache_key = collation_cache_key
                    else:
                        # store_with_locator (Fix #10): index by tool+args and
                        # record the source file's mtime/size so a later
                        # repeat read of the same unchanged file auto-recalls
                        # from cache instead of re-reading + re-burning
                        # tokens. Falls back to plain store when args are
                        # missing/unhashable.
                        try:
                            cache_key = (
                                session.tool_result_cache.store_with_locator(
                                    call_id=getattr(part, "tool_call_id", ""),
                                    tool_name=part.tool_name,
                                    tool_args=part.tool_args,
                                    result=source_result,
                                )
                            )
                        except Exception:
                            cache_key = session.tool_result_cache.store(
                                call_id=getattr(part, "tool_call_id", ""),
                                tool_name=part.tool_name,
                                result=source_result,
                            )
                except Exception:
                    # Defensive: best-effort path must not break the caller.
                    logger.debug("Suppressed exception", exc_info=True)

                # Range-memo bookkeeping (spec #7): record the supplied range
                # so a later overlapping read of the unchanged file dedups.
                # On writes, invalidate the memo for that path so the next
                # read re-executes against fresh content.
                try:
                    if part.tool_name in {
                        "write_file", "apply_diff", "search_and_replace_file"
                    }:
                        _wpath = part.tool_args.get("filename") if isinstance(
                            part.tool_args, dict
                        ) else None
                        if _wpath:
                            session.tool_result_cache.invalidate_path(_wpath)
                    elif cache_key is not None and session.variables.get(
                        "read_dedup_enabled", True
                    ):
                        session.tool_result_cache.record_read_range(
                            part.tool_name, part.tool_args, cache_key
                        )
                except Exception:
                    # Defensive: best-effort path must not break the caller.
                    logger.debug("Suppressed exception", exc_info=True)

                if session.variables.get("structured_tool_results", True):
                    if raw_result != source_result:
                        _, unwrapped_source = session._unwrap_tool_envelope(
                            source_result
                        )
                        source_text = str(unwrapped_source)
                        result = session._build_structured_tool_result(
                            part.tool_name,
                            part.tool_args,
                            raw_result,
                            execution_source="session",
                            cache_key=cache_key,
                        )
                        result["data"] = {
                            "collated": True,
                            "pending_items": len(session.collation_buffer.entries),
                            "source_char_count": len(source_text),
                            "source_line_count": len(source_text.splitlines()),
                        }
                        result["telemetry"].update(
                            {
                                "delivery_mode": "collated",
                                "visible_char_count": len(str(raw_result)),
                            }
                        )
                    else:
                        result = session._build_structured_tool_result(
                            part.tool_name,
                            part.tool_args,
                            source_result,
                            execution_source="session",
                            cache_key=cache_key,
                        )
                else:
                    result = raw_result

                session._sync_feature_state_for_tool(
                    part.tool_name,
                    part.tool_args,
                    source_result,
                    result,
                )

                # Efficiency metrics (spec #12): fold this result's telemetry
                # into the per-turn accumulators, and count retrieval-tool
                # calls (recall + result_* family) for the retrieval rate.
                try:
                    from mu.session.efficiency_metrics import (
                        accumulate_tool_result,
                        is_retrieval_tool,
                    )

                    accumulate_tool_result(session, result)
                    if is_retrieval_tool(part.tool_name):
                        session._eff_retrievals = int(
                            getattr(session, "_eff_retrievals", 0)
                        ) + 1
                except Exception:  # noqa: BLE001
                    pass

                tool_result_part = {
                    "type": "tool_result",
                    "tool_name": part.tool_name,
                    "tool_result": result,
                    "thought_signature": part.thought_signature,
                    "cache_key": cache_key,
                }
                # Keep a compact, first-class visualization descriptor beside
                # the tool result. Structured observation transforms and older
                # transports may reshape the result body; history replay should
                # not have to rediscover the chat card inside that envelope.
                if part.tool_name in {
                    "publish_visualization",
                    "create_visualization",
                    "render_visualization",
                }:
                    try:
                        from mu.artifact.history import extract_visualization

                        visualization = extract_visualization(source_result)
                        if visualization is not None:
                            tool_result_part["artifact"] = visualization
                    except Exception:
                        logger.debug(
                            "visualization history descriptor capture failed",
                            exc_info=True,
                        )
                tool_result_parts.append(tool_result_part)
                # Emit cache_key to the GUI so clicking the tool_result trace
                # event can fetch the full content from the cache endpoint
                # (the L5 history only carries a compact ref now).
                if cache_key and hasattr(session.ui, "_publish"):
                    try:
                        session.ui._publish({
                            "kind": "tool_result_cache",
                            "tool_name": part.tool_name,
                            "cache_key": cache_key,
                        })
                    except Exception:
                        # Defensive: best-effort path must not break the caller.
                        logger.debug("Suppressed exception", exc_info=True)
                # --- Trace: per-tool capture (latency, cache hit, result size) ---
                try:
                    _t_start = _tool_start_times.pop(i, None)
                    _lat = (
                        int((time.monotonic() - _t_start) * 1000)
                        if _t_start is not None
                        else 0
                    )
                    _arg_fp = session._tool_call_fingerprint(
                        part.tool_name, part.tool_args
                    )
                    _tok_path = ""
                    _ta = part.tool_args or {}
                    if isinstance(_ta, dict):
                        # Read/search tools name their target arg differently
                        # (read_file uses ``filename``, search_for_string uses
                        # ``search_string``, search_references uses ``query``…).
                        # Pull the first present one so the reads heatmap and
                        # redundant-read detection actually have paths to work
                        # with. Writes that share these keys are filtered out of
                        # the read view by tool name on the parser side.
                        for _pk in (
                            "path", "file_path", "filepath", "filename", "fp",
                            "search_string", "query", "pattern", "target",
                        ):
                            _pv = _ta.get(_pk)
                            if _pv:
                                _tok_path = str(_pv)
                                break
                    _res_preview = ""
                    try:
                        _res_preview = str(source_result or "")[:200]
                    except Exception:  # noqa: BLE001
                        _res_preview = ""
                    # Efficiency telemetry (spec #12): pull the observation
                    # transform's per-result metrics off the structured envelope
                    # so the trace tool record + efficiency panel can show raw vs
                    # injected tokens, delivery mode, omitted, and the store_key.
                    _eff_raw = 0
                    _eff_inj = 0
                    _eff_mode = ""
                    _eff_omitted = False
                    _eff_cr = None
                    if isinstance(result, dict):
                        _tele = result.get("telemetry") or {}
                        if isinstance(_tele, dict):
                            _eff_raw = int(_tele.get("raw_token_count") or 0)
                            _eff_inj = int(_tele.get("injected_token_count") or 0)
                            _eff_mode = str(_tele.get("delivery_mode") or "")
                            _crv = _tele.get("compression_ratio")
                            if _crv is not None:
                                try:
                                    _eff_cr = float(_crv)
                                except Exception:  # noqa: BLE001
                                    _eff_cr = None
                        _rd = result.get("data")
                        if isinstance(_rd, dict):
                            _eff_omitted = bool(_rd.get("omitted"))
                    # Round-51 T4: the ok/error_code pair must come from the
                    # tool envelope itself. The old string-prefix heuristic
                    # (ok = not raw.startswith("Error")) recorded ok=true for
                    # every JSON-envelope failure (545 invalid_args in one
                    # trace run) because envelopes never start with "Error".
                    _envelope, _ = session._unwrap_tool_envelope(
                        raw_result
                        if isinstance(raw_result, str)
                        else str(raw_result)
                    )
                    if isinstance(_envelope, dict):
                        _tool_ok = bool(_envelope.get("ok", True))
                        _tool_error_code = (
                            str(_envelope.get("error_code") or "") or None
                        )
                        if not _tool_ok and not _tool_error_code:
                            # Backstop: some handlers return ok=false with a
                            # message but no error_code. Classify from the
                            # message text so trace records always carry a
                            # code for failures (T4).
                            try:
                                from mu.tools._envelope import (
                                    infer_tool_error_code,
                                )
                                _tool_error_code = infer_tool_error_code(
                                    part.tool_name, raw_result
                                )
                            except Exception:  # noqa: BLE001
                                _tool_error_code = None
                    else:
                        _tool_ok = not str(raw_result).startswith("Error")
                        _tool_error_code = (
                            str(
                                getattr(
                                    session, "_last_retryable_error_code", ""
                                )
                                or ""
                            )
                            or None
                        )
                    emit_tool(
                        session,
                        iteration=iteration,
                        name=part.tool_name,
                        arg_fp=_arg_fp,
                        ok=_tool_ok,
                        error_code=_tool_error_code,
                        latency_ms=_lat,
                        cache_hit=bool(_auto_recall_hits.get(i)),
                        result_bytes=len(str(source_result or "")),
                        path=_tok_path,
                        preview=_res_preview,
                        store_key=cache_key or None,
                        stored=bool(cache_key),
                        raw_tokens=_eff_raw,
                        injected_tokens=_eff_inj,
                        delivery_mode=_eff_mode,
                        omitted=_eff_omitted,
                        compression_ratio=_eff_cr,
                    )
                except Exception:  # noqa: BLE001 — telemetry must not break the loop
                    pass
                current_tool_name = None
                current_tool_args = None

            tool_result_msg = {"role": "tool", "parts": tool_result_parts}
            session.session_manager.history.append(tool_result_msg)
            # Round-46 F1: throttled checkpoint — the first save of the turn
            # is immediate (baseline), afterwards at most one full-document
            # checkpoint per 2s / 10 iterations. Turn end and compaction
            # sites still save unconditionally.
            _now = time.monotonic()
            _iters_since_checkpoint += 1
            if (
                _last_history_checkpoint == 0.0
                or _iters_since_checkpoint >= _save_checkpoint_every_iters
                or (_now - _last_history_checkpoint)
                >= _save_checkpoint_min_interval
            ):
                session.session_manager.save_history_turn(session.folder_context)
                _last_history_checkpoint = _now
                _iters_since_checkpoint = 0
            turn_had_tool_execution = True

            # --- Fix #12: context-gathering stall detection -------------
            # Count consecutive iterations that re-cover already-read paths
            # without a concrete change. When the count hits the threshold
            # (and cooldown has passed), inject a re-orient nudge so the
            # next iteration's prompt steers the agent to act instead of
            # gathering more context it already has.
            if recoverage_threshold > 0:
                try:
                    _iter_read_paths = extract_read_paths(tool_calls)
                    _recovered = {
                        p for p in _iter_read_paths
                        if p in session._recoverage_seen_paths
                    }
                    _concrete = is_concrete_change_iter(tool_calls)
                    # Grow the seen set with this iteration's reads.
                    session._recoverage_seen_paths.update(_iter_read_paths)
                    if _recovered and not _concrete:
                        session._recoverage_stall_iters += 1
                    else:
                        # A concrete change or fresh-only reads reset the
                        # stall counter — the agent is making progress.
                        session._recoverage_stall_iters = 0
                    _cooldown_ok = (
                        iteration - session._recoverage_last_nudge_iter
                    ) >= recoverage_threshold
                    if (
                        session._recoverage_stall_iters >= recoverage_threshold
                        and _cooldown_ok
                    ):
                        _nudge = (
                            "CONTEXT-GATHERING STALL: you have re-read files "
                            f"{session._recoverage_stall_iters} iterations in a row "
                            "without making a concrete change. You already have "
                            "enough context to make progress. STOP re-reading and "
                            "re-checking the same files. Take a concrete action NOW: "
                            "apply the code change, run the test, or — if you're "
                            "blocked — raise_blocker with the exact missing "
                            "requirement. Re-reading will not unblock you."
                        )
                        session.session_manager.history.append({
                            "role": "user",
                            "parts": [{"type": "text", "text": _nudge}],
                        })
                        session.session_manager.save_history_turn(
                            session.folder_context
                        )
                        emit_nudge(
                            session,
                            "recoverage_stall",
                            iteration,
                            stall_iters=session._recoverage_stall_iters,
                        )
                        session._recoverage_last_nudge_iter = iteration
                        # Reset so we don't nudge every iteration; the next
                        # nudge fires only after another `threshold` stalled
                        # iterations.
                        session._recoverage_stall_iters = 0
                        if session.ui:
                            session.ui.show_error(
                                "Context-gathering stall detected — "
                                "re-orient nudge injected."
                            )
                        logger.warning(
                            "Recoverage stall at iteration %s/%s — nudge injected.",
                            iteration,
                            max_iterations,
                        )
                except Exception:
                    logger.debug(
                        "recoverage stall check failed", exc_info=True
                    )



            if loop_detection_enabled and iteration_tool_exact_fingerprints:
                exact_seq = " -> ".join(iteration_tool_exact_fingerprints)
                pattern_seq = " -> ".join(iteration_tool_pattern)
                exact_tool_sequence_history.append(exact_seq)
                pattern_tool_sequence_history.append(pattern_seq)

                exact_loop_detected = session._is_repeated_tool_sequence(
                    exact_tool_sequence_history,
                    repeat_threshold=loop_detection_repeat_threshold,
                )
                pattern_loop_detected = session._is_repeated_tool_sequence(
                    pattern_tool_sequence_history,
                    repeat_threshold=loop_detection_repeat_threshold,
                )
                # Periodic (non-consecutive) cycle detection on the pattern
                # lane (R7, FM-6). Catches read-loops where the agent
                # alternates between distinct iteration shapes — e.g.
                # [read a, edit, read a, edit] (period 2) — which never
                # produce `repeat_threshold` consecutive identical entries.
                # Path-aware collapsing makes multi-file read cycles
                # collapse to a repeated fingerprint in the pattern lane.
                pattern_periodic_detected = (
                    not pattern_loop_detected
                    and session._is_periodic_sequence(
                        pattern_tool_sequence_history,
                        max_period=int(
                            session.variables.get(
                                "loop_detection_periodic_max_period", 6
                            )
                        ),
                        min_repeats=2,
                    )
                )

                if exact_loop_detected or pattern_loop_detected or pattern_periodic_detected:
                    if exact_loop_detected:
                        loop_kind = "exact"
                    elif pattern_loop_detected:
                        loop_kind = "pattern"
                    else:
                        loop_kind = "periodic"
                    warning_text = (
                        "Loop detection triggered: repeated tool-call sequence "
                        f"detected {loop_detection_repeat_threshold}x ({loop_kind})."
                    )
                    if session.ui:
                        session.ui.show_error(warning_text)
                    logger.warning(warning_text)
                    loop_break_msg = {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": (
                                    "LOOP DETECTED: You repeated the same tool-call sequence three times. "
                                    "You MUST break out now. Do NOT repeat this sequence again. "
                                    "Take a materially different action: apply a concrete code change, "
                                    "run a different validation path, or raise_blocker with exact missing requirements. "
                                    "Then explain what changed and why this breaks the loop."
                                ),
                            }
                        ],
                    }
                    session.session_manager.history.append(loop_break_msg)
                    session.session_manager.save_history_turn(session.folder_context)
                    emit_nudge(session, "loop_detect", iteration)
                    messages = session._build_messages_from_history(
                        session._prepare_runtime_history(turn_start_index),
                        {"role": "system", "parts": []},
                    )[:-1]
                    continue

            # --- Retryable-failure cross-reference (R8, FM-6) -----------
            # Retryable storms where the model varies args each call (e.g.
            # read_file against f1.py, f2.py, f3.py, all returning a
            # retryable not_found envelope) evade both the normal pattern
            # lane (read_file isn't pattern-sensitive, so each filename
            # hashes to a distinct fingerprint) and the per-iteration
            # escalation below (which counts within a single iteration).
            # Feed a synthetic `retryable~<error_code>` fingerprint per
            # distinct code per iteration into its own history lane; a
            # storm spanning `loop_detection_repeat_threshold` iterations
            # trips `is_repeated_tool_sequence` here and breaks the loop.
            if (
                loop_detection_enabled
                and _iteration_retryable_codes
            ):
                for _code in sorted(_iteration_retryable_codes):
                    retryable_tool_sequence_history.append(f"retryable~{_code}")
                retryable_loop_detected = session._is_repeated_tool_sequence(
                    retryable_tool_sequence_history,
                    repeat_threshold=loop_detection_repeat_threshold,
                )
                if retryable_loop_detected:
                    warning_text = (
                        "Loop detection triggered: repeated retryable-failure "
                        f"sequence detected {loop_detection_repeat_threshold}x "
                        "(retryable)."
                    )
                    if session.ui:
                        session.ui.show_error(warning_text)
                    logger.warning(warning_text)
                    retryable_break_msg = {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": (
                                    "LOOP DETECTED: You hit the same retryable error "
                                    f"{loop_detection_repeat_threshold} iterations in a "
                                    "row while varying arguments. The tool will keep "
                                    "failing this way. You MUST break out now. Do NOT "
                                    "call the failing tool again with slightly different "
                                    "arguments. Re-read the target with read_file to "
                                    "confirm exact current state, take a materially "
                                    "different approach, or call raise_blocker with "
                                    "exact missing requirements."
                                ),
                            }
                        ],
                    }
                    session.session_manager.history.append(retryable_break_msg)
                    session.session_manager.save_history_turn(session.folder_context)
                    emit_nudge(session, "loop_detect_retryable", iteration)
                    messages = session._build_messages_from_history(
                        session._prepare_runtime_history(turn_start_index),
                        {"role": "system", "parts": []},
                    )[:-1]
                    _max_retryable_count_this_iter = 0
                    continue

            # Retryable-failure escalation: if any tool hit a retryable
            # error code too many times this iteration (even with different
            # args, which evades pattern-based loop detection above),
            # inject a corrective message telling the model to stop
            # retrying the same failing tool and try a different approach.
            # Threshold is configurable via `retryable_escalation_threshold`
            # (default 3 — lowered from 5 so a same-(tool,error) storm
            # escalates sooner; R8).
            _RETRYABLE_ESCALATION_THRESHOLD = int(
                session.variables.get("retryable_escalation_threshold", 3)
            )
            if _max_retryable_count_this_iter >= _RETRYABLE_ESCALATION_THRESHOLD:
                escalation_text = (
                    f"RETRYABLE FAILURE ESCALATION: A tool has hit the same retryable "
                    f"error {_max_retryable_count_this_iter}x this turn with different arguments. "
                    f"This means the approach itself is wrong — the tool will keep failing. "
                    f"Do NOT call this tool again with slightly different arguments. "
                    f"Instead: re-read the target file with read_file to confirm exact current "
                    f"state, or use a completely different tool/approach to achieve the goal. "
                    f"If you cannot proceed, call raise_blocker to ask the user for guidance."
                )
                if session.ui:
                    session.ui.show_error(escalation_text)
                logger.warning(escalation_text)
                escalation_msg = {
                    "role": "user",
                    "parts": [{"type": "text", "text": escalation_text}],
                }
                session.session_manager.history.append(escalation_msg)
                session.session_manager.save_history_turn(session.folder_context)
                emit_nudge(session, "retryable_escalation", iteration)
                messages = session._build_messages_from_history(
                    session._prepare_runtime_history(turn_start_index),
                    {"role": "system", "parts": []},
                )[:-1]
                _max_retryable_count_this_iter = 0
                continue

            messages = session._build_messages_from_history(
                session._prepare_runtime_history(turn_start_index),
                {"role": "system", "parts": []},
            )[:-1]

        except _HookAbort as abort_exc:
            # A `pre_provider_call` hook aborted the turn. The flag was
            # already set in `_provider_generate_with_retry`; just close
            # the turn cleanly without surfacing an "API Error" banner.
            reason = abort_exc.reason or "Hook requested abort"
            logger.info(f"Agentic loop aborted by hook: {reason}")
            if session.session_manager.get_feature_state():
                session._set_feature_state(status="hook_aborted")
            return session._collect_turn_response(
                initial_history_len,
                status="hook_aborted",
                total_in=total_in,
                total_out=total_out,
                total_cost=total_cost,
                error=reason,
            )
        except KeyboardInterrupt:
            if session.ui:
                session.ui.show_info("\nAgentic loop interrupted by user.")
            logger.warning("Agentic loop interrupted by user.")
            session.paused_execution_text = str(text or "")
            session.session_manager.history.append(
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_result",
                            "tool_name": "system",
                            "tool_result": "User interrupted execution.",
                        }
                    ],
                }
            )
            session.session_manager.save_history_turn(session.folder_context)
            if session.session_manager.get_feature_state():
                session._set_feature_state(status="interrupted")
            return session._collect_turn_response(
                initial_history_len,
                status="interrupted",
                total_in=total_in,
                total_out=total_out,
                total_cost=total_cost,
                error="User interrupted execution.",
            )
        except Exception as e:
            traceback_text = traceback.format_exc()
            tool_context = ""
            if current_tool_name:
                tool_context = (
                    f" | Last tool: {current_tool_name}("
                    f"{_shorten_tool_args(current_tool_args or {})})"
                )
            if session.ui:
                session.ui.show_error(f"API Error during agentic loop: {e}{tool_context}")
                session.ui.show_error(
                    "Traceback (most recent call last):\n"
                    + "\n".join(traceback_text.strip().splitlines()[-8:])
                )
            logger.error(f"Error in agentic loop: {e}", exc_info=True)

            status_code = session._extract_http_status_code(str(e).lower())
            if (
                not provider_bad_request_retried
                and not turn_had_tool_execution
                and status_code is not None
                and 400 <= status_code < 500
                and status_code not in {408, 409, 425, 429}
            ):
                # Roll back + retry once ONLY before any tool has run this
                # turn. Once a tool side effect exists, deleting its history
                # record orphans an irreversible action and retrying the
                # request can duplicate it — surface the error instead.
                provider_bad_request_retried = True
                session.session_manager.history = session.session_manager.history[:initial_history_len]
                session.session_manager.summary_anchor = min(
                    session.session_manager.summary_anchor,
                    len(session.session_manager.history),
                )
                session.session_manager.history.append(new_user_message)
                session.session_manager.save_history_turn(session.folder_context)
                messages = session._build_messages_from_history(
                    session._prepare_runtime_history(),
                    new_user_message,
                )
                iteration -= 1
                if session.ui:
                    session.ui.show_info(
                        f"Provider returned HTTP {status_code}. Rolled back the current turn and retrying once."
                    )
                continue

            choice = session._provider_error_recovery_choice()
            if choice == "rollback_retry":
                session.session_manager.history = session.session_manager.history[: turn_start_index + 1]
                session.session_manager.summary_anchor = min(
                    session.session_manager.summary_anchor,
                    len(session.session_manager.history),
                )
                session.session_manager.save_history_turn(session.folder_context)
                messages = session._build_messages_from_history(
                    session._prepare_runtime_history(turn_start_index),
                    {"role": "system", "parts": []},
                )[:-1]
                iteration -= 1
                continue
            if choice == "retry":
                iteration -= 1  # Decrement so the next loop run tries the same step
                continue

            session.session_manager.save_history_turn(session.folder_context)
            if session.session_manager.get_feature_state():
                session._set_feature_state(status="error")
            return session._collect_turn_response(
                initial_history_len,
                status="error",
                total_in=total_in,
                total_out=total_out,
                total_cost=total_cost,
                error=f"{e}{tool_context}",
            )

    session.session_manager.save_history_turn(session.folder_context)
    session.paused_execution_text = None

    # Fix #13: instead of silently stopping at max_iterations mid-work,
    # run ONE final consolidation turn — inject a user message telling the
    # model the budget is exhausted and asking it to write what it
    # accomplished, what's left, and save that to memory — then make a
    # final provider call with tools disabled so it can only respond, not
    # spin more tool calls. This turns an abrupt "max iterations reached"
    # cliff into a useful handoff the user can act on. Guarded by
    # `_consolidation_done` so it never recurses.
    if not getattr(session, "_consolidation_done", False):
        try:
            session._consolidation_done = True
            consolidation_text = (
                f"You have reached the maximum iteration budget ({max_iterations}). "
                "You will NOT get another tool call — this is your final response. "
                "Consolidate now:\n"
                "1. State what you actually accomplished this turn (files changed, "
                "tests run, concrete outcomes) — be specific.\n"
                "2. State exactly what remains to finish the task (next actionable steps).\n"
                "3. If any blocker stopped you, name it precisely and what you need.\n"
                "Do NOT call more tools. Do NOT re-read files. Summarize from what you "
                "already know and respond."
            )
            session.session_manager.history.append({
                "role": "user",
                "parts": [{"type": "text", "text": consolidation_text}],
            })
            session.session_manager.save_history_turn(session.folder_context)

            # Fresh L2/L3 for the consolidation call (cheap; reuses per-turn
            # cached L1B).
            _consol_prompt = session._inject_hierarchical_context(
                base_persona_prompt,
                cached_skills=session._turn_skills_block,
            )
            _consol_messages = session._build_messages_from_history(
                session._prepare_runtime_history(turn_start_index),
                {"role": "system", "parts": []},
            )[:-1]
            if session.ui and hasattr(session.ui, "build_live_status"):
                _cstatus = session.ui.build_live_status(
                    session, session.provider.model_name,
                    max_iterations, max_iterations,
                )
            else:
                _cstatus = (
                    f"Consolidating ({session.provider.model_name}) "
                    f"max_iter reached | {build_live_status_line(session)}"
                )
            if session.ui:
                with session.ui.show_status(_cstatus):
                    _consol_resp = session._provider_generate_with_retry(
                        messages=_consol_messages,
                        system_prompt=_consol_prompt,
                        thinking=session.thinking,
                        tools=None,  # no more tool calls — respond only
                    )
            else:
                _consol_resp = session._provider_generate_with_retry(
                    messages=_consol_messages,
                    system_prompt=_consol_prompt,
                    thinking=session.thinking,
                    tools=None,
                )
            total_in += int(getattr(_consol_resp, "input_tokens", 0) or 0)
            total_out += int(getattr(_consol_resp, "output_tokens", 0) or 0)
            # Render the consolidation text so the user sees it, and append
            # it to history as the assistant's final turn.
            _consol_parts = []
            for _p in getattr(_consol_resp, "parts", []) or []:
                if getattr(_p, "type", "") == "text" and getattr(_p, "text", ""):
                    if session.ui:
                        session.ui.render_message(
                            "assistant", _p.text, session.provider.model_name
                        )
                    _consol_parts.append({"type": "text", "text": _p.text})
            if _consol_parts:
                session.session_manager.history.append({
                    "role": "assistant",
                    "parts": _consol_parts,
                })
                session.session_manager.save_history_turn(session.folder_context)
            # Persist the consolidation into task memory so the next turn
            # inherits the handoff instead of re-deriving state from scratch.
            try:
                _consol_blob = " ".join(
                    str(p.get("text", "")) for p in _consol_parts if p.get("text")
                ).strip()
                if _consol_blob:
                    session.task_memory.save(
                        _consol_blob[:2000],
                        tags=["consolidation", "max_iterations"],
                        source="max_iterations_consolidation",
                        kind="consolidation",
                        # A consolidation is a handoff/audit record, not active
                        # working memory — save as DONE so it stays out of the
                        # default active+stale search and the active-first L3
                        # injection. It's still retrievable via
                        # search_memory(status="done") / include_all if a later
                        # turn needs the handoff. (anti-rot: audit-is-invisible.)
                        status="done",
                    )
            except Exception:
                logger.debug("consolidation memory persist failed", exc_info=True)
            logger.info(
                "Forced consolidation turn at max_iterations (%s).",
                max_iterations,
            )
        except Exception:
            logger.debug("consolidation turn failed", exc_info=True)

    if session.session_manager.get_feature_state():
        session._set_feature_state(status="max_iterations_reached")
    return session._collect_turn_response(
        initial_history_len,
        status="max_iterations_reached",
        total_in=total_in,
        total_out=total_out,
        total_cost=total_cost,
        error=(
            f"Reached maximum iterations ({max_iterations}). A final "
            "consolidation summary was generated — see the last assistant "
            "message for what was accomplished and what remains."
        ),
    )
