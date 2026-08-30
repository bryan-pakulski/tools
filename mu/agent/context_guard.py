"""Context-window guard: token estimates, preflight check, drift
calibration, and reactive overflow recovery.

Extracted from mu/agent/loop_body.py (which re-exports every name so the
test modules importing `from mu.agent.loop_body import _preflight_...`
keep working unchanged).
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from typing import Any, Dict

from mu.agent.retry import is_context_overflow_error, parse_overflow_token_counts
from providers.base import Message, MessagePart
from utils.logger import logger
from utils.token_estimator import estimate_tokens


def _reinject_refreshed_summary(session, prompt: str) -> str:
    """Rebuild the layered hierarchy from the ORIGINAL persona base after a
    compaction rewrote the L2 conversation summary.

    Compaction rewrites L2, but the incoming ``prompt`` is already
    fully injected (base persona + L1-L3 layers + loop-appended
    scratchpad/durable-recall/L3B blocks). Re-running
    ``_inject_hierarchical_context`` on that prompt appends a SECOND copy of
    the L1-L3 layers on top of the old ones, so an emergency recovery
    GROWS the prompt it is trying to shrink, and repeated rounds duplicate
    layer content while history was already compacted.

    The first successful layered injection stashes the ORIGINAL
    pre-injection base on ``session._system_prompt_base`` (loop_body sets
    the same attribute every turn start, so it is always current for the
    active turn). Rebuilding from that base produces ONE fresh set of
    L1-L3 layers over the persona, with the per-turn caches intact.

    Callers that append turn-scoped blocks AFTER the injection (durable
    recall, scratchpad snapshot/evictions, L3B subagent context) must
    preserve those themselves: pass the already-injected prompt as
    ``injected_prompt`` and this helper re-attaches exactly the tail that
    followed the injected base, so those blocks appear exactly once.
    Returns ``prompt`` unchanged when the base is unknown or the rebuild
    fails — callers treat this as best-effort and never crash the loop.
    """
    base = getattr(session, "_system_prompt_base", None)
    if not base:
        return prompt
    try:
        rebuilt = session._inject_hierarchical_context(
            base,
            cached_skills=getattr(session, "_turn_skills_block", None),
            cached_context_files=getattr(session, "_turn_context_files_block", None),
        )
    except Exception:
        logger.warning(
            "Emergency re-inject from base failed; keeping existing prompt.",
            exc_info=True,
        )
        return prompt
    if prompt.startswith(rebuilt):
        tail = prompt[len(rebuilt):]
    else:
        tail = ""
        logger.warning(
            "Emergency re-inject: rebuilt prompt is not a prefix of the "
            "incoming prompt; post-injection tail dropped.",
        )
    return rebuilt + tail


def _estimate_messages_tokens(messages) -> int:
    """Cheap token estimate for a list of Message objects."""
    total = 0
    for msg in messages:
        for part in msg.parts:
            if part.text:
                total += estimate_tokens(part.text)
            elif part.tool_result is not None:
                total += estimate_tokens(str(part.tool_result))
            elif part.tool_args is not None:
                total += estimate_tokens(json.dumps(part.tool_args))
    return total


_TOOL_SCHEMA_CACHE: Dict[tuple, int] = {}
_TOOL_SCHEMA_CACHE_LOCK = threading.Lock()
_TOOL_SCHEMA_CACHE_CAP = 8


def _estimate_tools_tokens(tools) -> int:
    """Estimate tool definitions, which are part of every agentic request.

    Round-46 F2: the JSON encoding + full tokenization is memoized by
    tool-set identity (names + schema hash), because the active tool set is
    stable across iterations — without the cache this re-encodes and
    re-tokenizes megabytes of unchanged schemas on EVERY provider call.
    Small LRU (insertion-order evict) keyed by the identity tuple.
    """
    if not tools:
        return 0
    payload = [
        {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
            "parameters": getattr(tool, "parameters", {}) or {},
        }
        for tool in tools
    ]
    identity = (
        tuple(entry["name"] for entry in payload),
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode(
                "utf-8", errors="replace"
            )
        ).hexdigest(),
    )
    with _TOOL_SCHEMA_CACHE_LOCK:
        cached = _TOOL_SCHEMA_CACHE.get(identity)
    if cached is not None:
        return cached
    count = estimate_tokens(json.dumps(payload, sort_keys=True, default=str))
    with _TOOL_SCHEMA_CACHE_LOCK:
        if len(_TOOL_SCHEMA_CACHE) >= _TOOL_SCHEMA_CACHE_CAP and identity not in _TOOL_SCHEMA_CACHE:
            _TOOL_SCHEMA_CACHE.pop(next(iter(_TOOL_SCHEMA_CACHE)), None)
        _TOOL_SCHEMA_CACHE[identity] = count
    return count


def _estimate_request_tokens(
    system_prompt: str, messages, tools=None
) -> Dict[str, Any]:
    """ONE structured estimate of a full provider request (Round-46 F2).

    Returns a manifest ``{"system", "messages", "tools", "total"}`` — the
    single tokenization shared by the preflight budget check, the loop's
    ``request_token_estimate``, and the trace ``build_request_record``.
    Previously these three consumers each re-tokenized the entire system
    prompt + all messages + tool schemas (three full traversals per
    provider call); the trace record additionally re-encoded the tool
    schemas a fourth time.
    """
    system_tokens = estimate_tokens(system_prompt)
    msg_tokens = _estimate_messages_tokens(messages)
    tool_tokens = _estimate_tools_tokens(tools)
    return {
        "system": system_tokens,
        "messages": msg_tokens,
        "tools": tool_tokens,
        "total": system_tokens + msg_tokens + tool_tokens,
    }


def _preflight_context_check(
    session, system_prompt, messages, turn_start_index=None, tools=None
):
    """Emergency-compact history if the assembled prompt exceeds the
    provider's context window.

    The normal compaction pass (``roll_history_summary_to_token_budget``)
    fires *before* the system prompt is finalized — resumption briefings,
    hierarchical context injection, and per-iteration memory/scratchpad
    layers all grow the prompt after that pass.  This pre-flight check
    runs right before the provider call with the *actual* prompt and
    messages, and triggers a second compaction if the total would
    overflow.

    ``turn_start_index`` is forwarded to ``_prepare_runtime_history`` so
    that tool-window compression applies the same pair-grouping logic
    used during normal (non-emergency) calls.  Without it the early-exit
    at ``turn_start_index is None`` skips compression and sends more
    tokens than necessary — which can cascade into repeated emergency
    compaction attempts.

    Returns (system_prompt, messages) — unchanged if within budget,
    or with rebuilt messages after emergency compaction.
    """
    from mu.session.budgets import (
        drift_corrected_context_limit,
        resolve_response_reserve,
        resolve_keep_recent,
        resolve_tool_result_floor,
    )

    # Use the drift-corrected limit so the last-line proactive defense fires
    # at the right point once real-token drift has been learned (see
    # budgets.effective_drift_ratio). Falls back to resolve_context_limit's
    # static safety factor when no drift has been observed yet.
    context_limit = drift_corrected_context_limit(session)
    response_reserve = resolve_response_reserve(session)
    max_prompt = context_limit - response_reserve

    # Round-46 F2: ONE tokenization pass for this request. The manifest is
    # stashed on the session so the loop's request_token_estimate and the
    # trace build_request_record reuse it instead of each re-tokenizing the
    # full prompt + messages + tool schemas (3→1 full traversals).
    estimate = _estimate_request_tokens(system_prompt, messages, tools)
    prompt_tokens = estimate["system"]
    msg_tokens = estimate["messages"]
    tool_tokens = estimate["tools"]
    total = estimate["total"]
    try:
        session._request_estimate_manifest = estimate
    except Exception:
        pass
    # Stash the cl100k estimate of the assembled prompt for the cold-cache
    # drift calibration in the response handler: when Ollama's
    # prompt_eval_count is a strong full-prompt signal (>= half the cl100k
    # estimate), we can learn the real/cl100k drift ratio from it.
    try:
        session._last_prompt_cl100k_est = int(total)
    except Exception:
        pass

    if total <= max_prompt:
        return system_prompt, messages

    overshoot = total - max_prompt
    logger.warning(
        "Pre-flight context check: estimated %d tokens "
        "(limit %d, reserve %d, max_prompt %d) — %d over. "
        "Emergency compaction triggered.",
        total, context_limit, response_reserve, max_prompt, overshoot,
    )
    emergency_budget = max(512, max_prompt - prompt_tokens - tool_tokens)
    base_keep_recent = resolve_keep_recent(session, emergency=True)
    # R3 / FM-8: even emergency compaction must respect the per-turn
    # tool-result floor so tool results just received stay verbatim.
    session.session_manager._tool_result_floor = resolve_tool_result_floor(session)

    # Escalating loop: compact, rebuild, re-estimate. If the first pass
    # doesn't reach the budget (the cl100k estimate under-counts Ollama's
    # real BPE ~2.2x, and a large recent tool result can keep the tail
    # fat), shrink keep_recent and compact again — up to 3 rounds. The
    # reactive overflow recovery in `_generate_with_overflow_recovery`
    # is the estimation-independent backstop if this still isn't enough.
    for round_idx in range(3):
        _before_anchor = int(getattr(session.session_manager, "summary_anchor", 0) or 0)
        try:
            session.session_manager._pending_compaction_kind = "emergency_preflight"
            session.session_manager._pending_compaction_iter = int(
                getattr(session, "_trace_current_iter", 0) or 0
            )
            session.session_manager.roll_history_summary_to_token_budget(
                int(emergency_budget * 0.85),
                keep_recent=max(2, base_keep_recent - round_idx * 2),
                max_passes=6,
                provider=session.provider,
            )
            session._compaction_watermark = len(session.session_manager.history)
            _after_anchor = int(getattr(session.session_manager, "summary_anchor", 0) or 0)
            if _after_anchor > _before_anchor:
                try:
                    from mu.trace.emitter import emit_context_artifact
                    emit_context_artifact(session,
                        iteration=int(getattr(session, "_trace_current_iter", 0) or 0),
                        artifact_id=f"history:{_before_anchor}-{_after_anchor}",
                        state="compacted", reason="hard_context_preflight")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Emergency compaction failed: %s", exc)
            return system_prompt, messages

        recent_history = session._prepare_runtime_history(
            turn_start_index=turn_start_index,
        )
        messages = session._build_messages_from_history(
            recent_history,
            {"role": "system", "parts": []},
        )[:-1]

        new_msg_tokens = _estimate_messages_tokens(messages)
        new_total = prompt_tokens + new_msg_tokens + tool_tokens
        if new_total <= max_prompt:
            logger.info(
                "Emergency compaction round %d complete: messages %d -> %d tokens.",
                round_idx + 1, msg_tokens, new_msg_tokens,
            )
            # Compaction may have rewritten the L2 conversation summary —
            # re-run the layered injection so the provider call sees the
            # refreshed summary. The rebuild starts from the ORIGINAL
            # persona base (stashed at the first successful injection) so
            # the refreshed L1-L3 layers REPLACE the old ones instead of
            # stacking on top of the already-injected prompt, and the
            # loop-appended post-injection tail (durable recall, scratchpad,
            # L3B subagent block) is re-attached exactly once.
            system_prompt = _reinject_refreshed_summary(session, system_prompt)
            return system_prompt, messages
        logger.warning(
            "Emergency compaction round %d: still %d over (est %d vs max_prompt %d); "
            "escalating keep_recent.",
            round_idx + 1, new_total - max_prompt, new_total, max_prompt,
        )

    # Couldn't get under budget in 3 rounds — return the most-compacted
    # state; the provider call may still overflow, at which point reactive
    # overflow recovery catches it. This avoids throwing pre-emptively.
    logger.warning(
        "Emergency compaction exhausted 3 rounds; est %d still over max_prompt %d. "
        "Reactive overflow recovery will backstop the provider call.",
        prompt_tokens + _estimate_messages_tokens(messages), max_prompt,
    )
    # Even when the budget could not be met, compaction may have rewritten
    # the L2 summary — re-run the layered injection so the provider call
    # reflects it. Same base-rebuild as the success path: the refreshed
    # L1-L3 layers replace the previously injected copies instead of
    # stacking on top of them, with the loop-appended post-injection tail
    # preserved exactly once.
    system_prompt = _reinject_refreshed_summary(session, system_prompt)
    return system_prompt, messages


# Per-turn cap on reactive overflow recoveries. Each recovery aggressively
# compacts history (keep_recent=4) and retries the provider call once. A
# later iteration in the same turn can overflow again as tool results
# accumulate — that's a *different* overflowing prompt and deserves its own
# recovery, so this is a count (not a boolean): each overflow gets a
# compaction attempt up to this many times per turn. The cap prevents a
# compact-fail loop when compaction genuinely can't shrink the prompt (e.g.
# a single active-turn tool result larger than the window, which the floor
# protects from degradation).
_MAX_OVERFLOW_RECOVERIES_PER_TURN = 3


def _resolve_real_context_window(session) -> int:
    """The provider's *real* (un-safety-factored) input-context ceiling.

    Reactive overflow recovery targets this number, not the safety-factored
    `resolve_context_limit`, because the wire-level "prompt too long" 400 is
    ground truth about the real window — the safety factor is only a
    pre-flight hint. Falls back to the factored limit + a default if the
    provider doesn't expose one.
    """
    try:
        window = session.provider.effective_context_window(
            session.provider.model_name
        )
    except Exception:
        window = None
    if window and window > 0:
        return int(window)
    from mu.session.budgets import resolve_context_limit
    return int(resolve_context_limit(session) or 200_000)


def _overflow_drift_ratio(session, system_prompt, messages, overflow_error) -> float:
    """Real-tokens / cl100k-tokens for the prompt that just overflowed.

    Ground truth from the error (the daemon's own BPE count of the prompt)
    divided by the harness's cl100k estimate of the same prompt. Clamped to
    a sane [1.0, 6.0] band so a mis-parsed error can't blow up the budget.
    Falls back to the provider's static `compaction_safety_factor` (2.5 for
    Ollama, 1.0 otherwise) when the error body doesn't carry real counts.
    """
    cl100k_full = estimate_tokens(system_prompt) + _estimate_messages_tokens(messages)
    real_prompt, _real_max = parse_overflow_token_counts(overflow_error)
    if real_prompt and cl100k_full and cl100k_full > 0:
        drift = real_prompt / cl100k_full
        # Clamp: drift < 1.0 is impossible (cl100k never over-counts Ollama by
        # that much) and means a parse error; > 6.0 is pathological content.
        return max(1.0, min(6.0, drift))
    try:
        factor = float(session.provider.compaction_safety_factor())
    except Exception:
        factor = 1.0
    return max(1.0, factor or 1.0)


def _calibrate_drift_from_response(session, response) -> None:
    """EWMA-update ``session._observed_drift_ratio`` from a cold-cache
    provider response, if the response's ``input_tokens`` is a strong
    full-prompt signal.

    For Ollama, ``response.input_tokens`` is the streamed
    ``prompt_eval_count`` — normally only the non-cached delta (near-zero in
    a warm loop, useless for drift). But on a cold cache (first call of a
    session, or after the prompt changed substantially) it reflects (close
    to) the FULL prompt, so ``real_tokens / cl100k_tokens`` is learnable from
    it. The guard rejects the warm-cache near-zero delta: only calibrate when
    the reported count is >= half the stashed cl100k estimate AND > 500
    tokens AND that estimate is itself substantial (> 1000). Never raises.
    """
    try:
        cl100k_est = int(getattr(session, "_last_prompt_cl100k_est", 0) or 0)
        reported_in = int(getattr(response, "input_tokens", 0) or 0)
        if cl100k_est > 1000 and reported_in > 500 and reported_in >= cl100k_est // 2:
            from mu.session.budgets import update_observed_drift

            update_observed_drift(session, reported_in / float(cl100k_est))
    except Exception:  # noqa: BLE001 — telemetry must not break the loop
        pass


def _aggressive_compact_for_overflow(
    session, system_prompt, messages, *,
    overflow_error=None,
    tools=None,
    keep_recent: int = 4,
    margin: float = 0.20,
    lift_floor: bool = False,
):
    """Claude Code Tier 5-style reactive compaction: keep only the last
    ``keep_recent`` messages verbatim and summarize everything older into
    the rolling summary, then degrade oversized payloads left in the tail.

    Estimation-independent: the budget is derived from the wire-level
    overflow's ground-truth real token count (parsed from the error) paired
    with the harness cl100k estimate of the same prompt — giving the real
    per-content drift ratio — so the retry targets a *real* token count the
    daemon will accept. The static `compaction_safety_factor` alone can't:
    drift varies ~2.2–3.2x by content, so targeting "half the safety-factored
    limit" in cl100k still overflows on the wire when drift outruns the
    factor.

    Escalation is driven by the caller (`_generate_with_overflow_recovery`),
    which raises the aggression per still-overflowing retry:
      * ``keep_recent`` shrinks (4 → 2 → 1) so more history is summarized,
      * ``margin`` grows (0.20 → 0.30 → 0.40) so the target real-token budget
        drops further below the window,
      * ``lift_floor`` (from the 2nd attempt on) forces the per-turn
        tool-result floor to 0 so the protected recent tool results that
        alone exceed the budget can be degraded — the lesser evil vs
        crashing the turn with a hard overflow. The floor is restored in
        the `finally` so later in-turn compaction keeps FM-8 protection.
    """
    from mu.session.budgets import (
        resolve_response_reserve,
        resolve_tool_result_floor,
        update_observed_drift,
    )

    real_max = _resolve_real_context_window(session)
    response_reserve = resolve_response_reserve(session)
    drift = _overflow_drift_ratio(session, system_prompt, messages, overflow_error)
    # Persist the measured real/cl100k drift so the NEXT turn's proactive
    # compaction gates (turn-start roll, auto-hook, preflight) fire at the
    # right point instead of relying on the static safety factor. This is the
    # fix for the repeat overflow: after one 400, every subsequent turn sees
    # the learned drift via budgets.effective_drift_ratio.
    update_observed_drift(session, drift)

    # Target (1 - margin) of the real window so drift variance across the
    # compacted content + tool schemas (not in the cl100k estimate) + the
    # response reserve all fit under the wire limit. margin escalates with
    # each recovery attempt (0.20 → 0.30 → 0.40).
    target_real = max(1024, int((real_max - response_reserve) * (1.0 - margin)))
    target_full_cl100k = max(512, int(target_real / drift))
    # The system prompt already carries L1–L4 (inject_hierarchical_context),
    # so the non-history cl100k cost is just the system prompt; the rest of
    # the budget is the L5 history ceiling the compaction loop gates on.
    non_history_cl100k = estimate_tokens(system_prompt) + _estimate_tools_tokens(tools)
    budget = max(512, target_full_cl100k - non_history_cl100k)

    logger.warning(
        "Reactive overflow compaction: real_max=%d reserve=%d drift=%.2f "
        "margin=%.2f keep_recent=%d lift_floor=%s target_real=%d "
        "target_full_cl100k=%d non_history=%d L5_budget=%d (cl100k_full~%d).",
        real_max, response_reserve, drift, margin, keep_recent, lift_floor,
        target_real, target_full_cl100k, non_history_cl100k, budget,
        non_history_cl100k + _estimate_messages_tokens(messages),
    )

    floor_value = resolve_tool_result_floor(session)
    # lift_floor: let `_degrade_oldest_runtime_payload` reach the protected
    # recent tool results that alone exceed the budget. Restored in `finally`.
    session.session_manager._tool_result_floor = 0 if lift_floor else floor_value
    if lift_floor:
        session.session_manager._pending_compaction_kind = (
            "reactive_overflow_floor_lift"
        )
    else:
        session.session_manager._pending_compaction_kind = "reactive_overflow"
    try:
        session.session_manager._pending_compaction_iter = int(
            getattr(session, "_trace_current_iter", 0) or 0
        )
        session.session_manager._compact_focus = (
            session.variables.get("compact_focus") or ""
        )
        # max_passes is generous (12) so a tight drift-corrected budget is
        # actually reached — each pass rolls a summary segment OR degrades
        # one oversized payload, and a fat recent tail needs several
        # degradation passes to get under.
        session.session_manager.roll_history_summary_to_token_budget(
            budget, keep_recent=keep_recent, max_passes=12,
            provider=session.provider,
        )

        session._compaction_watermark = len(session.session_manager.history)
    except Exception as exc:
        logger.warning("Aggressive overflow compaction failed: %s", exc)
    finally:
        # Restore the floor so later in-turn compaction keeps FM-8 protection.
        session.session_manager._tool_result_floor = floor_value


def _generate_with_overflow_recovery(
    session, *, messages, system_prompt, thinking, tools, turn_start_index,
):
    """Provider generate with reactive context-overflow recovery.

    Mirrors Claude Code's Tier 5 reactive compaction: if the provider
    rejects the prompt as too long (a non-transient 400/413), instead of
    surfacing a hard error we aggressively compact history, rebuild the
    prompt, and retry — and if the retry *still* overflows, compact harder
    and retry again, up to `_MAX_OVERFLOW_RECOVERIES_PER_TURN` times per
    turn. The retry lives inside the same try (it's the next loop
    iteration), so a too-generous compaction budget that overflows on the
    retry is caught and re-compacted instead of surfacing as "API Error
    during agentic loop" (the old code returned the retry outside the try).

    Escalation per still-overflowing attempt: shrink `keep_recent`
    (4 → 2 → 1), grow the drift margin (0.20 → 0.30 → 0.40), and from the
    2nd recovery on lift the per-turn tool-result floor so protected
    recent tool results that alone exceed the budget can be degraded. A
    per-turn count (`_overflow_recoveries_this_turn`, capped at
    `_MAX_OVERFLOW_RECOVERIES_PER_TURN`) is the circuit breaker — it's a
    count, not a boolean, so a *later* iteration that overflows again still
    gets its own recovery.

    This is the estimation-independent backstop behind the cl100k-based
    pre-flight check: when the cheap token estimate under-counts the
    model's real BPE (Ollama drifts ~2.2x), the wire-level 400 is the
    ground truth, and we recover from it rather than crashing the turn.
    The compaction budget is drift-corrected from that ground truth (see
    `_aggressive_compact_for_overflow`), so each retry targets a *real*
    token count the daemon accepts.
    """
    # Escalation ladder: each still-overflowing attempt compacts harder
    # before retrying. The retry is the next loop iteration — it sits
    # INSIDE the try, so a retry that still overflows is caught and
    # re-compacted instead of surfacing as "API Error during agentic loop"
    # (the old code returned the retry outside the try, so a too-generous
    # compaction budget blew context on the retry with no recovery).
    _KEEP_RECENT_LADDER = (4, 2, 1)
    _MARGIN_LADDER = (0.20, 0.30, 0.40)

    first_attempt = True
    while True:
        try:
            return session._provider_generate_with_retry(
                messages=messages,
                system_prompt=system_prompt,
                thinking=thinking,
                tools=tools,
            )
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            recoveries = int(getattr(session, "_overflow_recoveries_this_turn", 0) or 0)
            if recoveries >= _MAX_OVERFLOW_RECOVERIES_PER_TURN:
                logger.error(
                    "Context overflow persisted after %d reactive compactions this "
                    "turn; re-raising (circuit breaker): %s",
                    recoveries, str(exc)[:300],
                )
                raise
            session._overflow_recoveries_this_turn = recoveries + 1
            # Escalate per recovery: shrink keep_recent, grow the margin,
            # and (from the 2nd recovery on) lift the tool-result floor so
            # protected recent tool results that alone exceed the budget
            # can be degraded. Clamp the ladder index to its last rung.
            level = min(recoveries, len(_KEEP_RECENT_LADDER) - 1)
            keep_recent = _KEEP_RECENT_LADDER[level]
            margin = _MARGIN_LADDER[min(level, len(_MARGIN_LADDER) - 1)]
            lift_floor = level >= 1
            if first_attempt or session.ui:
                if session.ui:
                    session.ui.show_info(
                        "Context overflow from the model — compacting older "
                        "history into a summary and retrying (no data lost)."
                    )
                first_attempt = False
            logger.warning(
                "Reactive overflow recovery #%d: provider rejected prompt as "
                "too long. Compacting (keep_recent=%d, margin=%.2f, lift_floor=%s) "
                "and retrying. Error: %s",
                recoveries + 1, keep_recent, margin, lift_floor, str(exc)[:300],
            )
            # Late-bind through loop_body so monkeypatching
            # `loop_body._aggressive_compact_for_overflow` (the documented
            # public seam, used by tests) intercepts this call even though
            # the implementation moved here.
            _lb = sys.modules.get("mu.agent.loop_body")
            _compact_fn = getattr(_lb, "_aggressive_compact_for_overflow", None) or _aggressive_compact_for_overflow
            _compact_fn(
                session, system_prompt, messages, overflow_error=exc,
                keep_recent=keep_recent, margin=margin, lift_floor=lift_floor,
                tools=tools,
            )
            recent_history = session._prepare_runtime_history(
                turn_start_index=turn_start_index,
            )
            messages = session._build_messages_from_history(
                recent_history,
                {"role": "system", "parts": []},
            )[:-1]
            # Final pre-flight (re-checks + may compact more) before the retry.
            system_prompt, messages = _preflight_context_check(
                session, system_prompt, messages, turn_start_index=turn_start_index,
                tools=tools,
            )


from mu.agent.teacher_watcher import (  # noqa: F401
    _active_teacher_lesson,
    _run_teacher_watcher_user,
    _run_teacher_watcher_assistant,
    _persist_teacher_course,
    _render_learner_profile_block,
)
