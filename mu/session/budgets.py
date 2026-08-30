"""Token-budget arithmetic for the session compactor.

The three helpers here decide how aggressively the conversation
history gets compacted before each provider call:

  * `resolve_context_limit(session)`      — total token ceiling, min of
                                            user-set + provider window.
  * `resolve_response_reserve(session)`   — tokens to leave free for
                                            the model's reply.
  * `compaction_token_budget(session)`    — L5 (history) budget,
                                            = (ceiling − response reserve
                                              − non-L5 layers) × trim threshold.

The Ollama "prompt too long; exceeded max context length" bug that
drove `resolve_context_limit` is regression-pinned in
`tests/test_context_budget.py`. End-to-end coverage in
`tests/test_compaction_e2e.py`.

These functions take a `session` argument because they need access to
`session.provider` (for `effective_context_window` /
`effective_response_reserve`) and `session.variables` (for the
user-configurable knobs). They don't mutate the session.
"""

from __future__ import annotations

from typing import Any

from utils.config import _DEFAULT_CONTEXT_TOKEN_LIMIT


# ── Compaction keep-recent policy (R3, FM-8) ─────────────────────────────
# Single source of truth for how many trailing messages the compactor
# keeps verbatim. Previously these were inline magic numbers scattered
# across the call sites (12 / 12 / 2), which made the relationship
# between normal roll, auto-compaction, and emergency compaction opaque.
KEEP_RECENT_DEFAULT = 12
KEEP_RECENT_EMERGENCY = 2
# Per-turn tool-result floor: the last K tool-result messages of the
# active turn are never summarized or degraded, even under emergency
# compaction with a tiny keep_recent. Prevents the "compaction mid-turn
# drops tool results just received" failure mode (FM-8) at the cost of
# slightly higher steady-state token usage.
TOOL_RESULT_FLOOR_DEFAULT = 4


def resolve_keep_recent(session: Any, *, emergency: bool = False) -> int:
    """How many trailing messages the compactor keeps verbatim.

    Normal/auto compaction uses `compactor_keep_recent` (default
    `KEEP_RECENT_DEFAULT`). Emergency compaction (pre-flight context
    check) uses `emergency_keep_recent` (default `KEEP_RECENT_EMERGENCY`)
    — smaller so it can reclaim budget fast, but the tool-result floor
    (`resolve_tool_result_floor`) still protects recent tool results
    regardless of this value.
    """
    if emergency:
        raw = session.variables.get("emergency_keep_recent", KEEP_RECENT_EMERGENCY)
    else:
        raw = session.variables.get("compactor_keep_recent", KEEP_RECENT_DEFAULT)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return KEEP_RECENT_EMERGENCY if emergency else KEEP_RECENT_DEFAULT


def resolve_tool_result_floor(session: Any) -> int:
    """Number of trailing tool-result messages in the active turn that
    compaction must leave verbatim (R3, FM-8).

    Mode-aware (Fix #10): long-horizon modes (loop/feature) re-cover many
    files, so a larger floor keeps more recent read results verbatim in L5
    instead of compacting them away and forcing re-reads. Only raises the
    configured value — a user's explicit higher setting always wins.
    """
    raw = session.variables.get("tool_result_floor", TOOL_RESULT_FLOOR_DEFAULT)
    try:
        floor = max(0, int(raw))
    except (TypeError, ValueError):
        floor = TOOL_RESULT_FLOOR_DEFAULT
    mode = str(session.variables.get("agent_mode", "default") or "default").lower()
    if mode in ("loop", "feature"):
        floor = max(floor, 8)
    return floor


# Default tool-result cache bounds (Fix #10). Raised in long-horizon modes
# so more on-disk reads stay recallable instead of being evicted under the
# small default 50-entry / 512KB cap.
TOOL_CACHE_ENTRIES_DEFAULT = 50
TOOL_CACHE_BYTES_DEFAULT = 524_288  # 512 KB


def resolve_tool_cache_bounds(session: Any) -> tuple[int, int]:
    """(max_entries, max_bytes) for the tool-result sidecar cache.

    Mode-aware (Fix #10): loop/feature modes do far more read-only tool
    calls, so the cache is grown to keep more results recallable (and
    auto-recallable by locator). Only raises the configured bounds — a
    user's explicit larger setting always wins.
    """
    try:
        entries = max(1, int(
            session.variables.get("tool_result_cache_entries", TOOL_CACHE_ENTRIES_DEFAULT)
        ))
    except (TypeError, ValueError):
        entries = TOOL_CACHE_ENTRIES_DEFAULT
    try:
        nbytes = max(1, int(
            session.variables.get("tool_result_cache_bytes", TOOL_CACHE_BYTES_DEFAULT)
        ))
    except (TypeError, ValueError):
        nbytes = TOOL_CACHE_BYTES_DEFAULT
    mode = str(session.variables.get("agent_mode", "default") or "default").lower()
    if mode in ("loop", "feature"):
        entries = max(entries, 256)
        nbytes = max(nbytes, 2_097_152)  # 2 MB
    return entries, nbytes


def resolve_context_limit(session: Any) -> int:
    """Pick the smaller of (user-set `context_token_limit`, real
    provider window). Ollama models often have 4k–32k real windows
    while the user-set default is 256k, so without this the compactor
    never fires before the provider 400s with "prompt too long".
    """
    user_limit = max(
        1024,
        int(
            session.variables.get(
                "context_token_limit", _DEFAULT_CONTEXT_TOKEN_LIMIT
            )
            or _DEFAULT_CONTEXT_TOKEN_LIMIT
        ),
    )
    try:
        provider_window = session.provider.effective_context_window(
            session.provider.model_name
        )
    except Exception:
        provider_window = None
    if provider_window and provider_window > 0:
        limit = min(user_limit, int(provider_window))
    else:
        limit = user_limit
    # Apply a provider-aware safety factor so the compactor targets a reduced
    # ceiling for providers whose real tokenizer diverges from the harness's
    # cl100k_base estimate (notably Ollama, where cl100k under-counts ~2x and
    # streamed prompt_eval_count is only the non-cached delta). Without this
    # the real prompt overflows the window before the cl100k-based compaction
    # guard fires — the Ollama "prompt is too long" 400. 1.0 = trust cl100k.
    try:
        factor = float(session.provider.compaction_safety_factor())
    except Exception:
        factor = 1.0
    if factor > 1.0:
        limit = max(1024, int(limit / factor))
    return limit


def _static_safety_factor(session: Any) -> float:
    """The provider's static compaction safety factor, normalised to >=1.0
    (1.0 = trust the cl100k estimate verbatim)."""
    try:
        factor = float(session.provider.compaction_safety_factor())
    except Exception:
        factor = 1.0
    return factor if factor > 1.0 else 1.0


def effective_drift_ratio(session: Any) -> float:
    """The cl100k→real-token drift multiplier the compactor should assume.

    The provider's static ``compaction_safety_factor`` (baked into
    ``resolve_context_limit``) is a *seed*: it is returned only while no
    reliable drift observation has been recorded, so a cold session stays
    conservative (e.g. Ollama's 2.5x until proven otherwise). Once a
    measurement arrives — from an overflow 400's ground-truth real token
    count or a cold-cache ``prompt_eval_count`` — the *learned* ratio wins
    in both directions: it ratchets UP when the real drift is worse than
    the seed, and DOWN (to a floor of 1.0) when cl100k over-counts.

    The previous implementation floored at the static factor, which made the
    2.5x Ollama assumption permanent: a session whose real drift was ~0.83x
    (cl100k over-counting, evidenced by a consistently negative
    ``drift_pct``) still reported ``real_est = cl100k * 2.5`` and a Memory
    Map fill% of ~96%, even though the true fill was ~38%. The reactive
    overflow backstop + EWMA smoothing protect against a single spurious
    low reading making the proactive gates too lax.
    """
    static = _static_safety_factor(session)
    learned = getattr(session, "_observed_drift_ratio", None)
    if learned is None:
        return static
    try:
        learned = float(learned)
    except (TypeError, ValueError):
        return static
    return max(1.0, learned)


def drift_corrected_context_limit(session: Any) -> int:
    """The real-token context ceiling the compactor and fill% should use.

    ``resolve_context_limit`` divides the raw window by the static safety
    factor (2.5 for Ollama) as a conservative default. This replaces that
    static divisor with the *learned* effective drift ratio in both
    directions:

    * learned drift > static → the real tokenizer diverges worse than the
      seed assumed, so the ceiling shrinks (``limit * static / eff``).
    * learned drift < static → cl100k over-counts for this content, so the
      ceiling grows back toward the raw window. Previously the ceiling was
      pinned at ``raw / static`` forever, which made the Memory Map report
      ~96% full for a session whose true fill was ~38%.
    * no measurement yet → ``effective_drift_ratio`` returns the static
      factor, so ``limit * static / eff == limit`` (no change; stays
      conservative until a reliable reading lands).

    For providers with no safety factor (OpenAI/Gemini) this is a no-op.
    """
    static = _static_safety_factor(session)
    if static <= 1.0:
        return resolve_context_limit(session)
    # Derive the base from the RAW window, not the user-set token limit:
    # the whole point of drift correction is to retire the static safety
    # factor when measurement says the tokenizer tracks real tokens, so the
    # corrected ceiling recovers the provider's physical window (bounded by
    # it) rather than an arbitrarily smaller user default.
    try:
        raw_window = int(session.provider.effective_context_window())
    except Exception:
        raw_window = None
    if raw_window and raw_window > 0:
        base = max(1024, int(raw_window / static))
    else:
        base = resolve_context_limit(session)
    eff = effective_drift_ratio(session)
    if eff <= 0:
        return resolve_context_limit(session)
    corrected = int(base * static / eff)
    if raw_window and raw_window > 0:
        corrected = min(corrected, raw_window)
    return max(1024, corrected)


def update_observed_drift(session: Any, observed: float) -> None:
    """EWMA-smooth a real-token drift observation onto the session.

    ``observed`` is ``real_tokens / cl100k_tokens`` (clamped to [0.5, 6.0]).
    Values below 1.0 mean cl100k over-counts for this content (the provider's
    real tokenizer produces fewer tokens than cl100k estimates); capturing
    them lets ``effective_drift_ratio`` correct the previously-permanent 2.5x
    Ollama floor downward. Weight 0.5 blends a new observation with the prior
    so a single outlier doesn't whip the ratio, while still tracking a genuine
    shift in content type. The proactive compaction gates read this via
    ``effective_drift_ratio``; the reactive overflow path and the cold-cache
    response calibration write it.
    """
    try:
        observed = max(0.5, min(6.0, float(observed)))
    except (TypeError, ValueError):
        return
    prev = getattr(session, "_observed_drift_ratio", None)
    if prev is None:
        session._observed_drift_ratio = observed
    else:
        try:
            prev = float(prev)
        except (TypeError, ValueError):
            session._observed_drift_ratio = observed
            return
        session._observed_drift_ratio = 0.5 * prev + 0.5 * observed


def resolve_response_reserve(session: Any) -> int:
    """How many tokens to leave free for the model's output.

    Preferred source is the provider's `effective_response_reserve()`
    — which reads `ollama_num_predict` / `max_tokens` / etc. — so the
    reserve tracks the actual configured output cap instead of a
    guessed constant. Only falls back to the `response_token_reserve`
    session variable when the provider has no configured cap.
    """
    try:
        provider_reserve = session.provider.effective_response_reserve(
            session.provider.model_name
        )
    except Exception:
        provider_reserve = None
    if provider_reserve and provider_reserve > 0:
        return int(provider_reserve)
    raw = session.variables.get("response_token_reserve", 4096)
    try:
        return max(0, int(raw)) if raw is not None else 4096
    except (TypeError, ValueError):
        return 4096


def compaction_token_budget(session: Any) -> int:
    """The token ceiling the compactor targets for L5 (conversation
    history) specifically.

    The global cap (`context_token_limit`, or the provider's actual
    window when smaller) covers all 7 prompt layers PLUS the model's
    response reserve. L5 gets whatever the cap minus the non-L5
    layers (workspace files, skills, summary, goal context, recent
    tool activity, retrieval snippets) leaves room for, with the
    trim threshold applied to that residual.

    Computing the non-L5 layer tokens here means a heavy AGENTS.md or
    many auto-expanded skills tighten the compactor's threshold
    instead of being silently piled on top of the L5 budget.
    """
    context_limit = drift_corrected_context_limit(session)
    trim_threshold = float(
        session.variables.get("context_trim_threshold", 0.85) or 0.85
    )
    trim_threshold = max(0.10, min(trim_threshold, 1.0))
    response_reserve = resolve_response_reserve(session)

    non_l5_tokens = 0
    try:
        from utils.runtime_metrics import estimate_non_l5_context_tokens

        non_l5_tokens = int(estimate_non_l5_context_tokens(session) or 0)
    except Exception:
        non_l5_tokens = 0

    usable = context_limit - response_reserve - non_l5_tokens
    if usable <= 0:
        # No real capacity for history (codex round-7 F6): the old
        # max(1024, ...) floor returned tokens that don't exist, letting
        # compaction declare history "within budget" while the assembled
        # request was already over the provider window — producing
        # avoidable overflow/retry cycles. Zero means callers must
        # shrink non-L5 layers / reserve, or report the fixed prompt
        # cannot fit.
        return 0
    return max(512, int(usable * trim_threshold))


__all__ = [
    "resolve_context_limit",
    "drift_corrected_context_limit",
    "effective_drift_ratio",
    "update_observed_drift",
    "resolve_response_reserve",
    "compaction_token_budget",
    "resolve_keep_recent",
    "resolve_tool_result_floor",
    "resolve_tool_cache_bounds",
    "KEEP_RECENT_DEFAULT",
    "KEEP_RECENT_EMERGENCY",
    "TOOL_RESULT_FLOOR_DEFAULT",
    "TOOL_CACHE_ENTRIES_DEFAULT",
    "TOOL_CACHE_BYTES_DEFAULT",
]
