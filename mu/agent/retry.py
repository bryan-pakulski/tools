"""Provider-call retry with exponential backoff.

`provider_generate_with_retry(session, ...)` wraps a provider.stream()
call with retry-on-transient-error semantics. Transient errors are
classified by message-string heuristics and known HTTP status codes;
the loop is bounded by a cumulative-wait budget (default 120s) plus a
hard max-attempts ceiling (default 30) as a safety belt.

The retry loop also drives the `pre_provider_call` / `post_provider_call`
hook points and the `_HookAbort` exception that lets a hook stop the
turn before the provider is contacted.

Backoff with the defaults (base=0.4, max=30, budget=120):
    attempt 1: ~0.4s   (total ~0.4s)
    attempt 2: ~0.8s   (total ~1.2s)
    attempt 3: ~1.6s   (total ~2.8s)
    attempt 4: ~3.2s   (total ~6.0s)
    attempt 5: ~6.4s   (total ~12.4s)
    attempt 6: ~12.8s  (total ~25.2s)
    attempt 7: ~25.6s  (total ~50.8s)
    attempt 8+: 30s capped
    stops once cumulative >= 120s.

Tests: `tests/test_provider_retry.py` (5 regression pins).
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Optional

from .hooks import HookContext, default_registry


# ---------------------------------------------------------------- error classification


_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "temporary failure",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
    "connection reset",
    "connection aborted",
    "network",
    "econnreset",
    "ssl",
    "unexpected eof",
    "eof occurred",
    "broken pipe",
    "protocol error",
    "service unavailable",
    "try again",
    "overloaded",
    "capacity",
    "server error",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "server is",
)


_RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


# Context-overflow is NOT transient (the same prompt will fail identically),
# so it must not be retried by the transient loop. It is handled separately
# by reactive overflow recovery (compact-and-retry once) in `loop_body`.
_OVERFLOW_MARKERS = (
    "prompt too long",
    "prompt is too long",
    "maximum context length",
    "context length",
    "context window",
    "context overflow",
    "maximum context",
    "request entity too large",
    "request too large",
    "exceeded the context",
    "exceeds the context",
    "exceeded your available",
    "context_length_exceeded",
    "maximum_input_tokens",
)
# HTTP status codes that, when paired with a prompt/context marker, signal
# overflow rather than a generic client error.
_OVERFLOW_HTTP_STATUS = frozenset({400, 413})


def is_context_overflow_error(error: Exception) -> bool:
    """Detect a 'prompt too long' / context-window-exceeded provider error.

    These are non-transient — resending the same prompt fails identically —
    but they ARE recoverable by compacting history and retrying, so the
    agentic loop reacts to them (Claude Code Tier 5: keep last ~4 messages,
    summarize the rest, retry once) rather than surfacing a hard error.

    Matches on the error message and any chained ``__cause__`` so the real
    Ollama body (``"The prompt is too long: N, model maximum context length:
    M"``) is caught even when wrapped by a transport layer.
    """
    blob = str(error or "")
    cause = getattr(error, "__cause__", None)
    if cause is not None and cause is not error:
        blob = f"{blob}\n{cause}"
    # An `actionable` attribute (OllamaError) often carries the classified
    # phrasing ("Ollama context overflow for ...").
    actionable = getattr(error, "actionable", None)
    if actionable:
        blob = f"{blob}\n{actionable}"
    lowered = blob.lower()
    if any(marker in lowered for marker in _OVERFLOW_MARKERS):
        return True
    # Bare 413 ("Request Entity Too Large") is overflow by definition.
    status = extract_http_status_code(lowered)
    if status is not None and status in _OVERFLOW_HTTP_STATUS:
        # 400 is ambiguous — only treat as overflow if a context/prompt
        # marker is present (already covered above). 413 alone qualifies.
        if status == 413:
            return True
    return False


# Real prompt token count + the model's real maximum, parsed out of an
# overflow error body. Ollama's wording is:
#   "The prompt is too long: 1068887, model maximum context length: 1000000"
# This is ground truth — the daemon's own BPE count of the prompt that just
# overflowed — and is the key to estimation-independent reactive recovery:
# paired with the harness's cl100k estimate of the same prompt it gives the
# real per-content drift ratio, so the retry can target a budget that maps
# to a *real* token count the daemon will accept (the static safety factor
# alone can't, because drift varies ~2.2–3.2x by content).
_OVERFLOW_PROMPT_RE = re.compile(r"too long[^0-9]{0,12}(\d{4,})", re.IGNORECASE)
_OVERFLOW_MAX_RE = re.compile(
    r"maximum context length[^0-9]{0,12}(\d{4,})", re.IGNORECASE
)


def parse_overflow_token_counts(error: Exception) -> tuple:
    """Extract ``(real_prompt_tokens, real_max_tokens)`` from an overflow
    error body, or ``(None, None)`` if the counts aren't present.

    Both numbers are optional and independent — Ollama emits both, but other
    providers (or future Ollama wordings) may emit only one. ``real_prompt``
    is the daemon's count of the prompt that overflowed; ``real_max`` is the
    model's context ceiling.
    """
    blob = str(error or "")
    cause = getattr(error, "__cause__", None)
    if cause is not None and cause is not error:
        blob = f"{blob}\n{cause}"
    actionable = getattr(error, "actionable", None)
    if actionable:
        blob = f"{blob}\n{actionable}"
    real_prompt = None
    real_max = None
    m = _OVERFLOW_PROMPT_RE.search(blob)
    if m:
        try:
            real_prompt = int(m.group(1))
        except (TypeError, ValueError):
            real_prompt = None
    m = _OVERFLOW_MAX_RE.search(blob)
    if m:
        try:
            real_max = int(m.group(1))
        except (TypeError, ValueError):
            real_max = None
    return real_prompt, real_max


def is_transient_provider_error(error: Exception) -> bool:
    """Classify whether `error` is worth retrying. String-match against
    known transient markers, then fall back to extracting an HTTP status
    code from the message."""
    message = str(error or "").lower()
    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return True
    status = extract_http_status_code(message)
    if status is not None:
        return status in _RETRYABLE_HTTP_STATUS
    return False


def extract_http_status_code(message: str) -> Optional[int]:
    """Pull a 3-digit HTTP status code out of a provider error message.

    Patterns matched in order: `HTTP Error: 503`, `status_code=429`,
    bare `503`. Returns None if no plausible status is found."""
    patterns = (
        r"http error[: ]+(\d{3})",
        r"status_code[=: ]+(\d{3})",
        r"\b(?:http\s*)?(\d{3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        try:
            code = int(match.group(1))
            if 100 <= code <= 599:
                return code
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------- retry loop


def provider_generate_with_retry(
    session: Any,
    *,
    messages,
    system_prompt,
    thinking,
    tools,
):
    """Call `session.provider.stream(...)` with retry on transient
    errors. Fires `pre_provider_call` / `post_provider_call` hooks and
    honors `_HookAbort`.

    Returns the drained `ProviderResponse`. Re-raises any non-transient
    exception, and re-raises the last transient exception once the
    budget or attempt cap is exhausted.
    """
    # Lazy import to keep this module cheap at import time. Built-in
    # hooks (compactor, plan_mode, secret_guard, usage_tracker) are
    # auto-registered at `mu.agent` package load — no need to import
    # them here.
    from mu.ui.stream import build_default_renderer
    from mu.session.helpers import _HookAbort

    base_delay = float(
        session.variables.get("provider_retry_base_delay", 0.4) or 0.4
    )
    max_delay = float(
        session.variables.get("provider_retry_max_delay", 30.0) or 30.0
    )
    total_budget_s = float(
        session.variables.get("provider_retry_max_total_wait_seconds", 120.0)
        or 120.0
    )
    max_attempts = max(
        1, int(session.variables.get("provider_max_retries", 30) or 30)
    )

    attempt = 0
    elapsed = 0.0
    # Round-15 F13: baseline so we can detect a pre_provider_call hook
    # (auto-compactor crossing its threshold) that rolled/compacted
    # session_manager.history AFTER the caller assembled `messages` from
    # it — the in-memory list would otherwise go to the wire stale,
    # re-sending already-summarized content verbatim and defeating the
    # compaction for this call.
    # Round-17 F24: length alone is NOT a sufficient trigger — the rolling
    # compaction path (roll_history_summary_to_token_budget) advances
    # summary_anchor / conversation_summary WITHOUT shrinking the history
    # list, so a length-only check missed that compaction entirely.
    # Snapshot the compaction-visible triple instead.
    hist_len_before_hooks = len(session.session_manager.history)
    anchor_before_hooks = getattr(
        session.session_manager, "summary_anchor", 0
    )

    while True:
        try:
            pre_ctx = HookContext(
                point="pre_provider_call",
                session=session,
                variables=session.variables,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            )
            _, _, abort = default_registry.fire_with_signals(
                "pre_provider_call", pre_ctx
            )
            if abort is not None:
                session._record_hook_abort("pre_provider_call", abort)
                raise _HookAbort(session._hook_abort_reason)

            if (
                len(session.session_manager.history) != hist_len_before_hooks
                or getattr(
                    session.session_manager, "summary_anchor", 0
                ) != anchor_before_hooks
            ):
                # History changed under us (hook compaction or append) —
                # rebuild the wire messages from live history. Mirrors the
                # caller's build (runtime slice + trailing sentinel
                # stripped); on any fake/legacy session lacking the
                # builder, keep the original messages.
                try:
                    _live_recent = session._prepare_runtime_history(
                        turn_start_index=getattr(
                            session, "_current_turn_start_index", None
                        )
                    )
                    messages = session._build_messages_from_history(
                        _live_recent,
                        {"role": "system", "parts": []},
                    )[:-1]
                except AttributeError:
                    pass

            renderer = build_default_renderer(session.ui)
            events = session.provider.stream(
                messages=messages,
                system_prompt=system_prompt,
                thinking=thinking,
                tools=tools,
            )
            response = renderer.consume(session.provider, events)
            post_ctx = HookContext(
                point="post_provider_call",
                session=session,
                variables=session.variables,
                messages=messages,
                system_prompt=system_prompt,
                response=response,
            )
            _, _, abort = default_registry.fire_with_signals(
                "post_provider_call", post_ctx
            )
            if abort is not None:
                session._record_hook_abort("post_provider_call", abort)
            return response
        except Exception as exc:
            # Delegate the transient-error classification to the session.
            # Tests monkeypatch `Session._is_transient_provider_error`
            # to inject test-specific transient/non-transient policies,
            # so we must consult it via the session rather than calling
            # the module-level helper directly.
            classify = getattr(
                session, "_is_transient_provider_error", None
            ) or is_transient_provider_error
            if not classify(exc):
                raise
            if elapsed >= total_budget_s or attempt >= max_attempts:
                # Budget exhausted — bubble up so the outer turn loop
                # can surface a clear failure instead of stalling forever.
                if session.ui:
                    session.ui.show_error(
                        f"Provider retry budget exhausted "
                        f"({attempt} attempts, {elapsed:.1f}s slept). Aborting."
                    )
                raise
            attempt += 1
            # Exponential backoff with jitter, capped at max_delay.
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, min(1.0, delay * 0.25))
            # Don't oversleep past the remaining budget.
            remaining = max(0.0, total_budget_s - elapsed)
            delay = max(0.05, min(delay, remaining))
            if session.ui:
                session.ui.show_info(
                    f"Transient provider error; retry {attempt} "
                    f"in {delay:.1f}s ({elapsed:.1f}s of "
                    f"{total_budget_s:.0f}s budget used)."
                )
            time.sleep(delay)
            elapsed += delay


__all__ = [
    "is_transient_provider_error",
    "is_context_overflow_error",
    "extract_http_status_code",
    "provider_generate_with_retry",
]
