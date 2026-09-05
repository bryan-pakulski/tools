"""History → provider-message serialization helpers.

Four helpers, all consumed by the agent loop just before / after the
provider call:

  * `build_messages_from_history(recent, new_user)` — rehydrate the
    dict-shaped history records into the strongly-typed `Message` /
    `MessagePart` / `FileReference` / `ImageData` graph that providers
    accept. Handles text / file / image_input / tool_call / tool_result
    parts.

  * `prepare_runtime_history(session, turn_start_index)` — compute
    which slice of `session.session_manager.history` should be sent
    this turn. Walks backwards from the tail, summing per-message
    tokens, until the provider budget is hit. It never applies a
    second arbitrary tool-message window.

  * `summarize_message_parts(msg_dict)` — render one history entry as
    a single-line summary used by `prepare_runtime_history` when
    compressing old tool activity.

  * `clip_preview(text, limit)` — shorten a string with an ellipsis
    when it exceeds `limit` chars. Used in tool-result previews and
    history summaries.


Tests: `tests/test_session.py` (history compression, ordering, image
rehydration), `tests/test_vision_e2e.py` (image_input round-trip),
`tests/test_mu_session_history.py` (token estimation pinning).
"""

from __future__ import annotations

import base64
from typing import Any, Callable, List, Optional

from providers.base import (
    FileReference,
    ImageData,
    LLMProvider,
    MediaData,
    Message,
    MessagePart,
)

from .helpers import _shorten_tool_args


def _compact_tool_result_ref(part: dict) -> str:
    """Render a compact one-line ref for a tool_result part that has
    a cache_key, replacing the full structured envelope in provider
    messages. The model can recall(cache_key) or use chunk retrieval
    tools (result_range/head/tail/search/etc.) to fetch the full
    content on demand — keeping L5 context lean without losing
    retrievability.

    Format: ← tool call - name: X, result: success, ref: KEY
    """
    name = part.get("tool_name", "tool")
    cache_key = part.get("cache_key", "")
    raw = part.get("tool_result")
    if isinstance(raw, dict):
        ok = raw.get("ok")
        result_state = "success" if ok else "error"
    else:
        result_state = "success" if not str(raw or "").startswith("Error") else "error"
    return f"← tool call - name: {name}, result: {result_state}, ref: {cache_key}"


def _lean_tool_result(tool_result: Any) -> Any:
    """Collapse a structured tool-result dict for provider serialization.

    In-history structured envelopes carry duplicated content: ``summary``
    (a 220-char echo of ``raw``), ``error.message`` (an echo of raw),
    ``data.preview`` (a 240-char echo for read/get_chunk), the full
    ``telemetry.tool_envelope`` copy, and boolean bookkeeping flags. When
    the verbatim result ships in L5, all of that rides along on top of
    ``raw`` — pure duplication.

    This keeps the substance: ``ok``/``error_code``, the full ``raw``,
    per-tool structured ``data`` (minus preview/omitted flags), typed
    ``artifacts``, and compact telemetry counts. The original envelope in
    history is untouched — this applies only at serialization time, so
    GUI/metrics/state-capsule views keep their full fidelity.
    """
    if not isinstance(tool_result, dict) or "raw" not in tool_result:
        return tool_result
    lean: dict = {"ok": bool(tool_result.get("ok"))}
    if tool_result.get("error_code"):
        lean["error_code"] = tool_result["error_code"]
    raw = tool_result.get("raw")
    if raw is not None:
        lean["raw"] = raw
    data = tool_result.get("data")
    if isinstance(data, dict):
        data = {
            k: v for k, v in data.items()
            if k not in ("preview", "omitted", "stored_ref", "retrievable_via", "omission_note")
        }
        if data:
            lean["data"] = data
    if tool_result.get("artifacts"):
        lean["artifacts"] = tool_result["artifacts"]
    if tool_result.get("modified_files"):
        lean["modified_files"] = tool_result["modified_files"]
    return lean


def build_messages_from_history(
    recent_history_dicts: List[dict],
    new_user_message_dict: dict,
    *,
    tool_result_floor: int = 0,
    media_resolver: Optional[Callable[[dict[str, Any]], Optional[MediaData]]] = None,
) -> List[Message]:
    """Rehydrate dict-shaped history records into provider-typed
    `Message` objects. Pass-through for text; decodes base64 image
    payloads back into `ImageData`; threads provider-supplied
    `thought_signature` through tool_call / tool_result parts.

    When ``tool_result_floor > 0``, the last ``floor`` tool-result-bearing
    messages are kept verbatim (the model needs recent results in context
    without an extra recall() round-trip). Tool results beyond the floor
    that have a ``cache_key`` are replaced by a compact ref string — the
    model can ``recall(KEY)`` or use chunk retrieval tools to fetch the
    full content on demand. Results without a cache_key stay verbatim
    (no ref to recall)."""
    # Build the set of message indices (in recent_history_dicts only —
    # the new user message is never compacted) that are within the floor.
    floor_indices: set[int] = set()
    if tool_result_floor > 0:
        tr_indices = [
            i for i, md in enumerate(recent_history_dicts)
            if any(p.get("type") == "tool_result" for p in (md.get("parts") or []))
        ]
        if tr_indices:
            floor_indices = set(tr_indices[-tool_result_floor:])

    messages: List[Message] = []
    for hist_idx, msg_dict in enumerate(recent_history_dicts + [new_user_message_dict]):
        is_within_floor = hist_idx in floor_indices
        parts: List[MessagePart] = []
        for p in msg_dict.get("parts", []):
            p_type = p.get("type")
            if p_type == "text":
                parts.append(MessagePart(type="text", text=p["text"]))
            elif p_type == "file":
                fr_data = p.get("file_ref", {})
                parts.append(
                    MessagePart(
                        type="file",
                        file_ref=FileReference(
                            uri=fr_data.get("uri"),
                            mime_type=fr_data.get("mime_type"),
                            display_name=fr_data.get("display_name"),
                        ),
                    )
                )
            elif p_type == "attachment":
                attachment = p.get("attachment", {}) or {}
                attachment_id = str(attachment.get("attachment_id") or "")
                name = str(attachment.get("name") or "attachment")
                mime_type = str(attachment.get("mime_type") or "application/octet-stream")
                size = int(attachment.get("size") or 0)
                parts.append(MessagePart(
                    type="text",
                    text=(
                        "[User-uploaded attachment: "
                        f"id={attachment_id}; name={name}; mime={mime_type}; size={size} bytes. "
                        "Use read_attachment/search_attachments for bounded text, or download_attachment in workspace/container mode for a local copy.]"
                    ),
                ))
            elif p_type == "image_input":
                img_data = p.get("image", {}) or {}
                raw = img_data.get("data_b64") or ""
                try:
                    decoded = base64.b64decode(raw) if raw else b""
                except Exception:
                    decoded = b""
                if decoded:
                    parts.append(
                        MessagePart(
                            type="image_input",
                            image=ImageData(
                                data=decoded,
                                mime_type=img_data.get("mime_type", "image/png"),
                                source=img_data.get("source"),
                            ),
                        )
                    )
            elif p_type == "media_input":
                media_data = p.get("media", {}) or {}
                media: Optional[MediaData] = None
                raw = media_data.get("data_b64") or ""
                if raw:
                    try:
                        decoded = base64.b64decode(raw)
                    except Exception:
                        decoded = b""
                    if decoded:
                        media = MediaData(
                            data=decoded,
                            mime_type=media_data.get(
                                "mime_type", "application/octet-stream"
                            ),
                            source=media_data.get("source"),
                            display_name=media_data.get("display_name")
                            or media_data.get("name"),
                        )
                elif media_resolver is not None:
                    media = media_resolver(media_data)
                if media is not None:
                    parts.append(MessagePart(type="media_input", media=media))
            elif p_type == "tool_call":
                parts.append(
                    MessagePart(
                        type="tool_call",
                        tool_name=p["tool_name"],
                        tool_args=p.get("tool_args", {}),
                        thought_signature=p.get("thought_signature"),
                    )
                )
            elif p_type == "tool_result":
                cache_key = p.get("cache_key")
                media_inputs: List[MediaData] = []
                if media_resolver is not None and (not cache_key or is_within_floor):
                    for reference in p.get("media_inputs") or []:
                        media = media_resolver(reference)
                        if media is not None:
                            media_inputs.append(media)
                # Beyond the tool_result_floor: replace full envelope with
                # compact ref string when a cache_key is available. The model
                # can recall(cache_key) or use chunk retrieval tools to fetch
                # the full content on demand. Within the floor: keep verbatim
                # so the model has recent results without an extra round-trip.
                # No cache_key: keep verbatim (no ref to recall).
                if cache_key and not is_within_floor:
                    parts.append(
                        MessagePart(
                            type="tool_result",
                            tool_name=p.get("tool_name", "tool"),
                            tool_result=_compact_tool_result_ref(p),
                            thought_signature=p.get("thought_signature"),
                            media_inputs=media_inputs,
                        )
                    )
                else:
                    parts.append(
                        MessagePart(
                            type="tool_result",
                            tool_name=p.get("tool_name", "tool"),
                            tool_result=_lean_tool_result(
                                p.get("tool_result", "")
                            ),
                            thought_signature=p.get("thought_signature"),
                            media_inputs=media_inputs,
                        )
                    )
        messages.append(Message(role=msg_dict["role"], parts=parts))
    return messages


def clip_preview(text: Any, limit: int = 240) -> str:
    """Trim a string to `limit` chars, appending an ellipsis when
    truncated. Stripping leading/trailing whitespace first."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def summarize_message_parts(
    msg_dict: dict,
    provider: Optional[LLMProvider] = None,
) -> str:
    """Render one history entry as a single-line summary for
    compressed-history blocks. Returns `- <role>: <summaries>` or
    `- <role>: [no serializable content]`.

    Char limits expanded from 120/140 to 500 to preserve more context
    in the mechanical fallback path. The LLM summarization path
    (_llm_summarize_tool_batch) is preferred when a provider is
    available — see prepare_runtime_history."""
    role = msg_dict.get("role", "message")
    summaries: List[str] = []
    for part in msg_dict.get("parts", []):
        p_type = part.get("type")
        if p_type == "text":
            text = str(part.get("text", "")).strip().replace("\n", " ")
            if text:
                summaries.append(text[:500])
        elif p_type == "tool_call":
            summaries.append(
                f"tool_call:{part.get('tool_name')} "
                f"args={_shorten_tool_args(part.get('tool_args', {}))}"
            )
        elif p_type == "tool_result":
            # Action record (spec #4/#5): for cache_key'd / structured results,
            # emit a compact one-line record (decision + outcome + files +
            # error + cache_key) instead of a 500-char prose clip. The full
            # raw is recoverable via recall(cache_key).
            from mu.session.action_record import (
                is_action_record_eligible,
                render_action_record,
            )

            if is_action_record_eligible(part):
                summaries.append(render_action_record(part))
            else:
                raw_result = part.get("tool_result", "")
                if isinstance(raw_result, dict):
                    result = str(
                        raw_result.get("summary") or raw_result.get("raw", "")
                    )
                else:
                    result = str(raw_result)
                result = result.strip().replace("\n", " ")
                # Include cache key tag if present so model can recall full result
                cache_key = part.get("cache_key")
                cache_tag = f"[cache:{cache_key}] " if cache_key else ""
                if len(result) > 500:
                    result = f"{result[:497]}..."
                summaries.append(
                    f"tool_result:{part.get('tool_name')} => {cache_tag}{result}"
                )
        elif p_type == "file":
            fr = part.get("file_ref", {})
            summaries.append(
                f"file:{fr.get('display_name', fr.get('uri', 'unknown'))}"
            )
        elif p_type == "attachment":
            attachment = part.get("attachment", {}) or {}
            summaries.append(
                "attachment:"
                f"{attachment.get('name', 'unknown')}"
                f" [id={attachment.get('attachment_id', '')}]"
            )
        elif p_type == "image_input":
            img = part.get("image", {}) or {}
            source = img.get("source") or img.get("mime_type", "image")
            summaries.append(f"image:{source}")
        elif p_type == "media_input":
            media = part.get("media", {}) or {}
            source = (
                media.get("name")
                or media.get("display_name")
                or media.get("source")
                or media.get("mime_type", "media")
            )
            summaries.append(f"media:{source}")

    if not summaries:
        return f"- {role}: [no serializable content]"
    return f"- {role}: " + " | ".join(summaries)


# ── R4 / FM-2: first-class huge-message handling ────────────────────────
# When a single message (typically the user's turn prompt — a massive
# paste) alone consumes most of the L5 budget, the backward walk in
# `prepare_runtime_history` is forced to include it (the newest message
# is always kept), which overflows the provider window and triggers
# destructive mechanical truncation in `_degrade_oldest_runtime_payload`.
# Instead we chunk-summarize the oversized text via the provider (or a
# head+tail mechanical fallback when no provider is available) and
# substitute a labeled CONTEXT-OVERFLOW envelope so the model knows the
# full original is not in context.

_OVERFLOW_CHUNK_CHARS = 12_000  # ~3000 tokens per chunk
_OVERFLOW_MAX_CHUNKS = 12  # bound work on pathological inputs


def _chunk_summarize_text(
    provider: Optional[LLMProvider],
    text: str,
    budget_tokens: int,
) -> str:
    """Summarize an oversized text in chunks. Uses the provider when
    available (one generate() call per chunk, capped at
    `_OVERFLOW_MAX_CHUNKS`); falls back to a head+tail mechanical elision
    when no provider is present or a chunk call fails."""
    if not text:
        return ""
    if provider is None:
        return _mechanical_elide(text, budget_tokens)
    chunks = [
        text[i : i + _OVERFLOW_CHUNK_CHARS]
        for i in range(0, len(text), _OVERFLOW_CHUNK_CHARS)
    ][:_OVERFLOW_MAX_CHUNKS]
    system = (
        "You are a context summarizer for an AI coding agent. Summarize the "
        "following chunk concisely while preserving file paths, function "
        "names, error messages, and key findings VERBATIM. Be concise but "
        "complete. Output ONLY the summary, no headers or commentary."
    )
    summaries: List[str] = []
    for chunk in chunks:
        try:
            resp = provider.generate(
                messages=[
                    Message(
                        role="user",
                        parts=[MessagePart(type="text", text=chunk)],
                    )
                ],
                system_prompt=system,
                thinking=False,
                tools=None,
            )
            s = str(resp.text or "").strip()
            summaries.append(s if s else _mechanical_elide(chunk, budget_tokens))
        except Exception:
            summaries.append(_mechanical_elide(chunk, budget_tokens))
    if len(text) > _OVERFLOW_CHUNK_CHARS * _OVERFLOW_MAX_CHUNKS:
        elided = len(text) - _OVERFLOW_CHUNK_CHARS * _OVERFLOW_MAX_CHUNKS
        summaries.append(f"[...{elided} additional chars not summarized...]")
    return "\n\n".join(s for s in summaries if s).strip()


def _mechanical_elide(text: str, budget_tokens: int) -> str:
    """Head+tail elision fallback (no provider). Keeps a head and tail
    sized to roughly fit the budget (~4 chars/token)."""
    cap = max(4000, int(budget_tokens * 4))
    if len(text) <= cap:
        return text
    half = cap // 2
    return (
        text[:half]
        + f"\n[...{len(text) - cap} chars elided (no provider available "
        "for chunked summary)...]\n"
        + text[-half:]
    )


def _maybe_summarize_oversized(
    session: Any,
    abs_idx: int,
    msg: dict,
    budget_tokens: int,
    provider: Optional[LLMProvider],
    cache: dict,
) -> dict:
    """If `msg`'s text content alone exceeds the overflow threshold,
    substitute a chunk-summarized CONTEXT-OVERFLOW envelope. Returns the
    original message unchanged when it is not oversized. Summaries are
    cached per absolute history index so repeated `prepare_runtime_history`
    calls within a turn don't re-summarize."""
    if budget_tokens <= 0 or not isinstance(msg, dict):
        return msg
    text = None
    for p in msg.get("parts", []) or []:
        if p.get("type") == "text":
            text = p.get("text") or ""
            break
    if not text:
        return msg
    try:
        text_tokens = session.session_manager._estimate_tokens_from_text(text)
    except Exception:
        text_tokens = len(text) // 4
    # Threshold: a single message consuming more than 60% of the L5
    # budget AND non-trivially large. Below this, normal compaction
    # handles it; above it, proactive chunked summary avoids overflow.
    threshold = max(2000, int(budget_tokens * 0.6))
    if text_tokens <= threshold:
        return msg
    if abs_idx in cache:
        summarized = cache[abs_idx]
    else:
        summarized = _chunk_summarize_text(provider, text, budget_tokens)
        cache[abs_idx] = summarized
    envelope = (
        f"[CONTEXT-OVERFLOW — this message exceeded the context budget "
        f"(~{text_tokens} tokens) and was summarized in chunks. The full "
        f"original is NOT in context; ask the user to re-paste a specific "
        f"section if you need verbatim detail.]\n\n{summarized}"
    )
    new_parts = []
    replaced = False
    for p in msg.get("parts", []) or []:
        if (not replaced) and p.get("type") == "text" and (p.get("text") or "") == text:
            new_parts.append({"type": "text", "text": envelope})
            replaced = True
        else:
            new_parts.append(p)
    return {**msg, "parts": new_parts}


def prepare_runtime_history(
    session: Any,
    turn_start_index: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
) -> List[dict]:
    """Pick the slice of `session.session_manager.history` to send to
    the provider this turn, then (within the current-turn region)
    compress older `assistant`/`tool` message pairs into a single
    without arbitrary tool-message compression. Active evidence remains
    verbatim until model-directed compaction or a hard provider-limit recovery.

    Skips compression for any message carrying a thought signature —
    those must round-trip verbatim or the provider rejects subsequent
    calls."""
    session_manager = session.session_manager
    if session_manager.summary_anchor > len(session_manager.history):
        session_manager.summary_anchor = 0
    token_budget = session._compaction_token_budget()
    start_index = len(session_manager.history)
    running_tokens = 0
    while start_index > session_manager.summary_anchor:
        next_index = start_index - 1
        next_tokens = session_manager._estimate_message_tokens(
            session_manager.history[next_index]
        )
        if (
            running_tokens + next_tokens > token_budget
            and next_index < len(session_manager.history) - 1
        ):
            break
        running_tokens += next_tokens
        start_index = next_index
    # R4 / FM-2: if any single message in the runtime slice is oversized
    # relative to the L5 budget, substitute a chunk-summarized
    # CONTEXT-OVERFLOW envelope BEFORE assembling recent_history. Cached
    # per absolute history index on the session so repeated calls within
    # a turn (the loop calls this every iteration) don't re-summarize.
    raw_slice = session_manager.history[start_index:]
    cache = getattr(session, "_oversized_message_summaries", None)
    if cache is None:
        cache = {}
        session._oversized_message_summaries = cache
    runtime_slice = [
        _maybe_summarize_oversized(
            session, start_index + i, msg, token_budget, provider, cache
        )
        for i, msg in enumerate(raw_slice)
    ]
    # Inject protected messages that are below the summary anchor back
    # into the runtime history.  These messages were excluded from LLM
    # summarisation in roll_history_summary() and must appear verbatim
    # in L5 so the model retains the original user request and key
    # decisions even after compaction has advanced the anchor past them.
    protected = getattr(session_manager, "protected_indices", set())
    if protected:
        protected_below_anchor = [
            session_manager.history[idx]
            for idx in sorted(protected)
            if idx < start_index
        ]
        if protected_below_anchor:
            # Wrap protected messages in a labelled envelope so the model
            # understands these are intentionally preserved, not stale or
            # duplicated — they were excluded from L2 summarisation and
            # re-injected verbatim to keep important context (original user
            # request, key decisions) alive through compaction.
            preserved_marker = {
                "role": "user",
                "parts": [{
                    "type": "text",
                    "text": (
                        "[PRESERVED CONTEXT — These messages are kept verbatim "
                        "and protected from summarisation. They are NOT stale "
                        "or duplicated; they are intentionally preserved to "
                        "maintain important context through compaction.]"
                    ),
                }],
            }
            recent_history = (
                [preserved_marker] + protected_below_anchor + runtime_slice
            )
        else:
            recent_history = runtime_slice
    else:
        recent_history = runtime_slice
    return recent_history


__all__ = [
    "build_messages_from_history",
    "clip_preview",
    "summarize_message_parts",
    "prepare_runtime_history",
    "_compact_tool_result_ref",
]
