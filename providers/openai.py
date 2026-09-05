# OpenAI provider with streaming, prompt-cache telemetry, reasoning effort,
# and vision input. Uses the official `openai` Python SDK (Chat Completions
# API). Tool-call deltas arrive in chunks; we forward them as
# `tool_call_args_delta` events so the agent loop can render progress.
import base64
import json
import os
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

from .base import (
    CacheHint,
    FileReference,
    ImageData,
    LLMProvider,
    MediaData,
    Message,
    MessagePart,
    ProviderResponse,
    StreamEvent,
    ToolDefinition,
)
from utils.model_pricing import input_modality_for_mime


_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _sanitize_stream_error(exc: BaseException, *, cap: int = 300) -> str:
    """Bound and scrub an SDK exception message before it reaches a
    StreamEvent. SDK exceptions can embed upstream response bodies, request
    URLs, and prompt fragments; events are displayed/logged/traced, so the
    text is secret-redacted and length-capped. The exception class name is
    always preserved so callers can still distinguish error categories."""
    try:
        from mu.security.secret_paths import redact_secrets

        text, _ = redact_secrets(str(exc))
    except Exception:
        text = str(exc)
    text = " ".join(text.split())
    if len(text) > cap:
        text = text[: cap - 3].rstrip() + "..."
    return f"{type(exc).__name__}: {text}"



def _is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return any(n.startswith(p) for p in _REASONING_MODEL_PREFIXES)


_OPENAI_IMAGE_MIMES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
_OPENAI_AUDIO_FORMATS = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


def _media_content_part(media: MediaData) -> Optional[Dict[str, Any]]:
    """Serialize native media using Chat Completions content-part shapes."""
    mime_type = str(media.mime_type or "application/octet-stream").split(";", 1)[0].lower()
    encoded = base64.b64encode(media.data).decode("ascii")
    modality = input_modality_for_mime(mime_type, media.display_name or "")
    if modality == "image" and mime_type in _OPENAI_IMAGE_MIMES:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }
    if modality == "audio" and mime_type in _OPENAI_AUDIO_FORMATS:
        return {
            "type": "input_audio",
            "input_audio": {
                "data": encoded,
                "format": _OPENAI_AUDIO_FORMATS[mime_type],
            },
        }
    if modality == "document":
        return {
            "type": "file",
            "file": {
                "filename": media.display_name or "attachment",
                "file_data": f"data:{mime_type};base64,{encoded}",
            },
        }
    return None


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI ChatGPT models."""

    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    def __init__(self, model_name: str = "", api_key: Optional[str] = None):
        if not api_key:
            api_key = self.API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it via: export OPENAI_API_KEY='your-key'"
            )
        super().__init__(model_name)
        self.name = "openai"
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    # ------------------------------------------------------------------ models

    def get_available_models(self) -> List[str]:
        try:
            return [m.id for m in self._client.models.list().data]
        except Exception:
            return ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

    def supports_input_mime(self, mime_type: str, filename: str = "") -> bool:
        if not super().supports_input_mime(mime_type, filename):
            return False
        mime = str(mime_type or "").split(";", 1)[0].strip().lower()
        modality = input_modality_for_mime(mime, filename)
        if modality == "image":
            return mime in _OPENAI_IMAGE_MIMES
        if modality == "audio":
            return mime in _OPENAI_AUDIO_FORMATS
        # Chat Completions currently has no video content-part shape.
        return modality != "video"

    def native_media_request_limit(self) -> int:
        # OpenAI documents a 50 MB combined request limit for file inputs.
        return 50 * 1024 * 1024

    # ------------------------------------------------------- message conversion

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal Message format to OpenAI Chat Completions format.

        Tool calls and tool results require role/structure that differs from
        plain text turns:
          * assistant -> {"role": "assistant", "content": ..., "tool_calls": [...]}
          * tool      -> {"role": "tool", "tool_call_id": "...", "content": "..."}
        """

        out: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                # Emit one OpenAI tool message per tool_result part. OpenAI
                # requires a tool_call_id to match the originating call.
                returned_media: List[MediaData] = []
                for part in msg.parts:
                    if part.type != "tool_result":
                        continue
                    payload = part.tool_result
                    if isinstance(payload, (dict, list)):
                        payload = json.dumps(payload, indent=2, sort_keys=True)
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id or part.tool_name or "",
                            "content": str(payload),
                        }
                    )
                    returned_media.extend(part.media_inputs or [])
                media_parts = [
                    value
                    for value in (_media_content_part(media) for media in returned_media)
                    if value is not None
                ]
                if media_parts:
                    out.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Native media returned by the preceding tool call(s):",
                                },
                                *media_parts,
                            ],
                        }
                    )
                continue

            role = "user" if msg.role == "user" else "assistant" if msg.role == "assistant" else msg.role
            text_chunks: List[Dict[str, Any]] = []
            tool_calls: List[Dict[str, Any]] = []

            for part in msg.parts:
                if part.type == "text" and part.text:
                    text_chunks.append({"type": "text", "text": part.text})
                elif (
                    part.type == "image_input"
                    and part.image
                    and self.supports_input_mime(
                        part.image.mime_type, part.image.display_name or ""
                    )
                ):
                    b64 = base64.b64encode(part.image.data).decode("ascii")
                    text_chunks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{part.image.mime_type};base64,{b64}"},
                        }
                    )
                elif part.type == "media_input" and part.media:
                    media_part = _media_content_part(part.media)
                    if media_part is not None:
                        text_chunks.append(media_part)
                elif part.type == "file" and part.file_ref:
                    if str(part.file_ref.uri or "").startswith("file-"):
                        text_chunks.append(
                            {
                                "type": "file",
                                "file": {"file_id": part.file_ref.uri},
                            }
                        )
                    else:
                        text_chunks.append(
                            {"type": "text", "text": f"[File: {part.file_ref.display_name}]"}
                        )
                elif part.type == "tool_call" and role == "assistant":
                    args = part.tool_args or {}
                    tool_calls.append(
                        {
                            "id": part.tool_call_id or f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": part.tool_name or "",
                                "arguments": json.dumps(args, sort_keys=True),
                            },
                        }
                    )

            entry: Dict[str, Any] = {"role": role}
            if text_chunks:
                if len(text_chunks) == 1 and text_chunks[0]["type"] == "text":
                    entry["content"] = text_chunks[0]["text"]
                else:
                    entry["content"] = text_chunks
            else:
                entry["content"] = "" if role != "assistant" or not tool_calls else None
            if tool_calls:
                entry["tool_calls"] = tool_calls
                # OpenAI requires content to be either string or null when tool_calls present
                if entry.get("content") == "":
                    entry["content"] = None
            out.append(entry)

        return out

    def _convert_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # --------------------------------------------------------------- streaming

    def stream(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        thinking: bool = False,
        tools: Optional[List[ToolDefinition]] = None,
        cache_hint: Optional[CacheHint] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Iterator[StreamEvent]:

        model = self.model_name or "gpt-4o-mini"

        openai_msgs = self._convert_messages(messages)
        if system_prompt:
            openai_msgs.insert(0, {"role": "system", "content": system_prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": openai_msgs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        tool_defs = self._convert_tools(tools)
        if tool_defs:
            payload["tools"] = tool_defs
            payload["tool_choice"] = "auto"

        if _is_reasoning_model(model):
            effort = reasoning_effort or ("high" if thinking else "medium")
            payload["reasoning_effort"] = effort

        # OpenAI Chat Completions caches stable prefixes automatically; the
        # `cache_hint` argument is informational here. We still rely on stable
        # ordering: system prompt first, then tools, then history. That's the
        # order callers already build.
        _ = cache_hint  # explicitly unused

        # ------------------------------------------------------ stream events

        # Track partial tool calls. OpenAI uses per-index identifiers in deltas.
        partial_calls: Dict[int, Dict[str, Any]] = {}
        seen_ids: set = set()
        chunk_iter = None

        try:
            # Create the stream inside the try so connection / auth /
            # timeout failures take the same StreamEvent(error) path as
            # iteration failures.
            chunk_iter = self._client.chat.completions.create(**payload)
            for chunk in chunk_iter:
                # Usage chunk (only present when stream_options.include_usage)
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                    cached = 0
                    reasoning = 0
                    details = getattr(usage, "prompt_tokens_details", None)
                    if details is not None:
                        cached = getattr(details, "cached_tokens", 0) or 0
                    out_details = getattr(usage, "completion_tokens_details", None)
                    if out_details is not None:
                        reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
                    yield StreamEvent(
                        kind="usage",
                        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                        cached_tokens=cached,
                        reasoning_tokens=reasoning,
                    )

                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # Reasoning content (where the SDK surfaces it)
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    yield StreamEvent(kind="thinking_delta", text=rc)

                if delta.content:
                    yield StreamEvent(kind="text_delta", text=delta.content)

                tcs = getattr(delta, "tool_calls", None) or []
                for tc in tcs:
                    idx = getattr(tc, "index", None)
                    if idx is None:
                        idx = len(partial_calls)
                    state = partial_calls.setdefault(
                        idx,
                        {
                            "id": None,
                            "name": None,
                            "args": [],
                        },
                    )
                    if getattr(tc, "id", None):
                        state["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            state["name"] = fn.name
                        args_chunk = getattr(fn, "arguments", None)
                        if args_chunk:
                            state["args"].append(args_chunk)

                    cid = state["id"] or f"call_{idx}"
                    if cid not in seen_ids and state["name"]:
                        seen_ids.add(cid)
                        yield StreamEvent(
                            kind="tool_call_start",
                            tool_name=state["name"],
                            tool_call_id=cid,
                        )
                    if fn is not None and getattr(fn, "arguments", None):
                        yield StreamEvent(
                            kind="tool_call_args_delta",
                            text=fn.arguments,
                            tool_call_id=cid,
                            tool_name=state["name"],
                        )

                if choice.finish_reason in ("tool_calls", "stop", "length"):
                    # Finalize all known tool calls.
                    for idx, state in sorted(partial_calls.items()):
                        cid = state["id"] or f"call_{idx}"
                        joined = "".join(state["args"]).strip()
                        args: Dict[str, Any]
                        if joined:
                            try:
                                parsed = json.loads(joined)
                                args = parsed if isinstance(parsed, dict) else {"_value": parsed}
                            except json.JSONDecodeError:
                                args = {"_raw": joined}
                        else:
                            args = {}
                        yield StreamEvent(
                            kind="tool_call_complete",
                            tool_name=state["name"],
                            tool_args=args,
                            tool_call_id=cid,
                        )
                    partial_calls.clear()
                    seen_ids.clear()
        except Exception as exc:
            yield StreamEvent(kind="error", text=_sanitize_stream_error(exc))
            raise
        finally:
            # Close the underlying HTTP stream when iteration is abandoned
            # (error raised mid-stream, caller breaks early, or success).
            closer = getattr(chunk_iter, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

        yield StreamEvent(kind="done")

    # ----------------------------------------------------- non-streaming path

    def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        thinking: bool = False,
        tools: Optional[List[ToolDefinition]] = None,
    ) -> ProviderResponse:
        return self.drain_stream(
            self.stream(
                messages=messages,
                system_prompt=system_prompt,
                thinking=thinking,
                tools=tools,
            )
        )

    # ----------------------------------------------------------------- files

    def upload_file(self, file_path: str, mime_type: str) -> Optional[FileReference]:
        """OpenAI files endpoint is separate from chat API; we return a local ref."""
        return FileReference(uri=file_path, mime_type=mime_type, display_name=file_path)
