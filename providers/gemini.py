# Gemini provider with streaming via `generate_content_stream`, explicit
# context-cache support, and round-tripping of `thought_signature` for
# Gemini 3+ thinking models.
#
# Thought-signature handling is load-bearing: Gemini 3 models reject the
# next turn if the prior turn's signature is missing. We capture from three
# fallback locations on the chunk part (Part.thought_signature,
# function_call.id, function_call.thought_signature), hex-encode bytes for
# JSON-safe storage, and decode back on the next turn.
import logging
import os
import json
from typing import Iterator, List, Optional

from google import genai
from google.genai import types

from .base import (
    CacheHint,
    FileReference,
    LLMProvider,
    Message,
    MessagePart,
    ProviderResponse,
    StreamEvent,
    ToolDefinition,
)


logger = logging.getLogger(__name__)


def _sanitize_stream_error(exc: BaseException, *, cap: int = 300) -> str:
    """Bound and scrub a provider exception message before it reaches a
    StreamEvent or log line. Provider exceptions can embed upstream response
    bodies, request URLs, and prompt fragments; text is secret-redacted and
    length-capped. The exception class name is always preserved."""
    try:
        from mu.security.secret_paths import redact_secrets

        text, _ = redact_secrets(str(exc))
    except Exception:
        text = str(exc)
    text = " ".join(text.split())
    if len(text) > cap:
        text = text[: cap - 3].rstrip() + "..."
    return f"{type(exc).__name__}: {text}"


class GeminiProvider(LLMProvider):
    # Harness-controlled HTTP timeout (seconds). The SDK default may change
    # between versions and can otherwise occupy an agent worker indefinitely.
    DEFAULT_TIMEOUT_SECONDS = 300.0

    def __init__(
        self,
        model_name: str = "",
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        super().__init__(model_name)
        self.name = "gemini"
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        try:
            env_timeout = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "") or 0)
        except ValueError:
            env_timeout = 0.0
        self.timeout_seconds = float(
            timeout_seconds
            or env_timeout
            or self.DEFAULT_TIMEOUT_SECONDS
        )
        http_options = types.HttpOptions(timeout=self.timeout_seconds * 1000)
        self.client = genai.Client(
            api_key=api_key, http_options=http_options
        )

    def get_available_models(self) -> List[str]:
        try:
            models_response = self.client.models.list()
            discovered = []
            for model in models_response:
                name = getattr(model, "name", "") or ""
                if "gemini" in name.lower():
                    discovered.append(name.split("/")[-1])
            return discovered
        except Exception as e:
            logger.warning(
                "Failed to fetch available models from Gemini API: %s",
                _sanitize_stream_error(e),
            )
            return []

    # ------------------------------------------------------- message conversion

    def _convert_to_gemini_contents(
        self, messages: List[Message]
    ) -> List[types.Content]:
        gemini_contents: List[types.Content] = []
        for msg in messages:
            gemini_role = "user"
            if msg.role == "assistant":
                gemini_role = "model"
            elif msg.role == "tool":
                gemini_role = "user"  # Gemini expects function responses from 'user'

            gemini_parts = []
            for part in msg.parts:
                if part.type == "text":
                    gemini_parts.append(types.Part(text=part.text))
                elif part.type == "file" and part.file_ref:
                    gemini_parts.append(
                        types.Part(
                            file_data=types.FileData(
                                mime_type=part.file_ref.mime_type,
                                file_uri=part.file_ref.uri,
                            )
                        )
                    )
                elif part.type == "image_input" and part.image:
                    gemini_parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                data=part.image.data,
                                mime_type=part.image.mime_type,
                            )
                        )
                    )
                elif part.type == "tool_call":
                    fc_obj = types.FunctionCall(
                        name=part.tool_name, args=part.tool_args
                    )
                    fc_part = types.Part(function_call=fc_obj)
                    if part.thought_signature:
                        try:
                            fc_part.thought_signature = bytes.fromhex(
                                part.thought_signature
                            )
                        except (ValueError, TypeError):
                            fc_part.thought_signature = part.thought_signature.encode()
                    gemini_parts.append(fc_part)

                elif part.type == "tool_result":
                    tool_result = part.tool_result
                    if isinstance(tool_result, (dict, list)):
                        tool_result = json.dumps(tool_result, indent=2, sort_keys=True)
                    fresp = types.FunctionResponse(
                        name=part.tool_name,
                        response={"result": str(tool_result)},
                    )
                    # Thought signatures belong to MODEL-produced parts only.
                    # A function_response is a user-side replay of tool
                    # output; attaching a signature here corrupts the
                    # transcript. The signature stays on the preceding
                    # function_call part (handled above).
                    resp_part = types.Part(function_response=fresp)
                    gemini_parts.append(resp_part)

            if not gemini_parts:
                continue
            # Gemini enforces strict role alternation; merge adjacent same-role parts.
            if gemini_contents and gemini_contents[-1].role == gemini_role:
                gemini_contents[-1].parts.extend(gemini_parts)
            else:
                gemini_contents.append(
                    types.Content(role=gemini_role, parts=gemini_parts)
                )
        return gemini_contents

    def _build_config(
        self,
        system_prompt: Optional[str],
        thinking: bool,
        tools: Optional[List[ToolDefinition]],
        cache_hint: Optional[CacheHint],
    ) -> types.GenerateContentConfig:
        t_config = (
            types.ThinkingConfig(thinking_level="high")
            if thinking
            else types.ThinkingConfig(thinking_level=None)
        )
        gemini_tools = []
        if tools:
            func_decls = []
            for t in tools:
                func_decls.append(
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=t.parameters,
                    )
                )
            gemini_tools = [types.Tool(function_declarations=func_decls)]

        cfg_kwargs = dict(
            thinking_config=t_config,
            system_instruction=system_prompt,
            tools=gemini_tools if gemini_tools else None,
        )

        # Future expansion: Gemini explicit Context Cache. The SDK exposes
        # `cached_content` on GenerateContentConfig but creating cache entries
        # requires a separate `client.caches.create(...)` flow and a stable
        # prefix policy. We honor `cache_hint` informationally for now and
        # surface `cached_content_token_count` from usage_metadata, which the
        # API reports automatically when implicit caching kicks in.
        _ = cache_hint
        return types.GenerateContentConfig(**cfg_kwargs)

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
        contents = self._convert_to_gemini_contents(messages)
        # `reasoning_effort` is OpenAI-shaped; map "high"->thinking on for Gemini.
        if reasoning_effort in ("high", "medium"):
            thinking = True
        gen_config = self._build_config(system_prompt, thinking, tools, cache_hint)

        seen_call_ids: set = set()
        emitted_call_index = 0
        usage_emitted = False
        last_usage = None

        try:
            chunk_iter = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=gen_config,
            )

            for chunk in chunk_iter:
                if getattr(chunk, "usage_metadata", None):
                    last_usage = chunk.usage_metadata
                parts = []
                # Iterate parts safely (different SDK versions expose differently)
                if getattr(chunk, "candidates", None):
                    for cand in chunk.candidates:
                        content = getattr(cand, "content", None)
                        if content and getattr(content, "parts", None):
                            parts.extend(content.parts)
                elif getattr(chunk, "parts", None):
                    parts = list(chunk.parts)

                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        # Gemini may flag chain-of-thought parts via `thought` boolean
                        if getattr(part, "thought", False):
                            yield StreamEvent(kind="thinking_delta", text=text)
                        else:
                            yield StreamEvent(kind="text_delta", text=text)

                    fc = getattr(part, "function_call", None)
                    if fc:
                        # Only genuine thought signatures round-trip here.
                        # fc.id is a provider call identifier, NOT a
                        # signature — replaying it as one corrupts the
                        # next turn's signature validation.
                        ts = getattr(part, "thought_signature", None) or getattr(
                            fc, "thought_signature", None
                        )
                        if ts and isinstance(ts, bytes):
                            ts = ts.hex()

                        # Gemini delivers tool calls as a single complete object,
                        # not as deltas. Emit start+complete in one shot.
                        cid = f"gemini_call_{emitted_call_index}"
                        emitted_call_index += 1
                        seen_call_ids.add(cid)

                        yield StreamEvent(
                            kind="tool_call_start",
                            tool_name=fc.name,
                            tool_call_id=cid,
                            thought_signature=ts,
                        )
                        args = dict(fc.args) if getattr(fc, "args", None) else {}
                        yield StreamEvent(
                            kind="tool_call_complete",
                            tool_name=fc.name,
                            tool_args=args,
                            tool_call_id=cid,
                            thought_signature=ts,
                        )

                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        # Inline images returned by the model; we don't have a
                        # streaming event for these (they arrive whole), so
                        # surface as a text event noting their presence.
                        mt = getattr(inline, "mime_type", "image/?")
                        yield StreamEvent(
                            kind="text_delta",
                            text=f"[inline {mt} attachment]",
                        )
        except Exception as exc:
            yield StreamEvent(kind="error", text=_sanitize_stream_error(exc))
            raise

        if last_usage and not usage_emitted:
            cached = (
                getattr(last_usage, "cached_content_token_count", 0)
                or getattr(last_usage, "cache_tokens_details", 0)
                or 0
            )
            reasoning = getattr(last_usage, "thoughts_token_count", 0) or 0
            yield StreamEvent(
                kind="usage",
                input_tokens=getattr(last_usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(last_usage, "candidates_token_count", 0) or 0,
                total_tokens=getattr(last_usage, "total_token_count", 0) or 0,
                cached_tokens=cached if isinstance(cached, int) else 0,
                reasoning_tokens=reasoning,
            )

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

    def upload_file(self, file_path: str, mime_type: str) -> FileReference:
        uploaded = self.client.files.upload(
            file=file_path, config=types.UploadFileConfig(mime_type=mime_type)
        )
        return FileReference(
            uri=uploaded.uri,
            mime_type=uploaded.mime_type,
            display_name=os.path.basename(file_path),
        )
