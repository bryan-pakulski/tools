"""Ollama provider — first-class local LLM support.

Features:
  * **Standard env vars**: `OLLAMA_HOST` overrides the connection target
    (matches the official `ollama` CLI's convention). Falls back to
    `OLLAMA_API_KEY` + `https://ollama.com` if you want the hosted
    service, then `http://localhost:11434` if neither is set.
  * **Preflight check** with actionable error messages — distinguishes
    "Ollama not running" from "model not pulled" from generic transport
    errors, each with the exact CLI command to fix.
  * **Vision** — `MessagePart(type="image_input", image=ImageData)`
    rides on Ollama's `images: [base64]` message field, supported by
    llava, llava-llama3, qwen2-vl, llama3.2-vision, etc.
  * **Reasoning models** — parses `<think>…</think>` blocks out of the
    streamed content and emits them as `thinking_delta` events so the
    harness's existing thinking-tracking telemetry works against
    deepseek-r1, qwen-think, gpt-oss reasoning variants, etc.
  * **Native options** — passes through Ollama-specific tuning knobs
    (`num_ctx`, `num_predict`, `temperature`, `top_p`, `top_k`,
    `repeat_penalty`, `seed`, `mirostat`) from session variables
    prefixed `ollama_*`, plus an `OllamaOptions` constructor arg.
  * **Tool calling**, **keep_alive**, **NDJSON streaming**, and
    **structured tool_call deltas** as before.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

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
from utils.model_pricing import input_modality_for_mime


# Native Ollama options that we expose to callers. Map: kwarg name → JSON
# field name. The Ollama API's `options` object accepts these directly.
_OLLAMA_OPTION_KEYS = (
    "num_ctx",
    "num_predict",
    "temperature",
    "top_p",
    "top_k",
    "repeat_penalty",
    "seed",
    "mirostat",
    "mirostat_eta",
    "mirostat_tau",
    "tfs_z",
    "stop",
)


@dataclass
class OllamaOptions:
    """Provider-specific options. None values are omitted from the payload
    so Ollama applies its own defaults."""

    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    tfs_z: Optional[float] = None
    stop: Optional[List[str]] = None

    def as_payload(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in _OLLAMA_OPTION_KEYS:
            val = getattr(self, key, None)
            if val is not None:
                out[key] = val
        return out


class OllamaError(RuntimeError):
    """Raised when the Ollama daemon is unreachable, a model isn't pulled,
    or the API returns an unrecognized error. Carries an `actionable`
    message suitable for direct display to the user.
    """

    def __init__(self, message: str, *, actionable: Optional[str] = None):
        super().__init__(message)
        self.actionable = actionable or message


def _resolve_host(explicit: Optional[str] = None, mode: str = "auto") -> str:
    """Pick the right Ollama endpoint.

    ``mode`` is the ``ollama_mode`` session variable ("local" | "cloud" |
    "auto"). Priority:

      1. Explicit ``host`` argument / ``ollama_host`` variable (always wins —
         power-user override for a custom daemon)
      2. ``mode == "cloud"`` → ``https://ollama.com`` (hosted service; needs an
         API key via ``ollama_api_key`` / ``OLLAMA_API_KEY``)
      3. ``mode == "local"`` → ``OLLAMA_HOST`` env if set, else the local
         daemon. The legacy ``OLLAMA_API_KEY``→cloud auto-switch is
         *suppressed* here so "local" always means local.
      4. ``mode == "auto"`` (legacy default) → ``OLLAMA_HOST`` env →
         ``https://ollama.com`` if ``OLLAMA_API_KEY`` is set → localhost.
    """
    if explicit:
        host = explicit
    elif mode == "cloud":
        host = "https://ollama.com"
    elif mode == "local":
        host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    else:  # auto / legacy
        env_host = os.environ.get("OLLAMA_HOST")
        if env_host:
            host = env_host
        elif os.environ.get("OLLAMA_API_KEY"):
            host = "https://ollama.com"
        else:
            host = "http://localhost:11434"
    # OLLAMA_HOST often comes as `host:port` without a scheme; normalize.
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


_THINK_OPEN = "<" + "t" + "h" + "i" + "n" + "k" + ">"
_THINK_CLOSE = "<" + "/" + "t" + "h" + "i" + "n" + "k" + ">"


def _split_think_blocks(text: str, *, in_think: bool) -> tuple:
    """Split a streamed content chunk on think-block boundaries.

    Returns `(content_chunks, think_chunks, new_in_think)`. The boundary
    tracking is stateful - callers must thread `in_think` from the
    previous chunk so an opening delimiter in chunk N and the matching
    closing delimiter in chunk N+5 are handled correctly.
    """
    content_chunks: List[str] = []
    think_chunks: List[str] = []
    remaining = text
    while remaining:
        if in_think:
            close_idx = remaining.find(_THINK_CLOSE)
            if close_idx < 0:
                # Hold back a possible partial closing delimiter split
                # across the chunk boundary instead of emitting it as
                # thinking text.
                hold = _partial_suffix_len(remaining, _THINK_CLOSE)
                if hold:
                    think_chunks.append(remaining[:-hold])
                    remaining = ""
                else:
                    think_chunks.append(remaining)
                    remaining = ""
            else:
                think_chunks.append(remaining[:close_idx])
                remaining = remaining[close_idx + len(_THINK_CLOSE) :]
                in_think = False
        else:
            open_idx = remaining.find(_THINK_OPEN)
            if open_idx < 0:
                # Hold back a possible partial opening delimiter split
                # across the chunk boundary instead of leaking it as
                # visible text.
                hold = _partial_suffix_len(remaining, _THINK_OPEN)
                if hold:
                    content_chunks.append(remaining[:-hold])
                else:
                    content_chunks.append(remaining)
                remaining = ""
            else:
                if open_idx > 0:
                    content_chunks.append(remaining[:open_idx])
                remaining = remaining[open_idx + len(_THINK_OPEN) :]
                in_think = True
    return content_chunks, think_chunks, in_think


def _partial_suffix_len(text: str, delimiter: str) -> int:
    """Length of the longest proper suffix of `text` that is a proper
    prefix of `delimiter`. Used to hold back delimiters split across
    stream chunk boundaries."""
    max_len = min(len(text), len(delimiter) - 1)
    for n in range(max_len, 0, -1):
        if text[-n:] == delimiter[:n]:
            return n
    return 0


def _classify_url_error(host: str, exc: BaseException) -> OllamaError:
    """Turn an arbitrary URLError / OSError into a user-actionable message."""
    msg = str(exc)
    lowered = msg.lower()
    if "connection refused" in lowered or "name or service not known" in lowered or "no route to host" in lowered:
        return OllamaError(
            f"Ollama at {host} is not reachable.",
            actionable=(
                f"Ollama daemon not reachable at {host}.\n"
                f"Fix:\n"
                f"  - If running locally: `ollama serve` in another shell.\n"
                f"  - If remote: set `OLLAMA_HOST=<host:port>` and confirm the daemon is listening.\n"
                f"  - To check installed models: `ollama list`.\n"
                f"Underlying error: {msg}"
            ),
        )
    return OllamaError(f"Ollama transport error: {msg}", actionable=msg)


def _classify_api_error_body(host: str, model: str, body: str) -> OllamaError:
    """Distinguish 'model not pulled' / 'context overflow' from other API errors."""
    lowered = body.lower()
    if "model" in lowered and ("not found" in lowered or "could not be loaded" in lowered):
        return OllamaError(
            f"Ollama model '{model}' is not installed.",
            actionable=(
                f"The model '{model}' isn't installed on the Ollama daemon at {host}.\n"
                f"Fix: `ollama pull {model}` (then retry).\n"
                f"To list installed models: `ollama list`."
            ),
        )
    # Ollama's actual wording is "The prompt is too long: N, model maximum
    # context length: M" — match that plus the common variants. The harness
    # reacts to this with reactive overflow recovery (compact + retry), so
    # this classification must fire for the real daemon message, not just
    # the literal "prompt too long" substring.
    if (
        "prompt too long" in lowered
        or "prompt is too long" in lowered
        or "maximum context length" in lowered
        or ("too long" in lowered and "context" in lowered)
        or ("exceed" in lowered and "context" in lowered)
    ):
        return OllamaError(
            f"Ollama context overflow for '{model}': {body[:200]}",
            actionable=(
                f"The prompt exceeds the model's context window. The harness "
                f"compactor should prevent this — check that "
                f"`/set ollama_num_ctx <n>` matches your model's real window, "
                f"or `unset ollama_num_ctx` to let the harness auto-detect it "
                f"from `/api/show`.\n"
                f"Quick recovery: `/clear` to drop history, or aggressively "
                f"lower `context_trim_threshold` (e.g. `/set context_trim_threshold 0.5`)."
            ),
        )
    return OllamaError(f"Ollama API error: {body[:300]}", actionable=body[:500])




class OllamaProvider(LLMProvider):
    API_KEY = os.getenv("OLLAMA_API_KEY")

    def __init__(
        self,
        model_name: str = "",
        host: Optional[str] = None,
        *,
        mode: str = "auto",
        api_key: Optional[str] = None,
        options: Optional[OllamaOptions] = None,
        request_timeout: float = 300.0,
    ):
        super().__init__(model_name)
        self.name = "ollama"
        self.host = _resolve_host(host, mode)
        self.connection_mode = mode
        # Cloud auth: explicit per-session key wins, else env OLLAMA_API_KEY.
        # A local daemon ignores the bearer token.
        self.api_key = api_key if api_key is not None else os.getenv("OLLAMA_API_KEY")
        self.options = options or OllamaOptions()
        self.request_timeout = float(request_timeout)
        # Cache the preflight result so we don't probe the daemon on every
        # request. Reset by `invalidate_preflight()`.
        self._preflight_done = False
        self._preflight_error: Optional[OllamaError] = None
        self._cached_models: Optional[List[str]] = None
        # Model name → trained context length (tokens), or None if unknown.
        # Populated lazily by `_fetch_context_length`.
        self._context_length_cache: Dict[str, Optional[int]] = {}
        self._model_capabilities_cache: Dict[str, tuple[str, ...]] = {}

    # ----------------------------------------------------------- preflight

    def invalidate_preflight(self) -> None:
        self._preflight_done = False
        self._preflight_error = None
        self._cached_models = None
        self._context_length_cache: Dict[str, Optional[int]] = {}
        self._model_capabilities_cache = {}

    def _fetch_runtime_capabilities(self, model_name: str) -> tuple[str, ...]:
        """Read Ollama's authoritative /api/show capability list."""
        if not model_name:
            return ()
        if model_name in self._model_capabilities_cache:
            return self._model_capabilities_cache[model_name]
        try:
            payload = json.dumps({"model": model_name}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/show",
                data=payload,
                headers={"Content-Type": "application/json", **self._auth_headers()},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            capabilities = tuple(
                str(item).strip().lower()
                for item in (value.get("capabilities") or [])
                if str(item).strip()
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            json.JSONDecodeError,
        ):
            capabilities = ()
        self._model_capabilities_cache[model_name] = capabilities
        return capabilities

    def supports_input_modality(self, modality: str) -> bool:
        target = str(modality or "").strip().lower()
        if super().supports_input_modality(target):
            return True
        # Locally discovered models may not have a pricing row. Ollama's
        # /api/show response is authoritative for its vision capability.
        return target == "image" and "vision" in self._fetch_runtime_capabilities(
            self.model_name
        )

    def supports_input_mime(self, mime_type: str, filename: str = "") -> bool:
        mime = str(mime_type or "").split(";", 1)[0].strip().lower()
        if input_modality_for_mime(mime, filename) != "image":
            return False
        return mime in {"image/jpeg", "image/png", "image/webp"} and super().supports_input_mime(
            mime, filename
        )

    def _fetch_context_length(self, model_name: str) -> Optional[int]:
        """Hit `/api/show` and return the model's trained context length, or
        None if the endpoint is unreachable / the field is missing. Cached
        per model so we only probe once per process."""
        if not model_name:
            return None
        if not hasattr(self, "_context_length_cache"):
            self._context_length_cache = {}
        if model_name in self._context_length_cache:
            return self._context_length_cache[model_name]
        try:
            payload = json.dumps({"model": model_name}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/show",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    **self._auth_headers(),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
            self._context_length_cache[model_name] = None
            return None
        # Ollama's /api/show returns a `model_info` dict whose keys are
        # namespaced by the model architecture, e.g.
        # `llama.context_length`, `qwen2.context_length`. Grab whichever
        # one is present.
        ctx_len: Optional[int] = None
        model_info = data.get("model_info") or {}
        if isinstance(model_info, dict):
            for k, v in model_info.items():
                if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                    ctx_len = v
                    break
        self._context_length_cache[model_name] = ctx_len
        return ctx_len

    def effective_response_reserve(
        self, model_name: Optional[str] = None
    ) -> Optional[int]:
        """Compactor reserve for Ollama, derived from `ollama_num_predict`.

        Resolution:
          * num_predict > 0  → that's the explicit output cap; reserve it
                                exactly so the input budget gets the rest.
          * num_predict <= 0 → Ollama's "unlimited" / model-default mode.
                                Heuristically reserve ~⅛ of the window
                                (clamped to [512, 2048]) so a long
                                multi-tool-call output still has room.
          * No num_predict knob and no window → None (fall back to var).
        """
        try:
            vars_ = getattr(self, "_session_variables", None) or {}
            raw = vars_.get("ollama_num_predict")
            if raw is not None:
                num_predict = int(raw)
                if num_predict > 0:
                    return num_predict
        except (TypeError, ValueError):
            pass
        window = self.effective_context_window(model_name)
        if not window:
            return None
        return max(512, min(2048, window // 8))

    def effective_context_window(
        self, model_name: Optional[str] = None
    ) -> Optional[int]:
        """Real input-context ceiling for the active Ollama model.

        Resolution order:
          1. `ollama_num_ctx` session variable, if > 0 — the user is
             explicitly overriding (and is responsible for sanity).
          2. The model's trained `context_length` from `/api/show`.
          3. None (caller falls back to the harness-wide default).

        The session compactor calls this on every turn so we never send
        a prompt that's larger than the model can read — preventing the
        "prompt too long; exceeded max context length" 400 from Ollama.
        """
        # Resolve from session variables first.
        try:
            vars_ = getattr(self, "_session_variables", None) or {}
            raw = vars_.get("ollama_num_ctx")
            if raw is not None:
                num_ctx = int(raw)
                if num_ctx > 0:
                    return num_ctx
        except (TypeError, ValueError):
            pass
        target_model = model_name or self.model_name
        return self._fetch_context_length(target_model)

    def preflight(self) -> None:
        """Probe the daemon and cache the result.

        Raises `OllamaError` with an actionable message on failure. Safe
        to call multiple times — only the first call hits the network.
        Tests can force a re-check via `invalidate_preflight()`.
        """
        if self._preflight_done:
            if self._preflight_error is not None:
                raise self._preflight_error
            return
        try:
            models = self._fetch_models()
        except OllamaError as exc:
            self._preflight_done = True
            self._preflight_error = exc
            raise
        self._cached_models = models
        self._preflight_done = True

    # -------------------------------------------------------- API helpers

    def _auth_headers(self) -> Dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _fetch_models(self) -> List[str]:
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags", headers=self._auth_headers()
            )
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except urllib.error.URLError as exc:
            raise _classify_url_error(self.host, exc)
        except (json.JSONDecodeError, OSError) as exc:
            raise OllamaError(
                f"Could not parse Ollama tags response from {self.host}: {exc}",
                actionable=str(exc),
            )

    def get_available_models(self) -> List[str]:
        """Return installed models. Never raises — returns [] on failure.

        For an actionable error use `preflight()` instead.
        """
        if self._cached_models is not None:
            return list(self._cached_models)
        try:
            return self._fetch_models()
        except OllamaError:
            return []

    def is_model_installed(self, model: str) -> bool:
        if not model:
            return False
        installed = self.get_available_models()
        if model in installed:
            return True
        # Ollama tags include a `:latest` suffix that's often elided.
        normalized = model.split(":", 1)[0]
        return any(m.split(":", 1)[0] == normalized for m in installed)

    # ------------------------------------------------------- message conversion

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal Message format → Ollama /api/chat shape.

        * Text parts join into `content`.
        * `image_input` parts (raw bytes) get base64-encoded into the
          message's `images` field (vision models read this).
        * `tool_call` parts become `tool_calls` on the assistant message.
        * `tool_result` parts become a `tool` role message.
        """
        ollama_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            content = ""
            tool_calls: List[Dict[str, Any]] = []
            images: List[str] = []
            returned_images: List[str] = []
            role = msg.role

            for part in msg.parts:
                if part.type == "text":
                    content += (part.text or "") + "\n"
                elif (
                    part.type == "image_input"
                    and part.image is not None
                    and self.supports_input_mime(
                        part.image.mime_type, part.image.display_name or ""
                    )
                ):
                    try:
                        encoded = base64.b64encode(part.image.data).decode("ascii")
                    except Exception:
                        encoded = ""
                    if encoded:
                        images.append(encoded)
                elif (
                    part.type == "media_input"
                    and part.media is not None
                    and self.supports_input_mime(
                        part.media.mime_type, part.media.display_name or ""
                    )
                ):
                    encoded = base64.b64encode(part.media.data).decode("ascii")
                    if encoded:
                        images.append(encoded)
                elif part.type == "tool_call":
                    tool_calls.append(
                        {
                            "function": {
                                "name": part.tool_name,
                                "arguments": part.tool_args,
                            }
                        }
                    )
                    role = "assistant"
                elif part.type == "tool_result":
                    role = "tool"
                    if isinstance(part.tool_result, (dict, list)):
                        content = json.dumps(part.tool_result, indent=2, sort_keys=True)
                    else:
                        content = str(part.tool_result)
                    for media in part.media_inputs or []:
                        if not self.supports_input_mime(
                            media.mime_type, media.display_name or ""
                        ):
                            continue
                        encoded = base64.b64encode(media.data).decode("ascii")
                        if encoded:
                            returned_images.append(encoded)

            message_dict: Dict[str, Any] = {
                "role": role,
                "content": content.strip(),
            }
            if tool_calls:
                message_dict["tool_calls"] = tool_calls
            if images:
                message_dict["images"] = images
            ollama_msgs.append(message_dict)
            if returned_images:
                ollama_msgs.append(
                    {
                        "role": "user",
                        "content": "Native image(s) returned by the preceding tool call(s).",
                        "images": returned_images,
                    }
                )
        return ollama_msgs

    def _prepare_chat_messages(
        self, messages: List[Message], system_prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Return an Ollama-compatible transcript with one leading system message.

        Several Ollama model templates (including Qwen-derived templates) reject
        a system turn anywhere but the beginning of the conversation. The
        harness can add system messages later in a run for tool/recovery
        guidance, so simply inserting ``system_prompt`` at index zero is not
        sufficient. Fold every system turn into a single preamble while
        preserving its order, then leave the non-system transcript untouched.
        """
        converted = self._convert_messages(messages)
        system_parts: List[str] = []
        if system_prompt and system_prompt.strip():
            system_parts.append(system_prompt.strip())

        conversation: List[Dict[str, Any]] = []
        for message in converted:
            if message.get("role") == "system":
                content = str(message.get("content") or "").strip()
                if content:
                    system_parts.append(content)
            else:
                conversation.append(message)

        if system_parts:
            return [{"role": "system", "content": "\n\n".join(system_parts)}, *conversation]
        return conversation

    # -------------------------------------------------- option resolution

    def _build_options(
        self,
        *,
        thinking: bool,
        reasoning_effort: Optional[str],
        session_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the `options` dict for an /api/chat call.

        Precedence (later wins):
          1. Constructor `options=` arg
          2. Session variables prefixed `ollama_<option>`
          3. `thinking=True` → mild temperature bump if not otherwise set
        """
        opts = self.options.as_payload()
        if session_variables:
            for key in _OLLAMA_OPTION_KEYS:
                var_key = f"ollama_{key}"
                if var_key in session_variables:
                    val = session_variables[var_key]
                    # Treat None / "" / 0 as "no override" so the config-
                    # registry sentinel defaults (0) don't silently force
                    # Ollama into degenerate values. Users who actually
                    # want temperature=0 can still set 0.0 via the
                    # constructor's `options=OllamaOptions(temperature=0)`.
                    if val is None or val == "" or val == 0 or val == 0.0:
                        continue
                    opts[key] = val
        if thinking and "temperature" not in opts:
            opts["temperature"] = 0.7
        # `reasoning_effort` is intentionally NOT forwarded: Ollama's native
        # `options` object has no such knob, and unknown options can be
        # rejected or silently ignored depending on the server version.
        return opts

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
        # Surface preflight errors as actionable messages instead of opaque
        # transport errors mid-stream. We only check once per provider.
        try:
            self.preflight()
        except OllamaError as exc:
            yield StreamEvent(kind="error", text=exc.actionable)
            raise

        if self.model_name and not self.is_model_installed(self.model_name):
            err = OllamaError(
                f"Ollama model '{self.model_name}' is not installed.",
                actionable=(
                    f"The model '{self.model_name}' isn't installed at {self.host}.\n"
                    f"Fix: `ollama pull {self.model_name}` then retry.\n"
                    f"Installed: {', '.join(self.get_available_models()) or '(none)'}"
                ),
            )
            yield StreamEvent(kind="error", text=err.actionable)
            raise err

        # Pull session variables off the provider if a caller set them.
        session_variables = getattr(self, "_session_variables", None)
        ollama_messages = self._prepare_chat_messages(messages, system_prompt)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": True,
            "options": self._build_options(
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                session_variables=session_variables,
            ),
        }
        # keep_alive keeps the model warm across turns.
        keep_alive = cache_hint.keep_alive_seconds if cache_hint else 600
        payload["keep_alive"] = keep_alive
        if tools:
            payload["tools"] = [
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

        headers = {"Content-Type": "application/json", **self._auth_headers()}
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )

        emitted_tool_index = 0
        last_in = 0
        last_out = 0
        saw_done = False  # only a chunk with done:true ends a valid stream
        think_carry = ""  # partial delimiter held back across chunks
        in_think = False  # state for cross-chunk <think>…</think> tracking

        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                line_buffer = ""  # partial NDJSON record across reads
                for raw in response:
                    if not raw:
                        continue
                    # NDJSON records are newline-framed; the response reader
                    # may still surface a partial record split mid-line. Buffer
                    # by newline and parse only complete records.
                    line_buffer += raw.decode("utf-8", errors="replace")
                    while "\n" in line_buffer:
                        record, line_buffer = line_buffer.split("\n", 1)
                        if not record.strip():
                            continue
                        try:
                            chunk = json.loads(record)
                        except json.JSONDecodeError as exc:
                            # A complete but malformed record is a stream
                            # error, not something to silently skip.
                            err = _classify_api_error_body(
                                self.host,
                                self.model_name,
                                f"malformed NDJSON record in stream: {exc}",
                            )
                            yield StreamEvent(kind="error", text=err.actionable)
                            raise err

                        # Ollama returns errors mid-stream as { "error": "..." }.
                        if "error" in chunk and not chunk.get("message"):
                            err = _classify_api_error_body(
                                self.host, self.model_name, str(chunk.get("error", ""))
                            )
                            yield StreamEvent(kind="error", text=err.actionable)
                            raise err

                        msg = chunk.get("message") or {}
                        content = msg.get("content") or ""
                        if content:
                            # Prepend any partial-delimiter carry held back
                            # from the previous chunk so delimiters split
                            # across chunk boundaries reassemble.
                            content = think_carry + content
                            think_carry = ""
                            raw_tail = content
                            content_parts, think_parts, in_think = _split_think_blocks(
                                content, in_think=in_think
                            )
                            hold = max(
                                _partial_suffix_len(raw_tail, _THINK_OPEN),
                                _partial_suffix_len(raw_tail, _THINK_CLOSE),
                            )
                            if hold:
                                think_carry = raw_tail[-hold:]
                            for piece in content_parts:
                                if piece:
                                    yield StreamEvent(kind="text_delta", text=piece)
                            for piece in think_parts:
                                if piece:
                                    yield StreamEvent(kind="thinking_delta", text=piece)

                        # Models may also use a structured `thinking` field —
                        # surface it identically.
                        thought = msg.get("thinking") or msg.get("reasoning")
                        if thought:
                            yield StreamEvent(kind="thinking_delta", text=str(thought))

                        for tc in msg.get("tool_calls", []) or []:
                            fn = tc.get("function") or {}
                            cid = f"ollama_call_{emitted_tool_index}"
                            emitted_tool_index += 1
                            yield StreamEvent(
                                kind="tool_call_start",
                                tool_name=fn.get("name"),
                                tool_call_id=cid,
                            )
                            yield StreamEvent(
                                kind="tool_call_complete",
                                tool_name=fn.get("name"),
                                tool_args=fn.get("arguments") or {},
                                tool_call_id=cid,
                            )

                        last_in = chunk.get("prompt_eval_count", last_in) or last_in
                        last_out = chunk.get("eval_count", last_out) or last_out
                        if chunk.get("done"):
                            saw_done = True
                            break
                    if saw_done:
                        # done:true terminates the stream - do not keep
                        # reading the outer response loop.
                        break
                # EOF: a final record without a trailing newline is still
                # a valid NDJSON record - parse it before the saw_done check.
                residue = line_buffer.strip()
                if residue and not saw_done:
                    try:
                        chunk = json.loads(residue)
                    except json.JSONDecodeError as exc:
                        err = _classify_api_error_body(
                            self.host,
                            self.model_name,
                            f"malformed NDJSON record in stream: {exc}",
                        )
                        yield StreamEvent(kind="error", text=err.actionable)
                        raise err
                    if "error" in chunk and not chunk.get("message"):
                        err = _classify_api_error_body(
                            self.host, self.model_name, str(chunk.get("error", ""))
                        )
                        yield StreamEvent(kind="error", text=err.actionable)
                        raise err
                    if chunk.get("done"):
                        saw_done = True
                if think_carry:
                    if in_think:
                        yield StreamEvent(kind="thinking_delta", text=think_carry)
                    else:
                        yield StreamEvent(kind="text_delta", text=think_carry)
                    think_carry = ""
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                pass
            err = _classify_api_error_body(self.host, self.model_name, body or str(exc))
            yield StreamEvent(kind="error", text=err.actionable)
            raise err
        except urllib.error.URLError as exc:
            err = _classify_url_error(self.host, exc)
            yield StreamEvent(kind="error", text=err.actionable)
            raise err

        if not saw_done:
            # Truncated body (proxy cut, connection died cleanly). Never
            # present a partial answer as a successful completion.
            err = _classify_api_error_body(
                self.host, self.model_name, "stream ended without done marker"
            )
            yield StreamEvent(kind="error", text=err.actionable)
            raise err

        yield StreamEvent(
            kind="usage",
            input_tokens=last_in,
            output_tokens=last_out,
            total_tokens=last_in + last_out,
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

    def upload_file(self, file_path: str, mime_type: str) -> Optional[FileReference]:
        return FileReference(uri=file_path, mime_type=mime_type, display_name=file_path)

    # ------------------------------------------------------ session helper

    def bind_session_variables(self, variables: Dict[str, Any]) -> None:
        """Wire a session's variables dict so the provider can read
        `ollama_*` overrides on each call. Called by the harness when
        constructing the provider; safe to omit in standalone use.
        """
        self._session_variables = variables

    def apply_session_host(self, variables: Dict[str, Any]) -> None:
        """Recompute ``self.host`` / ``self.api_key`` from the session's
        ``ollama_host`` / ``ollama_mode`` / ``ollama_api_key`` variables and
        invalidate the preflight cache. Called by the harness
        (`mucli.sync_provider_settings`) on provider construction and whenever
        one of those variables changes via `/set` or the GUI.

        Priority: explicit ``ollama_host`` wins; otherwise ``ollama_mode``
        selects local (env/localhost) vs cloud (ollama.com) vs auto (legacy
        env-driven). The API key falls back to ``OLLAMA_API_KEY`` env.
        """
        host_var = variables.get("ollama_host") or None
        mode = variables.get("ollama_mode") or "auto"
        self.connection_mode = mode
        self.host = _resolve_host(host_var, mode)
        key = variables.get("ollama_api_key")
        self.api_key = key if key else os.getenv("OLLAMA_API_KEY")
        self.invalidate_preflight()

    def apply_session_variables(self, variables: Dict[str, Any]) -> None:
        """One-call wiring: bind the variables dict AND recompute host/key."""
        self._session_variables = variables
        self.apply_session_host(variables)

    def compaction_safety_factor(self) -> float:
        """Ollama's cl100k_base estimate under-counts the model's real
        tokenizer (observed ~2.2x for qwen-class models), and the streamed
        ``prompt_eval_count`` only reflects the non-cached prompt delta — so
        the compactor must target a reduced limit or the real prompt
        overflows before compaction fires (the "prompt is too long" 400).

        Tunable via the ``ollama_token_safety_factor`` session variable
        (``/set ollama_token_safety_factor <n>``); default 2.5 gives ~20%
        headroom beyond the observed drift. Set to 1.0 to disable.
        """
        try:
            vars_ = getattr(self, "_session_variables", None) or {}
            raw = vars_.get("ollama_token_safety_factor")
            if raw is not None:
                f = float(raw)
                if f > 0:
                    return f
        except (TypeError, ValueError):
            pass
        return 2.5


__all__ = [
    "OllamaError",
    "OllamaOptions",
    "OllamaProvider",
]
