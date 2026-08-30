"""Tests for the upgraded Ollama provider.

Coverage:
  * Host resolution: OLLAMA_HOST env, OLLAMA_API_KEY hosted fallback,
    localhost default, explicit-arg precedence, scheme normalization.
  * Preflight: caches success; surfaces actionable error for connection-
    refused; surfaces actionable error for model-not-pulled.
  * Message conversion: image_input → base64 in `images`; tool calls and
    tool results unchanged from prior behavior.
  * Think-tag parsing: stateful across chunk boundaries; mid-chunk open
    and close; multiple blocks; no-think passthrough.
  * Native options: constructor `options=` applied; session variables
    override; 0 / "" / None treated as "no override".
  * /ollama command: status, models, options subcommands.
"""

import io
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mu.commands.ollama import ollama_cmd
from providers.base import ImageData, Message, MessagePart
from providers.ollama import (
    OllamaError,
    OllamaOptions,
    OllamaProvider,
    _resolve_host,
    _split_think_blocks,
)


# ============================================================ host resolution


def test_resolve_host_explicit_wins(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert _resolve_host("http://example.com:9999") == "http://example.com:9999"


def test_resolve_host_env_var_used_when_no_explicit(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "ollama.lan:11434")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    # Scheme auto-prepended.
    assert _resolve_host() == "http://ollama.lan:11434"


def test_resolve_host_https_scheme_preserved(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.example.com")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert _resolve_host() == "https://ollama.example.com"


def test_resolve_host_hosted_when_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    assert _resolve_host() == "https://ollama.com"


def test_resolve_host_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert _resolve_host() == "http://localhost:11434"


def test_constructor_local_mode_ignores_cloud_api_key(monkeypatch):
    """Explicit local construction must not be redirected by cloud env auth."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "cloud-key")

    provider = OllamaProvider("local-model", mode="local")

    assert provider.host == "http://localhost:11434"


def test_resolve_host_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://foo:1234/")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert _resolve_host() == "http://foo:1234"


# ============================================================ think-tag parser


def test_split_think_blocks_no_think():
    content, think, in_think = _split_think_blocks("hello world", in_think=False)
    assert content == ["hello world"]
    assert think == []
    assert in_think is False


def test_split_think_blocks_full_block_in_one_chunk():
    content, think, in_think = _split_think_blocks(
        "before<think>plan</think>after", in_think=False
    )
    assert content == ["before", "after"]
    assert think == ["plan"]
    assert in_think is False


def test_split_think_blocks_open_only_keeps_state():
    content, think, in_think = _split_think_blocks(
        "before<think>partial", in_think=False
    )
    assert content == ["before"]
    assert think == ["partial"]
    assert in_think is True


def test_split_think_blocks_close_only_when_in_think_state():
    content, think, in_think = _split_think_blocks(
        "remaining</think>after", in_think=True
    )
    assert content == ["after"]
    assert think == ["remaining"]
    assert in_think is False


def test_split_think_blocks_multiple_blocks():
    content, think, in_think = _split_think_blocks(
        "a<think>x</think>b<think>y</think>c", in_think=False
    )
    assert content == ["a", "b", "c"]
    assert think == ["x", "y"]
    assert in_think is False


def test_split_think_blocks_pure_think_only():
    content, think, in_think = _split_think_blocks("<think>only</think>", in_think=False)
    assert content == []
    assert think == ["only"]
    assert in_think is False


# ============================================================ vision


def test_image_input_part_becomes_base64_in_images_field():
    provider = OllamaProvider("llava")
    img = ImageData(data=b"\x89PNG\r\n\x1a\nfake", mime_type="image/png")
    msgs = [
        Message(
            role="user",
            parts=[
                MessagePart(type="text", text="describe this"),
                MessagePart(type="image_input", image=img),
            ],
        )
    ]
    converted = provider._convert_messages(msgs)
    assert len(converted) == 1
    entry = converted[0]
    assert entry["role"] == "user"
    assert "describe this" in entry["content"]
    assert "images" in entry
    assert len(entry["images"]) == 1
    import base64

    decoded = base64.b64decode(entry["images"][0])
    assert decoded == b"\x89PNG\r\n\x1a\nfake"


def test_image_input_without_image_data_is_skipped():
    """`image_input` part with `image=None` should not crash and should
    not produce an empty entry in `images`."""
    provider = OllamaProvider("llava")
    msgs = [
        Message(
            role="user",
            parts=[MessagePart(type="image_input", image=None)],
        )
    ]
    converted = provider._convert_messages(msgs)
    assert converted == [{"role": "user", "content": ""}]


def test_prepare_chat_messages_merges_late_system_messages_into_preamble():
    """Some Ollama templates reject system messages after conversation turns.
    The provider must retain that guidance without sending an invalid shape."""
    provider = OllamaProvider("qwen")
    messages = [
        Message(role="user", parts=[MessagePart(type="text", text="first")]),
        Message(role="system", parts=[MessagePart(type="text", text="late guidance")]),
        Message(role="assistant", parts=[MessagePart(type="text", text="reply")]),
    ]

    prepared = provider._prepare_chat_messages(messages, "base guidance")

    assert [message["role"] for message in prepared] == ["system", "user", "assistant"]
    assert prepared[0]["content"] == "base guidance\n\nlate guidance"


# ============================================================ options


def test_options_from_constructor_applied():
    opts = OllamaOptions(num_ctx=16384, temperature=0.3, top_p=0.9)
    provider = OllamaProvider("x", options=opts)
    payload = provider._build_options(thinking=False, reasoning_effort=None)
    assert payload["num_ctx"] == 16384
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.9


def test_options_skip_none_fields():
    """OllamaOptions with everything None should produce an empty payload."""
    opts = OllamaOptions()
    provider = OllamaProvider("x", options=opts)
    assert provider._build_options(thinking=False, reasoning_effort=None) == {}


def test_session_variables_override_options():
    opts = OllamaOptions(num_ctx=4096)
    provider = OllamaProvider("x", options=opts)
    payload = provider._build_options(
        thinking=False,
        reasoning_effort=None,
        session_variables={"ollama_num_ctx": 32768, "ollama_temperature": 0.1},
    )
    assert payload["num_ctx"] == 32768  # session override beat constructor
    assert payload["temperature"] == 0.1


def test_session_variable_sentinel_zero_does_not_override():
    """Default config sentinels are 0 — they must NOT clobber Ollama's own
    defaults. The user has to explicitly set a non-zero value."""
    opts = OllamaOptions(num_ctx=8192)
    provider = OllamaProvider("x", options=opts)
    payload = provider._build_options(
        thinking=False,
        reasoning_effort=None,
        session_variables={"ollama_num_ctx": 0, "ollama_temperature": 0.0},
    )
    assert payload["num_ctx"] == 8192  # constructor preserved
    assert "temperature" not in payload  # zero sentinel ignored


def test_thinking_adds_temperature_when_unset():
    opts = OllamaOptions()  # no temperature
    provider = OllamaProvider("x", options=opts)
    payload = provider._build_options(thinking=True, reasoning_effort=None)
    assert payload.get("temperature") == 0.7


def test_thinking_does_not_clobber_explicit_temperature():
    opts = OllamaOptions(temperature=0.2)
    provider = OllamaProvider("x", options=opts)
    payload = provider._build_options(thinking=True, reasoning_effort=None)
    assert payload["temperature"] == 0.2


# ============================================================ preflight


def _mock_urlopen_factory(body_bytes, status=200):
    """Build a context-manager response object compatible with urllib.request.urlopen."""
    response = MagicMock()
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    response.read.return_value = body_bytes
    response.status = status
    # Stream path iterates NDJSON lines; a mock that iterates as empty
    # would previously be masked by the provider fabricating a `done`
    # event. Emit real lines so truncated-stream assertions are honest.
    response.__iter__ = MagicMock(
        return_value=iter(body_bytes.splitlines(keepends=True))
    )
    return response


def test_preflight_caches_success(monkeypatch):
    body = json.dumps({"models": [{"name": "llama3"}, {"name": "qwen3"}]}).encode("utf-8")
    call_count = {"n": 0}

    def fake_urlopen(*args, **kwargs):
        call_count["n"] += 1
        return _mock_urlopen_factory(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider("llama3", host="http://fake:1234")

    provider.preflight()
    provider.preflight()  # second call is cached
    assert call_count["n"] == 1
    assert provider.is_model_installed("llama3") is True
    assert provider.is_model_installed("qwen3") is True


def test_preflight_actionable_message_on_connection_refused(monkeypatch):
    import urllib.error

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("[Errno 111] Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider("llama3", host="http://localhost:11434")
    with pytest.raises(OllamaError) as exc_info:
        provider.preflight()
    msg = exc_info.value.actionable
    assert "not reachable" in msg
    assert "ollama serve" in msg
    assert "http://localhost:11434" in msg


def test_get_available_models_returns_empty_on_failure(monkeypatch):
    import urllib.error

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider("llama3")
    assert provider.get_available_models() == []


def test_is_model_installed_handles_latest_suffix(monkeypatch):
    body = json.dumps({"models": [{"name": "llama3:latest"}]}).encode("utf-8")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _mock_urlopen_factory(body),
    )
    provider = OllamaProvider("llama3")
    provider.preflight()
    # Both "llama3" and "llama3:latest" should match.
    assert provider.is_model_installed("llama3") is True
    assert provider.is_model_installed("llama3:latest") is True


# ============================================================ error classification


def test_classify_model_not_pulled_error():
    from providers.ollama import _classify_api_error_body

    err = _classify_api_error_body(
        "http://localhost:11434",
        "qwen3",
        '{"error": "model \\"qwen3\\" not found, try pulling it first"}',
    )
    assert isinstance(err, OllamaError)
    assert "qwen3" in err.actionable
    assert "ollama pull" in err.actionable


# ============================================================ /ollama command


class _RecordingUI:
    def __init__(self):
        self.info_calls = []
        self.error_calls = []
    def show_info(self, m):
        self.info_calls.append(str(m))
    def show_error(self, m):
        self.error_calls.append(str(m))


class _SessionStub:
    def __init__(self, provider):
        self.provider = provider
        self.ui = _RecordingUI()
        self.variables = {}


def test_ollama_command_status_when_not_using_ollama():
    class _NotOllama:
        name = "openai"

    sess = _SessionStub(_NotOllama())
    result = ollama_cmd(sess, "", allow_prompt=True)
    assert result.ok is False
    assert "not using the Ollama" in result.message


def test_ollama_command_status_when_using_ollama(monkeypatch):
    body = json.dumps({"models": [{"name": "llama3"}]}).encode("utf-8")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _mock_urlopen_factory(body),
    )
    provider = OllamaProvider("llama3", host="http://fake:1234")
    sess = _SessionStub(provider)
    result = ollama_cmd(sess, "status", allow_prompt=True)
    assert result.ok is True
    assert "llama3" in result.message
    assert "http://fake:1234" in result.message


def test_ollama_command_models_subcommand(monkeypatch):
    body = json.dumps({"models": [{"name": "a"}, {"name": "b"}]}).encode("utf-8")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _mock_urlopen_factory(body),
    )
    provider = OllamaProvider("a")
    sess = _SessionStub(provider)
    result = ollama_cmd(sess, "models", allow_prompt=True)
    assert result.ok is True
    assert "a" in result.data["models"]
    assert "b" in result.data["models"]


def test_ollama_command_options_subcommand():
    provider = OllamaProvider("llama3")
    sess = _SessionStub(provider)
    sess.variables["ollama_num_ctx"] = 32768
    result = ollama_cmd(sess, "options", allow_prompt=True)
    assert result.ok is True
    assert "32768" in result.message
    assert "ollama_temperature" in result.message  # all keys listed


def test_ollama_command_pull_missing_arg():
    provider = OllamaProvider("llama3")
    sess = _SessionStub(provider)
    result = ollama_cmd(sess, "pull", allow_prompt=True)
    assert result.ok is False
    assert "Usage" in result.message


def test_ollama_command_unknown_subcommand():
    provider = OllamaProvider("llama3")
    sess = _SessionStub(provider)
    result = ollama_cmd(sess, "wat", allow_prompt=True)
    assert result.ok is False
    assert "Unknown subcommand" in result.message


# ============================================================ session variables binding


def test_bind_session_variables_threads_through_to_stream(monkeypatch):
    """When the provider has bound session variables, they should appear in
    the payload's `options` dict."""
    sent_payloads = []

    def fake_urlopen(req, **kw):
        # Detect whether this is the /api/tags preflight or /api/chat call.
        if hasattr(req, "data") and req.data:
            sent_payloads.append(json.loads(req.data.decode("utf-8")))
            # Return a minimal NDJSON stream that exits cleanly.
            body = (
                json.dumps({"message": {"content": "ok"}, "done": True, "eval_count": 0, "prompt_eval_count": 0}).encode(
                    "utf-8"
                )
                + b"\n"
            )
        else:
            body = json.dumps({"models": [{"name": "llama3"}]}).encode("utf-8")
        return _mock_urlopen_factory(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider("llama3", host="http://fake:1234")
    provider.bind_session_variables({"ollama_num_ctx": 16384, "ollama_temperature": 0.2})

    events = list(
        provider.stream(messages=[Message(role="user", parts=[MessagePart(type="text", text="hi")])])
    )
    assert sent_payloads, "no /api/chat call was made"
    options = sent_payloads[0]["options"]
    assert options["num_ctx"] == 16384
    assert options["temperature"] == 0.2
    # And a `done` event was emitted last.
    assert events[-1].kind == "done"


# ===================================================== mode-aware host resolution


def test_resolve_host_mode_local_uses_env_or_localhost(monkeypatch):
    """mode='local' = OLLAMA_HOST env if set, else localhost — and must NOT
    auto-switch to cloud even when OLLAMA_API_KEY is set (the legacy
    auto-cloud is suppressed in local mode)."""
    monkeypatch.setenv("OLLAMA_API_KEY", "x")  # would trigger cloud in 'auto'
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert _resolve_host(None, "local") == "http://localhost:11434"

    monkeypatch.setenv("OLLAMA_HOST", "ollama.lan:11434")
    assert _resolve_host(None, "local") == "http://ollama.lan:11434"


def test_resolve_host_mode_cloud_is_ollama_com(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert _resolve_host(None, "cloud") == "https://ollama.com"


def test_resolve_host_explicit_host_beats_mode(monkeypatch):
    """An explicit `ollama_host` override wins over mode (custom daemon)."""
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    assert _resolve_host("http://my.daemon:11434", "cloud") == "http://my.daemon:11434"
    assert _resolve_host("http://my.daemon:11434", "local") == "http://my.daemon:11434"


def test_resolve_host_auto_legacy_env_host_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "ollama.lan:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    # auto: OLLAMA_HOST takes priority over the API-key cloud switch.
    assert _resolve_host(None, "auto") == "http://ollama.lan:11434"


def test_resolve_host_auto_legacy_api_key_cloud(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    assert _resolve_host(None, "auto") == "https://ollama.com"


def test_interactive_ollama_auto_mode_prompts_before_model_discovery(monkeypatch):
    """The saved ``auto`` default must not skip the local/cloud picker."""
    import mucli

    provider = SimpleNamespace(model_name="", get_available_models=lambda: ["local-model"])
    captured = {}

    def fake_init(name, model_name, host, mode, api_key):
        captured.update(name=name, model_name=model_name, host=host, mode=mode, api_key=api_key)
        return provider

    monkeypatch.setattr(mucli, "init_provider", fake_init)
    answers = iter(["local", "local-model"])
    monkeypatch.setattr(mucli, "prompt_choice", lambda *args, **kwargs: next(answers))

    selected = mucli.select_provider_and_model(
        "ollama", None, ollama_mode="auto", allow_prompt=True
    )

    assert captured["mode"] == "local"
    assert selected.model_name == "local-model"
    assert selected._mu_ollama_mode == "local"


# ============================================== api_key precedence + apply_session


def test_provider_api_key_arg_beats_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    p = OllamaProvider("llama3", api_key="explicit-key")
    assert p.api_key == "explicit-key"
    assert p._auth_headers() == {"Authorization": "Bearer explicit-key"}


def test_provider_api_key_defaults_to_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    p = OllamaProvider("llama3")
    assert p.api_key == "env-key"
    assert p._auth_headers() == {"Authorization": "Bearer env-key"}


def test_provider_no_api_key_no_auth_header(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    p = OllamaProvider("llama3")
    assert p.api_key in (None, "")
    assert p._auth_headers() == {}


def test_apply_session_host_cloud_sets_host_and_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    p = OllamaProvider("llama3")
    p.apply_session_host(
        {"ollama_mode": "cloud", "ollama_api_key": "sess-key", "ollama_host": ""}
    )
    assert p.host == "https://ollama.com"
    assert p.api_key == "sess-key"
    assert p._auth_headers() == {"Authorization": "Bearer sess-key"}


def test_apply_session_host_local_suppresses_env_cloud(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")  # would cloud in auto
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    p = OllamaProvider("llama3")
    p.apply_session_host({"ollama_mode": "local", "ollama_api_key": "", "ollama_host": ""})
    assert p.host == "http://localhost:11434"
    # local mode keeps the env key for auth if present (key is orthogonal
    # to host), but the host is NOT ollama.com.
    assert p.host != "https://ollama.com"


def test_apply_session_host_explicit_ollama_host_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    p = OllamaProvider("llama3")
    p.apply_session_host(
        {"ollama_mode": "cloud", "ollama_api_key": "", "ollama_host": "http://custom:11434"}
    )
    assert p.host == "http://custom:11434"


def test_apply_session_variables_binds_and_resolves(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    p = OllamaProvider("llama3")
    p.apply_session_variables(
        {
            "ollama_mode": "cloud",
            "ollama_api_key": "sess-key",
            "ollama_host": "",
            "ollama_num_ctx": 32768,
        }
    )
    assert p.host == "https://ollama.com"
    assert p.api_key == "sess-key"
    # bind_session_variables was threaded too — num_ctx is on the options.
    assert p._session_variables["ollama_num_ctx"] == 32768
