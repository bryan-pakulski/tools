"""Pin the provider-aware compaction-budget pipeline.

The bug this regression-tests: previously the compactor used a hardcoded
`context_token_limit` (default 256000), which is fine for Claude/Gemini
but silently overcommits on Ollama models that have 4k-32k real
windows. The session would happily build a 60k-token prompt for an 8k
model and Ollama would 400 with "prompt too long; exceeded max context
length".

Fix: `LLMProvider.effective_context_window()` lets each provider
declare its real ceiling, and `Session._compaction_token_budget()`
takes the min of (user-set limit, provider window) minus a response
reserve.
"""

from typing import Optional

import pytest

from mu.session.session import Session, SessionManager
from providers.base import LLMProvider, ProviderResponse


class _BaseDummy(LLMProvider):
    def get_available_models(self):
        return ["dummy"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        return ProviderResponse(
            text="ok", parts=[], input_tokens=0, output_tokens=0, total_tokens=0
        )

    def upload_file(self, file_path, mime_type):
        return None


class _UnboundedProvider(_BaseDummy):
    """Returns None from effective_context_window — same as the base
    default. Compactor should fall back to user-set context_token_limit."""


class _SmallWindowProvider(_BaseDummy):
    """Mimics an 8k Ollama model."""

    def effective_context_window(self, model_name: Optional[str] = None):
        return 8192


def _session(provider):
    sm = SessionManager()
    return Session(provider, False, "system instruction", sm)


def _isolate_compaction_math(session, monkeypatch):
    """Stub the non-L5 layer estimate to 0 so the compaction-budget math
    can be exercised in isolation from layer accounting. The
    layer-aware behavior is covered by tests/test_token_estimator.py."""
    monkeypatch.setattr(
        "utils.runtime_metrics.estimate_non_l5_context_tokens",
        lambda _session: 0,
    )


def test_provider_window_none_falls_back_to_user_limit():
    session = _session(_UnboundedProvider("dummy"))
    session.variables["context_token_limit"] = 100_000
    session.variables["response_token_reserve"] = 0
    session.variables["context_trim_threshold"] = 1.0
    assert session._resolve_context_limit() == 100_000


def test_provider_window_caps_user_limit():
    """If the provider says 8k but the user set 256k, we honor 8k. The
    user setting is a software ceiling; the provider's is hardware."""
    session = _session(_SmallWindowProvider("dummy"))
    session.variables["context_token_limit"] = 256_000
    assert session._resolve_context_limit() == 8192


def test_user_limit_can_go_lower_than_provider():
    """If the user wants to be conservative, that wins."""
    session = _session(_SmallWindowProvider("dummy"))
    session.variables["context_token_limit"] = 4096
    assert session._resolve_context_limit() == 4096


def test_compaction_budget_subtracts_response_reserve(monkeypatch):
    """The budget compactor targets must leave headroom for the model's
    own output — otherwise we pack the input to the edge and there's no
    room left to generate."""
    session = _session(_SmallWindowProvider("dummy"))
    _isolate_compaction_math(session, monkeypatch)
    session.variables["context_token_limit"] = 256_000
    session.variables["context_trim_threshold"] = 1.0
    session.variables["response_token_reserve"] = 2048
    # 8192 (provider) - 2048 (reserve) - 0 (non-L5) = 6144 usable, threshold=1.0.
    assert session._compaction_token_budget() == 6144


def test_compaction_budget_applies_trim_threshold(monkeypatch):
    session = _session(_SmallWindowProvider("dummy"))
    _isolate_compaction_math(session, monkeypatch)
    session.variables["context_token_limit"] = 256_000
    session.variables["context_trim_threshold"] = 0.5
    session.variables["response_token_reserve"] = 0
    # 8192 * 0.5 = 4096
    assert session._compaction_token_budget() == 4096


def test_compaction_budget_zero_when_no_capacity():
    """When reserve + non-L5 layers exceed the window, there is no real
    history capacity — the budget must be 0 (not a fabricated floor), so
    callers report the condition instead of looping (codex round-7 F6)."""
    session = _session(_SmallWindowProvider("dummy"))
    session.variables["context_token_limit"] = 1024
    session.variables["context_trim_threshold"] = 0.10
    session.variables["response_token_reserve"] = 100_000
    assert session._compaction_token_budget() == 0


def test_provider_window_exception_is_swallowed():
    """If a provider's effective_context_window raises, we fall back —
    never let a transient probe failure brick the compactor."""

    class _Flaky(_BaseDummy):
        def effective_context_window(self, model_name=None):
            raise RuntimeError("oops")

    session = _session(_Flaky("dummy"))
    session.variables["context_token_limit"] = 100_000
    assert session._resolve_context_limit() == 100_000


# --------------------------------------------------------- Ollama-specific


def test_ollama_effective_context_uses_num_ctx_override_when_set():
    """When `ollama_num_ctx` is set, the harness honors it directly
    without probing the daemon."""
    from providers.ollama import OllamaProvider

    provider = OllamaProvider("llama3", host="http://localhost:11434")
    provider.bind_session_variables({"ollama_num_ctx": 16384})
    assert provider.effective_context_window("llama3") == 16384


def test_ollama_effective_context_returns_none_when_unreachable(monkeypatch):
    """A daemon probe failure must not crash compaction — return None
    so the caller falls back to the user-set context_token_limit."""
    from providers.ollama import OllamaProvider
    import urllib.error

    provider = OllamaProvider("nonexistent-model", host="http://localhost:1")
    provider.bind_session_variables({"ollama_num_ctx": 0})

    def _fail(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _fail)
    assert provider.effective_context_window("nonexistent-model") is None


def test_ollama_effective_context_parses_model_info_context_length(monkeypatch):
    """The `/api/show` response has architecture-namespaced keys like
    `qwen2.context_length`. We must pick whichever one is present."""
    from providers.ollama import OllamaProvider
    import io
    import json

    provider = OllamaProvider("qwen2.5", host="http://localhost:11434")
    provider.bind_session_variables({"ollama_num_ctx": 0})

    body = json.dumps(
        {"model_info": {"qwen2.context_length": 32768, "qwen2.embedding_length": 4096}}
    ).encode("utf-8")

    class _FakeResp:
        def __init__(self, b): self._b = b
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: _FakeResp(body)
    )
    assert provider.effective_context_window("qwen2.5") == 32768


# ------------------------------------------------- compaction safety factor


class _DriftyProvider(_SmallWindowProvider):
    """A provider whose real tokenizer diverges from cl100k_base — models
    the Ollama case where the harness estimate under-counts the real prompt."""

    def compaction_safety_factor(self):
        return 2.5


def test_safety_factor_reduces_compaction_limit():
    """A provider with compaction_safety_factor > 1 gets its limit divided,
    so the compactor fires before the (under-counted) cl100k estimate would
    otherwise suggest — preventing the Ollama 'prompt is too long' overflow."""
    session = _session(_DriftyProvider("dummy"))
    session.variables["context_token_limit"] = 100_000
    # provider window 8192, then / 2.5 -> 3276
    assert session._resolve_context_limit() == 3276


def test_safety_factor_one_is_a_noop():
    """factor == 1.0 (the base default) must not change the limit — frontier
    providers that report true input_tokens are trusted verbatim."""
    session = _session(_SmallWindowProvider("dummy"))
    session.variables["context_token_limit"] = 100_000
    assert session._resolve_context_limit() == 8192


def test_ollama_safety_factor_default_and_override():
    """Ollama defaults to 2.5 (observed cl100k→real drift for qwen-class),
    and is tunable via the ollama_token_safety_factor session variable."""
    from providers.ollama import OllamaProvider

    provider = OllamaProvider("qwen3", host="http://localhost:11434")
    provider.bind_session_variables({})
    assert provider.compaction_safety_factor() == 2.5

    provider.bind_session_variables({"ollama_token_safety_factor": 1.0})
    assert provider.compaction_safety_factor() == 1.0

    provider.bind_session_variables({"ollama_token_safety_factor": 3.0})
    assert provider.compaction_safety_factor() == 3.0


def test_safety_factor_compaction_budget_tightens(monkeypatch):
    """The reduced limit flows through to the L5 compaction budget so the
    normal compaction pass (not just the emergency pre-flight) triggers
    earlier for drift-prone providers."""
    session = _session(_DriftyProvider("dummy"))
    _isolate_compaction_math(session, monkeypatch)
    session.variables["context_token_limit"] = 100_000
    session.variables["context_trim_threshold"] = 1.0
    session.variables["response_token_reserve"] = 0
    # 8192 / 2.5 = 3276 usable (non_l5 stubbed to 0), threshold 1.0
    assert session._compaction_token_budget() == 3276


def test_ollama_effective_context_caches_per_model(monkeypatch):
    """Don't hit the daemon every turn — cache per model."""
    from providers.ollama import OllamaProvider
    import io
    import json

    provider = OllamaProvider("llama3", host="http://localhost:11434")
    provider.bind_session_variables({"ollama_num_ctx": 0})

    call_count = {"n": 0}
    body = json.dumps({"model_info": {"llama.context_length": 8192}}).encode("utf-8")

    class _FakeResp:
        def __init__(self, b): self._b = b
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _spy(req, timeout=None):
        call_count["n"] += 1
        return _FakeResp(body)

    monkeypatch.setattr("urllib.request.urlopen", _spy)
    provider.effective_context_window("llama3")
    provider.effective_context_window("llama3")
    provider.effective_context_window("llama3")
    assert call_count["n"] == 1


def test_provider_reserve_overrides_session_variable(monkeypatch):
    """When the provider declares a real output cap (e.g. Ollama's
    `num_predict=512`), the compactor must honor that — not the
    user-set `response_token_reserve` default."""

    class _ProviderWithReserve(_BaseDummy):
        def effective_context_window(self, model_name=None):
            return 8192

        def effective_response_reserve(self, model_name=None):
            return 512

    session = _session(_ProviderWithReserve("dummy"))
    _isolate_compaction_math(session, monkeypatch)
    session.variables["response_token_reserve"] = 99999  # ignored
    session.variables["context_trim_threshold"] = 1.0
    # 8192 - 512 (provider reserve) - 0 (non-L5) = 7680, threshold = 1.0
    assert session._resolve_response_reserve() == 512
    assert session._compaction_token_budget() == 7680


def test_session_variable_fallback_when_provider_returns_none():
    """If the provider doesn't know its output cap, the session falls
    back to the `response_token_reserve` variable. This preserves the
    pre-Option-1 behavior for OpenAI / Gemini until they're wired up."""

    class _NoneReserve(_BaseDummy):
        def effective_context_window(self, model_name=None):
            return 32768

        def effective_response_reserve(self, model_name=None):
            return None

    session = _session(_NoneReserve("dummy"))
    session.variables["response_token_reserve"] = 2048
    assert session._resolve_response_reserve() == 2048


def test_provider_reserve_exception_falls_back_safely():
    """A provider that raises in `effective_response_reserve` must not
    crash the compactor — fall back to the session variable."""

    class _Flaky(_BaseDummy):
        def effective_response_reserve(self, model_name=None):
            raise RuntimeError("oops")

    session = _session(_Flaky("dummy"))
    session.variables["response_token_reserve"] = 1024
    assert session._resolve_response_reserve() == 1024


def test_ollama_reserve_honors_num_predict_when_set():
    """An explicit `ollama_num_predict` is the user's contract; the
    compactor reserves exactly that — no more, no less."""
    from providers.ollama import OllamaProvider

    provider = OllamaProvider("llama3", host="http://localhost:11434")
    provider.bind_session_variables(
        {"ollama_num_predict": 1500, "ollama_num_ctx": 16384}
    )
    assert provider.effective_response_reserve("llama3") == 1500


def test_ollama_reserve_heuristic_when_num_predict_is_zero():
    """`ollama_num_predict=0` means 'use model default' (i.e. potentially
    unlimited). We must pick a sane heuristic — ⅛ of the window, clamped
    to [512, 2048] — instead of leaving zero room for output."""
    from providers.ollama import OllamaProvider

    provider = OllamaProvider("llama3", host="http://localhost:11434")
    provider.bind_session_variables(
        {"ollama_num_predict": 0, "ollama_num_ctx": 16384}
    )
    # 16384 // 8 = 2048 (hits the upper clamp).
    assert provider.effective_response_reserve("llama3") == 2048


def test_ollama_reserve_heuristic_clamped_for_tiny_window():
    """For pathologically small windows (2k), the heuristic still
    reserves at least 512 — small but enough for a one-paragraph reply."""
    from providers.ollama import OllamaProvider

    provider = OllamaProvider("llama3", host="http://localhost:11434")
    provider.bind_session_variables(
        {"ollama_num_predict": 0, "ollama_num_ctx": 2048}
    )
    # 2048 // 8 = 256, clamped up to 512.
    assert provider.effective_response_reserve("llama3") == 512


def test_ollama_reserve_returns_none_when_window_unknown(monkeypatch):
    """If we can't determine the window AND num_predict isn't set,
    return None so the caller falls back to the session variable."""
    from providers.ollama import OllamaProvider
    import urllib.error

    provider = OllamaProvider("unknown", host="http://localhost:1")
    provider.bind_session_variables({"ollama_num_predict": 0, "ollama_num_ctx": 0})
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(
            urllib.error.URLError("connection refused")
        ),
    )
    assert provider.effective_response_reserve("unknown") is None


def test_ollama_context_overflow_error_classified_with_actionable_hint():
    """A "prompt too long" 400 body must be classified as a typed
    OllamaError with an actionable message — not the generic "API error"
    blob the user can't act on."""
    from providers.ollama import _classify_api_error_body

    err = _classify_api_error_body(
        host="http://localhost:11434",
        model="llama3",
        body=(
            '{"error":"prompt too long; exceeded max context length by '
            '2401 tokens (ref: 9d8b)"}'
        ),
    )
    assert "context overflow" in str(err).lower()
    assert "ollama_num_ctx" in err.actionable
    assert "context_trim_threshold" in err.actionable
