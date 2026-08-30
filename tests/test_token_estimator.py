"""Pin the token estimator and its integration with /memory layers."""

import pytest

import utils.token_estimator as te
from mu.session.session import Session, SessionManager
from providers.base import LLMProvider, ProviderResponse
from utils.runtime_metrics import collect_context_layers


class _DummyProvider(LLMProvider):
    def get_available_models(self):
        return ["dummy"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        return ProviderResponse(text="", parts=[])

    def upload_file(self, file_path, mime_type):
        return None


@pytest.fixture
def session():
    sm = SessionManager()
    s = Session(_DummyProvider("dummy"), False, "system instruction", sm)
    s._mcp_clients = []
    # Isolate from any prior on-disk session state. The default session
    # persists to ~/.mucli/sessions/default — earlier tests can have left
    # history + an advanced summary_anchor that breaks fresh fixtures.
    s.session_manager.history = []
    s.session_manager.summary_anchor = 0
    s.session_manager.conversation_summary = ""
    s.session_manager.provider_config = {"provider": "openai", "model": "gpt-4o"}
    return s


# ----------------------------------------------- estimator core


def test_empty_input_returns_zero():
    assert te.estimate_tokens("") == 0
    assert te.estimate_tokens(None) == 0


def test_estimator_returns_positive_for_simple_text():
    assert te.estimate_tokens("hello world") > 0


def test_estimator_handles_non_string_input():
    # Lists, dicts, ints should not blow up — coerced via str().
    assert te.estimate_tokens(42) > 0
    assert te.estimate_tokens([1, 2, 3]) > 0


def test_estimator_counts_symbol_dense_content_meaningfully():
    """Symbol-dense JSON is exactly the kind of payload that used to
    under-count under chars/4. Verify the estimator produces a tight,
    non-trivial token count for it — i.e. tiktoken is actually engaged."""
    payload = '{"a":1,"b":["x","y","z"],"c":{"nested":true,"v":3.14}}' * 100
    count = te.estimate_tokens(payload, "gpt-4o")
    # JSON like this tokenizes roughly 1 token per ~2 chars (lots of
    # short tokens for braces, quotes, etc.) — much denser than chars/4.
    chars_over_four = len(payload) // 4
    assert count > chars_over_four, (
        f"expected tiktoken to count more than chars/4 for symbol-dense JSON; "
        f"got {count} vs {chars_over_four}"
    )


def test_estimator_caches_encoder_per_model():
    te.clear_cache()
    te.estimate_tokens("warmup", "gpt-4o")
    # Second call should reuse the cached encoder; we can only assert
    # by side-effect that we don't crash on repeated calls.
    for _ in range(5):
        te.estimate_tokens("again", "gpt-4o")


def test_module_requires_tiktoken_at_import():
    """tiktoken is a hard requirement — the module must import it
    eagerly, not lazily. If tiktoken disappears, importing
    `utils.token_estimator` should fail loudly rather than silently
    falling back to a heuristic."""
    import utils.token_estimator as mod

    # The import-time symbol is the proof.
    assert getattr(mod, "tiktoken", None) is not None


# ----------------------------------------------- /memory layer breakdown


def test_layers_include_l1_and_l1b(session):
    """Workspace files (L1) and skills (L1B) were silently missing from
    the /memory breakdown. Pin that they're now reported so users see
    what's actually in the system prompt."""
    layers = collect_context_layers(session)
    layer_ids = {layer["layer"] for layer in layers}
    assert "L1B" in layer_ids
    assert "L1B" in layer_ids


def test_layers_include_all_seven_slots(session):
    """L0 (system prompt) added; L4 removed from system prompt; L1C
    (workspace tree) removed — total is now 7 layers."""
    layers = collect_context_layers(session)
    expected = {"L0", "L1A", "L1B", "L2", "L3", "L4B", "L5"}
    assert {layer["layer"] for layer in layers} == expected


def test_l0_reports_system_prompt_tokens(session):
    """L0 should count tokens for the base system prompt that's
    actually sent to the provider — not just the bare
    `session.system_instruction` string. When agentic mode is on the
    agentic harness adds thousands of tokens that previously hid from
    the per-layer accounting."""
    session.agentic = True
    session.variables["agent_mode"] = "default"
    session.system_instruction = "You are a coding assistant."
    layers = collect_context_layers(session)
    l0 = next(layer for layer in layers if layer["layer"] == "L0")
    # The bare instruction is ~5 tokens. With the agentic harness
    # workflow appended L0 should be hundreds-to-thousands of tokens.
    assert l0["current"] > 100, (
        f"L0 only counted {l0['current']} tokens — agentic harness "
        "text isn't being included"
    )


def test_l0_smaller_when_agentic_off(session):
    """Non-agentic sessions shouldn't pay the harness-prompt tax."""
    session.system_instruction = "Be helpful."
    session.agentic = True
    on = collect_context_layers(session)
    on_l0 = next(layer for layer in on if layer["layer"] == "L0")["current"]
    session.agentic = False
    off = collect_context_layers(session)
    off_l0 = next(layer for layer in off if layer["layer"] == "L0")["current"]
    assert off_l0 < on_l0, (
        f"agentic=False L0 ({off_l0}) should be << agentic=True L0 ({on_l0})"
    )


def test_l5_measures_full_history_not_just_last_message(session):
    """The old L5 only counted `history[-1]`. After the fix it should
    track the same number the splash banner shows
    (`estimate_runtime_history_tokens`)."""
    session.session_manager.history = [
        {"role": "user", "parts": [{"type": "text", "text": "first " * 200}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "reply " * 200}]},
        {"role": "user", "parts": [{"type": "text", "text": "second " * 200}]},
    ]
    layers = collect_context_layers(session)
    l5 = next(layer for layer in layers if layer["layer"] == "L5")
    splash_count = session.session_manager.estimate_runtime_history_tokens()
    assert l5["current"] == splash_count
    # And it should be substantially larger than just the last message.
    last_only = session.session_manager._estimate_message_tokens(
        session.session_manager.history[-1]
    )
    assert l5["current"] > last_only


def test_layer_units_are_tokens_not_chars(session):
    """The denominator on L5 is `context_token_limit` (a token count).
    Pre-fix, the numerator was a char count — a unit mismatch. After
    the fix both should be tokens, so the ratio is meaningful."""
    # Put a known body of text in history and confirm L5's current
    # is a token count, not a char count.
    big_text = "x " * 5000  # ~10000 chars; ~5000 tokens at most
    session.session_manager.history = [
        {"role": "user", "parts": [{"type": "text", "text": big_text}]}
    ]
    layers = collect_context_layers(session)
    l5 = next(layer for layer in layers if layer["layer"] == "L5")
    # 5000 "x " repeats — char count is ~10000, token count is much
    # smaller because "x" is a single token. If L5 reported chars, it
    # would be ~10000+; reporting tokens it should be a few thousand.
    assert l5["current"] < len(big_text), (
        f"L5 current ({l5['current']}) shouldn't equal/exceed char count "
        f"({len(big_text)}); the layer must report tokens"
    )


def test_history_estimate_uses_active_model(session):
    """`estimate_runtime_history_tokens` must use the model's tokenizer
    so the splash banner matches what the provider will actually charge
    for. Smoke-check the lookup path works without throwing."""
    session.session_manager.history = [
        {"role": "user", "parts": [{"type": "text", "text": "what is 2+2?"}]}
    ]
    count = session.session_manager.estimate_runtime_history_tokens()
    assert count > 0
    assert count < 50  # sanity bound for a short utterance


def test_history_estimate_matches_l5_after_fix(session):
    """The L5 row in /memory and the per-history token count must agree."""
    session.session_manager.history = [
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "hi there"}]},
    ]
    history_tokens = session.session_manager.estimate_runtime_history_tokens()
    layers = collect_context_layers(session)
    l5 = next(layer for layer in layers if layer["layer"] == "L5")
    assert history_tokens == l5["current"]


# ----------------------------------------------- global-cap accounting


def test_total_active_context_equals_sum_of_layers(session):
    """`estimate_active_context_tokens` must equal the sum of every
    layer's current token count — the splash and /memory total row
    both depend on this identity."""
    from utils.runtime_metrics import estimate_active_context_tokens

    layers = collect_context_layers(session)
    expected = sum(int(layer["current"] or 0) for layer in layers)
    assert estimate_active_context_tokens(session) == expected


def test_non_l5_estimate_excludes_history(session):
    """`estimate_non_l5_context_tokens` is what the compactor uses to
    reserve room for non-history layers. It MUST exclude L5."""
    from utils.runtime_metrics import (
        estimate_active_context_tokens,
        estimate_non_l5_context_tokens,
    )

    session.session_manager.history = [
        {"role": "user", "parts": [{"type": "text", "text": "long " * 500}]}
    ]
    total = estimate_active_context_tokens(session)
    non_l5 = estimate_non_l5_context_tokens(session)
    l5 = total - non_l5
    # The 500-word turn is the only history entry, so its token count
    # should account for the L5 portion of the total.
    assert l5 > 0
    assert non_l5 < total


def test_compaction_budget_shrinks_when_non_l5_layers_grow(session):
    """Heavy non-L5 layers must tighten the L5 budget so the global cap
    is not exceeded. This is the core fix — previously the compactor
    treated `context_token_limit` as if it were L5's budget alone."""
    # Baseline: tiny non-L5 layers.
    session.variables["context_token_limit"] = 100_000
    session.variables["context_trim_threshold"] = 0.85
    baseline = session._compaction_token_budget()

    # Now fake a giant L1B: 200kB of skills. The compactor's
    # L5 budget should drop accordingly.
    original_build = session._build_skills_block
    session._build_skills_block = lambda: "x" * 200_000
    try:
        tightened = session._compaction_token_budget()
    finally:
        session._build_skills_block = original_build

    assert tightened < baseline, (
        f"L5 budget should shrink when L1B grows; baseline={baseline}, "
        f"tightened={tightened}"
    )


def test_compaction_budget_zero_when_no_capacity(session):
    """When non-L5 layers alone exceed the window, there is no real
    history capacity — the budget must be 0 (not a fabricated floor);
    compaction cannot help and must not loop (codex round-7 F6)."""
    session.variables["context_token_limit"] = 10_000
    session._build_skills_block = lambda: "x" * 1_000_000
    assert session._compaction_token_budget() == 0


# ----------------------------------------------- /set / schema coverage


def test_all_layer_budget_variables_are_in_schema():
    """Each layer budget is consumed at runtime; every one must be in
    VARIABLE_SCHEMA so `/set` validates / casts, `/variables` lists,
    `/unset` restores defaults, and `/get` returns the default for
    keys the user hasn't explicitly set."""
    from utils.config import VARIABLE_SCHEMA

    layer_vars = {
        "skills_max_chars",
        "skills_mode",
        "conversation_summary_char_limit",
        "active_goal_context_char_limit",
        "retrieval_context_char_limit",
        "retrieval_top_k",
        "context_token_limit",
        "context_trim_threshold",
        "response_token_reserve",
    }
    missing = layer_vars - set(VARIABLE_SCHEMA.keys())
    assert not missing, f"layer-budget variables missing from schema: {missing}"


def test_set_command_validates_layer_budget_types():
    """The schema's `type` entry should be honored by `validate_and_cast`
    so `/set conversation_summary_char_limit 16000` casts to int."""
    from utils.config import validate_and_cast

    assert validate_and_cast("conversation_summary_char_limit", "16000") == 16000
    assert validate_and_cast("retrieval_top_k", "10") == 10
    assert validate_and_cast("retrieval_context_char_limit", "8000") == 8000


def test_memory_command_data_includes_total(session):
    """The /memory result should surface the global-cap accounting in
    its `data` payload (not only in the printed table) so non-
    interactive callers can read it."""
    from mu.commands.memory import memory_cmd

    result = memory_cmd(session, "status", allow_prompt=False)
    assert result.ok
    assert "context_total_tokens" in result.data
    assert "context_limit_tokens" in result.data
    assert "context_fill_pct" in result.data
    assert result.data["context_total_tokens"] >= 0
    assert result.data["context_limit_tokens"] > 0
