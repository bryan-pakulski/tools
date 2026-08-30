"""End-to-end verification that rolling compaction is actually firing
for small-context providers (Ollama 8k, etc.) — not just that the budget
arithmetic is correct.

`test_context_budget.py` covers the math (`_resolve_context_limit`,
`_compaction_token_budget`). This file closes the loop: given a real
Session whose provider declares an 8k window, stuff history with enough
chatter to blow that budget, and confirm:

  1. `_compaction_token_budget()` produces something an Ollama model
     can actually fit.
  2. `roll_history_summary_to_token_budget(budget)` invoked with that
     budget actually advances `summary_anchor`, populates
     `conversation_summary`, and shrinks runtime token estimate to
     under the budget.
  3. The pre-turn rolling site in the agent loop (line 2398) plumbs the
     provider-aware budget through — not the legacy 256000 default.
"""

from typing import List, Optional

import pytest

from mu.session.session import Session, SessionManager
from providers.base import LLMProvider, ProviderResponse


class _Tiny8kProvider(LLMProvider):
    """Mimics an 8k-window Ollama model."""

    def get_available_models(self) -> List[str]:
        return ["tiny-8k"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        return ProviderResponse(
            text="ok", parts=[], input_tokens=0, output_tokens=0, total_tokens=0
        )

    def upload_file(self, file_path, mime_type):
        return None

    def effective_context_window(self, model_name: Optional[str] = None):
        return 8192


def _make_session(provider=None):
    sm = SessionManager()
    session = Session(provider or _Tiny8kProvider("tiny-8k"), False, "sys", sm)
    # Isolate this test from L0 (system-prompt) accounting — the
    # agentic harness adds ~3-5k tokens to L0 which would dominate
    # the 8k-window budget and make the L5-trimming test flaky.
    # L0/agentic accounting is covered in test_token_estimator.py.
    session.agentic = False
    return session


def _stuff_history_over_budget(sm: SessionManager, n_turns: int = 40, size: int = 1500):
    """Append `n_turns` of (user, assistant) text bombs, each ~`size` chars.
    With the 4 chars/token heuristic, each message is ~`size/4` tokens, so
    n_turns=40 size=1500 → ~30k tokens total — well over any 8k budget."""
    sm.history.clear()
    sm.summary_anchor = 0
    sm.conversation_summary = ""
    for i in range(n_turns):
        sm.history.append(
            {"role": "user", "parts": [{"type": "text", "text": "u" * size}]}
        )
        sm.history.append(
            {"role": "assistant", "parts": [{"type": "text", "text": "a" * size}]}
        )


def test_compaction_budget_for_8k_provider_is_under_provider_window():
    """Sanity: the budget the compactor targets must be strictly smaller
    than the provider's real window — otherwise compaction "succeeds"
    at a budget Ollama still can't fit."""
    session = _make_session()
    budget = session._compaction_token_budget()
    assert budget < 8192, f"budget {budget} >= provider window 8192"
    # And not pathologically small either.
    assert budget >= 512


def test_compaction_actually_shrinks_oversized_history():
    """The smoking-gun test. Stuff history with ~30k tokens of chatter
    aimed at an 8k provider window, run the compactor with the
    provider-aware budget, and confirm the runtime-history token count
    drops below the budget."""
    session = _make_session()
    _stuff_history_over_budget(session.session_manager, n_turns=40, size=1500)

    before_tokens = session.session_manager.estimate_runtime_history_tokens()
    before_anchor = session.session_manager.summary_anchor
    budget = session._compaction_token_budget()
    assert before_tokens > budget, (
        f"test setup didn't blow the budget: tokens={before_tokens} budget={budget}"
    )

    changed = session.session_manager.roll_history_summary_to_token_budget(
        budget, keep_recent=4
    )
    assert changed is True

    after_tokens = session.session_manager.estimate_runtime_history_tokens()
    assert session.session_manager.summary_anchor > before_anchor, (
        "summary_anchor did not advance"
    )
    assert session.session_manager.conversation_summary, (
        "conversation_summary should be populated after compaction"
    )
    assert after_tokens <= budget, (
        f"compaction left {after_tokens} tokens for budget {budget}"
    )


def test_compaction_keeps_recent_turns_intact():
    """The last few turns must survive compaction — otherwise the model
    loses the live user message it's responding to. `keep_recent=4` is
    the harness default."""
    session = _make_session()
    sm = session.session_manager
    _stuff_history_over_budget(sm, n_turns=40, size=1500)
    # Stamp the last user turn so we can identify it post-compaction.
    sm.history[-2]["parts"][0]["text"] = "MARKER-LAST-USER"
    sm.history[-1]["parts"][0]["text"] = "MARKER-LAST-ASSISTANT"

    sm.roll_history_summary_to_token_budget(
        session._compaction_token_budget(), keep_recent=4
    )

    tails = [m["parts"][0]["text"] for m in sm.history[-2:]]
    assert tails == ["MARKER-LAST-USER", "MARKER-LAST-ASSISTANT"]


def test_compaction_falls_back_to_payload_truncation_when_one_huge_message():
    """If history has one giant message that can't be summarized away
    (it's the only thing left), the compactor must clip the oldest
    oversized payload in place so we still fit. Otherwise the model
    sees a 32k blob on an 8k window — exactly the original bug."""
    session = _make_session()
    sm = session.session_manager
    sm.history = [
        {
            "role": "assistant",
            "parts": [
                {"type": "tool_result", "tool_name": "read_file", "tool_result": "A" * 60000}
            ],
        }
    ]
    sm.summary_anchor = 0

    changed = sm.roll_history_summary_to_token_budget(
        session._compaction_token_budget(),
        keep_recent=1,
        max_passes=4,
    )
    assert changed is True
    payload = sm.history[0]["parts"][0]["tool_result"]
    assert "truncated_to_16000_chars_for_context_budget" in payload


def test_agent_loop_uses_provider_aware_budget_before_provider_call(monkeypatch):
    """The real bug was that the pre-turn rolling site in `send_message`
    used the 256k user-default — not the 8k provider window — so
    compaction never fired before the Ollama 400.

    Spy on `roll_history_summary_to_token_budget` and confirm the budget
    passed in is consistent with the provider's effective window, not
    the user-set `context_token_limit`."""
    session = _make_session()
    sm = session.session_manager
    session.variables["auto_compaction_enabled"] = True
    _stuff_history_over_budget(sm, n_turns=10, size=1500)

    # User has the harness-wide default (256k); provider says 8k.
    session.variables["context_token_limit"] = 256_000

    captured: dict = {}
    real = sm.roll_history_summary_to_token_budget

    def _spy(budget, *args, **kwargs):
        captured.setdefault("budgets", []).append(budget)
        return real(budget, *args, **kwargs)

    monkeypatch.setattr(sm, "roll_history_summary_to_token_budget", _spy)

    # Drive through the pre-turn rolling call site. We don't run a full
    # generation — just trigger send_message far enough that it hits the
    # pre-turn rolling call. The provider returns immediately on the
    # first generate() so the test stays cheap.
    session.send_message("ping")

    assert captured.get("budgets"), "compactor not invoked at all"
    for budget in captured["budgets"]:
        assert budget < 8192, (
            f"budget {budget} is at or above provider window — "
            f"compactor is using the legacy 256k limit, not the provider's"
        )


def test_compaction_no_op_when_history_already_fits():
    """If history already fits under the budget, compaction must return
    False and leave summary_anchor / conversation_summary untouched."""
    session = _make_session()
    sm = session.session_manager
    sm.history = [
        {"role": "user", "parts": [{"type": "text", "text": "short msg"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "short reply"}]},
    ]
    before_anchor = sm.summary_anchor
    before_summary = sm.conversation_summary

    changed = sm.roll_history_summary_to_token_budget(
        session._compaction_token_budget(), keep_recent=2
    )

    assert changed is False
    assert sm.summary_anchor == before_anchor
    assert sm.conversation_summary == before_summary


def test_auto_compact_hook_does_not_double_apply_threshold(monkeypatch):
    """The auto-compaction hook must pass `compaction_token_budget()` as the
    budget — NOT `compaction_token_budget() * threshold`. The latter
    double-applied `context_trim_threshold` (already applied inside
    `compaction_token_budget`), collapsing the target from 85% to ~72% of
    the residual window and firing compaction far too often.
    """
    from mu.agent.compactor import _compact_history
    from mu.agent.hooks import HookContext

    session = _make_session()
    sm = session.session_manager
    session.variables["auto_compaction_enabled"] = True
    _stuff_history_over_budget(sm, n_turns=10, size=1500)
    session.variables["context_token_limit"] = 256_000
    session._compaction_watermark = 0  # arm the watermark gate
    session._compacted_this_turn = False  # arm the once-per-turn gate

    captured: dict = {}
    real = sm.roll_history_summary_to_token_budget

    def _spy(budget, *args, **kwargs):
        captured.setdefault("budgets", []).append(budget)
        # Don't actually run the LLM summarizer in this unit test — return
        # False so the hook's "rolled" branch is exercised only when we want.
        return False

    monkeypatch.setattr(sm, "roll_history_summary_to_token_budget", _spy)

    ctx = HookContext(point="pre_provider_call", session=session)
    _compact_history(ctx)

    assert captured.get("budgets"), "hook did not invoke the roller"
    expected = int(session._compaction_token_budget())
    for budget in captured["budgets"]:
        # Must equal compaction_token_budget() exactly — not that × threshold.
        assert budget == expected, (
            f"hook budget {budget} != compaction_token_budget() {expected}; "
            f"threshold was double-applied (target collapsed to ~72%)"
        )
        # And it must be strictly larger than the buggy double-applied value.
        assert budget > int(expected * 0.85), (
            "hook budget is at or below the old double-applied value — "
            "the fix regressed"
        )


def test_auto_compact_hook_is_disabled_by_default(monkeypatch):
    """Normal cleanup is model-directed; automatic trimming is opt-in."""
    from mu.agent.compactor import _compact_history
    from mu.agent.hooks import HookContext

    session = _make_session()
    called = False

    def _unexpected(*args, **kwargs):
        nonlocal called
        called = True
        return False

    monkeypatch.setattr(
        session.session_manager, "roll_history_summary_to_token_budget", _unexpected
    )
    assert _compact_history(HookContext(point="pre_provider_call", session=session)) is None
    assert called is False


def test_auto_compact_hook_fires_at_most_once_per_turn(monkeypatch):
    """The per-iteration auto-compaction hook must fire at most once per turn
    (Claude Code fires autocompact once per turn at the boundary). Once
    `_compacted_this_turn` is set — by the turn-start roll or by the hook's
    own first compaction — subsequent `pre_provider_call` invocations skip.
    """
    from mu.agent.compactor import _compact_history
    from mu.agent.hooks import HookContext

    session = _make_session()
    # Round-9 F3: the hook is gated on the same auto_compaction_enabled
    # opt-in as the turn-start roll — opt this session in.
    session.variables["auto_compaction_enabled"] = True
    sm = session.session_manager
    _stuff_history_over_budget(sm, n_turns=10, size=1500)
    session._compaction_watermark = 0

    call_count = {"n": 0}
    real = sm.roll_history_summary_to_token_budget

    def _spy(budget, *args, **kwargs):
        call_count["n"] += 1
        return real(budget, *args, **kwargs)

    monkeypatch.setattr(sm, "roll_history_summary_to_token_budget", _spy)

    # First invocation: history is over budget → compacts → sets the flag.
    ctx = HookContext(point="pre_provider_call", session=session)
    res1 = _compact_history(ctx)
    assert res1 is not None, "first invocation should compact"
    assert session._compacted_this_turn is True, "hook did not set the flag"

    # Simulate history growing (another tool result landed) past the
    # watermark — the old behavior would re-fire here.
    sm.history.append(
        {"role": "user", "parts": [{"type": "text", "text": "more" * 1500}]}
    )
    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "more" * 1500}]}
    )
    assert len(sm.history) > session._compaction_watermark

    # Second + third invocations this turn: must skip (flag is set).
    assert _compact_history(ctx) is None
    assert _compact_history(ctx) is None
    assert call_count["n"] == 1, (
        f"hook compacted {call_count['n']} times this turn, expected once"
    )

    # After the turn ends (`_collect_turn_response`), the flag resets and the
    # next turn can fire again.
    session._compacted_this_turn = False
    assert _compact_history(ctx) is not None
