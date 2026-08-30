"""Test the pre-flight context overflow check in loop_body.py.

The pre-flight check catches prompt overflows that slip past the initial
compaction pass — e.g. when resumption briefings, hierarchical context,
or per-iteration memory/scratchpad layers grow the system prompt after
the initial ``roll_history_summary_to_token_budget`` call.
"""

import json
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from mu.agent.loop_body import (
    _estimate_messages_tokens,
    _estimate_tools_tokens,
    _preflight_context_check,
)
from providers.base import Message, MessagePart, ToolDefinition


# --------------------------------------------------------- helpers


def _make_messages(texts):
    """Build a list of Message objects from a list of strings."""
    msgs = []
    for t in texts:
        msgs.append(Message(role="user", parts=[MessagePart(type="text", text=t)]))
    return msgs


def _stub_session(context_limit=8192, response_reserve=2048):
    """Create a mock session with the budget helpers wired up."""
    session = MagicMock()
    session.variables = {
        "context_token_limit": context_limit,
    }

    # Wire up budget resolution
    provider = MagicMock()
    provider.effective_context_window.return_value = context_limit
    provider.effective_response_reserve.return_value = response_reserve
    session.provider = provider

    # Track compaction calls
    session._history_rolled_this_turn = True
    session.session_manager = MagicMock()
    session.session_manager.roll_history_summary_to_token_budget = MagicMock(
        return_value=True
    )

    # After compaction, rebuild returns shorter messages
    short_history = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
    session._prepare_runtime_history.return_value = short_history
    session._build_messages_from_history.return_value = _make_messages(["hi", ""])

    # Post-compaction prompt rebuild: echo the base prompt back unchanged
    # (the stub has no layered-context machinery). Tests assert identity.
    session.system_instruction = "stub system"
    session._inject_hierarchical_context = MagicMock(
        side_effect=lambda base, cached_skills=None: base
    )

    return session


# --------------------------------------------------------- token estimation


def test_estimate_messages_tokens_text():
    msgs = _make_messages(["hello world", "foo bar"])
    tokens = _estimate_messages_tokens(msgs)
    assert tokens > 0


def test_estimate_messages_tokens_tool_result():
    msg = Message(
        role="tool",
        parts=[
            MessagePart(
                type="tool_result",
                tool_result="some result text here",
                tool_name="test",
            )
        ],
    )
    tokens = _estimate_messages_tokens([msg])
    assert tokens > 0


def test_estimate_messages_tokens_tool_args():
    msg = Message(
        role="assistant",
        parts=[
            MessagePart(
                type="tool_call",
                tool_name="read_file",
                tool_args={"path": "/some/file.py"},
            )
        ],
    )
    tokens = _estimate_messages_tokens([msg])
    assert tokens > 0


def test_estimate_messages_tokens_empty():
    assert _estimate_messages_tokens([]) == 0


def test_tool_schemas_are_counted_by_preflight():
    tools = [ToolDefinition(
        name="verbose_tool",
        description="schema " * 5000,
        parameters={"type": "object", "properties": {}},
    )]
    assert _estimate_tools_tokens(tools) > 1000

    session = _stub_session(context_limit=500, response_reserve=100)
    _preflight_context_check(session, "small", _make_messages(["small"]), tools=tools)

    session.session_manager.roll_history_summary_to_token_budget.assert_called()
    assert session._last_prompt_cl100k_est > 1000


# --------------------------------------------------------- pre-flight check


def test_within_budget_returns_unchanged():
    session = _stub_session(context_limit=100_000, response_reserve=4096)
    system_prompt = "You are a helpful assistant."
    messages = _make_messages(["hello"])

    result_prompt, result_msgs = _preflight_context_check(
        session, system_prompt, messages
    )

    assert result_prompt is system_prompt
    assert result_msgs is messages
    session.session_manager.roll_history_summary_to_token_budget.assert_not_called()


def test_over_budget_triggers_emergency_compaction():
    session = _stub_session(context_limit=100, response_reserve=20)
    # 80 tokens of max_prompt, but we'll send way more
    big_prompt = "x " * 500  # ~500 tokens
    big_messages = _make_messages(["y " * 500])

    result_prompt, result_msgs = _preflight_context_check(
        session, big_prompt, big_messages
    )

    # Compaction should have been called at least once (the escalating
    # loop fires up to 3 rounds when the rebuilt prompt is still over).
    calls = session.session_manager.roll_history_summary_to_token_budget.call_args_list
    assert len(calls) >= 1
    first = calls[0]
    budget = first[0][0]
    assert budget > 0
    # Emergency keep_recent floor is KEEP_RECENT_EMERGENCY (2).
    assert first[1]["keep_recent"] == 2

    # Messages should have been rebuilt
    assert result_msgs is not big_messages
    # System prompt unchanged (only messages get compacted)
    assert result_prompt is big_prompt


def test_over_budget_escalates_keep_recent_when_still_over():
    """When one compaction pass doesn't reach the budget, the escalating
    loop shrinks keep_recent and compacts again (Claude Code-style
    progressive compaction)."""
    session = _stub_session(context_limit=100, response_reserve=20)
    # The stub's rebuild always returns a short message, but the system
    # prompt itself is far over budget so the re-check never succeeds —
    # exercising all 3 escalation rounds.
    big_prompt = "x " * 500
    big_messages = _make_messages(["y " * 500])

    _preflight_context_check(session, big_prompt, big_messages)

    calls = session.session_manager.roll_history_summary_to_token_budget.call_args_list
    # 3 rounds fire because the oversized system prompt keeps the
    # re-estimate over budget after each rebuild.
    assert len(calls) == 3
    # keep_recent shrinks across rounds: 2, 2, 2 (floor at max(2, ...)).
    keep_recents = [c[1]["keep_recent"] for c in calls]
    assert keep_recents == [2, 2, 2]


def test_over_budget_stops_early_when_rebuild_fits():
    """When the first compaction rebuild gets under budget, the loop stops
    after one round — no needless extra compaction passes."""
    session = _stub_session(context_limit=10_000, response_reserve=500)
    # max_prompt = 9500. A modestly oversized prompt that the stub's short
    # rebuild will bring well under budget.
    big_prompt = "x " * 600  # ~300 cl100k tokens
    # Push the messages well over budget (~10k tokens) so compaction
    # triggers, but the stub's short rebuild brings the total (~300)
    # under 9500 so the loop exits after round 0.
    big_messages = _make_messages(["y " * 20000])

    _preflight_context_check(session, big_prompt, big_messages)

    calls = session.session_manager.roll_history_summary_to_token_budget.call_args_list
    assert len(calls) == 1


def test_compaction_failure_returns_original():
    session = _stub_session(context_limit=100, response_reserve=20)
    session.session_manager.roll_history_summary_to_token_budget.side_effect = (
        RuntimeError("compaction broke")
    )
    big_prompt = "x " * 500
    big_messages = _make_messages(["y " * 500])

    result_prompt, result_msgs = _preflight_context_check(
        session, big_prompt, big_messages
    )

    # Should return originals on failure
    assert result_prompt is big_prompt
    assert result_msgs is big_messages


def test_history_rolled_flag_management():
    """The check temporarily clears _history_rolled_this_turn so the
    compactor can fire, then restores it."""
    session = _stub_session(context_limit=100, response_reserve=20)
    session._history_rolled_this_turn = True

    big_prompt = "x " * 500
    big_messages = _make_messages(["y " * 500])

    _preflight_context_check(session, big_prompt, big_messages)

    # After successful compaction, flag should be set back to True
    assert session._history_rolled_this_turn is True
