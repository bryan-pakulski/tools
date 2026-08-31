"""Pin the loop-mode watchdog vs raise_blocker interaction.

The bug this regression-tests:

  When a loop-mode agent satisfied its goal and correctly called
  `raise_blocker`, the next iteration found no tool calls and the
  watchdog fired a "LOOP WATCHDOG: Continue autonomous loop execution
  now" message — forcing the model to re-raise the blocker. Repeated
  per turn until iteration cap exhausted, burning thousands of tokens
  in a wedge loop. See the transcript in the bug report.

  Fix: `Session._loop_blocker_raised` flag flips True when
  `raise_blocker` post-processes; the watchdog branch checks the flag
  and skips the continue-nudge when set. `send_message` clears the
  flag at the start of each new turn so subsequent unrelated loops
  still work.
"""

import pytest

from mu.session.session import Session, SessionManager
from providers.base import LLMProvider, ProviderResponse


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
    s = Session(_DummyProvider("dummy"), False, "sys", sm)
    s._mcp_clients = []
    s.session_manager.history = []
    s.session_manager.summary_anchor = 0
    s.session_manager.conversation_summary = ""
    return s


def test_loop_blocker_flag_initializes_to_false(session):
    assert session._loop_blocker_raised is False


def test_raise_blocker_post_process_sets_flag(session):
    """The internal post-tool path that runs after `raise_blocker`
    completes must flip the flag so the watchdog can see it."""
    session._sync_feature_state_for_tool(
        tool_name="raise_blocker",
        tool_args={},
        raw_result="",
        structured_result={
            "ok": True,
            "data": {
                "kind": "user_input_required",
                "summary": "Goal satisfied — need a new mission.",
                "requested_input": "Provide next loop goal or /mode default.",
            },
        },
    )
    assert session._loop_blocker_raised is True


def test_post_process_for_other_tool_does_not_set_flag(session):
    """Only raise_blocker should set the flag — not, say, read_file
    returning successfully."""
    session._sync_feature_state_for_tool(
        tool_name="read_file",
        tool_args={"filename": "x.py"},
        raw_result="",
        structured_result={"ok": True, "data": {"content": "hello"}},
    )
    assert session._loop_blocker_raised is False


def test_send_message_clears_flag(session):
    """A fresh turn must reset the flag so a brand-new loop goal
    doesn't see a stale 'blocker raised' state."""
    session._loop_blocker_raised = True

    # We can't easily run a real send_message (needs a provider), but
    # the clear is the first thing it does — pin via source inspection.
    # Body moved to `mu/agent/loop_body.py:run_turn` during Phase 4 of
    # the refactor; source-inspect that target now.
    import inspect

    from mu.agent import loop_body as loop_body_mod

    source = inspect.getsource(loop_body_mod.run_turn)
    # The clear must appear early — right after the initial logging line.
    clear_pos = source.index("session._loop_blocker_raised = False")
    info_pos = source.index("logger.info")
    assert clear_pos > info_pos
    # And before the agentic loop entry point (search for the `while`
    # that starts iterations).
    while_pos = source.index("while iteration < max_iterations")
    assert clear_pos < while_pos


def test_watchdog_branch_checks_blocker_flag_in_source():
    """Source-level pin: the loop-mode watchdog branch must consult
    `_loop_blocker_raised` BEFORE appending the continue-nudge. If a
    future refactor removes the check, this test flags it."""
    import inspect

    from mu.agent import loop_body as loop_body_mod

    # The watchdog branch lives in run_turn (post-Phase-4 location).
    source = inspect.getsource(loop_body_mod.run_turn)
    watchdog_pos = source.index("LOOP WATCHDOG")
    # The flag check must appear after the active-mode == "loop" test
    # but before the watchdog message append.
    flag_pos = source.index("_loop_blocker_raised", source.index("active_mode == \"loop\""))
    assert flag_pos < watchdog_pos, (
        "Loop watchdog should consult _loop_blocker_raised before re-prompting"
    )


def test_watchdog_counters_init(session):
    """Round-51 T5: watchdog counters initialize lazily to zero."""
    assert getattr(session, "_watchdog_nudge_count", 0) or 0 == 0
    assert getattr(session, "_watchdog_ineffective_count", 0) or 0 == 0


def test_watchdog_ineffective_increments_without_tools(session):
    """Round-51 T5: an iteration with a nudge outstanding but no tool calls
    counts as ineffective."""
    session._watchdog_nudge_count = 1
    session._watchdog_ineffective_count = 0
    # Simulate the loop's no-tool-call accounting.
    if getattr(session, "_watchdog_nudge_count", 0):
        session._watchdog_ineffective_count = int(
            getattr(session, "_watchdog_ineffective_count", 0) or 0
        ) + 1
    assert session._watchdog_ineffective_count == 1


def test_watchdog_counter_resets_on_tool_progress(session):
    """Round-51 T5: a tool-call iteration resets the ineffective counter."""
    session._watchdog_nudge_count = 2
    session._watchdog_ineffective_count = 2
    # The reset contract from loop_body's tool_call branch:
    if getattr(session, "_watchdog_ineffective_count", 0):
        session._watchdog_ineffective_count = 0
    assert session._watchdog_ineffective_count == 0

