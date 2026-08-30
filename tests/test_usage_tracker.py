"""Pin the per-session usage tracker and /stats integration.

Three concerns:
  1. Every tool call updates `session.tool_stats` via the post_tool hook.
  2. `invoke_skill` produces a visible banner via the pre_tool hook AND
     bumps the per-skill counter on the tracker.
  3. `/stats` exposes the tracker data, and `/stats clear` wipes it.
"""

import time
from typing import Any, List

import pytest

import mu.commands as mc
# Importing the tracker registers its hooks on import.
import mu.agent.usage_tracker  # noqa: F401
from mu.session.session import Session, SessionManager
from mu.agent.hooks import HookContext, default_registry
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


def _fire(point, **kwargs):
    ctx = HookContext(point=point, **kwargs)
    default_registry.fire(point, ctx)
    return ctx


# ----------------------------------------------- tracker initialization


def test_tool_stats_initialized_on_session(session):
    assert isinstance(session.tool_stats, dict)
    assert session.tool_stats["tools"] == {}
    assert session.tool_stats["skills"] == {}
    assert session.tool_stats["first_call_at"] is None
    assert session.tool_stats["last_call_at"] is None


# ----------------------------------------------- post_tool counter


def test_post_tool_records_successful_call(session):
    pre = _fire(
        "pre_tool",
        session=session,
        tool_name="read_file",
        tool_args={"filename": "a.py"},
    )
    # Pre stamp should land on the same ctx.metadata.
    assert "_usage_start_ts" in pre.metadata

    # Post must use the SAME context (the metadata carries the start ts).
    pre.point = "post_tool"
    pre.tool_result = {"ok": True, "data": {"content": "hello"}}
    default_registry.fire("post_tool", pre)

    tools = session.tool_stats["tools"]
    assert "read_file" in tools
    assert tools["read_file"]["count"] == 1
    assert tools["read_file"]["success"] == 1
    assert tools["read_file"]["failed"] == 0
    assert tools["read_file"]["last_args"]  # non-empty preview
    assert tools["read_file"]["last_used_at"] is not None


def test_post_tool_records_failure_with_error_code(session):
    pre = _fire("pre_tool", session=session, tool_name="bash", tool_args={"command": "false"})
    pre.point = "post_tool"
    pre.tool_result = {"ok": False, "error_code": "exit_nonzero", "data": {}}
    default_registry.fire("post_tool", pre)

    bucket = session.tool_stats["tools"]["bash"]
    assert bucket["success"] == 0
    assert bucket["failed"] == 1
    assert session.tool_stats["errors"]["exit_nonzero"] == 1


def test_post_tool_aggregates_across_calls(session):
    for path in ("a.py", "b.py", "c.py"):
        pre = _fire("pre_tool", session=session, tool_name="read_file", tool_args={"filename": path})
        pre.point = "post_tool"
        pre.tool_result = {"ok": True}
        default_registry.fire("post_tool", pre)
    bucket = session.tool_stats["tools"]["read_file"]
    assert bucket["count"] == 3
    assert bucket["success"] == 3
    # last_args reflects the most recent call.
    assert "c.py" in bucket["last_args"]


def test_post_tool_records_elapsed_time(session, monkeypatch):
    """The post hook should compute (post_monotonic - pre_monotonic) and
    add it to total_ms. Mock time.monotonic so the elapsed is deterministic."""
    timeline = iter([100.0, 100.250])  # 250ms elapsed
    monkeypatch.setattr("mu.agent.usage_tracker.time.monotonic", lambda: next(timeline))

    pre = _fire("pre_tool", session=session, tool_name="search_for_string", tool_args={"string": "x"})
    pre.point = "post_tool"
    pre.tool_result = {"ok": True}
    default_registry.fire("post_tool", pre)

    bucket = session.tool_stats["tools"]["search_for_string"]
    assert 200 <= bucket["total_ms"] <= 300, bucket["total_ms"]


# ----------------------------------------------- invoke_skill specifics


def test_invoke_skill_bumps_skills_counter(session):
    pre = _fire(
        "pre_tool",
        session=session,
        tool_name="invoke_skill",
        tool_args={"name": "commit-message"},
    )
    pre.point = "post_tool"
    pre.tool_result = {"ok": True, "data": {}}
    default_registry.fire("post_tool", pre)

    skills = session.tool_stats["skills"]
    assert "commit-message" in skills
    assert skills["commit-message"]["invocations"] == 1
    assert skills["commit-message"]["last_used_at"] is not None


def test_invoke_skill_emits_visible_banner(session):
    """The pre_tool hook for `invoke_skill` must visibly highlight
    the activation through the UI's console surface."""
    captured: List[str] = []

    class _FakeConsole:
        def print(self, *args, **kwargs):
            for a in args:
                captured.append(a.plain if hasattr(a, "plain") else str(a))

    class _FakeUI:
        def __init__(self):
            self.console = _FakeConsole()

    session.ui = _FakeUI()
    _fire(
        "pre_tool",
        session=session,
        tool_name="invoke_skill",
        tool_args={"name": "code-review"},
    )
    assert any("SKILL ACTIVE" in line for line in captured), captured
    assert any("code-review" in line for line in captured), captured


def test_non_skill_tool_does_not_emit_banner(session):
    captured: List[str] = []

    class _FakeConsole:
        def print(self, *args, **kwargs):
            for a in args:
                captured.append(a.plain if hasattr(a, "plain") else str(a))

    class _FakeUI:
        def __init__(self):
            self.console = _FakeConsole()

    session.ui = _FakeUI()
    _fire("pre_tool", session=session, tool_name="read_file", tool_args={"filename": "x"})
    assert not any("SKILL ACTIVE" in line for line in captured)


# ----------------------------------------------- /stats integration


def test_stats_includes_tracker_in_data(session):
    pre = _fire("pre_tool", session=session, tool_name="read_file", tool_args={"filename": "x"})
    pre.point = "post_tool"
    pre.tool_result = {"ok": True}
    default_registry.fire("post_tool", pre)

    result = mc.dispatch(session, "/stats", allow_prompt=False)
    assert result.ok
    assert "tool_stats" in result.data
    assert "read_file" in result.data["tool_stats"]["tools"]


def test_stats_clear_wipes_tracker_but_keeps_tokens(session):
    # Populate tracker.
    pre = _fire("pre_tool", session=session, tool_name="bash", tool_args={"command": "ls"})
    pre.point = "post_tool"
    pre.tool_result = {"ok": True}
    default_registry.fire("post_tool", pre)
    # Spoof some lifetime token spend.
    session.session_manager.token_counts["total"] = 12_345
    session.session_manager.token_counts["total_cost"] = 1.50

    result = mc.dispatch(session, "/stats clear", allow_prompt=False)
    assert result.ok
    # Tracker is wiped …
    assert session.tool_stats["tools"] == {}
    assert session.tool_stats["skills"] == {}
    # … but lifetime token accounting stays.
    assert session.session_manager.token_counts["total"] == 12_345
    assert session.session_manager.token_counts["total_cost"] == 1.50


def test_stats_clear_resets_session_start_timestamp(session):
    """`session_started_at` resets so "Session age" reads from /stats clear."""
    old = session.tool_stats["session_started_at"]
    time.sleep(0.01)
    mc.dispatch(session, "/stats clear", allow_prompt=False)
    new = session.tool_stats["session_started_at"]
    assert new > old


def test_tool_stats_persist_through_session_reload(tmp_path, monkeypatch):
    from mu.session.session import Session, SessionManager

    monkeypatch.setattr("utils.config.HISTORY_DIR", str(tmp_path / "history"))
    manager = SessionManager(session_name="stats-persistence")
    session = Session(_DummyProvider("dummy"), False, "system", manager)
    pre = _fire("pre_tool", session=session, tool_name="read_file", tool_args={"filename": "a.py"})
    pre.point = "post_tool"
    pre.tool_result = {"ok": True}
    default_registry.fire("post_tool", pre)
    manager.save_history()

    restored = Session(
        _DummyProvider("dummy"), False, "system", SessionManager(session_name="stats-persistence")
    )
    assert restored.tool_stats["tools"]["read_file"]["count"] == 1


def test_stats_rejects_unknown_subcommand(session):
    result = mc.dispatch(session, "/stats wibble", allow_prompt=False)
    assert not result.ok
    assert "Usage" in result.message


def test_stats_autocomplete_includes_clear():
    """`/stats <Tab>` should offer `clear`."""
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from mu.ui.input import InputHandler

    handler = InputHandler()
    doc = Document(text="/stats ", cursor_position=len("/stats "))
    completions = list(
        handler.completer.get_completions(doc, CompleteEvent(completion_requested=True))
    )
    texts = {c.text for c in completions}
    assert "clear" in texts


def test_post_tool_parallel_no_lost_updates(session):
    """Round-20 F9: post_tool hooks run concurrently under parallel tool
    execution — the stats read-modify-write must not lose increments."""
    import threading

    from mu.agent import usage_tracker as ut
    from mu.agent.hooks import HookContext

    n_threads, n_incr = 8, 50
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()  # maximize contention
        for _ in range(n_incr):
            ctx = HookContext(
                point="post_tool",
                session=session,
                tool_name="bash",
                tool_args={"command": "x"},
                metadata={},
                tool_result={"ok": True},
            )
            ut.usage_tracker_post(ctx)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    bucket = session.tool_stats["tools"]["bash"]
    assert bucket["count"] == n_threads * n_incr
    assert bucket["success"] == n_threads * n_incr
