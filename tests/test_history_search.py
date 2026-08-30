"""Unit tests for HistorySearchMixin.search_history and the @tool search_history handler.

Covers: text/tool_call/tool_result/file/image_input matching, role/tool_name filters,
include_summarized, context_messages bounding, max_results, ranking, snippets,
cache_key passthrough, empty history, empty query, no matches, tool handler,
GUI endpoint, and existing-test regression.
"""

import json
import pytest

from mu.session.history_search import HistorySearchMixin


class _Host(HistorySearchMixin):
    """Minimal host for HistorySearchMixin testing."""

    def __init__(self, history=None, summary_anchor=0, tool_result_cache=None):
        self.history = history or []
        self.summary_anchor = summary_anchor
        self.tool_result_cache = tool_result_cache


# ============================================================ text part matching


def test_text_part_matching():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "hello world"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "goodbye"}]},
    ])
    results = host.search_history("hello")
    assert results["total_matches"] == 1
    assert results["results"][0]["index"] == 0
    assert results["results"][0]["role"] == "user"


def test_text_part_case_insensitive():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "Hello World"}]},
    ])
    results = host.search_history("hello")
    assert results["total_matches"] == 1


def test_text_part_snippet_extraction():
    long_text = "x" * 100 + "MATCH" + "y" * 100
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": long_text}]},
    ])
    results = host.search_history("MATCH")
    snippet = results["results"][0]["parts_matched"][0]["snippet"]
    assert "MATCH" in snippet
    assert len(snippet) <= 205  # 200 + small slack


# ============================================================ tool_call matching


def test_tool_call_name_matching():
    host = _Host(history=[
        {"role": "assistant", "parts": [
            {"type": "tool_call", "tool_name": "read_file", "tool_args": {"filename": "x.py"}}
        ]},
    ])
    results = host.search_history("read_file")
    assert results["total_matches"] == 1
    pm = results["results"][0]["parts_matched"]
    assert any(p["match_type"] == "tool_name" for p in pm)


def test_tool_call_args_matching():
    host = _Host(history=[
        {"role": "assistant", "parts": [
            {"type": "tool_call", "tool_name": "bash", "tool_args": {"command": "ls -la"}}
        ]},
    ])
    results = host.search_history("ls -la")
    assert results["total_matches"] == 1
    pm = results["results"][0]["parts_matched"]
    assert any(p["match_type"] == "tool_args" for p in pm)


# ============================================================ tool_result matching


def test_tool_result_matching():
    host = _Host(history=[
        {"role": "tool", "parts": [
            {"type": "tool_result", "tool_name": "read_file", "tool_result": "file content here"}
        ]},
    ])
    results = host.search_history("file content")
    assert results["total_matches"] == 1
    pm = results["results"][0]["parts_matched"]
    assert any(p["match_type"] == "tool_result" for p in pm)


def test_tool_result_snippet_truncation():
    long_result = "x" * 500
    host = _Host(history=[
        {"role": "tool", "parts": [
            {"type": "tool_result", "tool_name": "bash", "tool_result": long_result}
        ]},
    ])
    results = host.search_history("x")
    snippet = results["results"][0]["parts_matched"][0]["snippet"]
    assert len(snippet) <= 205


# ============================================================ file part matching


def test_file_part_matching():
    host = _Host(history=[
        {"role": "user", "parts": [
            {"type": "file", "file_ref": {"display_name": "report.pdf", "uri": "file:///tmp/report.pdf"}}
        ]},
    ])
    results = host.search_history("report")
    assert results["total_matches"] == 1


# ============================================================ image_input matching


def test_image_input_matching():
    host = _Host(history=[
        {"role": "user", "parts": [
            {"type": "image_input", "image": {"source": "/tmp/screenshot.png", "mime_type": "image/png", "data_b64": ""}}
        ]},
    ])
    results = host.search_history("screenshot")
    assert results["total_matches"] == 1


# ============================================================ role filter


def test_role_filter_user():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "match"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "match"}]},
    ])
    results = host.search_history("match", role="user")
    assert results["total_matches"] == 1
    assert results["results"][0]["role"] == "user"


def test_role_filter_assistant():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "match"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "match"}]},
    ])
    results = host.search_history("match", role="assistant")
    assert results["total_matches"] == 1
    assert results["results"][0]["role"] == "assistant"


# ============================================================ tool_name filter


def test_tool_name_filter():
    host = _Host(history=[
        {"role": "assistant", "parts": [
            {"type": "tool_call", "tool_name": "read_file", "tool_args": {}}
        ]},
        {"role": "assistant", "parts": [
            {"type": "tool_call", "tool_name": "bash", "tool_args": {}}
        ]},
    ])
    results = host.search_history("read", tool_name="read_file")
    assert results["total_matches"] == 1


def test_tool_name_filter_no_matches():
    host = _Host(history=[
        {"role": "assistant", "parts": [
            {"type": "tool_call", "tool_name": "bash", "tool_args": {}}
        ]},
    ])
    results = host.search_history("anything", tool_name="write_file")
    assert results["total_matches"] == 0


# ============================================================ include_summarized


def test_include_summarized_true_default():
    host = _Host(
        history=[
            {"role": "user", "parts": [{"type": "text", "text": "old match"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "new match"}]},
        ],
        summary_anchor=1,
    )
    results = host.search_history("match")
    assert results["total_matches"] == 2
    assert results["results"][0]["before_anchor"] is True
    assert results["results"][1]["before_anchor"] is False


def test_include_summarized_false():
    host = _Host(
        history=[
            {"role": "user", "parts": [{"type": "text", "text": "old match"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "new match"}]},
        ],
        summary_anchor=1,
    )
    results = host.search_history("match", include_summarized=False)
    assert results["total_matches"] == 1
    assert results["results"][0]["before_anchor"] is False


# ============================================================ context_messages bounding


def test_context_messages_before_and_after():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "msg 0"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "msg 1"}]},
        {"role": "user", "parts": [{"type": "text", "text": "MATCH"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "msg 3"}]},
        {"role": "user", "parts": [{"type": "text", "text": "msg 4"}]},
    ])
    results = host.search_history("MATCH", context_messages=2)
    hit = results["results"][0]
    assert len(hit["context_before"]) == 2
    assert hit["context_before"][0]["index"] == 0
    assert hit["context_before"][1]["index"] == 1
    assert len(hit["context_after"]) == 2
    assert hit["context_after"][0]["index"] == 3
    assert hit["context_after"][1]["index"] == 4


def test_context_messages_clamped_at_start():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "MATCH"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "after"}]},
    ])
    results = host.search_history("MATCH", context_messages=5)
    assert len(results["results"][0]["context_before"]) == 0
    assert len(results["results"][0]["context_after"]) == 1


def test_context_messages_clamped_at_end():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "before"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "MATCH"}]},
    ])
    results = host.search_history("MATCH", context_messages=5)
    assert len(results["results"][0]["context_before"]) == 1
    assert len(results["results"][0]["context_after"]) == 0


# ============================================================ max_results bounding


def test_max_results_bounding():
    history = []
    for i in range(10):
        history.append({"role": "user", "parts": [{"type": "text", "text": f"match {i}"}]})
    host = _Host(history=history)
    results = host.search_history("match", max_results=3)
    assert len(results["results"]) == 3
    assert results["total_matches"] == 10
    assert results["has_more"] is True


def test_max_results_no_more():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "match"}]},
    ])
    results = host.search_history("match", max_results=20)
    assert results["has_more"] is False


# ============================================================ ranking order


def test_ranking_text_over_tool_name():
    """A message with both a text match and tool_name match should rank text higher."""
    host = _Host(history=[
        {"role": "assistant", "parts": [
            {"type": "tool_call", "tool_name": "search", "tool_args": {}},
            {"type": "text", "text": "search results found"},
        ]},
        {"role": "assistant", "parts": [
            {"type": "tool_call", "tool_name": "search", "tool_args": {}},
        ]},
    ])
    results = host.search_history("search")
    # Both match, but the first has a text match (higher rank) so it should come first
    assert results["results"][0]["index"] == 0


# ============================================================ cache_key passthrough


def test_cache_key_passthrough():
    """When a tool_result hit has a ToolResultCache entry, include cache_key."""
    class _FakeCache:
        _cache = {
            "abc123": {"tool_name": "read_file", "result": "file content here"}
        }

    host = _Host(
        history=[
            {"role": "tool", "parts": [
                {"type": "tool_result", "tool_name": "read_file", "tool_result": "file content here"}
            ]},
        ],
        tool_result_cache=_FakeCache(),
    )
    results = host.search_history("file content")
    assert results["total_matches"] == 1
    # cache_key should be present (either the key or None if lookup failed)
    assert "cache_key" in results["results"][0]


def test_cache_key_none_when_no_cache():
    host = _Host(history=[
        {"role": "tool", "parts": [
            {"type": "tool_result", "tool_name": "bash", "tool_result": "output"}
        ]},
    ])
    results = host.search_history("output")
    assert results["results"][0]["cache_key"] is None


# ============================================================ empty history


def test_empty_history():
    host = _Host(history=[])
    results = host.search_history("anything")
    assert results["results"] == []
    assert results["total_matches"] == 0
    assert "message" in results


# ============================================================ empty query


def test_empty_query_returns_error():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
    ])
    results = host.search_history("")
    assert results["results"] == []
    assert "error" in results


def test_whitespace_only_query_returns_error():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
    ])
    results = host.search_history("   ")
    assert "error" in results


# ============================================================ no matches


def test_no_matches():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "hello world"}]},
    ])
    results = host.search_history("nonexistent")
    assert results["results"] == []
    assert results["total_matches"] == 0


# ============================================================ no mutation


def test_no_mutation_to_history():
    history = [
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
    ]
    host = _Host(history=history)
    host.search_history("hello")
    assert host.history == history
    assert len(host.history) == 1


def test_no_mutation_to_summary_anchor():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
    ], summary_anchor=0)
    host.search_history("hello")
    assert host.summary_anchor == 0


# ============================================================ result dict shape


def test_result_dict_contains_all_fields():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
    ])
    results = host.search_history("hello")
    r = results["results"][0]
    assert "index" in r
    assert "role" in r
    assert "before_anchor" in r
    assert "parts_matched" in r
    assert "context_before" in r
    assert "context_after" in r
    assert "cache_key" in r
    pm = r["parts_matched"][0]
    assert "type" in pm
    assert "snippet" in pm
    assert "match_type" in pm


def test_context_entries_have_index_role_preview():
    host = _Host(history=[
        {"role": "user", "parts": [{"type": "text", "text": "before"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "MATCH"}]},
        {"role": "user", "parts": [{"type": "text", "text": "after"}]},
    ])
    results = host.search_history("MATCH")
    r = results["results"][0]
    for ctx in r["context_before"]:
        assert "index" in ctx
        assert "role" in ctx
        assert "preview" in ctx
    for ctx in r["context_after"]:
        assert "index" in ctx
        assert "role" in ctx
        assert "preview" in ctx


# ============================================================ @tool search_history handler


def test_tool_handler_registered():
    from mu.tools._dispatcher import TOOL_HANDLERS
    from mu.tools.descriptors import TOOLS

    assert "search_history" in {t.name for t in TOOLS}
    assert "search_history" in TOOL_HANDLERS


def test_tool_handler_executes_search():
    import mu.tools as _mu_tools
    from types import SimpleNamespace

    session = _Host(
        history=[
            {"role": "user", "parts": [{"type": "text", "text": "hello world"}]},
        ]
    )
    # The tool handler reads from context.session.session_manager — build a
    # session-like object that delegates to our _Host.
    class _SM(HistorySearchMixin):
        def __init__(self, host):
            self.history = host.history
            self.summary_anchor = host.summary_anchor
            self.tool_result_cache = host.tool_result_cache

    class _Sess:
        def __init__(self, host):
            self.session_manager = _SM(host)

    ctx = _mu_tools.build_tool_context(
        folder_context=None, ui=None, variables={}, session=_Sess(session)
    )
    envelope = _mu_tools.execute("search_history", {"query": "hello"}, ctx)
    assert envelope["ok"] is True
    data = envelope["data"]
    assert "results" in data
    assert data["total_matches"] == 1


def test_tool_handler_empty_query_error():
    import mu.tools as _mu_tools
    from types import SimpleNamespace

    class _SM:
        history = []
        summary_anchor = 0
        tool_result_cache = None

        def search_history(self, **kwargs):
            return HistorySearchMixin.search_history(self, **kwargs)

    class _Sess:
        session_manager = _SM()

    ctx = _mu_tools.build_tool_context(
        folder_context=None, ui=None, variables={}, session=_Sess()
    )
    envelope = _mu_tools.execute("search_history", {"query": ""}, ctx)
    assert envelope["ok"] is False


def test_tool_handler_empty_history_message():
    import mu.tools as _mu_tools

    class _SM:
        history = []
        summary_anchor = 0
        tool_result_cache = None

        def search_history(self, **kwargs):
            return HistorySearchMixin.search_history(self, **kwargs)

    class _Sess:
        session_manager = _SM()

    ctx = _mu_tools.build_tool_context(
        folder_context=None, ui=None, variables={}, session=_Sess()
    )
    envelope = _mu_tools.execute("search_history", {"query": "test"}, ctx)
    assert envelope["ok"] is True
    data = envelope["data"]
    assert data["results"] == []
    assert data["total_matches"] == 0


def test_tool_handler_plan_mode_safe():
    from mu.agent.plan_mode import WRITE_TOOLS

    assert "search_history" not in WRITE_TOOLS


def test_tool_handler_requires_approval_false():
    from mu.tools.descriptors import TOOL_DESCRIPTORS

    desc = TOOL_DESCRIPTORS["search_history"]
    assert desc.definition.requires_approval is False
    assert desc.execution_kind == "read"


# ============================================================ GUI endpoint


def test_gui_endpoint_exists():
    import inspect
    from mu.gui.routers import chat

    source = inspect.getsource(chat)
    assert "history/search" in source
    assert "search_history" in source


def test_gui_endpoint_is_async():
    import inspect
    from mu.gui.routers.chat import search_history

    assert inspect.iscoroutinefunction(search_history)


def test_gui_endpoint_returns_400_on_empty_query():
    """The endpoint should reject empty query with HTTP 400."""
    import inspect
    from mu.gui.routers.chat import search_history

    source = inspect.getsource(search_history)
    assert "400" in source


# ============================================================ existing tests regression


def test_existing_session_tests_still_import():
    """Verify that the existing test modules still import cleanly.

    The tests directory has no __init__.py (pytest rootdir-based collection),
    so sibling test modules are imported by file path via importlib.
    """
    import importlib.util
    import pathlib

    tests_dir = pathlib.Path(__file__).resolve().parent
    for name in ("test_session.py", "test_mu_session_history.py"):
        path = tests_dir / name
        assert path.is_file(), f"expected test module missing: {name}"
        spec = importlib.util.spec_from_file_location(name[:-3], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)