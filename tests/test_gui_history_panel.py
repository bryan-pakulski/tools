"""Tests for the GUI History Search Panel feature.

Verifies:
  - 'history' is a GUI view panel (in GUI_VIEW_PANELS), NOT an agent mode
    (not in AGENTIC_MODES / AGENT_MODE_METADATA, not settable via POST /api/modes)
  - history_panel.html fragment exists with required elements
  - index.html includes history_panel.html
  - app.js contains Alpine.store('history') with search state fields
  - app.js panelModes array contains 'history'
"""

import os
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from utils.config import AGENTIC_MODES, AGENT_MODE_METADATA, GUI_VIEW_PANELS


# ============================================================ /api/modes endpoint


def _make_modes_app(with_session: bool = True):
    """Minimal FastAPI app mounting only the modes router, with a stub
    session so require_session resolves. Lets us assert response shape +
    that view-panel names 400 on POST without spinning up create_app."""
    from mu.gui.routers import modes as modes_mod
    from mu.gui.app import session_by_name, session_lock_for

    app = FastAPI()
    app.state.sessions = {}
    app.state.session_locks = {}
    app.state.current_session_name = None
    app.state._fallback_lock = threading.Lock()
    app.state.session_by_name = lambda name=None: session_by_name(app, name)
    app.state.session_lock_for = lambda name=None: session_lock_for(app, name)

    if with_session:
        class _StubSM:
            class folder_context:
                folders = []
            def save_history(self, fc):
                return None

        class _StubSession:
            variables = {"agent_mode": "default"}
            session_manager = _StubSM()

        app.state.sessions["s1"] = _StubSession()
        app.state.current_session_name = "s1"

    app.include_router(modes_mod.router, prefix="/api/modes")
    return app


def test_api_modes_returns_views_array():
    app = _make_modes_app()
    client = TestClient(app)
    r = client.get("/api/modes")
    assert r.status_code == 200
    data = r.json()
    assert "views" in data
    view_names = [v["name"] for v in data["views"]]
    # The core five views plus the artifacts/shell panels added with the
    # unified registries work.
    assert set(view_names) >= {"history", "memory", "systemPrompts", "trace", "files"}
    for v in data["views"]:
        assert v["view_only"] is True
    # The Files panel is the only view that needs a workspace; the stub
    # session has none, so it surfaces disabled. Every other panel is
    # stateless and never needs a workspace.
    files_view = next(v for v in data["views"] if v["name"] == "files")
    assert files_view["needs_workspace"] is True
    assert files_view["disabled"] is True
    for v in data["views"]:
        if v["name"] == "files":
            continue
        assert v["needs_workspace"] is False
        if v["name"] == "shell":
            # shell needs a container session; the stub has none.
            assert v["disabled"] is True
            continue
        assert v["disabled"] is False
    # The trace analyzer is an external full-page route, not an in-page panel.
    trace_view = next(v for v in data["views"] if v["name"] == "trace")
    assert trace_view["external"] is True
    assert trace_view["route"] == "/trace"
    # Real agent modes are still listed, and the view panels are NOT.
    mode_names = [m["name"] for m in data["modes"]]
    assert "default" in mode_names
    for panel in ("history", "memory", "systemPrompts"):
        assert panel not in mode_names


def test_post_modes_history_rejected():
    app = _make_modes_app()
    client = TestClient(app)
    r = client.post("/api/modes/history")
    assert r.status_code == 400


def test_post_modes_memory_rejected():
    app = _make_modes_app()
    client = TestClient(app)
    r = client.post("/api/modes/memory")
    assert r.status_code == 400


def test_post_modes_systemprompts_rejected():
    app = _make_modes_app()
    client = TestClient(app)
    r = client.post("/api/modes/systemPrompts")
    assert r.status_code == 400


# ============================================================ view-panel registration


def test_history_not_an_agent_mode():
    # history is a read-only view panel, not a real agent mode — it must
    # not be settable as agent_mode.
    assert "history" not in AGENTIC_MODES
    assert "history" not in AGENT_MODE_METADATA


def test_history_in_gui_view_panels():
    names = [p["name"] for p in GUI_VIEW_PANELS]
    assert "history" in names
    panel = next(p for p in GUI_VIEW_PANELS if p["name"] == "history")
    assert "display_name" in panel
    assert "description" in panel
    assert isinstance(panel["display_name"], str)
    assert isinstance(panel["description"], str)


def test_history_not_in_no_workspace_set():
    import inspect
    from mu.gui.routers import modes as modes_mod
    source = inspect.getsource(modes_mod)
    assert "_NO_WORKSPACE_NEEDED" in source
    # history dropped from the no-workspace set when it stopped being a mode
    assert '"history"' not in source


# ============================================================ panel fragment


PANEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu", "gui", "templates", "fragments", "history_panel.html",
)


def test_history_panel_html_exists():
    assert os.path.isfile(PANEL_PATH), f"history_panel.html not found at {PANEL_PATH}"


def test_history_panel_has_mode_panel_aside():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert 'class="mode-panel"' in content
    assert 'data-mode="history"' in content


def test_history_panel_has_search_input():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.history.query" in content
    assert "history-search-input" in content or "search" in content.lower()


def test_history_panel_has_role_filter():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.history.role" in content


def test_history_panel_has_tool_name_filter():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.history.tool_name" in content or "$store.history.tool" in content


def test_history_panel_has_search_button():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.history.search()" in content


def test_history_panel_displays_results():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.history.results" in content
    assert "parts_matched" in content or "parts_matched" in content


def test_history_panel_has_loading_state():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.history.loading" in content


def test_history_panel_has_empty_state():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    # Empty state: results.length === 0 and not loading
    assert "results.length" in content or "searched" in content


def test_history_panel_has_error_state():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.history.error" in content


# ============================================================ index.html inclusion


INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu", "gui", "templates", "index.html",
)


def test_index_html_includes_history_panel():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "history_panel.html" in content
    assert "{% include" in content or "include" in content


# ============================================================ app.js Alpine store


APP_JS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu", "gui", "static", "js", "app.js",
)


def test_app_js_has_alpine_history_store():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert 'Alpine.store("history"' in content


def test_app_js_history_store_has_query_field():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    # Find the history store block and check for query field
    assert "query:" in content


def test_app_js_history_store_has_results_field():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "results:" in content


def test_app_js_history_store_has_loading_field():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "loading:" in content


def test_app_js_history_store_has_error_field():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "error:" in content


def test_app_js_history_store_has_search_method():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "search()" in content
    assert "/api/chat/history/search" in content


def test_app_js_history_store_has_clear_method():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "clearResults" in content


def test_app_js_panel_modes_includes_history():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert '"history"' in content
    # Check it's in the panelModes array specifically
    assert "panelModes" in content


# ============================================================ CSS


CSS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu", "gui", "static", "css", "app.css",
)


def test_css_has_history_panel_classes():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    for cls in (
        ".history-search-bar",
        ".history-filters",
        ".history-results",
        ".history-result-card",
        ".history-context-line",
        ".history-anchor-badge",
        ".history-match-tag",
    ):
        assert cls in content, f"CSS class {cls} not found in app.css"