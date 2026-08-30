"""Tests for the GUI Memory Map panel feature.

Mirrors tests/test_gui_history_panel.py's structure:
  - 'memory' is a GUI view panel (in GUI_VIEW_PANELS), NOT an agent mode
    (not in AGENTIC_MODES / AGENT_MODE_METADATA, not settable via POST /api/modes)
  - memory_panel.html fragment exists with required elements
  - index.html includes memory_panel.html
  - app.js contains Alpine.store('memory') with grid/layers/load
  - app.js panelModes array contains 'memory'
  - app.js routes the context_snapshot SSE event
  - app.css has the memory panel classes
  - build_memory_snapshot: dims, determinism, layer change detection
"""

import os

from utils.config import AGENTIC_MODES, AGENT_MODE_METADATA, GUI_VIEW_PANELS

# ============================================================ view-panel registration


def test_memory_not_an_agent_mode():
    # memory is a read-only view panel, not a real agent mode — it must
    # not be settable as agent_mode.
    assert "memory" not in AGENTIC_MODES
    assert "memory" not in AGENT_MODE_METADATA


def test_memory_in_gui_view_panels():
    names = [p["name"] for p in GUI_VIEW_PANELS]
    assert "memory" in names
    panel = next(p for p in GUI_VIEW_PANELS if p["name"] == "memory")
    assert "display_name" in panel
    assert "description" in panel
    assert isinstance(panel["display_name"], str)
    assert isinstance(panel["description"], str)


def test_memory_not_in_no_workspace_set():
    from mu.gui.routers import modes as modes_mod
    import inspect

    source = inspect.getsource(modes_mod)
    assert "_NO_WORKSPACE_NEEDED" in source
    # memory dropped from the no-workspace set when it stopped being a mode
    assert '"memory"' not in source


# ============================================================ panel fragment


PANEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu",
    "gui",
    "templates",
    "fragments",
    "memory_panel.html",
)


def test_memory_panel_html_exists():
    assert os.path.isfile(PANEL_PATH)


def test_memory_panel_has_mode_panel_aside():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert 'class="mode-panel' in content
    assert 'data-mode="memory"' in content


def test_memory_panel_has_canvas():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "<canvas" in content
    assert "$store.memory.bindCanvas" in content
    assert "$store.memory.hoverCell" in content
    assert "memory-cell-tooltip" in content


def test_memory_panel_has_resolution_control():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.memory.resolution" in content


def test_memory_panel_has_legend():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.memory.layers" in content


# ============================================================ index.html inclusion


INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu",
    "gui",
    "templates",
    "index.html",
)


def test_index_html_includes_memory_panel():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "memory_panel.html" in content


# ============================================================ app.js Alpine store


APP_JS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu",
    "gui",
    "static",
    "js",
    "app.js",
)


def test_app_js_has_alpine_memory_store():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert 'Alpine.store("memory"' in content


def test_app_js_memory_store_has_load_method():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "/api/memory/state" in content


def test_app_js_memory_store_has_render_method():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "render()" in content
    assert "fillRect" in content
    assert "hoverCell(event)" in content
    assert "/api/memory/cell" in content


def test_app_js_memory_store_has_apply_snapshot():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "applySnapshot" in content


def test_app_js_has_temporal_context_renderers():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "/api/memory/timeline" in content
    assert "renderTimeline()" in content
    assert "_drawContextHeatmap" in content
    assert "_drawContextStream" in content
    assert "_drawContextChurn" in content


def test_app_js_context_canvases_are_theme_aware_and_redraw():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "_contextPalette()" in content
    assert 'getAttribute("data-theme") === "light"' in content
    assert "palette.light" in content
    assert "memory._scheduleRender()" in content


def test_app_js_panel_modes_includes_memory():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert '"memory"' in content
    assert "panelModes" in content


def test_app_js_routes_context_snapshot_event():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert 'case "context_snapshot"' in content


def test_app_js_memory_store_has_layer_modal_methods():
    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "layerModal" in content
    assert "openLayer" in content
    assert "closeLayer" in content
    assert "copyLayer" in content
    assert "/api/memory/content" in content


def test_memory_panel_legend_rows_are_clickable():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "$store.memory.openLayer(layer)" in content
    # Native buttons provide keyboard activation for the composition legend.
    assert "context-composition-legend" in content


def test_memory_panel_has_temporal_observability_views():
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Context Observatory" in content
    assert "Evolution heatmap" in content
    assert "$store.memory.setView('stream')" in content
    assert "$store.memory.setView('churn')" in content
    assert "$store.memory.bindTimelineCanvas" in content


def test_memory_layer_modal_fragment_exists():
    modal_path = os.path.join(
        os.path.dirname(PANEL_PATH),
        "memory_layer_modal.html",
    )
    assert os.path.isfile(modal_path)
    with open(modal_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "modal-backdrop" in content
    assert "$store.memory.layerModal" in content
    assert "$store.memory.copyLayer" in content
    assert "$store.memory.closeLayer" in content
    # copy/paste support: a copy button + selectable pre body
    assert "<pre" in content
    assert "copy" in content


def test_index_html_includes_memory_layer_modal():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "memory_layer_modal.html" in content


def test_css_has_memory_layer_modal_classes():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    for cls in (
        ".memory-layer-modal",
        ".memory-layer-pre",
        ".memory-legend-row",
    ):
        assert cls in content, f"CSS class {cls} not found in app.css"
    # the <pre> body is selectable so copy/paste works
    assert "user-select: text" in content


# ============================================================ CSS


CSS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mu",
    "gui",
    "static",
    "css",
    "app.css",
)


def test_css_has_memory_panel_classes():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    for cls in (
        ".memory-panel",
        ".memory-canvas-wrap",
        ".memory-canvas",
        ".memory-dividers",
        ".memory-divider",
        ".memory-legend",
        ".memory-legend-row",
    ):
        assert cls in content, f"CSS class {cls} not found in app.css"


def test_css_has_context_observatory_light_theme_palette():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert 'html[data-theme="light"] .context-observatory' in content
    for token in (
        "--context-card-surface",
        "--context-control-surface",
        "--context-plot-surface",
        "--context-tooltip-surface",
    ):
        assert token in content
    assert 'html[data-theme="light"] .context-pressure-pill' in content


# ============================================================ backend wiring


def test_memory_router_registered():
    import inspect
    from mu.gui import app as app_mod

    source = inspect.getsource(app_mod)
    assert "memory as memory_router" in source
    assert 'prefix="/api/memory"' in source


def test_memory_snapshot_hook_registered_idempotent():
    from mu.agent.hooks import default_registry
    from mu.gui.app import _register_memory_snapshot_hook

    _register_memory_snapshot_hook()
    _register_memory_snapshot_hook()  # second call must not duplicate
    names = [s.name for s in default_registry.list("pre_provider_call")]
    assert names.count("gui_memory_snapshot") == 1


def _make_content_app(session):
    """Minimal FastAPI app mounting only the memory router, wired so the
    /content endpoint resolves the given session as the focused one."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from mu.gui.routers import memory as memory_router

    app = FastAPI()
    app.state.sessions = {"s1": session}
    app.state.current_session_name = "s1"
    app.state.session_locks = {}
    app.state._fallback_lock = __import__("threading").Lock()

    def _session_by_name(name=None):
        return app.state.sessions.get(name or app.state.current_session_name)

    app.state.session_by_name = _session_by_name
    app.include_router(memory_router.router, prefix="/api/memory")
    return TestClient(app)


def test_memory_content_endpoint_returns_layer_body():
    session = _make_session()
    _add_user_turn(session, "alpha beta gamma delta epsilon zeta eta theta")
    client = _make_content_app(session)

    r = client.get("/api/memory/content?layer=L5")
    assert r.status_code == 200
    d = r.json()
    assert d["layer"] == "L5"
    assert "content" in d and isinstance(d["content"], str)
    # The L5 body is a human-readable conversation view, so the turn we
    # added must appear in the rendered contents.
    assert "alpha beta gamma" in d["content"]
    assert d["chars"] == len(d["content"])
    assert d["tokens"] >= 0
    assert d["error"] == ""
    # Hue is echoed so the modal header matches the legend swatch.
    assert d["hue"] == 358


def test_memory_content_endpoint_rejects_unknown_layer():
    session = _make_session()
    client = _make_content_app(session)
    r = client.get("/api/memory/content?layer=L9")
    assert r.status_code == 200
    d = r.json()
    assert d["error"] == "unknown layer"
    assert d["content"] == ""


def test_memory_timeline_endpoint_honors_explicit_session():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from mu.gui.memory_snapshot import build_memory_snapshot, record_context_snapshot
    from mu.gui.routers import memory as memory_router

    first = _make_session()
    second = _make_session()
    _add_user_turn(first, "first session context")
    _add_user_turn(second, "second session context with more words than first")
    first_snapshot = build_memory_snapshot(first, cols=32, rows=32)
    second_snapshot = build_memory_snapshot(second, cols=32, rows=32)
    record_context_snapshot(first, first_snapshot, recorded_at=1.0)
    record_context_snapshot(second, second_snapshot, recorded_at=2.0)

    app = FastAPI()
    app.state.sessions = {"first": first, "second": second}
    app.state.current_session_name = "first"
    app.state.session_by_name = lambda name=None: app.state.sessions.get(
        name or app.state.current_session_name
    )
    app.include_router(memory_router.router, prefix="/api/memory")
    client = TestClient(app)

    response = client.get("/api/memory/timeline?session_name=second")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["samples"] == 1
    assert body["points"][0]["at"] == 2.0
    assert body["points"][0]["total_tokens"] == second_snapshot["total_tokens"]


# ============================================================ builder behavior


def _make_session():
    from mu.session.session import Session, SessionManager
    from providers.base import LLMProvider, ProviderResponse

    class _DummyProvider(LLMProvider):
        def get_available_models(self):
            return ["dummy"]

        def generate(self, messages, system_prompt=None, thinking=False, tools=None):
            return ProviderResponse(
                text="ok", parts=[], input_tokens=0, output_tokens=0, total_tokens=0
            )

        def upload_file(self, file_path, mime_type):
            return None

    sm = SessionManager()
    return Session(_DummyProvider(), False, "you are a helpful assistant", sm)


def _add_user_turn(session, text):
    session.session_manager.history.append(
        {"role": "user", "parts": [{"type": "text", "text": text}]}
    )


def test_snapshot_empty_session_returns_grid_dims():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    snap = build_memory_snapshot(session, cols=64, rows=64)
    assert snap["active"] is True
    assert snap["cols"] == 64
    assert snap["rows"] == 64
    assert len(snap["grid"]) == 64
    assert all(len(r) == 64 for r in snap["grid"])
    # Every layer is represented in the legend, even if empty.
    ids = [l["id"] for l in snap["layers"]]
    assert ids == ["L0", "L1A", "L1B", "L2", "L3", "L4B", "L5"]


def test_snapshot_is_deterministic():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    _add_user_turn(session, "hello world this is a test of the memory map")
    a = build_memory_snapshot(session, cols=48, rows=48)
    b = build_memory_snapshot(session, cols=48, rows=48)
    assert a["grid"] == b["grid"]


def test_snapshot_uses_captured_request_estimate_for_trace_alignment(monkeypatch):
    import mu.gui.memory_snapshot as snapshot

    monkeypatch.setattr(
        snapshot,
        "collect_context_layers",
        lambda _session: [
            {
                "layer": lid,
                "name": lid,
                "current": 10 if lid == "L0" else 0,
                "maximum": 4096,
            }
            for lid in snapshot._LAYER_ORDER
        ],
    )

    class _Session:
        pass

    session = _Session()
    snap = snapshot.build_memory_snapshot(
        session, cols=48, rows=48, request_token_estimate=1234
    )

    assert snap["total_tokens"] == 1234
    assert snap["token_source"] == "pre_request_estimate"
    assert sum(layer["tokens"] for layer in snap["layers"]) == 1234


def test_snapshot_changed_history_changes_grid():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    _add_user_turn(session, "alpha beta gamma delta epsilon zeta eta theta")
    before = build_memory_snapshot(session, cols=48, rows=48)

    # Add a second turn — L5 (history) content changes, so the grid must
    # differ somewhere. Other layers are unchanged.
    _add_user_turn(session, "iota kappa lambda mu nu xi omicron pi")
    after = build_memory_snapshot(session, cols=48, rows=48)

    assert before["grid"] != after["grid"]
    # At least one layer registered a token change (L5 should grow).
    before_by_id = {l["id"]: l["tokens"] for l in before["layers"]}
    after_by_id = {l["id"]: l["tokens"] for l in after["layers"]}
    assert after_by_id["L5"] > before_by_id["L5"]


def test_context_timeline_tracks_provider_call_churn_without_raw_content():
    import json

    from mu.gui.memory_snapshot import (
        build_memory_snapshot,
        get_context_timeline,
        record_context_snapshot,
    )

    session = _make_session()
    secret_marker = "timeline-raw-content-must-not-leak"
    _add_user_turn(session, "alpha beta gamma")
    first = build_memory_snapshot(session, cols=32, rows=32)
    first_point = record_context_snapshot(session, first, recorded_at=10.0)

    _add_user_turn(session, secret_marker)
    second = build_memory_snapshot(session, cols=48, rows=48)
    second_point = record_context_snapshot(session, second, recorded_at=11.5)
    timeline = get_context_timeline(session)

    assert first_point["id"] == 1
    assert second_point["id"] == 2
    assert timeline["summary"]["samples"] == 2
    assert [point["at"] for point in timeline["points"]] == [10.0, 11.5]
    changed_l5 = next(
        layer for layer in timeline["points"][1]["layers"] if layer["id"] == "L5"
    )
    assert changed_l5["changed"] is True
    assert changed_l5["changed_chunks"] > 0
    assert timeline["points"][1]["changed_layers"] > 0
    assert timeline["points"][1]["churn_score"] > 0
    serialized = json.dumps(timeline)
    assert secret_marker not in serialized
    assert "_hashes" not in serialized


def test_context_timeline_detects_material_token_drop_as_compaction(monkeypatch):
    import mu.gui.memory_snapshot as snapshot

    class _Session:
        pass

    session = _Session()
    tokens = {layer_id: 0 for layer_id in snapshot._LAYER_ORDER}
    tokens["L5"] = 4000

    def _layers(_session):
        return [
            {
                "layer": layer_id,
                "name": layer_id,
                "current": tokens[layer_id],
                "maximum": 8192,
            }
            for layer_id in snapshot._LAYER_ORDER
        ]

    monkeypatch.setattr(snapshot, "collect_context_layers", _layers)
    monkeypatch.setattr(
        snapshot, "_layer_text", lambda _session, layer: layer * tokens[layer]
    )
    first = snapshot.build_memory_snapshot(session, cols=32, rows=32)
    snapshot.record_context_snapshot(session, first, recorded_at=1.0)
    tokens["L5"] = 2400
    second = snapshot.build_memory_snapshot(session, cols=32, rows=32)
    point = snapshot.record_context_snapshot(session, second, recorded_at=2.0)

    assert point["total_delta"] == -1600
    assert point["compaction"] is True


def test_snapshot_clamps_resolution():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    huge = build_memory_snapshot(session, cols=9999, rows=9999)
    assert huge["cols"] == 256
    assert huge["rows"] == 256
    tiny = build_memory_snapshot(session, cols=1, rows=1)
    assert tiny["cols"] == 16
    assert tiny["rows"] == 16


def test_snapshot_none_session_is_inactive():
    from mu.gui.memory_snapshot import build_memory_snapshot

    snap = build_memory_snapshot(None, cols=32, rows=32)
    assert snap["active"] is False
    assert len(snap["grid"]) == 32
    # Empty cells are 0 (int), not None — the grid is an int heatmap now.
    assert all(c == 0 for r in snap["grid"] for c in r)


def test_snapshot_layers_carry_hue_and_change_count():
    from mu.gui.memory_snapshot import build_memory_snapshot, LAYER_HUES

    session = _make_session()
    _add_user_turn(session, "a b c d e f g h i j k l m n o p")
    snap = build_memory_snapshot(session, cols=48, rows=48)
    for l in snap["layers"]:
        assert "hue" in l
        assert "change_count" in l
        assert l["hue"] == LAYER_HUES.get(l["id"], 0)
        assert isinstance(l["change_count"], int)


def test_snapshot_grid_is_int_heatmap():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    _add_user_turn(session, "a b c d e f g h i j k l m n o p")
    snap = build_memory_snapshot(session, cols=48, rows=48)
    flat = [c for r in snap["grid"] for c in r]
    # Every cell is an int in [0, 255]; 0 = empty, 1..255 = 1+heat.
    assert all(isinstance(c, int) for c in flat)
    assert all(0 <= c <= 255 for c in flat)


def test_snapshot_change_increments_heat():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    _add_user_turn(session, "alpha beta gamma delta epsilon zeta eta theta")
    before = build_memory_snapshot(session, cols=48, rows=48)
    before_max = max(c for r in before["grid"] for c in r)
    before_total_changes = sum(l["change_count"] for l in before["layers"])

    # Add a second turn — L5 content shifts, so some canonical chunks now
    # hash differently and their per-chunk change counter ticks up.
    _add_user_turn(session, "iota kappa lambda mu nu xi omicron pi rho sigma")
    after = build_memory_snapshot(session, cols=48, rows=48)
    after_max = max(c for r in after["grid"] for c in r)
    after_total_changes = sum(l["change_count"] for l in after["layers"])

    assert after_total_changes > before_total_changes
    assert after_max > before_max


def test_snapshot_reserves_a_real_free_capacity_band():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    # Only L5 (history) carries content; the other six layers are empty.
    _add_user_turn(session, "alpha beta gamma delta epsilon")
    snap = build_memory_snapshot(session, cols=48, rows=48)
    # Empty layers do not masquerade as allocated context. The map reserves
    # the unused window as a distinct FREE region instead.
    assert any((l["row_end"] - l["row_start"]) == 0 for l in snap["layers"])
    free = next(r for r in snap["regions"] if r["id"] == "FREE")
    assert free["tokens"] == snap["context_limit"] - snap["total_tokens"]
    assert free["row_end"] == 48
    assert snap["free_tokens"] == free["tokens"]


def test_snapshot_heat_keyed_per_resolution():
    from mu.gui.memory_snapshot import build_memory_snapshot

    session = _make_session()
    _add_user_turn(session, "alpha beta gamma delta epsilon zeta eta")
    build_memory_snapshot(session, cols=48, rows=48)  # seed 48 history

    # A second turn shifts L5 columns at the 48 resolution → heat accrues.
    _add_user_turn(session, "theta iota kappa lambda mu nu xi omicron")
    b48 = build_memory_snapshot(session, cols=48, rows=48)
    b64 = build_memory_snapshot(session, cols=64, rows=64)  # fresh resolution

    assert max(c for r in b48["grid"] for c in r) > 1  # 48 accumulated heat
    assert max(c for r in b64["grid"] for c in r) <= 1  # 64 starts clean


def test_snapshot_resize_keeps_change_signal():
    from mu.gui.memory_snapshot import build_memory_snapshot

    # Growing a layer resizes its band (re-chunk), but the changed regions
    # must still register as heat via fractional-position correspondence.
    session = _make_session()
    _add_user_turn(session, "alpha beta gamma delta epsilon zeta eta theta")
    before = build_memory_snapshot(session, cols=32, rows=32)
    before_changes = sum(l["change_count"] for l in before["layers"])

    _add_user_turn(session, "iota kappa lambda mu nu xi omicron pi rho sigma")
    after = build_memory_snapshot(session, cols=32, rows=32)
    after_changes = sum(l["change_count"] for l in after["layers"])

    assert after_changes > before_changes
    assert max(c for r in after["grid"] for c in r) > 1


def test_hash_color_is_deterministic_and_hex():
    from mu.gui.memory_snapshot import _hash_color

    a = _hash_color("some chunk")
    b = _hash_color("some chunk")
    assert a == b
    assert a.startswith("#") and len(a) == 7
    # Different input → (very likely) different color.
    assert _hash_color("different chunk") != a


def test_fingerprint_resolution_lru_cap():
    """Round-27 F2: per-session fingerprint map is LRU-bounded —
    requesting many distinct resolutions cannot grow it unboundedly."""
    from mu.gui import memory_snapshot as ms

    class _Session:
        pass

    session = _Session()
    for i in range(40):
        fp = ms._fingerprint(session)
        fp[(16 + i, 16 + i)] = {"L0": {"hashes": [1], "counts": [0]}}
    assert len(ms._fingerprint(session)) <= ms._MAX_RESOLUTIONS


def test_fingerprint_concurrent_access_thread_safe():
    """Round-27 F4: provider hooks (agent threads) and REST snapshots
    hammer _fingerprint concurrently — no exception, no lost keys."""
    import threading

    from mu.gui import memory_snapshot as ms

    class _Session:
        pass

    session = _Session()
    errors: list = []

    def hammer():
        try:
            for i in range(100):
                fp = ms._fingerprint(session)
                with ms._FINGERPRINTS_LOCK:
                    fp[(16, 16)] = fp.get((16, 16), {})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
