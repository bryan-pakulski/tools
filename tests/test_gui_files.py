"""GUI Files panel — workspace explorer + editor backend.

Covers ``mu/gui/routers/files.py`` (tree / read / save / create / rename /
delete) and the panel-registration wiring (config ``GUI_VIEW_PANELS`` +
``modes.py`` workspace gating + index.html disabled binding + the vendored
CodeMirror script tag + the panel template / store / CSS).

Path safety is the heart of this panel: every mutating endpoint funnels
through ``_resolve_within``, which realpath's first and containment-checks via
``os.path.commonpath`` (stronger than the agent's ``check_bounds`` startswith),
then refuses secret paths via ``is_denied_path`` and ignored paths via
``FolderContext.is_ignored``. Writes are atomic (temp + ``os.replace``) and
keep a ``.bak``; save guards on ``expected_mtime`` so the agent editing the
file under the user doesn't get silently clobbered.
"""

from __future__ import annotations

import os
import shutil
import threading
import types
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mu.gui.app import session_by_name
from mu.gui.routers import files as files_mod
from mu.gui.routers import modes as modes_mod
from mu.gui.routers import sessions as sessions_mod
from mu.workspace.folder_context import FolderContext

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_JS = os.path.join(REPO, "mu", "gui", "static", "js", "app.js")
INDEX_HTML = os.path.join(REPO, "mu", "gui", "templates", "index.html")
BASE_HTML = os.path.join(REPO, "mu", "gui", "templates", "base.html")
FILES_PANEL = os.path.join(REPO, "mu", "gui", "templates", "fragments", "files_panel.html")
APP_CSS = os.path.join(REPO, "mu", "gui", "static", "css", "app.css")
CM_JS = os.path.join(REPO, "mu", "gui", "static", "vendor", "codemirror.min.js")
CM_CSS = os.path.join(REPO, "mu", "gui", "static", "vendor", "codemirror.min.css")
CM_MODES = os.path.join(REPO, "mu", "gui", "static", "vendor", "codemirror-modes.min.js")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _fake_session(folder_context):
    sm = SimpleNamespace(folder_context=folder_context)
    return SimpleNamespace(
        folder_context=folder_context,
        variables={},
        session_manager=sm,
    )


def _make_app(router, prefix, session):
    app = FastAPI()
    app.state.sessions = {"s1": session}
    app.state.session_locks = {"s1": threading.Lock()}
    app.state.current_session_name = "s1"
    app.state._fallback_lock = threading.Lock()
    app.state.session_by_name = lambda name=None: session_by_name(app, name)
    app.include_router(router, prefix=prefix)
    return app


@pytest.fixture
def workspace(tmp_path):
    """A temp workspace with a couple of files, a secret, and a gitignore."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "hello.py").write_text("print('hi')\n")
    (root / "sub").mkdir()
    (root / "sub" / "note.md").write_text("# note\n")
    (root / ".env").write_text("SECRET=1\n")
    (root / "ignored.log").write_text("x" * 50)
    (root / ".gitignore").write_text("*.log\n")
    fc = FolderContext()
    fc.add_folder(str(root))
    return str(root), fc


@pytest.fixture
def client(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    return TestClient(app), root, fc


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------


def test_tree_lists_files_dirs_first(client):
    c, root, _ = client
    r = c.get("/api/files/tree")
    assert r.status_code == 200
    roots = r.json()["roots"]
    assert len(roots) == 1
    children = roots[0]["children"]
    names = [child["name"] for child in children]
    assert "hello.py" in names
    assert "sub" in names
    # gitignored file filtered out
    assert "ignored.log" not in names
    # secret file filtered out (FolderContext ignores .env by default)
    assert ".env" not in names
    # dirs first
    is_dirs = [child["is_dir"] for child in children]
    assert is_dirs == sorted(is_dirs, reverse=True)


def test_tree_no_workspace_returns_409(tmp_path):
    fc = FolderContext()  # no folders
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).get("/api/files/tree")
    assert r.status_code == 409


def test_tree_subtree_expand(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).get("/api/files/tree", params={"path": os.path.join(root, "sub")})
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert any(e["name"] == "note.md" and not e["is_dir"] for e in entries)


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_text_file(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).get("/api/files/read", params={"path": os.path.join(root, "hello.py")})
    assert r.status_code == 200
    d = r.json()
    assert d["content"] == "print('hi')\n"
    assert d["readonly"] is False
    assert "mtime" in d


def test_read_secret_path_refused(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).get("/api/files/read", params={"path": os.path.join(root, ".env")})
    assert r.status_code == 403


def test_read_out_of_workspace_refused(workspace, tmp_path):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("nope")
    r = TestClient(app).get("/api/files/read", params={"path": str(outside)})
    assert r.status_code == 403


def test_read_traversal_refused(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    # /etc/hosts via .. from the workspace root
    r = TestClient(app).get("/api/files/read", params={"path": os.path.join(root, "..", "..", "etc", "hosts")})
    assert r.status_code == 403


def test_read_binary_returns_readonly(workspace, tmp_path):
    root, fc = workspace
    (tmp_path / "ws" / "blob.bin").write_bytes(b"\x00\x01\x02\x00binary")
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).get("/api/files/read", params={"path": os.path.join(root, "blob.bin")})
    assert r.status_code == 200
    d = r.json()
    assert d["readonly"] is True
    assert "binary" in d["why"].lower()


def test_read_oversized_returns_readonly(workspace):
    root, fc = workspace
    big = os.path.join(root, "big.txt")
    with open(big, "w") as f:
        f.write("a" * (files_mod._MAX_EDITABLE_BYTES + 10))
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).get("/api/files/read", params={"path": big})
    assert r.status_code == 200
    d = r.json()
    assert d["readonly"] is True
    assert "large" in d["why"].lower()


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_save_atomic_with_backup(workspace):
    root, fc = workspace
    path = os.path.join(root, "hello.py")
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).post("/api/files/save", json={
        "path": path, "content": "print('bye')\n",
    })
    assert r.status_code == 200, r.text
    assert open(path).read() == "print('bye')\n"
    # backup of the previous content kept
    assert os.path.exists(path + ".bak")
    assert open(path + ".bak").read() == "print('hi')\n"


def test_save_stale_mtime_returns_409(workspace):
    root, fc = workspace
    path = os.path.join(root, "hello.py")
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).post("/api/files/save", json={
        "path": path, "content": "x\n", "expected_mtime": 1.0,
    })
    assert r.status_code == 409
    # file untouched
    assert open(path).read() == "print('hi')\n"


def test_save_out_of_workspace_refused(workspace, tmp_path):
    root, fc = workspace
    outside = tmp_path / "elsewhere.py"
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).post("/api/files/save", json={
        "path": str(outside), "content": "x\n",
    })
    assert r.status_code == 403
    assert not outside.exists()


def test_save_creates_new_file_no_backup(workspace):
    root, fc = workspace
    new = os.path.join(root, "brand_new.py")
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).post("/api/files/save", json={
        "path": new, "content": "new\n",
    })
    assert r.status_code == 200, r.text
    assert open(new).read() == "new\n"
    # brand-new file has no prior content → no .bak
    assert not os.path.exists(new + ".bak")


# ---------------------------------------------------------------------------
# create / rename / delete
# ---------------------------------------------------------------------------


def test_create_file_and_dir(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).post("/api/files/create", json={
        "path": os.path.join(root, "made.py"), "is_dir": False,
    })
    assert r.status_code == 200
    assert os.path.isfile(os.path.join(root, "made.py"))
    r = TestClient(app).post("/api/files/create", json={
        "path": os.path.join(root, "madefolder"), "is_dir": True,
    })
    assert r.status_code == 200
    assert os.path.isdir(os.path.join(root, "madefolder"))


def test_create_outside_workspace_refused(workspace, tmp_path):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).post("/api/files/create", json={
        "path": str(tmp_path / "outside_ws" / "x.py"), "is_dir": False,
    })
    assert r.status_code == 403


def test_rename_containment(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    src = os.path.join(root, "hello.py")
    dst = os.path.join(root, "hello_renamed.py")
    r = TestClient(app).post("/api/files/rename", json={"from": src, "to": dst})
    assert r.status_code == 200
    assert os.path.isfile(dst)
    assert not os.path.exists(src)


def test_rename_outside_refused(workspace, tmp_path):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    src = os.path.join(root, "hello.py")
    dst = str(tmp_path / "elsewhere_renamed.py")
    r = TestClient(app).post("/api/files/rename", json={"from": src, "to": dst})
    assert r.status_code == 403
    # source untouched
    assert os.path.isfile(src)


def test_delete_file_keeps_backup(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    path = os.path.join(root, "hello.py")
    r = TestClient(app).delete("/api/files", params={"path": path})
    assert r.status_code == 200
    assert not os.path.exists(path)
    # recoverable
    assert os.path.exists(path + ".bak")


def test_delete_nonempty_dir_refused_without_recursive(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).delete("/api/files", params={"path": os.path.join(root, "sub")})
    assert r.status_code == 409
    assert os.path.isdir(os.path.join(root, "sub"))


def test_delete_dir_recursive(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).delete(
        "/api/files", params={"path": os.path.join(root, "sub"), "recursive": "true"}
    )
    assert r.status_code == 200
    assert not os.path.exists(os.path.join(root, "sub"))


def test_delete_workspace_root_refused(workspace):
    root, fc = workspace
    app = _make_app(files_mod.router, "/api/files", _fake_session(fc))
    r = TestClient(app).delete("/api/files", params={"path": root})
    assert r.status_code == 400
    assert os.path.isdir(root)


# ---------------------------------------------------------------------------
# modes.py workspace gating
# ---------------------------------------------------------------------------


def test_modes_surfaces_files_panel_with_workspace(workspace):
    _, fc = workspace
    app = _make_app(modes_mod.router, "/api/modes", _fake_session(fc))
    d = TestClient(app).get("/api/modes").json()
    files_view = [v for v in d["views"] if v["name"] == "files"][0]
    assert files_view["needs_workspace"] is True
    assert files_view["disabled"] is False


def test_modes_disables_files_when_no_workspace():
    fc = FolderContext()
    app = _make_app(modes_mod.router, "/api/modes", _fake_session(fc))
    d = TestClient(app).get("/api/modes").json()
    files_view = [v for v in d["views"] if v["name"] == "files"][0]
    assert files_view["needs_workspace"] is True
    assert files_view["disabled"] is True


# ---------------------------------------------------------------------------
# static-content wiring
# ---------------------------------------------------------------------------


def test_gui_view_panels_has_files_with_needs_workspace():
    from utils.config import GUI_VIEW_PANELS
    files = [p for p in GUI_VIEW_PANELS if p["name"] == "files"]
    assert files, "files panel missing from GUI_VIEW_PANELS"
    assert files[0].get("needs_workspace") is True


def test_app_js_has_files_store_and_panel_mode():
    src = open(APP_JS, encoding="utf-8").read()
    assert "Alpine.store(\"files\"" in src
    assert '"files"' in src  # panelModes entry
    assert "openFile" in src
    assert "/api/files/tree" in src
    assert "/api/files/read" in src
    assert "/api/files/save" in src
    assert "/api/files/create" in src
    assert "/api/files/rename" in src


def test_base_html_loads_codemirror_vendor():
    """CodeMirror was removed from the web shell (971d101); pin the absence."""
    src = open(BASE_HTML, encoding="utf-8").read()
    assert "codemirror" not in src.lower()
    assert "/static/js/app.js" in src


def test_index_html_includes_files_panel_and_disabled_binding():
    src = open(INDEX_HTML, encoding="utf-8").read()
    assert "fragments/files_panel.html" in src
    # the non-external tools selector honors v.disabled — it lives in the
    # panel_tabs fragment that index.html includes.
    tabs = open(os.path.join(REPO, "mu", "gui", "templates", "fragments", "panel_tabs.html"), encoding="utf-8").read()
    assert "v.disabled" in tabs
    assert ":disabled=\"v.disabled\"" in tabs


def test_files_panel_template_exists_with_cm_host_and_store():
    """Lightweight browser panel (1b8ec71): no embedded CodeMirror editor."""
    src = open(FILES_PANEL, encoding="utf-8").read()
    assert 'data-mode="files"' in src
    assert "$store.files" in src
    assert "filesBrowserPanel()" in src
    # The embedded editor stayed removed: no CM host, no CodeMirror calls.
    assert "CodeMirror" not in src
    assert "cmHost" not in src
    assert "mountEditor" not in src
    assert "Browse the attached workspace" in src


def test_css_has_files_panel_and_codemirror_theme():
    src = open(APP_CSS, encoding="utf-8").read()
    assert ".files-panel" in src
    assert ".files-tree" in src
    assert "var(--bg)" in src


def test_todos_are_not_offered_as_a_separate_view():
    index = open(INDEX_HTML, encoding="utf-8").read()
    js = open(APP_JS, encoding="utf-8").read()
    assert 'class="mode-option todo-view-option"' not in index
    assert "Todos</span>" not in index
    assert "$store.mode.setView('loop'); open = false" not in index
    assert 'class="todo-field"' not in index
    assert 'Alpine.store("loop").load();' in js
    assert 'setInterval(() => Alpine.store("loop").load(), 5000)' in js


def test_shell_animates_side_panels_and_centers_settings_modal():
    index = open(INDEX_HTML, encoding="utf-8").read()
    css = open(APP_CSS, encoding="utf-8").read()
    inspector = open(
        os.path.join(REPO, "mu", "gui", "templates", "fragments", "inspector.html"),
        encoding="utf-8",
    ).read()

    assert "transition: grid-template-columns" in css
    assert ".app.sidebar-hidden .sidebar" in css
    assert "transform: translateX(-16px)" in css
    assert ".app.panel-hidden .mode-panel" in css
    assert "flex-basis 0.28s" in css
    assert "transform: translateX(16px)" in css
    assert 'class="inspector-backdrop"' in inspector
    assert "place-items: center" in css
    assert 'role="dialog"' in inspector
    assert 'aria-modal="true"' in inspector
    assert "$store.inspector.openDrawer()" in index


def test_prompt_picker_preserves_all_multi_select_values_and_recovery_choice():
    chat = open(
        os.path.join(REPO, "mu", "gui", "templates", "fragments", "chat.html"),
        encoding="utf-8",
    ).read()
    js = open(APP_JS, encoding="utf-8").read()
    # Explicit change handling keeps all checkbox values in the array rather
    # than allowing a reactive re-render to collapse the selection to one.
    assert "setChoice(optValue(opt), $event.target.checked)" in chat
    assert "isChoiceSelected(optValue(opt))" in chat
    assert "const selected = Array.isArray(this.value) ? [...this.value] : [];" in js
    # `prompt_choices` needs a scalar `value`, unlike ask_user_choice's list.
    assert 'this.shape === "choices"' in js
    assert ' ? { value: real[0] || (hasOther ? this.otherText : "") }' in js


def test_prompt_card_remains_visible_when_answer_or_cancel_post_fails():
    js = open(APP_JS, encoding="utf-8").read()
    assert "if (!answered) return;" in js
    assert "if (!cancelled) return;" in js
    # Queue removal happens only after the endpoint accepted the response.
    assert "if (!r.ok) {\n                    Alpine.store(\"chat\").addError" in js
    assert "this._remove(id);\n                return true;" in js


def test_busy_session_can_be_detached_without_waiting_for_its_turn():
    """Leaving a broken turn must not block on the per-session agent lock."""
    app = FastAPI()
    busy = threading.Event()
    busy.set()
    app.state.current_session_name = "stuck"
    app.state.sessions = {"stuck": SimpleNamespace()}
    app.state.session_busy = {"stuck": busy}
    app.state.session_locks = {"stuck": threading.Lock()}
    app.state._fallback_lock = threading.Lock()
    app.state.session_busy_for = lambda name=None: app.state.session_busy[name or "stuck"]
    app.state.session_lock_for = lambda name=None: app.state.session_locks[name or "stuck"]
    app.state.unload_session = lambda **_kwargs: pytest.fail("busy session was unloaded")
    app.include_router(sessions_mod.router, prefix="/api/sessions")
    client = TestClient(app)

    blocked = client.delete("/api/sessions/active")
    assert blocked.status_code == 409
    detached = client.post("/api/sessions/active/detach")
    assert detached.status_code == 200
    assert detached.json()["detached"] is True
    assert app.state.current_session_name is None


def test_session_controls_use_nonblocking_detach_and_show_unload_errors():
    js = open(APP_JS, encoding="utf-8").read()
    assert 'fetch("/api/sessions/active/detach", { method: "POST" })' in js
    assert "d.detail || `Unload failed (${r.status})`" in js


def test_codemirror_vendor_files_present():
    """Vendor bundles were deleted with the embedded editor (1b8ec71)."""
    for path in (CM_JS, CM_CSS, CM_MODES):
        assert not os.path.isfile(path), f"CodeMirror vendor should stay removed: {path}"
