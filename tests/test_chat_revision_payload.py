"""Round-31 F35: JS-safe revision transport in chat state payloads.

The per-session revision is an unbounded counter in session.json; JSON
numbers above 2^53-1 lose precision in JavaScript clients, silently
corrupting the If-Match optimistic-concurrency token. Producers clamp:
revisions within the JS-safe range travel as numbers, anything larger as
its decimal string (``Number(str)`` parses both identically in JS).

Covers:
- utils/revision.js_safe_revision: pass-through / string-clamp / negative
  and non-int rejection;
- parse_revision_token: round-trip of both transport forms;
- watcher session_updated events (immediate + deferred reload) carry the
  post-reload revision, using getattr fallback when the SessionManager
  lacks a revision attribute (stub compatibility);
- chat.py send responses (chat + container kinds) and the turn_complete
  publish carry the revision.
"""

import asyncio
import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.revision import JS_SAFE_MAX_REVISION, js_safe_revision, parse_revision_token  # noqa: E402


class TestJsSafeRevision:
    def test_small_revision_passes_through_as_int(self):
        assert js_safe_revision(0) == 0
        assert js_safe_revision(42) == 42
        assert isinstance(js_safe_revision(42), int)

    def test_max_safe_integer_boundary(self):
        assert js_safe_revision(JS_SAFE_MAX_REVISION) == JS_SAFE_MAX_REVISION
        assert isinstance(js_safe_revision(JS_SAFE_MAX_REVISION), int)
        # One past the boundary must travel as a decimal string.
        clamped = js_safe_revision(JS_SAFE_MAX_REVISION + 1)
        assert clamped == str(JS_SAFE_MAX_REVISION + 1)
        assert isinstance(clamped, str)

    def test_string_form_round_trips_through_number(self):
        # The JS consumer contract: Number(value) parses both forms to the
        # same value (Python stand-in for the ECMAScript semantics).
        huge = JS_SAFE_MAX_REVISION + 12345
        assert int(js_safe_revision(huge)) == huge
        assert parse_revision_token(str(huge)) == huge

    def test_negative_clamps_to_zero(self):
        assert js_safe_revision(-1) == 0

    def test_non_int_rejected(self):
        with pytest.raises(TypeError):
            js_safe_revision("7")
        with pytest.raises(TypeError):
            js_safe_revision(None)
        with pytest.raises(TypeError):
            js_safe_revision(True)  # bool is an int subclass — explicitly rejected
        with pytest.raises(TypeError):
            js_safe_revision(1.5)


class TestParseRevisionToken:
    def test_round_trip_number_form(self):
        assert parse_revision_token(js_safe_revision(7)) == 7

    def test_round_trip_string_form(self):
        assert parse_revision_token(js_safe_revision(JS_SAFE_MAX_REVISION + 1)) == JS_SAFE_MAX_REVISION + 1

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_revision_token("abc")
        with pytest.raises(ValueError):
            parse_revision_token("")
        with pytest.raises(ValueError):
            parse_revision_token(None)
        with pytest.raises(ValueError):
            parse_revision_token(True)


class _Bus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


class _SM:
    """Watcher-test SessionManager stand-in (test_gui_watcher.py pattern)."""

    def __init__(self, home, name, revision=5):
        self.home = home
        self.name = name
        self.revision = revision
        self.reloads = 0
        self.fail = False

    def _load_session(self, name):
        if self.fail:
            raise RuntimeError("boom")
        self.reloads += 1


class _NoRevSM(_SM):
    """SessionManager lacking a revision attribute (legacy stub pattern)."""

    def __init__(self, home, name):
        super().__init__(home, name)
        del self.revision


def _make_app(sm, busy_event=None):
    busy = busy_event or threading.Event()
    locks = {sm.name: threading.RLock()}
    state = types.SimpleNamespace(
        bus=_Bus(),
        session_busy={sm.name: busy},
        session_busy_for=lambda n: state.session_busy.setdefault(n, threading.Event()),
        session_lock_for=lambda n: locks.setdefault(n, threading.RLock()),
        sessions={sm.name: types.SimpleNamespace(session_manager=sm)},
        session_by_name=lambda n=None: state.sessions.get(n),
    )
    app = types.SimpleNamespace(state=state)
    return app, busy, state


@pytest.fixture()
def isolated_home(monkeypatch, tmp_path):
    home = str(tmp_path / "mucli-watch")
    monkeypatch.setenv("MUCLI_HOME", home)
    import utils.config as config

    monkeypatch.setattr(config, "HISTORY_DIR", home, raising=False)
    os.makedirs(os.path.join(home, "sessions"), exist_ok=True)
    return home


def _watcher_for(app, name="s1"):
    from mu.gui.watcher import SessionWatcher, _Track

    w = SessionWatcher(app, interval=0.05)
    w._tracks.setdefault(name, _Track())
    return w, w._tracks[name]


def test_session_updated_immediate_carries_revision(isolated_home):
    """Idle external write: event carries the post-reload revision."""
    sm = _SM(isolated_home, "s1", revision=11)
    app, busy, state = _make_app(sm)
    w, track = _watcher_for(app)

    asyncio.run(w._handle_external_write(app.state.sessions["s1"], "s1", track))
    ev = state.bus.events[-1]
    assert ev["kind"] == "session_updated"
    assert ev["reloaded"] is True
    assert ev["revision"] == 11


def test_session_updated_deferred_reload_carries_fresh_revision(isolated_home):
    """Deferred reload path publishes the revision too (second event)."""
    sm = _SM(isolated_home, "s1", revision=12)
    app, busy, state = _make_app(sm)
    w, track = _watcher_for(app)
    loaded = dict(app.state.sessions)

    busy.set()
    asyncio.run(w._handle_external_write(app.state.sessions["s1"], "s1", track))
    first = state.bus.events[-1]
    assert first["deferred"] is True
    # Round-32 F7: the busy-path first event carries NO revision — the
    # in-memory counter is still the pre-write value until the deferred
    # reload applies. A stale token adopted here would poison If-Match.
    assert "revision" not in first

    busy.clear()
    asyncio.run(w._check_deferred_reloads(loaded))
    second = state.bus.events[-1]
    assert second["reason"] == "deferred_reload"
    assert second["revision"] == 12


def test_session_updated_failed_reload_omits_revision(isolated_home):
    """Round-32 F7: a failed reload publishes reloaded:false and NO
    revision token — any token would be stale, and clients adopting it
    would send a wrong If-Match on their next write."""
    sm = _SM(isolated_home, "s1", revision=11)
    sm.fail = True
    app, busy, state = _make_app(sm)
    w, track = _watcher_for(app)

    asyncio.run(w._handle_external_write(app.state.sessions["s1"], "s1", track))
    ev = state.bus.events[-1]
    assert ev["kind"] == "session_updated"
    assert ev["reloaded"] is False
    assert "revision" not in ev


def test_session_updated_getattr_fallback_without_revision(isolated_home):
    """SessionManager stubs without .revision get 0, not AttributeError."""
    sm = _NoRevSM(isolated_home, "s1")
    app, busy, state = _make_app(sm)
    w, track = _watcher_for(app)

    asyncio.run(w._handle_external_write(app.state.sessions["s1"], "s1", track))
    ev = state.bus.events[-1]
    assert ev["revision"] == 0

    busy.set()
    asyncio.run(w._handle_external_write(app.state.sessions["s1"], "s1", track))
    busy.clear()
    asyncio.run(w._check_deferred_reloads(dict(app.state.sessions)))
    assert state.bus.events[-1]["revision"] == 0


def _chat_app(sm):
    """Minimal FastAPI-less app state for chat.py send_message."""
    lock = threading.RLock()
    session = types.SimpleNamespace(
        session_manager=sm,
        variables={"session_type": "workspace"},
        stage_attachment_ids=lambda ids: None,
    )
    state = types.SimpleNamespace(
        bus=_Bus(),
        session_busy={},
        session_busy_for=lambda n: state.session_busy.setdefault(n, threading.Event()),
        session_lock_for=lambda n: lock,
        session_by_name=lambda n=None: session if n in (None, sm.name) else None,
        sessions={sm.name: session},
    )
    return types.SimpleNamespace(state=state)


def test_chat_send_response_carries_revision(isolated_home):
    """/api/chat/send (chat kind) response carries a JS-safe revision."""
    from mu.gui.routers import chat as chat_router

    sm = types.SimpleNamespace(
        name="s1",
        revision=9,
        history=[],
        current_session_name="s1",
        append_message=lambda *a, **k: None,
    )
    app = _chat_app(sm)
    request = types.SimpleNamespace(app=app, headers={})

    import asyncio as _asyncio

    payload = {"text": "hi", "session_name": "s1"}
    # Run the coroutine far enough to read the response: send_message
    # schedules _drive() as a background task, so cancel pending tasks after.
    response = _asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        _send_and_drain(chat_router, request, payload)
    )
    assert response["accepted"] is True
    assert response["kind"] == "chat"
    assert response["session_name"] == "s1"
    assert response["revision"] == 9
    # turn_complete publish also carries the revision.
    completes = [e for e in app.state.bus.events if e.get("kind") == "turn_complete"]
    assert completes and completes[-1]["revision"] == 9


async def _send_and_drain(chat_router, request, payload):
    task = asyncio.ensure_future(chat_router.send_message(request, payload))
    # Give the background _drive() task a chance to publish; it exits fast
    # because _summarize_result + publish are the only awaited work.
    await asyncio.sleep(0.05)
    response = await task
    # Cancel any lingering _drive() tasks from this test.
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return response