"""G6 (§3.5): GUI deferred-reload re-check.

When the SessionWatcher detects an external write while a GUI turn is in
flight, it defers the reload. The fix under test: the reload must NOT be
dropped — the watcher re-arms its check and applies the reload (plus a
second session_updated event) once the busy flag clears.
"""

import asyncio
import os
import tempfile
import threading
import types

import pytest

from mu.gui.watcher import SessionWatcher, _Track


@pytest.fixture()
def isolated_home(monkeypatch):
    home = tempfile.mkdtemp(prefix="mucli-watch-")
    monkeypatch.setenv("MUCLI_HOME", home)
    import utils.config as config

    monkeypatch.setattr(config, "HISTORY_DIR", home, raising=False)
    os.makedirs(os.path.join(home, "sessions"), exist_ok=True)
    yield home


class _Bus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


class _SM:
    """Minimal SessionManager stand-in recording reload calls."""

    def __init__(self, home, name):
        self.home = home
        self.name = name
        self.reloads = 0
        self.fail = False

    def _load_session(self, name):
        if self.fail:
            raise RuntimeError("boom")
        self.reloads += 1


def _make_app(sm, busy_event=None):
    busy = busy_event or threading.Event()
    locks = {sm.name: threading.RLock()}

    state = types.SimpleNamespace(
        bus=_Bus(),
        session_busy={sm.name: busy},
        session_busy_for=lambda n: state.session_busy.setdefault(n, threading.Event()),
        session_lock_for=lambda n: locks.setdefault(n, threading.RLock()),
        sessions={sm.name: types.SimpleNamespace(session_manager=sm)},
    )
    app = types.SimpleNamespace(state=state)
    return app, busy, state


def _track_for(watcher, name):
    watcher._tracks.setdefault(name, _Track())
    return watcher._tracks[name]


def test_deferred_reload_fires_after_busy_clears(isolated_home):
    """Deferral arms the flag; next tick after busy clears applies the reload."""
    sm = _SM(isolated_home, "s1")
    app, busy, state = _make_app(sm)
    w = SessionWatcher(app, interval=0.05)
    track = _track_for(w, "s1")
    loaded = dict(app.state.sessions)

    # External write during a turn: busy set → deferred, no reload yet.
    busy.set()
    asyncio.run(w._handle_external_write(app.state.sessions["s1"], "s1", track))
    assert track.deferred_reload is True
    assert sm.reloads == 0

    # The deferral event tells clients a second event is coming.
    assert state.bus.events[-1]["deferred"] is True
    assert state.bus.events[-1]["reloaded"] is False

    # Turn finishes → next tick applies the skipped reload exactly once.
    busy.clear()
    asyncio.run(w._check_deferred_reloads(loaded))
    assert sm.reloads == 1
    assert track.deferred_reload is False

    # Second tick with no new external write: no double-fire.
    asyncio.run(w._check_deferred_reloads(loaded))
    assert sm.reloads == 1


def test_no_deferral_when_idle(isolated_home):
    """Not busy → immediate reload, no deferred flag, single event."""
    sm = _SM(isolated_home, "s1")
    app, busy, state = _make_app(sm)
    w = SessionWatcher(app, interval=0.05)
    track = _track_for(w, "s1")

    asyncio.run(w._handle_external_write(app.state.sessions["s1"], "s1", track))
    assert track.deferred_reload is False
    assert sm.reloads == 1
    ev = state.bus.events[-1]
    assert ev["deferred"] is False
    assert ev["reloaded"] is True


def test_deferred_reload_still_busy_keeps_waiting(isolated_home):
    """Flag stays armed while the turn is still in flight."""
    sm = _SM(isolated_home, "s1")
    app, busy, state = _make_app(sm)
    w = SessionWatcher(app, interval=0.05)
    track = _track_for(w, "s1")
    track.deferred_reload = True
    busy.set()

    asyncio.run(w._check_deferred_reloads(dict(app.state.sessions)))
    assert track.deferred_reload is True
    assert sm.reloads == 0


def test_deferred_reload_failure_keeps_flag_for_retry(isolated_home):
    """A failing reload keeps the flag armed (round-13 F6): the next tick
    retries instead of silently dropping the external write."""
    sm = _SM(isolated_home, "s1")
    sm.fail = True
    app, busy, state = _make_app(sm)
    w = SessionWatcher(app, interval=0.05)
    track = _track_for(w, "s1")
    track.deferred_reload = True

    # First tick: reload fails, flag stays armed for retry.
    asyncio.run(w._check_deferred_reloads(dict(app.state.sessions)))
    assert track.deferred_reload is True
    assert sm.reloads == 0
    assert state.bus.events == []  # no false reloaded=true event

    # Watcher survives; after the failure clears, the retry succeeds.
    sm.fail = False
    asyncio.run(w._check_deferred_reloads(dict(app.state.sessions)))
    assert track.deferred_reload is False
    assert sm.reloads == 1


def test_tick_wires_deferred_recheck(isolated_home):
    """_tick calls the re-check path (integration through the real entry point)."""
    sm = _SM(isolated_home, "s1")
    app, busy, state = _make_app(sm)
    w = SessionWatcher(app, interval=0.05)
    track = _track_for(w, "s1")
    track.deferred_reload = True

    called = []

    class _W(SessionWatcher):
        async def _check_deferred_reloads(self, loaded):
            called.append(True)
            # skip real reload — just prove the tick hook runs

    w2 = _W(app, interval=0.05)
    w2._tracks["s1"] = track
    asyncio.run(w2._tick())
    assert len(called) == 1


def test_track_default_off():
    assert _Track().deferred_reload is False