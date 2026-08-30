"""Tests for the CLI-side surface sync watcher (cross-surface phase 2, G1)."""

import json
import os
import threading
import time

import pytest

from mu.session.manager import SessionManager
from mu.session.surface_sync import SurfaceSync, surface_sync_enabled


@pytest.fixture()
def sm(tmp_path, monkeypatch):
    home = tmp_path / "mucli-home"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("MUCLI_HOME", str(home))
    import utils.config as _config

    monkeypatch.setattr(_config, "HISTORY_DIR", str(home))
    # Presence-gated by default (phase 3); force the watcher on for these
    # detection tests — the gate behaviour itself is covered separately.
    monkeypatch.setenv("MUCLI_SURFACE_SYNC", "1")
    mgr = SessionManager(session_name="sync-target")
    mgr.history = [{"role": "user", "content": "seed turn"}]
    mgr.save_history()
    return mgr


def _foreign_write(sm, *, history=None, pid=999999):
    """Simulate another surface writing session.json (foreign pid, rev+1)."""
    path = sm._get_filepath(sm.current_session_name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["__writer_pid__"] = pid
    data["__writer_at__"] = time.time()
    data["revision"] = int(data.get("revision", 0)) + 1
    if history is not None:
        data["history"] = history
    tmp = path + ".ext"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data["revision"]


class _RecorderUI:
    def __init__(self):
        self.infos = []

    def show_info(self, msg):
        self.infos.append(msg)


class _FakeSession:
    """Minimal Session stand-in: manager + turn-scoped busy marker."""

    def __init__(self, sm, turn_index=None):
        self.session_manager = sm
        self._current_turn_start_index = turn_index


# ---- detection -------------------------------------------------------------


def test_external_write_detected_and_reloaded(sm):
    session = _FakeSession(sm)
    sync = SurfaceSync(session)
    assert sync.check_once() is False  # first poll = baseline, no reload
    sm.history = [{"role": "user", "content": "stale local view"}]
    new_rev = _foreign_write(sm, history=[{"role": "user", "content": "from GUI"}])
    assert sync.check_once() is True
    # History reloaded from disk, revision hydrated.
    assert sm.history == [{"role": "user", "content": "from GUI"}]
    assert sm.revision == new_rev


def test_own_write_ignored(sm):
    session = _FakeSession(sm)
    sync = SurfaceSync(session)
    sync.check_once()  # baseline
    sm.save_history()  # our pid writes
    assert sync.check_once() is False
    assert sm.history == [{"role": "user", "content": "seed turn"}]


def test_equal_or_older_revision_ignored(sm):
    session = _FakeSession(sm)
    sync = SurfaceSync(sm)
    sync = SurfaceSync(session)
    sync.check_once()  # baseline
    # Foreign pid but revision NOT newer than ours (writer forgot to bump).
    path = sm._get_filepath(sm.current_session_name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["__writer_pid__"] = 424242
    data.pop("revision", None)  # no revision -> 0 on read
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    assert sync.check_once() is False


def test_no_session_name_noop(tmp_path, monkeypatch):
    home = tmp_path / "mucli-home"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("MUCLI_HOME", str(home))
    import utils.config as _config

    monkeypatch.setattr(_config, "HISTORY_DIR", str(home))
    mgr = SessionManager()  # no session loaded
    sync = SurfaceSync(_FakeSession(mgr))
    assert sync.check_once() is False


def test_gate_false_blocks_poll(sm):
    session = _FakeSession(sm)
    sync = SurfaceSync(session, gate=lambda: False)
    _foreign_write(sm)
    assert sync.check_once() is False
    # And the gate keeps it blocked even for the baseline poll.


def test_gate_true_allows_poll(sm):
    session = _FakeSession(sm)
    sync = SurfaceSync(session, gate=lambda: True)
    sync.check_once()
    _foreign_write(sm)
    assert sync.check_once() is True


# ---- mid-turn deferral (G6) -------------------------------------------------


def test_busy_defers_reload_and_recheck_applies(sm):
    session = _FakeSession(sm, turn_index=0)  # mid-turn
    sync = SurfaceSync(session)
    sync.check_once()
    _foreign_write(sm, history=[{"role": "user", "content": "during turn"}])
    assert sync.check_once() is False
    assert sync.pending is True
    # Turn ends (send_message finally clears the index).
    session._current_turn_start_index = None
    assert sync.check_once() is True  # deferred reload applied, not dropped
    assert sm.history == [{"role": "user", "content": "during turn"}]
    assert sync.pending is False


def test_apply_pending_boundary_hook(sm):
    session = _FakeSession(sm, turn_index=0)
    ui = _RecorderUI()
    sync = SurfaceSync(session, ui=ui)
    sync.check_once()
    _foreign_write(sm, history=[{"role": "user", "content": "queued"}])
    sync.check_once()
    assert sync.pending is True
    session._current_turn_start_index = None
    assert sync.apply_pending() is True
    assert sm.history == [{"role": "user", "content": "queued"}]
    assert sync.pending is False
    # Second call is a no-op.
    assert sync.apply_pending() is False


# ---- notification -----------------------------------------------------------


def test_ui_notified_once_per_reload(sm):
    session = _FakeSession(sm)
    ui = _RecorderUI()
    sync = SurfaceSync(session, ui=ui)
    sync.check_once()
    _foreign_write(sm)
    assert sync.check_once() is True
    assert len(ui.infos) == 1
    assert "another surface" in ui.infos[0]
    _foreign_write(sm)  # rev bumps again
    assert sync.check_once() is True
    assert len(ui.infos) == 2


# ---- thread lifecycle --------------------------------------------------------


def test_thread_start_stop_lifecycle(sm):
    session = _FakeSession(sm)
    sync = SurfaceSync(session, interval=0.2)
    sync.start()
    assert sync._thread is not None and sync._thread.is_alive()
    time.sleep(0.3)
    sync.stop()
    assert sync._thread is None
    sync.stop()  # idempotent


def test_thread_detects_foreign_write_while_running(sm):
    session = _FakeSession(sm)
    sync = SurfaceSync(session, interval=0.2)
    sync.start()
    time.sleep(0.3)  # let it baseline
    _foreign_write(sm, history=[{"role": "user", "content": "thread spotted"}])
    deadline = time.time() + 3.0
    while time.time() < deadline and sm.history != [
        {"role": "user", "content": "thread spotted"}
    ]:
        time.sleep(0.05)
    sync.stop()
    assert sm.history == [{"role": "user", "content": "thread spotted"}]


# ---- opt-in gate --------------------------------------------------------------


def test_surface_sync_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("MUCLI_SURFACE_SYNC", raising=False)
    assert surface_sync_enabled() is False
    monkeypatch.setenv("MUCLI_SURFACE_SYNC", "1")
    assert surface_sync_enabled() is True


# ---- real manager integration (not the fake) ---------------------------------


def test_real_session_manager_roundtrip(sm):
    """Reload via the real manager keeps the phase-1 contract intact."""
    session = _FakeSession(sm)
    sync = SurfaceSync(session)
    sync.check_once()
    _foreign_write(sm, history=[{"role": "user", "content": "real mgr"}])
    before_rev = sm.revision
    assert sync.check_once() is True
    assert sm.revision == before_rev + 1
    assert sm.current_session_name == "sync-target"