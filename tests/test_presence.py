"""Tests for presence beacons (cross-surface phase 3, G4)."""

import json
import os
import time

import pytest

from mu.session.presence import (
    PRESENCE_TTL_SECONDS,
    BeaconToucher,
    other_surfaces_active,
    prune_beacons,
    read_presence,
    write_beacon,
)
from mu.session.surface_sync import SurfaceSync


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / "mucli-home"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("MUCLI_HOME", str(home))
    import utils.config as _config

    # HISTORY_DIR is read from the config module object (not re-read from
    # the env per call) — patch it directly like test_session_revision does.
    monkeypatch.setattr(_config, "HISTORY_DIR", str(home))
    monkeypatch.delenv("MUCLI_SURFACE_SYNC", raising=False)
    return home


# ---- beacon write/read ------------------------------------------------------


def test_write_and_read_roundtrip(home):
    write_beacon("sess-a", "gui", pid=100)
    write_beacon("sess-a", "cli", pid=200)
    live = read_presence("sess-a")
    assert [b["pid"] for b in live] == [100, 200]  # sorted by started_at
    assert {b["surface"] for b in live} == {"gui", "cli"}
    assert all(b["busy"] is False for b in live)


def test_write_rejects_unknown_surface(home):
    with pytest.raises(ValueError):
        write_beacon("sess-a", "carrier-pigeon")


def test_write_rejects_invalid_names(home):
    for bad in ("", "../escape", "/abs", "a/b", "."):
        with pytest.raises(ValueError):
            write_beacon(bad, "cli")


def test_beacon_dir_separated_per_session(home):
    write_beacon("sess-a", "cli", pid=1)
    write_beacon("sess-b", "gui", pid=2)
    assert [b["pid"] for b in read_presence("sess-a")] == [1]
    assert [b["pid"] for b in read_presence("sess-b")] == [2]


def test_busy_flag_persists(home):
    write_beacon("sess-a", "cli", pid=1, busy=True)
    assert read_presence("sess-a")[0]["busy"] is True


def test_started_at_survives_touch(home):
    write_beacon("sess-a", "cli", pid=7)
    first = read_presence("sess-a")[0]["started_at"]
    time.sleep(0.01)
    write_beacon("sess-a", "cli", pid=7)  # touch
    again = read_presence("sess-a")[0]
    assert again["started_at"] == first  # not reset
    assert again["last_seen"] >= first  # refreshed


# ---- staleness / pruning ----------------------------------------------------


def test_stale_beacons_pruned(home):
    write_beacon("sess-a", "cli", pid=1)
    d = home / "sessions" / "sess-a" / "presence" / "1.json"
    data = json.load(open(d))
    data["last_seen"] = time.time() - (PRESENCE_TTL_SECONDS + 5)
    json.dump(data, open(d, "w"))
    assert read_presence("sess-a") == []
    assert not os.path.exists(d)  # pruned on read


def test_corrupt_beacon_treated_stale(home):
    d = home / "sessions" / "sess-a" / "presence"
    os.makedirs(d, exist_ok=True)
    (d / "42.json").write_text("{not json")
    assert prune_beacons("sess-a") == 1
    assert read_presence("sess-a") == []


def test_prune_returns_count(home):
    write_beacon("sess-a", "cli", pid=1)
    assert prune_beacons("sess-a") == 0  # fresh
    d = home / "sessions" / "sess-a" / "presence" / "1.json"
    data = json.load(open(d))
    data["last_seen"] = time.time() - 999
    json.dump(data, open(d, "w"))
    assert prune_beacons("sess-a") == 1


# ---- peer detection ----------------------------------------------------------


def test_other_surfaces_active(home):
    assert other_surfaces_active("sess-a") is False
    write_beacon("sess-a", "gui", pid=os.getpid())  # own pid only
    assert other_surfaces_active("sess-a") is False
    write_beacon("sess-a", "gui", pid=31337)
    assert other_surfaces_active("sess-a") is True


def test_missing_session_dir_is_absent(home):
    assert other_surfaces_active("never-created") is False
    assert read_presence("never-created") == []


# ---- SurfaceSync default gate -------------------------------------------------


def _fake_session(sm):
    class S:
        pass

    s = S()
    s.session_manager = sm
    s._current_turn_start_index = None
    return s


def test_surfacesync_gate_blocks_without_peer_beacon(home):
    from mu.session.manager import SessionManager

    mgr = SessionManager(session_name="gated")
    mgr.save_history()
    sync = SurfaceSync(_fake_session(mgr))  # default gate
    assert sync._gate() is False
    assert sync.check_once() is False  # gated out — not even baseline
    # Foreign peer appears → gate opens, poll proceeds.
    write_beacon("gated", "gui", pid=8888)
    assert sync._gate() is True
    assert sync.check_once() is False  # baseline now, not gated
    assert sync._initialized is True


def test_surfacesync_env_override_beats_gate(home, monkeypatch):
    from mu.session.manager import SessionManager

    monkeypatch.setenv("MUCLI_SURFACE_SYNC", "1")
    mgr = SessionManager(session_name="forced")
    mgr.save_history()
    sync = SurfaceSync(_fake_session(mgr))
    assert sync._gate() is True  # no beacon, env says run anyway


# ---- BeaconToucher thread ------------------------------------------------------


def test_toucher_writes_and_stops(home):
    name = "touchy"
    t = BeaconToucher(lambda: name, "cli", interval=0.2)
    t.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and not read_presence(name):
        time.sleep(0.05)
    assert read_presence(name), "toucher never wrote a beacon"
    t.stop()
    assert t._thread is None


def test_toucher_skips_when_no_session(home):
    t = BeaconToucher(lambda: None, "cli", interval=0.2)
    t.start()
    time.sleep(0.5)
    t.stop()  # no crash, nothing written anywhere readable
    assert t._thread is None


def test_toucher_busy_fn_reflects_state(home):
    state = {"busy": False}
    t = BeaconToucher(
        lambda: "busy-sess", "cli", busy_fn=lambda: state["busy"], interval=0.2
    )
    t.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and not read_presence("busy-sess"):
        time.sleep(0.05)
    assert read_presence("busy-sess")[0]["busy"] is False
    state["busy"] = True
    deadline = time.time() + 3.0
    while time.time() < deadline and read_presence("busy-sess")[0]["busy"] is False:
        time.sleep(0.05)
    assert read_presence("busy-sess")[0]["busy"] is True
    t.stop()