"""Round-31 F23: trace retention — the trace dir grew without bound
(371 MB / 2852 files live before the fix). Prune-on-create caps both
file count and total bytes, oldest-first, never deleting the active
file, and never raising into the agent loop."""

import os
import time

import pytest

from mu.trace import emitter as trace_emitter


@pytest.fixture()
def trace_home(tmp_path, monkeypatch):
    """Isolate the trace dir and reset the prune cooldown per test."""
    trace_dir = str(tmp_path / "trace")
    monkeypatch.setattr(trace_emitter, "trace_dir", lambda: trace_dir)
    monkeypatch.setattr(trace_emitter, "_last_prune_ts", 0.0)
    return trace_dir


def _mk(path, size, mtime):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * size)
    os.utime(path, (mtime, mtime))


def test_prune_keeps_newest_within_file_cap(trace_home):
    """More files than the cap -> oldest deleted, newest kept, active spared."""
    active = os.path.join(trace_home, "active_run.jsonl")
    for i in range(10):
        _mk(os.path.join(trace_home, f"s_run_{i}.jsonl"), 10, 1000 + i)
    _mk(active, 10, 9999)

    old_cap = trace_emitter._TRACE_MAX_FILES
    old_cd = trace_emitter._PRUNE_COOLDOWN_S
    trace_emitter._TRACE_MAX_FILES = 5
    trace_emitter._PRUNE_COOLDOWN_S = 0.0
    try:
        trace_emitter._enforce_trace_retention(active)
    finally:
        trace_emitter._TRACE_MAX_FILES = old_cap
        trace_emitter._PRUNE_COOLDOWN_S = old_cd

    remaining = sorted(os.listdir(trace_home))
    assert len(remaining) == 5  # 4 kept + active
    assert "active_run.jsonl" in remaining
    for i in range(6, 10):
        assert f"s_run_{i}.jsonl" in remaining  # newest 4 kept
    for i in range(0, 6):
        assert f"s_run_{i}.jsonl" not in remaining  # oldest pruned


def test_prune_respects_byte_cap(trace_home):
    """Total bytes over cap -> oldest-first deletion until under cap."""
    active = os.path.join(trace_home, "active_run.jsonl")
    # 4 files x 300 bytes = 1200; cap 700 -> keep newest 2 + active.
    for i in range(4):
        _mk(os.path.join(trace_home, f"s_run_{i}.jsonl"), 300, 1000 + i)
    _mk(active, 10, 9999)

    old_bytes = trace_emitter._TRACE_MAX_TOTAL_BYTES
    old_files = trace_emitter._TRACE_MAX_FILES
    old_cd = trace_emitter._PRUNE_COOLDOWN_S
    trace_emitter._TRACE_MAX_TOTAL_BYTES = 700
    trace_emitter._TRACE_MAX_FILES = 500
    trace_emitter._PRUNE_COOLDOWN_S = 0.0
    try:
        trace_emitter._enforce_trace_retention(active)
    finally:
        trace_emitter._TRACE_MAX_TOTAL_BYTES = old_bytes
        trace_emitter._TRACE_MAX_FILES = old_files
        trace_emitter._PRUNE_COOLDOWN_S = old_cd

    remaining = sorted(os.listdir(trace_home))
    assert "active_run.jsonl" in remaining
    assert "s_run_3.jsonl" in remaining  # newest
    assert "s_run_2.jsonl" in remaining  # 600 bytes total <= 700
    assert "s_run_1.jsonl" not in remaining  # 900 > 700, dropped
    assert "s_run_0.jsonl" not in remaining


def test_protected_bytes_consume_byte_budget(trace_home):
    """Round-33 F2: open/protected traces' SIZES consume the byte budget.

    The round-32 shape reserved protected files' slots but not their
    bytes, so a big open trace could coexist with a full cap of closed
    survivors (directory total ~2x the cap). Now the closed-survivor
    budget shrinks by the protected bytes.
    """
    open_trace = os.path.join(trace_home, "bigsession_run_open.jsonl")
    _mk(open_trace, 500, 8000)  # protected: open by another emitter
    trace_emitter._open_trace_paths.add(os.path.abspath(open_trace))
    try:
        # 3 closed x 300 bytes = 900; cap 700; protected eats 500 of it,
        # so closed survivors get only 200 -> just the newest one kept.
        for i in range(3):
            _mk(os.path.join(trace_home, f"k_run_{i}.jsonl"), 300, 1000 + i)
        active = os.path.join(trace_home, "active_run.jsonl")
        _mk(active, 10, 9999)

        old_bytes = trace_emitter._TRACE_MAX_TOTAL_BYTES
        old_files = trace_emitter._TRACE_MAX_FILES
        old_cd = trace_emitter._PRUNE_COOLDOWN_S
        trace_emitter._TRACE_MAX_TOTAL_BYTES = 700
        trace_emitter._TRACE_MAX_FILES = 500
        trace_emitter._PRUNE_COOLDOWN_S = 0.0
        try:
            trace_emitter._enforce_trace_retention(active)
        finally:
            trace_emitter._TRACE_MAX_TOTAL_BYTES = old_bytes
            trace_emitter._TRACE_MAX_FILES = old_files
            trace_emitter._PRUNE_COOLDOWN_S = old_cd

        remaining = sorted(os.listdir(trace_home))
        assert os.path.basename(open_trace) in remaining  # open: never deleted
        assert "active_run.jsonl" in remaining
        assert "k_run_2.jsonl" in remaining  # newest closed survivor
        # 500 (protected) + 300 (newest closed) = 800 > 700:
        # second-newest closed file must be drained now.
        assert "k_run_1.jsonl" not in remaining
        assert "k_run_0.jsonl" not in remaining
    finally:
        trace_emitter._open_trace_paths.discard(os.path.abspath(open_trace))


def test_prune_never_raises_and_spares_active(trace_home):
    """Garbage inputs (dir missing, active file absent) must not raise."""
    trace_emitter._enforce_trace_retention(
        os.path.join(trace_home, "missing_dir_subpath", "x.jsonl")
    )  # must not raise
    # Active path that doesn't exist yet: prune still runs, deletes nothing
    # it shouldn't.
    _mk(os.path.join(trace_home, "keep.jsonl"), 5, 1000)
    trace_emitter._enforce_trace_retention(os.path.join(trace_home, "new.jsonl"))
    assert os.path.exists(os.path.join(trace_home, "keep.jsonl"))


def test_prune_cooldown_prevents_rescan(trace_home, monkeypatch):
    """Within cooldown, prune is a no-op (no scan, no deletion)."""
    calls = {"n": 0}

    orig_listdir = os.listdir

    def counting_listdir(path):
        if str(path) == trace_home:
            calls["n"] += 1
        return orig_listdir(path)

    monkeypatch.setattr(trace_emitter.os, "listdir", counting_listdir)
    active = os.path.join(trace_home, "a.jsonl")
    _mk(active, 5, 1000)

    trace_emitter._last_prune_ts = time.time()  # just pruned
    trace_emitter._enforce_trace_retention(active)
    assert calls["n"] == 0  # cooldown skipped the scan

def test_open_trace_of_other_session_never_deleted(trace_home):
    """Round-31 F23 codex HIGH: a long-running session's OPEN trace file
    (not the active path of this prune) must survive keep-set pressure —
    unlinking it would send the live writes to an unlinked inode."""
    open_trace = os.path.join(trace_home, "longsession_run_old.jsonl")
    _mk(open_trace, 100, 1000)
    trace_emitter._open_trace_paths.add(os.path.abspath(open_trace))
    for i in range(8):
        _mk(os.path.join(trace_home, f"j_run_{i}.jsonl"), 100, 2000 + i)
    active = os.path.join(trace_home, "newsession_run_new.jsonl")
    _mk(active, 100, 9999)

    old_cap = trace_emitter._TRACE_MAX_FILES
    old_cd = trace_emitter._PRUNE_COOLDOWN_S
    trace_emitter._TRACE_MAX_FILES = 5  # 8 junk + open + active > 5
    trace_emitter._PRUNE_COOLDOWN_S = 0.0
    try:
        trace_emitter._enforce_trace_retention(active)
    finally:
        trace_emitter._TRACE_MAX_FILES = old_cap
        trace_emitter._PRUNE_COOLDOWN_S = old_cd
    trace_emitter._open_trace_paths.discard(os.path.abspath(open_trace))

    remaining = sorted(os.listdir(trace_home))
    assert "longsession_run_old.jsonl" in remaining  # open: spared
    assert "newsession_run_new.jsonl" in remaining  # active: spared
    # Newest junk kept up to the reduced budget.
    assert "j_run_7.jsonl" in remaining


def test_zero_byte_stubs_pruned_and_counted(trace_home):
    """Zero-byte .jsonl stubs are deleted first and consume file-cap
    budget — unbounded empty files must not defeat the cap."""
    for i in range(3):
        _mk(os.path.join(trace_home, f"z_run_{i}.jsonl"), 0, 5000 + i)
    active = os.path.join(trace_home, "active_run.jsonl")
    _mk(active, 10, 9999)

    old_cd = trace_emitter._PRUNE_COOLDOWN_S
    trace_emitter._PRUNE_COOLDOWN_S = 0.0
    try:
        trace_emitter._enforce_trace_retention(active)
    finally:
        trace_emitter._PRUNE_COOLDOWN_S = old_cd

    remaining = sorted(os.listdir(trace_home))
    assert remaining == ["active_run.jsonl"]  # all stubs gone


def test_byte_cap_always_keeps_newest_file(trace_home):
    """A single newest file larger than the byte cap is still kept —
    the drain must never delete ALL data."""
    active = os.path.join(trace_home, "active_run.jsonl")
    _mk(os.path.join(trace_home, "big_run_1.jsonl"), 10_000, 1000)
    _mk(os.path.join(trace_home, "big_run_2.jsonl"), 20_000, 2000)
    _mk(active, 10, 9999)

    old_bytes = trace_emitter._TRACE_MAX_TOTAL_BYTES
    old_cd = trace_emitter._PRUNE_COOLDOWN_S
    trace_emitter._TRACE_MAX_TOTAL_BYTES = 100  # smaller than any file
    trace_emitter._PRUNE_COOLDOWN_S = 0.0
    try:
        trace_emitter._enforce_trace_retention(active)
    finally:
        trace_emitter._TRACE_MAX_TOTAL_BYTES = old_bytes
        trace_emitter._PRUNE_COOLDOWN_S = old_cd

    remaining = sorted(os.listdir(trace_home))
    assert "big_run_2.jsonl" in remaining  # newest survivor kept
    assert "big_run_1.jsonl" not in remaining
    assert "active_run.jsonl" in remaining


def test_emitter_registers_and_unregisters_open_path(trace_home):
    """TraceEmitter._open adds its path to the protected registry;
    close() removes it so closed traces become prune-eligible."""
    path = os.path.join(trace_home, "reg_run_x.jsonl")
    em = trace_emitter.TraceEmitter("reg", "run_x", path)
    try:
        em.emit({"type": "iter"})
        assert os.path.abspath(path) in trace_emitter._open_trace_paths
    finally:
        em.close()
    assert os.path.abspath(path) not in trace_emitter._open_trace_paths
