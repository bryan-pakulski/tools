"""Unit tests for the best_of_codex agent tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mu.tools.agent import best_of


def test_rank_orders_by_success_then_substance_then_speed():
    attempts = [
        {"attempt": 1, "ok": True, "output_chars": 100, "elapsed_seconds": 5.0},
        {"attempt": 2, "ok": False, "output_chars": 500, "elapsed_seconds": 1.0},
        {"attempt": 3, "ok": True, "output_chars": 300, "elapsed_seconds": 9.0},
        {"attempt": 4, "ok": True, "output_chars": 300, "elapsed_seconds": 2.0},
    ]
    ranked = best_of._rank(attempts)
    assert [r["attempt"] for r in ranked] == [4, 3, 1, 2]


def test_handler_rejects_missing_task():
    result = best_of.best_of_codex({"task": ""}, context=None)
    assert result["ok"] is False
    assert result["error_code"] == "invalid_args"


def test_handler_rejects_bad_repo(tmp_path):
    result = best_of.best_of_codex(
        {"task": "x", "repo": str(tmp_path / "nope")}, context=None
    )
    assert result["ok"] is False
    assert result["error_code"] == "bad_repo"


def test_handler_reports_codex_unavailable():
    with patch.object(best_of, "_codex_available", return_value=False):
        result = best_of.best_of_codex({"task": "x"}, context=None)
    assert result["ok"] is False
    assert result["error_code"] == "codex_unavailable"


def test_tool_registered_in_descriptors():
    from mu.tools.descriptors import TOOLS

    assert any(t.name == "best_of_codex" for t in TOOLS)


def _fake_popen_write_once(out_path, bytes_to_write, then_hang=True):
    """Return a Popen stub whose 'process' writes bytes_to_write to out_path
    on first poll then stays 'running' until killed (exit -9)."""
    import os as _os

    class _Proc:
        def __init__(self):
            self.returncode = None
            self._polled = 0
            self.killed = False
            self.stdout = None
            self.stderr = None

        def poll(self):
            self._polled += 1
            if self._polled == 1 and not _os.path.exists(out_path):
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(bytes_to_write)
            return None if not self.killed else -9

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.returncode = -9 if self.killed else 0
            return self.returncode

    return _Proc


def test_run_attempt_stall_kill_preserves_partial_output(tmp_path, monkeypatch):
    """A wedged attempt (no output growth) is killed after the stall window;
    its partial output survives and the result is flagged as a stall timeout."""
    import time as _time

    out = tmp_path / "out.md"
    monkeypatch.setattr(best_of, "_POLL_INTERVAL_SECONDS", 0.05)
    proc = _fake_popen_write_once(str(out), "partial plan")()

    monkeypatch.setattr(best_of.subprocess, "Popen", lambda *a, **k: proc)
    result = best_of._run_attempt(
        "task", str(tmp_path), str(out), 1, stall_seconds=1
    )
    assert result["ok"] is False
    assert result["timeout"] is True
    assert result["timeout_reason"] == "stall"
    assert result["partial_output"] is True
    assert "partial plan" in result["output"]
    assert _time.monotonic()  # sanity: time module still imported


def test_run_attempt_completes_without_hard_cap(tmp_path, monkeypatch):
    """Default (no timeout, healthy process) — runs to completion, no stall
    kill, output captured from the file."""
    out = tmp_path / "out.md"

    class _Proc:
        returncode = None
        stdout = None
        stderr = None

        def poll(self):
            # write output on first poll then exit cleanly
            if not out.exists():
                out.write_text("full result", encoding="utf-8")
                return None
            self.returncode = 0
            return 0

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(best_of.subprocess, "Popen", lambda *a, **k: _Proc())
    result = best_of._run_attempt("task", str(tmp_path), str(out), 1)
    assert result["ok"] is True
    assert result["timeout"] is False
    assert result["output"] == "full result"


def test_future_exception_recorded_as_failed_attempt():
    """A crashing _run_attempt must not abort the race (codex round-3 F4)."""
    calls = {"n": 0}

    def flaky(task, repo, out_path, attempt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash")
        return {
            "attempt": attempt,
            "ok": True,
            "returncode": 0,
            "elapsed_seconds": 1.0,
            "output_chars": 5,
            "output_path": out_path,
            "output": "hello",
        }

    with patch.object(best_of, "_run_attempt", side_effect=flaky), \
         patch.object(best_of, "_codex_available", return_value=True):
        result = best_of.best_of_codex(
            {"task": "x", "repo": "/tmp", "attempts": 2}, context=None
        )
    assert result["ok"] is True
    # one failed (crashed) + one ok
    ok_count = sum(1 for r in result["data"]["ranking"] if r["ok"])
    assert ok_count == 1
