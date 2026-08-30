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


def test_run_attempt_handles_timeout(tmp_path):
    out = tmp_path / "out.md"
    with patch.object(
        best_of.subprocess,
        "run",
        side_effect=best_of.subprocess.TimeoutExpired(cmd="codex", timeout=1),
    ):
        result = best_of._run_attempt("task", str(tmp_path), str(out), 1)
    assert result["ok"] is False
    assert "timeout" in result["output"]


def test_future_exception_recorded_as_failed_attempt():
    """A crashing _run_attempt must not abort the race (codex round-3 F4)."""
    calls = {"n": 0}

    def flaky(task, repo, out_path, attempt):
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
