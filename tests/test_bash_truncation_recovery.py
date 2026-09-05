"""Bash truncation recoverability — oversized output must survive clipping.

The bash handler clips inline output at ``max_output_chars``. Historically the
full output was lost (no store write-through). These tests pin the recovery
contract: the FULL output is stored via ``tool_result_cache.store(force=True)``
and the inline text carries a recall hint with the cache key. Without a
session/store the legacy ``...[TRUNCATED]...`` behavior is preserved.
"""

from __future__ import annotations

from types import SimpleNamespace

from mu.tools.shell.handlers import bash_command


class _FakeCache:
    def __init__(self):
        self.stored = {}

    def store(self, call_id, tool_name, result, force=False):
        key = f"fakekey{len(self.stored) + 1}"
        self.stored[key] = (tool_name, result)
        return key


class _FakeSession:
    def __init__(self):
        self.tool_result_cache = _FakeCache()


class _FC:
    folders = ["/tmp"]

    def is_ignored(self, path):
        return False


def test_oversized_output_stored_with_recall_hint():
    sess = _FakeSession()
    out = bash_command(
        "python3 -c \"print('x' * 20000)\"",
        _FC(),
        timeout_seconds=30,
        max_output_chars=500,
        session=sess,
    )
    assert "...[TRUNCATED]..." in out
    assert "recall(" in out
    (key, (tool_name, full)) = next(iter(sess.tool_result_cache.stored.items()))
    assert tool_name == "bash"
    # Full un-truncated output survived in the store.
    assert len(full) >= 20000
    assert key in out


def test_timeout_partial_output_stored_with_recall_hint():
    sess = _FakeSession()
    out = bash_command(
        "python3 -c \"print('z' * 15000); import time; time.sleep(10)\"",
        _FC(),
        timeout_seconds=2,
        max_output_chars=3000,
        session=sess,
    )
    assert "timed out after 2 seconds" in out
    assert "recall(" in out
    (key, (tool_name, full)) = next(iter(sess.tool_result_cache.stored.items()))
    assert tool_name == "bash"
    assert len(full) >= 15000
    assert key in out


def test_small_output_not_stored():
    sess = _FakeSession()
    bash_command("echo hi", _FC(), timeout_seconds=15, max_output_chars=12000,
                 session=sess)
    assert sess.tool_result_cache.stored == {}


def test_no_session_falls_back_to_legacy_truncation():
    out = bash_command(
        "python3 -c \"print('y' * 20000)\"",
        _FC(),
        timeout_seconds=30,
        max_output_chars=500,
        session=None,
    )
    assert "...[TRUNCATED]..." in out
    assert "recall(" not in out