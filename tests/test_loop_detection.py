from collections import deque

from mu.session.session import Session
from utils.config import DEFAULT_VARIABLES


def test_tool_sequence_repetition_detection():
    history = [
        "read_file:aaa -> list_dir:bbb",
        "read_file:aaa -> list_dir:bbb",
        "read_file:aaa -> list_dir:bbb",
    ]
    assert Session._is_repeated_tool_sequence(history, repeat_threshold=3) is True


def test_tool_sequence_repetition_detection_accepts_deque():
    history = deque(["read_file:aaa"] * 3, maxlen=12)

    assert Session._is_repeated_tool_sequence(history, repeat_threshold=3) is True


def test_tool_fingerprint_pattern_mode_includes_argument_fingerprint():
    fp = Session._tool_call_fingerprint("read_file", {"filename": "a.py"})
    pattern = Session._tool_call_fingerprint(
        "read_file", {"filename": "different.py"}, pattern_only=True
    )
    assert fp.startswith("read_file:")
    assert pattern.startswith("read_file~")


def test_bash_pattern_fingerprint_differs_on_string_content():
    # bash is NOT a pattern-sensitive tool, so different string content
    # should produce DIFFERENT pattern fingerprints — legitimate
    # sequential bash calls with different commands must not trip loop
    # detection.
    first = Session._tool_call_fingerprint(
        "bash", {"command": "ls -la"}, pattern_only=True
    )
    second = Session._tool_call_fingerprint(
        "bash", {"command": "cat README.md"}, pattern_only=True
    )
    assert first != second


def test_search_tool_pattern_fingerprint_collapses_string_content():
    # Pattern-sensitive tools (search_for_string, etc.) SHOULD collapse
    # string content so repeated searches with different queries collide
    # on the same pattern fingerprint — the hallmark of a search-loop.
    first = Session._tool_call_fingerprint(
        "search_for_string", {"string": "foo"}, pattern_only=True
    )
    second = Session._tool_call_fingerprint(
        "search_for_string", {"string": "bar"}, pattern_only=True
    )
    assert first == second


def test_bash_pattern_fingerprint_differs_on_arg_structure():
    # Pattern fingerprints SHOULD differ when the argument STRUCTURE differs
    # (e.g. different keys, different non-string values).
    first = Session._tool_call_fingerprint(
        "bash", {"command": "ls -la"}, pattern_only=True
    )
    second = Session._tool_call_fingerprint(
        "bash", {"command": "ls -la", "cwd": "/tmp"}, pattern_only=True
    )
    assert first != second


def test_feature_bookkeeping_tools_are_excluded_from_loop_tracking():
    assert Session._track_tool_for_loop_detection("update_task_status", {}) is False
    assert Session._track_tool_for_loop_detection("get_execution_state", {}) is False
    assert Session._track_tool_for_loop_detection("create_task", {}) is False
    assert Session._track_tool_for_loop_detection("create_phases", {}) is False
    assert Session._track_tool_for_loop_detection("search_for_string", {}) is True


def test_loop_detection_variables_exist():
    assert "loop_detection_enabled" in DEFAULT_VARIABLES
    assert "loop_detection_repeat_threshold" in DEFAULT_VARIABLES


def test_transient_provider_error_does_not_retry_http_400():
    assert Session._is_transient_provider_error(Exception("HTTP Error 400: Bad Request")) is False


def test_transient_provider_error_retries_429_and_503():
    assert Session._is_transient_provider_error(Exception("HTTP Error 429: Too Many Requests")) is True
    assert Session._is_transient_provider_error(Exception("status_code=503 upstream unavailable")) is True
