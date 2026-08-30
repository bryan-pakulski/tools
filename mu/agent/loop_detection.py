"""Loop-detection helpers for the agentic loop.

Three primitives:

  * `coarse_tool_args(tool_args, tool_name="")` — produce a stable,
    recursive, digest-friendly representation of tool arguments.
    For **pattern-sensitive** tools (search/query tools), strings
    collapse to a fixed ``"str:*"`` placeholder so calls with the same
    shape but different content collide. For all other tools, strings
    get a short SHA1 hash prefix so different content produces
    different fingerprints — legitimate sequential calls (e.g.
    ``bash`` with different commands) won't trip pattern detection.

  * `tool_call_fingerprint(name, args, pattern_only=False)` — combine
    `(name, args)` into a compact string token. With
    `pattern_only=True`, the args are first coarsened so the
    fingerprint groups *similar* calls (same shape, different content)
    together; without it, the fingerprint is exact.

  * `track_tool_for_loop_detection(name)` — boolean filter excluding
    bookkeeping tools that can legitimately repeat during a feature
    progression (`update_task_status`, `get_tasks`, ...).

  * `is_repeated_tool_sequence(history, threshold)` — true when the
    last `threshold` fingerprints in `history` are all identical and
    non-empty. The session's iteration loop calls this each turn to
    break out of stuck-in-a-rut patterns.

Tests: `tests/test_loop_detection.py`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List


# Tools whose repeated invocation during normal feature progression
# should NOT count toward loop detection (they're bookkeeping calls).
_BOOKKEEPING_TOOLS = frozenset(
    {
        # Feature-mode mutators/inspectors
        "create_feature",
        "create_phases",
        "create_task",
        "update_task",
        "update_phases",
        "review_task",
        "review_all_completed_tasks",
        "review_completed_tasks",
        "update_task_status",
        "get_execution_state",
        "get_tasks",
        "get_current_task",
        # Meta/tooling calls that legitimately repeat every iteration
        "flush",
        "todo_write",
        "todo_set_status",
        "todo_list",
        "save_scratchpad",
        "search_scratchpad",
        "list_scratchpad",
        "clear_scratchpad",
        "save_memory",
        "search_memory",
        "list_memory",
        # Read-only context tools that fire every iteration
        "get_workspace_details",
        "get_course_state",
        "get_security_state",
        "get_due_reviews",
        # Blocking sub-agent wait: the parent legitimately calls this
        # repeatedly (await -> timeout -> re-await) while a child runs.
        # Unlike poll_subagent (which stays fingerprinted so a busy-poll
        # loop is still flagged), this is the sanctioned blocking wait,
        # so it must not itself trip the repeated-sequence detector.
        "await_subagent",
    }
)

# Tools where the model may legitimately try many different string
# queries/paths before finding what it needs. For these, string content
# is collapsed so repeated calls with different queries collide on the
# same pattern fingerprint (the hallmark of a search-loop).
_PATTERN_SENSITIVE_TOOLS = frozenset(
    {
        "search_for_string",
        "search_references",
        "retrieve_relevant_context",
        "web_search",
        "arxiv_search",
        "doi_resolve",
        "reddit_search",
        "stackoverflow_search",
        "hackernews_search",
        "search_memory",
        "search_scratchpad",
    }
)

# Pattern-sensitive tools whose string args are FILE PATHS rather than
# free-form queries. For these, distinct paths collapse to a ``"path:*"``
# placeholder so a multi-file read cycle (read_file f1 → f2 → f3 → f4 →
# f1 → …) collapses to one repeated fingerprint in the pattern lane and
# becomes detectable. Without this, non-pattern-sensitive read tools
# hash each filename to a distinct fingerprint and read-loops evade
# detection entirely (FM-6). The exact-fingerprint lane is unaffected,
# so legitimate distinct reads still fingerprint precisely there.
# See R7 in documentation/harness-investigation.md.
_PATH_SENSITIVE_TOOLS = frozenset(
    {
        "read_file",
        "get_chunk",
        "list_dir",
        "search_references",
    }
)


def _collapse_string(val: str, tool_name: str) -> str:
    """Collapse string for pattern fingerprint. For pattern-sensitive
    tools (search/query), all strings become ``"str:*"`` so repeated
    searches with different queries collide. For path-sensitive read
    tools (read_file/get_chunk/list_dir/search_references), all path
    strings become ``"path:*"`` so a multi-file read cycle collapses to
    one repeated fingerprint. For all other tools, strings get a short
    SHA1 hash prefix so different content produces different fingerprints
    — legitimate sequential calls (e.g. ``bash`` with different commands)
    won't trip pattern detection."""
    if tool_name in _PATH_SENSITIVE_TOOLS:
        return "path:*"
    if tool_name in _PATTERN_SENSITIVE_TOOLS:
        return "str:*"
    return "str:" + hashlib.sha1(val.encode("utf-8")).hexdigest()[:8]


def coarse_tool_args(tool_args: Any, tool_name: str = "") -> Any:
    """Build a stable, coarse-grained representation of tool args for
    loop pattern checks.

    For **pattern-sensitive** tools (search/query tools), string values
    collapse to a fixed ``"str:*"`` placeholder so that calls with the
    same tool and same argument *shape* but different string *content*
    produce the same fingerprint. This is what allows pattern-based
    loop detection to fire when the model keeps calling e.g.
    ``search_for_string`` with different query strings.

    For all other tools, strings get a short SHA1 hash prefix so
    different content produces different fingerprints — legitimate
    sequential calls (e.g. ``bash`` with different commands) won't
    trip pattern detection.

    ints/floats/bools/None pass through; nested dicts/lists recurse;
    unknown types collapse to their type name.

    The output is JSON-serializable, suitable for feeding into
    `tool_call_fingerprint(..., pattern_only=True)`.
    """
    if isinstance(tool_args, dict):
        coarse = {}
        for key in sorted(tool_args.keys()):
            val = tool_args.get(key)
            if isinstance(val, str):
                coarse[key] = _collapse_string(val, tool_name)
            elif isinstance(val, (int, float, bool)) or val is None:
                coarse[key] = val
            elif isinstance(val, list):
                coarse[key] = [
                    coarse_tool_args(item, tool_name) for item in val[:8]
                ]
            elif isinstance(val, dict):
                coarse[key] = coarse_tool_args(val, tool_name)
            else:
                coarse[key] = type(val).__name__
        return coarse
    if isinstance(tool_args, list):
        return [coarse_tool_args(item, tool_name) for item in tool_args[:8]]
    if isinstance(tool_args, str):
        return _collapse_string(tool_args, tool_name)
    if isinstance(tool_args, (int, float, bool)) or tool_args is None:
        return tool_args
    return type(tool_args).__name__


def tool_call_fingerprint(
    tool_name: str, tool_args: Any, *, pattern_only: bool = False
) -> str:
    """Compact fingerprint of a `(name, args)` tool call. The exact
    variant uses the raw args; the `pattern_only` variant first
    coarsens via `coarse_tool_args` so that two calls with different
    string content but the same shape collide on the same fingerprint."""
    name = str(tool_name or "").strip().lower() or "tool"
    payload_source = (
        coarse_tool_args(tool_args or {}, tool_name=name)
        if pattern_only
        else (tool_args or {})
    )
    try:
        payload = json.dumps(
            payload_source,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        payload = str(payload_source)
    digest = hashlib.sha1(f"{name}|{payload}".encode("utf-8")).hexdigest()[:12]
    return f"{name}:{digest}" if not pattern_only else f"{name}~{digest}"


def track_tool_for_loop_detection(tool_name: str, tool_args: Any = None) -> bool:
    """Return False for bookkeeping tools (feature-mode mutators, task
    inspectors) that legitimately repeat across iterations. The agent
    loop uses this to filter `tool_args` before adding to the
    fingerprint history."""
    name = str(tool_name or "").strip().lower()
    return name not in _BOOKKEEPING_TOOLS


def is_repeated_tool_sequence(
    sequence_history: List[str], repeat_threshold: int = 3
) -> bool:
    """True when the last `repeat_threshold` entries of
    `sequence_history` are all identical and non-empty — i.e. the
    agent has been firing the same tool call repeatedly and is
    probably stuck."""
    if len(sequence_history) < repeat_threshold:
        return False
    # Round-50 F8-hotfix: callers now pass bounded deques (r46 F8) —
    # deque supports indexing but NOT slicing, so materialize once (the
    # history is capped, this is cheap). Same fix is_periodic_sequence
    # already got.
    sequence_history = list(sequence_history)
    tail = sequence_history[-repeat_threshold:]
    if not all(tail):
        return False
    return len(set(tail)) == 1


def is_periodic_sequence(
    sequence_history: List[str],
    *,
    max_period: int = 6,
    min_repeats: int = 2,
) -> bool:
    """Detect a *periodic* (non-consecutive) repeat in the tail of
    `sequence_history` — any repeating sub-sequence of period
    ``2..max_period`` repeated at least ``min_repeats`` times.

    This complements `is_repeated_tool_sequence` (which only catches
    period-1 / consecutive-identical repeats). It catches read-loops
    where the agent alternates between two or three distinct iteration
    shapes, e.g. ``[A, B, A, B, A, B]`` (period 2) or
    ``[A, B, C, A, B, C]`` (period 3) — patterns that never produce
    ``repeat_threshold`` consecutive identical entries but are still
    stuck cycles (FM-6, R7).

    Period 1 is deliberately excluded: the consecutive detector owns
    that case with its own (higher) threshold, so this function won't
    fire on a 2-of-a-kind that the consecutive detector would reject.
    """
    n = len(sequence_history)
    if n < 2 * 2:  # need at least period-2 × min_repeats=2 = 4 entries
        return False
    # Round-46 F8: callers now pass bounded deques — slicing is not supported
    # on deque, so materialize once (the history is capped, this is cheap).
    sequence_history = list(sequence_history)
    for period in range(2, max_period + 1):
        if n < period * min_repeats:
            continue
        # Candidate repeating block = the last `period` entries.
        block = sequence_history[n - period : n]
        if not all(block):
            continue
        # Verify the preceding (min_repeats - 1) blocks match.
        ok = True
        for r in range(1, min_repeats):
            seg = sequence_history[n - period * (r + 1) : n - period * r]
            if seg != block:
                ok = False
                break
        if ok:
            return True
    return False


# Read-only tools whose arguments identify an on-disk source the agent is
# (re-)covering. Re-coverage of these is the long-horizon stall signature:
# the agent keeps re-reading files it already gathered instead of acting.
_RECOVERAGE_READ_TOOLS = frozenset({
    "read_file",
    "get_chunk",
    "list_dir",
    "search_for_string",
    "search_references",
    "retrieve_relevant_context",
    "get_workspace_details",
})


def extract_read_paths(tool_calls: Any) -> set:
    """Return the normalized set of source paths an iteration's read-only
    tool calls refer to (Fix #12).

    Used by the context-gathering stall detector: if the agent re-covers a
    path it already read earlier in the turn, that's a re-coverage event.
    Paths come from the ``path`` (or ``pattern``+``path``) arg; args without
    a path contribute nothing. Returns an empty set when there are no
    read-only calls or no path-bearing args.
    """
    paths: set = set()
    if not tool_calls:
        return paths
    for call in tool_calls:
        try:
            name = str(getattr(call, "tool_name", "") or "").strip().lower()
        except Exception:
            continue
        if name not in _RECOVERAGE_READ_TOOLS:
            continue
        try:
            args = getattr(call, "tool_args", None)
        except Exception:
            args = None
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if isinstance(path, str) and path:
            paths.add(path)
    return paths


def is_concrete_change_iter(tool_calls: Any) -> bool:
    """True if an iteration made (or would make) a concrete change, not just
    gathered context (Fix #12). Any tool that isn't a read-only /
    bookkeeping call counts: writes, bash, spawn_agent, apply_diff, etc.
    """
    if not tool_calls:
        return False
    for call in tool_calls:
        try:
            name = str(getattr(call, "tool_name", "") or "").strip().lower()
        except Exception:
            continue
        if not name:
            continue
        if name in _RECOVERAGE_READ_TOOLS or name in _BOOKKEEPING_TOOLS:
            continue
        return True
    return False


__all__ = [
    "coarse_tool_args",
    "tool_call_fingerprint",
    "track_tool_for_loop_detection",
    "is_repeated_tool_sequence",
    "is_periodic_sequence",
    "extract_read_paths",
    "is_concrete_change_iter",
]
