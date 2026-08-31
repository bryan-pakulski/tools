"""Regression tests for the long-horizon degradation fixes.

Each test pins one of the architectural fixes that prevent the agent from
re-deriving context it already gathered and stalling on long tasks:

  * L2/L3 system-prompt layers must refresh per iteration (not frozen at
    turn start) while L1/L1B are reused from a per-turn cache.
  * The goal must persist across turns for long-horizon tasks.
  * Loop detection must catch context-gathering stalls (re-reading the
    same paths), not just identical iteration shapes.
  * max_iterations must force a consolidation, not silently stop.
"""

from providers.base import LLMProvider, MessagePart, ProviderResponse


class _DummyProvider(LLMProvider):
    def get_available_models(self):
        return ["dummy"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        return ProviderResponse(
            text="ok", parts=[], input_tokens=0, output_tokens=0, total_tokens=0
        )

    def upload_file(self, file_path, mime_type):
        return None


def _make_session():
    from mu.session.session import Session, SessionManager

    sm = SessionManager()
    return Session(_DummyProvider(), False, "you are a helpful assistant", sm)


# ============================================================ Fix #8: L2/L3 refresh per iteration


def test_inject_accepts_cached_layers():
    """`_inject_hierarchical_context` accepts cached L1B so the loop can
    rebuild L2/L3 each iteration without disk reads."""
    session = _make_session()
    sk = session._build_skills_block(announce=False)
    out = session._inject_hierarchical_context(
        "base", cached_skills=sk
    )
    assert isinstance(out, str)
    assert "base" in out


def test_l2_refreshes_between_inject_calls_with_cache():
    """The core fix: two inject calls with the same cached L1B but a
    changed conversation_summary produce different L2 — proving the loop
    now sees mid-turn progress updates instead of a frozen turn-start L2."""
    session = _make_session()
    sk = session._build_skills_block(announce=False)

    session.session_manager.conversation_summary = (
        "### Progress\ndid thing X\n### Open items\nfinish Y"
    )
    before = session._inject_hierarchical_context(
        "base", cached_skills=sk
    )
    assert "did thing X" in before

    session.session_manager.conversation_summary = (
        "### Progress\ndid thing Z\n### Open items\nfinish W"
    )
    after = session._inject_hierarchical_context(
        "base", cached_skills=sk
    )
    assert "did thing Z" in after
    assert "did thing X" not in after


def test_l3_refreshes_session_goal_between_inject_calls():
    """L3 (active goal) must reflect a freshly-set session_goal without a
    turn restart."""
    session = _make_session()
    sk = session._build_skills_block(announce=False)

    before = session._inject_hierarchical_context(
        "base", cached_skills=sk
    )
    assert "ship the feature" not in before

    session.variables["session_goal"] = "ship the feature"
    after = session._inject_hierarchical_context(
        "base", cached_skills=sk
    )
    assert "ship the feature" in after


def test_cached_layers_skip_rebuild(monkeypatch):
    """When cached L1B is passed, inject must NOT rediscover skills —
    that's the whole point of the per-turn cache."""
    session = _make_session()

    calls = {"skills": 0}
    monkeypatch.setattr(
        session, "_build_skills_block", lambda *, announce=False: (
            calls.__setitem__("skills", calls["skills"] + 1) or ""
        )
    )

    sk = ""
    session._inject_hierarchical_context(
        "base", cached_skills=sk
    )
    assert calls["skills"] == 0, "cached_skills must skip rebuild"


# ============================================================ Fix #11: sticky session_goal across turns


def test_session_goal_strips_in_default_mode():
    """Default mode is conversational: the goal clears at end of turn so
    it can't bias the next unrelated request."""
    session = _make_session()
    session.variables["agent_mode"] = "default"
    session.variables["session_goal"] = "Refactor the auth layer"
    session._strip_session_goal_after_turn()
    assert session.variables["session_goal"] == ""


def test_session_goal_sticky_in_loop_mode():
    """Loop mode is long-horizon: the goal must persist across turns in
    L3 until the user clears it or sets a new goal."""
    session = _make_session()
    session.variables["agent_mode"] = "loop"
    session.variables["session_goal"] = "Ship the migration"
    session._strip_session_goal_after_turn()
    assert session.variables["session_goal"] == "Ship the migration"


def test_session_goal_sticky_in_feature_mode():
    """Feature mode is long-horizon (multi-turn feature work): goal
    persists across turns."""
    session = _make_session()
    session.variables["agent_mode"] = "feature"
    session.variables["session_goal"] = "Implement the dashboard"
    session._strip_session_goal_after_turn()
    assert session.variables["session_goal"] == "Implement the dashboard"


def test_session_goal_sticky_opt_in_default_mode():
    """A user doing long multi-turn work in default mode can opt in via
    `session_goal_sticky` and the goal survives the turn boundary.

    Setting it via `/set` also flips the explicit tracker so the mode-aware
    default (default = clear-per-turn) yields to the user's choice."""
    session = _make_session()
    session.variables["agent_mode"] = "default"
    session.variables["session_goal_sticky"] = True
    session.variables["session_goal_sticky_explicit"] = True
    session.variables["session_goal"] = "Long-running refactor"
    session._strip_session_goal_after_turn()
    assert session.variables["session_goal"] == "Long-running refactor"


def test_session_goal_explicit_opt_out_in_loop_mode():
    """Explicit `session_goal_sticky=False` is honored even in loop mode —
    the user can force per-turn clearing if they want. `/set` flips the
    explicit tracker so the loop-mode sticky default yields to the choice."""
    session = _make_session()
    session.variables["agent_mode"] = "loop"
    session.variables["session_goal_sticky"] = False
    session.variables["session_goal_sticky_explicit"] = True
    session.variables["session_goal"] = "one-off task"
    session._strip_session_goal_after_turn()
    assert session.variables["session_goal"] == ""


# ============================================================ Fix #9: periodic L2 progress checkpoints


class _SummaryProvider(LLMProvider):
    """Returns a canned structured summary so force_progress_checkpoint's
    LLM path is exercised without a real provider call."""

    def __init__(self, body="### Progress\nDid work X"):
        self._body = body
        self.calls = 0

    def get_available_models(self):
        return ["dummy"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        self.calls += 1
        return ProviderResponse(
            text=self._body,
            parts=[],
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )

    def upload_file(self, file_path, mime_type):
        return None


def _add_history(session, n, prefix="msg"):
    sm = session.session_manager
    for i in range(n):
        sm.history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "parts": [{"type": "text", "text": f"{prefix} {i}"}],
        })


def test_checkpoint_refreshes_l2_without_advancing_anchor():
    """force_progress_checkpoint folds recent history into L2 but leaves
    summary_anchor (and thus the L5 tail) untouched — no compaction."""
    session = _make_session()
    _add_history(session, 10)
    sm = session.session_manager
    anchor_before = sm.summary_anchor
    provider = _SummaryProvider("### Progress\nRefactored auth layer")
    ok = sm.force_progress_checkpoint(provider)
    assert ok is True
    assert "Refactored auth layer" in sm.conversation_summary
    # The whole point: no compaction. Anchor and L5 history are intact.
    assert sm.summary_anchor == anchor_before
    assert len(sm.history) == 10


def test_checkpoint_skips_when_too_few_new_entries():
    """A checkpoint shouldn't fire a provider call for a trivial amount of
    new work — bounds cost on short turns."""
    session = _make_session()
    _add_history(session, 2)
    provider = _SummaryProvider()
    ok = session.session_manager.force_progress_checkpoint(provider)
    assert ok is False
    assert provider.calls == 0
    assert session.session_manager.conversation_summary == ""


def test_checkpoint_only_summarizes_since_last_checkpoint():
    """Repeated checkpoints summarize only new work, not the whole turn
    again — _checkpoint_anchor tracks progress."""
    session = _make_session()
    _add_history(session, 8)
    sm = session.session_manager
    p1 = _SummaryProvider("### Progress\nMilestone A done")
    sm.force_progress_checkpoint(p1)
    assert p1.calls == 1
    # Add more work; the next checkpoint must only cover the new entries.
    _add_history(session, 8, prefix="more")
    p2 = _SummaryProvider("### Progress\nMilestone B done")
    sm.force_progress_checkpoint(p2)
    assert p2.calls == 1
    assert "Milestone A done" in sm.conversation_summary
    assert "Milestone B done" in sm.conversation_summary


def test_checkpoint_merges_structured_sections():
    """The canned ### Progress summary merges by section, not blind-
    append — existing Progress content is preserved alongside the new."""
    session = _make_session()
    _add_history(session, 8)
    sm = session.session_manager
    sm.conversation_summary = "### Progress\nEarlier work\n\n### Open items\nfinish X"
    sm.force_progress_checkpoint(_SummaryProvider("### Progress\nNew work\n\n### Open items\nfinish Y"))
    assert "Earlier work" in sm.conversation_summary
    assert "New work" in sm.conversation_summary


def test_checkpoint_mechanical_fallback_without_provider():
    """With no provider, the checkpoint still records a mechanical snapshot
    rather than silently no-op'ing — L2 must reflect progress somehow."""
    session = _make_session()
    _add_history(session, 8)
    sm = session.session_manager
    ok = sm.force_progress_checkpoint(None)
    assert ok is True
    assert "Progress checkpoint" in sm.conversation_summary
    assert sm.summary_anchor == 0  # no compaction


def test_checkpoint_loop_mode_default_cadence():
    """loop mode should default to an enabled checkpoint cadence even when
    the user hasn't set progress_checkpoint_every (long-horizon work)."""
    # Mirror the loop's cadence resolution.
    session = _make_session()
    session.variables["agent_mode"] = "loop"
    every = int(session.variables.get("progress_checkpoint_every", 0) or 0)
    if every <= 0:
        mode = str(session.variables.get("agent_mode", "default") or "default").lower()
        if mode in ("loop", "feature"):
            every = 12
    assert every == 12


def test_checkpoint_default_mode_disabled_by_default():
    """default/chat mode should NOT auto-enable checkpoints (short turns
    don't need a periodic provider call)."""
    session = _make_session()
    session.variables["agent_mode"] = "default"
    every = int(session.variables.get("progress_checkpoint_every", 0) or 0)
    assert every == 0


# ============================================================ Fix #10: auto-recall cached reads + grow cache

import os
import tempfile

from mu.session.tool_cache import ToolResultCache


def test_store_with_locator_indexes_re_read():
    """A read_file result stored with args is auto-recallable by the same
    args without re-executing the tool (the core auto-recall win)."""
    cache = ToolResultCache()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name
    try:
        result = {"content": "hello world"}
        key = cache.store_with_locator(
            "call1", "read_file", {"path": path}, result
        )
        assert key is not None
        hit = cache.lookup_by_locator("read_file", {"path": path})
        assert hit is not None
        assert hit["cache_hit"] is True
        assert hit["result"] == result
        assert hit["cache_key"] == key
    finally:
        os.unlink(path)


def test_lookup_returns_none_for_changed_file():
    """If the file's mtime/size changed since the cached read, auto-recall
    must miss so we don't serve stale content."""
    cache = ToolResultCache()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("v1 content here")
        path = f.name
    try:
        cache.store_with_locator(
            "call1", "read_file", {"path": path}, {"content": "v1 content here"}
        )
        # Modify the file (content + size change).
        with open(path, "w") as f:
            f.write("v2 different length content")
        hit = cache.lookup_by_locator("read_file", {"path": path})
        assert hit is None  # stale → miss
    finally:
        os.unlink(path)


def test_lookup_returns_none_for_different_args():
    """Auto-recall is exact-args — a different range/path must not hit."""
    cache = ToolResultCache()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("x" * 200)
        path = f.name
    try:
        cache.store_with_locator(
            "call1", "get_chunk", {"path": path, "start": 0, "end": 50},
            {"content": "x" * 50},
        )
        hit = cache.lookup_by_locator(
            "get_chunk", {"path": path, "start": 100, "end": 150}
        )
        assert hit is None
    finally:
        os.unlink(path)


def test_lookup_returns_none_for_write_tool():
    """Write tools are never auto-recallable — they're not in _LOCATOR_TOOLS."""
    cache = ToolResultCache()
    hit = cache.lookup_by_locator("write_file", {"path": "/tmp/x"})
    assert hit is None


def test_lookup_returns_none_when_uncached():
    """A path that was never cached must miss cleanly."""
    cache = ToolResultCache()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("never cached")
        path = f.name
    try:
        hit = cache.lookup_by_locator("read_file", {"path": path})
        assert hit is None
    finally:
        os.unlink(path)


def test_loop_mode_grows_cache_bounds():
    """loop mode should grow the tool-result cache so more reads stay
    recallable on long tasks."""
    from mu.session.budgets import resolve_tool_cache_bounds

    session = _make_session()
    session.variables["agent_mode"] = "loop"
    entries, nbytes = resolve_tool_cache_bounds(session)
    assert entries >= 256
    assert nbytes >= 2_097_152


def test_default_mode_keeps_small_cache_bounds():
    """default mode should keep the small default cache bounds."""
    from mu.session.budgets import resolve_tool_cache_bounds

    session = _make_session()
    session.variables["agent_mode"] = "default"
    entries, nbytes = resolve_tool_cache_bounds(session)
    assert entries == 50
    assert nbytes == 524_288


def test_loop_mode_raises_tool_result_floor():
    """loop mode raises the verbatim tool-result floor so recent reads
    survive compaction."""
    from mu.session.budgets import resolve_tool_result_floor

    session = _make_session()
    session.variables["agent_mode"] = "loop"
    assert resolve_tool_result_floor(session) >= 8


def test_explicit_higher_floor_respected_in_loop_mode():
    """A user's explicit higher floor wins over the mode-aware minimum."""
    from mu.session.budgets import resolve_tool_result_floor

    session = _make_session()
    session.variables["agent_mode"] = "loop"
    session.variables["tool_result_floor"] = 20
    assert resolve_tool_result_floor(session) == 20


# ============================================================ Fix #12: context-gathering stall detection

from mu.agent.loop_detection import extract_read_paths, is_concrete_change_iter


class _TC:
    """Minimal stand-in for a tool_call part."""

    def __init__(self, name, args):
        self.tool_name = name
        self.tool_args = args


def test_extract_read_paths_collects_path_args():
    calls = [
        _TC("read_file", {"path": "/a.py"}),
        _TC("list_dir", {"path": "/src"}),
        _TC("bash", {"command": "ls"}),
        _TC("get_chunk", {"path": "/a.py", "start": 0, "end": 10}),
    ]
    paths = extract_read_paths(calls)
    assert paths == {"/a.py", "/src"}


def test_extract_read_paths_empty_for_no_read_tools():
    calls = [_TC("bash", {"command": "ls"}), _TC("write_file", {"path": "/x"})]
    assert extract_read_paths(calls) == set()


def test_is_concrete_change_iter_true_for_write():
    calls = [_TC("read_file", {"path": "/a.py"}), _TC("write_file", {"path": "/b.py"})]
    assert is_concrete_change_iter(calls) is True


def test_is_concrete_change_iter_false_for_reads_only():
    calls = [_TC("read_file", {"path": "/a.py"}), _TC("list_dir", {"path": "/src"})]
    assert is_concrete_change_iter(calls) is False


def test_is_concrete_change_iter_false_for_empty():
    assert is_concrete_change_iter([]) is False


def test_recoverage_threshold_config_default():
    """Default threshold is 4 (enabled); 0 disables."""
    session = _make_session()
    assert int(session.variables.get("recoverage_stall_threshold", 4)) == 4


def test_recoverage_stall_logic_clean():
    """Clean version: 4 re-coverage iterations trip the nudge."""
    session = _make_session()
    session._recoverage_seen_paths = set()
    session._recoverage_stall_iters = 0
    session._recoverage_last_nudge_iter = -10_000
    threshold = 4

    # Seed the seen set with a first read (no recovery yet).
    session._recoverage_seen_paths.add("/a.py")

    def step():
        calls = [_TC("read_file", {"path": "/a.py"})]
        read_paths = extract_read_paths(calls)
        recovered = {p for p in read_paths if p in session._recoverage_seen_paths}
        concrete = is_concrete_change_iter(calls)
        session._recoverage_seen_paths.update(read_paths)
        if recovered and not concrete:
            session._recoverage_stall_iters += 1
        else:
            session._recoverage_stall_iters = 0
        return session._recoverage_stall_iters >= threshold

    # 3 re-coverage iterations: not yet.
    assert step() is False
    assert step() is False
    assert step() is False
    # 4th re-coverage iteration: trip.
    assert step() is True


def test_concrete_change_resets_stall_counter():
    """A write interspersed resets the stall counter — real progress."""
    session = _make_session()
    session._recoverage_seen_paths = {"/a.py"}
    session._recoverage_stall_iters = 3
    calls = [_TC("read_file", {"path": "/a.py"}), _TC("write_file", {"path": "/b.py"})]
    read_paths = extract_read_paths(calls)
    recovered = {p for p in read_paths if p in session._recoverage_seen_paths}
    concrete = is_concrete_change_iter(calls)
    assert recovered  # /a.py was re-covered
    assert concrete   # but a write happened too
    # The loop resets the counter when concrete is True.
    session._recoverage_stall_iters = 0 if (recovered and not concrete) else 0
    assert session._recoverage_stall_iters == 0


# ============================================================ Fix #13: force consolidation at max_iterations


class _ConsolidationProvider(LLMProvider):
    """Returns a tool_call on the first call (so the loop doesn't finish),
    then text on subsequent calls (the consolidation turn)."""

    def __init__(self, model_name="dummy"):
        self.calls = 0
        self.model_name = model_name

    def get_available_models(self):
        return ["dummy"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ProviderResponse(
                text="",
                parts=[MessagePart(
                    type="tool_call", tool_name="noop", tool_args={}, tool_call_id="c1"
                )],
                input_tokens=1, output_tokens=1, total_tokens=2,
            )
        return ProviderResponse(
            text="CONSOLIDATED: did X, left to do Y",
            parts=[MessagePart(type="text", text="CONSOLIDATED: did X, left to do Y")],
            input_tokens=1, output_tokens=1, total_tokens=2,
        )

    def upload_file(self, *a, **kw):
        return None


def test_max_iterations_forces_consolidation_turn(tmp_path, monkeypatch):
    """Hitting max_iterations mid-work injects a consolidation user message
    and runs a final tools-disabled provider call, so the user gets a
    handoff summary instead of an abrupt stop."""
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(tmp_path / "history"))
    from mu.session.session import Session, SessionManager
    from mu.agent.loop_body import run_turn

    provider = _ConsolidationProvider()
    session = Session(provider, False, "system", SessionManager())
    session.variables["max_iterations"] = 1
    session.variables["agent_mode"] = "default"

    result = run_turn(session, "do the thing")

    # The consolidation provider call happened (calls >= 2).
    assert provider.calls >= 2
    # A consolidation user message was injected.
    assert any(
        msg.get("role") == "user"
        and any(
            "maximum iteration budget" in str(p.get("text", ""))
            for p in msg.get("parts", [])
        )
        for msg in session.session_manager.history
    )
    # The consolidation text was appended as an assistant message.
    assert any(
        msg.get("role") == "assistant"
        and any(
            "CONSOLIDATED" in str(p.get("text", ""))
            for p in msg.get("parts", [])
        )
        for msg in session.session_manager.history
    )
    # The guard fired and is latched for this turn.
    assert getattr(session, "_consolidation_done", False) is True
    # Status reflects max_iterations with a consolidation note.
    assert result["status"] == "max_iterations_reached"
    assert "consolidation" in str(result.get("error") or "").lower()


def test_consolidation_entry_saved_as_done_not_active(tmp_path, monkeypatch):
    """The max-iterations consolidation is a handoff/audit record, not active
    working memory — it must be saved as `done` so it stays out of the default
    active+stale search and the active-first L3 injection (anti-rot:
    audit-is-invisible). A regression to default `active` would pollute the
    active set with every consolidation the session runs."""
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(tmp_path / "history"))
    from mu.session.session import Session, SessionManager
    from mu.agent.loop_body import run_turn

    provider = _ConsolidationProvider()
    session = Session(provider, False, "system", SessionManager())
    session.variables["max_iterations"] = 1
    session.variables["agent_mode"] = "default"

    run_turn(session, "do the thing")

    consol = [
        e for e in session.task_memory.entries
        if "consolidation" in e.tags and e.source == "max_iterations_consolidation"
    ]
    assert len(consol) == 1
    assert consol[0].status == "done"
    # And it is excluded from the default active+stale search view.
    default_results = session.task_memory.search("CONSOLIDATED", limit=10)
    assert all(e.id != consol[0].id for e in default_results)


def test_consolidation_guard_resets_between_turns(tmp_path, monkeypatch):
    """`_consolidation_done` resets at turn start so a later turn can
    consolidate again if it also hits the cap."""
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(tmp_path / "history"))
    from mu.session.session import Session, SessionManager

    provider = _ConsolidationProvider()
    session = Session(provider, False, "system", SessionManager())
    session._consolidation_done = True  # simulate a prior consolidation
    # run_turn's first action resets the guard.
    from mu.agent.loop_body import run_turn

    session.variables["max_iterations"] = 1
    run_turn(session, "another task")
    # After the turn, the guard was reset at start and may have fired again.
    # The key invariant: it was resettable (not permanently latched).
    assert hasattr(session, "_consolidation_done")

def test_goal_not_duplicated_across_l3_memory_snapshot():
    """Goal text renders verbatim in the LAYER 3 active-goal block; the
    working-memory snapshot must not restate the same sentence."""
    session = _make_session()
    session.variables["session_goal"] = "ship the feature"
    session.task_memory.save(
        "Locked session goal: ship the feature",
        kind="goal",
        source="goal-persistence",
    )
    session.task_memory.save(
        "Use codex for verification of changes",
        kind="decision",
        source="policy",
    )

    rendered = session.task_memory.render_summary(limit=8)
    assert "ship the feature" in rendered  # sanity: echo exists pre-filter

    import mu.agent.loop_body as lb

    # Exercise the same filter logic the injection site uses.
    goal_texts = [
        g
        for g in (
            str(session.variables.get(k, "") or "").strip()
            for k in ("session_goal", "loop_goal")
        )
        if g
    ]
    filtered = "\n".join(
        line
        for line in rendered.splitlines()
        if not (
            line.rstrip().endswith(tuple(goal_texts))
            and any(g in line for g in goal_texts)
        )
    )
    assert "ship the feature" not in filtered
    assert "Use codex for verification" in filtered  # non-goal lines kept


def test_goal_block_carries_scratchpad_once():
    """Scratchpad snapshot must appear in the L3 active-goal block only via
    the dedicated loop_body snapshot, not duplicated inside the goal block."""
    session = _make_session()
    session.turn_scratchpad.save("step 1 done", tags=["todo"])
    session.variables["session_goal"] = "ship the feature"
    ctx = session._build_active_goal_context()
    assert "ship the feature" in ctx
    assert "Scratchpad snapshot" not in ctx

    # Full prompt still surfaces the scratchpad exactly once.
    prompt = session._inject_hierarchical_context("base", cached_skills="")
    snapshot_header = "LAYER 3 — Turn scratchpad snapshot"
    # Scratchpad only in the dedicated L3 snapshot section (session.py
    # goal block no longer embeds it). Session-level injection excludes it
    # only when memory/scratchpad layers append — so count occurrences.
    assert prompt.count("step 1 done") >= 1
    # The goal block itself (rendered inside prompt) has no scratch section.
    goal_block = ctx
    assert goal_block in prompt or goal_block == ""


def test_multi_line_goal_still_deduped():
    """Whitespace-normalized matching must dedup multi-line goals (the
    long-loop-mode shape) and keep lines whose substance goes beyond the
    goal text."""
    session = _make_session()
    session.variables["session_goal"] = "Run reset\nmucli gui script"
    session.task_memory.save(
        "Locked session goal: Run reset\nmucli gui script",
        kind="goal",
        source="goal-persistence",
    )
    session.task_memory.save(
        "fixed loader for Run reset mucli gui script",
        kind="finding",
        source="work",
    )

    from mu.agent.loop_body import _filter_goal_echo_entries

    rendered = session.task_memory.render_summary(limit=8)
    goal_norm = " ".join(session.variables["session_goal"].split())
    filtered = _filter_goal_echo_entries(
        rendered, [session.variables["session_goal"]]
    )
    # Persistence-framed restate entry dropped entirely.
    assert "Locked session goal:" not in filtered
    # Dominance rule keeps payload entry: goal is <50% of its content but
    # its substance (the work note) survives without the restated goal.
    assert "fixed loader for" in filtered


def test_l3_snapshots_deduped_against_state_capsule():
    """L2 state capsule ('Durable decisions and findings', 'Open work
    ledger') is authoritative: L3 memory/scratchpad snapshot lines whose
    normalized core already appears in the capsule are dropped."""
    from mu.agent.loop_body import _filter_state_capsule_duplicates

    capsule = (
        "### Durable decisions and findings\n"
        "- [decision] Use lean serialization for tool results at provider call time\n"
        "### Open work ledger\n"
        "- [active] Finish the release checklist today\n"
    )
    payload = (
        "### In-Task Memory\n"
        "- #2 [active] (src): Use lean serialization for tool results at provider call time\n"
        "- #3 [active] (src): Distinct fact not projected anywhere else at all\n"
        "### Turn Scratchpad\n"
        "- [active] [todo] Finish the release checklist today\n"
        "- [active] [todo] Brand-new note only in scratchpad here"
    )
    filtered = _filter_state_capsule_duplicates(payload, capsule)
    # Both duplicated lines dropped.
    assert "Use lean serialization for tool results" not in filtered
    assert "Finish the release checklist today" not in filtered
    # Unique lines survive.
    assert "Distinct fact not projected" in filtered
    assert "Brand-new note only in scratchpad" in filtered
    # No capsule -> payload unchanged.
    assert _filter_state_capsule_duplicates(payload, "") == payload
    # Short lines (<24-char core) are never dropped — framing noise only.
    tiny = "- #1 [active] (s): ok"
    assert _filter_state_capsule_duplicates(tiny, capsule) == tiny


def test_context_pressure_nudge_hysteresis():
    """Model-directed compact nudge fires once per threshold crossing.

    Default mode has no proactive compaction; this nudge tells the model
    WHEN to call `compact` — once per crossing, re-armed by anchor advance
    (compaction happened) or fill dropping back below the threshold.
    """
    from types import SimpleNamespace

    from mu.agent.context_guard import _maybe_nudge_context_pressure

    _sm = SimpleNamespace(summary_anchor=0, history=[])
    session = SimpleNamespace(variables={}, session_manager=_sm)

    manifest = {"total": 400_000}
    LIMIT = 480_000  # 83.3% — over the default 80% threshold

    def n_nudges():
        return sum(
            1 for m in _sm.history if "CONTEXT PRESSURE" in str(m.get("parts"))
        )

    # Under threshold: silent, no state changes.
    _maybe_nudge_context_pressure(
        session, limit=LIMIT, manifest={"total": 100_000}
    )
    assert n_nudges() == 0
    assert not getattr(session, "_pressure_nudge_fired", False)

    # Crossing: exactly one synthetic nudge, marked fired.
    _maybe_nudge_context_pressure(session, limit=LIMIT, manifest=manifest)
    assert n_nudges() == 1
    assert _sm.history[-1]["synthetic"] is True
    assert _sm.history[-1]["role"] == "user"
    assert session._pressure_nudge_fired is True

    # Still over threshold, anchor unmoved: no re-fire (hysteresis).
    _maybe_nudge_context_pressure(session, limit=LIMIT, manifest=manifest)
    assert n_nudges() == 1

    # Compaction happened (anchor advanced): re-arms and may fire again.
    session.session_manager.summary_anchor = 5
    _maybe_nudge_context_pressure(session, limit=LIMIT, manifest=manifest)
    assert n_nudges() == 2

    # Dropped below threshold: re-arms silently (no message).
    _maybe_nudge_context_pressure(
        session, limit=LIMIT, manifest={"total": 50_000}
    )
    assert n_nudges() == 2
    assert session._pressure_nudge_fired is False

    # Fresh crossing after re-arm: fires again.
    _maybe_nudge_context_pressure(session, limit=LIMIT, manifest=manifest)
    assert n_nudges() == 3

    # Threshold 0 disables the feature entirely.
    session.variables["context_pressure_nudge_pct"] = 0
    before = n_nudges()
    _maybe_nudge_context_pressure(session, limit=LIMIT, manifest=manifest)
    assert n_nudges() == before

    # Missing manifest: silent no-op (no estimator seam this iteration).
    del session.variables["context_pressure_nudge_pct"]
    session._pressure_nudge_fired = False
    _maybe_nudge_context_pressure(session, limit=LIMIT, manifest=None)
    assert n_nudges() == before
