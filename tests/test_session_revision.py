"""Cross-surface continuity phase 1: session revision counter + CAS + If-Match.

Covers (design doc documentation/cross_surface_continuity.md §3.1):
- revision starts at 0, hydrates from session.json, increments per save;
- save_history(expected_revision=N) is a compare-and-swap: success bumps,
  stale expectation raises RevisionConflict without writing;
- legacy sessions without a revision field load as 0 and gain it on save;
- GUI routes expose revision on GET payloads and honor If-Match on
  mutating routes (409 + current_revision on mismatch).
"""

import asyncio
import json
import os
import sys
import tempfile
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mu.session.manager import RevisionConflict, SessionManager  # noqa: E402
from mu.memory.stores import TaskMemoryStore  # noqa: E402


@pytest.fixture()
def isolated_sessions(monkeypatch, tmp_path):
    # HISTORY_DIR is the mucli home root; sessions live under <root>/sessions.
    mucli_home = str(tmp_path / "mucli-home")
    os.makedirs(os.path.join(mucli_home, "sessions"), exist_ok=True)
    monkeypatch.setenv("MUCLI_HOME", mucli_home)
    import utils.config as _config

    monkeypatch.setattr(_config, "HISTORY_DIR", mucli_home)
    return mucli_home


class TestRevisionCounterUnit:
    def test_new_session_starts_at_zero(self, isolated_sessions):
        sm = SessionManager(session_name="rev-zero")
        assert sm.revision == 0

    def test_save_increments_and_persists(self, isolated_sessions):
        sm = SessionManager(session_name="rev-inc")
        sm.history = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
        sm.save_history()
        assert sm.revision == 1
        on_disk = json.load(
            open(os.path.join(isolated_sessions, "sessions", "rev-inc", "session.json"))
        )
        assert on_disk["revision"] == 1
        sm.save_history()
        assert sm.revision == 2

    def test_reload_hydrates_revision(self, isolated_sessions):
        sm = SessionManager(session_name="rev-hydrate")
        sm.history = [{"role": "user", "parts": [{"type": "text", "text": "x"}]}]
        sm.save_history()
        sm.save_history()
        sm2 = SessionManager(session_name="rev-hydrate")
        sm2._load_session("rev-hydrate")
        assert sm2.revision == 2

    def test_legacy_session_without_revision_loads_zero(self, isolated_sessions):
        d = os.path.join(isolated_sessions, "sessions", "rev-legacy")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "session.json"), "w") as fh:
            json.dump({"history": []}, fh)
        sm = SessionManager(session_name="rev-legacy")
        sm._load_session("rev-legacy")
        assert sm.revision == 0
        sm.save_history()
        assert sm.revision == 1

    def test_cas_success_bumps(self, isolated_sessions):
        sm = SessionManager(session_name="rev-cas")
        sm.save_history()
        assert sm.revision == 1
        sm.save_history(expected_revision=1)
        assert sm.revision == 2

    def test_cas_conflict_raises_without_writing(self, isolated_sessions):
        sm = SessionManager(session_name="rev-conflict")
        sm.save_history()
        sm.revision = 7  # simulate another surface having advanced on-disk
        with pytest.raises(RevisionConflict) as exc:
            sm.save_history(expected_revision=1)
        assert exc.value.expected == 1
        assert exc.value.current == 7
        # No write happened: on-disk revision still the pre-simulated one.
        on_disk = json.load(
            open(
                os.path.join(
                    isolated_sessions, "sessions", "rev-conflict", "session.json"
                )
            )
        )
        assert on_disk["revision"] == 1

    def test_malformed_revision_field_loads_zero(self, isolated_sessions):
        d = os.path.join(isolated_sessions, "sessions", "rev-malformed")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "session.json"), "w") as fh:
            json.dump({"history": [], "revision": "not-a-number"}, fh)
        sm = SessionManager(session_name="rev-malformed")
        sm._load_session("rev-malformed")
        assert sm.revision == 0


class TestIfMatchRoutes:
    def _build_app_state(self, isolated_sessions, name, sm):
        state = types.SimpleNamespace()
        state.current_session_name = name
        state.sessions = {name: types.SimpleNamespace(session_manager=sm)}
        state.session_busy_for = lambda _n: types.SimpleNamespace(is_set=lambda: False)
        state.session_lock_for = lambda _n=None: __import__("threading").Lock()
        state.unload_session = lambda name=None: state.sessions.pop(name, None)
        state.watcher = None
        return state

    def _request(self, app_state):
        from mu.gui.routers.sessions import _check_if_match

        return _check_if_match, app_state

    def test_check_if_match_mismatch_returns_409(self, isolated_sessions):
        from mu.gui.routers.sessions import _check_if_match
        from fastapi.responses import JSONResponse

        sm = SessionManager(session_name="rev-r1")
        sm.revision = 3
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=None))
        resp = _check_if_match(request, sm, "2")
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 409

    def test_check_if_match_match_returns_none(self, isolated_sessions):
        from mu.gui.routers.sessions import _check_if_match

        sm = SessionManager(session_name="rev-r2")
        sm.revision = 3
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=None))
        assert _check_if_match(request, sm, "3") is None

    def test_check_if_match_absent_is_noop(self, isolated_sessions):
        from mu.gui.routers.sessions import _check_if_match

        sm = SessionManager(session_name="rev-r3")
        sm.revision = 3
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=None))
        assert _check_if_match(request, sm, None) is None

    def test_check_if_match_invalid_header_400(self, isolated_sessions):
        from fastapi import HTTPException

        from mu.gui.routers.sessions import _check_if_match

        sm = SessionManager(session_name="rev-r4")
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=None))
        with pytest.raises(HTTPException) as exc:
            _check_if_match(request, sm, "xyz")
        assert exc.value.status_code == 400

    def test_check_if_match_on_disk_dict(self, isolated_sessions):
        from mu.gui.routers.sessions import _check_if_match
        from fastapi.responses import JSONResponse

        request = types.SimpleNamespace(app=types.SimpleNamespace(state=None))
        assert _check_if_match(request, {"revision": 4}, "4") is None
        resp = _check_if_match(request, {"revision": 4}, "3")
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 409

    def test_unload_active_conflict(self, isolated_sessions):
        from mu.gui.routers.sessions import unload_active_session

        sm = SessionManager(session_name="rev-unload")
        sm.revision = 5
        state = self._build_app_state(isolated_sessions, "rev-unload", sm)

        class _Req:
            app = types.SimpleNamespace(state=state)

        resp = asyncio.run(unload_active_session(_Req(), if_match="4"))
        assert resp.status_code == 409
        # Session NOT unloaded.
        assert "rev-unload" in state.sessions

    def test_unload_active_success(self, isolated_sessions):
        from mu.gui.routers.sessions import unload_active_session

        sm = SessionManager(session_name="rev-unload-ok")
        sm.revision = 5
        state = self._build_app_state(isolated_sessions, "rev-unload-ok", sm)

        class _Req:
            app = types.SimpleNamespace(state=state)

        result = asyncio.run(unload_active_session(_Req(), if_match="5"))
        assert result == {"ok": True, "active": False}
        assert "rev-unload-ok" not in state.sessions

# ---------------------------------------------------------------------------
# Phase-6 F47: turn-scoped CAS (begin_turn_cas / save_history_turn /
# end_turn_cas). A concurrent surface write DURING an agent turn must be
# detected and merged (winner adopted, turn messages re-applied without
# duplicating messages the turn already persisted) instead of clobbered.
# ---------------------------------------------------------------------------


def _load_document(home, name):
    return json.load(
        open(os.path.join(home, "sessions", name, "session.json"))
    )


def _texts(doc):
    return [
        part.get("text")
        for message in doc["history"]
        for part in message.get("parts", [])
    ]


def _make_manager(name, history):
    sm = SessionManager(session_name=name)
    sm.current_session_name = name
    sm.history = history
    return sm


def test_turn_cas_merges_winner_without_duplicates(isolated_sessions):
    """Conflict mid-turn: winner's document + turn's new messages both
    survive, already-persisted turn messages are not duplicated."""
    sm = _make_manager(
        "f47-merge",
        [{"role": "user", "parts": [{"type": "text", "text": "turn prompt"}]}],
    )
    sm._active_turn_start_index = 0
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()

    # Turn saves once successfully: tool result is now on disk (rev 2).
    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "tool result"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    # Concurrent GUI write at rev 2 -> rev 3.
    other = _make_manager(
        "f47-merge",
        [
            {"role": "user", "parts": [{"type": "text", "text": "turn prompt"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "tool result"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI edit mid-turn"}]},
        ],
    )
    other.save_history(folder_context_obj=None, expected_revision=2)

    # Turn appends and saves -> conflict -> winner adopted, turn msg kept.
    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "final answer"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    doc = _load_document(isolated_sessions, "f47-merge")
    texts = _texts(doc)
    assert texts.count("turn prompt") == 1
    assert texts.count("tool result") == 1  # not duplicated
    assert "GUI edit mid-turn" in texts  # winner preserved
    assert texts.count("final answer") == 1  # turn work preserved
    assert sm._turn_cas_baseline == doc["revision"]  # re-baselined


def test_turn_cas_disarms_in_end_and_lww_without_arm(isolated_sessions):
    """save_history_turn without an armed turn behaves as plain LWW, and
    end_turn_cas fully disarms."""
    sm = _make_manager(
        "f47-lww", [{"role": "user", "parts": [{"type": "text", "text": "x"}]}]
    )
    sm.save_history_turn(folder_context_obj=None)
    doc = _load_document(isolated_sessions, "f47-lww")
    assert doc["revision"] == 1

    sm.begin_turn_cas()
    assert sm._turn_cas_armed and sm._turn_cas_baseline == 1
    sm.end_turn_cas()
    assert not sm._turn_cas_armed and sm._turn_cas_baseline is None


def test_turn_cas_winner_adopted_without_turn_index(isolated_sessions):
    """No _active_turn_start_index: nothing is attributable to the turn,
    so the conflict path adopts the winner's document wholesale."""
    sm = _make_manager(
        "f47-noindex", [{"role": "user", "parts": [{"type": "text", "text": "x"}]}]
    )
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm._active_turn_start_index = None
    sm.begin_turn_cas()

    other = _make_manager(
        "f47-noindex", [{"role": "user", "parts": [{"type": "text", "text": "y"}]}]
    )
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.save_history_turn(folder_context_obj=None)
    doc = _load_document(isolated_sessions, "f47-noindex")
    assert _texts(doc) == ["y"]  # winner adopted wholesale


def test_turn_cas_merge_keeps_winner_repeats_multiset(isolated_sessions):
    """r21 F2: a legitimately DISTINCT winner message repeating content
    already in the pre-turn prefix must survive the merge (multiset
    diff, not global membership set)."""
    sm = _make_manager(
        "r21-repeat",
        [{"role": "user", "parts": [{"type": "text", "text": "retry"}]}],
    )
    sm._active_turn_start_index = 1  # turn starts AFTER existing "retry"
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()

    other = _make_manager(
        "r21-repeat",
        [
            {"role": "user", "parts": [{"type": "text", "text": "retry"}]},
            {"role": "user", "parts": [{"type": "text", "text": "retry"}]},
            {"role": "user", "parts": [{"type": "text", "text": "surface marker"}]},
        ],
    )
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "turn work"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    doc = _load_document(isolated_sessions, "r21-repeat")
    texts = _texts(doc)
    assert texts.count("retry") == 2  # both copies kept
    assert "surface marker" in texts
    assert texts.count("turn work") == 1


def test_save_history_if_current_refuses_stale_write(isolated_sessions):
    """r21 F5: post-turn best-effort save must not clobber a concurrent
    NEWER write once the turn CAS is disarmed."""
    sm = _make_manager(
        "r21-ifcur", [{"role": "user", "parts": [{"type": "text", "text": "p"}]}]
    )
    sm.save_history(folder_context_obj=None, expected_revision=0)  # rev 1

    other = _make_manager(
        "r21-ifcur",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "newer GUI write"}]},
        ],
    )
    other.save_history(folder_context_obj=None, expected_revision=1)  # rev 2

    assert sm.save_history_if_current(folder_context_obj=None) is False
    doc = _load_document(isolated_sessions, "r21-ifcur")
    assert "newer GUI write" in _texts(doc)
    assert doc["revision"] == 2


def test_turn_cas_merge_after_turn_collapse(isolated_sessions):
    """r22 F1: compact_completed_turn reindexes history; the armed turn
    start index must be remapped so the conflict-merge slice still
    covers the turn's messages (previously went stale -> turn msgs
    dropped from the merged document)."""
    sm = _make_manager(
        "r22-collapse",
        [{"role": "user", "parts": [{"type": "text", "text": f"old{i}"}]} for i in range(6)],
    )
    sm._active_turn_start_index = None
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    sm.history.append(
        {"role": "user", "parts": [{"type": "text", "text": "turn prompt"}]}
    )
    sm._active_turn_start_index = 6
    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "tool msg 1"}]}
    )
    sm.save_history_turn(folder_context_obj=None)  # rev 2

    sm.compact_completed_turn()  # collapses turn to prompt + final text
    assert sm._active_turn_start_index == 6  # remapped, still valid

    other = _make_manager(
        "r22-collapse", sm.history + [
            {"role": "user", "parts": [{"type": "text", "text": "GUI edit"}]}
        ]
    )
    other.save_history(folder_context_obj=None, expected_revision=2)
    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "final"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    doc = _load_document(isolated_sessions, "r22-collapse")
    texts = _texts(doc)
    assert "GUI edit" in texts and "final" in texts


def test_winner_reload_normalizes_like_load(isolated_sessions):
    """r22 F4: hydration of a legacy winner must apply the same
    normalization invariants as _load_session (registry dict-filter,
    active-state fallback, research-source validation, tool_stats
    normalize) — not naive dict() coercion."""
    sm = _make_manager(
        "r22-norm", [{"role": "user", "parts": [{"type": "text", "text": "p"}]}]
    )
    sm._active_turn_start_index = 1
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()

    path = os.path.join(isolated_sessions, "sessions", "r22-norm", "session.json")
    doc = json.load(open(path))
    doc["feature_state"] = None
    doc["feature_registry"] = {"f1": {"status": "ok"}, "f2": "not-a-dict"}
    doc["active_feature_id"] = "f1"
    doc["teacher_state"] = None
    doc["research_sources"] = "not-a-list"
    doc["tool_stats"] = None
    doc["revision"] = 2  # concurrent writer past the baseline
    json.dump(doc, open(path, "w"))

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    assert sm.feature_state == {"status": "ok"}  # registry fallback
    assert sm.feature_registry == {"f1": {"status": "ok"}}  # filtered
    assert sm.research_sources == []  # type-validated
    assert isinstance(sm.tool_stats, dict) and sm.tool_stats  # normalized


def test_goal_strip_save_merges_and_clear_survives(isolated_sessions):
    """r23 F1: the goal-strip save runs inside the armed turn window —
    it must merge against a concurrent write, and the cleared
    session_goal variable must survive the merge."""
    sm = _make_manager(
        "r23-goal", [{"role": "user", "parts": [{"type": "text", "text": "p"}]}]
    )
    sm._active_turn_start_index = 1
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    sm.variables["session_goal"] = ""

    other = _make_manager(
        "r23-goal",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI edit"}]},
        ],
    )
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.save_history_turn(sm.folder_context)
    doc = _load_document(isolated_sessions, "r23-goal")
    assert doc["variables"].get("session_goal") == ""
    assert "GUI edit" in _texts(doc)


def test_task_memory_overlay_survives_conflict(isolated_sessions):
    """r23 F2: local task_memory mutations made after the last
    successful save (durable_id marker + newly captured entry) must
    survive the conflict-merge's winner adoption."""
    from mu.session.manager import SessionManager as _SM  # noqa: F401

    sm = _make_manager(
        "r23-overlay", [{"role": "user", "parts": [{"type": "text", "text": "p"}]}]
    )
    sm._active_turn_start_index = 1
    e_old = sm.task_memory.save(content="existing entry", kind="observation")
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    e_old.durable_id = "dur-123"
    e_new = sm.task_memory.save(content="captured this turn", kind="finding")

    other = _make_manager(
        "r23-overlay",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI"}]},
        ],
    )
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    e_old_after = sm.task_memory.get_entry(e_old.id)
    e_new_after = sm.task_memory.get_entry(e_new.id)
    assert e_old_after is not None and e_old_after.durable_id == "dur-123"
    assert e_new_after is not None
    doc = _load_document(isolated_sessions, "r23-overlay")
    tm_ids = {e["id"] for e in doc["task_memory"]["entries"]}
    assert e_new.id in tm_ids


def test_winner_null_state_resets_stale_local_state(isolated_sessions):
    """r23 F3: hydration must RESET feature_state/teacher_state to None
    before conditional assignment, so a winner carrying intentional
    null (no registry fallback) is not overwritten by stale local
    state on the retry save."""
    sm = _make_manager(
        "r23-null", [{"role": "user", "parts": [{"type": "text", "text": "p"}]}]
    )
    sm._active_turn_start_index = 1
    sm.feature_state = {"old": True}
    sm.teacher_state = {"stale": 1}
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()

    path = os.path.join(isolated_sessions, "sessions", "r23-null", "session.json")
    doc = json.load(open(path))
    doc["feature_state"] = None
    doc["teacher_state"] = None
    doc["revision"] = 2
    json.dump(doc, open(path, "w"))

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    assert sm.feature_state is None
    assert sm.teacher_state is None
    doc = _load_document(isolated_sessions, "r23-null")
    assert doc["feature_state"] is None and doc["teacher_state"] is None


def test_task_memory_delta_winner_deletion_wins(isolated_sessions):
    """r24 F1: only THIS TURN's delta is re-applied onto the winner —
    a baseline entry the concurrent surface deleted/evicted stays
    deleted; a turn-local addition still survives."""
    sm = _make_manager(
        "r24-delta-del",
        [{"role": "user", "parts": [{"type": "text", "text": "p"}]}],
    )
    sm._active_turn_start_index = 1
    e_base = sm.task_memory.save(content="baseline entry", kind="observation")
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    e_new = sm.task_memory.save(content="captured this turn", kind="finding")

    other = _make_manager(
        "r24-delta-del",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI"}]},
        ],
    )
    other.task_memory.clear()
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    assert sm.task_memory.get_entry(e_base.id) is None
    assert sm.task_memory.get_entry(e_new.id) is not None
    doc = _load_document(isolated_sessions, "r24-delta-del")
    contents = [e["content"] for e in doc["task_memory"]["entries"]]
    assert "baseline entry" not in contents
    assert "captured this turn" in contents


def test_task_memory_delta_id_collision_imports_fresh(isolated_sessions):
    """r24 F2: concurrent surfaces can allocate the same numeric id
    for DIFFERENT entries. The conflict import must give the local
    entry a fresh id instead of adopting the winner's unrelated
    content as the local mutation target."""
    sm = _make_manager(
        "r24-delta-coll",
        [{"role": "user", "parts": [{"type": "text", "text": "p"}]}],
    )
    sm._active_turn_start_index = 1
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    sm.task_memory.save(content="LOCAL TURN CAPTURE", kind="finding")

    other = _make_manager(
        "r24-delta-coll",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI"}]},
        ],
    )
    other.task_memory.save(content="CONCURRENT CAPTURE", kind="finding")
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    winner1 = sm.task_memory.get_entry(1)
    assert winner1 is not None and winner1.content == "CONCURRENT CAPTURE"
    local = next(
        e for e in sm.task_memory.entries if e.content == "LOCAL TURN CAPTURE"
    )
    assert local is not None and local.id != 1
    doc = _load_document(isolated_sessions, "r24-delta-coll")
    contents = [e["content"] for e in doc["task_memory"]["entries"]]
    assert "CONCURRENT CAPTURE" in contents and "LOCAL TURN CAPTURE" in contents


def test_task_memory_delta_status_both_transitions_kept(isolated_sessions):
    """r24 F1 (mutation guard): a local status transition is adopted
    only when the winner still carries the baseline value. Winner-side
    transitions of other entries survive untouched."""
    sm = _make_manager(
        "r24-delta-status",
        [{"role": "user", "parts": [{"type": "text", "text": "p"}]}],
    )
    sm._active_turn_start_index = 1
    e1 = sm.task_memory.save(content="entry one", kind="observation")
    e2 = sm.task_memory.save(content="entry two", kind="observation")
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    sm.task_memory.update_status(e2.id, "done")

    other = _make_manager(
        "r24-delta-status",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI"}]},
        ],
    )
    other.task_memory.update_status(e1.id, "done")
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    assert sm.task_memory.get_entry(e1.id).status == "done"
    assert sm.task_memory.get_entry(e2.id).status == "done"


def test_import_entries_collision_remap_and_cap():
    """r24 F3: BaseNoteStore.import_entries remaps colliding ids to
    fresh ids (keeping free ids as-is), remaps intra-batch
    supersedes/superseded_by refs, advances _next_id, and enforces the
    store cap."""
    store = TaskMemoryStore(max_entries=4)
    for content in ("a", "b", "c", "d"):
        store.save(content=content, kind="finding")
    # 'b' gets a hit so eviction (lowest score, oldest first) targets
    # 'a' instead — makes the "winner entry intact" assertion
    # deterministic.
    store.get_entry(2).hits = 1
    imported = store.import_entries(
        [
            {"id": 2, "content": "LOCAL-2", "kind": "finding", "supersedes": 7},
            {"id": 7, "content": "LOCAL-7", "kind": "finding", "superseded_by": 2},
        ]
    )
    assert len(store.entries) == 4  # cap enforced after import
    assert store.get_entry(2).content == "b"  # winner entry intact
    local2 = next(e for e in store.entries if e.content == "LOCAL-2")
    local7 = next(e for e in store.entries if e.content == "LOCAL-7")
    assert local2.id == 5 and local7.id == 7  # collision remapped, free id kept
    assert local2.supersedes == 7 and local7.superseded_by == 5
    assert store._next_id == 8
    assert [e.id for e in imported] == [5, 7]


def test_task_memory_delta_supersedes_pointers_merge(isolated_sessions):
    """r25 F1: a local supersede(old, new) during the turn sets
    status=superseded AND the supersedes/superseded_by pointers. The
    delta merge must adopt the pointers too (baseline-guarded), or the
    winner keeps a superseded entry with a dangling back-pointer."""
    sm = _make_manager(
        "r25-supersedes",
        [{"role": "user", "parts": [{"type": "text", "text": "p"}]}],
    )
    sm._active_turn_start_index = 1
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    e_base = sm.task_memory.save(content="baseline obs", kind="observation")
    e_new = sm.task_memory.save(content="turn addition", kind="finding")
    sm.task_memory.supersede(e_base.id, e_new.id)

    other = _make_manager(
        "r25-supersedes",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI"}]},
        ],
    )
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    merged_old = sm.task_memory.get_entry(e_base.id)
    merged_new = sm.task_memory.get_entry(e_new.id)
    assert merged_old is not None and merged_old.status == "superseded"
    assert merged_old.superseded_by == e_new.id
    assert merged_new is not None and merged_new.supersedes == e_base.id
    doc = _load_document(isolated_sessions, "r25-supersedes")
    by_id = {e["id"]: e for e in doc["task_memory"]["entries"]}
    assert by_id[e_base.id]["superseded_by"] == e_new.id
    assert by_id[e_new.id]["supersedes"] == e_base.id


def test_import_entries_duplicate_ids_distinct(isolated_sessions):
    """r25 F2: duplicate ids inside one import batch (or two items
    without ids) must each get a DISTINCT id; refs to the duplicated
    id resolve to the first claimant."""
    store = TaskMemoryStore(max_entries=10)
    store.save(content="seed", kind="finding")
    imported = store.import_entries(
        [
            {"id": 5, "content": "dup-A", "kind": "finding"},
            {"id": 5, "content": "dup-B", "kind": "finding", "supersedes": 5},
        ]
    )
    ids = [e.id for e in imported]
    assert ids[0] != ids[1]
    assert ids[0] == 5  # first claimant keeps the id
    assert imported[1].supersedes == imported[0].id
    anon = store.import_entries(
        [
            {"content": "anon-1", "kind": "finding"},
            {"content": "anon-2", "kind": "finding"},
        ]
    )
    assert anon[0].id != anon[1].id
    all_ids = [e.id for e in store.entries]
    assert len(all_ids) == len(set(all_ids))


def test_delta_pointer_to_deleted_target_not_adopted(isolated_sessions):
    """r26 F1b: adopting a supersedes pointer whose TARGET the winner
    deleted/evicted would dangle — adoption is skipped entirely."""
    sm = _make_manager(
        "r26-no-dangle",
        [{"role": "user", "parts": [{"type": "text", "text": "p"}]}],
    )
    sm._active_turn_start_index = 1
    sm.save_history(folder_context_obj=None, expected_revision=0)
    sm.begin_turn_cas()
    victim = sm.task_memory.save(content="victim entry", kind="finding")
    pointer = sm.task_memory.save(content="pointer entry", kind="finding")
    sm.task_memory.supersede(victim.id, pointer.id)

    # Concurrent winner: fresh store without ANY baseline entries.
    other = _make_manager(
        "r26-no-dangle",
        [
            {"role": "user", "parts": [{"type": "text", "text": "p"}]},
            {"role": "user", "parts": [{"type": "text", "text": "GUI"}]},
        ],
    )
    other.task_memory.clear()
    other.save_history(folder_context_obj=None, expected_revision=1)

    sm.history.append(
        {"role": "assistant", "parts": [{"type": "text", "text": "w"}]}
    )
    sm.save_history_turn(folder_context_obj=None)

    for entry in sm.task_memory.entries:
        if entry.supersedes is not None:
            assert sm.task_memory.get_entry(entry.supersedes) is not None
        if entry.superseded_by is not None:
            assert sm.task_memory.get_entry(entry.superseded_by) is not None


def test_import_entries_evicted_items_not_in_return_semantics(isolated_sessions):
    """r26 F1a: when import_entries evicts entries (cap), every id the
    manager could map to must still exist in the store — i.e. the
    caller's get_entry filter (round-26 F1a) is load-bearing."""
    store = TaskMemoryStore(max_entries=3)
    for content in ("a", "b", "c"):
        store.save(content=content, kind="finding")
    store.get_entry(2).hits = 5  # protect 'b' from eviction
    store.import_entries(
        [
            {"id": 9, "content": "imported-x", "kind": "finding"},
            {"id": 10, "content": "imported-y", "kind": "observation"},
        ]
    )
    ids = [e.id for e in store.entries]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 3  # cap held
