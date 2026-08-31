from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from types import SimpleNamespace

from mu.agent.approval import build_approval_plan
from mu.agent.hooks import HookContext
from mu.agent.thread_guard import _guard
from mu.gui.bus import EventBus
from mu.session.manager import SessionManager
from mu.threads.coordinator import ThreadCoordinator
from mu.threads.model import ensure_thread_meta, new_child_thread_meta


def _register_pair(root: Path):
    parent = ensure_thread_meta("parent")
    child = new_child_thread_meta(parent, title="Peer")
    coordinator = ThreadCoordinator(parent.group_id, root=str(root))
    coordinator.register_thread(parent, "parent")
    coordinator.register_thread(child, "peer")
    return coordinator, parent, child


def test_create_thread_inherits_environment_but_not_conversation_state(tmp_path, monkeypatch):
    history_root = tmp_path / "state"
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(history_root))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    manager = SessionManager()
    manager.new_session("parent", "openai", "model-x")
    manager.variables["agent_mode"] = "research"
    manager.container_config = {"container_name": "shared-container"}
    manager.folder_context.add_folder(str(workspace))
    manager.history = [{"role": "user", "parts": [{"type": "text", "text": "secret history"}]}]
    manager.task_memory.save("parent-only state")

    created = manager.create_thread(title="Investigate parser", session_name="peer")
    child = SessionManager(session_name="peer")

    assert created["thread_meta"]["group_id"] == manager.thread_meta.group_id
    assert child.thread_meta.parent_thread_id == manager.thread_meta.thread_id
    assert child.provider_config == manager.provider_config
    assert child.variables["agent_mode"] == "research"
    assert child.container_config == {"container_name": "shared-container"}
    assert child.folder_context.folders == [str(workspace)]
    assert child.history == []
    assert child.conversation_summary == ""
    assert child.task_memory.entries == []


def test_messages_are_durable_scrubbed_and_create_one_coalesced_wake(tmp_path):
    coordinator, parent, child = _register_pair(tmp_path)
    first = coordinator.send_message(
        parent.thread_id,
        child.thread_id,
        "Use token ghp_abcdefghijklmnopqrstuvwxyz0123456789 on parser.py",
        related_paths=["parser.py"],
    )
    second = coordinator.send_message(parent.thread_id, child.thread_id, "One more detail")

    assert first["wake_id"] == second["wake_id"]
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in first["content"]
    wake = coordinator.claim_wake("runtime-test")
    assert wake and wake["target_thread_id"] == child.thread_id

    block = coordinator.context_block(child.thread_id)
    assert first["message_id"] in block
    assert "One more detail" in block
    with coordinator._connect() as conn:
        status = conn.execute(
            "SELECT status FROM wake_requests WHERE wake_id=?", (first["wake_id"],)
        ).fetchone()[0]
    assert status == "done"


def test_execution_lease_is_exclusive_and_status_goals_are_scrubbed(tmp_path):
    coordinator, parent, _child = _register_pair(tmp_path)

    assert coordinator.heartbeat(parent.thread_id, "runtime-one", ttl=120) is True
    assert coordinator.heartbeat(parent.thread_id, "runtime-two", ttl=120) is False

    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    coordinator.set_status(parent.thread_id, "running", goal=f"Use {secret}")
    assert secret not in coordinator.get_thread(parent.thread_id)["current_goal"]
    assert secret not in json.dumps(coordinator.activity())

    coordinator.release_execution_lease(parent.thread_id, "runtime-one")
    assert coordinator.heartbeat(parent.thread_id, "runtime-two", ttl=120) is True


def test_path_claim_conflict_requires_coordination_and_human_override(tmp_path):
    coordinator, parent, child = _register_pair(tmp_path)
    target = tmp_path / "shared.py"
    owner = coordinator.claim_paths(
        parent.thread_id, [str(target)], turn_id="turn-owner", ttl=3600
    )
    assert owner["ok"] is True
    collision = coordinator.claim_paths(
        child.thread_id, [str(target)], turn_id="turn-peer", ttl=3600
    )
    assert collision["ok"] is False
    conflict = collision["conflicts"][0]

    plan = build_approval_plan(
        "request_thread_claim_override",
        {"conflict_id": conflict["conflict_id"], "rationale": "Owner is unavailable"},
        folder_context=None,
        yolo=True,
    )
    assert plan.requires_approval is True
    assert plan.approval_policy == "always_human"

    batch_plan = build_approval_plan(
        "batch_job",
        {
            "commands": [
                {
                    "tool_name": "request_thread_claim_override",
                    "tool_args": {
                        "conflict_id": conflict["conflict_id"],
                        "rationale": "Owner is unavailable",
                    },
                }
            ]
        },
        folder_context=None,
        yolo=True,
    )
    assert batch_plan.requires_approval is True
    assert batch_plan.approval_policy == "always_human"

    overridden = coordinator.override_conflict(
        conflict["conflict_id"], child.thread_id, "Human approved after review"
    )
    assert overridden["state"] == "overridden"
    assert coordinator.active_claims()[0]["owner_thread_id"] == child.thread_id


def test_native_write_guard_blocks_peer_owned_file_and_notifies_owner(tmp_path):
    coordinator, parent, child = _register_pair(tmp_path)
    target = tmp_path / "shared.py"
    coordinator.claim_paths(parent.thread_id, [str(target)], turn_id="owner", ttl=3600)
    session = SimpleNamespace(
        thread_coordinator=coordinator,
        thread_meta=child,
        _thread_turn_id="requester",
        folder_context=SimpleNamespace(folders=[str(tmp_path)]),
    )
    result = _guard(
        HookContext(
            point="pre_tool",
            session=session,
            tool_name="write_file",
            tool_args={"filename": "shared.py", "content": "new"},
        )
    )

    assert result is not None and result.action == "short_circuit"
    assert result.payload["error_code"] == "thread_path_conflict"
    incoming = coordinator.open_messages(parent.thread_id)
    assert incoming[0]["kind"] == "path_conflict"
    assert str(target) in incoming[0]["content"]


def test_event_bus_delivers_group_events_without_cross_session_chat_leakage():
    async def scenario():
        bus = EventBus()
        queue = bus.subscribe(session_name="parent", thread_group_id="tg-test")
        await bus.publish(
            {"kind": "assistant_delta", "session_name": "unrelated", "text": "hidden"}
        )
        await bus.publish(
            {
                "kind": "thread_message",
                "session_name": "peer",
                "thread_group_id": "tg-test",
            }
        )
        event = queue.get_nowait()
        assert event["kind"] == "thread_message"
        assert queue.empty()

    asyncio.run(scenario())


def test_gui_and_tui_expose_thread_navigation_and_audit_panel():
    root = Path(__file__).resolve().parents[1]
    template = (root / "mu/gui/templates/index.html").read_text(encoding="utf-8")
    panel = (root / "mu/gui/templates/fragments/threads_panel.html").read_text(encoding="utf-8")
    javascript = (root / "mu/gui/static/js/app.js").read_text(encoding="utf-8")
    tui = (root / "mu/ui/input.py").read_text(encoding="utf-8")

    assert "$store.threads.groupRoster" in template
    assert "Agent conversations" in panel
    assert 'Alpine.store("threads"' in javascript
    assert "Keys.ShiftLeft" in tui
    assert 'result="/thread"' in tui


def test_gui_and_tui_expose_thread_delete():
    root = Path(__file__).resolve().parents[1]
    template = (root / "mu/gui/templates/index.html").read_text(encoding="utf-8")
    javascript = (root / "mu/gui/static/js/app.js").read_text(encoding="utf-8")
    commands = (root / "mu/commands/thread.py").read_text(encoding="utf-8")
    tui = (root / "mu/ui/input.py").read_text(encoding="utf-8")
    router = (root / "mu/gui/routers/threads.py").read_text(encoding="utf-8")

    # GUI: delete affordance + store action hitting the DELETE endpoint.
    assert "$store.threads.remove(t)" in template
    assert "method: \"DELETE\"" in javascript
    # TUI: /thread delete subcommand, picker shortcut, and completer entry.
    assert "delete" in commands and "thread delete" in commands
    assert '"delete": session_completer' in tui
    # Backend: DELETE route exists.
    assert '@router.delete("/{thread_id}")' in router


def test_tui_thread_delete_uses_session_manager_delete(tmp_path, monkeypatch):
    """The /thread delete flow removes both coordination rows and the session dir."""
    from mu.commands import thread as thread_cmd

    history_root = tmp_path / "state"
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(history_root))
    manager = SessionManager()
    manager.new_session("parent", "openai", "model-x")
    created = manager.create_thread(title="Doomed", session_name="doomed")

    child_meta = created["thread_meta"]
    coordinator = ThreadCoordinator(child_meta["group_id"], root=str(history_root))

    session = SimpleNamespace(
        session_manager=manager,
        ui=None,
        thread_meta=manager.thread_meta,
    )

    coordinator_for_calls = {}

    def fake_coordinator(target):
        coordinator_for_calls.setdefault(
            "session_name", target.session_manager.current_session_name
        )
        return coordinator

    monkeypatch.setattr(thread_cmd, "_coordinator", fake_coordinator)

    import mu.session as _session_pkg

    real_delete = SessionManager.delete_session

    def record_delete(self, name):
        record_delete.deleted = name
        return real_delete(self, name)

    record_delete.deleted = None
    monkeypatch.setattr(SessionManager, "delete_session", record_delete)

    result = thread_cmd._delete(session, "doomed", allow_prompt=False)

    assert result.ok is True
    assert "Deleted thread 'doomed'" in result.message
    assert record_delete.deleted == "doomed"
    assert not (history_root / "sessions" / "doomed").exists()
    assert coordinator.get_thread(child_meta["thread_id"]) is None


def test_delete_thread_purges_coordination_rows(tmp_path):
    coordinator, parent, child = _register_pair(tmp_path)
    coordinator.send_message(parent.thread_id, child.thread_id, "hello parser")
    coordinator.claim_paths(
        child.thread_id, ["parser.py"], turn_id="turn-1", note="hold"
    )
    coordinator.heartbeat(child.thread_id, "runtime-1")

    result = coordinator.delete_thread(child.thread_id)

    assert result["ok"] is True
    assert coordinator.get_thread(child.thread_id) is None
    names = {item["thread_id"] for item in coordinator.list_threads()}
    assert child.thread_id not in names
    assert parent.thread_id in names
    # Child's path claims released so peers are unblocked.
    assert coordinator.get_thread(child.thread_id) is None
    # Deleted thread's claims surfaced nowhere.
    for item in coordinator.list_threads():
        assert "parser.py" not in (item.get("claimed_paths") or [])
    # Audit trail keeps the tombstone event.
    kinds = [event["kind"] for event in coordinator.activity()]
    assert "thread_deleted" in kinds


def test_delete_thread_refuses_last_member(tmp_path):
    coordinator, parent, child = _register_pair(tmp_path)
    coordinator.delete_thread(child.thread_id)
    from mu.threads.coordinator import ThreadCoordinatorError

    with pytest.raises(ThreadCoordinatorError):
        coordinator.delete_thread(parent.thread_id)


def _write_thread_session(history_root: Path, name: str, provider: str | None = None,
                          model: str | None = None) -> None:
    """Materialize a minimal on-disk thread session for router-level tests."""
    from mu.threads.model import ThreadMeta
    import time as _time

    session_dir = history_root / "sessions" / name
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1,
        "thread_id": f"th-{name.replace('-', '')[:24]}",
        "group_id": "tg-testfixed",
        "title": name,
        "created_at": _time.time(),
        "parent_thread_id": "",
    }
    doc = {
        "name": name,
        "revision": 0,
        "thread_meta": meta,
        "variables": {"session_type": "workspace"},
        "provider_config": (
            {"provider": provider, "model": model}
            if provider and model else {}
        ),
        "history": [],
    }
    (session_dir / "session.json").write_text(json.dumps(doc))


def _bootstrap_group(history_root: Path, child_name: str) -> str:
    """Register parent+child rows in a coordination DB under history root."""
    parent_meta = ensure_thread_meta("mucli")
    child_meta = new_child_thread_meta(parent_meta, title=child_name)
    coordinator = ThreadCoordinator(parent_meta.group_id, root=str(history_root))
    coordinator.register_thread(parent_meta, "mucli")
    coordinator.register_thread(child_meta, child_name)
    _write_thread_session(history_root, "mucli", "openai", "model-x")
    _write_thread_session(history_root, child_name)
    return child_meta.thread_id


def test_delete_session_cleans_thread_coordination_row(tmp_path, monkeypatch):
    """Deleting a thread's session via the sessions API removes its roster row
    (bug 2 regression: orphaned ghost thread after sessions-list delete)."""
    from unittest.mock import patch as _patch

    import mu.gui.routers.sessions as sessions_router

    history_root = tmp_path / "state"
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(history_root))
    child_id = _bootstrap_group(history_root, "delete-thread")

    assert sessions_router._thread_row_for_session("delete-thread") == child_id

    # Simulate the sessions-list delete path (dir removal + coordination cleanup).
    import shutil as _shutil
    _shutil.rmtree(history_root / "sessions" / "delete-thread")
    # Reuse the router helper that delete_session now calls after rmtree.
    with _patch(
        "mu.gui.routers.sessions._iter_thread_group_dbs",
        sessions_router._iter_thread_group_dbs,
    ):
        coordinator = ThreadCoordinator(
            ensure_thread_meta("mucli").group_id, root=str(history_root)
        )
        from mu.gui.routers.threads import _prune_orphan_threads

        pruned = _prune_orphan_threads(coordinator)
    assert "delete-thread" in pruned
    assert sessions_router._thread_row_for_session("delete-thread") is None
    assert coordinator.get_thread(child_id) is None


def test_prune_orphans_keeps_last_remaining_thread(tmp_path):
    """A singleton thread with a missing dir must NOT be pruned away."""
    from mu.gui.routers.threads import _prune_orphan_threads

    history_root = tmp_path / "state"
    history_root.mkdir(parents=True, exist_ok=True)
    parent_meta = ensure_thread_meta("lonely")
    coordinator = ThreadCoordinator(parent_meta.group_id, root=str(history_root))
    coordinator.register_thread(parent_meta, "lonely")

    pruned = _prune_orphan_threads(coordinator)
    assert pruned == []
    assert coordinator.get_thread(parent_meta.thread_id) is not None


def test_provider_falls_back_to_thread_group_sibling(tmp_path, monkeypatch):
    """A providerless thread session resolves provider via a group sibling."""
    import mu.gui.routers.sessions as sessions_router

    history_root = tmp_path / "state"
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(history_root))
    _bootstrap_group(history_root, "old-thread")

    inherited = sessions_router._provider_from_thread_group("old-thread")
    assert inherited == {"provider": "openai", "model": "model-x"}


def test_orphan_thread_session_detected_for_404(tmp_path, monkeypatch):
    """A session dir deleted under a live thread row is detectable (bug 1)."""
    import mu.gui.routers.sessions as sessions_router

    history_root = tmp_path / "state"
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(history_root))
    _bootstrap_group(history_root, "ghost-thread")
    import shutil as _shutil

    _shutil.rmtree(history_root / "sessions" / "ghost-thread")
    assert sessions_router._thread_row_for_session("ghost-thread") is not None
