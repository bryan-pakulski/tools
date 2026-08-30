"""Cross-session Memory Ledger: persistence, scope, audit and UX contracts."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from mu.memory.ledger import SQLiteMemoryLedger
from mu.memory.service import (
    DurableMemoryService,
    MemoryRejectedError,
    get_memory_service,
)
from mu.memory.stores import TaskMemoryStore


def _session(folder, name="session-a", provider="openai"):
    manager = SimpleNamespace(current_session_name=name, active_feature_id="")
    return SimpleNamespace(
        folder_context=SimpleNamespace(folders=[str(folder)] if folder else []),
        session_manager=manager,
        active_feature_id="",
        provider=SimpleNamespace(name=provider),
        variables={
            "durable_memory_enabled": True,
            "durable_memory_auto_capture": True,
        },
        task_memory=TaskMemoryStore(),
    )


def _git_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "remote",
            "add",
            "origin",
            "git@github.com:Example/Project.git",
        ],
        check=True,
    )
    return path


def test_memory_survives_distinct_sessions_in_same_repository(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    first = _session(repo, "first")
    second = _session(repo, "second")

    item, created = service.remember(
        first,
        "Tests run with uv run pytest.",
        kind="procedure",
        verification="tool_verified",
    )
    receipt = service.recall(second, "How do tests run?", limit=4, budget_tokens=500)

    assert created is True
    assert item.scope_type == "repository"
    assert item.scope_key == "repository:github.com/example/project"
    assert [candidate.item.id for candidate in receipt.included] == [item.id]
    assert service.ledger.get_recall(receipt.id)["session_name"] == "second"


def test_repository_scope_prevents_cross_repo_leakage(tmp_path):
    repo_a = _git_repo(tmp_path / "a")
    repo_b = _git_repo(tmp_path / "b")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_b),
            "remote",
            "set-url",
            "origin",
            "git@github.com:Example/Other.git",
        ],
        check=True,
    )
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    item, _ = service.remember(
        _session(repo_a), "Deploy with make release.", kind="procedure"
    )

    receipt = service.recall(
        _session(repo_b, "other"), "How do I deploy with make?", budget_tokens=500
    )

    assert item.id not in [candidate.item.id for candidate in receipt.included]


def test_browsing_is_non_mutating_but_committed_recall_is_audited(tmp_path):
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "repo"))
    item, _ = service.remember(
        session, "Parser entry point is mucli.py.", kind="finding"
    )

    before = service.ledger.get(item.id)
    listed = service.list_for_session(session, query="Parser")
    after_browse = service.ledger.get(item.id)
    receipt = service.recall(
        session, "Where is the parser entry point?", budget_tokens=500
    )
    after_recall = service.ledger.get(item.id)

    assert listed[0].id == item.id
    assert after_browse.recall_count == before.recall_count == 0
    assert after_browse.updated_at == before.updated_at
    assert receipt.included
    assert after_recall.recall_count == 1
    assert after_recall.last_recalled_at is not None
    assert any(
        event["type"] == "recalled"
        for event in service.ledger.events(memory_id=item.id)
    )


def test_forget_purges_content_revisions_and_search_index(tmp_path):
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "repo"))
    item, _ = service.remember(session, "Use a blue deployment.", kind="decision")
    item = service.ledger.revise(item.id, {"statement": "Use a green deployment."})
    receipt = service.recall(
        session, "Which deployment should we use?", budget_tokens=500
    )
    assert receipt.included

    forgotten = service.ledger.action(item.id, "forget", actor="user")

    assert forgotten.lifecycle == "forgotten"
    assert forgotten.statement == ""
    assert service.ledger.revisions(item.id) == []
    assert service.list_for_session(session, query="deployment") == []
    assert service.ledger.events(memory_id=item.id)[0]["type"] == "forgotten"
    forgotten_receipt = service.ledger.get_recall(receipt.id)
    # Round-49 F1: receipts are COMPACT — id/version/score/token_cost only,
    # no memory body (statement etc. lives solely in the memories table).
    # The redaction invariant is preserved: the receipt references the
    # forgotten version without copying content.
    copied_memory = forgotten_receipt["included"][0]
    assert copied_memory["id"] == item.id
    assert "statement" not in str(forgotten_receipt["included"])
    assert "Use a green deployment." not in str(forgotten_receipt)


def test_secret_like_memory_is_never_persisted(tmp_path):
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "repo"))

    with pytest.raises(MemoryRejectedError):
        service.remember(session, "api_key=sk-test-secret-value", kind="finding")

    assert service.ledger.stats()["total"] == 0


def test_model_managed_task_entries_promote_without_goal_noise(tmp_path):
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "repo"))
    finding = session.task_memory.save(
        "The GUI router is registered in mu/gui/app.py.",
        tags=["architecture"],
        source="read_file",
        kind="finding",
    )
    goal = session.task_memory.save(
        "Locked session goal: implement memory",
        tags=["goal", "session-goal", "locked"],
        source="session_goal",
        kind="goal",
    )

    captured = service.capture_task_entries(session)

    assert len(captured) == 1
    assert finding.durable_id == captured[0].id
    assert goal.durable_id == ""
    assert service.ledger.stats()["total"] == 1


def test_model_save_memory_commits_without_an_approval_round_trip(tmp_path):
    import mu.tools as memory_tools
    from mu.tools.memory.handlers import manage_durable_memory, save_memory

    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "model-repo"), "model-session")
    session.get_durable_memory_service = lambda: service
    context = SimpleNamespace(session=session)
    assert memory_tools.get("manage_durable_memory") is not None

    result = save_memory(
        {
            "content": "The formatter is configured in pyproject.toml.",
            "kind": "finding",
            "scope": "repository",
            "verification": "source_backed",
            "source": "pyproject.toml",
        },
        context,
    )

    assert "No approval required" in result
    assert len(session.task_memory.entries) == 1
    durable_id = session.task_memory.entries[0].durable_id
    assert service.ledger.get(durable_id).statement.endswith("pyproject.toml.")
    assert [item.id for item in session._turn_durable_writes] == [durable_id]

    managed = manage_durable_memory(
        {
            "memory_id": durable_id[:8],
            "action": "archive",
            "reason": "finding no longer applies",
        },
        context,
    )
    assert "now archived" in managed
    assert service.ledger.get(durable_id).lifecycle == "archived"


def test_exact_duplicate_reinforces_one_record_with_new_revision(tmp_path):
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "repo"))
    first, created = service.remember(session, "Use Ruff for linting.", tags=["lint"])
    second, created_again = service.remember(
        session,
        "Use Ruff for linting.",
        tags=["python"],
        source_refs=[{"type": "file", "path": "pyproject.toml"}],
    )

    assert created is True
    assert created_again is False
    assert first.id == second.id
    assert second.version == 2
    assert second.tags == ["lint", "python"]
    assert len(service.ledger.revisions(first.id)) == 2


def test_supersession_is_audited_on_both_sides(tmp_path):
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "supersede-repo"))
    old, _ = service.remember(session, "Deployments use the blue pool.")
    new, _ = service.remember(
        session,
        "Deployments use the green pool.",
        supersedes_id=old.id,
        reason="deployment policy changed",
    )

    replaced = service.ledger.get(old.id)
    assert replaced.lifecycle == "superseded"
    assert len(service.ledger.revisions(old.id)) == 2
    assert replaced.relations == [{"type": "superseded_by", "target_id": new.id}]
    assert service.ledger.graph(new.id)["edges"] == [
        {"source": new.id, "target": old.id, "type": "supersedes"}
    ]


def test_compact_id_lookup_is_scoped_and_safe_edits_reject_secrets(tmp_path):
    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "repo"))
    item, _ = service.remember(session, "Use the repository formatter.")

    assert service.get_for_session(session, item.id[:8]).id == item.id
    with pytest.raises(MemoryRejectedError):
        service.revise_for_session(
            session, item.id[:8], {"statement": "password=do-not-store-this"}
        )

    unchanged = service.ledger.get(item.id)
    assert unchanged.statement == item.statement


def test_memory_center_contract_present_in_all_clients():
    root = os.path.dirname(os.path.dirname(__file__))
    web = open(
        os.path.join(root, "mu", "gui", "templates", "fragments", "memory_panel.html"),
        encoding="utf-8",
    ).read()
    js = open(
        os.path.join(root, "mu", "gui", "static", "js", "app.js"),
        encoding="utf-8",
    ).read()
    mobile = open(
        os.path.join(root, "mobile", "android", "src", "screens", "MemoryScreen.tsx"),
        encoding="utf-8",
    ).read()

    assert "cross-session memory" in web
    assert "/api/v1/memories" in js
    assert "Why last recall" in mobile
    assert "Context Observatory" in mobile
    assert "const { colors, isDark } = useTheme()" in mobile


def test_tui_memory_center_supports_compact_model_managed_workflow(tmp_path):
    from mu.commands.memory import memory_cmd, remember_cmd

    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    session = _session(_git_repo(tmp_path / "tui-repo"), "tui")
    session.get_durable_memory_service = lambda: service
    session.ui = None

    stored = remember_cmd(
        session,
        "Use make verify --scope repository --kind procedure --pin",
        allow_prompt=False,
    )
    assert stored.ok
    memory = stored.data["memory"]
    assert memory["pinned"] is True
    assert memory["scope"]["type"] == "repository"

    listed = memory_cmd(session, "list durable", allow_prompt=False)
    assert listed.ok and listed.data["memories"][0]["id"] == memory["id"]
    shown = memory_cmd(session, f"show {memory['id'][:8]}", allow_prompt=False)
    assert shown.ok and shown.data["id"] == memory["id"]

    service.recall(session, "How should we verify?", budget_tokens=500)
    why = memory_cmd(session, "why last", allow_prompt=False)
    assert why.ok and why.data["included"]


def test_versioned_memory_api_uses_explicit_session_scope_and_etags(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mu.gui.bus import EventBus
    from mu.gui.routers import memories as memories_router

    service = DurableMemoryService(SQLiteMemoryLedger(tmp_path / "memory.db"))
    repo_a = _git_repo(tmp_path / "api-a")
    repo_b = _git_repo(tmp_path / "api-b")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_b),
            "remote",
            "set-url",
            "origin",
            "git@github.com:Example/Api-Other.git",
        ],
        check=True,
    )
    session_a = _session(repo_a, "api-a")
    session_b = _session(repo_b, "api-b")
    session_a.get_durable_memory_service = lambda: service
    session_b.get_durable_memory_service = lambda: service

    app = FastAPI()
    app.state.sessions = {"api-a": session_a, "api-b": session_b}
    app.state.session_by_name = lambda name=None: app.state.sessions.get(name)
    app.state.bus = EventBus()
    app.include_router(memories_router.router, prefix="/api/v1")
    client = TestClient(app)

    created = client.post(
        "/api/v1/memories?session_name=api-a",
        json={"statement": "Use API memory receipts.", "kind": "decision"},
    )
    assert created.status_code == 200
    memory = created.json()["memory"]

    other = client.post(
        "/api/v1/memories?session_name=api-b",
        json={"statement": "This belongs to another repository."},
    )
    assert other.status_code == 200

    listed = client.get("/api/v1/memories?session_name=api-a").json()["memories"]
    assert [row["id"] for row in listed] == [memory["id"]]

    revised = client.patch(
        f"/api/v1/memories/{memory['id']}?session_name=api-a",
        headers={"If-Match": memory["etag"]},
        json={"changes": {"statement": "Use auditable API memory receipts."}},
    )
    assert revised.status_code == 200
    assert revised.json()["memory"]["version"] == 2

    missing_etag = client.patch(
        f"/api/v1/memories/{memory['id']}?session_name=api-a",
        json={"changes": {"statement": "Blind overwrite."}},
    )
    assert missing_etag.status_code == 428

    stale = client.patch(
        f"/api/v1/memories/{memory['id']}?session_name=api-a",
        headers={"If-Match": memory["etag"]},
        json={"changes": {"statement": "A stale edit must not win."}},
    )
    assert stale.status_code == 409

    events = client.get("/api/v1/memory-events?session_name=api-a").json()["events"]
    assert events
    assert {event["memory_id"] for event in events} == {memory["id"]}
