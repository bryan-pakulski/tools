"""Container sessions expose the same strategy harness and Mode OS surface."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mu.commands.mode import mode_cmd
from mu.container import worker
from mu.container.supervisor import ContainerSupervisor
from mu.gui.container_mode_proxy import proxy_container_mode_request
from mu.gui.memory_snapshot import (
    get_context_timeline,
    ingest_context_timeline_point,
)
from mu.gui.routers import modes as modes_router
from mu.gui.routers import containers as containers_router
from mu.session.manager import SessionManager


class _FolderContext:
    folders: list[str] = []


class _Manager:
    current_session_name = "container-demo"
    folder_context = _FolderContext()

    def save_history(self, *_args, **_kwargs):
        return None


class _ContainerSession:
    def __init__(self, mode: str = "default") -> None:
        self.variables = {"session_type": "container", "agent_mode": mode}
        self.session_manager = _Manager()
        self.folder_context = self.session_manager.folder_context
        self.container_ref = SimpleNamespace(name="mucli-demo")
        self.provider = SimpleNamespace(name="openai", model_name="test-model")
        self.system_instruction = "system"


def _modes_app(session):
    app = FastAPI()
    app.state.sessions = {"container-demo": session}
    app.state.current_session_name = "container-demo"
    app.state.session_locks = {"container-demo": threading.Lock()}
    app.state.session_by_name = lambda name=None: app.state.sessions.get(
        name or app.state.current_session_name
    )
    app.state.session_lock_for = lambda name=None: app.state.session_locks[
        name or app.state.current_session_name
    ]
    class Bus:
        async def publish(self, _event):
            return None

    app.state.bus = Bus()
    app.include_router(modes_router.router, prefix="/api/modes")
    return app


def test_container_is_a_valid_strategy_execution_workspace():
    session = _ContainerSession()
    with TestClient(_modes_app(session)) as client:
        payload = client.get("/api/modes").json()
        response = client.post("/api/modes/feature")

    feature = next(mode for mode in payload["modes"] if mode["name"] == "feature")
    assert payload["has_workspace"] is False
    assert payload["has_execution_workspace"] is True
    assert payload["session_type"] == "container"
    assert payload["execution_boundary"] == "container"
    assert feature["disabled"] is False
    assert response.status_code == 200
    assert session.variables["agent_mode"] == "feature"


def test_mode_selection_targets_named_container_not_focused_session():
    focused = _ContainerSession()
    focused.session_manager.current_session_name = "focused"
    focused.variables["session_type"] = "chat"
    focused.container_ref = None
    container = _ContainerSession()
    app = _modes_app(container)
    app.state.sessions = {"focused": focused, "container-demo": container}
    app.state.current_session_name = "focused"
    app.state.session_locks["focused"] = threading.Lock()

    with TestClient(app) as client:
        listing = client.get(
            "/api/modes", params={"session_name": "container-demo"}
        )
        selected = client.post(
            "/api/modes/feature", params={"session_name": "container-demo"}
        )

    feature = next(
        mode for mode in listing.json()["modes"] if mode["name"] == "feature"
    )
    assert feature["disabled"] is False
    assert selected.status_code == 200
    assert container.variables["agent_mode"] == "feature"
    assert focused.variables["agent_mode"] == "default"


def test_tui_mode_command_accepts_container_without_host_folder():
    session = _ContainerSession()
    session.ui = None
    result = mode_cmd(session, "feature", allow_prompt=False)
    assert result.ok is True
    assert session.variables["agent_mode"] == "feature"


def test_cached_worker_session_synchronizes_agent_mode():
    class UI:
        variables = {}

        def set_variables(self, values):
            self.variables = dict(values)

    session = SimpleNamespace(
        variables={"agent_mode": "default"},
        provider=SimpleNamespace(model_name="old"),
        system_instruction="old system",
        disabled_tools=["feature_plan"],
        ui=UI(),
    )
    request = worker.SendRequest(
        session_name="demo",
        text="build it",
        provider="openai",
        model="new-model",
        agent_mode="feature",
        system_instruction="new system",
    )

    worker._sync_request_context(session, request)

    assert session.variables["agent_mode"] == "feature"
    assert session.variables["session_type"] == "container"
    assert session.variables["lazy_tools_enabled"] is True
    assert session.provider.model_name == "new-model"
    assert session.system_instruction == "new system"
    assert session.disabled_tools == []
    assert session.ui.variables["agent_mode"] == "feature"


def test_container_binding_persists_lazy_mode_filtering(monkeypatch, tmp_path):
    monkeypatch.setattr(containers_router._config, "HISTORY_DIR", str(tmp_path))
    session_dir = tmp_path / "sessions" / "container-demo"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "session.json"
    session_path.write_text(
        json.dumps({"variables": {"agent_mode": "default"}}),
        encoding="utf-8",
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(sessions={})))

    containers_router._persist_session_container_binding(
        request,
        "container-demo",
        {"container_name": "mucli-demo"},
    )

    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert saved["variables"]["session_type"] == "container"
    assert saved["variables"]["lazy_tools_enabled"] is True


def test_worker_mode_router_is_authenticated_and_uses_worker_state(monkeypatch):
    monkeypatch.setenv("MUCLI_WORKER_TOKEN", "worker-token")
    manager = SessionManager(session_name="mode-worker")
    manager.variables.update({"agent_mode": "debug", "session_type": "container"})
    session = SimpleNamespace(session_manager=manager, variables=manager.variables)
    worker._sessions["mode-worker"] = session
    worker._locks["mode-worker"] = threading.Lock()
    worker._busy["mode-worker"] = threading.Event()
    try:
        with TestClient(worker.app) as client:
            denied = client.get(
                "/api/debug/state", params={"session_name": "mode-worker"}
            )
            response = client.get(
                "/api/debug/state",
                params={"session_name": "mode-worker"},
                headers={"X-MuCLI-Worker-Token": "worker-token"},
            )
            manager.variables["agent_mode"] = "feature"
            feature = client.get(
                "/api/feature/state",
                params={"session_name": "mode-worker"},
                headers={"X-MuCLI-Worker-Token": "worker-token"},
            )
    finally:
        worker._sessions.pop("mode-worker", None)
        worker._locks.pop("mode-worker", None)
        worker._busy.pop("mode-worker", None)

    assert denied.status_code == 401
    assert response.status_code == 200
    assert response.json()["workspace"]["mode"] == "debug"
    assert response.json()["workspace"]["status"]["label"] == "investigating"
    assert feature.status_code == 200
    assert feature.json()["workspace"]["mode"] == "feature"


def test_supervisor_mode_proxy_syncs_runtime_then_forwards(monkeypatch):
    ref = SimpleNamespace(worker_token="token")
    supervisor = ContainerSupervisor(registry=SimpleNamespace())
    monkeypatch.setattr(supervisor, "container_for_session", lambda _name: ref)
    monkeypatch.setattr(supervisor, "ensure_running", lambda value: value)
    monkeypatch.setattr(supervisor, "worker_url", lambda _ref: "http://worker:30312")
    calls = []

    class Response:
        status_code = 200
        content = b'{"workspace":{"mode":"feature"}}'
        headers = {"content-type": "application/json"}

        def json(self):
            return {"ok": True}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs))
            return Response()

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return Response()

    monkeypatch.setattr("mu.container.supervisor.httpx.Client", Client)

    result = supervisor.proxy_mode_api(
        "demo",
        "/api/feature/state",
        query=[("session_name", "demo")],
        provider="openai",
        model="test",
        agent_mode="feature",
    )

    assert calls[0][1].endswith("/runtime/sync")
    assert calls[0][2]["json"]["agent_mode"] == "feature"
    assert calls[1][1].endswith("/api/feature/state")
    assert result["status_code"] == 200
    assert json.loads(result["content"])["workspace"]["mode"] == "feature"


def test_host_proxy_uses_container_boundary_and_preserves_workspace_sessions():
    container = _ContainerSession("feature")
    workspace = _ContainerSession("feature")
    workspace.variables["session_type"] = "workspace"
    workspace.container_ref = None
    calls = []

    class Supervisor:
        def proxy_mode_api(self, session_name, path, **kwargs):
            calls.append((session_name, path, kwargs))
            return {
                "status_code": 200,
                "content": b'{"workspace":{"mode":"feature"}}',
                "content_type": "application/json",
            }

    app = FastAPI()
    app.middleware("http")(proxy_container_mode_request)
    app.state.sessions = {"container-demo": container, "host-demo": workspace}
    app.state.current_session_name = "container-demo"
    app.state.session_by_name = lambda name=None: app.state.sessions.get(
        name or app.state.current_session_name
    )
    app.state.container_supervisor = Supervisor()

    @app.get("/api/feature/state")
    async def host_state():
        return {"source": "host"}

    with TestClient(app) as client:
        proxied = client.get(
            "/api/feature/state", params={"session_name": "container-demo"}
        )
        local = client.get("/api/feature/state", params={"session_name": "host-demo"})

    assert proxied.headers["x-mucli-execution-boundary"] == "container"
    assert proxied.json()["workspace"]["mode"] == "feature"
    assert calls[0][0:2] == ("container-demo", "/api/feature/state")
    assert local.json() == {"source": "host"}


def test_container_context_points_survive_worker_counter_restart():
    class Session:
        pass

    session = Session()
    point = {
        "id": 1,
        "at": 1.0,
        "total_tokens": 100,
        "context_limit": 1000,
        "free_tokens": 900,
        "fill_pct": 10.0,
        "layers": [],
    }
    ingest_context_timeline_point(session, point)
    restarted = ingest_context_timeline_point(
        session, {**point, "at": 2.0, "total_tokens": 140, "total_delta": 40}
    )
    timeline = get_context_timeline(session)

    assert restarted["id"] == 2
    assert [item["total_tokens"] for item in timeline["points"]] == [100, 140]
