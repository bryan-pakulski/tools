from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
from fastapi import HTTPException

from mu.container.builder import build_create_command
from mu.container.docker_cli import ContainerRuntimeError
from mu.container.ref import (
    DEFAULT_WORKER_PORT,
    WORKER_PROTOCOL_VERSION,
    ContainerRef,
)
from mu.container.registry import ContainerRegistry
from mu.container.supervisor import ContainerSupervisor
from mu.container import worker
from mu.container.tui import send_tui_container_message

ROOT = Path(__file__).resolve().parents[1]


def make_ref(name: str, *, port: int = DEFAULT_WORKER_PORT) -> ContainerRef:
    return ContainerRef(
        container_id="abc",
        name=name,
        image="mucli/test:latest",
        dockerfile_hash="hash",
        network_name=f"{name}-net",
        proxy_name=f"{name}-proxy",
        proxy_ip="172.31.0.2",
        worker_token="secret",
        worker_port=port,
        worker_protocol=WORKER_PROTOCOL_VERSION,
        attached_sessions=["demo"],
    )


def test_worker_ports_allocate_from_30312(tmp_path):
    registry = ContainerRegistry(str(tmp_path))
    registry.upsert(make_ref("mucli-one", port=30312))
    registry.upsert(make_ref("mucli-three", port=30314))
    assert registry.allocate_worker_port() == 30313


def test_legacy_registry_record_is_marked_for_worker_migration():
    payload = make_ref("mucli-old", port=9090).to_dict()
    payload.pop("worker_protocol", None)
    loaded = ContainerRef.from_dict(payload)
    assert loaded.worker_protocol == 0
    assert loaded.worker_port == 9090


def test_create_command_exports_selected_worker_port():
    ref = make_ref("mucli-port", port=30317)
    command = build_create_command(ref)
    assert "MUCLI_WORKER_PORT=30317" in command


def test_worker_build_marks_provider_as_prevalidated(monkeypatch):
    captured = {}

    class Manager:
        history = []
        folder_context = SimpleNamespace(folders=[])

        def save_history(self, *_args, **_kwargs):
            return None

    class Session:
        variables = {}
        session_manager = Manager()
        folder_context = session_manager.folder_context

        def sync_runtime_state(self):
            return None

    def fake_build_session(args, ui, allow_prompt=True):
        captured["args"] = args
        captured["allow_prompt"] = allow_prompt
        captured["ui"] = ui
        return Session()

    worker._sessions.clear()
    worker._locks.clear()
    worker._busy.clear()
    fake_mucli = ModuleType("mucli")
    fake_mucli.build_session = fake_build_session
    monkeypatch.setitem(sys.modules, "mucli", fake_mucli)

    request = worker.SendRequest(
        session_name="demo",
        text="hello",
        provider="ollama",
        model="glm-5.2",
    )
    session = worker._build_session(request)

    assert session is not None
    assert captured["allow_prompt"] is False
    assert captured["args"].provider_prevalidated is True
    assert captured["args"].session_type == "container"


def test_worker_initialization_error_is_returned_as_http_detail(monkeypatch):
    monkeypatch.setenv("MUCLI_WORKER_TOKEN", "token")
    monkeypatch.setattr(
        worker,
        "_build_session",
        lambda _request: (_ for _ in ()).throw(RuntimeError("broken provider")),
    )
    request = worker.SendRequest(
        session_name="demo",
        text="hello",
        provider="ollama",
        model="glm-5.2",
    )
    with pytest.raises(HTTPException) as exc:
        worker.send_sync(request, "token")
    assert exc.value.status_code == 500
    assert "worker session initialisation failed" in str(exc.value.detail)
    assert "broken provider" in str(exc.value.detail)


def test_supervisor_worker_error_includes_response_and_log_tail(monkeypatch):
    ref = make_ref("mucli-demo")
    registry = SimpleNamespace(list_containers=lambda: [ref])
    supervisor = ContainerSupervisor(registry=registry)
    monkeypatch.setattr(supervisor, "ensure_running", lambda _ref: _ref)
    monkeypatch.setattr(supervisor, "worker_url", lambda _ref: "http://172.20.0.3:30312")
    monkeypatch.setattr(supervisor, "_worker_log_tail", lambda _ref: "traceback line")

    captured = {}

    class Response:
        status_code = 500
        text = '{"detail":"provider unavailable"}'
        reason_phrase = "Internal Server Error"

        def json(self):
            return {"detail": "provider unavailable"}

    class Client:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("mu.container.supervisor.httpx.Client", Client)

    with pytest.raises(ContainerRuntimeError) as exc:
        supervisor.send_sync(
            "demo",
            "hello",
            provider="ollama",
            model="glm-5.2",
        )
    assert captured["trust_env"] is False
    assert "provider unavailable" in str(exc.value)
    assert "traceback line" in str(exc.value)


def test_supervisor_forwards_attachment_ids_to_container_worker(monkeypatch):
    ref = make_ref("mucli-demo")
    supervisor = ContainerSupervisor(
        registry=SimpleNamespace(list_containers=lambda: [ref])
    )
    monkeypatch.setattr(supervisor, "ensure_running", lambda _ref: _ref)
    monkeypatch.setattr(
        supervisor, "worker_url", lambda _ref: "http://172.20.0.3:30312"
    )
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"ok": True}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **kwargs):
            captured.update(kwargs.get("json") or {})
            return Response()

    monkeypatch.setattr("mu.container.supervisor.httpx.Client", Client)
    supervisor.send_sync(
        "demo",
        "inspect this",
        provider="ollama",
        model="glm-5.3-flash:cloud",
        attachment_ids=["attachment-a"],
    )
    assert captured["attachment_ids"] == ["attachment-a"]


def test_tui_worker_error_does_not_crash(monkeypatch):
    errors = []

    class Supervisor:
        def send_sync(self, *_args, **_kwargs):
            raise ContainerRuntimeError("worker failed")

    session = SimpleNamespace(
        _container_supervisor=Supervisor(),
        session_manager=SimpleNamespace(current_session_name="demo"),
        provider=SimpleNamespace(name="ollama", model_name="glm-5.2"),
        variables={"agent_mode": "default"},
        system_instruction="system",
        ui=SimpleNamespace(show_error=lambda message: errors.append(message)),
    )
    monkeypatch.setattr("mu.container.tui.ensure_tui_container", lambda _session: None)
    result = send_tui_container_message(session, "hello")
    assert result["status"] == "error"
    assert errors == ["worker failed"]


def test_gui_container_turn_uses_background_sync_bridge():
    source = (ROOT / "mu/gui/routers/chat.py").read_text(encoding="utf-8")
    assert "container_supervisor.send_sync" in source
    assert "asyncio.create_task(_drive_container())" in source
    assert '"kind": "history_refresh"' in source


def test_web_artifact_panel_is_visible_when_empty():
    source = (ROOT / "mu/gui/templates/fragments/artifacts_panel.html").read_text(encoding="utf-8")
    assert 'data-mode="artifacts"' in source
    assert "No artifacts yet" in source
    assert "$store.artifacts.current.length" in source


def test_default_worker_image_uses_30312():
    dockerfile = (ROOT / "mu/container/Dockerfile.mucli").read_text(encoding="utf-8")
    assert "EXPOSE 30312" in dockerfile
    worker_source = (ROOT / "mu/container/worker.py").read_text(encoding="utf-8")
    assert 'os.getenv("MUCLI_WORKER_PORT", "30312")' in worker_source


def test_container_manager_displays_worker_port():
    source = (ROOT / "mu/gui/static/js/containers.js").read_text(encoding="utf-8")
    assert "ref.worker_port" in source


def test_prevalidated_provider_path_avoids_model_discovery():
    source = (ROOT / "mucli.py").read_text(encoding="utf-8")
    assert 'getattr(args, "provider_prevalidated", False)' in source
    branch = source.split('getattr(args, "provider_prevalidated", False)', 1)[1][:900]
    assert "init_provider(" in branch


def test_worker_runtime_uses_proxy_ip_and_offline_tiktoken_cache():
    ref = make_ref("mucli-proxy")
    command = build_create_command(ref)
    assert "HTTP_PROXY=http://172.31.0.2:3128" in command
    assert "MUCLI_PROXY_URL=http://172.31.0.2:3128" in command
    dockerfile = (ROOT / "mu/container/Dockerfile.mucli").read_text(encoding="utf-8")
    assert "MUCLI_TIKTOKEN_PREWARM_V1" in dockerfile
    assert "tiktoken.get_encoding('cl100k_base')" in dockerfile
    assert "TIKTOKEN_CACHE_DIR=/opt/mucli/.cache/tiktoken" in dockerfile


def test_gui_container_failure_emits_terminal_event():
    source = (ROOT / "mu/gui/routers/chat.py").read_text(encoding="utf-8")
    failure_branch = source.split('error_text = f"container send failed:', 1)[1][:1400]
    assert '"kind": "error"' in failure_branch
    assert '"kind": "turn_complete"' in failure_branch
    assert '"kind": "history_refresh"' in failure_branch


def test_current_protocol_forces_proxy_ip_migration():
    assert WORKER_PROTOCOL_VERSION >= 3
    legacy = make_ref("mucli-legacy")
    legacy.proxy_ip = ""
    legacy.worker_protocol = WORKER_PROTOCOL_VERSION - 1
    assert legacy.worker_protocol < WORKER_PROTOCOL_VERSION


def test_web_and_mobile_render_terminal_container_errors():
    web = (ROOT / "mu/gui/static/js/app.js").read_text(encoding="utf-8")
    mobile = (ROOT / "mobile/android/src/hooks/useChatSession.ts").read_text(encoding="utf-8")
    assert 'ev.result.status === "error"' in web
    assert "result?.status === 'error'" in mobile
    assert "setError(String(result.error))" in mobile


def test_current_worker_protocol_rebuilds_broken_proxy_entrypoint():
    from mu.container.ref import WORKER_PROTOCOL_VERSION

    assert WORKER_PROTOCOL_VERSION >= 4
