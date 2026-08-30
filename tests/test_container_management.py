from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from mu.commands import get
from mu.container import ContainerSupervisor
from mu.container.docker_cli import CommandRunner
from mu.container.ref import ContainerRef
from mu.container.registry import ContainerRegistry
from mu.container.templates import ContainerTemplate, TemplateRegistry
from mu.gui.routers import containers


def _ref(name: str = "mucli-demo") -> ContainerRef:
    return ContainerRef(
        container_id="abc",
        name=name,
        image="mucli/demo:test",
        dockerfile_hash="hash",
        network_name=f"{name}-net",
        worker_token="secret",
        root_volume=f"{name}-home",
        workspace_volume=f"{name}-workspace",
        status="running",
        standalone=True,
    )


def test_template_registry_round_trip(tmp_path):
    registry = TemplateRegistry(str(tmp_path))
    item = ContainerTemplate(
        name="python-tools",
        image="mucli/template-python-tools:1",
        source_container="mucli-demo",
        description="Configured tools",
    )
    registry.upsert(item)
    loaded = TemplateRegistry(str(tmp_path)).get("python-tools")
    assert loaded is not None
    assert loaded.image == item.image
    assert registry.list_templates()[0].description == "Configured tools"
    assert registry.remove("python-tools") is True


def test_snapshot_commits_and_registers_template(tmp_path):
    runner = CommandRunner(dry_run=True)
    container_registry = ContainerRegistry(str(tmp_path / "containers"))
    template_registry = TemplateRegistry(str(tmp_path / "templates"))
    container_registry.upsert(_ref())
    supervisor = ContainerSupervisor(
        registry=container_registry,
        template_registry=template_registry,
        runner=runner,
    )

    item = supervisor.snapshot("demo", "python-tools", description="Configured tools")

    assert item.name == "python-tools"
    assert item.source_container == "mucli-demo"
    assert template_registry.get("python-tools") is not None
    commit = next(command for command in runner.commands if command[1] == "commit")
    # Round-18 F31 + round-20 F45: the ledger stores the REDACTED argv as
    # proper tokens — ENV keys must be present, secret values must never
    # be. Substring match: a rendered token may be `ENV KEY=<redacted>`.
    assert any("OPENAI_API_KEY=<redacted>" in part for part in commit)
    assert any("MUCLI_WORKER_TOKEN=<redacted>" in part for part in commit)
    assert not any("OPENAI_API_KEY=" in part and "<redacted>" not in part for part in commit)


def test_create_standalone_environment_from_template_refreshes_worker_layer(tmp_path):
    runner = CommandRunner(dry_run=True)
    container_registry = ContainerRegistry(str(tmp_path / "containers"))
    template_registry = TemplateRegistry(str(tmp_path / "templates"))
    template_registry.upsert(
        ContainerTemplate(
            name="python-tools",
            image="mucli/template-python-tools:1",
            source_container="mucli-source",
            egress_allow=["example.com"],
        )
    )
    supervisor = ContainerSupervisor(
        registry=container_registry,
        template_registry=template_registry,
        runner=runner,
    )

    ref = supervisor.create_environment(
        container_name="standalone",
        template_name="python-tools",
        source_path=str(Path(__file__).resolve().parents[1]),
    )

    assert ref.standalone is True
    assert ref.template_name == "python-tools"
    assert ref.image.startswith("mucli/standalone:")
    assert any(command[1] == "build" for command in runner.commands)
    assert any(command[1:3] == ["image", "inspect"] for command in runner.commands)


def test_shutdown_keeps_standalone_environment_running(tmp_path):
    runner = CommandRunner(dry_run=True)
    registry = ContainerRegistry(str(tmp_path / "containers"))
    registry.upsert(_ref())
    supervisor = ContainerSupervisor(
        registry=registry,
        template_registry=TemplateRegistry(str(tmp_path / "templates")),
        runner=runner,
    )

    supervisor.shutdown()

    assert not any(len(command) > 1 and command[1] == "stop" for command in runner.commands)


def test_container_and_template_commands_are_registered():
    assert get("/container") is not None
    assert get("/template") is not None
    assert get("/templates") is not None


def test_environment_creation_job_returns_incremental_logs():
    class FakeRef:
        def to_dict(self, *, include_secret=True):
            return {"name": "mucli-demo", "status": "running"}

    def create_environment(**kwargs):
        kwargs["progress"]("building_image", "Building image")
        kwargs["output"]("stdout", "step one")
        return FakeRef()

    state = SimpleNamespace(
        port=30311,
        container_supervisor=SimpleNamespace(create_environment=create_environment),
        container_environment_jobs={},
        container_environment_tasks={},
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    asyncio.run(
        containers._run_environment_creation(
            request,
            "job-one",
            {"name": "demo", "dockerfile": "FROM ubuntu:24.04"},
        )
    )

    job = state.container_environment_jobs["job-one"]
    assert job["state"] == "ready"
    assert job["container"]["name"] == "mucli-demo"
    assert job["logs"][0]["text"] == "step one"


def test_container_manager_page_has_shell_and_templates():
    root = Path(__file__).resolve().parents[1]
    markup = (root / "mu/gui/templates/containers.html").read_text()
    script = (root / "mu/gui/static/js/containers.js").read_text()
    assert "Container environments" in markup
    assert "Templates" in markup
    assert "shell-modal" in markup
    assert "/api/containers/" in script
    assert "snapshot" in script
