"""Interactive terminal setup and synchronous bridge for container sessions.

The browser/mobile GUI receives worker events through the host FastAPI bridge.
The TUI has no callback server, so it creates/attaches the worker with an empty
supervisor URL and uses the worker's synchronous turn endpoint.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Prompt

from mu.ui.choice_picker import prompt_choice, prompt_confirm

from .builder import default_dockerfile
from .docker_cli import ContainerRuntimeError
from .load_errors import describe_container_load_error, format_container_load_error
from .network import DEFAULT_EGRESS_ALLOW
from .supervisor import ContainerSupervisor

console = Console()


def _domain_list(value: Any, fallback: list[str] | None = None) -> list[str]:
    if isinstance(value, str):
        raw = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = fallback or []
    return list(
        dict.fromkeys(
            str(item or "").strip().lower()
            for item in raw
            if str(item or "").strip()
        )
    )


def _prompt_domain_list(label: str, current: list[str]) -> list[str]:
    value = Prompt.ask(
        label,
        default=", ".join(current),
        show_default=bool(current),
    )
    return _domain_list(value)


def _edit_dockerfile(initial: str) -> str:
    """Open the template in $VISUAL/$EDITOR, or accept a file path fallback."""
    editor = str(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "").strip()
    if editor:
        fd, path = tempfile.mkstemp(prefix="mucli-Dockerfile-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(initial)
            command = [*shlex.split(editor), path]
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                console.print(
                    f"[yellow]editor exited with status {result.returncode}; keeping the previous template[/yellow]"
                )
                return initial
            return Path(path).read_text(encoding="utf-8")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    console.print(
        "[dim]Set $EDITOR or $VISUAL to edit inline. You can also load a Dockerfile from disk.[/dim]"
    )
    source = Prompt.ask("Dockerfile path (blank keeps template)", default="").strip()
    if not source:
        return initial
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        console.print(f"[yellow]not a file: {path}; keeping the template[/yellow]")
        return initial
    return path.read_text(encoding="utf-8")


def _browse_host_folder(label: str = "Host folder") -> str:
    """Prompt with directory completion and require an existing host folder."""
    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.completion import PathCompleter

        value = pt_prompt(
            f"{label}> ",
            completer=PathCompleter(expanduser=True, only_directories=True),
            default="",
        ).strip()
    except (ImportError, EOFError, KeyboardInterrupt):
        value = Prompt.ask(label, default="").strip()
    if not value:
        return ""
    path = str(Path(value).expanduser().resolve())
    if not Path(path).is_dir():
        console.print(f"[yellow]not a directory: {path}[/yellow]")
        return ""
    return path


def _prompt_mounts(current: list[dict[str, str]]) -> list[dict[str, str]]:
    mounts = [dict(item) for item in current]
    if mounts:
        console.print("[dim]Current bind mounts:[/dim]")
        for item in mounts:
            console.print(
                f"  {item.get('host_path')} → {item.get('container_path')} ({item.get('mode', 'rw')})",
                markup=False,
            )
        action = prompt_choice(
            "Bind mounts",
            [
                ("keep", "Keep mounts", "Leave the current bind mounts unchanged"),
                ("add", "Add mount", "Select another host folder"),
                ("clear", "Clear mounts", "Remove all configured bind mounts"),
            ],
            default="keep",
        )
        if action == "keep":
            return mounts
        if action == "clear":
            mounts = []
    elif not prompt_confirm("Add a host bind mount?", default=False):
        return mounts

    while True:
        host_path = _browse_host_folder("Host folder (blank finishes)")
        if not host_path:
            break
        container_path = Prompt.ask(
            "Container path",
            default=f"/workspace/{Path(host_path).name or 'project'}",
        ).strip()
        mode = prompt_choice(
            "Mount mode",
            [
                ("rw", "Read and write", "The container can modify the selected folder"),
                ("ro", "Read only", "The selected folder cannot be modified"),
            ],
            default="rw",
        )
        mounts.append(
            {
                "host_path": host_path,
                "container_path": container_path,
                "mode": mode,
            }
        )
        if not prompt_confirm("Add another bind mount?", default=False):
            break
    return mounts


# MUCLI_CONTAINER_HARDWARE_V1
def _prompt_hardware(
    supervisor: ContainerSupervisor,
    current: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Interactive hardware configuration shared by session and host managers."""
    current = dict(current or {})
    capability = supervisor.hardware_capabilities()
    gpu = dict(capability.get("gpu") or {})
    existing_gpu = str(current.get("gpu_request") or "")
    console.print("\n[bold cyan]Hardware[/bold cyan]")
    console.print(f"[dim]GPU → {gpu.get('reason') or 'not detected'}[/dim]")

    gpu_options = [("none", "No GPU", "Do not attach host GPUs")]
    if gpu.get("supported"):
        gpu_options.extend([
            ("all", "All GPUs", "Expose every detected NVIDIA GPU"),
            ("selected", "Selected GPUs", "Choose indexes or UUIDs"),
        ])
    default_gpu = "all" if existing_gpu == "all" and gpu.get("supported") else (
        "selected" if existing_gpu and gpu.get("supported") else "none"
    )
    gpu_mode = prompt_choice("GPU access", gpu_options, default=default_gpu)
    if gpu_mode == "all":
        gpu_request = "all"
    elif gpu_mode == "selected":
        inventory = gpu.get("devices") or []
        for item in inventory:
            console.print(
                f"  {item.get('index')} · {item.get('name')} · {item.get('uuid')}",
                markup=False,
            )
        previous = existing_gpu.replace("device=", "")
        selected = Prompt.ask("GPU indexes or UUIDs (comma-separated)", default=previous).strip()
        gpu_request = f"device={selected}" if selected else ""
    else:
        gpu_request = ""

    devices = [dict(item) for item in (current.get("devices") or [])]
    if devices:
        console.print("[dim]Current device mappings:[/dim]")
        for item in devices:
            console.print(
                f"  {item.get('host_path')} → {item.get('container_path')} "
                f"({item.get('permissions', 'rwm')})",
                markup=False,
            )
        action = prompt_choice(
            "Device mappings",
            [
                ("keep", "Keep devices", "Leave current device mappings unchanged"),
                ("add", "Add device", "Attach another host device"),
                ("clear", "Clear devices", "Remove all device mappings"),
            ],
            default="keep",
        )
        if action == "keep":
            return gpu_request, devices
        if action == "clear":
            devices = []
            if not prompt_confirm("Add a device now?", default=False):
                return gpu_request, devices
    elif not prompt_confirm("Attach a host device?", default=False):
        return gpu_request, devices

    discovered = capability.get("devices") or []
    while True:
        choices = [
            (
                str(item.get("host_path")),
                str(item.get("name") or item.get("host_path")),
                str(item.get("kind") or "device"),
            )
            for item in discovered
        ]
        choices.extend([
            ("__custom__", "Custom path", "Enter another /dev path"),
            ("__done__", "Done", "Finish hardware configuration"),
        ])
        selected = prompt_choice("Host device", choices, default="__done__")
        if selected == "__done__":
            break
        host_path = (
            Prompt.ask("Host device path", default="/dev/video0").strip()
            if selected == "__custom__"
            else selected
        )
        if not host_path:
            continue
        container_path = Prompt.ask("Container device path", default=host_path).strip()
        permissions = prompt_choice(
            "Device permissions",
            [
                ("r", "Read", "Read only"),
                ("rw", "Read/write", "Read and write"),
                ("rwm", "Read/write/mknod", "Full Docker device permissions"),
            ],
            default="rwm",
        )
        devices.append({
            "host_path": host_path,
            "container_path": container_path or host_path,
            "permissions": permissions,
        })
        if not prompt_confirm("Attach another device?", default=False):
            break
    return gpu_request, devices


def _select_existing_container(supervisor: ContainerSupervisor):
    refs = supervisor.registry.list_containers()
    if not refs:
        return None
    console.print("\n[bold cyan]Managed containers:[/bold cyan]")
    for index, ref in enumerate(refs, 1):
        sessions = ", ".join(ref.attached_sessions) or "none"
        console.print(
            f" {index}. {ref.name}  [{ref.status}]  sessions: {sessions}",
            markup=False,
        )
    selected = prompt_choice(
        "Select a container",
        [
            (ref.name, ref.name, f"{ref.status} · sessions: {', '.join(ref.attached_sessions) or 'none'}")
            for ref in refs
        ],
        default=refs[0].name,
    )
    return next(ref for ref in refs if ref.name == selected)


def configure_tui_container(
    session: Any,
    *,
    supervisor: ContainerSupervisor | None = None,
) -> dict[str, Any]:
    """Prompt through container selection and editable creation stages."""
    manager = session.session_manager
    name = manager.current_session_name
    config = dict(getattr(manager, "container_config", {}) or {})
    supervisor = supervisor or getattr(session, "_container_supervisor", None) or ContainerSupervisor()
    session._container_supervisor = supervisor

    refs = supervisor.registry.list_containers()
    template_registry = getattr(supervisor, "template_registry", None)
    templates = template_registry.list_templates() if template_registry else []
    console.print("\n[bold cyan]Container setup[/bold cyan]")
    choices = ["new"]
    if refs:
        choices.append("existing")
    if templates:
        choices.append("template")
    source_options = [("new", "Create new", "Build a fresh managed environment")]
    if "existing" in choices:
        source_options.append(("existing", "Attach existing", "Connect this session to a managed environment"))
    if "template" in choices:
        source_options.append(("template", "Use template", "Create from a saved container template"))
    source = prompt_choice("Container source", source_options, default=choices[0])

    if source == "template":
        names = [item.name for item in templates]
        console.print("Templates: " + ", ".join(names), markup=False)
        selected = prompt_choice(
            "Template",
            [(item_name, item_name) for item_name in names],
            default=names[0],
        )
        template = template_registry.get(selected) if template_registry else None
        default_name = str(config.get("container_name") or f"mucli-{name}").strip()
        config = {
            "container_name": Prompt.ask("Container name", default=default_name).strip(),
            "dockerfile": None,
            "template_name": template.name if template else selected,
            "mounts": _prompt_mounts([]),
            "egress_allow": list(template.egress_allow if template else DEFAULT_EGRESS_ALLOW),
            "egress_deny": list(template.egress_deny if template else []),
        }
        config["gpu_request"], config["devices"] = _prompt_hardware(supervisor, config)

    if source == "existing":
        ref = _select_existing_container(supervisor)
        if ref is None:
            source = "new"
        else:
            config = {
                "container_name": ref.name,
                "dockerfile": None,
                "template_name": ref.template_name or None,
                "mounts": [mount.to_dict() for mount in ref.mounts],
                "gpu_request": ref.gpu_request,
                "devices": [item.to_dict() for item in ref.devices],
                "egress_allow": list(ref.egress_allow),
                "egress_deny": list(ref.egress_deny),
            }
            console.print(f"[dim]selected → {ref.name}[/dim]", markup=False)

    if source == "new":
        default_name = str(config.get("container_name") or f"mucli-{name}").strip()
        config["container_name"] = Prompt.ask("Container name", default=default_name).strip()

        dockerfile = str(config.get("dockerfile") or default_dockerfile())
        if prompt_confirm("Edit the Dockerfile template?", default=False):
            dockerfile = _edit_dockerfile(dockerfile)
        config["dockerfile"] = dockerfile

        config["mounts"] = _prompt_mounts(list(config.get("mounts") or []))
        config["gpu_request"], config["devices"] = _prompt_hardware(supervisor, config)

        allow = _domain_list(config.get("egress_allow"), DEFAULT_EGRESS_ALLOW)
        console.print(f"[dim]egress allowlist → {', '.join(allow) or 'empty'}[/dim]")
        if prompt_confirm("Edit the egress allowlist?", default=False):
            allow = _prompt_domain_list("Allowed domains (comma-separated)", allow)
        config["egress_allow"] = allow

        deny = _domain_list(config.get("egress_deny"), [])
        console.print(f"[dim]egress blocklist → {', '.join(deny) or 'empty'}[/dim]")
        if prompt_confirm("Edit the egress blocklist?", default=False):
            deny = _prompt_domain_list("Blocked domains (comma-separated)", deny)
        config["egress_deny"] = deny

    manager.container_config = config
    manager.save_history(session.folder_context)
    return config


def _show_environment_table(supervisor: ContainerSupervisor):
    refs = supervisor.list_environments()
    if not refs:
        console.print("[dim]No managed container environments.[/dim]")
        return []
    console.print("\n[bold cyan]Managed environments[/bold cyan]")
    for index, ref in enumerate(refs, 1):
        attached = ", ".join(ref.attached_sessions) or "none"
        console.print(
            f" {index}. {ref.name}  [{ref.status}]  template={ref.template_name or 'custom'}  sessions={attached}",
            markup=False,
        )
    return refs


def _select_environment(supervisor: ContainerSupervisor):
    refs = _show_environment_table(supervisor)
    if not refs:
        return None
    environment_options = [
        (
            ref.name,
            ref.name,
            f"{ref.status} · template: {ref.template_name or 'custom'}",
        )
        for ref in refs
    ]
    environment_options.append(
        ("__back__", "Back", "Return to container management")
    )
    selected = prompt_choice(
        "Select environment",
        environment_options,
        default=refs[0].name,
    )
    if selected == "__back__":
        return None
    return next(ref for ref in refs if ref.name == selected)


def _prompt_environment_config(
    supervisor: ContainerSupervisor,
    *,
    config: dict[str, Any] | None = None,
    default_name: str = "",
) -> dict[str, Any]:
    current = dict(config or {})
    templates = supervisor.template_registry.list_templates()
    source_choices = ["dockerfile"] + (["template"] if templates else [])
    source_default = "template" if current.get("template_name") and templates else "dockerfile"
    source_options = [("dockerfile", "Dockerfile", "Build from the editable MuCLI worker template")]
    if "template" in source_choices:
        source_options.append(("template", "Template", "Create from a saved snapshot"))
    source = prompt_choice("Environment base", source_options, default=source_default)
    name = Prompt.ask("Environment name", default=default_name or current.get("container_name") or "sandbox").strip()
    result: dict[str, Any] = {"container_name": name, "mounts": _prompt_mounts(list(current.get("mounts") or []))}
    result["gpu_request"], result["devices"] = _prompt_hardware(supervisor, current)
    if source == "template":
        names = [item.name for item in templates]
        template_name = prompt_choice(
            "Template",
            [(item_name, item_name) for item_name in names],
            default=current.get("template_name") if current.get("template_name") in names else names[0],
        )
        result.update({"template_name": template_name, "dockerfile": None, "egress_allow": None, "egress_deny": None})
    else:
        dockerfile = str(current.get("dockerfile") or default_dockerfile())
        if prompt_confirm("Edit Dockerfile?", default=False):
            dockerfile = _edit_dockerfile(dockerfile)
        allow = _domain_list(current.get("egress_allow"), DEFAULT_EGRESS_ALLOW)
        deny = _domain_list(current.get("egress_deny"), [])
        if prompt_confirm("Edit network policy?", default=False):
            allow = _prompt_domain_list("Allowed domains", allow)
            deny = _prompt_domain_list("Blocked domains", deny)
        result.update({"template_name": None, "dockerfile": dockerfile, "egress_allow": allow, "egress_deny": deny})
    return result


def run_container_manager(supervisor: ContainerSupervisor | None = None) -> None:
    """Host-level interactive container manager available before loading a session."""
    supervisor = supervisor or ContainerSupervisor()
    while True:
        console.print("\n[bold cyan]Container management[/bold cyan]")
        action = prompt_choice(
            "Container management",
            [
                ("list", "List environments", "Show all managed containers"),
                ("create", "Create environment", "Build a standalone managed container"),
                ("manage", "Manage environment", "Shell, edit, clone, snapshot, or remove"),
                ("templates", "Templates", "Use or delete saved container templates"),
                ("back", "Back", "Return to the MuCLI launcher"),
            ],
            default="list",
        )
        if action == "back":
            return
        if action == "list":
            _show_environment_table(supervisor)
            continue
        if action == "create":
            config = _prompt_environment_config(supervisor)
            try:
                supervisor.create_environment(
                    container_name=config["container_name"],
                    dockerfile=config.get("dockerfile"),
                    template_name=config.get("template_name"),
                    mounts=config.get("mounts"),
                    gpu_request=config.get("gpu_request"),
                    devices=config.get("devices"),
                    egress_allow=config.get("egress_allow"),
                    egress_deny=config.get("egress_deny"),
                    start=True,
                    progress=_progress,
                )
                console.print(f"[green]Created {config['container_name']}[/green]")
            except Exception as exc:
                console.print(f"[red]{exc}[/red]")
            continue
        if action == "templates":
            templates = supervisor.template_registry.list_templates()
            if not templates:
                console.print("[dim]No templates.[/dim]")
                continue
            for index, item in enumerate(templates, 1):
                console.print(f" {index}. {item.name} — {item.description or item.image}", markup=False)
            template_action = prompt_choice(
                "Template action",
                [
                    ("use", "Use template", "Create a new environment from this template"),
                    ("delete", "Delete template", "Remove the saved template image reference"),
                    ("back", "Back", "Return to container management"),
                ],
                default="back",
            )
            if template_action == "back":
                continue
            names = [item.name for item in templates]
            selected = prompt_choice(
            "Template",
            [(item_name, item_name) for item_name in names],
            default=names[0],
        )
            if template_action == "delete":
                if prompt_confirm(f"Delete template {selected}?", default=False):
                    supervisor.remove_template(selected)
            else:
                name = Prompt.ask("New environment name", default=f"{selected}-env")
                try:
                    supervisor.create_environment(container_name=name, template_name=selected, start=True, progress=_progress)
                except Exception as exc:
                    console.print(f"[red]{exc}[/red]")
            continue

        ref = _select_environment(supervisor)
        if ref is None:
            continue
        operation = prompt_choice(
            f"Manage {ref.name}",
            [
                ("shell", "Interactive shell", "Open docker exec -it inside the environment"),
                ("start", "Start", "Start the environment"),
                ("stop", "Stop", "Stop the environment"),
                ("restart", "Restart", "Restart the environment"),
                ("edit", "Edit and recreate", "Change its base, mounts, or network policy"),
                ("clone", "Clone", "Create another environment from this configuration"),
                ("snapshot", "Create template", "Snapshot the writable filesystem as a template"),
                ("remove", "Remove", "Delete the environment and its volumes"),
                ("back", "Back", "Return to container management"),
            ],
            default="shell",
        )
        if operation == "back":
            continue
        try:
            if operation == "shell":
                supervisor.interactive_shell(ref.name)
            elif operation == "start":
                supervisor.start(ref.name)
            elif operation == "stop":
                supervisor.stop(ref.name)
            elif operation == "restart":
                supervisor.restart(ref.name)
            elif operation in {"edit", "clone"}:
                config = supervisor.configuration(ref.name)
                default_name = ref.name if operation == "edit" else f"{ref.name.replace('mucli-', '')}-copy"
                updated = _prompt_environment_config(supervisor, config=config, default_name=default_name)
                if operation == "edit":
                    supervisor.reconfigure_environment(
                        ref.name,
                        dockerfile=updated.get("dockerfile"), template_name=updated.get("template_name"),
                        mounts=updated.get("mounts"), gpu_request=updated.get("gpu_request"),
                        devices=updated.get("devices"), egress_allow=updated.get("egress_allow"),
                        egress_deny=updated.get("egress_deny"), start=True, progress=_progress,
                    )
                else:
                    supervisor.create_environment(
                        container_name=updated["container_name"], dockerfile=updated.get("dockerfile"),
                        template_name=updated.get("template_name"), mounts=updated.get("mounts"),
                        gpu_request=updated.get("gpu_request"), devices=updated.get("devices"),
                        egress_allow=updated.get("egress_allow"), egress_deny=updated.get("egress_deny"),
                        start=True, progress=_progress,
                    )
            elif operation == "snapshot":
                template_name = Prompt.ask("Template name", default=ref.name.replace("mucli-", ""))
                description = Prompt.ask("Description", default="")
                supervisor.snapshot(ref.name, template_name, description=description)
            elif operation == "remove" and prompt_confirm(f"Remove {ref.name} and its volumes?", default=False):
                supervisor.remove(ref.name, force=True)
        except Exception as exc:
            console.print(f"[red]{exc}[/red]")


def _progress(stage: str, message: str) -> None:
    label = stage.replace("_", " ")
    console.print(f"[cyan]container · {label}[/cyan]  {message}")


def ensure_tui_container(session: Any) -> Any:
    manager = session.session_manager
    name = manager.current_session_name
    config = dict(getattr(manager, "container_config", {}) or {})
    selected_ollama_mode = getattr(
        getattr(session, "provider", None), "_mu_ollama_mode", None
    )
    if (
        getattr(getattr(session, "provider", None), "name", None) == "ollama"
        and selected_ollama_mode in {"local", "cloud"}
        and session.variables.get("ollama_mode") != selected_ollama_mode
    ):
        session.variables["ollama_mode"] = selected_ollama_mode
        if selected_ollama_mode == "local":
            session.variables["ollama_host"] = ""
        manager.save_history(session.folder_context)
    supervisor = getattr(session, "_container_supervisor", None) or ContainerSupervisor()
    existing = supervisor.container_for_session(name)
    container_name = str(
        config.get("container_name")
        or (existing.name if existing is not None else f"mucli-{name}")
    ).strip()
    ref = supervisor.create(
        container_name=container_name,
        session_name=name,
        dockerfile=config.get("dockerfile"),
        template_name=config.get("template_name"),
        mounts=list(config.get("mounts") or []),
        gpu_request=config.get("gpu_request"),
        devices=list(config.get("devices") or []),
        egress_allow=list(config.get("egress_allow") or DEFAULT_EGRESS_ALLOW),
        egress_deny=list(config.get("egress_deny") or []),
        supervisor_url="",
        progress=_progress,
    )
    if existing is None or ref.name != existing.name or ref.worker_protocol != existing.worker_protocol:
        config.update(
            {
                "container_name": ref.name,
                "dockerfile": config.get("dockerfile"),
                "template_name": ref.template_name or config.get("template_name"),
                "mounts": [m.to_dict() for m in ref.mounts],
                "gpu_request": ref.gpu_request,
                "devices": [item.to_dict() for item in ref.devices],
                "egress_allow": list(ref.egress_allow),
                "egress_deny": list(ref.egress_deny),
            }
        )
        manager.container_config = config
        manager.save_history(session.folder_context)
    session._container_supervisor = supervisor
    session.container_ref = ref
    return ref


def send_tui_container_message(
    session: Any, text: str, *, origin: str = "user"
) -> dict[str, Any]:
    supervisor = getattr(session, "_container_supervisor", None) or ContainerSupervisor()
    session._container_supervisor = supervisor
    try:
        ensure_tui_container(session)
        response = supervisor.send_sync(
            session.session_manager.current_session_name,
            text,
            provider=session.provider.name,
            model=session.provider.model_name,
            agent_mode=str(session.variables.get("agent_mode", "default")),
            system_instruction=session.system_instruction,
            origin=origin,
        )
    except (ContainerRuntimeError, OSError, RuntimeError, ValueError) as exc:
        failure = describe_container_load_error(
            exc,
            session_name=session.session_manager.current_session_name,
            container_name=str((getattr(session.session_manager, "container_config", {}) or {}).get("container_name") or ""),
        )
        rendered = format_container_load_error(failure)
        if session.ui:
            # UI surfaces get the concise raw error; the long resolution
            # narrative is for the plain console path only.
            session.ui.show_error(str(exc))
        else:
            console.print(f"[red]{rendered}[/red]")
        return {"status": "error", "error": str(exc), "failure": failure}
    assistant_text = str(response.get("assistant_text") or "")
    if assistant_text and session.ui:
        session.ui.render_message("assistant", assistant_text, session.provider.model_name)
    # The worker wrote the mounted session JSON. Reload the host mirror so TUI
    # memory, traces, feature state, and token totals remain current.
    name = session.session_manager.current_session_name
    session.session_manager._load_session(name)
    session.sync_runtime_state()
    result = response.get("result")
    return result if isinstance(result, dict) else {"status": "complete"}


def interrupt_tui_container(session: Any) -> dict[str, Any]:
    supervisor = getattr(session, "_container_supervisor", None)
    if supervisor is None:
        return {"ok": False, "detail": "No container supervisor is active."}
    return supervisor.interrupt(session.session_manager.current_session_name)


def shutdown_tui_container(session: Any) -> None:
    supervisor = getattr(session, "_container_supervisor", None)
    if supervisor is not None:
        supervisor.shutdown()
