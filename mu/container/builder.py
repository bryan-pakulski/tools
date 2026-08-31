"""Image build and container creation for MuCLI workers."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from utils.config import HISTORY_DIR

from .docker_cli import CommandRunner, ContainerRuntimeError, OutputCallback, run_with_output
from .network import DEFAULT_EGRESS_ALLOW, create_isolated_network, teardown_network
from .ref import (
    DEFAULT_WORKER_PORT,
    ContainerRef,
    DeviceSpec,
    MountSpec,
    WORKER_PROTOCOL_VERSION,
)
from .hardware import validate_hardware  # MUCLI_CONTAINER_HARDWARE_V1
from .registry import ContainerRegistry

ProgressCallback = Callable[[str, str], None]


def _report(progress: ProgressCallback | None, stage: str, message: str) -> None:
    if progress is not None:
        progress(stage, message)


_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OLLAMA_API_KEY",
    "OLLAMA_HOST",
)


def container_slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9_.-]+", "-", str(name or "").strip().lower()).strip("-.")
    if not value:
        raise ValueError("container name is required")
    return value[:48]


def _default_source_path() -> str:
    # mu/container/builder.py -> repository root
    return str(Path(__file__).resolve().parents[2])


def default_dockerfile() -> str:
    """Return the maintained MuCLI worker Dockerfile template."""
    return Path(__file__).with_name("Dockerfile.mucli").read_text(encoding="utf-8")


def template_overlay_dockerfile(base_image: str) -> str:
    """Overlay the current MuCLI worker onto a saved environment template.

    Templates intentionally preserve user-installed packages and writable-layer
    configuration.  Re-copying the current source prevents an old snapshot from
    reviving a stale worker bridge after a MuCLI upgrade.
    """
    return f"""FROM {base_image}

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    MUCLI_CONTAINER_MODE=1 \\
    PYTHONPATH=/opt/mucli \\
    TIKTOKEN_CACHE_DIR=/opt/mucli/.cache/tiktoken

COPY . /opt/mucli
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /opt/mucli/requirements.txt

# MUCLI_TIKTOKEN_PREWARM_V1: keep token estimation offline at runtime.
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" && python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

WORKDIR /workspace
EXPOSE {DEFAULT_WORKER_PORT}
ENTRYPOINT [\"python3\", \"-m\", \"mu.container.worker\"]
"""


def provider_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in _PROVIDER_ENV_KEYS if os.environ.get(key)}


_TIKTOKEN_RUNTIME_LAYER = r'''

# MUCLI_TIKTOKEN_PREWARM_V1: keep token estimation offline at runtime.
ENV TIKTOKEN_CACHE_DIR=/opt/mucli/.cache/tiktoken
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \\
    && python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
'''


def prepare_worker_dockerfile(content: str) -> str:
    """Ensure every editable worker Dockerfile contains the offline tokenizer cache."""
    value = str(content or "").rstrip()
    if "MUCLI_TIKTOKEN_PREWARM_V1" in value:
        return value + "\n"
    return value + _TIKTOKEN_RUNTIME_LAYER + "\n"


def build_create_command(
    ref: ContainerRef,
    *,
    environment: dict[str, str] | None = None,
) -> list[str]:
    command = [
        "docker",
        "create",
        "--name",
        ref.name,
        "--hostname",
        ref.name,
        "--network",
        ref.network_name,
        "--restart",
        "unless-stopped",
        "--cap-drop",
        "NET_ADMIN",
        "--cap-drop",
        "SYS_ADMIN",
        "--cap-drop",
        "SYS_PTRACE",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "2048",
        "--add-host",
        "host.docker.internal:host-gateway",
        "--label",
        "io.mucli.managed=true",
        "--label",
        f"io.mucli.container={ref.name}",
    ]
    if ref.gpu_request:
        command += ["--gpus", ref.gpu_request]
    for device in ref.devices:
        command += [
            "--device",
            f"{device.host_path}:{device.container_path}:{device.permissions}",
        ]
    if ref.root_volume:
        command += ["-v", f"{ref.root_volume}:{ref.container_volume}:rw"]
    for session_name in ref.attached_sessions:
        session_dir = os.path.join(HISTORY_DIR, "sessions", session_name)
        os.makedirs(session_dir, exist_ok=True)
        command += [
            "-v",
            f"{session_dir}:{ref.container_volume}/sessions/{session_name}:rw",
        ]
    # Bind-mount the host trace directory into the container so trace JSONL
    # files written by the in-container TraceEmitter are immediately visible
    # to the host's trace router (/api/traces) and CLI.  Without this mount
    # traces land in the Docker named volume (mucli-*-home) and are invisible
    # to the host — the "trace stats not available in container mode" bug.
    host_trace_dir = os.path.join(os.path.abspath(os.path.expanduser(HISTORY_DIR)), "trace")
    os.makedirs(host_trace_dir, exist_ok=True)
    command += ["-v", f"{host_trace_dir}:{ref.container_volume}/trace:rw"]
    # Peer threads may be split across host and container runtimes. Mount the
    # coordination journals so both sides share one durable message/claim
    # ledger without exposing unrelated session transcripts.
    host_thread_dir = os.path.join(
        os.path.abspath(os.path.expanduser(HISTORY_DIR)), "thread-groups"
    )
    os.makedirs(host_thread_dir, exist_ok=True)
    command += [
        "-v",
        f"{host_thread_dir}:{ref.container_volume}/thread-groups:rw",
    ]
    if ref.workspace_volume:
        command += ["-v", f"{ref.workspace_volume}:/workspace:rw"]
    for mount in ref.mounts:
        command += ["-v", f"{mount.host_path}:{mount.container_path}:{mount.mode}"]
    proxy_host = ref.proxy_ip or ref.proxy_name
    proxy_url = f"http://{proxy_host}:{ref.proxy_port}" if proxy_host else ""
    env = {
        "MUCLI_CONTAINER_MODE": "1",
        "MUCLI_CONTAINER_NAME": ref.name,
        "MUCLI_SUPERVISOR_URL": ref.supervisor_url,
        "MUCLI_WORKER_TOKEN": ref.worker_token,
        "MUCLI_WORKER_PORT": str(ref.worker_port),
        "MUCLI_PROXY_URL": proxy_url,
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "TIKTOKEN_CACHE_DIR": "/opt/mucli/.cache/tiktoken",
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
        "MUCLI_EGRESS_ALLOW": __import__("json").dumps(ref.egress_allow),
        "MUCLI_EGRESS_DENY": __import__("json").dumps(ref.egress_deny),
        "MUCLI_GPU_REQUEST": ref.gpu_request,
        "MUCLI_DEVICES": __import__("json").dumps([item.to_dict() for item in ref.devices]),
        "MUCLI_WORKSPACES": __import__("json").dumps(
            list(dict.fromkeys(["/workspace", *[mount.container_path for mount in ref.mounts]]))
        ),
        **(environment or {}),
    }
    for key, value in env.items():
        if value:
            command += ["-e", f"{key}={value}"]
    command.append(ref.image)
    return command


def build_container(
    name: str,
    dockerfile_content: str | None = None,
    *,
    base_image: str | None = None,
    template_name: str | None = None,
    mounts: Iterable[MountSpec | dict] | None = None,
    gpu_request: str | None = None,
    devices: Iterable[DeviceSpec | dict] | None = None,
    egress_allow: list[str] | None = None,
    egress_deny: list[str] | None = None,
    mucli_source_path: str | None = None,
    supervisor_url: str = "http://host.docker.internal:30311",
    session_name: str | None = None,
    registry: ContainerRegistry | None = None,
    runner: CommandRunner | None = None,
    start: bool = True,
    progress: ProgressCallback | None = None,
    output: OutputCallback | None = None,
) -> ContainerRef:
    runner = runner or CommandRunner()
    registry = registry or ContainerRegistry()
    _report(progress, "checking_docker", "Checking Docker and container configuration…")
    docker = runner.require("docker")
    normalized_gpu, parsed_devices = validate_hardware(
        gpu_request, devices, runner=runner
    )
    slug = container_slug(name)
    managed_name = slug if slug.startswith("mucli-") else f"mucli-{slug}"
    source_path = os.path.abspath(mucli_source_path or _default_source_path())
    if not os.path.isdir(source_path):
        raise ValueError(f"MuCLI source directory does not exist: {source_path}")

    content = prepare_worker_dockerfile(
        template_overlay_dockerfile(base_image)
        if base_image
        else (dockerfile_content if dockerfile_content is not None else default_dockerfile())
    )
    dockerfile_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    container_dir = os.path.join(registry.root, managed_name)
    os.makedirs(container_dir, exist_ok=True)
    dockerfile_path = os.path.join(container_dir, "Dockerfile")
    Path(dockerfile_path).write_text(content, encoding="utf-8")
    image = f"mucli/{slug}:{dockerfile_hash[:12]}"
    network_name = f"{managed_name}-net"
    root_volume = f"{managed_name}-home"
    workspace_volume = f"{managed_name}-workspace"
    parsed_mounts = [
        item if isinstance(item, MountSpec) else MountSpec.from_dict(item)
        for item in (mounts or [])
    ]

    ref = ContainerRef(
        container_id="",
        name=managed_name,
        image=image,
        dockerfile_hash=dockerfile_hash,
        mounts=parsed_mounts,
        gpu_request=normalized_gpu,
        devices=parsed_devices,
        egress_allow=list(dict.fromkeys(egress_allow or DEFAULT_EGRESS_ALLOW)),
        egress_deny=list(dict.fromkeys(egress_deny or [])),
        network_name=network_name,
        session_volume=(
            os.path.join(os.path.abspath(os.path.expanduser(HISTORY_DIR)), "sessions", session_name)
            if session_name else ""
        ),
        container_volume="/root/.mucli",
        worker_token=secrets.token_urlsafe(32),
        worker_port=registry.allocate_worker_port(exclude_name=managed_name),
        worker_protocol=WORKER_PROTOCOL_VERSION,
        supervisor_url=supervisor_url.rstrip("/"),
        status="building",
        attached_sessions=[session_name] if session_name else [],
        root_volume=root_volume,
        workspace_volume=workspace_volume,
        template_name=str(template_name or ""),
        standalone=session_name is None,
    )
    registry.upsert(ref)

    policy = None
    try:
        if base_image:
            _report(progress, "using_template", f"Using template image {base_image}…")
            run_with_output(
                runner, [docker, "image", "inspect", "--format", "{{.Id}}", base_image], output_callback=output
            )
        _report(
            progress,
            "building_image",
            "Refreshing the MuCLI worker layer…" if base_image else "Building the MuCLI worker image…",
        )
        run_with_output(
            runner,
            [docker, "build", "--pull", "-t", image, "-f", dockerfile_path, source_path],
            output_callback=output,
        )
        _report(progress, "creating_storage", "Creating persistent container storage…")
        run_with_output(
            runner, [docker, "volume", "create", root_volume], output_callback=output
        )
        run_with_output(
            runner, [docker, "volume", "create", workspace_volume], output_callback=output
        )
        host_allow: dict[str, list[int]] = {}
        if ref.supervisor_url:
            parsed_supervisor = urlparse(ref.supervisor_url)
            if parsed_supervisor.hostname:
                host_allow.setdefault(parsed_supervisor.hostname, []).append(
                    parsed_supervisor.port
                    or (443 if parsed_supervisor.scheme == "https" else 80)
                )
        ollama_host = os.environ.get("OLLAMA_HOST", "")
        if ollama_host:
            parsed_ollama = urlparse(ollama_host)
            if parsed_ollama.hostname in {"host.docker.internal", "localhost", "127.0.0.1"} or (parsed_ollama.hostname or "").startswith("172."):
                host = "host.docker.internal" if parsed_ollama.hostname in {"localhost", "127.0.0.1"} else parsed_ollama.hostname
                host_allow.setdefault(str(host), []).append(parsed_ollama.port or 11434)
        _report(
            progress,
            "configuring_network",
            "Creating an internal network and unprivileged egress proxy…",
        )
        policy = create_isolated_network(
            network_name,
            ref.egress_allow,
            egress_deny=ref.egress_deny,
            host_allow=host_allow,
            proxy_image=ref.image,
            runner=runner,
            output_callback=output,
        )
        ref.network_subnet = policy.subnet
        ref.proxy_name = policy.proxy_name
        ref.proxy_ip = policy.proxy_ip
        ref.proxy_port = policy.proxy_port
        ref.proxy_image = policy.proxy_image
        ref.egress_network_name = policy.egress_network_name
        _report(progress, "creating_container", "Creating the worker container…")
        create_cmd = build_create_command(ref, environment=provider_environment())
        create_cmd[0] = docker
        result = run_with_output(runner, create_cmd, output_callback=output)
        ref.container_id = result.stdout.strip() or (f"dry-{managed_name}" if runner.dry_run else "")
        if start:
            _report(progress, "starting_worker", "Starting the MuCLI worker…")
            run_with_output(
                runner, [docker, "start", managed_name], output_callback=output
            )
            ref.status = "running"
        else:
            if ref.proxy_name:
                run_with_output(
                    runner, [docker, "stop", "-t", "10", ref.proxy_name], output_callback=output
                )
            ref.status = "stopped"
        registry.upsert(ref)
        _report(progress, "worker_ready", "Worker started; attaching the session…")
        return ref
    except Exception:
        ref.status = "error"
        registry.upsert(ref)
        try:
            # MUCLI_PRESERVE_CONTAINER_VOLUMES_ON_FAILURE: a failed worker/image
            # rebuild must never destroy the user's durable home/workspace data.
            runner.run([docker, "rm", "-f", managed_name], check=False)
        except Exception:
            pass
        if policy is not None:
            teardown_network(
                network_name,
                policy.subnet,
                proxy_name=policy.proxy_name,
                egress_network_name=policy.egress_network_name,
                runner=runner,
            )
        raise
