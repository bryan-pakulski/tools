"""Host-side lifecycle and message bridge for container sessions."""
from __future__ import annotations

import ipaddress
import os
import subprocess
import time
from typing import Any, Callable

import httpx
import utils.config as _config

from .builder import build_container, container_slug
from .docker_cli import CommandRunner, ContainerRuntimeError, OutputCallback
from .network import teardown_network
from .hardware import detect_hardware, normalize_device_specs, normalize_gpu_request, validate_hardware  # MUCLI_CONTAINER_HARDWARE_V1
from .ref import ContainerRef, DeviceSpec, DEFAULT_WORKER_PORT, WORKER_PROTOCOL_VERSION
from .registry import ContainerRegistry
from .templates import ContainerTemplate, TemplateRegistry
from .runner import attach_session_folder, detach_session_folder, mount_folder
from utils.logger import logger


class ContainerSupervisor:
    def __init__(
        self,
        *,
        registry: ContainerRegistry | None = None,
        runner: CommandRunner | None = None,
        request_timeout: float = 15.0,
        template_registry: TemplateRegistry | None = None,
    ):
        self.registry = registry or ContainerRegistry()
        self.runner = runner or CommandRunner()
        self.request_timeout = request_timeout
        self.template_registry = template_registry or TemplateRegistry()

    def resolve(self, name: str) -> ContainerRef | None:
        """Resolve a managed name with or without the ``mucli-`` prefix."""
        value = str(name or "").strip()
        if not value:
            return None
        ref = self.registry.get(value)
        if ref is None and not value.startswith("mucli-"):
            ref = self.registry.get(f"mucli-{value}")
        return ref

    def list_environments(self, *, refresh: bool = True) -> list[ContainerRef]:
        refs = self.registry.list_containers()
        if not refresh:
            return refs
        for ref in refs:
            try:
                exists, running = self._container_state(ref.name)
                ref.status = "running" if running else ("stopped" if exists else "missing")
                self.registry.upsert(ref)
            except Exception:
                # Defensive: best-effort path must not break the caller.
                logger.debug("Suppressed exception", exc_info=True)
        return refs

    def _validate_rebuild_inputs(
        self,
        *,
        template_name: str | None,
        mounts: list[dict] | None,
        gpu_request: str | None = None,
        devices: list[dict] | list[DeviceSpec] | None = None,
        source_path: str | None = None,
    ) -> None:
        """Fail before deleting an old worker when rebuild inputs are invalid."""
        if source_path:
            resolved_source = os.path.abspath(os.path.expanduser(source_path))
            if not os.path.isdir(resolved_source):
                raise ContainerRuntimeError(
                    f"MuCLI source directory does not exist: {resolved_source}"
                )
        if template_name and self.template_registry.get(template_name) is None:
            raise ContainerRuntimeError(f"container template not found: {template_name}")
        for item in mounts or []:
            host_path = str((item or {}).get("host_path") or "").strip()
            if host_path and not os.path.isdir(os.path.abspath(os.path.expanduser(host_path))):
                raise ContainerRuntimeError(f"container bind mount is missing: {host_path}")
        validate_hardware(gpu_request, devices, runner=getattr(self, "runner", None))

    def create_environment(
        self,
        *,
        container_name: str,
        dockerfile: str | None = None,
        template_name: str | None = None,
        mounts: list[dict] | None = None,
        gpu_request: str | None = None,
        devices: list[dict] | list[DeviceSpec] | None = None,
        egress_allow: list[str] | None = None,
        egress_deny: list[str] | None = None,
        supervisor_url: str = "",
        source_path: str | None = None,
        start: bool = True,
        progress: Callable[[str, str], None] | None = None,
        output: OutputCallback | None = None,
    ) -> ContainerRef:
        """Create a managed environment without creating or attaching a session."""
        self._validate_rebuild_inputs(
            template_name=template_name,
            mounts=mounts,
            gpu_request=gpu_request,
            devices=devices,
            source_path=source_path,
        )
        existing = self.resolve(container_name)
        if existing is not None:
            exists, _running = self._container_state(existing.name)
            if exists:
                raise ContainerRuntimeError(f"managed container already exists: {existing.name}")
            self._discard_stale_registration(existing)

        template = None
        if template_name:
            template = self.template_registry.get(template_name)
            if template is None:
                raise ContainerRuntimeError(f"container template not found: {template_name}")

        return build_container(
            container_name,
            dockerfile,
            base_image=template.image if template else None,
            template_name=template.name if template else None,
            mounts=mounts,
            gpu_request=gpu_request,
            devices=devices,
            egress_allow=(egress_allow if egress_allow is not None else (template.egress_allow if template else None)),
            egress_deny=(egress_deny if egress_deny is not None else (template.egress_deny if template else None)),
            mucli_source_path=source_path,
            supervisor_url=supervisor_url,
            session_name=None,
            registry=self.registry,
            runner=self.runner,
            start=start,
            progress=progress,
            output=output,
        )

    def reconfigure_environment(
        self,
        container_name: str,
        *,
        dockerfile: str | None = None,
        template_name: str | None = None,
        mounts: list[dict] | None = None,
        gpu_request: str | None = None,
        devices: list[dict] | list[DeviceSpec] | None = None,
        egress_allow: list[str] | None = None,
        egress_deny: list[str] | None = None,
        supervisor_url: str = "",
        start: bool = True,
        progress: Callable[[str, str], None] | None = None,
        output: OutputCallback | None = None,
    ) -> ContainerRef:
        """Recreate a managed environment while retaining its named volumes and sessions."""
        ref = self.resolve(container_name)
        if ref is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        attached_sessions = list(ref.attached_sessions)
        standalone = bool(ref.standalone)
        resolved_gpu = getattr(ref, "gpu_request", "") if gpu_request is None else gpu_request
        resolved_devices = (
            [item.to_dict() for item in getattr(ref, "devices", [])]
            if devices is None
            else devices
        )
        self._validate_rebuild_inputs(
            template_name=template_name,
            mounts=mounts,
            gpu_request=resolved_gpu,
            devices=resolved_devices,
        )
        if progress is not None:
            progress("reconfiguring_container", "Stopping and recreating the managed environment…")
        self._discard_stale_registration(ref)
        rebuilt = self.create_environment(
            container_name=ref.name,
            dockerfile=dockerfile,
            template_name=template_name,
            mounts=mounts,
            gpu_request=resolved_gpu,
            devices=resolved_devices,
            egress_allow=egress_allow,
            egress_deny=egress_deny,
            supervisor_url=supervisor_url,
            start=start,
            progress=progress,
            output=output,
        )
        rebuilt.standalone = standalone
        self.registry.upsert(rebuilt)
        for session_name in attached_sessions:
            rebuilt = self.attach_session(rebuilt.name, session_name)
        return rebuilt

    def create(
        self,
        *,
        container_name: str,
        session_name: str,
        dockerfile: str | None = None,
        template_name: str | None = None,
        mounts: list[dict] | None = None,
        gpu_request: str | None = None,
        devices: list[dict] | list[DeviceSpec] | None = None,
        egress_allow: list[str] | None = None,
        egress_deny: list[str] | None = None,
        supervisor_url: str,
        source_path: str | None = None,
        progress: Callable[[str, str], None] | None = None,
        output: OutputCallback | None = None,
    ) -> ContainerRef:
        def report(stage: str, message: str) -> None:
            if progress is not None:
                progress(stage, message)

        report("resolving_container", "Checking for an existing managed container…")
        existing = self.resolve(container_name)
        requested_gpu = normalize_gpu_request(
            existing.gpu_request if existing is not None and gpu_request is None else gpu_request
        )
        requested_devices = (
            list(existing.devices)
            if existing is not None and devices is None
            else normalize_device_specs(devices or [])
        )
        restore_sessions: list[str] = []
        recovery_ref: ContainerRef | None = None
        if existing is not None:
            exists, _running = self._container_state(existing.name)
            actual_proxy_ip = (
                self._network_ip(existing.proxy_name, existing.network_name)
                if exists and existing.proxy_name and existing.network_name
                else ""
            )
            if (
                existing.status == "error"
                or not exists
                or not existing.proxy_name
                or not existing.proxy_ip
                or not actual_proxy_ip
                or actual_proxy_ip != existing.proxy_ip
                or int(existing.worker_protocol or 0) < WORKER_PROTOCOL_VERSION
                or int(existing.worker_port or 0) < DEFAULT_WORKER_PORT
                or requested_gpu != existing.gpu_request
                or requested_devices != existing.devices
            ):
                restore_sessions = list(existing.attached_sessions)
                recovery_ref = existing
                self._validate_rebuild_inputs(
                    template_name=template_name or existing.template_name or None,
                    mounts=mounts if mounts is not None else [item.to_dict() for item in existing.mounts],
                    gpu_request=requested_gpu,
                    devices=requested_devices,
                    source_path=source_path,
                )
                report(
                    "recovering_container",
                    "Migrating the container worker runtime and rebuilding without host privileges…",
                )
                self._discard_stale_registration(existing)
                existing = None
        if existing is None:
            template = self.template_registry.get(template_name) if template_name else None
            if template_name and template is None:
                raise ContainerRuntimeError(f"container template not found: {template_name}")
            try:
                ref = build_container(
                    container_name,
                    dockerfile,
                    base_image=template.image if template else None,
                    template_name=template.name if template else None,
                    mounts=mounts,
                    gpu_request=requested_gpu,
                    devices=requested_devices,
                    egress_allow=egress_allow,
                    egress_deny=egress_deny,
                    mucli_source_path=source_path,
                    supervisor_url=supervisor_url,
                    session_name=session_name,
                    registry=self.registry,
                    runner=self.runner,
                    progress=progress,
                    output=output,
                )
            except Exception:
                if recovery_ref is not None:
                    recovery_ref.status = "error"
                    self.registry.upsert(recovery_ref)
                raise
            for attached_name in restore_sessions:
                if attached_name == session_name or attached_name in ref.attached_sessions:
                    continue
                ref = attach_session_folder(
                    ref,
                    attached_name,
                    registry=self.registry,
                    runner=self.runner,
                )
            report("checking_worker", f"Waiting for worker API on port {ref.worker_port}…")
            self.wait_worker_ready(ref)
        else:
            report("reusing_container", "Reusing the existing managed container…")
            ref = existing
            self.ensure_running(ref)
            if session_name not in ref.attached_sessions:
                report("attaching_session", "Attaching the session data to the worker…")
                ref = attach_session_folder(
                    ref,
                    session_name,
                    registry=self.registry,
                    runner=self.runner,
                )
        attached = self.registry.attach_session(ref.name, session_name)
        report("container_ready", "Container is ready; loading the session…")
        return attached

    def container_for_session(self, session_name: str) -> ContainerRef | None:
        for ref in self.registry.list_containers():
            if session_name in ref.attached_sessions:
                return ref
        return None

    def _container_state(self, name: str) -> tuple[bool, bool]:
        """Return ``(exists, running)`` for a Docker container name."""
        docker = self.runner.require("docker")
        inspect = self.runner.run(
            [docker, "inspect", "-f", "{{.State.Running}}", name],
            check=False,
        )
        if inspect.returncode != 0:
            return False, False
        return True, inspect.stdout.strip().lower() == "true"

    def _network_ip(self, container_name: str, network_name: str) -> str:
        """Return a normalized Docker-network address or an empty string.

        Docker inspect can return placeholders or malformed text when a
        container has stale network metadata.  Treat those values as a missing
        attachment so callers can rebuild the topology instead of persisting an
        unusable proxy URL such as ``http://invalid IP:3128``.
        """
        if not container_name or not network_name:
            return ""
        docker = self.runner.require("docker")
        result = self.runner.run(
            [
                docker,
                "inspect",
                "-f",
                f'{{{{(index .NetworkSettings.Networks "{network_name}").IPAddress}}}}',
                container_name,
            ],
            check=False,
        )
        if result.returncode != 0:
            return ""
        value = str(result.stdout or "").strip()
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            return ""

    def _discard_stale_registration(self, ref: ContainerRef) -> None:
        """Remove registry/network state before rebuilding a managed worker.

        Persistent volumes are deliberately retained. A replacement worker uses
        the same volume names, so user-installed packages and workspace data are
        not discarded during recovery.
        """
        docker = self.runner.require("docker")
        self.runner.run([docker, "rm", "-f", ref.name], check=False)

        if ref.network_name:
            teardown_network(
                ref.network_name,
                ref.network_subnet,
                proxy_name=ref.proxy_name,
                egress_network_name=ref.egress_network_name,
                runner=self.runner,
            )
        # MUCLI_CONTAINER_PERSISTENCE_V1: keep the durable registry
        # record during repair. A replacement build will overwrite it; a failed
        # build leaves enough metadata to retry without recreating the session.
        ref.status = "rebuilding"
        self.registry.upsert(ref)

    def _ensure_proxy_running(self, ref: ContainerRef) -> None:
        if not ref.proxy_name:
            raise ContainerRuntimeError(
                f"managed container uses the legacy host-firewall network model: {ref.name}; "
                "reload or recreate it to migrate to the unprivileged proxy network"
            )
        exists, running = self._container_state(ref.proxy_name)
        if not exists:
            raise ContainerRuntimeError(
                f"egress proxy is missing from Docker: {ref.proxy_name}; "
                "reload or recreate the environment"
            )
        if not running:
            self.runner.run([self.runner.require("docker"), "start", ref.proxy_name])
        proxy_ip = self._network_ip(ref.proxy_name, ref.network_name)
        if not proxy_ip:
            raise ContainerRuntimeError(
                f"egress proxy {ref.proxy_name!r} is not attached to {ref.network_name!r}; "
                "reload the session to rebuild its network"
            )
        if not ref.proxy_ip or proxy_ip != ref.proxy_ip:
            raise ContainerRuntimeError(
                f"egress proxy address changed for {ref.name}: "
                f"registered={ref.proxy_ip or 'missing'} current={proxy_ip}; "
                "reload the session to rebuild the worker"
            )

    def ensure_running(self, ref: ContainerRef) -> ContainerRef:
        exists, running = self._container_state(ref.name)
        if not exists:
            raise ContainerRuntimeError(
                f"managed container is missing from Docker: {ref.name}; "
                "reload the session to rebuild it"
            )
        self._ensure_proxy_running(ref)
        if not running:
            self.runner.run([self.runner.require("docker"), "start", ref.name])
        ref.status = "running"
        ref = self.registry.upsert(ref)
        self.wait_worker_ready(ref)
        return ref

    def worker_url(self, ref: ContainerRef) -> str:
        docker = self.runner.require("docker")
        result = self.runner.run(
            [
                docker,
                "inspect",
                "-f",
                f'{{{{(index .NetworkSettings.Networks "{ref.network_name}").IPAddress}}}}',
                ref.name,
            ]
        )
        host = result.stdout.strip()
        if not host:
            raise ContainerRuntimeError(f"worker container has no network address: {ref.name}")
        return f"http://{host}:{ref.worker_port}"

    def _proxy_log_tail(self, ref: ContainerRef, *, lines: int = 80) -> str:
        if not ref.proxy_name:
            return ""
        try:
            result = self.runner.run(
                [
                    self.runner.require("docker"),
                    "logs",
                    "--tail",
                    str(max(1, int(lines))),
                    ref.proxy_name,
                ],
                check=False,
            )
        except Exception:
            return ""
        return "\n".join(
            value.strip()
            for value in (result.stdout, result.stderr)
            if str(value or "").strip()
        )[-12000:]

    def _worker_log_tail(self, ref: ContainerRef, *, lines: int = 80) -> str:
        try:
            result = self.runner.run(
                [
                    self.runner.require("docker"),
                    "logs",
                    "--tail",
                    str(max(1, int(lines))),
                    ref.name,
                ],
                check=False,
            )
        except Exception:
            return ""
        text = "\n".join(
            value.strip()
            for value in (result.stdout, result.stderr)
            if str(value or "").strip()
        )[-12000:]
        secrets = [
            ref.worker_token,
            *(os.environ.get(key, "") for key in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "OLLAMA_API_KEY",
            )),
        ]
        for secret in secrets:
            if secret and len(secret) >= 4:
                text = text.replace(secret, "<redacted>")
        return text

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error") or payload.get("message")
            if detail:
                return str(detail)
        return str(response.text or "").strip() or response.reason_phrase

    def _raise_worker_response(self, ref: ContainerRef, response: httpx.Response) -> None:
        detail = self._response_detail(response)
        logs = self._worker_log_tail(ref)
        message = f"container worker returned HTTP {response.status_code}: {detail}"
        if logs:
            message += f"\n\nworker log tail:\n{logs}"
        raise ContainerRuntimeError(message)

    def wait_worker_ready(
        self,
        ref: ContainerRef,
        *,
        timeout: float = 45.0,
        interval: float = 0.25,
    ) -> None:
        """Wait until the worker API is accepting authenticated requests."""
        if type(self.runner) is not CommandRunner or bool(getattr(self.runner, "dry_run", False)):
            return
        deadline = time.monotonic() + max(0.1, float(timeout))
        last_error = "worker did not answer"
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=2.0, trust_env=False) as client:
                    response = client.get(
                        f"{self.worker_url(ref)}/health",
                        headers={"X-MuCLI-Worker-Token": ref.worker_token},
                    )
                if response.status_code == 200:
                    payload = response.json()
                    actual_protocol = int(
                        payload.get("worker_protocol") or 0
                        if isinstance(payload, dict)
                        else 0
                    )
                    if actual_protocol == WORKER_PROTOCOL_VERSION:
                        return
                    last_error = (
                        f"worker protocol {actual_protocol} does not match "
                        f"required protocol {WORKER_PROTOCOL_VERSION}"
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {self._response_detail(response)}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(max(0.05, float(interval)))
        logs = self._worker_log_tail(ref)
        proxy_logs = self._proxy_log_tail(ref)
        message = f"container worker did not become ready on port {ref.worker_port}: {last_error}"
        if logs:
            message += f"\n\nworker log tail:\n{logs}"
        if proxy_logs:
            message += f"\n\negress proxy log tail:\n{proxy_logs}"
        raise ContainerRuntimeError(message)

    def _post(self, ref: ContainerRef, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_running(ref)
        with httpx.Client(timeout=self.request_timeout, trust_env=False) as client:
            response = client.post(
                f"{self.worker_url(ref)}{path}",
                json=payload,
                headers={"X-MuCLI-Worker-Token": ref.worker_token},
            )
        if response.status_code >= 400:
            self._raise_worker_response(ref, response)
        value = response.json()
        return value if isinstance(value, dict) else {"ok": True}

    def send(
        self,
        session_name: str,
        text: str,
        *,
        provider: str,
        model: str,
        agent_mode: str = "default",
        system_instruction: str = "You are a helpful assistant.",
        origin: str = "user",
    ) -> dict[str, Any]:
        ref = self.container_for_session(session_name)
        if ref is None:
            raise ContainerRuntimeError(f"no container attached to session {session_name!r}")
        return self._post(
            ref,
            "/send",
            {
                "session_name": session_name,
                "text": text,
                "provider": provider,
                "model": model,
                "agent_mode": agent_mode,
                "system_instruction": system_instruction,
                "origin": origin,
            },
        )

    def send_sync(
        self,
        session_name: str,
        text: str,
        *,
        provider: str,
        model: str,
        agent_mode: str = "default",
        system_instruction: str = "You are a helpful assistant.",
        timeout: float | None = None,
        origin: str = "user",
    ) -> dict[str, Any]:
        """Run one worker turn synchronously for terminal clients.

        Browser/mobile clients use the event callback bridge. A standalone TUI
        has no FastAPI callback server, so this endpoint returns the final
        assistant text and compact turn result directly.
        """
        ref = self.container_for_session(session_name)
        if ref is None:
            raise ContainerRuntimeError(f"no container attached to session {session_name!r}")
        self.ensure_running(ref)
        with httpx.Client(timeout=timeout or None, trust_env=False) as client:
            response = client.post(
                f"{self.worker_url(ref)}/send-sync",
                json={
                    "session_name": session_name,
                    "text": text,
                    "provider": provider,
                    "model": model,
                    "agent_mode": agent_mode,
                    "system_instruction": system_instruction,
                    "origin": origin,
                },
                headers={"X-MuCLI-Worker-Token": ref.worker_token},
            )
        if response.status_code >= 400:
            self._raise_worker_response(ref, response)
        value = response.json()
        return value if isinstance(value, dict) else {"ok": True}

    def proxy_mode_api(
        self,
        session_name: str,
        path: str,
        *,
        method: str = "GET",
        query: list[tuple[str, str]] | None = None,
        body: bytes = b"",
        content_type: str = "application/json",
        provider: str,
        model: str,
        agent_mode: str = "default",
        system_instruction: str = "You are a helpful assistant.",
    ) -> dict[str, Any]:
        """Forward one Mode OS request to the owning container worker.

        Only the six versioned mode surfaces are accepted.  Runtime sync runs
        first so a freshly restarted worker can serve an explorer before the
        next model turn and so cached workers always use the host-selected
        strategy.
        """
        allowed = (
            "/api/feature",
            "/api/research",
            "/api/security",
            "/api/debug",
            "/api/loop",
            "/api/teacher",
            "/api/memory",
        )
        clean_path = str(path or "")
        if not any(
            clean_path == prefix or clean_path.startswith(prefix + "/")
            for prefix in allowed
        ):
            raise ContainerRuntimeError("refusing to proxy a non-mode worker path")
        verb = str(method or "GET").upper()
        if verb not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ContainerRuntimeError(f"unsupported mode API method: {verb}")
        if len(body or b"") > 2 * 1024 * 1024:
            raise ContainerRuntimeError("mode API request exceeds 2 MiB")

        ref = self.container_for_session(session_name)
        if ref is None:
            raise ContainerRuntimeError(
                f"no container attached to session {session_name!r}"
            )
        self.ensure_running(ref)
        headers = {"X-MuCLI-Worker-Token": ref.worker_token}
        base_url = self.worker_url(ref)
        with httpx.Client(
            timeout=max(30.0, float(self.request_timeout)), trust_env=False
        ) as client:
            synchronized = client.post(
                f"{base_url}/runtime/sync",
                json={
                    "session_name": session_name,
                    "provider": provider,
                    "model": model,
                    "agent_mode": agent_mode,
                    "system_instruction": system_instruction,
                },
                headers=headers,
            )
            if synchronized.status_code >= 400:
                self._raise_worker_response(ref, synchronized)

            forwarded_headers = dict(headers)
            if body:
                forwarded_headers["Content-Type"] = str(
                    content_type or "application/json"
                )
            response = client.request(
                verb,
                f"{base_url}{clean_path}",
                params=query or [],
                content=body or None,
                headers=forwarded_headers,
            )
        return {
            "status_code": int(response.status_code),
            "content": bytes(response.content),
            "content_type": str(
                response.headers.get("content-type") or "application/json"
            ),
        }

    def interrupt(self, session_name: str) -> dict[str, Any]:
        ref = self.container_for_session(session_name)
        if ref is None:
            return {"ok": False, "detail": "No container attached."}
        return self._post(ref, "/interrupt", {"session_name": session_name})

    def add_mount(
        self, session_name: str, host_path: str, container_path: str, mode: str = "rw"
    ) -> ContainerRef:
        ref = self.container_for_session(session_name)
        if ref is None:
            raise ContainerRuntimeError(f"no container attached to session {session_name!r}")
        return mount_folder(
            ref,
            host_path,
            container_path,
            mode,
            registry=self.registry,
            runner=self.runner,
        )

    def detach(self, session_name: str, *, stop_if_idle: bool = True) -> ContainerRef | None:
        ref = self.container_for_session(session_name)
        if ref is None:
            return None
        ref = detach_session_folder(
            ref,
            session_name,
            registry=self.registry,
            runner=self.runner,
            recreate_if_empty=not stop_if_idle,
        )
        if stop_if_idle and not ref.attached_sessions:
            docker = self.runner.require("docker")
            self.runner.run([docker, "stop", "-t", "20", ref.name], check=False)
            if ref.proxy_name:
                self.runner.run([docker, "stop", "-t", "10", ref.proxy_name], check=False)
            ref.status = "stopped"
            self.registry.upsert(ref)
        return ref

    def remove(self, container_name: str, *, force: bool = False) -> bool:
        ref = self.resolve(container_name)
        if ref is None:
            return False
        if ref.attached_sessions and not force:
            raise RuntimeError("detach all sessions before removing the container")
        docker = self.runner.require("docker")
        self.runner.run([docker, "rm", "-f", ref.name], check=False)
        teardown_network(
            ref.network_name,
            ref.network_subnet,
            proxy_name=ref.proxy_name,
            egress_network_name=ref.egress_network_name,
            runner=self.runner,
        )
        if ref.workspace_volume:
            self.runner.run([docker, "volume", "rm", ref.workspace_volume], check=False)
        if ref.root_volume:
            self.runner.run([docker, "volume", "rm", ref.root_volume], check=False)
        return self.registry.remove(ref.name, force=force)

    def start(self, container_name: str) -> ContainerRef:
        ref = self.resolve(container_name)
        if ref is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        return self.ensure_running(ref)

    def stop(self, container_name: str) -> ContainerRef:
        ref = self.resolve(container_name)
        if ref is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        docker = self.runner.require("docker")
        self.runner.run([docker, "stop", "-t", "20", ref.name], check=False)
        if ref.proxy_name:
            self.runner.run([docker, "stop", "-t", "10", ref.proxy_name], check=False)
        ref.status = "stopped"
        return self.registry.upsert(ref)

    def restart(self, container_name: str) -> ContainerRef:
        ref = self.resolve(container_name)
        if ref is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        docker = self.runner.require("docker")
        self._ensure_proxy_running(ref)
        self.runner.run([docker, "restart", "-t", "20", ref.name])
        ref.status = "running"
        return self.registry.upsert(ref)

    def _recoverable_runtime_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "missing from docker",
                "legacy host-firewall",
                "egress proxy is missing",
                "is not attached to",
                "egress proxy address changed",
            )
        )

    def _rebuild_environment_runtime(
        self,
        ref: ContainerRef,
        *,
        supervisor_url: str = "",
        progress: Callable[[str, str], None] | None = None,
        output: OutputCallback | None = None,
    ) -> ContainerRef:
        """Recreate a managed environment from its persisted configuration.

        Named home/workspace volumes and attached session directories are
        preserved by ``reconfigure_environment``.  This is used when Docker has
        reassigned or lost the egress-proxy address, because the worker's proxy
        environment is fixed at container creation time and cannot be repaired
        by merely updating the registry.
        """
        config = self.configuration(ref.name)
        if progress is not None:
            progress(
                "recovering_container",
                "Repairing the container network and worker proxy configuration…",
            )
        if output is not None:
            output(
                "stdout",
                "Detected stale or invalid container network metadata; rebuilding the worker topology.",
            )
        return self.reconfigure_environment(
            ref.name,
            dockerfile=config.get("dockerfile"),
            template_name=config.get("template_name"),
            mounts=list(config.get("mounts") or []),
            egress_allow=list(config.get("egress_allow") or []),
            egress_deny=list(config.get("egress_deny") or []),
            supervisor_url=(supervisor_url or ref.supervisor_url),
            start=True,
            progress=progress,
            output=output,
        )

    def attach_session(
        self,
        container_name: str,
        session_name: str,
        *,
        supervisor_url: str = "",
        progress: Callable[[str, str], None] | None = None,
        output: OutputCallback | None = None,
    ) -> ContainerRef:
        ref = self.resolve(container_name)
        if ref is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        session_file = os.path.join(
            os.path.expanduser(_config.HISTORY_DIR),
            "sessions",
            session_name,
            "session.json",
        )
        if not os.path.isfile(session_file):
            raise ContainerRuntimeError(f"saved session not found: {session_name}")
        try:
            ref = self.ensure_running(ref)
        except ContainerRuntimeError as exc:
            if not self._recoverable_runtime_error(exc):
                raise
            ref = self._rebuild_environment_runtime(
                ref,
                supervisor_url=supervisor_url,
                progress=progress,
                output=output,
            )
        if session_name not in ref.attached_sessions:
            ref = attach_session_folder(
                ref, session_name, registry=self.registry, runner=self.runner
            )
        return self.registry.attach_session(ref.name, session_name)

    def hardware_capabilities(self) -> dict[str, Any]:
        """Return host GPU/device capability information for creation UIs."""
        return detect_hardware(self.runner)

    def configuration(self, container_name: str) -> dict[str, Any]:
        ref = self.resolve(container_name)
        if ref is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        dockerfile = None
        if not ref.template_name:
            path = os.path.join(self.registry.root, ref.name, "Dockerfile")
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    dockerfile = handle.read()
            except OSError:
                dockerfile = None
        return {
            "container_name": ref.name,
            "worker_port": ref.worker_port,
            "worker_protocol": ref.worker_protocol,
            "dockerfile": dockerfile,
            "template_name": ref.template_name or None,
            "mounts": [item.to_dict() for item in ref.mounts],
            "gpu_request": ref.gpu_request,
            "devices": [item.to_dict() for item in ref.devices],
            "egress_allow": list(ref.egress_allow),
            "egress_deny": list(ref.egress_deny),
            "network_isolation": "internal-proxy" if ref.proxy_name else "legacy",
            "proxy_name": ref.proxy_name or None,
            "proxy_ip": ref.proxy_ip or None,
        }

    def detach_session(
        self, container_name: str, session_name: str, *, stop_if_idle: bool = False
    ) -> ContainerRef:
        requested = self.resolve(container_name)
        attached = self.container_for_session(session_name)
        if requested is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        if attached is None or attached.name != requested.name:
            raise ContainerRuntimeError(
                f"session {session_name!r} is not attached to {requested.name}"
            )
        result = self.detach(session_name, stop_if_idle=stop_if_idle)
        if result is None:
            raise ContainerRuntimeError(f"session is not attached: {session_name}")
        return result

    def snapshot(
        self, container_name: str, template_name: str, *, description: str = ""
    ) -> ContainerTemplate:
        ref = self.resolve(container_name)
        if ref is None:
            raise ContainerRuntimeError(f"managed container not found: {container_name}")
        exists, _running = self._container_state(ref.name)
        if not exists:
            raise ContainerRuntimeError(f"Docker container is missing: {ref.name}")
        slug = container_slug(template_name)
        previous = self.template_registry.get(slug)
        image = f"mucli/template-{slug}:{int(time.time())}"
        docker = self.runner.require("docker")
        scrub_keys = (
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
            "GOOGLE_API_KEY", "OLLAMA_API_KEY", "OLLAMA_HOST",
            "MUCLI_WORKER_TOKEN", "MUCLI_SUPERVISOR_URL",
            "MUCLI_WORKER_PORT",
            "MUCLI_CONTAINER_NAME", "MUCLI_EGRESS_ALLOW",
            "MUCLI_EGRESS_DENY", "MUCLI_WORKSPACES",
            "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "MUCLI_PROXY_URL", "TIKTOKEN_CACHE_DIR",
            "NO_PROXY", "no_proxy",
        )
        commit_command = [docker, "commit"]
        for key in scrub_keys:
            commit_command.extend(["--change", f"ENV {key}="])
        commit_command.extend([ref.name, image])
        self.runner.run(commit_command)
        if previous is not None and previous.image != image:
            self.runner.run([docker, "image", "rm", previous.image], check=False)
        template = ContainerTemplate(
            name=slug,
            image=image,
            source_container=ref.name,
            description=str(description or ""),
            dockerfile_hash=ref.dockerfile_hash,
            egress_allow=list(ref.egress_allow),
            egress_deny=list(ref.egress_deny),
        )
        return self.template_registry.upsert(template)

    def remove_template(self, template_name: str, *, remove_image: bool = True) -> bool:
        template = self.template_registry.get(template_name)
        if template is None:
            return False
        if remove_image:
            self.runner.run(
                [self.runner.require("docker"), "image", "rm", template.image], check=False
            )
        return self.template_registry.remove(template.name)

    def interactive_shell(self, container_name: str, *, shell: str = "/bin/bash") -> int:
        """Attach the current terminal to a managed Docker environment."""
        ref = self.start(container_name)
        docker = self.runner.require("docker")
        command = [docker, "exec", "-it", ref.name, shell]
        result = subprocess.run(command, check=False)
        if result.returncode == 126 and shell != "/bin/sh":
            result = subprocess.run([docker, "exec", "-it", ref.name, "/bin/sh"], check=False)
        return int(result.returncode)

    def validate_token(self, container_name: str, token: str) -> bool:
        ref = self.registry.get(container_name)
        return bool(ref and token and __import__("hmac").compare_digest(ref.worker_token, token))

    def shutdown(self) -> None:
        """Leave managed Docker environments untouched on MuCLI shutdown.

        Server restarts and UI lifecycle changes are not container-management
        operations. Containers are stopped or removed only through explicit
        container actions or session deletion.
        """
        return None
