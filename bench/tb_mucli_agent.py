"""Terminal-Bench installed-agent adapter for mucli.

Wraps mucli as a Terminal-Bench agent: the harness installs mucli into the
task container, feeds it the task instruction headlessly, and mucli works in
``/app`` until done. Grading stays with the task's own verifier scripts —
mucli's JSONL traces are additionally captured into the agent logs dir for
post-run harness analysis (tokens, wall-clock, compactions).

Usage (TB 0.2.x):
    tb run --dataset-path ~/.cache/terminal-bench/terminal-bench-core/0.1.1 \
        --agent-import-path bench.tb_mucli_agent:MucliAgent \
        --task-id hello-world --model <provider/model>

Provider keys are forwarded verbatim (OPENAI_API_KEY / GEMINI_API_KEY /
OLLAMA_HOST / OLLAMA_API_KEY). The current tracked worktree is snapshotted into
each task and fingerprinted in ``/logs/mucli/source.json``.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

from bench.tb_support import build_source_tarball, read_trace_usage, stop_mucli_process

# Where mucli's own repo lives on the host (bench/ sits inside it).
_MUCLI_REPO_HOST = Path(__file__).resolve().parent.parent

_FORWARD_KEYS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OLLAMA_HOST",
    "OLLAMA_API_KEY",
    "OPENAI_BASE_URL",
    "MUCLI_BENCH_PROVIDER",
    "MUCLI_BENCH_MODEL",
)

_BENCHMARK_PROMPTS = {
    "none": "",
    "verify-v1": (
        "Before finishing, verify the final state against every requirement "
        "with direct checks. For services and configuration, exercise the "
        "changed behavior end to end rather than relying only on a syntax "
        "check. When restoring existing files or commits, use version-control "
        "operations instead of reconstructing content so exact bytes are "
        "preserved. Prefer the smallest direct solution, stop exploration once "
        "the acceptance criteria pass, and do not report success while any "
        "check fails."
    ),
}


class MucliAgent(AbstractInstalledAgent):
    """Run mucli headlessly inside the TB task container."""

    @staticmethod
    def name() -> str:
        return "mucli"

    def __init__(
        self,
        model_name: str | None = None,
        execution_timeout_sec: float | str | None = None,
        setup_timeout_sec: float | str | None = 180,
        benchmark_prompt: str = "verify-v1",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model_name = model_name or os.environ.get("MUCLI_BENCH_MODEL", "")
        self._version = kwargs.get("version", "bench-2")
        self._execution_timeout_sec = self._positive_timeout(execution_timeout_sec)
        self._setup_timeout_sec = self._positive_timeout(setup_timeout_sec) or 180.0
        if benchmark_prompt not in _BENCHMARK_PROMPTS:
            choices = ", ".join(sorted(_BENCHMARK_PROMPTS))
            raise ValueError(f"benchmark_prompt must be one of: {choices}")
        self._benchmark_prompt = benchmark_prompt

    @staticmethod
    def _positive_timeout(value: float | str | None) -> float | None:
        if value is None:
            return None
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution_timeout_sec must be a number") from exc
        if timeout <= 0:
            raise ValueError("execution_timeout_sec must be positive")
        return timeout

    @staticmethod
    def _write_execution_metrics(
        logging_dir: Path | None,
        *,
        setup_seconds: float,
        execution_seconds: float,
        execution_timeout_seconds: float | None,
        completed: bool,
        error_type: str | None,
        error_phase: str | None = None,
    ) -> None:
        """Persist the timing boundary separately from TB's inclusive timer."""

        if logging_dir is None:
            return
        payload = {
            "schema": 1,
            "setup_seconds": round(setup_seconds, 6),
            "execution_seconds": round(execution_seconds, 6),
            "execution_timeout_seconds": execution_timeout_seconds,
            "completed": completed,
            "timed_out": error_type == "TimeoutError" and error_phase == "execution",
            "setup_timed_out": error_type == "TimeoutError" and error_phase == "setup",
            "error_type": error_type,
            "error_phase": error_phase,
        }
        path = Path(logging_dir) / "mucli-execution.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError:
            # Timing telemetry must never change the task outcome.
            temporary.unlink(missing_ok=True)

    # -- repo delivery ----------------------------------------------------

    def _build_repo_tarball(self) -> Path:
        """Snapshot the current tracked worktree, including local edits."""
        return build_source_tarball(_MUCLI_REPO_HOST)

    @staticmethod
    def _make_logs_readable(session) -> None:
        # The directory may not exist when startup failed; artifact recovery is
        # best-effort and must never mask the benchmark outcome.
        session.container.exec_run(
            ["sh", "-c", "chmod -R a+rwX /logs/mucli 2>/dev/null || true"]
        )

    @staticmethod
    def _stop_running_agent(session) -> None:
        """Stop MuCLI before TB starts the verifier after an inner timeout.

        TB 0.2.x times out its ``tmux wait`` process but leaves the command in
        the pane alive.  Interrupt the foreground process first, then use
        narrowly scoped TERM/KILL fallbacks inside the disposable task
        container.  Without this, an agent can keep mutating ``/app`` while the
        verifier is reading it.
        """

        stop_mucli_process(session)

    def _setup_failure(
        self,
        logging_dir: Path | None,
        setup_started: float,
        error_type: str,
    ) -> AgentResult:
        self._write_execution_metrics(
            logging_dir,
            setup_seconds=time.monotonic() - setup_started,
            execution_seconds=0.0,
            execution_timeout_seconds=self._execution_timeout_sec,
            completed=False,
            error_type=error_type,
            error_phase="setup",
        )
        return AgentResult(
            total_input_tokens=0,
            total_output_tokens=0,
            failure_mode=FailureMode.AGENT_INSTALLATION_FAILED,
        )

    def perform_task(self, instruction, session, logging_dir=None):
        """Copy, install, and preflight mucli before running the task.

        The base implementation only copies the setup script; mucli's source
        must also land at /mucli-src (extracted) so the offline install works
        without a registry round-trip. This follows the base installation flow
        so a container-level import can be checked after setup and before the
        benchmark command is allowed to run.
        """
        setup_started = time.monotonic()
        tarball = self._build_repo_tarball()
        try:
            session.copy_to_container(
                paths=tarball,
                container_dir="/tmp",
                container_filename="mucli-src.tar.gz",
            )
        finally:
            tarball.unlink(missing_ok=True)
        # Offline wheelhouse (pre-baked manylinux wheels): eliminates
        # pip's registry round-trip — the 200s+ install cost that starved the
        # 360s per-task gates. Falls back to network if payload is missing.
        wheelhouse = (
            _MUCLI_REPO_HOST / "bench" / "artifacts" / "mucli-wheelhouse.tar.gz"
        )
        if wheelhouse.exists():
            session.copy_to_container(
                paths=wheelhouse,
                container_dir="/tmp",
                container_filename="mucli-wheelhouse.tar.gz",
            )
        extraction = session.container.exec_run(
            [
                "sh",
                "-c",
                "mkdir -p /mucli-src && "
                "tar -xzf /tmp/mucli-src.tar.gz -C /mucli-src",
            ]
        )
        if extraction.exit_code != 0:
            return self._setup_failure(
                logging_dir, setup_started, "SourceExtractionError"
            )

        setup_script = Path(self._install_agent_script_path)
        try:
            session.copy_to_container(
                setup_script,
                container_dir="/installed-agent",
                container_filename="install-agent.sh",
            )
        finally:
            setup_script.unlink(missing_ok=True)

        env_setup_content = self._create_env_setup_file()
        session.container.exec_run(
            [
                "sh",
                "-c",
                (
                    f"echo {shlex.quote(env_setup_content)} > "
                    "/installed-agent/setup-env.sh"
                ),
            ]
        )
        try:
            session.send_keys(
                ["source /installed-agent/setup-env.sh", "Enter"],
                block=True,
                max_timeout_sec=max(
                    0.1,
                    self._setup_timeout_sec - (time.monotonic() - setup_started),
                ),
            )
            session.send_keys(
                [
                    (
                        "if bash /installed-agent/install-agent.sh; then "
                        "printf '0' > /tmp/mucli-install-status; "
                        "else status=$?; printf '%s' \"$status\" "
                        "> /tmp/mucli-install-status; "
                        "echo 'INSTALL_FAIL_STATUS'; fi"
                    ),
                    "Enter",
                ],
                block=True,
                max_timeout_sec=max(
                    0.1,
                    self._setup_timeout_sec - (time.monotonic() - setup_started),
                ),
            )
        except TimeoutError as exc:
            self._stop_running_agent(session)
            self._make_logs_readable(session)
            return self._setup_failure(
                logging_dir, setup_started, type(exc).__name__
            )

        installation_status = session.container.exec_run(
            ["sh", "-c", 'test "$(cat /tmp/mucli-install-status)" = 0']
        )
        if installation_status.exit_code != 0:
            return self._setup_failure(
                logging_dir, setup_started, "InstallScriptError"
            )

        preflight = session.container.exec_run(
            ["sh", "-c", "cd /opt/mucli && python3 -c 'import mucli'"]
        )
        if preflight.exit_code != 0:
            return self._setup_failure(logging_dir, setup_started, "PreflightError")

        rendered_instruction = self._render_instruction(instruction)
        prompt_suffix = _BENCHMARK_PROMPTS[self._benchmark_prompt]
        if prompt_suffix:
            rendered_instruction = f"{rendered_instruction.rstrip()}\n\n{prompt_suffix}"
        setup_seconds = time.monotonic() - setup_started
        execution_started = time.monotonic()
        completed = False
        error_type = None
        try:
            for command in self._run_agent_commands(rendered_instruction):
                session.send_command(command)
            completed = True
        except TimeoutError as exc:
            error_type = type(exc).__name__
            self._stop_running_agent(session)
            self._make_logs_readable(session)
            input_tokens, output_tokens = read_trace_usage(logging_dir)
            return AgentResult(
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
                failure_mode=FailureMode.AGENT_TIMEOUT,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            self._write_execution_metrics(
                logging_dir,
                setup_seconds=setup_seconds,
                execution_seconds=time.monotonic() - execution_started,
                execution_timeout_seconds=self._execution_timeout_sec,
                completed=completed,
                error_type=error_type,
                error_phase="execution" if error_type else None,
            )

        # Files written through the bind mount are created by the container
        # user. Make the captured artifacts readable/removable by the host
        # benchmark user before parsing them.
        self._make_logs_readable(session)
        input_tokens, output_tokens = read_trace_usage(logging_dir)
        return AgentResult(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
        )

    # -- AbstractInstalledAgent contract ---------------------------------

    @property
    def _env(self) -> dict[str, str]:
        # /logs is mounted by every TB 0.2.x core task. Keeping MuCLI's state
        # there preserves traces and makes token accounting available without
        # touching the host's normal ~/.mucli state.
        env = {"MUCLI_HOME": "/logs/mucli"}
        env.update(
            {key: os.environ[key] for key in _FORWARD_KEYS if os.environ.get(key)}
        )
        if self._model_name:
            env.setdefault("MUCLI_BENCH_MODEL", self._model_name)
        return env

    def _create_env_setup_file(self) -> str:
        """Render shell-safe exports, including values containing quotes."""
        return "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in self._env.items()
        )

    @property
    def _install_agent_script_path(self) -> os.PathLike:
        expected = Path(__file__).parent / "mucli-setup.sh.j2"
        if not expected.exists():
            raise FileNotFoundError(
                f"missing Terminal-Bench setup template: {expected}"
            )
        return self._get_templated_script_path("mucli-setup.sh.j2")

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped = shlex.quote(instruction)
        provider_part = ""
        if self._model_name and "/" in self._model_name:
            provider, model = self._model_name.split("/", 1)
            if provider == "ollama":
                # Ollama cloud: the api key resolves the host; model id is bare.
                provider_part = f" --provider ollama --model {shlex.quote(model)}"
            else:
                provider_part = (
                    f" --provider {shlex.quote(provider)}"
                    f" --model {shlex.quote(model)}"
                )
        return [
            TerminalCommand(
                command=(
                    "cd /app && mucli --headless-prompt "
                    f"{escaped}{provider_part} --tool-profile terminal-bench --yolo"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=self._execution_timeout_sec or float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
