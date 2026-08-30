"""Small injectable subprocess wrapper for Docker commands."""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence


class ContainerRuntimeError(RuntimeError):
    pass


OutputCallback = Callable[[str, str], None]


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def _redact_command_list(command: Sequence[str]) -> list[str]:
    """Return a copy of argv safe for diagnostics objects.

    Round-20 F45: create commands carry `-e KEY=value` provider secrets.
    Anything that outlives the run — result objects, ledgers, exception
    payloads — must hold the redacted form; raw argv stays execution-local.
    """
    values = [str(item) for item in command]
    rendered: list[str] = []
    redact_next = False
    for index, value in enumerate(values):
        if redact_next:
            key, separator, _secret = value.partition("=")
            rendered.append(f"{key}=<redacted>" if separator else "<redacted>")
            redact_next = False
            continue
        lower = value.lower()
        if value == "-e" and index + 1 < len(values):
            next_value = values[index + 1]
            key = next_value.split("=", 1)[0].upper()
            rendered.append(value)
            if any(token in key for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                redact_next = True
            continue
        if any(token in lower for token in ("worker_token=", "api_key=", "password=")):
            key = value.split("=", 1)[0]
            rendered.append(f"{key}=<redacted>")
        else:
            rendered.append(value)
    return rendered


def _redact_command(command: Sequence[str]) -> str:
    """Render a command for diagnostics without exposing provider secrets."""
    return " ".join(_redact_command_list(command))


def run_with_output(
    runner,
    args: Sequence[str],
    *,
    check: bool = True,
    input_text: str | None = None,
    timeout: float | None = None,
    output_callback: OutputCallback | None = None,
) -> CommandResult:
    """Run through an injectable runner while retaining old fake-runner support."""
    if output_callback is None:
        return runner.run(args, check=check, input_text=input_text, timeout=timeout)
    try:
        return runner.run(
            args,
            check=check,
            input_text=input_text,
            timeout=timeout,
            output_callback=output_callback,
        )
    except TypeError as exc:
        if "output_callback" not in str(exc):
            raise
        output_callback("command", f"$ {_redact_command(args)}")
        result = runner.run(args, check=check, input_text=input_text, timeout=timeout)
        for line in str(getattr(result, "stdout", "") or "").splitlines():
            output_callback("stdout", line)
        for line in str(getattr(result, "stderr", "") or "").splitlines():
            output_callback("stderr", line)
        return result


class CommandRunner:
    def __init__(self, *, dry_run: bool = False):
        self.dry_run = dry_run
        self.commands: list[list[str]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        input_text: str | None = None,
        timeout: float | None = None,
        output_callback: OutputCallback | None = None,
    ) -> CommandResult:
        command = [str(item) for item in args]
        # Round-18 F31: the ledger previously stored the RAW argv —
        # create commands carry `-e KEY=value` provider secrets, so any
        # introspection path reading runner.commands got live API keys
        # even though rendered diagnostics were redacted. Store the
        # redacted rendering instead; the ledger is diagnostics-only.
        self.commands.append(_redact_command_list(command))
        if output_callback is not None:
            output_callback("command", f"$ {_redact_command(command)}")
        if self.dry_run:
            return CommandResult(_redact_command_list(command), 0, "", "")
        if output_callback is None:
            proc = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            # Round-20 F45: result objects outlive the run and are used
            # for diagnostics — never carry the raw secret-bearing argv.
            result = CommandResult(
                _redact_command_list(command), proc.returncode, proc.stdout, proc.stderr
            )
        else:
            result = self._run_streaming(
                command,
                input_text=input_text,
                timeout=timeout,
                output_callback=output_callback,
            )
        if check and result.returncode != 0:
            raise ContainerRuntimeError(
                f"command failed ({result.returncode}): {_redact_command(command)}\n"
                f"{result.stderr.strip()}"
            )
        return result

    @staticmethod
    def _run_streaming(
        command: list[str],
        *,
        input_text: str | None,
        timeout: float | None,
        output_callback: OutputCallback,
    ) -> CommandResult:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if input_text is not None and proc.stdin is not None:
            proc.stdin.write(input_text)
            proc.stdin.close()

        events: queue.Queue[tuple[str, str | None]] = queue.Queue()

        def pump(stream, label: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    events.put((label, line))
            finally:
                events.put((label, None))
                stream.close()

        threads = [
            threading.Thread(target=pump, args=(proc.stdout, "stdout"), daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        open_streams = 2
        started = time.monotonic()
        while open_streams:
            if timeout is not None and time.monotonic() - started > timeout:
                proc.kill()
                proc.wait()
                # Round-20 F45: TimeoutExpired embeds cmd in its payload;
                # callers render str(exc) into logs/diagnostics. Raise
                # with the REDACTED argv so secrets never enter the
                # exception chain.
                raise subprocess.TimeoutExpired(
                    _redact_command_list(command), timeout
                )
            try:
                label, line = events.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None and not any(thread.is_alive() for thread in threads):
                    break
                continue
            if line is None:
                open_streams -= 1
                continue
            if label == "stdout":
                stdout_parts.append(line)
            else:
                stderr_parts.append(line)
            output_callback(label, line.rstrip("\r\n"))

        returncode = proc.wait()
        for thread in threads:
            thread.join(timeout=0.2)
        return CommandResult(
            _redact_command_list(command),
            returncode,
            "".join(stdout_parts),
            "".join(stderr_parts),
        )

    def require(self, executable: str) -> str:
        found = shutil.which(executable)
        if found:
            return found
        if self.dry_run:
            return executable
        raise ContainerRuntimeError(f"required executable not found: {executable}")

    def docker_json(self, args: Sequence[str]) -> dict:
        result = self.run([self.require("docker"), *args])
        try:
            return json.loads(result.stdout or "{}")
        except ValueError as exc:
            raise ContainerRuntimeError("docker returned invalid JSON") from exc
