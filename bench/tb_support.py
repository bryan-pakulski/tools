"""Helpers shared by the Terminal-Bench adapter and its local tests.

This module intentionally has no ``terminal_bench`` dependency.  Keeping the
source snapshot and trace parsing here lets the normal MuCLI test environment
exercise benchmark-critical behavior without installing Terminal-Bench.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

_SOURCE_EXCLUDES = (
    "bench/artifacts/",
    "bench/results/",
)

_TASK_TIMEOUT_RE = re.compile(
    r"^max_agent_timeout_sec:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$",
    re.MULTILINE,
)


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=text,
    )


def _tracked_paths(repo: Path) -> list[Path]:
    """Return existing tracked files, excluding generated benchmark payloads.

    Only tracked files are eligible.  This includes edits to tracked files but
    deliberately excludes untracked files, which could contain local secrets.
    """

    raw = _git(repo, "ls-files", "--cached", "-z", text=False).stdout
    paths: list[Path] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        rel_text = os.fsdecode(item)
        posix = PurePosixPath(rel_text)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"unsafe tracked path: {rel_text!r}")
        if any(rel_text.startswith(prefix) for prefix in _SOURCE_EXCLUDES):
            continue
        rel = Path(*posix.parts)
        target = repo / rel
        # Deleted tracked files should be absent from the worktree snapshot.
        if target.is_file() or target.is_symlink():
            paths.append(rel)
    return sorted(paths, key=lambda path: path.as_posix())


def _snapshot_metadata(repo: Path, paths: list[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for rel in paths:
        target = repo / rel
        digest.update(rel.as_posix().encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(f"{target.lstat().st_mode & 0o7777:o}".encode("ascii"))
        digest.update(b"\0")
        if target.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(target).encode("utf-8", errors="surrogateescape"))
        else:
            digest.update(b"file\0")
            with target.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(b"\0")

    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(
        _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    )
    return {
        "schema": 1,
        "git_commit": commit,
        "tracked_worktree_dirty": dirty,
        "source_sha256": digest.hexdigest(),
        "file_count": len(paths),
    }


def source_snapshot_metadata(repo: Path) -> dict[str, Any]:
    """Describe the exact tracked worktree used by a benchmark run."""

    repo = Path(repo).resolve()
    return _snapshot_metadata(repo, _tracked_paths(repo))


def build_source_tarball(repo: Path) -> Path:
    """Create a safe snapshot of the current tracked worktree.

    ``git archive HEAD`` silently omitted in-progress fixes.  This snapshot
    reads tracked files from the worktree, so modified files are benchmarked,
    while ignored and untracked files (including likely credentials) are never
    copied into task containers.
    """

    repo = repo.resolve()
    paths = _tracked_paths(repo)
    metadata = _snapshot_metadata(repo, paths)

    handle = tempfile.NamedTemporaryFile(
        prefix="mucli-bench-", suffix=".tar.gz", delete=False
    )
    handle.close()
    output = Path(handle.name)
    try:
        with tarfile.open(output, mode="w:gz") as archive:
            for rel in paths:
                archive.add(
                    repo / rel,
                    arcname=rel.as_posix(),
                    recursive=False,
                )
            payload = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
            info = tarfile.TarInfo(".mucli-benchmark-source.json")
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def read_trace_usage(logging_dir: Path | None) -> tuple[int, int]:
    """Return token totals from the newest usable MuCLI trace, if any.

    Completed traces expose authoritative cumulative totals in ``run_end`` or
    ``turn_end``.  A command killed exactly at its execution deadline may have
    neither, so retain a sum of its completed ``iter`` records as a fallback.
    A completed trace is still preferred over a newer incomplete trace; this
    avoids counting an abandoned startup attempt when a later run completed.
    """

    if logging_dir is None:
        return 0, 0
    logging_dir = Path(logging_dir)
    # TB 0.2.18 advertises /agent-logs but its core compose files only mount
    # /logs, whose host path is the sibling ``sessions`` directory. Newer task
    # definitions may mount the dedicated agent directory, so support both.
    trace_dirs = (
        logging_dir / "mucli" / "trace",
        logging_dir.parent / "sessions" / "mucli" / "trace",
    )
    candidates = sorted(
        (path for directory in trace_dirs for path in directory.glob("*.jsonl")),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    incomplete_fallback: tuple[int, int] | None = None
    for path in candidates:
        latest: tuple[int, int] | None = None
        iter_input = 0
        iter_output = 0
        iter_count = 0
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if event.get("type") == "run_end":
                        try:
                            latest = (
                                int(event.get("tokens_in") or 0),
                                int(event.get("tokens_out") or 0),
                            )
                        except (TypeError, ValueError):
                            continue
                    elif event.get("type") == "turn_end" and latest is None:
                        try:
                            latest = (
                                int(event.get("total_in") or 0),
                                int(event.get("total_out") or 0),
                            )
                        except (TypeError, ValueError):
                            continue
                    elif event.get("type") == "iter":
                        tokens = event.get("tokens")
                        if not isinstance(tokens, dict):
                            continue
                        try:
                            iter_input += int(tokens.get("in") or 0)
                            iter_output += int(tokens.get("out") or 0)
                            iter_count += 1
                        except (TypeError, ValueError):
                            continue
        except OSError:
            continue
        if latest is not None:
            return latest
        if incomplete_fallback is None and iter_count:
            incomplete_fallback = (iter_input, iter_output)
    return incomplete_fallback or (0, 0)


def read_task_execution_timeout(task_yaml: Path) -> float:
    """Read the task's native agent budget without adding a YAML dependency."""

    try:
        text = Path(task_yaml).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read task config: {task_yaml}") from exc
    match = _TASK_TIMEOUT_RE.search(text)
    if match is None:
        raise ValueError(f"missing max_agent_timeout_sec in {task_yaml}")
    timeout = float(match.group(1))
    if timeout <= 0:
        raise ValueError(f"invalid max_agent_timeout_sec in {task_yaml}")
    return timeout


def stop_mucli_process(
    session,
    *,
    polls_per_signal: int = 20,
    poll_interval: float = 0.1,
    sleep=time.sleep,
) -> None:
    """Stop a timed-out MuCLI process before Terminal-Bench starts grading."""

    try:
        session.send_keys(["C-c"], block=False, min_timeout_sec=0.0)
    except Exception:
        pass

    def is_running() -> bool:
        result = session.container.exec_run(
            ["sh", "-c", "pgrep -f '[/]opt/mucli/mucli.py' >/dev/null 2>&1"]
        )
        return result.exit_code == 0

    for signal_name in ("TERM", "KILL"):
        for _ in range(polls_per_signal):
            if not is_running():
                return
            sleep(poll_interval)
        session.container.exec_run(
            [
                "sh",
                "-c",
                f"pkill -{signal_name} -f '[/]opt/mucli/mucli.py' "
                "2>/dev/null || true",
            ]
        )


__all__ = [
    "build_source_tarball",
    "read_task_execution_timeout",
    "read_trace_usage",
    "source_snapshot_metadata",
    "stop_mucli_process",
]
