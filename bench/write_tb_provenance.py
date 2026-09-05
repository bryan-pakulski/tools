#!/usr/bin/env python3
"""Write reproducibility metadata for one MuCLI Terminal-Bench pack run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.tb_support import source_snapshot_metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tb_version(tb_executable: Path) -> str:
    python = tb_executable.parent / "python"
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.metadata; "
                "print(importlib.metadata.version('terminal-bench'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _wheelhouse_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"path": str(path.resolve())}
    try:
        metadata["sha256"] = _sha256(path)
        with tarfile.open(path, "r:gz") as archive:
            raw = archive.extractfile("wheelhouse/manifest.json")
            if raw is not None:
                metadata["manifest"] = json.load(raw)
    except (OSError, KeyError, json.JSONDecodeError, tarfile.TarError):
        metadata["error"] = "unreadable"
    return metadata


def _prepared_images(path: Path | None, tasks: list[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {"manifest_path": str(path.resolve()), "error": "unreadable"}
    entries = manifest.get("tasks") if isinstance(manifest, dict) else None
    selected = {
        task: entries[task]
        for task in tasks
        if isinstance(entries, dict) and task in entries
    }
    return {
        "manifest_path": str(path.resolve()),
        "dataset": manifest.get("dataset"),
        "tasks": selected,
    }


def _harness_file_hashes(repo: Path) -> dict[str, str]:
    relative_paths = (
        "bench/run_pack.sh",
        "bench/tb_mucli_agent.py",
        "bench/tb_support.py",
        "bench/mucli-setup.sh.j2",
        "bench/prepare_tb.py",
        "bench/write_tb_provenance.py",
    )
    hashes = {}
    for relative in relative_paths:
        path = repo / relative
        if path.is_file():
            hashes[relative] = _sha256(path)
    return hashes


def write_provenance(
    output: Path,
    *,
    repo: Path,
    dataset: Path,
    model: str,
    attempts: int,
    tasks: list[str],
    tb_executable: Path,
    setup_allowance_seconds: float,
    outer_cleanup_margin_seconds: float,
    prepared_manifest: Path | None,
    run_label: str,
) -> dict[str, Any]:
    payload = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "harness": {
            "name": "mucli",
            "adapter_version": "bench-2",
            "tool_profile": "terminal-bench",
            "benchmark_prompt": "verify-v1",
            "model": model,
            "source": source_snapshot_metadata(repo),
            "files_sha256": _harness_file_hashes(repo),
        },
        "terminal_bench": {
            "version": _tb_version(tb_executable),
            "dataset": "terminal-bench-core==0.1.1",
            "dataset_path": str(dataset.resolve()),
            "tasks": tasks,
            "attempts_per_task": attempts,
            "n_concurrent": 1,
            "setup_allowance_seconds": setup_allowance_seconds,
            "outer_cleanup_margin_seconds": outer_cleanup_margin_seconds,
            "execution_time_excludes_agent_setup": True,
        },
        "task_images": _prepared_images(prepared_manifest, tasks),
        "wheelhouse": _wheelhouse_metadata(
            repo / "bench" / "artifacts" / "mucli-wheelhouse.tar.gz"
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--tb", type=Path, required=True)
    parser.add_argument("--setup-allowance-seconds", type=float, required=True)
    parser.add_argument("--outer-cleanup-margin-seconds", type=float, required=True)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--run-label", default="")
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    write_provenance(
        args.output,
        repo=args.repo,
        dataset=args.dataset,
        model=args.model,
        attempts=args.attempts,
        tasks=args.tasks,
        tb_executable=args.tb,
        setup_allowance_seconds=args.setup_allowance_seconds,
        outer_cleanup_margin_seconds=args.outer_cleanup_margin_seconds,
        prepared_manifest=args.prepared_manifest,
        run_label=args.run_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
