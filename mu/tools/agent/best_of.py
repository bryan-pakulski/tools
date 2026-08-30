"""Best-of-N codex runs: race N parallel `codex exec` attempts on one
task, then rank the results.

Each attempt runs `codex exec -s read-only --ephemeral` with its own
output file, in parallel, on the same repository. Attempts never write
to the repo — they produce plans/proposals. The caller (agent loop) can
then implement the winning plan itself, or hand it to a workspace-write
codex run.

This mirrors Codex's `--attempts` best-of-N feature, which this codex
build does not expose natively.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from mu.tools import tool
from mu.tools.agent._repo_gate import check_repo_gate, resolve_repo
from utils.logger import logger

DEFAULT_ATTEMPTS = 3
MAX_ATTEMPTS = 5
PER_ATTEMPT_TIMEOUT_SECONDS = 600


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def _run_attempt(
    task: str, repo: str, out_path: str, attempt: int
) -> Dict[str, Any]:
    started = time.monotonic()
    cmd = [
        "codex",
        "exec",
        "-s",
        "read-only",
        "--ephemeral",
        "-C",
        repo,
        "-o",
        out_path,
        "--",
        task,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PER_ATTEMPT_TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - started
        ok = proc.returncode == 0
        output = ""
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
                output = fh.read().strip()
        # Scrub returned text (codex round-8 F3): attempt output and
        # stderr go straight to the model — secrets printed by codex
        # must be redacted like shell-tool results.
        from mu.tools._scrub import scrub_and_annotate as _scrub

        output = str(_scrub(output)) if output else ""
        stderr_tail = str(_scrub(proc.stderr[-1500:])) if proc.stderr else ""
        return {
            "attempt": attempt,
            "ok": ok,
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "output_chars": len(output),
            "output_path": out_path if output else None,
            "output": output[:4000] if output else stderr_tail,
        }
    except subprocess.TimeoutExpired:
        return {
            "attempt": attempt,
            "ok": False,
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "output_chars": 0,
            "output_path": None,
            "output": f"timeout after {PER_ATTEMPT_TIMEOUT_SECONDS}s",
        }
    except OSError as exc:
        return {
            "attempt": attempt,
            "ok": False,
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "output_chars": 0,
            "output_path": None,
            "output": f"launch failed: {exc}",
        }


def _rank(attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank by success, then output substance, then speed."""
    def key(a: Dict[str, Any]):
        return (
            0 if a["ok"] else 1,
            -a["output_chars"],
            a["elapsed_seconds"],
        )

    return sorted(attempts, key=key)


TOOL_NAME = "best_of_codex"


def _envelope(ok: bool, error_code: str | None, message: str, data: dict | None = None) -> Dict[str, Any]:
    return {
        "ok": ok,
        "error_code": error_code,
        "message": message,
        "data": data or {},
        "artifacts": [],
        "telemetry": {"tool_name": TOOL_NAME},
    }


@tool(
    name=TOOL_NAME,
    description=(
        "Race N parallel read-only codex attempts on one task and return the "
        "best-ranked proposal. Use for high-stakes plans where a single "
        "attempt may be weak: 2-5 attempts, ranked by success, then output "
        "substance, then speed. Attempts never modify the repo."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task/prompt given to every codex attempt.",
            },
            "repo": {
                "type": "string",
                "description": "Repository path to run codex against.",
            },
            "attempts": {
                "type": "integer",
                "description": "Parallel attempts (1-5, default 3).",
            },
        },
        "required": ["task"],
    },
    requires_approval=False,
    execution_kind="io",
    result_mode="json",
)
def best_of_codex(args: Dict[str, Any], context) -> Dict[str, Any]:
    task = str(args.get("task") or "").strip()
    if not task:
        return _envelope(False, "invalid_args", "best_of_codex requires non-empty 'task'.")
    repo = str(args.get("repo") or ".").strip() or "."
    try:
        attempts = int(args.get("attempts", DEFAULT_ATTEMPTS))
    except (TypeError, ValueError):
        attempts = DEFAULT_ATTEMPTS
    attempts = max(1, min(attempts, MAX_ATTEMPTS))

    if not _codex_available():
        return _envelope(False, "codex_unavailable", "codex CLI not found on PATH.")

    repo_abs = resolve_repo(repo)
    if repo_abs is None:
        return _envelope(False, "bad_repo", f"repo path not found: {repo}")
    # Capability boundary (codex round-8 F3 / round-9 F1): repo must pass the
    # same workspace/secret-path gate as filesystem tools — an approval-free
    # tool must not direct codex's sandbox at arbitrary host paths.
    allowed, reason = check_repo_gate(repo_abs, context)
    if not allowed:
        return _envelope(False, "bad_repo", reason)

    with tempfile.TemporaryDirectory(prefix="mucli-best-of-") as tmp:
        out_paths = {
            i: os.path.join(tmp, f"attempt-{i}.md")
            for i in range(1, attempts + 1)
        }
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=attempts) as pool:
            futures = {
                pool.submit(_run_attempt, task, repo_abs, path, i): i
                for i, path in out_paths.items()
            }
            for fut in as_completed(futures):
                attempt_no = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001 — one broken attempt
                    # must not abort the whole race; record a failed attempt
                    logger.warning("best_of_codex attempt %s raised: %s", futures[fut], exc)
                    results.append({
                        "attempt": futures[fut],
                        "ok": False,
                        "returncode": None,
                        "elapsed_seconds": 0.0,
                        "output_chars": 0,
                        "output_path": None,
                        "output": f"attempt crashed: {exc}",
                    })

        ranked = _rank(results)
        winner = ranked[0] if ranked else None
        any_ok = any(r["ok"] for r in results)
        logger.info(
            "best_of_codex: %d attempts, winner=attempt %s (%d chars, %.1fs)",
            len(ranked),
            winner.get("attempt") if winner else "?",
            winner.get("output_chars", 0) if winner else 0,
            winner.get("elapsed_seconds", 0.0) if winner else 0.0,
        )
        return _envelope(
            any_ok,
            None if any_ok else "all_failed",
            (
                f"best-of-{attempts}: winner attempt {winner['attempt']} "
                f"({winner['output_chars']} chars, {winner['elapsed_seconds']}s)"
                if winner
                else "all attempts failed"
            ),
            data={
                "attempts": attempts,
                "ranking": [
                    {
                        "attempt": r["attempt"],
                        "ok": r["ok"],
                        "output_chars": r["output_chars"],
                        "elapsed_seconds": r["elapsed_seconds"],
                    }
                    for r in ranked
                ],
                "winner_attempt": winner["attempt"] if winner else None,
                "winner_output": winner["output"] if winner else "",
            },
        )
