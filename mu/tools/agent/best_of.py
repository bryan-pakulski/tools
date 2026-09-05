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
# Legacy hard cap. No longer used as a kill switch by default: codex
# attempts may legitimately run for hours on large repos. The attempt
# runner now polls liveness + output growth and only gives up when the
# process dies, the configured hard cap elapses, or the run stalls (no
# output growth for ``PER_ATTEMPT_STALL_SECONDS``). 0 = unlimited.
PER_ATTEMPT_TIMEOUT_SECONDS = 0
# A codex attempt that has produced no output growth for this long is
# presumed wedged and killed; its partial output is preserved.
PER_ATTEMPT_STALL_SECONDS = 900
_POLL_INTERVAL_SECONDS = 5.0


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def _read_output(out_path: str) -> str:
    if not os.path.exists(out_path):
        return ""
    try:
        with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _run_attempt(
    task: str,
    repo: str,
    out_path: str,
    attempt: int,
    *,
    timeout_seconds: int | None = None,
    stall_seconds: int = PER_ATTEMPT_STALL_SECONDS,
) -> Dict[str, Any]:
    """Run one codex attempt with a liveness/progress poll loop instead of a
    blind ``subprocess.run(timeout=...)`` kill.

    Termination conditions:
      * process exits normally (returncode recorded);
      * ``timeout_seconds`` hard cap elapsed (0/None = unlimited);
      * no output-file growth for ``stall_seconds`` — presumed wedged, killed,
        partial output preserved.

    Codex attempts on large tasks can legitimately run for hours, so the
    historical hard 600s kill produced spurious timeouts. The stall check
    replaces it: a live run that is still writing output is left alone.
    """
    from mu.tools._scrub import scrub_and_annotate as _scrub

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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return {
            "attempt": attempt,
            "ok": False,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "output_chars": 0,
            "output_path": None,
            "output": f"launch failed: {exc}",
        }

    hard_cap = timeout_seconds if (timeout_seconds or 0) > 0 else 0
    stalled = False
    capped = False
    last_size = -1
    last_progress = time.monotonic()
    try:
        while proc.poll() is None:
            time.sleep(_POLL_INTERVAL_SECONDS)
            size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            if size > last_size:
                last_size = size
                last_progress = time.monotonic()
            now = time.monotonic()
            if hard_cap and (now - started) >= hard_cap:
                capped = True
                proc.kill()
                break
            if stall_seconds > 0 and (now - last_progress) >= stall_seconds:
                stalled = True
                proc.kill()
                break
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001 — poll/kill path must not raise
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.monotonic() - started
    output = _read_output(out_path)
    stderr_tail = ""
    try:
        if proc.stderr is not None:
            stderr_tail = (proc.stderr.read() or "")[-1500:]
    except Exception:  # noqa: BLE001
        stderr_tail = ""
    try:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    except Exception:  # noqa: BLE001
        pass

    # Scrub returned text (codex round-8 F3): attempt output and
    # stderr go straight to the model — secrets printed by codex
    # must be redacted like shell-tool results.
    output = str(_scrub(output)) if output else ""
    stderr_tail = str(_scrub(stderr_tail)) if stderr_tail else ""

    if capped:
        return {
            "attempt": attempt,
            "ok": False,
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "output_chars": len(output),
            "output_path": out_path if output else None,
            "output": (
                output[:4000]
                if output
                else f"hard timeout after {int(hard_cap)}s (partial output preserved)"
            ),
            "timeout": True,
            "timeout_reason": "hard_cap",
            "partial_output": bool(output),
        }
    if stalled:
        return {
            "attempt": attempt,
            "ok": False,
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "output_chars": len(output),
            "output_path": out_path if output else None,
            "output": (
                output[:4000]
                if output
                else (
                    f"killed: no output growth for {int(stall_seconds)}s "
                    f"(stall); partial output preserved at {out_path}"
                )
            ),
            "timeout": True,
            "timeout_reason": "stall",
            "partial_output": bool(output),
        }
    ok = proc.returncode == 0
    return {
        "attempt": attempt,
        "ok": ok,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "output_chars": len(output),
        "output_path": out_path if output else None,
        "output": output[:4000] if output else stderr_tail,
        "timeout": False,
        "timeout_reason": None,
        "partial_output": False,
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
            "timeout_seconds": {
                "type": "integer",
                "description": (
                    "Hard cap per attempt in seconds (0 or omitted = no cap; "
                    "stalled attempts are killed earlier by the stall "
                    "health-check)."
                ),
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

    # Per-attempt hard cap: explicit arg wins, else the session variable
    # ``codex_attempt_timeout_seconds`` (0 = unlimited, the default so long
    # codex runs are no longer killed at an arbitrary 600s).
    timeout_seconds = 0
    if args.get("timeout_seconds") is not None:
        try:
            timeout_seconds = max(0, int(args.get("timeout_seconds") or 0))
        except (TypeError, ValueError):
            timeout_seconds = 0
    else:
        session = getattr(context, "session", None)
        try:
            timeout_seconds = max(
                0, int((getattr(session, "variables", {}) or {}).get(
                    "codex_attempt_timeout_seconds", 0
                ) or 0)
            )
        except (TypeError, ValueError):
            timeout_seconds = 0

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
                pool.submit(
                    _run_attempt, task, repo_abs, path, i,
                    timeout_seconds=timeout_seconds,
                ): i
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
