"""Deterministic verification and durable evidence manifests for jobs."""

from __future__ import annotations

import json
import os
import sqlite3
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from utils.config import HISTORY_DIR

from .models import Job
from .service import JobService
from .store import JobStore


OUTPUT_LIMIT = 24000


def _bounded(text: str, limit: int = OUTPUT_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    half = max(1, (limit - 80) // 2)
    return value[:half] + "\n… [verification output truncated] …\n" + value[-half:]


@dataclass(frozen=True)
class VerificationCheck:
    command: str
    return_code: Optional[int]
    passed: bool
    timed_out: bool
    duration_ms: int
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationRun:
    id: str
    job_id: str
    status: str
    passed: bool
    started_at: float
    finished_at: float
    duration_ms: int
    base_sha: str
    head_sha: str
    branch: str
    worktree: str
    checks: List[VerificationCheck] = field(default_factory=list)
    changed_files: List[Dict[str, Any]] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    diff_stat: str = ""
    dirty: bool = False
    dirty_status: str = ""
    manifest_path: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["checks"] = [check.to_dict() for check in self.checks]
        return value


class VerificationStore:
    def __init__(self, store: JobStore, *, evidence_root: Optional[str] = None):
        self.store = store
        self.evidence_root = os.path.abspath(
            os.path.expanduser(evidence_root or os.path.join(HISTORY_DIR, "jobs", "evidence"))
        )
        os.makedirs(self.evidence_root, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.store.path, timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def initialize(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS job_verifications (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    base_sha TEXT NOT NULL DEFAULT '',
                    head_sha TEXT NOT NULL DEFAULT '',
                    branch TEXT NOT NULL DEFAULT '',
                    worktree TEXT NOT NULL DEFAULT '',
                    checks_json TEXT NOT NULL DEFAULT '[]',
                    changed_files_json TEXT NOT NULL DEFAULT '[]',
                    additions INTEGER NOT NULL DEFAULT 0,
                    deletions INTEGER NOT NULL DEFAULT 0,
                    diff_stat TEXT NOT NULL DEFAULT '',
                    dirty INTEGER NOT NULL DEFAULT 0,
                    dirty_status TEXT NOT NULL DEFAULT '',
                    manifest_path TEXT NOT NULL DEFAULT '',
                    summary_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS job_verifications_job_idx
                    ON job_verifications(job_id, finished_at DESC);
                """
            )
        finally:
            conn.close()

    def save(self, run: VerificationRun) -> VerificationRun:
        job_dir = os.path.join(self.evidence_root, run.job_id)
        os.makedirs(job_dir, exist_ok=True)
        manifest_path = run.manifest_path or os.path.join(job_dir, f"{run.id}.json")
        value = run.to_dict()
        value["manifest_path"] = manifest_path

        # DB row first, then the manifest: the row is the index of record;
        # a crash before the rename leaves only an orphan tmp file, never a
        # half-written manifest at its final path or a truncated replacement
        # of prior evidence.
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR REPLACE INTO job_verifications (
                    id, job_id, status, passed, started_at, finished_at, duration_ms,
                    base_sha, head_sha, branch, worktree, checks_json,
                    changed_files_json, additions, deletions, diff_stat, dirty,
                    dirty_status, manifest_path, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id, run.job_id, run.status, int(run.passed), run.started_at,
                    run.finished_at, run.duration_ms, run.base_sha, run.head_sha,
                    run.branch, run.worktree,
                    json.dumps([c.to_dict() for c in run.checks], ensure_ascii=False),
                    json.dumps(run.changed_files, ensure_ascii=False),
                    int(run.additions), int(run.deletions), run.diff_stat, int(run.dirty),
                    run.dirty_status, manifest_path,
                    json.dumps(run.summary, ensure_ascii=False, default=str),
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

        # Atomically publish the manifest only after the row is durable:
        # tmp file in the same directory, fsync, rename.
        tmp_path = f"{manifest_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, manifest_path)
        return self.get(run.id)

    def get(self, verification_id: str) -> VerificationRun:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM job_verifications WHERE id = ?",
                (verification_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(verification_id)
        return self._from_row(row)

    def list(self, job_id: str, *, limit: int = 50) -> List[VerificationRun]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM job_verifications
                WHERE job_id = ? ORDER BY finished_at DESC LIMIT ?
                """,
                (job_id, max(1, min(int(limit), 500))),
            ).fetchall()
        finally:
            conn.close()
        return [self._from_row(row) for row in rows]

    def latest(self, job_id: str) -> Optional[VerificationRun]:
        values = self.list(job_id, limit=1)
        return values[0] if values else None

    @staticmethod
    def _from_row(row: sqlite3.Row) -> VerificationRun:
        checks_raw = json.loads(row["checks_json"] or "[]")
        return VerificationRun(
            id=row["id"],
            job_id=row["job_id"],
            status=row["status"],
            passed=bool(row["passed"]),
            started_at=float(row["started_at"]),
            finished_at=float(row["finished_at"]),
            duration_ms=int(row["duration_ms"]),
            base_sha=row["base_sha"],
            head_sha=row["head_sha"],
            branch=row["branch"],
            worktree=row["worktree"],
            checks=[VerificationCheck(**item) for item in checks_raw],
            changed_files=json.loads(row["changed_files_json"] or "[]"),
            additions=int(row["additions"]),
            deletions=int(row["deletions"]),
            diff_stat=row["diff_stat"],
            dirty=bool(row["dirty"]),
            dirty_status=row["dirty_status"],
            manifest_path=row["manifest_path"],
            summary=json.loads(row["summary_json"] or "{}"),
        )


def _kill_process_group(pgid: int) -> None:
    """Best-effort SIGTERM→SIGKILL of an entire process group. Used after a
    validation-command timeout: subprocess kills only the shell, so without
    this its spawned descendants would survive and could keep mutating the
    worktree after verification inspects it."""
    try:
        import signal
        import time as _time

        os.killpg(pgid, signal.SIGTERM)
        _time.sleep(0.2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # Group already gone or restricted - nothing further to do.
        pass
    except AttributeError:
        # Non-POSIX platform without os.killpg: at minimum kill the shell
        # itself (pid == pgid here) so the caller's follow-up communicate()
        # cannot block forever. Callers also pass a bounded timeout to the
        # follow-up communicate() as a second net.
        try:
            os.kill(pgid, signal.SIGTERM)
            _time.sleep(0.2)
            os.kill(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def ensure_safe_job_id(job_id: str) -> str:
    """Reject job ids that could escape the worktree/evidence roots when
    joined into paths (absolute ids, '.'/'..' path components, separators).
    Harmless consecutive dots inside one component (e.g. release..v2) pass."""
    if not job_id or not _SAFE_JOB_ID.match(job_id):
        raise ValueError(
            f"unsafe job_id {job_id!r}: must match {_SAFE_JOB_ID.pattern}"
        )
    # The regex already bans '/' and '\', so the only traversal risk is a
    # '.'/'..' component: a dot-leading id or an id made entirely of dots.
    if job_id.startswith(".") or set(job_id) == {"."}:
        raise ValueError(
            f"unsafe job_id {job_id!r}: '.'/'..' path components are not allowed"
        )
    return job_id


_STREAM_BUF_SIZE = 1 << 16  # 64 KiB read chunks
_MAX_CAPTURE = 8 << 20  # 8 MiB hard cap per stream before truncation


class _CappedReader:
    """Drain a pipe on a thread, retaining head+tail within a hard cap so a
    command emitting unlimited output cannot exhaust worker memory."""

    def __init__(self, pipe, cap: int = _MAX_CAPTURE):
        self._pipe = pipe
        self._cap = cap
        self._head = deque(maxlen=200)
        self._tail = deque(maxlen=200)
        self._total = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            for line in iter(self._pipe.readline, ""):
                self._total += len(line)
                if self._total <= self._cap:
                    self._head.append(line)
                self._tail.append(line)
        except Exception:
            pass
        finally:
            try:
                self._pipe.close()
            except Exception:
                pass

    def join(self, timeout: float = 5.0) -> tuple:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._total <= self._cap:
            return "".join(self._head), False
        # Over the cap: head + marker + tail, bounded.
        head_text = "".join(self._head)
        tail_text = "".join(self._tail)
        marker = f"\n... [{self._total} bytes total, output truncated]\n"
        return head_text[: self._cap // 2] + marker + tail_text[-self._cap // 2 :], True


class DeterministicVerifier:
    def __init__(self, service: JobService, *, store: Optional[VerificationStore] = None):
        self.service = service
        self.store = store or VerificationStore(service.store)

    @staticmethod
    def _git(worktree: str, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", worktree, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "git command failed").strip())
        return (result.stdout or "").strip()

    @staticmethod
    def _changed_files(worktree: str, base_sha: str, head_sha: str) -> tuple[List[Dict[str, Any]], int, int]:
        if not base_sha or not head_sha:
            return [], 0, 0
        status_text = DeterministicVerifier._git(
            worktree, "diff", "--name-status", f"{base_sha}..{head_sha}", check=False
        )
        numstat_text = DeterministicVerifier._git(
            worktree, "diff", "--numstat", f"{base_sha}..{head_sha}", check=False
        )
        stats: Dict[str, tuple[Optional[int], Optional[int]]] = {}
        additions = 0
        deletions = 0
        for line in numstat_text.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            add_raw, del_raw, path = parts
            add = int(add_raw) if add_raw.isdigit() else None
            delete = int(del_raw) if del_raw.isdigit() else None
            stats[path] = (add, delete)
            additions += add or 0
            deletions += delete or 0
        changed: List[Dict[str, Any]] = []
        for line in status_text.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            change = parts[0]
            path = parts[-1]
            add, delete = stats.get(path, (None, None))
            changed.append({
                "status": change,
                "path": path,
                "additions": add,
                "deletions": delete,
            })
        return changed, additions, deletions

    def verify(self, job: Job) -> VerificationRun:
        if not job.worktree or not os.path.isdir(job.worktree):
            raise RuntimeError(f"Job worktree is unavailable: {job.worktree or 'not prepared'}")
        started_at = time.time()
        monotonic_start = time.monotonic()
        verification_id = uuid.uuid4().hex
        head_before = self._git(job.worktree, "rev-parse", "HEAD^{commit}")
        checks: List[VerificationCheck] = []
        timeout_seconds = int(job.execution.get("validation_timeout_seconds", 600) or 600)
        timeout_seconds = max(1, min(timeout_seconds, 3600))

        for command in job.validation_commands:
            command_start = time.monotonic()
            try:
                # Own process group: on timeout we must kill the entire
                # tree (shell + spawned descendants), not just the shell —
                # survivors could keep mutating the worktree after
                # verification inspects it.
                child = subprocess.Popen(
                    command,
                    cwd=job.worktree,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,  # child leads its own process group
                    bufsize=_STREAM_BUF_SIZE,
                )
                # Round-35 F5: initialize BEFORE the child starts — on a
                # timeout these are referenced by the outer TimeoutExpired
                # handler; when the FIRST command times out they were never
                # assigned, raising UnboundLocalError and aborting the whole
                # verifier (job stuck VERIFYING, relaunched forever).
                stdout = ""
                stderr = ""
                out_reader = _CappedReader(child.stdout)
                err_reader = _CappedReader(child.stderr)
                out_reader.start()
                err_reader.start()
                try:
                    child.wait(timeout=timeout_seconds)
                    stdout, _trunc1 = out_reader.join(timeout=5)
                    stderr, _trunc2 = err_reader.join(timeout=5)
                    result = subprocess.CompletedProcess(
                        command, child.returncode, stdout, stderr
                    )
                except subprocess.TimeoutExpired:
                    # subprocess.run would kill only the shell; kill the
                    # whole group so descendants die too.
                    _kill_process_group(child.pid)
                    # A descendant that escaped the group may still hold the
                    # pipe descriptors; bound this drain so the worker can
                    # never hang forever on it.
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            child.kill()
                        except OSError:
                            pass
                    # Round-35 F5: capture whatever the readers drained so
                    # the timed-out check's diagnostics are not lost.
                    stdout, _trunc1 = out_reader.join(timeout=2)
                    stderr, _trunc2 = err_reader.join(timeout=2)
                    raise
                checks.append(VerificationCheck(
                    command=command,
                    return_code=int(result.returncode),
                    passed=result.returncode == 0,
                    timed_out=False,
                    duration_ms=int((time.monotonic() - command_start) * 1000),
                    stdout=_bounded(result.stdout),
                    stderr=_bounded(result.stderr),
                ))
            except subprocess.TimeoutExpired:
                checks.append(VerificationCheck(
                    command=command,
                    return_code=None,
                    passed=False,
                    timed_out=True,
                    duration_ms=int((time.monotonic() - command_start) * 1000),
                    stdout=_bounded(stdout or ""),
                    stderr=_bounded(stderr or ""),
                    error=f"timed out after {timeout_seconds}s (process group killed)",
                ))
            except Exception as exc:
                checks.append(VerificationCheck(
                    command=command,
                    return_code=None,
                    passed=False,
                    timed_out=False,
                    duration_ms=int((time.monotonic() - command_start) * 1000),
                    error=str(exc),
                ))

        head_sha = self._git(job.worktree, "rev-parse", "HEAD^{commit}")
        dirty_status = self._git(job.worktree, "status", "--porcelain", check=False)
        dirty = bool(dirty_status.strip())
        changed_files, additions, deletions = self._changed_files(
            job.worktree, job.base_sha, head_sha
        )
        diff_stat = self._git(
            job.worktree,
            "diff",
            "--stat",
            f"{job.base_sha}..{head_sha}",
            check=False,
        ) if job.base_sha else ""

        has_contract = bool(job.validation_commands)
        checks_passed = has_contract and all(check.passed for check in checks)
        head_changed = head_before != head_sha
        # A validation command that commits (or anything else advancing HEAD
        # mid-run) means the final commit was never exercised by the checks
        # above — that is a verification failure, not a pass.
        passed = checks_passed and not dirty and not head_changed
        if not has_contract:
            status = "missing_contract"
        elif head_changed:
            status = "head_moved_during_verification"
        elif dirty:
            status = "dirty_worktree"
        elif passed:
            status = "passed"
        else:
            status = "failed"
        finished_at = time.time()
        summary = {
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.passed),
            "checks_failed": sum(1 for check in checks if not check.passed),
            "acceptance_criteria": list(job.acceptance_criteria),
            "acceptance_criteria_count": len(job.acceptance_criteria),
            "acceptance_criteria_machine_verified": False,
            "changed_files": len(changed_files),
            "additions": additions,
            "deletions": deletions,
            "head_changed_during_verification": head_changed,
            "head_moved_is_failure": True,
            "dirty_worktree": dirty,
        }
        run = VerificationRun(
            id=verification_id,
            job_id=job.id,
            status=status,
            passed=passed,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((time.monotonic() - monotonic_start) * 1000),
            base_sha=job.base_sha,
            head_sha=head_sha,
            branch=job.branch,
            worktree=job.worktree,
            checks=checks,
            changed_files=changed_files,
            additions=additions,
            deletions=deletions,
            diff_stat=diff_stat,
            dirty=dirty,
            dirty_status=_bounded(dirty_status, 8000),
            summary=summary,
        )
        saved = self.store.save(run)
        self.service.store.append_event(
            job.id,
            "verification_evidence_created",
            reason=saved.status,
            payload={
                "verification_id": saved.id,
                "passed": saved.passed,
                "manifest_path": saved.manifest_path,
                "summary": saved.summary,
            },
        )
        return saved
