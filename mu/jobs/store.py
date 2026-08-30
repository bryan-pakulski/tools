"""SQLite/WAL persistence for engineering jobs.

The store owns durability and transaction boundaries. It deliberately exposes
no UI concepts so GUI, TUI and mobile can share exactly the same lifecycle.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

from utils.config import HISTORY_DIR

from .models import AttentionReason, Job, JobAttempt, JobEvent, JobSpec, JobStatus, coerce_status


SCHEMA_VERSION = 2


def default_db_path() -> str:
    root = os.path.join(HISTORY_DIR, "jobs")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "jobs.sqlite3")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _loads(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


class JobStore:
    """Small durable repository with one SQLite transaction per mutation."""

    def __init__(self, path: Optional[str] = None, *, clock=time.time):
        self.path = os.path.abspath(os.path.expanduser(path or default_db_path()))
        self._clock = clock
        self._init_lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._init_lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS job_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        repository TEXT NOT NULL DEFAULT '',
                        base_branch TEXT NOT NULL DEFAULT 'main',
                        base_sha TEXT NOT NULL DEFAULT '',
                        acceptance_json TEXT NOT NULL DEFAULT '[]',
                        validation_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL,
                        attention_reason TEXT NOT NULL DEFAULT '',
                        attention_detail TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        started_at REAL,
                        completed_at REAL,
                        max_cost_usd REAL,
                        max_runtime_seconds INTEGER,
                        max_iterations INTEGER,
                        max_retries INTEGER NOT NULL DEFAULT 2,
                        max_subagents INTEGER,
                        cost_usd REAL NOT NULL DEFAULT 0,
                        branch TEXT NOT NULL DEFAULT '',
                        worktree TEXT NOT NULL DEFAULT '',
                        environment_json TEXT NOT NULL DEFAULT '{}',
                        execution_json TEXT NOT NULL DEFAULT '{}',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        session_name TEXT NOT NULL DEFAULT '',
                        worker_id TEXT NOT NULL DEFAULT '',
                        lease_expires_at REAL,
                        heartbeat_at REAL,
                        version INTEGER NOT NULL DEFAULT 1
                    );

                    CREATE INDEX IF NOT EXISTS jobs_status_idx
                        ON jobs(status, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS jobs_attention_idx
                        ON jobs(attention_reason, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS jobs_lease_idx
                        ON jobs(lease_expires_at);

                    CREATE TABLE IF NOT EXISTS job_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        event_type TEXT NOT NULL,
                        from_status TEXT,
                        to_status TEXT,
                        reason TEXT NOT NULL DEFAULT '',
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS job_events_job_idx
                        ON job_events(job_id, id);

                    CREATE TABLE IF NOT EXISTS job_attempts (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        number INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        session_name TEXT NOT NULL DEFAULT '',
                        worker_id TEXT NOT NULL DEFAULT '',
                        started_at REAL NOT NULL,
                        finished_at REAL,
                        error TEXT NOT NULL DEFAULT '',
                        cost_usd REAL NOT NULL DEFAULT 0,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(job_id, number)
                    );
                    CREATE INDEX IF NOT EXISTS job_attempts_job_idx
                        ON job_attempts(job_id, number);
                    """
                )
                columns = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if "execution_json" not in columns:
                    conn.execute(
                        "ALTER TABLE jobs ADD COLUMN execution_json TEXT NOT NULL DEFAULT '{}'"
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO job_meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            finally:
                conn.close()

    def create_job(self, spec: JobSpec, *, job_id: Optional[str] = None) -> Job:
        normalized = spec.normalized()
        now = float(self._clock())
        identifier = job_id or uuid.uuid4().hex
        # Job ids feed worktree/evidence/receipt paths; reject ids that
        # could escape those roots (absolute paths, '..', separators).
        from .verification import ensure_safe_job_id

        ensure_safe_job_id(identifier)
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, title, description, repository, base_branch, base_sha,
                    acceptance_json, validation_json, status, created_at, updated_at,
                    max_cost_usd, max_runtime_seconds, max_iterations, max_retries,
                    max_subagents, environment_json, execution_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    normalized.title,
                    normalized.description,
                    normalized.repository,
                    normalized.base_branch,
                    normalized.base_sha,
                    _json(normalized.acceptance_criteria),
                    _json(normalized.validation_commands),
                    JobStatus.QUEUED.value,
                    now,
                    now,
                    normalized.max_cost_usd,
                    normalized.max_runtime_seconds,
                    normalized.max_iterations,
                    normalized.max_retries,
                    normalized.max_subagents,
                    _json(normalized.environment),
                    _json(normalized.execution),
                    _json(normalized.metadata),
                ),
            )
            self._insert_event(
                conn,
                identifier,
                "job_created",
                None,
                JobStatus.QUEUED,
                "",
                {"title": normalized.title, "execution": normalized.execution},
                now,
            )
        return self.get_job(identifier)

    def get_job(self, job_id: str) -> Job:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(job_id)
        return self._job_from_row(row)

    def list_jobs(
        self,
        *,
        statuses: Optional[Iterable[JobStatus | str]] = None,
        limit: int = 200,
    ) -> List[Job]:
        params: List[Any] = []
        query = "SELECT * FROM jobs"
        if statuses:
            values = [coerce_status(status).value for status in statuses]
            query += " WHERE status IN (%s)" % ",".join("?" for _ in values)
            params.extend(values)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        conn = self._connect()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        return [self._job_from_row(row) for row in rows]

    def list_unarchived_jobs(
        self,
        *,
        statuses: Optional[Iterable[JobStatus | str]] = None,
        limit: int = 200,
    ) -> List[Job]:
        """Jobs not archived in job_management (round-49 F10).

        The archived filter runs as a NOT EXISTS subquery — the board
        previously materialized the entire archived-id set in Python on
        every GUI poll, and that set grows without bound.
        """
        params: List[Any] = []
        query = (
            "SELECT * FROM jobs WHERE NOT EXISTS ("
            "SELECT 1 FROM job_management m WHERE m.job_id = jobs.id "
            "AND m.archived_at IS NOT NULL)"
        )
        if statuses:
            values = [coerce_status(status).value for status in statuses]
            query += " AND status IN (%s)" % ",".join("?" for _ in values)
            params.extend(values)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        conn = self._connect()
        try:
            try:
                rows = conn.execute(query, params).fetchall()
            except sqlite3.OperationalError:
                # job_management is created lazily by JobManagementService —
                # a store touched before any management op has no table, so
                # nothing is archived and the plain listing applies.
                fallback = "SELECT * FROM jobs"
                fallback_params: List[Any] = []
                if statuses:
                    values = [coerce_status(status).value for status in statuses]
                    fallback += " WHERE status IN (%s)" % ",".join("?" for _ in values)
                    fallback_params.extend(values)
                fallback += " ORDER BY updated_at DESC LIMIT ?"
                fallback_params.append(max(1, min(int(limit), 1000)))
                rows = conn.execute(fallback, fallback_params).fetchall()
        finally:
            conn.close()
        return [self._job_from_row(row) for row in rows]

    def transition(
        self,
        job_id: str,
        target: JobStatus | str,
        *,
        reason: str = "",
        payload: Optional[Dict[str, Any]] = None,
        attention_reason: AttentionReason | str = AttentionReason.NONE,
        attention_detail: str = "",
        expected_version: Optional[int] = None,
    ) -> Job:
        target_status = coerce_status(target)
        now = float(self._clock())
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = JobStatus(row["status"])
            version = int(row["version"])
            if expected_version is not None and version != int(expected_version):
                raise RuntimeError(
                    f"job {job_id} version changed: expected {expected_version}, found {version}"
                )
            # Validate the transition against the row actually observed
            # inside this write transaction — not a snapshot read earlier —
            # so a concurrent cancellation/recovery cannot be overwritten by
            # a transition that is illegal from the true current state.
            from .models import can_transition

            if current != target_status and not can_transition(current, target_status):
                from .service import JobStateError

                raise JobStateError(
                    f"cannot transition job {job_id} from {current.value} to {target_status.value}"
                )

            attention = (
                attention_reason
                if isinstance(attention_reason, AttentionReason)
                else AttentionReason(str(attention_reason or ""))
            )
            started_at = row["started_at"]
            completed_at = row["completed_at"]
            if target_status == JobStatus.RUNNING and started_at is None:
                started_at = now
            if target_status in {JobStatus.MERGED, JobStatus.CANCELLED}:
                completed_at = now
            elif target_status not in {JobStatus.READY_FOR_REVIEW}:
                completed_at = None

            conn.execute(
                """
                UPDATE jobs
                SET status = ?, attention_reason = ?, attention_detail = ?,
                    updated_at = ?, started_at = ?, completed_at = ?, version = version + 1
                WHERE id = ?
                """,
                (
                    target_status.value,
                    attention.value,
                    str(attention_detail or ""),
                    now,
                    started_at,
                    completed_at,
                    job_id,
                ),
            )
            self._insert_event(
                conn, job_id, "status_changed", current, target_status,
                reason, payload or {}, now,
            )
        return self.get_job(job_id)

    def update_runtime_fields(self, job_id: str, **fields: Any) -> Job:
        allowed = {
            "base_sha", "cost_usd", "branch", "worktree", "environment_json",
            "execution_json", "metadata_json", "session_name",
        }
        updates: Dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"unsupported job field: {key}")
            if key in {"environment_json", "execution_json", "metadata_json"} and not isinstance(value, str):
                value = _json(value or {})
            updates[key] = value
        if not updates:
            return self.get_job(job_id)
        updates["updated_at"] = float(self._clock())
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            conn.execute(
                f"UPDATE jobs SET {assignments}, version = version + 1 WHERE id = ?",
                [*updates.values(), job_id],
            )
        return self.get_job(job_id)

    def append_event(
        self,
        job_id: str,
        event_type: str,
        *,
        reason: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> JobEvent:
        now = float(self._clock())
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            event_id = self._insert_event(
                conn, job_id, event_type, None, None, reason, payload or {}, now
            )
        return self.get_event(event_id)

    def claim_interaction_response(
        self,
        job_id: str,
        response_event_id: int,
        *,
        kind: str = "",
        tool_name: str = "",
    ) -> bool:
        """Round-41 F7: atomically claim an interaction response. Exactly
        ONE caller across all JobUI instances/processes wins: the consumed
        marker is inserted inside a write transaction only if no earlier
        consumption of the same response_event_id exists. The losing
        claimant writes nothing and must re-poll for new input.

        This IS the consumption event (round-41 F7 fix-up): the caller
        must NOT append a second interaction_response_consumed event, or
        downstream consumers counting consumed markers double-count."""
        now = float(self._clock())
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            already = conn.execute(
                """
                SELECT 1 FROM job_events
                WHERE job_id = ? AND event_type = 'interaction_response_consumed'
                  AND json_extract(payload_json, '$.response_event_id') = ?
                """,
                (job_id, int(response_event_id)),
            ).fetchone()
            if already is not None:
                return False
            self._insert_event(
                conn, job_id, "interaction_response_consumed", None, None, "",
                {
                    "response_event_id": int(response_event_id),
                    "kind": str(kind or ""),
                    "tool_name": str(tool_name or ""),
                },
                now,
            )
        return True

    def get_event(self, event_id: int) -> JobEvent:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM job_events WHERE id = ?", (event_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(event_id)
        return self._event_from_row(row)

    def list_events(self, job_id: str, *, after_id: int = 0, limit: int = 500) -> List[JobEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (job_id, max(0, int(after_id)), max(1, min(int(limit), 5000))),
            ).fetchall()
        finally:
            conn.close()
        return [self._event_from_row(row) for row in rows]

    def acquire_lease(self, job_id: str, worker_id: str, *, ttl_seconds: int = 60) -> bool:
        now = float(self._clock())
        expires = now + max(5, int(ttl_seconds))
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT worker_id, lease_expires_at FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current_worker = str(row["worker_id"] or "")
            current_expiry = row["lease_expires_at"]
            if current_worker and current_worker != worker_id and current_expiry and float(current_expiry) > now:
                return False
            conn.execute(
                """
                UPDATE jobs SET worker_id = ?, lease_expires_at = ?, heartbeat_at = ?,
                    updated_at = ?, version = version + 1 WHERE id = ?
                """,
                (worker_id, expires, now, now, job_id),
            )
            self._insert_event(
                conn, job_id, "worker_lease_acquired", None, None, "",
                {"worker_id": worker_id, "lease_expires_at": expires}, now,
            )
        return True

    def heartbeat(self, job_id: str, worker_id: str, *, ttl_seconds: int = 60) -> bool:
        now = float(self._clock())
        expires = now + max(5, int(ttl_seconds))
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE id = ? AND worker_id = ?
                """,
                (expires, now, now, job_id, worker_id),
            )
            return cursor.rowcount == 1

    def assert_lease(self, job_id: str, worker_id: str) -> bool:
        """Transactional ownership check: True iff worker_id currently owns
        the job's lease AND the lease has not expired, evaluated inside a
        write transaction. Used by workers/verifiers as the final gate
        before applying results.

        Round-36 F2: an expired lease is NOT ownership. Without the expiry
        clause, a worker whose heartbeat thread died (GC pause, blocked
        event loop) passed this gate and applied results to a job whose
        lease had already lapsed — racing the takeover worker."""
        now = float(self._clock())
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT worker_id, lease_expires_at FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["worker_id"] != worker_id:
                return False
            expires = row["lease_expires_at"]
            return expires is not None and float(expires) >= now

    def finish_attempt_owned(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        status: str,
        error: str = "",
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        target_status: Optional["JobStatus"] = None,
        transition_reason: str = "",
        transition_payload: Optional[Dict[str, Any]] = None,
        attention_reason: Optional["AttentionReason"] = None,
        attention_detail: str = "",
        cost_add: float = 0.0,
    ) -> bool:
        """Round-36 F2: lease-conditioned atomic attempt finish + optional
        job transition. Returns False (nothing written) when the job's
        lease is no longer held unexpired by `worker_id` — closing the
        check-then-act window between assert_lease() and finish_attempt()
        where a takeover could acquire the lease and this worker would
        still stamp its superseded outcome onto the replacement's state.

        When target_status is given, the transition runs in the SAME
        transaction, revalidating can_transition against the row observed
        under the lock (a cancelled/merged job is left untouched and False
        is returned only if the lease itself was lost)."""
        from .models import JobStatus as _JS, can_transition
        from .service import JobStateError

        now = float(self._clock())
        with self._transaction() as conn:
            attempt_row = conn.execute(
                "SELECT * FROM job_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if attempt_row is None:
                return False
            job_id = attempt_row["job_id"]
            row = conn.execute(
                "SELECT worker_id, lease_expires_at FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return False
            if row["worker_id"] != worker_id:
                return False
            expires = row["lease_expires_at"]
            if expires is None or float(expires) < now:
                return False
            combined = _loads(attempt_row["metadata_json"], {})
            combined.update(metadata or {})
            # Round-40 F1: accumulate job cost INSIDE the same
            # lease-conditioned transaction — a superseded worker must not
            # add its cost to the replacement's job. The resulting total is
            # stamped into the attempt metadata (callers no longer need a
            # separate pre-transaction cost read).
            new_total = float(cost_usd)
            if cost_add:
                cost_row = conn.execute(
                    "SELECT cost_usd FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                new_total = float(cost_row["cost_usd"] or 0.0) + max(0.0, float(cost_add or 0.0))
                conn.execute(
                    "UPDATE jobs SET cost_usd = ?, updated_at = ? WHERE id = ?",
                    (new_total, now, job_id),
                )
                combined.setdefault("total_job_cost_usd", new_total)
            conn.execute(
                """
                UPDATE job_attempts SET status = ?, finished_at = ?, error = ?,
                    cost_usd = ?, metadata_json = ? WHERE id = ?
                """,
                (str(status), now, str(error or ""), float(cost_usd), _json(combined), attempt_id),
            )
            self._insert_event(
                conn, job_id, "attempt_finished", None, None, str(status),
                {"attempt_id": attempt_id, "cost_usd": float(cost_usd), "error": str(error or "")}, now,
            )
            if target_status is not None:
                target = coerce_status(target_status)
                current = JobStatus(conn.execute(
                    "SELECT status FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()["status"])
                if current != target and not can_transition(current, target):
                    raise JobStateError(
                        f"cannot transition job {job_id} from {current.value} to {target.value}"
                    )
                attention_value = (
                    attention_reason.value
                    if isinstance(attention_reason, AttentionReason)
                    else (str(attention_reason) if attention_reason else None)
                )
                if attention_value is not None:
                    conn.execute(
                        """
                        UPDATE jobs SET status = ?, attention_reason = ?,
                            attention_detail = ?, updated_at = ?, version = version + 1
                        WHERE id = ?
                        """,
                        (target.value, attention_value, str(attention_detail or ""), now, job_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs SET status = ?, updated_at = ?, version = version + 1
                        WHERE id = ?
                        """,
                        (target.value, now, job_id),
                    )
                self._insert_event(
                    conn, job_id, "status_changed", current, target,
                    transition_reason, transition_payload or {}, now,
                )
        return True

    def release_lease(self, job_id: str, worker_id: str, *, reason: str = "") -> bool:
        now = float(self._clock())
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET worker_id = '', lease_expires_at = NULL, heartbeat_at = ?,
                    updated_at = ?, version = version + 1
                WHERE id = ? AND worker_id = ?
                """,
                (now, now, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                return False
            self._insert_event(
                conn, job_id, "worker_lease_released", None, None, reason,
                {"worker_id": worker_id}, now,
            )
        return True

    def release_expired_lease(
        self, job_id: str, worker_id: str, *, reason: str = ""
    ) -> bool:
        """CAS release: clear the lease only if it is still held by
        `worker_id` AND still expired at write time. Prevents the TOCTOU
        where a worker renews its lease between the caller's
        `expired_leases()` read and this release, losing the fresh lease."""
        now = float(self._clock())
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET worker_id = '', lease_expires_at = NULL, heartbeat_at = ?,
                    updated_at = ?, version = version + 1
                WHERE id = ? AND worker_id = ?
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (now, now, job_id, worker_id, now),
            )
            if cursor.rowcount != 1:
                return False
            self._insert_event(
                conn, job_id, "worker_lease_released", None, None, reason,
                {"worker_id": worker_id}, now,
            )
        return True

    def expired_leases(self, *, now: Optional[float] = None, limit: int = 100) -> List[Job]:
        """Expired leases, oldest expiry first, BOUNDED (round-49 F11).

        Recovery after downtime previously materialized EVERY expired lease
        in one query — hundreds of jobs made the restart pass unbounded
        (plus a per-job attempt query downstream). Bounded batches ordered
        by expiry let the controller drain the backlog incrementally per
        poll; callers needing the full set pass a larger limit.
        """
        timestamp = float(self._clock() if now is None else now)
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE worker_id != '' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                ORDER BY lease_expires_at ASC
                LIMIT ?
                """,
                (timestamp, max(1, min(int(limit), 500))),
            ).fetchall()
        finally:
            conn.close()
        return [self._job_from_row(row) for row in rows]

    def start_attempt(
        self,
        job_id: str,
        *,
        worker_id: str = "",
        session_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JobAttempt:
        now = float(self._clock())
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(number), 0) AS n FROM job_attempts WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            number = int(row["n"]) + 1
            attempt_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO job_attempts (
                    id, job_id, number, status, session_name, worker_id,
                    started_at, metadata_json
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (attempt_id, job_id, number, session_name, worker_id, now, _json(metadata or {})),
            )
            self._insert_event(
                conn, job_id, "attempt_started", None, None, "",
                {"attempt_id": attempt_id, "number": number, "worker_id": worker_id}, now,
            )
        return self.get_attempt(attempt_id)

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        error: str = "",
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JobAttempt:
        now = float(self._clock())
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM job_attempts WHERE id = ?", (attempt_id,)).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            combined = _loads(row["metadata_json"], {})
            combined.update(metadata or {})
            conn.execute(
                """
                UPDATE job_attempts SET status = ?, finished_at = ?, error = ?,
                    cost_usd = ?, metadata_json = ? WHERE id = ?
                """,
                (str(status), now, str(error or ""), float(cost_usd), _json(combined), attempt_id),
            )
            self._insert_event(
                conn, row["job_id"], "attempt_finished", None, None, str(status),
                {"attempt_id": attempt_id, "cost_usd": float(cost_usd), "error": str(error or "")}, now,
            )
        return self.get_attempt(attempt_id)

    def get_attempt(self, attempt_id: str) -> JobAttempt:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM job_attempts WHERE id = ?", (attempt_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(attempt_id)
        return self._attempt_from_row(row)

    def list_attempts(self, job_id: str) -> List[JobAttempt]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY number ASC", (job_id,)
            ).fetchall()
        finally:
            conn.close()
        return [self._attempt_from_row(row) for row in rows]

    @staticmethod
    def _insert_event(
        conn: sqlite3.Connection,
        job_id: str,
        event_type: str,
        from_status: Optional[JobStatus],
        to_status: Optional[JobStatus],
        reason: str,
        payload: Dict[str, Any],
        created_at: float,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO job_events (
                job_id, event_type, from_status, to_status, reason, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, event_type,
                from_status.value if from_status else None,
                to_status.value if to_status else None,
                str(reason or ""), _json(payload or {}), created_at,
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"], title=row["title"], description=row["description"],
            repository=row["repository"], base_branch=row["base_branch"], base_sha=row["base_sha"],
            acceptance_criteria=_loads(row["acceptance_json"], []),
            validation_commands=_loads(row["validation_json"], []),
            status=JobStatus(row["status"]),
            attention_reason=AttentionReason(row["attention_reason"] or ""),
            attention_detail=row["attention_detail"],
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
            started_at=row["started_at"], completed_at=row["completed_at"],
            max_cost_usd=row["max_cost_usd"], max_runtime_seconds=row["max_runtime_seconds"],
            max_iterations=row["max_iterations"], max_retries=int(row["max_retries"]),
            max_subagents=row["max_subagents"], cost_usd=float(row["cost_usd"] or 0),
            branch=row["branch"], worktree=row["worktree"],
            environment=_loads(row["environment_json"], {}),
            execution=_loads(row["execution_json"], {}),
            metadata=_loads(row["metadata_json"], {}),
            session_name=row["session_name"], worker_id=row["worker_id"],
            lease_expires_at=row["lease_expires_at"], heartbeat_at=row["heartbeat_at"],
            version=int(row["version"]),
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> JobEvent:
        return JobEvent(
            id=int(row["id"]), job_id=row["job_id"], event_type=row["event_type"],
            from_status=JobStatus(row["from_status"]) if row["from_status"] else None,
            to_status=JobStatus(row["to_status"]) if row["to_status"] else None,
            reason=row["reason"], payload=_loads(row["payload_json"], {}),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> JobAttempt:
        return JobAttempt(
            id=row["id"], job_id=row["job_id"], number=int(row["number"]),
            status=row["status"], session_name=row["session_name"], worker_id=row["worker_id"],
            started_at=float(row["started_at"]), finished_at=row["finished_at"], error=row["error"],
            cost_usd=float(row["cost_usd"] or 0), metadata=_loads(row["metadata_json"], {}),
        )
