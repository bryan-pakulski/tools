"""History management, reporting and debug export for durable engineering jobs.

Archiving is deliberately orthogonal to the execution state machine. A stopped
job can be hidden from the operational board without inventing another runtime
status, while its durable events/evidence remain queryable and exportable.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import time
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .diagnostics import build_job_diagnostics
from .models import Job, JobStatus
from .receipt import JobReceiptBuilder
from .review import JobReviewError, build_job_diff
from .service import JobService
from .verification import VerificationStore
from .worktree import JobWorktreeManager, WorktreeError


HISTORIC_STATUSES = frozenset(
    {
        JobStatus.FAILED,
        JobStatus.TIMED_OUT,
        JobStatus.BUDGET_EXCEEDED,
        JobStatus.ENVIRONMENT_ERROR,
        JobStatus.CANCELLED,
        JobStatus.MERGED,
    }
)


class JobManagementError(RuntimeError):
    pass


class JobManagementService:
    """Query/archive/delete/report facade shared by control planes."""

    def __init__(self, service: JobService):
        self.service = service
        self.store = service.store
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.store.path, timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def initialize(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS job_management (
                    job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                    archived_at REAL,
                    archived_reason TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS job_management_archived_idx
                    ON job_management(archived_at, updated_at DESC);
                """
            )
        finally:
            conn.close()

    @staticmethod
    def _status_values(values: Optional[Iterable[JobStatus | str]]) -> List[str]:
        if not values:
            return []
        out: List[str] = []
        for value in values:
            status = value if isinstance(value, JobStatus) else JobStatus(str(value))
            if status.value not in out:
                out.append(status.value)
        return out

    def _where(
        self,
        *,
        q: str = "",
        statuses: Optional[Iterable[JobStatus | str]] = None,
        repository: str = "",
        archive: str = "all",
        scope: str = "history",
        created_after: Optional[float] = None,
        created_before: Optional[float] = None,
    ) -> tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []

        status_values = self._status_values(statuses)
        if status_values:
            clauses.append("j.status IN (%s)" % ",".join("?" for _ in status_values))
            params.extend(status_values)

        scope_value = str(scope or "history").strip().lower()
        history_values = [status.value for status in HISTORIC_STATUSES]
        if scope_value == "history":
            clauses.append("j.status IN (%s)" % ",".join("?" for _ in history_values))
            params.extend(history_values)
        elif scope_value == "active":
            clauses.append("j.status NOT IN (%s)" % ",".join("?" for _ in history_values))
            params.extend(history_values)
        elif scope_value != "all":
            raise ValueError("scope must be history, active, or all")

        archive_value = str(archive or "all").strip().lower()
        if archive_value == "archived":
            clauses.append("m.archived_at IS NOT NULL")
        elif archive_value in {"current", "unarchived", "active"}:
            clauses.append("m.archived_at IS NULL")
        elif archive_value != "all":
            raise ValueError("archive must be all, archived, or current")

        text = str(q or "").strip()
        if text:
            needle = f"%{text}%"
            clauses.append(
                "(" + " OR ".join(
                    [
                        "j.id LIKE ?",
                        "j.title LIKE ?",
                        "j.description LIKE ?",
                        "j.repository LIKE ?",
                        "j.branch LIKE ?",
                        "j.session_name LIKE ?",
                    ]
                ) + ")"
            )
            params.extend([needle] * 6)

        repo = str(repository or "").strip()
        if repo:
            clauses.append("j.repository LIKE ?")
            params.append(f"%{repo}%")

        if created_after is not None:
            clauses.append("j.created_at >= ?")
            params.append(float(created_after))
        if created_before is not None:
            clauses.append("j.created_at <= ?")
            params.append(float(created_before))

        return (" AND ".join(clauses) if clauses else "1=1"), params

    def archived_ids(self) -> set[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT job_id FROM job_management WHERE archived_at IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        return {str(row["job_id"]) for row in rows}

    def state(self, job_id: str) -> Dict[str, Any]:
        self.service.get(job_id)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT archived_at, archived_reason, updated_at FROM job_management WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        return {
            "archived": bool(row and row["archived_at"] is not None),
            "archived_at": float(row["archived_at"]) if row and row["archived_at"] is not None else None,
            "archived_reason": str(row["archived_reason"] or "") if row else "",
            "management_updated_at": float(row["updated_at"]) if row else None,
        }

    def query_jobs(
        self,
        *,
        q: str = "",
        statuses: Optional[Iterable[JobStatus | str]] = None,
        repository: str = "",
        archive: str = "all",
        scope: str = "history",
        created_after: Optional[float] = None,
        created_before: Optional[float] = None,
        sort: str = "updated",
        order: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        where, params = self._where(
            q=q,
            statuses=statuses,
            repository=repository,
            archive=archive,
            scope=scope,
            created_after=created_after,
            created_before=created_before,
        )
        sort_column = {
            "updated": "j.updated_at",
            "created": "j.created_at",
            "cost": "j.cost_usd",
            "title": "j.title",
            "status": "j.status",
        }.get(str(sort or "updated").lower())
        if not sort_column:
            raise ValueError("sort must be updated, created, cost, title, or status")
        direction = "ASC" if str(order or "desc").lower() == "asc" else "DESC"
        bounded_limit = max(1, min(int(limit), 5000))
        bounded_offset = max(0, int(offset))

        conn = self._connect()
        try:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM jobs j LEFT JOIN job_management m ON m.job_id = j.id WHERE {where}",
                    params,
                ).fetchone()["n"]
            )
            rows = conn.execute(
                f"""
                SELECT j.*,
                       m.archived_at AS management_archived_at,
                       m.archived_reason AS management_archived_reason,
                       (SELECT COUNT(*) FROM job_attempts a WHERE a.job_id = j.id) AS attempt_count,
                       CASE WHEN j.started_at IS NULL THEN 0
                            ELSE MAX(0, COALESCE(j.completed_at, j.updated_at) - j.started_at)
                       END AS elapsed_seconds
                FROM jobs j
                LEFT JOIN job_management m ON m.job_id = j.id
                WHERE {where}
                ORDER BY {sort_column} {direction}, j.id ASC
                LIMIT ? OFFSET ?
                """,
                [*params, bounded_limit, bounded_offset],
            ).fetchall()
        finally:
            conn.close()

        jobs: List[Dict[str, Any]] = []
        for row in rows:
            item = self.store._job_from_row(row).to_dict()
            archived_at = row["management_archived_at"]
            item.update(
                {
                    "archived": archived_at is not None,
                    "archived_at": float(archived_at) if archived_at is not None else None,
                    "archived_reason": str(row["management_archived_reason"] or ""),
                    "attempt_count": int(row["attempt_count"] or 0),
                    "elapsed_seconds": float(row["elapsed_seconds"] or 0.0),
                    "manageable": item["status"] in {status.value for status in HISTORIC_STATUSES},
                }
            )
            jobs.append(item)
        return {
            "jobs": jobs,
            "total": total,
            "limit": bounded_limit,
            "offset": bounded_offset,
            "filters": {
                "q": q,
                "statuses": self._status_values(statuses),
                "repository": repository,
                "archive": archive,
                "scope": scope,
                "created_after": created_after,
                "created_before": created_before,
                "sort": sort,
                "order": direction.lower(),
            },
        }

    def report(self, **filters: Any) -> Dict[str, Any]:
        where, params = self._where(
            q=filters.get("q", ""),
            statuses=filters.get("statuses"),
            repository=filters.get("repository", ""),
            archive=filters.get("archive", "all"),
            scope=filters.get("scope", "history"),
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
        )
        failed_values = [
            JobStatus.FAILED.value,
            JobStatus.TIMED_OUT.value,
            JobStatus.BUDGET_EXCEEDED.value,
            JobStatus.ENVIRONMENT_ERROR.value,
        ]
        conn = self._connect()
        try:
            summary = conn.execute(
                f"""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN m.archived_at IS NOT NULL THEN 1 ELSE 0 END) AS archived,
                       COALESCE(SUM(j.cost_usd), 0) AS total_cost,
                       COALESCE(AVG(j.cost_usd), 0) AS average_cost,
                       COALESCE(AVG(CASE WHEN j.started_at IS NULL THEN NULL
                           ELSE MAX(0, COALESCE(j.completed_at, j.updated_at) - j.started_at) END), 0) AS average_elapsed
                FROM jobs j
                LEFT JOIN job_management m ON m.job_id = j.id
                WHERE {where}
                """,
                params,
            ).fetchone()
            status_rows = conn.execute(
                f"""
                SELECT j.status, COUNT(*) AS n
                FROM jobs j LEFT JOIN job_management m ON m.job_id = j.id
                WHERE {where}
                GROUP BY j.status ORDER BY n DESC, j.status ASC
                """,
                params,
            ).fetchall()
            repo_rows = conn.execute(
                f"""
                SELECT CASE WHEN TRIM(j.repository) = '' THEN '(no repository)' ELSE j.repository END AS repository,
                       COUNT(*) AS n, COALESCE(SUM(j.cost_usd), 0) AS cost
                FROM jobs j LEFT JOIN job_management m ON m.job_id = j.id
                WHERE {where}
                GROUP BY repository ORDER BY n DESC, cost DESC LIMIT 12
                """,
                params,
            ).fetchall()
            attempt_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) AS n
                    FROM job_attempts a
                    JOIN jobs j ON j.id = a.job_id
                    LEFT JOIN job_management m ON m.job_id = j.id
                    WHERE {where}
                    """,
                    params,
                ).fetchone()["n"]
            )
            failure_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) AS n
                    FROM jobs j LEFT JOIN job_management m ON m.job_id = j.id
                    WHERE {where} AND j.status IN ({','.join('?' for _ in failed_values)})
                    """,
                    [*params, *failed_values],
                ).fetchone()["n"]
            )
            verification_total = 0
            verification_passed = 0
            try:
                verification_row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS n, COALESCE(SUM(v.passed), 0) AS passed
                    FROM job_verifications v
                    JOIN jobs j ON j.id = v.job_id
                    LEFT JOIN job_management m ON m.job_id = j.id
                    WHERE {where}
                    """,
                    params,
                ).fetchone()
                verification_total = int(verification_row["n"] or 0)
                verification_passed = int(verification_row["passed"] or 0)
            except sqlite3.OperationalError:
                pass
        finally:
            conn.close()

        total = int(summary["total"] or 0)
        return {
            "total_jobs": total,
            "archived_jobs": int(summary["archived"] or 0),
            "total_cost_usd": float(summary["total_cost"] or 0.0),
            "average_cost_usd": float(summary["average_cost"] or 0.0),
            "average_elapsed_seconds": float(summary["average_elapsed"] or 0.0),
            "attempts": attempt_count,
            "failure_count": failure_count,
            "failure_rate": (failure_count / total) if total else 0.0,
            "verification_runs": verification_total,
            "verification_passed": verification_passed,
            "verification_pass_rate": (verification_passed / verification_total) if verification_total else 0.0,
            "status_counts": {str(row["status"]): int(row["n"]) for row in status_rows},
            "repositories": [
                {
                    "repository": str(row["repository"]),
                    "jobs": int(row["n"]),
                    "cost_usd": float(row["cost"] or 0.0),
                }
                for row in repo_rows
            ],
        }

    def archive(self, job_id: str, *, reason: str = "") -> Dict[str, Any]:
        job = self.service.get(job_id)
        if job.status not in HISTORIC_STATUSES:
            raise JobManagementError(
                f"Job {job.id} cannot be archived while it is {job.status.value}. Stop or finish it first."
            )
        if job.worker_id:
            raise JobManagementError("A job with an active worker lease cannot be archived.")
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO job_management(job_id, archived_at, archived_reason, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    archived_at = excluded.archived_at,
                    archived_reason = excluded.archived_reason,
                    updated_at = excluded.updated_at
                """,
                (job.id, now, str(reason or ""), now),
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
        self.store.append_event(
            job.id,
            "job_archived",
            reason=str(reason or "archived from Engineering Work"),
            payload={"archived_at": now},
        )
        return self.state(job.id)

    def restore(self, job_id: str) -> Dict[str, Any]:
        job = self.service.get(job_id)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM job_management WHERE job_id = ?", (job.id,))
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        self.store.append_event(
            job.id,
            "job_restored",
            reason="restored to Engineering Work history",
        )
        return self.state(job.id)

    def _artifact_paths(self, job_id: str) -> tuple[str, str]:
        root = os.path.dirname(self.store.path)
        return os.path.join(root, "evidence", job_id), os.path.join(root, "logs", f"{job_id}.log")

    def delete(self, job_id: str, *, purge_artifacts: bool = True) -> Dict[str, Any]:
        job = self.service.get(job_id)
        state = self.state(job_id)
        if job.status not in HISTORIC_STATUSES:
            raise JobManagementError(
                f"Job {job.id} cannot be deleted while it is {job.status.value}. Stop or finish it first."
            )
        if not state["archived"]:
            raise JobManagementError("Archive a historic job before deleting it.")
        if job.worker_id:
            raise JobManagementError("A job with an active worker lease cannot be deleted.")

        # Round-37 F1: DELETE the row FIRST (inside the transaction that
        # revalidates status/lease/archive), and only remove the worktree
        # afterwards. The old order — rmtree the worktree, then revalidate
        # and DELETE the row — had a destructive TOCTOU: a concurrent
        # requeue could move this archived job back to QUEUED and acquire
        # a lease after the snapshot checks but before the removal; the
        # deletion then destroyed the ACTIVE worker's checkout and the
        # transaction (correctly) refused the row delete, leaving a live
        # job with no execution environment. Now the row delete commits
        # first; the worktree of a deleted job is unreachable by workers
        # (start/retry require the row), so removal afterwards is safe.
        with self.store._transaction() as conn:
            # Re-verify inside the write transaction: a concurrent lease
            # acquisition, requeue, or archive flip between the checks
            # above and this DELETE must not let an active job be deleted
            # under a running worker.
            row = conn.execute(
                "SELECT status, worker_id FROM jobs WHERE id = ?",
                (job.id,),
            ).fetchone()
            if row is None:
                raise KeyError(job.id)
            if row["status"] not in {s.value for s in HISTORIC_STATUSES}:
                raise JobManagementError(
                    f"Job {job.id} can no longer be deleted: it is {row['status']}."
                )
            if row["worker_id"]:
                raise JobManagementError(
                    "A job with an active worker lease cannot be deleted."
                )
            mg = conn.execute(
                "SELECT archived_at FROM job_management WHERE job_id = ?",
                (job.id,),
            ).fetchone()
            if mg is None or mg["archived_at"] is None:
                raise JobManagementError(
                    "Archive a historic job before deleting it."
                )
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job.id,))
            if cursor.rowcount != 1:
                raise KeyError(job.id)

        worktree_removed = False
        if purge_artifacts and job.worktree and os.path.exists(job.worktree):
            try:
                worktree_removed = JobWorktreeManager(self.service).remove(job, force=False)
            except WorktreeError as exc:
                # The job row is already gone; the worktree is orphaned but
                # harmless (unregistered from any live job). Surface the
                # cleanup failure without corrupting the delete result.
                worktree_removed = False

        removed: List[str] = []
        warnings: List[str] = []
        if purge_artifacts:
            evidence_dir, log_path = self._artifact_paths(job.id)
            try:
                if os.path.isdir(evidence_dir):
                    shutil.rmtree(evidence_dir)
                    removed.append(evidence_dir)
            except OSError as exc:
                warnings.append(f"Could not remove evidence directory: {exc}")
            try:
                if os.path.isfile(log_path):
                    os.remove(log_path)
                    removed.append(log_path)
            except OSError as exc:
                warnings.append(f"Could not remove worker log: {exc}")
        return {
            "job_id": job.id,
            "deleted": True,
            "worktree_removed": worktree_removed,
            "artifacts_removed": removed,
            "warnings": warnings,
            "branch_preserved": job.branch,
        }

    def bulk(self, action: str, job_ids: Sequence[str], *, reason: str = "") -> Dict[str, Any]:
        operation = str(action or "").strip().lower()
        if operation not in {"archive", "restore", "delete"}:
            raise ValueError("action must be archive, restore, or delete")
        unique = list(dict.fromkeys(str(value or "").strip() for value in job_ids if str(value or "").strip()))
        if not unique:
            raise ValueError("job_ids is required")
        if len(unique) > 250:
            raise ValueError("bulk operations are limited to 250 jobs")
        results: List[Dict[str, Any]] = []
        for job_id in unique:
            try:
                if operation == "archive":
                    value = self.archive(job_id, reason=reason)
                elif operation == "restore":
                    value = self.restore(job_id)
                else:
                    value = self.delete(job_id, purge_artifacts=True)
                results.append({"job_id": job_id, "ok": True, "result": value})
            except Exception as exc:
                results.append({"job_id": job_id, "ok": False, "error": str(exc)})
        return {
            "action": operation,
            "requested": len(unique),
            "succeeded": sum(1 for item in results if item["ok"]),
            "failed": sum(1 for item in results if not item["ok"]),
            "results": results,
        }

    def _all_events(self, job_id: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self.store._event_from_row(row).to_dict() for row in rows]

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")

    def debug_bundle(self, job_id: str, *, max_log_bytes: int = 5 * 1024 * 1024) -> bytes:
        job = self.service.get(job_id)
        events = self._all_events(job.id)
        attempts = [attempt.to_dict() for attempt in self.service.attempts(job.id)]
        verifications = [value.to_dict() for value in VerificationStore(self.store).list(job.id, limit=500)]
        receipt = JobReceiptBuilder(self.service).build(job.id)
        diagnostics = build_job_diagnostics(
            self.service,
            job.id,
            event_limit=1000,
            log_tail_bytes=524_288,
        ).to_dict()
        management = self.state(job.id)
        diff: Optional[Dict[str, Any]] = None
        diff_error = ""
        try:
            diff = build_job_diff(self.service, job.id, max_chars=2_000_000).to_dict()
        except JobReviewError as exc:
            diff_error = str(exc)

        log_path = str(diagnostics.get("worker_log_path") or "")
        log_bytes = b""
        log_size = 0
        log_truncated = False
        if log_path and os.path.isfile(log_path):
            try:
                log_size = int(os.path.getsize(log_path))
                take = max(64 * 1024, min(int(max_log_bytes), 20 * 1024 * 1024))
                with open(log_path, "rb") as handle:
                    if log_size > take:
                        handle.seek(-take, os.SEEK_END)
                        log_truncated = True
                    log_bytes = handle.read(take)
            except OSError:
                log_bytes = b""

        generated_at = time.time()
        manifest = {
            "schema": "mucli-job-debug-bundle/v1",
            "generated_at": generated_at,
            "job_id": job.id,
            "status": job.status.value,
            "event_count": len(events),
            "attempt_count": len(attempts),
            "verification_count": len(verifications),
            "worker_log_size": log_size,
            "worker_log_exported_bytes": len(log_bytes),
            "worker_log_truncated": log_truncated,
            "diff_available": diff is not None,
            "diff_error": diff_error,
        }

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", self._json_bytes(manifest))
            archive.writestr("job.json", self._json_bytes({**job.to_dict(), **management}))
            archive.writestr("work-receipt.json", self._json_bytes(receipt))
            archive.writestr("diagnostics.json", self._json_bytes(diagnostics))
            archive.writestr("events.json", self._json_bytes(events))
            archive.writestr(
                "events.ndjson",
                "".join(json.dumps(event, ensure_ascii=False, default=str) + "\n" for event in events).encode("utf-8"),
            )
            archive.writestr("attempts.json", self._json_bytes(attempts))
            archive.writestr("verifications.json", self._json_bytes(verifications))
            if diff is not None:
                archive.writestr("git-diff.json", self._json_bytes(diff))
                archive.writestr("git.diff", str(diff.get("patch") or "").encode("utf-8"))
            else:
                archive.writestr("git-diff-unavailable.txt", diff_error.encode("utf-8"))
            if log_bytes:
                archive.writestr("worker.log", log_bytes)
            archive.writestr(
                "README.txt",
                (
                    "MuCLI engineering-job debug bundle\n\n"
                    "events.ndjson is convenient for grep/jq/log ingestion. diagnostics.json contains the "
                    "structured execution/preflight view. worker.log is bounded and may contain provider/runtime "
                    "output; review before sharing externally.\n"
                ).encode("utf-8"),
            )
        return buffer.getvalue()

    def report_export(self, format: str = "json", **filters: Any) -> tuple[bytes, str, str]:
        export_format = str(format or "json").strip().lower()
        query = self.query_jobs(limit=5000, offset=0, **filters)
        report = self.report(**filters)
        generated_at = time.time()
        if export_format == "json":
            payload = {
                "schema": "mucli-job-report/v1",
                "generated_at": generated_at,
                "report": report,
                "query": query,
                "truncated": query["total"] > len(query["jobs"]),
            }
            return self._json_bytes(payload), "application/json", "mucli-job-report.json"
        if export_format != "csv":
            raise ValueError("format must be json or csv")

        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id", "title", "status", "archived", "repository", "branch", "base_sha",
                "created_at", "updated_at", "elapsed_seconds", "attempts", "cost_usd",
                "attention_reason", "session_name",
            ]
        )
        for job in query["jobs"]:
            writer.writerow(
                [
                    job["id"], job["title"], job["status"], job["archived"], job["repository"],
                    job["branch"], job["base_sha"], job["created_at"], job["updated_at"],
                    job["elapsed_seconds"], job["attempt_count"], job["cost_usd"],
                    job["attention_reason"], job["session_name"],
                ]
            )
        return stream.getvalue().encode("utf-8"), "text/csv; charset=utf-8", "mucli-job-report.csv"
