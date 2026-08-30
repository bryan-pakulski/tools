"""Canonical Git repository registry for durable engineering jobs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .store import JobStore


class RepositoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepositoryRecord:
    id: str
    canonical_path: str
    git_common_dir: str
    origin_url: str
    default_branch: str
    created_at: float
    updated_at: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepositoryRegistry:
    """Durable registry keyed by Git common-dir identity.

    A job may be submitted from the primary checkout or any existing worktree;
    both resolve to one repository record and one canonical primary worktree.
    The inspection result separately retains the submitted worktree's HEAD so a
    job starts from the exact committed state the user delegated from.
    """

    def __init__(self, store: JobStore):
        self.store = store
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
                CREATE TABLE IF NOT EXISTS job_repositories (
                    id TEXT PRIMARY KEY,
                    canonical_path TEXT NOT NULL UNIQUE,
                    git_common_dir TEXT NOT NULL UNIQUE,
                    origin_url TEXT NOT NULL DEFAULT '',
                    default_branch TEXT NOT NULL DEFAULT 'main',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS job_repositories_updated_idx
                    ON job_repositories(updated_at DESC);
                """
            )
        finally:
            conn.close()

    @staticmethod
    def _git(path: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", "-C", path, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if check and result.returncode != 0:
            raise RepositoryError((result.stderr or result.stdout or "git command failed").strip())
        return result

    @classmethod
    def inspect(cls, path: str) -> Dict[str, Any]:
        candidate = os.path.abspath(os.path.expanduser(str(path or "")))
        if not os.path.isdir(candidate):
            raise RepositoryError(f"Repository does not exist: {candidate or path}")
        submitted = (cls._git(candidate, "rev-parse", "--show-toplevel").stdout or "").strip()
        if not submitted:
            raise RepositoryError(f"Not a Git repository: {candidate}")
        common_raw = (cls._git(submitted, "rev-parse", "--git-common-dir").stdout or "").strip()
        common_dir = common_raw if os.path.isabs(common_raw) else os.path.join(submitted, common_raw)
        common_dir = os.path.realpath(os.path.abspath(common_dir))

        # The first porcelain worktree is Git's primary checkout. Use it as the
        # stable repository identity/canonical path, but DO NOT use its HEAD as
        # the delegated starting point when the user supplied another worktree.
        listing = (cls._git(submitted, "worktree", "list", "--porcelain").stdout or "").splitlines()
        primary = submitted
        for line in listing:
            if line.startswith("worktree "):
                primary = os.path.abspath(line.split(" ", 1)[1].strip())
                break

        origin = cls._git(primary, "config", "--get", "remote.origin.url", check=False)
        origin_url = (origin.stdout or "").strip()

        primary_branch_result = cls._git(
            primary,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        primary_branch = (primary_branch_result.stdout or "").strip()

        submitted_branch_result = cls._git(
            submitted,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        current_branch = (submitted_branch_result.stdout or "").strip()

        # Prefer origin/HEAD for the repository-level default. Fall back to
        # the primary checkout branch and finally the submitted worktree branch.
        remote_head = cls._git(
            primary,
            "symbolic-ref",
            "--quiet",
            "--short",
            "refs/remotes/origin/HEAD",
            check=False,
        )
        remote_default_ref = (remote_head.stdout or "").strip()
        remote_default_branch = (
            remote_default_ref.split("/", 1)[1]
            if remote_default_ref.startswith("origin/")
            else remote_default_ref
        )
        default_branch = remote_default_branch or primary_branch or current_branch or "main"

        # These are deliberately taken from `submitted`, not `primary`.
        head = cls._git(submitted, "rev-parse", "--verify", "HEAD^{commit}", check=False)
        head_sha = (head.stdout or "").strip()
        status = cls._git(submitted, "status", "--porcelain", check=False)
        clean = status.returncode == 0 and not bool((status.stdout or "").strip())

        identity = hashlib.sha256(common_dir.encode("utf-8")).hexdigest()[:24]
        return {
            "id": identity,
            "canonical_path": primary,
            "submitted_path": submitted,
            "git_common_dir": common_dir,
            "origin_url": origin_url,
            "default_branch": default_branch,
            "primary_branch": primary_branch,
            "current_branch": current_branch,
            "remote_default_ref": remote_default_ref,
            "head_sha": head_sha,
            "clean": clean,
        }

    def register(self, path: str, *, metadata: Optional[Dict[str, Any]] = None) -> RepositoryRecord:
        info = self.inspect(path)
        now = time.time()
        conn = self._connect()
        try:
            # Round-41 F9: BEGIN IMMEDIATE BEFORE the read-merge — the old
            # order (read + merge metadata, then lock) let two concurrent
            # registrations of the same repository both read the same old
            # metadata, merge different updates, and upsert sequentially:
            # the later writer silently discarded the earlier one's
            # metadata. Under the immediate lock the read-merge-upsert is
            # one atomic critical section.
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM job_repositories WHERE id = ?",
                (info["id"],),
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            previous_metadata: Dict[str, Any] = {}
            if existing:
                try:
                    previous_metadata = json.loads(existing["metadata_json"] or "{}")
                except (TypeError, ValueError):
                    previous_metadata = {}
            previous_metadata.update(
                {
                    "last_submitted_path": info.get("submitted_path", ""),
                    "current_branch": info.get("current_branch", ""),
                    "primary_branch": info.get("primary_branch", ""),
                    "remote_default_ref": info.get("remote_default_ref", ""),
                    "head_sha": info.get("head_sha", ""),
                    "clean": bool(info.get("clean", False)),
                }
            )
            previous_metadata.update(metadata or {})
            conn.execute(
                """
                INSERT INTO job_repositories (
                    id, canonical_path, git_common_dir, origin_url, default_branch,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    canonical_path = excluded.canonical_path,
                    git_common_dir = excluded.git_common_dir,
                    origin_url = excluded.origin_url,
                    default_branch = excluded.default_branch,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    info["id"], info["canonical_path"], info["git_common_dir"],
                    info["origin_url"], info["default_branch"], created_at, now,
                    json.dumps(previous_metadata, ensure_ascii=False, default=str),
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
        return self.get(info["id"])

    def get(self, repository_id: str) -> RepositoryRecord:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM job_repositories WHERE id = ?",
                (repository_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(repository_id)
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        return RepositoryRecord(
            id=row["id"],
            canonical_path=row["canonical_path"],
            git_common_dir=row["git_common_dir"],
            origin_url=row["origin_url"],
            default_branch=row["default_branch"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            metadata=metadata,
        )

    def list(self, *, limit: int = 200) -> List[RepositoryRecord]:
        # Round-49 F8: this was an N+1 — ids fetched, then get() per row,
        # each get() opening its own SQLite connection. One query, one
        # connection; rows map directly to records.
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM job_repositories ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        finally:
            conn.close()
        out: List[RepositoryRecord] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                metadata = {}
            out.append(
                RepositoryRecord(
                    id=row["id"],
                    canonical_path=row["canonical_path"],
                    git_common_dir=row["git_common_dir"],
                    origin_url=row["origin_url"],
                    default_branch=row["default_branch"],
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                    metadata=metadata,
                )
            )
        return out
