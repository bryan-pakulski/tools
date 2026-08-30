"""Git isolation and checkpointing for durable engineering jobs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from utils.config import HISTORY_DIR

from .models import Job
from .repository import RepositoryRecord, RepositoryRegistry
from .service import JobService


class WorktreeError(RuntimeError):
    """Structured Git/worktree failure safe to expose in job diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "worktree",
        command: Optional[Sequence[str]] = None,
        return_code: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(str(message or "worktree operation failed"))
        self.stage = str(stage or "worktree")
        self.command = [str(part) for part in (command or [])]
        self.return_code = return_code
        self.stdout = str(stdout or "")[-8000:]
        self.stderr = str(stderr or "")[-8000:]
        self.context = dict(context or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "error": str(self),
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            **self.context,
        }


@dataclass(frozen=True)
class WorktreeInfo:
    repository_id: str
    repository: str
    worktree: str
    branch: str
    base_sha: str
    base_ref: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "repository_id": self.repository_id,
            "repository": self.repository,
            "worktree": self.worktree,
            "branch": self.branch,
            "base_sha": self.base_sha,
            "base_ref": self.base_ref,
        }


class JobWorktreeManager:
    def __init__(self, service: JobService, *, root: Optional[str] = None):
        self.service = service
        self.repositories = RepositoryRegistry(service.store)
        self.root = os.path.abspath(
            os.path.expanduser(root or os.path.join(HISTORY_DIR, "jobs", "worktrees"))
        )
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def _run(
        repo: str,
        *args: str,
        check: bool = True,
        stage: str = "git",
    ) -> subprocess.CompletedProcess:
        command = ["git", "-C", repo, *args]
        try:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorktreeError(
                f"Git command timed out during {stage}",
                stage=stage,
                command=command,
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or ""),
            ) from exc
        except OSError as exc:
            raise WorktreeError(
                f"Could not run Git during {stage}: {exc}",
                stage=stage,
                command=command,
            ) from exc
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "git command failed").strip()
            raise WorktreeError(
                detail,
                stage=stage,
                command=command,
                return_code=int(result.returncode),
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        return result

    def _event(
        self,
        job_id: str,
        event_type: str,
        *,
        reason: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self.service.store.append_event(
                job_id,
                event_type,
                reason=reason,
                payload=payload or {},
            )
        except Exception:
            pass

    @staticmethod
    def _slug(text: str, limit: int = 28) -> str:
        value = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
        return (value or "job")[:limit].rstrip("-")

    def branch_name(self, job: Job) -> str:
        return job.branch or f"mu/job-{job.id[:10]}-{self._slug(job.title)}"

    def worktree_path(self, job: Job) -> str:
        return job.worktree or os.path.join(self.root, job.id)

    @staticmethod
    def _persisted_base_ref(job: Job) -> str:
        metadata = dict(job.metadata or {})
        existing = str(metadata.get("resolved_base_ref") or "").strip()
        if existing:
            return existing
        submission = metadata.get("submission_repository_preflight")
        if isinstance(submission, dict):
            return str(
                submission.get("current_branch")
                or submission.get("default_branch")
                or job.base_branch
                or ""
            ).strip()
        return ""

    def _resolve_base(
        self,
        repo: str,
        job: Job,
        repository: RepositoryRecord,
    ) -> tuple[str, str]:
        if job.base_sha:
            result = self._run(
                repo,
                "rev-parse",
                "--verify",
                f"{job.base_sha}^{{commit}}",
                stage="base_sha_resolution",
            )
            sha = (result.stdout or "").strip()
            # Preserve the originally resolved human-readable ref on retries.
            # Newly-created GUI/mobile jobs already carry their submitted branch
            # in the repository-preflight metadata; historical jobs gain
            # resolved_base_ref after their first successful preparation.
            base_ref = self._persisted_base_ref(job) or str(job.base_sha)
            return sha, base_ref

        requested = str(job.base_branch or "main").strip() or "main"
        candidates: List[str] = []

        def add(ref: str) -> None:
            value = str(ref or "").strip()
            if value and value not in candidates:
                candidates.append(value)

        if requested.lower() != "auto":
            add(requested)
            if not requested.startswith("origin/"):
                add(f"origin/{requested}")

        # `main` was historically injected when callers did not specify a base.
        # Only that implicit legacy value (plus explicit `auto`) may fall back.
        allow_fallback = requested.lower() in {"", "auto", "main"}
        if allow_fallback:
            add(repository.default_branch)
            if repository.default_branch and not repository.default_branch.startswith("origin/"):
                add(f"origin/{repository.default_branch}")

            remote_head = self._run(
                repo,
                "symbolic-ref",
                "--quiet",
                "--short",
                "refs/remotes/origin/HEAD",
                check=False,
                stage="remote_default_branch_detection",
            )
            add((remote_head.stdout or "").strip())

            current = self._run(
                repo,
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
                check=False,
                stage="current_branch_detection",
            )
            add((current.stdout or "").strip())
            add("HEAD")

        attempted: List[Dict[str, Any]] = []
        for ref in candidates:
            result = self._run(
                repo,
                "rev-parse",
                "--verify",
                f"{ref}^{{commit}}",
                check=False,
                stage="base_ref_resolution",
            )
            attempted.append(
                {
                    "ref": ref,
                    "return_code": int(result.returncode),
                    "stderr": (result.stderr or "").strip()[-1200:],
                }
            )
            sha = (result.stdout or "").strip()
            if result.returncode == 0 and sha:
                return sha, ref

        raise WorktreeError(
            f"Could not resolve base branch {requested!r} in {repo}",
            stage="base_resolution",
            context={
                "requested_base_branch": requested,
                "detected_default_branch": repository.default_branch,
                "attempted_refs": attempted,
            },
        )

    def _registered_worktrees(self, repo: str) -> Dict[str, Dict[str, str]]:
        self._run(repo, "worktree", "prune", check=False, stage="worktree_prune")
        result = self._run(
            repo,
            "worktree",
            "list",
            "--porcelain",
            stage="worktree_inventory",
        )
        entries: Dict[str, Dict[str, str]] = {}
        current: Dict[str, str] = {}
        for raw in (result.stdout or "").splitlines() + [""]:
            line = raw.strip()
            if not line:
                path = current.get("worktree")
                if path:
                    entries[os.path.abspath(path)] = dict(current)
                current = {}
                continue
            key, _, value = line.partition(" ")
            current[key] = value
        return entries

    def prepare(self, job: Job) -> WorktreeInfo:
        branch = self.branch_name(job)
        worktree = os.path.abspath(self.worktree_path(job))
        self._event(
            job.id,
            "worktree_preflight_started",
            payload={
                "repository_input": job.repository,
                "requested_base_branch": job.base_branch,
                "requested_base_sha": job.base_sha,
                "managed_branch": branch,
                "managed_worktree": worktree,
            },
        )

        try:
            try:
                repository = self.repositories.register(job.repository)
            except Exception as exc:
                raise WorktreeError(
                    str(exc),
                    stage="repository_inspection",
                    context={"repository_input": job.repository},
                ) from exc

            repo = repository.canonical_path
            repo_meta = dict(repository.metadata or {})
            self._event(
                job.id,
                "repository_inspected",
                payload={
                    "repository_id": repository.id,
                    "repository_input": job.repository,
                    "submitted_path": repo_meta.get("last_submitted_path", job.repository),
                    "canonical_path": repo,
                    "git_common_dir": repository.git_common_dir,
                    "origin_url": repository.origin_url,
                    "detected_default_branch": repository.default_branch,
                    "current_branch": repo_meta.get("current_branch", ""),
                    "primary_branch": repo_meta.get("primary_branch", ""),
                    "head_sha": repo_meta.get("head_sha", ""),
                    "source_worktree_clean": repo_meta.get("clean"),
                },
            )

            base_sha, base_ref = self._resolve_base(repo, job, repository)
            requested = str(job.base_branch or "main").strip() or "main"
            self._event(
                job.id,
                "job_base_resolved",
                payload={
                    "requested_base_branch": requested,
                    "resolved_base_ref": base_ref,
                    "base_sha": base_sha,
                    "fallback_used": bool(
                        not job.base_sha
                        and base_ref not in {requested, f"origin/{requested}"}
                    ),
                },
            )

            registered = self._registered_worktrees(repo)
            self._event(
                job.id,
                "worktree_inventory",
                payload={
                    "registered_count": len(registered),
                    "managed_worktree": worktree,
                    "managed_path_registered": worktree in registered,
                    "managed_path_exists": os.path.exists(worktree),
                },
            )

            existing = registered.get(worktree)
            if existing:
                expected_ref = f"refs/heads/{branch}"
                if existing.get("branch") and existing.get("branch") != expected_ref:
                    raise WorktreeError(
                        f"Managed worktree {worktree} is attached to {existing.get('branch')}, expected {expected_ref}",
                        stage="existing_worktree_validation",
                        context={
                            "managed_worktree": worktree,
                            "actual_branch": existing.get("branch"),
                            "expected_branch": expected_ref,
                        },
                    )
                head = self._run(
                    worktree,
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                    stage="existing_worktree_head",
                )
                if not (head.stdout or "").strip():
                    raise WorktreeError(
                        f"Registered worktree has no HEAD: {worktree}",
                        stage="existing_worktree_head",
                    )
            else:
                if os.path.exists(worktree):
                    if os.path.isdir(worktree) and not os.listdir(worktree):
                        os.rmdir(worktree)
                    else:
                        raise WorktreeError(
                            f"Refusing to replace unregistered non-empty job worktree: {worktree}",
                            stage="managed_path_collision",
                            context={"managed_worktree": worktree},
                        )
                os.makedirs(os.path.dirname(worktree), exist_ok=True)
                branch_exists = self._run(
                    repo,
                    "show-ref",
                    "--verify",
                    "--quiet",
                    f"refs/heads/{branch}",
                    check=False,
                    stage="managed_branch_lookup",
                ).returncode == 0
                self._event(
                    job.id,
                    "worktree_add_started",
                    payload={
                        "repository": repo,
                        "worktree": worktree,
                        "branch": branch,
                        "branch_exists": branch_exists,
                        "base_ref": base_ref,
                        "base_sha": base_sha,
                    },
                )
                if branch_exists:
                    self._run(
                        repo,
                        "worktree",
                        "add",
                        worktree,
                        branch,
                        stage="worktree_add_existing_branch",
                    )
                else:
                    self._run(
                        repo,
                        "worktree",
                        "add",
                        "-b",
                        branch,
                        worktree,
                        base_sha,
                        stage="worktree_add_new_branch",
                    )

            metadata = {
                **dict(job.metadata or {}),
                "repository_id": repository.id,
                "repository_root": repo,
                "repository_origin": repository.origin_url,
                "repository_default_branch": repository.default_branch,
                "resolved_base_ref": base_ref,
                "worktree_managed": True,
            }
            environment = {
                **dict(job.environment or {}),
                "kind": "host_git_worktree",
                "repository_id": repository.id,
                "repository_root": repo,
                "worktree": worktree,
                "branch": branch,
                "base_ref": base_ref,
            }
            updated = self.service.store.update_runtime_fields(
                job.id,
                base_sha=base_sha,
                branch=branch,
                worktree=worktree,
                environment_json=environment,
                metadata_json=metadata,
            )
            self._event(
                job.id,
                "worktree_ready",
                payload={
                    "repository_id": repository.id,
                    "repository": repo,
                    "worktree": updated.worktree,
                    "branch": updated.branch,
                    "base_ref": base_ref,
                    "base_sha": updated.base_sha,
                },
            )
            return WorktreeInfo(
                repository.id,
                repo,
                updated.worktree,
                updated.branch,
                updated.base_sha,
                base_ref,
            )
        except WorktreeError as exc:
            payload = {
                **exc.to_dict(),
                "repository_input": job.repository,
                "requested_base_branch": job.base_branch,
                "requested_base_sha": job.base_sha,
                "managed_branch": branch,
                "managed_worktree": worktree,
            }
            self._event(
                job.id,
                "worktree_prepare_failed",
                reason=str(exc),
                payload=payload,
            )
            raise

    def checkpoint(self, job: Job, *, label: str) -> Optional[str]:
        worktree = str(job.worktree or "")
        if not worktree or not os.path.isdir(worktree):
            return None
        status = self._run(
            worktree,
            "status",
            "--porcelain",
            stage="checkpoint_status",
        )
        if not (status.stdout or "").strip():
            head = self._run(
                worktree,
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
                stage="checkpoint_head",
            )
            return (head.stdout or "").strip() or None

        self._run(worktree, "add", "-A", stage="checkpoint_stage")
        env = os.environ.copy()
        env.setdefault("GIT_AUTHOR_NAME", "MuCLI")
        env.setdefault("GIT_AUTHOR_EMAIL", "mucli@localhost")
        env.setdefault("GIT_COMMITTER_NAME", "MuCLI")
        env.setdefault("GIT_COMMITTER_EMAIL", "mucli@localhost")
        command = [
            "git", "-C", worktree, "commit", "--no-gpg-sign",
            "-m", f"mu checkpoint: {label}",
        ]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "checkpoint commit failed").strip()
            raise WorktreeError(
                detail,
                stage="checkpoint_commit",
                command=command,
                return_code=int(result.returncode),
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        head = self._run(
            worktree,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            stage="checkpoint_head",
        )
        sha = (head.stdout or "").strip()
        self._event(
            job.id,
            "checkpoint_created",
            payload={
                "sha": sha,
                "label": label,
                "branch": job.branch,
                "worktree": worktree,
            },
        )
        return sha

    def _validate_removable_path(self, worktree: str, job: Job, repository: Any) -> str:
        """Round-37 F2: identity checks for a worktree path before ANY
        destructive action. A corrupted/imported DB row (or a stale path
        reused by another job) must never let rmtree/--force erase an
        unrelated directory. Rules (root-INDEPENDENT — the manager root is
        per-instance configuration, so a persisted path created under a
        different root is still this job's own checkout):
        - the REALPATH must not be or live inside the repository itself;
        - the final path component must be the job's id (this job's own
          checkout directory name);
        - git's porcelain worktree inventory of THIS repository must list
          the path as a registered worktree."""
        repo_real = os.path.realpath(os.path.abspath(repository.canonical_path))
        real = os.path.realpath(os.path.abspath(worktree))
        if real == repo_real:
            # Never delete the repository itself.
            raise WorktreeError(
                f"refusing to remove {worktree!r}: path is the repository itself",
                stage="worktree_remove",
                context={"worktree": worktree},
            )
        if os.path.basename(real) != job.id:
            raise WorktreeError(
                f"refusing to remove {worktree!r}: not this job's checkout "
                f"(expected directory name {job.id!r})",
                stage="worktree_remove",
                context={"worktree": worktree, "job_id": job.id},
            )
        listing = self._run(
            repository.canonical_path,
            "worktree",
            "list",
            "--porcelain",
            stage="worktree_remove_inventory",
            check=False,
        )
        registered = {
            line[len("worktree "):].strip()
            for line in (listing.stdout or "").splitlines()
            if line.startswith("worktree ")
        }
        real_norm = os.path.normcase(real)
        if not any(os.path.normcase(os.path.abspath(w)) == real_norm for w in registered):
            raise WorktreeError(
                f"refusing to remove {worktree!r}: not a registered worktree "
                "of this job's repository",
                stage="worktree_remove",
                context={"worktree": worktree},
            )
        return real

    def remove(self, job: Job, *, force: bool = False) -> bool:
        try:
            repository = self.repositories.register(job.repository)
        except Exception as exc:
            raise WorktreeError(
                str(exc),
                stage="repository_inspection",
                context={"repository_input": job.repository},
            ) from exc
        worktree = str(job.worktree or self.worktree_path(job))
        if not os.path.exists(worktree):
            return False
        real = self._validate_removable_path(worktree, job, repository)
        args: List[str] = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(worktree)
        self._run(repository.canonical_path, *args, stage="worktree_remove")
        if os.path.exists(worktree) and force:
            # Round-40 F3/F5: capture the identity LOCALLY right here and
            # re-lstat immediately before the delete — fail CLOSED. A
            # same-pathname swap (original removed, unrelated directory
            # created in between) has an identical realpath but a different
            # (st_dev, st_ino); an unreadable path is refused rather than
            # trusted. No instance cache — nothing to leak across jobs.
            def _identity() -> tuple[int, int] | None:
                try:
                    st = os.lstat(worktree)
                except OSError:
                    return None
                return (st.st_dev, st.st_ino)

            pre_identity = _identity()
            if pre_identity is None:
                raise WorktreeError(
                    f"refusing rmtree: {worktree!r} disappeared before fallback delete",
                    stage="worktree_remove",
                    context={"worktree": worktree},
                )
            shutil.rmtree(worktree, ignore_errors=False)
            post_identity = _identity()
            if post_identity is not None:
                # The rmtree above already removed the tree; a surviving
                # entry means the directory was swapped mid-delete — refuse
                # loudly rather than silently accepting a replaced path.
                raise WorktreeError(
                    f"refusing rmtree: {worktree!r} was recreated during removal",
                    stage="worktree_remove",
                    context={"worktree": worktree},
                )
        self._event(
            job.id,
            "worktree_removed",
            payload={"worktree": worktree, "force": bool(force)},
        )
        return True
