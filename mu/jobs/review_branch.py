"""Materialize verified durable-job work as a normal Git review branch.

Worktrees are an execution primitive, not a review surface.  Once deterministic
verification passes, the job branch is made authoritative, the temporary
worktree is retired, and subsequent review/diff operations use the branch from
the canonical repository.  If a reviewer requests changes later, the normal
worker preparation path re-creates a fresh temporary worktree on the same
branch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from .service import JobService
from .worktree import JobWorktreeManager, WorktreeError


@dataclass(frozen=True)
class ReviewBranchInfo:
    repository: str
    branch: str
    head_sha: str
    base_sha: str
    retired_worktree: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository": self.repository,
            "branch": self.branch,
            "head_sha": self.head_sha,
            "base_sha": self.base_sha,
            "retired_worktree": self.retired_worktree,
        }


def materialize_review_branch(
    service: JobService,
    job_id: str,
    *,
    expected_head_sha: Optional[str] = None,
) -> ReviewBranchInfo:
    """Make the job branch the sole review artifact and retire its worktree.

    Safety contract:
    - branch + captured base must exist;
    - when `expected_head_sha` is supplied (the SHA the verification run
      actually tested), both the branch ref and any present worktree HEAD
      must equal it — otherwise an unverified commit slipped in between
      the checks finishing and this materialization;
    - any still-present worktree must be clean;
    - worktree HEAD must equal the branch ref before removal;
    - branch must descend from the captured base;
    - only then is the managed worktree force-removed (safe because clean).
    """

    job = service.get(job_id)
    if not job.branch:
        raise WorktreeError(
            "Verified job has no managed branch to review.",
            stage="review_branch_resolution",
            context={"job_id": job.id},
        )
    if not job.base_sha:
        raise WorktreeError(
            "Verified job has no captured base commit.",
            stage="review_branch_resolution",
            context={"job_id": job.id, "branch": job.branch},
        )

    manager = JobWorktreeManager(service)
    try:
        repository = manager.repositories.register(job.repository)
    except Exception as exc:
        raise WorktreeError(
            str(exc),
            stage="review_repository_resolution",
            context={"repository_input": job.repository},
        ) from exc

    repo = repository.canonical_path
    branch_ref = f"refs/heads/{job.branch}"
    branch_result = manager._run(
        repo,
        "rev-parse",
        "--verify",
        f"{branch_ref}^{{commit}}",
        stage="review_branch_head",
    )
    branch_head = (branch_result.stdout or "").strip()
    if not branch_head:
        raise WorktreeError(
            f"Review branch {job.branch!r} has no commit.",
            stage="review_branch_head",
        )
    if expected_head_sha and branch_head != expected_head_sha:
        raise WorktreeError(
            "Verification evidence is stale: branch HEAD advanced after the "
            "checks ran. Refusing to materialize an unverified commit.",
            stage="review_branch_head_stale",
            context={
                "branch": job.branch,
                "expected_head_sha": expected_head_sha,
                "branch_head": branch_head,
            },
        )

    ancestor = manager._run(
        repo,
        "merge-base",
        "--is-ancestor",
        job.base_sha,
        branch_head,
        check=False,
        stage="review_branch_base_check",
    )
    if ancestor.returncode != 0:
        raise WorktreeError(
            f"Review branch {job.branch!r} no longer descends from captured base {job.base_sha}.",
            stage="review_branch_base_check",
            command=["git", "-C", repo, "merge-base", "--is-ancestor", job.base_sha, branch_head],
            return_code=int(ancestor.returncode),
            stdout=ancestor.stdout or "",
            stderr=ancestor.stderr or "",
        )

    retired_worktree = os.path.abspath(os.path.expanduser(str(job.worktree or ""))) if job.worktree else ""
    if retired_worktree and os.path.isdir(retired_worktree):
        status = manager._run(
            retired_worktree,
            "status",
            "--porcelain",
            stage="review_worktree_cleanliness",
        )
        dirty = (status.stdout or "").strip()
        if dirty:
            raise WorktreeError(
                "Refusing to retire a verified job worktree that still has uncommitted changes.",
                stage="review_worktree_cleanliness",
                stdout=dirty,
                context={"worktree": retired_worktree, "branch": job.branch},
            )
        wt_head = (
            manager._run(
                retired_worktree,
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
                stage="review_worktree_head",
            ).stdout
            or ""
        ).strip()
        if wt_head != branch_head:
            raise WorktreeError(
                "Worktree HEAD does not match the review branch; refusing cleanup.",
                stage="review_worktree_branch_mismatch",
                context={
                    "worktree": retired_worktree,
                    "worktree_head": wt_head,
                    "branch": job.branch,
                    "branch_head": branch_head,
                },
            )

        # A clean, branch-matching worktree can be removed forcefully without
        # risking uncommitted data. Unlock first so stale lock files cannot
        # strand a completed job in worktree-only review mode.
        manager._run(
            repo,
            "worktree",
            "unlock",
            retired_worktree,
            check=False,
            stage="review_worktree_unlock",
        )
        # Round-37 F3: the cleanliness/HEAD checks above are a snapshot; a
        # concurrent process could modify the worktree between them and the
        # removal (verified-but-uncommitted work would be silently
        # discarded, or an unverified commit destroyed while metadata
        # records the stale SHA). Re-verify BOTH invariants immediately
        # before the destructive remove and abort loudly on any change.
        recheck_status = manager._run(
            retired_worktree,
            "status",
            "--porcelain",
            stage="review_worktree_cleanliness_recheck",
            check=False,
        )
        if (recheck_status.stdout or "").strip():
            raise WorktreeError(
                "Worktree changed during review materialization: uncommitted "
                "changes appeared after the first cleanliness check.",
                stage="review_worktree_cleanliness_recheck",
                stdout=(recheck_status.stdout or "").strip(),
                context={"worktree": retired_worktree, "branch": job.branch},
            )
        recheck_head = (
            manager._run(
                retired_worktree,
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
                stage="review_worktree_head_recheck",
                check=False,
            ).stdout
            or ""
        ).strip()
        if recheck_head != branch_head:
            raise WorktreeError(
                "Worktree HEAD moved during review materialization; refusing "
                "to discard an unverified commit.",
                stage="review_worktree_branch_mismatch_recheck",
                context={
                    "worktree": retired_worktree,
                    "worktree_head": recheck_head,
                    "branch_head": branch_head,
                },
            )
        # Round-39 F5: remove WITHOUT --force — git re-checks dirtiness at
        # removal time, closing the recheck-to-removal window for new
        # modifications (a --force removal would delete them silently).
        # If the removal still fails because the tree changed in the
        # microseconds between recheck and remove, the WorktreeError
        # surfaces and materialization fails loudly — the correct outcome
        # when the invariants cannot be held.
        manager.remove(job, force=False)
    else:
        manager._run(repo, "worktree", "prune", check=False, stage="review_worktree_prune")

    now = float(service.store._clock())
    metadata = {
        **dict(job.metadata or {}),
        "worktree_managed": False,
        "review_branch": job.branch,
        "review_head_sha": branch_head,
        "review_base_sha": job.base_sha,
        "review_branch_materialized_at": now,
        "retired_worktree": retired_worktree,
    }
    environment = {
        **dict(job.environment or {}),
        "kind": "host_git_review_branch",
        "repository_id": repository.id,
        "repository_root": repo,
        "branch": job.branch,
        "head_sha": branch_head,
        "base_sha": job.base_sha,
        "worktree": "",
    }
    service.store.update_runtime_fields(
        job.id,
        worktree="",
        metadata_json=metadata,
        environment_json=environment,
    )
    info = ReviewBranchInfo(
        repository=repo,
        branch=job.branch,
        head_sha=branch_head,
        base_sha=job.base_sha,
        retired_worktree=retired_worktree,
    )
    service.store.append_event(
        job.id,
        "review_branch_ready",
        reason="verified implementation materialized as branch; worktree retired",
        payload=info.to_dict(),
    )
    return info
