from __future__ import annotations

import subprocess

import pytest

from mu.jobs import JobService, JobSpec, JobStore
from mu.jobs.worktree import JobWorktreeManager, WorktreeError


def git(path, *args):
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def make_repo(tmp_path, *, branch="main"):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", branch)
    git(repo, "config", "user.name", "Test User")
    git(repo, "config", "user.email", "test@example.com")
    (repo / "app.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "app.txt")
    git(repo, "commit", "-m", "base")
    return repo


def make_service(tmp_path):
    return JobService(JobStore(str(tmp_path / "jobs.sqlite3")))


def test_remove_rejects_unregistered_decoy_path(tmp_path):
    """Round-37 F2: a corrupted job.worktree path (directory named after
    the job id, but NOT a registered git worktree of the repository) must
    be refused instead of rmtree'd. The decoy here holds a sentinel file
    that would have been destroyed by the old unconditional rmtree."""
    repo = make_repo(tmp_path)
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Decoy victim", repository=str(repo)))
    decoy = tmp_path / "worktrees" / job.id
    decoy.mkdir(parents=True)
    sentinel = decoy / "sentinel.txt"
    sentinel.write_text("precious\n", encoding="utf-8")
    service.store.update_runtime_fields(job.id, worktree=str(decoy))
    fresh = service.get(job.id)
    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))
    with pytest.raises(WorktreeError) as excinfo:
        manager.remove(fresh, force=True)
    assert "not a registered worktree" in str(excinfo.value)
    assert sentinel.read_text(encoding="utf-8") == "precious\n"


def test_prepare_creates_branch_and_worktree_without_touching_primary_checkout(tmp_path):
    repo = make_repo(tmp_path)
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Change greeting", repository=str(repo)))
    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))

    info = manager.prepare(job)
    current = service.get(job.id)

    assert info.repository == str(repo)
    assert info.base_ref == "main"
    assert current.worktree == info.worktree
    assert current.branch.startswith(f"mu/job-{job.id[:10]}-")
    assert current.base_sha == git(repo, "rev-parse", "main^{commit}")
    assert git(current.worktree, "rev-parse", "--show-toplevel") == current.worktree
    assert git(current.worktree, "branch", "--show-current") == current.branch
    assert (repo / "app.txt").read_text(encoding="utf-8") == "base\n"


def test_implicit_main_falls_back_to_detected_repository_branch(tmp_path):
    """Historical jobs default to `main`; healthy non-main repos must still run."""
    repo = make_repo(tmp_path, branch="develop")
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Non-main repository", repository=str(repo)))
    assert job.base_branch == "main"  # historical implicit default

    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))
    info = manager.prepare(job)
    current = service.get(job.id)

    assert info.base_ref == "develop"
    assert current.base_sha == git(repo, "rev-parse", "develop^{commit}")
    assert git(current.worktree, "branch", "--show-current") == current.branch
    events = service.events(job.id)
    resolved = next(event for event in events if event.event_type == "job_base_resolved")
    assert resolved.payload["resolved_base_ref"] == "develop"
    assert resolved.payload["fallback_used"] is True
    inspected = next(event for event in events if event.event_type == "repository_inspected")
    assert inspected.payload["detected_default_branch"] == "develop"


def test_explicit_missing_non_main_branch_does_not_silently_fallback(tmp_path):
    repo = make_repo(tmp_path, branch="develop")
    service = make_service(tmp_path)
    job = service.create(
        JobSpec(
            title="Explicit release branch",
            repository=str(repo),
            base_branch="release-does-not-exist",
        )
    )
    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))

    with pytest.raises(WorktreeError) as caught:
        manager.prepare(job)

    assert caught.value.stage == "base_resolution"
    assert "release-does-not-exist" in str(caught.value)
    failed = [event for event in service.events(job.id) if event.event_type == "worktree_prepare_failed"]
    assert len(failed) == 1
    assert failed[0].payload["stage"] == "base_resolution"
    assert failed[0].payload["requested_base_branch"] == "release-does-not-exist"
    assert failed[0].payload["attempted_refs"]


def test_worktree_preflight_records_user_visible_git_diagnostics(tmp_path):
    repo = make_repo(tmp_path)
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Trace preparation", repository=str(repo)))
    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))

    manager.prepare(job)
    events = service.events(job.id)
    names = [event.event_type for event in events]

    assert "worktree_preflight_started" in names
    assert "repository_inspected" in names
    assert "job_base_resolved" in names
    assert "worktree_inventory" in names
    assert "worktree_add_started" in names
    assert "worktree_ready" in names

    inspected = next(event for event in events if event.event_type == "repository_inspected")
    assert inspected.payload["canonical_path"] == str(repo)
    assert inspected.payload["git_common_dir"]
    ready = next(event for event in events if event.event_type == "worktree_ready")
    assert ready.payload["base_sha"]
    assert ready.payload["branch"].startswith("mu/job-")


def test_checkpoint_commits_job_changes_only_on_managed_branch(tmp_path):
    repo = make_repo(tmp_path)
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Checkpoint me", repository=str(repo)))
    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))
    manager.prepare(job)
    current = service.get(job.id)

    (tmp_path / "worktrees" / job.id / "app.txt").write_text("job change\n", encoding="utf-8")
    sha = manager.checkpoint(current, label="attempt-1-implementation")

    assert sha
    assert git(current.worktree, "rev-parse", "HEAD") == sha
    assert git(current.worktree, "status", "--porcelain") == ""
    assert (repo / "app.txt").read_text(encoding="utf-8") == "base\n"
    assert git(repo, "rev-parse", "main^{commit}") == current.base_sha
    events = service.events(job.id)
    assert any(event.event_type == "checkpoint_created" and event.payload["sha"] == sha for event in events)


def test_prepare_is_idempotent_for_resumed_job(tmp_path):
    repo = make_repo(tmp_path)
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Resume", repository=str(repo)))
    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))

    first = manager.prepare(job)
    (tmp_path / "worktrees" / job.id / "resume.txt").write_text("saved\n", encoding="utf-8")
    checkpoint = manager.checkpoint(service.get(job.id), label="blocked")
    second = manager.prepare(service.get(job.id))

    # The first prepare resolves the symbolic branch and persists an immutable
    # base_sha. A resumed prepare may therefore report that pinned SHA as its
    # base_ref. The isolation invariants themselves must remain identical.
    assert second.repository == first.repository
    assert second.worktree == first.worktree
    assert second.branch == first.branch
    assert second.base_sha == first.base_sha
    assert first.base_ref == "main"
    assert second.base_ref in {first.base_ref, first.base_sha}
    assert checkpoint == git(second.worktree, "rev-parse", "HEAD")
    assert (tmp_path / "worktrees" / job.id / "resume.txt").read_text(encoding="utf-8") == "saved\n"
    assert (repo / "app.txt").read_text(encoding="utf-8") == "base\n"


def test_two_jobs_get_separate_worktrees_and_branches(tmp_path):
    repo = make_repo(tmp_path)
    service = make_service(tmp_path)
    manager = JobWorktreeManager(service, root=str(tmp_path / "worktrees"))
    a = service.create(JobSpec(title="Ticket A", repository=str(repo)))
    b = service.create(JobSpec(title="Ticket B", repository=str(repo)))

    wa = manager.prepare(a)
    wb = manager.prepare(b)
    assert wa.worktree != wb.worktree
    assert wa.branch != wb.branch
    assert wa.base_sha == wb.base_sha

    (tmp_path / "worktrees" / a.id / "app.txt").write_text("A\n", encoding="utf-8")
    (tmp_path / "worktrees" / b.id / "app.txt").write_text("B\n", encoding="utf-8")
    sha_a = manager.checkpoint(service.get(a.id), label="A")
    sha_b = manager.checkpoint(service.get(b.id), label="B")

    assert sha_a != sha_b
    assert (repo / "app.txt").read_text(encoding="utf-8") == "base\n"
    assert (tmp_path / "worktrees" / a.id / "app.txt").read_text(encoding="utf-8") == "A\n"
    assert (tmp_path / "worktrees" / b.id / "app.txt").read_text(encoding="utf-8") == "B\n"
