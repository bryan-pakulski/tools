from __future__ import annotations

import sqlite3

import pytest

from mu.jobs import AttentionReason, JobService, JobSpec, JobStateError, JobStatus, JobStore
from mu.jobs.board import build_job_board
from mu.jobs.management import JobManagementError, JobManagementService


class Clock:
    def __init__(self, value: float = 1000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_service(tmp_path, clock=None):
    return JobService(JobStore(str(tmp_path / "jobs.sqlite3"), clock=clock or Clock()))


def test_job_persists_across_store_instances(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, clock)
    job = service.create(JobSpec(
        title="Fix checkout race",
        description="Make checkout idempotent.",
        repository="/repo",
        acceptance_criteria=["duplicate callback is safe"],
        validation_commands=["pytest tests/checkout"],
        max_cost_usd=5,
        execution={
            "provider": "openai",
            "model": "test-model",
            "agent_mode": "default",
            "session_type": "workspace",
            "auto_approve_writes": True,
        },
    ))
    assert job.status == JobStatus.QUEUED
    assert job.version == 1
    assert service.events(job.id)[0].event_type == "job_created"

    reopened = JobService(JobStore(str(tmp_path / "jobs.sqlite3"), clock=clock)).get(job.id)
    assert reopened.title == "Fix checkout race"
    assert reopened.acceptance_criteria == ["duplicate callback is safe"]
    assert reopened.validation_commands == ["pytest tests/checkout"]
    assert reopened.max_cost_usd == 5
    assert reopened.execution["provider"] == "openai"
    assert reopened.execution["model"] == "test-model"
    assert reopened.execution["auto_approve_writes"] is True


def test_store_uses_wal_and_schema_v2(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    JobStore(path)
    conn = sqlite3.connect(path)
    try:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        version = conn.execute("SELECT value FROM job_meta WHERE key='schema_version'").fetchone()[0]
        assert version == "2"
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        assert "execution_json" in columns
    finally:
        conn.close()


def test_state_machine_rejects_illegal_transition_and_requires_attention_reason(tmp_path):
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Ticket"))
    with pytest.raises(JobStateError):
        service.transition(job.id, JobStatus.READY_FOR_REVIEW)

    service.transition(job.id, JobStatus.PREPARING)
    service.transition(job.id, JobStatus.RUNNING)
    with pytest.raises(JobStateError):
        service.transition(job.id, JobStatus.NEEDS_HUMAN)

    blocked = service.require_human(
        job.id,
        AttentionReason.AMBIGUOUS_REQUIREMENT,
        "Which compatibility behavior should win?",
    )
    assert blocked.status == JobStatus.NEEDS_HUMAN
    assert blocked.needs_attention
    resumed = service.resume(job.id, detail="Preserve current API behavior")
    assert resumed.status == JobStatus.QUEUED
    assert resumed.attention_reason == AttentionReason.NONE
    assert any(event.event_type == "human_response" for event in service.events(job.id))


def test_transition_is_versioned_and_event_is_written_atomically(tmp_path):
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Versioned"))
    next_job = service.transition(
        job.id,
        JobStatus.PREPARING,
        expected_version=job.version,
        reason="scheduler claimed job",
    )
    assert next_job.version == job.version + 1
    with pytest.raises(RuntimeError):
        service.transition(job.id, JobStatus.RUNNING, expected_version=job.version)
    transition = service.events(job.id)[-1]
    assert transition.from_status == JobStatus.QUEUED
    assert transition.to_status == JobStatus.PREPARING
    assert transition.reason == "scheduler claimed job"


def test_worker_lease_is_exclusive_and_expired_run_becomes_recovering(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, clock)
    job = service.create(JobSpec(title="Unattended"))
    service.transition(job.id, JobStatus.PREPARING)
    service.transition(job.id, JobStatus.RUNNING)
    assert service.acquire(job.id, "worker-a", ttl_seconds=10)
    assert not service.acquire(job.id, "worker-b", ttl_seconds=10)
    clock.advance(11)
    recovered = service.recover_expired_leases()
    assert [item.id for item in recovered] == [job.id]
    current = service.get(job.id)
    assert current.status == JobStatus.RECOVERING
    assert current.worker_id == ""
    assert any(
        event.reason == "implementation worker lease expired"
        for event in service.events(job.id)
    )


def test_recovered_lease_that_renewed_is_left_alone(tmp_path):
    """Round-36 F1: recovery must CLAIM the expired lease (CAS) BEFORE
    transitioning/replaying. A worker that renewed between the
    expired_leases() snapshot and the claim keeps its fresh lease and its
    RUNNING job — the old act-then-release order yanked the job to
    RECOVERING while the live worker kept going (two owners)."""
    clock = Clock()
    service = make_service(tmp_path, clock)
    job = service.create(JobSpec(title="Renewer"))
    service.transition(job.id, JobStatus.PREPARING)
    service.transition(job.id, JobStatus.RUNNING)
    assert service.acquire(job.id, "worker-a", ttl_seconds=10)
    clock.advance(11)  # lease now expired from the snapshot's perspective
    assert service.heartbeat(job.id, "worker-a", ttl_seconds=10)  # renewal
    assert service.recover_expired_leases() == []
    current = service.get(job.id)
    assert current.status == JobStatus.RUNNING  # untouched
    assert current.worker_id == "worker-a"  # fresh lease preserved


def test_assert_lease_rejects_expired_lease(tmp_path):
    """Round-36 F2: an expired lease is NOT ownership — a worker whose
    heartbeat died must fail the final gate instead of applying results
    to a job that recovery may already be reassigning."""
    clock = Clock()
    service = make_service(tmp_path, clock)
    job = service.create(JobSpec(title="Lapsed"))
    service.transition(job.id, JobStatus.PREPARING)
    service.transition(job.id, JobStatus.RUNNING)
    assert service.acquire(job.id, "worker-a", ttl_seconds=10)
    assert service.store.assert_lease(job.id, "worker-a") is True
    clock.advance(11)
    assert service.store.assert_lease(job.id, "worker-a") is False
    # finish_attempt_owned refuses to write under an expired lease.
    attempt = service.start_attempt(job.id, worker_id="worker-a")
    assert service.store.finish_attempt_owned(
        attempt.id, worker_id="worker-a", status="completed"
    ) is False
    stored = service.store.get_attempt(attempt.id)
    assert stored.status == "running"  # untouched


def test_timed_out_reachable_from_all_worker_active_states(tmp_path):
    """Round-38 F3: the whole-job runtime deadline must be enforceable in
    EVERY worker-active phase — PREPARING, VERIFYING, and RECOVERING were
    missing TIMED_OUT transitions, so a hung worktree prepare or verifier
    could never be reaped."""
    for setup in ("prepare", "verify", "recover"):
        clock = Clock()
        service = make_service(tmp_path / setup, clock)
        job = service.create(JobSpec(title=f"Timeout {setup}"))
        if setup == "prepare":
            service.transition(job.id, JobStatus.PREPARING)
        elif setup == "verify":
            service.transition(job.id, JobStatus.PREPARING)
            service.transition(job.id, JobStatus.RUNNING)
            service.transition(job.id, JobStatus.VERIFYING)
        else:
            service.transition(job.id, JobStatus.PREPARING)
            service.transition(job.id, JobStatus.RECOVERING)
        current = service.transition(job.id, JobStatus.TIMED_OUT)
        assert current.status == JobStatus.TIMED_OUT, setup


def test_attempt_records_survive_and_are_numbered(tmp_path):
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Attempts"))
    first = service.start_attempt(job.id, worker_id="worker-a", session_name="job-1")
    completed = service.finish_attempt(first.id, status="failed", error="provider 500", cost_usd=0.12)
    second = service.start_attempt(job.id, worker_id="worker-b", session_name="job-1-retry")
    assert completed.number == 1
    assert completed.status == "failed"
    assert completed.cost_usd == pytest.approx(0.12)
    assert second.number == 2
    assert [attempt.number for attempt in service.attempts(job.id)] == [1, 2]


def test_retryable_failure_states_return_to_queue(tmp_path):
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Retry"))
    service.transition(job.id, JobStatus.PREPARING)
    service.transition(job.id, JobStatus.ENVIRONMENT_ERROR, reason="docker unavailable")
    assert service.retry(job.id).status == JobStatus.QUEUED


def test_job_spec_rejects_invalid_budgets_and_execution_type(tmp_path):
    service = make_service(tmp_path)
    with pytest.raises(ValueError):
        service.create(JobSpec(title="Bad", max_cost_usd=0))
    with pytest.raises(ValueError):
        service.create(JobSpec(title="Bad", max_runtime_seconds=-1))
    with pytest.raises(ValueError):
        service.create(JobSpec(title="Bad", max_retries=-1))
    with pytest.raises(ValueError):
        service.create(JobSpec(title="Bad", execution={"session_type": "spaceship"}))


def test_archived_history_is_hidden_from_board_and_queryable(tmp_path):
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Historic drone job", repository="/repo"))
    service.cancel(job.id, reason="done")
    management = JobManagementService(service)

    assert any(item.id == job.id for item in build_job_board(service).done)
    management.archive(job.id, reason="old")
    assert not any(item.id == job.id for item in build_job_board(service).done)

    history = management.query_jobs(q="drone", archive="archived", scope="history")
    report = management.report(q="drone", archive="archived", scope="history")
    assert history["total"] == 1
    assert history["jobs"][0]["archived"] is True
    assert report["total_jobs"] == 1
    assert report["archived_jobs"] == 1


def test_historic_job_requires_archive_before_delete_and_cascades_events(tmp_path):
    service = make_service(tmp_path)
    job = service.create(JobSpec(title="Delete history", repository="/repo"))
    service.cancel(job.id, reason="done")
    management = JobManagementService(service)

    with pytest.raises(JobManagementError, match="Archive a historic job"):
        management.delete(job.id, purge_artifacts=False)

    management.archive(job.id)
    result = management.delete(job.id, purge_artifacts=False)
    assert result["deleted"] is True
    with pytest.raises(KeyError):
        service.get(job.id)
    conn = service.store._connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM job_events WHERE job_id = ?", (job.id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_management WHERE job_id = ?", (job.id,)).fetchone()[0] == 0
    finally:
        conn.close()
