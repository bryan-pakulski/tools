from __future__ import annotations

from types import SimpleNamespace

from mu.jobs import JobService, JobSpec, JobStore
from mu.jobs.runner import SessionJobRunner
from mu.ui.exceptions import InteractionRequired


class FakeManager:
    def __init__(self):
        self.token_counts = {"total_cost": 1.0}
        self.saved = 0
        self.ui = None
        self.revision = 0

    def save_history(self, folder_context, expected_revision=None):
        self.saved += 1
        self.revision += 1


class FakeSession:
    def __init__(self, *, outcome=None, gate=None):
        self.variables = {}
        self.folder_context = SimpleNamespace(folders=[])
        self.session_manager = FakeManager()
        self.ui = None
        self.outcome = outcome or {"status": "completed"}
        self.gate = gate
        self.prompts = []
        self.shutdown_called = False

    def send_message(self, text):
        self.prompts.append(text)
        self.session_manager.token_counts["total_cost"] = 1.75
        if self.gate is not None:
            raise self.gate
        return self.outcome

    def shutdown(self):
        self.shutdown_called = True


def make_service(tmp_path):
    return JobService(JobStore(str(tmp_path / "jobs.sqlite3")))


def make_job(service, repo, execution=None):
    return service.create(JobSpec(
        title="Implement durable thing",
        description="Do the requested change.",
        repository=str(repo),
        acceptance_criteria=["works"],
        validation_commands=["pytest"],
        execution=execution or {
            "provider": "openai",
            "model": "test-model",
            "agent_mode": "default",
            "session_type": "workspace",
        },
    ))


def test_runner_uses_existing_session_runtime_and_persists_job_context(tmp_path):
    service = make_service(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    job = make_job(service, repo)
    attempt = service.start_attempt(job.id, worker_id="worker", session_name=f"job-{job.id[:20]}")
    built = []
    session = FakeSession()

    def build(args, ui, allow_prompt):
        built.append((args, ui, allow_prompt))
        session.folder_context.folders = list(args.workspace)
        return session

    base_args = SimpleNamespace(gui=True, trace=False, workspace=[], yolo=False)
    runner = SessionJobRunner(service, build_session_fn=build, base_args=base_args)
    outcome = runner.run(job, attempt)

    assert outcome.kind == "completed"
    assert outcome.cost_usd == 0.75
    assert built[0][0].session == f"job-{job.id[:20]}"
    assert built[0][0].provider == "openai"
    assert built[0][0].model == "test-model"
    assert built[0][0].workspace == [str(repo)]
    assert built[0][0].trace is True
    assert built[0][2] is False
    assert session.variables["durable_job_id"] == job.id
    assert "Acceptance criteria:" in session.prompts[0]
    assert "Validation expected by the controller:" in session.prompts[0]
    assert session.shutdown_called


def test_runner_converts_noninteractive_gate_to_needs_human(tmp_path):
    service = make_service(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    job = make_job(service, repo)
    attempt = service.start_attempt(job.id)
    session = FakeSession(gate=InteractionRequired(
        "approval_required",
        "Approval required for write_file",
        payload={"tool_name": "write_file"},
    ))

    runner = SessionJobRunner(
        service,
        build_session_fn=lambda args, ui, allow_prompt: session,
        base_args=SimpleNamespace(gui=True, trace=False, workspace=[], yolo=False),
    )
    outcome = runner.run(job, attempt)
    assert outcome.kind == "needs_human"
    assert outcome.attention_reason.value == "approval_required"
    assert outcome.attention_payload["tool_name"] == "write_file"
    assert outcome.cost_usd == 0.75


def test_runner_missing_execution_profile_gates_instead_of_guessing(tmp_path):
    service = make_service(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    job = make_job(service, repo, execution={"session_type": "workspace"})
    attempt = service.start_attempt(job.id)
    runner = SessionJobRunner(
        service,
        build_session_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
        base_args=SimpleNamespace(gui=True, trace=False, workspace=[], yolo=False),
    )
    outcome = runner.run(job, attempt)
    assert outcome.kind == "needs_human"
    assert outcome.attention_payload["shape"] == "execution_profile"


def test_runner_missing_workspace_is_environment_failure(tmp_path):
    service = make_service(tmp_path)
    job = make_job(service, tmp_path / "does-not-exist")
    attempt = service.start_attempt(job.id)
    runner = SessionJobRunner(
        service,
        build_session_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
        base_args=SimpleNamespace(gui=True, trace=False, workspace=[], yolo=False),
    )
    outcome = runner.run(job, attempt)
    assert outcome.kind == "failed"
    assert outcome.status == "environment_error"
