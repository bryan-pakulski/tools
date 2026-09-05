from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

from bench.prepare_tb import check_prepared, stage_task, task_source_sha256
from bench.summarize_tb import summarize
from bench.write_tb_provenance import _prepared_images
from bench.tb_support import (
    build_source_tarball,
    read_task_execution_timeout,
    read_trace_usage,
    stop_mucli_process,
)
from mucli import apply_tool_profile


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_source_snapshot_uses_tracked_worktree_without_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "app.py").write_text("committed\n", encoding="utf-8")
    (repo / "bench" / "results").mkdir(parents=True)
    (repo / "bench" / "results" / "old.json").write_text("{}", encoding="utf-8")
    _git(repo, "add", "app.py", "bench/results/old.json")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MuCLI tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    # The benchmark must exercise tracked edits, while an untracked credential
    # must never be copied into an arbitrary task container.
    (repo / "app.py").write_text("working tree\n", encoding="utf-8")
    (repo / ".env").write_text("API_KEY=secret\n", encoding="utf-8")

    archive_path = build_source_tarball(repo)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            names = set(archive.getnames())
            assert archive.extractfile("app.py").read() == b"working tree\n"
            metadata = json.load(archive.extractfile(".mucli-benchmark-source.json"))
    finally:
        archive_path.unlink(missing_ok=True)

    assert ".env" not in names
    assert "bench/results/old.json" not in names
    assert metadata["tracked_worktree_dirty"] is True
    assert metadata["file_count"] == 1
    assert len(metadata["source_sha256"]) == 64


def test_trace_usage_prefers_run_end_and_skips_incomplete_newer_trace(tmp_path):
    logging_dir = tmp_path / "agent-logs"
    trace_dir = tmp_path / "sessions" / "mucli" / "trace"
    trace_dir.mkdir(parents=True)
    complete = trace_dir / "complete.jsonl"
    complete.write_text(
        "not-json\n"
        + json.dumps({"type": "turn_end", "total_in": 10, "total_out": 2})
        + "\n"
        + json.dumps({"type": "run_end", "tokens_in": 12, "tokens_out": 3})
        + "\n",
        encoding="utf-8",
    )
    incomplete = trace_dir / "newer.jsonl"
    incomplete.write_text(
        json.dumps({"type": "iter", "tokens": {"in": 99, "out": 99}}) + "\n",
        encoding="utf-8",
    )
    incomplete.touch()

    assert read_trace_usage(logging_dir) == (12, 3)


def test_trace_usage_sums_iterations_when_timeout_prevents_run_end(tmp_path):
    logging_dir = tmp_path / "agent-logs"
    trace_dir = tmp_path / "sessions" / "mucli" / "trace"
    trace_dir.mkdir(parents=True)
    (trace_dir / "timeout.jsonl").write_text(
        json.dumps({"type": "iter", "tokens": {"in": 100, "out": 7}})
        + "\n"
        + "truncated-json"
        + "\n"
        + json.dumps({"type": "iter", "tokens": {"in": 250, "out": 11}})
        + "\n",
        encoding="utf-8",
    )

    assert read_trace_usage(logging_dir) == (350, 18)


def test_timed_out_mucli_is_stopped_before_verification():
    class Container:
        def __init__(self):
            self.calls = []

        def exec_run(self, command):
            self.calls.append(command)
            return SimpleNamespace(exit_code=0)

    container = Container()
    sent = []
    session = SimpleNamespace(
        container=container,
        send_keys=lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    stop_mucli_process(
        session,
        polls_per_signal=1,
        poll_interval=0,
        sleep=lambda _seconds: None,
    )

    assert sent == [((["C-c"],), {"block": False, "min_timeout_sec": 0.0})]
    shell_commands = [call[2] for call in container.calls if call[:2] == ["sh", "-c"]]
    assert any("pkill -TERM" in command for command in shell_commands)
    assert any("pkill -KILL" in command for command in shell_commands)


def test_prepared_task_uses_content_addressed_image_and_detects_drift(tmp_path):
    source = tmp_path / "source"
    task = source / "hello-world"
    task.mkdir(parents=True)
    (task / "Dockerfile").write_text("FROM example/base:1\n", encoding="utf-8")
    (task / "task.yaml").write_text("max_agent_timeout_sec: 360\n", encoding="utf-8")
    (task / "docker-compose.yaml").write_text(
        "services:\n"
        "  client:\n"
        "    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}\n",
        encoding="utf-8",
    )
    source_sha = task_source_sha256(task)
    image_name = f"mucli-test/hello:{source_sha[:16]}"
    output = tmp_path / "prepared"
    stage_task(task, output / "hello-world", image_name)
    (output / "prepare-manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "tasks": {
                    "hello-world": {
                        "source_sha256": source_sha,
                        "image_name": image_name,
                        "image_id": "sha256:image-one",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    compose = (output / "hello-world" / "docker-compose.yaml").read_text(
        encoding="utf-8"
    )
    assert f"image: {image_name}" in compose
    healthy, errors = check_prepared(
        source,
        output,
        ["hello-world"],
        inspect_image=lambda _name: {"Id": "sha256:image-one"},
    )
    assert healthy is True
    assert errors == []

    (task / "Dockerfile").write_text("FROM example/base:2\n", encoding="utf-8")
    healthy, errors = check_prepared(
        source,
        output,
        ["hello-world"],
        inspect_image=lambda _name: {"Id": "sha256:image-one"},
    )
    assert healthy is False
    assert errors == ["hello-world: source task changed since image preparation"]


def test_provenance_includes_selected_prepared_image_ids(tmp_path):
    manifest = tmp_path / "prepare-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "dataset": "terminal-bench-core==0.1.1",
                "tasks": {
                    "hello-world": {"image_id": "sha256:hello"},
                    "fix-git": {"image_id": "sha256:git"},
                },
            }
        ),
        encoding="utf-8",
    )

    images = _prepared_images(manifest, ["hello-world"])

    assert images["dataset"] == "terminal-bench-core==0.1.1"
    assert images["tasks"] == {"hello-world": {"image_id": "sha256:hello"}}


def test_terminal_bench_tool_profile_removes_unrelated_schemas():
    session = SimpleNamespace(disabled_tools=["already-disabled"])
    tools = [
        SimpleNamespace(name="bash"),
        SimpleNamespace(name="write_file"),
        SimpleNamespace(name="web_search"),
        SimpleNamespace(name="create_course"),
    ]

    apply_tool_profile(session, "terminal-bench", tools=tools)

    assert session.disabled_tools == [
        "already-disabled",
        "create_course",
        "web_search",
    ]


def test_summary_reports_execution_time_separately_from_setup(tmp_path):
    run_dir = tmp_path / "hello-world" / "run-1"
    trial_dir = run_dir / "hello-world" / "trial-1"
    trial_dir.mkdir(parents=True)
    result = {
        "task_id": "hello-world",
        "is_resolved": True,
        "failure_mode": "unset",
        "agent_started_at": "2026-09-04T01:00:00+00:00",
        "agent_ended_at": "2026-09-04T01:00:02.500000+00:00",
    }
    (run_dir / "results.json").write_text(
        json.dumps({"results": [result]}), encoding="utf-8"
    )
    # This nested copy caused the old summary implementation to double-count.
    (trial_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
    agent_logs = trial_dir / "agent-logs"
    agent_logs.mkdir()
    (agent_logs / "mucli-execution.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "execution_seconds": 1.25,
                "setup_seconds": 9.75,
                "execution_timeout_seconds": 360.0,
                "completed": True,
                "timed_out": False,
                "error_type": None,
            }
        ),
        encoding="utf-8",
    )

    lines, healthy = summarize(tmp_path, ["hello-world", "missing-task"])

    assert healthy is False
    assert sum("hello-world" in line for line in lines) == 1
    assert any(
        "PASS" in line and "1.2s execution (+9.8s setup excluded)" in line
        for line in lines
    )
    assert any("missing-task" in line and "NO-RESULT" in line for line in lines)
    assert lines[-1] == "pack score: 1/1"


def test_summary_labels_terminal_bench_fallback_as_setup_inclusive(tmp_path):
    run_dir = tmp_path / "hello-world" / "run-1"
    run_dir.mkdir(parents=True)
    result = {
        "task_id": "hello-world",
        "is_resolved": True,
        "failure_mode": "unset",
        "agent_started_at": "2026-09-04T01:00:00+00:00",
        "agent_ended_at": "2026-09-04T01:00:02.500000+00:00",
    }
    (run_dir / "results.json").write_text(
        json.dumps({"results": [result]}), encoding="utf-8"
    )

    lines, healthy = summarize(tmp_path, ["hello-world"])

    assert healthy is True
    assert any("2.5s TB-agent (setup included)" in line for line in lines)


def test_task_execution_timeout_uses_native_task_budget(tmp_path):
    config = tmp_path / "task.yaml"
    config.write_text(
        "instruction: do the thing\nmax_agent_timeout_sec: 480.5\n",
        encoding="utf-8",
    )

    assert read_task_execution_timeout(config) == 480.5


def test_tb_suite_uses_terminal_bench_dataset_config_schema():
    config = Path("bench/tb_suite.yaml").read_text(encoding="utf-8")
    assert "name: terminal-bench-core" in config
    assert "version: 0.1.1" in config
    assert "task_ids:\n  - hello-world" in config
    assert "\ndataset:" not in config
    assert "\nagent:" not in config


def test_pack_requires_python_314_and_supports_repeated_prebuilt_runs():
    wheelhouse = Path("bench/build_wheelhouse.sh").read_text(encoding="utf-8")
    setup = Path("bench/mucli-setup.sh.j2").read_text(encoding="utf-8")
    runner = Path("bench/run_pack.sh").read_text(encoding="utf-8")

    assert "3.10 3.11 3.12 3.13 3.14" in wheelhouse
    assert "MUCLI_WHEELHOUSE_PYTHON_UNSUPPORTED" in setup
    assert '--n-attempts "$ATTEMPTS"' in runner
    assert "--no-rebuild --no-cleanup" in runner
    assert "write_tb_provenance.py" in runner
