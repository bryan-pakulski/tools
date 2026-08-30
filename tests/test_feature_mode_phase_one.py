import os
import pytest
import tempfile

from mu.feature.engine import (
    STATUS_ARCHIVED,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    STATUS_NOT_STARTED,
    create_feature_plan,
    is_valid_task_transition,
    load_feature_plan,
    transition_task_status,
)


@pytest.fixture(autouse=True)
def _isolate_feature_writes(tmp_path, monkeypatch):
    """The feature engine resolves its workspace root from `os.getcwd()`
    when `folder_context=None`. Chdir into the per-test tmp_path so the
    `feature_plan.json` file lands in /tmp instead
    of the repo's `documentation/`."""
    monkeypatch.chdir(tmp_path)


def test_create_feature_plan_populates_phase_and_event_foundation(tmp_path):
    plan = create_feature_plan(
        feature_name="Board",
        feature_request="Implement feature board",
        tasks_data=[{"title": "Task A", "objectives": ["Goal A"]}],
        folder_context=None,
        metadata_path=str(tmp_path / "feature_plan.json"),
        feature_id="board",
    )

    assert len(plan.tasks) == 1
    assert len(plan.phases_meta) == 1
    assert plan.phases_meta[0].task_ids == [plan.tasks[0].id]
    assert plan.event_log == []


def test_transition_task_status_records_event_and_updates_state(tmp_path):
    plan = create_feature_plan(
        feature_name="Transitions",
        feature_request="Test transitions",
        tasks_data=[{"title": "Task A"}],
        folder_context=None,
        metadata_path=str(tmp_path / "feature_plan.json"),
        feature_id="transitions",
    )

    updated_task = transition_task_status(
        plan,
        task_id=1,
        to_status=STATUS_IN_PROGRESS,
        notes="Started",
        actor="agent",
    )
    assert updated_task.status == STATUS_IN_PROGRESS
    assert len(plan.event_log) == 1
    assert plan.event_log[0].payload["to_status"] == STATUS_IN_PROGRESS



def test_status_transition_matrix_supports_execution_lifecycle():
    assert is_valid_task_transition(STATUS_NOT_STARTED, STATUS_IN_PROGRESS)
    assert is_valid_task_transition(STATUS_IN_PROGRESS, STATUS_BLOCKED)
    assert is_valid_task_transition(STATUS_BLOCKED, STATUS_IN_PROGRESS)
    assert is_valid_task_transition(STATUS_IN_PROGRESS, STATUS_COMPLETED)
    assert is_valid_task_transition(STATUS_COMPLETED, STATUS_ARCHIVED)
    assert not is_valid_task_transition(STATUS_ARCHIVED, STATUS_IN_PROGRESS)



def test_load_feature_plan_backfills_phase_metadata_for_legacy_files(tmp_path):
    plan = create_feature_plan(
        feature_name="Legacy",
        feature_request="Backfill",
        tasks_data=[{"title": "Task A"}],
        folder_context=None,
        metadata_path=str(tmp_path / "feature_plan.json"),
        feature_id="legacy",
    )

    # Simulate an older metadata format by removing newer fields.
    import json

    with open(plan.metadata_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("phases_meta", None)
    data.pop("event_log", None)
    with open(plan.metadata_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)

    loaded = load_feature_plan(plan.metadata_path)
    assert len(loaded.phases_meta) == 1
    assert loaded.phases_meta[0].title == "Task A"


def test_update_task_content_refuses_empty_exit_criteria():
    """Round-30 F1: stripping all exit criteria would make the
    completion gate vacuously pass (`missing` is empty when `expected`
    is empty). The engine refuses empty criteria on update; the create
    path already required non-empty."""
    import pytest as _pytest

    from mu.feature.engine import (
        FeaturePlan,
        FeatureTask,
        load_feature_plan,
        save_feature_plan,
        update_task_content,
    )

    plan = FeaturePlan(
        feature_id="f1",
        feature_name="n",
        feature_request="r",
        directory=tempfile.mkdtemp(),
    )
    plan.tasks.append(
        FeatureTask(id=1, title="task", exit_criteria=["criterion A"])
    )
    metadata_path = os.path.join(plan.directory, "feature_plan.json")
    plan.metadata_path = metadata_path
    save_feature_plan("", plan)

    with _pytest.raises(ValueError, match="exit_criteria cannot be emptied"):
        update_task_content(metadata_path, 1, exit_criteria=[])

    # Non-empty replacement still works.
    update_task_content(metadata_path, 1, exit_criteria=["A", "B"])
    reloaded = load_feature_plan(metadata_path)
    assert reloaded.tasks[0].exit_criteria == ["A", "B"]
