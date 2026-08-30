import os
import pytest
import json
from mu.tools._dispatcher import execute_tool, TOOL_HANDLERS
from mu.tools.descriptors import ToolExecutionContext, TOOLS
from mu.tools._bounds import check_bounds as _check_bounds
from mu.tools.feature.handlers import _handle_create_feature_task, _handle_approve_feature_task
from mu.tools.research.handlers import stackoverflow_search, web_search
from mu.tools.workspace.handlers import read_file
from mu.workspace.folder_context import FolderContext


@pytest.fixture(autouse=True)
def _isolate_feature_writes(tmp_path, monkeypatch):
    """Feature handlers fall back to `os.getcwd()` for the workspace
    root. Chdir into tmp_path so `feature_plan.json` files
    land in /tmp instead of the repo."""
    monkeypatch.chdir(tmp_path)


class _FeatureSessionManagerStub:
    def __init__(self, metadata_path: str):
        self._metadata_path = metadata_path
        self.feature = None
        self.active_feature_id = None
        self.saved = False
        self.record = None
        self.feature_state = None

    def get_feature(self, feature_id):
        return self.feature

    def get_feature_metadata_path(self, feature_id):
        return self._metadata_path

    def allocate_feature_id(self, requested_id):
        return requested_id

    def upsert_feature(self, feature_record):
        self.record = feature_record

    def activate_feature(self, feature_id):
        self.active_feature_id = feature_id

    def save_history(self):
        self.saved = True

    def save_history_turn(self, folder_context_obj=None):
        self.save_history()

    def get_feature_state(self):
        return self.feature_state or self.record

    def set_feature_state(self, state, folder_context=None):
        self.feature_state = state


class _SessionStub:
    def __init__(self, metadata_path: str):
        self.session_manager = _FeatureSessionManagerStub(metadata_path)


def _assert_tool_envelope(payload: dict):
    required = {"ok", "error_code", "message", "data", "artifacts", "telemetry"}
    assert required.issubset(payload.keys())


def test_workspace_boundaries(tmp_path):
    ctx = FolderContext()
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()

    # Add 'safe' directory to tracked folders
    ctx.add_folder(str(safe_dir))

    # Safe file inside
    assert _check_bounds(str(safe_dir / "test.txt"), ctx) is True

    # File outside should be blocked
    outside_file = tmp_path / "outside.txt"
    assert _check_bounds(str(outside_file), ctx) is False

    # Relative path navigation outside workspace
    hacked_path = str(safe_dir / ".." / "outside.txt")
    assert _check_bounds(hacked_path, ctx) is False


def test_read_file_boundary_enforcement(tmp_path):
    ctx = FolderContext()
    safe_dir = tmp_path / "workspace"
    safe_dir.mkdir()
    ctx.add_folder(str(safe_dir))

    out_file = tmp_path / "secret.txt"
    out_file.write_text("top secret")

    result = read_file(str(out_file), ctx)
    assert "Error: Access denied" in result
    assert "top secret" not in result


def test_read_file_not_found(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))

    result = read_file(str(tmp_path / "missing.py"), ctx)
    assert "Error: File" in result
    assert "not found" in result


def test_create_feature_task_accepts_json_string_tasks(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))
    session = _SessionStub(str(tmp_path / "feature_plan.json"))
    tool_ctx = ToolExecutionContext(folder_context=ctx, session=session)
    tasks = [
        {
            "title": "Initialize",
            "objectives": ["Set up"],
            "action_points": ["Create files"],
            "exit_criteria": ["Files exist"],
        }
    ]

    result = _handle_create_feature_task(
        {
            "feature_name": "Visibility",
            "feature_request": "Add failure visibility",
            "tasks": json.dumps(tasks),
        },
        tool_ctx,
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["task_count"] == 1
    assert session.session_manager.record is not None


def test_create_feature_task_rejects_non_object_task_items(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))
    session = _SessionStub(str(tmp_path / "feature_plan.json"))
    tool_ctx = ToolExecutionContext(folder_context=ctx, session=session)

    result = _handle_create_feature_task(
        {
            "feature_name": "Visibility",
            "feature_request": "Add failure visibility",
            "tasks": [1, 2],
        },
        tool_ctx,
    )

    assert "tasks must be an array of objects" in result


def test_execute_tool_rejects_non_dict_args(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))

    result = execute_tool("read_file", "not-a-dict", ctx)
    payload = json.loads(result)
    _assert_tool_envelope(payload)
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_args"
    assert "arguments must be an object/dict" in payload["message"]


def test_execute_tool_converts_handler_exception_to_error(tmp_path, monkeypatch):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))

    def _boom(args, context):
        raise RuntimeError("boom")

    monkeypatch.setitem(TOOL_HANDLERS, "read_file", _boom)
    result = execute_tool("read_file", {"filename": "x.txt"}, ctx)
    payload = json.loads(result)
    _assert_tool_envelope(payload)
    assert payload["ok"] is False
    assert payload["error_code"] == "execution_failed"
    assert "Tool 'read_file' failed with RuntimeError: boom" in payload["message"]


def test_all_tools_return_schema_valid_envelope_on_invalid_args(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))

    for tool in TOOLS:
        raw = execute_tool(tool.name, "bad-args", ctx)
        payload = json.loads(raw)
        _assert_tool_envelope(payload)
        assert payload["ok"] is False
        assert payload["error_code"] == "invalid_args"


def test_error_code_coverage_core_categories(tmp_path, monkeypatch):
    ctx = FolderContext()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx.add_folder(str(workspace))

    # invalid_args
    invalid_payload = json.loads(execute_tool("read_file", "bad-args", ctx))
    assert invalid_payload["error_code"] == "invalid_args"

    # not_found
    not_found_payload = json.loads(execute_tool("missing_tool", {}, ctx))
    assert not_found_payload["error_code"] == "not_found"

    # access_denied
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret")
    denied_payload = json.loads(
        execute_tool("read_file", {"filename": str(outside_file)}, ctx)
    )
    assert denied_payload["error_code"] == "access_denied"

    # execution_failed
    def _boom(args, context):
        raise RuntimeError("boom")

    monkeypatch.setitem(TOOL_HANDLERS, "read_file", _boom)
    failed_payload = json.loads(
        execute_tool("read_file", {"filename": str(workspace / "x.txt")}, ctx)
    )
    assert failed_payload["error_code"] == "execution_failed"


def test_bash_tool_executes_raw_command(tmp_path):
    ctx = FolderContext()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx.add_folder(str(workspace))

    payload = json.loads(execute_tool("bash", {"command": "pwd"}, ctx))
    _assert_tool_envelope(payload)
    assert payload["ok"] is True
    assert str(workspace) in payload["message"]


def test_bash_tool_rejects_out_of_bounds_cwd(tmp_path):
    ctx = FolderContext()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx.add_folder(str(workspace))

    outside = tmp_path / "outside"
    outside.mkdir()
    payload = json.loads(
        execute_tool("bash", {"command": "pwd", "cwd": str(outside)}, ctx)
    )
    _assert_tool_envelope(payload)
    assert payload["ok"] is False
    assert payload["error_code"] == "access_denied"


def test_web_search_duckduckgo_fallback_works_without_ddgs(monkeypatch):
    class _FakeResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(request, timeout=0):
        return _FakeResponse(
            {
                "Heading": "DuckDuckGo",
                "AbstractURL": "https://duckduckgo.com/about",
                "AbstractText": "DuckDuckGo overview",
                "RelatedTopics": [
                    {"FirstURL": "https://example.com/a", "Text": "Example A - snippet"},
                    {"FirstURL": "https://example.com/b", "Text": "Example B - snippet"},
                ],
            }
        )

    def _raise_import_error(query: str, max_results: int):
        raise ImportError("forced missing ddgs")

    def _httpx_offline(*args, **kwargs):
        # The HTML-scrape fallback fetches via httpx (not urllib). Force it
        # to fail so we fall through to the InstantAnswer fallback, which
        # uses urllib.request.urlopen (patched below) — keeping the test
        # offline and deterministic.
        raise RuntimeError("forced offline: httpx disabled in test")

    monkeypatch.setattr("mu.tools.research.handlers._ddgs_text_search", _raise_import_error)
    monkeypatch.setattr("httpx.get", _httpx_offline)
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    payload = json.loads(web_search("privacy search", engine="duckduckgo", num_results=3))

    assert payload.get("error") is None
    assert payload["num_results"] >= 1
    assert payload["results"][0]["url"].startswith("https://")
    # InstantAnswer abstract is the first result.
    assert payload["results"][0]["url"] == "https://duckduckgo.com/about"


def test_normalize_ddg_href_unwraps_redirect_and_protocol_relative():
    """DuckDuckGo HTML results come back as protocol-relative redirect
    wrappers; the normalizer must unwrap them to absolute https URLs."""
    from mu.tools.research.handlers import _normalize_ddg_href

    # Redirect wrapper with url-encoded target → decoded target.
    assert (
        _normalize_ddg_href("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath&rut=x")
        == "https://example.com/path"
    )
    # Plain protocol-relative → https.
    assert _normalize_ddg_href("//example.com/foo") == "https://example.com/foo"
    # Already absolute → unchanged.
    assert _normalize_ddg_href("https://example.com/x") == "https://example.com/x"
    # Empty / falsy → empty.
    assert _normalize_ddg_href("") == ""
    assert _normalize_ddg_href(None) == ""


def test_web_search_returns_unknown_engine_error():
    payload = json.loads(web_search("hello", engine="bogus"))
    assert "Unknown search engine" in payload["error"]
    assert payload["results"] == []


def test_stackoverflow_search_rejects_empty_query():
    payload = json.loads(stackoverflow_search(""))
    assert "error" in payload
    assert payload["results"] == []


def test_create_feature_create_phases_create_task_staged_flow(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))
    session = _SessionStub(str(tmp_path / "feature_plan.json"))

    create_result = execute_tool(
        "create_feature",
        {
            "feature_name": "Staged Feature",
            "feature_request": "Plan feature in staged tools",
            "feature_id": "staged_feature",
            "design_plan": "Initial design",
        },
        ctx,
        session=session,
    )
    create_payload = json.loads(create_result)
    assert create_payload["ok"] is True
    assert create_payload["feature_id"] == "staged_feature"

    # Approve the feature plan before creating phases/tasks
    approve_result = execute_tool(
        "approve_feature_task",
        {"approved": True},
        ctx,
        session=session,
    )
    approve_payload = json.loads(approve_result)
    assert approve_payload["ok"] is True

    phases_result = execute_tool(
        "create_phases",
        {
            "feature_id": "staged_feature",
            "phases": [
                {"id": 1, "title": "Phase 1", "goal": "Ship planning API", "order": 1}
            ],
        },
        ctx,
        session=session,
    )
    phases_payload = json.loads(phases_result)
    assert phases_payload["ok"] is True
    assert phases_payload["phase_count"] == 1

    task_result = execute_tool(
        "create_task",
        {
            "feature_id": "staged_feature",
            "phase_id": 1,
            "title": "Create stage-one planner",
            "overview": "Build stage one dialogue loop",
            "design": ["Handle ambiguity options", "Persist review notes"],
            "exit_criteria": ["Tool payload is persisted"],
        },
        ctx,
        session=session,
    )
    task_payload = json.loads(task_result)
    assert task_payload["ok"] is True
    assert task_payload["task_id"] == 1

    execution_result = execute_tool(
        "get_execution_state",
        {"feature_id": "staged_feature"},
        ctx,
        session=session,
    )
    execution_payload = json.loads(execution_result)
    assert execution_payload["ok"] is True
    assert execution_payload["execution"]["next_phase"]["id"] == 1
    assert execution_payload["execution"]["next_task"]["id"] == 1

    blocked_result = execute_tool(
        "block_task",
        {
            "feature_id": "staged_feature",
            "task_id": 1,
            "reason": "Need API key",
            "requested_input": "Provide test API key",
        },
        ctx,
        session=session,
    )
    blocked_payload = json.loads(blocked_result)
    assert blocked_payload["ok"] is True
    assert blocked_payload["status"] == "blocked"

    resumed_result = execute_tool(
        "resume_task",
        {
            "feature_id": "staged_feature",
            "task_id": 1,
            "notes": "User provided the missing API key",
        },
        ctx,
        session=session,
    )
    resumed_payload = json.loads(resumed_result)
    assert resumed_payload["ok"] is True
    assert resumed_payload["status"] == "in_progress"

    completed_result = execute_tool(
        "update_task_status",
        {
            "task_id": 1,
            "status": "completed",
            "verified_exit_criteria": ["Tool payload is persisted"],
        },
        ctx,
        session=session,
    )
    completed_payload = json.loads(completed_result)
    assert completed_payload["ok"] is True

    review_all_result = execute_tool(
        "review_all_completed_tasks",
        {},
        ctx,
        session=session,
    )
    review_all_payload = json.loads(review_all_result)
    assert review_all_payload["ok"] is True
    assert review_all_payload["created_review_count"] >= 1

    review_result = execute_tool(
        "review_completed_tasks",
        {
            "task_id": 1,
            "summary": "Task delivered with one follow-up risk.",
            "limitations": ["No retry fallback"],
            "issues": [
                {"id": "risk-1", "title": "Retry fallback missing", "category": "risk"}
            ],
        },
        ctx,
        session=session,
    )
    review_payload = json.loads(review_result)
    assert review_payload["ok"] is True
    review_id = review_payload["review"]["id"]

    proposal_result = execute_tool(
        "propose_task_diff",
        {
            "review_id": review_id,
            "issue_id": "risk-1",
            "diff": "--- a/demo.py\n+++ b/demo.py\n@@\n+retry = True\n",
        },
        ctx,
        session=session,
    )
    proposal_payload = json.loads(proposal_result)
    assert proposal_payload["ok"] is True
    proposal_id = proposal_payload["proposal"]["id"]

    decision_result = execute_tool(
        "decide_task_diff",
        {
            "proposal_id": proposal_id,
            "decision": "approved",
            "reason": "Looks good.",
        },
        ctx,
        session=session,
    )
    decision_payload = json.loads(decision_result)
    assert decision_payload["ok"] is True
    assert decision_payload["proposal"]["status"] == "approved"

    archive_result = execute_tool(
        "archive_task",
        {"task_id": 1},
        ctx,
        session=session,
    )
    archive_payload = json.loads(archive_result)
    assert archive_payload["ok"] is True
    assert archive_payload["status"] == "archived"


def test_update_task_status_requires_verified_exit_criteria_for_completion(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))
    session = _SessionStub(str(tmp_path / "feature_plan.json"))
    tool_ctx = ToolExecutionContext(folder_context=ctx, session=session)
    _handle_create_feature_task(
        {
            "feature_name": "Verification",
            "feature_request": "Ensure completion checks",
            "tasks": [
                {
                    "title": "Task A",
                    "objectives": ["Goal A"],
                    "action_points": ["Action A"],
                    "exit_criteria": ["Criterion A"],
                }
            ],
        },
        tool_ctx,
    )

    # Approve the feature before updating task status
    _handle_approve_feature_task({"approved": True}, tool_ctx)

    result = execute_tool(
        "update_task_status",
        {"task_id": 1, "status": "completed"},
        ctx,
        session=session,
    )

    assert "Cannot mark task completed" in result


def test_update_task_status_persists_incremental_verified_exit_criteria(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))
    metadata_path = str(tmp_path / "feature_plan.json")
    session = _SessionStub(metadata_path)
    tool_ctx = ToolExecutionContext(folder_context=ctx, session=session)
    _handle_create_feature_task(
        {
            "feature_name": "Progress",
            "feature_request": "Track exit criteria progress",
            "tasks": [
                {
                    "title": "Task A",
                    "objectives": ["Goal A"],
                    "action_points": ["Action A"],
                    "exit_criteria": ["Criterion A", "Criterion B"],
                }
            ],
        },
        tool_ctx,
    )

    # Approve the feature before updating task status
    _handle_approve_feature_task({"approved": True}, tool_ctx)

    first = execute_tool(
        "update_task_status",
        {"task_id": 1, "status": "in_progress", "verified_exit_criteria": ["Criterion A"]},
        ctx,
        session=session,
    )
    first_payload = json.loads(first)
    assert first_payload["ok"] is True
    assert first_payload["plan"]["phases"][0]["verified_exit_criteria"] == ["Criterion A"]

    second = execute_tool(
        "update_task_status",
        {"task_id": 1, "status": "completed", "verified_exit_criteria": ["Criterion B"]},
        ctx,
        session=session,
    )
    second_payload = json.loads(second)
    assert second_payload["ok"] is True
    assert second_payload["status"] == "completed"
    assert sorted(second_payload["plan"]["phases"][0]["verified_exit_criteria"]) == [
        "Criterion A",
        "Criterion B",
    ]


def test_update_task_status_recovers_missing_feature_metadata_path(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))
    session = _SessionStub(str(tmp_path / "feature_plan.json"))
    tool_ctx = ToolExecutionContext(folder_context=ctx, session=session)
    _handle_create_feature_task(
        {
            "feature_name": "Recovery",
            "feature_request": "Recover metadata path",
            "tasks": [
                {
                    "title": "Task A",
                    "objectives": ["Goal A"],
                    "action_points": ["Action A"],
                    "exit_criteria": ["Criterion A"],
                }
            ],
        },
        tool_ctx,
    )
    session.session_manager.feature_state = {
        "feature_id": session.session_manager.record["feature_id"],
        "directory": session.session_manager.record["directory"],
        "metadata_path": "",
    }

    # Approve the feature before updating task status
    _handle_approve_feature_task({"approved": True}, tool_ctx)

    result = execute_tool(
        "update_task_status",
        {"task_id": 1, "status": "in_progress"},
        ctx,
        session=session,
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["status"] == "in_progress"


def test_apply_diff_requires_approved_proposal_in_review_mode(tmp_path):
    ctx = FolderContext()
    ctx.add_folder(str(tmp_path))
    session = _SessionStub(str(tmp_path / "feature_plan.json"))
    tool_ctx = ToolExecutionContext(folder_context=ctx, session=session)
    _handle_create_feature_task(
        {
            "feature_name": "Review Gate",
            "feature_request": "Check apply_diff guard",
            "tasks": [
                {
                    "title": "Task A",
                    "objectives": ["Goal A"],
                    "action_points": ["Action A"],
                    "exit_criteria": ["Criterion A"],
                }
            ],
        },
        tool_ctx,
    )

    # Approve the feature before updating task status
    _handle_approve_feature_task({"approved": True}, tool_ctx)

    execute_tool(
        "update_task_status",
        {
            "task_id": 1,
            "status": "completed",
            "verified_exit_criteria": ["Criterion A"],
        },
        ctx,
        session=session,
    )

    blocked = execute_tool(
        "apply_diff",
        {"filename": str(tmp_path / "demo.txt"), "diff": "@@ -1 +1 @@\n-a\n+b\n"},
        ctx,
        session=session,
    )
    assert "requires proposal_id" in blocked
