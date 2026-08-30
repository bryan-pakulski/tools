"""Tests for the dual-presentation layer: lecture transcript + exercise files.

Covers:
  * Engine: write_lecture_transcript / register_exercise_file write to disk,
    set lesson fields, persist via save_course, survive round-trip.
  * Path-traversal guards in the engine layer.
  * Handler @tool wrappers: registered, requires_approval=False, guard
    against pending lessons.
  * GUI router: lecture + exercises endpoints, _summarize_lessons carries
    the new fields, missing artifacts → 404 / empty list.
  * Backward compat: old course without the new fields loads fine.
  * Plan-mode blocks write_lecture_transcript / register_exercise_file.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

import mu.teacher.engine as engine
from mu.teacher.engine import (
    Course,
    LESSON_PENDING,
    LESSON_PRESENTING,
    Lesson,
    Module,
    create_course,
    find_lesson,
    load_course,
    register_exercise_file,
    save_course,
    write_lecture_transcript,
)
from mu.teacher import storage as _storage


# =========================================================== fixtures


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _seed_course(folder_context=None, subject="Perl") -> engine.Course:
    """Create a course with one module + one lesson in PRESENTING state."""
    course = create_course(subject=subject, folder_context=folder_context)
    course.modules.append(
        Module(module_id="m1", title="Basics", goal="Learn the basics", order=1,
               lesson_ids=["l1"], mastery_threshold=70)
    )
    course.lessons.append(
        Lesson(lesson_id="l1", module_id="m1", title="Hello World",
               learning_objectives=["print"], concept_brief="")
    )
    course.lessons[0].status = LESSON_PRESENTING
    save_course(course)
    return course


# =========================================================== engine: fields


def test_lesson_dataclass_has_new_fields_with_safe_defaults():
    lesson = Lesson(lesson_id="x", module_id="m", title="t")
    assert lesson.lecture_transcript_path is None
    assert lesson.exercise_file_paths == []


def test_lesson_from_dict_defaults_missing_fields():
    raw = {
        "lesson_id": "x",
        "module_id": "m",
        "title": "t",
        # no lecture_transcript_path or exercise_file_paths
    }
    lesson = engine._lesson_from_dict(raw)
    assert lesson.lecture_transcript_path is None
    assert lesson.exercise_file_paths == []


def test_lesson_from_dict_reads_new_fields():
    raw = {
        "lesson_id": "x",
        "module_id": "m",
        "title": "t",
        "lecture_transcript_path": "lessons/x/lecture.md",
        "exercise_file_paths": ["lessons/x/exercises/example_01.py"],
    }
    lesson = engine._lesson_from_dict(raw)
    assert lesson.lecture_transcript_path == "lessons/x/lecture.md"
    assert lesson.exercise_file_paths == ["lessons/x/exercises/example_01.py"]


# =========================================================== engine: write_lecture_transcript


def test_write_lecture_transcript_creates_file_and_sets_path(isolated_workspace):
    course = _seed_course()
    path = write_lecture_transcript(course, "l1", "# Hello World\n\nThis is the lecture.")
    assert os.path.isfile(path)
    assert path.endswith("lessons/l1/lecture.md")
    lesson = find_lesson(course, "l1")
    assert lesson.lecture_transcript_path is not None
    assert "lecture.md" in lesson.lecture_transcript_path


def test_write_lecture_transcript_content_matches(isolated_workspace):
    course = _seed_course()
    content = "# Lecture\n\n## Part 1\n\nExplanation here."
    write_lecture_transcript(course, "l1", content)
    lesson = find_lesson(course, "l1")
    path = os.path.join(course.directory, lesson.lecture_transcript_path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == content


def test_write_lecture_transcript_persists_via_save_course(isolated_workspace):
    course = _seed_course()
    write_lecture_transcript(course, "l1", "content")
    save_course(course)
    reloaded = load_course(course.course_id)
    lesson = find_lesson(reloaded, "l1")
    assert lesson.lecture_transcript_path is not None


def test_write_lecture_transcript_does_not_change_lesson_status(isolated_workspace):
    course = _seed_course()
    lesson = find_lesson(course, "l1")
    assert lesson.status == LESSON_PRESENTING
    write_lecture_transcript(course, "l1", "content")
    lesson = find_lesson(course, "l1")
    assert lesson.status == LESSON_PRESENTING  # unchanged


def test_write_lecture_transcript_unknown_lesson_raises(isolated_workspace):
    course = _seed_course()
    with pytest.raises(ValueError, match="not found"):
        write_lecture_transcript(course, "nonexistent", "content")


def test_write_lecture_transcript_empty_content_writes_empty_file(isolated_workspace):
    course = _seed_course()
    write_lecture_transcript(course, "l1", "")
    lesson = find_lesson(course, "l1")
    path = os.path.join(course.directory, lesson.lecture_transcript_path)
    assert os.path.isfile(path)
    with open(path) as f:
        assert f.read() == ""


# =========================================================== engine: register_exercise_file


def test_register_exercise_file_creates_file_and_appends_path(isolated_workspace):
    course = _seed_course()
    path = register_exercise_file(course, "l1", "example_01.py", "print('hello')")
    assert os.path.isfile(path)
    assert path.endswith("lessons/l1/exercises/example_01.py")
    lesson = find_lesson(course, "l1")
    assert len(lesson.exercise_file_paths) == 1
    assert "example_01.py" in lesson.exercise_file_paths[0]


def test_register_exercise_file_content_matches(isolated_workspace):
    course = _seed_course()
    register_exercise_file(course, "l1", "example.pl", "print 'hello';")
    lesson = find_lesson(course, "l1")
    path = os.path.join(course.directory, lesson.exercise_file_paths[0])
    with open(path) as f:
        assert f.read() == "print 'hello';"


def test_register_exercise_file_multiple_files(isolated_workspace):
    course = _seed_course()
    register_exercise_file(course, "l1", "example_01.py", "a = 1")
    register_exercise_file(course, "l1", "example_02.py", "b = 2")
    lesson = find_lesson(course, "l1")
    assert len(lesson.exercise_file_paths) == 2


def test_register_exercise_file_dedupes_same_path(isolated_workspace):
    course = _seed_course()
    register_exercise_file(course, "l1", "example.py", "v1")
    register_exercise_file(course, "l1", "example.py", "v2")
    lesson = find_lesson(course, "l1")
    assert len(lesson.exercise_file_paths) == 1  # not duplicated


def test_register_exercise_file_does_not_change_lesson_status(isolated_workspace):
    course = _seed_course()
    register_exercise_file(course, "l1", "example.py", "x")
    lesson = find_lesson(course, "l1")
    assert lesson.status == LESSON_PRESENTING


def test_register_exercise_file_unknown_lesson_raises(isolated_workspace):
    course = _seed_course()
    with pytest.raises(ValueError, match="not found"):
        register_exercise_file(course, "nonexistent", "example.py", "x")


def test_register_exercise_file_rejects_absolute_path(isolated_workspace):
    course = _seed_course()
    with pytest.raises(ValueError, match="absolute paths not allowed"):
        register_exercise_file(course, "l1", "/etc/passwd", "x")


def test_register_exercise_file_rejects_parent_traversal(isolated_workspace):
    course = _seed_course()
    with pytest.raises(ValueError, match=r"'\.\.' segments not allowed"):
        register_exercise_file(course, "l1", "../../../etc/passwd", "x")


def test_register_exercise_file_rejects_empty_path(isolated_workspace):
    course = _seed_course()
    with pytest.raises(ValueError, match="relative_path is required"):
        register_exercise_file(course, "l1", "", "x")


def test_register_exercise_file_persists_via_save_course(isolated_workspace):
    course = _seed_course()
    register_exercise_file(course, "l1", "example.py", "x = 1")
    save_course(course)
    reloaded = load_course(course.course_id)
    lesson = find_lesson(reloaded, "l1")
    assert len(lesson.exercise_file_paths) == 1


def test_register_exercise_file_creates_subdirectories(isolated_workspace):
    course = _seed_course()
    register_exercise_file(course, "l1", "subdir/nested.py", "x = 1")
    lesson = find_lesson(course, "l1")
    path = os.path.join(course.directory, lesson.exercise_file_paths[0])
    assert os.path.isfile(path)


# =========================================================== engine: backward compat


def test_old_course_without_new_fields_loads_safely(isolated_workspace):
    course = _seed_course()
    save_course(course)
    # Tamper with the JSON to remove the new fields (simulate old course).
    import json
    state_path = _storage.course_state_path(course.course_id, None)
    with open(state_path) as f:
        data = json.load(f)
    for lesson in data.get("lessons", []):
        lesson.pop("lecture_transcript_path", None)
        lesson.pop("exercise_file_paths", None)
    with open(state_path, "w") as f:
        json.dump(data, f)
    # Reload — must not crash.
    reloaded = load_course(course.course_id)
    lesson = find_lesson(reloaded, "l1")
    assert lesson.lecture_transcript_path is None
    assert lesson.exercise_file_paths == []


# =========================================================== engine: __all__ exports


def test_engine_exports_new_functions():
    assert "write_lecture_transcript" in engine.__all__
    assert "register_exercise_file" in engine.__all__


def test_teacher_package_reexports_new_functions():
    from mu.teacher import write_lecture_transcript, register_exercise_file
    assert write_lecture_transcript is engine.write_lecture_transcript
    assert register_exercise_file is engine.register_exercise_file


# =========================================================== handler: registration


def test_handlers_registered_in_tool_registry():
    from mu.tools.descriptors import TOOL_DESCRIPTORS, TOOLS
    from mu.tools._dispatcher import TOOL_HANDLERS
    for name in ("write_lecture_transcript", "register_exercise_file"):
        assert name in TOOL_DESCRIPTORS, f"{name} missing from TOOL_DESCRIPTORS"
        assert name in TOOL_HANDLERS, f"{name} missing from TOOL_HANDLERS"
        assert any(t.name == name for t in TOOLS), f"{name} missing from TOOLS"


def test_handlers_requires_approval_is_false():
    from mu.tools.descriptors import TOOL_DESCRIPTORS
    desc1 = TOOL_DESCRIPTORS["write_lecture_transcript"]
    desc2 = TOOL_DESCRIPTORS["register_exercise_file"]
    assert desc1.definition.requires_approval is False
    assert desc2.definition.requires_approval is False


def test_handlers_exported_in_all():
    from mu.tools.teacher import handlers
    assert "write_lecture_transcript_tool" in handlers.__all__
    assert "register_exercise_file_tool" in handlers.__all__


# =========================================================== handler: guard logic


def test_write_lecture_transcript_refuses_pending_lesson(isolated_workspace):
    """The handler must refuse if lesson status is pending."""
    from mu.tools.teacher.handlers import write_lecture_transcript_tool
    from types import SimpleNamespace

    course = _seed_course()
    lesson = find_lesson(course, "l1")
    lesson.status = LESSON_PENDING
    save_course(course)

    # Build a minimal fake session that _load_active_course + _persist need.
    fake_session = SimpleNamespace(
        session_manager=SimpleNamespace(
            active_course_id=course.course_id,
            teacher_state={},
            folder_context=None,
            get_teacher_state=lambda: {},
            get_course=lambda cid: {
                "course_path": _storage.course_state_path(course.course_id, None),
            },
            upsert_teacher_course=lambda rec: None,
            save_history_turn=lambda fc: None,
        ),
        folder_context=None,
        variables={},
        ui=None,
    )
    context = SimpleNamespace(
        session=fake_session,
        folder_context=None,
        ui=None,
        variables={},
        invocation_source="test",
    )
    import json as _json
    raw = write_lecture_transcript_tool(
        {"lesson_id": "l1", "content": "test"},
        context,
    )
    payload = _json.loads(raw)
    assert payload["ok"] is False
    err = payload.get("error") or payload.get("message") or ""
    assert "presenting" in err.lower() or "pending" in err.lower()


def test_register_exercise_file_refuses_pending_lesson(isolated_workspace):
    from mu.tools.teacher.handlers import register_exercise_file_tool
    from types import SimpleNamespace

    course = _seed_course()
    lesson = find_lesson(course, "l1")
    lesson.status = LESSON_PENDING
    save_course(course)

    fake_session = SimpleNamespace(
        session_manager=SimpleNamespace(
            active_course_id=course.course_id,
            teacher_state={},
            folder_context=None,
            get_teacher_state=lambda: {},
            get_course=lambda cid: {
                "course_path": _storage.course_state_path(course.course_id, None),
            },
            upsert_teacher_course=lambda rec: None,
            save_history_turn=lambda fc: None,
        ),
        folder_context=None,
        variables={},
        ui=None,
    )
    context = SimpleNamespace(
        session=fake_session,
        folder_context=None,
        ui=None,
        variables={},
        invocation_source="test",
    )
    import json as _json
    raw = register_exercise_file_tool(
        {"lesson_id": "l1", "path": "example.py", "content": "x"},
        context,
    )
    payload = _json.loads(raw)
    assert payload["ok"] is False


# =========================================================== handler: happy path via tool


def test_write_lecture_transcript_tool_writes_file(isolated_workspace):
    from mu.tools.teacher.handlers import write_lecture_transcript_tool
    from types import SimpleNamespace

    course = _seed_course()
    save_course(course)

    fake_session = SimpleNamespace(
        session_manager=SimpleNamespace(
            active_course_id=course.course_id,
            teacher_state={},
            folder_context=None,
            get_teacher_state=lambda: {},
            get_course=lambda cid: {
                "course_path": _storage.course_state_path(course.course_id, None),
            },
            upsert_teacher_course=lambda rec: None,
            save_history_turn=lambda fc: None,
        ),
        folder_context=None,
        variables={},
        ui=None,
    )
    context = SimpleNamespace(
        session=fake_session,
        folder_context=None,
        ui=None,
        variables={},
        invocation_source="test",
    )
    import json as _json
    raw = write_lecture_transcript_tool(
        {"lesson_id": "l1", "content": "# Lecture via tool"},
        context,
    )
    payload = _json.loads(raw)
    assert payload["ok"] is True
    assert payload["lesson_id"] == "l1"
    assert payload["lecture_transcript_path"] is not None
    # File should exist on disk.
    reloaded = load_course(course.course_id)
    lesson = find_lesson(reloaded, "l1")
    path = os.path.join(reloaded.directory, lesson.lecture_transcript_path)
    assert os.path.isfile(path)
    with open(path) as f:
        assert f.read() == "# Lecture via tool"


def test_register_exercise_file_tool_writes_file(isolated_workspace):
    from mu.tools.teacher.handlers import register_exercise_file_tool
    from types import SimpleNamespace

    course = _seed_course()
    save_course(course)

    fake_session = SimpleNamespace(
        session_manager=SimpleNamespace(
            active_course_id=course.course_id,
            teacher_state={},
            folder_context=None,
            get_teacher_state=lambda: {},
            get_course=lambda cid: {
                "course_path": _storage.course_state_path(course.course_id, None),
            },
            upsert_teacher_course=lambda rec: None,
            save_history_turn=lambda fc: None,
        ),
        folder_context=None,
        variables={},
        ui=None,
    )
    context = SimpleNamespace(
        session=fake_session,
        folder_context=None,
        ui=None,
        variables={},
        invocation_source="test",
    )
    import json as _json
    raw = register_exercise_file_tool(
        {"lesson_id": "l1", "path": "example.py", "content": "print('hello')"},
        context,
    )
    payload = _json.loads(raw)
    assert payload["ok"] is True
    assert payload["lesson_id"] == "l1"
    assert len(payload["exercise_file_paths"]) == 1
    # File on disk.
    reloaded = load_course(course.course_id)
    lesson = find_lesson(reloaded, "l1")
    path = os.path.join(reloaded.directory, lesson.exercise_file_paths[0])
    assert os.path.isfile(path)
    with open(path) as f:
        assert f.read() == "print('hello')"


# =========================================================== plan-mode


def test_plan_mode_blocks_write_lecture_transcript():
    from mu.agent.plan_mode import WRITE_TOOLS
    assert "write_lecture_transcript" in WRITE_TOOLS


def test_plan_mode_blocks_register_exercise_file():
    from mu.agent.plan_mode import WRITE_TOOLS
    assert "register_exercise_file" in WRITE_TOOLS


# =========================================================== GUI router: _summarize_lessons


def test_summarize_lessons_includes_new_fields():
    from mu.gui.routers.teacher import _summarize_lessons

    course_dict = {
        "lessons": [
            {
                "lesson_id": "l1",
                "module_id": "m1",
                "title": "Hello",
                "lecture_transcript_path": "lessons/l1/lecture.md",
                "exercise_file_paths": ["lessons/l1/exercises/example_01.py"],
            }
        ]
    }
    result = _summarize_lessons(course_dict)
    assert len(result) == 1
    assert result[0]["lecture_transcript_path"] == "lessons/l1/lecture.md"
    assert result[0]["exercise_file_paths"] == ["lessons/l1/exercises/example_01.py"]


def test_summarize_lessons_defaults_missing_fields():
    from mu.gui.routers.teacher import _summarize_lessons

    course_dict = {
        "lessons": [
            {
                "lesson_id": "l1",
                "module_id": "m1",
                "title": "Hello",
                # no lecture_transcript_path or exercise_file_paths
            }
        ]
    }
    result = _summarize_lessons(course_dict)
    assert result[0]["lecture_transcript_path"] is None
    assert result[0]["exercise_file_paths"] == []


# =========================================================== GUI router: endpoints


def _fake_request(course_dir):
    """Build a mock Request whose app.state.session_by_name() resolves
    to a session with teacher_state carrying the given course directory."""
    from types import SimpleNamespace

    session = SimpleNamespace(
        session_manager=SimpleNamespace(
            teacher_state={"directory": course_dir},
            active_course_id="test-course",
            teacher_registry={},
        )
    )
    app_state = SimpleNamespace(session_by_name=lambda: session)
    app = SimpleNamespace(state=app_state)
    return SimpleNamespace(app=app)


def test_get_lecture_transcript_returns_markdown(isolated_workspace):
    import asyncio
    from mu.gui.routers.teacher import get_lecture_transcript

    course = _seed_course()
    write_lecture_transcript(course, "l1", "# Hello World\n\nLecture content.")
    save_course(course)

    request = _fake_request(course.directory)
    result = asyncio.run(
        get_lecture_transcript(request, "l1")
    )
    assert result["lesson_id"] == "l1"
    assert result["content"] == "# Hello World\n\nLecture content."


def test_get_lecture_transcript_404_when_absent(isolated_workspace):
    import asyncio
    from fastapi import HTTPException
    from mu.gui.routers.teacher import get_lecture_transcript

    course = _seed_course()
    request = _fake_request(course.directory)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            get_lecture_transcript(request, "l1")
        )
    assert exc_info.value.status_code == 404


def test_get_lecture_transcript_404_no_active_course(isolated_workspace):
    import asyncio
    from fastapi import HTTPException
    from types import SimpleNamespace
    from mu.gui.routers.teacher import get_lecture_transcript

    # Session with no teacher_state → no course dir
    session = SimpleNamespace(
        session_manager=SimpleNamespace(
            teacher_state=None,
            active_course_id=None,
            teacher_registry={},
        )
    )
    app_state = SimpleNamespace(session_by_name=lambda: session)
    request = SimpleNamespace(app=SimpleNamespace(state=app_state))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            get_lecture_transcript(request, "l1")
        )
    assert exc_info.value.status_code == 404


def test_get_exercises_listing_returns_files(isolated_workspace):
    import asyncio
    from mu.gui.routers.teacher import get_exercises_listing

    course = _seed_course()
    register_exercise_file(course, "l1", "example_01.py", "print('hello')")
    register_exercise_file(course, "l1", "example_02.py", "x = 42")
    save_course(course)

    request = _fake_request(course.directory)
    result = asyncio.run(
        get_exercises_listing(request, "l1")
    )
    assert result["lesson_id"] == "l1"
    assert len(result["files"]) == 2
    paths = [f["path"] for f in result["files"]]
    assert "example_01.py" in paths
    assert "example_02.py" in paths
    contents = {f["path"]: f["content"] for f in result["files"]}
    assert contents["example_01.py"] == "print('hello')"
    assert contents["example_02.py"] == "x = 42"


def test_get_exercises_listing_empty_when_no_dir(isolated_workspace):
    import asyncio
    from mu.gui.routers.teacher import get_exercises_listing

    course = _seed_course()
    request = _fake_request(course.directory)
    result = asyncio.run(
        get_exercises_listing(request, "l1")
    )
    assert result["lesson_id"] == "l1"
    assert result["files"] == []


def test_get_exercise_file_returns_content(isolated_workspace):
    import asyncio
    from mu.gui.routers.teacher import get_exercise_file

    course = _seed_course()
    register_exercise_file(course, "l1", "example.py", "print('hello')")
    save_course(course)

    request = _fake_request(course.directory)
    result = asyncio.run(
        get_exercise_file(request, "l1", "example.py")
    )
    assert result["lesson_id"] == "l1"
    assert result["path"] == "example.py"
    assert result["content"] == "print('hello')"


def test_get_exercise_file_404_not_found(isolated_workspace):
    import asyncio
    from fastapi import HTTPException
    from mu.gui.routers.teacher import get_exercise_file

    course = _seed_course()
    request = _fake_request(course.directory)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            get_exercise_file(request, "l1", "nonexistent.py")
        )
    assert exc_info.value.status_code == 404


def test_get_exercise_file_rejects_traversal(isolated_workspace):
    import asyncio
    from fastapi import HTTPException
    from mu.gui.routers.teacher import get_exercise_file

    course = _seed_course()
    register_exercise_file(course, "l1", "example.py", "x = 1")
    save_course(course)

    request = _fake_request(course.directory)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            get_exercise_file(request, "l1", "../../../etc/passwd")
        )
    assert exc_info.value.status_code == 403


def test_get_exercise_file_rejects_absolute_path(isolated_workspace):
    import asyncio
    from fastapi import HTTPException
    from mu.gui.routers.teacher import get_exercise_file

    course = _seed_course()
    request = _fake_request(course.directory)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            get_exercise_file(request, "l1", "/etc/passwd")
        )
    assert exc_info.value.status_code == 403


def test_get_exercise_file_handles_subdirectory(isolated_workspace):
    import asyncio
    from mu.gui.routers.teacher import get_exercise_file

    course = _seed_course()
    register_exercise_file(course, "l1", "subdir/nested.py", "y = 2")
    save_course(course)

    request = _fake_request(course.directory)
    result = asyncio.run(
        get_exercise_file(request, "l1", "subdir/nested.py")
    )
    assert result["path"] == os.path.join("subdir", "nested.py")
    assert result["content"] == "y = 2"


# =========================================================== lecture references exercises


def test_lecture_transcript_contains_relative_path_references(isolated_workspace):
    """lecture.md must embed relative-path references to exercise files."""
    course = _seed_course()
    register_exercise_file(course, "l1", "example_01.py", "print('hello')")
    register_exercise_file(course, "l1", "example_02.py", "x = 42")
    lecture_md = (
        "# Hello World\n\n"
        "See exercises/example_01.py for a basic example.\n\n"
        "Also see exercises/example_02.py for variables.\n"
    )
    write_lecture_transcript(course, "l1", lecture_md)
    save_course(course)

    # Verify lecture.md on disk contains relative path references
    lesson = find_lesson(course, "l1")
    lecture_path = os.path.join(course.directory, lesson.lecture_transcript_path)
    with open(lecture_path) as f:
        content = f.read()
    assert "exercises/example_01.py" in content
    assert "exercises/example_02.py" in content

    # Verify exercise files exist on disk at the referenced paths
    exercises_dir = os.path.join(course.directory, "lessons", "l1", "exercises")
    assert os.path.isfile(os.path.join(exercises_dir, "example_01.py"))
    assert os.path.isfile(os.path.join(exercises_dir, "example_02.py"))


# =========================================================== existing assignment flow unchanged


def test_assign_exercise_tool_works_after_dual_presentation(isolated_workspace):
    """Existing assignment flow must work on a lesson that has dual-presentation artifacts."""
    from mu.tools.teacher.handlers import assign_exercise_tool

    course = _seed_course()
    register_exercise_file(course, "l1", "example.py", "print('hello')")
    write_lecture_transcript(course, "l1", "# Lecture\n\nSee exercises/example.py")
    save_course(course)

    fake_session = SimpleNamespace(
        session_manager=SimpleNamespace(
            active_course_id=course.course_id,
            teacher_state={},
            folder_context=None,
            get_teacher_state=lambda: {},
            get_course=lambda cid: {
                "course_path": _storage.course_state_path(course.course_id, None),
            },
            upsert_teacher_course=lambda rec: None,
            save_history_turn=lambda fc: None,
        ),
        folder_context=None,
        variables={},
        ui=None,
    )
    context = SimpleNamespace(
        session=fake_session,
        folder_context=None,
        ui=None,
        variables={},
        invocation_source="test",
    )
    import json as _json
    raw = assign_exercise_tool(
        {"lesson_id": "l1", "kind": "short-answer", "prompt": "What does print('hello') output?"},
        context,
    )
    payload = _json.loads(raw)
    assert payload["ok"] is True
    assert payload["lesson_id"] == "l1"
    # Dual presentation fields still intact
    reloaded = load_course(course.course_id)
    lesson = find_lesson(reloaded, "l1")
    assert lesson.lecture_transcript_path is not None
    assert len(lesson.exercise_file_paths) == 1