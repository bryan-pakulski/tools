"""Teacher-mode slash command: `/teach <subcommand>`.

Subcommands:
    /teach                            — list courses (alias of /teach list)
    /teach list                       — list courses for this workspace
    /teach new <subject>              — create a new course (does not switch mode)
    /teach load <id>                  — activate an existing course
    /teach exit | unload              — clear active course without deleting
    /teach status                     — print current board: module/lesson, %, scores
    /teach next                       — show the next pending lesson and prompt the agent
    /teach grades                     — markdown table of every graded assignment
    /teach curriculum                 — render the course's curriculum.md
    /teach delete <id>                — delete a course
    /teach help                       — workflow help text
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from mu.teacher import (
    Course,
    course_metrics,
    create_course,
    find_lesson,
    load_course,
    next_pending_lesson,
)
from mu.teacher import storage as _storage

from . import CommandResult, command


_HELP_TEXT = """\
/teach — manage teacher-mode courses (independent of `/mode teacher`).

Most workflows:
  /teach new <subject>      — open a new course (run /mode teacher to drive it)
  /teach list               — list this workspace's courses
  /teach load <id>          — activate an existing course (read-only or under /mode teacher)
  /teach status             — current module/lesson, completion %, average score
  /teach next               — show the next pending lesson
  /teach grades             — every graded assignment, newest first
  /teach curriculum         — render the syllabus
  /teach exit               — clear active course (course stays on disk)
  /teach delete <id>        — permanently delete a course directory
"""


def _folder_context(session: Any):
    return getattr(session, "folder_context", None)


def _print(session: Any, message: str) -> None:
    """Echo a command's user-facing text to the terminal.

    The REPL does not auto-print `CommandResult.message` (it's the
    machine-readable payload). Each slash-command handler is responsible
    for surfacing what the user sees.
    """
    if not message:
        return
    ui = getattr(session, "ui", None)
    if ui is None:
        return
    console = getattr(ui, "console", None)
    if console is not None:
        try:
            console.print(message, markup=False, highlight=False)
            return
        except Exception:
            pass
    if hasattr(ui, "show_info"):
        try:
            ui.show_info(message)
        except Exception:
            pass


def _active_course(session: Any) -> Course | None:
    record = session.session_manager.get_teacher_state() or {}
    course_id = (record or {}).get("course_id") or session.session_manager.active_course_id
    if not course_id:
        return None
    path = (record or {}).get("course_path") or _storage.course_state_path(
        course_id, _folder_context(session)
    )
    if os.path.exists(path):
        return load_course(path, _folder_context(session))
    return load_course(course_id, _folder_context(session))


def _sync_record(session: Any, course: Course) -> dict:
    fc = _folder_context(session)
    metrics = course_metrics(course)
    record = {
        "type": "course",
        "course_id": course.course_id,
        "subject": course.subject,
        "directory": course.directory,
        "course_path": _storage.course_state_path(course.course_id, fc),
        "status": course.status,
        "metrics": metrics,
        "updated_at": course.updated_at,
        "created_at": course.created_at,
    }
    session.session_manager.upsert_teacher_course(record)
    return record


# ---------------------------------------------------------- subcommands


def _list(session: Any) -> CommandResult:
    fc = _folder_context(session)
    courses = []
    for course_id in _storage.list_courses(fc):
        course = load_course(course_id, fc)
        if course is None:
            continue
        metrics = course_metrics(course)
        courses.append(
            {
                "course_id": course.course_id,
                "subject": course.subject,
                "status": course.status,
                "overall_pct": metrics["overall_pct"],
                "average_score_pct": metrics["average_score_pct"],
            }
        )
    if not courses:
        return CommandResult(
            ok=True,
            message="No courses yet. Use `/teach new <subject>` to create one.",
            data={"courses": []},
        )
    lines = ["Courses (this workspace):"]
    for c in courses:
        lines.append(
            f"  - {c['course_id']:24}  {c['subject']:24}  "
            f"{c['status']:22}  progress {c['overall_pct']}%  "
            f"avg {c['average_score_pct']}%"
        )
    return CommandResult(
        ok=True,
        message="\n".join(lines),
        data={"courses": courses},
    )


def _new(session: Any, rest: str) -> CommandResult:
    subject = rest.strip()
    if not subject:
        return CommandResult(
            ok=False,
            message="Usage: /teach new <subject> (e.g. /teach new Perl)",
        )
    fc = _folder_context(session)
    course = create_course(subject=subject, folder_context=fc)
    record = _sync_record(session, course)
    session.session_manager.active_course_id = course.course_id
    session.session_manager.teacher_state = dict(record)
    session.session_manager.save_history_turn(fc)
    msg = (
        f"Created course `{course.course_id}` for subject {course.subject!r}.\n"
        f"Directory: {course.directory}\n"
        "Next: run `/mode teacher` to enter teacher mode, then ask the "
        "agent to begin the diagnostic."
    )
    return CommandResult(ok=True, message=msg, data={"course": record})


def _load(session: Any, rest: str) -> CommandResult:
    course_id = rest.strip()
    if not course_id:
        return CommandResult(ok=False, message="Usage: /teach load <course_id>")
    fc = _folder_context(session)
    course = load_course(course_id, fc)
    if course is None:
        return CommandResult(
            ok=False,
            message=f"No course with id `{course_id}` in this workspace.",
        )
    record = _sync_record(session, course)
    session.session_manager.active_course_id = course.course_id
    session.session_manager.teacher_state = dict(record)
    session.session_manager.save_history_turn(fc)
    _queue_course_resumption_briefing(session, course)
    return CommandResult(
        ok=True,
        message=f"Loaded course `{course.course_id}` ({course.subject}).",
        data={"course": record},
    )


def _queue_course_resumption_briefing(session: Any, course: Course) -> None:
    """Tell the next-turn agent that it just resumed an in-flight course.

    Includes: subject, status, current lesson, latest grade, and the
    next pending action. The model uses this to skip re-asking the
    user where they were and pick up the lesson loop directly.
    """
    if not hasattr(session, "queue_resumption_briefing"):
        return
    metrics = course_metrics(course)
    nxt = next_pending_lesson(course)
    lines = [
        f"You just resumed teacher-mode course **{course.course_id}** "
        f"(subject: {course.subject!r}, target_level: {course.target_level}).",
        f"Course status: {course.status}.",
        f"Progress: {metrics['lessons_completed']}/{metrics['total_lessons']} "
        f"lessons completed ({metrics['overall_pct']}%); "
        f"average grade {metrics['average_score_pct']}%.",
    ]
    if course.current_lesson_id:
        current = find_lesson(course, course.current_lesson_id)
        if current is not None:
            lines.append(
                f"Current lesson: `{current.lesson_id}` — {current.title!r} "
                f"(status: {current.status})."
            )
    if nxt is not None and (
        course.current_lesson_id is None or nxt.lesson_id != course.current_lesson_id
    ):
        lines.append(
            f"Next pending lesson: `{nxt.lesson_id}` — {nxt.title!r} "
            f"(status: {nxt.status})."
        )
    # Surface the most recent graded assignment so the agent has
    # something concrete to reference.
    latest_grade = None
    for a in reversed(course.assignments):
        if a.grade is not None:
            latest_grade = a
            break
    if latest_grade is not None and latest_grade.grade is not None:
        verdict = "passed" if latest_grade.grade.passed else "failed"
        lines.append(
            f"Most recent grade: assignment `{latest_grade.assignment_id}` "
            f"scored {latest_grade.grade.score_pct}% ({verdict})."
        )
    # Action hint based on the current state.
    if course.status == "diagnosing":
        lines.append("ACTION: finish the diagnostic and call propose_curriculum.")
    elif course.status == "curriculum_proposed":
        lines.append(
            "ACTION: the learner needs to call approve_curriculum before the "
            "lesson loop can begin."
        )
    elif course.status == "completed":
        lines.append("ACTION: course is complete. Surface the report card.")
    elif nxt is not None:
        lines.append(
            f"ACTION: start the next pending lesson with start_lesson("
            f"lesson_id='{nxt.lesson_id}'). For non-trivial topics, run the "
            f"lecture phase before assigning."
        )
    session.queue_resumption_briefing("\n".join(lines))


def _exit(session: Any) -> CommandResult:
    if session.session_manager.active_course_id is None:
        return CommandResult(ok=True, message="No active course to exit.")
    session.session_manager.clear_teacher_state(_folder_context(session))
    return CommandResult(ok=True, message="Cleared active course (still on disk).")


def _status(session: Any) -> CommandResult:
    course = _active_course(session)
    if course is None:
        return CommandResult(
            ok=True,
            message="No active course. /teach load <id> or /teach new <subject>.",
        )
    metrics = course_metrics(course)
    nxt = next_pending_lesson(course)
    lines = [
        f"Course: {course.course_id} — {course.subject!r}",
        f"Status: {course.status}",
        f"Modules: {metrics['total_modules']}, Lessons: "
        f"{metrics['lessons_completed']}/{metrics['total_lessons']} "
        f"({metrics['overall_pct']}%)",
        f"Average grade: {metrics['average_score_pct']}%",
    ]
    if course.current_lesson_id:
        lines.append(f"Current lesson: {course.current_lesson_id}")
    if nxt is not None and nxt.lesson_id != course.current_lesson_id:
        lines.append(f"Next pending lesson: {nxt.lesson_id} — {nxt.title}")
    return CommandResult(
        ok=True,
        message="\n".join(lines),
        data={"course_id": course.course_id, "metrics": metrics},
    )


def _next(session: Any) -> CommandResult:
    course = _active_course(session)
    if course is None:
        return CommandResult(ok=True, message="No active course.")
    nxt = next_pending_lesson(course)
    if nxt is None:
        return CommandResult(
            ok=True,
            message="Every lesson is complete. Run finalize_course to wrap up.",
            data={"course_id": course.course_id, "next_lesson_id": None},
        )
    return CommandResult(
        ok=True,
        message=(
            f"Next pending lesson: `{nxt.lesson_id}` — {nxt.title}\n"
            f"(module: {nxt.module_id}, status: {nxt.status})\n"
            f"Switch to teacher mode and ask the agent to start it: "
            f"`start_lesson(lesson_id='{nxt.lesson_id}')`"
        ),
        data={
            "course_id": course.course_id,
            "next_lesson_id": nxt.lesson_id,
            "module_id": nxt.module_id,
            "title": nxt.title,
        },
    )


def _grades(session: Any) -> CommandResult:
    course = _active_course(session)
    if course is None:
        return CommandResult(ok=True, message="No active course.")
    rows = []
    for assignment in course.assignments:
        if assignment.grade is None:
            continue
        lesson = find_lesson(course, assignment.lesson_id)
        rows.append(
            {
                "assignment_id": assignment.assignment_id,
                "lesson_id": assignment.lesson_id,
                "lesson_title": lesson.title if lesson else "",
                "kind": assignment.kind,
                "score_pct": assignment.grade.score_pct,
                "passed": assignment.grade.passed,
                "graded_at": assignment.grade.graded_at,
            }
        )
    rows.sort(key=lambda r: r["graded_at"], reverse=True)
    if not rows:
        return CommandResult(ok=True, message="No graded assignments yet.")
    lines = ["| Lesson | Assignment | Kind | Score | Passed |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['lesson_id']} | {r['assignment_id']} | {r['kind']} | "
            f"{r['score_pct']}% | {'✓' if r['passed'] else '✗'} |"
        )
    return CommandResult(
        ok=True,
        message="\n".join(lines),
        data={"grades": rows},
    )


def _curriculum(session: Any) -> CommandResult:
    course = _active_course(session)
    if course is None:
        return CommandResult(ok=True, message="No active course.")
    path = os.path.join(course.directory, "curriculum.md")
    if not os.path.exists(path):
        return CommandResult(
            ok=True,
            message="No curriculum.md yet — propose_curriculum hasn't run.",
        )
    with open(path, encoding="utf-8") as handle:
        body = handle.read()
    return CommandResult(ok=True, message=body, data={"path": path})


def _delete(session: Any, rest: str) -> CommandResult:
    course_id = rest.strip()
    if not course_id:
        return CommandResult(ok=False, message="Usage: /teach delete <course_id>")
    fc = _folder_context(session)
    directory = _storage.course_directory(course_id, fc)
    if not os.path.isdir(directory):
        return CommandResult(
            ok=False, message=f"No course directory at {directory}"
        )
    import shutil

    shutil.rmtree(directory, ignore_errors=True)
    session.session_manager.delete_course(course_id)
    return CommandResult(
        ok=True,
        message=f"Deleted course `{course_id}` ({directory}).",
    )


def _help(session: Any) -> CommandResult:
    return CommandResult(ok=True, message=_HELP_TEXT)


@command(
    "/teach",
    "/t",
    help=(
        "Teacher-mode courses: list/new/load/exit, status/next/grades, "
        "curriculum, delete. Run /mode teacher to drive a loaded course."
    ),
)
def teach_cmd(session: Any, args: str, *, allow_prompt: bool = True) -> CommandResult:
    raw = (args or "").strip()
    if not raw:
        result = _list(session)
    else:
        head, _, rest = raw.partition(" ")
        sub = head.lower()
        rest = rest.strip()

        if sub == "list":
            result = _list(session)
        elif sub == "new":
            result = _new(session, rest)
        elif sub == "load":
            result = _load(session, rest)
        elif sub in ("exit", "unload"):
            result = _exit(session)
        elif sub == "status":
            result = _status(session)
        elif sub == "next":
            result = _next(session)
        elif sub == "grades":
            result = _grades(session)
        elif sub == "curriculum":
            result = _curriculum(session)
        elif sub == "delete":
            result = _delete(session, rest)
        elif sub == "help":
            result = _help(session)
        else:
            result = CommandResult(
                ok=False,
                message=f"Unknown teach subcommand: {sub}. Use `/teach help` for guidance.",
            )

    if allow_prompt:
        _print(session, result.message)
    return result
