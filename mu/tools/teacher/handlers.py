"""Teacher-mode `@tool` handlers.

The engine (`mu.teacher.engine`) is UI-free; these handlers perform the
session sync + side effects (writing artifact files, launching the
optional live quiz UI, persisting graded artifacts to disk).

Each handler returns a JSON string envelope so the agent gets a
structured result. State mutations save via
`session.session_manager.upsert_teacher_course()` then
`session.session_manager.save_history_turn()` so courses survive across
session restarts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any

from mu.teacher import storage as _storage
from mu.teacher.engine import (
    ASSIGNMENT_PRESENTED,
    ASSIGNMENT_SUBMITTED,
    COURSE_COMPLETED,
    COURSE_CURRICULUM_PROPOSED,
    COURSE_DIAGNOSING,
    COURSE_IN_PROGRESS,
    LESSON_ASSIGNED,
    LESSON_COMPLETED,
    LESSON_GRADED,
    LESSON_LECTURING,
    LESSON_PENDING,
    LESSON_PRESENTING,
    Assignment,
    Course,
    Lesson,
    Module,
    RubricItem,
    VerificationSpec,
    add_event,
    advance_lesson_status,
    close_socratic_dialog,
    complete_review,
    conclude_lecture,
    course_metrics,
    create_course,
    decide_next,
    due_reviews,
    find_assignment,
    find_lesson,
    find_module,
    find_review,
    load_course,
    next_pending_lesson,
    record_dialog_turn,
    record_lecture_turn,
    save_course,
    schedule_review,
    start_lecture,
    write_lecture_transcript,
    register_exercise_file,
)
from mu.teacher.grading import grade as grade_assignment_payload
from mu.tools import tool


# ----------------------------------------------------------- helpers


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps({"ok": True, **payload}, indent=2, sort_keys=True, default=str)


def _err(message: str, **extra: Any) -> str:
    return json.dumps(
        {"ok": False, "error": message, **extra}, indent=2, sort_keys=True, default=str
    )


def _not_found_err(kind: str, requested: str, known: list[str]) -> str:
    """Friendly not-found error that lists the IDs that DO exist so the
    agent can self-correct instead of concluding state is lost."""
    sample = ", ".join(f"`{k}`" for k in known[:10]) if known else "(none)"
    hint = (
        f"{kind} {requested!r} not found. "
        f"Known {kind}s on this course: {sample}. "
        "If you just created one, use the id returned by the create call "
        "verbatim — do not re-slugify it."
    )
    return _err(hint, known_ids=known)


def _session_from_context(context: Any):
    session = getattr(context, "session", None)
    if session is None:
        raise RuntimeError("teacher tool requires an active session in the context")
    return session


def _folder_context(context: Any):
    fc = getattr(context, "folder_context", None)
    if fc is not None:
        return fc
    session = getattr(context, "session", None)
    return getattr(session, "folder_context", None) if session else None


def _persist(session, course: Course) -> dict[str, Any]:
    """Save course state to disk AND mirror summary into session registry."""
    save_course(course)
    fc = getattr(session, "folder_context", None)
    record = _summary_record(course, fc)
    session.session_manager.upsert_teacher_course(record)
    if session.session_manager.active_course_id == course.course_id:
        session.session_manager.teacher_state = dict(record)
    session.session_manager.save_history_turn(fc)
    return record


def _summary_record(course: Course, folder_context: Any) -> dict[str, Any]:
    metrics = course_metrics(course)
    state_path = _storage.course_state_path(course.course_id, folder_context)
    return {
        "type": "course",
        "course_id": course.course_id,
        "subject": course.subject,
        "directory": course.directory,
        "course_path": state_path,
        "status": course.status,
        "metrics": metrics,
        "updated_at": course.updated_at,
        "created_at": course.created_at,
    }


def _load_active_course(session, context: Any, course_id: str | None = None) -> Course:
    """Find and load the active course, trying every persistence layer.

    Resolution order:
      1. Caller-supplied `course_id` arg (slugified).
      2. SessionManager.active_course_id.
      3. SessionManager.teacher_state['course_id'] (set by /teach load).

    Once a course_id is in hand we try paths in this order so old layouts
    keep working: session_manager.get_course(...)['course_path'],
    storage.course_state_path(workspace), then a direct on-disk search.
    """
    fc = _folder_context(context)
    target_id = course_id
    if target_id:
        target_id = _storage.slugify(str(target_id))
    if not target_id:
        target_id = session.session_manager.active_course_id
    if not target_id:
        state = session.session_manager.get_teacher_state() or {}
        target_id = state.get("course_id")
    if not target_id:
        raise ValueError(
            "no active course — call create_course (for a new course) or "
            "tell the user to run '/teach load <course_id>' to resume one."
        )

    # Candidate paths to try, in order.
    candidates: list[str] = []
    record = session.session_manager.get_course(target_id) or {}
    persisted = str(record.get("course_path") or "").strip()
    if persisted:
        candidates.append(persisted)
    candidates.append(_storage.course_state_path(target_id, fc))

    for path in candidates:
        if path and os.path.exists(path):
            course = load_course(path, fc)
            if course is not None:
                return course

    # Final fallback: load_course with just the id will also recompute
    # the path. If it returns None, we genuinely can't find it.
    course = load_course(target_id, fc)
    if course is not None:
        return course

    # If a course_id was supplied but didn't resolve, surface the active
    # course (if any) so the agent can recover without thinking state was
    # lost.
    active = session.session_manager.active_course_id
    if course_id and active and active != target_id:
        raise ValueError(
            f"course {target_id!r} not found, but active course is "
            f"{active!r}. Drop the course_id arg or pass course_id={active!r} "
            "instead of guessing from the subject."
        )

    tried = "\n  ".join(candidates) if candidates else "(none)"
    raise ValueError(
        f"could not load course {target_id!r}. Tried:\n  {tried}\n"
        "If you have an existing course directory, move it under "
        "<workspace>/courses/<course_id>/ or run /teach load with the "
        "correct id."
    )


def _activate(session, course: Course, folder_context: Any) -> None:
    record = _summary_record(course, folder_context)
    session.session_manager.upsert_teacher_course(record)
    session.session_manager.active_course_id = course.course_id
    session.session_manager.teacher_state = dict(record)
    session.session_manager.save_history_turn(folder_context)


def _write_assignment_artifacts(
    course: Course,
    assignment: Assignment,
    artifact_files: list[dict[str, str]],
) -> list[str]:
    """Write engine-supplied artifact files into the assignment's work/ dir.

    Each entry is `{"path": "lesson_03.pl", "content": "..."}`. Absolute
    paths are rejected — artifacts live inside `work/`.
    """
    written: list[str] = []
    if not artifact_files:
        return written
    work_dir = os.path.join(
        course.directory,
        "assignments",
        _storage.slugify(assignment.assignment_id),
        "work",
    )
    os.makedirs(work_dir, exist_ok=True)
    for entry in artifact_files:
        rel = str(entry.get("path", "") or "").strip()
        if not rel or rel.startswith("/") or ".." in rel.split(os.sep):
            continue
        target = os.path.join(work_dir, rel)
        os.makedirs(os.path.dirname(target) or work_dir, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(str(entry.get("content", "") or ""))
        written.append(target)
    return written


def _write_curriculum_md(course: Course) -> None:
    if not course.directory:
        return
    path = os.path.join(course.directory, "curriculum.md")
    lines = [f"# {course.subject} — Curriculum", ""]
    lines.append(f"- **Target level**: {course.target_level}")
    lines.append(f"- **Status**: {course.status}")
    lines.append("")
    ordered = sorted(course.modules, key=lambda m: m.order)
    for module in ordered:
        lines.append(f"## Module {module.order}: {module.title}")
        if module.goal:
            lines.append(f"_Goal_: {module.goal}")
        lines.append("")
        for lesson_id in module.lesson_ids:
            lesson = find_lesson(course, lesson_id)
            if lesson is None:
                continue
            lines.append(f"- **{lesson.title}** ({lesson.status})")
            for objective in lesson.learning_objectives:
                lines.append(f"  - {objective}")
            for source in lesson.sources:
                title = source.get("title") or source.get("url") or "Source"
                kind = source.get("kind") or "reference"
                lines.append(f"  - Source: [{title}]({source.get('url')}) ({kind})")
                if source.get("note"):
                    lines.append(f"    - {source['note']}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_report_card(course: Course) -> str:
    path = os.path.join(course.directory, "report_card.md")
    metrics = course_metrics(course)
    lines = [
        f"# Report Card — {course.subject}",
        "",
        f"- **Course ID**: `{course.course_id}`",
        f"- **Lessons completed**: {metrics['lessons_completed']} / {metrics['total_lessons']}",
        f"- **Overall progress**: {metrics['overall_pct']}%",
        f"- **Average graded score**: {metrics['average_score_pct']}%",
        "",
        "## Per-lesson detail",
        "",
    ]
    ordered = sorted(course.modules, key=lambda m: m.order)
    for module in ordered:
        lines.append(f"### {module.title}")
        for lesson_id in module.lesson_ids:
            lesson = find_lesson(course, lesson_id)
            if lesson is None:
                continue
            assignments = [
                a for a in course.assignments if a.lesson_id == lesson.lesson_id
            ]
            best = None
            for a in assignments:
                if a.grade is None:
                    continue
                if best is None or a.grade.score_pct > best.grade.score_pct:
                    best = a
            score = best.grade.score_pct if best and best.grade else 0
            lines.append(
                f"- {lesson.title}: status `{lesson.status}`, best score {score}%"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


# ============================================================ tool handlers
# ----- course lifecycle ------------------------------------------------


@tool(
    name="create_course",
    description=(
        "Open a new teacher-mode course. Returns the course record. "
        "The course starts in 'diagnosing' state — call record_diagnostic "
        "after probing the learner."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Subject to teach, e.g. 'Perl', 'Kubernetes'."},
            "target_level": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced"],
                "default": "beginner",
            },
            "learner_summary": {
                "type": "string",
                "description": "One-line summary of what the learner said when starting.",
            },
            "course_id": {"type": "string", "description": "Optional explicit slug."},
        },
        "required": ["subject"],
    },
    requires_approval=False,
    phase="teacher",
)
def create_course_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        fc = _folder_context(context)
        course = create_course(
            subject=str(args.get("subject", "")).strip(),
            target_level=str(args.get("target_level", "beginner") or "beginner").strip(),
            learner_summary=str(args.get("learner_summary", "") or "").strip(),
            folder_context=fc,
            course_id=(str(args.get("course_id", "")).strip() or None),
        )
        _activate(session, course, fc)
        return _ok({"course": _summary_record(course, fc)})
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="record_diagnostic",
    description=(
        "Save the learner's diagnostic profile after the upfront Q&A. "
        "Required before propose_curriculum. Captures not just prior "
        "experience/gaps/goals but also HOW the learner learns: preferred "
        "modalities, pace, jargon tolerance, motivation, analogy anchors "
        "from adjacent fields, and personality cues. Every field is "
        "optional — fill what you've actually heard. Lessons reference "
        "this profile on every turn, so be specific."
    ),
    parameters={
        "type": "object",
        "properties": {
            "course_id": {"type": "string"},
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Things the learner already knows well in or near this subject.",
            },
            "gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific things they don't know yet that the course must cover.",
            },
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What they want to be able to do after the course (be concrete).",
            },
            "modality": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Preferred learning modalities — pick from: analogy, hands-on, visual, socratic, worked-example, formal-definition, story. Multi-select fine.",
            },
            "pace": {
                "type": "string",
                "enum": ["slow", "moderate", "fast"],
                "description": "How fast they want to move through material.",
            },
            "jargon_tolerance": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "How comfortable they are with field-specific terminology and notation up front.",
            },
            "motivation": {
                "type": "string",
                "description": "Why they're learning this — career, curiosity, specific project, exam, etc.",
            },
            "background": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Other fields/languages/tools they know well — used as analogy anchors (e.g. 'Python', 'electrical engineering', 'piano').",
            },
            "personality": {
                "type": "string",
                "description": "Tone cues — terse vs chatty, likes being pushed back / gentle, humor preferences, anything that should shape voice.",
            },
            "anchors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific concepts they spontaneously cited as known/comfortable — useful for grounding new ideas.",
            },
            "notes": {
                "type": "string",
                "description": "Free-form catch-all for everything else worth remembering.",
            },
        },
    },
    requires_approval=False,
    phase="teacher",
)
def record_diagnostic_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context, args.get("course_id"))
        profile: dict[str, Any] = {
            "strengths": [str(x) for x in (args.get("strengths") or [])],
            "gaps": [str(x) for x in (args.get("gaps") or [])],
            "goals": [str(x) for x in (args.get("goals") or [])],
            "notes": str(args.get("notes", "") or ""),
            "modality": [str(x) for x in (args.get("modality") or [])],
            "pace": str(args.get("pace", "") or ""),
            "jargon_tolerance": str(args.get("jargon_tolerance", "") or ""),
            "motivation": str(args.get("motivation", "") or ""),
            "background": [str(x) for x in (args.get("background") or [])],
            "personality": str(args.get("personality", "") or ""),
            "anchors": [str(x) for x in (args.get("anchors") or [])],
            "stumbling_blocks": [],
            "recorded_at": time.time(),
            "calibration_history": [],
        }
        course.learner_profile = profile
        add_event(
            course,
            kind="diagnostic_recorded",
            entity="course",
            entity_id=course.course_id,
            payload=dict(profile),
        )
        record = _persist(session, course)
        return _ok({"course": record, "learner_profile": profile})
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="update_learner_profile",
    description=(
        "Update the learner profile mid-course when you observe a pattern. "
        "Call this when an analogy lands particularly well, the learner "
        "prefers a different pace than originally set, a new stumbling "
        "block emerges, or any other personality / learning-style signal "
        "shifts. Only pass fields you actually want to change — list "
        "fields are MERGED (existing items kept, new items appended, "
        "deduped); scalar fields OVERWRITE. Every call appends a "
        "calibration_history entry with the observation that triggered "
        "the update so the audit trail is preserved."
    ),
    parameters={
        "type": "object",
        "properties": {
            "course_id": {"type": "string"},
            "observation": {
                "type": "string",
                "description": "What you saw that prompted the update (e.g. 'learner answered the cooking-recipe analogy correctly within seconds — analogy-driven works'). Required.",
            },
            "strengths": {"type": "array", "items": {"type": "string"}},
            "gaps": {"type": "array", "items": {"type": "string"}},
            "goals": {"type": "array", "items": {"type": "string"}},
            "modality": {"type": "array", "items": {"type": "string"}},
            "pace": {"type": "string", "enum": ["slow", "moderate", "fast"]},
            "jargon_tolerance": {"type": "string", "enum": ["low", "medium", "high"]},
            "motivation": {"type": "string"},
            "background": {"type": "array", "items": {"type": "string"}},
            "personality": {"type": "string"},
            "anchors": {"type": "array", "items": {"type": "string"}},
            "stumbling_blocks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete patterns where the learner is getting stuck (e.g. 'confuses references and pointers').",
            },
            "notes": {"type": "string"},
        },
        "required": ["observation"],
    },
    requires_approval=False,
    phase="teacher",
)
def update_learner_profile_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context, args.get("course_id"))
        observation = str(args.get("observation", "") or "").strip()
        if not observation:
            return _err("observation is required — describe what you saw.")

        profile = dict(course.learner_profile or {})
        list_fields = (
            "strengths",
            "gaps",
            "goals",
            "modality",
            "background",
            "anchors",
            "stumbling_blocks",
        )
        scalar_fields = (
            "pace",
            "jargon_tolerance",
            "motivation",
            "personality",
            "notes",
        )

        deltas: dict[str, Any] = {}
        for field_name in list_fields:
            incoming = args.get(field_name)
            if incoming is None:
                continue
            existing = list(profile.get(field_name) or [])
            for item in incoming:
                item_str = str(item).strip()
                if item_str and item_str not in existing:
                    existing.append(item_str)
            profile[field_name] = existing
            deltas[field_name] = existing

        for field_name in scalar_fields:
            incoming = args.get(field_name)
            if incoming is None:
                continue
            new_value = str(incoming).strip()
            if not new_value:
                continue
            profile[field_name] = new_value
            deltas[field_name] = new_value

        history = list(profile.get("calibration_history") or [])
        history.append(
            {
                "at": time.time(),
                "observation": observation,
                "deltas": deltas,
            }
        )
        profile["calibration_history"] = history

        course.learner_profile = profile
        add_event(
            course,
            kind="learner_profile_updated",
            entity="course",
            entity_id=course.course_id,
            payload={"observation": observation, "deltas": deltas},
        )
        record = _persist(session, course)
        return _ok(
            {
                "course": record,
                "learner_profile": profile,
                "deltas": deltas,
            }
        )
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="propose_curriculum",
    description=(
        "Replace the course's modules + lessons with a proposed curriculum. "
        "Sets course status to 'curriculum_proposed'. The learner must call "
        "approve_curriculum before the lesson loop can begin."
    ),
    parameters={
        "type": "object",
        "properties": {
            "course_id": {"type": "string"},
            "modules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "module_id": {"type": "string"},
                        "title": {"type": "string"},
                        "goal": {"type": "string"},
                        "order": {"type": "integer"},
                        "mastery_threshold": {"type": "integer", "default": 70},
                        "lessons": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "lesson_id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "learning_objectives": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "concept_brief": {"type": "string"},
                                    "sources": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "title": {"type": "string"},
                                                "url": {"type": "string"},
                                                "kind": {"type": "string"},
                                                "note": {"type": "string"},
                                            },
                                            "required": ["title", "url"],
                                        },
                                    },
                                },
                                "required": ["lesson_id", "title"],
                            },
                        },
                    },
                    "required": ["module_id", "title", "lessons"],
                },
            },
        },
        "required": ["modules"],
    },
    requires_approval=False,
    phase="teacher",
)
def propose_curriculum_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context, args.get("course_id"))
        course.modules = []
        course.lessons = []
        for idx, raw_module in enumerate(args.get("modules") or [], start=1):
            module_id = _storage.slugify(str(raw_module.get("module_id") or f"module_{idx}"))
            module = Module(
                module_id=module_id,
                title=str(raw_module.get("title") or module_id),
                goal=str(raw_module.get("goal", "") or ""),
                order=int(raw_module.get("order", idx) or idx),
                mastery_threshold=int(raw_module.get("mastery_threshold", 70) or 70),
            )
            course.modules.append(module)
            for raw_lesson in raw_module.get("lessons") or []:
                lesson_id = _storage.slugify(
                    str(raw_lesson.get("lesson_id") or raw_lesson.get("title") or "")
                )
                if not lesson_id:
                    continue
                lesson = Lesson(
                    lesson_id=lesson_id,
                    module_id=module_id,
                    title=str(raw_lesson.get("title") or lesson_id),
                    learning_objectives=[
                        str(x) for x in (raw_lesson.get("learning_objectives") or [])
                    ],
                    concept_brief=str(raw_lesson.get("concept_brief", "") or ""),
                    sources=[
                        {
                            str(key): str(value)
                            for key, value in source.items()
                            if key in {"title", "url", "kind", "note"}
                        }
                        for source in (raw_lesson.get("sources") or [])
                        if isinstance(source, dict) and str(source.get("url") or "").strip()
                    ],
                )
                course.lessons.append(lesson)
                module.lesson_ids.append(lesson_id)
        course.status = COURSE_CURRICULUM_PROPOSED
        add_event(
            course,
            kind="curriculum_proposed",
            entity="course",
            entity_id=course.course_id,
            payload={"modules": len(course.modules), "lessons": len(course.lessons)},
        )
        _write_curriculum_md(course)
        record = _persist(session, course)
        return _ok(
            {
                "course": record,
                "module_count": len(course.modules),
                "lesson_count": len(course.lessons),
                "next_step": "ask the learner to call `/teach approve` or run approve_curriculum",
            }
        )
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="approve_curriculum",
    description=(
        "Mark the proposed curriculum as approved by the learner. Flips the "
        "course to 'in_progress' and unlocks the lesson loop. Refuses unless "
        "the course is currently in 'curriculum_proposed'."
    ),
    parameters={
        "type": "object",
        "properties": {"course_id": {"type": "string"}},
    },
    requires_approval=True,
    phase="teacher",
)
def approve_curriculum_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context, args.get("course_id"))
        if course.status != COURSE_CURRICULUM_PROPOSED:
            return _err(
                f"approve_curriculum refused: course status is "
                f"{course.status!r}, expected 'curriculum_proposed'"
            )
        course.status = COURSE_IN_PROGRESS
        first = next_pending_lesson(course)
        if first is not None:
            course.current_module_id = first.module_id
            course.current_lesson_id = first.lesson_id
        add_event(
            course,
            kind="curriculum_approved",
            entity="course",
            entity_id=course.course_id,
            payload={},
            actor="learner",
        )
        record = _persist(session, course)
        return _ok({"course": record, "first_lesson_id": first.lesson_id if first else None})
    except Exception as exc:
        return _err(str(exc))


# ----- lesson loop -----------------------------------------------------


@tool(
    name="start_lesson",
    description=(
        "Begin a lesson — flips its status from 'pending' to 'presenting' "
        "and marks it the current lesson on the course."
    ),
    parameters={
        "type": "object",
        "properties": {
            "course_id": {"type": "string"},
            "lesson_id": {"type": "string"},
        },
        "required": ["lesson_id"],
    },
    requires_approval=False,
    phase="teacher",
)
def start_lesson_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context, args.get("course_id"))
        lesson_id = _storage.slugify(str(args.get("lesson_id", "")).strip())
        lesson = find_lesson(course, lesson_id)
        if lesson is None:
            return _not_found_err("lesson", lesson_id, [l.lesson_id for l in course.lessons])
        advance_lesson_status(course, lesson_id, LESSON_PRESENTING)
        course.current_module_id = lesson.module_id
        course.current_lesson_id = lesson.lesson_id
        record = _persist(session, course)
        return _ok({"course": record, "lesson": asdict(lesson)})
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="assign_exercise",
    description=(
        "Create an assignment for the active lesson. The engine writes any "
        "artifact_files into the assignment's work/ directory and persists "
        "the verification spec. For socratic-dialog kinds, supply min_turns "
        "and required_concepts inside verification; for code kinds, supply "
        "verify_cmd + expected_markers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lesson_id": {"type": "string"},
            "assignment_id": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": [
                    "fix-broken-code",
                    "implement-from-scratch",
                    "predict-output",
                    "multiple-choice",
                    "fill-blank",
                    "command-output",
                    "short-answer",
                    "explain-trace",
                    "socratic-dialog",
                ],
            },
            "prompt": {"type": "string"},
            "rubric": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "weight": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["criterion"],
                },
            },
            "verification": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "verify_cmd": {"type": "string"},
                    "expected_markers": {"type": "array", "items": {"type": "string"}},
                    "forbidden_markers": {"type": "array", "items": {"type": "string"}},
                    "expected_answer": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                    "rubric_keywords": {"type": "array", "items": {"type": "string"}},
                    "min_turns": {"type": "integer"},
                    "required_concepts": {"type": "array", "items": {"type": "string"}},
                    "timeout_seconds": {"type": "integer"},
                    "working_dir": {"type": "string"},
                    "use_live_quiz_ui": {"type": "boolean"},
                },
            },
            "artifact_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
            "pass_threshold": {"type": "integer", "default": 70},
            "quiz_questions": {
                "type": "array",
                "description": (
                    "For multiple-choice / fill-blank assignments: the "
                    "per-question payload that drives the live quiz UI."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "qid": {"type": "string"},
                        "prompt": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["multiple_choice", "fill_blank"],
                        },
                        "options": {"type": "array", "items": {"type": "string"}},
                        "correct_index": {"type": "integer"},
                        "expected_pattern": {"type": "string"},
                        "case_sensitive": {"type": "boolean"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["qid", "prompt", "kind"],
                },
            },
        },
        "required": ["lesson_id", "kind", "prompt"],
    },
    requires_approval=False,
    phase="teacher",
)
def assign_exercise_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        lesson_id = _storage.slugify(str(args.get("lesson_id", "")).strip())
        lesson = find_lesson(course, lesson_id)
        if lesson is None:
            return _not_found_err("lesson", lesson_id, [l.lesson_id for l in course.lessons])
        assignment_id = _storage.slugify(
            str(args.get("assignment_id", "") or f"{lesson_id}_a{len(lesson.assignment_ids) + 1}")
        )
        verification_raw = args.get("verification") or {}
        rubric = [
            RubricItem(
                criterion=str(item.get("criterion", "")).strip(),
                weight=int(item.get("weight", 1) or 1),
                description=str(item.get("description", "")).strip(),
            )
            for item in (args.get("rubric") or [])
        ]
        spec = VerificationSpec(
            method=str(verification_raw.get("method", "")).strip() or _default_method(args["kind"]),
            verify_cmd=verification_raw.get("verify_cmd"),
            expected_markers=[str(m) for m in (verification_raw.get("expected_markers") or [])],
            forbidden_markers=[str(m) for m in (verification_raw.get("forbidden_markers") or [])],
            expected_answer=verification_raw.get("expected_answer"),
            case_sensitive=bool(verification_raw.get("case_sensitive", False)),
            rubric_keywords=[str(k) for k in (verification_raw.get("rubric_keywords") or [])],
            min_turns=int(verification_raw.get("min_turns", 0) or 0),
            required_concepts=[
                str(c) for c in (verification_raw.get("required_concepts") or [])
            ],
            timeout_seconds=int(verification_raw.get("timeout_seconds", 30) or 30),
            working_dir=str(verification_raw.get("working_dir", "") or ""),
            use_live_quiz_ui=bool(verification_raw.get("use_live_quiz_ui", False)),
        )
        # Default to live quiz UI for MC/fill-blank when the agent didn't opt out.
        if args["kind"] in {"multiple-choice", "fill-blank"} and "use_live_quiz_ui" not in verification_raw:
            spec.use_live_quiz_ui = True

        assignment = Assignment(
            assignment_id=assignment_id,
            lesson_id=lesson_id,
            kind=str(args["kind"]),
            prompt=str(args.get("prompt", "")).strip(),
            rubric=rubric,
            verification=spec,
            pass_threshold=int(args.get("pass_threshold", 70) or 70),
        )
        # Persist quiz questions onto the assignment submission scaffold so
        # grade_assignment can later compute correctness without re-receiving
        # them from the agent.
        quiz_questions = args.get("quiz_questions") or []
        if quiz_questions:
            quiz_keys: dict[str, Any] = {}
            for q in quiz_questions:
                qid = str(q.get("qid"))
                if not qid:
                    continue
                kind = q.get("kind", "multiple_choice")
                if kind == "multiple_choice":
                    options = q.get("options") or []
                    idx = int(q.get("correct_index", -1) or -1)
                    quiz_keys[qid] = options[idx] if 0 <= idx < len(options) else ""
                    quiz_keys[f"{qid}__method"] = "exact_match"
                else:
                    quiz_keys[qid] = str(q.get("expected_pattern") or "")
                    quiz_keys[f"{qid}__method"] = "regex_match" if q.get("expected_pattern") else "exact_match"
            assignment.submission = {
                "quiz_questions": quiz_questions,
                "quiz_keys": quiz_keys,
            }

        course.assignments.append(assignment)
        lesson.assignment_ids.append(assignment_id)
        course.current_assignment_id = assignment_id
        artifacts = _write_assignment_artifacts(course, assignment, args.get("artifact_files") or [])
        assignment.artifact_paths = artifacts

        # Persist the prompt for human inspection.
        prompt_dir = os.path.join(
            course.directory, "assignments", _storage.slugify(assignment_id)
        )
        os.makedirs(prompt_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "prompt.md"), "w", encoding="utf-8") as handle:
            handle.write(f"# Assignment {assignment_id}\n\n{assignment.prompt}\n")

        if lesson.status in {LESSON_PRESENTING, LESSON_LECTURING}:
            advance_lesson_status(course, lesson_id, LESSON_ASSIGNED)
        elif lesson.status == "remediating":
            # Re-assignment during remediation flips back to 'assigned'.
            advance_lesson_status(course, lesson_id, LESSON_ASSIGNED)

        add_event(
            course,
            kind="assignment_created",
            entity="assignment",
            entity_id=assignment_id,
            payload={"kind": assignment.kind, "artifact_count": len(artifacts)},
        )

        # Live quiz UI: if this is a quiz-kind assignment with the live
        # UI flag set, try to launch the quiz right now so the tool
        # result reflects what actually happened. The agent's narration
        # then matches reality ("quiz launched" vs "fell back to chat
        # flow because no TTY"). Failure here is non-fatal: the
        # grade_assignment path remains as a retry surface.
        launch_envelope = {
            "attempted": False,
            "launched": False,
            "reason": None,
            "questions_shown": 0,
            "answers_collected": 0,
        }
        is_quiz_kind = assignment.kind in {"multiple-choice", "fill-blank"}
        if is_quiz_kind and spec.use_live_quiz_ui:
            launch_envelope["attempted"] = True
            ui = getattr(session, "ui", None) or getattr(context, "ui", None)
            quiz_questions = (assignment.submission or {}).get("quiz_questions") or []
            if ui is None or not hasattr(ui, "run_quiz"):
                launch_envelope["reason"] = (
                    "no UI with run_quiz available — assignment was created but "
                    "the live quiz did not launch; the learner will need to "
                    "answer in chat or grade_assignment will retry."
                )
            elif not quiz_questions:
                launch_envelope["reason"] = (
                    "no quiz_questions provided on the assignment — supply them "
                    "to assign_exercise or the live quiz cannot launch."
                )
            else:
                try:
                    submitted = ui.run_quiz(quiz_questions)
                    if isinstance(submitted, dict) and submitted:
                        existing = assignment.submission or {}
                        assignment.submission = {**existing, "answers": submitted}
                        assignment.status = ASSIGNMENT_SUBMITTED
                        launch_envelope["launched"] = True
                        launch_envelope["questions_shown"] = len(quiz_questions)
                        launch_envelope["answers_collected"] = len(submitted)
                    else:
                        launch_envelope["reason"] = (
                            "quiz UI returned no answers — the learner may have "
                            "cancelled (q/Esc) or the picker fell back to chat-flow."
                        )
                except NotImplementedError:
                    launch_envelope["reason"] = (
                        "ui.run_quiz is not implemented on the active UI — "
                        "fell back to chat-flow."
                    )
                except Exception as exc:
                    launch_envelope["reason"] = (
                        f"quiz UI raised {type(exc).__name__}: {exc}"
                    )

        _persist(session, course)
        return _ok(
            {
                "assignment_id": assignment_id,
                "lesson_id": lesson_id,
                "artifact_paths": artifacts,
                "live_quiz": is_quiz_kind and spec.use_live_quiz_ui,
                "live_quiz_launch": launch_envelope,
            }
        )
    except Exception as exc:
        return _err(str(exc))


def _default_method(kind: str) -> str:
    return {
        "fix-broken-code": "exec_markers",
        "implement-from-scratch": "exec_markers",
        "command-output": "exec_markers",
        "predict-output": "exact_match",
        "multiple-choice": "exact_match",
        "fill-blank": "regex_match",
        "short-answer": "rubric_judge",
        "explain-trace": "rubric_judge",
        "socratic-dialog": "dialog_close",
    }.get(kind, "exec_markers")


@tool(
    name="submit_assignment",
    description=(
        "Record the learner's submission for an assignment. For code kinds "
        "the submission may carry an inline payload or simply confirm the "
        "learner edited the artifact files in work/."
    ),
    parameters={
        "type": "object",
        "properties": {
            "assignment_id": {"type": "string"},
            "submission": {
                "type": "object",
                "description": (
                    "Free-form submission payload. Conventions: "
                    "{answer: str} for predict-output / short-answer; "
                    "{answers: {qid: value}} for multi-question quizzes; "
                    "{notes: str} for code fixes done in-place on the work/ files."
                ),
            },
        },
        "required": ["assignment_id"],
    },
    requires_approval=False,
    phase="teacher",
)
def submit_assignment_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        assignment_id = _storage.slugify(str(args.get("assignment_id", "")).strip())
        assignment = find_assignment(course, assignment_id)
        if assignment is None:
            return _not_found_err(
                "assignment",
                assignment_id,
                [a.assignment_id for a in course.assignments],
            )
        submission = args.get("submission") or {}
        existing = assignment.submission or {}
        merged: dict[str, Any] = {**existing, **submission}
        assignment.submission = merged
        assignment.status = ASSIGNMENT_SUBMITTED
        sub_dir = os.path.join(
            course.directory,
            "assignments",
            _storage.slugify(assignment_id),
            "submission",
        )
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "submission.json"), "w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, default=str)
        add_event(
            course,
            kind="assignment_submitted",
            entity="assignment",
            entity_id=assignment_id,
            payload={"submission_keys": sorted(merged.keys())},
            actor="learner",
        )
        _persist(session, course)
        return _ok({"assignment_id": assignment_id, "status": assignment.status})
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="grade_assignment",
    description=(
        "Grade a previously submitted assignment. For exec_markers kinds, "
        "runs the verify_cmd and checks the markers. For multiple-choice / "
        "fill-blank with live UI enabled and no submission yet recorded, "
        "launches the live quiz Application via session.ui.run_quiz. For "
        "rubric kinds, pass llm_rubric_score and feedback after evaluating "
        "the prose answer against the rubric."
    ),
    parameters={
        "type": "object",
        "properties": {
            "assignment_id": {"type": "string"},
            "llm_rubric_score": {"type": "integer"},
            "feedback": {"type": "string"},
        },
        "required": ["assignment_id"],
    },
    requires_approval=True,
    phase="teacher",
)
def grade_assignment_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        assignment_id = _storage.slugify(str(args.get("assignment_id", "")).strip())
        assignment = find_assignment(course, assignment_id)
        if assignment is None:
            return _not_found_err(
                "assignment",
                assignment_id,
                [a.assignment_id for a in course.assignments],
            )
        if assignment.kind == "socratic-dialog":
            return _err(
                "socratic-dialog assignments are graded via close_dialog, not grade_assignment"
            )

        # Live quiz UI launch path.
        needs_quiz = (
            assignment.kind in {"multiple-choice", "fill-blank"}
            and assignment.verification.use_live_quiz_ui
            and not (assignment.submission or {}).get("answers")
        )
        if needs_quiz:
            ui = getattr(session, "ui", None) or getattr(context, "ui", None)
            quiz_questions = (assignment.submission or {}).get("quiz_questions") or []
            if ui is not None and hasattr(ui, "run_quiz") and quiz_questions:
                try:
                    submitted = ui.run_quiz(quiz_questions)
                    if isinstance(submitted, dict):
                        existing = assignment.submission or {}
                        assignment.submission = {**existing, "answers": submitted}
                        assignment.status = ASSIGNMENT_SUBMITTED
                except Exception:
                    pass  # fall through to chat-flow grading

        grade_obj = grade_assignment_payload(
            assignment,
            submission=assignment.submission,
            feedback_override=str(args.get("feedback") or "") or None,
            llm_rubric_score=(
                int(args["llm_rubric_score"]) if "llm_rubric_score" in args else None
            ),
        )
        lesson = find_lesson(course, assignment.lesson_id)
        if lesson is not None and lesson.status == LESSON_ASSIGNED:
            advance_lesson_status(course, lesson.lesson_id, LESSON_GRADED)
        add_event(
            course,
            kind="assignment_graded",
            entity="assignment",
            entity_id=assignment.assignment_id,
            payload={"score_pct": grade_obj.score_pct, "passed": grade_obj.passed},
        )
        # Persist the grade record.
        grade_path = os.path.join(
            course.directory,
            "assignments",
            _storage.slugify(assignment.assignment_id),
            "grade.json",
        )
        os.makedirs(os.path.dirname(grade_path), exist_ok=True)
        with open(grade_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(grade_obj), handle, indent=2, default=str)
        _persist(session, course)
        return _ok(
            {
                "assignment_id": assignment.assignment_id,
                "grade": asdict(grade_obj),
            }
        )
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="decide_next",
    description=(
        "After a lesson is graded, advance to the next lesson or trigger "
        "remediation. Refuses advance unless the most recent assignment "
        "passed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lesson_id": {"type": "string"},
            "action": {"type": "string", "enum": ["advance", "remediate"]},
        },
        "required": ["lesson_id", "action"],
    },
    requires_approval=True,
    phase="teacher",
)
def decide_next_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        lesson_id = _storage.slugify(str(args.get("lesson_id", "")).strip())
        action = str(args.get("action", "")).strip()
        decide_next(course, lesson_id, action)
        if action == "advance":
            nxt = next_pending_lesson(course)
            if nxt is not None:
                course.current_module_id = nxt.module_id
                course.current_lesson_id = nxt.lesson_id
            else:
                course.current_lesson_id = None
        record = _persist(session, course)
        return _ok(
            {
                "course": record,
                "action": action,
                "next_lesson_id": course.current_lesson_id,
            }
        )
    except Exception as exc:
        return _err(str(exc))


# ----- socratic dialog -------------------------------------------------


@tool(
    name="record_dialog_turn",
    description=(
        "Append one turn (agent question OR learner answer) to a "
        "socratic-dialog assignment. Call once per turn so the engine "
        "can later enforce min_turns and required_concepts coverage."
    ),
    parameters={
        "type": "object",
        "properties": {
            "assignment_id": {"type": "string"},
            "role": {"type": "string", "enum": ["agent_question", "learner_answer"]},
            "content": {"type": "string"},
            "quality_signal": {"type": "string"},
        },
        "required": ["assignment_id", "role", "content"],
    },
    requires_approval=False,
    phase="teacher",
)
def record_dialog_turn_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        turn = record_dialog_turn(
            course,
            _storage.slugify(str(args.get("assignment_id", "")).strip()),
            role=str(args.get("role", "")).strip(),
            content=str(args.get("content", "")).strip(),
            quality_signal=(
                str(args.get("quality_signal")).strip()
                if args.get("quality_signal")
                else None
            ),
        )
        _persist(session, course)
        return _ok({"turn": asdict(turn)})
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="close_dialog",
    description=(
        "Finalize a socratic-dialog assignment with an honest mastery "
        "score, a summary, and a list of gaps. Refuses unless min_turns "
        "and required_concepts coverage thresholds are met."
    ),
    parameters={
        "type": "object",
        "properties": {
            "assignment_id": {"type": "string"},
            "mastery_pct": {"type": "integer"},
            "summary": {"type": "string"},
            "gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["assignment_id", "mastery_pct", "summary"],
    },
    requires_approval=True,
    phase="teacher",
)
def close_dialog_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        assignment_id = _storage.slugify(str(args.get("assignment_id", "")).strip())
        grade_obj = close_socratic_dialog(
            course,
            assignment_id,
            mastery_pct=int(args.get("mastery_pct", 0)),
            summary=str(args.get("summary", "")).strip(),
            gaps=[str(x) for x in (args.get("gaps") or [])],
        )
        assignment = find_assignment(course, assignment_id)
        lesson = find_lesson(course, assignment.lesson_id) if assignment else None
        if lesson is not None and lesson.status == LESSON_ASSIGNED:
            advance_lesson_status(course, lesson.lesson_id, LESSON_GRADED)
        _persist(session, course)
        return _ok(
            {
                "assignment_id": assignment_id,
                "grade": asdict(grade_obj),
            }
        )
    except Exception as exc:
        return _err(str(exc))


# ----- inspection / control -------------------------------------------


@tool(
    name="get_course_state",
    description=(
        "Return a snapshot of the active course: status, current "
        "module/lesson/assignment, completion %, average score, and "
        "the most recently graded assignment."
    ),
    parameters={
        "type": "object",
        "properties": {"course_id": {"type": "string"}},
    },
    requires_approval=False,
    phase="teacher",
)
def get_course_state_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context, args.get("course_id"))
        metrics = course_metrics(course)
        next_lesson = next_pending_lesson(course)
        latest_graded = None
        for a in reversed(course.assignments):
            if a.grade is not None:
                latest_graded = {
                    "assignment_id": a.assignment_id,
                    "lesson_id": a.lesson_id,
                    "score_pct": a.grade.score_pct,
                    "passed": a.grade.passed,
                }
                break
        return _ok(
            {
                "course_id": course.course_id,
                "subject": course.subject,
                "status": course.status,
                "metrics": metrics,
                "current_module_id": course.current_module_id,
                "current_lesson_id": course.current_lesson_id,
                "current_assignment_id": course.current_assignment_id,
                "next_pending_lesson_id": next_lesson.lesson_id if next_lesson else None,
                "latest_graded": latest_graded,
            }
        )
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="complete_module",
    description=(
        "Mark a module complete. Refuses unless every lesson in the module "
        "is in 'completed' status AND the aggregate score across the "
        "module's graded assignments meets the module's mastery_threshold."
    ),
    parameters={
        "type": "object",
        "properties": {"module_id": {"type": "string"}},
        "required": ["module_id"],
    },
    requires_approval=True,
    phase="teacher",
)
def complete_module_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        module_id = _storage.slugify(str(args.get("module_id", "")).strip())
        module = find_module(course, module_id)
        if module is None:
            return _not_found_err("module", module_id, [m.module_id for m in course.modules])
        incomplete = []
        for lesson_id in module.lesson_ids:
            lesson = find_lesson(course, lesson_id)
            if lesson is None or lesson.status != LESSON_COMPLETED:
                incomplete.append(lesson_id)
        if incomplete:
            return _err(
                f"complete_module refused: lessons not yet completed: "
                + ", ".join(incomplete)
            )
        # Aggregate score across passed assignments in this module.
        lesson_ids = set(module.lesson_ids)
        graded = [
            a
            for a in course.assignments
            if a.lesson_id in lesson_ids and a.grade is not None
        ]
        if not graded:
            agg = 0
        else:
            agg = int(round(sum(a.grade.score_pct for a in graded) / len(graded)))
        if agg < module.mastery_threshold:
            return _err(
                f"complete_module refused: aggregate score {agg}% < "
                f"mastery_threshold {module.mastery_threshold}%"
            )
        module.status = LESSON_COMPLETED
        add_event(
            course,
            kind="module_completed",
            entity="module",
            entity_id=module_id,
            payload={"aggregate_score_pct": agg},
        )
        _persist(session, course)
        return _ok({"module_id": module_id, "aggregate_score_pct": agg})
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="finalize_course",
    description=(
        "Mark the course completed, write the report card, and save a "
        "durable `user_skill:<subject>` memory so future courses can recall "
        "what the learner already knows."
    ),
    parameters={
        "type": "object",
        "properties": {"course_id": {"type": "string"}},
    },
    requires_approval=True,
    phase="teacher",
)
def finalize_course_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context, args.get("course_id"))
        course.status = COURSE_COMPLETED
        add_event(
            course,
            kind="course_completed",
            entity="course",
            entity_id=course.course_id,
            payload=course_metrics(course),
        )
        report_path = _write_report_card(course)
        # Persist a durable memory tag for cross-session recall.
        try:
            memory = getattr(session.session_manager, "task_memory", None)
            if memory is not None and hasattr(memory, "save"):
                metrics = course_metrics(course)
                memory.save(
                    content=(
                        f"Completed course '{course.subject}' "
                        f"(level: {course.target_level}, "
                        f"avg score: {metrics['average_score_pct']}%, "
                        f"lessons: {metrics['lessons_completed']}/{metrics['total_lessons']})"
                    ),
                    tags=[f"user_skill:{_storage.slugify(course.subject)}", "course_complete"],
                    source="teacher_mode",
                )
        except Exception:
            pass
        record = _persist(session, course)
        return _ok({"course": record, "report_card_path": report_path})
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="raise_teacher_blocker",
    description=(
        "Signal that the agent needs clarification from the learner before "
        "proceeding (e.g. unclear goal, missing background, environment "
        "issue). Mirror of feature mode's raise_blocker."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "details": {"type": "string"},
            "questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary"],
    },
    requires_approval=False,
    phase="teacher",
)
def raise_teacher_blocker_tool(args: dict[str, Any], context) -> str:
    payload = {
        "kind": "teacher_blocker",
        "summary": str(args.get("summary", "")).strip(),
        "details": str(args.get("details", "")).strip(),
        "questions": [
            str(item).strip() for item in (args.get("questions") or []) if str(item).strip()
        ],
    }
    return _ok(payload)


# ----- spaced review --------------------------------------------------


@tool(
    name="schedule_review",
    description=(
        "Queue a spaced-repetition checkpoint for a finished lesson. The "
        "review fires after `after_n_lessons` more lessons have been "
        "completed — so a value of 3 means 'check the learner can still "
        "do this concept three lessons from now'. Use after concluding a "
        "lesson with a non-perfect grade, after a remediation, or at the "
        "end of a module's worth of lessons. The agent is responsible "
        "for surfacing due reviews via get_due_reviews and complete_review "
        "once the learner has answered."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source_lesson_id": {"type": "string"},
            "after_n_lessons": {
                "type": "integer",
                "description": (
                    "How many MORE lessons must complete before this review "
                    "comes due. 2–5 is the sweet spot — close enough that "
                    "the concept is still fresh-ish, far enough that "
                    "actual retention is being tested."
                ),
            },
            "review_id": {
                "type": "string",
                "description": "Optional explicit slug; auto-generated otherwise.",
            },
            "notes": {
                "type": "string",
                "description": (
                    "Optional context: why this review is being scheduled "
                    "(e.g. 'learner barely passed; recheck pointer arithmetic')."
                ),
            },
        },
        "required": ["source_lesson_id", "after_n_lessons"],
    },
    requires_approval=False,
    phase="teacher",
)
def schedule_review_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        review = schedule_review(
            course,
            source_lesson_id=_storage.slugify(str(args.get("source_lesson_id", "")).strip()),
            after_n_lessons=int(args.get("after_n_lessons", 0)),
            review_id=str(args.get("review_id") or "").strip() or None,
            notes=str(args.get("notes", "") or ""),
        )
        _persist(session, course)
        return _ok(
            {
                "review_id": review.review_id,
                "source_lesson_id": review.source_lesson_id,
                "due_at_lesson_count": review.due_at_lesson_count,
                "current_lesson_count": course.lessons_completed_count,
            }
        )
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="get_due_reviews",
    description=(
        "Return spaced-review checkpoints whose due-date has passed. "
        "Surface these to the learner — at the start of a lesson is the "
        "natural spot, or at module boundaries. Empty list means "
        "nothing's due. Each review has `source_lesson_id` so you can "
        "re-issue the original lesson's exercise as a review prompt; "
        "after grading, call complete_review with the score."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
    requires_approval=False,
    phase="teacher",
    execution_kind="read",
)
def get_due_reviews_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        items = [
            {
                "review_id": r.review_id,
                "source_lesson_id": r.source_lesson_id,
                "due_at_lesson_count": r.due_at_lesson_count,
                "lessons_overdue_by": max(
                    0, course.lessons_completed_count - r.due_at_lesson_count
                ),
                "notes": r.notes,
            }
            for r in due_reviews(course)
        ]
        return _ok(
            {
                "due_reviews": items,
                "current_lesson_count": course.lessons_completed_count,
                "total_scheduled": len(course.scheduled_reviews),
            }
        )
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="complete_review",
    description=(
        "Finalize a spaced-review checkpoint after the learner has answered. "
        "Pass the score the learner achieved (0–100). If the learner skipped "
        "or you decided to defer it, set skipped=true — the review is still "
        "marked complete, but the audit trail records the choice. Schedule "
        "a fresh review with schedule_review if you want a retake."
    ),
    parameters={
        "type": "object",
        "properties": {
            "review_id": {"type": "string"},
            "score_pct": {"type": "integer"},
            "skipped": {"type": "boolean", "default": False},
            "notes": {
                "type": "string",
                "description": (
                    "Optional post-mortem: what did the learner still "
                    "remember, what gaps showed up."
                ),
            },
        },
        "required": ["review_id", "score_pct"],
    },
    requires_approval=True,
    phase="teacher",
)
def complete_review_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        review = complete_review(
            course,
            str(args.get("review_id", "")).strip(),
            score_pct=int(args.get("score_pct", 0)),
            notes=str(args.get("notes", "") or ""),
            skipped=bool(args.get("skipped", False)),
        )
        _persist(session, course)
        return _ok(
            {
                "review_id": review.review_id,
                "status": review.status,
                "score_pct": review.score_pct,
            }
        )
    except Exception as exc:
        return _err(str(exc))


# ── Dual presentation artifacts ──────────────────────────────────


@tool(
    name="write_lecture_transcript",
    description=(
        "Write the authored lecture transcript for a lesson. Creates "
        "lessons/<lesson_id>/lecture.md on disk. The transcript is the "
        "primary served content; exercises are referenced by relative "
        "path. Does NOT alter lesson lifecycle status."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lesson_id": {
                "type": "string",
                "description": "The lesson to write the transcript for.",
            },
            "content": {
                "type": "string",
                "description": "Markdown content of the lecture transcript.",
            },
        },
        "required": ["lesson_id", "content"],
    },
    requires_approval=False,
    phase="teacher",
)
def write_lecture_transcript_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        lesson_id = str(args.get("lesson_id", "")).strip()
        content = str(args.get("content", "") or "")
        if not lesson_id:
            return _err("lesson_id is required")
        lesson = find_lesson(course, lesson_id)
        if lesson is None:
            return _err(f"lesson {lesson_id!r} not found")
        if lesson.status == LESSON_PENDING:
            return _err(
                "lesson must be at least 'presenting' before writing a "
                "lecture transcript"
            )
        path = write_lecture_transcript(course, lesson_id, content)
        _persist(session, course)
        return _ok(
            {
                "lesson_id": lesson_id,
                "lecture_transcript_path": lesson.lecture_transcript_path,
                "path": path,
            }
        )
    except Exception as exc:
        return _err(str(exc))


@tool(
    name="register_exercise_file",
    description=(
        "Write an exercise/example file for a lesson. Creates "
        "lessons/<lesson_id>/exercises/<relative_path> on disk and "
        "appends to the lesson's exercise_file_paths. Does NOT alter "
        "lesson lifecycle status and is NOT a graded assignment."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lesson_id": {
                "type": "string",
                "description": "The lesson to register the exercise for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Relative filename within the exercises/ directory "
                    "(e.g. 'example_01.py'). Must not be absolute or "
                    "contain '..' segments."
                ),
            },
            "content": {
                "type": "string",
                "description": "File content to write.",
            },
        },
        "required": ["lesson_id", "path", "content"],
    },
    requires_approval=False,
    phase="teacher",
)
def register_exercise_file_tool(args: dict[str, Any], context) -> str:
    try:
        session = _session_from_context(context)
        course = _load_active_course(session, context)
        lesson_id = str(args.get("lesson_id", "")).strip()
        rel_path = str(args.get("path", "")).strip()
        content = str(args.get("content", "") or "")
        if not lesson_id:
            return _err("lesson_id is required")
        if not rel_path:
            return _err("path is required")
        lesson = find_lesson(course, lesson_id)
        if lesson is None:
            return _err(f"lesson {lesson_id!r} not found")
        if lesson.status == LESSON_PENDING:
            return _err(
                "lesson must be at least 'presenting' before registering "
                "exercise files"
            )
        path = register_exercise_file(course, lesson_id, rel_path, content)
        _persist(session, course)
        return _ok(
            {
                "lesson_id": lesson_id,
                "exercise_file_paths": lesson.exercise_file_paths,
                "path": path,
            }
        )
    except Exception as exc:
        return _err(str(exc))


__all__ = [
    "approve_curriculum_tool",
    "assign_exercise_tool",
    "close_dialog_tool",
    "complete_module_tool",
    "complete_review_tool",
    "create_course_tool",
    "decide_next_tool",
    "finalize_course_tool",
    "get_course_state_tool",
    "get_due_reviews_tool",
    "grade_assignment_tool",
    "propose_curriculum_tool",
    "raise_teacher_blocker_tool",
    "record_diagnostic_tool",
    "record_dialog_turn_tool",
    "register_exercise_file_tool",
    "schedule_review_tool",
    "start_lesson_tool",
    "submit_assignment_tool",
    "update_learner_profile_tool",
    "write_lecture_transcript_tool",
]
