"""Teacher-mode introspection.

Exposes the live course state (curriculum, learner profile, current
lesson, grades) so the GUI can render a side-panel showing what the
agent has been working on. Read-only — no mutation endpoints. State is
mutated through the existing teacher tools via the chat send path.

Returns `null`-shaped payloads when no session or course is active so
the panel can mount empty without erroring.

Data sources: the SessionManager only persists a lightweight metadata
stub for each course (course_id, subject, directory, status, metrics).
The rich data — modules, lessons, learner_profile, assignments — lives
in ``<course_directory>/course.json``, written by the teacher engine.
This router hydrates from that file when the stub points to a directory
we can read, so the GUI sees what's actually on disk rather than only
the in-memory metadata.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from ..mode_workspace import teacher_workspace
from ..mode_session import mode_session

router = APIRouter()
_logger = logging.getLogger(__name__)


def _hydrate_from_disk(stub: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """If `stub` carries a `directory` and `<directory>/course.json`
    exists, return the merged disk view. Falls back to the stub unchanged.

    The disk file is the source of truth — modules/lessons/learner_profile
    are only stored there. We merge so any fields present in the in-memory
    stub but not the file (rare) are preserved.
    """
    if not isinstance(stub, dict):
        return stub
    directory = stub.get("directory")
    if not directory:
        return stub
    path = os.path.join(directory, "course.json")
    if not os.path.exists(path):
        return stub
    try:
        with open(path, encoding="utf-8") as fh:
            disk = json.load(fh)
    except (OSError, ValueError) as exc:
        _logger.warning("teacher: could not read %s: %s", path, exc)
        return stub
    if not isinstance(disk, dict):
        return stub
    merged = {**stub, **disk}
    return merged


def _summarize_assignments(course: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact assignment list with the bits the panel actually shows:
    id, lesson, kind, status, grade summary, plus the prompt/rubric/
    verification/artifact_paths/dialog_turns so the panel can render
    full detail rows when an assignment is expanded."""
    out: List[Dict[str, Any]] = []
    for a in course.get("assignments", []) or []:
        if not isinstance(a, dict):
            continue
        grade = a.get("grade") or {}
        verification = a.get("verification") or {}
        # Compact the dialog turns so the panel can show a turn-by-turn
        # transcript without shipping the full submission payload.
        dialog_turns = []
        for t in a.get("dialog_turns", []) or []:
            if not isinstance(t, dict):
                continue
            dialog_turns.append(
                {
                    "role": t.get("role"),
                    "content": (t.get("content") or "").strip(),
                    "quality_signal": t.get("quality_signal"),
                }
            )
        # Compact the rubric so each criterion is present but stripped
        # of any heavyweight metadata the panel does not render.
        rubric = []
        for r in a.get("rubric", []) or []:
            if not isinstance(r, dict):
                continue
            rubric.append(
                {
                    "criterion": r.get("criterion"),
                    "description": r.get("description"),
                    "weight": r.get("weight"),
                }
            )
        out.append(
            {
                "assignment_id": a.get("assignment_id"),
                "lesson_id": a.get("lesson_id"),
                "kind": a.get("kind"),
                "status": a.get("status"),
                "prompt": a.get("prompt") or "",
                "pass_threshold": a.get("pass_threshold"),
                "artifact_paths": list(a.get("artifact_paths") or []),
                "rubric": rubric,
                "verification": {
                    "method": verification.get("method"),
                    "verify_cmd": verification.get("verify_cmd"),
                    "expected_markers": list(verification.get("expected_markers") or []),
                    "expected_answer": verification.get("expected_answer"),
                    "min_turns": verification.get("min_turns"),
                    "required_concepts": list(verification.get("required_concepts") or []),
                }
                if isinstance(verification, dict) and verification
                else {},
                "dialog_turns": dialog_turns,
                "submission": a.get("submission") or {},
                "grade": {
                    "score_pct": grade.get("score_pct"),
                    "passed": grade.get("passed"),
                    "feedback": grade.get("feedback"),
                }
                if grade
                else None,
            }
        )
    return out


def _summarize_reviews(course: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize scheduled_reviews so the panel template can render them
    without knowing the engine's exact field names.

    The engine stores: review_id, source_lesson_id, due_at_lesson_count,
    status, score_pct, completed_at, notes, created_at. We surface those
    plus a couple of display-friendly derived fields (status_label,
    source_lesson_title when joinable).
    """
    lessons_by_id = {
        l.get("lesson_id"): l
        for l in course.get("lessons", []) or []
        if isinstance(l, dict) and l.get("lesson_id")
    }
    out: List[Dict[str, Any]] = []
    for r in course.get("scheduled_reviews", []) or []:
        if not isinstance(r, dict):
            continue
        status = (r.get("status") or "").lower()
        source_lesson_id = r.get("source_lesson_id")
        source_lesson = lessons_by_id.get(source_lesson_id) or {}
        out.append(
            {
                "review_id": r.get("review_id"),
                "source_lesson_id": source_lesson_id,
                "source_lesson_title": source_lesson.get("title") or source_lesson_id or "?",
                "due_at_lesson_count": r.get("due_at_lesson_count"),
                "status": r.get("status"),
                "status_label": {
                    "pending": "due",
                    "done": "done",
                    "skipped": "skipped",
                }.get(status, r.get("status") or "—"),
                "score_pct": r.get("score_pct"),
                "completed_at": r.get("completed_at"),
                "created_at": r.get("created_at"),
                "notes": r.get("notes") or "",
            }
        )
    return out


def _summarize_lesson_turns(raw_turns: Any) -> List[Dict[str, Any]]:
    """Compact the lecture_turns list for the panel: keep role + content +
    timestamp + comprehension markers, drop any heavyweight fields."""
    turns: List[Dict[str, Any]] = []
    for t in raw_turns or []:
        if not isinstance(t, dict):
            continue
        turns.append(
            {
                "role": t.get("role"),
                "content": (t.get("content") or "").strip(),
                "timestamp": t.get("timestamp"),
                "comprehension_pct": t.get("comprehension_pct"),
                "gaps": list(t.get("gaps") or []),
            }
        )
    return turns


def _summarize_lessons(course: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for l in course.get("lessons", []) or []:
        if not isinstance(l, dict):
            continue
        lecture_turns = l.get("lecture_turns") or []
        out.append(
            {
                "lesson_id": l.get("lesson_id"),
                "module_id": l.get("module_id"),
                "title": l.get("title"),
                "status": l.get("status"),
                "concept_brief": l.get("concept_brief", ""),
                "learning_objectives": list(l.get("learning_objectives") or []),
                "lecture_turn_count": len(lecture_turns),
                "lecture_turns": _summarize_lesson_turns(lecture_turns),
                "lecture_comprehension_pct": l.get("lecture_comprehension_pct"),
                "lecture_gaps": list(l.get("lecture_gaps") or []),
                "lecture_concluded": bool(l.get("lecture_concluded")),
                "assignment_ids": list(l.get("assignment_ids") or []),
                "remediation_count": l.get("remediation_count", 0),
                "lecture_transcript_path": l.get("lecture_transcript_path"),
                "exercise_file_paths": list(l.get("exercise_file_paths") or []),
            }
        )
    return out


def _summarize_modules(course: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Modules with their lessons embedded.

    The frontend renders the curriculum tree as `module → lessons`, so
    pre-joining here keeps the template trivial (avoids fragile
    nested-find expressions in Alpine).

    Round-44 F11: module-embedded lessons are CARD copies — the curriculum
    tree renders only title/status/comprehension. The full lecture
    transcript (``lecture_turns``) stays ONLY on the top-level lessons
    list, which backs the lesson detail view; embedding it again here
    serialized the whole lecture history twice per /state refresh.
    """
    summarized_lessons = _summarize_lessons(course)
    lessons_by_module: Dict[str, List[Dict[str, Any]]] = {}
    for l in summarized_lessons:
        mid = str(l.get("module_id") or "")
        card = {
            k: v for k, v in l.items()
            if k != "lecture_turns"  # F11: transcript lives on top-level lessons only
        }
        lessons_by_module.setdefault(mid, []).append(card)

    out: List[Dict[str, Any]] = []
    seen_module_ids: set[str] = set()
    for m in course.get("modules", []) or []:
        if not isinstance(m, dict):
            continue
        module_id = str(m.get("module_id") or "")
        # Defensive: skip duplicate module_ids (breaks Alpine x-for :key)
        if module_id in seen_module_ids:
            continue
        seen_module_ids.add(module_id)
        lesson_ids = list(m.get("lesson_ids") or [])
        # Preserve the module's explicit lesson order when given; fall
        # back to lookup-order otherwise.
        module_lessons: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for lid in lesson_ids:
            for l in lessons_by_module.get(module_id, []):
                if l.get("lesson_id") == lid and l.get("lesson_id") not in seen:
                    module_lessons.append(l)
                    seen.add(l.get("lesson_id"))
                    break
        # Append any lessons claimed by this module but missing from
        # lesson_ids (shouldn't happen, but defensive).
        for l in lessons_by_module.get(module_id, []):
            if l.get("lesson_id") not in seen:
                module_lessons.append(l)
                seen.add(l.get("lesson_id"))
        out.append(
            {
                "module_id": m.get("module_id"),
                "title": m.get("title"),
                "goal": m.get("goal", ""),
                "order": m.get("order", 0),
                "status": m.get("status"),
                "lesson_ids": lesson_ids,
                "mastery_threshold": m.get("mastery_threshold"),
                "lessons": module_lessons,
            }
        )

    # Lessons not claimed by any module land in a synthetic "_loose"
    # module so they still render.
    placed_ids = {l.get("lesson_id") for m in out for l in m.get("lessons", [])}
    loose = [l for l in summarized_lessons if l.get("lesson_id") not in placed_ids]
    if loose:
        out.append(
            {
                "module_id": "_loose",
                "title": "(unassigned)",
                "goal": "",
                "order": 999,
                "status": None,
                "lesson_ids": [l.get("lesson_id") for l in loose],
                "mastery_threshold": None,
                "lessons": loose,
            }
        )

    out.sort(key=lambda m: (m.get("order") or 0, str(m.get("module_id") or "")))
    return out


def _course_payload(course: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(course, dict):
        return None
    lessons = _summarize_lessons(course)
    modules = _summarize_modules(course)
    assignments = _summarize_assignments(course)
    completed = sum(1 for l in lessons if (l.get("status") or "") == "completed")
    return {
        "course_id": course.get("course_id"),
        "subject": course.get("subject"),
        "target_level": course.get("target_level"),
        "status": course.get("status"),
        "directory": course.get("directory"),
        "learner_profile": course.get("learner_profile") or {},
        "current_module_id": course.get("current_module_id"),
        "current_lesson_id": course.get("current_lesson_id"),
        "current_assignment_id": course.get("current_assignment_id"),
        "lessons_completed_count": course.get("lessons_completed_count", completed),
        "lesson_total": len(lessons),
        "modules": modules,
        "lessons": lessons,
        "assignments": assignments,
        "scheduled_reviews": _summarize_reviews(course),
    }


@router.get("/state")
async def get_teacher_state(request: Request) -> Dict[str, Any]:
    session = mode_session(request)
    if session is None:
        return {
            "active": False,
            "active_course_id": None,
            "course": None,
            "courses": [],
            "raw_teacher_state_present": False,
            "registry_size": 0,
            "workspace": teacher_workspace(None, [], active=False),
        }
    sm = session.session_manager
    mode_active = sm.variables.get("agent_mode", "default") == "teacher"

    # Pick the active course's metadata stub. SessionManager keeps a
    # registry; the active id points into it.
    teacher_state = sm.teacher_state
    if teacher_state is None and sm.active_course_id:
        registry_record = (sm.teacher_registry or {}).get(sm.active_course_id)
        if isinstance(registry_record, dict):
            teacher_state = registry_record

    # Stubs only carry course-level metadata. The actual curriculum +
    # learner data lives in course.json under the course directory —
    # hydrate from there so the panel sees the real picture.
    teacher_state = _hydrate_from_disk(teacher_state)
    course = _course_payload(teacher_state)

    courses = []
    for cid, record in (sm.teacher_registry or {}).items():
        if not isinstance(record, dict):
            continue
        hydrated = _hydrate_from_disk(record) or record
        courses.append(
            {
                "course_id": cid,
                "subject": hydrated.get("subject") or record.get("subject"),
                "status": hydrated.get("status") or record.get("status"),
                "is_active": cid == sm.active_course_id,
                "lesson_total": len(hydrated.get("lessons") or []),
                "lessons_completed_count": hydrated.get("lessons_completed_count", 0),
                "directory": hydrated.get("directory") or record.get("directory"),
            }
        )
    courses.sort(key=lambda c: (not c["is_active"], str(c["course_id"] or "")))

    return {
        "active": session is not None,
        "active_course_id": sm.active_course_id,
        "course": course,
        "courses": courses,
        # Diagnostics so the GUI can show a clear "saved but not loaded"
        # state when registry has data but no course is active.
        "raw_teacher_state_present": isinstance(sm.teacher_state, dict),
        "registry_size": len(sm.teacher_registry or {}),
        # Where the hydrated course data was read from, so we can debug
        # cases where the panel still looks empty.
        "course_path": (
            os.path.join(teacher_state["directory"], "course.json")
            if isinstance(teacher_state, dict) and teacher_state.get("directory")
            else None
        ),
        "workspace": teacher_workspace(course, courses, active=mode_active),
    }


# ── Dual presentation endpoints ─────────────────────────────────


def _resolve_course_dir(request: Request) -> Optional[str]:
    """Resolve the active course directory from the session."""
    session = mode_session(request)
    if session is None:
        return None
    sm = session.session_manager
    teacher_state = sm.teacher_state
    if teacher_state is None and sm.active_course_id:
        registry_record = (sm.teacher_registry or {}).get(sm.active_course_id)
        if isinstance(registry_record, dict):
            teacher_state = registry_record
    teacher_state = _hydrate_from_disk(teacher_state)
    if isinstance(teacher_state, dict):
        return teacher_state.get("directory")
    return None


def _lesson_dir(course_dir: str, lesson_id: str) -> str:
    """Return the per-lesson directory path."""
    from mu.teacher.storage import slugify
    return os.path.join(course_dir, "lessons", slugify(lesson_id))


@router.get("/lessons/{lesson_id}/lecture")
async def get_lecture_transcript(request: Request, lesson_id: str) -> Dict[str, Any]:
    """Return the authored lecture.md content for a lesson, or 404."""
    course_dir = _resolve_course_dir(request)
    if not course_dir:
        raise HTTPException(status_code=404, detail="No active course.")
    path = os.path.join(_lesson_dir(course_dir, lesson_id), "lecture.md")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No lecture transcript for this lesson.")
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    return {"lesson_id": lesson_id, "content": content}


@router.get("/lessons/{lesson_id}/exercises")
async def get_exercises_listing(request: Request, lesson_id: str) -> Dict[str, Any]:
    """Return a listing of exercise file paths plus each file's contents."""
    course_dir = _resolve_course_dir(request)
    if not course_dir:
        raise HTTPException(status_code=404, detail="No active course.")
    exercises_dir = os.path.join(_lesson_dir(course_dir, lesson_id), "exercises")
    if not os.path.isdir(exercises_dir):
        return {"lesson_id": lesson_id, "files": []}
    files = []
    for root, dirs, filenames in os.walk(exercises_dir):
        dirs.sort()
        for fname in sorted(filenames):
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, exercises_dir)
            try:
                with open(full, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except Exception:
                content = ""
            files.append({"path": rel, "content": content})
    return {"lesson_id": lesson_id, "files": files}


@router.get("/lessons/{lesson_id}/exercises/{path:path}")
async def get_exercise_file(request: Request, lesson_id: str, path: str) -> Dict[str, Any]:
    """Return a single exercise file's contents (path-validated)."""
    course_dir = _resolve_course_dir(request)
    if not course_dir:
        raise HTTPException(status_code=404, detail="No active course.")
    exercises_dir = os.path.join(_lesson_dir(course_dir, lesson_id), "exercises")
    target = os.path.normpath(os.path.join(exercises_dir, path))
    # Path traversal guard: resolved path must stay under exercises_dir.
    if not os.path.abspath(target).startswith(os.path.abspath(exercises_dir)):
        raise HTTPException(status_code=403, detail="Path traversal not allowed.")
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="Exercise file not found.")
    with open(target, "r", encoding="utf-8") as fh:
        content = fh.read()
    rel = os.path.relpath(target, exercises_dir)
    return {"lesson_id": lesson_id, "path": rel, "content": content}
