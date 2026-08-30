"""Teacher-mode transcript watchers, extracted from mu/agent/loop_body.py.

These hooks observe user/assistant turns during an active teacher lesson
and persist course state. loop_body imports them; no other module does.
"""

from __future__ import annotations

import json

from utils.logger import logger


def _active_teacher_lesson(session):
    """Return ``(course, lesson_id)`` if teacher mode has a lesson the
    watcher should classify against; ``(None, None)`` otherwise.

    Eligible iff there's a teacher_state with a current_lesson_id whose
    lesson is in presenting/lecturing status. Other states are skipped
    so we don't fabricate transcript entries for completed lessons or
    off-topic chatter.
    """
    state = getattr(session.session_manager, "teacher_state", None)
    if not isinstance(state, dict):
        return None, None
    lesson_id = state.get("current_lesson_id")
    if not lesson_id:
        return None, None
    try:
        from dataclasses import is_dataclass

        from mu.teacher.engine import Course
        from mu.teacher.watcher import is_watcher_eligible

        # The session_manager stores the course as a dict; the engine
        # functions take a Course dataclass. Hydrate one lazily.
        course = Course(**{k: v for k, v in state.items() if k in Course.__dataclass_fields__})  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("watcher: could not hydrate course: %s", exc)
        return None, None
    if not is_watcher_eligible(course, lesson_id):
        return None, None
    return course, lesson_id


def _run_teacher_watcher_user(session, user_text: str) -> None:
    course, lesson_id = _active_teacher_lesson(session)
    if course is None:
        return
    try:
        from mu.teacher.engine import find_lesson
        from mu.teacher.watcher import (
            apply_learner_classification,
            classify_user_message,
        )

        lesson = find_lesson(course, lesson_id)
        classification = classify_user_message(
            session.provider, user_text, lesson=lesson
        )
        if not classification:
            return
        result = apply_learner_classification(course, lesson_id, classification)
        _persist_teacher_course(session, course)
        if result.events_applied and session.ui:
            session.ui.show_info(
                f"📚 watcher: recorded {len(result.events_applied)} "
                f"learner turn{'s' if len(result.events_applied) != 1 else ''}"
            )
    except Exception as exc:
        logger.warning("watcher (user): %s", exc)


def _run_teacher_watcher_assistant(session, assistant_text: str) -> None:
    course, lesson_id = _active_teacher_lesson(session)
    if course is None:
        return
    try:
        from mu.teacher.engine import find_lesson
        from mu.teacher.watcher import (
            apply_assistant_classification,
            classify_assistant_message,
        )

        lesson = find_lesson(course, lesson_id)
        classification = classify_assistant_message(
            session.provider, assistant_text, lesson=lesson
        )
        if not classification:
            return
        result = apply_assistant_classification(course, lesson_id, classification)
        _persist_teacher_course(session, course)
        if result.agent_feedback and session.ui:
            session.ui.show_info(f"📚 {result.agent_feedback}")
        if result.events_applied and session.ui:
            kinds = ", ".join(e["kind"] for e in result.events_applied)
            session.ui.show_info(f"📚 watcher: recorded {kinds}")
        if result.auto_concluded_lecture and session.ui:
            session.ui.show_info("📚 lecture auto-concluded by watcher")
    except Exception as exc:
        logger.warning("watcher (assistant): %s", exc)


def _persist_teacher_course(session, course) -> None:
    """Mirror engine course state back onto session_manager so the
    next iteration's prompt + L3 reflect the watcher's updates.

    Uses the same persistence path the existing tool handlers use
    (`_persist` in `mu/tools/teacher/handlers.py`) by going through
    `session_manager.upsert_teacher_course`.
    """
    try:
        from dataclasses import asdict

        from mu.teacher.engine import save_course

        record = asdict(course)
        # Write course.json to disk so subsequent reads see the new
        # lecture turns.
        save_course(course)
        session.session_manager.upsert_teacher_course(record)
        # Make sure teacher_state mirrors registry so L3 injection
        # picks up the new lecture turns.
        session.session_manager.teacher_state = record
        session.session_manager.active_course_id = course.course_id
        session.session_manager.save_history(session.folder_context)
    except Exception as exc:
        logger.warning("watcher: persist course failed: %s", exc)


def _render_learner_profile_block(session) -> str:
    """Compact LEARNER PROFILE block for the teacher-mode system prompt.

    Read from the active course's `learner_profile`. Returns "" when no
    course is active OR when the profile has no usable fields yet
    (pre-diagnostic), which is the right behavior — the model shouldn't
    confabulate context the diagnostic hasn't established.
    """
    state = getattr(session.session_manager, "teacher_state", None)
    if not isinstance(state, dict):
        return ""
    profile = state.get("learner_profile") or {}
    if not isinstance(profile, dict):
        return ""

    def _list(key: str) -> list[str]:
        val = profile.get(key)
        return [str(x).strip() for x in val if str(x).strip()] if isinstance(val, list) else []

    def _scalar(key: str) -> str:
        return str(profile.get(key, "") or "").strip()

    sections: list[str] = []
    if _list("strengths"):
        sections.append(f"- Strengths: {', '.join(_list('strengths'))}")
    if _list("gaps"):
        sections.append(f"- Gaps: {', '.join(_list('gaps'))}")
    if _list("goals"):
        sections.append(f"- Goals: {', '.join(_list('goals'))}")
    if _list("modality"):
        sections.append(f"- Preferred modalities: {', '.join(_list('modality'))}")
    if _scalar("pace"):
        sections.append(f"- Pace: {_scalar('pace')}")
    if _scalar("jargon_tolerance"):
        sections.append(f"- Jargon tolerance: {_scalar('jargon_tolerance')}")
    if _scalar("motivation"):
        sections.append(f"- Motivation: {_scalar('motivation')}")
    if _list("background"):
        sections.append(
            f"- Background (use as analogy anchors): {', '.join(_list('background'))}"
        )
    if _scalar("personality"):
        sections.append(f"- Personality / voice cues: {_scalar('personality')}")
    if _list("anchors"):
        sections.append(f"- Known concepts (anchor new ideas here): {', '.join(_list('anchors'))}")
    if _list("stumbling_blocks"):
        sections.append(
            f"- Recurring stumbling blocks (slow down + re-anchor): "
            f"{', '.join(_list('stumbling_blocks'))}"
        )
    if _scalar("notes"):
        sections.append(f"- Notes: {_scalar('notes')}")

    if not sections:
        return ""

    body = "\n".join(sections)
    return (
        "\n\nLEARNER PROFILE (active course — consult on EVERY teaching turn):\n"
        f"{body}\n"
        "Voice, examples, analogies, exercise difficulty, and pace MUST reflect this profile. "
        "If you observe a new pattern (an analogy lands, a stumbling block surfaces, pace is "
        "wrong, etc.), call `update_learner_profile` with a one-line `observation` and the "
        "delta — the profile is a living document."
    )
