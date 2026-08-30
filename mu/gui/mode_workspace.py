"""Typed presentation contracts for MuCLI's mode workspaces.

The agent modes deliberately share a harness, but the GUI must not pretend
that their outputs mean the same thing.  This module is the adapter between
mode-owned state (task memory, scratchpad, feature plans, security reports,
and courses) and a small shared piece of UI chrome.

The contract only contains *presentation metadata*.  The underlying mode
payload remains the source of truth and keeps its native shape.  In
particular, no synthetic "accuracy percentage" is generated: when a mode has
not actually measured accuracy or relevance, the contract says so.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA_VERSION = 1


def _metric(
    metric_id: str,
    label: str,
    value: Any,
    *,
    tone: str = "neutral",
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "id": metric_id,
        "label": label,
        "value": value,
        "tone": tone,
        "detail": detail,
    }


def _view(view_id: str, label: str, count: Optional[int] = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {"id": view_id, "label": label}
    if count is not None:
        item["count"] = count
    return item


def _quality(
    quality_id: str,
    label: str,
    state: str,
    detail: str,
    *,
    value: Optional[Any] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "id": quality_id,
        "label": label,
        "state": state,
        "detail": detail,
    }
    if value is not None:
        result["value"] = value
    return result


def _workspace(
    mode: str,
    title: str,
    objective: str,
    *,
    status_label: str,
    status_tone: str,
    views: Sequence[Dict[str, Any]],
    metrics: Sequence[Dict[str, Any]],
    quality: Sequence[Dict[str, Any]],
    search_placeholder: str,
    provenance: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "title": title,
        "objective": objective,
        "status": {"label": status_label, "tone": status_tone},
        "views": list(views),
        "metrics": list(metrics),
        "quality": list(quality),
        "search_placeholder": search_placeholder,
        "provenance": provenance,
    }


def research_workspace(
    sources: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    *,
    active: bool = True,
) -> Dict[str, Any]:
    typed_claims = [
        finding for finding in findings if finding.get("record_type") == "claim"
    ]
    sourced_findings = sum(1 for finding in typed_claims if finding.get("source"))
    high_credibility = sum(
        1 for source in sources if float(source.get("credibility_score") or 0) >= 0.75
    )
    source_types = len(
        {str(source.get("source_type") or "unknown") for source in sources}
    )
    topics = sorted({str(source.get("topic") or "general") for source in sources})
    assessed = sum(
        1 for source in sources if float(source.get("credibility_score") or 0) > 0
    )
    return _workspace(
        "research",
        "Evidence desk",
        "Turn sources into traceable claims, expose coverage gaps, and keep source quality separate from claim correctness. Sources are grouped by research topic.",
        status_label="collecting" if active else "idle",
        status_tone="active" if active else "neutral",
        views=(
            _view("overview", "Brief"),
            _view("claims", "Findings", len(findings)),
            _view("sources", "Sources", len(sources)),
            _view("bibliography", "Citations", len(sources)),
        ),
        metrics=(
            _metric(
                "claims",
                "typed claims",
                len(typed_claims),
                detail="Research claims saved with typed research/claim tags.",
            ),
            _metric(
                "sourced",
                "source-linked",
                sourced_findings,
                tone="good" if sourced_findings else "neutral",
                detail="Typed claims with an explicit source reference.",
            ),
            _metric(
                "credible",
                "high-credibility",
                high_credibility,
                tone="good" if high_credibility else "neutral",
                detail="Sources scoring 0.75+ credibility — the threshold for citing a claim as well-evidenced.",
            ),
            _metric(
                "sources",
                "sources",
                len(sources),
                detail=f"Across {source_types} source type(s) and {len(topics)} topic(s).",
            ),
            _metric(
                "topics",
                "research topics",
                len(topics),
                detail="Distinct rabbit holes / sub-questions with at least one source.",
            ),
            _metric(
                "assessed",
                "AI-assessed",
                assessed,
                tone="good" if assessed else "neutral",
                detail="Sources the AI has graded via assess_source; unassessed score 0.0.",
            ),
        ),
        quality=(
            _quality(
                "accuracy",
                "Claim accuracy",
                "unassessed",
                "MuCLI has not fact-checked a finding merely because it is in memory.",
            ),
            _quality(
                "relevance",
                "Relevance",
                "topic-grouped",
                "Sources are grouped by research topic so the bibliography stays organized by ask.",
            ),
            _quality(
                "evidence",
                "Evidence quality",
                "ai-assessed",
                "Credibility is AI-assessed per source (assess_source), bounded by a per-type safety cap; unassessed sources score 0.0.",
            ),
        ),
        search_placeholder="Filter claims, sources, authors, topics, or tags",
        provenance="Citation ledger (topic-grouped) + research-mode task memory",
    )


def security_workspace(
    report: Optional[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    *,
    active: bool = True,
) -> Dict[str, Any]:
    proof_verified = sum(1 for finding in findings if finding.get("proof_verified"))
    fixes_verified = sum(
        1 for finding in findings if finding.get("remediation_verified")
    )
    critical_high = sum(
        1
        for finding in findings
        if str(finding.get("severity") or "").lower() in {"critical", "high"}
    )
    approved = sum(1 for finding in findings if finding.get("status") == "approved")
    title = str((report or {}).get("title") or "Verification bench")
    report_status = str((report or {}).get("status") or ("ready" if active else "idle"))
    return _workspace(
        "security",
        title,
        "Prioritise risk, inspect the exploit chain, and gate conclusions on reproducible proof and verified remediation.",
        status_label=report_status,
        status_tone="risk" if critical_high else ("active" if active else "neutral"),
        views=(
            _view("overview", "Risk"),
            _view("findings", "Findings", len(findings)),
            _view("evidence", "Proof", proof_verified),
            _view("remediation", "Fixes", fixes_verified),
        ),
        metrics=(
            _metric(
                "risk",
                "critical / high",
                critical_high,
                tone="risk" if critical_high else "good",
                detail="Findings requiring the fastest attention.",
            ),
            _metric(
                "proof",
                "proof verified",
                proof_verified,
                tone="good" if proof_verified else "neutral",
                detail="Findings with a reproducible verified proof.",
            ),
            _metric(
                "fixes",
                "fixes verified",
                fixes_verified,
                tone="good" if fixes_verified else "neutral",
                detail="Remediations with verification evidence.",
            ),
            _metric(
                "approved",
                "approved",
                approved,
                tone="good" if approved else "neutral",
                detail="Findings that passed the mode's approval gate.",
            ),
        ),
        quality=(
            _quality(
                "accuracy",
                "Finding accuracy",
                "verification-gated",
                "Approval requires both verified exploit proof and verified remediation.",
            ),
            _quality(
                "relevance",
                "Affected surface",
                "traceable",
                "Paths and exploit paths show where each finding applies; no generic relevance score is invented.",
            ),
            _quality(
                "evidence",
                "Evidence chain",
                "measured",
                "Proof and remediation verification are independently visible.",
            ),
        ),
        search_placeholder="Filter risks, paths, classes, or evidence",
        provenance="Security report, proof commands, and remediation records",
    )


def debug_workspace(
    target: str,
    hypotheses: Sequence[Mapping[str, Any]],
    suspects: Sequence[Mapping[str, Any]],
    notes: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    *,
    active: bool = True,
) -> Dict[str, Any]:
    supported = sum(
        1 for item in hypotheses if item.get("status") in {"supported", "confirmed"}
    )
    disproved = sum(1 for item in hypotheses if item.get("status") == "disproved")
    untested = sum(1 for item in hypotheses if item.get("status") == "untested")
    return _workspace(
        "debug",
        target or "Investigation board",
        "Move from symptom to root cause by testing competing hypotheses and preserving the evidence that changes belief.",
        status_label="investigating" if active else "idle",
        status_tone="active" if active else "neutral",
        views=(
            _view("overview", "Case"),
            _view("hypotheses", "Hypotheses", len(hypotheses)),
            _view("evidence", "Observations", len(notes) + len(suspects)),
            _view("findings", "Root causes", len(findings)),
        ),
        metrics=(
            _metric(
                "untested",
                "untested",
                untested,
                tone="warn" if untested else "neutral",
                detail="Hypotheses still needing a discriminating test.",
            ),
            _metric(
                "supported",
                "supported",
                supported,
                tone="good" if supported else "neutral",
                detail="Hypotheses marked supported or confirmed.",
            ),
            _metric(
                "disproved",
                "ruled out",
                disproved,
                detail="Hypotheses rejected by the investigation.",
            ),
            _metric(
                "suspects",
                "suspect sites",
                len(suspects),
                detail="Files, lines, or symbols under investigation.",
            ),
        ),
        quality=(
            _quality(
                "accuracy",
                "Root-cause accuracy",
                "unassessed",
                "A status tag records investigation state; it is not an independent accuracy measurement.",
            ),
            _quality(
                "relevance",
                "Diagnostic relevance",
                "case-linked",
                "Hypotheses and observations are scoped to the current debug target.",
            ),
            _quality(
                "evidence",
                "Test strength",
                "partially-structured",
                "Support and refutation are visible; explicit test artifacts are shown when captured as notes or sources.",
            ),
        ),
        search_placeholder="Filter hypotheses, symptoms, files, or observations",
        provenance="Debug scratchpad + durable root-cause memory",
    )


def loop_workspace(
    goal: str,
    backlog: Sequence[Mapping[str, Any]],
    features: Sequence[Any],
    memory: Sequence[Mapping[str, Any]],
    *,
    loop_active: bool,
    active: bool = True,
) -> Dict[str, Any]:
    completed = sum(1 for item in backlog if item.get("status") == "completed")
    in_progress = sum(1 for item in backlog if item.get("status") == "in_progress")
    blocked = sum(1 for item in backlog if item.get("status") == "blocked")
    return _workspace(
        "loop",
        goal or "Mission control",
        "Keep autonomous work oriented around a stable goal, visible queue, blockers, and checkable outcomes.",
        status_label="running" if loop_active else ("paused" if active else "idle"),
        status_tone="active" if loop_active else "neutral",
        views=(
            _view("overview", "Mission"),
            _view("backlog", "Queue", len(backlog)),
            _view("features", "Workstreams", len(features)),
            _view("memory", "Checkpoints", len(memory)),
        ),
        metrics=(
            _metric(
                "active",
                "in progress",
                in_progress,
                tone="active" if in_progress else "neutral",
                detail="Queue items currently being worked.",
            ),
            _metric(
                "completed",
                "completed",
                completed,
                tone="good" if completed else "neutral",
                detail="Queue items reported completed.",
            ),
            _metric(
                "blocked",
                "blocked",
                blocked,
                tone="risk" if blocked else "good",
                detail="Items that cannot currently advance.",
            ),
            _metric(
                "workstreams",
                "workstreams",
                len(features),
                detail="Features spawned from this loop.",
            ),
        ),
        quality=(
            _quality(
                "accuracy",
                "Outcome accuracy",
                "unassessed",
                "A completed queue status is not proof that the result is correct.",
            ),
            _quality(
                "relevance",
                "Goal relevance",
                "goal-scoped",
                "Queue and checkpoints are presented against the active loop goal.",
            ),
            _quality(
                "evidence",
                "Completion evidence",
                "partially-structured",
                "Verification notes may be stored in checkpoints; completion currently has no mandatory receipt.",
            ),
        ),
        search_placeholder="Filter queue, workstreams, blockers, or checkpoints",
        provenance="Loop variables + todo scratchpad + recent memory",
    )


def _plan_tasks(plan: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not plan:
        return []
    tasks: List[Mapping[str, Any]] = []
    for phase in plan.get("phase_columns") or []:
        if isinstance(phase, Mapping):
            tasks.extend(
                item for item in (phase.get("tasks") or []) if isinstance(item, Mapping)
            )
    return tasks


def feature_workspace(
    plan: Optional[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    *,
    active: bool = True,
) -> Dict[str, Any]:
    tasks = _plan_tasks(plan)
    completed = sum(1 for task in tasks if task.get("status") == "completed")
    blocked = sum(1 for task in tasks if task.get("status") == "blocked")
    criteria = sum(len(task.get("exit_criteria") or []) for task in tasks)
    verified = sum(len(task.get("verified_exit_criteria") or []) for task in tasks)
    title = str((plan or {}).get("feature_name") or "Delivery workspace")
    status = str((plan or {}).get("overall_status") or ("ready" if active else "idle"))
    return _workspace(
        "feature",
        title,
        str(
            (plan or {}).get("feature_request")
            or "Plan delivery, expose dependencies, and verify exit criteria before work is considered complete."
        ),
        status_label=status,
        status_tone="risk" if blocked else ("active" if plan else "neutral"),
        views=(
            _view("overview", "Plan"),
            _view("board", "Tasks", len(tasks)),
            _view("verification", "Criteria", criteria),
            _view("reviews", "Reviews", len((plan or {}).get("review_records") or [])),
        ),
        metrics=(
            _metric(
                "progress",
                "tasks complete",
                f"{completed}/{len(tasks)}",
                tone="good" if tasks and completed == len(tasks) else "active",
                detail="Completed tasks over all plan tasks.",
            ),
            _metric(
                "criteria",
                "criteria verified",
                f"{verified}/{criteria}",
                tone="good" if criteria and verified == criteria else "warn",
                detail="Explicitly checked exit criteria; not a proxy for code quality.",
            ),
            _metric(
                "blocked",
                "blocked",
                blocked,
                tone="risk" if blocked else "good",
                detail="Tasks stopped by a recorded blocker.",
            ),
            _metric(
                "plans",
                "available plans",
                len(features),
                detail="Active and archived feature plans.",
            ),
        ),
        quality=(
            _quality(
                "accuracy",
                "Implementation correctness",
                "unassessed",
                "Task state alone does not prove implementation correctness.",
            ),
            _quality(
                "relevance",
                "Scope relevance",
                "plan-linked",
                "Tasks and criteria are tied to the approved feature request.",
            ),
            _quality(
                "evidence",
                "Acceptance evidence",
                "measured",
                "Verified exit criteria and review records expose why work can advance.",
            ),
        ),
        search_placeholder="Filter tasks, criteria, blockers, or reviews",
        provenance="Feature plan engine + task events + review records",
    )


def teacher_workspace(
    course: Optional[Mapping[str, Any]],
    courses: Sequence[Mapping[str, Any]],
    *,
    active: bool = True,
) -> Dict[str, Any]:
    lessons = list((course or {}).get("lessons") or [])
    assignments = list((course or {}).get("assignments") or [])
    reviews = list((course or {}).get("scheduled_reviews") or [])
    completed = sum(1 for lesson in lessons if lesson.get("status") == "completed")
    graded = [
        item for item in assignments if isinstance(item, Mapping) and item.get("grade")
    ]
    scores = [
        float(item["grade"]["score_pct"])
        for item in graded
        if isinstance(item.get("grade"), Mapping)
        and item["grade"].get("score_pct") is not None
    ]
    avg_score: Any = "—" if not scores else f"{round(sum(scores) / len(scores))}%"
    title = str((course or {}).get("subject") or "Learning studio")
    status = str((course or {}).get("status") or ("ready" if active else "idle"))
    return _workspace(
        "teacher",
        title,
        "Make the learning path, demonstrated mastery, gaps, and next review visible without conflating completion with understanding.",
        status_label=status,
        status_tone="active" if course else "neutral",
        views=(
            _view("overview", "Path"),
            _view("curriculum", "Curriculum", len(lessons)),
            _view("mastery", "Mastery", len(graded)),
            _view("reviews", "Reviews", len(reviews)),
        ),
        metrics=(
            _metric(
                "lessons",
                "lessons complete",
                f"{completed}/{len(lessons)}",
                tone="good" if lessons and completed == len(lessons) else "active",
                detail="Course progression, not a mastery score.",
            ),
            _metric(
                "assignments",
                "graded work",
                len(graded),
                detail="Assignments with recorded grades.",
            ),
            _metric(
                "score",
                "graded average",
                avg_score,
                tone="good" if scores else "neutral",
                detail="Average of recorded assignment scores only.",
            ),
            _metric(
                "courses",
                "courses",
                len(courses),
                detail="Courses available in this session.",
            ),
        ),
        quality=(
            _quality(
                "accuracy",
                "Mastery estimate",
                "measured-where-tested",
                "Grades and comprehension checks are evidence; untested material remains unassessed.",
            ),
            _quality(
                "relevance",
                "Learner relevance",
                "profile-linked",
                "Curriculum and examples are tied to the learner profile and course target.",
            ),
            _quality(
                "evidence",
                "Learning evidence",
                "measured",
                "Assignments, rubrics, dialog checks, and scheduled reviews support the mastery view.",
            ),
        ),
        search_placeholder="Filter lessons, objectives, gaps, or assignments",
        provenance="Course engine + learner profile + grades + review schedule",
    )
