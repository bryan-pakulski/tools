"""Execute one durable job attempt through the existing MuCLI Session runtime."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from mu.session.manager import RevisionConflict
from mu.ui.exceptions import InteractionRequired
from utils.model_pricing import estimate_model_cost

from .models import AttentionReason, Job, JobAttempt
from .service import JobService
from .ui import JobUI


@dataclass
class JobRunOutcome:
    kind: str
    status: str = ""
    error: str = ""
    cost_usd: float = 0.0
    attention_reason: AttentionReason = AttentionReason.NONE
    attention_detail: str = ""
    attention_payload: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)


_TOKEN_KEYS = ("input", "output", "total", "cached", "reasoning")


class SessionJobRunner:
    def __init__(self, service: JobService, *, build_session_fn: Callable, base_args: Any):
        self.service = service
        self.build_session_fn = build_session_fn
        self.base_args = base_args

    @staticmethod
    def session_name(job: Job) -> str:
        return f"job-{job.id[:20]}"

    @staticmethod
    def workspace_path(job: Job) -> str:
        return str(job.worktree or job.repository or "")

    @staticmethod
    def _token_snapshot(session) -> Dict[str, float]:
        counts = getattr(getattr(session, "session_manager", None), "token_counts", {}) or {}
        value: Dict[str, float] = {
            key: float(counts.get(key, 0) or 0) for key in _TOKEN_KEYS
        }
        value["total_cost"] = float(counts.get("total_cost", 0.0) or 0.0)
        return value

    @staticmethod
    def _token_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, int]:
        return {
            key: max(0, int(round(float(after.get(key, 0)) - float(before.get(key, 0)))))
            for key in _TOKEN_KEYS
        }

    def _usage_result(
        self,
        job: Job,
        session,
        before: Dict[str, float],
        result: Dict[str, Any] | None = None,
    ) -> tuple[float, Dict[str, Any]]:
        """Return authoritative attempt API cost + persistence-ready result.

        The inner ReAct loop historically priced only a small Gemini map. A
        durable engineering job instead recomputes its attempt from actual
        provider token deltas and the versioned pricing registry. The pricing
        key/rates/version are persisted with the attempt so historical job
        economics remain explainable when list prices change later.
        """
        after = self._token_snapshot(session)
        tokens = self._token_delta(before, after)
        execution = dict(job.execution or {})
        provider_obj = getattr(session, "provider", None)
        provider_name = str(
            execution.get("provider") or getattr(provider_obj, "name", "") or ""
        )
        model_name = str(
            execution.get("model") or getattr(provider_obj, "model_name", "") or ""
        )
        endpoint = str(
            getattr(provider_obj, "host", "")
            or getattr(provider_obj, "BASE_URL", "")
            or ""
        )
        pricing = estimate_model_cost(
            provider=provider_name,
            model_name=model_name,
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            cached_tokens=tokens["cached"],
            reasoning_tokens=tokens["reasoning"],
            ollama_mode=str(session.variables.get("ollama_mode", "") or ""),
            endpoint=endpoint,
        )
        mapped_cost = pricing.get("api_cost_usd")
        legacy_delta = max(
            0.0,
            float(after.get("total_cost", 0.0)) - float(before.get("total_cost", 0.0)),
        )
        attributable_cost = (
            max(0.0, float(mapped_cost))
            if mapped_cost is not None
            else legacy_delta if legacy_delta > 0 else 0.0
        )
        if mapped_cost is not None:
            session.session_manager.token_counts["total_cost"] = (
                float(before.get("total_cost", 0.0)) + attributable_cost
            )
        output = dict(result or {})
        output["tokens"] = tokens
        output["cost"] = {
            **pricing,
            "api_cost_usd": mapped_cost,
            "attributed_cost_usd": attributable_cost,
            "legacy_loop_cost_usd": legacy_delta,
        }
        return attributable_cost, output

    def _args_for(self, job: Job):
        execution = dict(job.execution or {})
        provider = str(execution.get("provider") or "").strip()
        model = str(execution.get("model") or "").strip()
        if not provider or not model:
            raise InteractionRequired(
                "question",
                "This job needs a provider and model before it can run.",
                payload={"shape": "execution_profile"},
            )
        args = copy.copy(self.base_args)
        args.session = self.session_name(job)
        args.provider = provider
        args.model = model
        args.provider_prevalidated = True
        args.session_type = str(execution.get("session_type") or "workspace")
        workspace = self.workspace_path(job)
        args.workspace = [workspace] if workspace and args.session_type == "workspace" else []
        args.yolo = bool(execution.get("auto_approve_writes", False))
        args.gui = False
        # Durable jobs need the same provider/iteration/context telemetry as
        # interactive sessions so retrospective Job Trace can drill into the
        # actual agent loop rather than only coarse controller events.
        args.trace = True
        return args

    @staticmethod
    def _verification_feedback(payload: Dict[str, Any]) -> list[str]:
        lines = ["", "The previous implementation failed deterministic verification."]
        verification_id = str(payload.get("verification_id") or "")
        if verification_id:
            lines.append(f"Verification: {verification_id}")
        for check in list(payload.get("failed_checks") or [])[:4]:
            if not isinstance(check, dict):
                continue
            command = str(check.get("command") or "verification command")
            lines.append(f"- FAILED: {command}")
            if check.get("timed_out"):
                lines.append("  timed out")
            error = str(check.get("error") or "").strip()
            if error:
                lines.append(f"  error: {error[:1200]}")
            output = str(check.get("stderr") or check.get("stdout") or "").strip()
            if output:
                lines.append("  output:")
                lines.append(output[-2500:])
        dirty = str(payload.get("dirty_status") or "").strip()
        if dirty:
            lines.extend(["- Verification left the worktree dirty:", dirty[-2000:]])
        lines.append("Repair the implementation in the existing job branch and re-run relevant checks.")
        return lines

    def _prompt(self, job: Job) -> str:
        lines = ["DURABLE ENGINEERING JOB", f"Title: {job.title}"]
        if job.description:
            lines.extend(["", "Description:", job.description])
        if job.acceptance_criteria:
            lines.extend(["", "Acceptance criteria:", *[f"- {v}" for v in job.acceptance_criteria]])
        if job.validation_commands:
            lines.extend(["", "Validation expected by the controller:", *[f"- {v}" for v in job.validation_commands]])
        if job.branch:
            lines.extend(["", f"Managed job branch: {job.branch}"])

        events = self.service.events(job.id)
        for event in reversed(events):
            if event.event_type == "verification_failed":
                lines.extend(self._verification_feedback(event.payload))
                break
        for event in reversed(events):
            if event.event_type == "human_response":
                detail = str(event.payload.get("detail") or "").strip()
                if detail:
                    lines.extend(["", "Latest human response:", detail])
                break

        lines.extend([
            "",
            "Implement the ticket and validate the result where possible.",
            "Work only inside the attached job workspace; do not modify the user's primary checkout.",
            "The controller, not the agent, decides whether the job is ready for review.",
        ])
        return "\n".join(lines)

    def run(self, job: Job, attempt: JobAttempt) -> JobRunOutcome:
        session = None
        initial_usage: Dict[str, float] = {
            key: 0.0 for key in (*_TOKEN_KEYS, "total_cost")
        }
        try:
            execution = dict(job.execution or {})
            session_type = str(execution.get("session_type") or "workspace")
            workspace = self.workspace_path(job)
            if session_type == "workspace":
                if not workspace:
                    raise InteractionRequired(
                        "question",
                        "This job needs a repository/workspace path.",
                        payload={"shape": "repository"},
                    )
                if not os.path.isdir(os.path.expanduser(workspace)):
                    return JobRunOutcome(
                        kind="failed",
                        status="environment_error",
                        error=f"Job workspace does not exist: {workspace}",
                    )
            if session_type == "container":
                return JobRunOutcome(
                    kind="needs_human",
                    status="needs_human",
                    attention_reason=AttentionReason.ENVIRONMENT_FAILURE,
                    attention_detail="Container-backed durable jobs need the per-job container adapter before autonomous execution is safe.",
                    attention_payload={"session_type": "container"},
                )

            ui = JobUI(self.service, job.id)
            session = self.build_session_fn(self._args_for(job), ui, allow_prompt=False)
            session.ui = ui
            session.session_manager.ui = ui
            ui.set_variables(session.variables)
            session.variables["agent_mode"] = str(execution.get("agent_mode") or "default")
            session.variables["session_type"] = session_type
            session.variables["yolo"] = bool(execution.get("auto_approve_writes", False))
            session.variables["durable_job_id"] = job.id
            session.variables["durable_job_attempt"] = attempt.number
            session.variables["durable_job_branch"] = job.branch
            session.variables["durable_job_base_sha"] = job.base_sha
            if job.max_iterations is not None:
                session.variables["max_iterations"] = int(job.max_iterations)
            # Round-16 F18: durable-job sessions live in the same sessions
            # directory as user sessions — saving without expected_revision
            # selects LWW and can silently overwrite a concurrent GUI/CLI
            # write to this session document. CAS against the revision we
            # loaded; on conflict SKIP this pre-run save (the job variables
            # are already in memory for this run; the finally-block save
            # persists them) instead of clobbering the newer document.
            expected = int(
                getattr(session.session_manager, "revision", 0) or 0
            )
            try:
                session.session_manager.save_history(
                    session.folder_context, expected_revision=expected
                )
            except RevisionConflict as exc:
                logging.getLogger("mucli").warning(
                    "Job %s: session revision conflict on initial save "
                    "(disk=%s); skipping pre-run save",
                    job.id, getattr(exc, "current", "?"),
                )
            self.service.store.update_runtime_fields(
                job.id, session_name=self.session_name(job)
            )

            initial_usage = self._token_snapshot(session)
            raw_result = session.send_message(self._prompt(job)) or {}
            base_result = dict(raw_result) if isinstance(raw_result, dict) else {}
            cost, result = self._usage_result(job, session, initial_usage, base_result)
            status = str(result.get("status") or "completed")
            error = str(result.get("error") or "")
            if status == "completed":
                return JobRunOutcome(
                    kind="completed", status=status, cost_usd=cost, result=result
                )
            return JobRunOutcome(
                kind="failed",
                status=status,
                error=error or f"Agent stopped with status {status}",
                cost_usd=cost,
                result=result,
            )

        except InteractionRequired as gate:
            cost = 0.0
            result: Dict[str, Any] = {}
            if session is not None:
                cost, result = self._usage_result(job, session, initial_usage)
            reason = (
                AttentionReason.APPROVAL_REQUIRED
                if gate.kind == "approval_required"
                else AttentionReason.QUESTION
            )
            return JobRunOutcome(
                kind="needs_human",
                status="needs_human",
                cost_usd=cost,
                attention_reason=reason,
                attention_detail=gate.detail,
                attention_payload=gate.payload,
                result=result,
            )
        except Exception as exc:
            cost = 0.0
            result: Dict[str, Any] = {}
            if session is not None:
                cost, result = self._usage_result(job, session, initial_usage)
            return JobRunOutcome(
                kind="failed", status="error", error=str(exc), cost_usd=cost, result=result
            )
        finally:
            if session is not None:
                try:
                    # Round-16 F18: CAS the final save too — same
                    # last-writer-wins hazard as the pre-run save; on
                    # conflict, the concurrent writer's newer document
                    # wins and we log instead of clobbering it.
                    expected = int(
                        getattr(session.session_manager, "revision", 0) or 0
                    )
                    session.session_manager.save_history(
                        session.folder_context, expected_revision=expected
                    )
                except RevisionConflict as exc:
                    logging.getLogger("mucli").warning(
                        "Job %s: session revision conflict on final save "
                        "(disk=%s); concurrent writer wins",
                        job.id, getattr(exc, "current", "?"),
                    )
                except Exception:
                    pass
                try:
                    session.shutdown()
                except Exception:
                    pass
