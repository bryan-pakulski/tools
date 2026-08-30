"""UI adapter for autonomous durable job execution.

The adapter never blocks on stdin or a browser. Human interactions become a
control-flow signal which the outer job runner persists as NEEDS_HUMAN. A later
control-plane response is persisted and consumed exactly once by the next
isolated worker attempt.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

from mu.ui.base import BaseUI
from mu.ui.exceptions import InteractionRequired

from .service import JobService


class JobUI(BaseUI):
    def __init__(self, service: JobService, job_id: str):
        self.service = service
        self.job_id = job_id
        self._variables: Dict[str, Any] = {}
        self._responses = self._load_pending_responses()

    def _event(self, event_type: str, *, payload: Dict[str, Any] | None = None, reason: str = "") -> None:
        self.service.store.append_event(
            self.job_id,
            event_type,
            reason=reason,
            payload=payload or {},
        )

    def _load_pending_responses(self):
        events = self.service.events(self.job_id)
        consumed = {
            int(event.payload.get("response_event_id"))
            for event in events
            if event.event_type == "interaction_response_consumed"
            and str(event.payload.get("response_event_id") or "").isdigit()
        }
        return [
            event for event in events
            if event.event_type == "interaction_response" and event.id not in consumed
        ]

    def _take_response(self, kind: str, *, tool_name: str = "") -> Dict[str, Any] | None:
        for index, event in enumerate(list(self._responses)):
            payload = dict(event.payload or {})
            response_kind = str(payload.get("kind") or "question")
            if kind == "approval_required":
                if response_kind != "approval_required":
                    continue
                target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
                target_tool = str(target.get("tool_name") or "")
                if tool_name and target_tool and target_tool != tool_name:
                    continue
            elif response_kind not in {kind, "question"}:
                continue
            self._responses.pop(index)
            # Round-41 F7: ATOMIC claim — two JobUI instances (GUI + worker
            # session) previously both popped the same in-memory response
            # and both consumed it, double-approving write approvals. The
            # store-level claim lets exactly one caller win; the loser
            # re-loads pending responses (the winner's consumption is now
            # visible) and keeps looking.
            if not self.service.store.claim_interaction_response(
                self.job_id, event.id, kind=kind, tool_name=tool_name
            ):
                self._responses = self._load_pending_responses()
                continue
            # The claim itself wrote the interaction_response_consumed
            # event — do NOT append a second one (double-count).
            return payload
        return None

    def render_message(self, role, content, model_name=None):
        text = str(content or "")
        self._event(
            "agent_message",
            payload={"role": str(role), "text": text[:24000], "model": model_name or ""},
        )

    def get_input(self, session_name, staged_files, agent_mode="default", current_task=None, feature_context=None):
        return ""

    def show_error(self, message):
        self._event("runtime_error", payload={"text": str(message or "")[:12000]})

    def show_info(self, message):
        self._event("runtime_info", payload={"text": str(message or "")[:8000]})

    @contextmanager
    def show_status(self, message):
        self._event("runtime_status", payload={"text": str(message or "")[:1000]})
        yield

    def show_tool_result(self, result_str):
        self._event("tool_result_ui", payload={"preview": str(result_str or "")[:8000]})

    def show_diff(self, filename, original_content, modified_content):
        self._event(
            "approval_diff",
            payload={
                "filename": str(filename or ""),
                "original_chars": len(str(original_content or "")),
                "modified_chars": len(str(modified_content or "")),
            },
        )

    def set_variables(self, variables_dict):
        self._variables = variables_dict or {}

    def stream_tool_call(self, tool_name: str):
        self._event("tool_call_ui", payload={"tool_name": str(tool_name or "")})

    def request_tool_approval(self, **kwargs):
        tool_name = str(kwargs.get("tool_name") or "tool")
        response = self._take_response("approval_required", tool_name=tool_name)
        if response is not None:
            decision = str(response.get("decision") or "").lower()
            detail = str(response.get("detail") or "").strip() or None
            if decision == "approve":
                return "y", None
            if decision == "explain":
                return "e", detail
            return "n", None
        raise InteractionRequired(
            "approval_required",
            f"Approval required for {tool_name}",
            payload={
                "tool_name": tool_name,
                "tool_args": kwargs.get("tool_args") or kwargs.get("display_args") or {},
                "can_approve": bool(kwargs.get("can_approve", True)),
                "preview_error": kwargs.get("preview_error"),
                "error_code": kwargs.get("error_code"),
                "modifications": kwargs.get("modifications") or [],
            },
        )

    def prompt(self, message, default=None):
        response = self._take_response("question")
        if response is not None:
            if response.get("value") is not None:
                return response.get("value")
            detail = str(response.get("detail") or "").strip()
            return detail if detail else default
        raise InteractionRequired(
            "question",
            str(message or "Input required"),
            payload={"shape": "input", "default": default},
        )

    def confirm(self, message, default=True):
        response = self._take_response("question")
        if response is not None:
            value = response.get("value")
            if isinstance(value, bool):
                return value
            decision = str(response.get("decision") or "").lower()
            if decision in {"approve", "yes", "confirm", "true"}:
                return True
            if decision in {"deny", "no", "false"}:
                return False
            return bool(default)
        raise InteractionRequired(
            "question",
            str(message or "Confirmation required"),
            payload={"shape": "confirm", "default": bool(default)},
        )

    def prompt_choices(self, message, choices, default=None):
        response = self._take_response("question")
        if response is not None:
            if response.get("value") is not None:
                return response.get("value")
            selected = list(response.get("selected") or [])
            if selected:
                return selected[0]
            detail = str(response.get("detail") or "").strip()
            return detail or default
        raise InteractionRequired(
            "question",
            str(message or "Choice required"),
            payload={"shape": "choices", "choices": list(choices or []), "default": default},
        )

    def run_quiz(self, questions):
        response = self._take_response("question")
        if response is not None and isinstance(response.get("value"), dict):
            return dict(response["value"])
        raise InteractionRequired(
            "question",
            "Quiz response required",
            payload={"shape": "quiz", "questions": list(questions or [])},
        )

    def ask_user_choice(self, question, options, *, multi_select=False, description="", allow_other=False):
        response = self._take_response("question")
        if response is not None:
            selected = list(response.get("selected") or [])
            if not selected and response.get("value") is not None:
                selected = [response.get("value")]
            return {
                "selected": selected,
                "other_text": str(response.get("detail") or ""),
                "cancelled": False,
            }
        raise InteractionRequired(
            "question",
            str(question or "Choice required"),
            payload={
                "shape": "choice",
                "options": list(options or []),
                "multi_select": bool(multi_select),
                "description": str(description or ""),
                "allow_other": bool(allow_other),
            },
        )
