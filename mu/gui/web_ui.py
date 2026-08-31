"""``WebUI`` — :class:`BaseUI` adapter that streams agent events to the bus.

Renders nothing locally; every UI side-effect becomes an event the
browser receives via SSE. Blocking prompts pause the agent thread on
a :class:`threading.Event` held by :class:`PromptStore`.

Three non-obvious bits:

- :meth:`stream_assistant_delta` lazily emits ``assistant_start`` on
  the first token because ``mu.ui.stream.build_default_renderer``
  doesn't probe for a start callback. Without this the frontend never
  sees a "new bubble" signal.
- :meth:`stream_assistant_end` is a no-op when no delta arrived (e.g.
  provider erred before any text), so there's no orphan bubble close.
- **Delta coalescing.** Providers stream per-token (Ollama from a local
  daemon is the worst case — hundreds of chunks/sec, no API rate limit),
  and each delta used to be its own ``publish_threadsafe`` → its own
  event-loop callback + its own SSE chunk. A long generation flooded the
  event loop with per-token publishes and *starved every other HTTP
  request* (loading traces, navigating sessions, reopening the GUI)
  while the existing SSE stream kept trickpling — the GUI looked frozen
  even though the live turn kept updating. :meth:`_flush_deltas` now
  buffers ``assistant_delta``/``thinking_delta`` text on the agent
  thread and publishes at most ~one batch every
  :data:`_DELTAS_FLUSH_MS` (plus a size cap and a forced flush on every
  boundary event). The frontend already concatenates deltas and
  re-renders on a rAF, so batching is invisible to the user — but it
  cuts event-loop publishes 10-50× and unblocks the rest of the server.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

from mu.ui.base import BaseUI

from .bus import EventBus
from .prompts import PromptStore


# Delta coalescing thresholds. Flush a batch at most every this many
# milliseconds (so ≤ ~20 publishes/sec regardless of provider chunk
# rate), or when the buffered text exceeds this many characters, or
# immediately on any boundary (tool_call / assistant_end / any non-delta
# event, which routes through the flush-first :meth:`_publish`).
_DELTAS_FLUSH_MS = 50.0
_DELTAS_MAX_CHARS = 8192


class WebUI(BaseUI):
    def __init__(
        self,
        bus: EventBus,
        prompts: PromptStore,
        *,
        session_name: Optional[str] = None,
    ):
        self._bus = bus
        self._prompts = prompts
        self._variables: Dict[str, Any] = {}
        self._current_turn_id: Optional[str] = None
        # The session this WebUI belongs to. Stamped onto every event
        # so the frontend can route streaming/prompt traffic to the
        # right per-session chat slot when multiple sessions are loaded.
        self.session_name: Optional[str] = session_name
        # Delta coalescer state. Only the parent agent thread appends
        # (subagents use a SubagentUI that doesn't override the streaming
        # hooks), but the lock guards the buffer against any future
        # cross-thread caller and costs nothing uncontended.
        self._delta_lock = threading.Lock()
        self._asst_buf: list[str] = []
        self._think_buf: list[str] = []
        self._last_delta_flush = 0.0
        # Mirrors RichUI._streamed_any_text: when True, the assistant text
        # was already delivered token-by-token via assistant_delta events,
        # so the post-stream render_message("assistant", full_text) call in
        # loop_body must be suppressed to avoid a duplicate bubble.
        self._streamed_any_text = False

    def _publish_raw(self, event: Dict[str, Any]) -> None:
        """Low-level publish — stamps the session name and hands off to
        the bus. Does NOT flush buffered deltas, so the coalescer can
        publish batches without recursing through :meth:`_publish`."""
        if self.session_name is not None and "session_name" not in event:
            event = {**event, "session_name": self.session_name}
        self._bus.publish_threadsafe(event)

    def _publish(self, event: Dict[str, Any]) -> None:
        """Publish a non-delta event, draining any buffered deltas first
        so ordering is preserved (the buffered text always precedes the
        boundary / status / error / tool-result event)."""
        self._flush_deltas(force=True)
        self._publish_raw(event)

    def publish_event(self, event: Dict[str, Any]) -> None:
        """Public event transport shared with the container worker bridge."""
        self._publish(event)

    def _flush_deltas(self, *, force: bool = False) -> None:
        """Publish + clear the buffered assistant/thinking deltas if the
        time or size threshold is met (or ``force``). One batched
        ``assistant_delta`` and one ``thinking_delta`` per flush, so the
        event loop sees ≤ ~2 publishes per flush window instead of one
        per provider chunk."""
        now = time.monotonic()
        with self._delta_lock:
            size = sum(len(s) for s in self._asst_buf) + sum(
                len(s) for s in self._think_buf
            )
            if not (
                force
                or size >= _DELTAS_MAX_CHARS
                or (now - self._last_delta_flush) * 1000.0 >= _DELTAS_FLUSH_MS
            ):
                return
            asst = "".join(self._asst_buf)
            self._asst_buf.clear()
            think = "".join(self._think_buf)
            self._think_buf.clear()
            self._last_delta_flush = now
        turn_id = self._current_turn_id
        # Thinking first so it reads above the assistant bubble in the
        # trace; the two render in separate frontend regions, so the
        # inter-kind order within one flush window is cosmetic.
        if think:
            self._publish_raw(
                {"kind": "thinking_delta", "turn_id": turn_id, "text": think}
            )
        if asst:
            self._publish_raw(
                {"kind": "assistant_delta", "turn_id": turn_id, "text": asst}
            )

    def _new_turn(self) -> str:
        self._current_turn_id = uuid.uuid4().hex[:12]
        return self._current_turn_id

    # --- BaseUI surface ---------------------------------------------------

    def render_message(self, role, content, model_name=None):
        # Suppress the duplicate assistant bubble: loop_body calls
        # render_message("assistant", full_text) right after the streaming
        # loop delivers the same text token-by-token. If we already streamed
        # it, skip. _streamed_any_text resets on the next stream start.
        if role == "assistant" and self._streamed_any_text:
            return
        self._publish(
            {
                "kind": "message",
                "role": role,
                "content": str(content),
                "model": model_name,
            }
        )

    def get_input(
        self,
        session_name,
        staged_files,
        agent_mode="default",
        current_task=None,
        feature_context=None,
    ):
        # Browser drives input via POST /api/chat/send. Stub for interface.
        return ""

    def show_error(self, message):
        self._publish({"kind": "error", "text": str(message)})

    def show_info(self, message):
        self._publish({"kind": "info", "text": str(message)})

    def show_status(self, message):
        return _NullStatus(self, str(message))

    def show_tool_result(self, result_str):
        self._publish({"kind": "tool_result", "text": str(result_str)})

    # --- streaming hooks (duck-typed; see mu/ui/stream.py) ----------------

    def stream_assistant_delta(self, text: str):
        if not text:
            return
        if self._current_turn_id is None:
            # New turn — reset the dedup flag from the previous turn so
            # history replay / non-streaming callers still render normally.
            self._streamed_any_text = False
            self._new_turn()
            self._publish_raw(
                {"kind": "assistant_start", "turn_id": self._current_turn_id}
            )
        self._streamed_any_text = True
        with self._delta_lock:
            self._asst_buf.append(text)
        self._flush_deltas()

    def stream_thinking_delta(self, text: str):
        if not text:
            return
        if self._current_turn_id is None:
            # Thinking before any assistant text (reasoning models) — no
            # turn bubble yet, so publish immediately the way we always
            # did; the frontend renders it on the trace, not a turn.
            self._publish_raw(
                {"kind": "thinking_delta", "turn_id": None, "text": text}
            )
            return
        with self._delta_lock:
            self._think_buf.append(text)
        self._flush_deltas()

    def stream_tool_call(self, tool_name: str):
        # _publish flushes any buffered deltas first so the tool-call
        # marker lands after the text that preceded it.
        self._publish(
            {
                "kind": "tool_call",
                "turn_id": self._current_turn_id,
                "tool_name": tool_name,
            }
        )

    def stream_assistant_end(self):
        if self._current_turn_id is None:
            return  # No deltas this turn — nothing to close.
        self._publish({"kind": "assistant_end", "turn_id": self._current_turn_id})
        self._current_turn_id = None
        # _streamed_any_text stays True so the immediately-following
        # render_message("assistant", full_text) in loop_body is suppressed.
        # It resets on the next stream_assistant_delta (new turn).

    def set_variables(self, variables_dict):
        self._variables = dict(variables_dict or {})

    # --- blocking prompts -------------------------------------------------

    def prompt(self, message, default=None):
        result = self._ask_prompt(
            {
                "shape": "input",
                "message": str(message),
                "default": "" if default is None else str(default),
            }
        )
        if isinstance(result, dict) and result.get("cancelled"):
            return default
        if isinstance(result, dict):
            return result.get("value", default)
        return default

    def confirm(self, message, default=True):
        result = self._ask_prompt(
            {
                "shape": "confirm",
                "message": str(message),
                "default": bool(default),
            }
        )
        if isinstance(result, dict) and result.get("cancelled"):
            return bool(default)
        if isinstance(result, dict) and "value" in result:
            return bool(result["value"])
        return bool(default)

    def prompt_choices(self, message, choices, default=None):
        result = self._ask_prompt(
            {
                "shape": "choices",
                "message": str(message),
                "choices": list(choices),
                "default": default,
            }
        )
        if isinstance(result, dict) and result.get("cancelled"):
            return default
        if isinstance(result, dict):
            return result.get("value", default)
        return default

    def request_tool_approval(
        self,
        tool_name=None,
        tool_args=None,
        *,
        description=None,
        risk=None,
        display_args=None,
        approval_policy="default",
        **_kwargs,
    ):
        result = self._ask_prompt(
            {
                "shape": "tool_approval",
                "tool_name": tool_name,
                "tool_args": tool_args or display_args,
                "description": description,
                "risk": risk,
                "approval_policy": approval_policy,
            }
        )
        if isinstance(result, dict) and result.get("cancelled"):
            return {"approved": False, "remember": False}
        if isinstance(result, dict):
            return {
                "approved": bool(result.get("approved", False)),
                "remember": bool(result.get("remember", False)),
            }
        return {"approved": False, "remember": False}

    def run_quiz(self, questions):
        result = self._ask_prompt(
            {"shape": "quiz", "questions": list(questions or [])}
        )
        if isinstance(result, dict) and result.get("cancelled"):
            return {}
        if isinstance(result, dict):
            return dict(result.get("answers") or {})
        return {}

    def ask_user_choice(
        self,
        question,
        options,
        *,
        multi_select=False,
        description="",
        allow_other=False,
    ):
        result = self._ask_prompt(
            {
                "shape": "choice",
                "question": str(question),
                "options": list(options),
                "multi_select": bool(multi_select),
                "description": str(description or ""),
                "allow_other": bool(allow_other),
            }
        )
        if isinstance(result, dict) and result.get("cancelled"):
            return {"selected": [], "other_text": "", "cancelled": True}
        if isinstance(result, dict):
            return {
                "selected": list(result.get("selected") or []),
                "other_text": str(result.get("other_text") or ""),
                "cancelled": False,
            }
        return {"selected": [], "other_text": "", "cancelled": True}

    def show_diff(self, filename, original_content, new_content):
        self._publish(
            {
                "kind": "diff",
                "filename": str(filename),
                "original": str(original_content or ""),
                "new": str(new_content or ""),
            }
        )

    # --- prompt plumbing --------------------------------------------------

    def _ask_prompt(self, payload: Dict[str, Any], timeout: float = 600.0) -> Any:
        # Tag the prompt payload itself so it's discoverable later
        # via /api/prompts (for reconnection / debugging).
        tagged_payload = dict(payload)
        if self.session_name is not None:
            tagged_payload.setdefault("session_name", self.session_name)
        prompt_id, event = self._prompts.open(tagged_payload)
        self._publish({"kind": "prompt", "id": prompt_id, "prompt": tagged_payload})
        if not event.wait(timeout=timeout):
            self._prompts.cancel(prompt_id)
            self._publish({"kind": "prompt_cancelled", "id": prompt_id})
        result = self._prompts.take(prompt_id)
        self._publish({"kind": "prompt_resolved", "id": prompt_id})
        return result


class _NullStatus:
    """Context manager surrogate for `show_status` — emits start/end events."""

    def __init__(self, ui: WebUI, message: str):
        self._ui = ui
        self._message = message

    def __enter__(self):
        self._ui._publish({"kind": "status_start", "text": self._message})
        return self

    def __exit__(self, exc_type, exc, tb):
        self._ui._publish({"kind": "status_end", "text": self._message})
        return False

    def update(self, message: str) -> None:
        self._message = str(message)
        self._ui._publish({"kind": "status_update", "text": self._message})
