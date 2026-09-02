"""Headless UI for benchmark/CI one-shot runs.

Implements the small UI surface the Session + agent loop touch so mucli can
run a turn without a terminal: every interactive request fails closed
(approval denied, prompt aborted) unless ``approve_all`` is set, in which
case tool approvals are auto-granted (YOLO benchmarking).
"""

from __future__ import annotations


class HeadlessUI:
    """Minimal UI duck-type for headless one-shot runs."""

    def __init__(self, approve_all: bool = False):
        self.approve_all = approve_all
        self.notes: list[str] = []
        self.errors: list[str] = []

    # -- informational ----------------------------------------------------

    def show_info(self, message, *args, **kwargs):
        text = str(message)
        self.notes.append(text)
        return text

    def show_error(self, message, *args, **kwargs):
        text = str(message)
        self.errors.append(text)
        return text

    # -- interactive (fail closed) ----------------------------------------

    def prompt(self, *args, **kwargs):
        raise EOFError("headless: interactive prompt unavailable")

    def prompt_choices(self, *args, **kwargs):
        """Return the default choice instead of raising — provider-error
        recovery and tool-approval flows treat an interactive headless run
        as 'use the default' (retry), matching server-mode behavior."""
        return kwargs.get("default") or (args[2] if len(args) > 2 else "retry")

    def confirm(self, *args, **kwargs):
        return bool(self.approve_all)

    def request_tool_approval(
        self,
        *,
        tool_name: str = "",
        tool_args=None,
        display_args=None,
        count_info: str = "",
        can_approve: bool = True,
        modifications=None,
        preview_error=None,
        error_code=None,
        approval_policy=None,
        prompt_text: str = "",
        choices=None,
        default: str = "n",
        **kwargs,
    ):
        """Return an approval-shaped decision dict.

        Approves when approve_all (YOLO bench), else denies so write tools
        never execute unattended.
        """
        if self.approve_all and can_approve:
            return {"approved": True, "remember": True}
        return {"approved": False, "reason": "headless: approval denied"}

    # -- optional hooks the TUI provides (no-ops here) ----------------------

    import contextlib as _cl

    @_cl.contextmanager
    def show_status(self, message):
        """No-op context manager for the TUI status spinner."""
        yield self

    def build_live_status(self, *args, **kwargs):
        return "headless"

    def render_message(self, *args, **kwargs):
        return None

    def show_tool_result(self, *args, **kwargs):
        return None

    def show_diff(self, *args, **kwargs):
        return None

    def emit_tool_trace(self, *args, **kwargs):
        return None

    def set_variables(self, variables):  # TUI HUD sync
        return None

    def build_prompt_markup(self, *args, **kwargs):
        return ""

    def build_input_toolbar_text(self):
        return ""

    def build_choice_toolbar_text(self):
        return ""

    def get_input(self, *args, **kwargs):
        raise EOFError("headless: no interactive input")

    def show_info_markup(self, message, *args, **kwargs):
        return self.show_info(message, *args, **kwargs)

    def __getattr__(self, name):  # TUI-only methods degrade to no-ops
        def _noop(*args, **kwargs):
            return None

        return _noop