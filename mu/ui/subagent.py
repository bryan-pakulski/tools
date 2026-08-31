"""Sub-agent UI wrapper.

When the parent agent spawns a child, the child needs a UI so its tool
calls and progress messages reach the user's terminal — otherwise the
parent's REPL goes silent for the duration of the child run with no
feedback at all.

`SubagentUI` wraps a parent UI and forwards the subset of events that
make sense to surface, prefixing them with `[subagent d=N]` so the user
can tell at a glance which agent is talking. Methods that don't make
sense to bubble up (nested spinners, diff previews for YOLO tools,
duplicate panel renders, interactive prompts the child can't answer)
are silenced or replaced with a single-line info message.

Nesting: if the parent UI is itself a `SubagentUI` (a grandchild spawn),
we forward through to the *root* UI and use the most-nested depth as
the label, rather than stacking prefixes like `[subagent][subagent]`.
"""

from __future__ import annotations

from typing import Any, Optional

from mu.security.secret_paths import redact_secrets


def _extract_tool_name(text: str) -> Optional[str]:
    """Parse the tool name out of the 'Running tool: X(args)' UI format.

    The Session loop emits:
        f"🔨 Running tool: {tool_name}({shortened_args})"
    We pull out `tool_name` so the progress tracker can show it.
    Returns None if the line doesn't match the expected shape.
    """
    marker = "Running tool:"
    idx = text.find(marker)
    if idx < 0:
        return None
    rest = text[idx + len(marker):].lstrip()
    paren = rest.find("(")
    if paren < 0:
        candidate = rest.strip()
    else:
        candidate = rest[:paren].strip()
    return candidate or None


def _extract_tool_detail(text: str, tool_name: str) -> str:
    """Return the already-shortened display args for an action row.

    The agent loop shortens large ``content``/``diff`` values before this UI
    boundary. We additionally run the non-bypassable secret scrubber and cap
    the result so the activity drill-down remains safe and compact.
    """
    marker = f"{tool_name}("
    start = text.find(marker)
    if start < 0:
        return ""
    detail = text[start + len(marker):]
    if detail.endswith(")"):
        detail = detail[:-1]
    detail, _ = redact_secrets(detail.strip())
    return detail[:240]


class _NoopStatus:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class SubagentUI:
    def __init__(
        self,
        parent_ui: Any,
        depth: int = 1,
        *,
        tracker: Any = None,
        agent_id: Optional[str] = None,
    ):
        self._parent = parent_ui
        self._depth = max(1, int(depth))
        self._prefix = f"[subagent d={self._depth}]"
        # If a `SubagentProgressTracker` is attached, "🔨 Running tool: X"
        # messages get routed to the live panel instead of flooding the
        # parent's terminal with one line per child tool call. All other
        # messages still bubble up (errors, completion banners, etc.).
        self._tracker = tracker
        self._agent_id = agent_id

    # -------------------------------------------------------------- helpers

    def _root(self) -> Optional[Any]:
        """Walk up nested SubagentUI wrappers to the real UI."""
        cur = self._parent
        while isinstance(cur, SubagentUI):
            cur = cur._parent
        return cur

    def _emit(self, kind: str, message: str) -> None:
        """Forward a styled info/error/status message to the root UI.

        Builds a pre-styled ``Text`` with explicit style spans so that:
          * The ``[dim cyan]`` prefix renders as a Rich style, not literal text.
          * Literal square brackets in the message (e.g. ``[Collated: get_chunk]``
            or ``[subagent d=1]``) are preserved verbatim — ``Text`` bypasses
            markup parsing, unlike ``Text.from_markup()``.
        """
        from rich.text import Text

        root = self._root()
        if root is None:
            return
        styled = Text()
        styled.append(self._prefix, style="dim cyan")
        styled.append(f"   {message}")
        if kind == "error" and hasattr(root, "show_error"):
            root.show_error(styled)
        elif hasattr(root, "show_info"):
            root.show_info(styled)

    # -------------------------------------------------- forwarded surface

    def show_info(self, message: Any) -> None:
        text = str(message)
        # Suppress per-iteration token/cost chatter — the spawn-agent end
        # banner reports the cumulative total once at the end, which is
        # cleaner than N intermediate "Tokens: In X | Out Y" lines.
        if text.startswith("Tokens:") or text.startswith("Final session tokens:"):
            return
        # If a progress tracker is attached, the per-tool-call announcement
        # becomes a live-panel update instead of a terminal log line.
        if self._tracker is not None and self._agent_id is not None:
            tool_name = _extract_tool_name(text)
            if tool_name is not None:
                try:
                    self._tracker.update_tool(
                        self._agent_id,
                        tool_name,
                        _extract_tool_detail(text, tool_name),
                    )
                except Exception:
                    pass
                return
        self._emit("info", text)

    def show_error(self, message: Any) -> None:
        self._emit("error", str(message))

    def show_status(self, message: Any) -> _NoopStatus:
        """No nested Live spinners — surface as a one-line info message instead.

        Most session status messages are noise we want to suppress:
          * 'Generating (model) it N/M | ...' — per-iteration spinner text
            (fires on every iter, would flood the parent terminal).
          * '' / None — falls through when `build_live_status` is None or
            session is between status updates.
        """
        if message is None:
            return _NoopStatus()
        text = str(message).strip()
        if not text or text.startswith("Generating "):
            return _NoopStatus()
        self._emit("info", text)
        return _NoopStatus()

    def emit_tool_trace(self, *args: Any, **kwargs: Any) -> None:
        # The loop invokes this after a tool result exists. Complete the
        # matching action without bubbling full tool output into the parent.
        if self._tracker is None or self._agent_id is None or not args:
            return None
        tool_name = str(args[0] or "")
        raw_result = args[3] if len(args) > 3 else None
        if isinstance(raw_result, dict):
            ok = bool(raw_result.get("ok", not raw_result.get("error_code")))
        else:
            result_text = str(raw_result or "").lstrip().lower()
            ok = not (
                result_text.startswith(("error", "refused"))
                or '"ok": false' in result_text[:240]
                or "'ok': false" in result_text[:240]
            )
        try:
            self._tracker.complete_tool(self._agent_id, tool_name, ok=ok)
        except Exception:
            pass
        return None

    # -------------------------------------------------- silenced surface

    def show_diff(self, filename: Any, original: Any, modified: Any) -> None:
        # Subagent runs YOLO; the user already approved the spawn. Diff
        # previews would only add noise.
        return None

    def show_tool_result(self, result: Any) -> None:
        # Full tool-result panels would double the noise — the per-tool
        # "Running tool: X" line plus the final returned summary is enough.
        return None

    def render_message(self, role: str, content: Any, model_name: Optional[str] = None) -> None:
        # The child's final assistant text reaches the parent via the
        # spawn_agent tool result envelope — no need to also render a
        # panel for it inside the child's UI surface.
        return None

    # -------------------------------------------------- interactive (YOLO)

    def request_tool_approval(self, **kwargs: Any):
        if kwargs.get("approval_policy") == "always_human":
            root = self._root()
            if root is not None and hasattr(root, "request_tool_approval"):
                return root.request_tool_approval(**kwargs)
            # No human surface is attached; fail closed.
            return "n", "Human approval is unavailable."
        # Ordinary subagent writes remain covered by the spawn's YOLO
        # authorization; only always-human tools escape this path.
        return "y", None

    def confirm(self, message: Any, default: bool = True) -> bool:
        # Non-interactive: a child must not block forever on a confirmation.
        return False

    def prompt(self, message: Any, default: Optional[Any] = None) -> Optional[Any]:
        return default

    def prompt_choices(
        self,
        message: Any,
        choices: Any,
        default: Optional[Any] = None,
    ) -> Optional[Any]:
        """Choose 'abort' if it's available, else None.

        Returning `default` here was a foot-gun: the agent-loop recovery flow
        defaults to 'retry', which combined with a child whose provider
        keeps failing produced an infinite retry loop. A child has no human
        to escalate to — failing fast is the only correct answer.
        """
        try:
            if choices and "abort" in choices:
                return "abort"
        except (TypeError, ValueError):
            pass
        return None

    # -------------------------------------------------- variables / status line

    def set_variables(self, variables_dict: Any) -> None:
        # The child runs with its own variables; nothing to broadcast.
        return None

    def build_live_status(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return None

    def show_memory_monitor(self, session: Any) -> None:
        return None


__all__ = ["SubagentUI"]
