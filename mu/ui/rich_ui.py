from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import select
import sys
import threading
import time

from utils.threads import NamedThread
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .input import InputHandler
from .render import render_response
from utils.config import AGENT_MODE_METADATA
from utils.helpers import safe_markup
from utils.runtime_metrics import build_live_status_line, collect_runtime_metrics


class RichUI:
    def __init__(self):
        self.console = Console()
        self.input_handler = InputHandler()
        # Per-turn streaming state. `_gen_live` is set when an active
        # `_GenerationLive` context manager is open — streaming-text and
        # thinking deltas append into that Live's buffer so the Rich Live
        # region (not bare `console.print`) handles cursor placement.
        # `_streamed_any_text` survives until the next stream starts so
        # `render_message` can suppress the duplicate panel.
        self._gen_live = None
        self._streamed_any_text = False
        self._variables = None  # set via set_variables()

    # -------------------------------------------------- live streaming surface

    def _streaming_enabled(self) -> bool:
        """Honor `variables["streaming_enabled"]` (default True)."""
        if self._variables is None:
            return True
        try:
            return bool(self._variables.get("streaming_enabled", True))
        except AttributeError:
            return True

    def stream_assistant_start(self, model_name=None):
        """No-op: header rendering is owned by the active `_GenerationLive`.
        Kept for API compatibility."""
        return None

    def stream_assistant_delta(self, text: str):
        """Append a token chunk to the active generation Live, if any."""
        if not self._streaming_enabled() or not text:
            return
        if self._gen_live is not None:
            self._gen_live.append_text(text)
            self._streamed_any_text = True
            return
        # No active Live — happens in tests or non-streaming paths. Fall
        # back to a direct print so the text isn't lost.
        self.console.print(text, end="", soft_wrap=True, highlight=False, markup=False)
        self._streamed_any_text = True

    def stream_thinking_delta(self, text: str):
        """Append a reasoning chunk to the active generation Live, styled
        in dim italic so it visually separates from user-facing text.

        Gated on `show_thinking` (default True), with one mode-aware
        wrinkle: teacher mode hides thinking by default unless the user
        has explicitly toggled `/show-thinking on` (tracked via
        `show_thinking_explicit`). The model still GENERATES reasoning
        when `session.thinking=True` — this gate only controls rendering.
        """
        if not self._streaming_enabled() or not text:
            return
        if not self._show_thinking_effective():
            return
        if self._gen_live is not None:
            self._gen_live.append_thinking(text)
            return
        self.console.print(
            Text(text, style="dim italic"), end="", soft_wrap=True, highlight=False
        )

    def _show_thinking_effective(self) -> bool:
        """Resolve `show_thinking` against per-mode defaults.

        Teacher mode hides thinking by default — the lecture cadence
        works better without dim-italic reasoning blocks competing with
        agent_explanation turns for the learner's attention. Any other
        mode falls back to the stored `show_thinking` value.

        The user's `/show-thinking` toggle wins: once they set the
        explicit flag, mode-based overrides stop applying.
        """
        variables = self._variables or {}
        explicit = bool(variables.get("show_thinking_explicit", False))
        stored = bool(variables.get("show_thinking", True))
        if explicit:
            return stored
        mode = str(variables.get("agent_mode", "default") or "default").lower()
        if mode == "teacher":
            return False
        return stored

    def stream_tool_call(self, tool_name: str):
        """Note a tool call inside the generation Live's text region.
        Outside an active Live, fall back to a one-line print."""
        if not self._streaming_enabled() or not tool_name:
            return
        if self._gen_live is not None:
            self._gen_live.note_tool_call(tool_name)
            return
        self.console.print(
            Text.assemble(("\n→ ", "cyan"), (str(tool_name), "cyan")),
            highlight=False,
        )

    def stream_assistant_end(self):
        """End-of-stream notification. The Live keeps its rendered text in
        scrollback (transient=False) when its CM exits, so we don't have to
        print anything here. `_streamed_any_text` is deliberately preserved
        so `render_message` can suppress the duplicate panel; it resets on
        the next Live start."""
        return None

    # -------------------------------------------------- render_message

    def render_message(self, role, content, model_name=None):
        local_now = datetime.now().astimezone()
        ts = local_now.strftime(f"%H:%M:%S {local_now.tzname() or 'local'}")
        if role == "user":
            self.console.print(
                Panel(
                    Text(str(content)),
                    title="User",
                    style="blue",
                    box=box.ROUNDED,
                    title_align="right",
                )
            )
            return
        # Assistant path. If the text already streamed token-by-token, the
        # user has seen it — `_GenerationLive.__exit__` already re-emitted
        # it via `render_response`. Don't redraw the panel (duplicate).
        # The `_streamed_any_text` flag survives until the next stream's
        # `__enter__` clears it. History replay and other non-streaming
        # callers never set the flag, so they always render normally.
        if self._streamed_any_text:
            return
        if model_name:
            self.console.print(f"\nAssistant ({model_name}) [{ts}]:")
        else:
            self.console.print(f"\nAssistant [{ts}]:")
        render_response(content)

    def render_visualization(self, artifact, *, local_path=None):
        """Render a compact TUI wrapper with an OSC-8 browser link."""
        descriptor = artifact if isinstance(artifact, dict) else {}
        title = str(descriptor.get("title") or descriptor.get("name") or "Visualization")
        url = str(descriptor.get("view_url") or "")
        if local_path:
            try:
                url = Path(local_path).resolve().as_uri()
            except (OSError, ValueError):
                pass

        body = Text()
        body.append("Interactive visualization ready.\n", style="dim")
        if url:
            body.append("Open visualization", style=f"bold underline link {url}")
            body.append(f"\n{url}", style="dim")
        else:
            body.append("Open it from the web or mobile Artifacts view.", style="dim")
        self.console.print(
            Panel(
                body,
                title=title,
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    def get_input(
        self,
        session_name,
        staged_files,
        agent_mode="default",
        current_task=None,
        feature_context=None,
    ):
        return self.input_handler.get_input(
            session_name,
            staged_files,
            agent_mode=agent_mode,
            current_task=current_task,
            feature_context=feature_context,
        )

    def set_variables(self, variables_dict):
        self._variables = variables_dict
        self.input_handler.set_variables(variables_dict)

    def confirm(self, message, default=True):
        return Confirm.ask(message, default=default)

    def prompt_choices(self, message, choices, default=None):
        return Prompt.ask(message, choices=choices, default=default)

    def prompt(self, message, default=None):
        return Prompt.ask(message, default=default)

    def request_tool_approval(
        self,
        *,
        tool_name=None,
        tool_args=None,
        display_args=None,
        count_info="",
        can_approve=True,
        modifications=None,
        preview_error=None,
        error_code=None,
        approval_policy="default",
        prompt_text,
        choices,
        default,
    ):
        self.console.print(prompt_text)
        if approval_policy != "always_human":
            self.console.print(
                "[dim]Tip: press Shift+Tab here to toggle YOLO for this and subsequent approvals in the current loop.[/dim]"
            )
        choice = self.input_handler.prompt_choice(
            "Approval choice",
            choices=choices,
            default=default,
        )
        reason = None
        if choice == "e":
            reason = self.prompt("Provide an explanation to the model")
        return choice, reason

    def show_error(self, message):
        if isinstance(message, Text):
            self.console.print(message, style="red")
        else:
            self.console.print(Text(str(message), style="red"))

    def show_info(self, message):
        if isinstance(message, Text):
            text = message.plain
        else:
            text = str(message)
        if self._is_silenced_in_compact_mode(text):
            return
        if isinstance(message, Text):
            self.console.print(message, style="blue")
        else:
            self.console.print(Text(text, style="blue"))

    def _is_silenced_in_compact_mode(self, text: str) -> bool:
        """Filter noisy diagnostic messages when verbose rendering is off.

        Compact mode (the default) hides per-turn token lines, the
        post-iteration final-totals line, the compaction notice, the
        full tool-args echo, and the collated-tool indicator. The
        compact streaming `→ tool_name` indicator stays — it comes
        from `stream_tool_call`, not from `show_info`.
        """
        variables = self._variables or {}
        if variables.get("verbose", False):
            return False
        stripped = text.lstrip()
        if stripped.startswith("🔨 Running tool:"):
            return True
        if stripped.startswith("Tokens:"):
            return True
        if stripped.startswith("Final session tokens:"):
            return True
        if "Compacting turn history" in stripped:
            return True
        if stripped.startswith("[Collated:"):
            return True
        return False

    def build_meter(
        self,
        label,
        current,
        maximum,
        *,
        color="cyan",
        width=16,
        warning_threshold=0.75,
        danger_threshold=0.9,
    ):
        maximum = max(1, int(maximum or 1))
        current = max(0, int(current or 0))
        ratio = min(current / maximum, 1.0)
        filled = min(width, int(round(width * ratio)))

        bar = Text()
        active_color = color
        if ratio >= danger_threshold:
            active_color = "red"
        elif ratio >= warning_threshold:
            active_color = "yellow"

        bar.append("█" * filled, style=f"bold {active_color}")
        bar.append("░" * (width - filled), style="grey30")

        line = Text()
        line.append(f"{label:<8}", style="bold white")
        line.append(" ")
        line.append(bar)
        line.append(f" {current}/{maximum}", style="dim white")
        return line

    def build_memory_monitor(self, session):
        metrics = collect_runtime_metrics(session)

        meters = [
            self.build_meter(
                "CTX",
                metrics["ctx"]["current"],
                metrics["ctx"]["maximum"],
                color="cyan",
            ),
            self.build_meter(
                "MEM",
                metrics["mem"]["current"],
                metrics["mem"]["maximum"],
                color="magenta",
            ),
            self.build_meter(
                "SCRATCH",
                metrics["scratch"]["current"],
                metrics["scratch"]["maximum"],
                color="green",
            ),
            self.build_meter(
                "QUEUE",
                metrics["queue"]["current"],
                metrics["queue"]["maximum"],
                color="yellow",
            ),
        ]

        current_mode = metrics["mode"]["name"]
        mode_description = AGENT_MODE_METADATA.get(current_mode, {}).get(
            "description", ""
        )

        meta = Text()
        meta.append("tokens ", style="dim white")
        meta.append(str(metrics["tokens"]["total"]), style="bold cyan")
        meta.append("  |  queue ", style="dim white")
        meta.append(str(metrics["queue_items"]), style="bold yellow")
        meta.append(" item", style="dim white")
        if metrics["queue_items"] != 1:
            meta.append("s", style="dim white")
        meta.append("  |  mode ", style="dim white")
        meta.append(current_mode, style="bold magenta")

        token_line = Text()
        token_line.append("in ", style="dim white")
        token_line.append(str(metrics["tokens"]["input"]), style="bold cyan")
        token_line.append("  out ", style="dim white")
        token_line.append(str(metrics["tokens"]["output"]), style="bold green")
        token_line.append("  cost ", style="dim white")
        token_line.append(
            f"${metrics['tokens']['total_cost']:.5f}",
            style="bold yellow",
        )

        mode_line = Text()
        mode_line.append("mode info ", style="dim white")
        mode_line.append(mode_description or "No description available.", style="white")

        feature_group = []
        feature_metrics = metrics.get("feature")
        if feature_metrics:
            feature_state = feature_metrics.get("state") or {}
            feature_plan = feature_metrics.get("plan") or {}
            feature_name = feature_plan.get(
                "feature_name", feature_state.get("directory", "Unknown feature")
            )
            feature_status = feature_state.get("status", "unknown")
            feature_group.append(Text(""))

            feature_header = Text()
            feature_header.append("feature ", style="dim white")
            feature_header.append(str(feature_name), style="bold cyan")
            feature_group.append(feature_header)

            feature_meta = Text()
            feature_meta.append("loop ", style="dim white")
            feature_meta.append(str(feature_status), style="bold yellow")
            if feature_plan:
                feature_meta.append("  |  review ", style="dim white")
                feature_meta.append(
                    str(feature_plan.get("review_status", "unknown")),
                    style="bold magenta",
                )
            feature_group.append(feature_meta)

            phases = feature_plan.get("phases", [])
            progress = feature_metrics.get("progress") or {}
            completed_phases = sum(
                1 for phase in phases if phase.get("status") == "completed"
            )
            phase_count = max(1, int(feature_plan.get("phase_count", len(phases)) or 1))
            feature_group.append(
                self.build_meter(
                    "PHASES",
                    completed_phases,
                    phase_count,
                    color="blue",
                )
            )

            for phase in phases[:4]:
                counts = phase.get("task_counts", {})
                total_tasks = max(1, sum(int(value or 0) for value in counts.values()))
                completed_tasks = int(counts.get("completed", 0) or 0)
                phase_color = {
                    "completed": "green",
                    "in_progress": "yellow",
                    "not_started": "blue",
                }.get(phase.get("status"), "cyan")
                feature_group.append(
                    self.build_meter(
                        f"P{phase.get('number', '?')}",
                        completed_tasks,
                        total_tasks,
                        color=phase_color,
                        width=12,
                    )
                )

            if progress:
                next_phase = progress.get("next_phase") or {}
                active_label = next_phase.get("title") or "Review"
                elapsed_seconds = int(progress.get("elapsed_seconds", 0) or 0)
                elapsed = self._format_elapsed(elapsed_seconds)
                token_delta = int(progress.get("token_delta", 0) or 0)
                activity = Text()
                activity.append(f"Implementing {active_label}… ", style="white")
                activity.append(
                    f"({elapsed} · ↓ {self._format_token_delta(token_delta)} tokens)",
                    style="dim white",
                )
                feature_group.append(activity)

                completed = int(progress.get("completed_tasks", 0) or 0)
                remaining = max(0, int(progress.get("total_tasks", 0) or 0) - completed)
                summary_line = Text()
                summary_line.append("✔ ", style="green")
                summary_line.append(f"{completed} completed", style="white")
                summary_line.append("  ◻ ", style="cyan")
                summary_line.append(f"{remaining} remaining", style="white")
                feature_group.append(summary_line)

            max_visible_tasks = 10
            visible_tasks = phases[:max_visible_tasks]
            hidden_tasks = phases[max_visible_tasks:]
            hidden_completed = sum(
                1 for task in hidden_tasks if task.get("status") == "completed"
            )
            for phase in visible_tasks:
                icon = {
                    "completed": "✔",
                    "in_progress": "◼",
                    "not_started": "◻",
                }.get(phase.get("status"), "◻")
                icon_style = {
                    "completed": "green",
                    "in_progress": "yellow",
                    "not_started": "grey70",
                }.get(phase.get("status"), "grey70")
                task_line = Text()
                task_line.append(f"{icon} ", style=icon_style)
                task_line.append(
                    str(phase.get("title", "")),
                    style="white",
                )
                feature_group.append(task_line)
            if hidden_completed > 0:
                hidden_line = Text()
                hidden_line.append("… ", style="dim white")
                hidden_line.append(f"+{hidden_completed} completed", style="dim green")
                feature_group.append(hidden_line)

            next_phase = feature_plan.get("next_phase")
            if next_phase:
                next_phase_line = Text()
                next_phase_line.append("next ", style="dim white")
                next_phase_line.append(
                    f"P{next_phase.get('number')}: {next_phase.get('title', '')}",
                    style="white",
                )
                feature_group.append(next_phase_line)

        legend = Text.from_markup(
            "[cyan]context[/cyan] [magenta]memory[/magenta] [green]scratchpad[/green] [yellow]collation[/yellow]"
        )

        return Align.center(
            Panel(
                Group(
                    *meters,
                    Text(""),
                    meta,
                    token_line,
                    mode_line,
                    *feature_group,
                    legend,
                ),
                title="[bold white]/stats[/bold white]",
                border_style="bright_black",
                box=box.ROUNDED,
                width=76,
            )
        )

    def show_memory_monitor(self, session):
        self.console.print(self.build_memory_monitor(session))

    @staticmethod
    def _format_elapsed(total_seconds):
        total_seconds = max(0, int(total_seconds or 0))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        return f"{minutes}m {seconds}s"

    @staticmethod
    def _format_token_delta(tokens):
        tokens = max(0, int(tokens or 0))
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}k"
        return str(tokens)

    def show_diff(self, filename, original_content, new_content):
        """Displays a side-by-side diff with context-aware hunks and Git-style highlighting."""
        import difflib
        import os

        ext = os.path.splitext(filename)[1][1:] or "txt"

        orig_lines = original_content.splitlines()
        new_lines = new_content.splitlines()

        sm = difflib.SequenceMatcher(None, orig_lines, new_lines)
        grouped_opcodes = sm.get_grouped_opcodes(n=3)

        table = Table(
            title=f"PROPOSED CHANGES: {filename}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            expand=True,
            pad_edge=False,
            collapse_padding=True,
        )

        # We'll use 4 columns: Line L, Content L, Line R, Content R
        table.add_column("L#", justify="right", style="dim", width=5)
        table.add_column("CURRENT STATE", ratio=1)
        table.add_column("R#", justify="right", style="dim", width=5)
        table.add_column("PROPOSED STATE", ratio=1)

        for group in grouped_opcodes:
            # Hunk separator
            table.add_row(
                Text("...", style="cyan"),
                Text(
                    f"@@ hunk starting at L{group[0][1]+1} R{group[0][3]+1} @@",
                    style="cyan dim",
                ),
                Text("...", style="cyan"),
                Text("", style="cyan dim"),
            )

            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    for k in range(i2 - i1):
                        table.add_row(
                            str(i1 + k + 1),
                            Syntax(
                                orig_lines[i1 + k],
                                ext,
                                theme="monokai",
                                background_color="default",
                            ),
                            str(j1 + k + 1),
                            Syntax(
                                new_lines[j1 + k],
                                ext,
                                theme="monokai",
                                background_color="default",
                            ),
                        )
                elif tag == "delete":
                    for k in range(i2 - i1):
                        table.add_row(
                            Text(str(i1 + k + 1), style="red"),
                            Text("- " + orig_lines[i1 + k], style="red on #3a0000"),
                            "",
                            "",
                            style="on #2a0000",
                        )
                elif tag == "insert":
                    for k in range(j2 - j1):
                        table.add_row(
                            "",
                            "",
                            Text(str(j1 + k + 1), style="green"),
                            Text("+ " + new_lines[j1 + k], style="green on #002b00"),
                            style="on #001b00",
                        )
                elif tag == "replace":
                    # For replace, we show deletions then insertions to keep L/R aligned if possible,
                    # but side-by-side replace is tricky in 4 columns if we want to align corresponding lines.
                    # Simplest is to show L on left and R on right in the same row.
                    max_range = max(i2 - i1, j2 - j1)
                    for k in range(max_range):
                        l_idx = i1 + k
                        r_idx = j1 + k

                        l_num = str(l_idx + 1) if l_idx < i2 else ""
                        l_text = orig_lines[l_idx] if l_idx < i2 else ""

                        r_num = str(r_idx + 1) if r_idx < j2 else ""
                        r_text = new_lines[r_idx] if r_idx < j2 else ""

                        table.add_row(
                            Text(l_num, style="red" if l_num else ""),
                            Text(
                                "- " + l_text, style="red on #3a0000" if l_text else ""
                            ),
                            Text(r_num, style="green" if r_num else ""),
                            Text(
                                "+ " + r_text,
                                style="green on #002b00" if r_text else "",
                            ),
                            style="on #1a1a1a",  # Neutral dark background for mixed rows
                        )

        # Summary calculation
        diff_list = list(difflib.unified_diff(orig_lines, new_lines))
        additions = len(
            [l for l in diff_list if l.startswith("+") and not l.startswith("+++")]
        )
        deletions = len(
            [l for l in diff_list if l.startswith("-") and not l.startswith("---")]
        )
        summary = f"[bold green]+{additions} lines[/bold green]  [bold red]-{deletions} lines[/bold red]"

        self.console.print("\n")
        self.console.print(table)
        self.console.print(
            Panel(summary, title="Change Summary", expand=False, border_style="dim")
        )
        self.console.print("\n")

    def show_status(self, message):
        """Open a unified streaming + status `_GenerationLive`.

        Replaces the old `console.status` spinner. The returned context
        manager opens a single Rich `Live` containing:
          * top: the accumulating assistant text (token-streamed)
          * middle: any accumulating thinking content (dim italic)
          * bottom: a spinner + the status message

        Streaming deltas pumped through `stream_assistant_delta` /
        `stream_thinking_delta` / `stream_tool_call` write into the same
        Live so token output never fights with the spinner for cursor
        position. The status line stays anchored at the bottom.

        Returned object supports `update(new_message)` so callers that
        previously used the rich Status API keep working.
        """
        return _GenerationLive(self, message)

    def build_live_status(self, session, model_name, iteration, max_iterations):
        metrics = collect_runtime_metrics(session)
        status = Text()
        status.append(f"Generating ({model_name}) it {iteration}/{max_iterations} | ")
        if metrics["yolo"]["enabled"]:
            status.append("✦", style="bold yellow blink")
            status.append(" YOLO | ", style="bold yellow")
        else:
            status.append("YOLO:off | ", style="dim")
        status.append(build_live_status_line(session), style="white")
        return status

    @staticmethod
    def _status_with_yolo(base_message, enabled):
        suffix = "✦ YOLO" if enabled else "YOLO:off"
        raw = base_message.plain if isinstance(base_message, Text) else str(base_message)
        raw = raw.replace("YOLO:off", suffix)
        raw = raw.replace("✦ YOLO", suffix)
        return raw

    def _start_yolo_status_watcher(self, *, status, base_message, stop_event):
        fd = None
        old_attrs = None
        try:
            if not sys.stdin.isatty():
                return None
            fd = sys.stdin.fileno()
            import termios
            import tty

            old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return None

        def _watch():
            buffer = ""
            try:
                while not stop_event.is_set():
                    ready, _, _ = select.select([fd], [], [], 0.1)
                    if not ready:
                        continue
                    chunk = os.read(fd, 32).decode(errors="ignore")
                    if not chunk:
                        continue
                    combined = buffer + chunk
                    while "\x1b[Z" in combined:
                        combined = combined.replace("\x1b[Z", "", 1)
                        enabled = self.input_handler.toggle_yolo_mode()
                        status.update(self._status_with_yolo(base_message, enabled))
                    buffer = combined[-6:]
                    time.sleep(0.02)
            finally:
                if old_attrs is not None and fd is not None:
                    try:
                        import termios

                        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
                    except Exception:
                        pass

        watcher = NamedThread(target=_watch, daemon=True, name="rich-ui-watcher")
        watcher.start()
        return watcher

    def show_tool_result(self, result_str):
        """Displays the tool result preview with green for success and red for Error:."""
        variables = self._variables or {}
        if not variables.get("verbose", False):
            # Compact mode: tool results are silent — the `→ tool_name`
            # streaming indicator already showed the call. Errors still
            # surface because `show_error` is a separate channel.
            return
        res_preview = str(result_str).replace("\n", " ")[:60]
        char_count = len(str(result_str))
        color = (
            "red"
            if "Error" in str(res_preview) or "User denied" in str(res_preview)
            else "green"
        )
        self.console.print(
            f"[{color}]  ↳ Result: {safe_markup(res_preview)}... ({char_count} chars)[/{color}]"
        )

    def run_quiz(self, questions):
        """Launch the live quiz UI. On any failure (no TTY, prompt-toolkit
        bail, etc.) we re-raise so the caller can fall back to chat-flow
        question/answer."""
        from mu.ui.quiz_picker import run_interactive_quiz

        return run_interactive_quiz(questions)

    def ask_user_choice(
        self,
        question,
        options,
        *,
        multi_select=False,
        description="",
        allow_other=False,
    ):
        """Run the multi-choice prompt picker. On TTY failure, falls back
        to a numbered prompt on stdin/stdout so headless contexts still
        get a useful response shape."""
        try:
            from mu.ui.choice_prompt import run_interactive_choice_prompt

            return run_interactive_choice_prompt(
                question,
                list(options),
                multi_select=bool(multi_select),
                description=str(description or ""),
                allow_other=bool(allow_other),
            )
        except Exception:
            return self._fallback_choice_prompt(
                question,
                list(options),
                multi_select=bool(multi_select),
                description=str(description or ""),
                allow_other=bool(allow_other),
            )

    def _fallback_choice_prompt(
        self, question, options, *, multi_select, description, allow_other=False
    ):
        """Numbered-list prompt for when prompt-toolkit can't drive the TTY."""
        other_index = len(options) + 1 if allow_other else None
        try:
            self.console.print(f"\n[bold]{safe_markup(question)}[/bold]")
            if description:
                self.console.print(f"[dim]{safe_markup(description)}[/dim]")
            for i, option in enumerate(options, start=1):
                self.console.print(f"  {i}. {safe_markup(option)}")
            if other_index is not None:
                self.console.print(
                    f"  {other_index}. [italic dim]Other (type your own)…[/italic dim]"
                )
            if multi_select:
                self.console.print(
                    "[dim]Enter comma-separated numbers (e.g. 1,3) or blank to cancel:[/dim]"
                )
            else:
                self.console.print(
                    "[dim]Enter a number, or blank to cancel:[/dim]"
                )
            raw = input("> ").strip()
        except Exception:
            return {"selected": [], "other_text": "", "cancelled": True}
        if not raw:
            return {"selected": [], "other_text": "", "cancelled": True}
        picks: list[str] = []
        wants_other = False
        for token in raw.split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            idx = int(token) - 1
            if other_index is not None and idx == other_index - 1:
                wants_other = True
                if not multi_select:
                    break
                continue
            if 0 <= idx < len(options):
                picks.append(options[idx])
            if not multi_select and picks:
                break
        other_text = ""
        if wants_other:
            try:
                other_text = input(f"Other ({question.rstrip('?:')}): ").strip()
            except Exception:
                other_text = ""
        return {
            "selected": picks,
            "other_text": other_text,
            "cancelled": not picks and not other_text,
        }


class _GenerationLive:
    """Context manager that owns a single Rich Live region for one agent
    generation iteration.

    Layout (top → bottom):
      * accumulating assistant text (streamed)
      * accumulating thinking text (dim italic) — usually empty
      * tool-call markers as they fire ("→ tool_name" lines)
      * status footer (spinner + the status message)

    Why one Live instead of `console.status` + bare `console.print` for
    tokens: a Rich Status is itself a Live, and `console.print(text, end="")`
    fired while a Live is active doesn't reliably keep tokens on the same
    line — each print competes with the Live's redraw and cursor tracking
    breaks (the bug the user reported: scattered tokens, status pushed
    around).  By routing every streaming source through this one Live,
    the spinner stays anchored at the bottom and text flows naturally
    above it. When the CM exits, the rendered content remains in
    terminal scrollback (transient=False).
    """

    def __init__(self, ui: "RichUI", status_message: str):
        self.ui = ui
        self._status_message = str(status_message or "")
        self._text_buf: list = []
        self._thinking_buf: list = []
        self._tool_call_log: list = []
        self._live = None
        self._lock = threading.Lock()
        self._watcher_stop = threading.Event()
        self._watcher = None

    # ---------------------------------------------------------- lifecycle

    def __enter__(self):
        # Wire the UI's streaming handlers to route into us.
        self.ui._gen_live = self
        # Start of a new stream — clear the persistent flag so a previous
        # iteration's panel-suppression hint doesn't bleed forward.
        self.ui._streamed_any_text = False
        # `transient=True` so the live streaming region clears on exit;
        # we then re-print the accumulated content as properly-rendered
        # rich Markdown (otherwise `**bold**` / headers / code fences
        # would land in scrollback as raw markdown characters).
        self._live = Live(
            self._render(),
            console=self.ui.console,
            refresh_per_second=10,
            transient=True,
            auto_refresh=True,
        )
        self._live.start()
        # YOLO toggle watcher (Shift+Tab) still works through the same
        # update mechanism — it changes the status footer text.
        self._watcher = self.ui._start_yolo_status_watcher(
            status=self,
            base_message=self._status_message,
            stop_event=self._watcher_stop,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self._watcher_stop.set()
        if self._watcher is not None:
            try:
                self._watcher.join(timeout=0.2)
            except Exception:
                pass
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        # Live region was transient — it's gone from the terminal now.
        # Re-emit the accumulated buffers as properly-styled output so
        # final scrollback shows rendered Markdown (headers/lists/code
        # fences) instead of the raw character stream the Live was using
        # for fast token append.
        with self._lock:
            text = "".join(self._text_buf).strip()
            thinking = "".join(self._thinking_buf).strip()
            tool_calls = list(self._tool_call_log)
        try:
            if thinking:
                # Reasoning content: keep dim italic, no Markdown — partial
                # markup is common and would render erratically.
                self.ui.console.print(Text(thinking, style="dim italic"))
            for name in tool_calls:
                self.ui.console.print(
                    Text.assemble(("→ ", "cyan"), (str(name), "cyan")),
                    highlight=False,
                )
            if text:
                from .render import render_response
                render_response(text)
        except Exception:
            # Never let a render bug eat the turn — fall back to raw print.
            if text:
                try:
                    self.ui.console.print(text)
                except Exception:
                    pass
        self.ui._gen_live = None
        return False

    # ----------------------------------------------- streaming sinks

    def append_text(self, text: str) -> None:
        with self._lock:
            self._text_buf.append(text)
        self._refresh()

    def append_thinking(self, text: str) -> None:
        with self._lock:
            self._thinking_buf.append(text)
        self._refresh()

    def note_tool_call(self, tool_name: str) -> None:
        with self._lock:
            self._tool_call_log.append(str(tool_name or ""))
        self._refresh()

    def update(self, new_message: str) -> None:
        """Compat with the old `console.status` API + the YOLO watcher."""
        self._status_message = str(new_message or "")
        self._refresh()

    # ------------------------------------------------- render

    def _refresh(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:
                pass

    def _render(self, *, final: bool = False):
        with self._lock:
            text = "".join(self._text_buf)
            thinking = "".join(self._thinking_buf)
            tool_calls = list(self._tool_call_log)
        parts = []
        if thinking:
            parts.append(Text(thinking, style="dim italic"))
        if text:
            parts.append(Text(text))
        for name in tool_calls:
            parts.append(Text(f"→ {name}", style="cyan"))
        # Bottom-anchored status footer — ONLY while the Live is active.
        # On the final render (Live exiting) we deliberately drop the
        # status: `transient=False` means the final render persists to
        # scrollback, so including the footer here leaks a duplicate
        # "Generating ... it N/1000 | ..." stub on every turn.
        if not final and self._status_message:
            spinner = Spinner("dots", text=Text(f" {self._status_message}", style="dim"))
            parts.append(spinner)
        return Group(*parts) if parts else Text("")
