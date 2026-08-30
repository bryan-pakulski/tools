# InputHandler (prompt_toolkit)
import os
import glob
import re
import json
from html import escape
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import (
    Completer,
    Completion,
    FuzzyWordCompleter,
    NestedCompleter,
    PathCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

from utils.config import HISTORY_DIR, KNOWN_MODELS, VARIABLE_SCHEMA


MODE_PROMPT_STYLES = {
    "debug": "mode-debug",
    "feature": "mode-feature",
    "research": "mode-research",
    "security": "mode-security",
    "teacher": "mode-teacher",
}

def _mode_choices():
    try:
        from utils.config import AGENT_MODE_METADATA

        return {name: None for name in AGENT_MODE_METADATA}
    except Exception:
        return {"default": None}


MODE_CHOICES = _mode_choices()


def get_session_names():
    if not os.path.exists(HISTORY_DIR):
        return []
    session_files = glob.glob(os.path.join(HISTORY_DIR, "sessions", "*", "session.json"))
    sessions = [os.path.basename(os.path.dirname(path)) for path in session_files]

    # Backward compatibility for legacy single-file session storage.
    legacy_files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    for path in legacy_files:
        sessions.append(os.path.basename(path).replace(".json", ""))

    return sorted(set(sessions))


class DynamicSessionCompleter(Completer):
    def get_completions(self, document, complete_event):
        sessions = get_session_names()
        if not sessions:
            return
        completer = FuzzyWordCompleter(sessions)
        yield from completer.get_completions(document, complete_event)


class DynamicVariableCompleter(Completer):
    def __init__(self, input_handler):
        self.input_handler = input_handler

    def get_completions(self, document, complete_event):
        if self.input_handler.variables_dict is None:
            return
        # We always get the latest keys from the dictionary reference
        completer = FuzzyWordCompleter(list(self.input_handler.variables_dict.keys()))
        yield from completer.get_completions(document, complete_event)


class GetCompleter(Completer):
    """Position-aware completer for `/get [<var> | layer [<id>]]`.

    Without a space → variable names plus the literal `layer`.
    After `/get layer <Tab>` → layer IDs (L1, L1B, ...).
    """

    def __init__(self, variable_completer):
        self._variable_completer = variable_completer
        try:
            from mu.commands.variables import LAYER_BUDGET_VARS

            self._layer_ids = tuple(LAYER_BUDGET_VARS.keys())
        except Exception:
            self._layer_ids = ("L1A", "L1B", "L2", "L3", "L5")

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if " " not in text:
            yield from self._variable_completer.get_completions(
                document, complete_event
            )
            if "layer".startswith(text):
                yield Completion("layer", start_position=-len(text), display="layer")
            return

        first, _, remainder = text.partition(" ")
        if first.lower() != "layer":
            return
        for layer_id in self._layer_ids:
            if layer_id.upper().startswith(remainder.upper()):
                yield Completion(
                    layer_id,
                    start_position=-len(remainder),
                    display=layer_id,
                )


class DynamicFeatureIdCompleter(Completer):
    def get_completions(self, document, complete_event):
        feature_ids = set()

        # Session-managed feature metadata records.
        for path in glob.glob(
            os.path.join(HISTORY_DIR, "sessions", "*", "features", "*.json")
        ):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                feature_id = str(payload.get("feature_id", "")).strip()
                if feature_id:
                    feature_ids.add(feature_id)
            except (OSError, json.JSONDecodeError, AttributeError):
                continue

        if not feature_ids:
            return

        completer = FuzzyWordCompleter(sorted(feature_ids))
        yield from completer.get_completions(document, complete_event)


class _SkillNameCompleter(Completer):
    """Completes `/skills <name>` from the discovered skills registry."""

    def get_completions(self, document, complete_event):
        try:
            from mu.skills import discover_skills
        except ImportError:
            return
        names = sorted({skill.name for skill in discover_skills([])})
        if not names:
            return
        yield from FuzzyWordCompleter(names).get_completions(document, complete_event)


class _DocsNameCompleter(Completer):
    """Completes `/docs <name>` from files under `documentation/`."""

    def get_completions(self, document, complete_event):
        try:
            from mu.commands.docs import list_doc_names
        except ImportError:
            return
        names = list_doc_names()
        if not names:
            return
        yield from FuzzyWordCompleter(names).get_completions(document, complete_event)


class DynamicToolCompleter(Completer):
    def get_completions(self, document, complete_event):
        try:
            from mu.tools.descriptors import TOOLS

            tool_names = sorted({tool.name for tool in TOOLS if getattr(tool, "name", "")})
        except Exception:
            tool_names = []
        if not tool_names:
            return
        completer = FuzzyWordCompleter(tool_names)
        yield from completer.get_completions(document, complete_event)


class MergedCompleter(Completer):
    """Custom class to merge multiple completers to avoid import errors across versions."""

    def __init__(self, completers):
        self.completers = completers

    def get_completions(self, document, complete_event):
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)


class SetCompleter(Completer):
    """Position-aware completer for `/set <var> <value>`.

    Without a space typed yet → suggest variable names (delegates to
    `variable_completer`), plus the literal `layer` for the
    `/set layer <id> <chars>` shortcut. After a space:

      * `/set layer <Tab>`  → suggest layer IDs (L1, L1B, L2, …)
      * `/set <bool_var> <Tab>` → `true` | `false`
      * `/set agent_mode <Tab>` → registered mode names
      * everything else → no suggestion (numeric / free-form string)
    """

    def __init__(self, variable_completer, variable_schema, mode_choices):
        self._variable_completer = variable_completer
        self._schema = variable_schema
        self._mode_choices = mode_choices
        try:
            from mu.commands.variables import LAYER_BUDGET_VARS

            self._layer_ids = tuple(LAYER_BUDGET_VARS.keys())
        except Exception:
            self._layer_ids = ("L1A", "L1B", "L2", "L3", "L5")

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if " " not in text:
            # Still typing the variable name — offer normal variables
            # and the `layer` keyword.
            yield from self._variable_completer.get_completions(
                document, complete_event
            )
            if "layer".startswith(text):
                yield Completion("layer", start_position=-len(text), display="layer")
            return

        # Past the name — special-case `/set layer <id> <chars>`.
        first, _, remainder = text.partition(" ")
        if first.lower() == "layer":
            if " " not in remainder:
                for layer_id in self._layer_ids:
                    if layer_id.upper().startswith(remainder.upper()):
                        yield Completion(
                            layer_id,
                            start_position=-len(remainder),
                            display=layer_id,
                        )
            return

        # Generic `/set <var> <value>` value-side completion.
        spec = self._schema.get(first)
        if spec is None:
            return
        vtype = spec.get("type")
        candidates: list = []
        if vtype is bool:
            candidates = ["true", "false"]
        elif first == "agent_mode":
            candidates = list(self._mode_choices.keys())
        if not candidates:
            return
        for c in candidates:
            if c.startswith(remainder):
                yield Completion(
                    c, start_position=-len(remainder), display=c
                )


class SlashCommandCompleter(Completer):
    """Top-level completer for slash commands.

    `prompt_toolkit.NestedCompleter` doesn't handle `/`-prefixed keys
    well: it uses `WordCompleter` with the default `\\w+` word pattern,
    so when you type `/me` the "word before cursor" is `me` (sans
    slash) and `/memory` doesn't prefix-match.

    This completer:
      * suggests slash commands when the user has typed nothing (or only
        a partial command — `/m`, `/me`, etc.);
      * descends into the per-command sub-completer once the user has
        typed a full command followed by a space (`/memory `).
    """

    def __init__(self, command_completions: dict):
        self.command_completions = command_completions
        self._commands = sorted(command_completions.keys())

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # Once a space appears, the command is "committed" — delegate to
        # its registered sub-completer (or show no completions if the
        # command takes no args).
        if " " in text:
            cmd, _, remainder = text.partition(" ")
            sub_completer = self.command_completions.get(cmd)
            if sub_completer is None:
                return
            sub_doc = Document(text=remainder, cursor_position=len(remainder))
            yield from sub_completer.get_completions(sub_doc, complete_event)
            return

        # No space yet — completing the command name itself. Prefix-match
        # against the slash-leading keys directly.
        for cmd in self._commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                )


class InputHandler:
    def __init__(self):
        self.active_session_name = None
        self.history_root = os.path.expanduser("~/.mucli_history")
        if os.path.exists(self.history_root) and not os.path.isdir(self.history_root):
            self.history_root = os.path.expanduser("~/.mucli_history_sessions")
        os.makedirs(self.history_root, exist_ok=True)
        self.variables_dict = None  # Will be set via set_variables
        path_completer = PathCompleter(expanduser=True)
        directory_completer = PathCompleter(expanduser=True, only_directories=True)
        session_completer = DynamicSessionCompleter()
        variable_completer = DynamicVariableCompleter(self)
        feature_id_completer = DynamicFeatureIdCompleter()
        tool_name_completer = DynamicToolCompleter()

        model_completer = NestedCompleter.from_nested_dict(
            {m: None for m in KNOWN_MODELS}
        )

        provider_completer = NestedCompleter.from_nested_dict(
            {"gemini": None, "ollama": None, "openai": None}
        )

        tool_command_completer = NestedCompleter.from_nested_dict(
            {
                "enable": tool_name_completer,
                "disable": tool_name_completer,
                "list": None,
            }
        )

        feature_completer = NestedCompleter.from_nested_dict(
            {
                "list": None,
                "show": None,
                "new": None,
                "load": feature_id_completer,
                "delete": feature_id_completer,
                "status": feature_id_completer,
                "phases": feature_id_completer,
                "exit": None,
                "unload": None,
            }
        )

        teach_completer = NestedCompleter.from_nested_dict(
            {
                "list": None,
                "new": None,
                "load": None,
                "delete": None,
                "exit": None,
                "unload": None,
                "status": None,
                "next": None,
                "grades": None,
                "curriculum": None,
                "help": None,
            }
        )

        try:
            from mu.commands.memory import LIST_TARGETS as _MEMORY_LIST_TARGETS
        except ImportError:
            _MEMORY_LIST_TARGETS = (
                "all", "task", "scratchpad",
                "L1A", "L1B", "L2", "L3", "L5",
            )
        memory_completer = NestedCompleter.from_nested_dict(
            {
                "status": None,
                "list": {target: None for target in _MEMORY_LIST_TARGETS},
                "clear": {"task": None, "scratchpad": None, "all": None},
            }
        )

        mode_completer = NestedCompleter.from_nested_dict(MODE_CHOICES)
        research_completer = NestedCompleter.from_nested_dict(
            {
                "status": None,
                "sources": NestedCompleter.from_nested_dict(
                    {"--type": None, "--min": None, "--query": None}
                ),
                "show": None,
                "bibliography": None,
                "biblio": None,
                "bib": None,
                "stats": None,
                "clear": None,
                "new": None,
            }
        )
        unset_completer = MergedCompleter(
            [
                variable_completer,
                FuzzyWordCompleter(["--all"]),
            ]
        )
        folder_subcompleter = MergedCompleter(
            [
                NestedCompleter.from_nested_dict(
                    {
                        "remove": directory_completer,
                        "clear": None,
                    }
                ),
                directory_completer,
            ]
        )
        file_subcompleter = MergedCompleter(
            [
                FuzzyWordCompleter(["clear"]),
                path_completer,
            ]
        )
        workspace_completer = NestedCompleter.from_nested_dict(
            {
                "folder": folder_subcompleter,
                "file": file_subcompleter,
                "clear": None,
            }
        )
        container_completer = NestedCompleter.from_nested_dict(
            {
                "list": None, "create": None, "start": None, "stop": None,
                "restart": None, "shell": None, "attach": None, "detach": None,
                "snapshot": None, "remove": None,
            }
        )
        template_completer = NestedCompleter.from_nested_dict(
            {"list": None, "snapshot": None, "use": None, "delete": None}
        )
        session_subcommand_completer = NestedCompleter.from_nested_dict(
            {
                "list": None,
                "load": session_completer,
                "new": NestedCompleter.from_nested_dict({
                    "--type": {"chat": None, "workspace": None, "container": None}
                }),
                "delete": session_completer,
            }
        )
        plan_completer = NestedCompleter.from_nested_dict(
            {"on": None, "off": None, "toggle": None}
        )

        # `/set <var> <value>` — position-aware: variable names without a
        # space, value suggestions for bool / agent_mode after a space.
        set_completer = SetCompleter(variable_completer, VARIABLE_SCHEMA, MODE_CHOICES)
        ollama_completer = NestedCompleter.from_nested_dict(
            {
                "status": None,
                "models": None,
                "options": None,
                "pull": None,
            }
        )

        # Single source of truth for slash-command autocomplete.
        #
        # Curated to ~30 unique commands — no aliases, no dead entries.
        # Removed (with rationale): /exit /h /c /v (aliases of /quit /help
        # /clear /view), /f /add (aliases of /file), /cf (alias of
        # /clearfiles), /dir (alias of /folder), /sys (alias of /system),
        # /ls /rm (aliases of /list /delete), /open (alias of /load),
        # /features /tools (plural aliases), /splash (auto-runs at boot,
        # nothing else to do), /update (command was deleted earlier),
        # /clear-workspace /cw (use `/workspace clear`).
        self.command_completions = {
            # session control
            "/help": None,
            "/h": None,
            "/quit": None,
            "/q": None,
            "/clear": None,
            "/history": NestedCompleter.from_nested_dict({"clear": None, "show": None}),
            "/session": session_subcommand_completer,
            "/container": container_completer,
            "/template": template_completer,
            "/templates": template_completer,
            "/continue": None,
            # manual compaction (focus is free-form text — no subcommands)
            "/compact": None,
            # workspace
            "/workspace": workspace_completer,
            # model / provider
            "/model": model_completer,
            "/provider": provider_completer,
            "/ollama": ollama_completer,
            # variables
            "/set": set_completer,
            "/get": GetCompleter(variable_completer),
            "/unset": unset_completer,
            "/variables": None,
            # modes & toggles
            "/mode": mode_completer,
            "/plan": plan_completer,
            "/yolo": None,
            "/agentic": None,
            "/thinking": None,
            "/verbose": NestedCompleter.from_nested_dict({"on": None, "off": None, "toggle": None}),
            "/show-thinking": NestedCompleter.from_nested_dict({"on": None, "off": None, "toggle": None}),
            "/goal": NestedCompleter.from_nested_dict(
                {"set": None, "clear": None, "show": None, "help": None}
            ),
            "/profile": NestedCompleter.from_nested_dict(
                {"save": None, "use": None, "show": None, "delete": None}
            ),
            # /bash takes a free-form shell command — no useful completion list.
            "/bash": None,
            "/sh": None,
            "/!": None,
            "/research": research_completer,
            # memory / tools
            "/memory": memory_completer,
            "/tool": tool_command_completer,
            "/feature": feature_completer,
            "/teach": teach_completer,
            "/t": teach_completer,
            # diagnostics
            "/stats": NestedCompleter.from_nested_dict({"clear": None}),
            "/costs": None,
            "/pricing": None,
            "/trace": None,
            "/jobtrace": None,
            "/jobdiag": None,
            "/job-diagnostics": None,
            "/job-analysis": None,
            "/jobs": None,
            "/job": None,
            # memory quick-write
            "/remember": None,
            # skills
            "/skills": _SkillNameCompleter(),
            # docs
            "/docs": _DocsNameCompleter(),
            # prompts (file-based system-prompt overrides)
            "/prompts": NestedCompleter.from_nested_dict(
                {
                    "reload": None,
                    "init": None,
                    "show": None,
                    "validate": None,
                    "edit": None,
                }
            ),
        }

        # Use our slash-aware top-level completer instead of NestedCompleter
        # directly. Stock NestedCompleter uses WordCompleter under the hood
        # with a default word pattern that treats `/` as a boundary, so
        # `/me` never autocompletes to `/memory`. SlashCommandCompleter
        # prefix-matches the leading slash and only descends to the
        # per-command sub-completer once a space has been typed.
        self.completer = SlashCommandCompleter(self.command_completions)

        self.style = Style.from_dict(
            {
                "prompt": "ansiblue bold",
                "rprompt": "bg:ansiblue ansiwhite",
                "files": "ansiyellow",
                "yolo-indicator": "ansiyellow bold blink",
                # Plan mode: high-contrast lock indicator so the user can never
                # miss it. Persists across the prompt while plan_mode=True.
                "plan-indicator": "bg:ansicyan ansiblack bold",
                "mode-debug": "ansiyellow bold",
                "mode-feature": "ansiblue bold",
                "mode-research": "ansimagenta bold",
                "mode-security": "ansired bold",
            }
        )

        self.kb = KeyBindings()

        @self.kb.add("enter")
        def _(event):
            buff = event.current_buffer
            text = buff.text.strip()
            if text.startswith("/"):
                buff.validate_and_handle()
            else:
                buff.insert_text("\n")

        @self.kb.add("escape", "enter")
        def _(event):
            event.current_buffer.validate_and_handle()

        @self.kb.add("s-tab")
        def _(event):
            self.toggle_yolo_mode()
            if event.app:
                event.app.invalidate()

        self.session = self._build_prompt_session(
            self._history_file_for_session("default")
        )

    def _build_prompt_session(self, history_file):
        return PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            style=self.style,
            key_bindings=self.kb,
            multiline=True,
        )

    @staticmethod
    def _safe_session_name(session_name):
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(session_name or "").strip())
        return cleaned or "default"

    def _history_file_for_session(self, session_name):
        safe = self._safe_session_name(session_name)
        return os.path.join(self.history_root, f"{safe}.history")

    def _ensure_session_history(self, session_name):
        safe_session = self._safe_session_name(session_name)
        if self.active_session_name == safe_session:
            return
        self.active_session_name = safe_session
        self.session = self._build_prompt_session(
            self._history_file_for_session(safe_session)
        )

    def set_variables(self, variables_dict):
        """Update the reference to the variables dictionary for completion."""
        self.variables_dict = variables_dict

    def is_yolo_enabled(self):
        if self.variables_dict is None:
            return False
        return bool(self.variables_dict.get("yolo", False))

    def is_plan_mode_enabled(self):
        if self.variables_dict is None:
            return False
        return bool(self.variables_dict.get("plan_mode", False))

    def toggle_yolo_mode(self):
        if self.variables_dict is None:
            return False
        enabled = not bool(self.variables_dict.get("yolo", False))
        self.variables_dict["yolo"] = enabled
        return enabled

    @staticmethod
    def _progress_bar(done, total, width=8):
        total = max(1, int(total or 1))
        done = max(0, min(int(done or 0), total))
        ratio = done / total
        filled = min(width, int(round(width * ratio)))
        percent = int(round(ratio * 100))
        return f"{'█' * filled}{'░' * (width - filled)} {percent:>3}%"

    def build_prompt_markup(
        self,
        session_name,
        staged_files,
        agent_mode="default",
        current_task=None,
        feature_context=None,
    ):
        files_text = ""
        if staged_files:
            # Note the updated accessor here for our new FileReference schema
            f_names = ", ".join([f["file_ref"]["display_name"] for f in staged_files])
            files_text = f" [Files: {f_names}]"

        mode_name = str(agent_mode or "default").lower()
        mode_text = ""
        if mode_name != "default":
            mode_style = MODE_PROMPT_STYLES.get(mode_name, "prompt")
            mode_text = f" <{mode_style}>{mode_name}</{mode_style}>"

        yolo_text = ""
        if self.is_yolo_enabled():
            yolo_text = " <yolo-indicator>✦</yolo-indicator>"

        plan_text = ""
        if self.is_plan_mode_enabled():
            plan_text = " <plan-indicator> 🔒 PLAN MODE </plan-indicator>"

        task_text = ""
        if current_task:
            task = str(current_task).strip()
            if len(task) > 48:
                task = f"{task[:45]}…"
            task_text = f" <files>[Task: {escape(task)}]</files>"

        feature_text = ""
        if isinstance(feature_context, dict):
            phase_bar = self._progress_bar(
                feature_context.get("phase_done", 0),
                feature_context.get("phase_total", 1),
            )
            overall_bar = self._progress_bar(
                feature_context.get("overall_done", 0),
                feature_context.get("overall_total", 1),
            )
            feature_text = (
                f" <files>[P {phase_bar} | O {overall_bar}]</files>"
            )

        return (
            f"<prompt>[{session_name}]</prompt>"
            f"{mode_text}"
            f"{plan_text}"
            f"{yolo_text}"
            f"{task_text}"
            f"{feature_text}"
            f"<files>{files_text}</files>\n"
            f"<prompt>>></prompt> "
        )

    def build_input_toolbar_text(self):
        yolo_status = "ON" if self.is_yolo_enabled() else "OFF"
        plan_segment = " | 🔒 PLAN MODE ACTIVE — writes blocked" if self.is_plan_mode_enabled() else ""
        return (
            "[Meta+Enter] or [Esc] [Enter] to submit | "
            f"[Shift+Tab] toggles YOLO ({yolo_status})"
            f"{plan_segment} | "
            "/help for commands"
        )

    def build_choice_toolbar_text(self):
        yolo_status = "ON" if self.is_yolo_enabled() else "OFF"
        return f"[Shift+Tab] toggles YOLO ({yolo_status})"

    def get_input(
        self,
        session_name,
        staged_files,
        agent_mode="default",
        current_task=None,
        feature_context=None,
    ):
        self._ensure_session_history(session_name)
        message = HTML(
            self.build_prompt_markup(
                session_name,
                staged_files,
                agent_mode=agent_mode,
                current_task=current_task,
                feature_context=feature_context,
            )
        )

        def bottom_toolbar():
            return self.build_input_toolbar_text()

        try:
            return self.session.prompt(
                message,
                bottom_toolbar=bottom_toolbar,
                prompt_continuation=self._prompt_continuation,
            ).strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            raise EOFError

    def prompt_choice(self, prompt_text, *, choices, default=None):
        plain_prompt = re.sub(r"\[[^\]]+\]", "", str(prompt_text)).strip()
        choices_str = "/".join(str(choice) for choice in choices)
        default_suffix = f" default={default}" if default else ""
        message = HTML(
            f"<prompt>{plain_prompt} [{choices_str}]{default_suffix}</prompt> "
        )

        def bottom_toolbar():
            return self.build_choice_toolbar_text()

        while True:
            value = self.session.prompt(
                message,
                bottom_toolbar=bottom_toolbar,
                multiline=False,
                default=default or "",
            ).strip()
            if value in choices:
                return value

    def _prompt_continuation(self, width, line_number, is_soft_wrap):
        return HTML("<prompt>    </prompt>")
