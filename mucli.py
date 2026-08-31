#!/usr/bin/env python

import argparse
import copy
import json
import os
import re
import sys
import time
import threading

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.text import Text
from rich.table import Table
from rich import box

# Import from our new modular structure
from providers.gemini import GeminiProvider
from providers.ollama import OllamaProvider
from utils.helpers import safe_markup
from utils.logger import logger
from providers.openai import OpenAIProvider
from mu.session.session import SessionManager, Session, derive_feature_state_status
from mu.feature.engine import (
    load_feature_plan,
    refresh_and_persist_feature_plan,
    save_feature_plan,
    summarize_feature_plan,
)
from mu.tools._dispatcher import execute_tool
from mu.ui.rich_ui import RichUI
from mu.ui.choice_picker import prompt_choice, prompt_confirm
from utils.config import AGENT_MODE_METADATA, _DEFAULT_CONTEXT_TOKEN_LIMIT
from mu.tools.capabilities import normalize_session_type

console = Console()


def refresh_memory_hud(session, ui, *, force=False):
    if force and ui and hasattr(ui, "show_memory_monitor"):
        ui.show_memory_monitor(session)


def print_mode_overview(session):
    current_mode = str(session.variables.get("agent_mode", "default"))
    table = Table(title="Available Agent Modes", box=box.SIMPLE_HEAVY)
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Current", style="yellow", justify="center")
    table.add_column("Description", style="white")
    table.add_column("Docs", style="magenta")

    for mode_name, metadata in AGENT_MODE_METADATA.items():
        table.add_row(
            mode_name,
            "*" if mode_name == current_mode else "",
            metadata.get("description", ""),
            metadata.get("documentation", ""),
        )

    console.print(table)
    console.print(f"[dim]Current mode: {safe_markup(current_mode)}[/dim]")


def _research_tool_names():
    return [
        "web_search",
        "url_grounding",
        "arxiv_search",
        "doi_resolve",
        "reddit_search",
        "stackoverflow_search",
        "hackernews_search",
        "read_document",
    ]


def _extract_recent_sources(history, limit=12):
    urls = []
    seen = set()
    pattern = re.compile(r"https?://[^\s)\]>\"']+")
    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        for part in message.get("parts", []) if isinstance(message.get("parts"), list) else []:
            if not isinstance(part, dict):
                continue
            text_blob = json.dumps(part, ensure_ascii=False, default=str)
            for match in pattern.findall(text_blob):
                if match in seen:
                    continue
                seen.add(match)
                urls.append(match)
                if len(urls) >= limit:
                    return urls
    return urls


# Feature-mode helpers moved to mu/cli/feature.py — re-exported here so
# existing `from mucli import X` call sites (mu/commands/*, tests) are
# unaffected. mucli.py remains the stable public surface.
from mu.cli.feature import (  # noqa: F401
    _slugify_feature_id,
    _default_feature_directory,
    refresh_feature_record,
    get_current_feature_task_label,
    get_feature_prompt_context,
    build_feature_markdown,
    _feature_three_option_prompt,
    _log_feature_cli_event,
    _feature_prompt_with_logging,
    _feature_confirm_deny_edit_loop,
    _monitor_compact_line,
    _execute_feature_tool,
)

# Stats/help/splash rendering moved to mu/cli/display.py — re-exported
# so `from mucli import ...` call sites (mu/commands/misc.py, tests) keep
# working. mucli.py remains the stable public surface.
from mu.cli.display import (  # noqa: F401
    _HELP_GROUPS,
    build_stats_snapshot,
    _curated_commands,
    _uncurated_commands_section,
    print_help,
    print_splash,
)

def init_provider(
    provider_name, model_name, ollama_host=None, ollama_mode=None, ollama_api_key=None
):
    # Init provider contextually
    if provider_name == "ollama":
        provider = OllamaProvider(
            model_name=model_name,
            host=ollama_host or None,
            mode=ollama_mode or "auto",
            api_key=ollama_api_key,
        )
    elif provider_name == "gemini":
        provider = GeminiProvider(model_name=model_name)
    elif provider_name == "openai":
        provider = OpenAIProvider(model_name=model_name)
    else:
        return None
    return provider


def select_provider_and_model(
    args_provider,
    args_model,
    ollama_host=None,
    ollama_mode=None,
    ollama_api_key=None,
    allow_prompt=True,
):
    providers = ["gemini", "ollama", "openai"]
    provider_name = args_provider

    if provider_name not in providers:
        if not allow_prompt:
            raise ValueError("A valid --provider is required in non-interactive mode.")
        provider_name = prompt_choice(
            "Select a provider",
            [
                ("gemini", "Gemini", "Google Gemini models"),
                ("ollama", "Ollama", "Local daemon or Ollama cloud"),
                ("openai", "OpenAI", "OpenAI API models"),
            ],
            default="gemini",
            subtitle="Use the arrow keys, then press Enter.",
        )

    # An Ollama API key traditionally selected ollama.com implicitly.  Ask
    # terminal users explicitly so local models remain selectable even when
    # OLLAMA_API_KEY is present in their environment.
    # ``auto`` is the persisted legacy default, not a user choice.  Treat it
    # like an unset mode in the interactive TUI so the connection target is
    # chosen before model discovery (and, crucially, before a cloud API key
    # can make discovery list cloud models by default).
    if (
        provider_name == "ollama"
        and allow_prompt
        and ollama_mode not in {"local", "cloud"}
    ):
        ollama_mode = prompt_choice(
            "Ollama connection",
            [
                ("local", "Local", "Use OLLAMA_HOST or the local Ollama daemon"),
                ("cloud", "Cloud", "Use ollama.com with an API key"),
            ],
            default="local",
        )

    provider = init_provider(
        provider_name,
        "",
        ollama_host,
        ollama_mode,
        ollama_api_key,
    )
    if not provider:
        raise ValueError(f"Unknown provider: {provider_name}")

    models = sorted(
        provider.get_available_models() or [],
        key=lambda m: str(m).lower(),
    )
    model_name = args_model

    if not models:
        if not model_name:
            if not allow_prompt:
                raise ValueError(
                    f"A model name is required for provider '{provider_name}' in "
                    "non-interactive mode."
                )
            model_name = Prompt.ask(f"Enter model name manually for {provider_name}")
    elif model_name not in models:
        if not allow_prompt and model_name:
            raise ValueError(
                f"Model '{model_name}' is not available for provider '{provider_name}'."
            )
        if not allow_prompt:
            raise ValueError(
                f"A valid --model is required for provider '{provider_name}' in "
                "non-interactive mode."
            )
        model_name = prompt_choice(
            f"Select a {provider_name} model",
            [(model, model) for model in models],
            default=models[0],
            subtitle=f"{len(models)} models available",
        )

    provider.model_name = model_name
    # Callers that own session variables use this to persist the interactive
    # connection choice after the selected provider is returned.
    if provider_name == "ollama" and ollama_mode:
        provider._mu_ollama_mode = ollama_mode
    return provider


def _safe_delete_session(session_manager, name: str, *, silent: bool = False) -> None:
    """Drop a session at startup, bypassing the active-session guard.

    `SessionManager.delete_session` refuses to remove the currently-
    active session — but at startup `current_session_name` is just the
    bootstrap default placeholder, the user hasn't loaded anything
    yet. Temporarily clear it so any session can be deleted, then
    restore.

    If we just deleted the session named in `prior_active`, leave
    `current_session_name` empty — the caller (choose_session) will
    prompt for a new one.

    When `silent=True` the SessionManager's UI is detached for the
    duration of the call so its `show_info("Deleted session: ...")`
    print doesn't punch a hole through an active TUI render (the
    interactive picker uses this).
    """
    prior_active = session_manager.current_session_name
    prior_ui = getattr(session_manager, "ui", None)
    session_manager.current_session_name = None
    if silent:
        session_manager.ui = None
    try:
        session_manager.delete_session(name)
    finally:
        if prior_active and prior_active != name:
            session_manager.current_session_name = prior_active
        else:
            session_manager.current_session_name = ""
        if silent:
            session_manager.ui = prior_ui


def choose_session(session_manager):
    """Top-level arrow-key launcher mirroring the web/mobile welcome flow."""
    while True:
        sessions = session_manager.get_session_list()
        default_action = "sessions" if sessions else "create"
        action = prompt_choice(
            "MuCLI",
            [
                ("sessions", "Sessions", "Open, inspect, or delete a saved session"),
                ("create", "Create new", "Create a chat, workspace, or container session"),
                ("containers", "Container management", "Create and manage standalone environments and templates"),
                ("quit", "Quit", "Exit MuCLI"),
            ],
            default=default_action,
            subtitle="Choose where to begin.",
        )
        if action == "quit":
            raise SystemExit(0)
        if action == "containers":
            from mu.container.tui import run_container_manager

            run_container_manager()
            continue
        if action == "create":
            raw = Prompt.ask(
                "Session name (optional, press enter for default)", default=""
            ).strip()
            session_manager._startup_session_type = prompt_choice(
                "Session type",
                [
                    ("chat", "○ Chat", "Conversation only; no filesystem tools"),
                    ("workspace", "▱ Workspace", "Attach a host folder and enable workspace tools"),
                    ("container", "◇ Container", "Run tools inside an isolated managed environment"),
                ],
                default="workspace",
            )
            return "new", raw or None

        if not sessions:
            console.print("[dim]No saved sessions. Choose create to begin.[/dim]")
            continue
        # Build options with session-type glyph prefix.
        _session_type_glyph = {
            "chat": "○",
            "workspace": "▱",
            "container": "◇",
        }
        sessions_typed = session_manager.get_session_list_with_type()
        session_options = [
            (
                name,
                f"{_session_type_glyph.get(st, '▱')} {name}",
                f"{st} · Load or manage this session",
            )
            for name, st in sessions_typed
        ]
        session_options.append(
            ("__back__", "Back", "Return to the MuCLI launcher")
        )
        selected = prompt_choice(
            "Saved sessions",
            session_options,
            default=sessions[0],
            subtitle=f"{len(sessions)} saved session{'s' if len(sessions) != 1 else ''}",
        )
        if selected == "__back__":
            continue
        session_action = prompt_choice(
            selected,
            [
                ("load", "Load session", "Open this session"),
                ("delete", "Delete session", "Permanently remove its saved state"),
                ("back", "Back", "Return to the session list"),
            ],
            default="load",
        )
        if session_action == "back":
            continue
        if session_action == "delete":
            if prompt_confirm(f"Delete session {selected!r}?", default=False):
                _safe_delete_session(session_manager, selected)
            continue
        return "load", selected

def _choose_session_numbered(session_manager):
    """Numbered fallback for environments where the prompt-toolkit picker
    can't run. Same behavior as before: numbered list + delete sub-flow."""
    while True:
        sessions = session_manager.get_session_list()
        if not sessions:
            return "new", None

        console.print("\n[bold cyan]Available Sessions:[/bold cyan]")
        for i, s in enumerate(sessions, 1):
            console.print(f" {i}. {s}", markup=False)
        new_idx = len(sessions) + 1
        delete_idx = len(sessions) + 2
        console.print(f" {new_idx}. [bold green][New Session][/bold green]")
        console.print(f" {delete_idx}. [bold red][Delete a session…][/bold red]")

        choice = IntPrompt.ask(
            "Select a session",
            choices=[str(i) for i in range(1, delete_idx + 1)],
        )

        if choice == new_idx:
            from rich.prompt import Prompt

            name = Prompt.ask(
                "Enter name for new session (optional, press enter for default)"
            )
            return "new", name if name else None
        if choice == delete_idx:
            _delete_session_flow(session_manager, sessions)
            continue
        return "load", sessions[choice - 1]


def _delete_session_flow(session_manager, sessions):
    """Numbered prompt → confirm → delete. Used by the fallback picker."""
    if not sessions:
        return

    console.print("\n[bold red]Delete a session[/bold red]")
    for i, s in enumerate(sessions, 1):
        console.print(f" {i}. {s}", markup=False)
    cancel_idx = len(sessions) + 1
    console.print(f" {cancel_idx}. [dim][Cancel][/dim]")

    choice = IntPrompt.ask(
        "Pick a session to delete",
        choices=[str(i) for i in range(1, cancel_idx + 1)],
    )
    if choice == cancel_idx:
        console.print("[dim]Cancelled.[/dim]")
        return

    target = sessions[choice - 1]
    from rich.prompt import Confirm

    if not Confirm.ask(
        f"Delete session [bold red]{target!r}[/bold red]? This cannot be undone.",
        default=False,
    ):
        console.print("[dim]Cancelled.[/dim]")
        return

    _safe_delete_session(session_manager, target)


def sync_provider_settings(session):
    if isinstance(session.provider, OllamaProvider):
        # `apply_session_variables` binds the variables dict AND recomputes
        # host + api_key from `ollama_host` / `ollama_mode` / `ollama_api_key`
        # in one shot — so a `/set ollama_mode cloud`, a GUI local/cloud
        # toggle, or a provider switch all live-update the running provider.
        if hasattr(session.provider, "apply_session_variables"):
            session.provider.apply_session_variables(session.variables)
            return
        # Fallback for older provider objects without the unified applier.
        host_override = session.variables.get("ollama_host")
        if host_override:
            session.provider.host = host_override
            session.provider.invalidate_preflight()
        if hasattr(session.provider, "bind_session_variables"):
            session.provider.bind_session_variables(session.variables)


def build_session(args, ui, allow_prompt=True):
    session_manager = SessionManager(ui=ui, session_name=args.session)
    created_new_session = False
    requested_session_type = getattr(args, "session_type", None)
    selected_session_type = normalize_session_type(requested_session_type)
    if ui and hasattr(ui, "set_variables"):
        ui.set_variables(session_manager.variables)
    
    # Let automatic selection occur in provider class
    host_str = session_manager.variables.get("ollama_host")
    ollama_host = host_str if host_str != "" else None
    ollama_mode = session_manager.variables.get("ollama_mode") or "auto"
    ollama_api_key = session_manager.variables.get("ollama_api_key") or None

    if allow_prompt and not args.session:
        action, session_name = choose_session(session_manager)
        startup_session_type = getattr(session_manager, "_startup_session_type", None)
        if action == "new" and requested_session_type is None:
            selected_session_type = normalize_session_type(
                startup_session_type
                or prompt_choice(
                    "Session type",
                    [
                        ("chat", "Chat", "Conversation only; no filesystem tools"),
                        ("workspace", "Workspace", "Attach a host folder and enable workspace tools"),
                        ("container", "Container", "Run tools inside an isolated managed environment"),
                    ],
                    default="workspace",
                )
            )
        if action == "load":
            session_manager.switch_session(session_name)
            provider_config = session_manager.provider_config
            if provider_config.get("provider") and provider_config.get("model"):
                provider = init_provider(
                    provider_config["provider"],
                    provider_config["model"],
                    ollama_host=ollama_host,
                    ollama_mode=ollama_mode,
                    ollama_api_key=ollama_api_key,
                )
            else:
                provider = select_provider_and_model(
                    args.provider,
                    args.model,
                    ollama_host=ollama_host,
                    ollama_mode=ollama_mode,
                    ollama_api_key=ollama_api_key,
                    allow_prompt=allow_prompt,
                )
                session_manager.provider_config = {
                    "provider": provider.name,
                    "model": provider.model_name,
                }
                session_manager.save_history()
        else:
            provider = select_provider_and_model(
                args.provider,
                args.model,
                ollama_host=ollama_host,
                ollama_mode=ollama_mode,
                ollama_api_key=ollama_api_key,
                allow_prompt=allow_prompt,
            )
            session_manager.new_session(
                session_name,
                provider.name,
                provider.model_name,
                session_type=selected_session_type,
            )
            created_new_session = True
    else:
        provider = None
        provider_name = args.provider
        model_name = args.model
        provider_config = session_manager.provider_config

        if provider_name and model_name:
            if getattr(args, "provider_prevalidated", False):
                provider = init_provider(
                    provider_name,
                    model_name,
                    ollama_host=ollama_host,
                    ollama_mode=ollama_mode,
                    ollama_api_key=ollama_api_key,
                )
                if provider is None:
                    raise ValueError(f"Unknown provider: {provider_name}")
            else:
                provider = select_provider_and_model(
                    provider_name,
                    model_name,
                    ollama_host=ollama_host,
                    ollama_mode=ollama_mode,
                    ollama_api_key=ollama_api_key,
                    allow_prompt=allow_prompt,
                )
            session_manager.provider_config = {
                "provider": provider.name,
                "model": provider.model_name,
            }
            session_manager.save_history()
        elif provider_config.get("provider") and provider_config.get("model"):
            provider = init_provider(
                provider_config["provider"],
                provider_config["model"],
                ollama_host=ollama_host,
                ollama_mode=ollama_mode,
                ollama_api_key=ollama_api_key,
            )
        elif provider_name and model_name:
            provider = init_provider(
                provider_name, model_name, ollama_host=ollama_host,
                ollama_mode=ollama_mode, ollama_api_key=ollama_api_key,
            )

        if not provider:
            raise ValueError(
                "Unable to determine provider/model. Supply --provider and --model, "
                "or reuse a saved session with provider configuration."
            )

    session = Session(
        provider=provider,
        thinking=False,
        system_instruction=args.system,
        session_manager=session_manager,
        ui=ui,
        debug=args.debug,
    )

    if created_new_session and provider.name == "ollama":
        selected_ollama_mode = getattr(provider, "_mu_ollama_mode", None)
        if selected_ollama_mode in {"local", "cloud"}:
            session.variables["ollama_mode"] = selected_ollama_mode
            if selected_ollama_mode == "local":
                session.variables["ollama_host"] = ""
            session.session_manager.save_history(session.folder_context)

    # File-based system-prompt overrides. --system-file replaces the
    # non-agentic base instruction; --mode-prompt NAME=PATH installs a
    # runtime override for the base agentic prompt or a per-mode workflow
    # (same keys /set uses: agentic_system_base_override /
    # agentic_mode_prompt_<mode>). Both accept '-' for stdin.
    try:
        from mu.commands._prompt_flags import apply_prompt_flags

        apply_prompt_flags(session, args)
    except ImportError:
        pass

    # If the session we just loaded has in-flight teacher / feature
    # state, queue a resumption briefing so the agent's first turn
    # knows what's already running without making the user re-explain.
    try:
        from mu.commands.session import _queue_session_resumption_briefing

        _queue_session_resumption_briefing(session)
    except ImportError:
        pass

    if allow_prompt and session.variables.get("session_type") == "container":
        from mu.container.tui import configure_tui_container, ensure_tui_container

        if created_new_session or not getattr(session.session_manager, "container_config", None):
            configure_tui_container(session)
        ensure_tui_container(session)

    if args.workspace and session.variables.get("session_type") == "workspace":
        for workspace in args.workspace:
            session.folder_context.add_folder(workspace)
        session.session_manager.save_history(session.folder_context)
    elif (
        not session.folder_context.folders
        and allow_prompt
        and session.variables.get("session_type") == "workspace"
    ):
        try:
            from prompt_toolkit import prompt as _pt_prompt
            from prompt_toolkit.completion import PathCompleter as _PathCompleter

            console.print(
                "[dim]Workspace folder (optional — press enter to skip)\n"
                "Without a workspace the agent runs in chat-only mode[/dim]"
            )
            ws_path = _pt_prompt(
                "workspace> ",
                completer=_PathCompleter(expanduser=True, only_directories=True),
                default="",
            ).strip()
        except (ImportError, EOFError, KeyboardInterrupt):
            ws_path = ""
        if ws_path:
            ws_path = os.path.expanduser(ws_path)
            if os.path.isdir(ws_path):
                session.folder_context.add_folder(ws_path)
                session.session_manager.save_history(session.folder_context)
                console.print(f"[dim]workspace → {ws_path}[/dim]")
            else:
                console.print(f"[yellow]not a directory: {ws_path} — skipping[/yellow]")

    if args.yolo:
        session.variables["yolo"] = True
        session.session_manager.save_history(session.folder_context)

    # Auto-load hooks.json from `.mu/`. Failures log a warning and continue.
    try:
        from mu.agent.hooks_config import load_hooks_from_config

        load_hooks_from_config()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("hooks.json: load failed: %s", exc)
    sync_provider_settings(session)
    return session


def serialize_command_result(session, command, ok=True, message=None, data=None):
    return {
        "ok": ok,
        "command": command,
        "message": message,
        "data": data or {},
        "session_name": session.session_manager.current_session_name,
        "provider": session.provider.name,
        "model": session.provider.model_name,
        "variables": dict(session.variables),
        "folders": list(session.folder_context.folders),
        "history_length": len(session.session_manager.history),
    }


def handle_command(session, user_input, allow_prompt=True):
    """Thin shim around `mu.commands.dispatch`.

    Every slash command lives in `mu/commands/<module>.py`. This function
    exists only to serialize the registry's `CommandResult` into the
    dict shape callers (REPL loop, web UI, JSON output) expect.
    """
    parts = user_input.split(" ", 1)
    cmd = parts[0].lower()

    import mu.commands as _mu_commands

    new_result = _mu_commands.dispatch(session, user_input, allow_prompt=allow_prompt)
    if new_result is not None:
        data = dict(new_result.data or {})
        if new_result.exit:
            data["exit"] = True
        return serialize_command_result(
            session,
            cmd,
            ok=new_result.ok,
            message=new_result.message,
            data=data,
        )

    ui = session.ui
    if ui:
        ui.show_error(f"Unknown command: {cmd}")
    return serialize_command_result(
        session, cmd, ok=False, message=f"Unknown command: {cmd}"
    )


def _trace_analyze_cli(path: str) -> int:
    """Headless terminal summary of one trace JSONL file (``--trace-analyze``).

    Thin wrapper over the shared ``mu.trace`` parser + summary builder so the
    CLI and the GUI dashboard show identical numbers. Prints overview cards
    and the headline compaction/drift/nudge/tool counts; returns an exit code.
    """
    import os

    from mu.trace import build_series, build_summary, parse_trace

    if not os.path.exists(path):
        console.print(f"[red]trace file not found: {safe_markup(path)}[/red]")
        return 1
    run = parse_trace(path)
    if not run.header and not run.iters:
        console.print(
            f"[red]no trace records parsed from: {safe_markup(path)}[/red]"
        )
        return 1
    series = build_series(run)
    s = build_summary(run, series)

    console.print(
        f"[bold]Trace run[/bold] {s['run_id']}  "
        f"session=[cyan]{safe_markup(s['session'])}[/cyan]  "
        f"model={safe_markup(s['model'])}  mode={safe_markup(s['mode'])}"
    )
    console.print(
        f"  iters={s['iters']}  status={s['status']}  "
        f"tokens in={s['total_in']} out={s['total_out']}  "
        f"cost=${s['total_cost']}"
    )
    console.print(
        f"  peak_context={s['peak_context']}  peak_estimated={s['peak_estimated']}  "
        f"peak|drift|={s['peak_drift_abs']}%  mean_drift={s['mean_drift']}%  "
        f"context_limit={s['context_limit']}"
    )
    console.print(
        f"  compactions={s['compaction_count']} by_kind={s['compaction_by_kind']}  "
        f"mechanical_fallbacks={s['mechanical_fallback_count']}"
    )
    console.print(
        f"  nudges={s['nudge_count']} by_kind={s['nudge_by_kind']}  "
        f"broke_loop={s['nudges_broken']}  "
        f"redundant_reads={s['redundant_reads']}  tool_calls={s['tool_calls']}"
    )
    if series["tool_histogram"]:
        console.print("  tools:")
        for h in sorted(series["tool_histogram"], key=lambda x: -x["count"]):
            console.print(
                f"    {h['name']:<24} n={h['count']:<4} ok={h['ok']:<4} "
                f"err={h['error']:<3} avg_lat={h['avg_latency_ms']}ms "
                f"cache={h['cache_hit_rate']}"
            )
    if series["compaction_timeline"]:
        console.print("  compaction timeline:")
        for c in series["compaction_timeline"]:
            console.print(
                f"    iter={c['iter']:<4} {c['kind']:<20} "
                f"{c['tokens_before']}→{c['tokens_after']} "
                f"(saved {c['tokens_saved']})  summarizer={c['summarizer']}"
            )
    return 0


def main():
    logger.info("μCLI starting...")

    parser = argparse.ArgumentParser(description="Interactive AI CLI")
    parser.add_argument("--model", default=None, help="Default model")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["gemini", "ollama", "openai"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Load the specified saved session instead of prompting.",
    )
    parser.add_argument(
        "--session-type",
        dest="session_type",
        choices=["chat", "workspace", "container"],
        default=None,
        help="Capability boundary for a newly created session.",
    )
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        help="Attach a workspace folder at startup. May be provided multiple times.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Enable YOLO mode at startup.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the browser GUI in the background (frees the terminal). Default port 30311.",
    )
    parser.add_argument(
        "--gui-stop",
        action="store_true",
        help="Stop the running GUI daemon and exit.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Launch the GUI and open the Trace Analyzer dashboard (/trace) "
            "to visualize per-run context growth, tokenizer drift, "
            "compaction/nudge/tool timelines, and more."
        ),
    )
    parser.add_argument(
        "--trace-analyze",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Print a terminal summary of a trace JSONL file and exit "
            "(headless quick-look — no GUI)."
        ),
    )
    parser.add_argument(
        "--gui-foreground",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for --gui (default 30311).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Bind address for --gui (default 127.0.0.1; use 0.0.0.0 for LAN access).",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="""You are a helpful assistant, answer all questions succinctly.
        When providing code changes or file content:

  1. Always use a standard markdown pattern for code blocks: (```language ... ```).
  2. For code modifications/diffs, use the same code block style as point .1
  3. For new files or partial snippets, use the specific language tag (e.g., 'python', 'cpp')
  4. Always precede the code block with a clear header including the file path, for example: \"### File: src/main.cpp\".
  5. Only provide the new code or specific changes; do not regenerate whole files unless specifically asked.
  """,
        help="Initial system instruction",
    )
    parser.add_argument(
        "--system-file",
        type=str,
        default=None,
        help=(
            "Load the initial system instruction from a file (overrides --system). "
            "Use '-' to read from stdin."
        ),
    )
    parser.add_argument(
        "--mode-prompt",
        type=str,
        action="append",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Load a file-based prompt override for the base or a mode. Repeatable. "
            "NAME is 'base' or a mode name (default, debug, feature, research, "
            "loop, security, history, teacher); PATH is a file (use '-' for stdin). "
            "Examples: --mode-prompt base=./prompts/base.md "
            "--mode-prompt default=./prompts/default.md"
        ),
    )
    args = parser.parse_args()

    if getattr(args, "trace_analyze", None):
        sys.exit(_trace_analyze_cli(args.trace_analyze))

    if getattr(args, "trace", False):
        # Launch the GUI and point the user at the Trace Analyzer dashboard.
        args.gui = True
        from mu.gui.launcher import run_gui

        host = getattr(args, "host", None) or "127.0.0.1"
        port = int(getattr(args, "port", None) or 30311)
        try:
            run_gui(args, build_session)
        except Exception as exc:
            console.print(
                f"[red]Failed to launch GUI: {safe_markup(str(exc))}[/red]"
            )
            sys.exit(1)
        console.print(
            f"  Trace Analyzer → http://{host}:{port}/trace"
        )
        return

    if args.gui_stop:
        from mu.gui.launcher import stop_gui

        sys.exit(stop_gui(args.port))

    if args.gui:
        from mu.gui.launcher import run_gui

        try:
            run_gui(args, build_session)
        except Exception as exc:
            console.print(
                f"[red]Failed to launch GUI: {safe_markup(str(exc))}[/red]"
            )
            sys.exit(1)
        return

    ui = RichUI()

    try:
        session = build_session(args, ui, allow_prompt=True)
    except Exception as exc:
        console.print(f"[red]Failed to initialize Session/Provider: {safe_markup(str(exc))}[/red]")
        sys.exit(1)

    print_splash(session)
    refresh_memory_hud(session, ui)

    # Cross-surface continuity phases 2+3 (G1+G4): inbound session watcher +
    # presence beacon. The watcher polls session.json only while another
    # surface (GUI/mobile) holds a live beacon (§3.4); the toucher publishes
    # this CLI's own beacon every ~5s so peers see us. MUCLI_SURFACE_SYNC=1
    # forces the watcher on regardless of presence (single-surface opt-in).
    # Reloads are deferred to the turn boundary while a turn executes.
    _surface_sync = None
    _presence_toucher = None
    _thread_wake_scheduler = None
    _thread_runtime_cache = {}
    _thread_runtime_lock = threading.RLock()
    _thread_turn_locks = {}
    try:
        from mu.session.presence import BeaconToucher
        from mu.session.surface_sync import SurfaceSync

        _surface_sync = SurfaceSync(session, ui)
        session._surface_sync = _surface_sync
        _surface_sync.start()
        _presence_toucher = BeaconToucher(
            lambda: session.session_manager.current_session_name,
            "cli",
            busy_fn=lambda: getattr(session, "_current_turn_start_index", None)
            is not None,
        )
        _presence_toucher.start()
    except Exception:
        _surface_sync = None
        _presence_toucher = None

    try:
        from mu.threads.scheduler import ThreadWakeScheduler

        def _run_tui_thread_wake(coordinator, wake):
            target = coordinator.get_thread(
                str(wake.get("target_thread_id") or "")
            )
            if not target:
                return True
            name = str(target.get("session_name") or "")
            if not name:
                return True
            with _thread_runtime_lock:
                peer = _thread_runtime_cache.get(name)
                if peer is None:
                    peer_args = copy.copy(args)
                    peer_args.session = name
                    peer_args.workspace = []
                    peer = build_session(peer_args, ui, allow_prompt=False)
                    _thread_runtime_cache[name] = peer
                turn_lock = _thread_turn_locks.setdefault(name, threading.RLock())
            with turn_lock:
                wake_text = (
                    "A peer thread sent coordination updates. Inspect LAYER 3C, "
                    "respond or acknowledge each incoming message, and continue "
                    "your existing task where appropriate."
                )
                if peer.variables.get("session_type") == "container":
                    from mu.container.tui import send_tui_container_message

                    result = send_tui_container_message(
                        peer, wake_text, origin="thread_wake"
                    )
                else:
                    result = peer.send_message(wake_text, origin="thread_wake")
            coordinator.record_event(
                "thread_wake_completed",
                actor_thread_id=target["thread_id"],
                message_id=str(wake.get("message_id") or ""),
                payload={"status": (result or {}).get("status") if isinstance(result, dict) else "complete"},
            )
            try:
                ui.set_variables(session.variables)
            except Exception:
                pass
            return True

        _thread_wake_scheduler = ThreadWakeScheduler(_run_tui_thread_wake)
        _thread_wake_scheduler.start()
    except Exception:
        logger.debug("TUI thread wake scheduler failed to start", exc_info=True)

    try:
        while True:
            try:
                ui.set_variables(session.variables)
                current_task = get_current_feature_task_label(session)
                feature_context = get_feature_prompt_context(session)
                user_input = ui.get_input(
                    session.session_manager.current_session_name,
                    session.staged_files,
                    agent_mode=session.variables.get("agent_mode", "default"),
                    current_task=current_task,
                    feature_context=feature_context,
                )

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    result = handle_command(session, user_input, allow_prompt=True)
                    if result.get("data", {}).get("exit"):
                        break
                    continue

                if session.variables.get("session_type") == "container":
                    from mu.container.tui import send_tui_container_message

                    name = session.session_manager.current_session_name
                    turn_lock = _thread_turn_locks.setdefault(name, threading.RLock())
                    with turn_lock:
                        send_result = send_tui_container_message(session, user_input)
                else:
                    name = session.session_manager.current_session_name
                    turn_lock = _thread_turn_locks.setdefault(name, threading.RLock())
                    with turn_lock:
                        send_result = session.send_message(user_input)
                if send_result.get("status") == "interrupted":
                    console.print(
                        "[dim]Execution paused. Type /continue to resume, or enter a new prompt to re-guide the agent.[/dim]"
                    )
                refresh_memory_hud(session, ui)

            except KeyboardInterrupt:
                if session.variables.get("session_type") == "container":
                    try:
                        from mu.container.tui import interrupt_tui_container

                        interrupt_tui_container(session)
                    except Exception:
                        pass
                console.print("\n(Interrupted. Type /quit to exit)")
            except EOFError:
                console.print("\nGoodbye!")
                break
    finally:
        if _thread_wake_scheduler is not None:
            try:
                _thread_wake_scheduler.stop(wait=False)
            except Exception:
                pass
        for _peer in list(_thread_runtime_cache.values()):
            try:
                _peer.shutdown()
            except Exception:
                pass
        if _surface_sync is not None:
            try:
                _surface_sync.stop()
            except Exception:
                pass
        if _presence_toucher is not None:
            try:
                _presence_toucher.stop()
            except Exception:
                pass
        # Cancel any still-running async sub-agents and reap background bash
        # tasks so nothing outlives the session. No-op when none are active.
        try:
            session.shutdown()
        except Exception:
            pass
        try:
            from mu.container.tui import shutdown_tui_container

            shutdown_tui_container(session)
        except Exception:
            pass


if __name__ == "__main__":
    main()
