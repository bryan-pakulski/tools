import json
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from types import SimpleNamespace

from mu.ui.input import InputHandler, get_session_names


def test_prompt_markup_hides_default_mode():
    handler = InputHandler()
    handler.set_variables({"yolo": False})

    markup = handler.build_prompt_markup("demo", [], agent_mode="default")

    assert "[demo]" in markup
    assert "mode-feature" not in markup
    assert ">>>" in markup


def test_prompt_markup_shows_non_default_mode():
    handler = InputHandler()
    handler.set_variables({"yolo": False})

    markup = handler.build_prompt_markup("demo", [], agent_mode="feature")

    assert "[demo]" in markup
    assert "mode-feature" in markup
    assert ">feature<" in markup


def test_prompt_markup_shows_yolo_indicator_when_enabled():
    handler = InputHandler()
    handler.set_variables({"yolo": True})

    markup = handler.build_prompt_markup("demo", [], agent_mode="default")

    assert "yolo-indicator" in markup
    assert "✦" in markup


def test_prompt_markup_includes_current_task_when_present():
    handler = InputHandler()
    handler.set_variables({"yolo": False})

    markup = handler.build_prompt_markup(
        "demo",
        [],
        agent_mode="feature",
        current_task="Implement fixtures/pcap.py",
    )

    assert "[Task: Implement fixtures/pcap.py]" in markup


def test_prompt_markup_compacts_feature_progress_bars():
    handler = InputHandler()
    handler.set_variables({"yolo": False})

    markup = handler.build_prompt_markup(
        "demo",
        [],
        agent_mode="feature",
        feature_context={
            "status": "awaiting_input",
            "task": "Implement fixtures/pcap.py",
            "phase_done": 2,
            "phase_total": 4,
            "overall_done": 3,
            "overall_total": 10,
        },
    )

    assert "Feature:" not in markup
    assert "Task: Implement fixtures/pcap.py" not in markup
    assert "P ████░░░░  50%" in markup
    assert "O ██░░░░░░  30%" in markup


def test_input_toolbar_shows_plain_yolo_status_text():
    handler = InputHandler()
    handler.set_variables({"yolo": True})

    toolbar = handler.build_input_toolbar_text()

    assert toolbar == (
        "[Meta+Enter] or [Esc] [Enter] to submit | "
        "[Shift+Left] threads | "
        "[Shift+Tab] toggles YOLO (ON) | "
        "/help for commands"
    )


def test_choice_toolbar_shows_plain_yolo_status_text():
    handler = InputHandler()
    handler.set_variables({"yolo": False})

    toolbar = handler.build_choice_toolbar_text()

    assert toolbar == "[Shift+Tab] toggles YOLO (OFF)"


def test_toggle_yolo_mode_flips_bound_variable():
    handler = InputHandler()
    variables = {"yolo": False}
    handler.set_variables(variables)

    enabled = handler.toggle_yolo_mode()
    assert enabled is True
    assert variables["yolo"] is True

    enabled = handler.toggle_yolo_mode()
    assert enabled is False
    assert variables["yolo"] is False


def test_shift_tab_keybinding_is_registered_for_yolo_toggle():
    handler = InputHandler()

    bindings = [binding.keys for binding in handler.kb.bindings]

    assert any(len(keys) == 1 and keys[0] == "s-tab" for keys in bindings)


def test_command_completion_covers_curated_command_set():
    """The autocomplete dict must contain every canonical command after the
    alias cleanup. Dropped aliases (/exit /h /c /v /f /add /cf /dir /sys
    /ls /rm /open /features /tools /splash /update /clear-workspace /cw)
    are deliberately absent — `test_command_surface.py` pins THAT direction."""
    handler = InputHandler()

    expected_commands = {
        # session
        "/help",
        "/quit",
        "/q",
        "/clear",
        "/history",
        "/session",
        "/continue",
        # workspace
        "/workspace",
        # model & provider
        "/model",
        "/provider",
        "/ollama",
        # variables
        "/set",
        "/get",
        "/unset",
        "/variables",
        # modes & toggles
        "/mode",
        "/plan",
        "/yolo",
        "/agentic",
        "/thinking",
        "/research",
        # memory / tools / features
        "/memory",
        "/tool",
        "/feature",
        # diagnostics
        "/stats",
    }

    assert expected_commands.issubset(set(handler.command_completions.keys()))


def test_unset_completion_includes_all_keyword():
    handler = InputHandler()
    document = Document(
        text="/unset --",
        cursor_position=len("/unset --"),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "--all" in completion_texts


def test_workspace_folder_completion_includes_clear_subcommand():
    handler = InputHandler()
    document = Document(
        text="/workspace folder c",
        cursor_position=len("/workspace folder c"),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "clear" in completion_texts


def test_tool_enable_completion_suggests_tool_names(monkeypatch):
    monkeypatch.setattr(
        "mu.tools.descriptors.TOOLS",
        [SimpleNamespace(name="read_file"), SimpleNamespace(name="write_file")],
    )
    handler = InputHandler()
    document = Document(
        text="/tool enable wr",
        cursor_position=len("/tool enable wr"),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "write_file" in completion_texts


def test_research_completion_includes_status_and_sources():
    handler = InputHandler()
    document = Document(
        text="/research s",
        cursor_position=len("/research s"),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "status" in completion_texts
    assert "sources" in completion_texts


def test_memory_clear_completion_suggests_scratchpad():
    """`/memory clear scr` → suggests `scratchpad` (the alias `scratch`
    was removed in the cleanup pass)."""
    handler = InputHandler()
    document = Document(
        text="/memory clear scr",
        cursor_position=len("/memory clear scr"),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}
    assert "scratchpad" in completion_texts
    # `scratch` was an alias and is no longer offered.
    assert "scratch" not in completion_texts


def test_memory_list_completion_includes_layers():
    """`/memory list <Tab>` should offer all 8 layer IDs plus the
    stores (task, scratchpad, all)."""
    handler = InputHandler()
    document = Document(
        text="/memory list ",
        cursor_position=len("/memory list "),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}
    for target in (
        # L4 (recent tool activity) and L4B (auto-retrieval) were removed
        # from the layered context architecture — tool activity now lives
        # in messages and retrieval is on-demand via retrieve_relevant_context.
        "all", "task", "scratchpad",
        "L0", "L1B", "L2", "L3", "L5",
    ):
        assert target in completion_texts, f"/memory list {target!r} not suggested"


def test_input_history_is_isolated_per_session():
    handler = InputHandler()

    handler._ensure_session_history("alpha")
    first_session_obj = handler.session
    first_history_file = handler._history_file_for_session("alpha")

    handler._ensure_session_history("beta")
    second_session_obj = handler.session
    second_history_file = handler._history_file_for_session("beta")

    assert first_history_file != second_history_file
    assert first_session_obj is not second_session_obj
    assert handler.active_session_name == "beta"


def test_workspace_completion_includes_clear_subcommand():
    handler = InputHandler()
    document = Document(
        text="/workspace c",
        cursor_position=len("/workspace c"),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "clear" in completion_texts


def test_feature_completion_includes_exit_subcommand():
    handler = InputHandler()
    document = Document(
        text="/feature e",
        cursor_position=len("/feature e"),
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "exit" in completion_texts


def test_feature_delete_completion_suggests_feature_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("mu.ui.input.HISTORY_DIR", str(tmp_path))
    features_dir = tmp_path / "sessions" / "my_session" / "features"
    features_dir.mkdir(parents=True)
    (features_dir / "alpha.json").write_text(
        json.dumps({"feature_id": "alpha"}), encoding="utf-8"
    )
    (features_dir / "beta.json").write_text(
        json.dumps({"feature_id": "beta"}), encoding="utf-8"
    )

    handler = InputHandler()
    document = Document(text="/feature delete a", cursor_position=len("/feature delete a"))
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "alpha" in completion_texts
    assert "beta" in completion_texts


def test_get_session_names_supports_session_directory_layout(tmp_path, monkeypatch):
    monkeypatch.setattr("mu.ui.input.HISTORY_DIR", str(tmp_path))
    (tmp_path / "sessions" / "alpha").mkdir(parents=True)
    (tmp_path / "sessions" / "alpha" / "session.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sessions" / "beta").mkdir(parents=True)
    (tmp_path / "sessions" / "beta" / "session.json").write_text("{}", encoding="utf-8")

    sessions = get_session_names()

    assert sessions == ["alpha", "beta"]


def test_session_load_completion_suggests_saved_session_names(tmp_path, monkeypatch):
    monkeypatch.setattr("mu.ui.input.HISTORY_DIR", str(tmp_path))
    (tmp_path / "sessions" / "my_session").mkdir(parents=True)
    (tmp_path / "sessions" / "my_session" / "session.json").write_text(
        "{}",
        encoding="utf-8",
    )

    handler = InputHandler()
    document = Document(
        text="/session load my", cursor_position=len("/session load my")
    )
    completions = list(
        handler.completer.get_completions(
            document,
            CompleteEvent(completion_requested=True),
        )
    )
    completion_texts = {completion.text for completion in completions}

    assert "my_session" in completion_texts
