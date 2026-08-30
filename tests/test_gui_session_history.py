from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import utils.config as config
from mu.gui.routers.session_history import get_authoritative_history


ROOT = Path(__file__).resolve().parents[1]


def _request_with_live_history(name: str, history: list[dict]):
    manager = SimpleNamespace(current_session_name=name, history=history)
    session = SimpleNamespace(session_manager=manager)
    state = SimpleNamespace(session_by_name=lambda requested=None: session)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _history(text: str):
    return [
        {
            "role": "user",
            "parts": [{"type": "text", "text": text}],
        },
        {
            "role": "assistant",
            "parts": [{"type": "text", "text": f"reply to {text}"}],
        },
    ]


def _write_saved_history(tmp_path, name: str, history: list[dict]):
    session_dir = tmp_path / "sessions" / name
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / "session.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history}, handle)


def test_named_gui_history_uses_saved_session_when_live_copy_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_DIR", str(tmp_path))
    name = "saved-session"
    _write_saved_history(tmp_path, name, _history("persisted prompt"))

    request = _request_with_live_history(name, [])
    payload = asyncio.run(
        get_authoritative_history(
            request,
            session_name=name,
            limit_turns=None,
            artifact_limit=None,
            before_index=None,
        )
    )

    assert payload["name"] == name
    assert payload["history_source"] == "durable_session"
    assert payload["history_recovered"] is True
    assert payload["total_turns"] == 2
    assert payload["turns"][0]["parts"][0]["text"] == "persisted prompt"
    assert payload["turns"][1]["parts"][0]["text"] == "reply to persisted prompt"


def test_unsaved_history_request_can_still_use_live_session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_DIR", str(tmp_path))
    name = "live-session"
    request = _request_with_live_history(name, _history("live prompt"))

    payload = asyncio.run(
        get_authoritative_history(
            request,
            session_name=name,
            limit_turns=None,
            artifact_limit=None,
            before_index=None,
        )
    )

    assert payload["history_source"] == "live_session"
    assert payload["history_recovered"] is False
    assert payload["total_turns"] == 2
    assert payload["turns"][0]["parts"][0]["text"] == "live prompt"


def test_newer_live_history_wins_over_older_saved_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_DIR", str(tmp_path))
    name = "newer-live-session"
    saved = _history("older persisted prompt")
    live = [*saved, *_history("new live prompt")]
    _write_saved_history(tmp_path, name, saved)
    request = _request_with_live_history(name, live)

    payload = asyncio.run(
        get_authoritative_history(
            request,
            session_name=name,
            limit_turns=None,
            artifact_limit=None,
            before_index=None,
        )
    )

    assert payload["history_source"] == "live_session"
    assert payload["history_recovered"] is False
    assert payload["total_turns"] == 4
    assert payload["turns"][-2]["parts"][0]["text"] == "new live prompt"


def test_authoritative_history_forwards_bidirectional_window_parameters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "HISTORY_DIR", str(tmp_path))
    name = "paged-live-session"
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "parts": [{"type": "text", "text": f"turn {index}"}],
        }
        for index in range(8)
    ]
    request = _request_with_live_history(name, history)

    forward = asyncio.run(
        get_authoritative_history(
            request,
            session_name=name,
            limit_turns=2,
            artifact_limit=None,
            before_index=None,
            after_index=3,
        )
    )

    assert [turn["index"] for turn in forward["turns"]] == [3, 4]
    assert forward["window_end"] == 5
    assert forward["total_turns"] == 8


def test_authoritative_history_pages_begin_at_user_checkpoints(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "HISTORY_DIR", str(tmp_path))
    name = "checkpoint-session"
    history = [
        {
            "role": "user" if index in {0, 4, 8, 12, 16, 20} else "assistant",
            "parts": [{"type": "text", "text": f"turn {index}"}],
        }
        for index in range(24)
    ]
    request = _request_with_live_history(name, history)

    payload = asyncio.run(
        get_authoritative_history(
            request,
            session_name=name,
            limit_turns=20,
            artifact_limit=None,
            before_index=None,
            checkpoint_count=5,
        )
    )

    assert payload["start_index"] == 4
    assert [
        turn["index"] for turn in payload["turns"] if turn["role"] == "user"
    ] == [4, 8, 12, 16, 20]


def test_web_reload_groups_newly_hydrated_transcript_not_old_browser_slot():
    """Regression for refresh rendering an empty conversation.

    app.js rebuilds the durable transcript and passes the old turn array only so
    collapsed-group open state can be preserved.  The hydration guard must group
    slot.turns (the newly rebuilt transcript), never that pre-hydration array.
    """
    template = (ROOT / "mu/gui/templates/index.html").read_text(encoding="utf-8")
    guard = (ROOT / "mu/gui/static/js/history_hydration.js").read_text(encoding="utf-8")

    web_shell = '/static/js/web_shell.js'
    hydration = '/static/js/history_hydration.js'
    assert hydration in template
    assert template.index(web_shell) < template.index(hydration)
    assert "previousTurns !== slot.turns" in guard
    assert "const result = coreGroup(slot, slot.turns);" in guard
    assert "openByGroup" in guard
    assert "openByUser" in guard


def test_watcher_publishes_session_deleted_when_document_removed():
    """Round-27 F5: when the watched session.json vanishes, the watcher
    publishes session_deleted once and drops the tracker (no repeats)."""
    import asyncio
    import threading

    from mu.gui.watcher import SessionWatcher

    class _Bus:
        def __init__(self):
            self.events = []

        async def publish(self, event):
            self.events.append(event)

    class _SM:
        @staticmethod
        def _get_filepath(name):
            return "/nonexistent/session.json"

    class _Session:
        session_manager = _SM()

    class _State:
        def __init__(self, bus):
            self.bus = bus

        def session_busy_for(self, name):
            return threading.Event()

    class _App:
        def __init__(self, bus):
            self.state = _State(bus)

    bus = _Bus()
    watcher = SessionWatcher.__new__(SessionWatcher)
    watcher._app = _App(bus)
    watcher._interval = 1.0
    watcher._tracks = {"gone": object()}

    asyncio.run(watcher._tick_one("gone", _Session()))
    kinds = [e["kind"] for e in bus.events]
    assert kinds == ["session_deleted"]
    assert "gone" not in watcher._tracks

    # Second tick: no duplicate event.
    asyncio.run(watcher._tick_one("gone", _Session()))
    assert len(bus.events) == 1
