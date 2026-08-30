import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

from mu.gui.routers import sessions as sessions_router


ROOT = Path(__file__).resolve().parents[1]


def test_history_projection_does_not_block_event_loop(monkeypatch):
    def slow_projection(*_args, **_kwargs):
        time.sleep(0.1)
        return {"name": "large", "turns": []}

    monkeypatch.setattr(sessions_router, "_get_history_sync", slow_projection)

    async def exercise():
        request = SimpleNamespace()
        loop_advanced = asyncio.Event()
        asyncio.get_running_loop().call_later(0.01, loop_advanced.set)
        result = await sessions_router.get_history(
            request,
            session_name="large",
            limit_turns=100,
            artifact_limit=None,
            before_index=None,
            after_index=None,
            full=False,
        )
        assert result == {"name": "large", "turns": []}
        # This callback represents SSE heartbeat/delta and interrupt handling.
        # It must execute while the history worker is still blocked.
        assert loop_advanced.is_set()

    asyncio.run(exercise())


def test_history_clients_hydrate_in_cooperative_batches():
    browser = (ROOT / "mu/gui/static/js/app.js").read_text(encoding="utf-8")
    mobile = (ROOT / "mobile/android/src/hooks/useChatSession.ts").read_text(
        encoding="utf-8"
    )

    assert "historyChunkIndex % 8" in browser
    assert "_historyTurnsToTimeline" in browser
    assert "WEB_HISTORY_PAGE_TURNS = 200" in browser
    assert "WEB_HISTORY_CHECKPOINT_BATCH = 5" in browser
    assert "WEB_HISTORY_CHECKPOINT_SCAN_PAGES = 6" in browser
    assert "historyToMessagesCooperatively" in mobile
    assert "await new Promise<void>(resolve => setTimeout(resolve, 0))" in mobile


def test_subagent_history_reads_only_states_referenced_by_page(tmp_path, monkeypatch):
    subagents = tmp_path / "subagents"
    subagents.mkdir()
    loaded = []

    class Store:
        def __init__(self, _session_dir):
            pass

        def load(self, task_id):
            loaded.append(task_id)
            return {
                "task_id": task_id,
                "batch_id": task_id,
                "status": "done",
            }

        def list(self):  # pragma: no cover - a regression must never call it
            raise AssertionError("history page enumerated every subagent")

    monkeypatch.setattr(sessions_router, "SubagentArtifactStore", Store)
    history = [
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_result",
                    "tool_name": "spawn_agent",
                    "tool_result": {"task_id": "sa-visible"},
                }
            ],
        }
    ]

    exact, _fallback = sessions_router._subagent_history_anchors(
        str(tmp_path), history, start_index=0, end_index=1
    )

    assert loaded == ["sa-visible"]
    assert exact[(0, 0)][0]["agents"][0]["task_id"] == "sa-visible"
