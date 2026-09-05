from pathlib import Path

import pytest

from mu.artifact import ArtifactError, ArtifactRegistry


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_registry_persists_visualization_metadata(tmp_path):
    registry = ArtifactRegistry(str(tmp_path / "sessions" / "demo"))
    artifact = registry.add(
        name="chart.html",
        content="<!doctype html><h1>chart</h1>",
        mime_type="text/html",
        kind="visualization",
        display="inline",
        title="Latency chart",
        height=640,
    )

    assert artifact["kind"] == "visualization"
    assert artifact["display"] == "inline"
    assert artifact["title"] == "Latency chart"
    assert artifact["height"] == 640
    assert artifact["view_url"].endswith(f"/{artifact['artifact_id']}/view")
    assert registry.get(artifact["artifact_id"]) == artifact


def test_registry_clamps_visualization_height(tmp_path):
    registry = ArtifactRegistry(str(tmp_path / "sessions" / "demo"))
    artifact = registry.add(
        name="chart.html",
        content="<html></html>",
        mime_type="text/html",
        kind="visualization",
        height=9999,
    )
    assert artifact["height"] == 1200


def test_registry_refuses_non_html_visualization(tmp_path):
    registry = ArtifactRegistry(str(tmp_path / "sessions" / "demo"))
    with pytest.raises(ArtifactError, match="HTML mime type"):
        registry.add(
            name="chart.json",
            content="{}",
            mime_type="application/json",
            kind="visualization",
        )


def test_visualization_tool_is_registered_and_publishes_inline_metadata():
    source = read("mu/tools/artifact/handlers.py")
    assert 'name="publish_visualization"' in source
    assert 'kind="visualization"' in source
    assert 'display="inline"' in source
    assert "render_visualization" in source
    assert '"publish_visualization"' in read("mu/tools/capabilities.py")


def test_web_chat_renders_sandboxed_iframe():
    template = read("mu/gui/templates/fragments/chat.html")
    router = read("mu/gui/routers/artifacts.py")
    javascript = read("mu/gui/static/js/app.js")
    assert "t.role === 'visualization'" in template
    assert "sandbox=\"allow-scripts allow-forms allow-modals allow-downloads\"" in template
    assert "allow-same-origin" not in template
    assert '/artifacts/{artifact_id}/view' in router
    assert "Content-Security-Policy" in router
    assert "chat.addVisualization(ev.artifact" in javascript


def test_history_replays_visualization_descriptors():
    sessions = read("mu/gui/routers/sessions.py")
    web = read("mu/gui/static/js/app.js")
    mobile = read("mobile/android/src/hooks/useChatSession.ts")
    assert "_visualization_from_tool_result" in sessions
    assert 'result_part["artifact"] = visualization' in sessions
    assert 'part.type === "visualization"' in web
    assert '"type": "visualization", "artifact": visualization' in sessions
    assert 'tool_result_part["artifact"] = visualization' in read(
        "mu/agent/loop_body.py"
    )
    assert "part.artifact" in web
    assert "asVisualization(part.artifact)" in mobile


def test_mobile_uses_native_webview_card():
    package = read("mobile/android/package.json")
    card = read("mobile/android/src/components/VisualizationCard.tsx")
    screen = read("mobile/android/src/screens/ChatScreen.tsx")
    assert '"react-native-webview"' in package
    assert "from 'react-native-webview'" in card
    assert "allowFileAccess={false}" in card
    assert "VisualizationCard" in screen


def test_tui_prints_clickable_browser_wrapper():
    source = read("mu/ui/rich_ui.py")
    assert "def render_visualization" in source
    assert "Path(local_path).resolve().as_uri()" in source
    assert "link {url}" in source


def test_container_worker_forwards_visualization_metadata():
    from mu.container.worker import WorkerBridgeUI

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "ok": True,
                "artifact": {
                    "artifact_id": "viz-1",
                    "name": "chart.html",
                    "kind": "visualization",
                    "display": "inline",
                    "view_url": "/api/sessions/demo/artifacts/viz-1/view",
                },
            }

    class Client:
        def post(self, url, *, params, content, headers, timeout):
            captured.update(
                url=url,
                params=dict(params),
                content=content,
                headers=dict(headers),
                timeout=timeout,
            )
            return Response()

    ui = WorkerBridgeUI("demo")
    ui._client.close()
    ui._client = Client()
    ui.supervisor_url = "http://host.docker.internal:30311"
    ui.container_name = "mucli-demo"
    ui.token = "worker-token"

    artifact = ui.publish_artifact(
        name="chart.html",
        content="<!doctype html><h1>chart</h1>",
        mime_type="text/html",
        kind="visualization",
        display="inline",
        title="Container chart",
        height=560,
    )

    assert artifact["kind"] == "visualization"
    assert artifact["view_url"].endswith("/viz-1/view")
    assert captured["params"]["kind"] == "visualization"
    assert captured["params"]["display"] == "inline"
    assert captured["params"]["title"] == "Container chart"
    assert captured["params"]["height"] == "560"
    assert captured["content"].startswith(b"<!doctype html>")


def test_container_protocol_requires_visualization_bridge_upgrade():
    worker = read("mu/container/worker.py")
    supervisor = read("mu/container/supervisor.py")
    endpoint = read("mu/gui/routers/containers.py")
    assert "WORKER_PROTOCOL_VERSION = 10" in read("mu/container/ref.py")
    assert '"worker_protocol": WORKER_PROTOCOL_VERSION' in worker
    assert "did not preserve visualization metadata" in worker
    assert "actual_protocol == WORKER_PROTOCOL_VERSION" in supervisor
    assert 'kind: str = "file"' in endpoint
    assert "kind=kind" in endpoint
