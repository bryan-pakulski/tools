from __future__ import annotations

import asyncio
from pathlib import Path

import mu.gui.routers.artifacts as artifact_router

from mu.artifact import ArtifactRegistry


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_visualization_view_route_builds_html_response(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions" / "demo"
    registry = ArtifactRegistry(str(session_dir))
    artifact = registry.add(
        name="view.html",
        content="<!doctype html><p>ok</p>",
        mime_type="text/html",
        kind="visualization",
    )
    monkeypatch.setattr(artifact_router, "HISTORY_DIR", str(tmp_path))

    response = asyncio.run(
        artifact_router.view_artifact("demo", artifact["artifact_id"])
    )

    assert response.media_type == "text/html"
    assert "sandbox allow-scripts" in response.headers["content-security-policy"]
    assert '\n+        "Content-Security-Policy"' not in read(
        "mu/gui/routers/artifacts.py"
    )


def test_history_keeps_collapsed_intermediate_information():
    sessions = read("mu/gui/routers/sessions.py")
    web = read("mu/gui/static/js/app.js")
    loop = read("mu/agent/loop_body.py")

    # Round-44 F2: the registry read is bounded (limit kwarg) and scoped to
    # the requested window — the guard now pins the bounded-read contract
    # instead of the old unbounded .list() call.
    assert "ArtifactRegistry(session_dir).list(limit=registry_read_limit)" in sessions
    assert '"tool_args": part.get("tool_args")' in sessions
    assert 'ptype in {"thinking", "reasoning", "thought"}' in sessions
    assert 'part.type in {"thinking", "reasoning", "thought"}' in loop
    assert "const rebuiltTurns = []" in web
    assert "dst.turns = rebuiltTurns" in web
    assert 'part.type === "thinking"' in web


def test_visualization_guidance_and_codemirror_dependency():
    """Guidance pins plus the CodeMirror-removal contract.

    Commit 1b8ec71 removed the embedded file editor and every CodeMirror
    dependency (971d101 base.html cleanup, 650f4e8/8ae58b9/4709bdb/e6ae5cc
    vendor removals). The visualization guidance assertions are unaffected;
    the CodeMirror portion now pins the ABSENCE contract that
    test_web_product_ui guards: no vendor files, no base.html tags.
    """
    config = read("utils/config.py")
    base = read("mu/gui/templates/base.html")

    assert config.count("Use `publish_visualization` proactively") == 3
    assert "do not wait for the user to nudge a tool call" in config
    # CodeMirror stays removed: no vendor script tags in the web shell.
    assert "codemirror" not in base.lower()
    assert not (ROOT / "mu/gui/static/vendor/codemirror-simple.js").exists()
