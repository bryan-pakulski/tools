"""Unit tests for the browser_snapshot agent tool."""

from __future__ import annotations

from mu.tools.agent import browser


def test_resolve_target_rejects_empty():
    url, err = browser._resolve_target("")
    assert url is None
    assert err and "requires" in err


def test_resolve_target_local_file(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html></html>")
    url, err = browser._resolve_target(str(f))
    assert err is None
    assert url == "file://" + str(f)


def test_resolve_target_http_passthrough():
    url, err = browser._resolve_target("https://example.com")
    assert err is None
    assert url == "https://example.com"


def test_resolve_target_missing_path(tmp_path):
    url, err = browser._resolve_target(str(tmp_path / "nope.html"))
    assert url is None
    assert "not found" in err


def test_resolve_target_directory_rejected(tmp_path):
    url, err = browser._resolve_target(str(tmp_path))
    assert url is None
    assert "directory" in err


def test_handler_rejects_invalid_target():
    result = browser.browser_snapshot({"url": ""}, context=None)
    assert result["ok"] is False
    assert result["error_code"] == "invalid_target"


def test_tool_registered_in_descriptors():
    from mu.tools.descriptors import TOOLS

    assert any(t.name == "browser_snapshot" for t in TOOLS)
