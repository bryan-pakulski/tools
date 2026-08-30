"""Regression tests for session-type icons in session list + during creation.

Covers three surfaces:
1. Web GUI welcome.html — sessionGlyph() for chat/workspace/container.
2. Mobile SwipeSessionsDrawer — typeIcon() maps session_type to Ionicons.
3. TUI mucli — session picker shows glyph prefix; creation prompt shows glyph.
4. SessionManager.get_session_list_with_type() — reads session_type from disk.

Glyphs used across surfaces:
  chat      → ○  (Ionicons: chatbubble-ellipses-outline)
  workspace → ▱  (Ionicons: folder-open-outline)
  container → ◇  (Ionicons: cube-outline)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

WELCOME_HTML = ROOT / "mu/gui/templates/fragments/welcome.html"
SWIPE_DRAWER = ROOT / "mobile/android/src/components/SwipeSessionsDrawer.tsx"
NEW_SESSION = ROOT / "mobile/android/src/components/NewSessionSheet.tsx"
MUCLI = ROOT / "mucli"
# print_splash moved to mu/cli/display.py (facade re-exports it from mucli);
# the glyph f-string lives in the implementation module now.
MUCLI_DISPLAY = ROOT / "mu/cli/display.py"
MANAGER = ROOT / "mu/session/manager.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── Web GUI ──────────────────────────────────────────────────────────────────


class TestWebGuiGlyphs:
    """welcome.html must render session-type glyphs in both list + creation."""

    def test_session_glyph_function_exists(self):
        src = read(WELCOME_HTML)
        assert "sessionGlyph" in src, "sessionGlyph function missing from welcome.html"

    def test_session_glyph_maps_all_three_types(self):
        src = read(WELCOME_HTML)
        # sessionGlyph returns ◇ for container, ○ for chat, ▱ for workspace
        assert "◇" in src or "\\u25c7" in src, "container glyph ◇ missing"
        assert "○" in src or "\\u25cb" in src, "chat glyph ○ missing"
        assert "▱" in src or "\\u25b1" in src, "workspace glyph ▱ missing"

    def test_session_list_uses_session_glyph(self):
        src = read(WELCOME_HTML)
        assert "sessionGlyph(s.session_type)" in src, (
            "session list row does not call sessionGlyph()"
        )

    def test_creation_wizard_has_icons(self):
        src = read(WELCOME_HTML)
        assert "option.icon" in src, "creation wizard does not render option.icon"
        # sessionTypes array must have icon field for each type
        assert "icon:" in src or "icon:" in src


# ── Mobile ────────────────────────────────────────────────────────────────────


class TestMobileSessionDrawer:
    """SwipeSessionsDrawer must show type icons in session rows."""

    def test_type_icon_function_exists(self):
        src = read(SWIPE_DRAWER)
        assert "typeIcon" in src, "typeIcon function missing from SwipeSessionsDrawer"

    def test_type_icon_maps_all_three_types(self):
        src = read(SWIPE_DRAWER)
        assert "cube-outline" in src, "container icon cube-outline missing"
        assert "chatbubble-ellipses-outline" in src, (
            "chat icon chatbubble-ellipses-outline missing"
        )
        assert "folder-open-outline" in src, "workspace icon folder-open-outline missing"

    def test_type_icon_wrap_style_exists(self):
        src = read(SWIPE_DRAWER)
        assert "typeIconWrap" in src, "typeIconWrap style missing"

    def test_session_type_in_subtitle(self):
        src = read(SWIPE_DRAWER)
        assert "session_type" in src, "session_type not shown in row subtitle"


class TestMobileNewSessionSheet:
    """NewSessionSheet already has icons — verify they still exist."""

    def test_session_types_array_has_icons(self):
        src = read(NEW_SESSION)
        assert "chatbubble-ellipses-outline" in src
        assert "folder-open-outline" in src
        assert "cube-outline" in src


# ── TUI ──────────────────────────────────────────────────────────────────────


class TestTuiSessionPicker:
    """mucli session picker + creation prompt must show glyphs."""

    def test_session_list_uses_get_session_list_with_type(self):
        src = read(MUCLI)
        assert "get_session_list_with_type" in src, (
            "TUI picker does not call get_session_list_with_type()"
        )

    def test_session_list_has_glyph_prefix(self):
        src = read(MUCLI)
        # Glyph dict must exist in the picker
        assert '"chat": "○"' in src or '"chat": "\\u25cb"' in src, (
            "chat glyph ○ missing from TUI picker"
        )
        assert '"workspace": "▱"' in src or '"workspace": "\\u25b1"' in src, (
            "workspace glyph ▱ missing from TUI picker"
        )
        assert '"container": "◇"' in src or '"container": "\\u25c7"' in src, (
            "container glyph ◇ missing from TUI picker"
        )

    def test_creation_prompt_has_glyphs(self):
        src = read(MUCLI)
        # The creation prompt_choice labels must include glyph prefix
        assert "○ Chat" in src, "creation prompt missing ○ Chat"
        assert "▱ Workspace" in src, "creation prompt missing ▱ Workspace"
        assert "◇ Container" in src, "creation prompt missing ◇ Container"

    def test_welcome_status_shows_glyph(self):
        src = read(MUCLI_DISPLAY)
        assert "session_type_glyph" in src, (
            "welcome status panel does not compute session_type_glyph"
        )
        assert "{session_type_glyph}" in src, (
            "welcome status f-string does not reference session_type_glyph"
        )


# ── SessionManager.get_session_list_with_type ────────────────────────────────


class TestGetSessionListWithType:
    """SessionManager.get_session_list_with_type() reads session_type from disk."""

    def test_method_exists(self):
        src = read(MANAGER)
        assert "def get_session_list_with_type" in src, (
            "get_session_list_with_type method missing from manager.py"
        )

    def test_returns_name_type_pairs(self, tmp_path):
        """Create fake session dirs, verify method reads session_type correctly."""
        from mu.session.manager import SessionManager

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Create three sessions with different types
        for name, st in [("chat-sess", "chat"), ("ws-sess", "workspace"), ("ctr-sess", "container")]:
            sdir = sessions_dir / name
            sdir.mkdir()
            (sdir / "session.json").write_text(
                json.dumps({"variables": {"session_type": st}})
            )

        # Create a session with no session_type (should default to workspace)
        sdir = sessions_dir / "default-sess"
        sdir.mkdir()
        (sdir / "session.json").write_text(json.dumps({"variables": {}}))

        # Create a session with corrupt JSON (should default to workspace)
        sdir = sessions_dir / "corrupt-sess"
        sdir.mkdir()
        (sdir / "session.json").write_text("NOT JSON")

        with patch("mu.session.manager._history_dir", return_value=str(tmp_path)):
            mgr = SessionManager.__new__(SessionManager)
            result = mgr.get_session_list_with_type()

        result_dict = dict(result)
        assert result_dict["chat-sess"] == "chat"
        assert result_dict["ws-sess"] == "workspace"
        assert result_dict["ctr-sess"] == "container"
        assert result_dict["default-sess"] == "workspace"
        assert result_dict["corrupt-sess"] == "workspace"

    def test_returns_sorted_by_name(self, tmp_path):
        from mu.session.manager import SessionManager

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        for name in ["zebra", "alpha", "mike"]:
            sdir = sessions_dir / name
            sdir.mkdir()
            (sdir / "session.json").write_text(
                json.dumps({"variables": {"session_type": "workspace"}})
            )

        with patch("mu.session.manager._history_dir", return_value=str(tmp_path)):
            mgr = SessionManager.__new__(SessionManager)
            result = mgr.get_session_list_with_type()

        names = [n for n, _ in result]
        assert names == sorted(names)