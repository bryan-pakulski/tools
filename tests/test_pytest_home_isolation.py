"""Regression coverage for suite-wide MuCLI runtime isolation."""

from __future__ import annotations

import os
from pathlib import Path

import utils.config as config
from mu.session.manager import SessionManager


def test_pytest_uses_disposable_mucli_home_outside_real_user_state():
    isolated = Path(os.environ["MUCLI_HOME"]).resolve()
    production = (Path.home() / ".mucli").resolve()

    assert isolated != production
    assert isolated.name.startswith("mucli-pytest-")
    assert Path(config.HISTORY_DIR).resolve() == isolated
    assert Path(config.SESSION_DIR).resolve() == isolated / "sessions"


def test_pytest_mucli_home_is_inherited_by_subprocess_environment():
    # Every subprocess launched without an explicit environment receives this
    # value, keeping CLI/GUI worker tests inside the same disposable boundary.
    assert os.environ.get("MUCLI_HOME") == str(Path(config.HISTORY_DIR))


def test_default_session_writes_stay_inside_disposable_home():
    name = "pytest-home-isolation-probe"
    manager = SessionManager()
    manager.new_session(name, "openai", "test-model")

    isolated_session = Path(config.HISTORY_DIR) / "sessions" / name / "session.json"
    production_session = Path.home() / ".mucli" / "sessions" / name / "session.json"
    assert isolated_session.is_file()
    assert not production_session.exists()
