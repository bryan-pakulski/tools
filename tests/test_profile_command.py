"""Unit tests for the /profile slash command (named variable bundles)."""

from __future__ import annotations

import json

import pytest

from mu.commands import profile as prof
from mu.session.session import Session, SessionManager
from providers.base import LLMProvider, ProviderResponse


class DummyProvider(LLMProvider):
    def get_available_models(self):
        return ["m"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        return ProviderResponse(text="ok", parts=[], input_tokens=0, output_tokens=0, total_tokens=0)

    def upload_file(self, path, mime):
        return None


@pytest.fixture()
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("MUCLI_HOME", str(tmp_path))
    return Session(DummyProvider("dummy"), False, "sys", SessionManager())


def test_save_and_use_roundtrip(session, tmp_path):
    session.variables["max_iterations"] = 77
    r1 = prof.profile_cmd(session, "save fast")
    assert r1.ok

    session.variables["max_iterations"] = 5
    r2 = prof.profile_cmd(session, "use fast")
    assert r2.ok
    assert session.variables["max_iterations"] == 77


def test_transient_vars_excluded(session, tmp_path):
    session.variables["session_goal"] = "transient goal"
    session.variables["agent_mode"] = "loop"
    r = prof.profile_cmd(session, "save snap")
    assert r.ok
    payload = json.load(open(prof._profile_path("snap")))
    assert "session_goal" not in payload
    assert "agent_mode" not in payload


def test_use_missing_profile(session, tmp_path):
    r = prof.profile_cmd(session, "use nope")
    assert r.ok is False
    assert "not found" in r.message


def test_invalid_name_rejected(session, tmp_path):
    r = prof.profile_cmd(session, "save ../evil")
    assert r.ok is False


def test_unknown_action(session, tmp_path):
    r = prof.profile_cmd(session, "frobnicate x")
    assert r.ok is False


def test_show_lists_values(session, tmp_path):
    session.variables["max_iterations"] = 42
    prof.profile_cmd(session, "save showme")
    r = prof.profile_cmd(session, "show showme")
    assert r.ok
    assert "max_iterations = 42" in r.message


def test_delete(session, tmp_path):
    prof.profile_cmd(session, "save gone")
    r = prof.profile_cmd(session, "delete gone")
    assert r.ok
    r2 = prof.profile_cmd(session, "use gone")
    assert r2.ok is False


def test_apply_skips_unknown_variables(session, tmp_path, monkeypatch):
    prof._ensure_dir()
    with open(prof._profile_path("corrupt"), "w") as fh:
        json.dump({"max_iterations": 10, "bogus_var": 1}, fh)
    r = prof.profile_cmd(session, "use corrupt")
    assert r.ok
    assert session.variables["max_iterations"] == 10
    assert any("bogus_var" in s for s in r.data.get("skipped", []))
