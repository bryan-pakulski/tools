"""Phased tool exposure / lazy schemas (spec #9).

Specialist phases stay lazy in general, but choosing a strategy mode activates
that mode's registry automatically.
"""

from types import SimpleNamespace

import pytest

from mu.tools.descriptors import (
    TOOL_DESCRIPTORS,
    ToolDescriptor,
    filter_tools_by_phase,
    resolve_active_tool_phases,
)
from mu.tools.memory.handlers import load_tools
from providers.base import ToolDefinition


def _def(name):
    return ToolDefinition(name=name, description="d", parameters={"type": "object"})


def test_filter_keeps_core_tools():
    tools = [_def("read_file"), _def("arxiv_search")]
    TOOL_DESCRIPTORS["read_file"] = ToolDescriptor(
        definition=tools[0], execution_kind="read", preview_policy="none",
        server_policy="session_only", result_mode="raw", handler_key="read_file",
        phase="core",
    )
    TOOL_DESCRIPTORS["arxiv_search"] = ToolDescriptor(
        definition=tools[1], execution_kind="read", preview_policy="none",
        server_policy="session_only", result_mode="raw", handler_key="arxiv_search",
        phase="research",
    )
    kept = filter_tools_by_phase(tools, ["core"])
    assert [t.name for t in kept] == ["read_file"]


def test_filter_includes_phase_when_active():
    tools = [_def("read_file"), _def("arxiv_search")]
    kept = filter_tools_by_phase(tools, ["core", "research"])
    assert {t.name for t in kept} == {"read_file", "arxiv_search"}


def test_filter_always_includes_core():
    # Even if caller forgets "core", it's added.
    tools = [_def("read_file")]
    kept = filter_tools_by_phase(tools, ["research"])
    assert [t.name for t in kept] == ["read_file"]


def test_filter_no_descriptor_defaults_to_core():
    # Tools without a registered descriptor pass (treated as core).
    tools = [_def("mystery_tool")]
    kept = filter_tools_by_phase(tools, ["core"])
    assert [t.name for t in kept] == ["mystery_tool"]


@pytest.mark.parametrize(
    ("mode", "expected_phase"),
    [
        ("feature", "feature"),
        ("research", "research"),
        ("security", "security"),
        ("teacher", "teacher"),
    ],
)
def test_strategy_mode_automatically_activates_its_tool_phase(mode, expected_phase):
    phases = resolve_active_tool_phases(
        {"agent_mode": mode, "active_tool_phases": ["core"]}
    )

    assert phases == ["core", expected_phase]


def test_mode_phase_is_additive_with_configured_and_loaded_phases():
    phases = resolve_active_tool_phases(
        {"agent_mode": "feature", "active_tool_phases": ["core", "research"]},
        loaded_phases=["security"],
    )

    assert phases == ["core", "feature", "research", "security"]


def test_load_tools_adds_phase_to_session():
    session = SimpleNamespace()
    ctx = SimpleNamespace(session=session)
    out = load_tools({"phase": "research"}, ctx)
    assert "research" in session._loaded_tool_phases
    assert "Activated" in out


def test_load_tools_reports_mode_required_phase_as_active():
    session = SimpleNamespace(
        variables={"agent_mode": "feature", "active_tool_phases": ["core"]}
    )
    ctx = SimpleNamespace(session=session)

    out = load_tools({"phase": "research"}, ctx)

    assert session._active_tool_phases == ("core", "feature", "research")
    assert "Active phases: core, feature, research" in out


def test_load_tools_requires_phase_arg():
    session = SimpleNamespace()
    ctx = SimpleNamespace(session=session)
    out = load_tools({}, ctx)
    assert "phase" in out.lower()
