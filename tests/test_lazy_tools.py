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
    filter_tools_for_mode,
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


def test_mode_filter_keeps_only_the_selected_mode_registry():
    tools = [
        _def("read_file"),
        _def("create_feature"),
        _def("web_search"),
        _def("create_security_report"),
        _def("create_course"),
    ]

    kept = filter_tools_for_mode(tools, "feature")

    assert {tool.name for tool in kept} == {"read_file", "create_feature"}


def test_mode_filter_hides_all_mode_registries_in_default_mode():
    tools = [
        _def("read_file"),
        _def("create_feature"),
        _def("web_search"),
        _def("create_security_report"),
        _def("create_course"),
    ]

    kept = filter_tools_for_mode(tools, "default")

    assert [tool.name for tool in kept] == ["read_file"]


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


def test_mode_phase_excludes_configured_and_loaded_phases_owned_by_other_modes():
    phases = resolve_active_tool_phases(
        {"agent_mode": "feature", "active_tool_phases": ["core", "research"]},
        loaded_phases=["security"],
    )

    assert phases == ["core", "feature"]


def test_load_tools_refuses_mode_phase_outside_owning_mode():
    session = SimpleNamespace(
        variables={"agent_mode": "default"},
        _loaded_tool_phases=[],
    )
    ctx = SimpleNamespace(session=session)
    out = load_tools({"phase": "research"}, ctx)
    assert session._loaded_tool_phases == []
    assert "mode-owned" in out
    assert "research" in out


def test_load_tools_cannot_add_another_mode_phase():
    session = SimpleNamespace(
        variables={"agent_mode": "feature", "active_tool_phases": ["core"]},
        _loaded_tool_phases=[],
    )
    ctx = SimpleNamespace(session=session)

    out = load_tools({"phase": "research"}, ctx)

    assert session._loaded_tool_phases == []
    assert not hasattr(session, "_active_tool_phases")
    assert "mode-owned" in out


def test_load_tools_accepts_current_mode_phase_without_cross_mode_exposure():
    session = SimpleNamespace(
        variables={"agent_mode": "feature", "active_tool_phases": ["core"]},
        _loaded_tool_phases=[],
    )
    ctx = SimpleNamespace(session=session)

    out = load_tools({"phase": "feature"}, ctx)

    assert session._loaded_tool_phases == ["feature"]
    assert session._active_tool_phases == ("core", "feature")
    assert "Active phases: core, feature" in out


def test_load_tools_retains_support_for_non_mode_phases():
    session = SimpleNamespace(
        variables={"agent_mode": "default", "active_tool_phases": ["core"]},
        _loaded_tool_phases=[],
    )
    ctx = SimpleNamespace(session=session)

    out = load_tools({"phase": "experimental"}, ctx)

    assert session._loaded_tool_phases == ["experimental"]
    assert session._active_tool_phases == ("core", "experimental")
    assert "Active phases: core, experimental" in out


def test_load_tools_requires_phase_arg():
    session = SimpleNamespace()
    ctx = SimpleNamespace(session=session)
    out = load_tools({}, ctx)
    assert "phase" in out.lower()
