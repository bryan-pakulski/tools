"""Tool registry descriptors and approval metadata.

This module owns:
  * `ToolDescriptor` / `ToolExecutionContext` dataclasses
  * `TOOLS` / `TOOL_DESCRIPTORS` / `COLLATED_TOOLS` registry data
  * Descriptor introspection (`get_tool_descriptor`, `get_tool_definition`,
    `list_tool_descriptors`, `serialize_tool_descriptor`)
  * Approval metadata (`tool_requires_approval`, `get_modifications`)
  * Factory helpers (`_build_descriptor`, `build_tool_context`)

Every concrete tool registers via the `@tool` decorator in
`mu/tools/<group>/handlers.py`. `TOOLS` and `TOOL_DESCRIPTORS` are
initially empty here — the bridge in `mu/tools/__init__.py` mirrors
each `@tool` registration into both when its module imports.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from providers.base import ToolDefinition


# =================================================================== dataclasses


@dataclass(frozen=True)
class ToolDescriptor:
    definition: ToolDefinition
    execution_kind: str
    preview_policy: str
    server_policy: str
    result_mode: str
    handler_key: str
    error_mode: str = "text_error"
    summary_builder: str | None = None
    # Phased exposure (spec #9): "core" tools are always exposed; specialist
    # phases ("feature", "research", "security", ...) are filtered out of
    # the schema unless `lazy_tools_enabled` is on AND the phase is active.
    # Default "core" so untagged tools stay visible under the phase filter.
    phase: str = "core"
    group: str = ""
    # "always_human" cannot be bypassed by YOLO/container automation.
    approval_policy: str = "default"


@dataclass(frozen=True)
class ToolExecutionContext:
    folder_context: Any
    ui: Any = None
    variables: dict | None = None
    invocation_source: str = "session"
    session: Any = None


def _build_descriptor(
    definition: ToolDefinition,
    *,
    execution_kind: str,
    preview_policy: str,
    server_policy: str,
    result_mode: str,
    handler_key: str,
    error_mode: str = "text_error",
    summary_builder: str | None = None,
    phase: str = "core",
    group: str = "",
    approval_policy: str = "default",
) -> ToolDescriptor:
    return ToolDescriptor(
        definition=definition,
        execution_kind=execution_kind,
        preview_policy=preview_policy,
        server_policy=server_policy,
        result_mode=result_mode,
        handler_key=handler_key,
        error_mode=error_mode,
        summary_builder=summary_builder,
        phase=phase,
        group=group,
        approval_policy=approval_policy,
    )


def build_tool_context(
    folder_context,
    ui=None,
    variables: dict | None = None,
    *,
    invocation_source: str = "session",
    session: Any = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        folder_context=folder_context,
        ui=ui,
        variables=variables,
        invocation_source=invocation_source,
        session=session,
    )


# =================================================================== registry


# Empty placeholders — the `@tool` decorator in `mu/tools/__init__.py`
# mirrors every registration into both maps.
TOOLS: list[ToolDefinition] = []
TOOL_DESCRIPTORS: dict[str, ToolDescriptor] = {}
TOOL_DESCRIPTOR_OVERRIDES: dict[str, dict] = {}


def filter_tools_by_phase(tools, active_phases) -> list:
    """Spec #9: keep only tools whose descriptor ``phase`` is in
    ``active_phases``. Tools without a registered descriptor (phase defaults
    to "core") always pass. Used by the loop's ``active_tools`` filter when
    ``lazy_tools_enabled`` is on; a no-op when all phases are active."""
    try:
        phases = set(active_phases or ["core"])
        if "core" not in phases:
            phases.add("core")
        out = []
        for t in tools:
            desc = TOOL_DESCRIPTORS.get(t.name)
            phase = desc.phase if desc is not None else "core"
            if phase in phases:
                out.append(t)
        return out
    except Exception:  # noqa: BLE001
        return list(tools)


def resolve_active_tool_phases(
    variables: dict | None,
    loaded_phases: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return effective lazy-tool phases for one provider request.

    Explicit ``active_tool_phases`` and phases activated through ``load_tools``
    remain additive.  A strategy mode's declared ``tool_phases`` are always
    included, so selecting feature/research/security/teacher mode cannot leave
    its engine absent from the provider schema.
    """
    values = variables or {}
    phases = {
        str(phase).strip().lower()
        for phase in (values.get("active_tool_phases") or ["core"])
        if str(phase).strip()
    }
    phases.update(
        str(phase).strip().lower()
        for phase in (loaded_phases or [])
        if str(phase).strip()
    )
    phases.add("core")

    mode = str(values.get("agent_mode", "default") or "default").strip().lower()
    try:
        from utils.config import AGENT_MODE_METADATA

        metadata = AGENT_MODE_METADATA.get(mode, {})
        phases.update(
            str(phase).strip().lower()
            for phase in (metadata.get("tool_phases") or [])
            if str(phase).strip()
        )
    except (ImportError, AttributeError, TypeError):
        pass

    return ["core", *sorted(phase for phase in phases if phase != "core")]


_COLLATED_TOOL_NAMES = {
    "get_workspace_details",
    "read_file",
    "search_for_string",
    "search_references",
    "retrieve_relevant_context",
    "get_chunk",
    "list_dir",
    "url_grounding",
    "web_search",
    "arxiv_search",
    "doi_resolve",
    "reddit_search",
    "stackoverflow_search",
    "hackernews_search",
    "read_document",
    "read_attachment",
    "search_attachments",
    "get_tasks",
    "get_current_task",
}


COLLATED_TOOLS = sorted(_COLLATED_TOOL_NAMES)


def _default_result_mode(tool_name: str) -> str:
    return "structured+collated" if tool_name in _COLLATED_TOOL_NAMES else "structured"


def _default_server_policy(tool_name: str) -> str:
    if tool_name in {"flush"}:
        return "session_only"
    return "allowed"


# =================================================================== introspection


def get_tool_definition(tool_name: str) -> ToolDefinition | None:
    descriptor = get_tool_descriptor(tool_name)
    return descriptor.definition if descriptor else None


def get_tool_descriptor(tool_name: str) -> ToolDescriptor | None:
    descriptor = TOOL_DESCRIPTORS.get(tool_name)
    if descriptor is not None:
        return descriptor
    # Fall through to the unified `mu.tools` registry — covers tools that
    # registered through `@tool` before this module's descriptors dict was
    # populated.
    try:
        from mu import tools as _mut
    except ImportError:
        return None
    return _mut.get(tool_name)


def list_tool_descriptors() -> list[ToolDescriptor]:
    return [TOOL_DESCRIPTORS[tool.name] for tool in TOOLS if tool.name in TOOL_DESCRIPTORS]


def serialize_tool_descriptor(tool_name: str) -> dict | None:
    descriptor = get_tool_descriptor(tool_name)
    if not descriptor:
        return None

    definition = descriptor.definition
    return {
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
        "requires_approval": definition.requires_approval,
        "execution_kind": descriptor.execution_kind,
        "preview_policy": descriptor.preview_policy,
        "server_policy": descriptor.server_policy,
        "result_mode": descriptor.result_mode,
        "handler_key": descriptor.handler_key,
        "error_mode": descriptor.error_mode,
        "summary_builder": descriptor.summary_builder,
        "phase": descriptor.phase,
        "group": descriptor.group,
        "approval_policy": descriptor.approval_policy,
    }


# =================================================================== approval metadata


def tool_requires_approval(tool_name: str, args: dict) -> bool:
    """Checks if a tool call requires user approval."""
    tool_def = get_tool_definition(tool_name)
    if not tool_def:
        return False

    if tool_name == "batch_job":
        commands = args.get("commands", [])
        for cmd in commands:
            if tool_requires_approval(cmd.get("tool_name"), cmd.get("tool_args", {})):
                return True
        return False

    return tool_def.requires_approval


def tool_approval_policy(tool_name: str, args: dict) -> str:
    """Return the strongest approval policy for a call and its children."""

    descriptor = get_tool_descriptor(tool_name)
    policy = descriptor.approval_policy if descriptor is not None else "default"
    if tool_name != "batch_job":
        return policy
    for command in args.get("commands", []) or []:
        if not isinstance(command, dict):
            continue
        child_policy = tool_approval_policy(
            str(command.get("tool_name") or ""),
            command.get("tool_args") or {},
        )
        if child_policy == "always_human":
            return child_policy
    return policy


def get_modifications(
    tool_name: str, args: dict, folder_context
) -> list[tuple[str, str, str]]:
    """Returns a list of (original_content, new_content, filename) tuples
    for tools that modify files. Used to render diffs at approval time."""
    if tool_name == "batch_job":
        results = []
        commands = args.get("commands", [])
        for cmd in commands:
            mod = get_modifications(
                cmd.get("tool_name"), cmd.get("tool_args", {}), folder_context
            )
            if mod:
                results.extend(mod)
        return results

    filename = args.get("filename") or args.get("file")
    if not filename:
        return []

    abs_path = os.path.abspath(os.path.expanduser(filename))

    original_content = ""
    if os.path.exists(abs_path):
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                original_content = f.read()
        except Exception:
            pass

    if tool_name == "write_file":
        return [(original_content, args.get("content", ""), filename)]

    elif tool_name == "apply_diff":
        diff = args.get("diff", "")
        if not original_content:
            return [("", "", filename)]

        from mu.tools.file.handlers import _sanitize_diff

        diff = _sanitize_diff(diff, filename)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_orig:
            tmp_orig.write(original_content)
            tmp_orig_path = tmp_orig.name
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_diff:
            tmp_diff.write(diff)
            tmp_diff_path = tmp_diff.name

        try:
            result = subprocess.run(
                [
                    "patch", "-u", "-l", "-F3",
                    tmp_orig_path, "-i", tmp_diff_path, "-o", "-",
                ],
                capture_output=True,
                text=True,
            )
            new_content = (
                result.stdout if result.returncode == 0 else f"ERROR: {result.stderr}"
            )
            return [(original_content, new_content, filename)]
        except FileNotFoundError:
            return [
                (
                    original_content,
                    "ERROR: 'patch' utility not found on system.",
                    filename,
                )
            ]
        except Exception as e:
            return [(original_content, f"ERROR: {e}", filename)]
        finally:
            if os.path.exists(tmp_orig_path):
                os.unlink(tmp_orig_path)
            if os.path.exists(tmp_diff_path):
                os.unlink(tmp_diff_path)

    return []
