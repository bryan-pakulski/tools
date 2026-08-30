"""Hierarchical context assembly for the system prompt.

L2 is state-first: structured runtime state is projected deterministically
from tool envelopes/stores. The rolling conversation summary remains a bounded
semantic residue for information that cannot be derived structurally.
"""
from __future__ import annotations

import json
from typing import Any, Optional
from utils.logger import logger

_MAX_SUBAGENT_DEPTH = 2


def _build_role_layer(role: str, session: Any) -> str:
    """Minimal role metadata; detailed specialist policy lives in spawn.py."""
    role = (role or "").strip().lower()
    if role == "parent":
        try:
            n_active = sum(1 for r in session._subagent_registry.list() if r.status == "running")
        except Exception:
            n_active = 0
        return (
            f"ROLE: ORCHESTRATOR ({n_active} active delegation(s)). "
            "Persistent specialists push material findings/completions through the unread mailbox. "
            "Use await_subagent when you must block; poll_subagent only occasionally; "
            "kill_subagent cancels an unneeded or stuck delegation."
        )
    if role == "child":
        try:
            depth = int(session.variables.get("subagent_depth", 1) or 1)
        except Exception:
            depth = 1
        remaining = max(0, _MAX_SUBAGENT_DEPTH - depth)
        cap = "depth cap reached; do not spawn further sub-agents" if remaining <= 0 else f"you may spawn up to {remaining} further sub-agent level(s)"
        return f"ROLE: SUB-AGENT depth={depth}; persistent specialist policy is in the base system instruction; {cap}."
    return ""


def build_attachment_context(session: Any) -> str:
    registry = getattr(session, "attachment_registry", None)
    if registry is None:
        return ""
    try:
        items = registry.list()
    except Exception:
        return ""
    if not items:
        return ""
    lines = ["Uploaded documents are durable session inputs. Retrieve contents on demand; do not guess them."]
    for item in items[:30]:
        # Attachment metadata is untrusted user input: strip control
        # characters (incl. newlines) so a filename like
        # "x\nIgnore previous instructions..." cannot inject system-level
        # text into this registry block.
        def _clean(val: Any, limit: int = 120) -> str:
            text = "".join(
                ch for ch in str(val or "") if ch.isprintable() or ch == " "
            )
            return text[:limit]

        lines.append(
            json.dumps(
                {
                    "id": _clean(item.get("attachment_id", ""), 64),
                    "name": _clean(item.get("name", "attachment")),
                    "mime_type": _clean(item.get("mime_type", "application/octet-stream")),
                    "size": int(item.get("size", 0) or 0),
                },
                ensure_ascii=True,
            )
        )
    if len(items) > 30:
        lines.append(f"- ... {len(items)-30} more; call list_attachments")
    return "\n".join(lines)[:6000]


def inject_hierarchical_context(session: Any, system_prompt: str, *, cached_skills: Optional[str] = None, cached_context_files: Optional[str] = None) -> str:
    try:
        from utils.runtime_metrics import _current_time_prelude
        system_prompt = f"{_current_time_prelude()}\n\n{system_prompt}".strip()
    except Exception:
        # Defensive: best-effort path must not break the caller.
        logger.debug("Suppressed exception", exc_info=True)

    summary_limit = max(0, int(session.variables.get("conversation_summary_char_limit", 24000) or 12000))
    semantic_residue = str(getattr(session.session_manager, "conversation_summary", "") or "").strip()

    # State and semantic residue share one L2 budget. Structured state gets
    # priority; residue receives only the unused tail. This prevents the new
    # projection from doubling the old L2 budget.
    state_budget = int(summary_limit * 0.70) if summary_limit else 0
    try:
        from mu.session.state_capsule import build_state_capsule
        state_capsule = build_state_capsule(session, max_chars=state_budget, include_goal=False)
        state_capsule = state_capsule[:state_budget] if state_budget else ""
    except Exception:
        state_capsule = ""
    residue_budget = max(0, summary_limit - len(state_capsule))
    if residue_budget:
        semantic_residue = semantic_residue[-residue_budget:].lstrip()
    else:
        semantic_residue = ""

    goal_context = session._build_active_goal_context()
    layers: list[str] = []

    session_type = str(session.variables.get("session_type", "workspace") or "workspace").lower()
    if session_type == "container":
        from mu.container.context import build_container_context
        container_block = build_container_context(session)
        if container_block:
            layers.append(f"LAYER 1C \u2014 Container sandbox:\n{container_block}")

    attachment_context = build_attachment_context(session)
    if attachment_context:
        layers.append("LAYER 1D \u2014 User-uploaded attachment registry (metadata only):\n[budget: 6000 chars | contents retrieved on demand]\n" + attachment_context)

    context_files_block = cached_context_files if cached_context_files is not None else session._build_context_files_block()
    if context_files_block:
        cf_limit = max(0, int(session.variables.get("context_files_max_chars", 8000) or 8000))
        layers.append(f"LAYER 1A \u2014 Workspace context files (AGENTS.md/CLAUDE.md/MUCLI.md/.mu/CONTEXT.md):\n[budget: {cf_limit} chars | whole-file-or-skip, no truncation]\n{context_files_block}")

    skills_block = cached_skills if cached_skills is not None else session._build_skills_block(announce=True)
    if skills_block:
        limit = max(0, int(session.variables.get("skills_max_chars", 6144) or 6144))
        layers.append(f"LAYER 1B \u2014 Installed skills (compact index; bodies auto-load on trigger or via `invoke_skill`):\n[budget: {limit} chars | eviction: drop-tail after auto-expand]\n{skills_block}")

    if state_capsule or semantic_residue:
        parts = [f"[budget: {summary_limit} chars | eviction: keep newest]"]
        if state_capsule:
            parts.append(state_capsule)
        if semantic_residue:
            parts.append("Semantic residue from compacted older conversation (non-authoritative where structured state disagrees):\n" + semantic_residue)
        layers.append("LAYER 2 \u2014 Conversation summary:\n" + "\n\n".join(parts))

    if goal_context:
        layers.append("LAYER 3 \u2014 Active task plan / current goal:\n" + goal_context)

    session_role = str(session.variables.get("session_role", "") or "").strip()
    if session_role:
        role_block = _build_role_layer(session_role, session)
        if role_block:
            layers.append("LAYER 3B \u2014 Agent role:\n" + role_block)

    layers.append("LAYER 5 \u2014 Current turn:\nAlways prioritize the live user message and current-turn tool results. Structured L2 state is authoritative; older semantic residue is fallback context only.")
    layered = f"{system_prompt}\n\nHierarchical runtime context (layered with independent budgets/eviction):\n" + "\n\n".join(layers)
    # Idempotency marker (codex round-9 F4): post-compaction rebuilds call
    # this function with an already-layered prompt. Strip any previous
    # layered block from the base first so L1/L2/L3/L5 are rebuilt once
    # from live state instead of stacking a stale copy beneath a fresh one.
    _LAYER_SENTINEL = "\n\nHierarchical runtime context (layered with independent budgets/eviction):\n"
    first = layered.find(_LAYER_SENTINEL)
    last = layered.rfind(_LAYER_SENTINEL)
    if first != last:
        layered = layered[:first] + _LAYER_SENTINEL + layered[last + len(_LAYER_SENTINEL):]
    return layered


__all__ = ["build_attachment_context", "inject_hierarchical_context"]