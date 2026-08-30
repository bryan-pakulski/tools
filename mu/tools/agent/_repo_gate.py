"""Shared repository-containment gate for codex-driven agent tools.

best_of_codex and verify_html both launch an approval-free ``codex``
subprocess with ``-C <repo>``. A caller-controlled repo path would let the
subprocess read arbitrary host paths (secret files included) and leak them
through the tool result. Every codex-facing tool must resolve the repo and
pass it through :func:`check_repo_containment` first.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple


def resolve_repo(repo: str) -> Optional[str]:
    """Resolve a caller-supplied repo string to an absolute realpath, or
    ``None`` if it does not exist as a directory."""
    repo_abs = os.path.realpath(os.path.abspath(os.path.expanduser(repo)))
    if not os.path.isdir(repo_abs):
        return None
    return repo_abs


def check_repo_gate(repo_abs: str, context: Any) -> Tuple[bool, str]:
    """Capability boundary for approval-free codex tools: the repo must pass
    the same workspace/secret-path gate as filesystem tools.

    Returns (allowed, reason). ``reason`` is empty when allowed.
    """
    try:
        from mu.security.secret_paths import is_denied_path
        from mu.tools._bounds import check_bounds
    except ImportError:  # pragma: no cover — security modules always present
        return True, ""

    session = getattr(context, "session", None)
    # Prefer the dispatch context's folder_context, then the session's.
    folder_context = getattr(context, "folder_context", None) or getattr(
        session, "folder_context", None
    ) or getattr(
        getattr(session, "session_manager", None), "folder_context", None
    )
    session_type = str(
        (getattr(session, "variables", None) or {}).get("session_type", "workspace")
        or "workspace"
    )
    denied, reason = is_denied_path(repo_abs)
    if denied:
        return False, f"blocked: {reason}"
    # Approval-free codex subprocess tools must FAIL CLOSED when invoked
    # through dispatch (context present): without an explicit workspace
    # boundary there is no containment argument, and the historical
    # open-for-workspace-less behavior of check_bounds would let the
    # subprocess read arbitrary host paths.
    if context is not None and not (
        folder_context and getattr(folder_context, "folders", None)
    ):
        return False, (
            "blocked: no workspace folder attached; "
            "codex tools require an explicit workspace boundary"
        )
    if not check_bounds(repo_abs, folder_context, session_type=session_type):
        return False, f"repo outside workspace bounds: {repo_abs}"
    return True, ""
