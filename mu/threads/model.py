"""Stable identities for sessions participating in a thread group.

Threads intentionally reuse the existing session persistence model.  The
metadata here is the small link between ``session.json`` and a group's
coordination journal.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping


_LEGACY_NAMESPACE = uuid.UUID("c3c78a2d-65d1-4f89-a9ec-1af244dd0f98")


def _clean_id(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    if text.startswith(prefix) and len(text) <= 80:
        return text
    return ""


def _legacy_uuid(prefix: str, session_name: str) -> str:
    value = uuid.uuid5(_LEGACY_NAMESPACE, f"{prefix}:{session_name}")
    return f"{prefix}-{value.hex}"


@dataclass(frozen=True)
class ThreadMeta:
    schema_version: int
    thread_id: str
    group_id: str
    title: str
    created_at: float
    parent_thread_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_thread_meta(
    session_name: str,
    value: Mapping[str, Any] | None = None,
) -> ThreadMeta:
    """Normalize saved metadata or derive a deterministic singleton group.

    Deterministic fallback IDs let read-only session listing and a later
    normal save agree without forcing a migration write during startup.
    """

    raw = value if isinstance(value, Mapping) else {}
    thread_id = _clean_id(raw.get("thread_id"), "th")
    group_id = _clean_id(raw.get("group_id"), "tg")
    if not thread_id:
        thread_id = _legacy_uuid("th", session_name)
    if not group_id:
        # A legacy session is deliberately its own explicit family.
        group_id = _legacy_uuid("tg", session_name)
    title = str(raw.get("title") or session_name or "Thread").strip()[:120]
    try:
        created_at = float(raw.get("created_at") or 0.0)
    except (TypeError, ValueError):
        created_at = 0.0
    if created_at <= 0:
        created_at = time.time()
    parent = _clean_id(raw.get("parent_thread_id"), "th")
    return ThreadMeta(
        schema_version=1,
        thread_id=thread_id,
        group_id=group_id,
        title=title,
        created_at=created_at,
        parent_thread_id=parent,
    )


def new_child_thread_meta(parent: ThreadMeta, *, title: str) -> ThreadMeta:
    return ThreadMeta(
        schema_version=1,
        thread_id=f"th-{uuid.uuid4().hex}",
        group_id=parent.group_id,
        title=str(title or "New thread").strip()[:120] or "New thread",
        created_at=time.time(),
        parent_thread_id=parent.thread_id,
    )


__all__ = ["ThreadMeta", "ensure_thread_meta", "new_child_thread_meta"]
