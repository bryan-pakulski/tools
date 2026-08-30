"""Collation buffer for deferring read‑only tool results.

The buffer stores the raw output of tools that only read data (e.g. ``read_file``
or ``search_for_string``). The model receives a short status message during the
agentic loop, and the full payload can be injected later with a *flush* command.

The buffer is persisted as part of the session JSON file, so a session reload
restores any pending collation entries.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple


def _artifact_identifier(tool_name: str, args: Dict[str, Any], result: str) -> str:
    """Stable opaque identifier for one deferred result.

    Exposed to the model and trace. Derived from the tool name, args, and the
    FULL raw result, so identical evidence keeps a stable id across turns.
    """
    payload = json.dumps([tool_name, args, result], default=str, sort_keys=True)
    return "ctx_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _result_byte_count(result: str) -> int:
    """UTF-8 byte length of a deferred result (replacement-char safe)."""
    return len(result.encode("utf-8", errors="replace"))


class CollationBuffer:
    """Collects tool outputs until the user triggers a flush.

    Each entry is a tuple ``(tool_name, args, result)``. ``args`` is stored as a
    plain ``dict`` to make JSON (de)serialisation straightforward.
    """

    def __init__(self) -> None:
        # A deferred result is evidence, not a cache entry. Keep it until the
        # model explicitly delivers or discards it.
        self.entries: List[Tuple[str, Dict[str, Any], str]] = []
        # Cached manifest rows, same order as self.entries. Each entry's
        # artifact id (json serialize + sha256 over the FULL raw result) and
        # UTF-8 byte count are computed exactly once — in add()/from_dict() —
        # instead of being re-derived on every manifest() call. The loop calls
        # manifest()[-1] after every add; recomputing identifiers there made
        # cumulative manifest() cost Theta(N^2) and re-hashed megabyte-scale
        # unchanged evidence on every turn.
        self._manifest_cache: List[Dict[str, Any]] | None = None

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the buffer for storage in the session JSON file."""
        return {
            "entries": [
                {
                    "tool_name": name,
                    "args": args,
                    "result": result,
                }
                for name, args, result in self.entries
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollationBuffer":
        buf = cls()
        for entry in data.get("entries", []):
            buf.entries.append(
                (
                    entry.get("tool_name", ""),
                    entry.get("args", {}),
                    entry.get("result", ""),
                )
            )
        buf._manifest_cache = buf._build_manifest()
        return buf

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def add(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        """Add a result; only explicit model cleanup removes it."""
        self.entries.append((tool_name, args, result))
        if self._manifest_cache is not None:
            self._manifest_cache.append(
                {
                    "id": _artifact_identifier(tool_name, args, result),
                    "tool_name": tool_name,
                    "args": args,
                    "bytes": _result_byte_count(result),
                }
            )
        else:
            # Cache invalidated by a partial removal since the last build —
            # stay invalidated; the next manifest() rebuilds once.
            pass

    def _build_manifest(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": _artifact_identifier(name, args, result),
                "tool_name": name,
                "args": args,
                "bytes": _result_byte_count(result),
            }
            for name, args, result in self.entries
        ]

    def artifact_id(self, index: int) -> str:
        """Stable opaque identifier exposed to the model and trace."""
        if self._manifest_cache is not None:
            return self._manifest_cache[index]["id"]
        name, args, result = self.entries[index]
        return _artifact_identifier(name, args, result)

    def manifest(self) -> List[Dict[str, Any]]:
        if self._manifest_cache is None:
            self._manifest_cache = self._build_manifest()
        return self._manifest_cache

    def last_manifest_entry(self) -> Dict[str, Any] | None:
        """Newest manifest row without recomputing any identifiers.

        Returns ``None`` when the buffer is empty. Callers that only need the
        latest entry (e.g. the deferred-result announcement after ``add()``)
        use this instead of ``manifest()[-1]`` so no per-entry metadata is
        recomputed and no full list is copied.
        """
        if not self.entries:
            return None
        if self._manifest_cache is None:
            self._manifest_cache = self._build_manifest()
        return self._manifest_cache[-1]

    def flush_selected(self, artifact_ids: List[str] | None = None) -> List[Tuple[str, str]]:
        """Deliver selected artifacts (or all) and remove only those entries."""
        wanted = set(artifact_ids or [])
        selected = []
        kept = []
        for i, entry in enumerate(self.entries):
            aid = self.artifact_id(i)
            if not wanted or aid in wanted:
                name, args, result = entry
                header = f"### Collated Data – {name}\n**Parameters:**\n```json\n{json.dumps(args, indent=2, sort_keys=True)}\n```\n**Result:**\n{result}"
                selected.append((aid, header))
            else:
                kept.append(entry)
        if len(kept) != len(self.entries):
            self.entries = kept
            self._manifest_cache = None
        return selected

    def discard(self, artifact_ids: List[str]) -> List[str]:
        """Explicit model-directed cleanup with an auditable return value."""
        wanted = set(artifact_ids)
        removed, kept = [], []
        for i, entry in enumerate(self.entries):
            aid = self.artifact_id(i)
            if aid in wanted:
                removed.append(aid)
            else:
                kept.append(entry)
        if removed:
            self.entries = kept
            self._manifest_cache = None
        return removed