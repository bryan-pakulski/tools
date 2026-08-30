"""Durable, session-scoped artifact registry.

Artifacts are intentionally separate from ``session.json``.  A corrupt or large
artifact registry therefore cannot prevent a conversation from loading.
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Round-50 F2: parse cache for registry reads — bounded by path count,
# validated by (size, mtime_ns). Registries are append-mostly, so history
# polling hits the cache instead of json.load-ing the file every time.
_REGISTRY_CACHE: dict[str, dict[str, Any]] = {}
_REGISTRY_CACHE_LOCK = threading.Lock()
_REGISTRY_CACHE_CAP = 256


class ArtifactError(ValueError):
    """Raised for invalid artifact inputs or registry operations."""


def _safe_name(name: str) -> str:
    raw = str(name or "").strip().replace("\\", "/")
    candidate = raw.rsplit("/", 1)[-1]
    if not candidate or candidate in {".", ".."} or "\x00" in candidate:
        raise ArtifactError("artifact name must be a non-empty file name")
    return candidate[:240]


class ArtifactRegistry:
    def __init__(self, session_dir: str, *, max_bytes: int | None = None):
        self.session_dir = os.path.abspath(os.path.expanduser(session_dir))
        self.artifacts_dir = os.path.join(self.session_dir, "artifacts")
        self.registry_path = os.path.join(self.artifacts_dir, "registry.json")
        self.max_bytes = int(
            max_bytes
            if max_bytes is not None
            else os.getenv("MUCLI_ARTIFACT_MAX_BYTES", 2000 * 1024 * 1024)
        )
        self._lock = threading.RLock()
        os.makedirs(self.artifacts_dir, exist_ok=True)

    @property
    def session_name(self) -> str:
        return os.path.basename(self.session_dir.rstrip(os.sep))

    def _read(self) -> list[dict[str, Any]]:
        # Round-50 F2: bounded listing was not bounded I/O — every list()
        # json.load-ed the whole registry. Parse results are cached per
        # (path, size, mtime_ns); an unchanged registry (the common case —
        # registries are append-mostly) is a stat + dict hit. Bounded by
        # path count; each cached value IS the entry list the caller may
        # mutate copies of, so we return a shallow copy of the list shell.
        try:
            st = os.stat(self.registry_path)
        except OSError:
            _REGISTRY_CACHE.pop(self.registry_path, None)
            return []
        size, mtime_ns = st.st_size, st.st_mtime_ns
        with _REGISTRY_CACHE_LOCK:
            cached = _REGISTRY_CACHE.get(self.registry_path)
        if cached and cached["size"] == size and cached["mtime_ns"] == mtime_ns:
            return list(cached["entries"])
        try:
            with open(self.registry_path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            value = []
        except (OSError, ValueError):
            # Keep the invalid file for forensic recovery; start a usable view.
            value = []
        entries = value if isinstance(value, list) else []
        with _REGISTRY_CACHE_LOCK:
            if (
                len(_REGISTRY_CACHE) >= _REGISTRY_CACHE_CAP
                and self.registry_path not in _REGISTRY_CACHE
            ):
                _REGISTRY_CACHE.pop(next(iter(_REGISTRY_CACHE)), None)
            _REGISTRY_CACHE[self.registry_path] = {
                "size": size,
                "mtime_ns": mtime_ns,
                "entries": entries,
            }
        return list(entries)

    def _write(self, entries: list[dict[str, Any]]) -> None:
        os.makedirs(self.artifacts_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="registry-", suffix=".json.tmp", dir=self.artifacts_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entries, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.registry_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        # Round-50 F2: writes change (size, mtime) — drop the stale cache
        # entry so the next read re-parses.
        with _REGISTRY_CACHE_LOCK:
            _REGISTRY_CACHE.pop(self.registry_path, None)

    def _descriptor(
        self,
        artifact_id: str,
        name: str,
        size: int,
        mime_type: str,
        *,
        kind: str = "file",
        display: str = "download",
        title: str | None = None,
        height: int | None = None,
        timeline_turn_id: str | None = None,
        timeline_history_index: int | None = None,
        timeline_part_index: int | None = None,
    ) -> dict[str, Any]:
        descriptor = {
            "artifact_id": artifact_id,
            "name": name,
            "size": int(size),
            "mime_type": mime_type or "application/octet-stream",
            "created_at": time.time(),
            "kind": kind,
            "display": display,
            "download_url": (
                f"/api/sessions/{self.session_name}/artifacts/"
                f"{artifact_id}/download"
            ),
        }
        if kind == "visualization":
            descriptor.update(
                {
                    "title": str(title or name)[:240],
                    "height": self._clamp_height(height),
                    "view_url": (
                        f"/api/sessions/{self.session_name}/artifacts/"
                        f"{artifact_id}/view"
                    ),
                }
            )
            turn_id = str(timeline_turn_id or "").strip()[:200]
            if turn_id:
                descriptor["timeline_turn_id"] = turn_id
            for key, value in (
                ("timeline_history_index", timeline_history_index),
                ("timeline_part_index", timeline_part_index),
            ):
                try:
                    parsed = int(value) if value is not None else -1
                except (TypeError, ValueError):
                    parsed = -1
                if parsed >= 0:
                    descriptor[key] = parsed
        return descriptor

    @staticmethod
    def _clamp_height(value: Any) -> int:
        try:
            return max(180, min(1200, int(value or 480)))
        except (TypeError, ValueError):
            return 480

    def _normalize_descriptor(self, entry: dict[str, Any]) -> dict[str, Any]:
        fresh = dict(entry)
        kind = str(fresh.get("kind") or "file").strip().lower()
        fresh["kind"] = kind
        fresh.setdefault("display", "inline" if kind == "visualization" else "download")
        if kind == "visualization":
            fresh.setdefault("title", fresh.get("name") or "Visualization")
            fresh["height"] = self._clamp_height(fresh.get("height"))
            fresh.setdefault(
                "view_url",
                (
                    f"/api/sessions/{self.session_name}/artifacts/"
                    f"{fresh.get('artifact_id')}/view"
                ),
            )
        return fresh

    def add(
        self,
        name: str,
        source_path: str | None = None,
        content: str | bytes | None = None,
        mime_type: str = "application/octet-stream",
        kind: str = "file",
        display: str = "download",
        title: str | None = None,
        height: int | None = None,
        timeline_turn_id: str | None = None,
        timeline_history_index: int | None = None,
        timeline_part_index: int | None = None,
    ) -> dict[str, Any]:
        if (source_path is None) == (content is None):
            raise ArtifactError("provide exactly one of source_path or content")
        safe_name = _safe_name(name)
        artifact_id = uuid.uuid4().hex
        target_dir = os.path.join(self.artifacts_dir, artifact_id)
        target_path = os.path.join(target_dir, safe_name)

        if source_path is not None:
            source = os.path.abspath(os.path.expanduser(str(source_path)))
            if not os.path.isfile(source):
                raise ArtifactError(f"artifact source is not a file: {source}")
            size = os.path.getsize(source)
        else:
            payload = content.encode("utf-8") if isinstance(content, str) else bytes(content or b"")
            size = len(payload)

        if size > self.max_bytes:
            raise ArtifactError(
                f"artifact is {size} bytes; maximum is {self.max_bytes} bytes"
            )

        resolved_mime = (
            mime_type
            if mime_type and mime_type != "application/octet-stream"
            else mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        )

        resolved_kind = str(kind or "file").strip().lower()
        if resolved_kind not in {"file", "visualization"}:
            raise ArtifactError("artifact kind must be file or visualization")
        resolved_display = str(display or "download").strip().lower()
        if resolved_display not in {"download", "inline"}:
            raise ArtifactError("artifact display must be download or inline")
        if resolved_kind == "visualization":
            base_mime = resolved_mime.split(";", 1)[0].strip().lower()
            if base_mime not in {"text/html", "application/xhtml+xml"}:
                raise ArtifactError("visualizations must use an HTML mime type")
            resolved_display = "inline"

        with self._lock:
            os.makedirs(target_dir, exist_ok=False)
            try:
                if source_path is not None:
                    shutil.copy2(source, target_path)
                else:
                    with open(target_path, "wb") as handle:
                        handle.write(payload)
                descriptor = self._descriptor(
                    artifact_id,
                    safe_name,
                    os.path.getsize(target_path),
                    resolved_mime,
                    kind=resolved_kind,
                    display=resolved_display,
                    title=title,
                    height=height,
                    timeline_turn_id=timeline_turn_id,
                    timeline_history_index=timeline_history_index,
                    timeline_part_index=timeline_part_index,
                )
                entries = self._read()
                entries.append(descriptor)
                self._write(entries)
                return dict(descriptor)
            except Exception:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return descriptors newest-first.

        Round-44 F2: ``limit`` bounds the work — the registry is sorted
        newest-first and only the newest ``limit`` entries are stat()'d and
        normalized. A history-page request that wants the 20 newest
        visualizations no longer stats and rewrites a registry holding
        thousands of artifacts. Only the unbounded variant performs the
        prune/rewrite self-heal, because bounded reads cannot vouch for
        entries they never examined.
        """
        with self._lock:
            entries = []
            changed = False
            raw = self._read()
            if limit is not None and limit >= 0:
                # Round-45 F5: legacy entries can lack created_at — a bare
                # created_at sort would drop them all into bucket 0 in file
                # order (append order), which IS the recency order for
                # legacy registries. Sort by (created_at, append_index) so
                # missing timestamps keep append order instead of colliding.
                indexed = list(enumerate(raw))
                indexed.sort(
                    key=lambda pair: (
                        float(pair[1].get("created_at", 0) or 0),
                        pair[0],
                    ),
                    reverse=True,
                )
                raw = [entry for _, entry in indexed[:limit]]
            for entry in raw:
                artifact_id = str(entry.get("artifact_id") or "")
                path = self.resolve_path(artifact_id, _entry=entry)
                if path and os.path.isfile(path):
                    fresh = self._normalize_descriptor(entry)
                    fresh["size"] = os.path.getsize(path)
                    if fresh != entry:
                        changed = True
                    entries.append(fresh)
                else:
                    changed = True
            if changed and limit is None:
                self._write(entries)
            if limit is not None:
                # Bounded reads are already newest-first from the pre-sort.
                return entries
            return sorted(
                entries, key=lambda item: float(item.get("created_at", 0) or 0), reverse=True
            )

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        target = str(artifact_id or "").strip()
        with self._lock:
            for entry in self._read():
                if entry.get("artifact_id") == target:
                    return self._normalize_descriptor(entry)
        return None

    def resolve_path(
        self, artifact_id: str, *, _entry: dict[str, Any] | None = None
    ) -> str | None:
        entry = _entry or self.get(artifact_id)
        if not entry:
            return None
        candidate = Path(self.artifacts_dir, str(entry["artifact_id"]), _safe_name(entry["name"]))
        resolved = candidate.resolve()
        root = Path(self.artifacts_dir).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None
        return str(resolved)

    def remove(self, artifact_id: str) -> bool:
        target = str(artifact_id or "").strip()
        with self._lock:
            entries = self._read()
            kept = [entry for entry in entries if entry.get("artifact_id") != target]
            if len(kept) == len(entries):
                return False
            shutil.rmtree(os.path.join(self.artifacts_dir, target), ignore_errors=True)
            self._write(kept)
            return True
