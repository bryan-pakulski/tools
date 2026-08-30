
"""Durable registry for immutable files uploaded by a user.

Attachments are inputs, not generated artifacts. They live outside session.json
so large uploads cannot prevent conversation recovery and remain retrievable
after compaction or process restart.
"""
from __future__ import annotations

import hashlib
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


class AttachmentError(ValueError):
    """Raised for invalid attachment inputs or registry operations."""


def _safe_name(name: str) -> str:
    raw = str(name or "").strip().replace("\\", "/")
    candidate = raw.rsplit("/", 1)[-1]
    if not candidate or candidate in {".", ".."} or "\x00" in candidate:
        raise AttachmentError("attachment name must be a non-empty file name")
    return candidate[:240]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class AttachmentRegistry:
    def __init__(self, session_dir: str, *, max_bytes: int | None = None):
        self.session_dir = os.path.abspath(os.path.expanduser(session_dir))
        self.attachments_dir = os.path.join(self.session_dir, "attachments")
        self.registry_path = os.path.join(self.attachments_dir, "registry.json")
        # MUCLI_UNBOUNDED_ATTACHMENT_UPLOADS_V1
        # ``max_bytes`` remains accepted for source compatibility with older
        # callers, but user-file registry uploads are intentionally unbounded.
        # Available storage and any external reverse-proxy limits are the only
        # remaining constraints.
        self.max_bytes: int | None = None
        self._lock = threading.RLock()
        os.makedirs(self.attachments_dir, exist_ok=True)

    @property
    def session_name(self) -> str:
        return os.path.basename(self.session_dir.rstrip(os.sep))

    def _read(self) -> list[dict[str, Any]]:
        try:
            with open(self.registry_path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            return []
        return value if isinstance(value, list) else []

    def _write(self, entries: list[dict[str, Any]]) -> None:
        os.makedirs(self.attachments_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="registry-", suffix=".json.tmp", dir=self.attachments_dir
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

    def _descriptor(
        self,
        attachment_id: str,
        name: str,
        size: int,
        mime_type: str,
        sha256: str,
    ) -> dict[str, Any]:
        return {
            "attachment_id": attachment_id,
            "name": name,
            "size": int(size),
            "mime_type": mime_type or "application/octet-stream",
            "sha256": sha256,
            "created_at": time.time(),
            "download_url": (
                f"/api/sessions/{self.session_name}/attachments/"
                f"{attachment_id}/download"
            ),
        }

    def add(
        self,
        name: str,
        source_path: str,
        mime_type: str = "",
        *,
        move_source: bool = False,
    ) -> dict[str, Any]:
        safe_name = _safe_name(name)
        source = os.path.realpath(os.path.abspath(os.path.expanduser(source_path)))
        if not os.path.isfile(source):
            raise AttachmentError(f"attachment source is not a file: {source}")
        size = os.path.getsize(source)
        digest = _sha256(source)
        resolved_mime = (
            str(mime_type or "").split(";", 1)[0].strip().lower()
            or mimetypes.guess_type(safe_name)[0]
            or "application/octet-stream"
        )

        with self._lock:
            entries = self._read()
            for entry in entries:
                if entry.get("sha256") == digest and entry.get("name") == safe_name:
                    path = self.resolve_path(str(entry.get("attachment_id") or ""), _entry=entry)
                    if path and os.path.isfile(path):
                        fresh = dict(entry)
                        fresh["deduplicated"] = True
                        return fresh

            attachment_id = uuid.uuid4().hex
            target_dir = os.path.join(self.attachments_dir, attachment_id)
            target_path = os.path.join(target_dir, safe_name)
            os.makedirs(target_dir, exist_ok=False)
            try:
                if move_source:
                    os.replace(source, target_path)
                else:
                    shutil.copy2(source, target_path)
                descriptor = self._descriptor(
                    attachment_id,
                    safe_name,
                    os.path.getsize(target_path),
                    resolved_mime,
                    digest,
                )
                entries.append(descriptor)
                self._write(entries)
                return dict(descriptor)
            except Exception:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return attachment descriptors newest-first.

        Round-44 F9: ``limit`` bounds the work — only the newest ``limit``
        entries are stat()'d and normalized. Bounded reads skip the
        prune/rewrite self-heal (they cannot vouch for entries they never
        examined), mirroring ArtifactRegistry.list().
        """
        with self._lock:
            entries: list[dict[str, Any]] = []
            changed = False
            raw = self._read()
            if limit is not None and limit >= 0:
                # Round-45 F5: legacy entries can lack created_at — sort by
                # (created_at, append_index) so missing timestamps keep
                # append order (the recency order for legacy registries)
                # instead of colliding in bucket 0.
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
                attachment_id = str(entry.get("attachment_id") or "")
                path = self.resolve_path(attachment_id, _entry=entry)
                if path and os.path.isfile(path):
                    fresh = dict(entry)
                    fresh["size"] = os.path.getsize(path)
                    fresh.setdefault(
                        "download_url",
                        f"/api/sessions/{self.session_name}/attachments/"
                        f"{attachment_id}/download",
                    )
                    entries.append(fresh)
                    changed = changed or fresh != entry
                else:
                    changed = True
            if changed and limit is None:
                self._write(entries)
            if limit is not None:
                return entries  # pre-sorted newest-first
            return sorted(
                entries,
                key=lambda item: float(item.get("created_at", 0) or 0),
                reverse=True,
            )

    def get(self, attachment_id: str) -> dict[str, Any] | None:
        target = str(attachment_id or "").strip()
        with self._lock:
            for entry in self._read():
                if entry.get("attachment_id") == target:
                    fresh = dict(entry)
                    fresh.setdefault(
                        "download_url",
                        f"/api/sessions/{self.session_name}/attachments/"
                        f"{target}/download",
                    )
                    return fresh
        return None

    def resolve_path(
        self,
        attachment_id: str,
        *,
        _entry: dict[str, Any] | None = None,
    ) -> str | None:
        entry = _entry or self.get(attachment_id)
        if not entry:
            return None
        candidate = Path(
            self.attachments_dir,
            str(entry["attachment_id"]),
            _safe_name(str(entry["name"])),
        ).resolve()
        root = Path(self.attachments_dir).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return str(candidate)

    def remove(self, attachment_id: str) -> bool:
        target = str(attachment_id or "").strip()
        with self._lock:
            entries = self._read()
            kept = [entry for entry in entries if entry.get("attachment_id") != target]
            if len(kept) == len(entries):
                return False
            shutil.rmtree(os.path.join(self.attachments_dir, target), ignore_errors=True)
            self._write(kept)
            return True
