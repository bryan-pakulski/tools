"""Capability-gated resolution of durable attachments/artifacts into media.

History stores registry identifiers, never arbitrary paths or repeated base64
payloads. Immediately before a provider request, these helpers re-resolve the
identifier through the registry's path boundary, re-check the active model's
declared modality support, and read the original bytes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from providers.base import MediaData
from utils.model_pricing import input_modality_for_mime


def _provider_accepts(session: Any, descriptor: dict[str, Any]) -> bool:
    provider = getattr(session, "provider", None)
    if provider is None or not hasattr(provider, "supports_input_mime"):
        return False
    mime_type = str(descriptor.get("mime_type") or "application/octet-stream")
    name = str(descriptor.get("name") or "")
    try:
        return bool(provider.supports_input_mime(mime_type, name))
    except Exception:
        return False


def native_media_reference(
    session: Any,
    descriptor: dict[str, Any],
    *,
    registry_kind: str,
) -> Optional[dict[str, Any]]:
    """Return a safe history reference when the active model accepts it."""
    if not _provider_accepts(session, descriptor):
        return None
    id_key = "attachment_id" if registry_kind == "attachment" else "artifact_id"
    item_id = str(descriptor.get(id_key) or "").strip()
    if not item_id:
        return None
    size = max(0, int(descriptor.get("size") or 0))
    provider = getattr(session, "provider", None)
    try:
        max_bytes = max(1, int(provider.native_media_request_limit()))
    except Exception:
        max_bytes = 20 * 1024 * 1024
    if size > max_bytes:
        return None
    return {
        id_key: item_id,
        "name": str(descriptor.get("name") or f"{registry_kind}-{item_id[:8]}"),
        "mime_type": str(descriptor.get("mime_type") or "application/octet-stream"),
        "size": size,
    }


def _resolve_reference(session: Any, reference: dict[str, Any]) -> Optional[MediaData]:
    if not isinstance(reference, dict):
        return None
    if reference.get("attachment_id"):
        registry = getattr(session, "attachment_registry", None)
        item_id = str(reference.get("attachment_id") or "")
        kind = "attachment"
    elif reference.get("artifact_id"):
        registry = getattr(session, "artifact_registry", None)
        item_id = str(reference.get("artifact_id") or "")
        kind = "artifact"
    else:
        return None
    if registry is None or not item_id:
        return None
    try:
        descriptor = registry.get(item_id)
        path = registry.resolve_path(item_id) if descriptor else None
    except Exception:
        return None
    if not descriptor or not path or not os.path.isfile(path):
        return None
    if not _provider_accepts(session, descriptor):
        return None
    provider = getattr(session, "provider", None)
    try:
        max_bytes = max(1, int(provider.native_media_request_limit()))
    except Exception:
        max_bytes = 20 * 1024 * 1024
    size = os.path.getsize(path)
    if size > max_bytes:
        return None
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    return MediaData(
        data=data,
        mime_type=str(descriptor.get("mime_type") or "application/octet-stream"),
        source=f"{kind}:{item_id}",
        display_name=str(descriptor.get("name") or os.path.basename(path)),
    )


def media_resolver_for_session(
    session: Any,
) -> Callable[[dict[str, Any]], Optional[MediaData]]:
    """Build a request-scoped resolver enforcing the aggregate byte limit."""
    provider = getattr(session, "provider", None)
    try:
        remaining = max(1, int(provider.native_media_request_limit()))
    except Exception:
        remaining = 20 * 1024 * 1024

    def resolve(reference: dict[str, Any]) -> Optional[MediaData]:
        nonlocal remaining
        media = _resolve_reference(session, reference)
        if media is None or len(media.data) > remaining:
            return None
        remaining -= len(media.data)
        return media

    return resolve


def tool_media_references(session: Any, result: Any) -> list[dict[str, Any]]:
    """Extract provider-supported binary artifacts from a tool result.

    This is what turns a Chromium PNG artifact into pixels on the next model
    iteration. Text-only models simply keep the normal tool result and artifact
    descriptor; no OCR or hidden conversion service is introduced.
    """
    value = result
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if not isinstance(value, dict):
        return []
    artifacts = value.get("artifacts") or []
    if not isinstance(artifacts, list):
        artifacts = []
    # Composed tools such as verify_html historically return the screenshot's
    # artifact_id in data while keeping their own artifact list empty. Resolve
    # that canonical ID rather than losing the rendered pixels.
    data = value.get("data") or {}
    fallback_id = str(data.get("artifact_id") or "") if isinstance(data, dict) else ""
    if fallback_id and not any(
        str(item.get("artifact_id") or "") == fallback_id
        for item in artifacts
        if isinstance(item, dict)
    ):
        registry = getattr(session, "artifact_registry", None)
        try:
            descriptor = registry.get(fallback_id) if registry is not None else None
        except Exception:
            descriptor = None
        if descriptor:
            artifacts = [*artifacts, descriptor]
    references: list[dict[str, Any]] = []
    for descriptor in artifacts[:8]:
        if not isinstance(descriptor, dict):
            continue
        mime_type = str(descriptor.get("mime_type") or "application/octet-stream")
        modality = input_modality_for_mime(mime_type, str(descriptor.get("name") or ""))
        # Raw text/code artifacts already ride in bounded tool results and are
        # better retrieved with file tools. Media and PDFs benefit from native
        # provider interpretation.
        if modality not in {"image", "audio", "video"} and mime_type != "application/pdf":
            continue
        reference = native_media_reference(
            session, descriptor, registry_kind="artifact"
        )
        if reference is not None:
            references.append(reference)
    return references


__all__ = [
    "media_resolver_for_session",
    "native_media_reference",
    "tool_media_references",
]
