"""Artifact list/download/delete endpoints."""
from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse

from mu.artifact import ArtifactRegistry
from utils.config import HISTORY_DIR

router = APIRouter()

_THEME_BOOTSTRAP = r"""<script id="mucli-visualization-theme">
(() => {
  const allowed = new Set(['light', 'dark']);
  const requested = new URLSearchParams(location.search).get('mucli_theme');
  const initial = allowed.has(requested)
    ? requested
    : (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  const apply = (theme) => {
    if (!allowed.has(theme)) return;
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.__MUCLI_THEME__ = theme;
    window.dispatchEvent(new CustomEvent('mucli-theme-change', { detail: { theme } }));
  };
  apply(initial);
  addEventListener('message', (event) => {
    if (event.data && event.data.type === 'mucli-theme') apply(event.data.theme);
  });
})();
</script>"""


def _inject_visualization_theme(html: str) -> str:
    """Install the shared theme contract before visualization content runs."""
    if 'id="mucli-visualization-theme"' in html:
        return html
    head = re.search(r"<head(?:\s[^>]*)?>", html, flags=re.IGNORECASE)
    if head:
        return html[: head.end()] + _THEME_BOOTSTRAP + html[head.end() :]
    root = re.search(r"<html(?:\s[^>]*)?>", html, flags=re.IGNORECASE)
    if root:
        return html[: root.end()] + _THEME_BOOTSTRAP + html[root.end() :]
    return _THEME_BOOTSTRAP + html


def _registry(session_name: str) -> ArtifactRegistry:
    safe = str(session_name or "").strip()
    if not safe or os.path.basename(safe) != safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid session name")
    session_dir = os.path.join(HISTORY_DIR, "sessions", safe)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail="session not found")
    return ArtifactRegistry(session_dir)


@router.get("/{name}/artifacts")
async def list_artifacts(name: str, response: Response, limit: int = 100):
    # Artifact registries are written by both the host and mounted container
    # workers. Never let a mobile HTTP cache hide a newly published entry.
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    # Round-44 F9: bounded list — newest `limit` descriptors (0 = all).
    # A 200k-message session that publishes artifacts repeatedly no longer
    # transfers and stat()s every descriptor on each refresh.
    registry = _registry(name)
    if limit and limit > 0:
        return {"artifacts": registry.list(limit=limit), "total": len(registry._read())}
    return {"artifacts": registry.list()}


@router.get("/{name}/artifacts/{artifact_id}/view")
async def view_artifact(name: str, artifact_id: str):
    registry = _registry(name)
    descriptor = registry.get(artifact_id)
    path = registry.resolve_path(artifact_id)
    if descriptor is None or path is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    if descriptor.get("kind") != "visualization":
        raise HTTPException(status_code=415, detail="artifact is not a visualization")
    mime_type = str(descriptor.get("mime_type") or "").split(";", 1)[0].lower()
    if mime_type not in {"text/html", "application/xhtml+xml"}:
        raise HTTPException(status_code=415, detail="visualization is not HTML")

    # The document runs with scripts enabled, but without allow-same-origin.
    # It therefore cannot read the parent DOM, cookies, localStorage, or MuCLI
    # API responses. Network access remains available for charting CDNs/data.
    headers = {
        "Cache-Control": "no-store, max-age=0",
        "Content-Security-Policy": (
            "sandbox allow-scripts allow-forms allow-modals allow-downloads; "
            "default-src 'none'; "
            "script-src 'unsafe-inline' 'unsafe-eval' https: http:; "
            "style-src 'unsafe-inline' https: http:; "
            "img-src data: blob: https: http:; "
            "font-src data: https: http:; "
            "connect-src https: http: ws: wss:; "
            "media-src data: blob: https: http:; "
            "worker-src blob:; frame-ancestors 'self'"
        ),
        "Cross-Origin-Resource-Policy": "same-origin",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
    }
    # Inline the tiny bootstrap for normal visualization documents so every
    # artifact receives MuCLI's current light/dark contract. Very large HTML
    # remains streamed; conforming visualizations can still read the same
    # ``mucli_theme`` query parameter without loading the file into memory.
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    # Keep XHTML byte-for-byte valid: the JavaScript bootstrap is authored for
    # HTML parsing (for example, it contains raw ``&&`` tokens).
    if mime_type == "text/html" and size <= 10 * 1024 * 1024:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            html = _inject_visualization_theme(handle.read())
        return HTMLResponse(html, media_type=mime_type, headers=headers)
    return FileResponse(path, media_type=mime_type, headers=headers)


@router.get("/{name}/artifacts/{artifact_id}/download")
async def download_artifact(name: str, artifact_id: str):
    registry = _registry(name)
    descriptor = registry.get(artifact_id)
    path = registry.resolve_path(artifact_id)
    if descriptor is None or path is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(
        path,
        media_type=descriptor.get("mime_type") or "application/octet-stream",
        filename=descriptor.get("name") or "artifact",
    )


@router.delete("/{name}/artifacts/{artifact_id}")
async def delete_artifact(name: str, artifact_id: str, request: Request):
    session = request.app.state.session_by_name(name)
    if session is not None and request.app.state.session_busy_for(name).is_set():
        raise HTTPException(status_code=409, detail="cannot delete artifacts during an active turn")
    if not _registry(name).remove(artifact_id):
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"ok": True, "artifact_id": artifact_id}
