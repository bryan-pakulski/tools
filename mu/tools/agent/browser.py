"""browser_snapshot: headless-Chromium screenshot + page text for self-review.

Distinct from research/url_grounding (text extraction for citations):
this tool renders a URL or local HTML file, captures a full-page PNG
into the session artifact registry, and returns a compact page summary.
Primary uses: reviewing published visualizations, debugging the GUI,
checking web layouts the agent just wrote.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from mu.tools import tool
from utils.logger import logger

TOOL_NAME = "browser_snapshot"

_NAV_TIMEOUT_MS = 30_000
_MAX_TEXT_CHARS = 4000


class _SSRFBlocked(Exception):
    """Raised by the page-level request hook when a redirect or subresource
    targets a non-public address (codex round-9 F2)."""


def _envelope(ok: bool, error_code: str | None, message: str, data: dict | None = None, artifacts: list | None = None) -> Dict[str, Any]:
    return {
        "ok": ok,
        "error_code": error_code,
        "message": message,
        "data": data or {},
        "artifacts": artifacts or [],
        "telemetry": {"tool_name": TOOL_NAME},
    }


def _ssrf_check_hostname(hostname: str) -> str | None:
    """SSRF gate for a URL hostname (codex round-9 F2).

    Resolves every hostname — raw IPs and names alike — and refuses the
    request if ANY resolved address is non-public. This closes the
    round-8 gap where a DNS name pointing at loopback/private/link-local
    space (or cloud metadata endpoints) passed the literal-IP check.
    Returns an error message, or None when allowed.
    """
    import ipaddress as _ip
    import socket

    lowered = hostname.lower()
    if lowered in {"metadata.google.internal", "metadata"} or lowered.endswith(
        (".internal", ".local", ".localhost")
    ):
        return f"blocked internal hostname: {hostname}"
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        return f"cannot resolve hostname {hostname}: {exc}"
    seen: set[str] = set()
    for info in infos:
        addr = _ip.ip_address(info[4][0])
        key = str(addr)
        if key in seen:
            continue
        seen.add(key)
        if (
            addr.is_loopback
            or addr.is_link_local
            or addr.is_private
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return f"blocked non-public address: {hostname} -> {key}"
    return None


def _resolve_target(target: str, context=None) -> tuple[str | None, str | None]:
    """Return (playwright_url, error). Local file paths become file:// URLs.

    Security (codex round-8 F2): local targets must pass the same
    capability boundary as filesystem tools (workspace bounds + secret
    path denylist); remote targets must not be loopback/link-local/
    metadata addresses (SSRF).
    """
    target = str(target or "").strip()
    if not target:
        return None, "browser_snapshot requires a url or file path"
    if target.startswith(("http://", "https://")):
        from urllib.parse import urlsplit

        try:
            parsed = urlsplit(target)
            hostname = parsed.hostname or ""
        except ValueError:
            return None, f"unparseable URL: {target}"
        if not hostname:
            return None, f"URL has no hostname: {target}"
        err = _ssrf_check_hostname(hostname)
        if err:
            return None, err
        return target, None
    if target.startswith("file://"):
        path = target[len("file://") :]
    else:
        path = os.path.abspath(os.path.expanduser(target))
    path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    if not os.path.isfile(path):
        return None, f"target not found: {target}"
    # Capability boundary: same gate the file tools use.
    try:
        from mu.security.secret_paths import is_denied_path
        from mu.tools._bounds import check_bounds

        session = getattr(context, "session", None)
        folder_context = getattr(context, "folder_context", None) or getattr(
            session, "folder_context", None
        ) or getattr(
            getattr(session, "session_manager", None), "folder_context", None
        )
        session_type = str(
            (getattr(session, "variables", None) or {}).get("session_type", "workspace")
            or "workspace"
        )
        denied, reason = is_denied_path(path)
        if denied:
            return None, f"blocked: {reason}"
        if not check_bounds(path, folder_context, session_type=session_type):
            return None, f"path outside workspace bounds: {path}"
        # Fail closed for local files when a dispatch context exists but
        # carries no workspace boundary. A direct call with no context at
        # all (unit helpers, ad-hoc resolution) keeps the historical
        # open-but-denylisted behavior; the dispatcher always supplies a
        # context, so production calls are covered.
        if context is not None and not (
            folder_context and getattr(folder_context, "folders", None)
        ):
            return None, (
                "blocked: no workspace folder attached; "
                "local file targets require an explicit workspace boundary"
            )
    except ImportError:
        pass
    return "file://" + path, None


def _registry(context) -> Any | None:
    session = getattr(context, "session", None)
    registry = getattr(session, "artifact_registry", None)
    if registry is not None:
        return registry
    # No live session (unit tests, ad-hoc runs): use a stable scratch
    # session directory. ArtifactRegistry expects a SESSION dir and appends
    # artifacts/ itself — do not pass an artifacts path directly.
    home = os.environ.get("MUCLI_HOME", os.path.expanduser("~/.mucli"))
    from mu.artifact.registry import ArtifactRegistry

    return ArtifactRegistry(os.path.join(home, "sessions", "tool-snapshots"))


@tool(
    name=TOOL_NAME,
    description=(
        "Render a URL or local HTML file in headless Chromium and return a "
        "full-page screenshot registered as a session artifact plus a text "
        "summary of the page. Use to self-review published visualizations, "
        "debug web UI, or verify layouts. Read-only: nothing on the page is "
        "submitted."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "HTTP(S) URL, file:// URL, or local HTML file path.",
            },
            "name": {
                "type": "string",
                "description": "Artifact name for the screenshot (default: page snapshot).",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full scrollable page (default true).",
            },
        },
        "required": ["url"],
    },
    requires_approval=False,
    execution_kind="io",
    result_mode="json",
)
def browser_snapshot(args: Dict[str, Any], context) -> Dict[str, Any]:
    url, err = _resolve_target(args.get("url"), context)
    if err:
        return _envelope(False, "invalid_target", err)
    name = str(args.get("name") or "browser-snapshot").strip() or "browser-snapshot"
    full_page = bool(args.get("full_page", True))

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return _envelope(
            False,
            "playwright_missing",
            "playwright not installed; add the 'browser' extra (pip install .[browser]).",
        )

    tmp_png = None
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:
                return _envelope(
                    False,
                    "chromium_unavailable",
                    f"chromium launch failed: {exc}. Run 'playwright install chromium'.",
                )
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                # Round-9 F2: re-check every network request (redirects and
                # subresources included) — the initial gate only sees the
                # entry URL, but a page can redirect or pull scripts/images
                # from internal addresses.
                from urllib.parse import urlsplit as _split

                def _deny_internal(request_url: str) -> str | None:
                    try:
                        host = _split(request_url).hostname or ""
                    except ValueError:
                        return "unparseable subresource URL"
                    if not host:
                        return None  # data:, blob:, about: — no host to check
                    return _ssrf_check_hostname(host)

                # Route-based enforcement: Playwright event listeners do
                # not abort requests, so raising from "request" would let
                # the fetch complete before the exception surfaced. A
                # context route aborts denied requests before any bytes
                # are sent, then reports through the shared flag.
                ssrf_error: list = []

                def _route_handler(route, request):
                    err = _deny_internal(request.url)
                    if err:
                        if not ssrf_error:
                            ssrf_error.append(err)
                        try:
                            route.abort()
                        except Exception:
                            pass
                        return
                    route.continue_()

                context_routes = getattr(page, "context", None)
                if context_routes is not None and hasattr(context_routes, "route"):
                    context_routes.route("**/*", _route_handler)
                else:  # pragma: no cover - older playwright without routing
                    def _on_request(request):
                        err = _deny_internal(request.url)
                        if err:
                            raise _SSRFBlocked(err)

                    page.on("request", _on_request)
                page.goto(url, wait_until="networkidle", timeout=_NAV_TIMEOUT_MS)
                if ssrf_error:
                    raise _SSRFBlocked(ssrf_error[0])
                title = page.title()
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                    tmp_png = fh.name
                page.screenshot(path=tmp_png, full_page=full_page)
                text = page.evaluate(
                    "() => document.body ? document.body.innerText.slice(0, %d) : ''"
                    % _MAX_TEXT_CHARS
                )
                final_url = page.url
            finally:
                browser.close()
    except _SSRFBlocked as exc:
        if tmp_png and os.path.exists(tmp_png):
            try:
                os.unlink(tmp_png)
            except OSError:
                pass
        return _envelope(False, "ssrf_blocked", f"blocked: {exc}")
    except Exception as exc:
        if tmp_png and os.path.exists(tmp_png):
            try:
                os.unlink(tmp_png)
            except OSError:
                pass
        return _envelope(False, "render_failed", f"page render failed: {exc}")

    try:
        registry = _registry(context)
        descriptor = registry.add(
            name=name if name.endswith(".png") else f"{name}.png",
            source_path=tmp_png,
            mime_type="image/png",
            kind="file",
            display="download",
            title=f"Browser snapshot: {title or url}",
        )
    except Exception as exc:
        if tmp_png and os.path.exists(tmp_png):
            try:
                os.unlink(tmp_png)
            except OSError:
                pass
        return _envelope(False, "artifact_failed", f"could not register screenshot: {exc}")

    # The artifact registry has already copied the PNG into its own store;
    # remove the temp file so successful runs don't leak /tmp files.
    png_size = os.path.getsize(tmp_png)
    try:
        os.unlink(tmp_png)
        tmp_png = None
    except OSError:
        pass

    logger.info("browser_snapshot: %s -> artifact %s", url, descriptor.get("artifact_id"))
    return _envelope(
        True,
        None,
        f"Captured {title or final_url} ({png_size} bytes).",
        data={
            "artifact_id": descriptor.get("artifact_id"),
            "artifact_name": descriptor.get("name"),
            "final_url": final_url,
            "page_title": title,
            "page_text": text or "",
            "full_page": full_page,
        },
        artifacts=[descriptor],
    )
