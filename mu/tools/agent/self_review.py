"""verify_html: composed self-review loop for HTML artifacts.

Chains the two Codex-gap tools into one verify step:
1. browser_snapshot renders the page → screenshot artifact + body text.
2. A read-only codex attempt reviews the page SOURCE against an
   optional checklist and the rendered text.
The agent gets a verdict + the screenshot artifact_id, so it can
iterate: edit HTML → verify_html again → compare.

This is the mucli equivalent of Codex's browser self-review habit:
publish → look at it → fix → re-check.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

from mu.tools import tool
from utils.logger import logger

TOOL_NAME = "verify_html"


def _envelope(ok: bool, error_code: str | None, message: str, data: dict | None = None) -> Dict[str, Any]:
    return {
        "ok": ok,
        "error_code": error_code,
        "message": message,
        "data": data or {},
        "artifacts": [],
        "telemetry": {"tool_name": TOOL_NAME},
    }


_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(PASS|ISSUES)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_verdict(review_output: str) -> str:
    """Tolerant verdict scan: any standalone VERDICT: PASS/ISSUES line wins,
    case-insensitive, whitespace-tolerant, ignoring fences/preamble."""
    m = _VERDICT_RE.search(review_output or "")
    if not m:
        return "unknown"
    return "pass" if m.group(1).upper() == "PASS" else "issues"


def _run_codex_review(html_path: str, rendered_text: str, checklist: str, repo: str) -> Dict[str, Any]:
    """One read-only codex attempt reviewing the HTML source. Best-effort:
    a codex failure degrades to 'snapshot only', never blocks the verdict."""
    from mu.tools.agent.best_of import _run_attempt

    prompt_parts = [
        "You are reviewing an HTML page for quality. Be concrete and terse.",
        f"Checklist: {checklist or 'visual correctness, valid markup, no obvious layout bugs, text readable'}",
    ]
    if rendered_text:
        prompt_parts.append(
            "Rendered body text (first ~3000 chars):\n" + rendered_text[:3000]
        )
    prompt_parts.append("Review the HTML file at " + html_path + " against the checklist.")
    prompt_parts.append("Reply with VERDICT: PASS or VERDICT: ISSUES on the first line, then a short bullet list.")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as fh:
        out_path = fh.name
    try:
        result = _run_attempt("\n".join(prompt_parts), repo, out_path, 1)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    return result


@tool(
    name=TOOL_NAME,
    description=(
        "Self-review loop for HTML pages: render the page in headless "
        "Chromium (screenshot registered as a session artifact), extract "
        "body text, then run one read-only codex review of the source "
        "against a checklist. Returns PASS/ISSUES verdict + screenshot "
        "artifact id so you can iterate: edit, re-verify, compare."
    ),
    parameters={
        "type": "object",
        "properties": {
            "html_path": {
                "type": "string",
                "description": "Path to the HTML file to verify.",
            },
            "checklist": {
                "type": "string",
                "description": "What to check, e.g. 'dark theme tokens used, chart has axis labels'.",
            },
            "repo": {
                "type": "string",
                "description": "Repo context for the codex reviewer.",
            },
        },
        "required": ["html_path"],
    },
    requires_approval=False,
    execution_kind="io",
    result_mode="json",
)
def verify_html(args: Dict[str, Any], context) -> Dict[str, Any]:
    from mu.tools.agent.browser import browser_snapshot, _resolve_target

    html_path = str(args.get("html_path") or "").strip()
    url, err = _resolve_target(html_path, context)
    if err:
        return _envelope(False, "invalid_target", err)

    # 1. Render + capture.
    snap = browser_snapshot(
        {"url": html_path, "name": "verify-html", "full_page": True},
        context=context,
    )
    if not snap["ok"]:
        return _envelope(
            False,
            snap.get("error_code") or "snapshot_failed",
            f"render step failed: {snap.get('message')}",
        )
    page_text = (snap["data"] or {}).get("page_text") or ""
    artifact_id = (snap["data"] or {}).get("artifact_id")
    page_title = (snap["data"] or {}).get("page_title") or ""

    # 2. Codex review of the source (best-effort). Repo must pass the same
    # containment gate as best_of_codex (codex round-9 F1): an approval-free
    # tool must not aim codex's sandbox at arbitrary host paths.
    from mu.tools.agent._repo_gate import check_repo_gate, resolve_repo

    checklist = str(args.get("checklist") or "").strip()
    repo = str(args.get("repo") or ".").strip() or "."
    repo_abs = resolve_repo(repo)
    if repo_abs is None:
        return _envelope(False, "bad_repo", f"repo path not found: {repo}")
    allowed, reason = check_repo_gate(repo_abs, context)
    if not allowed:
        return _envelope(False, "bad_repo", reason)
    review = _run_codex_review(os.path.abspath(html_path), page_text, checklist, repo_abs)
    review_output = (review.get("output") or "") if review.get("ok") else ""
    verdict = _parse_verdict(review_output)

    if not review.get("ok"):
        # Degrade gracefully: snapshot succeeded, review unavailable.
        return _envelope(
            True,
            None,
            f"Snapshot captured (artifact {artifact_id}); codex reviewer unavailable "
            f"({review.get('output', 'error')[:120]}). Inspect the screenshot manually.",
            data={
                "verdict": "unreviewed",
                "artifact_id": artifact_id,
                "page_title": page_title,
                "page_text": page_text[:2000],
                "review_output": "",
            },
        )

    logger.info("verify_html: %s -> verdict=%s artifact=%s", html_path, verdict, artifact_id)
    return _envelope(
        True,
        None,
        f"verify_html: {verdict.upper()} — {page_title or html_path} (screenshot artifact {artifact_id})",
        data={
            "verdict": verdict,
            "artifact_id": artifact_id,
            "page_title": page_title,
            "page_text": page_text[:2000],
            "review_output": review_output[:4000],
        },
    )
