"""Shared helpers for JS-safe session revision transport.

Cross-surface continuity (round-31 F35/F12): the per-session revision is an
unbounded counter persisted in ``session.json``. JSON numbers above
``2^53 - 1`` lose precision once parsed by JavaScript clients (mobile app,
browser GUI), silently corrupting the optimistic-concurrency token carried
in ``If-Match``. Producers therefore clamp: revisions within the JS-safe
range travel as numbers, anything larger travels as its decimal string form.
Consumers that parse with ``Number(value)`` get identical semantics either
way; strict consumers can use the string form to detect the oversized case.

The session revision counter is document-local; hitting 2^53 requires
~9 quadrillion saves, so the string path is a defensive clamp (F12 is a
documented-accepted risk, this keeps it honest).
"""

from __future__ import annotations

# Number.MAX_SAFE_INTEGER (ECMA-262): the largest integer JavaScript can
# represent without precision loss.
JS_SAFE_MAX_REVISION = 2 ** 53 - 1


def js_safe_revision(revision: int) -> int | str:
    """Return a JS-safe transport form of ``revision``.

    - ``revision <= 2^53 - 1``: passed through as ``int`` (JSON number).
    - larger values: decimal string, e.g. ``"9007199254740992"``.

    Raises TypeError for non-int input (defensive: callers must pass the
    SessionManager's integer revision).
    """
    if not isinstance(revision, int) or isinstance(revision, bool):
        raise TypeError(f"revision must be int, got {type(revision).__name__}")
    if revision < 0:
        # Revision counter never goes negative; clamp instead of shipping
        # a malformed token to clients.
        return 0
    if revision > JS_SAFE_MAX_REVISION:
        return str(revision)
    return revision

def parse_revision_token(raw: object) -> int:
    """Parse a revision transported via js_safe_revision back into an int.

    Accepts int (JSON number form) and str (string form for values above
    2^53-1). Raises ValueError on anything else so callers can 400/409 the
    request rather than silently mis-compare.
    """
    if isinstance(raw, bool):
        raise ValueError("revision token must be an integer")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if not raw or not raw.lstrip("-").isdigit():
            raise ValueError(f"revision token is not an integer: {raw!r}")
        return int(raw)
    raise ValueError(f"revision token must be int or str, got {type(raw).__name__}")