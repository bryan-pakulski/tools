"""Round-44 F10: bounded summary reads for session listing cards.

A saved session.json embeds the full conversation history; a 200k-message
session is many MB. The listing needs only a handful of metadata fields
(``revision``, ``variables.session_type``, ``container_config``) which the
writer emits near the top of the file — decoding the whole document per
card is the dominant cost of GET /api/sessions once histories grow.
"""

from __future__ import annotations

import json


_MAX_BYTES = 262_144
# Fields the six-field session card needs. Everything else is skipped, so a
# multi-MB `history` value in the head window costs one scan, not one decode.
_CARD_KEYS = ("revision", "variables", "container_config", "thread_meta")


def _skip_value(blob: str, index: int) -> tuple[object, int] | None:
    """Parse one JSON value starting at ``index``; None when truncated."""
    n = len(blob)
    while index < n and blob[index].isspace():
        index += 1
    if index >= n:
        return None
    ch = blob[index]
    if ch == "{":
        depth = 0
        in_string = False
        escaped = False
        for i in range(index, n):
            c = blob[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(blob[index : i + 1]), i + 1
                    except ValueError:
                        return None
        return None
    if ch == "[":
        depth = 0
        in_string = False
        escaped = False
        for i in range(index, n):
            c = blob[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(blob[index : i + 1]), i + 1
                    except ValueError:
                        return None
        return None
    if ch == '"':
        escaped = False
        for i in range(index + 1, n):
            c = blob[i]
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                try:
                    return json.loads(blob[index : i + 1]), i + 1
                except ValueError:
                    return None
        return None
    # Scalar literal: true/false/null/number — ends at , or } at this level.
    end = index
    while end < n and blob[end] not in ",}":
        end += 1
    token = blob[index:end].strip()
    try:
        return json.loads(token), end
    except ValueError:
        return None


def read_session_summary(path: str, *, max_bytes: int = _MAX_BYTES) -> dict:
    """Read the card-relevant top-level fields from session.json cheaply.

    Reads at most ``max_bytes`` from the head of the file, scans the root
    object's top-level ``"key": value`` pairs, and returns only the card
    keys (``revision`` / ``variables`` / ``container_config``) that fit
    completely inside the window. A giant ``history`` value is skipped by
    scanning, not decoding, so a 200k-message file costs one bounded read
    plus a character scan. Returns {} when nothing usable parses; callers
    fall back to a full decode only for legacy layouts that hide the card
    fields beyond the cap.
    """
    try:
        with open(path, "rb") as fh:
            blob = fh.read(max_bytes)
    except OSError:
        return {}
    text = blob.decode("utf-8", errors="replace")
    n = len(text)
    index = 0
    while index < n and text[index].isspace():
        index += 1
    if index >= n or text[index] != "{":
        return {}
    index += 1
    out: dict = {}
    in_string = False
    escaped = False
    key_start = -1
    while index < n:
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                if depth_guard := (key_start >= 0):
                    pass
                key_end = index
            index += 1
            continue
        if ch == '"':
            in_string = True
            escaped = False
            if key_start < 0:
                key_start = index
            index += 1
            continue
        if ch == ":" and key_start >= 0:
            key = text[key_start + 1 : index - 1] if False else None
            # Decode the key properly from the raw slice.
            try:
                key = json.loads(text[key_start : index])
            except ValueError:
                key = None
            value_index = index + 1
            parsed = _skip_value(text, value_index)
            if parsed is None:
                # Value truncated by the window — stop; anything after it
                # cannot be reliably located.
                break
            value, next_index = parsed
            key_name = key if isinstance(key, str) else None
            # Round-45 F7: match json.loads semantics — duplicate keys
            # overwrite (LAST one wins), so never early-break on having
            # found every card key once; keep scanning to root close/trunc.
            if key_name in _CARD_KEYS:
                out[key_name] = value
            index = next_index
            key_start = -1
            continue
        if ch == ",":
            key_start = -1
            index += 1
            continue
        if ch == "}":
            break
        index += 1
    return out
