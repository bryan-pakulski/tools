"""Tool-result envelope construction and normalization.

The canonical envelope shape every tool result must satisfy:

    {
        "ok":          bool,
        "error_code":  str | None,
        "message":     str,
        "data":        Any,
        "artifacts":   list,
        "hint":        str | None,
        "retryable":   bool,
        "telemetry":   {"tool_name": str, ...},
    }

Tests in `tests/test_harness_layers.py` and `tests/test_envelope_hints_retry.py`
pin this shape. Handlers can return any of:

  * a plain string                           — wrapped automatically
  * a dict with a partial `ok`/`error_code`  — completed via `_ensure_envelope_shape`
  * a fully-formed envelope dict             — passed through unchanged
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------- error-code heuristics


def infer_tool_error_code(tool_name: str, result: Any) -> Optional[str]:
    """Best-effort mapping from a raw handler-output string to an
    `error_code` token. Used when a handler returned a plain string and
    didn't tell us *why* it failed. The matcher is intentionally
    conservative — if nothing matches, we return None and leave the
    caller's `ok` flag as the source of truth."""
    raw_text = str(result or "")
    lowered = raw_text.lower()

    if not raw_text:
        return None

    if "disabled for this session" in lowered:
        return "access_denied"
    if "access denied" in lowered or "outside boundaries" in lowered:
        return "access_denied"
    if "nested batch_job not allowed" in lowered:
        return "unsupported"
    if "unknown tool" in lowered or "tool_name missing" in lowered:
        return "not_found"
    if "does not exist" in lowered or "no such file" in lowered:
        return "not_found"
    if "field '" in lowered and "required" in lowered:
        return "invalid_args"
    if "argument is empty" in lowered or "must be a list" in lowered:
        return "invalid_args"
    if (
        "malformed patch" in lowered
        or "'patch' utility not found" in lowered
        or "patch: ****" in lowered
        or "only garbage was found in the patch input" in lowered
    ):
        return "preview_failed"
    if raw_text.startswith("Error"):
        return "execution_failed"
    return None


def _hint_lookup(
    tool_name: str, error_code: Optional[str]
) -> Tuple[Optional[str], bool]:
    """Return `(hint, retryable)` for the (tool, error_code) pair.

    Imported lazily so this module stays load-cheap and doesn't pull in
    the hint registry on cold import."""
    if not error_code:
        return None, False
    try:
        from mu.tools._hints import hint_for, retryable_for_code

        return hint_for(tool_name, error_code), retryable_for_code(error_code)
    except Exception:  # pragma: no cover — defensive
        return None, False


# ---------------------------------------------------------------- builder


def _build_tool_envelope(
    *,
    tool_name: str,
    ok: bool,
    message: str,
    data: Any = None,
    error_code: Optional[str] = None,
    artifacts: Optional[list] = None,
    telemetry: Optional[dict] = None,
    hint: Optional[str] = None,
    retryable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Construct a canonical tool-result envelope.

    Auto-derives `hint` and `retryable` from the hint registry when the
    envelope represents a failure and the caller didn't supply them.
    """
    if not ok:
        derived_hint, derived_retryable = _hint_lookup(tool_name, error_code)
        if hint is None:
            hint = derived_hint
        if retryable is None:
            retryable = derived_retryable
    else:
        if retryable is None:
            retryable = False
    return {
        "ok": bool(ok),
        "error_code": error_code,
        "message": str(message or ""),
        "data": data if data is not None else {},
        "artifacts": artifacts or [],
        "hint": hint,
        "retryable": bool(retryable),
        "telemetry": {
            "tool_name": tool_name,
            **(telemetry or {}),
        },
    }


# ---------------------------------------------------------------- normalizer


def _envelope_from_handler_result(
    tool_name: str, handler_result: Any
) -> Dict[str, Any]:
    """Normalize whatever a handler returned into the canonical envelope.

    Accepts strings, dicts with `ok`, fully-formed 6-key envelopes, and
    tool-local JSON (e.g. `{"success": ..., "error": ...}`). Backfills
    `hint` and `retryable` from the hint registry when missing.
    """

    def _ensure_envelope_shape(payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        if "error_code" not in out:
            out["error_code"] = (
                None
                if out.get("ok")
                else infer_tool_error_code(tool_name, out)
            )
        if "message" not in out:
            if isinstance(out.get("error"), str):
                out["message"] = out.get("error", "")
            elif out.get("ok"):
                out["message"] = "ok"
            else:
                out["message"] = str(out.get("error") or "")
        if "data" not in out:
            out["data"] = {}
        if "artifacts" not in out:
            out["artifacts"] = []
        # Backfill hint + retryable from the registry when missing. Handlers
        # that emit their own structured envelopes can override either
        # value; we never clobber a non-None hint or an explicit retryable.
        if not out.get("ok"):
            derived_hint, derived_retryable = _hint_lookup(
                tool_name, out.get("error_code")
            )
            if "hint" not in out or out.get("hint") is None:
                out["hint"] = derived_hint
            if "retryable" not in out:
                out["retryable"] = derived_retryable
        else:
            out.setdefault("hint", None)
            out.setdefault("retryable", False)
        telemetry = out.get("telemetry")
        out["telemetry"] = telemetry if isinstance(telemetry, dict) else {}
        out["telemetry"].setdefault("tool_name", tool_name)
        return out

    if isinstance(handler_result, dict):
        # Already-canonical 6-key envelope — backfill hint/retryable.
        if {
            "ok",
            "error_code",
            "message",
            "data",
            "artifacts",
            "telemetry",
        }.issubset(handler_result.keys()):
            out = dict(handler_result)
            if not out.get("ok"):
                derived_hint, derived_retryable = _hint_lookup(
                    tool_name, out.get("error_code")
                )
                if "hint" not in out or out.get("hint") is None:
                    out["hint"] = derived_hint
                if "retryable" not in out:
                    out["retryable"] = derived_retryable
            else:
                out.setdefault("hint", None)
                out.setdefault("retryable", False)
            return out
        if "ok" in handler_result:
            return _ensure_envelope_shape(handler_result)
        # Free-form dict — wrap as envelope, inferring success from
        # error-code matching.
        error_code = infer_tool_error_code(
            tool_name, json.dumps(handler_result)
        )
        return _build_tool_envelope(
            tool_name=tool_name,
            ok=error_code is None,
            error_code=error_code,
            message=json.dumps(handler_result, sort_keys=True),
            data=handler_result,
        )

    raw_text = str(handler_result or "")
    parsed_data = None
    if raw_text:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                if {
                    "ok",
                    "error_code",
                    "message",
                    "data",
                    "artifacts",
                    "telemetry",
                }.issubset(parsed.keys()):
                    return parsed
                if "ok" in parsed:
                    return _ensure_envelope_shape(parsed)
                # Tool-local structured JSON payload (e.g. `{"success": ...}`).
                success_value = parsed.get("success")
                parsed_error = parsed.get("error")
                if success_value is not None or parsed_error:
                    ok = bool(success_value) and not parsed_error
                    error_code = (
                        None
                        if ok
                        else infer_tool_error_code(
                            tool_name, parsed_error or raw_text
                        )
                    )
                    envelope = _build_tool_envelope(
                        tool_name=tool_name,
                        ok=ok,
                        error_code=error_code,
                        message=str(
                            parsed_error or ("success" if ok else raw_text)
                        ),
                        data=parsed,
                    )
                    envelope.update(parsed)
                    return envelope
            parsed_data = parsed
        except Exception:
            parsed_data = None

    error_code = infer_tool_error_code(tool_name, raw_text)
    return _build_tool_envelope(
        tool_name=tool_name,
        ok=error_code is None,
        error_code=error_code,
        message=raw_text,
        data=parsed_data if isinstance(parsed_data, (dict, list)) else {},
    )


__all__ = [
    "infer_tool_error_code",
    "_build_tool_envelope",
    "_envelope_from_handler_result",
]
