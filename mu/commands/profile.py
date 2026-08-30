"""Profile slash commands: named bundles of session variables.

Codex-parity feature ([profiles.NAME] in Codex config.toml). A profile
is a JSON file under ~/.mucli/profiles/ capturing a snapshot of the
session's VARIABLE_SCHEMA values. /profile use applies it to the live
session (validated through the same VARIABLE_SCHEMA gate as /set).

/profile            list profiles
/profile save NAME   snapshot current variables into profile NAME
/profile use NAME    apply profile NAME to the live session
/profile show NAME   print the profile's variable values (secrets redacted)
/profile delete NAME remove profile NAME
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any

from utils.config import VARIABLE_SCHEMA, validate_and_cast
from utils.logger import logger

from . import CommandResult, command

PROFILE_DIR = os.path.join(
    os.path.expanduser(os.getenv("MUCLI_HOME", "~/.mucli")), "profiles"
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# Variables never captured: runtime/transient state, not configuration.
_EXCLUDED_VARS = {"session_goal", "session_type", "agent_mode"}

# Secrets never captured to disk nor shown by /profile show. Matched by
# exact name or by suffix so future secret variables stay covered.
_SENSITIVE_SUFFIXES = ("api_key", "api_token", "secret", "password")
_REDACTED = "***redacted***"


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(lowered.endswith(s) for s in _SENSITIVE_SUFFIXES)


def _valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))


def _profile_path(name: str) -> str:
    """Path for a profile, rejecting traversal and symlink escapes.

    Every operation funnels through here with a validated name; realpath
    containment defends against symlinked entries planted inside
    PROFILE_DIR pointing outside it.
    """
    if not _valid_name(name):
        raise ValueError(f"invalid profile name: {name!r}")
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if os.path.islink(path) or os.path.islink(PROFILE_DIR):
        raise ValueError("profile path must not be a symlink")
    real_dir = os.path.realpath(PROFILE_DIR)
    real_path = os.path.realpath(path)
    if os.path.commonpath([real_dir, real_path]) != real_dir:
        raise ValueError("profile path escapes profile directory")
    return path


def _ensure_dir() -> None:
    os.makedirs(PROFILE_DIR, exist_ok=True)


def list_profiles() -> list[str]:
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith(".json")
    )


def load_profile(name: str) -> dict | None:
    """Load a profile. Returns None for missing; raises ValueError for
    invalid/traversal names; raises json/OSError for unreadable files."""
    path = _profile_path(name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"profile {name!r} root is not an object")
    return data


def save_profile(name: str, variables: dict) -> dict:
    """Atomically snapshot variables into profile NAME (mode 0600).

    Sensitive variables are excluded from the snapshot. Write goes to a
    same-directory temp file (fsynced) then os.replace, so an interrupted
    save never truncates an existing profile.
    """
    payload = {
        k: variables[k]
        for k in VARIABLE_SCHEMA
        if k in variables
        and k not in _EXCLUDED_VARS
        and not _is_sensitive(k)
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    _ensure_dir()
    path = _profile_path(name)
    fd, tmp_path = tempfile.mkstemp(
        dir=PROFILE_DIR, prefix=f".{name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return payload


def apply_profile(session: Any, name: str) -> tuple[int, list[str]]:
    """Apply a profile to the live session. Returns (applied_count, skipped)."""
    payload = load_profile(name)
    if payload is None:
        raise FileNotFoundError(f"profile not found: {name}")
    applied, skipped = 0, []
    for key, value in payload.items():
        if key not in VARIABLE_SCHEMA:
            skipped.append(f"{key} (unknown variable)")
            continue
        try:
            session.variables[key] = validate_and_cast(key, value)
            applied += 1
        except (TypeError, ValueError) as exc:
            skipped.append(f"{key}={value!r} ({exc})")
    # Session-variable hooks (e.g. layer budgets) may need re-derivation.
    if applied:
        sync = getattr(session, "sync_runtime_state", None)
        if callable(sync):
            sync()
    return applied, skipped


def _bad_name(name: str) -> CommandResult | None:
    if _valid_name(name):
        return None
    return CommandResult(
        ok=False,
        message=f"Invalid profile name: {name!r} (use letters, digits, ., _, -)",
    )


@command(
    "/profile",
    help=(
        "Named bundles of session variables. Usage: /profile [save|use|show|delete NAME] "
        "— no args lists saved profiles."
    ),
)
def profile_cmd(session: Any, args: str, *, allow_prompt: bool = True) -> CommandResult:
    parts = (args or "").strip().split()
    if not parts:
        names = list_profiles()
        if not names:
            return CommandResult(ok=True, message="No saved profiles. Create one: /profile save <name>")
        return CommandResult(
            ok=True,
            message="Saved profiles:\n" + "\n".join(f"  - {n}" for n in names),
            data={"profiles": names},
        )

    action, rest = parts[0].lower(), parts[1:]
    name = rest[0] if rest else ""

    if action == "save":
        bad = _bad_name(name)
        if bad:
            return bad
        payload = save_profile(name, session.variables)
        return CommandResult(
            ok=True,
            message=f"Profile '{name}' saved ({len(payload)} variables). Apply later: /profile use {name}",
            data={"name": name, "variables": payload},
        )

    if action == "use":
        bad = _bad_name(name)
        if bad:
            return bad
        try:
            applied, skipped = apply_profile(session, name)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
            logger.warning("profile apply failed: %s", exc)
            return CommandResult(ok=False, message=f"Failed to apply profile '{name}': {exc}")
        msg = f"Profile '{name}' applied ({applied} variables)."
        if skipped:
            msg += " Skipped: " + "; ".join(skipped)
        return CommandResult(ok=True, message=msg, data={"applied": applied, "skipped": skipped})

    if action == "show":
        bad = _bad_name(name)
        if bad:
            return bad
        try:
            payload = load_profile(name)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            logger.warning("profile read failed: %s", exc)
            return CommandResult(ok=False, message=f"Failed to read profile '{name}': {exc}")
        if payload is None:
            return CommandResult(ok=False, message=f"Profile not found: {name}")
        shown = {k: (_REDACTED if _is_sensitive(k) else v) for k, v in payload.items()}
        lines = [f"{k} = {v!r}" for k, v in sorted(shown.items())]
        return CommandResult(ok=True, message=f"Profile '{name}':\n" + "\n".join(lines), data=shown)

    if action == "delete":
        bad = _bad_name(name)
        if bad:
            return bad
        try:
            path = _profile_path(name)
        except ValueError as exc:
            return CommandResult(ok=False, message=str(exc))
        if not os.path.isfile(path):
            return CommandResult(ok=False, message=f"Profile not found: {name}")
        os.unlink(path)
        return CommandResult(ok=True, message=f"Profile '{name}' deleted.")

    return CommandResult(
        ok=False,
        message=f"Unknown action '{action}'. Use list (default), save, use, show, or delete.",
    )