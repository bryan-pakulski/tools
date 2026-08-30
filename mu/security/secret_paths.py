"""Always-on protections against accessing or leaking common secret material.

This module is the single source of truth for two related capabilities:

1. **Path denylist** — `is_denied_path(path)` decides whether a filesystem
   path points at material that should never be read or written, regardless
   of workspace scope (SSH keys, cloud creds, shell history, `.env*` files,
   `/etc/shadow`, etc.).

2. **Output scrubber** — `redact_secrets(text)` finds and replaces a curated
   set of high-confidence secret patterns (AWS keys, GitHub tokens, PEM
   blocks, JWTs, ...) before tool output is handed back to the model.

Both are *defense in depth*. They are heuristic and intentionally
conservative — false positives are preferable to leaks, but neither layer
should be the only line of defense.

Override: when the session variable `security_allow_secret_paths` is truthy,
`is_denied_path` always returns `(False, None)`. The scrubber is *not*
overridable — even with the path denylist relaxed, we still redact known
secret patterns from output.
"""

from __future__ import annotations

import fnmatch
import os
import re
from typing import Iterable, Optional, Tuple


# ---------------------------------------------------------------------------
# Path denylist
# ---------------------------------------------------------------------------

# Directories where the entire subtree is denied. Resolved against `~` and
# matched as prefixes of the absolute path.
_DENIED_DIR_ROOTS: Tuple[str, ...] = (
    "~/.ssh",
    "~/.aws",
    "~/.azure",
    "~/.config/gcloud",
    "~/.kube",
    "~/.gnupg",
    "~/.config/gh",
    "~/.cargo/credentials.d",
    "/etc/ssh",
    "/etc/sudoers.d",
)

# Round-28 F1: absolute glob roots covering OTHER users' home dirs (the
# invoking user's ~ only covers one account). fnmatch-matched against
# the resolved path.
_DENIED_DIR_ROOT_GLOBS: Tuple[str, ...] = (
    "/home/*/.ssh",
    "/home/*/.aws",
    "/home/*/.azure",
    "/home/*/.config/gcloud",
    "/home/*/.kube",
    "/home/*/.gnupg",
    "/home/*/.config/gh",
    "/root/.ssh",
    "/root/.aws",
    "/root/.gnupg",
    "/root/.kube",
)

# Exact files (after ~ expansion). Compared by absolute path equality.
_DENIED_EXACT_FILES: Tuple[str, ...] = (
    "~/.docker/config.json",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "~/.bash_history",
    "~/.zsh_history",
    "~/.netrc",
    "~/.npmrc",
    "~/.pypirc",
    "~/.cargo/credentials",
    "~/.cargo/credentials.toml",
    "/etc/shadow",
    "/etc/sudoers",
)

# Basename / glob patterns. Matched against the basename of the path with
# `fnmatch`. Any path whose basename matches is denied regardless of dir.
_DENIED_BASENAME_GLOBS: Tuple[str, ...] = (
    # SSH / GPG keys by conventional name
    "id_rsa", "id_rsa.*",
    "id_ed25519", "id_ed25519.*",
    "id_ecdsa", "id_ecdsa.*",
    "id_dsa", "id_dsa.*",
    "known_hosts", "known_hosts.*",
    "authorized_keys",
    # Certs and key bundles
    "*.pem", "*.key", "*.pfx", "*.p12", "*.jks", "*.keystore",
    # Dotenv files
    ".env", ".env.*",
    # Common credential JSONs
    "credentials*.json",
    "service-account*.json",
    "service_account*.json",
    "gcp-key*.json",
)

# Glob patterns matched against the *full path* (so we can target things like
# `/proc/<pid>/environ` which only make sense as full-path matches).
_DENIED_FULLPATH_GLOBS: Tuple[str, ...] = (
    "/proc/*/environ",
    "/proc/*/cmdline",
)


def _expand(p: str) -> str:
    """Expand `~`, normalize, and resolve symlinks. Returns absolute path.

    Uses `realpath` so a symlink inside the workspace pointing at
    `~/.ssh/id_rsa` is denied just like a direct path.
    """
    expanded = os.path.expanduser(p)
    try:
        return os.path.realpath(expanded)
    except OSError:
        return os.path.abspath(expanded)


def _is_within(child: str, parent: str) -> bool:
    """True when `child` is `parent` or lives somewhere beneath it."""
    parent = parent.rstrip("/")
    return child == parent or child.startswith(parent + "/")


def _override_active(session_variables: Optional[dict]) -> bool:
    if not session_variables:
        return False
    val = session_variables.get("security_allow_secret_paths")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def is_denied_path(
    path: str,
    session_variables: Optional[dict] = None,
) -> Tuple[bool, Optional[str]]:
    """Decide whether a path should be blocked from any read/write access.

    Returns `(denied, reason)`. `reason` is a short human-readable string
    suitable for surfacing to the user / model when blocked.

    The check is always-on by default. Pass `session_variables` with
    `security_allow_secret_paths=True` to bypass (with a single UI warning
    expected to fire elsewhere).
    """
    if not path:
        return False, None
    if _override_active(session_variables):
        return False, None

    abs_path = _expand(str(path))
    # Check the basename of *both* the resolved path and the input path so a
    # symlink named `id_rsa` pointing at `/elsewhere/whatever` is still caught.
    raw_basename = os.path.basename(os.path.expanduser(str(path)).rstrip("/"))
    resolved_basename = os.path.basename(abs_path)

    # 1. Exact files
    for candidate in _DENIED_EXACT_FILES:
        if abs_path == _expand(candidate):
            return True, f"denied secret file ({candidate})"

    # 2. Denied directory subtrees
    for root in _DENIED_DIR_ROOTS:
        root_abs = _expand(root)
        if _is_within(abs_path, root_abs):
            return True, f"denied secret directory ({root})"
    # Round-28 F1: glob roots (other users' home directories).
    for pattern in _DENIED_DIR_ROOT_GLOBS:
        if fnmatch.fnmatch(abs_path, pattern + "/*") or fnmatch.fnmatch(
            abs_path, pattern
        ):
            return True, f"denied secret directory ({pattern})"

    # 3. Full-path globs
    for pattern in _DENIED_FULLPATH_GLOBS:
        if fnmatch.fnmatch(abs_path, pattern):
            return True, f"denied path pattern ({pattern})"

    # 4. Basename globs (`.env`, `*.pem`, `id_rsa`, ...). Check both the
    #    original (pre-symlink) basename and the resolved one — either match
    #    means the path is denied.
    for pattern in _DENIED_BASENAME_GLOBS:
        if fnmatch.fnmatch(raw_basename, pattern) or fnmatch.fnmatch(
            resolved_basename, pattern
        ):
            return True, f"denied filename pattern ({pattern})"

    return False, None


def extract_paths_from_command(command: str) -> Iterable[str]:
    """Tokenize a shell command and yield arguments that look like paths.

    Heuristic — we don't try to fully parse shell syntax. We look for
    tokens that:
      * start with `/`, `~`, or `./`,
      * or contain a `/` after stripping leading punctuation,
      * or are bare basenames matching denied basename globs.

    The denylist itself handles redirect/pipe punctuation (`>>`, `|`, etc.)
    by rejecting it as not-a-path.
    """
    import shlex

    if not command:
        return
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        # Unbalanced quotes — fall back to whitespace split; the denylist
        # will still flag anything that looks pathy.
        tokens = command.split()
    for tok in tokens:
        cleaned = tok.strip()
        if not cleaned or cleaned in {"|", "||", "&", "&&", ";", ">", ">>", "<"}:
            continue
        # Strip a leading `-` so we don't mistake `-rf` for a flag-shaped path.
        if cleaned.startswith("-"):
            continue
        # Round-28 F1: shell punctuation wraps paths in ways that break
        # exact/glob matching — `$(echo /home/u/.ssh/id_rsa)` tokenizes
        # as `$(echo` + `/home/u/.ssh/id_rsa)`. Split on parens and
        # strip wrapping quotes so every inner fragment gets checked.
        for fragment in cleaned.replace("(", " ").replace(")", " ").split():
            frag = fragment.strip("'\"")
            if not frag or frag.startswith("-"):
                continue
            if (
                frag.startswith("/")
                or frag.startswith("~")
                or frag.startswith("./")
                or "/" in frag
            ):
                yield frag
            else:
                yield frag


# ---------------------------------------------------------------------------
# Secret scrubber
# ---------------------------------------------------------------------------

# Each entry: (label, compiled regex). Order matters only for overlapping
# patterns; the PEM block goes first so its multi-line capture isn't fragmented
# by other matches.
_SECRET_PATTERNS = [
    (
        "PEM private key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
    ),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "AWS secret access key",
        re.compile(
            r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?"
        ),
    ),
    ("GitHub PAT", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("GitHub OAuth token", re.compile(r"\bgho_[A-Za-z0-9]{30,}\b")),
    ("GitHub user token", re.compile(r"\bghu_[A-Za-z0-9]{30,}\b")),
    ("GitHub server token", re.compile(r"\bghs_[A-Za-z0-9]{30,}\b")),
    ("GitHub refresh token", re.compile(r"\bghr_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("GitLab PAT", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-api[0-9]{2}-[A-Za-z0-9_\-]{60,}\b")),
    ("OpenAI/sk-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{30,}\b")),
    ("Google API key", re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b")),
    (
        "JWT",
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
        ),
    ),
]


def redact_secrets(text: str) -> Tuple[str, int]:
    """Replace known secret patterns with `[REDACTED:<label>]`.

    Returns `(redacted_text, count)`. The count is the total number of
    substitutions across all patterns. The caller can decide whether to
    append a trailer noting that scrubbing happened.

    Patterns are intentionally specific enough to keep false positives low.
    A non-secret string that happens to look like e.g. `AKIA[16 random
    uppercase alphanumerics]` will still be redacted — that's an acceptable
    cost for not leaking the real thing.
    """
    if not text or not isinstance(text, str):
        return text, 0
    total = 0
    out = text
    for label, pattern in _SECRET_PATTERNS:
        replacement = f"[REDACTED:{label}]"
        out, n = pattern.subn(replacement, out)
        total += n
    return out, total


__all__ = [
    "is_denied_path",
    "extract_paths_from_command",
    "redact_secrets",
]
