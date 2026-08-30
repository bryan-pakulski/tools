"""Unit tests for the always-on path denylist.

These cover `is_denied_path` and the related path-tokenizer used by the
bash secret-guard. The denylist intentionally errs on the side of caution
— a false positive on a non-secret file is cheaper than a leak.
"""

import os

import pytest

from mu.security.secret_paths import (
    extract_paths_from_command,
    is_denied_path,
)


# ----------------------------------------------------- positive (denied) cases


@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/id_rsa",
        "~/.ssh/id_ed25519",
        "~/.ssh/authorized_keys",
        "~/.ssh/known_hosts",
        "~/.aws/credentials",
        "~/.aws/config",
        "~/.config/gcloud/credentials.db",
        "~/.kube/config",
        "~/.docker/config.json",
        "~/.gnupg/private-keys-v1.d/abc.key",
        "~/.bashrc",
        "~/.zshrc",
        "~/.bash_history",
        "~/.netrc",
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/ssh/sshd_config",
        "/proc/1/environ",
    ],
)
def test_denies_known_secret_locations(path):
    denied, reason = is_denied_path(path)
    assert denied is True, f"{path} should be denied"
    assert reason


@pytest.mark.parametrize(
    "name",
    [
        "id_rsa", "id_rsa.pub",
        "id_ed25519", "id_ed25519.pub",
        "server.pem", "private.key", "store.jks", "wallet.pfx",
        ".env", ".env.local", ".env.production",
        "credentials.json", "service-account.json", "service_account.json",
    ],
)
def test_denies_basename_patterns(tmp_path, name):
    target = tmp_path / name
    target.write_text("dummy")
    denied, _ = is_denied_path(str(target))
    assert denied is True, f"basename {name!r} should be denied"


def test_denies_via_symlink(tmp_path):
    """A symlink inside the workspace pointing at a denied target is denied."""
    real_secret = tmp_path / "real_id_rsa"
    real_secret.write_text("PRIVATE")
    # Symlink target uses the denied basename so the basename check fires
    # regardless of `realpath` capabilities on the platform.
    link = tmp_path / "id_rsa"
    os.symlink(real_secret, link)
    denied, _ = is_denied_path(str(link))
    assert denied is True


# ----------------------------------------------------- negative (allowed) cases


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/foo.txt",
        "/var/log/syslog",
        "src/main.py",
        "docs/README.md",
        "/home/user/project/config.yaml",
        "package.json",
    ],
)
def test_allows_ordinary_paths(path):
    denied, reason = is_denied_path(path)
    assert denied is False, f"{path} should be allowed (got reason: {reason})"


def test_allows_when_override_set():
    """The `security_allow_secret_paths` flag fully bypasses the denylist."""
    denied, _ = is_denied_path(
        "~/.ssh/id_rsa",
        session_variables={"security_allow_secret_paths": True},
    )
    assert denied is False


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_override_accepts_string_truthy(truthy):
    denied, _ = is_denied_path(
        "~/.ssh/id_rsa",
        session_variables={"security_allow_secret_paths": truthy},
    )
    assert denied is False


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", None])
def test_override_rejects_falsy(falsy):
    denied, _ = is_denied_path(
        "~/.ssh/id_rsa",
        session_variables={"security_allow_secret_paths": falsy},
    )
    assert denied is True


# ----------------------------------------------------- command tokenizer


def test_extract_paths_from_command_pulls_path_args():
    cmd = "cat /etc/shadow > /tmp/out && grep -r foo /home/user"
    tokens = list(extract_paths_from_command(cmd))
    assert "/etc/shadow" in tokens
    assert "/tmp/out" in tokens
    assert "/home/user" in tokens


def test_extract_paths_ignores_bare_words():
    """Bare words like 'cat' or 'grep' are yielded as candidate tokens but
    they aren't paths — `is_denied_path` correctly returns False for them."""
    cmd = "cat foo.txt"
    tokens = list(extract_paths_from_command(cmd))
    assert "cat" in tokens
    assert "foo.txt" in tokens
    # Bare words don't match any denied pattern.
    assert is_denied_path("cat") == (False, None)


def test_extract_paths_handles_tilde_expansion():
    cmd = "less ~/.bashrc"
    tokens = list(extract_paths_from_command(cmd))
    assert any("~/.bashrc" in t or "/.bashrc" in t for t in tokens)


def test_extract_paths_skips_flag_tokens():
    """Flags like `-rf` should not be misclassified as paths."""
    cmd = "rm -rf /tmp/foo"
    tokens = list(extract_paths_from_command(cmd))
    assert "-rf" not in tokens
    assert "/tmp/foo" in tokens


def test_extract_paths_with_unbalanced_quotes_falls_back():
    """Malformed input shouldn't raise — fall back to whitespace split."""
    cmd = 'cat "/etc/shadow'  # missing closing quote
    tokens = list(extract_paths_from_command(cmd))
    # The fallback should still surface the suspicious token, possibly
    # quoted; either way, a downstream denied-path check would fire.
    assert any("shadow" in t for t in tokens)


def test_other_user_home_ssh_denied():
    """Round-28 F1: /home/<other>/.ssh subtrees are denied even though
    the invoking user's ~ expansion only covers one account."""
    denied, why = is_denied_path("/home/alice/.ssh/id_rsa")
    assert denied and ".ssh" in why
    denied, why = is_denied_path("/root/.ssh/authorized_keys")
    assert denied


def test_command_substitution_path_extracted():
    """Round-28 F1: `$(echo /home/u/.ssh/id_rsa)` tokenizes as `$(echo`
    + `path)` — paren-splitting must expose the inner path to the
    denylist."""
    from mu.security.secret_paths import extract_paths_from_command

    tokens = list(extract_paths_from_command("cat $(echo /home/u/.ssh/id_rsa)"))
    denied = [t for t in tokens if is_denied_path(t)[0]]
    assert denied, tokens
