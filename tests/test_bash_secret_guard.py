"""Unit tests for the bash secret-guard pre_tool hook.

The hook short-circuits `bash` / `bash_background` calls whose command
references denied paths or matches one of the risky-command patterns.
"""

import pytest

import mu.agent.secret_guard as guard
from mu.agent.hooks import HookContext, HookRegistry
from mu.agent.secret_guard import install as install_guard


@pytest.fixture
def registry():
    reg = HookRegistry()
    install_guard(reg)
    return reg


def _fire(registry, command, *, tool_name="bash", variables=None):
    ctx = HookContext(
        point="pre_tool",
        tool_name=tool_name,
        tool_args={"command": command},
        variables=variables or {},
    )
    return registry.first_short_circuit("pre_tool", ctx)


# ----------------------------------------------------- denied-path arguments


@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.ssh/id_rsa",
        "less /etc/shadow",
        "cp ~/.aws/credentials /tmp/x",
        "tar czf - ~/.ssh",
        "head -c 64 ~/.bash_history",
        "openssl rsa -in /home/user/private.pem -out /tmp/dec.pem",
    ],
)
def test_blocks_commands_targeting_denied_paths(registry, command):
    result = _fire(registry, command)
    assert result is not None, f"expected block for {command!r}"
    envelope = result.payload
    assert envelope["ok"] is False
    assert envelope["error_code"] == "secret_guard_blocked"


# ----------------------------------------------------- risky patterns


@pytest.mark.parametrize(
    "command",
    [
        "env",
        "printenv",
        "env | grep AWS",
        "cat ~/.zsh_history",
        "find / -name id_rsa",
        "find / -name '*.pem'",
        "cat /proc/1234/environ",
        "cat ~/.ssh/id_ed25519 | base64",
    ],
)
def test_blocks_risky_command_patterns(registry, command):
    result = _fire(registry, command)
    assert result is not None, f"expected block for {command!r}"
    assert result.payload["error_code"] == "secret_guard_blocked"


# ----------------------------------------------------- pass-through


@pytest.mark.parametrize(
    "command",
    [
        "ls /tmp",
        "echo hello",
        "git status",
        "python -c 'print(1)'",
        "grep -r TODO src/",
        "find . -name '*.py'",
    ],
)
def test_allows_ordinary_commands(registry, command):
    result = _fire(registry, command)
    assert result is None, f"unexpected block of {command!r}"


def test_non_bash_tool_unaffected(registry):
    """Read-only tools or unrelated tool calls pass straight through."""
    result = _fire(registry, "anything goes", tool_name="read_file")
    assert result is None


# ----------------------------------------------------- override


def test_override_variable_bypasses_guard(registry):
    """`security_allow_secret_paths=True` lets the guarded command through.
    The output scrubber still runs downstream, so this is opt-in, not
    a wholesale disable."""
    result = _fire(
        registry,
        "cat ~/.ssh/id_rsa",
        variables={"security_allow_secret_paths": True},
    )
    assert result is None


@pytest.mark.parametrize("truthy", ["true", "1", "yes", "on"])
def test_override_accepts_string_values(registry, truthy):
    result = _fire(
        registry,
        "cat ~/.ssh/id_rsa",
        variables={"security_allow_secret_paths": truthy},
    )
    assert result is None


def test_quoted_python_reader_blocked():
    """Round-28 F1: paths hidden inside quoted program text
    (`python3 -c "open('/home/u/.ssh/id_rsa')"`) must be caught by the
    quoted-string scan."""
    reason = guard._check_command(
        'python3 -c "print(open(\'/home/u/.ssh/id_rsa\').read())"'
    )
    assert reason is not None and "denied" in reason
