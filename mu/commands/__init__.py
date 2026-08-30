"""Slash-command registry for the REPL.

Replaces the 33-branch `if/elif` chain at `mucli.py:1001-2395` with a
decorator-based registry. Each command is registered with one or more
aliases and a help string. Dispatch is by aliasing the leading token of
a user input line.

Usage:

    from mu.commands import command, CommandResult

    @command("/help", "/h", help="Show help")
    def help_cmd(session, args, *, allow_prompt=True):
        ...
        return CommandResult(ok=True, message="...")

    # In the REPL:
    from mu.commands import dispatch
    result = dispatch(session, "/help")
    if result is not None and result.exit:
        break

The registry coexists with the legacy dispatcher: if `dispatch()` returns
`None`, the legacy `mucli.handle_command()` should be invoked as a
fallback. As commands are ported one-by-one, the legacy dispatcher
shrinks.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class CommandResult:
    ok: bool = True
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    exit: bool = False


CommandHandler = Callable[..., Optional[CommandResult]]


@dataclass
class CommandSpec:
    names: Tuple[str, ...]
    help: str
    handler: CommandHandler
    completer: Optional[Callable[..., List[str]]] = None


_REGISTRY: Dict[str, CommandSpec] = {}


def command(
    *names: str,
    help: str,
    completer: Optional[Callable[..., List[str]]] = None,
) -> Callable[[CommandHandler], CommandHandler]:
    """Register a slash command under one or more aliases."""
    if not names:
        raise ValueError("@command requires at least one name")

    def decorator(func: CommandHandler) -> CommandHandler:
        spec = CommandSpec(
            names=tuple(names),
            help=help,
            handler=func,
            completer=completer,
        )
        for alias in names:
            if not alias.startswith("/"):
                raise ValueError(f"Command name must start with '/': {alias!r}")
            _REGISTRY[alias] = spec
        return func

    return decorator


def get(name: str) -> Optional[CommandSpec]:
    return _REGISTRY.get(name)


def list_commands() -> List[CommandSpec]:
    seen: set = set()
    out: List[CommandSpec] = []
    for spec in _REGISTRY.values():
        if id(spec) in seen:
            continue
        seen.add(id(spec))
        out.append(spec)
    return out


def dispatch(
    session: Any,
    line: str,
    *,
    allow_prompt: bool = True,
) -> Optional[CommandResult]:
    """Parse `line` and dispatch to a registered handler.

    Returns `None` when no command matches — the caller should treat the
    line as either user input (no leading slash) or as a legacy command
    that has not been ported yet.
    """

    if not line:
        return None
    stripped = line.strip()
    if not stripped or not stripped.startswith("/"):
        return None

    parts = stripped.split(None, 1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    spec = _REGISTRY.get(cmd)
    if spec is None:
        return None

    return spec.handler(session, args, allow_prompt=allow_prompt)


def _load_command_modules() -> None:
    from . import misc  # noqa: F401
    from . import stats  # noqa: F401
    from . import costs  # noqa: F401
    from . import mode  # noqa: F401
    from . import ollama  # noqa: F401
    from . import skills  # noqa: F401
    from . import docs  # noqa: F401
    from . import session  # noqa: F401
    from . import workspace  # noqa: F401
    from . import runtime  # noqa: F401
    from . import variables  # noqa: F401
    from . import provider  # noqa: F401
    from . import tool  # noqa: F401
    from . import memory  # noqa: F401
    from . import research  # noqa: F401
    from . import feature  # noqa: F401
    from . import teach  # noqa: F401
    from . import trace  # noqa: F401
    from . import shell  # noqa: F401
    from . import goal  # noqa: F401
    from . import prompts  # noqa: F401
    from . import compact  # noqa: F401
    from . import container  # noqa: F401
    from . import job  # noqa: F401 — registers /job and /jobs
    from . import job_diagnostics  # noqa: F401 — registers /jobdiag
    from . import profile  # noqa: F401 — registers /profile
    from . import job_analysis  # noqa: F401 — registers /jobtrace


_load_command_modules()


__all__ = [
    "CommandResult",
    "CommandSpec",
    "command",
    "dispatch",
    "get",
    "list_commands",
]
