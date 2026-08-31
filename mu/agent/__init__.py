"""Agent-loop infrastructure.

Modules:
  * hooks         — lifecycle hook registry (pre/post tool, pre/post provider, on_stop)
  * parallel      — asyncio-based parallel tool dispatch
  * compactor     — auto-compaction trigger for history → token budget
  * plan_mode     — read-only enforcement for tool dispatch
  * secret_guard  — bash secret-path guard
  * usage_tracker — per-session tool counters, latencies, skill invocations
  * loop          — `AgentLoop` typed wrapper around `Session.send_message`
"""

from .hooks import HookRegistry, HookSpec, default_registry
from .loop import AgentLoop, TurnResult

# Side-effect imports — register built-in hooks at package load so they
# fire from the first turn, not the first call to retry/tools_glue.
from . import compactor  # noqa: F401 — auto_compact_pre_call
from . import plan_mode  # noqa: F401 — plan_mode pre_tool guard
from . import secret_guard  # noqa: F401 — bash secret-path pre_tool guard
from . import thread_guard  # noqa: F401 — peer path ownership guard
from . import usage_tracker  # noqa: F401 — per-session usage counters

__all__ = ["AgentLoop", "HookRegistry", "HookSpec", "TurnResult", "default_registry"]
