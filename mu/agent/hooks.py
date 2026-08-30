"""Lifecycle hook registry for the agent loop.

Five hook points cover the loop's natural boundaries:

  * pre_provider_call   - About to send to the model. Use for context
                          assembly, auto-compaction, telemetry.
  * post_provider_call  - Just got a response. Use for usage logging,
                          cost accounting, response decoration.
  * pre_tool            - About to execute a tool. Return a non-None
                          result here to short-circuit (e.g. plan-mode
                          enforcement, sandbox checks). Return None to
                          allow normal execution.
  * post_tool           - Tool returned a result. Use for telemetry,
                          collation, structured-result rewrap.
  * on_stop             - The loop is ending (success, max_iter, error).

Registration:

    @default_registry.register("pre_tool", priority=10)
    def block_writes_in_plan_mode(ctx: HookContext) -> HookResult: ...

Multiple hooks at the same point fire in *priority* order (lower runs
first); within equal priorities the order of registration is preserved.
A hook can either return `None` (no effect), a dict patch to merge into
the loop's state for the next step, or a `HookResult` carrying:

  * action='short_circuit', payload=...   (only meaningful for pre_tool)
  * action='abort', payload=reason
  * action='continue', data={...}

The registry is intentionally lightweight — no async, no thread safety
(hooks fire synchronously from the loop's own thread).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)


HookPoint = str  # one of the literals listed in HOOK_POINTS

HOOK_POINTS: Tuple[str, ...] = (
    "pre_provider_call",
    "post_provider_call",
    "pre_tool",
    "post_tool",
    "on_stop",
)

# Points whose hooks enforce security/policy gates. An exception raised by
# a hook at these points is treated as a denial (fail closed) instead of a
# skip (fail open) — see HookRegistry.fire.
ENFORCEMENT_HOOK_POINTS: frozenset = frozenset({"pre_tool", "pre_provider_call"})


@dataclass
class HookContext:
    """State passed to every hook.

    Not every field is populated at every hook point — `tool_name` and
    `tool_args` are only present at `pre_tool` / `post_tool`. `response`
    is only present at `post_provider_call`. Hooks should treat missing
    fields as `None`.
    """

    point: HookPoint
    session: Any = None
    variables: Dict[str, Any] = field(default_factory=dict)

    # pre_tool / post_tool
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Any = None

    # pre_provider_call
    messages: Optional[List[Any]] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[Any]] = None

    # post_provider_call
    response: Any = None

    # on_stop
    stop_reason: Optional[str] = None

    # Arbitrary metadata bag the loop or earlier hooks may have populated.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Return value from a hook.

    * `action="continue"` (default) - do nothing special.
    * `action="short_circuit"`      - skip normal execution; use `payload`
                                      as the result. Only honored at
                                      `pre_tool`.
    * `action="abort"`              - request the loop to stop after the
                                      current step. `payload` is a reason.
    """

    action: str = "continue"
    payload: Any = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookSpec:
    name: str
    point: HookPoint
    priority: int
    handler: Callable[[HookContext], Optional[HookResult]]


class HookRegistry:
    def __init__(self) -> None:
        self._by_point: Dict[HookPoint, List[HookSpec]] = {p: [] for p in HOOK_POINTS}

    def register(
        self,
        point: HookPoint,
        *,
        name: Optional[str] = None,
        priority: int = 100,
    ) -> Callable[[Callable[[HookContext], Optional[HookResult]]], Callable]:
        """Decorator-style registration:

            @registry.register("pre_tool", priority=10)
            def block_writes(ctx: HookContext) -> HookResult: ...
        """

        if point not in HOOK_POINTS:
            raise ValueError(
                f"Unknown hook point {point!r}; valid: {HOOK_POINTS}"
            )

        def decorator(func: Callable[[HookContext], Optional[HookResult]]):
            spec = HookSpec(
                name=name or func.__name__,
                point=point,
                priority=priority,
                handler=func,
            )
            bucket = self._by_point[point]
            bucket.append(spec)
            bucket.sort(key=lambda s: s.priority)
            return func

        return decorator

    def add(self, spec: HookSpec) -> None:
        if spec.point not in HOOK_POINTS:
            raise ValueError(f"Unknown hook point {spec.point!r}")
        bucket = self._by_point[spec.point]
        bucket.append(spec)
        bucket.sort(key=lambda s: s.priority)

    def remove(self, name: str) -> int:
        """Remove hooks by registered name. Returns count removed."""
        removed = 0
        for point in HOOK_POINTS:
            before = len(self._by_point[point])
            self._by_point[point] = [s for s in self._by_point[point] if s.name != name]
            removed += before - len(self._by_point[point])
        return removed

    def list(self, point: Optional[HookPoint] = None) -> List[HookSpec]:
        if point is None:
            return [s for specs in self._by_point.values() for s in specs]
        return list(self._by_point.get(point, []))

    def fire(self, point: HookPoint, ctx: HookContext) -> List[HookResult]:
        """Fire every hook registered at `point`. Returns each hook's result
        in firing order. Hooks that return `None` are skipped from the list.

        Any exception in a hook is caught and logged but does not stop other
        hooks — the loop must not break because of a buggy user hook.
        """
        if point not in HOOK_POINTS:
            raise ValueError(f"Unknown hook point {point!r}")
        results: List[HookResult] = []
        for spec in self._by_point[point]:
            try:
                out = spec.handler(ctx)
            except Exception as exc:
                # Enforcement points (pre_tool gates, retries) must FAIL
                # CLOSED: a raising security hook (plan-mode guard, secret
                # scrubber) denies the action rather than waving it
                # through. Non-enforcement points stay advisory — a buggy
                # user hook must not break the loop.
                if point in ENFORCEMENT_HOOK_POINTS:
                    logging.getLogger("mucli").error(
                        "enforcement hook %s at %s raised %s; denying action",
                        spec.name,
                        point,
                        exc,
                    )
                    results.append(
                        HookResult(
                            action="short_circuit" if point == "pre_tool" else "abort",
                            payload=(
                                f"Enforcement hook {spec.name!r} failed: {exc}"
                            ),
                        )
                    )
                else:
                    logging.getLogger("mucli").warning(
                        "hook %s at %s raised %s; continuing", spec.name, point, exc
                    )
                continue
            if out is None:
                continue
            if isinstance(out, HookResult):
                results.append(out)
            elif isinstance(out, dict):
                results.append(HookResult(action="continue", data=out))
        return results

    def first_short_circuit(
        self, point: HookPoint, ctx: HookContext
    ) -> Optional[HookResult]:
        """Convenience: fire `point` and return the first short_circuit
        result. Used by the loop at `pre_tool` to detect plan-mode blocks.
        """
        for result in self.fire(point, ctx):
            if result.action == "short_circuit":
                return result
        return None

    def fire_with_signals(
        self, point: HookPoint, ctx: HookContext
    ) -> Tuple[List[HookResult], Optional[HookResult], Optional[HookResult]]:
        """Fire `point` and return (all_results, first_short_circuit, first_abort).

        Hooks fire in priority order; the first short_circuit and the
        first abort are extracted into their own slots so the caller
        doesn't have to walk the list. `short_circuit` is only acted on
        at `pre_tool`; `abort` is a stop-the-loop signal honored
        anywhere (the caller wires it to a session flag the agent loop
        checks at its iteration boundary).
        """
        results = self.fire(point, ctx)
        short: Optional[HookResult] = None
        abort: Optional[HookResult] = None
        for result in results:
            if short is None and result.action == "short_circuit":
                short = result
            if abort is None and result.action == "abort":
                abort = result
            if short is not None and abort is not None:
                break
        return results, short, abort


# A module-level registry. The default loop wiring reads from here so most
# callers don't need to pass a registry around.
default_registry = HookRegistry()


__all__ = [
    "HOOK_POINTS",
    "HookContext",
    "HookRegistry",
    "HookResult",
    "HookSpec",
    "default_registry",
]
