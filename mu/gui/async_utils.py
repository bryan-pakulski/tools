"""Small async helpers shared by GUI request handlers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import anyio


_T = TypeVar("_T")


async def run_sync_responsive(func: Callable[..., _T], *args: Any) -> _T:
    """Run blocking work without owning the event loop's default executor.

    The short bounded wait also gives selector-based loops a liveness deadline
    if a cross-thread completion notification is coalesced or missed.  Normal
    completions return immediately; the deadline is only a safety net.
    """

    task = asyncio.create_task(anyio.to_thread.run_sync(func, *args))
    while not task.done():
        await asyncio.wait({task}, timeout=0.02)
    return task.result()


__all__ = ["run_sync_responsive"]
