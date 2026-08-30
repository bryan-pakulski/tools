"""In-process event bus for the GUI.

Producers (the agent thread, via ``WebUI``) call ``publish_threadsafe``
to push events. Each subscribed browser tab owns one
``asyncio.Queue`` and receives every event.

The loop is bound lazily by ``mu.gui.app.create_app``'s startup hook —
``create_app`` constructs the bus before uvicorn has a running loop,
so the loop must be attached AFTER uvicorn's loop is up. Publishing
before the loop is bound silently drops the event (only the agent
thread fires that early, and it doesn't run until a user message
arrives).

Round-47 hardening (perf-at-scale mission):
* F1 — cross-thread publishes enqueue into ONE bounded ingress deque via
  ``loop.call_soon_threadsafe`` instead of spawning an unbounded chain of
  ``run_coroutine_threadsafe`` coroutine objects. A fast token stream can
  no longer outpace the loop and grow RAM without bound; a slow/absent
  loop coalesces replaceable events (deltas/snapshots) instead of
  queueing every stale one.
* F2 — every event carries a monotonic per-bus ``seq``; the bus keeps a
  bounded replay ring so a reconnecting client can resume from its last
  ``Last-Event-ID`` (SSE ``id:`` field) instead of silently missing the
  events that overflowed a slow subscriber's queue.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# F1: cross-thread producer backlog bound. Replaceable event kinds are
# coalesced when the ingress buffer is full; non-replaceable kinds still
# push the oldest item out (drop-oldest, bounded).
_INGRESS_CAP = 1024
# F2: replay ring size — how far back a reconnecting client can resume.
# Round-48 F4: must EXCEED subscriber queue capacity (2048) so events a
# slow subscriber hasn't consumed yet are still replayable after an
# overflow-triggered reconnect; 512 < 2048 left up to 1536 events
# permanently unrecoverable.
_REPLAY_CAP = 4096
# Event kinds whose OLDER instance can be dropped when the ingress or a
# subscriber queue is full (a newer delta/snapshot supersedes the old one).
_REPLACEABLE_KINDS = frozenset({
    "assistant_delta",
    "memory_snapshot",
    "trace_snapshot",
    "ping",
})


class _Subscriber:
    __slots__ = ("queue", "filters", "last_seq")

    def __init__(self, queue: "asyncio.Queue[Dict[str, Any]]",
                 filters: Optional[set]) -> None:
        self.queue = queue
        self.filters = filters
        self.last_seq = 0


class EventBus:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: List[_Subscriber] = []
        # Per-subscription session filter (codex round-6 F4): the bus used
        # to broadcast every session's events (assistant text, tool
        # results, prompts) to every tab — cross-session content and
        # approval leakage. None = receive all (legacy behavior for
        # trusted single-user loopback use).
        self._session_filters: Dict[asyncio.Queue, Optional[set]] = {}
        # F2: monotonic sequence + bounded replay ring (thread-safe: the
        # seq allocation happens in publish() on the loop thread; the ring
        # is only mutated there).
        self._seq = itertools.count(1)
        self._replay: Deque[Tuple[int, Dict[str, Any]]] = deque(
            maxlen=_REPLAY_CAP
        )
        # F1: cross-thread ingress. Written from producer threads under
        # this lock, drained by the loop task.
        self._ingress: Deque[Dict[str, Any]] = deque()
        self._ingress_lock = threading.Lock()
        self._drain_scheduled = False
        self._dropped_count = 0

    # ------------------------------------------------------------- loop

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach uvicorn's running loop. Called from the FastAPI
        startup hook so cross-thread publishes target the right loop."""
        self._loop = loop

    # ----------------------------------------------------- subscription

    def subscribe(
        self,
        session_name: Optional[str] = None,
        last_event_id: Optional[int] = None,
    ) -> "asyncio.Queue[Dict[str, Any]]":
        """Subscribe. With session_name, the queue only receives events
        belonging to that session (plus session-agnostic events).

        ``last_event_id`` (from the SSE ``Last-Event-ID`` header, F2):
        when present and still inside the replay ring, the missed events
        are re-queued ahead of live traffic.
        """
        # Lazy fallback if subscribe lands before bind_loop fires.
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        queue: asyncio.Queue = asyncio.Queue(maxsize=2048)
        sub = _Subscriber(queue, {session_name} if session_name else None)
        self._subscribers.append(sub)
        self._session_filters[queue] = sub.filters
        if last_event_id is not None:
            self._replay_into(sub, last_event_id)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        for sub in list(self._subscribers):
            if sub.queue is queue:
                self._subscribers.remove(sub)
                break
        self._session_filters.pop(queue, None)

    @staticmethod
    def _event_matches(event: Dict[str, Any], allowed: Optional[set]) -> bool:
        if allowed is None:
            return True
        name = event.get("session_name")
        # Session-agnostic events (no session stamp) always pass.
        return name is None or name in allowed

    def _replay_into(self, sub: _Subscriber, last_event_id: int) -> None:
        """Re-queue replayed events newer than last_event_id (F2)."""
        for seq, event in list(self._replay):
            if seq <= last_event_id:
                continue
            if not self._event_matches(event, sub.filters):
                continue
            try:
                sub.queue.put_nowait(dict(event, replayed=True))
            except asyncio.QueueFull:
                # Reconnect with a too-old cursor — replay would push out
                # live events. Stop replaying; the client resyncs via the
                # next session snapshot.
                break

    def replay_since(self, last_event_id: int) -> List[Dict[str, Any]]:
        """Events after last_event_id still in the ring (oldest first)."""
        return [
            dict(event, seq=seq)
            for seq, event in self._replay
            if seq > last_event_id
        ]

    def oldest_seq(self) -> Optional[int]:
        return self._replay[0][0] if self._replay else None

    # ---------------------------------------------------------- publish

    async def publish(self, event: Dict[str, Any]) -> Dict[str, Any]:
        stamp = dict(event)
        stamp["seq"] = next(self._seq)
        self._replay.append((stamp["seq"], stamp))
        for sub in list(self._subscribers):
            if not self._event_matches(stamp, sub.filters):
                continue
            sub.last_seq = stamp["seq"]
            try:
                sub.queue.put_nowait(stamp)
            except asyncio.QueueFull:
                # F2: slow-consumer overflow now drops the OLDEST queued
                # event (same as before) but the client can detect the gap
                # via the monotonic seq and the reconnect can replay from
                # the ring — an overflow is no longer an unrecoverable
                # silent loss.
                try:
                    sub.queue.get_nowait()
                    sub.queue.put_nowait(stamp)
                except Exception:
                    pass
        return stamp

    def publish_threadsafe(self, event: Dict[str, Any]) -> None:
        """Schedule publish on the bound loop from any thread.

        F1 (round-48): ONE loop callback per burst, not one per event —
        the r47 shape scheduled _drain_ingress on every publish, so a
        fast producer queued thousands of loop callbacks. The
        _drain_scheduled latch schedules only on the false→true
        transition and re-arms in the drain.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            with self._ingress_lock:
                if len(self._ingress) >= _INGRESS_CAP:
                    # Coalesce: drop the oldest REPLACEMENTABLE event to
                    # make room; if none, drop the oldest event outright
                    # (bounded backlog beats unbounded growth).
                    dropped = False
                    for i, pending in enumerate(self._ingress):
                        if pending.get("kind") in _REPLACEABLE_KINDS:
                            del self._ingress[i]
                            dropped = True
                            break
                    if not dropped:
                        self._ingress.popleft()
                    self._dropped_count += 1
                self._ingress.append(event)
                if self._drain_scheduled:
                    return
                self._drain_scheduled = True
            loop.call_soon_threadsafe(self._drain_ingress)
        except RuntimeError:
            pass

    def _drain_ingress(self) -> None:
        """Loop-side drain: hand the batch to ONE drain coroutine.

        Round-48 F3: the r48 first cut called the async publish() directly
        (un-awaited coroutines — caught by probe). publish() is async (it
        may be awaited by in-loop callers), so the drain runs it via
        ensure_future on a SHARED task that covers the whole batch — one
        task per burst, not one per event.
        """
        with self._ingress_lock:
            batch = list(self._ingress)
            self._ingress.clear()
            self._drain_scheduled = False
        if not batch:
            return
        loop = self._loop

        async def _publish_batch() -> None:
            for event in batch:
                try:
                    await self.publish(event)
                except Exception:  # noqa: BLE001 — telemetry must not break
                    return

        try:
            asyncio.ensure_future(_publish_batch(), loop=loop)
        except RuntimeError:
            return