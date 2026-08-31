"""Background delivery scheduler for durable peer-thread wake requests."""

from __future__ import annotations

import glob
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import utils.config as _config

from .coordinator import ThreadCoordinator


class ThreadWakeScheduler:
    """Claim pending wakes and hand each to a bounded runtime callback.

    SQLite claims make multiple MuCLI processes safe: only one scheduler can
    own a wake, while the target thread's execution lease prevents overlapping
    turns. Failed/busy callbacks return the wake to the pending queue.
    """

    def __init__(
        self,
        handler: Callable[[ThreadCoordinator, dict], bool],
        *,
        interval: float = 0.35,
        max_workers: int = 8,
        root: str | None = None,
    ) -> None:
        self.handler = handler
        self.interval = max(0.1, float(interval))
        self.root = os.path.abspath(os.path.expanduser(root or _config.HISTORY_DIR))
        self.runtime_id = f"wake-{os.getpid()}-{uuid.uuid4().hex}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="mucli-thread-wake",
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="mucli-thread-wake-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        self._stop.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=2.0)
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def _coordinators(self):
        pattern = os.path.join(
            self.root, "thread-groups", "tg-*", "coordination.sqlite3"
        )
        for path in glob.glob(pattern):
            group_id = os.path.basename(os.path.dirname(path))
            try:
                yield ThreadCoordinator(group_id, root=self.root)
            except Exception:
                continue

    def _deliver(self, coordinator: ThreadCoordinator, wake: dict) -> None:
        try:
            success = bool(self.handler(coordinator, wake))
        except Exception:
            success = False
        if success:
            coordinator.finish_wake(wake["wake_id"], success=True)
        else:
            self._stop.wait(self.interval)
            coordinator.requeue_wake(wake["wake_id"])

    def _run(self) -> None:
        while not self._stop.is_set():
            claimed = False
            for coordinator in self._coordinators():
                if self._stop.is_set():
                    break
                try:
                    wake = coordinator.claim_wake(self.runtime_id)
                except Exception:
                    continue
                if wake is None:
                    continue
                claimed = True
                self._pool.submit(self._deliver, coordinator, wake)
            self._stop.wait(0.05 if claimed else self.interval)


__all__ = ["ThreadWakeScheduler"]
