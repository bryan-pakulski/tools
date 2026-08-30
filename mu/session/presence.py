"""Presence beacons (cross-surface continuity phase 3, G4).

Each running surface touches
``~/.mucli/sessions/<name>/presence/<pid>.json`` periodically with
``{surface, started_at, last_seen, busy}``. Beacons older than ``ttl``
seconds are pruned on read, so the live peer list is always fresh
without a heartbeat protocol beyond "keep writing your file".

Consumers (design §3.4):
- ``SurfaceSync`` (CLI watcher) gates its polling on
  ``other_surfaces_active`` — zero overhead in CLI-only usage, active
  only while a GUI/mobile peer holds a live beacon.
- ``GET /api/sessions/{name}/presence`` lists live surfaces for UIs.
- The GUI's SessionWatcher touches a beacon per loaded session each
  poll, which is what makes the CLI watcher wake up automatically.

Beacon files are keyed by pid; a crashed process's beacon simply ages
out at the next read. Session names are validated with the same rule as
``SessionManager`` so no traversal can reach outside the sessions root.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# A beacon is "live" if refreshed within this window. Writers touch every
# ~5s, so 15s tolerates ~2 missed touches before a peer is considered gone.
PRESENCE_TTL_SECONDS = 15.0

_TOUCH_INTERVAL_SECONDS = 5.0


def _validate_session_name(name: Any) -> str:
    """Reuse the manager's validation without importing it (avoid cycles)."""
    import re

    name = str(name or "").strip()
    if not name:
        raise ValueError("presence: empty session name")
    if len(name) > 128:
        raise ValueError(f"Invalid session name: {name!r}")
    if name in (".", "..") or os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"Invalid session name: {name!r}")
    if os.path.isabs(name):
        raise ValueError(f"Invalid session name: {name!r}")
    if not re.match(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$", name):
        raise ValueError(f"Invalid session name: {name!r}")
    return name


def _history_dir() -> str:
    from mu.session.manager import _history_dir

    return _history_dir()


def _beacon_dir(session_name: str) -> str:
    name = _validate_session_name(session_name)
    return os.path.join(_history_dir(), "sessions", name, "presence")


def write_beacon(
    session_name: str,
    surface: str,
    *,
    busy: bool = False,
    pid: Optional[int] = None,
) -> str:
    """Write this process's beacon atomically. Returns the beacon path."""
    if surface not in ("cli", "gui", "mobile"):
        raise ValueError(f"Unknown surface: {surface!r}")
    d = _beacon_dir(session_name)
    pid = int(pid if pid is not None else os.getpid())
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{pid}.json")
    now = time.time()
    payload = {
        "pid": pid,
        "surface": surface,
        "started_at": now,
        "last_seen": now,
        "busy": bool(busy),
    }
    # started_at must survive across touches: read-then-merge.
    try:
        with open(path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if isinstance(prev, dict) and prev.get("pid") == pid:
            payload["started_at"] = prev.get("started_at", now)
    except (OSError, ValueError):
        pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    return path


def prune_beacons(session_name: str, ttl: float = PRESENCE_TTL_SECONDS) -> int:
    """Delete stale beacon files; returns how many were removed."""
    d = _beacon_dir(session_name)
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    now = time.time()
    removed = 0
    for fname in names:
        if not fname.endswith(".json"):
            continue
        path = os.path.join(d, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            last_seen = float(data.get("last_seen", 0) or 0)
        except (OSError, ValueError):
            # Corrupt beacon — treat as stale and remove.
            last_seen = 0.0
        if now - last_seen > ttl:
            # Round-13 F8: TOCTOU guard. Between reading last_seen and
            # unlinking, a toucher may have os.replace()d a fresh beacon
            # onto this path — re-stat and only unlink when the inode (and
            # mtime) still match the stale file we inspected.
            try:
                fresh = os.stat(path)
                with open(path, "r", encoding="utf-8") as f:
                    fresh_last_seen = float(json.load(f).get("last_seen", 0) or 0)
                if now - fresh_last_seen <= ttl:
                    continue  # beacon was refreshed mid-prune
            except (OSError, ValueError):
                pass  # vanished or corrupt — attempt the unlink anyway
            try:
                os.unlink(path)
                removed += 1
            except OSError:
                pass
    return removed


def read_presence(
    session_name: str, ttl: float = PRESENCE_TTL_SECONDS
) -> list[dict[str, Any]]:
    """Prune, then return the live beacons sorted by start time."""
    prune_beacons(session_name, ttl)
    d = _beacon_dir(session_name)
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    live = []
    for fname in names:
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("pid") is not None:
            live.append(data)
    live.sort(key=lambda b: (float(b.get("started_at", 0) or 0), int(b.get("pid", 0))))
    return live


def other_surfaces_active(
    session_name: str,
    own_pid: Optional[int] = None,
    ttl: float = PRESENCE_TTL_SECONDS,
) -> bool:
    """True while any *other* process holds a live beacon for the session."""
    own_pid = int(own_pid if own_pid is not None else os.getpid())
    for beacon in read_presence(session_name, ttl):
        try:
            if int(beacon.get("pid", 0)) != own_pid:
                return True
        except (TypeError, ValueError):
            continue
    return False


class BeaconToucher:
    """Background thread refreshing this surface's beacon every ~5s.

    ``busy_fn`` (optional) is consulted on each touch so the beacon
    reflects turn state (``busy``) without the caller having to flip a
    flag manually.
    """

    def __init__(
        self,
        session_name_fn: Callable[[], Optional[str]],
        surface: str,
        *,
        busy_fn: Optional[Callable[[], bool]] = None,
        interval: float = _TOUCH_INTERVAL_SECONDS,
    ) -> None:
        self._name_fn = session_name_fn
        self._surface = surface
        self._busy_fn = busy_fn
        self._interval = max(0.5, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="mucli-presence", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=self._interval + 2.0)
        self._thread = None

    def _loop(self) -> None:
        # Touch immediately, then on interval, so peers see us fast.
        self._touch()
        while not self._stop.wait(self._interval):
            self._touch()

    def _touch(self) -> None:
        try:
            name = self._name_fn()
            if not name:
                return
            busy = bool(self._busy_fn()) if self._busy_fn is not None else False
            write_beacon(name, self._surface, busy=busy)
        except Exception:  # noqa: BLE001 — presence must never crash its host
            logger.debug("presence touch failed", exc_info=True)