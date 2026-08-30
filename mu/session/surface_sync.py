"""CLI-side surface sync watcher (cross-surface continuity phase 2, G1).

Mirrors the GUI's ``SessionWatcher`` (mu/gui/watcher.py) in reverse: the
GUI already reloads sessions when the CLI (or another process) writes
``session.json``; this module gives the *CLI* the same inbound awareness so
a GUI/mobile edit made while a CLI session is open no longer gets silently
clobbered by the CLI's next ``save_history``.

Design (documentation/cross_surface_continuity.md §3.2):
- A daemon thread polls the session file every ``interval`` seconds.
- An external write is one whose ``__writer_pid__`` differs from this
  process AND whose on-disk ``revision`` is newer than the last one we
  applied (revision fast-path dedups equal-mtime rewrites).
- Mid-turn policy: while a turn is executing (``session._current_turn_start_index``
  is set — cleared in ``send_message``'s finally on every exit path) the
  watcher NEVER reloads; it records ``pending`` and re-checks on the next
  poll after the turn ends (deferred, not dropped — G6).
- Reload = ``SessionManager._load_session(name)``: full state hydration
  including the phase-1 revision counter. The TUI gets a one-line notice
  instead of silently mutating.

Presence gating (§3.4): until presence beacons exist, the watcher is
opt-in via ``MUCLI_SURFACE_SYNC=1`` — zero overhead in CLI-only usage.
When beacons land, gating switches to "GUI beacon present" automatically
via ``gate`` callable injection.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def surface_sync_enabled() -> bool:
    """Opt-in override; presence beacons (§3.4) are the primary gate."""
    return os.environ.get("MUCLI_SURFACE_SYNC", "") == "1"


class SurfaceSync:
    """Watch session.json for writes from other surfaces; reload on change.

    ``session`` is the Session object (needs ``.session_manager`` and the
    turn-scoped ``_current_turn_start_index`` busy marker). ``ui`` is any
    object with ``show_info`` (notified once per applied reload). ``gate``
    is an optional ``() -> bool``; when provided and returning False the
    poll is a no-op (presence gating hook for §3.4).
    """

    def __init__(
        self,
        session: Any,
        ui: Any = None,
        *,
        interval: float = 2.0,
        gate: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._session = session
        self._ui = ui
        self._interval = max(0.2, float(interval))
        # Default gate (§3.4): poll only while another surface holds a live
        # presence beacon — zero overhead in CLI-only usage. An explicit
        # gate wins; MUCLI_SURFACE_SYNC=1 forces polling on.
        if gate is not None:
            self._gate: Callable[[], bool] = gate
        elif surface_sync_enabled():
            self._gate = lambda: True
        else:
            self._gate = self._presence_gate
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.pending = False  # deferred external reload (G6 re-check)
        self._track_mtime = 0.0
        self._track_size = 0
        self._track_revision = 0
        self._initialized = False
        self._our_pid = os.getpid()

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="mucli-surface-sync", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=self._interval + 2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.check_once()
            except Exception:  # noqa: BLE001 — watcher must never die
                logger.debug("surface_sync poll failed", exc_info=True)

    # ---- core poll ---------------------------------------------------------

    def _filepath(self) -> Optional[str]:
        sm = self._session.session_manager
        name = getattr(sm, "current_session_name", None)
        if not name:
            return None
        try:
            return sm._get_filepath(name)
        except Exception:  # noqa: BLE001 — invalid names etc. just skip
            return None

    def _busy(self) -> bool:
        """True while a turn is executing on this surface.

        ``_current_turn_start_index`` is set in run_turn and cleared in
        send_message's finally on every exit path — unlike
        ``_active_turn_start_index`` which intentionally persists after
        the turn for compaction bookkeeping.
        """
        return getattr(self._session, "_current_turn_start_index", None) is not None

    def _presence_gate(self) -> bool:
        """Default gate: another surface holds a live beacon (§3.4)."""
        try:
            from mu.session.presence import other_surfaces_active

            sm = self._session.session_manager
            name = getattr(sm, "current_session_name", None)
            if not name:
                return False
            return other_surfaces_active(name, self._our_pid)
        except Exception:  # noqa: BLE001
            return False

    def check_once(self) -> bool:
        """Single poll. Returns True if a reload was applied now.

        Safe to call directly (tests, turn-boundary hooks); the thread
        loop just calls this repeatedly.

        Round-13 F7 ordering: tracker initialization and pending-reload
        re-checks happen BEFORE gate evaluation. Initializing while the
        gate is closed prevents the first external write from being
        silently absorbed as the baseline; applying a pending reload does
        not depend on the peer still being present (its beacon may have
        expired between deferral and turn end).
        """
        filepath = self._filepath()
        if not filepath:
            return False
        try:
            st = os.stat(filepath)
        except OSError:
            return False

        if not self._initialized:
            self._track_mtime, self._track_size = st.st_mtime, st.st_size
            self._track_revision = int(
                getattr(self._session.session_manager, "revision", 0) or 0
            )
            self._initialized = True
            return False

        # G6 re-check: a reload deferred during a busy turn must be applied
        # at the first idle poll — even though the file's mtime/size were
        # already folded into the tracker during the busy poll.
        if self.pending and not self._busy():
            return self._apply_reload()

        if self._gate is not None and not self._gate():
            return False

        if (st.st_mtime, st.st_size) == (self._track_mtime, self._track_size):
            return False  # nothing changed on disk

        # File changed — read attribution + revision in one pass.
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            # Mid-write or unreadable; skip this poll (file may be the
            # writer's tmp — os.replace is atomic so next poll sees it).
            return False
        if not isinstance(data, dict):
            return False

        try:
            on_disk_rev = int(data.get("revision", 0) or 0)
        except (TypeError, ValueError):
            on_disk_rev = 0
        writer_pid = data.get("__writer_pid__")

        self._track_mtime, self._track_size = st.st_mtime, st.st_size

        external = writer_pid is not None and int(writer_pid) != self._our_pid
        newer = on_disk_rev > self._track_revision
        if not (external and newer):
            return False  # our own write, or an older/equal write

        self._track_revision = on_disk_rev
        if self._busy():
            self.pending = True  # deferred — re-checked at next poll (G6)
            return False
        return self._apply_reload()

    def _apply_reload(self) -> bool:
        sm = self._session.session_manager
        name = getattr(sm, "current_session_name", None)
        if not name:
            return False
        try:
            sm._load_session(name)
        except Exception:  # noqa: BLE001 — never kill the watcher
            logger.error("surface_sync reload of %r failed", name, exc_info=True)
            return False
        self.pending = False
        if self._ui is not None:
            try:
                rev = int(getattr(sm, "revision", 0) or 0)
                self._ui.show_info(
                    f"↻ Session updated by another surface (revision {rev}) — reloaded"
                )
            except Exception:  # noqa: BLE001
                logger.debug("surface_sync notify failed", exc_info=True)
        return True

    def apply_pending(self) -> bool:
        """Turn-boundary hook: apply a deferred reload if one is pending."""
        if self.pending and not self._busy():
            return self._apply_reload()
        return False