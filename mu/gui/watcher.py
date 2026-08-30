"""Cross-process session file watcher (multi-session aware).

Polls every loaded session's ``session.json`` every few seconds. When
another process writes a file (different ``__writer_pid__`` than ours),
reloads that session's SessionManager from disk and emits
``session_updated`` on the SSE bus so connected browsers refresh.

Without this, two mucli processes (TUI + GUI, or two GUI tabs) writing
to the same session.json silently clobber each other. The last writer
wins on disk; the loser's in-memory state is stale until the next page
reload.

Each loaded session gets its own watcher entry so cross-process sync
works for every session in the daemon's cache — not just whichever one
the user is currently focused on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.revision import js_safe_revision  # noqa: E402 - config path bootstrap

logger = logging.getLogger(__name__)

# External writes are activity pulses, not durable busy state. A container
# worker saves session.json at turn completion; retaining this flag forever
# makes reconnecting clients display an infinite "thinking" state.
_EXTERNAL_ACTIVITY_TTL_SECONDS = 8.0


@dataclass
class _Track:
    mtime: float = 0.0
    size: int = 0
    initialized: bool = False
    external_active: bool = False
    external_last_at: float = 0.0
    deferred_reload: bool = False  # G6: an external-write reload was skipped while busy
    # Round-47 F5: revision of the external write that armed deferred_reload.
    # The deferred reload must adopt THIS version (or a strictly newer one
    # from the same external surface) — if the local writer saved since, the
    # external change was lost and the reload is skipped instead of
    # acknowledging a version that never contained the external update.
    external_revision: Optional[int] = None


class SessionWatcher:
    def __init__(self, app, *, interval: float = 2.0) -> None:
        self._app = app
        self._interval = interval
        self._task: Optional[asyncio.Task] = None
        self._tracks: Dict[str, _Track] = {}
        self._our_pid: int = os.getpid()

    # ---- compat shims -----------------------------------------------------
    # Old single-session UI reads these flags. Surface the focused session's
    # values so the existing /api/sessions/active endpoint keeps working.
    @property
    def external_active(self) -> bool:
        cur = self._app.state.current_session_name
        return self.external_active_for(cur) if cur else False

    @property
    def external_last_at(self) -> float:
        track = self._focused_track()
        return float(track.external_last_at) if track else 0.0

    def external_active_for(self, name: str) -> bool:
        track = self._tracks.get(name)
        if not track or not track.external_active:
            return False
        if time.time() - float(track.external_last_at or 0.0) <= _EXTERNAL_ACTIVITY_TTL_SECONDS:
            return True
        track.external_active = False
        return False

    def external_last_at_for(self, name: str) -> float:
        track = self._tracks.get(name)
        return float(track.external_last_at) if track else 0.0

    def _focused_track(self) -> Optional[_Track]:
        cur = self._app.state.current_session_name
        return self._tracks.get(cur) if cur else None

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval)
                try:
                    await self._tick()
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning("session watcher tick failed: %s", exc)
        except asyncio.CancelledError:
            return

    async def _tick(self) -> None:
        state = self._app.state
        loaded = dict(state.sessions)  # snapshot

        # Drop trackers for sessions that were unloaded.
        for stale in list(self._tracks.keys()):
            if stale not in loaded:
                self._tracks.pop(stale, None)

        # Presence beacons (§3.4): the GUI touches one beacon per loaded
        # session on every poll. This is what lets the CLI's SurfaceSync
        # watcher know a GUI peer is attached (and lets it stay idle when
        # nobody else is watching the session).
        self._touch_presence(loaded)

        # G6 (§3.5): if a previous external write was skipped because a turn
        # was in flight, re-check now that the busy flag may have cleared.
        await self._check_deferred_reloads(loaded)

        for name, session in loaded.items():
            try:
                await self._tick_one(name, session)
            except Exception as exc:
                logger.warning("watcher: %s failed: %s", name, exc)

    def _touch_presence(self, loaded: dict) -> None:
        try:
            from mu.session.presence import write_beacon

            for name in loaded:
                try:
                    write_beacon(name, "gui")
                except Exception:  # noqa: BLE001 — invalid names etc.
                    logger.debug("presence touch skipped for %r", name)
        except Exception:  # noqa: BLE001
            logger.debug("presence module unavailable", exc_info=True)

    async def _tick_one(self, name: str, session) -> None:
        sm = session.session_manager
        path = sm._get_filepath(name)
        if not os.path.exists(path):
            # Round-27 F5: the watched document vanished (deleted by
            # another surface, or the session was renamed). Without a
            # published event clients kept the stale in-memory session
            # forever. Emit once (tracker drop), then stop tracking —
            # a recreated file is re-discovered as a fresh track.
            if name in self._tracks:
                self._tracks.pop(name, None)
                await self._app.state.bus.publish(
                    {
                        "kind": "session_deleted",
                        "name": name,
                        "session_name": name,
                        "reason": "document_removed",
                    }
                )
            return

        track = self._tracks.setdefault(name, _Track())
        st = os.stat(path)
        mtime = st.st_mtime
        size = st.st_size

        if not track.initialized:
            track.mtime = mtime
            track.size = size
            track.initialized = True
            return

        if abs(mtime - track.mtime) < 0.05 and size == track.size:
            return

        # Round-47 F4: writer identity now comes from the fixed-size
        # .meta.json sidecar written at save time (~100 bytes) instead of
        # json.load()-ing the ENTIRE session document on the event loop —
        # on a 100k-message session that parse blocked SSE/HTTP for the
        # duration of every observed write. The sidecar also carries the
        # saved revision (F5): the deferred-reload path pins THIS revision
        # so a later reload can verify it is adopting the version the
        # external write actually produced, not whatever the local writer
        # saved since. Sidecar absent/older-format → fall back to a bounded
        # header read on a worker thread (never the event loop).
        writer_pid = None
        external_revision = None
        meta_path = path + ".meta.json"
        try:
            meta_st = os.stat(meta_path)
            # Round-48 F6: STRICT freshness — the sidecar is written AFTER
            # the document replace, so a sidecar older than the document is
            # by definition from the PREVIOUS save; the old ±50ms tolerance
            # let that stale sidecar pass on closely spaced saves and
            # misattribute writer PID / revision.
            if meta_st.st_mtime >= mtime:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                writer_pid = meta.get("__writer_pid__")
                external_revision = meta.get("revision")
        except (OSError, json.JSONDecodeError, ValueError):
            writer_pid = None
        if writer_pid is None:
            def _read_writer_pid() -> "tuple[Optional[int], Optional[int]]":
                try:
                    with open(path, "rb") as fh:
                        head = fh.read(65536)
                    text = head.decode("utf-8", errors="ignore")
                    idx = text.find("__writer_pid__")
                    if idx < 0:
                        return None, None
                    colon = text.find(":", idx)
                    comma = text.find(",", colon)
                    raw = text[colon + 1 : comma if comma > colon else colon + 24]
                    pid = int(raw.strip().strip('"'))
                    ridx = text.find('"revision"')
                    rev = None
                    if ridx >= 0:
                        rcolon = text.find(":", ridx)
                        rcomma = text.find(",", rcolon)
                        rev = int(text[rcolon + 1 : rcomma if rcomma > rcolon else rcolon + 24].strip())
                    return pid, rev
                except (OSError, ValueError):
                    return None, None
            writer_pid, external_revision = await asyncio.to_thread(_read_writer_pid)

        if writer_pid is None:
            # Race with a write in progress — skip this tick.
            return

        logger.info(
            "watcher: %r changed (mtime %.3f→%.3f) writer_pid=%s our_pid=%d",
            name, track.mtime, mtime, writer_pid, self._our_pid,
        )
        track.mtime = mtime
        track.size = size
        # Round-48 F8: pin the external revision ONLY for external writers.
        # The r47 shape assigned it before the local-writer check, so a
        # local save during a deferred external reload overwrote the pin
        # with the local revision — the lost-write detection then compared
        # the local version against itself and never fired.
        if writer_pid != self._our_pid:
            track.external_revision = external_revision

        if writer_pid == self._our_pid:
            return

        logger.info("watcher: external write detected on %r — reloading", name)
        await self._handle_external_write(session, name, track)

    async def _handle_external_write(self, session, name: str, track: _Track) -> None:
        state = self._app.state
        bus = state.bus
        lock = state.session_lock_for(name)
        busy = state.session_busy_for(name)

        # Round-32 F7: report reload success so the publish below can omit
        # the revision token when no fresh state was actually loaded — a
        # stale token adopted by a client would poison its next If-Match.
        def _do_reload() -> bool:
            try:
                with lock:
                    session.session_manager._load_session(name)
                return True
            except Exception as exc:
                logger.warning("session reload failed: %s", exc)
                return False

        reloaded = False
        if busy.is_set():
            # G6 (§3.5): the reload is deferred, not dropped. Arm a flag so
            # the next tick after the busy flag clears re-arms the check and
            # applies the skipped external write.
            track.deferred_reload = True
            logger.info("watcher: turn in flight on %r — deferring reload (re-check armed)", name)
        else:
            reloaded = await asyncio.to_thread(_do_reload)

        track.external_active = True
        track.external_last_at = time.time()
        # Round-31 F35: publish the post-reload revision so clients can
        # capture it as an If-Match token. The reload (immediate or just
        # applied deferred) rehydrates sm.revision from the winner's write.
        # Round-32 F7: the revision field is published ONLY when a reload
        # actually succeeded — on the busy/deferred path the in-memory
        # revision is still the pre-write value (the deferred-reload event
        # carries the fresh token later), and on a failed reload any token
        # would be stale. Absent field = "no trustworthy token yet".
        event = {
            "kind": "session_updated",
            "name": name,
            "session_name": name,
            "reason": "external_write",
            "reloaded": reloaded,
            "deferred": busy.is_set(),
        }
        if reloaded:
            event["revision"] = js_safe_revision(
                getattr(session.session_manager, "revision", 0) or 0
            )
        await bus.publish(event)

    async def _check_deferred_reloads(self, loaded: dict) -> None:
        """G6 (§3.5) re-check: after a busy turn ends, apply the reload we
        skipped. Called from _tick each poll — the busy flag lives on
        app.state, so this re-arm works regardless of which request cleared
        it (normal completion, error, worker failure)."""
        for name, session in loaded.items():
            track = self._tracks.get(name)
            if track is None or not track.deferred_reload:
                continue
            busy = self._app.state.session_busy_for(name)
            if busy.is_set():
                continue  # still in the turn — keep waiting
            logger.info("watcher: %r turn finished — applying deferred reload", name)
            # Round-13 F6: the flag is cleared only on a successful reload —
            # a transient load failure keeps it armed so the next tick
            # retries instead of silently dropping the external write.
            if await self._apply_deferred_reload(session, name, track):
                track.deferred_reload = False

    async def _apply_deferred_reload(self, session, name: str, track: Optional[_Track] = None) -> bool:
        """Apply a skipped reload under the session lock. Returns True on
        success; on failure the caller keeps the deferred flag armed.

        Round-47 F5: when the external write's revision is known (from the
        save-time sidecar), the reload only counts as applied if the disk
        document still carries that revision or a NEWER one written by the
        same external surface. If the local writer saved over the external
        version while the turn was in flight, the external change is LOST —
        adopting the local file and reporting success would acknowledge a
        version that never contained the external update. In that case the
        reload is skipped (returns False, flag stays armed) and the lost
        external write is surfaced as an external_write_lost conflict event
        so surfaces can reconcile explicitly.
        """
        state = self._app.state
        lock = state.session_lock_for(name)
        wanted_revision = track.external_revision if track else None

        def _locked_reload() -> "Optional[int]":
            # Round-27 F6: _load_session does blocking disk I/O — run
            # it in a worker thread so a large session.json or slow
            # storage cannot stall the event loop (SSE + all HTTP
            # routes) while the deferred reload applies. The session
            # lock is taken inside the worker thread (same ordering as
            # the previous synchronous version).
            # Round-47 F5: peek the on-disk revision BEFORE the load
            # overwrites in-memory state — cheap sidecar read. Absent
            # sidecar OR a session manager without _get_filepath (test
            # stubs) → disk_rev None = no version pin, reload applies.
            disk_rev = None
            try:
                with open(session.session_manager._get_filepath(name) + ".meta.json",
                          "r", encoding="utf-8") as mf:
                    disk_rev = int(json.load(mf).get("revision") or 0) or None
            except (OSError, AttributeError, json.JSONDecodeError, ValueError):
                disk_rev = None
            session.session_manager._load_session(name)
            return disk_rev

        # Round-48 F7 (CRITICAL): the pin check must gate the LOAD — the
        # r47 shape called _load_session() first and only then compared
        # revisions, so the "skipped" branch had already clobbered
        # in-memory state with the very version we rejected. Peek the
        # sidecar FIRST; load only when the external version is still
        # on disk (or unknown).
        disk_rev = None
        try:
            with open(session.session_manager._get_filepath(name) + ".meta.json",
                      "r", encoding="utf-8") as mf:
                disk_rev = int(json.load(mf).get("revision") or 0) or None
        except (OSError, AttributeError, json.JSONDecodeError, ValueError):
            disk_rev = None
        if wanted_revision is not None and disk_rev is not None and disk_rev < wanted_revision:
            # The external version was overwritten before we could adopt it —
            # in-memory state is still the local writer's (correct); do NOT
            # load. Surface the lost external write explicitly.
            logger.warning(
                "watcher: %r deferred reload skipped — external rev %s was "
                "overwritten by a local save (disk rev %s); publishing "
                "external_write_lost",
                name, wanted_revision, disk_rev,
            )
            await self._app.state.bus.publish({
                "kind": "external_write_lost",
                "name": name,
                "session_name": name,
                "lost_revision": js_safe_revision(wanted_revision),
                "current_revision": js_safe_revision(disk_rev),
            })
            # The skipped external write will never be loadable again —
            # clear the armed flag to avoid an eternal re-check. State was
            # never touched, so no reload bookkeeping is needed.
            return True
        try:
            await asyncio.to_thread(_locked_reload)
        except Exception as exc:
            logger.warning("session reload failed: %s", exc)
            return False
        # Round-13 F5: fold the external write into the tracker so the
        # _tick_one pass right after us does not re-detect the same mtime
        # change and fire a duplicate session_updated event.
        if track is not None:
            try:
                sm = session.session_manager
                path = sm._get_filepath(name)
                st = os.stat(path)
                track.mtime = st.st_mtime
                track.size = st.st_size
            except Exception:  # noqa: BLE001 — tracker fold is best-effort
                pass
        try:
            import asyncio as _asyncio

            _asyncio.get_running_loop().create_task(
                state.bus.publish(
                    {
                        "kind": "session_updated",
                        "name": name,
                        "session_name": name,
                        "reason": "deferred_reload",
                        "reloaded": True,
                        # Round-31 F35: JS-safe post-reload revision token.
                        "revision": js_safe_revision(
                            getattr(session.session_manager, "revision", 0) or 0
                        ),
                    }
                )
            )
        except RuntimeError:
            pass  # no running loop (sync probe) — skip the notification
        return True
