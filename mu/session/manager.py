"""`SessionManager` — persistent per-session state.

Owns the history list, conversation summary, provider config, token
accounting, feature-mode registry, and the on-disk JSON store at
`~/.mucli/sessions/<name>/session.json`. Inherits the
token-budget-aware history-roll helpers from `HistoryMixin`.

`mu.session.session.SessionManager` re-exports this class for backward
compatibility — callers that import via the old path still work.

Tests: `tests/test_session.py` (load/save/switch/delete),
`tests/test_session_picker_state.py` (session-list interactions),
`tests/test_startup_session_picker.py` (numbered-picker fallback),
`tests/test_mu_session_history.py` (HistoryMixin round-trip via this class).
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import glob
import json
import os
import re
import shutil
import tempfile
import time
from copy import deepcopy
from typing import Any

from mu.agent.collation import CollationBuffer
from mu.memory.stores import MemoryEntry, ScratchpadStore, TaskMemoryStore
from mu.workspace.folder_context import FolderContext
from utils.config import (
    DEFAULT_VARIABLES,
    validate_and_cast,
)
from utils.logger import logger

import utils.config as _config

from .helpers import _slugify_feature_id, derive_feature_state_status
from .history import HistoryMixin
from .history_search import HistorySearchMixin
from .tool_cache import ToolResultCache


def _history_dir() -> str:
    """Resolve the active HISTORY_DIR via `utils.config.HISTORY_DIR`.
    Read dynamically (not `from utils.config import HISTORY_DIR`) so
    `monkeypatch.setattr("utils.config.HISTORY_DIR", ...)` reaches
    this code path during tests."""
    return _config.HISTORY_DIR


def _message_identity(message: Any) -> str:
    """Stable identity key for a history message (phase-6 F47 dedupe).

    JSON-canonical hash of role+parts. Used to detect which of the
    turn's in-memory messages the winning on-disk document ALREADY
    contains (persisted through the turn's earlier successful saves) so
    the conflict-path re-append does not duplicate them. Non-dict
    messages fall back to their repr; unhashable payloads degrade to
    that repr as well — identity is best-effort, duplication is the
    failure mode we are preventing, not a crash.
    """
    try:
        if isinstance(message, dict):
            payload = {"role": message.get("role"), "parts": message.get("parts")}
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
            ).hexdigest()
        return hashlib.sha256(repr(message).encode("utf-8")).hexdigest()
    except Exception:  # pragma: no cover — identity must never crash the save
        return repr(message)


def _empty_tool_stats() -> dict[str, Any]:
    return {
        "session_started_at": time.time(),
        "first_call_at": None,
        "last_call_at": None,
        "tools": {},
        "skills": {},
        "approvals": {"approved": 0, "denied": 0},
        "errors": {},
    }


def _normalize_tool_stats(value: Any) -> dict[str, Any]:
    stats = _empty_tool_stats()
    if not isinstance(value, dict):
        return stats
    for key in ("session_started_at", "first_call_at", "last_call_at"):
        if key in value:
            stats[key] = value[key]
    for key in ("tools", "skills", "approvals", "errors"):
        if isinstance(value.get(key), dict):
            stats[key].update(value[key])
    return stats


class RevisionConflict(Exception):
    """Raised when a compare-and-swap save detects a concurrent write.

    Cross-surface continuity phase 1: callers pass ``expected_revision`` to
    ``save_history`` (or an ``If-Match`` header on mutating GUI routes) to
    assert the on-disk session state they based their edit on. A mismatch
    means another surface (CLI/GUI/mobile) wrote the session in between —
    the caller must reload and reapply rather than silently clobbering.
    """

    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(
            f"Revision conflict: expected {expected}, current on-disk revision is {current}"
        )


class SessionManager(HistoryMixin, HistorySearchMixin):
    def __init__(self, ui=None, session_name=None):
        self.ui = ui
        logger.info(f"Initializing SessionManager (session_name={session_name})")
        self.current_session_name = session_name or ""
        self.history = []
        # Cross-surface continuity phase 1 (G2): monotonically increasing
        # per-session revision. Loaded from session.json (0 for legacy
        # sessions without one), incremented on every successful save.
        self.revision = 0
        self.conversation_summary = ""
        self.provider_config = {}
        self.collation_buffer = CollationBuffer()
        self.summary_anchor = 0
        self.protected_indices: set[int] = set()
        # Active-turn start index, mirrored from the agent loop so the
        # compaction paths can compute the per-turn tool-result floor
        # (R3, FM-8). None outside an active turn.
        self._active_turn_start_index: int | None = None
        # Phase-6 F47: turn-scoped compare-and-swap. When armed by
        # begin_turn_cas(), every save during the turn verifies that the
        # on-disk document still reflects the revision this turn has been
        # building on — a concurrent surface write (GUI/mobile/CLI) makes
        # the turn's intermediate saves fail LOUDLY instead of silently
        # clobbering the other surface's document.
        self._turn_cas_armed = False
        self._turn_cas_baseline: int | None = None
        self.folder_context = FolderContext()
        self.task_memory = TaskMemoryStore()
        self.turn_scratchpad = ScratchpadStore()
        self.tool_result_cache = ToolResultCache()
        self.token_counts = {
            "input": 0,
            "output": 0,
            "total": 0,
            "total_cost": 0.0,
            "cached": 0,
            "reasoning": 0,
        }
        self.feature_state = None
        self.feature_registry = {}
        self.active_feature_id = None
        self.teacher_state = None
        self.teacher_registry = {}
        self.active_course_id = None
        # Structured research sources are session-owned.  The citation engine
        # itself is process-global, so retaining this snapshot prevents an
        # unload/reload from silently dropping a research trail.
        self.research_sources: list[dict] = []
        self.tool_stats = _empty_tool_stats()
        self.variables = DEFAULT_VARIABLES.copy()
        self.container_config: dict[str, Any] = {}

        if session_name:
            self._load_session(session_name)

    def _get_filepath(self, name):
        return os.path.join(self._get_session_dir(name), "session.json")

    @staticmethod
    def session_lock_path(name) -> str:
        """Round-14 F1: stable per-session lock file OUTSIDE the session dir.

        The delete route rmtree's the session directory while holding the
        CAS lock; a lock file living inside that directory would be unlinked
        mid-hold, letting a concurrent writer recreate the path and acquire
        a DIFFERENT inode (lock split-brain). Locks live in a sibling
        .locks/ directory that survives deletion.
        """
        validated = SessionManager._validate_session_name(name)
        if not validated:
            raise ValueError("Session lock requires a session name")
        locks_dir = os.path.join(_history_dir(), "sessions", ".locks")
        os.makedirs(locks_dir, exist_ok=True)
        return os.path.join(locks_dir, f"{validated}.lock")

    def _get_session_dir(self, name):
        return os.path.join(_history_dir(), "sessions", self._validate_session_name(name))

    @staticmethod
    def _validate_session_name(name) -> str:
        """Canonical session-name validation (codex round-7 F1).

        Session names reach os.path.join/rmtree/rename from CLI input,
        not just the (already-validated) GUI routes. A name like
        '/tmp/victim' or '../../target' previously deleted or created
        directories outside the sessions root. Names must be a single
        safe path component.
        """
        name = str(name or "").strip()
        if not name:
            # Empty name joins to the sessions root itself — callers use
            # this for root-relative paths (e.g. feature metadata root).
            # Harmless: no traversal, operations just resolve to the root.
            return name
        if len(name) > 128:
            raise ValueError(f"Invalid session name: {name!r}")
        if name in (".", "..") or os.sep in name or (os.altsep and os.altsep in name):
            raise ValueError(f"Invalid session name: {name!r}")
        if os.path.isabs(name):
            raise ValueError(f"Invalid session name: {name!r}")
        if not re.match(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$", name):
            raise ValueError(f"Invalid session name: {name!r}")
        return name

    def get_durable_memory_service(self):
        """Return the process-shared cross-session memory service lazily.

        Resolving HISTORY_DIR at call time preserves the existing test and
        deployment behaviour where MUCLI_HOME may be changed after import.
        """

        from mu.memory.service import get_memory_service

        return get_memory_service(_history_dir())

    def _load_session(self, name):
        filepath = self._get_filepath(name)
        legacy_filepath = os.path.join(_history_dir(), f"{name}.json")
        self.current_session_name = name
        self.history = []
        self.conversation_summary = ""
        self.summary_anchor = 0
        # Phase 1 (G2): reset revision alongside the rest of the loadable
        # state; hydrated from session.json below (0 if absent/legacy).
        self.revision = 0
        self.protected_indices = set()
        self.provider_config = {}
        self.collation_buffer = CollationBuffer()
        self.folder_context = FolderContext()
        self.task_memory = TaskMemoryStore()
        self.turn_scratchpad = ScratchpadStore()
        self.tool_result_cache = ToolResultCache()
        self.variables.clear()
        self.token_counts = {
            "input": 0,
            "output": 0,
            "total": 0,
            "total_cost": 0.0,
            "cached": 0,
            "reasoning": 0,
        }
        self.feature_state = None
        self.feature_registry = {}
        self.active_feature_id = None
        self.teacher_state = None
        self.teacher_registry = {}
        self.active_course_id = None
        self.research_sources = []
        self.tool_stats = _empty_tool_stats()
        self.container_config = {}
        self.variables.update(DEFAULT_VARIABLES)

        data = self.read_session_data(name)
        if data is not None:
            try:
                if isinstance(data, list):
                    self.history = data
                elif isinstance(data, dict):
                    self.history = data.get("history", [])
                    self.conversation_summary = str(
                        data.get("conversation_summary", "") or ""
                    )
                    self.summary_anchor = data.get("summary_anchor", 0)
                    try:
                        self.revision = int(data.get("revision", 0) or 0)
                    except (TypeError, ValueError):
                        self.revision = 0
                    self.protected_indices = set(data.get("protected_indices", []))
                    self.provider_config = data.get("provider_config", {})
                    saved_container_config = data.get("container_config", {})
                    self.container_config = (
                        dict(saved_container_config)
                        if isinstance(saved_container_config, dict)
                        else {}
                    )
                    self.collation_buffer = CollationBuffer.from_dict(
                        data.get("collation_buffer", {})
                    )
                    self.folder_context.from_dict(data.get("folder_context", {}))
                    self.task_memory = TaskMemoryStore.from_dict(
                        data.get("task_memory", {})
                    )
                    self.turn_scratchpad = ScratchpadStore.from_dict(
                        data.get("turn_scratchpad", {})
                    )
                    self.token_counts = data.get(
                        "token_counts",
                        {"input": 0, "output": 0, "total": 0, "total_cost": 0.0},
                    )
                    feature_state = data.get("feature_state")
                    if isinstance(feature_state, dict):
                        self.feature_state = feature_state
                    self.feature_registry = {
                        str(key): value
                        for key, value in (
                            data.get("feature_registry", {}) or {}
                        ).items()
                        if isinstance(value, dict)
                    }
                    self.active_feature_id = data.get("active_feature_id")
                    if (
                        self.feature_state is None
                        and self.active_feature_id in self.feature_registry
                    ):
                        self.feature_state = deepcopy(
                            self.feature_registry[self.active_feature_id]
                        )

                    teacher_state = data.get("teacher_state")
                    if isinstance(teacher_state, dict):
                        self.teacher_state = teacher_state
                    self.teacher_registry = {
                        str(key): value
                        for key, value in (
                            data.get("teacher_registry", {}) or {}
                        ).items()
                        if isinstance(value, dict)
                    }
                    self.active_course_id = data.get("active_course_id")
                    if (
                        self.teacher_state is None
                        and self.active_course_id
                        and self.active_course_id in self.teacher_registry
                    ):
                        self.teacher_state = deepcopy(
                            self.teacher_registry[self.active_course_id]
                        )

                    saved_sources = data.get("research_sources", [])
                    self.research_sources = (
                        saved_sources if isinstance(saved_sources, list) else []
                    )
                    self.restore_research_sources()

                    self.tool_stats = _normalize_tool_stats(data.get("tool_stats"))

                    saved_vars = data.get("variables", {})
                    for k, v in saved_vars.items():
                        try:
                            self.variables[k] = validate_and_cast(k, v)
                        except ValueError:
                            # If saved data is corrupt or schema changed, keep default
                            pass
            except (json.JSONDecodeError, IOError):
                self.history = []

    def read_session_data(self, name):
        filepath = self._get_filepath(name)
        legacy_filepath = os.path.join(_history_dir(), f"{name}.json")
        source_filepath = filepath if os.path.exists(filepath) else legacy_filepath
        if not os.path.exists(source_filepath):
            return None
        try:
            with open(source_filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def get_session_history(self, name):
        data = self.read_session_data(name)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("history", [])
        return []

    # ----- Protected messages: key messages preserved through compaction -----
    _PROTECTED_CAP = 20

    def _maybe_protect(self, idx: int, role: str, text: str, *, is_turn_prompt: bool = False) -> None:
        """Mark a history index as protected if it meets importance criteria.

        Rules:
        - No index is permanently protected — index 0 follows the same rules.
        - The turn's starting prompt (is_turn_prompt=True) is always protected during active turn.
        - Other user messages >50 chars and not starting with '/' are protected.
        - Slash commands and short messages are NOT protected.
        - After adding, enforce cap: if > _PROTECTED_CAP protected indices,
          evict oldest (smallest idx), no special case for index 0.
        - Call _cleanup_protected() at end of turn to unset turn-prompt
          protections that are no longer needed.
        """
        if role != "user":
            return
        text = (text or "").strip()
        # Index 0 is NOT permanently protected — it follows the same rules
        # as any other message: protected during active turn (is_turn_prompt)
        # or if substantial (>50 chars, not a /command). After the task/
        # feature completes, index 0 can be wiped/summarized away.
        if is_turn_prompt or (len(text) > 50 and not text.startswith("/")):
            self.protected_indices.add(idx)
        # Enforce cap with age-based eviction (keep newest, no special case)
        self._enforce_protected_cap()

    def _enforce_protected_cap(self) -> None:
        """Evict oldest protected indices when count exceeds cap."""
        while len(self.protected_indices) > self._PROTECTED_CAP:
            evictable = sorted(self.protected_indices)
            if evictable:
                self.protected_indices.discard(evictable[0])
            else:
                break

    def tool_result_floor_indices(
        self, turn_start_index: int | None, floor: int
    ) -> set[int]:
        """Indices of the last ``floor`` tool-result-bearing messages
        within the active turn (``>= turn_start_index``).

        These are protected from summarization/degradation (R3, FM-8) so
        compaction mid-turn cannot drop tool results just received.
        Returns an empty set when ``floor <= 0`` or no active turn is set.
        """
        if floor <= 0 or turn_start_index is None:
            return set()
        start = max(0, int(turn_start_index))
        if start >= len(self.history):
            return set()
        indices: list[int] = []
        for idx in range(start, len(self.history)):
            msg = self.history[idx]
            parts = msg.get("parts") or []
            if any(p.get("type") == "tool_result" for p in parts):
                indices.append(idx)
        if len(indices) <= floor:
            return set(indices)
        return set(indices[-floor:])

    def _cleanup_protected(self, turn_start_index: int) -> None:
        """Clean up protected indices after a turn ends.

        - Remove the turn-prompt protection (turn_start_index) if it's
          not otherwise worthy of long-term protection (not a substantial
          user message >50 chars).
        - Index 0 is NOT specially protected — it follows the same rules.
        - This keeps the protected set bounded: only genuinely important
          messages survive across turns; the turn's starting prompt is
          protected only while that turn is active.
        """
        # Check if the turn prompt qualifies for long-term protection
        if turn_start_index < len(self.history):
            msg = self.history[turn_start_index]
            text = ""
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    text = (part.get("text") or "").strip()
                    break
            if len(text) > 50 and not text.startswith("/"):
                return  # qualifies on its own merits, keep protected
        # Not otherwise worthy — unprotect the turn prompt
        self.protected_indices.discard(turn_start_index)

    def begin_turn_cas(self) -> None:
        """Arm turn-scoped CAS with the current revision as the baseline.

        Phase-6 F47: called at the START of an agent turn. Every turn
        save (save_history_turn) then verifies the on-disk revision
        still matches the baseline this turn has been building on; a
        concurrent surface write (GUI/mobile/CLI) makes saves fail
        LOUDLY instead of silently clobbering the other surface's
        document. The baseline re-baselines after every successful
        turn save, so only INTER-surface conflicts fire — never the
        turn's own saves.
        """
        self._turn_cas_armed = True
        self._turn_cas_baseline = self.revision
        # Phase-6 r24 F1: baseline snapshot of task_memory at turn
        # start. The conflict path applies only the DELTA (snapshot
        # before the failed save minus this baseline) onto the winner,
        # so winner-side deletions/evictions/status transitions of
        # baseline entries are never resurrected or overwritten.
        self._task_memory_baseline = self._snapshot_task_memory()

    def _bind_runtime_owner(self, owner) -> None:
        """Bind the Session that owns this manager (weakref, phase-6 r21 F4).

        reload_winner_state replaces manager state OBJECTS (folder_context,
        task_memory, turn_scratchpad, collation buffer). Session holds
        aliases to those objects (sync_runtime_state); after a winner
        reload those aliases are stale. The owner is re-synced after a
        winner reload so subsequent tool calls in the turn see adopted
        state, not pre-conflict objects.
        """
        import weakref

        try:
            self._runtime_owner = weakref.ref(owner)
        except TypeError:
            self._runtime_owner = None

    def _resync_runtime_owner(self) -> None:
        """Re-sync the bound Session's aliases after a winner reload."""
        ref = getattr(self, "_runtime_owner", None)
        owner = ref() if ref is not None else None
        if owner is None:
            return
        try:
            owner.sync_runtime_state()
        except Exception:
            # Defensive: resync is best-effort; the merge itself already
            # succeeded and must not be rolled back by alias churn.
            logger.debug("runtime owner resync failed", exc_info=True)

    def _snapshot_task_memory(self):
        """Serialize the local task_memory store (or None on failure)."""
        try:
            return deepcopy(self.task_memory.to_dict())
        except Exception:
            return None

    def _apply_task_memory_delta(
        self, presave: dict | None, baseline: dict | None
    ) -> None:
        """Re-apply THIS TURN's task_memory mutations onto the winner.

        Phase-6 r24 F1/F2/F3 (supersedes the r23 whole-store overlay):
        the overlay snapshot resurrected winner-side deletions and
        overwrote winner status transitions because it replayed the
        ENTIRE local store. The delta instead computes, per entry:

        - ADDED this turn (id absent from the baseline): imported into
          the winner's store with a FRESH id (the local id may collide
          with a concurrent surface's entry — numeric ids are
          session-local counters), supersedes/superseded_by refs
          remapped within the batch, store cap enforced once.
        - MUTATED this turn (present in both snapshots): the local
          durable_id / status change is adopted ONLY when the winner
          entry still carries the baseline value — a value the winner
          itself changed wins.

        Winner-side deletions (entry in baseline, absent from winner)
        win: the entry is not resurrected.
        """
        if not presave or not baseline:
            return
        try:
            base_by_id = {
                int(item.get("id", 0)): item
                for item in baseline.get("entries", [])
                if isinstance(item, dict)
            }
            snap_by_id = {
                int(item.get("id", 0)): item
                for item in presave.get("entries", [])
                if isinstance(item, dict)
            }
            # 1. Entries added this turn: absent from the baseline.
            added = [
                item
                for entry_id, item in snap_by_id.items()
                if entry_id and entry_id not in base_by_id
            ]
            local_id_map: dict[int, int] = {}
            if added:
                imported = self.task_memory.import_entries(added)
                # Round-25 F1: turn-added entries may land on a FRESH id
                # (collision with a concurrent surface's entry). Pointer
                # adoption below must translate local ids to the fresh
                # ones so supersedes refs stay resolvable.
                # Round-26 F1a: _enforce_limit() inside import_entries
                # can EVICT an imported entry immediately (fresh ids get
                # status weights like any entry) — a mapping to an
                # evicted id would dangle. Drop those translations.
                local_id_map = {
                    int(item.get("id", 0)): entry.id
                    for item, entry in zip(added, imported)
                    if int(item.get("id", 0) or 0)
                    and self.task_memory.get_entry(entry.id) is not None
                }
            # 2. Mutations on baseline entries: adopt only when the
            # winner still carries the baseline value.
            for entry_id, base_item in base_by_id.items():
                snap_item = snap_by_id.get(entry_id)
                if snap_item is None:
                    continue
                winner_entry = self.task_memory.get_entry(entry_id)
                if winner_entry is None:
                    # Winner deleted/evicted it — deletion wins.
                    continue
                base_durable = str(base_item.get("durable_id", "") or "")
                snap_durable = str(snap_item.get("durable_id", "") or "")
                winner_durable = str(winner_entry.durable_id or "")
                if snap_durable != base_durable and winner_durable == base_durable:
                    winner_entry.durable_id = snap_durable
                base_status = str(base_item.get("status", "") or "")
                snap_status = str(snap_item.get("status", "") or "")
                if snap_status != base_status and winner_entry.status == base_status:
                    winner_entry.status = snap_status
                # Round-25 F1: supersedes/superseded_by pointers are part
                # of the mutation set too — a local supersede(old, new)
                # sets status=superseded AND the back-pointers. Without
                # this the winner kept a 'superseded' status with a
                # dangling/missing pointer. Same 3-way guard: adopt only
                # when the winner still carries the baseline pointer.
                for field_name in ("supersedes", "superseded_by"):
                    base_ref = base_item.get(field_name)
                    snap_ref = snap_item.get(field_name)
                    try:
                        snap_ref = None if snap_ref is None else int(snap_ref)
                    except (TypeError, ValueError):
                        continue
                    if snap_ref is not None and snap_ref in local_id_map:
                        snap_ref = local_id_map[snap_ref]
                    # Round-26 F1b: the pointer TARGET must exist in the
                    # winner store too — a ref pointing at a baseline
                    # entry the winner deleted/evicted would dangle.
                    # Skip adoption entirely (winner value stands).
                    if snap_ref is not None and (
                        self.task_memory.get_entry(snap_ref) is None
                    ):
                        continue
                    winner_ref = getattr(winner_entry, field_name, None)
                    if snap_ref != base_ref and winner_ref == base_ref:
                        setattr(winner_entry, field_name, snap_ref)
        except Exception:
            # Defensive: delta apply is best-effort; never crash the
            # merge — the winner document is already persisted.
            logger.debug("task_memory delta apply failed", exc_info=True)

    def end_turn_cas(self) -> None:
        """Disarm turn-scoped CAS (turn finally)."""
        self._turn_cas_armed = False
        self._turn_cas_baseline = None
        self._task_memory_baseline = None
        self._pending_presave_snapshot = None

    def save_history_turn(self, folder_context_obj=None) -> None:
        """Phase-6 F47: turn save with auto-CAS + one conflict recovery.

        When turn CAS is armed, this is a compare-and-swap save against
        the turn baseline. On RevisionConflict: reload the WINNING disk
        document (the concurrent surface's write — history/summary/
        protected/variables/provider_config/container_config), keep the
        turn's messages (they were appended in memory AFTER the winner's
        history prefix, so they are preserved by the reload + re-append),
        re-baseline to the winner's revision, and retry ONCE. A second
        conflict re-raises (the other surface is actively racing).

        Outside an armed turn this behaves exactly like save_history()
        with no expected_revision (plain LWW).
        """
        if not self._turn_cas_armed or self._turn_cas_baseline is None:
            return self.save_history(folder_context_obj)
        # Phase-6 r24: snapshot task_memory BEFORE the CAS save. On
        # conflict the delta (presave minus turn-start baseline) is
        # re-applied onto the adopted winner; on success the baseline
        # rolls forward so only post-save mutations are deltas.
        presave = self._snapshot_task_memory()
        self._pending_presave_snapshot = presave
        try:
            self.save_history(folder_context_obj, self._turn_cas_baseline)
            self._turn_cas_baseline = self.revision
            self._task_memory_baseline = presave
            return
        except RevisionConflict:
            pass
        # Conflict path: the winner's document is on disk. Rebuild the
        # turn view: reload winner state, then re-append what the turn
        # added in memory on top of the winner's history. Messages the
        # turn already persisted through EARLIER successful saves are
        # part of the winner's document — re-appending them would
        # duplicate.
        #
        # Phase-6 r21 F2: dedupe is a MULTISET DIFF against the winner's
        # FULL history, not a membership set and not a prefix slice. A
        # global set would drop a legitimately DISTINCT message that
        # repeats content already in the winner (same prompt submitted
        # twice); a prefix slice would re-append turn-persisted messages
        # that live inside the winner's turn window. The multiset diff
        # re-appends exactly max(0, mem_count - winner_count) copies of
        # each identity: messages the turn already persisted (winner has
        # them) contribute 0; genuinely new or extra copies contribute
        # their surplus. Identical-content collisions collapse benignly —
        # same bytes, no information lost.
        turn_start = self._active_turn_start_index
        turn_messages = (
            self.history[turn_start:] if turn_start is not None else []
        )
        winner_state = self.reload_winner_state()
        winner_history = winner_state.get("history") or []
        if turn_start is not None:
            winner_counts: dict[str, int] = {}
            for message in winner_history:
                key = _message_identity(message)
                winner_counts[key] = winner_counts.get(key, 0) + 1
            mem_counts: dict[str, int] = {}
            for message in turn_messages:
                key = _message_identity(message)
                mem_counts[key] = mem_counts.get(key, 0) + 1
            for message in turn_messages:
                key = _message_identity(message)
                if mem_counts.get(key, 0) > winner_counts.get(key, 0):
                    self.history.append(message)
                    mem_counts[key] -= 1
        # Phase-6 r21 F-c: the caller's folder_context_obj is the
        # PRE-RELOAD object; save_history would re-assign it over the
        # winner's adopted folder context. The reload already adopted
        # the winner's folder_context — retry with it (None = keep).
        self._turn_cas_baseline = self.revision
        # Retry ONCE against the winner's revision. A second conflict
        # means the other surface is actively racing — propagate.
        self.save_history(None, self._turn_cas_baseline)
        self._turn_cas_baseline = self.revision

    def reload_winner_state(self) -> dict:
        """Reload the winning on-disk document into memory (conflict path).

        Adopts the concurrent winner's document. Phase-6 r21 F3: must
        hydrate the FULL persisted schema — any field written by
        _save_history_locked but not adopted here is silently reverted
        to this manager's stale in-memory value by the retry save.
        Raises FileNotFoundError when the document vanished entirely.
        """
        name = self.current_session_name
        data = self.read_session_data(name)
        if data is None:
            raise FileNotFoundError(
                f"session document for {name!r} disappeared; cannot reload winner"
            )
        if isinstance(data, list):
            self.history = data
            return {"history": self.history}
        if not isinstance(data, dict):
            raise ValueError(f"corrupt session document for {name!r}")
        self.history = data.get("history", [])
        self.conversation_summary = str(data.get("conversation_summary", "") or "")
        self.summary_anchor = int(data.get("summary_anchor", 0) or 0)
        try:
            self.revision = int(data.get("revision", 0) or 0)
        except (TypeError, ValueError):
            self.revision = 0
        self.protected_indices = set(data.get("protected_indices", []) or [])
        self.provider_config = dict(data.get("provider_config", {}) or {})
        self.container_config = dict(data.get("container_config", {}) or {})
        self.collation_buffer = CollationBuffer.from_dict(
            data.get("collation_buffer", {}) or {}
        )
        fc = FolderContext()
        fc.from_dict(data.get("folder_context", {}) or {})
        self.folder_context = fc
        variables = data.get("variables", {})
        self.variables.clear()
        self.variables.update(DEFAULT_VARIABLES)
        if isinstance(variables, dict):
            self.variables.update(variables)
        task_memory = TaskMemoryStore.from_dict(data.get("task_memory", {}) or {})
        self.task_memory = task_memory
        turn_scratchpad = ScratchpadStore.from_dict(
            data.get("turn_scratchpad", {}) or {}
        )
        self.turn_scratchpad = turn_scratchpad
        # Phase-6 r21 F3: hydrate the REMAINING persisted fields. The
        # retry save in save_history_turn rewrites the whole document —
        # any field not adopted here would be reverted to this
        # manager's stale in-memory value (e.g. a GUI feature-state
        # update surviving in history but silently reverted in the
        # merged save). Mirrors the schema of _save_history_locked.
        # Phase-6 r22 F4: and mirrors the NORMALIZATION invariants of
        # _load_session — None-defaults for the two state dicts,
        # registry entries filtered to dict records, active-registry
        # fallback when the surface state is absent, research-source
        # validation + restore, normalized tool stats. A legacy or
        # partially populated winner must hydrate to the same state a
        # cold _load_session would produce.
        self.token_counts = data.get(
            "token_counts",
            {"input": 0, "output": 0, "total": 0, "total_cost": 0.0},
        )
        feature_state = data.get("feature_state")
        if isinstance(feature_state, dict):
            self.feature_state = feature_state
        self.feature_registry = {
            str(key): value
            for key, value in (data.get("feature_registry", {}) or {}).items()
            if isinstance(value, dict)
        }
        self.active_feature_id = data.get("active_feature_id")
        if (
            self.feature_state is None
            and self.active_feature_id in self.feature_registry
        ):
            self.feature_state = deepcopy(self.feature_registry[self.active_feature_id])
        teacher_state = data.get("teacher_state")
        if isinstance(teacher_state, dict):
            self.teacher_state = teacher_state
        self.teacher_registry = {
            str(key): value
            for key, value in (data.get("teacher_registry", {}) or {}).items()
            if isinstance(value, dict)
        }
        self.active_course_id = data.get("active_course_id")
        if (
            self.teacher_state is None
            and self.active_course_id
            and self.active_course_id in self.teacher_registry
        ):
            self.teacher_state = deepcopy(self.teacher_registry[self.active_course_id])
        saved_sources = data.get("research_sources", [])
        self.research_sources = (
            saved_sources if isinstance(saved_sources, list) else []
        )
        self.restore_research_sources()
        self.tool_stats = _normalize_tool_stats(data.get("tool_stats"))
        # Phase-6 r22 F4 + r23 F3: mirror _load_session normalization
        # INCLUDING None-defaults. The r22 shape only assigned when the
        # winner value was a dict, which left the manager's STALE
        # pre-conflict value in place when the winner intentionally
        # carries null — the retry then wrote the stale state back over
        # the winner's null. Reset to None first, exactly like a cold
        # load, then conditionally assign + registry fallback.
        self.feature_state = None
        if isinstance(feature_state, dict):
            self.feature_state = feature_state
        if (
            self.feature_state is None
            and self.active_feature_id in self.feature_registry
        ):
            self.feature_state = deepcopy(self.feature_registry[self.active_feature_id])
        self.teacher_state = None
        if isinstance(teacher_state, dict):
            self.teacher_state = teacher_state
        if (
            self.teacher_state is None
            and self.active_course_id
            and self.active_course_id in self.teacher_registry
        ):
            self.teacher_state = deepcopy(self.teacher_registry[self.active_course_id])
        # Phase-6 r24: re-apply only THIS TURN's task_memory delta
        # (added entries + baseline-value-guarded mutations) onto the
        # adopted winner. presave/baseline are bound by the caller
        # (save_history_turn); fall back to the manager's rolled
        # baseline when invoked directly.
        self._apply_task_memory_delta(
            getattr(self, "_pending_presave_snapshot", None),
            getattr(self, "_task_memory_baseline", None),
        )
        # Phase-6 r21 F4: Session aliases these manager objects
        # (sync_runtime_state). After the winner reload they would be
        # stale — re-sync the bound owner if one is bound.
        self._resync_runtime_owner()
        return {
            "history": self.history,
            "revision": self.revision,
            "summary_anchor": self.summary_anchor,
        }

    def save_history_if_current(self, folder_context_obj=None) -> bool:
        """Best-effort LWW save that refuses to clobber a NEWER write.

        Phase-6 r21 F5: post-turn best-effort saves (e.g. the GUI
        _run_send fallback after send_message returns, turn CAS already
        disarmed) used plain save_history — a concurrent surface write
        landing between the final turn save and this fallback was
        overwritten with stale in-memory state, re-opening exactly the
        window the turn CAS closed. This CASes against the manager's
        own in-memory revision: if the disk moved on, SKIP (return
        False) — the concurrent writer's document stays authoritative
        and the GUI watcher reloads it.
        """
        try:
            self.save_history(folder_context_obj, self.revision)
            return True
        except RevisionConflict:
            logger.debug(
                "save_history_if_current skipped: concurrent write is newer"
            )
            return False

    def save_history(self, folder_context_obj=None, expected_revision=None):
        """Persist the session document.

        Cross-surface continuity phase 1: when ``expected_revision`` is not
        None, the save is a compare-and-swap — if ``self.revision`` does not
        equal it (another surface wrote and reloaded in between), raise
        ``RevisionConflict`` WITHOUT writing. Callers that don't opt in keep
        last-writer-wins semantics. On success the revision is incremented
        (in memory and on disk).
        """
        if not self.current_session_name:
            logger.debug("save_history skipped — no session name set")
            return
        if expected_revision is not None and self.revision != expected_revision:
            raise RevisionConflict(expected_revision, self.revision)
        logger.debug(f"Saving history for session: {self.current_session_name}")
        filepath = self._get_filepath(self.current_session_name)
        # Round-13 F1: cross-process compare-and-swap. The in-memory check
        # above only guards this process; two surfaces (CLI + GUI) loaded at
        # revision N would both pass it. When the caller opts into CAS, take
        # a per-session advisory file lock, re-read the revision from disk
        # under the lock, and only then write — so a stale writer is
        # rejected against the state actually on disk.
        # Round-14 F2: locking failures FAIL CLOSED — a save that cannot
        # verify the on-disk revision is rejected rather than silently
        # degrading to an unlocked (stale-overwrite-prone) write.
        # Round-14 F3: a corrupt/unreadable session.json is NOT revision 0;
        # only a genuinely absent file (ENOENT) means an empty session.
        lock_fh = None
        if expected_revision is not None:
            # Round-15 F11: acquire/verify/write/release is ONE try — the
            # round-14 shape ran the unlock/close finally only around the
            # write, so a RevisionConflict raised during verification (and
            # any other early exit) leaked the open flock file object until
            # GC. A same-process CAS retry could then block on the leaked
            # fd until the exception's traceback (which pins the file
            # object) happened to be released.
            try:
                lock_fh = open(
                    self.session_lock_path(self.current_session_name), "w"
                )
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        disk = json.load(f)
                    disk_rev = int(disk.get("revision", 0) or 0)
                except FileNotFoundError:
                    disk_rev = 0  # genuinely absent session = revision 0
                except (OSError, ValueError):
                    # Corrupt or temporarily unreadable — cannot verify the
                    # caller's expectation; refusing is the safe direction.
                    disk_rev = -1
                if disk_rev != expected_revision:
                    raise RevisionConflict(expected_revision, disk_rev)
                if folder_context_obj:
                    self.folder_context = folder_context_obj
                return self._save_history_locked(
                    filepath, folder_context_obj, lock_fh
                )
            finally:
                if lock_fh is not None:
                    try:
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                    lock_fh.close()
        if folder_context_obj:
            self.folder_context = folder_context_obj
        return self._save_history_locked(filepath, folder_context_obj, None)

    def _save_history_locked(self, filepath, folder_context_obj, lock_fh) -> None:
        """Write the session document (round-14 F4: called with the CAS lock
        already held when lock_fh is not None; this method owns no locking).
        Returns True when a write happened, False when skipped."""
        if not self.current_session_name:
            return False
        tmp_path: str | None = None  # round-9 F5: except handler may fire
        try:                         # before tmp_path is assigned
            os.makedirs(self._get_session_dir(self.current_session_name), exist_ok=True)
            data = {
                # Cross-process write attribution. The GUI's file watcher
                # uses this to tell its OWN writes apart from concurrent
                # TUI writes on the same session.json. Safe to ignore on
                # load — it's just a marker.
                "__writer_pid__": os.getpid(),
                "__writer_at__": time.time(),
                "revision": self.revision + 1,
                # Round-50 F1: card metadata (variables, container_config)
                # is serialized BEFORE history. json.dump preserves
                # insertion order, and the GUI session-list's bounded
                # reader scans keys in file order — with history first it
                # hit the 256 KiB cap before the card keys appeared and the
                # listing endpoint fell back to a FULL document decode
                # (O(sessions × history) per GUI poll). History is the
                # biggest blob, so it goes LAST; key order in the saved
                # file is irrelevant to json.load semantics.
                "variables": self.variables,
                "container_config": self.container_config,
                "conversation_summary": self.conversation_summary,
                "summary_anchor": self.summary_anchor,
                "protected_indices": sorted(self.protected_indices),
                "provider_config": self.provider_config,
                "folder_context": self.folder_context.to_dict(),
                "collation_buffer": self.collation_buffer.to_dict(),
                "task_memory": self.task_memory.to_dict(),
                "turn_scratchpad": self.turn_scratchpad.to_dict(),
                "token_counts": self.token_counts,
                "feature_state": self.feature_state,
                "feature_registry": self.feature_registry,
                "active_feature_id": self.active_feature_id,
                "teacher_state": self.teacher_state,
                "teacher_registry": self.teacher_registry,
                "active_course_id": self.active_course_id,
                "research_sources": self.research_sources,
                "tool_stats": self.tool_stats,
                # Largest payload last — see the F1 note above.
                "history": self.history,
            }
            # Atomic save (codex round-7 F2): the old open(filepath, "w")
            # truncated session.json BEFORE serialization — a crash, disk
            # full, or concurrent reader could see an empty/partial file
            # and the only saved copy was lost. Write to a temp file in
            # the same directory, fsync, then atomically replace.
            # Round-13 F1: the temp file is unique per save (mkstemp) so two
            # processes saving under the session lock cannot clobber each
            # other's staging file.
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(filepath), prefix=".session-", suffix=".tmp"
            )
            # Round-15 F12: fd + staging path are owned resources — every
            # failure between mkstemp and the atomic replace must close the
            # fd and unlink the tmp file (the old shape leaked both when
            # fdopen, json.dump, or fsync raised).
            tmp_owned_fd = True
            tmp_owned_path = True
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    tmp_owned_fd = False  # fdopen now owns the descriptor
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, filepath)
                tmp_owned_path = False  # consumed by the replace
            finally:
                if tmp_owned_fd:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                if tmp_owned_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            self.revision = int(data["revision"])
            # Round-47 F4: fixed-size sidecar for the file watcher. The GUI
            # watcher previously json.load()-ed the ENTIRE session document
            # on its event loop just to read __writer_pid__; on a 100k-message
            # session that parse blocked SSE/HTTP on every observed write.
            # The sidecar carries the metadata the watcher needs in ~100
            # bytes; session.json remains the source of truth for state.
            try:
                meta_path = filepath + ".meta.json"
                meta_fd, meta_tmp = tempfile.mkstemp(
                    dir=os.path.dirname(filepath), prefix=".meta-", suffix=".tmp"
                )
                try:
                    with os.fdopen(meta_fd, "w") as mf:
                        json.dump({
                            "__writer_pid__": os.getpid(),
                            "revision": self.revision,
                        }, mf)
                        mf.flush()
                        os.fsync(mf.fileno())
                    os.replace(meta_tmp, meta_path)
                finally:
                    if os.path.exists(meta_tmp):
                        try:
                            os.unlink(meta_tmp)
                        except OSError:
                            pass
            except OSError:
                pass  # sidecar is an optimization; never break the save
            return True
        except (OSError, ValueError, TypeError) as e:
            if self.ui:
                self.ui.show_error(f"Warning: Could not save chat history: {e}")
            logger.error(f"Failed to save history: {e}")
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def restore_research_sources(self) -> None:
        """Hydrate the global citation engine from this session's snapshot."""
        try:
            from utils.citation_manager import get_citation_manager

            get_citation_manager().load_dict(self.research_sources)
        except Exception:
            logger.debug("Could not restore persisted research sources", exc_info=True)

    def snapshot_research_sources(self) -> None:
        """Capture sources immediately after a research tool updates them."""
        try:
            from utils.citation_manager import get_citation_manager

            self.research_sources = get_citation_manager().to_dict()
        except Exception:
            logger.debug("Could not snapshot research sources", exc_info=True)

    def get_feature_state(self):
        return (
            deepcopy(self.feature_state)
            if isinstance(self.feature_state, dict)
            else None
        )

    def get_feature_metadata_root(self) -> str:
        return os.path.join(self._get_session_dir(self.current_session_name), "features")

    def get_feature_metadata_path(self, feature_id: str) -> str:
        return os.path.join(
            self.get_feature_metadata_root(),
            f"{_slugify_feature_id(feature_id)}.json",
        )

    def get_feature_metadata_index(self) -> dict[str, str]:
        index = {}
        for feature in self.feature_registry.values():
            directory = str(feature.get("directory", "") or "").strip()
            metadata_path = str(feature.get("metadata_path", "") or "").strip()
            if directory and metadata_path:
                index[directory] = metadata_path
        return index

    def list_features(self) -> list[dict]:
        features = [deepcopy(feature) for feature in self.feature_registry.values()]
        features.sort(
            key=lambda feature: float(feature.get("updated_at", 0) or 0), reverse=True
        )
        return features

    def get_feature(self, feature_id: str | None = None) -> dict | None:
        resolved_feature_id = feature_id or self.active_feature_id
        if not resolved_feature_id:
            return None
        feature = self.feature_registry.get(str(resolved_feature_id))
        return deepcopy(feature) if isinstance(feature, dict) else None

    def upsert_feature(self, feature: dict | None) -> dict | None:
        if not isinstance(feature, dict):
            return None
        feature_id = str(
            feature.get("feature_id")
            or feature.get("id")
            or feature.get("feature_name")
            or ""
        ).strip()
        if not feature_id:
            return None
        feature_id = _slugify_feature_id(feature_id)
        record = deepcopy(feature)
        record["feature_id"] = feature_id
        # Preserve created_at from existing registry entry (if any)
        existing = self.feature_registry.get(feature_id)
        if isinstance(existing, dict) and existing.get("created_at"):
            record["created_at"] = existing["created_at"]
        elif not record.get("created_at"):
            record["created_at"] = time.time()
        record["updated_at"] = float(
            record.get("updated_at", time.time()) or time.time()
        )
        self.feature_registry[feature_id] = record
        return deepcopy(record)

    def activate_feature(self, feature_id: str) -> dict | None:
        record = self.get_feature(feature_id)
        if not record:
            return None
        self.active_feature_id = record["feature_id"]
        self.feature_state = deepcopy(record)
        self.save_history()
        return deepcopy(record)

    def archive_feature(self, feature_id: str) -> dict | None:
        resolved = _slugify_feature_id(feature_id)
        record = self.feature_registry.get(resolved)
        if not isinstance(record, dict):
            return None
        record["archived"] = True
        record["updated_at"] = time.time()
        if self.active_feature_id == resolved:
            self.active_feature_id = None
            self.feature_state = None
        self.save_history()
        return deepcopy(record)

    def unarchive_feature(self, feature_id: str) -> dict | None:
        resolved = _slugify_feature_id(feature_id)
        record = self.feature_registry.get(resolved)
        if not isinstance(record, dict):
            return None
        record.pop("archived", None)
        record["updated_at"] = time.time()
        self.save_history()
        return deepcopy(record)

    def delete_feature(self, feature_id: str) -> dict | None:
        resolved_feature_id = _slugify_feature_id(feature_id)
        record = self.feature_registry.pop(resolved_feature_id, None)
        if not isinstance(record, dict):
            return None
        metadata_path = str(record.get("metadata_path", "") or "").strip()
        if metadata_path and os.path.exists(metadata_path):
            os.remove(metadata_path)
        if self.active_feature_id == resolved_feature_id:
            self.active_feature_id = None
            self.feature_state = None
        self.save_history()
        return deepcopy(record)

    def create_feature_record(
        self,
        feature_name: str,
        *,
        directory: str,
        feature_request: str = "",
    ) -> dict:
        feature_id = self.allocate_feature_id(feature_name)
        metadata_path = self.get_feature_metadata_path(feature_id)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        record = {
            "type": "feature",
            "status": "draft",
            "feature_id": feature_id,
            "feature_name": feature_name.strip() or feature_id,
            "directory": directory,
            "metadata_path": metadata_path,
            "feature_plan": {
                "feature_id": feature_id,
                "feature_name": feature_name.strip() or feature_id,
                "feature_request": feature_request.strip()
                or feature_name.strip()
                or feature_id,
                "directory": directory,
                "metadata_path": metadata_path,
                "approved": False,
                "review_status": "pending",
                "review_notes": "",
                "overall_status": "not_started",
                "phases_completed": False,
                "phase_count": 0,
                "phases": [],
                "next_phase": None,
            },
            "blocker": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(record["feature_plan"], handle, indent=2)
        self.upsert_feature(record)
        self.active_feature_id = feature_id
        self.feature_state = deepcopy(record)
        self.save_history()
        return deepcopy(record)

    def allocate_feature_id(self, requested_id: str) -> str:
        base = _slugify_feature_id(requested_id)
        if base not in self.feature_registry:
            return base
        suffix = 2
        while True:
            candidate = f"{base}_{suffix}"
            if candidate not in self.feature_registry:
                return candidate
            suffix += 1

    def set_feature_state(self, state: dict | None, folder_context_obj=None):
        """Persist the active feature state.

        A completed feature is terminal for the active workflow.  Preserve its
        final plan in the registry, but archive it and clear the active state
        so the next interaction starts from the feature list rather than a
        finished board.
        """
        if isinstance(state, dict):
            # Re-derive status from feature_plan when caller did not provide
            # an explicit status override.
            feature_plan = state.get("feature_plan")
            explicit_status = str(state.get("status", "") or "").strip()
            should_derive = (not explicit_status) or explicit_status == "completed"
            if isinstance(feature_plan, dict) and should_derive:
                derived = derive_feature_state_status(feature_plan)
                state = {**state, "status": derived}
        self.feature_state = deepcopy(state) if isinstance(state, dict) else None
        if isinstance(self.feature_state, dict):
            record = self.upsert_feature(self.feature_state)
            if record:
                feature_id = record["feature_id"]
                if record.get("status") == "completed":
                    # Keep the completed record available under Archived, but
                    # do not leave a completed feature loaded.
                    archived_record = self.feature_registry[feature_id]
                    archived_record["archived"] = True
                    archived_record["updated_at"] = time.time()
                    self.feature_state = None
                    self.active_feature_id = None
                else:
                    self.active_feature_id = feature_id
        self.save_history(folder_context_obj)

    def clear_feature_state(self, folder_context_obj=None):
        self.feature_state = None
        self.active_feature_id = None
        self.save_history(folder_context_obj)

    # -------------------------------------------------- teacher mode

    def get_teacher_state(self) -> dict | None:
        return (
            deepcopy(self.teacher_state)
            if isinstance(self.teacher_state, dict)
            else None
        )

    def list_courses(self) -> list[dict]:
        courses = [deepcopy(course) for course in self.teacher_registry.values()]
        courses.sort(
            key=lambda course: float(course.get("updated_at", 0) or 0), reverse=True
        )
        return courses

    def get_course(self, course_id: str | None = None) -> dict | None:
        resolved = course_id or self.active_course_id
        if not resolved:
            return None
        record = self.teacher_registry.get(str(resolved))
        return deepcopy(record) if isinstance(record, dict) else None

    def upsert_teacher_course(self, record: dict | None) -> dict | None:
        if not isinstance(record, dict):
            return None
        course_id = str(record.get("course_id") or record.get("id") or "").strip()
        if not course_id:
            return None
        stored = deepcopy(record)
        stored["course_id"] = course_id
        stored["updated_at"] = float(stored.get("updated_at", time.time()) or time.time())
        self.teacher_registry[course_id] = stored
        return deepcopy(stored)

    def activate_course(self, course_id: str) -> dict | None:
        record = self.get_course(course_id)
        if not record:
            return None
        self.active_course_id = record["course_id"]
        self.teacher_state = deepcopy(record)
        self.save_history()
        return deepcopy(record)

    def clear_teacher_state(self, folder_context_obj=None):
        self.teacher_state = None
        self.active_course_id = None
        self.save_history(folder_context_obj)

    def delete_course(self, course_id: str) -> dict | None:
        record = self.teacher_registry.pop(str(course_id), None)
        if not isinstance(record, dict):
            return None
        if self.active_course_id == course_id:
            self.active_course_id = None
            self.teacher_state = None
        self.save_history()
        return deepcopy(record)

    def switch_session(self, name):
        logger.info(f"Switching to session: {name}")
        self.save_history()
        self._load_session(name)
        if self.ui:
            self.ui.show_info(f"Switched to session: '{name}'")
        self.view_history()

    def new_session(self, name=None, provider_name=None, model_name=None, session_type="workspace"):
        logger.info(
            f"Creating new session: {name} (provider={provider_name}, model={model_name})"
        )
        self.save_history()
        if not name:
            name = f"chat_{int(time.time())}"
        self.folder_context = FolderContext()
        self.current_session_name = name
        self.collation_buffer = CollationBuffer()
        self.task_memory = TaskMemoryStore()
        self.turn_scratchpad = ScratchpadStore()
        self.feature_state = None
        self.feature_registry = {}
        self.active_feature_id = None
        self.teacher_state = None
        self.teacher_registry = {}
        self.active_course_id = None
        self.research_sources = []
        self.restore_research_sources()
        self.conversation_summary = ""
        self.summary_anchor = 0
        self.history = []
        self.provider_config = {"provider": provider_name, "model": model_name}
        self.token_counts = {
            "input": 0,
            "output": 0,
            "total": 0,
            "total_cost": 0.0,
            "cached": 0,
            "reasoning": 0,
        }
        self.variables.clear()
        self.variables.update(DEFAULT_VARIABLES)
        self.container_config = {}
        from mu.tools.capabilities import normalize_session_type

        self.variables["session_type"] = normalize_session_type(session_type)
        if self.variables["session_type"] == "container":
            self.variables["yolo"] = True
            self.variables["strict_mode"] = False
            self.variables["plan_mode"] = False
            self.variables["security_allow_secret_paths"] = False
        self.save_history()
        if self.ui:
            self.ui.show_info(f"Started new session: '{name}'")

    def list_sessions(self):
        logger.debug("Listing sessions")
        if self.current_session_name and not os.path.exists(self._get_filepath(self.current_session_name)):
            self.save_history()

        files = glob.glob(os.path.join(_history_dir(), "sessions", "*", "session.json"))
        if self.ui:
            # We might want a specific UI method for listing sessions
            self.ui.show_info("\n=== Available Conversations ===")
            for f in files:
                name = os.path.basename(os.path.dirname(f))
                indicator = "*" if name == self.current_session_name else " "
                mod_time = datetime.datetime.fromtimestamp(
                    os.path.getmtime(f)
                ).strftime("%Y-%m-%d %H:%M")
                self.ui.show_info(f" {indicator} {name:<20} ({mod_time})")

    def get_session_list(self):
        files = glob.glob(os.path.join(_history_dir(), "sessions", "*", "session.json"))
        sessions = []
        for f in files:
            sessions.append(os.path.basename(os.path.dirname(f)))
        return sorted(sessions)

    def get_session_list_with_type(self):
        """Return ``[(name, session_type), …]`` sorted by name.

        Reads ``variables.session_type`` from each saved session.json.
        Falls back to ``"workspace"`` when the key is missing or unreadable.
        """
        files = glob.glob(os.path.join(_history_dir(), "sessions", "*", "session.json"))
        result = []
        for f in files:
            name = os.path.basename(os.path.dirname(f))
            st = "workspace"
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                st = str((data.get("variables") or {}).get("session_type") or "workspace")
            except (OSError, ValueError):
                pass
            result.append((name, st))
        return sorted(result, key=lambda pair: pair[0])

    def delete_session(self, name):
        logger.info(f"Deleting session: {name}")
        if name == self.current_session_name:
            if self.ui:
                self.ui.show_error("Cannot delete active session.")
            return

        session_dir = self._get_session_dir(name)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            if self.ui:
                self.ui.show_info(f"Deleted session: '{name}'")
        else:
            if self.ui:
                self.ui.show_error(f"Session '{name}' not found.")

    def rename_session(self, old_name: str, new_name: str) -> bool:
        old_name = str(old_name or "").strip()
        new_name = str(new_name or "").strip()
        if not old_name or not new_name:
            raise ValueError("Both old_name and new_name are required.")
        if old_name == new_name:
            return True

        old_dir = self._get_session_dir(old_name)
        new_dir = self._get_session_dir(new_name)
        if not os.path.exists(old_dir):
            raise FileNotFoundError(f"Session '{old_name}' not found.")
        if os.path.exists(new_dir):
            raise FileExistsError(f"Session '{new_name}' already exists.")

        os.rename(old_dir, new_dir)
        if self.current_session_name == old_name:
            self.current_session_name = new_name
            self.save_history()
        if self.ui:
            self.ui.show_info(f"Renamed session '{old_name}' to '{new_name}'.")
        return True

    def clear_current_history(self):
        logger.info(f"Clearing history for session: {self.current_session_name}")
        self.history = []
        self.conversation_summary = ""
        self.summary_anchor = 0
        self.token_counts = {
            "input": 0,
            "output": 0,
            "total": 0,
            "total_cost": 0.0,
            "cached": 0,
            "reasoning": 0,
        }
        self.save_history()
        if self.ui:
            self.ui.show_info("Current chat history cleared.")

    def reset_current_session_state(self):
        logger.info(f"Resetting session state for session: {self.current_session_name}")
        self.history = []
        self.conversation_summary = ""
        self.summary_anchor = 0
        self.folder_context = FolderContext()
        self.collation_buffer = CollationBuffer()
        self.task_memory = TaskMemoryStore()
        self.turn_scratchpad = ScratchpadStore()
        self.token_counts = {
            "input": 0,
            "output": 0,
            "total": 0,
            "total_cost": 0.0,
            "cached": 0,
            "reasoning": 0,
        }
        self.feature_state = None
        self.feature_registry = {}
        self.active_feature_id = None
        self.teacher_state = None
        self.teacher_registry = {}
        self.active_course_id = None

        feature_root = self.get_feature_metadata_root()
        if os.path.isdir(feature_root):
            for entry in glob.glob(os.path.join(feature_root, "*.json")):
                os.remove(entry)

        self.save_history()

    # History summarization & token-budget rolling moved to
    # mu/session/history.py (HistoryMixin). See top of file for the import.

    def view_history(self):
        if not self.history:
            if self.ui:
                self.ui.show_info("No history in this session.")
            return

        if self.ui:
            self.ui.show_info(f"\nConversation History ({self.current_session_name})\n")

            for turn in self.history:
                role = turn["role"]
                for part in turn.get("parts", []):
                    p_type = part.get("type")
                    if p_type == "text":
                        self.ui.render_message(role, part["text"])
                    elif p_type == "file":
                        mime = part.get("file_ref", {}).get("mime_type", "file")
                        self.ui.show_info(f"[Attached File: {mime}]")
                    elif p_type == "image_input":
                        img = part.get("image", {}) or {}
                        src = img.get("source") or img.get("mime_type", "image")
                        self.ui.show_info(f"[Attached Image: {src}]")
                    elif p_type == "tool_call":
                        self.ui.show_info(f"  [Tool Call: {part.get('tool_name')}]")
                    elif p_type == "tool_result":
                        res_preview = str(part.get("tool_result", ""))[:50].replace(
                            "\n", ""
                        )
                        self.ui.show_info(f"  [Tool Result: {res_preview}...]")

    def compact_completed_turn(self):
        """
        Collapses the most recent agentic turn.
        Identifies the last 'user' prompt and the last 'assistant' text response,
        then removes all intermediate tool calls and results between them.
        """
        if len(self.history) < 2:
            return

        # 1. Find the index of the last 'user' message that started this turn.
        # Round-14 F10: synthetic messages (loop-watchdog nudges, empty-response
        # pokes) are stored as user messages but are NOT turn boundaries —
        # skip them so the collapse covers the whole turn instead of only the
        # suffix after the last nudge.
        last_user_idx = -1
        for i in range(len(self.history) - 1, -1, -1):
            msg = self.history[i]
            if msg.get("role") == "user" and not msg.get("synthetic"):
                last_user_idx = i
                break

        if last_user_idx == -1:
            return

        # 2. Extract the final assistant text parts from the end of history
        final_assistant_parts = []
        for i in range(len(self.history) - 1, last_user_idx, -1):
            if self.history[i]["role"] == "assistant":
                # Collect text parts only
                text_parts = [
                    p for p in self.history[i]["parts"] if p["type"] == "text"
                ]
                if text_parts:
                    # We reverse them back because we are iterating backwards
                    final_assistant_parts = text_parts + final_assistant_parts
                    # If we found the "final" response message, we stop looking for more text
                    break

        # 3. Reconstruct history
        # Keep everything before the current turn
        new_history = self.history[: last_user_idx + 1]

        # Append the collapsed assistant response if we found text
        if final_assistant_parts:
            new_history.append({"role": "assistant", "parts": final_assistant_parts})

        self.history = new_history
        self.summary_anchor = min(self.summary_anchor, len(self.history))
        # Phase-6 r22 F1: the collapse reindexes history. The armed
        # turn's start index points at the turn prompt in the OLD
        # list; after removal it can point past the end (empty
        # conflict-merge slice -> turn messages lost) or mid-turn
        # (wrong dedupe multiset). Remap to the boundary index of the
        # collapsed turn so the suffix still covers exactly the turn.
        if self._active_turn_start_index is not None:
            self._active_turn_start_index = min(
                self._active_turn_start_index,
                len(self.history) - 1 if self.history else 0,
            )
