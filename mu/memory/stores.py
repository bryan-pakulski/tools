"""Persistent memory and turn-local scratchpad stores for agentic sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Valid lifecycle statuses for memory entries.
ACTIVE = "active"
DONE = "done"
SUPERSEDED = "superseded"
ARCHIVED = "archived"
STALE = "stale"
ALLOWED_STATUSES = {ACTIVE, DONE, SUPERSEDED, ARCHIVED, STALE}

# Status weights for eviction scoring. Lower weight = evicted sooner.
STATUS_EVIC_WEIGHTS: Dict[str, float] = {
    ACTIVE: 1.0,
    STALE: 0.8,
    DONE: 0.5,
    SUPERSEDED: 0.3,
    ARCHIVED: 0.1,
}


@dataclass
class MemoryEntry:
    id: int
    content: str
    tags: List[str] = field(default_factory=list)
    source: str = ""
    kind: str = "observation"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    hits: int = 0
    status: str = ACTIVE
    superseded_by: Optional[int] = None
    supersedes: Optional[int] = None
    # Stable UUID in the cross-session Memory Ledger. Empty means this
    # working-memory entry has not yet been promoted; "rejected" means the
    # safety floor refused credential-like content and must not retry it.
    durable_id: str = ""
    # Turn index of the last explicit retrieval/save (active reliance), used
    # by staleness decay (`apply_staleness_decay`). 0 = never touched since
    # load. Passive L3 injection does NOT bump this (injection is not active
    # reliance), so entries that are only ever surfaced — never searched or
    # re-saved — eventually decay to STALE and drop out of the active set.
    last_hit_turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "source": self.source,
            "kind": self.kind,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hits": self.hits,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "durable_id": self.durable_id,
            "last_hit_turn": self.last_hit_turn,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=int(data.get("id", 0)),
            content=str(data.get("content", "")),
            tags=list(data.get("tags", [])),
            source=str(data.get("source", "")),
            kind=str(data.get("kind") or "observation"),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            hits=int(data.get("hits", 0)),
            status=str(data.get("status") or ACTIVE),
            superseded_by=data.get("superseded_by"),
            supersedes=data.get("supersedes"),
            durable_id=str(data.get("durable_id", "") or ""),
            last_hit_turn=int(data.get("last_hit_turn", 0)),
        )


class BaseNoteStore:
    title = "Notes"

    # Default kind weights for eviction scoring. Higher weight = kept longer.
    # Decisions (architectural choices, design rationale) are most valuable.
    # Findings (root causes, verified facts) are next.
    # Observations (general notes) are lowest priority.
    DEFAULT_EVIC_KIND_WEIGHTS: Dict[str, float] = {
        "decision": 3.0,
        "finding": 2.0,
        "observation": 1.0,
        "goal": 2.5,
    }

    def __init__(
        self,
        max_entries: int = 64,
        summary_char_limit: int = 2_000,
        eviction_kind_weights: Dict[str, float] | None = None,
    ) -> None:
        self.max_entries = max_entries
        self.summary_char_limit = summary_char_limit
        self.entries: List[MemoryEntry] = []
        self._next_id = 1
        self.eviction_kind_weights = eviction_kind_weights or dict(
            self.DEFAULT_EVIC_KIND_WEIGHTS
        )
        # Monotonic turn counter for staleness decay. The agent loop calls
        # `advance_turn()` once per turn before `apply_staleness_decay()`.
        # Persisted so decay stays consistent across session resume.
        self.turn_count: int = 0
        # Transient log of entries evicted by `_enforce_limit` (R12, FM-11).
        # The agent loop drains this into an L3 notice block so the model
        # learns that a memory it relied on is gone (preventing silent
        # re-derivation). NOT persisted — `to_dict`/`from_dict` skip it.
        self.eviction_log: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "summary_char_limit": self.summary_char_limit,
            "next_id": self._next_id,
            "turn_count": self.turn_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseNoteStore":
        store = cls(
            max_entries=int(data.get("max_entries", 64)),
            summary_char_limit=int(data.get("summary_char_limit", 2_000)),
        )
        store._next_id = int(data.get("next_id", 1))
        store.turn_count = int(data.get("turn_count", 0))
        store.entries = [
            MemoryEntry.from_dict(item) for item in data.get("entries", [])
        ]
        if store.entries and store._next_id <= max(entry.id for entry in store.entries):
            store._next_id = max(entry.id for entry in store.entries) + 1
        return store

    def clear(self) -> None:
        self.entries.clear()
        self._next_id = 1

    def clear_excluding(self, except_tags: set[str]) -> int:
        """Drop every entry NOT carrying one of ``except_tags``.

        Used at turn start to wipe ephemeral scratchpad notes while
        preserving the persistent `todo`-tagged ledger (the agent's
        self-managed task plan survives across turns). Returns the number
        of entries removed. ``_next_id`` is recomputed so new saves don't
        collide with retained entries.
        """
        if not except_tags:
            return self._do_clear()
        kept = [e for e in self.entries if any(t in except_tags for t in e.tags)]
        removed = len(self.entries) - len(kept)
        self.entries = kept
        self._next_id = (max(e.id for e in kept) + 1) if kept else 1
        return removed

    def _do_clear(self) -> int:
        n = len(self.entries)
        self.entries.clear()
        self._next_id = 1
        return n

    def advance_turn(self) -> None:
        """Bump the monotonic turn counter. Called once per turn by the
        agent loop, before `apply_staleness_decay`."""
        self.turn_count += 1

    def apply_staleness_decay(self, stale_after_turns: int) -> int:
        """Demote ACTIVE entries not hit in ``stale_after_turns`` turns to
        STALE. The single mechanism that keeps the active set honest: "active"
        means "recently mattered", not "ever saved". Reversible — a search
        hit or re-save promotes a STALE entry back to ACTIVE automatically
        (see `search` / `save`), so decay never loses information the agent
        is actually using.

        Only ACTIVE entries are touched; done/superseded/archived are left
        alone (they're already off the active set). Returns the count
        demoted. No-op when ``stale_after_turns <= 0`` (decay disabled).
        """
        if stale_after_turns <= 0:
            return 0
        threshold = self.turn_count - stale_after_turns
        demoted = 0
        for entry in self.entries:
            if entry.status == ACTIVE and entry.last_hit_turn <= threshold:
                entry.status = STALE
                entry.updated_at = time.time()
                demoted += 1
        return demoted

    def save(
        self,
        content: str,
        tags: List[str] | None = None,
        source: str = "",
        kind: str = "observation",
        status: str = ACTIVE,
    ) -> MemoryEntry:
        tags = [str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()]
        content = str(content or "").strip()

        existing = next(
            (
                entry
                for entry in self.entries
                if entry.content == content and sorted(entry.tags) == sorted(tags)
            ),
            None,
        )
        if existing:
            existing.updated_at = time.time()
            existing.hits += 1
            existing.last_hit_turn = self.turn_count
            if source and not existing.source:
                existing.source = source
            if kind and not existing.kind:
                existing.kind = kind
            # Re-saving identical content is active reliance — if the entry
            # had decayed to STALE, promote it back to ACTIVE (decay is
            # reversible through use, not just through reactivate_memory).
            if existing.status == STALE:
                existing.status = status if status != STALE else ACTIVE
            elif status != existing.status:
                # Update status if content+tags match but status differs
                existing.status = status
            return existing

        entry = MemoryEntry(
            id=self._next_id,
            content=content,
            tags=tags,
            source=source,
            kind=kind,
            status=status,
            last_hit_turn=self.turn_count,
        )
        self._next_id += 1
        self.entries.append(entry)
        self._enforce_limit()
        return entry

    def get_entry(self, entry_id: int) -> Optional[MemoryEntry]:
        return next(
            (entry for entry in self.entries if entry.id == entry_id), None
        )

    def import_entries(self, items: List[Dict[str, Any]]) -> List[MemoryEntry]:
        """Import serialized entries with collision-safe id remapping.

        Used by the session-manager conflict path to restore entries a
        turn captured locally. The numeric id space is session-local,
        so an imported id may already exist (a concurrent surface
        appended a different entry with the same counter value): such
        items get a FRESH id. Cross-entry references (supersedes /
        superseded_by) inside the batch are remapped to the fresh ids.
        Returns the imported entries (new ids for remapped items).
        """
        if not items:
            return []
        existing_ids = {entry.id for entry in self.entries}
        # Round-25 F2: allocate one id PER ITEM (a list keyed by item
        # index), not per id. Duplicate ids within the batch — or two
        # items both lacking an id — previously overwrote each other's
        # id_map entry, so both entries landed on the same fresh id in
        # the second pass.
        item_ids: List[int] = []
        assigned: List[int] = []

        def _fresh() -> int:
            fresh = max(existing_ids) + 1 if existing_ids else 1
            existing_ids.add(fresh)
            return fresh

        for item in items:
            try:
                item_id = int(item.get("id", 0))
            except (TypeError, ValueError):
                item_id = 0
            item_ids.append(item_id)
            if item_id and item_id not in existing_ids:
                assigned.append(item_id)
                existing_ids.add(item_id)
            else:
                assigned.append(_fresh())
        # Reference remap: first item claiming an id wins; a later
        # item whose id collided got a fresh id, so refs to the
        # ORIGINAL id resolve to the first claimant. Kept ids map to
        # themselves; ids outside the batch fall through unchanged.
        ref_map: Dict[int, int] = {}
        for item_id, new_id in zip(item_ids, assigned):
            if item_id and item_id not in ref_map:
                ref_map[item_id] = new_id

        def _remap(ref: Optional[int]) -> Optional[int]:
            if ref is None:
                return None
            try:
                return ref_map.get(int(ref), int(ref))
            except (TypeError, ValueError):
                return ref

        imported: List[MemoryEntry] = []
        for item, item_id, new_id in zip(items, item_ids, assigned):
            entry = MemoryEntry.from_dict(item)
            entry.id = new_id
            entry.supersedes = _remap(entry.supersedes)
            entry.superseded_by = _remap(entry.superseded_by)
            imported.append(entry)
            existing_ids.add(entry.id)
        self.entries.extend(imported)
        max_id = max(entry.id for entry in self.entries)
        if self._next_id <= max_id:
            self._next_id = max_id + 1
        self._enforce_limit()
        return imported

    def update_status(self, entry_id: int, status: str) -> Optional[MemoryEntry]:
        if status not in ALLOWED_STATUSES:
            return None
        entry = self.get_entry(entry_id)
        if entry is None:
            return None
        entry.status = status
        entry.updated_at = time.time()
        return entry

    def supersede(self, old_id: int, new_id: int) -> Optional[tuple]:
        old = self.get_entry(old_id)
        new = self.get_entry(new_id)
        if old is None or new is None:
            return None
        old_status = old.status
        new_status = new.status
        old.status = SUPERSEDED
        old.superseded_by = new_id
        new.supersedes = old_id
        old.updated_at = time.time()
        new.updated_at = time.time()
        return (old, new, old_status, new_status)

    def _rank_by_relevance(
        self,
        query: str,
        *,
        status_set: set[str] | None,
        kind_set: set[str] | None,
        exclude_tags: set[str],
    ) -> List[tuple]:
        """Score every (filter-passing) entry against `query` WITHOUT
        mutating hits/updated_at. Returns ``(score, updated_at, entry)``
        tuples sorted by (score desc, updated_at desc). A non-empty query
        with no term matches yields score 0 for that entry; an empty
        query yields score 1 for every passing entry (recency-only).

        Extracted from `search` so `render_summary` can bias L3 injection
        toward the current turn's topic without the `hits += 1` /
        `updated_at = now` side effects that `search` applies (R6, FM-7).
        """
        terms = [term for term in str(query or "").lower().split() if term]
        ranked = []
        for entry in self.entries:
            # Status filtering
            if status_set is not None and entry.status not in status_set:
                continue
            # Kind filtering
            if kind_set is not None and entry.kind not in kind_set:
                continue
            # Tag exclusion filtering
            if exclude_tags and any(tag in exclude_tags for tag in entry.tags):
                continue

            haystack = " ".join(
                [entry.content, " ".join(entry.tags), entry.source]
            ).lower()
            score = 0
            for term in terms:
                if term in haystack:
                    score += 2
                if term in entry.content.lower():
                    score += 1
            if not terms:
                score = 1
            ranked.append((score, entry.updated_at, entry))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked

    def search(
        self,
        query: str = "",
        limit: int = 5,
        status_filter: str | list[str] | None = None,
        kind_filter: str | list[str] | None = None,
        tags_exclude: list[str] | None = None,
        include_all: bool = False,
    ) -> List[MemoryEntry]:
        # Normalize status_filter to a set
        # Default search surfaces ACTIVE + STALE (decayed-but-recent). A
        # search hit on a STALE entry reactivates it to ACTIVE (see below), so
        # decay is self-correcting through use — the agent doesn't need
        # `include_all` to find and revive relevant-but-decayed knowledge.
        # L3 *injection* (`render_summary`) keeps the stricter active-first
        # priority, so the auto-surfaced context window stays clean even
        # though search can still surface stale entries on demand.
        if status_filter is None and not include_all:
            status_set = {ACTIVE, STALE}
        elif status_filter is None and include_all:
            status_set = None  # no status filtering
        else:
            if isinstance(status_filter, str):
                status_set = {status_filter}
            else:
                status_set = set(status_filter)

        # Normalize kind_filter to a set
        if kind_filter is None:
            kind_set = None
        elif isinstance(kind_filter, str):
            kind_set = {kind_filter}
        else:
            kind_set = set(kind_filter)

        # Normalize tags_exclude
        exclude_tags = set(tags_exclude or [])

        ranked = self._rank_by_relevance(
            query,
            status_set=status_set,
            kind_set=kind_set,
            exclude_tags=exclude_tags,
        )
        results = [entry for score, _, entry in ranked if score > 0][: max(1, limit)]
        for entry in results:
            entry.hits += 1
            entry.updated_at = time.time()
            entry.last_hit_turn = self.turn_count
            # An explicit search hit is proof of relevance — if the entry had
            # decayed to STALE, reactivate it. Decay is self-correcting: use
            # it and it comes back to the active set automatically.
            if entry.status == STALE:
                entry.status = ACTIVE
        return results

    def search_readonly(
        self,
        query: str = "",
        limit: int = 5,
        status_filter: str | list[str] | None = None,
        kind_filter: str | list[str] | None = None,
        tags_exclude: list[str] | None = None,
        include_all: bool = False,
    ) -> List[MemoryEntry]:
        """Pure search for human browsing and API reads.

        Unlike ``search`` this never increments hits, changes timestamps or
        reactivates stale entries. Only committed model reliance may mutate
        working-memory usage metadata.
        """

        if status_filter is None and not include_all:
            status_set = {ACTIVE, STALE}
        elif status_filter is None:
            status_set = None
        elif isinstance(status_filter, str):
            status_set = {status_filter}
        else:
            status_set = set(status_filter)
        if kind_filter is None:
            kind_set = None
        elif isinstance(kind_filter, str):
            kind_set = {kind_filter}
        else:
            kind_set = set(kind_filter)
        ranked = self._rank_by_relevance(
            query,
            status_set=status_set,
            kind_set=kind_set,
            exclude_tags=set(tags_exclude or []),
        )
        return [entry for score, _, entry in ranked if score > 0][: max(1, limit)]

    def list_entries(
        self, limit: int = 10, status_filter: str | list[str] | None = None
    ) -> List[MemoryEntry]:
        # Normalize status_filter
        if status_filter is None:
            status_set = None
        elif isinstance(status_filter, str):
            status_set = {status_filter}
        else:
            status_set = set(status_filter)

        filtered = self.entries
        if status_set is not None:
            filtered = [e for e in filtered if e.status in status_set]

        return sorted(filtered, key=lambda entry: entry.updated_at, reverse=True)[
            : max(1, limit)
        ]

    def render_summary(
        self,
        limit: int = 8,
        include_archived: bool = False,
        query: str = "",
    ) -> str:
        # Partition by status (recency-ordered within each partition).
        active_entries: List[MemoryEntry] = []
        done_entries: List[MemoryEntry] = []
        other_entries: List[MemoryEntry] = []
        for entry in sorted(self.entries, key=lambda e: e.updated_at, reverse=True):
            if entry.status == ACTIVE:
                active_entries.append(entry)
            elif entry.status == DONE:
                done_entries.append(entry)
            elif entry.status == ARCHIVED and not include_archived:
                continue
            else:
                other_entries.append(entry)

        ordered: List[MemoryEntry] = []
        seen: set[int] = set()

        # R6 / FM-7: when a query (the current turn's user text) is given,
        # bias L3 injection toward relevance hits FIRST, then fill the
        # remaining slots by the recency partition. Uses the non-mutating
        # `_rank_by_relevance` so injection doesn't perturb hits/eviction.
        terms = [t for t in str(query or "").lower().split() if t]
        if terms:
            ranked = self._rank_by_relevance(
                query,
                status_set=None,
                kind_set=None,
                exclude_tags=set(),
            )
            relevant = [entry for score, _, entry in ranked if score > 0]
            for entry in relevant:
                if len(ordered) >= limit:
                    break
                # Respect the archived-exclusion policy even for relevant hits.
                if entry.status == ARCHIVED and not include_archived:
                    continue
                if entry.id in seen:
                    continue
                ordered.append(entry)
                seen.add(entry.id)

        # Fill remaining slots by the recency partition (active first,
        # done capped at 2, then other non-archived). When there is no
        # query this branch alone reproduces the original recency-only
        # ordering exactly.
        if len(ordered) < limit:
            remaining = limit - len(ordered)
            done_cap = min(2, remaining)
            fill = active_entries + done_entries[:done_cap] + other_entries
            for entry in fill:
                if len(ordered) >= limit:
                    break
                if entry.id in seen:
                    continue
                ordered.append(entry)
                seen.add(entry.id)

        if not ordered:
            return ""

        lines = [f"### {self.title}"]
        for entry in ordered:
            tags = f" [{', '.join(entry.tags)}]" if entry.tags else ""
            source = f" ({entry.source})" if entry.source else ""
            lines.append(
                f"- #{entry.id} [{entry.status}]{tags}{source}: {entry.content}"
            )

        summary = "\n".join(lines)
        if len(summary) <= self.summary_char_limit:
            return summary
        return summary[: self.summary_char_limit - 3] + "..."

    def format_results(self, entries: List[MemoryEntry]) -> str:
        if not entries:
            return f"No {self.title.lower()} entries matched."

        lines = []
        for entry in entries:
            tags = json.dumps(entry.tags)
            source = entry.source or "n/a"
            lines.append(f"#{entry.id} [{entry.status}] kind={entry.kind} tags={tags} source={source} :: {entry.content}")
        return "\n".join(lines)

    def status_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in ALLOWED_STATUSES}
        for entry in self.entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return counts

    def _eviction_score(self, entry: MemoryEntry) -> float:
        """Score an entry for eviction. Lower score = evicted first.

        Combines hits (access frequency) with kind weight (semantic
        importance) and status weight (lifecycle relevance). Uses (hits + 1)
        so that kind/status weight matters even for newly-created entries.
        """
        kind_weight = self.eviction_kind_weights.get(entry.kind, 1.0)
        status_weight = STATUS_EVIC_WEIGHTS.get(entry.status, 1.0)
        return float(entry.hits + 1) * kind_weight * status_weight

    def _enforce_limit(self) -> None:
        if len(self.entries) <= self.max_entries:
            return

        # Status-aware + kind-aware eviction: sort by eviction score (lowest
        # evicted first). Falls back to updated_at as tiebreaker — older
        # entries evicted before newer ones with the same score.
        # Eviction order: archived → superseded → done → stale → active
        # (lower STATUS_EVIC_WEIGHTS = evicted first)
        self.entries.sort(
            key=lambda entry: (self._eviction_score(entry), entry.updated_at)
        )
        evicted: List[MemoryEntry] = []
        while len(self.entries) > self.max_entries:
            evicted.append(self.entries.pop(0))
        if evicted:
            # Record a one-line notice per evicted entry (R12, FM-11) so the
            # agent loop can surface the eviction to the model. Cap the log
            # to the last 20 evictions to bound memory across long sessions.
            for entry in evicted:
                preview = entry.content[:80]
                self.eviction_log.append(
                    f"#{entry.id} [{entry.kind}] evicted to make room: {preview}"
                )
            if len(self.eviction_log) > 20:
                del self.eviction_log[: len(self.eviction_log) - 20]

    def drain_eviction_log(self) -> List[str]:
        """Return and clear pending eviction notices (R12, FM-11).

        The agent loop calls this each turn after rendering L3 memory so
        eviction events are surfaced exactly once, then forgotten."""
        if not self.eviction_log:
            return []
        pending = list(self.eviction_log)
        self.eviction_log.clear()
        return pending


class TaskMemoryStore(BaseNoteStore):
    title = "In-Task Memory"

    def __init__(self, max_entries: int = 1024, summary_char_limit: int = 16_000) -> None:
        super().__init__(max_entries=max_entries, summary_char_limit=summary_char_limit)


class ScratchpadStore(BaseNoteStore):
    title = "Turn Scratchpad"

    def __init__(self, max_entries: int = 256, summary_char_limit: int = 8_000) -> None:
        super().__init__(max_entries=max_entries, summary_char_limit=summary_char_limit)

    def save(
        self,
        content: str,
        tags: List[str] | None = None,
        source: str = "",
        kind: str = "",
        status: str = ACTIVE,
    ) -> MemoryEntry:
        # Scratchpad is ephemeral — lifecycle does not apply. Force status=active.
        return super().save(content, tags=tags, source=source, kind=kind, status=ACTIVE)
