"""SQLite DDL and row mapping for the durable memory ledger.

Extracted from mu/memory/ledger.py: the schema executescript and the
row->MemoryItem mapper are pure functions of a connection/row, so they
live here while the ledger class keeps lifecycle orchestration.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .models import MemoryItem


def _loads(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


SCHEMA_DDL = """                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    statement TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    scope_label TEXT NOT NULL DEFAULT '',
                    lifecycle TEXT NOT NULL DEFAULT 'active',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    trust_origin TEXT NOT NULL DEFAULT 'model',
                    verification TEXT NOT NULL DEFAULT 'unverified',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    sensitivity TEXT NOT NULL DEFAULT 'normal',
                    egress_policy TEXT NOT NULL DEFAULT 'any',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    source_refs_json TEXT NOT NULL DEFAULT '[]',
                    relations_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_recalled_at REAL,
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    etag TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                    ON memories(scope_type, scope_key, lifecycle, updated_at DESC);
                -- Round-49 F5: the per-turn fallback query orders by
                -- (pinned DESC, updated_at DESC) within eligible scopes —
                -- the scope index above forced a full sort of every active
                -- row. pinned sits between lifecycle and updated_at so the
                -- ordering is served directly by the index.
                CREATE INDEX IF NOT EXISTS idx_memories_scope_pinned
                    ON memories(scope_type, scope_key, lifecycle, pinned DESC, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_hash
                    ON memories(scope_type, scope_key, content_hash);
                CREATE INDEX IF NOT EXISTS idx_memories_kind
                    ON memories(kind, lifecycle, updated_at DESC);

                CREATE TABLE IF NOT EXISTS memory_revisions (
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    statement TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    PRIMARY KEY(memory_id, version),
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    memory_id TEXT,
                    version INTEGER NOT NULL DEFAULT 0,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    device_id TEXT NOT NULL DEFAULT 'local',
                    before_hash TEXT NOT NULL DEFAULT '',
                    after_json TEXT NOT NULL DEFAULT '{}',
                    reason TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_item
                    ON memory_events(memory_id, created_at DESC);
                -- Round-49 F7: unscoped recent-events queries order by
                -- created_at but only the memory_id-prefixed index existed —
                -- years-long event tables sorted fully on every listing.
                CREATE INDEX IF NOT EXISTS idx_memory_events_created
                    ON memory_events(created_at DESC, event_id DESC);

                CREATE TABLE IF NOT EXISTS memory_recall_receipts (
                    id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL DEFAULT '',
                    query_text TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    budget_tokens INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    included_json TEXT NOT NULL,
                    excluded_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_receipts_session
                    ON memory_recall_receipts(session_name, created_at DESC);

                CREATE TABLE IF NOT EXISTS memory_sync_tombstones (
                    memory_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    forgotten_at REAL NOT NULL
                );"""


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_DDL)


def row_to_item(row: sqlite3.Row | None) -> MemoryItem | None:
        if row is None:
            return None
        return MemoryItem(
            id=str(row["id"]),
            version=int(row["version"]),
            statement=str(row["statement"] or ""),
            kind=str(row["kind"] or "observation"),
            scope_type=str(row["scope_type"]),
            scope_key=str(row["scope_key"]),
            scope_label=str(row["scope_label"] or ""),
            lifecycle=str(row["lifecycle"] or "active"),
            pinned=bool(row["pinned"]),
            trust_origin=str(row["trust_origin"] or "model"),
            verification=str(row["verification"] or "unverified"),
            confidence=float(row["confidence"] or 0.0),
            sensitivity=str(row["sensitivity"] or "normal"),
            egress_policy=str(row["egress_policy"] or "any"),
            tags=list(_loads(row["tags_json"], [])),
            source_refs=list(_loads(row["source_refs_json"], [])),
            relations=list(_loads(row["relations_json"], [])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_recalled_at=(
                float(row["last_recalled_at"])
                if row["last_recalled_at"] is not None
                else None
            ),
            recall_count=int(row["recall_count"] or 0),
            content_hash=str(row["content_hash"] or ""),
            etag=str(row["etag"] or ""),
            metadata=dict(_loads(row["metadata_json"], {})),
        )
