"""Recall receipts, event queries, revision history, graph, and stats.

Mixin extracted from mu/memory/ledger.py; SQLiteMemoryLedger composes it.
Methods rely on the host class's _read()/_write() context managers and the
module-level _json()/_loads() helpers (re-imported from ledger).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

from .models import RecallReceipt, uuid7

# Host module owns the schema version; imported lazily to avoid a cycle.
def _schema_version() -> int:
    from .ledger import SCHEMA_VERSION

    return SCHEMA_VERSION


def _json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _loads(value: Any, fallback: Any) -> Any:
    import json

    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


class LedgerRecallMixin:
    """Recall receipts + read-only inspection endpoints."""

    def record_recall(self, receipt: RecallReceipt, *, actor: str = "system") -> None:
        # Round-49 F1: receipts previously stored FULL MemoryItem dicts for
        # included AND excluded candidates — on the latency-critical
        # per-turn path this serialized megabytes of statement/metadata/
        # relations JSON into an unboundedly-growing table. Receipts now
        # carry compact projections (ids, versions, scores, reasons, token
        # costs) sufficient for audit/replay; the durable memory content
        # lives in the memories table itself.
        def _compact(candidates):
            return [
                {
                    "id": c.item.id,
                    "version": c.item.version,
                    "score": round(c.score, 4),
                    "token_cost": c.token_cost,
                    "reason": getattr(c, "reason", None),
                }
                for c in candidates
            ]

        included_payload = _compact(receipt.included)
        excluded_payload = _compact(receipt.excluded)
        with self._write() as connection:
            connection.execute(
                "INSERT INTO memory_recall_receipts(id, session_name, query_text, scopes_json, "
                "budget_tokens, token_count, included_json, excluded_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.id,
                    receipt.session_name,
                    receipt.query,
                    _json(receipt.scopes),
                    receipt.budget_tokens,
                    receipt.token_count,
                    _json(included_payload),
                    _json(excluded_payload),
                    receipt.created_at,
                ),
            )
            # Round-49 F2: the per-candidate UPDATE + event INSERT loop held
            # the single-writer lock for N round-trips per recall — bulk
            # via executemany; events assembled inline to keep one tx.
            if receipt.included:
                connection.executemany(
                    "UPDATE memories SET recall_count=recall_count+1, last_recalled_at=? WHERE id=?",
                    [(receipt.created_at, c.item.id) for c in receipt.included],
                )
                event_rows = [
                    (
                        uuid7(),
                        c.item.id,
                        c.item.version,
                        "recalled",
                        actor,
                        "",
                        _json({
                            "receipt_id": receipt.id,
                            "session_name": receipt.session_name,
                            "score": round(c.score, 4),
                            "token_cost": c.token_cost,
                        }),
                        "",
                        receipt.created_at,
                    )
                    for c in receipt.included
                ]
                connection.executemany(
                    "INSERT INTO memory_events(event_id, memory_id, version, event_type, actor, "
                    "before_hash, after_json, reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    event_rows,
                )

    def get_recall(
        self, receipt_id: str = "", *, session_name: str = ""
    ) -> Dict[str, Any] | None:
        with self._read() as connection:
            if receipt_id:
                where_session = " AND session_name=?" if session_name else ""
                params: List[Any] = [receipt_id]
                if session_name:
                    params.append(session_name)
                row = connection.execute(
                    "SELECT * FROM memory_recall_receipts WHERE id=?" + where_session,
                    params,
                ).fetchone()
                # TUI receipts deliberately display compact IDs. Accept a
                # unique prefix there while retaining canonical IDs in data.
                if row is None and len(str(receipt_id)) >= 6:
                    prefix_params: List[Any] = [f"{receipt_id}%"]
                    if session_name:
                        prefix_params.append(session_name)
                    matches = connection.execute(
                        "SELECT * FROM memory_recall_receipts WHERE id LIKE ?"
                        + where_session
                        + " ORDER BY created_at DESC LIMIT 2",
                        prefix_params,
                    ).fetchall()
                    row = matches[0] if len(matches) == 1 else None
            elif session_name:
                row = connection.execute(
                    "SELECT * FROM memory_recall_receipts WHERE session_name=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (session_name,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM memory_recall_receipts ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "session_name": row["session_name"],
                "query": row["query_text"],
                "scopes": _loads(row["scopes_json"], []),
                "budget_tokens": row["budget_tokens"],
                "token_count": row["token_count"],
                "included": _loads(row["included_json"], []),
                "excluded": _loads(row["excluded_json"], []),
                "created_at": row["created_at"],
            }

    def events(
        self,
        *,
        memory_id: str = "",
        scopes: Sequence[tuple[str, str]] | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT e.* FROM memory_events e"
        params: List[Any] = []
        clauses: List[str] = []
        if scopes:
            sql += " JOIN memories m ON m.id=e.memory_id"
            clauses.append(
                "("
                + " OR ".join("(m.scope_type=? AND m.scope_key=?)" for _ in scopes)
                + ")"
            )
            for scope_type, scope_key in scopes:
                params.extend([scope_type, scope_key])
        if memory_id:
            clauses.append("e.memory_id=?")
            params.append(memory_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY e.created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        with self._read() as connection:
            return [
                {
                    "event_id": row["event_id"],
                    "memory_id": row["memory_id"],
                    "version": row["version"],
                    "type": row["event_type"],
                    "actor": row["actor"],
                    "device_id": row["device_id"],
                    "before_hash": row["before_hash"],
                    "after": _loads(row["after_json"], {}),
                    "reason": row["reason"],
                    "created_at": row["created_at"],
                }
                for row in connection.execute(sql, params).fetchall()
            ]

    def revisions(self, memory_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT memory_id, version, statement, snapshot_json, actor, reason, created_at "
                "FROM memory_revisions WHERE memory_id=? ORDER BY version DESC LIMIT ?",
                (memory_id, max(1, min(int(limit), 1000))),
            ).fetchall()
            return [
                {
                    "memory_id": row["memory_id"],
                    "version": row["version"],
                    "statement": row["statement"],
                    "snapshot": _loads(row["snapshot_json"], {}),
                    "actor": row["actor"],
                    "reason": row["reason"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def graph(self, memory_id: str) -> Dict[str, Any]:
        item = self.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        target_ids = {
            str(relation.get("target_id") or "")
            for relation in item.relations
            if isinstance(relation, dict) and relation.get("target_id")
        }
        nodes = [item.to_dict()]
        for target_id in sorted(target_ids):
            target = self.get(target_id)
            if target is not None:
                nodes.append(target.to_dict())
        edges = [
            {
                "source": item.id,
                "target": relation.get("target_id"),
                "type": relation.get("type", "related_to"),
            }
            for relation in item.relations
            if isinstance(relation, dict) and relation.get("target_id")
        ]
        return {"center": item.id, "nodes": nodes, "edges": edges}

    def stats(
        self, *, scopes: Sequence[tuple[str, str]] | None = None
    ) -> Dict[str, Any]:
        clauses = ["lifecycle!='forgotten'"]
        params: List[Any] = []
        if scopes:
            clauses.append(
                "("
                + " OR ".join("(scope_type=? AND scope_key=?)" for _ in scopes)
                + ")"
            )
            for scope_type, scope_key in scopes:
                params.extend([scope_type, scope_key])
        where = " AND ".join(clauses)
        with self._read() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM memories WHERE {where}", params
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT lifecycle, COUNT(*) AS n FROM memories WHERE {where} GROUP BY lifecycle",
                params,
            ).fetchall()
            pinned = connection.execute(
                f"SELECT COUNT(*) FROM memories WHERE {where} AND pinned=1", params
            ).fetchone()[0]
        return {
            "total": int(total),
            "pinned": int(pinned),
            "by_lifecycle": {str(row["lifecycle"]): int(row["n"]) for row in rows},
            "database": self.path,
            "schema_version": _schema_version(),
            "fts_enabled": self._fts_enabled,
        }
