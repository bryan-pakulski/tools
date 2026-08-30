"""Compact evidence receipt for understanding a durable job at a glance."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict

from utils.config import HISTORY_DIR

from .service import JobService
from .verification import VerificationStore


RECEIPT_SCHEMA_VERSION = 3


class JobReceiptBuilder:
    def __init__(self, service: JobService, *, root: str | None = None):
        self.service = service
        self.root = os.path.abspath(
            os.path.expanduser(root or os.path.join(HISTORY_DIR, "jobs", "evidence"))
        )
        os.makedirs(self.root, exist_ok=True)
        self.verifications = VerificationStore(service.store, evidence_root=self.root)

    @staticmethod
    def _token_totals(attempts) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for attempt in attempts:
            result = attempt.metadata.get("agent_result") if isinstance(attempt.metadata, dict) else None
            tokens = result.get("tokens") if isinstance(result, dict) else None
            if not isinstance(tokens, dict):
                continue
            for key, value in tokens.items():
                if isinstance(value, (int, float)):
                    totals[str(key)] = totals.get(str(key), 0.0) + float(value)
        return totals

    @staticmethod
    def _cost_summary(attempts, fallback_cost: float) -> Dict[str, Any]:
        records: list[Dict[str, Any]] = []
        for attempt in attempts:
            result = attempt.metadata.get("agent_result") if isinstance(attempt.metadata, dict) else None
            cost = result.get("cost") if isinstance(result, dict) else None
            if isinstance(cost, dict):
                records.append(dict(cost))

        unpriced = [record for record in records if record.get("api_cost_usd") is None]
        local_zero = [record for record in records if record.get("billing") == "local"]
        billing_modes = sorted({str(record.get("billing") or "unknown") for record in records})
        pricing_versions = sorted({str(record.get("pricing_version") or "") for record in records if record.get("pricing_version")})
        if records:
            priced_api_cost = sum(
                float(record.get("api_cost_usd") or 0.0)
                for record in records
                if record.get("api_cost_usd") is not None
            )
        else:
            # Pre-pricing-ledger historical jobs retain their old accumulated
            # value but are explicitly marked legacy rather than pretending we
            # know the rates/cached-token treatment that produced it.
            priced_api_cost = float(fallback_cost or 0.0)

        component_totals: Dict[str, float] = {}
        for record in records:
            components = record.get("cost_components")
            if not isinstance(components, dict):
                continue
            for key, value in components.items():
                if isinstance(value, (int, float)):
                    component_totals[str(key)] = component_totals.get(str(key), 0.0) + float(value)

        if not records:
            status = "legacy"
        elif unpriced and len(unpriced) == len(records):
            status = "unpriced"
        elif unpriced:
            status = "partial"
        elif local_zero and len(local_zero) == len(records) and priced_api_cost == 0:
            status = "local_zero"
        else:
            status = "metered"

        return {
            "api_cost_usd": priced_api_cost,
            "cost_complete": not bool(unpriced) if records else False,
            "status": status,
            "billing_modes": billing_modes,
            "unpriced_attempts": len(unpriced),
            "local_zero_attempts": len(local_zero),
            "pricing_versions": pricing_versions,
            "cost_components": component_totals,
            "records": records,
            "note": "Workspace CPU/GPU/storage/network economics are separate from model/API spend.",
        }

    def build(self, job_id: str) -> Dict[str, Any]:
        job = self.service.get(job_id)
        attempts = self.service.attempts(job_id)
        events = self.service.events(job_id)
        verification = self.verifications.latest(job_id)
        elapsed = 0.0
        if job.started_at is not None:
            end = job.completed_at or job.updated_at or time.time()
            elapsed = max(0.0, float(end) - float(job.started_at))

        event_counts: Dict[str, int] = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        metadata = dict(job.metadata or {})
        review_branch = str(metadata.get("review_branch") or "").strip()
        if not review_branch and job.status.value == "ready_for_review":
            review_branch = str(job.branch or "").strip()
        review_head_sha = str(metadata.get("review_head_sha") or "").strip()
        if review_branch and not review_head_sha and verification:
            review_head_sha = str(verification.head_sha or "").strip()
        retired_worktree = str(metadata.get("retired_worktree") or "").strip()
        review_artifact = "branch" if review_branch else ("worktree" if job.worktree else "none")

        cost_summary = self._cost_summary(attempts, float(job.cost_usd or 0.0))
        receipt: Dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "generated_at": time.time(),
            "job": {
                "id": job.id,
                "title": job.title,
                "description": job.description,
                "status": job.status.value,
                "needs_attention": job.needs_attention,
                "attention_reason": job.attention_reason.value,
                "attention_detail": job.attention_detail,
            },
            "outcome": {
                "ready_for_review": job.status.value == "ready_for_review",
                "review_artifact": review_artifact,
                "attempts": len(attempts),
                "elapsed_seconds": elapsed,
                "cost_usd": float(job.cost_usd or 0.0),
                "cost_status": cost_summary["status"],
                "cost_complete": cost_summary["cost_complete"],
                "terminal": job.terminal,
            },
            "ticket": {
                "acceptance_criteria": list(job.acceptance_criteria),
                "validation_commands": list(job.validation_commands),
            },
            "git": {
                "repository": job.repository,
                "repository_id": metadata.get("repository_id"),
                "base_branch": job.base_branch,
                "base_sha": job.base_sha,
                # `branch` / `worktree` remain for receipt compatibility.  The
                # review_* fields below define the authoritative completed
                # artifact: execution happens in a worktree, review happens on
                # a normal branch after that worktree is retired.
                "branch": job.branch,
                "worktree": job.worktree,
                "review_artifact": review_artifact,
                "review_branch": review_branch,
                "review_head_sha": review_head_sha,
                "execution_worktree": job.worktree,
                "retired_worktree": retired_worktree,
                "head_sha": review_head_sha or (verification.head_sha if verification else ""),
                "changed_files": verification.changed_files if verification else [],
                "additions": verification.additions if verification else 0,
                "deletions": verification.deletions if verification else 0,
                "diff_stat": verification.diff_stat if verification else "",
                "dirty": verification.dirty if verification else None,
            },
            "verification": verification.to_dict() if verification else None,
            "attempts": [attempt.to_dict() for attempt in attempts],
            "usage": {
                "cost_usd": float(job.cost_usd or 0.0),
                "model_api": cost_summary,
                "tokens": self._token_totals(attempts),
            },
            "activity": {
                "events": len(events),
                "agent_messages": event_counts.get("agent_message", 0),
                "tool_calls": event_counts.get("tool_call_ui", 0),
                "human_responses": event_counts.get("human_response", 0),
                "checkpoints": event_counts.get("checkpoint_created", 0),
                "verification_runs": event_counts.get("verification_evidence_created", 0),
            },
        }
        return receipt

    def write(self, job_id: str) -> str:
        job_dir = os.path.join(self.root, job_id)
        os.makedirs(job_dir, exist_ok=True)
        path = os.path.join(job_dir, "work-receipt.json")
        receipt = self.build(job_id)
        # Round-35 F4: atomic write — serialize to a temp file in the same
        # directory, fsync, then os.replace. A crash mid-dump or a
        # concurrent reader previously saw a truncated receipt (and a
        # failed rewrite destroyed the last valid one).
        tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(receipt, fh, ensure_ascii=False, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self.service.store.append_event(
            job_id,
            "work_receipt_updated",
            payload={"path": path, "schema_version": RECEIPT_SCHEMA_VERSION},
        )
        return path
