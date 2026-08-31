"""Per-run JSONL trace emitter.

Records one structured event per line to
``$MUCLI_HOME/trace/<session>_run_<hex12>.jsonl``. The record types are:

  * ``run_start``  — one header line per run (model, provider, mode, limits)
  * ``iter``       — one per agent-loop iteration, captured at the post-response
                     seam (context layers, real vs estimated tokens, drift,
                     subagent/memory snapshots, compaction summary)
  * ``tool``       — one per tool call (standalone; joined to iters by ``iter``)
  * ``nudge``      — one per corrective nudge injection (standalone)
  * ``compaction`` — one per compaction pass (standalone; drained from the
                     session manager's ``_compaction_log``)
  * ``request``    — privacy-safe component and per-message token manifest for
                     the exact provider request
  * ``turn_end``   — one per turn, with totals; flushes the file

The headline field is ``iter.context.drift_pct``: the signed percent difference
between the provider's *real* prompt token count (``response.input_tokens``) and
the harness's tiktoken ``cl100k_base`` estimate (the sum of context-layer
estimates). On a model whose tokenizer is not cl100k_base (e.g. glm), this drift
is systematic and is the primary signal for diagnosing long-horizon compaction
failures.

All public methods are no-ops once disabled/closed, and every write is wrapped
so an I/O error cannot propagate into the agent loop.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional


logger = logging.getLogger("mucli")

# Round-47 F8: flush cadence — trace records flush to disk at most every N
# records (or on turn_end / close). Bounded loss window on a crash.
_FLUSH_EVERY_RECORDS = 64

# Round-47 F6/F7: build_request_record caches. Per-part detail (bytes/tokens)
# for already-seen messages is stable — memoize by message identity and
# re-tokenize only NEW/CHANGED suffix messages each iteration. Tool-schema
# hash/bytes memoized by tool-set identity (names + canonical JSON hash).
_REQUEST_PART_CACHE: Dict[Any, Dict[str, Any]] = {}
_REQUEST_PART_CACHE_LOCK = threading.Lock()
_REQUEST_PART_CACHE_CAP = 4096
# Round-48 F10: per-part content digest memoized by object id — the
# identity itself must not re-hash MB-scale tool results every call.
_PART_ID_HASH: Dict[int, str] = {}
_PART_ID_HASH_CAP = 16384
_TOOL_HASH_CACHE: Dict[tuple, Dict[str, Any]] = {}
_TOOL_HASH_CACHE_LOCK = threading.Lock()
_TOOL_HASH_CACHE_CAP = 8


def trace_dir() -> str:
    """Return the trace directory under ``$MUCLI_HOME`` (lazy-created on write)."""
    from utils.config import HISTORY_DIR

    return os.path.join(os.path.expanduser(str(HISTORY_DIR)), "trace")


def new_run_id() -> str:
    """Generate a run id matching the codebase's ``turn_id``/``sa-`` conventions."""
    return "run_" + uuid.uuid4().hex[:12]


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "session").strip("_") or "session"
    return s[:64]


# ---------------------------------------------------------------------------
# Trace retention (round-31 F23). The trace dir grows without bound (every
# agent-loop iteration, tool call, and request manifest is a JSONL line, so
# heavy sessions accumulate hundreds of MB). Prune-on-create: when a new run
# emitter is built, a lock- and cooldown-gated sweep deletes the OLDEST
# closed trace files until the directory is within BOTH caps. Invariants:
#   * a trace file still OPEN by any emitter (this session's active file,
#     or another GUI session's long-running run) is NEVER deleted — POSIX
#     would keep the writes going to the unlinked inode and silently lose
#     the whole trace at close;
#   * the newest non-open file is always kept (unless it alone exceeds the
#     byte cap, in which case it is kept anyway as the newest survivor);
#   * zero-byte stubs count toward the file cap (unbounded empty files
#     would defeat it) and are pruned first;
#   * best-effort only — a prune failure must never break the loop.
_TRACE_MAX_FILES = 500
_TRACE_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB
_PRUNE_COOLDOWN_S = 300.0  # at most one directory scan per 5 minutes
_prune_lock = threading.Lock()
_last_prune_ts = 0.0
# Registry of trace files that have been opened for writing by a live
# emitter in THIS process (absolute paths). Populated in TraceEmitter._open,
# cleared in close(). Protects concurrent GUI sessions' open traces from
# each other's prunes.
_open_trace_paths: set = set()


def _enforce_trace_retention(active_path: str) -> None:
    """Prune old trace files to keep the trace dir within caps. Never raises."""
    global _last_prune_ts
    try:
        import time as _time

        now = _time.time()
        with _prune_lock:
            if now - _last_prune_ts < _PRUNE_COOLDOWN_S:
                return
            # Round-32 F3: mark the attempt BEFORE scanning so a repeatedly
            # failing sweep cannot turn every emitter creation into an
            # immediate directory rescan — the cooldown bounds scan attempts,
            # not just successful sweeps.
            _last_prune_ts = now
            directory = trace_dir()
            if not os.path.isdir(directory):
                return
            active_abspath = os.path.abspath(active_path)
            entries = []  # (mtime, size, path)
            # Round-32 F2: protected files (open by any emitter, or the
            # active path) occupy directory slots but are absent from
            # ``entries`` — count them so the survivor budget below reflects
            # the real remaining capacity.
            protected = 0
            # Round-33 F2: protected files' sizes must consume the byte
            # budget too — previously only their file-count was reserved,
            # so many large open traces could coexist with a full 512 MiB
            # of closed survivors (directory total ~2× the cap).
            protected_bytes = 0
            for name in os.listdir(directory):
                if not name.endswith(".jsonl"):
                    continue
                path = os.path.abspath(os.path.join(directory, name))
                if path in _open_trace_paths or path == active_abspath:
                    protected += 1
                    try:
                        protected_bytes += os.stat(path).st_size
                    except OSError:
                        pass  # vanished mid-scan: it is closed now, eligible next sweep
                    continue  # open by some emitter: never delete
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, path))
            # Zero-byte stubs first (they defeat the file cap if skipped).
            # They are DELETED here, so they no longer consume file-cap
            # budget below (round-32 F2: the old code subtracted len(zero)
            # again after deletion, shrinking the survivor budget by the
            # stub count for no reason).
            zero = [e for e in entries if e[1] == 0]
            nonzero = [e for e in entries if e[1] > 0]
            for _, _, path in zero:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            # ``protected`` includes the active file, so subtracting it
            # reserves the active slot AND every other emitter's open file
            # (old code's bare ``-1`` only covered the active path and
            # ignored other sessions' open traces).
            keep = max(_TRACE_MAX_FILES - protected, 0)
            # Newest-first; the newest survivor is kept even if it alone
            # exceeds the byte cap (never delete the newest data we have).
            newest = sorted(nonzero, reverse=True)[:keep]
            keep_set = {p for _, _, p in newest}
            keep_bytes = sum(sz for _, sz, _ in newest)
            # Byte cap: drain OLDEST-first (end of the newest-first list);
            # always retain at least the single newest CLOSED file (open
            # traces are untouchable by design — if protected files alone
            # exceed the cap, the overage is accepted until they close and
            # the next sweep drains; round-33 F2 documents this behavior).
            keep_bytes += protected_bytes
            survivors = list(newest)
            while len(survivors) > 1 and keep_bytes > _TRACE_MAX_TOTAL_BYTES:
                _, sz, path = survivors.pop()
                keep_set.discard(path)
                keep_bytes -= sz
            for _, _, path in nonzero:
                if path not in keep_set:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            _last_prune_ts = now
    except Exception as exc:  # noqa: BLE001
        logger.debug("trace retention prune failed: %s", exc)


# Round-51 T2: run-level run_end idempotency across emitter rebuilds.
_RUN_END_REGISTRY: set = set()
_RUN_END_REGISTRY_LOCK = threading.Lock()


class TraceEmitter:
    """Append-only JSONL writer for one run. Thread-safe; lazy-opens the file."""

    def __init__(self, session_name: str, run_id: str, path: str) -> None:
        self.session_name = session_name
        self.run_id = run_id
        self.path = path
        self._lock = threading.Lock()
        self._fh: Optional[Any] = None
        self._closed = False
        self.iter_count = 0
        # Round-47 F9: monotonic per-run sequence allocated under the lock in
        # emit(); consumers and the parser key ordering on it.
        self._seq = 0
        # Round-47 F8: buffered flush cadence.
        self._records_since_flush = 0

    # ----------------------------------------------------------- low-level

    def _open(self) -> None:
        if self._fh is None:
            # Round-33 F1: open AND register atomically under _prune_lock.
            # The round-32 shape opened first and only then took the lock to
            # register — a sweep could acquire the lock inside that window,
            # classify the freshly created file as closed, and unlink it
            # while writes continued to the unlinked inode. Holding the
            # prune lock across open+register closes the window: the sweep
            # either classifies before the file exists (not in listdir yet)
            # or after registration (protected). Lock order emitter._lock →
            # _prune_lock preserved (close() uses the same order).
            with _prune_lock:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8")
                _open_trace_paths.add(os.path.abspath(self.path))

    def emit(self, record: Dict[str, Any]) -> None:
        """Append one JSON line. Swallows all errors (telemetry must not break runs).

        Round-47 F8: writes are buffered — each emit appends to the file
        handle's buffer and flushes only on the turn_end cadence (or every
        _FLUSH_EVERY_RECORDS records), instead of an fsync-ish flush under
        the lock per record. A crash loses at most the unflushed buffer
        (bounded by the threshold); durability is preserved at turn
        boundaries where it matters.
        """
        try:
            # Round-47 F9 + Round-48 F12: seq + counter + write ordering
            # under the lock, but JSON serialization happens OUTSIDE it —
            # a MB-scale record previously held the lock across the whole
            # dumps, stalling concurrent emitter threads. The seq is spliced
            # into the pre-serialized line (string surgery on the closing
            # brace) instead of re-serializing the whole record under lock.
            line = json.dumps(record, default=str, ensure_ascii=False)
            with self._lock:
                if self._closed:
                    return
                self._seq += 1
                if record.get("type") == "iter":
                    # Round-47 F9: counter incremented under the lock — was
                    # outside, losing increments under concurrent emission.
                    self.iter_count += 1
                # Round-47 F9: seq rides INSIDE the record (flat JSONL schema
                # preserved — the parser and trace consumers see the same
                # shape; the copy-on-write avoids mutating the caller's dict).
                self._open()
                self._fh.write(f'{line[:-1]},"seq":{self._seq}}}' + "\n")
                self._records_since_flush += 1
                if self._records_since_flush >= _FLUSH_EVERY_RECORDS:
                    self._fh.flush()
                    self._records_since_flush = 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("trace emit failed: %s", exc)

    def flush(self) -> None:
        try:
            with self._lock:
                if self._fh is not None:
                    self._fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            with self._lock:
                if self._fh is not None:
                    # Round-17 F26: mark closed BEFORE closing — if the
                    # close's implicit flush raises, the old order left
                    # _closed False with a dead handle, so later emits
                    # would try to write through it (or double-close).
                    self._closed = True
                    try:
                        self._fh.close()
                    finally:
                        self._fh = None
                else:
                    self._closed = True
            # Round-31 F23: unregister so prune sweeps may reclaim the
            # (now closed) trace file if it falls outside the caps.
            # Round-32 F1: under the prune lock, same as registration —
            # a sweep running concurrently sees either "open" (registered)
            # or "closed + unregistered", never a half-state.
            with _prune_lock:
                _open_trace_paths.discard(os.path.abspath(self.path))
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------- typed events

    def run_start(self, meta: Dict[str, Any]) -> None:
        rec: Dict[str, Any] = {"type": "run_start", "run_id": self.run_id}
        rec.update(meta)
        self.emit(rec)

    def iter_record(self, rec: Dict[str, Any]) -> None:
        out = {"type": "iter", "run_id": self.run_id}
        out.update(rec)
        self.emit(out)

    def tool(self, rec: Dict[str, Any]) -> None:
        out = {"type": "tool", "run_id": self.run_id}
        out.update(rec)
        self.emit(out)

    def nudge(self, kind: str, iteration: int, **extra: Any) -> None:
        out = {"type": "nudge", "run_id": self.run_id, "kind": kind, "iteration": iteration}
        out.update(extra)
        self.emit(out)

    def compaction(self, rec: Dict[str, Any]) -> None:
        out = {"type": "compaction", "run_id": self.run_id}
        out.update(rec)
        self.emit(out)

    def context_artifact(self, rec: Dict[str, Any]) -> None:
        out = {"type": "context_artifact", "run_id": self.run_id}
        out.update(rec)
        self.emit(out)

    def request(self, rec: Dict[str, Any]) -> None:
        out = {"type": "request", "run_id": self.run_id}
        out.update(rec)
        self.emit(out)

    def turn_end(self, rec: Dict[str, Any]) -> None:
        out = {"type": "turn_end", "run_id": self.run_id}
        out.update(rec)
        self.emit(out)
        self.flush()

    def run_end(self, rec: Dict[str, Any]) -> None:
        """Terminal run bootstrap record (Round-51 T2).

        Written exactly once per emitter: emitted by the ``send_message``
        finally block on every exit path (completion, max-iterations,
        exception, stop/unblock). The ``status`` field records how the run
        terminated so the parser can retire the eternal 'running' default.
        Guards + flush + mark closed so late emitters cannot re-open or
        double-append.
        """
        if getattr(self, "_run_end_emitted", False):
            return
        # Round-51 T2: run-level idempotency survives emitter rebuilds —
        # a late caller that re-acquires a (fresh) emitter for the same
        # run id must not append a second terminal record into the run's
        # trace file.
        with _RUN_END_REGISTRY_LOCK:
            if self.run_id in _RUN_END_REGISTRY:
                self._run_end_emitted = True
                return
            _RUN_END_REGISTRY.add(self.run_id)
        self._run_end_emitted = True
        out = {"type": "run_end", "run_id": self.run_id}
        out.update(rec)
        self.emit(out)
        self.flush()
        self.close()


# Round-51 T2: run-level run_end idempotency across emitter rebuilds.
# ----------------------------------------------------------- accessor

def get_emitter(session: Any) -> Optional[TraceEmitter]:
    """Return the session's cached emitter, or build one. ``None`` when disabled.

    Caches on ``session._trace_emitter``. Generates ``session._trace_run_id``
    lazily. Never raises — a trace failure must not break the agent loop.

    Round-47 F10: get-or-create is serialized by a per-session init lock so
    two concurrent first-callers cannot both build an emitter for the same
    path (the loser previously overwrote the winner's cached emitter and
    both kept appending to the same file through separate handles).
    """
    try:
        variables = getattr(session, "variables", None) or {}
        if not bool(variables.get("trace_enabled", True)):
            return None
        em = getattr(session, "_trace_emitter", None)
        if em is not None and not em._closed:
            return em
        init_lock = getattr(session, "_trace_emitter_lock", None)
        if init_lock is None:
            init_lock = threading.Lock()
            session._trace_emitter_lock = init_lock
        with init_lock:
            # Double-check: the racing caller may have won the race while we
            # waited for the lock.
            em = getattr(session, "_trace_emitter", None)
            if em is not None and not em._closed:
                return em
            sm = getattr(session, "session_manager", None)
            run_id = getattr(session, "_trace_run_id", None) or new_run_id()
            session._trace_run_id = run_id
            name = _safe_name(getattr(sm, "current_session_name", "") or "session")
            path = os.path.join(trace_dir(), f"{name}_{run_id}.jsonl")
            em = TraceEmitter(name, run_id, path)
            session._trace_emitter = em
            _enforce_trace_retention(path)
            return em
    except Exception as exc:  # noqa: BLE001
        logger.debug("trace emitter init failed: %s", exc)
        return None


# ----------------------------------------------------------- convenience emit

def emit_nudge(session: Any, kind: str, iteration: int, **extra: Any) -> None:
    """One-line nudge emit for the loop's injection sites. Never raises."""
    try:
        em = get_emitter(session)
        if em is not None:
            em.nudge(kind, iteration, **extra)
    except Exception:  # noqa: BLE001
        pass


def emit_tool(
    session: Any,
    *,
    iteration: int,
    name: str,
    arg_fp: str = "",
    ok: Optional[bool] = None,
    error_code: Optional[str] = None,
    latency_ms: int = 0,
    cache_hit: bool = False,
    result_bytes: int = 0,
    path: str = "",
    preview: str = "",
    # Tool-output / context-management telemetry (spec #12). Populated from
    # the structured result's telemetry + cache_key by the loop-body call site.
    # All optional + defaulted so older callers and missing data degrade
    # gracefully (the parser treats absent fields as zero/None).
    store_key: Optional[str] = None,
    stored: bool = False,
    raw_tokens: int = 0,
    injected_tokens: int = 0,
    delivery_mode: str = "",
    omitted: bool = False,
    compression_ratio: Optional[float] = None,
) -> None:
    """One-line per-tool emit for the post-execution capture site. Never raises."""
    try:
        em = get_emitter(session)
        if em is not None:
            rec = {
                "iter": iteration,
                "name": name,
                "arg_fp": arg_fp,
                "ok": ok,
                "error_code": error_code,
                "latency_ms": int(latency_ms or 0),
                "cache_hit": bool(cache_hit),
                "result_bytes": int(result_bytes or 0),
                "path": path,
                "preview": (preview or "")[:200],
                "stored": bool(stored),
                "omitted": bool(omitted),
                "raw_tokens": int(raw_tokens or 0),
                "injected_tokens": int(injected_tokens or 0),
            }
            if store_key:
                rec["store_key"] = store_key
            if delivery_mode:
                rec["delivery_mode"] = delivery_mode
            if compression_ratio is not None:
                rec["compression_ratio"] = float(compression_ratio)
            em.tool(rec)
    except Exception:  # noqa: BLE001
        pass


def emit_context_artifact(session: Any, *, iteration: int, artifact_id: str,
                          state: str, tool_name: str = "", path: str = "",
                          bytes: int = 0, reason: str = "") -> None:
    """Record model-visible context lifecycle, including explicit discard."""
    try:
        em = get_emitter(session)
        if em is not None:
            em.context_artifact({"iter": iteration, "artifact_id": artifact_id,
                "state": state, "tool_name": tool_name, "path": path,
                "bytes": int(bytes or 0), "reason": reason})
    except Exception:
        pass


def _summarize_messages(message_parts, keep_recent=20):
    """Bounded per-message summary for iterations > 1 (Round-51 T6).

    Keeps per-message totals for the most recent ``keep_recent`` messages;
    older messages collapse into one aggregate row (role=older, count and
    byte/token totals). This bounds a request record to O(keep) independent
    of history length — the 517-iter run's 92KB-per-record request dumps
    become <2KB.
    """
    try:
        n = len(message_parts)
        if n <= keep_recent:
            return [
                {
                    "index": msg.get("index"),
                    "role": msg.get("role"),
                    "parts": msg.get("parts"),
                    "bytes": msg.get("bytes"),
                    "tokens": msg.get("tokens"),
                }
                for msg in message_parts
            ]
        newer = message_parts[-keep_recent:]
        older = message_parts[:-keep_recent]
        agg = {
            "index": 0,
            "role": "older",
            "parts": 0,
            "bytes": 0,
            "tokens": 0,
            "collapsed_count": 0,
        }
        for m in message_parts[:-keep_recent]:
            agg["bytes"] += int(m.get("bytes") or 0)
            agg["tokens"] += int(m.get("tokens") or 0)
            agg["collapsed_count"] += 1
        agg["parts"] = 0
        return [agg] + [
            {
                "index": msg.get("index"),
                "role": msg.get("role"),
                "parts": msg.get("parts"),
                "bytes": msg.get("bytes"),
                "tokens": msg.get("tokens"),
            }
            for msg in newer
        ]
    except Exception:  # noqa: BLE001
        return message_parts


def build_request_record(
    *,
    iteration: int,
    system_prompt: str,
    messages: Any,
    tools: Any,
    token_estimate: int,
    estimate_manifest: Optional[Dict[str, int]] = None,
    summarize: bool = False,
) -> Dict[str, Any]:
    """Privacy-preserving immutable manifest of the exact provider request.

    Raw prompts are intentionally not copied into telemetry; hashes, byte
    counts and message-part structure let traces prove which request was sent
    without leaking repository contents into another retention surface.

    Round-46 F2: ``estimate_manifest`` — the structured estimate produced by
    ``context_guard._estimate_request_tokens`` in preflight — supplies the
    shared tokenization. When omitted (legacy callers), the manifest is
    computed locally exactly as before. The per-part tokenization walk still
    runs here (it produces per-part byte/token detail the estimate does not),
    but the schema JSON is re-encoded only when no cached manifest was given.
    """
    from utils.token_estimator import estimate_tokens

    def _hash(value: Any) -> str:
        return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:16]

    def _get(value: Any, key: str, default: Any = None) -> Any:
        return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)

    def _serialized(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)

    component_tokens = {
        "system": estimate_tokens(system_prompt),
        "user": 0,
        "assistant": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "files_images": 0,
        "other": 0,
        "tool_schemas": 0,
    }
    message_parts = []
    # Round-47 F6: per-message part detail is memoized by content identity.
    # The dominant per-iteration cost was re-TOKENIZING (tiktoken) every old
    # message's parts on every provider call — quadratic over a turn.
    # Round-48 F10: the r47 identity itself rebuilt role + FULL text chunks
    # and hashed MB-scale bytes for every OLD message on every call — the
    # quadratic walk survived as the cache key. Identity is now (role,
    # id(part) per part) with a content hash computed ONLY for parts we
    # haven't seen before; hash results are memoized by object id. Message
    # objects are immutable once appended to history, so id() is stable.
    _part_digest: Dict[int, str] = {}

    def _part_identity(part: Any) -> str:
        pid = id(part)
        with _REQUEST_PART_CACHE_LOCK:
            known = _PART_ID_HASH.get(pid)
        if known is not None:
            return known
        if _get(part, "text") is not None:
            body = "t:" + str(_get(part, "text"))
        elif _get(part, "tool_result") is not None:
            body = "r:" + str(_get(part, "tool_result"))
        elif _get(part, "tool_args") is not None:
            body = "a:" + json.dumps(_get(part, "tool_args"), sort_keys=True, default=str)
        elif _get(part, "inline_data") is not None:
            body = "i:" + _serialized(_get(part, "inline_data"))
        else:
            body = "e:"
        h = hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()
        with _REQUEST_PART_CACHE_LOCK:
            if len(_PART_ID_HASH) >= _PART_ID_HASH_CAP and pid not in _PART_ID_HASH:
                _PART_ID_HASH.pop(next(iter(_PART_ID_HASH)), None)
            _PART_ID_HASH[pid] = h
        return h

    def _message_identity(msg: Any) -> tuple:
        return (
            str(_get(msg, "role", "") or ""),
            tuple(_part_identity(p) for p in (_get(msg, "parts", []) or [])),
        )

    def _part_details_for(msg: Any) -> Dict[str, Any]:
        """Cached {part_details, bytes, tokens, buckets} for one message."""
        key = _message_identity(msg)
        with _REQUEST_PART_CACHE_LOCK:
            cached = _REQUEST_PART_CACHE.get(key)
            if cached is not None:
                return cached
        role = str(_get(msg, "role", "") or "")
        part_records = []
        buckets = {k: 0 for k in ("user", "assistant", "tool_calls", "tool_results", "files_images", "other")}
        total_bytes = 0
        total_tokens = 0
        for part_index, part in enumerate(_get(msg, "parts", []) or []):
            part_type = str(_get(part, "type", "") or "other")
            if _get(part, "text") is not None:
                serialized = str(_get(part, "text"))
            elif _get(part, "tool_result") is not None:
                # Match loop_body._estimate_messages_tokens exactly so the
                # component stack adds up to the compaction estimate.
                serialized = str(_get(part, "tool_result"))
            elif _get(part, "tool_args") is not None:
                serialized = json.dumps(_get(part, "tool_args"))
            elif _get(part, "inline_data") is not None:
                serialized = _serialized(_get(part, "inline_data"))
            else:
                serialized = ""
            byte_count = len(serialized.encode("utf-8", errors="replace"))
            token_count = estimate_tokens(serialized)
            if part_type == "tool_result":
                bucket = "tool_results"
            elif part_type == "tool_call":
                bucket = "tool_calls"
            elif part_type in {"file", "image_inline", "image_input"}:
                bucket = "files_images"
            elif part_type == "text" and role == "user":
                bucket = "user"
            elif part_type == "text" and role == "assistant":
                bucket = "assistant"
            else:
                bucket = "other"
            buckets[bucket] += token_count
            total_bytes += byte_count
            total_tokens += token_count
            part_records.append({
                "index": part_index,
                "type": part_type,
                "tool_name": str(_get(part, "tool_name", "") or ""),
                "bytes": byte_count,
                "tokens": token_count,
            })
        entry = {
            "role": role,
            "part_details": part_records,
            "bytes": total_bytes,
            "tokens": total_tokens,
            "buckets": buckets,
        }
        with _REQUEST_PART_CACHE_LOCK:
            if len(_REQUEST_PART_CACHE) >= _REQUEST_PART_CACHE_CAP and key not in _REQUEST_PART_CACHE:
                _REQUEST_PART_CACHE.pop(next(iter(_REQUEST_PART_CACHE)), None)
            _REQUEST_PART_CACHE[key] = entry
        return entry

    for message_index, msg in enumerate(messages or []):
        entry = _part_details_for(msg)
        for bucket, count in entry["buckets"].items():
            component_tokens[bucket] += count
        message_parts.append({
            "index": message_index,
            "role": entry["role"],
            # Keep the original numeric field for trace-schema compatibility;
            # the new detail lives alongside it.
            "parts": len(entry["part_details"]),
            "part_details": entry["part_details"],
            "bytes": entry["bytes"],
            "tokens": entry["tokens"],
        })

    tool_payload = [{
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "parameters": getattr(tool, "parameters", {}) or {},
    } for tool in (tools or [])]
    tool_names = [tool["name"] for tool in tool_payload]
    # Round-47 F7 + Round-48 F11: schema hash + bytes memoized by tool-set
    # identity. The r48 review caught the identity itself re-serializing the
    # full schema (json.dumps of the whole payload) on EVERY call — the
    # exact cost the cache claims to avoid. Identity is now (names tuple,
    # tuple of tool object ids); tool objects are stable for the process
    # lifetime, and the payload is serialized only on a cache miss.
    identity = (
        tuple(tool_names),
        tuple(id(t) for t in (tools or [])),
    )
    with _TOOL_HASH_CACHE_LOCK:
        meta = _TOOL_HASH_CACHE.get(identity)
    if meta is None:
        tool_json = json.dumps(tool_payload, sort_keys=True, default=str, ensure_ascii=False)
        meta = {
            "hash": _hash(tool_payload),
            "bytes": len(tool_json.encode("utf-8")) if tool_payload else 0,
        }
        with _TOOL_HASH_CACHE_LOCK:
            if len(_TOOL_HASH_CACHE) >= _TOOL_HASH_CACHE_CAP and identity not in _TOOL_HASH_CACHE:
                _TOOL_HASH_CACHE.pop(next(iter(_TOOL_HASH_CACHE)), None)
            _TOOL_HASH_CACHE[identity] = meta
    if estimate_manifest is not None:
        component_tokens["tool_schemas"] = int(
            estimate_manifest.get("tools", 0) or 0
        )
    else:
        from utils.token_estimator import estimate_tokens as _et
        component_tokens["tool_schemas"] = _et(json.dumps(
            tool_payload, sort_keys=True, default=str, ensure_ascii=False
        )) if tool_payload else 0
    return {
        "iter": iteration,
        "system_prompt_bytes": len(system_prompt.encode("utf-8", errors="replace")),
        "system_prompt_hash": _hash(system_prompt),
        "messages": (
            # Round-51 T6: bounded summary — per-message totals only, no
            # part_details, and older messages beyond the most recent 50
            # collapse to a single total row. Keeps size/token drift
            # diagnostics derivable; full dumps are it1-only.
            (
                _summarize_messages(message_parts)
            )
            if summarize
            else message_parts
        ),
        "summarized": summarize,
        "messages_hash": _hash([(msg["role"], msg["bytes"]) for msg in message_parts]),
        "tool_names": tool_names,
        "tool_schema_bytes": meta["bytes"],
        "tools_hash": meta["hash"],
        "component_tokens": component_tokens,
        "component_total_tokens": sum(component_tokens.values()),
        "token_estimate": int(token_estimate),
    }


# ----------------------------------------------------------- record builders

def _layer_tokens(session: Any) -> Dict[str, Any]:
    """Sum context-layer estimates via the harness's own estimator (cl100k_base).

    Returns ``{l0,l1,l1c,l1b,l2,l3,l4b,l5,total_est}``. Each value is the layer's
    estimated token ``current``; ``total_est`` is their sum — the harness's
    estimate of the assembled prompt, directly comparable to
    ``response.input_tokens``.
    """
    try:
        from utils.runtime_metrics import collect_context_layers

        layers = collect_context_layers(session) or []
    except Exception:  # noqa: BLE001
        layers = []
    out: Dict[str, int] = {}
    total = 0
    for layer in layers:
        key = (layer.get("layer") or "").lower()  # "l0","l1","l1c","l1b","l2","l3","l4b","l5"
        try:
            val = int(layer.get("current") or 0)
        except Exception:  # noqa: BLE001
            val = 0
        out[key] = val
        total += val
    return {
        "l0": out.get("l0", 0),
        "l1c": out.get("l1c", 0),
        "l1a": out.get("l1a", 0),
        "l1b": out.get("l1b", 0),
        "l2": out.get("l2", 0),
        "l3": out.get("l3", 0),
        "l4b": out.get("l4b", 0),
        "l5": out.get("l5", 0),
        "total_est": total,
    }


def _subagent_snapshot(session: Any) -> Dict[str, Any]:
    try:
        reg = getattr(session, "_subagent_registry", None)
        if reg is None:
            return {"active": 0, "stuck": 0, "stall": 0, "children": []}
        snap = reg.snapshot_all()
        children = []
        for s in snap:
            children.append(
                {
                    "task_id": s.get("task_id"),
                    "depth": s.get("depth"),
                    "status": s.get("status"),
                    "stuck": bool(s.get("stuck")),
                    "stall": bool(s.get("stall")),
                    "tool_calls": s.get("tool_calls"),
                    "elapsed": s.get("elapsed"),
                }
            )
        return {
            "active": sum(1 for s in snap if s.get("status") == "running"),
            "stuck": sum(1 for s in snap if s.get("stuck")),
            "stall": sum(1 for s in snap if s.get("stall")),
            "children": children,
        }
    except Exception:  # noqa: BLE001
        return {"active": 0, "stuck": 0, "stall": 0, "children": []}


def _memory_counts(session: Any) -> Dict[str, Any]:
    try:
        sm = session.session_manager
        tm = getattr(sm, "task_memory", None)
        entries = list(getattr(tm, "entries", []) or [])
        by_status: Dict[str, int] = {}
        for e in entries:
            st = getattr(e, "status", None) or "unknown"
            by_status[st] = by_status.get(st, 0) + 1
        sp = getattr(sm, "turn_scratchpad", None)
        scratch = len(getattr(sp, "entries", []) or [])
        return {
            "task_memory_count": len(entries),
            "by_status": by_status,
            "scratchpad_count": scratch,
        }
    except Exception:  # noqa: BLE001
        return {"task_memory_count": 0, "by_status": {}, "scratchpad_count": 0}


def drain_compactions(session: Any) -> List[Dict[str, Any]]:
    """Pull pending compaction entries off the session manager and clear the log.

    Called by the loop at the post-response seam so each compaction pass that
    fired before this iteration's provider response is emitted as a standalone
    trace line.
    """
    try:
        sm = session.session_manager
        log = getattr(sm, "_compaction_log", None)
        if not log:
            return []
        out = list(log)
        log.clear()
        return out
    except Exception:  # noqa: BLE001
        return []


def build_iter_record(
    session: Any,
    *,
    iteration: int,
    max_iter: int,
    response: Any,
    total_in: int,
    total_out: int,
    total_cost: float,
    has_text: bool,
    has_tool_call: bool,
    iter_start: float,
    cost_delta: float = 0.0,
    compaction: Optional[Dict[str, Any]] = None,
    status: str = "running",
    request_token_estimate: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the per-iteration trace record from in-scope loop state.

    Called at the post-response seam (after token accounting, before the
    ``if not has_tool_call`` branch). ``compaction`` is the last drained
    compaction entry for this iteration (standalone lines for every entry are
    emitted by the caller). All gathering is defensive.
    """
    import time as _time

    wall_ms = int((_time.monotonic() - iter_start) * 1000)
    layers = _layer_tokens(session)
    # This record is emitted after the provider returns, by which point the
    # response has already been appended to history.  Measuring L5 from that
    # mutable state counts assistant output that was *not* in this request and
    # made the advertised actual-vs-predicted comparison invalid.  The loop
    # therefore snapshots the exact pre-request estimate and supplies it here.
    # Keep the layer values for the UI breakdown, but make the headline total
    # (and drift) use the request snapshot whenever it is available.
    total_est = (
        max(0, int(request_token_estimate))
        if request_token_estimate is not None
        else layers["total_est"]
    )
    actual = int(getattr(response, "input_tokens", 0) or 0)
    # drift_pct compares the provider's reported prompt size (actual) to the
    # cl100k estimate (total_est). It is only meaningful when `actual` is a
    # reliable FULL-prompt signal. For Ollama, `actual` is the streamed
    # prompt_eval_count — the non-cached prompt DELTA, near-zero in a warm
    # loop and far smaller than total_est. Normalising by that near-zero
    # value ((actual−est)/actual) blew up to ±thousands of percent even
    # though the prompt was fine. Gate: actual is a real full-prompt count
    # when it is NOT a tiny fraction of the estimate (actual*4 >= total_est,
    # i.e. actual is at least ~25% of the estimate). When the estimate is 0
    # (missing/zero layer sum) any nonzero actual is reliable. When gated
    # out (Ollama warm cache), zero drift_pct and flag unreliable so the UI
    # doesn't paint "0% drift = perfect estimate"; the learned cl100k→real
    # `drift_ratio` is the representative diagnostic instead.
    drift_pct_reliable = bool(
        actual > 0 and (total_est == 0 or actual * 4 >= total_est)
    )
    if drift_pct_reliable:
        drift_pct = round((actual - total_est) / max(1, actual) * 100, 2)
    else:
        drift_pct = 0.0
    # Drift-corrected real-prompt estimate + the drift ratio the compactor is
    # assuming. `actual` (Ollama prompt_eval_count) is the non-cached delta —
    # near-zero in a warm loop and a misleading "real" prompt size. The
    # drift-corrected cl100k estimate is the representative real fill; the
    # ratio makes the cl100k undercount auditable in the trace.
    try:
        from mu.session.budgets import effective_drift_ratio
        eff_drift = float(effective_drift_ratio(session))
    except Exception:  # noqa: BLE001
        eff_drift = 1.0
    real_est = int(total_est * eff_drift)

    tokens = {
        "in": int(getattr(response, "input_tokens", 0) or 0),
        "out": int(getattr(response, "output_tokens", 0) or 0),
        "cached": int(getattr(response, "cached_tokens", 0) or 0),
        "reasoning": int(getattr(response, "reasoning_tokens", 0) or 0),
        "cost_delta": round(float(cost_delta or 0.0), 6),
    }

    # Assistant text preview (first text part, truncated).
    preview = ""
    try:
        for p in getattr(response, "parts", []) or []:
            if getattr(p, "type", None) == "text" and getattr(p, "text", ""):
                preview = (p.text or "").strip()[:240]
                break
    except Exception:  # noqa: BLE001
        preview = ""

    compactions: List[Dict[str, Any]] = []
    # Embed a compact summary of the latest compaction this iteration; full
    # entries are emitted as standalone lines by the caller via drain_compactions.
    last_compaction = compaction

    return {
        "iter": iteration,
        "max_iter": max_iter,
        "wall_ms": wall_ms,
        "context": {
            "l0": layers["l0"],
            "l1a": layers.get("l1a", 0),
            "l1c": layers["l1c"],
            "l1b": layers["l1b"],
            "l2": layers["l2"],
            "l3": layers["l3"],
            "l4b": layers["l4b"],
            "l5": layers["l5"],
            "total_est": total_est,
            "estimate_source": (
                "pre_request" if request_token_estimate is not None else "post_response_layers"
            ),
            "prompt_tokens_actual": actual,
            "prompt_tokens_real_est": real_est,
            "drift_ratio": round(eff_drift, 3),
            "drift_pct": drift_pct,
            "drift_pct_reliable": drift_pct_reliable,
        },
        "tokens": tokens,
        "has_text": bool(has_text),
        "has_tool_call": bool(has_tool_call),
        "assistant_preview": preview,
        "subagents": _subagent_snapshot(session),
        "memory": _memory_counts(session),
        "compaction": last_compaction,
        "status": status,
    }
