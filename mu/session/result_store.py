"""Durable on-disk store for full raw tool results (spec #1, #11).

The model context carries only a compact observation + a ``stored_ref``
cache key; the authoritative full raw result lives here, on disk, so it
survives LRU eviction and session restarts and is retrievable on demand
via bounded operations (spec #6). The trace JSONL stays telemetry-only
(200-char preview); this store is the raw-output record.

Layout::

    $MUCLI_HOME/results/<run_id>/
        <key>.json        # one file per stored result (atomic write)
        index.jsonl       # append-only index: {key, tool, args_digest, ...}

Everything here is defensive — a store failure must NEVER break the agent
loop (mirrors the trace-emitter contract). All public methods swallow
exceptions and return a benign fallback (None / empty) so callers can treat
the store as best-effort.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Diagnostic line patterns — the "extract diagnostics" op returns the unique
# subset of lines matching these, deduped. Order matters only for the regex
# alternation; dedup preserves first-seen order.
_DIAGNOSTIC_RE = re.compile(
    r"(error|warning|warn|fail|traceback|exception|denied|not found|"
    r"cannot|undefined|unresolved|fatal|panic)",
    re.IGNORECASE,
)

# Lines that are pure progress noise — dropped by the "logs" reduction and
# by diagnostics so a build's spinning percentage lines don't drown the
# real signals. Anchored to common forms; intentionally conservative.
_NOISE_RE = re.compile(
    r"^\s*(\[?\d{1,3}%\]|\r|building\.\.\.|compiling\.\.\.|downloading\.\.\.|"
    r"running\.\.\.|fetching\.\.\.|loading\.\.\.)",
    re.IGNORECASE,
)


def _results_root() -> str:
    """``$MUCLI_HOME/results`` — sibling of the trace dir."""
    from utils.config import HISTORY_DIR

    return os.path.join(os.path.expanduser(str(HISTORY_DIR)), "results")


def _render_text(result: Any) -> str:
    """Render a stored result payload to text for line-oriented ops.

    Strings pass through; structured payloads are pretty-JSON-dumped so line
    ranges / search / diagnostics operate on a stable textual form.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, indent=2, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(result)


class ResultStore:
    """Durable, best-effort store of full raw tool results for one run.

    Writes are atomic (temp file + rename). Reads are on-demand from disk;
    the in-memory ``ToolResultCache`` is the hot layer that writes through
    here. Bounds: a per-run byte cap prunes the oldest stored keys; a
    cross-run GC drops run directories older than ``gc_age_days``.
    """

    def __init__(
        self,
        run_id: str,
        *,
        root: Optional[str] = None,
        max_bytes: int = 16 * 1024 * 1024,
        gc_age_days: int = 7,
        enabled: bool = True,
    ) -> None:
        self.run_id = run_id
        self.root = root or _results_root()
        self.run_dir = os.path.join(self.root, run_id)
        self.max_bytes = max(1024, int(max_bytes))
        self.gc_age_days = max(0, int(gc_age_days))
        self.enabled = bool(enabled)
        self._index_path = os.path.join(self.run_dir, "index.jsonl")
        # key -> index entry (dict); lazily loaded.
        self._index: Optional[Dict[str, dict]] = None
        self._current_bytes = 0
        # Counters for efficiency metrics (#12).
        self.puts = 0
        self.disk_hits = 0
        self.evictions = 0

    # --------------------------------------------------------------- lifecycle

    def _ensure_dir(self) -> None:
        if not self.enabled:
            return
        try:
            os.makedirs(self.run_dir, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def _load_index(self) -> Dict[str, dict]:
        if self._index is not None:
            return self._index
        idx: Dict[str, dict] = {}
        if not self.enabled:
            self._index = idx
            return idx
        try:
            with open(self._index_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    key = entry.get("key")
                    if not key:
                        continue
                    idx[key] = entry
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass
        # Byte accounting must reflect LIVE entries only (codex round-7
        # F5): duplicate index records for the same key used to add every
        # historical record's size, inflating _current_bytes and causing
        # valid results to be evicted after restart.
        self._current_bytes = sum(
            int(entry.get("bytes", 0) or 0) for entry in idx.values()
        )
        self._index = idx
        return idx

    def _append_index(self, entry: dict) -> None:
        try:
            with open(self._index_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ put

    def put(
        self,
        key: str,
        tool_name: str,
        tool_args: Any,
        result: Any,
        *,
        iteration: int = 0,
    ) -> Optional[str]:
        """Persist a full raw result. Returns the file path, or None on any
        failure / when disabled. Idempotent: re-putting an existing key
        overwrites the file and refreshes the index entry."""
        if not self.enabled or not key:
            return None
        self._ensure_dir()
        path = os.path.join(self.run_dir, f"{key}.json")
        payload = {
            "key": key,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            "stored_at": time.time(),
            "iter": int(iteration),
        }
        try:
            blob = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        except Exception:  # noqa: BLE001
            return None
        size = len(blob)
        try:
            # Atomic write.
            fd, tmp = tempfile.mkstemp(dir=self.run_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(blob)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            return None

        idx = self._load_index()
        old = idx.get(key)
        if old is not None:
            self._current_bytes -= int(old.get("bytes", 0) or 0)
        idx[key] = {
            "key": key,
            "tool": tool_name,
            "args_digest": _args_digest(tool_args),
            "path": path,
            "bytes": size,
            "iter": int(iteration),
            "stored_at": payload["stored_at"],
        }
        self._current_bytes += size
        self._append_index(idx[key])
        self.puts += 1
        self._enforce_bounds(idx)
        # Index compaction (codex round-7 F4): index.jsonl is append-only
        # and replacements keep appending records. Without periodic
        # compaction the file grows unboundedly even when payload bytes
        # stay within cap. Compact when the index itself outgrows a
        # fraction of the payload cap.
        self._maybe_compact_index(idx)
        return path

    def _maybe_compact_index(self, idx: Dict[str, dict]) -> None:
        """Rewrite index.jsonl with only live entries when it balloons."""
        try:
            if not os.path.exists(self._index_path):
                return
            index_bytes = os.path.getsize(self._index_path)
            if index_bytes <= max(1 << 20, self.max_bytes // 4):
                return
            tmp_path = self._index_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                for entry in idx.values():
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self._index_path)
        except Exception:  # noqa: BLE001 — compaction is best-effort
            pass

    def _enforce_bounds(self, idx: Dict[str, dict]) -> None:
        """Prune oldest stored keys (by stored_at) until under the byte cap."""
        if self._current_bytes <= self.max_bytes:
            return
        ordered = sorted(idx.values(), key=lambda e: e.get("stored_at", 0))
        for entry in ordered:
            if self._current_bytes <= self.max_bytes:
                break
            key = entry.get("key")
            path = entry.get("path")
            if path:
                try:
                    os.unlink(path)
                except Exception:  # noqa: BLE001
                    pass
            self._current_bytes -= int(entry.get("bytes", 0) or 0)
            idx.pop(key, None)
            self.evictions += 1
        # Rewrite the index cleanly after a prune so it stays consistent.
        self._rewrite_index(idx)

    def _rewrite_index(self, idx: Dict[str, dict]) -> None:
        try:
            fd, tmp = tempfile.mkstemp(dir=self.run_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for entry in idx.values():
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.replace(tmp, self._index_path)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ get

    def get(self, key: str) -> Optional[dict]:
        """Return the stored payload (with ``result``), or None if missing."""
        if not self.enabled or not key:
            return None
        idx = self._load_index()
        entry = idx.get(key)
        path = entry.get("path") if entry else None
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.disk_hits += 1
            return payload
        except Exception:  # noqa: BLE001
            return None

    def get_text(self, key: str) -> Optional[str]:
        """Convenience: the stored result rendered to text for line ops."""
        payload = self.get(key)
        if payload is None:
            return None
        return _render_text(payload.get("result"))

    def has(self, key: str) -> bool:
        if not self.enabled or not key:
            return False
        idx = self._load_index()
        entry = idx.get(key)
        return bool(entry and entry.get("path") and os.path.exists(entry["path"]))

    # ------------------------------------------------------- bounded retrieval

    def line_range(self, key: str, start: int, end: int) -> Optional[str]:
        text = self.get_text(key)
        if text is None:
            return None
        lines = text.splitlines()
        start = max(1, int(start))
        end = int(end) if end is not None else len(lines)
        end = min(end, len(lines))
        if start > end:
            return ""
        return "\n".join(lines[start - 1 : end])

    def head(self, key: str, n: int = 20) -> Optional[str]:
        text = self.get_text(key)
        if text is None:
            return None
        n = max(1, int(n))
        return "\n".join(text.splitlines()[:n])

    def tail(self, key: str, n: int = 20) -> Optional[str]:
        text = self.get_text(key)
        if text is None:
            return None
        n = max(1, int(n))
        return "\n".join(text.splitlines()[-n:])

    def search(self, key: str, query: str, max_matches: int = 20) -> Optional[str]:
        text = self.get_text(key)
        if text is None:
            return None
        if not query:
            return ""
        q = query.lower()
        matches: List[str] = []
        for i, line in enumerate(text.splitlines(), start=1):
            if q in line.lower():
                matches.append(f"{i}: {line}")
                if len(matches) >= max(1, int(max_matches)):
                    break
        return "\n".join(matches)

    def diagnostics(self, key: str, max_lines: int = 40) -> Optional[str]:
        text = self.get_text(key)
        if text is None:
            return None
        seen: List[str] = []
        for line in text.splitlines():
            if _NOISE_RE.match(line):
                continue
            if not _DIAGNOSTIC_RE.search(line):
                continue
            if line in seen:
                continue
            seen.append(line)
            if len(seen) >= max(1, int(max_lines)):
                break
        return "\n".join(seen)

    def json_path(self, key: str, pointer: str) -> Optional[Any]:
        """Extract a sub-value by a JSON path. Accepts either slash-style
        JSON pointer (``/data/matches/0/file``) or dotted (``data.matches.0.file``).
        If the stored ``result`` is a JSON string, it is parsed first.

        Returns the value, or None if missing/not JSON. Brackets not
        supported — kept minimal and dependency-free."""
        payload = self.get(key)
        if payload is None:
            return None
        target = payload.get("result")
        # Auto-parse JSON strings so this works on stored tool output that
        # was a JSON string (the common case for structured tool results).
        if isinstance(target, str) and target.lstrip().startswith(("{", "[")):
            try:
                import json as _json
                target = _json.loads(target)
            except Exception:  # noqa: BLE001
                pass
        if not pointer:
            return target
        # Normalize slash-style pointer to dotted segments.
        segs = [s for s in pointer.replace("/", ".").split(".") if s != ""]
        cur: Any = target
        for part in segs:
            if cur is None:
                return None
            if isinstance(cur, list):
                try:
                    cur = cur[int(part)]
                except (ValueError, IndexError):
                    return None
            elif isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur

    def compare(self, key_a: str, key_b: str) -> Optional[str]:
        """A compact unified-diff-style comparison of two stored results.

        Uses difflib (stdlib). Returns the diff text, or None if either key
        is missing."""
        import difflib

        a = self.get_text(key_a)
        b = self.get_text(key_b)
        if a is None or b is None:
            return None
        diff = difflib.unified_diff(
            (a or "").splitlines(),
            (b or "").splitlines(),
            fromfile=key_a,
            tofile=key_b,
            lineterm="",
        )
        return "\n".join(diff)

    # ------------------------------------------------------------------ GC

    def gc(self, *, now: Optional[float] = None) -> int:
        """Drop run directories older than ``gc_age_days``. Returns the count
        of removed run dirs. Best-effort."""
        if not self.enabled or self.gc_age_days <= 0:
            return 0
        if now is None:
            now = time.time()
        cutoff = now - self.gc_age_days * 86400
        removed = 0
        try:
            for name in os.listdir(self.root):
                d = os.path.join(self.root, name)
                try:
                    st = os.stat(d)
                except Exception:  # noqa: BLE001
                    continue
                if os.path.isdir(d) and st.st_mtime < cutoff and name != self.run_id:
                    try:
                        import shutil

                        shutil.rmtree(d)
                        removed += 1
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        return removed

    # ----------------------------------------------------------- introspection

    def summary(self) -> Dict[str, int]:
        idx = self._load_index() if self.enabled else {}
        return {
            "entries": len(idx),
            "bytes": int(self._current_bytes),
            "puts": int(self.puts),
            "disk_hits": int(self.disk_hits),
            "evictions": int(self.evictions),
        }


def _args_digest(tool_args: Any) -> str:
    """Short stable digest of tool args for the index (not a security hash)."""
    import hashlib

    try:
        blob = json.dumps(tool_args, sort_keys=True, default=str).encode("utf-8")
    except Exception:  # noqa: BLE001
        blob = str(tool_args).encode("utf-8", "replace")
    return hashlib.sha256(blob).hexdigest()[:12]