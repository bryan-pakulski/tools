"""Context-window observability for the GUI Memory Center.

``build_memory_snapshot`` turns the layered system prompt the harness
assembles each turn into a 2-D grid: one horizontal band per layer
(L0..L5), plus an explicit free-space band. Band height is proportional
to the *total available context window*, not merely the currently used
layers. Each
**cell carries a heat value** (0..255) rather than a raw color, so the
frontend can render either a change-frequency heatmap or a solid
per-layer color-coding from the same payload.

Two signals, one grid:

* **Per-layer hue** — every layer gets a fixed hue (L0 blue, L1 green,
  …, L5 red). The band's color identifies *which* layer it is at a
  glance.
* **Change frequency (hash-based)** — each layer's text is split into one
  chunk **per grid cell in its band** (``r * cols`` row-major slices, so
  the chunk count equals the band's cell count — one hash per displayed
  cell) and SHA-256 hashed. The fingerprint remembers each cell's last
  hash + a per-cell change counter, keyed by the session object and the
  ``(cols, rows)`` resolution. Every snapshot compares the current cell
  hashes to the stored ones; a cell whose hash changed increments its
  counter. Cell brightness encodes that counter (sqrt curve, capped), so
  the regions of a layer that change every iteration *glow* and the
  regions stable since first contact stay *dim*. That is the whole point
  of the panel: watching **which regions of each layer's memory churn**
  in real time, and how often — not just that the layer grew.

  Because chunking tracks the band's actual cell count, a layer whose
  band height changes (its token share drifted) is re-chunked. To keep
  the change signal across that resize, cells are compared by
  *fractional position* against the previous layout (the cell at "30%
  through the layer" still corresponds to "30% through"), so the regions
  that shifted still light up — at the cost of some boundary noise from
  the moved slice edges on that one snapshot. While the band height is
  stable (the common case within a turn), cells compare directly and the
  per-cell heat accumulates cleanly.

Each resolution keeps its own change history (keyed by ``(cols, rows)``),
so switching resolution starts a fresh heat for that view. Identical
content between two snapshots at the same resolution yields an identical
grid (deterministic); only changed cells brighten. Empty space is still
space: every non-empty layer gets at least one band row while genuinely
empty layers consume no capacity. A cell whose slice has no text renders
as a dim layer-colored cell rather than vanishing.

The same builder feeds the REST endpoint (``/api/memory/state``) and the
``pre_provider_call`` hook that pushes a live snapshot per iteration, so
the layout is identical between the live and final frames.

The hook also calls ``record_context_snapshot`` once per real provider call.
That bounded timeline stores token and fixed-slice hash measurements only,
enabling heatmap, flow and churn views without retaining another copy of the
prompt text.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
import weakref
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from utils.runtime_metrics import collect_context_layers

_logger = logging.getLogger(__name__)

# Canonical layer order — matches the assembly in
# mu/session/context.py:inject_hierarchical_context.
_LAYER_ORDER: Tuple[str, ...] = ("L0", "L1A", "L1B", "L2", "L3", "L4B", "L5")

# A distinct hue per layer (HSL degrees) so bands are visually identifiable.
# Spread around the wheel; L5 (the volatile history) lands on red.
LAYER_HUES: Dict[str, int] = {
    "L0": 210,  # blue  — system prompt (stable)

    "L1B": 168,  # teal  — installed skills
    "L2": 280,  # purple — conversation summary
    "L3": 25,  # orange — active goal
    "L4B": 50,  # yellow — retrieved snippets
    "L5": 358,  # red   — conversation history (churns most)
}

# Resolution bounds the router enforces. 256×256 is the documented max
# (65,536 cells); the live SSE push uses a smaller grid to stay light.
_MIN_RES = 16
_MAX_RES = 256
_DEFAULT_RES = 128

# Live-push resolution — capped so a per-iteration SSE event stays well
# under ~100 KB even at 256-wide display. The canvas scales to the same
# on-screen size as the full-res frame, so the handoff at turn end is
# visually seamless.
_LIVE_RES = 96

# Change-count at which a cell reaches max brightness (sqrt curve below).
# ~8 observed changes → fully hot; stable content stays dim.
_HEAT_REF = 8.0

# Neutral blue-grey reserved for the unoccupied portion of the context
# window. It is deliberately not a layer hue: it means capacity available to
# every layer, rather than another prompt component.
_FREE_HUE = 215

# The temporal view records one compact point per provider call.  It keeps
# hashes and aggregate measurements only: raw prompt/layer text remains in the
# live session and is never copied into the history.  A bounded runtime ledger
# is intentional here -- this is an observability trace for a session run, not
# another durable memory store.
_TIMELINE_MAX_POINTS = 360
_TIMELINE_CHUNKS = 64
_TIMELINES: "weakref.WeakKeyDictionary[Any, deque]" = weakref.WeakKeyDictionary()
_TIMELINE_LOCK = threading.RLock()


def _clamp_res(value: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if v < _MIN_RES:
        return _MIN_RES
    if v > _MAX_RES:
        return _MAX_RES
    return v


def _hash_color(chunk: str) -> str:
    """Deterministic chunk → ``#rrggbb`` (kept for the debug/identity view
    and for tests; the live grid uses heat ints, not this).

    SHA-256 of the UTF-8 bytes, take the first three bytes as RGB, then
    lift each channel into [40, 255] so colors are never near-invisible
    on a dark theme. Purely a function of the chunk text.
    """
    digest = hashlib.sha256(chunk.encode("utf-8", "replace")).digest()
    r, g, b = digest[0], digest[1], digest[2]

    def lift(byte: int) -> int:
        return 40 + (byte * 215 // 255)

    return f"#{lift(r):02x}{lift(g):02x}{lift(b):02x}"


def _chunk_hash(chunk: str) -> str:
    """Stable hash of one canonical chunk (hex digest prefix)."""
    return hashlib.sha256(chunk.encode("utf-8", "replace")).hexdigest()[:16]


def _sample_chunks(text: str, n: int) -> List[str]:
    """Split ``text`` into ``n`` evenly-spaced slices.

    Even sampling (rather than a prefix split) means a band represents its
    whole layer, not just its opening words — important for L5, where a
    long conversation's recent tail is usually what changed. Empty slices
    (when ``n`` exceeds the text length, or the layer is empty) yield
    ``""``, which the caller maps to a 0 (absent/transparent) cell.
    """
    if n <= 0:
        return []
    if not text:
        return ["" for _ in range(n)]
    tlen = len(text)
    out: List[str] = []
    for i in range(n):
        start = i * tlen // n
        end = (i + 1) * tlen // n
        out.append(text[start:end])
    return out


def _layer_text(session: Any, layer_id: str) -> str:
    """Best-effort text body for one layer.

    Imports ``_layer_content`` lazily so importing this module never pulls
    the commands package (and its CLI deps). Swallows builder failures —
    a layer that can't be materialized just contributes an empty band.
    """
    try:
        from mu.commands.memory import _layer_content

        return str(_layer_content(session, layer_id) or "")
    except Exception as exc:  # defensive — runs on the FastAPI thread too
        _logger.debug("memory_snapshot: layer %s read failed: %s", layer_id, exc)
        return ""


# ---------------------------------------------------------------- change tracking
#
# Per-session fingerprint: for each layer, the last hash of every canonical
# chunk plus a per-chunk change counter. Keyed by the session *object* via a
# WeakKeyDictionary, so the entry auto-evicts the moment the session is
# garbage-collected — change frequency is a runtime observation tied to one
# live session, not persisted history, and weak-keying means a recycled
# id() can never inherit a dead session's counts (which would otherwise
# make a fresh session's first snapshot look "already churning").
#
# Structure: { session(obj): { (cols, rows): { <layer>: { "hashes": [...], "counts": [...] } } } }
# Round-27 F4: provider hooks (agent threads) and REST snapshots (server
# thread) mutate this structure concurrently — every access goes through
# _FINGERPRINTS_LOCK. Round-27 F2: each resolution key holds up to
# r*cols cell hashes; without a cap, requesting varying resolutions
# grows the per-session map unboundedly — _MAX_RESOLUTIONS evicts the
# least-recently-used resolution.
_FINGERPRINTS: "weakref.WeakKeyDictionary[Any, Dict[str, Dict[str, List[Any]]]]" = (
    weakref.WeakKeyDictionary()
)
_FINGERPRINTS_LOCK = threading.Lock()
_MAX_RESOLUTIONS = 16


def _fingerprint(session: Any) -> Dict[str, Dict[str, List[Any]]]:
    with _FINGERPRINTS_LOCK:
        fp = _FINGERPRINTS.get(session)
        if fp is None:
            fp = {}
            _FINGERPRINTS[session] = fp
        # Round-27 F2: LRU bound on resolutions per session. dict
        # preserves insertion order; re-inserting on hit refreshes
        # recency, popping from the front evicts the oldest use.
        if len(fp) > _MAX_RESOLUTIONS:
            while len(fp) > _MAX_RESOLUTIONS:
                fp.pop(next(iter(fp)))
        return fp


def _heat_value(count: int) -> int:
    """Map a per-chunk change count to a 0..254 heat magnitude.

    sqrt curve so the first few changes spread out perceptually; capped at
    ``_HEAT_REF`` changes → max. A stable (count 0) chunk that has content
    still renders (value 1 = dim presence) so the band is visible — only
    truly empty chunks map to 0 (absent/transparent).
    """
    if count <= 0:
        return 0
    t = math.sqrt(min(count, _HEAT_REF) / _HEAT_REF)
    return max(1, round(t * 254))


def _changed(new_hash: str, prev_hash: str) -> bool:
    """Did this chunk's content change between snapshots?

    A hash diff is a change; a chunk going present→absent or absent→present
    (one side empty, the other not) is also a change — the region's content
    materially moved in or out. Both-empty is no change.
    """
    if new_hash and prev_hash:
        return new_hash != prev_hash
    return bool(new_hash) != bool(prev_hash)


def _prop_index(j: int, new_n: int, old_n: int) -> int:
    """Index in a length-``old_n`` array for fractional position ``j`` of a
    length-``new_n`` array — used to correspond cells across a band resize,
    so "30% through the layer" still maps to "30% through" after re-chunking.
    """
    if old_n <= 0:
        return 0
    idx = int((j + 0.5) * old_n / new_n)
    return 0 if idx < 0 else (old_n - 1 if idx >= old_n else idx)


def _empty_state(cols: int, rows: int) -> Dict[str, Any]:
    grid: List[List[int]] = [[0] * cols for _ in range(rows)]
    return {
        "active": False,
        "cols": cols,
        "rows": rows,
        "layers": [],
        "regions": [],
        "grid": grid,
        "total_tokens": 0,
        "context_limit": 0,
        "free_tokens": 0,
        "fill_pct": 0.0,
        "updated_at": None,
    }


def _capacity_rows(amounts: Dict[str, int], rows: int) -> Dict[str, int]:
    """Allocate raster rows by each region's share of total capacity.

    Positive regions receive one minimum row before largest-remainder
    allocation.  Without that floor a real but small layer (usually L5 early
    in a long-context session) rounds to zero against the large FREE region,
    making its changes completely invisible. Empty layers still receive no
    rows, so the picture does not pretend unused layers occupy capacity.
    """
    total = sum(max(0, value) for value in amounts.values())
    out = {key: 0 for key in amounts}
    if total <= 0:
        return out
    positive = [key for key, value in amounts.items() if value > 0]
    if len(positive) > rows:
        positive = []
    assigned = 0
    for key in positive:
        out[key] = 1
        assigned += 1

    remaining = rows - assigned
    if remaining <= 0:
        return out

    fractions = []
    distributed = 0
    for index, (key, value) in enumerate(amounts.items()):
        exact = remaining * max(0, value) / total
        base = int(exact)
        out[key] += base
        distributed += base
        fractions.append((exact - base, -index, key))
    for _, _, key in sorted(fractions, reverse=True)[: remaining - distributed]:
        out[key] += 1
    return out


def _public_timeline_point(point: Dict[str, Any]) -> Dict[str, Any]:
    """Strip comparison-only fingerprints before returning a history point."""
    return {key: value for key, value in point.items() if not key.startswith("_")}


def record_context_snapshot(
    session: Any,
    snapshot: Dict[str, Any],
    *,
    phase: str = "provider_call",
    recorded_at: float | None = None,
) -> Dict[str, Any]:
    """Append one privacy-preserving context observation for ``session``.

    Temporal churn is measured with a fixed 64-slice fingerprint per layer,
    independent of the current square-raster resolution.  This makes history
    comparable across browser sizes and avoids a read of ``/state`` changing
    the trace.  The returned point is safe for the API and SSE payload.
    """
    if session is None or not snapshot.get("active"):
        return {}

    layer_by_id = {item.get("id"): item for item in snapshot.get("layers", [])}
    hashes_by_layer: Dict[str, List[str]] = {}
    fingerprints: Dict[str, str] = {}
    for layer_id in _LAYER_ORDER:
        text = _layer_text(session, layer_id)
        hashes_by_layer[layer_id] = [
            _chunk_hash(chunk) if chunk else ""
            for chunk in _sample_chunks(text, _TIMELINE_CHUNKS)
        ]
        fingerprints[layer_id] = _chunk_hash(text) if text else ""

    with _TIMELINE_LOCK:
        history = _TIMELINES.get(session)
        if history is None:
            history = deque(maxlen=_TIMELINE_MAX_POINTS)
            _TIMELINES[session] = history
        previous = history[-1] if history else None
        previous_hashes = (previous or {}).get("_hashes", {})
        previous_fingerprints = (previous or {}).get("_fingerprints", {})
        previous_layers = {
            item.get("id"): item for item in (previous or {}).get("layers", [])
        }

        layers: List[Dict[str, Any]] = []
        total_changed = 0
        comparable_chunks = 0
        changed_layers = 0
        for layer_id in _LAYER_ORDER:
            current = layer_by_id.get(layer_id, {})
            hashes = hashes_by_layer[layer_id]
            old_hashes = previous_hashes.get(layer_id, [])
            if previous is None:
                changed_chunks = 0
                changed = False
            else:
                changed_chunks = sum(
                    1
                    for index, value in enumerate(hashes)
                    if _changed(
                        value,
                        old_hashes[index] if index < len(old_hashes) else "",
                    )
                )
                changed = fingerprints[layer_id] != previous_fingerprints.get(
                    layer_id, ""
                )
            active_chunks = sum(
                1
                for index, value in enumerate(hashes)
                if value or (index < len(old_hashes) and old_hashes[index])
            )
            if active_chunks:
                comparable_chunks += active_chunks
                total_changed += changed_chunks
            if changed:
                changed_layers += 1
            previous_tokens = int(
                (previous_layers.get(layer_id, {}) or {}).get("tokens") or 0
            )
            tokens = int(current.get("tokens") or 0)
            layers.append(
                {
                    "id": layer_id,
                    "name": current.get("name", layer_id),
                    "hue": LAYER_HUES.get(layer_id, 0),
                    "tokens": tokens,
                    "token_delta": tokens - previous_tokens if previous else 0,
                    "changed": changed,
                    "changed_chunks": changed_chunks,
                    "sampled_chunks": _TIMELINE_CHUNKS,
                    "change_ratio": (
                        round(changed_chunks / active_chunks, 4)
                        if active_chunks
                        else 0.0
                    ),
                }
            )

        total_tokens = int(snapshot.get("total_tokens") or 0)
        previous_total = int((previous or {}).get("total_tokens") or 0)
        total_delta = total_tokens - previous_total if previous else 0
        compaction_threshold = max(256, round(previous_total * 0.08))
        point = {
            "id": int((previous or {}).get("id") or 0) + 1,
            "at": float(recorded_at if recorded_at is not None else time.time()),
            "phase": phase,
            "total_tokens": total_tokens,
            "total_delta": total_delta,
            "context_limit": int(snapshot.get("context_limit") or 0),
            "free_tokens": int(snapshot.get("free_tokens") or 0),
            "fill_pct": float(snapshot.get("fill_pct") or 0),
            "token_source": snapshot.get("token_source", "layer_estimate"),
            "churn_score": (
                round(100.0 * total_changed / comparable_chunks, 1)
                if comparable_chunks
                else 0.0
            ),
            "changed_layers": changed_layers,
            "compaction": bool(previous and total_delta <= -compaction_threshold),
            "layers": layers,
            "_hashes": hashes_by_layer,
            "_fingerprints": fingerprints,
        }
        history.append(point)
        return _public_timeline_point(point)


def get_context_timeline(session: Any, *, limit: int = 240) -> Dict[str, Any]:
    """Return the bounded provider-call history and a small insight summary."""
    if session is None:
        return {"active": False, "points": [], "summary": {"samples": 0}}
    try:
        safe_limit = max(1, min(int(limit), _TIMELINE_MAX_POINTS))
    except (TypeError, ValueError):
        safe_limit = 240
    with _TIMELINE_LOCK:
        history = list(_TIMELINES.get(session) or [])[-safe_limit:]
        points = [_public_timeline_point(point) for point in history]

    if not points:
        return {
            "active": True,
            "points": [],
            "summary": {
                "samples": 0,
                "net_delta": 0,
                "peak_tokens": 0,
                "peak_fill_pct": 0.0,
                "compactions": 0,
                "hottest_layer": None,
            },
        }

    churn_by_layer = {layer_id: 0 for layer_id in _LAYER_ORDER}
    for point in points:
        for layer in point.get("layers", []):
            churn_by_layer[layer["id"]] += int(layer.get("changed_chunks") or 0)
    hottest_id, hottest_churn = max(churn_by_layer.items(), key=lambda item: item[1])
    summary = {
        "samples": len(points),
        "first_tokens": points[0]["total_tokens"],
        "last_tokens": points[-1]["total_tokens"],
        "net_delta": points[-1]["total_tokens"] - points[0]["total_tokens"],
        "peak_tokens": max(point["total_tokens"] for point in points),
        "peak_fill_pct": max(point["fill_pct"] for point in points),
        "compactions": sum(1 for point in points if point.get("compaction")),
        "hottest_layer": hottest_id if hottest_churn else None,
        "hottest_layer_changes": hottest_churn,
        "max_churn_score": max(point["churn_score"] for point in points),
    }
    return {"active": True, "points": points, "summary": summary}


def ingest_context_timeline_point(
    session: Any,
    point: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Mirror a content-free observation produced by a container worker.

    The worker owns the real prompt and computes churn there.  The host keeps a
    bounded public copy so page refreshes and the normal `/api/memory/timeline`
    endpoint behave exactly like a host workspace while the session remains
    loaded. Raw prompt text and private comparison hashes are never accepted.
    """
    if session is None or not isinstance(point, dict) or not point.get("id"):
        return {}
    layers = []
    for raw in point.get("layers", []) or []:
        if not isinstance(raw, dict):
            continue
        layers.append(
            {
                "id": str(raw.get("id") or ""),
                "name": str(raw.get("name") or raw.get("id") or ""),
                "hue": int(raw.get("hue") or 0),
                "tokens": int(raw.get("tokens") or 0),
                "token_delta": int(raw.get("token_delta") or 0),
                "changed": bool(raw.get("changed")),
                "changed_chunks": int(raw.get("changed_chunks") or 0),
                "sampled_chunks": int(raw.get("sampled_chunks") or 0),
                "change_ratio": float(raw.get("change_ratio") or 0.0),
            }
        )
    safe = {
        "id": int(point.get("id") or 0),
        "at": float(point.get("at") or time.time()),
        "phase": str(point.get("phase") or "provider_call"),
        "total_tokens": int(point.get("total_tokens") or 0),
        "total_delta": int(point.get("total_delta") or 0),
        "context_limit": int(point.get("context_limit") or 0),
        "free_tokens": int(point.get("free_tokens") or 0),
        "fill_pct": float(point.get("fill_pct") or 0.0),
        "token_source": str(point.get("token_source") or "layer_estimate"),
        "churn_score": float(point.get("churn_score") or 0.0),
        "changed_layers": int(point.get("changed_layers") or 0),
        "compaction": bool(point.get("compaction")),
        "layers": layers,
    }
    with _TIMELINE_LOCK:
        history = _TIMELINES.get(session)
        if history is None:
            history = deque(maxlen=_TIMELINE_MAX_POINTS)
            _TIMELINES[session] = history
        existing_index = next(
            (
                index
                for index, value in enumerate(history)
                if int(value.get("id") or 0) == safe["id"]
            ),
            None,
        )
        if existing_index is not None and abs(
            float(history[existing_index].get("at") or 0.0) - safe["at"]
        ) > 0.001:
            # A rebuilt worker starts its local counter at one. Preserve the
            # earlier host timeline and assign the incoming observation the
            # next session-wide id instead of overwriting history.
            safe["id"] = max(
                (int(value.get("id") or 0) for value in history),
                default=0,
            ) + 1
            existing_index = None
        if existing_index is None:
            history.append(safe)
        else:
            history[existing_index] = safe
    return dict(safe)


def build_memory_snapshot(
    session: Any,
    cols: int = _DEFAULT_RES,
    rows: int = _DEFAULT_RES,
    *,
    request_token_estimate: int | None = None,
) -> Dict[str, Any]:
    """Build a layer-banded heat-grid snapshot of the active context.

    Returns a JSON-serializable dict with:
      * ``layers`` — per-layer ``{id, name, tokens, max, fill_pct, hue,
        change_count, row_start, row_end}`` for the legend.
      * ``grid`` — ``rows`` × ``cols`` of ints. ``0`` = no content for
        that cell (the frontend renders it as a dim "empty space"); ``1..255``
        = present, magnitude encodes change frequency (1 = stable, 255 =
        churning).
      * ``total_tokens`` / ``context_limit`` / ``fill_pct`` for the header.

    The frontend derives each cell's color from the layer hue + the heat
    value + the active view mode (heatmap vs. solid layer color), so this
    single payload supports both render modes without a re-fetch.
    """
    cols = _clamp_res(cols, _DEFAULT_RES)
    rows = _clamp_res(rows, _DEFAULT_RES)

    if session is None:
        return _empty_state(cols, rows)

    try:
        token_layers = collect_context_layers(session)
    except Exception as exc:
        _logger.warning("memory_snapshot: collect_context_layers failed: %s", exc)
        return _empty_state(cols, rows)

    by_id = {entry["layer"]: entry for entry in token_layers}
    layer_tokens = {
        lid: int((by_id.get(lid, {}) or {}).get("current") or 0) for lid in _LAYER_ORDER
    }
    layer_total = sum(layer_tokens.values())

    # The layer inspector is assembled independently from the exact provider
    # request.  In particular, it cannot infer provider message framing or
    # transient prompt additions.  Prefer the request estimate captured at
    # the pre-provider seam (the same value written to the Trace Analyzer),
    # and reconcile that small remainder into L0 rather than displaying a
    # context total that disagrees with the trace.
    if request_token_estimate is None:
        request_token_estimate = getattr(
            session, "_memory_map_request_token_estimate", None
        )
    try:
        request_total = max(0, int(request_token_estimate))
    except (TypeError, ValueError):
        request_total = 0
    if request_total:
        layer_tokens["L0"] = max(0, layer_tokens["L0"] + request_total - layer_total)
        total_tokens = sum(layer_tokens.values())
        token_source = "pre_request_estimate"
    else:
        total_tokens = layer_total
        token_source = "layer_estimate"

    context_limit = 0
    try:
        context_limit = int(by_id.get("L5", {}).get("maximum") or 0)
    except Exception:
        context_limit = 0
    if context_limit <= 0:
        context_limit = sum(
            int((by_id.get(lid, {}) or {}).get("maximum") or 0) for lid in _LAYER_ORDER
        )
    # The raw window (context_token_limit) is the provider's real capacity,
    # but the compactor fires on the drift-corrected effective ceiling (raw
    # ÷ safety/drift factor — 2.5 for Ollama). Size the map against the
    # SAME effective ceiling so the Memory Map's fill matches /memory, the
    # splash banner, and the compactor's actual trigger — not the raw
    # window, which made Ollama look half-empty while emergency compaction
    # was already firing. No-op for providers with no safety factor.
    context_limit_raw = context_limit
    try:
        from mu.session.budgets import drift_corrected_context_limit

        context_limit = max(1, int(drift_corrected_context_limit(session)))
    except Exception:  # noqa: BLE001
        pass

    fill_pct = (
        round(100.0 * total_tokens / context_limit, 1) if context_limit > 0 else 0.0
    )

    # --- Capacity layout. The grid is a scaled picture of the complete
    # context window: used layers consume their current tokens and FREE is
    # the genuinely available remainder. This is intentionally unlike a
    # composition chart, where the used layers would always fill 100%.
    band_rows: Dict[str, int] = {}
    displayed_tokens = (
        min(total_tokens, context_limit) if context_limit > 0 else total_tokens
    )
    free_tokens = max(0, context_limit - displayed_tokens)
    capacity_amounts = {lid: layer_tokens[lid] for lid in _LAYER_ORDER}
    # If the prompt is over budget, retain the full used-layer proportions;
    # there is no fictional free area to display.
    if total_tokens > context_limit > 0:
        capacity_amounts = layer_tokens.copy()
    else:
        capacity_amounts["FREE"] = free_tokens
    band_rows = _capacity_rows(capacity_amounts, rows)

    # --- change tracking: one hash per GRID CELL in the band (r*cols
    # row-major slices — one chunk per displayed cell), so changes light up
    # at cell granularity and you can see *which regions* of a layer churn.
    # Keyed by (cols, rows) so each resolution keeps its own change history.
    # Round-47 F11/F12: layer texts were materialized TWICE per provider call
    # (once here for the grid, once in record_context_snapshot for the 64-
    # slice timeline) — both walks are O(history). The hook now assembles
    # them once and passes them via session._memory_layer_texts; only an
    # explicit /state request without the precomputed texts re-materializes.
    layer_texts = getattr(session, "_memory_layer_texts", None)
    if not isinstance(layer_texts, dict) or set(layer_texts) < set(_LAYER_ORDER):
        layer_texts = {lid: _layer_text(session, lid) for lid in _LAYER_ORDER}
    fp = _fingerprint(session)
    with _FINGERPRINTS_LOCK:
        # Refresh recency for the LRU bound (round-27 F2). Snapshot the
        # current state + generation; the swap at the end is validated
        # against the generation (round-48 F13) so a concurrent build
        # cannot erase another's counts.
        res_fp = fp.pop((cols, rows), {})
        fp[(cols, rows)] = res_fp
        base_generation = int(res_fp.get("__generation__", 0) or 0)
    layer_heat: Dict[str, List[int]] = {}  # per-cell heat (len == r*cols)
    layer_change_count: Dict[str, int] = {}
    # Round-47 F12: compute the full replacement state locally, then swap
    # each layer's entry under the lock — previously res_fp was read and
    # mutated OUTSIDE the lock, so a concurrent /state request and the
    # provider hook at the same resolution could interleave and lose or
    # double-count changes.
    new_state: Dict[str, Dict[str, Any]] = {}
    for lid in _LAYER_ORDER:
        r = band_rows.get(lid, 0)
        n = r * cols  # one chunk per band cell (row-major)
        text = layer_texts[lid]
        chunks = _sample_chunks(text, n) if n > 0 else []
        hashes = [_chunk_hash(c) for c in chunks]
        present = [bool(c) for c in chunks]

        state = res_fp.get(lid)
        prev_hashes = (state or {}).get("hashes") or []
        prev_counts = (state or {}).get("counts") or []
        prev_r = (state or {}).get("band_rows")

        counts = [0] * n
        if state is not None and prev_hashes:
            if prev_r == r and len(prev_hashes) == n:
                # Band height unchanged → direct cell-by-cell correspondence;
                # carry each cell's accumulated count forward.
                for i in range(n):
                    pc = prev_counts[i] if i < len(prev_counts) else 0
                    counts[i] = pc + (1 if _changed(hashes[i], prev_hashes[i]) else 0)
            else:
                # Band resized → re-chunked. Correspond by fractional position
                # so we still detect *which regions* shifted (with some boundary
                # noise from the moved slice edges on this one snapshot).
                # Counts start fresh for the new layout.
                old_n = len(prev_hashes)
                for i in range(n):
                    if _changed(hashes[i], prev_hashes[_prop_index(i, n, old_n)]):
                        counts[i] = 1
        # else: first snapshot for this (resolution, layer) → counts stay 0.

        new_state[lid] = {"band_rows": r, "hashes": hashes, "counts": counts}
        # 0 = empty cell (no text in that slice); 1..255 = present, where the
        # magnitude encodes change frequency (1 = stable since first seen,
        # 255 = churning).
        layer_heat[lid] = [
            (0 if not present[i] else 1 + _heat_value(counts[i])) for i in range(n)
        ]
        layer_change_count[lid] = sum(counts)

    # Round-47 F12 + Round-48 F13: CAS swap — commit only if no other
    # build advanced the generation while we computed. On mismatch the
    # OTHER build's state wins (ours derives from a stale snapshot; the
    # other one is at least as fresh).
    with _FINGERPRINTS_LOCK:
        current = fp.get((cols, rows)) or {}
        fp[(cols, rows)] = {**current, **new_state, "__generation__": base_generation + 1}
        while len(fp) > _MAX_RESOLUTIONS:
            fp.pop(next(iter(fp)))

    # --- build the grid: each band cell carries its own per-cell heat
    # (row-major within the band), so changes show up as spatial regions,
    # not uniform stripes. Empty cells stay 0 so the frontend renders them
    # as dim "empty space" rather than transparent — the band's full extent
    # is always visible.
    grid: List[List[int]] = [[0] * cols for _ in range(rows)]
    layers_out: List[Dict[str, Any]] = []
    row_cursor = 0
    for lid in _LAYER_ORDER:
        meta = by_id.get(lid, {}) or {}
        tokens = layer_tokens[lid]
        maximum = int(meta.get("maximum") or 0)
        r = band_rows.get(lid, 0)
        row_start = row_cursor
        row_end = min(row_cursor + r, rows)
        heat = layer_heat.get(lid) or []
        for ri in range(row_start, row_end):
            row = grid[ri]
            base = (ri - row_start) * cols
            for ci in range(cols):
                idx = base + ci
                row[ci] = heat[idx] if idx < len(heat) else 0
        row_cursor = row_end

        layers_out.append(
            {
                "id": lid,
                "name": meta.get("name", lid),
                "tokens": tokens,
                "chars": len(layer_texts[lid]),
                "max": maximum,
                "fill_pct": round(100.0 * tokens / maximum, 1) if maximum > 0 else 0.0,
                "hue": LAYER_HUES.get(lid, 0),
                "change_count": layer_change_count.get(lid, 0),
                "row_start": row_start,
                "row_end": row_end,
            }
        )

    regions = list(layers_out)
    free_rows = band_rows.get("FREE", 0)
    if free_rows:
        regions.append(
            {
                "id": "FREE",
                "name": "Available space",
                "tokens": free_tokens,
                "max": context_limit,
                "fill_pct": round(100.0 * free_tokens / context_limit, 1),
                "hue": _FREE_HUE,
                "change_count": 0,
                "row_start": row_cursor,
                "row_end": min(row_cursor + free_rows, rows),
                "free": True,
            }
        )

    return {
        "active": True,
        "cols": cols,
        "rows": rows,
        "layers": layers_out,
        "regions": regions,
        "grid": grid,
        "total_tokens": total_tokens,
        "token_source": token_source,
        "context_limit": context_limit,
        "context_limit_raw": context_limit_raw,
        "free_tokens": free_tokens,
        "fill_pct": fill_pct,
        "updated_at": time.time(),
    }


__all__ = [
    "build_memory_snapshot",
    "get_context_timeline",
    "ingest_context_timeline_point",
    "record_context_snapshot",
    "LIVE_RESOLUTION",
    "LAYER_HUES",
]
# Constant the hook/router import for the live push resolution.
LIVE_RESOLUTION = _LIVE_RES
