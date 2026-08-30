# Cross-Surface Session Continuity — Design

Status: **design only** (no implementation in this document).
Scope: how a single MuCLI session stays consistent across the three live
surfaces — terminal CLI (TUI), browser GUI, and the mobile Android client —
including concurrent-write safety, presence, and conflict policy.

## 1. Current architecture (verified against code)

### Persistence model
- One session = one directory under `~/.mucli/sessions/<name>/` containing
  `session.json`. `SessionManager.save_history` (mu/session/manager.py)
  serializes **the entire session state as one JSON document**:
  history, conversation_summary, summary_anchor, protected_indices,
  provider_config, folder_context, variables, container_config,
  collation_buffer, task_memory, turn_scratchpad, token_counts,
  feature/teacher/research state, tool_stats.
- Writes are **atomic** (tmp + rename, codex round-7 F2) and carry two
  attribution markers: `__writer_pid__` and `__writer_at__`.

### Surface topology
- **CLI (TUI)** — owns a `SessionManager` in-process; writes session.json at
  turn boundaries and on state changes. **No file watcher**: the CLI never
  reloads a session that another surface modified while it was open.
- **GUI (FastAPI)** — holds loaded sessions in `app.state.sessions`; a
  `SessionWatcher` (mu/gui/watcher.py) polls every 2s, compares
  mtime+size, reads `__writer_pid__`, and on *external* writes (pid ≠ its
  own) reloads the session under the per-session lock — deferring if a turn
  is in flight — then publishes `session_updated` on the SSE bus.
- **Mobile** — stateless HTTP + SSE client of the GUI. It never touches the
  filesystem; every write goes through GUI routes, so the GUI's in-memory
  session object is authoritative for mobile-originated changes.

### What already works
- **GUI ↔ CLI sync**: CLI writes session.json → watcher detects (≤2s) →
  GUI reloads under `state.session_lock_for(name)` (deferred while a GUI
  turn is busy) → SSE `session_updated` fans out to browser and mobile.
- **Mobile ↔ GUI sync**: mobile is a thin client; nothing to sync beyond
  the SSE stream it already consumes.
- **Whole-file last-writer-wins** with busy-deferral prevents the most
  common corruption cases.

## 2. Gap analysis

| # | Gap | Consequence |
|---|-----|-------------|
| G1 | CLI has **no inbound watcher** | CLI keeps stale in-memory state after a GUI/mobile write; its next `save_history` **clobbers** the newer GUI/mobile state (silent lost update). |
| G2 | **No revision counter** | Watchers use mtime+size heuristics; equal-mtime writes are undetectable; no way for a writer to do an atomic compare-and-swap. |
| G3 | **Whole-document ownership** | Two surfaces writing "concurrently" (e.g. user typing in GUI file editor while CLI turn is running) interleave at document granularity: one side's variables/task_memory edits are lost even though they touched disjoint keys. |
| G4 | **No presence model** | Surfaces can't tell whether anyone else is *actively* attached to a session (only the watcher's transient `external_active` flag inside the GUI). |
| G5 | Mobile has **no offline queue** | Actions attempted while unreachable are dropped rather than replayed. |
| G6 | Deferred GUI reloads during busy turns are **dropped, not queued** — after the turn completes, the GUI does not re-check for the external write it skipped. |

## 3. Design

### 3.1 Revision counter (closes G2, prerequisite for G1/G3)
- Add `revision: int` to session.json. `save_history` writes
  `revision = loaded_revision + 1` (read-modify-write under the session
  file lock); loaders remember it as `sm.revision`.
- Add optional `expected_revision` to save paths:
  `save_history(expected_revision=None)`. When provided and
  `expected_revision != on-disk revision`, the save **fails** with a
  `RevisionConflict` instead of writing. Callers that opt in get CAS
  semantics ("If-Match"); callers that don't keep last-writer-wins.
- HTTP surface: `GET /api/sessions/{name}` returns `revision`;
  mutating endpoints accept `If-Match: <rev>` and answer `409 Conflict`
  with the current revision on mismatch. Mobile/GUI then reload + replay.

### 3.2 CLI-side watcher (closes G1)
- New `mu/session/surface_sync.py`: a small daemon thread per CLI process
  (started lazily when a session with a live GUI is detected — presence
  file, §3.4) that mirrors the GUI watcher's logic in reverse:
  poll mtime+size of session.json (2s), read `__writer_pid__`, and when
  `pid != os.getpid()` **and the CLI is not mid-turn**, reload
  (`_load_session`) and refresh the TUI history view.
- Mid-turn policy: the CLI never reloads while a turn is executing;
  it marks `pending_external_reload = True` and applies it at the next
  turn boundary (after its own save), exactly mirroring the GUI's
  busy-deferral — but with the re-check G6 requires.
- The TUI renders a one-line notice ("session updated by another
  surface") instead of silently mutating.

### 3.3 Key-granularity ownership (closes G3; phase 2)
- Split session.json's single document into two files:
  - `session.core.json` — conversational state (history, summary, anchor,
    protected_indices, token_counts). Owner: the surface running the turn.
  - `session.meta.json` — preference-ish state (variables, provider/model,
    container_config, feature/teacher/research registries, tool_stats).
- `load_session` merges both; `save_history` gains a `scope="core"|"meta"`
  parameter so each surface saves only the file it owns changes to.
- Conflicts within `meta` are per-key last-writer-wins using per-key
  `__writer_at__` stamps (kept inside the file) — good enough because
  meta keys are small and independent.
- The revision counter (§3.1) becomes two counters: `core_revision`,
  `meta_revision`.

### 3.4 Presence beacons (closes G4)
- Each running surface touches `~/.mucli/sessions/<name>/presence/<pid>.json`
  every 5s: `{surface: cli|gui|mobile, started_at, last_seen, busy}`.
  Stale beacons (>15s) are pruned on read.
- GUI exposes `GET /api/sessions/{name}/presence` (list of live surfaces)
  and includes `presence` in `session_updated` SSE payloads. Mobile shows
  a "also active on: desktop" hint; CLI `/status` lists peers.
- Presence also gates §3.2: the CLI watcher only runs while a GUI beacon
  exists (zero overhead in CLI-only usage).

### 3.5 Conflict policy (normative rules)
1. **Turn granularity ownership**: during a turn, the surface executing
   the turn owns `core`; other surfaces treat that session as read-only
   (their writes to `core` fail CAS with the turn surface's revision).
2. **Meta is cooperative**: any surface may write `meta` at any time;
   per-key stamps resolve ties; UI surfaces a "changed elsewhere" toast
   when a key it displayed was overwritten.
3. **Never merge history**: history/summary/anchor only ever move forward
   via the turn owner. A loser's in-flight user input is preserved in its
   composer (GUI/mobile) or prompt line (CLI), never injected into a
   reloaded history.
4. **Reloads are deferred, not dropped** (fixes G6): every surface that
   defers a reload re-arms its watcher check at turn end; the SSE
   `session_updated` event carries `deferred: true` so clients know to
   expect a second event.

### 3.6 Mobile offline queue (closes G5; phase 3)
- Mobile keeps an outbound queue in AsyncStorage: chat sends, yolo
  toggles, profile switches attempted while `isConnected=false`.
- On reconnect (existing autoReconnect success path), the queue replays in
  order; each request carries `If-Match` from §3.1 — a 409 drops the item
  and surfaces a conflict notice instead of applying stale state.
- Queue cap: 50 items / 24h TTL, FIFO, user-visible count badge.

## 4. Migration and compatibility

- Old sessions without `revision` load as revision 0 and gain it on first
  save — no backfill needed.
- The two-file split (§3.3) ships behind `session.variables["split_meta"]`
  (default off); loader reads the single-file layout transparently until
  the flag flips, so downgrade is a flag flip, not a migration.
- GUI/mobile changes are API-additive (new field + new endpoints); the
  existing SSE event gains optional keys only.

## 5. Test plan (when implemented)

1. **Unit**: revision CAS success/conflict; two-file load/save round-trip;
   per-key stamp merge.
2. **Watcher symmetry**: fake external write (write session.json from a
   subprocess) → CLI reloads and reports; GUI reload of CLI writes
   (already covered by existing watcher tests — extend with deferred
   reload re-check).
3. **Concurrent surfaces**: pytest harness runs SessionManager A (simulated
   CLI) + GUI TestClient on the same session dir; interleave a turn with a
   meta edit; assert no lost update for either.
4. **Presence**: beacon TTL expiry; presence endpoint shape.
5. **Mobile replay**: jest test with mocked fetch — queue drains in order
   on reconnect; 409 mid-queue surfaces conflict and aborts replay.

## 6. Explicit non-goals

- Real-time per-token streaming merge across surfaces (turn ownership
  makes this unnecessary).
- CRDT/OT machinery — document granularity and turn-ownership cover the
  realistic concurrency here.
- Multi-host sync (all surfaces share one filesystem via the GUI host;
  mobile reaches it through the GUI API only).