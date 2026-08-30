import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  discardConflicted,
  enqueueOffline,
  loadQueue,
  requeueConflicted,
  replayQueue,
  revisionToken,
  type ReplayOutcome,
  type RevisionToken,
} from '../api/offlineQueue';

const STORAGE_KEY = '@mucli/connection';
let reconnectInFlight: Promise<void> | null = null;
let storageWriteQueue: Promise<void> = Promise.resolve();
let queueReplayInFlight: Promise<ReplayOutcome> | null = null;

export interface ConnectionState {
  baseUrl: string;
  activeSessionName: string | null;
  activeProvider: string | null;
  activeModel: string | null;
  isConnected: boolean;
  yolo: boolean;
  /** F11: last-known server-side session revision (If-Match for queued
   * offline mutations). Refreshed from GET /api/sessions/active.
   * Round-32b F4: number|string token — never Number()-converted. */
  sessionRevision: RevisionToken | null;
  /** G5: offline mutation queue depth (user-visible badge). */
  pendingMutations: number;
  /** G5: last replay summary — conflicts>0 means server state moved on. */
  lastReplay: ReplayOutcome | null;
  /** Round-31 F40 / round-32b F10: queue item ids parked as conflicted
   * (409 on If-Match), scoped to the ACTIVE session. */
  conflictIds: string[];
  /**
   * Round-32b F10: conflicted ids per session. Queue items are
   * session-scoped, so the conflict state must survive session switches:
   * switching away parks the ids here; switching back restores them.
   * `null` = no conflicts recorded for that session.
   */
  conflictIdsBySession: Record<string, string[] | null>;
  /**
   * Round-34 F1: conflicted ids for items with NO session attribution
   * (sessionName absent AND no active session at conflict time). These are
   * global — the banner shows them regardless of which session is active.
   * Persisted nowhere (rebuilt from the durable queue on refresh); cleared
   * only by requeue/discard/drain, never by session switches.
   */
  globalConflictIds: string[];
  /** Round-31 F40 / round-32b F11: retry a conflicted item with a fresh
   * If-Match token. Undefined token + keepIfMatch=false strips the guard;
   * the banner only requeues when it holds a fresh token. */
  requeueConflict: (id: string, ifMatch?: RevisionToken) => Promise<void>;
  /** Round-31 F40 / round-32b F11: drop a conflicted item permanently. */
  discardConflict: (id: string) => Promise<void>;
  /** F11: record the session revision seen in the last state sync.
   * Round-33b F8: external input is normalized through revisionToken() —
   * a non-canonical value (rounded float, decorated string, object) is
   * DROPPED to null instead of being stored as a bogus If-Match guard. */
  setSessionRevision: (revision: RevisionToken | null) => void;
  setBaseUrl: (url: string) => void;
  setActiveSession: (name: string | null) => void;
  setActiveProviderModel: (provider: string | null, model: string | null) => void;
  setConnected: (connected: boolean) => void;
  setYolo: (yolo: boolean) => void;
  loadFromStorage: () => Promise<void>;
  saveToStorage: () => Promise<void>;
  /** G5: replay the offline queue (FIFO, If-Match CAS on each item). */
  replayPending: () => Promise<ReplayOutcome | null>;
  /** Best-effort background reconnection after cold start. */
  autoReconnect: () => Promise<void>;
}

/** G5: refresh the pending-mutation badge from persisted queue state.
 * Round-32b F8: cold-start hydration — derive conflictIds for the active
 * session from persisted conflicted queue items so the banner survives an
 * app restart (the durable queue is still blocked; the banner must show).
 * Round-33b F7: REBUILD, don't union — the durable queue is the source of
 * truth. Each session's id set is REPLACED from the queue and empty entries
 * pruned, so a discarded item can never resurrect as a conflict after a
 * session switch (the previous union-forever behavior let the map grow
 * unboundedly). The union with in-memory state is kept ONLY for
 * session-less items when a durable sessionName is not recorded. */
export async function refreshPendingMutations(): Promise<number> {
  const queue = await loadQueue();
  const state = useConnectionStore.getState();
  const active = state.activeSessionName;

  // F7: rebuild every session's set from the durable queue (replace, then
  // prune empties). Session-less items attribute to the active session.
  const rebuilt: Record<string, string[]> = {};
  const rebuiltGlobal: string[] = [];
  for (const item of queue) {
    if (item.conflicted !== true) continue;
    const owner = item.sessionName ?? active;
    if (!owner) {
      // Round-34 F1: session-less items with no active session are GLOBAL
      // conflicts — they were previously skipped, hiding the banner even
      // though the durable queue stayed blocked.
      rebuiltGlobal.push(item.id);
      continue;
    }
    (rebuilt[owner] ??= []).push(item.id);
  }
  // Durable queue is the source of truth: the map IS the rebuild (stale
  // in-memory entries for sessions with no conflicted items are pruned).
  const nextMap = rebuilt;
  const activeIds = (active ? nextMap[active] : null) ?? [];
  // Round-34 F1: when a session IS active, its own ids own the banner —
  // global ids stay parked until no session claims them.
  const shownIds = active ? activeIds : rebuiltGlobal;

  useConnectionStore.setState({
    pendingMutations: queue.length,
    conflictIds: shownIds,
    globalConflictIds: rebuiltGlobal,
    conflictIdsBySession: nextMap,
  });
  return queue.length;
}

/** G5 (§3.6): call at mutation entry points. When the client is offline,
 * enqueue the mutation for replay-on-reconnect and return true; when
 * online, return false and the caller performs the request as usual. */
export async function queueIfOffline(
  kind: 'chat_send' | 'set_variable' | 'provider_switch',
  payload: Record<string, unknown>,
  options: { sessionName?: string; ifMatch?: RevisionToken } = {},
): Promise<boolean> {
  if (useConnectionStore.getState().isConnected) return false;
  await enqueueOffline(kind, payload, options);
  await refreshPendingMutations();
  return true;
}

export const useConnectionStore = create<ConnectionState>((set, get) => ({
  baseUrl: 'http://192.168.20.14:30311',
  activeSessionName: null,
  activeProvider: null,
  activeModel: null,
  isConnected: false,
  yolo: false,
  sessionRevision: null,
  pendingMutations: 0,
  lastReplay: null,
  conflictIds: [],
  conflictIdsBySession: {},
  globalConflictIds: [],

  setSessionRevision: (revision: RevisionToken | null) => {
    // Round-33b F8: never store a non-canonical token — an invalid input
    // would later ride out as `If-Match: [object Object]`. null clears.
    set({ sessionRevision: revision === null ? null : revisionToken(revision) });
  },

  setBaseUrl: (url: string) => {
    set({ baseUrl: url.replace(/\/$/, '') });
    get().saveToStorage();
  },

  setActiveSession: (name: string | null) => {
    // Round-19 F43: sessionRevision was global but never cleared on
    // session change — a write issued before the new session's state
    // sync could carry the PREVIOUS session's revision as If-Match,
    // corrupting CAS semantics (or failing 409 forever). Drop the stale
    // revision synchronously; guarded mutations stay disabled until the
    // new session's revision is loaded by the active-state probe.
    // Round-32b F10: conflicts are session-scoped facts. PARK the current
    // ids under the outgoing session and RESTORE the incoming session's
    // ids — the banner must not leak another session's conflicts, but a
    // returning session must see its own again (documented spec change:
    // the old test asserted a plain reset).
    const current = get();
    const parkedMap = { ...current.conflictIdsBySession };
    if (current.activeSessionName) {
      parkedMap[current.activeSessionName] = current.conflictIds.length > 0 ? current.conflictIds : null;
    }
    const restored = name ? parkedMap[name] ?? null : null;
    // Round-34 F1: the global bucket is session-independent — it survives
    // switches untouched (session-less conflicts are shown when no session
    // is active; while a session is active, session ids own the banner).
    const globalIds = current.activeSessionName
      ? current.globalConflictIds
      : (current.conflictIds as string[] | typeof current.globalConflictIds);
    set({
      activeSessionName: name,
      sessionRevision: null,
      conflictIds: restored ? [...restored] : globalIds,
      globalConflictIds: globalIds,
      conflictIdsBySession: parkedMap,
    });
    get().saveToStorage();
  },

  setActiveProviderModel: (provider: string | null, model: string | null) => {
    set({ activeProvider: provider, activeModel: model });
    get().saveToStorage();
  },

  setConnected: (connected: boolean) => {
    set({ isConnected: connected });
    get().saveToStorage();
  },

  setYolo: (yolo: boolean) => {
    set({ yolo });
    get().saveToStorage();
  },

  loadFromStorage: async () => {
    try {
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        set({
          baseUrl: (parsed.baseUrl && !parsed.baseUrl.includes('localhost') ? parsed.baseUrl : 'http://192.168.20.14:30311'),
          activeSessionName: parsed.activeSessionName || null,
          activeProvider: parsed.activeProvider || null,
          activeModel: parsed.activeModel || null,
          // isConnected is live state only — always cold-start disconnected.
          isConnected: false,
          yolo: parsed.yolo || false,
        });
      }
    } catch {
      // AsyncStorage not available or corrupt — keep defaults
    }
  },

  saveToStorage: async () => {
    // MUCLI_MOBILE_RECONNECT_YOLO_V1: serialize snapshots so rapid toggles and
    // connection changes cannot finish AsyncStorage writes out of order.
    const state = get();
    // isConnected is NEVER persisted: it is a live network fact, not a
    // preference. Restoring it made cold starts show connected-only UI
    // (PromptHost etc.) before any probe succeeded.
    const snapshot = JSON.stringify({
      baseUrl: state.baseUrl,
      activeSessionName: state.activeSessionName,
      activeProvider: state.activeProvider,
      activeModel: state.activeModel,
      yolo: state.yolo,
    });
    storageWriteQueue = storageWriteQueue
      .catch(() => undefined)
      .then(() => AsyncStorage.setItem(STORAGE_KEY, snapshot));
    try {
      await storageWriteQueue;
    } catch {
      // Persistence is best-effort. Keep the in-memory connection usable.
    }
  },

  replayPending: async () => {
    // G5 (§3.6): replay queued offline mutations in FIFO order, each with
    // its captured If-Match (§3.1). Round-31 F38: a conflict now marks the
    // item conflicted and HALTS the replay (user resolves via banner);
    // other failures stop the replay and keep the remainder queued.
    // Single-flight so a flapping connection cannot run overlapping replays.
    if (queueReplayInFlight) return queueReplayInFlight;
    const task = (async () => {
      const outcome = await replayQueue();
      // Round-32b F8: MERGE, don't overwrite. The replay reports the parked
      // head (F8 fix) so this only fires when the durable queue truly
      // still blocks; a fully-drained queue clears the ids. Without the
      // merge a shorter/empty outcome list would hide a still-blocked
      // queue's banner.
      const state = get();
      const active = state.activeSessionName;
      // Round-33b F5: attribution comes from the outcome's conflictRecords
      // (id + the item's TARGET sessionName), NOT from whichever session is
      // active after the awaited replay. A session switch DURING the replay
      // can no longer park the id under the wrong session. Records without
      // a sessionName fall back to the active session at completion time.
      // The round-32b merge semantics are preserved via the parked map:
      // conflict records union into their session's set; a fully drained
      // queue clears the ACTIVE session's set (its ids left the durable
      // queue); a blocked queue keeps every other session's parked ids.
      const parkedMap: Record<string, string[] | null> = { ...state.conflictIdsBySession };
      // Round-33b F5: a session-LESS item with no active session cannot be
      // attributed to any parked bucket — it is a GLOBAL conflict (the
      // queue is global). Keep a global banner list for that case instead
      // of silently dropping the id (which hid the banner entirely).
      let globalIds: string[] | null = null;
      if (outcome.conflictRecords.length > 0) {
        for (const record of outcome.conflictRecords) {
          const owner = record.sessionName ?? active;
          if (!owner) {
            globalIds = Array.from(new Set([...(globalIds ?? []), record.id]));
            continue;
          }
          const existing = parkedMap[owner] ?? [];
          parkedMap[owner] = Array.from(new Set([...existing, record.id]));
        }
      } else if (outcome.remaining === 0) {
        // Fully drained: clear everything — every conflicted id has left
        // the durable queue, so NO session (active or parked) and no
        // global bucket can still have a live conflict. Round-33b F5: the
        // old active-only clear leaked ids for other sessions after a drain.
        for (const session of Object.keys(parkedMap)) parkedMap[session] = null;
        globalIds = [];
      }
      // Active session: show its parked ids. No active session: show the
      // global (session-less) conflicts — there is nothing to scope to.
      const activeIds = active ? (parkedMap[active] ?? []) : (globalIds ?? []);
      // Round-34 F1: persist the global bucket in state (it was a local
      // variable before — global conflicts vanished after this setState
      // and were never rebuilt by refreshPendingMutations).
      set({
        lastReplay: outcome,
        pendingMutations: outcome.remaining,
        conflictIds: activeIds,
        globalConflictIds: globalIds ?? [],
        conflictIdsBySession: parkedMap,
      });
      return outcome;
    })();
    queueReplayInFlight = task;
    try {
      return await task;
    } finally {
      if (queueReplayInFlight === task) queueReplayInFlight = null;
    }
  },

  requeueConflict: async (id: string, ifMatch?: RevisionToken) => {
    // Round-31 F40 (round-32b F9/F10/F11): re-queue via the queue mutex;
    // storage failures from the durable persist PROPAGATE to the caller
    // (the banner keeps its action-in-progress state until this resolves).
    await requeueConflicted(id, ifMatch);
    // Round-32b F10 (round-34 F1): update the ACTIVE session's ids AND the
    // global bucket — a session-less conflict requeued from the global
    // banner must leave the banner instead of lingering forever.
    const state = get();
    const remaining = state.conflictIds.filter(existing => existing !== id);
    const remainingGlobal = state.globalConflictIds.filter(existing => existing !== id);
    const active = state.activeSessionName;
    set({
      conflictIds: remaining,
      globalConflictIds: remainingGlobal,
      conflictIdsBySession: active
        ? { ...state.conflictIdsBySession, [active]: remaining.length > 0 ? remaining : null }
        : state.conflictIdsBySession,
    });
    await refreshPendingMutations();
  },

  discardConflict: async (id: string) => {
    // Round-31 F40 (round-32b F9/F10/F11): discard via the queue mutex;
    // durable persist failures propagate so the UI can retry.
    await discardConflicted(id);
    // Round-34 F1: same global-bucket handling as requeueConflict.
    const state = get();
    const remaining = state.conflictIds.filter(existing => existing !== id);
    const remainingGlobal = state.globalConflictIds.filter(existing => existing !== id);
    const active = state.activeSessionName;
    set({
      conflictIds: remaining,
      globalConflictIds: remainingGlobal,
      conflictIdsBySession: active
        ? { ...state.conflictIdsBySession, [active]: remaining.length > 0 ? remaining : null }
        : state.conflictIdsBySession,
    });
    await refreshPendingMutations();
  },

  autoReconnect: async () => {
    // MUCLI_MOBILE_RECONNECT_YOLO_V1: non-destructive reconnect. A temporary
    // Wi-Fi/VPN/background outage must not erase a known-good remote host or
    // eject the user from a server-side session that is still running.
    if (reconnectInFlight) return reconnectInFlight;

    const task = (async () => {
      const state = get();
      const baseUrl = state.baseUrl.replace(/\/$/, '');
      if (!baseUrl) return;

      const url = baseUrl + '/healthz';
      const MAX_ATTEMPTS = 3;
      const BACKOFF_MS = 1_500;

      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10_000);
        try {
          const resp = await fetch(url, { method: 'GET', signal: controller.signal });
          if (resp.ok) {
            if (!get().isConnected) {
              set({ isConnected: true });
              await get().saveToStorage();
            }
            // G5 (§3.6): reconnect success path replays the offline queue.
            void get().replayPending();
            return;
          }
        } catch {
          // Retry. Existing connection state remains intact on failure.
        } finally {
          clearTimeout(timeout);
        }
        if (attempt < MAX_ATTEMPTS) {
          await new Promise(resolve => setTimeout(resolve, BACKOFF_MS * attempt));
        }
      }
      // Definitive probe failure after all attempts: reflect reality. SSE and
      // foreground health probes will set it true again on recovery.
      if (get().isConnected) {
        set({ isConnected: false });
      }
    })();

    reconnectInFlight = task;
    try {
      await task;
    } finally {
      if (reconnectInFlight === task) reconnectInFlight = null;
    }
  },
}));