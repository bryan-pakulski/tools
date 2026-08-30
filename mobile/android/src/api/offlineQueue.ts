/**
 * MUCLI_MOBILE_OFFLINE_QUEUE_V1 — G5 (cross-surface design doc §3.6).
 *
 * Outbound queue for mutations attempted while the mobile client is
 * offline: chat sends, session-variable toggles (e.g. yolo), and provider
 * switches. Items persist in AsyncStorage, replay in FIFO order on
 * reconnect, and each replayed mutation carries `If-Match` (§3.1 session
 * revision CAS) when the revision was captured — a 409 means the server
 * moved on: the item is marked CONFLICTED and the replay halts (round-31
 * F38/F40) so the user decides re-queue vs discard instead of stale state
 * silently overwriting or silently dropping.
 *
 * Queue discipline: cap 50 items, 24h TTL, FIFO, user-visible count badge.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { api, ApiError } from './client';

export const QUEUE_STORAGE_KEY = 'MUCLI_OFFLINE_QUEUE_V1';
export const QUEUE_MAX = 50;
export const QUEUE_TTL_MS = 24 * 60 * 60 * 1000;

export type QueuedKind = 'chat_send' | 'set_variable' | 'provider_switch';

/**
 * Round-32b F4: session revision token, lossless end-to-end.
 *
 * The server clamps revisions above 2^53-1 to their decimal string form
 * (utils/revision.js_safe_revision) because JS numbers lose precision
 * past Number.MAX_SAFE_INTEGER — 9007199254740993 parses to ...992 and a
 * rounded If-Match would CAS-mismatch forever. Tokens therefore travel as
 * `number | string` and are NEVER converted with Number()/parseInt:
 * strings pass through verbatim into If-Match. Only numeric strings
 * (canonical decimal form, no sign/whitespace) are accepted as tokens.
 */
export type RevisionToken = number | string;

export function revisionToken(raw: unknown): RevisionToken | null {
  if (typeof raw === 'number') {
    // Round-34 F2: Number.isSafeInteger, not Number.isInteger — an unsafe
    // integer-valued number has ALREADY lost precision at JSON.parse time
    // (9007199254740993 parses to ...992), so accepting it would send a
    // wrong If-Match and CAS-mismatch forever. Revisions beyond 2^53-1
    // must arrive as canonical decimal strings.
    return Number.isSafeInteger(raw) && raw >= 0 ? raw : null;
  }
  if (typeof raw === 'string') {
    return /^(0|[1-9][0-9]*)$/.test(raw) ? raw : null;
  }
  return null;
}

export interface QueuedItem {
  id: string;
  kind: QueuedKind;
  /** Epoch ms when the mutation was attempted offline. */
  enqueuedAt: number;
  /** Session the mutation targets (chat sends are session-scoped). */
  sessionName?: string;
  /** Kind-specific request payload, ready to replay verbatim. */
  payload: Record<string, unknown>;
  /** §3.1 session revision token captured at enqueue time, sent as If-Match. */
  ifMatch?: RevisionToken;
  /**
   * Round-31 F38/F40: item hit a 409 (stale If-Match). Excluded from
   * replay until the user re-queues (clears flag) or discards (removes).
   */
  conflicted?: boolean;
}

export interface ReplayOutcome {
  applied: number;
  conflicts: number;
  failed: number;
  /** Items still queued after the replay stopped. */
  remaining: number;
  /**
   * Round-31 F38/F40: ids of items marked conflicted by this replay.
   * UI reads this to show the conflict banner with re-queue/discard
   * actions. Empty when nothing conflicted.
   *
   * Round-33b F5: conflicts can also carry the item's TARGET session —
   * see conflictRecords.
   */
  conflictIds: string[];
  /**
   * Round-33b F5: conflict attribution records — BOTH the parked item id
   * and the sessionName the item targets (undefined for session-less
   * items). The store derives conflictIdsBySession from these records
   * (falling back to the active session only when sessionName is absent),
   * so a session switch DURING a replay can no longer park the id under
   * the wrong session.
   */
  conflictRecords: Array<{ id: string; sessionName?: string }>;
  /**
   * Round-32b F9: a durable persist failed mid-replay — the replay
   * HALTED (no further sends) and the just-applied/flagged state was NOT
   * durably recorded, so the outcome must not be treated as an ack. The
   * caller surfaces this (badge stays, next replay retries).
   */
  persistFailed?: true;
}

function makeId(): string {
  return `q_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function isQueuedItem(value: unknown): value is QueuedItem {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Partial<QueuedItem>;
  return (
    typeof item.id === 'string' &&
    typeof item.kind === 'string' &&
    (item.kind === 'chat_send' || item.kind === 'set_variable' || item.kind === 'provider_switch') &&
    typeof item.enqueuedAt === 'number' &&
    typeof item.payload === 'object' &&
    item.payload !== null
  );
}

/** Read + prune (TTL) + cap the persisted queue. Never throws. */
export async function loadQueue(): Promise<QueuedItem[]> {
  return withQueueMutex(() => loadQueueUnlocked());
}

/**
 * Round-33b F8: loadQueue validates each persisted item's ifMatch through
 * revisionToken() — corrupted/foreign JSON must never reach the replay
 * path, where a bogus token would ride out as `If-Match: [object Object]`
 * or a precision-rounded number. Invalid tokens are DROPPED (unguarded
 * item — the server decides), matching the requeue default.
 */
function normalizeItem(item: QueuedItem): QueuedItem | null {
  if (item.ifMatch === undefined) return item;
  const token = revisionToken(item.ifMatch);
  if (token === null) return { ...item, ifMatch: undefined };
  return token === item.ifMatch ? item : { ...item, ifMatch: token };
}

async function loadQueueUnlocked(): Promise<QueuedItem[]> {
  try {
    const raw = await AsyncStorage.getItem(QUEUE_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const now = Date.now();
    const live = parsed.filter(isQueuedItem).filter(item => now - item.enqueuedAt < QUEUE_TTL_MS).map(normalizeItem) as QueuedItem[];
    if (live.length > QUEUE_MAX) live.splice(0, live.length - QUEUE_MAX);
    return live;
  } catch {
    return [];
  }
}

export async function persistQueue(items: QueuedItem[]): Promise<void> {
  try {
    await AsyncStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(items));
  } catch {
    // Storage full/unavailable: queue becomes best-effort in-memory only.
  }
}

/**
 * Round-32b F9: durable persist contract. persistQueue() is best-effort
 * (swallows storage failures) — fine for TTL pruning, but the replay loop
 * MUST know whether an acknowledged item was actually durably recorded:
 * a crash after an unrecorded removal would re-send an applied chat turn.
 * Throws on storage failure so callers can halt instead of acking.
 */
async function persistQueueDurable(items: QueuedItem[]): Promise<void> {
  try {
    await AsyncStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(items));
  } catch (err) {
    throw err instanceof Error ? err : new Error(String(err));
  }
}

export async function queueCount(): Promise<number> {
  return (await loadQueue()).length;
}

/** FIFO enqueue with cap enforcement (oldest dropped beyond QUEUE_MAX).
 * Round-33b F4: runs under the queue mutex with the unlocked primitive —
 * an enqueue racing a replay can no longer interleave a load/modify/persist
 * cycle between the replay's read and write.
 * Round-33b F8: an externally supplied ifMatch is normalized through
 * revisionToken() — an invalid token is dropped (undefined), never
 * persisted, so no bogus If-Match can ever ride out to the server. */
export async function enqueueOffline(
  kind: QueuedKind,
  payload: Record<string, unknown>,
  options: { sessionName?: string; ifMatch?: RevisionToken } = {},
): Promise<QueuedItem> {
  return withQueueMutex(async () =>
    enqueueOfflineUnlocked(kind, payload, options),
  );
}

/** Mutex-free enqueue primitive (F4): replay uses this under its own hold. */
async function enqueueOfflineUnlocked(
  kind: QueuedKind,
  payload: Record<string, unknown>,
  options: { sessionName?: string; ifMatch?: RevisionToken } = {},
): Promise<QueuedItem> {
  const queue = await loadQueueUnlocked();
  const normalizedIfMatch =
    options.ifMatch === undefined ? undefined : revisionToken(options.ifMatch) ?? undefined;
  const item: QueuedItem = {
    id: makeId(),
    kind,
    enqueuedAt: Date.now(),
    sessionName: options.sessionName,
    payload,
    ...(normalizedIfMatch !== undefined ? { ifMatch: normalizedIfMatch } : {}),
  };
  queue.push(item);
  if (queue.length > QUEUE_MAX) queue.splice(0, queue.length - QUEUE_MAX);
  await persistQueue(queue);
  return item;
}

/** Round-33b F4: single remove runs under the queue mutex (F11 semantics
 * already held for requeue/discard; dequeue joins the same discipline). */
export async function dequeueOffline(id: string): Promise<void> {
  await withQueueMutex(async () => {
    const queue = await loadQueueUnlocked();
    const next = queue.filter(item => item.id !== id);
    if (next.length !== queue.length) await persistQueue(next);
  });
}

/** Round-33b F4: full clear under the queue mutex. */
export async function clearQueue(): Promise<void> {
  await withQueueMutex(async () => {
    await persistQueue([]);
  });
}

/** Drop everything older than the TTL (badge refresh path).
 * Round-33b F4: load→persist refresh runs under the queue mutex so it
 * cannot interleave with a concurrent replay or enqueue. */
export async function pruneQueue(): Promise<number> {
  return withQueueMutex(async () => {
    const before = await loadQueueUnlocked();
    await persistQueue(before);
    return before.length;
  });
}

/**
 * Round-31 F38 (updated round-32b F11): clear the conflicted flag on an
 * item so the next replay retries it. Single-flight via the queue mutex
 * so a double-tap cannot run two overlapping load/modify/persist cycles.
 *
 * If-Match semantics (round-32b F11): a provided token replaces the CAS
 * guard; when NO fresh token is available the caller must pass
 * `{ keepIfMatch: true }` to retry against the ORIGINAL captured token —
 * otherwise (default) the guard is stripped (unguarded single-item retry).
 * Callers that have no token and must not strip the guard simply don't
 * call this — the item stays parked (see ConflictBanner).
 */
export async function requeueConflicted(
  id: string,
  ifMatch?: RevisionToken,
  opts: { keepIfMatch?: boolean } = {},
): Promise<void> {
  await withQueueMutex(async () => {
    // Round-33b F4: load the queue via the UNLOCKED primitive — this task
    // already holds the mutex; calling the public loadQueue() here would
    // enqueue a nested mutex task and deadlock the chain.
    const queue = await loadQueueUnlocked();
    const next = queue.map(item => {
      if (item.id !== id) return item;
      const updated: QueuedItem = { ...item };
      delete updated.conflicted;
      if (ifMatch !== undefined) {
        // Round-34 F3: normalize the replacement through revisionToken() —
        // the same guard enqueue/load/store use (round-33b F8). An invalid
        // token (object, decorated string, unsafe number) DROPS the guard
        // instead of persisting a bogus If-Match, matching the enqueue
        // path's invalid-means-unguarded semantics.
        const token = revisionToken(ifMatch);
        if (token !== null) updated.ifMatch = token;
        else delete updated.ifMatch;
      } else if (!opts.keepIfMatch) {
        delete updated.ifMatch;
      }
      return updated;
    });
    if (next.some((item, index) => item !== queue[index])) await persistQueueDurable(next);
  });
}

/**
 * Round-31 F40 (updated round-32b F9/F11): user chose to drop a conflicted
 * mutation permanently. Durable persist (throws on storage failure so the
 * store action surfaces it) and single-flight via the queue mutex.
 */
export async function discardConflicted(id: string): Promise<void> {
  await withQueueMutex(async () => {
    // Round-33b F4: UNLOCKED primitive under an already-held mutex (see
    // requeueConflicted note).
    const queue = await loadQueueUnlocked();
    const next = queue.filter(item => item.id !== id);
    if (next.length !== queue.length) await persistQueueDurable(next);
  });
}

/**
 * Round-31 F38: count of conflicted items currently in the queue. The
 * replay halts on the first conflict, so at most one item is newly
 * conflicted per replay — but a user can accumulate several across
 * replays by re-queueing into further conflicts.
 */
export async function conflictedCount(): Promise<number> {
  const queue = await loadQueue();
  return queue.filter(item => item.conflicted === true).length;
}

export interface ReplayExecutorResult {
  status: 'applied' | 'conflict' | 'failed';
}

export type ReplayExecutor = (item: QueuedItem) => Promise<ReplayExecutorResult>;

/** Default executors use the shared api client. */
async function defaultExecutor(item: QueuedItem): Promise<ReplayExecutorResult> {
  const headers: Record<string, string> = {};
  if (item.ifMatch !== undefined) headers['If-Match'] = String(item.ifMatch);
  try {
    if (item.kind === 'chat_send') {
      await api.post('/api/chat/send', item.payload, { headers });
    } else if (item.kind === 'set_variable') {
      const key = String(item.payload.key ?? '');
      const { key: _drop, ...body } = item.payload;
      await api.post(`/api/variables/${encodeURIComponent(key)}`, body, { headers, query: { session_name: item.sessionName } });
    } else {
      await api.put('/api/providers/switch', item.payload, { headers });
    }
    return { status: 'applied' };
  } catch (err) {
    if (err instanceof ApiError && (err.status === 409 || err.status === 412)) {
      return { status: 'conflict' };
    }
    throw err;
  }
}

/**
 * Round-13 F10: every queue mutation goes through this promise-chain mutex
 * so a concurrent enqueue cannot be lost between replay's load and persist
 * (AsyncStorage reads/writes are independent read-modify-write operations).
 */
let queueMutation: Promise<unknown> = Promise.resolve();

function withQueueMutex<T>(task: () => Promise<T>): Promise<T> {
  const run = queueMutation.then(task, task);
  // Keep the chain alive regardless of task outcome.
  queueMutation = run.catch(() => undefined);
  return run;
}

/**
 * Replay the queue in FIFO order.
 *
 * Round-31 F38 semantics change: a conflict (409/412 — server state moved
 * past the captured If-Match) now HALTS the replay and marks the item
 * conflicted. Everything after the conflicted item stays queued untouched:
 * later items were built on the same (now stale) session view, so firing
 * them at the server would either duplicate work or produce more stale
 * writes. The user resolves via the conflict banner (re-queue with a
 * fresh revision, or discard). Any other failure keeps the failed item
 * plus the remainder queued for the next reconnect.
 *
 * Round-13 F9 preserved: each acknowledged result is durably persisted
 * BEFORE the next item is sent — a crash mid-replay can never replay an
 * already-applied chat send (which would duplicate a user turn).
 */
export async function replayQueue(executor: ReplayExecutor = defaultExecutor): Promise<ReplayOutcome> {
  return withQueueMutex(async () => {
    // Round-33b F4: UNLOCKED primitive under an already-held mutex.
    const queue = await loadQueueUnlocked();
    const outcome: ReplayOutcome = { applied: 0, conflicts: 0, failed: 0, remaining: 0, conflictIds: [], conflictRecords: [] };

    for (let index = 0; index < queue.length; index++) {
      const item = queue[index];
      if (item.conflicted === true) {
        // Round-32b F8: report the parked head so the store can keep the
        // banner visible (an empty list would look like "nothing wrong").
        // Conflicted items are parked, not replayed. Halt here: FIFO order
        // means everything after is blocked behind an unresolved item.
        // Round-33b F5: the record carries the item's target session.
        outcome.conflictIds = [item.id];
        outcome.conflictRecords = [{ id: item.id, sessionName: item.sessionName }];
        outcome.remaining = queue.length - index;
        return outcome;
      }
      let status: ReplayExecutorResult['status'];
      try {
        status = (await executor(item)).status;
      } catch {
        status = 'failed';
      }
      if (status === 'applied') {
        outcome.applied += 1;
        // Round-13 F9 / round-32b: durably record the removal BEFORE the
        // next item is sent. On storage failure the replay HALTS with
        // persistFailed: no further sends, and the outcome is not an ack
        // (the item is still in durable storage, so the next replay
        // retries it). The failure is surfaced via the outcome rather
        // than thrown so a single bad persist cannot crash the caller.
        try {
          await persistQueueDurable(queue.slice(index + 1));
        } catch {
          outcome.persistFailed = true;
          outcome.remaining = queue.length - index; // item NOT durably removed
          return outcome;
        }
        // Continue: next item.
      } else if (status === 'conflict') {
        outcome.conflicts += 1;
        outcome.conflictIds = [item.id];
        // Round-33b F5: the record carries the item's target session so
        // the store can attribute the conflict even if the user switches
        // sessions while the replay's awaited executor is in flight.
        outcome.conflictRecords = [{ id: item.id, sessionName: item.sessionName }];
        // Round-31 F38/F40: mark conflicted + halt. Persist the flagged
        // item BEFORE stopping so the conflict state survives a crash.
        item.conflicted = true;
        try {
          await persistQueueDurable(queue.slice(index));
        } catch {
          // The flag never reached storage — revert the in-memory mark so
          // storage and memory agree (no false ack), and surface the
          // durable failure in the outcome.
          delete item.conflicted;
          outcome.persistFailed = true;
        }
        outcome.remaining = queue.length - index;
        return outcome;
      } else {
        // failed: keep this item AND everything after it queued.
        outcome.failed += 1;
        try {
          await persistQueueDurable(queue.slice(index));
        } catch {
          outcome.persistFailed = true;
        }
        outcome.remaining = queue.length - index;
        return outcome;
      }
    }
    outcome.remaining = 0;
    return outcome;
  });
}