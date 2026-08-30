import { useCallback, useEffect, useRef, useState } from 'react';
import { chatApi } from '../api/chat';
import {
  sessionsApi,
  type SessionHistoryResponse,
  type SessionHistoryTurn,
} from '../api/sessions';
import { subscribeToEvents, type SSESubscription } from '../api/sse';
import { queueIfOffline, useConnectionStore } from '../store/connection';
import { revisionToken } from '../api/offlineQueue';
import type { ArtifactDescriptor } from '../api/artifacts';
import type { AttachmentDescriptor } from '../api/attachments';

const SESSION_POLL_MS = 5000;
// Initial window size — the latest N turns to load on session open.
// Kept small to avoid OOM on mid-range Android; older turns load on demand
// via loadOlderHistory() (sliding window) when the user scrolls up.
const MOBILE_HISTORY_TURN_LIMIT = 80;
const MOBILE_HISTORY_CHECKPOINT_BATCH = 5;
const MOBILE_HISTORY_CHECKPOINT_SCAN_PAGES = 6;

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'visualization' | 'subagent_panel' | 'collapse';
  text: string;
  turnId?: string;
  /** Durable server turn index; user messages sharing it are one checkpoint. */
  historyIndex?: number;
  streaming?: boolean;
  origin?: 'history' | 'local' | 'stream';
  artifact?: ArtifactDescriptor;
  subagents?: LiveSubagent[];
  subagentBatchId?: string;
  attachments?: AttachmentDescriptor[];
  childTurns?: ChatMessage[];
  collapseCount?: number;
  collapseElapsed?: string;
  collapseTokens?: string;
  collapseOpen?: boolean;
  collapseGroupKey?: string;
  collapseLive?: boolean;
  collapseUserId?: string;
  handoff?: 'entering' | 'leaving';
}

export interface LiveSubagentAction {
  seq: number;
  tool: string;
  detail: string;
  status: string;
  elapsed: number;
  at: number;
}

export interface LiveSubagent {
  task_id: string;
  batch_id: string;
  task: string;
  title: string;
  depth: number;
  model: string;
  specialist_key: string;
  status: string;
  tool_count: number;
  last_tool: string | null;
  elapsed: number;
  context_pct: number;
  iter: number;
  max_iter: number;
  tokens_in: number;
  summary: string;
  error: string | null;
  actions: LiveSubagentAction[];
  observed_at: number;
}

type StreamEvent = { kind: string; [key: string]: unknown };
type ActiveSessionState = { active?: boolean; is_busy?: boolean; external_active?: boolean; external_last_at?: number; revision?: number | string };

function asVisualization(value: unknown): ArtifactDescriptor | null {
  if (!value || typeof value !== 'object') return null;
  const artifact = value as ArtifactDescriptor;
  return artifact.kind === 'visualization' && typeof artifact.artifact_id === 'string'
    ? artifact
    : null;
}

export function historyToMessages(turns: SessionHistoryTurn[]): ChatMessage[] {
  // MUCLI_MOBILE_VISUALIZATION_HISTORY_V1: tool-result turns can contain durable visualization
  // descriptors even when their role is `tool`. Preserve those cards while
  // continuing to render ordinary text only for user/assistant turns.
  return turns.flatMap(turn => {
    const messageRole = turn.role === 'user' || turn.role === 'assistant'
      ? turn.role as 'user' | 'assistant'
      : null;
    const messages: ChatMessage[] = [];
    let pendingText: string[] = [];
    const pendingAttachments: AttachmentDescriptor[] = [];
    let partIndex = 0;
    const flushText = () => {
      const text = pendingText.join('\n\n').trim();
      pendingText = [];
      if (!messageRole || !text) return;
      messages.push({
        id: `history-${turn.index}-${partIndex++}`,
        role: messageRole,
        text,
        historyIndex: turn.index,
        streaming: false,
        origin: 'history',
        attachments: messageRole === 'user' ? [...pendingAttachments] : undefined,
      });
    };

    for (const part of turn.parts) {
      if (part.type === 'text' && typeof part.text === 'string') {
        if (messageRole) pendingText.push(String(part.text));
        continue;
      }
      if (
        messageRole === 'user'
        && part.type === 'attachment'
        && part.attachment
        && typeof part.attachment === 'object'
      ) {
        pendingAttachments.push(part.attachment as AttachmentDescriptor);
        continue;
      }
      const artifact = asVisualization(part.artifact);
      if (part.type === 'subagent_panel' && Array.isArray(part.agents)) {
        flushText();
        const agents = part.agents
          .map(value => subagentFromEvent(value as StreamEvent))
          .filter((value): value is LiveSubagent => Boolean(value));
        if (agents.length > 0) {
          const batchId = typeof part.batch_id === 'string'
            ? part.batch_id
            : (agents[0].batch_id || agents[0].task_id);
          messages.push({
            id: `subagent-${batchId}`,
            role: 'subagent_panel',
            text: '',
            streaming: false,
            origin: 'history',
            subagents: agents,
            subagentBatchId: batchId,
          });
        }
        continue;
      }
      if (!artifact) continue;
      flushText();
      messages.push({
        id: `visualization-${artifact.artifact_id}`,
        role: 'visualization',
        text: '',
        streaming: false,
        origin: 'history',
        artifact,
      });
    }
    flushText();
    if (messageRole === 'user' && pendingAttachments.length > 0 && messages.length === 0) {
      messages.push({
        id: `history-${turn.index}-attachments`,
        role: 'user',
        text: '',
        historyIndex: turn.index,
        streaming: false,
        origin: 'history',
        attachments: [...pendingAttachments],
      });
    }
    return messages;
  });
}

async function historyToMessagesCooperatively(
  turns: SessionHistoryTurn[],
  signal?: AbortSignal,
): Promise<ChatMessage[]> {
  const messages: ChatMessage[] = [];
  const batchSize = 8;
  for (let start = 0; start < turns.length; start += batchSize) {
    if (signal?.aborted) return [];
    messages.push(...historyToMessages(turns.slice(start, start + batchSize)));
    if (start + batchSize < turns.length) {
      // Let gestures, navigation, and incoming SSE deltas run between
      // conversion batches. A large history response must not monopolize
      // React Native's JS thread even though FlatList virtualizes its cells.
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
  }
  return messages;
}

/**
 * Fetch a backward history window containing five distinct user prompts.
 *
 * The server's limit is expressed in raw provider/tool turns, so one page can
 * legitimately contain no user turns. Keep scanning bounded pages until the
 * rail has a useful checkpoint batch or the start of history is reached.
 */
export async function fetchCheckpointHistory(
  sessionName: string,
  options: {
    beforeIndex?: number;
    signal?: AbortSignal;
    timeoutMs?: number;
  } = {},
): Promise<SessionHistoryResponse> {
  const turnsByIndex = new Map<number, SessionHistoryTurn>();
  const checkpointIndexes = new Set<number>();
  let cursor = options.beforeIndex;
  let firstResponse: SessionHistoryResponse | null = null;
  let oldestResponse: SessionHistoryResponse | null = null;

  for (let page = 0; page < MOBILE_HISTORY_CHECKPOINT_SCAN_PAGES; page += 1) {
    const remaining = Math.max(1, MOBILE_HISTORY_CHECKPOINT_BATCH - checkpointIndexes.size);
    const response = await sessionsApi.getHistory(sessionName, {
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      limitTurns: MOBILE_HISTORY_TURN_LIMIT,
      beforeIndex: cursor,
      checkpointCount: remaining,
    });
    if (!firstResponse) firstResponse = response;
    oldestResponse = response;
    for (const turn of response.turns || []) {
      turnsByIndex.set(turn.index, turn);
      if (turn.role === 'user') checkpointIndexes.add(turn.index);
    }

    const nextCursor = response.start_index;
    if (
      checkpointIndexes.size >= MOBILE_HISTORY_CHECKPOINT_BATCH
      || !response.has_more
      || nextCursor == null
      || nextCursor <= 0
      || nextCursor === cursor
    ) break;
    cursor = nextCursor;
  }

  const base = firstResponse || {
    name: sessionName,
    turns: [],
    start_index: options.beforeIndex ?? 0,
    has_more: false,
  };
  return {
    ...base,
    turns: [...turnsByIndex.values()].sort((left, right) => left.index - right.index),
    start_index: oldestResponse?.start_index ?? base.start_index,
    has_more: oldestResponse?.has_more ?? false,
  };
}

function eventBelongsToSession(event: StreamEvent, activeSessionName: string | null): boolean {
  const eventSession = typeof event.session_name === 'string' ? event.session_name : null;
  return !eventSession || !activeSessionName || eventSession === activeSessionName;
}

function numberOr(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function isSubagentActive(status: string): boolean {
  return status === 'running' || status === 'stuck' || status === 'stall';
}

function mergeSubagentActions(
  existing: LiveSubagentAction[],
  event: StreamEvent,
): LiveSubagentAction[] {
  const incoming: Record<string, unknown>[] = [];
  if (Array.isArray(event.actions)) {
    incoming.push(...event.actions.filter(value => value && typeof value === 'object') as Record<string, unknown>[]);
  }
  if (event.action && typeof event.action === 'object') {
    incoming.push(event.action as Record<string, unknown>);
  }
  if (incoming.length === 0) return existing;
  const merged = new Map(existing.map(action => [action.seq, action]));
  for (const raw of incoming) {
    const seq = numberOr(raw.seq, 0);
    if (seq <= 0) continue;
    const previous = merged.get(seq);
    merged.set(seq, {
      seq,
      tool: typeof raw.tool === 'string' ? raw.tool : (previous?.tool || 'tool'),
      detail: typeof raw.detail === 'string' ? raw.detail : (previous?.detail || ''),
      status: typeof raw.status === 'string' ? raw.status : (previous?.status || 'running'),
      elapsed: numberOr(raw.elapsed, previous?.elapsed || 0),
      at: numberOr(raw.at, previous?.at || Date.now() / 1000),
    });
  }
  return [...merged.values()].sort((left, right) => left.seq - right.seq).slice(-100);
}

function subagentFromEvent(
  event: StreamEvent,
  existing?: LiveSubagent,
): LiveSubagent | null {
  const taskId = typeof event.task_id === 'string' ? event.task_id : existing?.task_id;
  if (!taskId) return null;
  const tokens = event.tokens && typeof event.tokens === 'object'
    ? event.tokens as Record<string, unknown>
    : null;
  const readString = (key: string, fallback: string) => (
    typeof event[key] === 'string' ? String(event[key]) : fallback
  );
  const readNullableString = (key: string, fallback: string | null) => (
    event[key] === null ? null : (typeof event[key] === 'string' ? String(event[key]) : fallback)
  );
  return {
    task_id: taskId,
    batch_id: readString('batch_id', existing?.batch_id || ''),
    task: readString('task', existing?.task || ''),
    title: readString('title', existing?.title || ''),
    depth: numberOr(event.depth, existing?.depth || 1),
    model: readString('model', existing?.model || ''),
    specialist_key: readString('specialist_key', existing?.specialist_key || ''),
    status: readString('status', existing?.status || 'running'),
    tool_count: numberOr(event.tool_count ?? event.tool_calls, existing?.tool_count || 0),
    last_tool: readNullableString('last_tool', existing?.last_tool || null),
    elapsed: numberOr(event.elapsed, existing?.elapsed || 0),
    context_pct: numberOr(event.context_pct, existing?.context_pct || 0),
    iter: numberOr(event.iter, existing?.iter || 0),
    max_iter: numberOr(event.max_iter, existing?.max_iter || 0),
    tokens_in: numberOr(event.tokens_in ?? tokens?.['in'], existing?.tokens_in || 0),
    summary: readString('summary', existing?.summary || ''),
    error: readNullableString('error', existing?.error || null),
    actions: mergeSubagentActions(existing?.actions || [], event),
    observed_at: Date.now(),
  };
}

// Collapse every completed exchange independently. Visualizations are timeline
// anchors: compact disclosures may form on either side, but an artifact card
// itself always remains top-level and in chronological order.
function flattenCollapsedMessages(messages: ChatMessage[]): ChatMessage[] {
  const flattened: ChatMessage[] = [];
  for (const message of messages) {
    if (message.role === 'collapse') {
      flattened.push(...flattenCollapsedMessages(message.childTurns || []));
    } else {
      flattened.push(message);
    }
  }
  return flattened;
}

function collapseGroupKey(user: ChatMessage, finalResponse: ChatMessage): string {
  return JSON.stringify([user.text, finalResponse.text]);
}

function isTimelineAnchor(message: ChatMessage): boolean {
  return message.role === 'visualization' || message.role === 'subagent_panel';
}

function upsertSubagentPanelMessage(
  messages: ChatMessage[],
  event: StreamEvent,
): ChatMessage[] {
  const taskId = typeof event.task_id === 'string' ? event.task_id : '';
  if (!taskId) return messages;
  const suppliedBatch = typeof event.batch_id === 'string' ? event.batch_id : '';
  let index = messages.findIndex(message => (
    message.role === 'subagent_panel'
    && (
      (suppliedBatch && message.subagentBatchId === suppliedBatch)
      || (message.subagents || []).some(agent => agent.task_id === taskId)
    )
  ));
  const currentPanel = index >= 0 ? messages[index] : null;
  const currentAgents = currentPanel?.subagents || [];
  const agentIndex = currentAgents.findIndex(agent => agent.task_id === taskId);
  const nextAgent = subagentFromEvent(event, agentIndex >= 0 ? currentAgents[agentIndex] : undefined);
  if (!nextAgent) return messages;
  const nextAgents = [...currentAgents];
  if (agentIndex >= 0) nextAgents[agentIndex] = nextAgent;
  else nextAgents.push(nextAgent);
  const batchId = suppliedBatch || currentPanel?.subagentBatchId || nextAgent.batch_id || taskId;
  const panel: ChatMessage = {
    id: currentPanel?.id || `subagent-${batchId}`,
    role: 'subagent_panel',
    text: '',
    streaming: nextAgents.some(agent => isSubagentActive(agent.status)),
    origin: currentPanel?.origin || 'stream',
    subagents: nextAgents,
    subagentBatchId: batchId,
  };
  const updated = [...messages];
  if (index >= 0) updated[index] = panel;
  else {
    index = updated.length;
    updated.push(panel);
  }
  return updated;
}

function appendCollapsedSegments(
  target: ChatMessage[],
  segmentSource: ChatMessage[],
  options: {
    idPrefix: string;
    groupKey: string;
    userId: string;
    live: boolean;
    previousOpen: Map<string, boolean>;
    defaultOpen: boolean;
  },
) {
  let segment: ChatMessage[] = [];
  let segmentIndex = 0;
  const flush = () => {
    if (segment.length === 0) return;
    const childTurns = segment;
    segment = [];
    const key = `${options.groupKey}:${segmentIndex}`;
    segmentIndex += 1;
    target.push({
      id: `${options.idPrefix}-${segmentIndex}`,
      role: 'collapse',
      text: '',
      childTurns,
      collapseCount: childTurns.length,
      collapseElapsed: '',
      collapseTokens: '',
      collapseOpen: options.previousOpen.get(key) ?? options.defaultOpen,
      collapseGroupKey: key,
      collapseLive: options.live,
      collapseUserId: options.userId,
    });
  };
  for (const message of segmentSource) {
    if (isTimelineAnchor(message)) {
      flush();
      target.push(message);
    } else {
      segment.push(message);
    }
  }
  flush();
}

function groupIntermediateTurns(
  messages: ChatMessage[],
  previous: ChatMessage[] = messages,
): ChatMessage[] {
  const source = flattenCollapsedMessages(messages);
  if (source.length < 3) return source;

  const previousOpen = new Map<string, boolean>();
  const previousLiveOpen = new Map<string, boolean>();
  for (const message of previous) {
    if (message.role === 'collapse' && message.collapseGroupKey) {
      previousOpen.set(message.collapseGroupKey, Boolean(message.collapseOpen));
    }
    if (message.role === 'collapse' && message.collapseLive && message.collapseUserId) {
      previousLiveOpen.set(message.collapseUserId, Boolean(message.collapseOpen));
    }
  }

  const grouped: ChatMessage[] = [];
  let index = 0;
  while (index < source.length) {
    const userMessage = source[index];
    if (userMessage.role !== 'user') {
      grouped.push(userMessage);
      index += 1;
      continue;
    }

    grouped.push(userMessage);
    let nextUserIndex = index + 1;
    while (nextUserIndex < source.length && source[nextUserIndex].role !== 'user') {
      nextUserIndex += 1;
    }

    const exchange = source.slice(index + 1, nextUserIndex);
    let finalOffset = -1;
    for (let offset = exchange.length - 1; offset >= 0; offset -= 1) {
      const candidate = exchange[offset];
      if (candidate.role === 'assistant' && !candidate.streaming) {
        finalOffset = offset;
        break;
      }
    }

    if (finalOffset > 0) {
      const finalResponse = exchange[finalOffset];
      const groupKey = collapseGroupKey(userMessage, finalResponse);
      appendCollapsedSegments(grouped, exchange.slice(0, finalOffset), {
        idPrefix: `collapse-${userMessage.id}-${finalResponse.id}`,
        groupKey,
        userId: userMessage.id,
        live: false,
        previousOpen,
        defaultOpen:
          previousOpen.get(groupKey)
          ?? previousLiveOpen.get(userMessage.id)
          ?? false,
      });
      grouped.push(finalResponse, ...exchange.slice(finalOffset + 1));
    } else {
      grouped.push(...exchange);
    }

    index = nextUserIndex;
  }
  return grouped;
}

// A completed response stays readable until replacement text exists. Once the
// successor arrives, fold only the content before it; visualization anchors
// remain top-level and keep their exact chronological position.
function foldLiveInterim(messages: ChatMessage[], currentTurnId: string): ChatMessage[] {
  let userIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user') {
      userIndex = index;
      break;
    }
  }
  if (userIndex < 0) return messages;

  const userMessage = messages[userIndex];
  const originalTail = messages.slice(userIndex + 1);
  const previousOpen = new Map<string, boolean>();
  for (const message of originalTail) {
    if (message.role === 'collapse' && message.collapseGroupKey) {
      previousOpen.set(message.collapseGroupKey, Boolean(message.collapseOpen));
    }
  }
  const tail = flattenCollapsedMessages(originalTail).map(message => ({
    ...message,
    handoff: undefined,
  }));
  let currentIndex = -1;
  for (let index = tail.length - 1; index >= 0; index -= 1) {
    const message = tail[index];
    if (message.role === 'assistant' && message.turnId === currentTurnId) {
      currentIndex = index;
      break;
    }
  }
  if (currentIndex < 0) return messages;

  const regrouped: ChatMessage[] = [];
  appendCollapsedSegments(regrouped, tail.slice(0, currentIndex), {
    idPrefix: `live-collapse-${userMessage.id}`,
    groupKey: `live:${userMessage.id}`,
    userId: userMessage.id,
    live: true,
    previousOpen,
    defaultOpen: false,
  });
  regrouped.push(...tail.slice(currentIndex));
  return [...messages.slice(0, userIndex + 1), ...regrouped];
}

// Pure whole-turn eviction helper retained for callers that need a bounded
// offline snapshot. Interactive checkpoint history intentionally does not
// use it: every exposed dot must remain a valid navigation target.
// Keeps the
// oldest `maxHistoryMessages` history-origin messages and drops the rest
// (newest first eviction); live/stream messages always survive. Returns the
// trimmed array plus the server index the forward cursor should point at
// (null when nothing was evicted). History ids are `history-{turnIndex}-...`
// so the newest surviving history message names the eviction boundary.
export function applyHistoryWindowEviction(
  messages: ChatMessage[],
  options: {
    maxHistoryMessages: number;
  },
): { messages: ChatMessage[]; forwardCursor: number | null } {
  const { maxHistoryMessages } = options;
  const parseTurnIndex = (message: ChatMessage): number | null => {
    const parsed = /^history-(\d+)-/.exec(message.id);
    return parsed ? Number(parsed[1]) : null;
  };

  const historyIndexes: number[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    if (messages[index].origin === 'history') historyIndexes.push(index);
  }
  if (historyIndexes.length <= maxHistoryMessages) {
    // Nothing evicted. Report the index after the newest retained turn.
    if (historyIndexes.length === 0) return { messages, forwardCursor: null };
    const lastTurn = parseTurnIndex(messages[historyIndexes[historyIndexes.length - 1]]);
    // Without a parseable id we can't know whether newer content exists;
    // null keeps any armed cursor as-is (caller decides).
    return { messages, forwardCursor: lastTurn === null ? null : lastTurn + 1 };
  }

  // Round-45 F8: evict WHOLE server turns, never a partial turn. The old
  // message-count slice could split a multi-part turn — half its parts
  // evicted, cursor set past the whole turn, and the evicted remainder
  // unrecoverable. Group history messages by parsed turn index and evict
  // the NEWEST complete turns until the kept history fits the budget
  // (a turn whose group crosses the budget is kept whole — it may push
  // the kept count slightly over, which is fine: the budget is a target).
  const groupsByTurn = new Map<number, number[]>();
  const turnOrder: number[] = [];
  for (const index of historyIndexes) {
    const turnIndex = parseTurnIndex(messages[index]);
    if (turnIndex === null) continue; // unparseable history ids are always kept
    if (!groupsByTurn.has(turnIndex)) {
      groupsByTurn.set(turnIndex, []);
      turnOrder.push(turnIndex);
    }
    groupsByTurn.get(turnIndex)!.push(index);
  }
  // Walk turns OLDEST-first, keeping complete turns while the count stays
  // within budget; the FIRST turn that would cross the budget starts the
  // evicted tail. A turn is never split — it is kept whole or evicted whole.
  let keptCount = 0;
  let evictFromOrder = turnOrder.length;
  for (let position = 0; position < turnOrder.length; position += 1) {
    const group = groupsByTurn.get(turnOrder[position])!;
    if (position > 0 && keptCount + group.length > maxHistoryMessages) {
      evictFromOrder = position;
      break;
    }
    keptCount += group.length;
    evictFromOrder = position + 1;
  }
  const evictedSet = new Set<number>();
  for (let position = evictFromOrder; position < turnOrder.length; position += 1) {
    for (const index of groupsByTurn.get(turnOrder[position])!) {
      evictedSet.add(index);
    }
  }
  // Everything NOT in evictedSet survives — live/stream messages were never
  // added to it, so they always survive.
  const result = messages.filter((_, index) => !evictedSet.has(index));

  // Forward cursor: server index AFTER the newest wholly-KEPT turn. Every
  // evicted turn is complete, so reloading from this cursor can never
  // re-fetch half a turn or skip evicted parts.
  const newestKeptOrder = evictFromOrder - 1;
  if (newestKeptOrder < 0) return { messages: result, forwardCursor: null };
  const newestKeptTurn = turnOrder[newestKeptOrder];
  return {
    messages: result,
    forwardCursor: newestKeptTurn == null ? null : newestKeptTurn + 1,
  };
}

function prepareForAssistantTurn(
  messages: ChatMessage[],
  turnId: string,
): ChatMessage[] {
  const retired = messages.map(message =>
    message.role === 'assistant'
    && message.streaming
    && message.turnId !== turnId
      ? { ...message, streaming: false }
      : message
  );
  return retired;
}

export function useChatSession(activeSessionName: string | null) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [waitingForFirstToken, setWaitingForFirstToken] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [activityLabel, setActivityLabel] = useState('Thinking');
  const [sseConnected, setSseConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnectKey, setReconnectKey] = useState(0);
  const [artifactRevision, setArtifactRevision] = useState(0);
  // MUCLI_SLIDING_WINDOW_V1: track whether older turns exist on the server
  // and whether a backward-pagination request is in flight. Mobile ChatScreen
  // triggers loadOlderHistory when the user scrolls near the top.
  const [hasMore, setHasMore] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);

  const subscriptionRef = useRef<SSESubscription | null>(null);
  const messageIdRef = useRef(0);
  const seenAssistantTurnsRef = useRef(new Set<string>());
  const messagesRef = useRef<ChatMessage[]>([]);
  const busyRef = useRef(false);
  const sseConnectedRef = useRef(false);
  const lastSessionRef = useRef<string | null>(null);
  const historyHydratedRef = useRef<string | null>(null);
  const historyRequestRef = useRef<{ sessionName: string; promise: Promise<void> } | null>(null);
  const historyAbortRef = useRef<AbortController | null>(null);
  const olderHistoryAbortRef = useRef<AbortController | null>(null);
  const stateAbortRef = useRef<AbortController | null>(null);
  const externalWriteAtRef = useRef(0);
  const completionProbeRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handoffTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Fallback: if history_refresh doesn't arrive within 3s after
  // turn_complete, force a history reload. Safety net for server
  // bugs or network issues that drop the event.
  const historyFallbackRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // MUCLI_SLIDING_WINDOW_V1: the absolute server index of the oldest turn
  // currently in the messages array. Used as the `before_index` cursor for
  // backward pagination. Reset on session change / full reload.
  const oldestLoadedIndexRef = useRef<number | null>(null);
  const loadingOlderRef = useRef(false);
  // Round-44 F5: coalesced streaming deltas — pending text + flush cadence.
  const pendingDeltaRef = useRef<{ turnId: string; text: string; first: boolean } | null>(null);
  const deltaFlushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    sseConnectedRef.current = sseConnected;
  }, [sseConnected]);

  const nextId = useCallback((prefix: string) => {
    messageIdRef.current += 1;
    return `${prefix}-${Date.now().toString(36)}-${messageIdRef.current}`;
  }, []);

  const appendUserMessage = useCallback((text: string, origin: 'local' | 'stream', attachments: AttachmentDescriptor[] = []) => {
    setMessages(current => {
      const last = current[current.length - 1];
      if (origin === 'stream' && last?.role === 'user' && last.text === text && last.origin === 'local') {
        const updated = [...current];
        updated[updated.length - 1] = { ...last, origin: 'stream' };
        return updated;
      }
      return [...current, {
        id: nextId('user'),
        role: 'user',
        text,
        streaming: false,
        origin,
        attachments,
      }];
    });
  }, [nextId]);

  const scheduleAssistantHandoff = useCallback((turnId: string) => {
    if (handoffTimerRef.current) clearTimeout(handoffTimerRef.current);
    handoffTimerRef.current = setTimeout(() => {
      handoffTimerRef.current = null;
      setMessages(current => foldLiveInterim(current, turnId));
    }, 260);
  }, []);

  // Round-44 F5: streaming deltas are coalesced. Each SSE delta used to
  // run prepareForAssistantTurn (full .map), a findIndex, and a full array
  // copy — O(loaded history) per token, with a new array identity per token
  // forcing FlatList reconciliation every frame. Deltas now accumulate in
  // a ref and flush to React state at a fixed 32ms cadence (and on turn
  // end), so a 200k-message session pays O(1) amortized per token instead
  // of O(n).
  const flushPendingDeltas = useCallback(() => {
    if (deltaFlushTimerRef.current) {
      clearTimeout(deltaFlushTimerRef.current);
      deltaFlushTimerRef.current = null;
    }
    const pending = pendingDeltaRef.current;
    if (!pending) return;
    pendingDeltaRef.current = null;
    const { turnId, text, first } = pending;
    setMessages(current => {
      const prepared = prepareForAssistantTurn(current, turnId);
      const index = prepared.findIndex(
        message => message.role === 'assistant' && message.turnId === turnId,
      );
      if (index >= 0) {
        const updated = [...prepared];
        updated[index] = {
          ...updated[index],
          text: updated[index].text + text,
          streaming: true,
          origin: 'stream',
          handoff: first ? 'entering' : updated[index].handoff,
        };
        return updated;
      }
      let lastUserIndex = -1;
      for (let scan = prepared.length - 1; scan >= 0; scan -= 1) {
        if (prepared[scan].role === 'user') { lastUserIndex = scan; break; }
      }
      const previousMarked = first
        ? prepared.map((message, messageIndex) => (
          messageIndex > lastUserIndex
          && message.role === 'assistant'
          && !message.streaming
          && message.text.trim().length > 0
            ? { ...message, handoff: 'leaving' as const }
            : message
        ))
        : prepared;
      return [...previousMarked, {
        id: nextId('assistant'),
        role: 'assistant',
        text,
        turnId,
        streaming: true,
        origin: 'stream',
        handoff: first ? 'entering' : undefined,
      }];
    });
    if (first) scheduleAssistantHandoff(turnId);
  }, [nextId, scheduleAssistantHandoff]);

  const appendAssistantDelta = useCallback((turnId: string, delta: string) => {
    if (!delta) return;
    const safeTurnId = turnId || 'active-turn';
    const firstDelta = !seenAssistantTurnsRef.current.has(safeTurnId);
    seenAssistantTurnsRef.current.add(safeTurnId);
    const pending = pendingDeltaRef.current;
    if (pending && pending.turnId === safeTurnId) {
      pending.text += delta;
    } else {
      if (pending) flushPendingDeltas();
      pendingDeltaRef.current = { turnId: safeTurnId, text: delta, first: firstDelta };
    }
    if (!deltaFlushTimerRef.current) {
      deltaFlushTimerRef.current = setTimeout(() => {
        deltaFlushTimerRef.current = null;
        flushPendingDeltas();
      }, 32);
    }
  }, [flushPendingDeltas]);

  const finalizeAssistant = useCallback((turnId: string) => {
    setMessages(current => current.map(message =>
      message.role === 'assistant' && message.turnId === turnId
        ? { ...message, streaming: false }
        : message,
    ));
  }, []);

  const upsertSubagent = useCallback((event: StreamEvent) => {
    setMessages(current => upsertSubagentPanelMessage(current, event));
  }, []);

  const replaceSubagentSnapshot = useCallback((event: StreamEvent) => {
    const children = Array.isArray(event.children)
      ? event.children.filter(value => value && typeof value === 'object') as StreamEvent[]
      : [];
    const active = children.filter(child => isSubagentActive(String(child.status || 'running')));
    if (active.length === 0) return;
    setMessages(current => active.reduce(
      (updated, child) => upsertSubagentPanelMessage(updated, child),
      current,
    ));
  }, []);

  const loadHistory = useCallback(async (preserveLive = true) => {
    if (!activeSessionName) {
      historyAbortRef.current?.abort();
      historyAbortRef.current = null;
      historyHydratedRef.current = null;
      setMessages([]);
      setHistoryLoading(false);
      return;
    }

    const inFlight = historyRequestRef.current;
    if (inFlight?.sessionName === activeSessionName) {
      await inFlight.promise;
      return;
    }

    const initialLoad = historyHydratedRef.current !== activeSessionName;
    if (initialLoad) setHistoryLoading(true);

    olderHistoryAbortRef.current?.abort();
    olderHistoryAbortRef.current = null;
    historyAbortRef.current?.abort();
    const controller = new AbortController();
    historyAbortRef.current = controller;
    const request = (async () => {
      try {
        const response = await fetchCheckpointHistory(activeSessionName, {
          signal: controller.signal,
          timeoutMs: 15_000,
        });
        if (controller.signal.aborted || lastSessionRef.current !== activeSessionName) return;
        const historyMessages = await historyToMessagesCooperatively(
          response.turns || [],
          controller.signal,
        );
        if (controller.signal.aborted || lastSessionRef.current !== activeSessionName) return;
        // Track the oldest loaded turn index for backward pagination.
        const oldestIdx = response.start_index ?? null;
        oldestLoadedIndexRef.current = oldestIdx;
        setHasMore(response.has_more ?? false);
        setMessages(current => {
          const hasLiveContent = current.some(message => message.origin !== 'history' || message.streaming);
          if (preserveLive && busyRef.current && hasLiveContent) return current;
          // Preserve existing message IDs where content matches so FlatList
          // keyExtractor returns stable keys → no cell remount → no scroll
          // jump. When a history_refresh arrives after a turn, the live
          // messages (origin: 'stream', id: 'assistant-xxx') are replaced by
          // history messages (id: 'history-N-N'). Without ID reuse, every
          // key changes → FlatList unmounts/remounts all cells → visible
          // "jumping up and down" effect.
          const currentFlat = flattenCollapsedMessages(current);
          // Fast path: identical history refresh → keep current array identity
          // so FlatList sees no change and does not re-render any cell.
          const unchanged = current.length === historyMessages.length
            && historyMessages.every((next, index) => {
              const prev = currentFlat[index];
              return !!prev
                && prev.role === next.role
                && prev.text === next.text;
            });
          if (unchanged) {
            return unchanged ? current : historyMessages;
          }
          const merged = historyMessages.map((next, index) => {
            const prev = currentFlat[index];
            if (!prev) return next;
            const contentMatches = prev.role === next.role
              && prev.text === next.text
              && (next.role === 'visualization'
                ? prev.artifact?.artifact_id === next.artifact?.artifact_id
                : (next.role === 'subagent_panel'
                  ? prev.subagentBatchId === next.subagentBatchId
                  : true));
            if (contentMatches) return { ...next, id: prev.id };
            return next;
          });
          return groupIntermediateTurns(merged, current);
        });
        historyHydratedRef.current = activeSessionName;
        setError(null);
      } catch (historyError) {
        if (controller.signal.aborted) return;
        if (messagesRef.current.length === 0) {
          setError(`Could not load conversation: ${String(historyError)}`);
        }
      } finally {
        if (historyAbortRef.current === controller) historyAbortRef.current = null;
        if (initialLoad && lastSessionRef.current === activeSessionName) {
          setHistoryLoading(false);
        }
      }
    })();

    historyRequestRef.current = { sessionName: activeSessionName, promise: request };
    try {
      await request;
    } finally {
      if (historyRequestRef.current?.promise === request) historyRequestRef.current = null;
    }
  }, [activeSessionName]);

  // MUCLI_SLIDING_WINDOW_V1: load older turns and prepend them to the
  // messages array. Called by ChatScreen when the user scrolls near the top.
  // Uses `before_index` = oldestLoadedIndexRef to request the page of turns
  // immediately older than what's currently loaded. The server returns
  // has_more + start_index so we can update the cursor.
  const loadOlderHistory = useCallback(async (): Promise<number> => {
    if (!activeSessionName || loadingOlderRef.current) return 0;
    const cursor = oldestLoadedIndexRef.current;
    if (cursor === null || cursor <= 0 || !hasMore) return 0;

    loadingOlderRef.current = true;
    setLoadingOlder(true);
    const controller = new AbortController();
    olderHistoryAbortRef.current = controller;
    try {
      const response = await fetchCheckpointHistory(activeSessionName, {
        signal: controller.signal,
        timeoutMs: 15_000,
        beforeIndex: cursor,
      });
      if (controller.signal.aborted || lastSessionRef.current !== activeSessionName) return 0;
      const olderMessages = await historyToMessagesCooperatively(
        response.turns || [],
        controller.signal,
      );
      if (controller.signal.aborted || lastSessionRef.current !== activeSessionName) return 0;
      const olderCheckpointCount = new Set(
        olderMessages
          .filter(message => message.role === 'user' && message.historyIndex != null)
          .map(message => message.historyIndex),
      ).size;
      const newOldest = response.start_index ?? null;
      oldestLoadedIndexRef.current = newOldest;
      setHasMore(response.has_more ?? false);
      if (olderMessages.length > 0) {
        setMessages(current => {
          return groupIntermediateTurns(
            [...olderMessages, ...flattenCollapsedMessages(current)],
            current,
          );
        });
      }
      return olderCheckpointCount;
    } catch {
      return 0;
    } finally {
      if (olderHistoryAbortRef.current === controller) olderHistoryAbortRef.current = null;
      loadingOlderRef.current = false;
      setLoadingOlder(false);
    }
  }, [activeSessionName, hasMore]);

  const syncSessionState = useCallback(async () => {
    if (!activeSessionName) return;
    stateAbortRef.current?.abort();
    const controller = new AbortController();
    stateAbortRef.current = controller;
    try {
      const response = await sessionsApi.getActive(activeSessionName, {
        signal: controller.signal,
        timeoutMs: 8_000,
      }) as ActiveSessionState;
      if (controller.signal.aborted || lastSessionRef.current !== activeSessionName) return;
      // `external_active` is a short cross-process write pulse, not evidence
      // that an agent turn is still running. Only the server-owned busy event
      // controls the mobile generating state.
      const busy = Boolean(response.active && response.is_busy);
      const externalWriteAt = Number(response.external_last_at || 0);
      const sawExternalWrite = externalWriteAt > externalWriteAtRef.current;
      if (sawExternalWrite) externalWriteAtRef.current = externalWriteAt;
      // F11 (round-13): track the server-side session revision so offline
      // mutations can queue with If-Match CAS protection. Round-32b F4:
      // capture the token losslessly — number stays number, a decimal
      // string (revision > 2^53-1) passes through verbatim; Number()
      // would round 9007199254740993 to ...992 and CAS would never match.
      const stateRevision = revisionToken(response.revision);
      if (stateRevision !== null) {
        useConnectionStore.getState().setSessionRevision(stateRevision);
      }
      const wasBusy = busyRef.current;
      busyRef.current = busy;
      setStreaming(busy);

      if (busy) {
        const hasStreamingText = messagesRef.current.some(message =>
          message.role === 'assistant' && message.streaming && message.text.trim().length > 0,
        );
        if (!hasStreamingText) setWaitingForFirstToken(true);
        // Round-20 F46: reconciliation restored busy/streaming but only
        // replaced the activity label when SSE was disconnected — after a
        // 'Send failed' label was set (F42 path) with SSE still connected,
        // the UI showed a failed send while a server turn was actually
        // running. Busy is authoritative for the label whenever the
        // current label is a terminal failure marker.
        setActivityLabel(prev => (prev === 'Send failed' ? 'Thinking' : prev));
        if (!sseConnectedRef.current) setActivityLabel('Reconnecting');
      } else {
        setWaitingForFirstToken(false);
        setActivityLabel('Thinking');
        if (wasBusy && sseConnectedRef.current) {
          // SSE will deliver history_refresh; skip.
        } else if (sawExternalWrite || historyHydratedRef.current !== activeSessionName) {
          await loadHistory(false);
        }
      }
    } catch {
      if (controller.signal.aborted) return;
      if (busyRef.current) {
        setStreaming(true);
        setWaitingForFirstToken(true);
        setActivityLabel('Reconnecting');
      }
    } finally {
      if (stateAbortRef.current === controller) stateAbortRef.current = null;
    }
  }, [activeSessionName, loadHistory]);

  const handleEvent = useCallback((event: StreamEvent) => {
    if (!eventBelongsToSession(event, activeSessionName)) return;
    const kind = event.kind;

    if (kind === 'hello') {
      const busyNames = Array.isArray(event.busy) ? event.busy.map(String) : [];
      const busy = Boolean(activeSessionName && busyNames.includes(activeSessionName));
      busyRef.current = busy;
      setStreaming(busy);
      setWaitingForFirstToken(busy);
      // MUCLI_MOBILE_RECONNECT_YOLO_V1: reconnect history recovery. If
      // the phone missed turn_complete/history_refresh while suspended, a new
      // hello with this session no longer busy must reconcile from history.
      if (!busy) void loadHistory(false);
      return;
    }

    if (kind === 'user_message') {
      const attachments = Array.isArray(event.attachments)
        ? event.attachments.filter(value => value && typeof value === 'object') as AttachmentDescriptor[]
        : [];
      appendUserMessage(String(event.text || ''), 'stream', attachments);
      return;
    }

    if (kind === 'assistant_start') {
      const turnId = String(event.turn_id || 'active-turn');
      setMessages(current => prepareForAssistantTurn(current, turnId));
      busyRef.current = true;
      setStreaming(true);
      setWaitingForFirstToken(true);
      setActivityLabel('Generating');
      return;
    }

    if (kind === 'assistant_delta') {
      busyRef.current = true;
      setStreaming(true);
      setWaitingForFirstToken(false);
      setActivityLabel('Generating');
      appendAssistantDelta(String(event.turn_id || 'active-turn'), String(event.text || ''));
      return;
    }

    if (kind === 'thinking_delta') {
      busyRef.current = true;
      setStreaming(true);
      setWaitingForFirstToken(true);
      setActivityLabel('Thinking');
      return;
    }

    if (kind === 'tool_call') {
      busyRef.current = true;
      setStreaming(true);
      setWaitingForFirstToken(true);
      const toolName = typeof event.tool_name === 'string' ? event.tool_name : '';
      setActivityLabel(toolName ? `Running ${toolName}` : 'Running tool');
      return;
    }

    if (kind === 'tool_result') {
      busyRef.current = true;
      setStreaming(true);
      setWaitingForFirstToken(true);
      setActivityLabel('Thinking');
      return;
    }

    if (kind === 'subagent_start' || kind === 'subagent_progress' || kind === 'subagent_end') {
      upsertSubagent({
        ...event,
        status: kind === 'subagent_start'
          ? 'running'
          : (kind === 'subagent_end' ? String(event.status || 'done') : event.status),
      });
      return;
    }

    if (kind === 'subagent_snapshot') {
      replaceSubagentSnapshot(event);
      return;
    }

    if (kind === 'assistant_end') {
      // Round-44 F5: flush coalesced deltas BEFORE finalizing so the full
      // streamed text is in state when streaming is retired.
      flushPendingDeltas();
      finalizeAssistant(String(event.turn_id || 'active-turn'));
      if (busyRef.current) {
        setWaitingForFirstToken(true);
        setActivityLabel('Finishing');
        if (completionProbeRef.current) clearTimeout(completionProbeRef.current);
        completionProbeRef.current = setTimeout(() => {
          void syncSessionState();
        }, 500);
      }
      return;
    }

    if (kind === 'prompt') {
      busyRef.current = true;
      setStreaming(true);
      setWaitingForFirstToken(true);
      setActivityLabel('Waiting for approval');
      return;
    }

    if (kind === 'prompt_resolved' || kind === 'prompt_cancelled') {
      if (busyRef.current) {
        setWaitingForFirstToken(true);
        setActivityLabel('Generating');
      }
      return;
    }

    if (kind === 'turn_complete') {
      // Round-44 F5: never drop pending deltas on turn completion.
      flushPendingDeltas();
      if (handoffTimerRef.current) {
        clearTimeout(handoffTimerRef.current);
        handoffTimerRef.current = null;
      }
      if (completionProbeRef.current) {
        clearTimeout(completionProbeRef.current);
        completionProbeRef.current = null;
      }
      busyRef.current = false;
      setStreaming(false);
      setWaitingForFirstToken(false);
      setActivityLabel('Thinking');
      // Round-31 F35: the turn_complete publish carries the post-turn
      // revision (chat.py _drive); capture it as the fresh If-Match token
      // so the next offline mutation CASes against the state just written.
      // Round-32b F4: lossless token capture (never Number()-convert).
      const completedRevision = revisionToken(event.revision);
      if (completedRevision !== null) {
        useConnectionStore.getState().setSessionRevision(completedRevision);
      }
      const result = event.result && typeof event.result === 'object'
        ? event.result as Record<string, unknown>
        : null;
      if (result?.status === 'error' && result.error) {
        setError(String(result.error));
      }
      setMessages(current => {
        const retired = current.map(message =>
          message.role === 'assistant' && message.streaming
            ? { ...message, streaming: false, handoff: undefined }
            : { ...message, handoff: undefined }
        );
        return groupIntermediateTurns(retired, current);
      });
      setArtifactRevision(value => value + 1);
      // Do NOT call loadHistory here. The server may not have persisted
      // the final assistant message yet, so loadHistory would replace
      // the live-streamed message with stale history (missing the last
      // turn) — causing the screen to flash and the final output to
      // vanish. Instead, wait for the `history_refresh` SSE event,
      // which the server emits after the session is safely persisted.
      // Safety net: if history_refresh doesn't arrive in 3s, force reload.
      if (historyFallbackRef.current) clearTimeout(historyFallbackRef.current);
      historyFallbackRef.current = setTimeout(() => {
        historyFallbackRef.current = null;
        void loadHistory(false);
      }, 3000);
      return;
    }

    if (kind === 'artifact_created') {
      const artifact = asVisualization(event.artifact);
      if (artifact) {
        setMessages(current => {
          const updated = [...current];
          const existing = updated.findIndex(message =>
            message.role === 'visualization'
            && message.artifact?.artifact_id === artifact.artifact_id,
          );
          const next: ChatMessage = {
            id: `visualization-${artifact.artifact_id}`,
            role: 'visualization',
            text: '',
            streaming: false,
            origin: 'stream',
            artifact,
          };
          if (existing >= 0) {
            updated[existing] = next;
            return updated;
          }

          // MUCLI_VISUALIZATION_TIMELINE_V2: close the current assistant segment
          // at the artifact boundary. Future deltas create a new segment after
          // the visualization instead of mutating text above it.
          let insertAt = updated.length;
          for (let index = updated.length - 1; index >= 0; index -= 1) {
            const message = updated[index];
            if (message.role === 'assistant' && message.streaming) {
              if (message.turnId) seenAssistantTurnsRef.current.delete(message.turnId);
              updated[index] = {
                ...message,
                id: `${message.id}-segment-${artifact.artifact_id}`,
                turnId: `${message.turnId || 'active-turn'}-segment-${artifact.artifact_id}`,
                streaming: false,
              };
              insertAt = index + 1;
              break;
            }
          }
          updated.splice(insertAt, 0, next);
          return updated;
        });
      }
      setArtifactRevision(value => value + 1);
      return;
    }

    if (kind === 'history_refresh') {
      if (historyFallbackRef.current) {
        clearTimeout(historyFallbackRef.current);
        historyFallbackRef.current = null;
      }
      setArtifactRevision(value => value + 1);
      if (!busyRef.current) void loadHistory(false);
      return;
    }

    if (kind === 'session_updated') {
      // Round-31 F35: the watcher now publishes the post-reload session
      // revision in session_updated; capture it as the If-Match token so
      // offline mutations CAS against the server state we just learned
      // about. Round-32b F4: lossless token capture — the decimal string
      // form (revision above 2^53-1) must pass through verbatim.
      // Round-33b F3: a session_updated WITHOUT a valid revision is the
      // deferred/failed-reload path (round-32 F7) — the server state this
      // event describes was NOT re-read, so any previously captured token
      // is now known-stale. Reset it to null: the next offline send then
      // goes out UNGUARDED (server decides) rather than 409-ing forever
      // against a dead token. The successful deferred_reload event DOES
      // carry a fresh revision and re-arms the guard here.
      const updatedRevision = revisionToken(event.revision);
      useConnectionStore.getState().setSessionRevision(updatedRevision);
      if (!busyRef.current) void loadHistory(false);
      return;
    }

    if (kind === 'error') {
      if (completionProbeRef.current) {
        clearTimeout(completionProbeRef.current);
        completionProbeRef.current = null;
      }
      busyRef.current = false;
      setStreaming(false);
      setWaitingForFirstToken(false);
      setMessages(current => current.map(message =>
        message.role === 'assistant' && message.streaming
          ? { ...message, streaming: false }
          : message
      ));
      setError(String(event.text || 'Agent error'));
      // Same as turn_complete: skip loadHistory when SSE connected.
      // The server will emit history_refresh after persisting.
      if (!sseConnectedRef.current) void loadHistory(false);
    }
  }, [
    activeSessionName,
    appendAssistantDelta,
    appendUserMessage,
    finalizeAssistant,
    flushPendingDeltas,
    loadHistory,
    replaceSubagentSnapshot,
    syncSessionState,
    upsertSubagent,
  ]);

  useEffect(() => {
    subscriptionRef.current?.close();
    subscriptionRef.current = null;
    historyAbortRef.current?.abort();
    historyAbortRef.current = null;
    olderHistoryAbortRef.current?.abort();
    olderHistoryAbortRef.current = null;
    historyRequestRef.current = null;
    stateAbortRef.current?.abort();
    stateAbortRef.current = null;
    const sessionChanged = lastSessionRef.current !== activeSessionName;
    lastSessionRef.current = activeSessionName;
    if (sessionChanged) {
      setMessages([]);
      setError(null);
      setActivityLabel('Thinking');
      busyRef.current = false;
      historyHydratedRef.current = null;
      externalWriteAtRef.current = 0;
      oldestLoadedIndexRef.current = null;
      loadingOlderRef.current = false;
      setHasMore(false);
      setLoadingOlder(false);
      setArtifactRevision(value => value + 1);
      seenAssistantTurnsRef.current.clear();
      if (completionProbeRef.current) {
        clearTimeout(completionProbeRef.current);
        completionProbeRef.current = null;
      }
      if (handoffTimerRef.current) {
        clearTimeout(handoffTimerRef.current);
        handoffTimerRef.current = null;
      }
      if (historyFallbackRef.current) {
        clearTimeout(historyFallbackRef.current);
        historyFallbackRef.current = null;
      }
    }
    setSseConnected(false);

    if (!activeSessionName) {
      setHistoryLoading(false);
      setStreaming(false);
      setWaitingForFirstToken(false);
      return undefined;
    }

    void loadHistory(false);
    subscriptionRef.current = subscribeToEvents({
      onOpen: () => {
        setSseConnected(true);
        void syncSessionState();
      },
      onMessage: handleEvent,
      onError: () => {
        setSseConnected(false);
        if (busyRef.current) {
          setStreaming(true);
          setWaitingForFirstToken(true);
          setActivityLabel('Reconnecting');
        }
        void syncSessionState();
      },
      onClose: () => {
        setSseConnected(false);
        void syncSessionState();
      },
    }, { sessionName: activeSessionName });

    void syncSessionState();
    const poll = setInterval(() => {
      // A connected event stream already carries busy, completion, prompt,
      // artifact, and external session updates. Skip only while connected AND
      // busy so idle sessions still reconcile; this keeps mobile from
      // hammering the host while an agent runs.
      if (sseConnectedRef.current && busyRef.current) return;
      void syncSessionState();
    }, SESSION_POLL_MS);

    return () => {
      clearInterval(poll);
      if (completionProbeRef.current) {
        clearTimeout(completionProbeRef.current);
        completionProbeRef.current = null;
      }
      if (historyFallbackRef.current) {
        clearTimeout(historyFallbackRef.current);
        historyFallbackRef.current = null;
      }
      if (handoffTimerRef.current) {
        clearTimeout(handoffTimerRef.current);
        handoffTimerRef.current = null;
      }
      subscriptionRef.current?.close();
      subscriptionRef.current = null;
      historyAbortRef.current?.abort();
      historyAbortRef.current = null;
      olderHistoryAbortRef.current?.abort();
      olderHistoryAbortRef.current = null;
      stateAbortRef.current?.abort();
      stateAbortRef.current = null;
    };
  }, [activeSessionName, handleEvent, loadHistory, reconnectKey, syncSessionState]);

  const sendMessage = useCallback(async (text: string, attachments: AttachmentDescriptor[] = []) => {
    let trimmed = text.trim();
    if (!trimmed && attachments.length > 0) trimmed = 'Please review the attached document(s).';
    if (!trimmed || !activeSessionName || busyRef.current) return false;

    setError(null);
    busyRef.current = true;
    setStreaming(true);
    setWaitingForFirstToken(true);
    setActivityLabel(sseConnectedRef.current ? 'Thinking' : 'Connecting');
    appendUserMessage(trimmed, 'local', attachments);

    // G5 (§3.6): offline sends join the outbound queue and replay on the
    // next reconnect; the optimistic user bubble above keeps the chat UX.
    // F11: carry the last-known session revision as If-Match so a stale
    // send is rejected (409) instead of silently clobbering.
    if (await queueIfOffline('chat_send', {
      text: trimmed,
      session_name: activeSessionName,
      attachment_ids: attachments.map(item => item.attachment_id),
    }, {
      sessionName: activeSessionName,
      ifMatch: useConnectionStore.getState().sessionRevision ?? undefined,
    })) {
      busyRef.current = false;
      setStreaming(false);
      setWaitingForFirstToken(false);
      setActivityLabel('Queued (offline)');
      return true;
    }

    try {
      const response = await chatApi.send(trimmed, activeSessionName, attachments.map(item => item.attachment_id));
      // Round-32b F5: capture the send response's revision — covers the
      // SSE-missed-terminal-event case (the phone never saw turn_complete,
      // so this response is the only fresh If-Match source). Same lossless
      // token handling as F4; only valid tokens update the store.
      const sendRevision = revisionToken(response.revision);
      if (sendRevision !== null) {
        useConnectionStore.getState().setSessionRevision(sendRevision);
      }
      return true;
    } catch (sendError) {
      setError(String(sendError));
      // Round-19 F42: the send never reached/failed at the server — the
      // turn is NOT in flight. Clear the local busy state BEFORE the
      // reconciliation attempt; if syncSessionState also fails it
      // deliberately preserves a busy presentation, which dead-locked
      // the send guard (busyRef stays true forever, no further sends).
      busyRef.current = false;
      setStreaming(false);
      setWaitingForFirstToken(false);
      setActivityLabel('Send failed');
      try {
        await syncSessionState();
      } catch {
        // Reconciliation failed — the local state above is still the
        // truthful one (nothing is running server-side).
      }
      return false;
    }
  }, [activeSessionName, appendUserMessage, syncSessionState]);

  const stop = useCallback(async () => {
    if (!activeSessionName) return;
    setActivityLabel('Stopping');
    try {
      await chatApi.interrupt(activeSessionName);
    } catch (stopError) {
      setError(String(stopError));
    }
    await syncSessionState();
  }, [activeSessionName, syncSessionState]);

  const retry = useCallback(() => {
    setError(null);
    setReconnectKey(value => value + 1);
    void loadHistory(false);
    void syncSessionState();
  }, [loadHistory, syncSessionState]);

  return {
    messages,
    setMessages,
    streaming,
    waitingForFirstToken,
    historyLoading,
    activityLabel,
    sseConnected,
    error,
    artifactRevision,
    hasMore,
    loadingOlder,
    sendMessage,
    stop,
    retry,
    loadOlderHistory,
  };
}
