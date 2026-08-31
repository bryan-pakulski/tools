import { api } from './client';

export type ThreadStatus =
  | 'idle'
  | 'running'
  | 'waiting_peer'
  | 'awaiting_approval'
  | 'interrupted'
  | 'error'
  | 'closed';

export interface ThreadSummary {
  thread_id: string;
  session_name: string;
  title: string;
  status: ThreadStatus;
  current_goal: string;
  run_origin: string;
  runtime_id: string;
  last_seen: number;
  created_at: number;
  updated_at: number;
  unread_count: number;
  claimed_paths: string[];
}

export interface ThreadListResponse {
  session_name: string;
  thread_group_id: string;
  current_thread_id: string;
  threads: ThreadSummary[];
}

export interface ThreadActivityEvent {
  event_id: number;
  kind: string;
  actor_thread_id: string;
  actor_title?: string | null;
  target_thread_id: string;
  target_title?: string | null;
  message_id: string;
  conflict_id: string;
  payload: Record<string, unknown>;
  created_at: number;
}

export interface ThreadActivityResponse {
  thread_group_id: string;
  events: ThreadActivityEvent[];
  last_event_id: number;
}

export interface CreateThreadRequest {
  parentSessionName: string;
  title: string;
  name?: string;
  activate?: boolean;
}

export interface CreateThreadResponse {
  ok: boolean;
  active: boolean;
  session_name: string;
  thread_meta: {
    thread_id: string;
    group_id: string;
    parent_thread_id: string;
    title: string;
    created_at: number;
  };
}

// Compatibility names used by the mobile UI tests and older call sites.
export type ThreadListItem = ThreadSummary;
export type ThreadsListResponse = ThreadListResponse;
export type ThreadMeta = CreateThreadResponse['thread_meta'];
export type CreateThreadOptions = CreateThreadRequest;
export type CreateThreadResult = CreateThreadResponse;

export interface ThreadRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface DeleteThreadResponse {
  ok: boolean;
  thread_id: string;
  deleted_session_name: string;
}

export const threadsApi = {
  list: (sessionName: string, options?: ThreadRequestOptions) =>
    api.get<ThreadListResponse>('/api/threads', {
      ...options,
      query: { session_name: sessionName },
    }),
  activity: (
    sessionName: string,
    options?: ThreadRequestOptions & { afterId?: number; limit?: number },
  ) => api.get<ThreadActivityResponse>('/api/threads/activity', {
    signal: options?.signal,
    timeoutMs: options?.timeoutMs,
    query: {
      session_name: sessionName,
      after_id: options?.afterId,
      limit: options?.limit,
    },
  }),
  create: (request: CreateThreadRequest, options?: ThreadRequestOptions) =>
    api.post<CreateThreadResponse>('/api/threads', {
      parent_session_name: request.parentSessionName,
      title: request.title,
      name: request.name,
      activate: request.activate ?? true,
    }, {
      ...options,
      query: { session_name: request.parentSessionName },
    }),
  remove: (threadId: string, sessionName?: string, options?: ThreadRequestOptions) =>
    api.delete<DeleteThreadResponse>(`/api/threads/${encodeURIComponent(threadId)}`, {
      ...options,
      query: { session_name: sessionName ?? '' },
    }),
};
