import { api } from './client';

export interface ChatSendResponse {
  accepted: boolean;
  kind: string;
  session_name: string;
  /**
   * Round-32b F4/F5: JS-safe session revision token from the send
   * response. number for revisions <= 2^53-1, decimal string above —
   * capture verbatim (never Number()-convert) as the next If-Match.
   */
  revision?: number | string;
}

export interface HistorySearchResult {
  index: number;
  role: string;
  before_anchor: boolean;
  parts_matched: Array<Record<string, unknown>>;
  context_before: Array<Record<string, unknown>>;
  context_after: Array<Record<string, unknown>>;
  cache_key: string | null;
}

export interface HistorySearchResponse {
  results: HistorySearchResult[];
  total_matches: number;
  has_more: boolean;
}

export interface CommandSpec {
  names: string[];
  help: string;
}

export const chatApi = {
  send: (text: string, sessionName?: string, attachmentIds: string[] = []) =>
    api.post<ChatSendResponse>('/api/chat/send', {
      text,
      session_name: sessionName,
      attachment_ids: attachmentIds,
    }),
  interrupt: (sessionName?: string) =>
    api.post<Record<string, unknown>>('/api/chat/interrupt', { session_name: sessionName }),
  getCommands: () => api.get<{ commands: CommandSpec[] }>('/api/chat/commands'),
  getCompletions: (kind: string) =>
    api.get<{ items: string[] }>('/api/chat/completions', { query: { kind } }),
  searchHistory: (query: string, role?: string, toolName?: string, maxResults?: number) =>
    api.get<HistorySearchResponse>('/api/chat/history/search', {
      query: { query, role, tool_name: toolName, max_results: maxResults },
    }),
};