import { useConnectionStore } from '../store/connection';

export const DEFAULT_REQUEST_TIMEOUT_MS = 12_000;

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

export function baseUrl(): string {
  return useConnectionStore.getState().baseUrl;
}

type QueryValue = string | number | boolean | undefined;

export interface RequestOptions {
  body?: unknown;
  signal?: AbortSignal;
  query?: Record<string, QueryValue>;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

function timeoutMessage(timeoutMs: number): string {
  const seconds = Math.max(1, Math.round(timeoutMs / 1000));
  return `Request timed out after ${seconds}s`;
}

async function request<T>(
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  path: string,
  opts?: RequestOptions,
): Promise<T> {
  const base = baseUrl();
  let url = `${base}${path}`;

  // Append the active session for reads and mutations when the caller did not
  // provide one explicitly. Mode controls are session-scoped too; limiting
  // this to GET/DELETE made mobile POSTs mutate whichever session happened to
  // be focused in the web daemon instead of the mobile-selected container.
  {
    const sep = url.includes('?') ? '&' : '?';
    const sn = useConnectionStore.getState().activeSessionName;
    const explicitSession = Object.prototype.hasOwnProperty.call(
      opts?.query || {},
      'session_name',
    );
    if (sn && !path.includes('session_name') && !explicitSession) {
      url += `${sep}session_name=${encodeURIComponent(sn)}`;
    }
  }

  if (opts?.query) {
    for (const [key, value] of Object.entries(opts.query)) {
      if (value !== undefined) {
        const sep = url.includes('?') ? '&' : '?';
        url += `${sep}${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`;
      }
    }
  }

  const headers: Record<string, string> = { ...(opts?.headers || {}) };
  let bodyStr: string | undefined;
  if (opts?.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    bodyStr = JSON.stringify(opts.body);
  }

  const timeoutMs = opts?.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const controller = new AbortController();
  let timedOut = false;
  const forwardAbort = () => controller.abort();
  if (opts?.signal?.aborted) controller.abort();
  else opts?.signal?.addEventListener('abort', forwardAbort, { once: true });
  const timeout = timeoutMs > 0
    ? setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs)
    : null;

  try {
    const response = await fetch(url, {
      method,
      headers,
      body: bodyStr,
      signal: controller.signal,
    });

    if (!response.ok) {
      // Response bodies are single-consumption streams: read once as text,
      // then parse guarded. (json() failure followed by text() always failed.)
      let errorBody: unknown;
      let errorMsg = `HTTP ${response.status}`;
      try {
        const raw = await response.text();
        if (raw) {
          try {
            errorBody = JSON.parse(raw);
          } catch {
            errorBody = raw;
          }
          if (errorBody && typeof errorBody === 'object' && 'detail' in errorBody) {
            const structuredDetail = (errorBody as Record<string, unknown>).detail;
            if (typeof structuredDetail === 'string') {
              errorMsg = structuredDetail;
            } else if (structuredDetail && typeof structuredDetail === 'object') {
              const record = structuredDetail as Record<string, unknown>;
              errorMsg = String(record.message || record.title || errorMsg);
            }
          }
        }
      } catch {
        // Body unreadable (stream consumed or aborted) — status-only error.
      }
      throw new ApiError(response.status, errorMsg, errorBody);
    }

    if (response.status === 204) return undefined as T;

    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return response.json() as Promise<T>;
    }
    return (await response.text()) as unknown as T;
  } catch (error) {
    if (timedOut) {
      throw new ApiError(0, timeoutMessage(timeoutMs), { timeout: true, url });
    }
    throw error;
  } finally {
    if (timeout) clearTimeout(timeout);
    opts?.signal?.removeEventListener('abort', forwardAbort);
  }
}

export const api = {
  get<T>(path: string, opts?: Omit<RequestOptions, 'body'>): Promise<T> {
    return request<T>('GET', path, opts);
  },
  // MUCLI_MOBILE_RECONNECT_YOLO_V1: POST query support lets mutating
  // inspector routes target the mobile-selected session explicitly.
  post<T>(path: string, body?: unknown, opts?: Omit<RequestOptions, 'body'>): Promise<T> {
    return request<T>('POST', path, { ...opts, body });
  },
  put<T>(path: string, body?: unknown, opts?: Omit<RequestOptions, 'body'>): Promise<T> {
    return request<T>('PUT', path, { ...opts, body });
  },
  patch<T>(path: string, body?: unknown, opts?: Omit<RequestOptions, 'body'>): Promise<T> {
    return request<T>('PATCH', path, { ...opts, body });
  },
  delete<T>(path: string, opts?: Omit<RequestOptions, 'body'>): Promise<T> {
    return request<T>('DELETE', path, opts);
  },
};

export async function checkHealth(
  targetBaseUrl?: string,
  opts?: { timeoutMs?: number },
): Promise<boolean> {
  const url = (targetBaseUrl || useConnectionStore.getState().baseUrl) + '/healthz';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), opts?.timeoutMs ?? 5_000);
  try {
    const resp = await fetch(url, { method: 'GET', signal: controller.signal });
    return resp.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}
