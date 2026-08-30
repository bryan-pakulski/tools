import { AppState, type AppStateStatus } from 'react-native';
import EventSource from 'react-native-sse';

export const DEFAULT_RECONNECT_DELAY_MS = 2_500;

export interface SSEHandlers {
  onMessage?: (event: { kind: string; [key: string]: unknown }) => void;
  onOpen?: () => void;
  onError?: (error: Error) => void;
  onClose?: () => void;
}

export interface SSESubscription {
  close: () => void;
}

export interface SSEOptions {
  sessionName?: string | null;
  reconnectDelayMs?: number;
}

export function subscribeToEvents(
  handlers: SSEHandlers,
  options: SSEOptions = {},
): SSESubscription {
  // MUCLI_MOBILE_RECONNECT_YOLO_V1: explicit lifecycle reconnect. Android can
  // suspend the native socket without react-native-sse recreating it when the
  // app returns to the foreground. Own the reconnect loop and rebuild the
  // EventSource with the latest configured host on every attempt.
  const sessionName = options.sessionName === undefined
    ? require('../store/connection').useConnectionStore.getState().activeSessionName
    : options.sessionName;
  const reconnectDelayMs = Math.max(
    1_000,
    options.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS,
  );

  let closed = false;
  let source: EventSource | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectAttempt = 0;
  let appState: AppStateStatus = AppState.currentState;

  const isForeground = () => appState !== 'background' && appState !== 'inactive';

  const buildUrl = () => {
    // Import lazily to avoid circular deps and read the newest base URL after
    // the user edits connection settings.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const connection = require('../store/connection').useConnectionStore.getState();
    let url = `${connection.baseUrl}/api/events`;
    if (sessionName) url += `?session_name=${encodeURIComponent(sessionName)}`;
    return url;
  };

  const clearReconnect = () => {
    if (!reconnectTimer) return;
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  };

  const destroySource = () => {
    const current = source;
    source = null;
    if (!current) return;
    current.removeAllEventListeners();
    current.close();
  };

  let connect: () => void;

  const scheduleReconnect = (immediate = false) => {
    if (closed || !isForeground() || reconnectTimer) return;
    const delay = immediate
      ? 0
      : Math.min(15_000, reconnectDelayMs * (2 ** Math.min(reconnectAttempt, 3)));
    reconnectAttempt += 1;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  };

  connect = () => {
    if (closed || !isForeground()) return;
    clearReconnect();
    destroySource();

    const next = new EventSource(buildUrl(), {
      headers: { Accept: 'text/event-stream' },
      // SSE push is the primary transport: pollingInterval: 0 disables the
      // library's hidden polling/retry loop entirely. Reconnection is owned
      // exclusively by our foreground-aware backoff loop below — a nonzero
      // pollingInterval here meant two competing reconnect mechanisms racing
      // on the same transport drop.
      pollingInterval: 0,
    });
    source = next;

    next.addEventListener('open', () => {
      if (closed || source !== next) return;
      reconnectAttempt = 0;
      handlers.onOpen?.();
    });

    next.addEventListener('message', (event) => {
      if (closed || source !== next || !event.data) return;
      // Parse and dispatch are separate concerns: a malformed server payload
      // must not be conflated with (and swallow) a reducer exception.
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        handlers.onError?.(new Error(`SSE: malformed JSON payload: ${String(event.data).slice(0, 120)}`));
        return;
      }
      if (!parsed || typeof parsed !== 'object' || typeof (parsed as { kind?: unknown }).kind !== 'string') {
        handlers.onError?.(new Error('SSE: event missing string "kind" field'));
        return;
      }
      handlers.onMessage?.(parsed as { kind: string; [key: string]: unknown });
    });

    next.addEventListener('error', (event) => {
      if (closed || source !== next) return;
      const msg = (event as unknown as { message?: string })?.message || 'SSE error';
      handlers.onError?.(new Error(msg));
      destroySource();
      scheduleReconnect();
    });

    next.addEventListener('close', () => {
      if (closed || source !== next) return;
      handlers.onClose?.();
      destroySource();
      scheduleReconnect();
    });
  };

  const appStateSubscription = AppState.addEventListener('change', nextState => {
    const wasForeground = isForeground();
    appState = nextState;
    if (!isForeground()) {
      clearReconnect();
      destroySource();
      return;
    }
    if (!wasForeground) {
      reconnectAttempt = 0;
      scheduleReconnect(true);
    }
  });

  connect();

  return {
    close: () => {
      if (closed) return;
      closed = true;
      clearReconnect();
      appStateSubscription.remove();
      destroySource();
    },
  };
}

export function subscribeToKind(
  kind: string,
  onEvent: (data: Record<string, unknown>) => void,
  options?: SSEOptions,
): SSESubscription {
  return subscribeToEvents({
    onMessage: (event) => {
      if (event.kind === kind) onEvent(event);
    },
  }, options);
}
