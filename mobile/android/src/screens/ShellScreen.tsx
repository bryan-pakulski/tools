import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  NativeSyntheticEvent,
  NativeScrollEvent,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../theme/ThemeContext';
import { Text, Badge } from '../components';
import { useConnectionStore } from '../store/connection';
import { sessionsApi } from '../api/sessions';
import { spacing } from '../theme/tokens';

const MUCLI_SHELL_QOL_V1 = true;
const SHELL_HISTORY = new Map<string, string[]>();

// Strip ANSI escape sequences for plain-text display.
const ANSI_RE = /\u001b\[[0-9;]*[a-zA-Z]|\u001b\][^\u0007]*\u0007|\u001b[()][AB012]|\x07/g;

function stripAnsi(s: string): string {
  return s.replace(ANSI_RE, '');
}

interface CompletionMessage {
  type: 'shell_completion';
  request_id?: string;
  source?: string;
  start?: number;
  end?: number;
  replacement?: string;
  candidates?: string[];
}

export function ShellScreen() {
  const { colors } = useTheme();
  const { baseUrl, activeSessionName } = useConnectionStore();
  const [output, setOutput] = useState<string>('');
  const [input, setInputState] = useState('');
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [containerName, setContainerName] = useState<string | null>(null);
  const [followingOutput, setFollowingOutput] = useState(true);

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<ScrollView>(null);
  const inputRef = useRef<TextInput>(null);
  const outputBuf = useRef<string[]>([]);
  const followOutputRef = useRef(true);
  const inputValueRef = useRef('');
  const historyRef = useRef<string[]>([]);
  const historyIndexRef = useRef(0);
  const historyDraftRef = useRef('');
  const completionSeqRef = useRef(0);

  const setInput = useCallback((value: string) => {
    inputValueRef.current = value;
    setInputState(value);
  }, []);

  const setFollow = useCallback((value: boolean) => {
    followOutputRef.current = value;
    setFollowingOutput(value);
  }, []);

  const jumpToEnd = useCallback((animated = true) => {
    setFollow(true);
    requestAnimationFrame(() => scrollRef.current?.scrollToEnd({ animated }));
  }, [setFollow]);

  const appendOutput = useCallback((text: string, forceFollow = false) => {
    const clean = stripAnsi(text);
    if (!clean) return;
    outputBuf.current.push(clean);

    // Bound by approximate lines while retaining chunk boundaries.
    let joined = outputBuf.current.join('');
    const lines = joined.split('\n');
    if (lines.length > 2000) {
      joined = lines.slice(-2000).join('\n');
      outputBuf.current = [joined];
    }
    setOutput(joined);

    if (forceFollow || followOutputRef.current) {
      requestAnimationFrame(() => scrollRef.current?.scrollToEnd({ animated: false }));
    }
  }, []);

  const handleCompletion = useCallback((message: CompletionMessage) => {
    const current = inputValueRef.current;
    if (message.source !== current) return;
    const start = Math.max(0, Math.min(Number(message.start) || 0, current.length));
    const end = Math.max(start, Math.min(Number(message.end) || current.length, current.length));
    const replacement = String(message.replacement || '');
    const candidates = Array.isArray(message.candidates) ? message.candidates.map(String) : [];

    if (replacement) {
      setInput(current.slice(0, start) + replacement + current.slice(end));
      requestAnimationFrame(() => inputRef.current?.focus());
      return;
    }
    if (candidates.length > 1) {
      appendOutput(`\n${candidates.join('  ')}\n`);
    }
  }, [appendOutput, setInput]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }
    setConnected(false);
    setConnecting(false);
  }, []);

  const connect = useCallback(async () => {
    if (connecting || connected) return;
    setConnecting(true);
    setError(null);
    outputBuf.current = [];
    setOutput('');
    setFollow(true);

    let container = containerName;
    if (!container && activeSessionName) {
      try {
        const info = await sessionsApi.getContainer(activeSessionName);
        const name = (info as Record<string, unknown>).name;
        if (typeof name === 'string') {
          container = name;
          setContainerName(name);
        }
      } catch {
        // fall through — no container attached
      }
    }

    if (!container) {
      setError('No container attached to the active session.');
      setConnecting(false);
      return;
    }

    historyRef.current = [...(SHELL_HISTORY.get(container) || [])];
    historyIndexRef.current = historyRef.current.length;
    historyDraftRef.current = '';

    const wsBase = baseUrl
      .replace(/^http:\/\//i, 'ws://')
      .replace(/^https:\/\//i, 'wss://');
    const wsUrl = `${wsBase}/api/containers/${encodeURIComponent(container)}/shell`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setConnecting(false);
        appendOutput(`Connected to ${container}\n`, true);
      };

      ws.onmessage = (event: WebSocketMessageEvent) => {
        const data = typeof event.data === 'string' ? event.data : '';
        if (!data) return;
        if (data.startsWith('{')) {
          try {
            const message = JSON.parse(data) as CompletionMessage;
            if (message.type === 'shell_completion') {
              handleCompletion(message);
              return;
            }
          } catch {
            // Normal shell output may itself begin with a JSON object.
          }
        }
        appendOutput(data);
      };

      ws.onerror = () => {
        setError('WebSocket error — check the server is reachable.');
        setConnecting(false);
        setConnected(false);
      };

      ws.onclose = (event: CloseEvent) => {
        setConnected(false);
        setConnecting(false);
        if (event.code !== 1000) {
          appendOutput(`\n[Connection closed: ${event.code}${event.reason ? ' ' + event.reason : ''}]\n`);
        } else {
          appendOutput('\n[Connection closed]\n');
        }
      };
    } catch (e) {
      setError(String(e));
      setConnecting(false);
    }
  }, [
    activeSessionName,
    appendOutput,
    baseUrl,
    connected,
    connecting,
    containerName,
    handleCompletion,
    setFollow,
  ]);

  const recordHistory = useCallback((command: string) => {
    const value = command.trim();
    if (!value) return;
    const next = historyRef.current.filter(item => item !== value);
    next.push(value);
    historyRef.current = next.slice(-200);
    historyIndexRef.current = historyRef.current.length;
    historyDraftRef.current = '';
    if (containerName) SHELL_HISTORY.set(containerName, historyRef.current);
  }, [containerName]);

  const send = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const command = input || inputValueRef.current;
    recordHistory(command);
    setFollow(true);
    appendOutput(`$ ${command}\n`, true);
    ws.send(command + '\n');
    setInput('');
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [input, appendOutput]);

  const cycleHistory = useCallback((direction: -1 | 1) => {
    const history = historyRef.current;
    if (!history.length) return;
    if (historyIndexRef.current === history.length) {
      historyDraftRef.current = inputValueRef.current;
    }
    const next = Math.max(0, Math.min(history.length, historyIndexRef.current + direction));
    historyIndexRef.current = next;
    setInput(next === history.length ? historyDraftRef.current : history[next]);
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [setInput]);

  const complete = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const line = inputValueRef.current;
    const requestId = String(++completionSeqRef.current);
    ws.send(JSON.stringify({
      type: 'shell_complete',
      request_id: requestId,
      line,
      cursor: line.length,
    }));
  }, []);

  const clear = useCallback(() => {
    outputBuf.current = [];
    setOutput('');
    setFollow(true);
  }, [setFollow]);

  const onOutputScroll = useCallback((event: NativeSyntheticEvent<NativeScrollEvent>) => {
    const { contentOffset, contentSize, layoutMeasurement } = event.nativeEvent;
    const distance = contentSize.height - layoutMeasurement.height - contentOffset.y;
    setFollow(distance < 72);
  }, [setFollow]);

  useFocusEffect(
    useCallback(() => {
      return () => disconnect();
    }, [disconnect]),
  );

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        try { wsRef.current.close(); } catch { /* ignore */ }
        wsRef.current = null;
      }
    };
  }, []);

  const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.bg },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: spacing.base,
      paddingVertical: spacing.sm,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: colors.border,
    },
    headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    headerActions: { flexDirection: 'row', gap: 12 },
    actionBtn: { padding: 4 },
    outputWrap: { flex: 1, paddingHorizontal: spacing.sm },
    output: {
      fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
      fontSize: 12,
      lineHeight: 16,
      color: colors.text,
      includeFontPadding: false,
    },
    inputBar: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: spacing.sm,
      paddingVertical: spacing.xs,
      borderTopWidth: StyleSheet.hairlineWidth,
      borderTopColor: colors.border,
      gap: 5,
    },
    prompt: {
      fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
      fontSize: 13,
      color: colors.accent,
    },
    input: {
      flex: 1,
      fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
      fontSize: 13,
      color: colors.text,
      paddingVertical: 6,
      paddingHorizontal: 8,
      backgroundColor: colors.card,
      borderRadius: 6,
    },
    keyButton: {
      minWidth: 32,
      height: 32,
      borderRadius: 7,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: colors.card,
      paddingHorizontal: 5,
    },
    errorBar: {
      paddingHorizontal: spacing.base,
      paddingVertical: spacing.xs,
      backgroundColor: colors.errorBg || colors.card,
    },
    emptyWrap: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: spacing.base },
  });

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Ionicons name="terminal-outline" size={18} color={colors.accent} />
          <Text variant="base" style={{ fontWeight: '600' }}>Shell</Text>
          <Badge
            label={connected ? 'Connected' : connecting ? 'Connecting…' : 'Disconnected'}
            variant={connected ? 'accent' : 'neutral'}
          />
        </View>
        <View style={styles.headerActions}>
          {!followingOutput && output ? (
            <TouchableOpacity onPress={() => jumpToEnd()} style={styles.actionBtn} accessibilityLabel="Jump to latest shell output">
              <Ionicons name="arrow-down-circle-outline" size={21} color={colors.accent} />
            </TouchableOpacity>
          ) : null}
          <TouchableOpacity onPress={clear} style={styles.actionBtn} accessibilityLabel="Clear output">
            <Ionicons name="trash-outline" size={20} color={colors.textDim} />
          </TouchableOpacity>
          {connected ? (
            <TouchableOpacity onPress={disconnect} style={styles.actionBtn} accessibilityLabel="Disconnect">
              <Ionicons name="close-circle-outline" size={20} color={colors.error || colors.textDim} />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              onPress={connect}
              disabled={connecting}
              style={styles.actionBtn}
              accessibilityLabel="Connect"
            >
              <Ionicons name="play-circle-outline" size={20} color={colors.accent} />
            </TouchableOpacity>
          )}
        </View>
      </View>

      {error ? (
        <View style={styles.errorBar}>
          <Text variant="xs" style={{ color: colors.error || colors.textDim }}>{error}</Text>
        </View>
      ) : null}

      {connected || output ? (
        <KeyboardAvoidingView
          style={{ flex: 1 }}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          keyboardVerticalOffset={90}
        >
          <ScrollView
            ref={scrollRef}
            style={styles.outputWrap}
            contentContainerStyle={{ paddingVertical: spacing.sm }}
            onScroll={onOutputScroll}
            scrollEventThrottle={80}
            onContentSizeChange={() => {
              if (followOutputRef.current) scrollRef.current?.scrollToEnd({ animated: false });
            }}
          >
            <Text style={styles.output}>{output}</Text>
          </ScrollView>

          <View style={styles.inputBar}>
            <Text style={styles.prompt}>$</Text>
            <TextInput
              ref={inputRef}
              style={styles.input}
              value={input}
              onChangeText={setInput}
              placeholder="Type a command…"
              placeholderTextColor={colors.textDim}
              autoCapitalize="none"
              autoCorrect={false}
              returnKeyType="send"
              blurOnSubmit={false}
              onSubmitEditing={send}
              onKeyPress={event => {
                const key = event.nativeEvent.key;
                if (key === 'Tab') complete();
                else if (key === 'ArrowUp') cycleHistory(-1);
                else if (key === 'ArrowDown') cycleHistory(1);
              }}
              editable={connected}
            />
            <TouchableOpacity onPress={complete} disabled={!connected} style={styles.keyButton} accessibilityLabel="Complete shell input">
              <Text variant="xs" style={{ color: colors.textDim, fontWeight: '700' }}>TAB</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => cycleHistory(-1)} disabled={!connected} style={styles.keyButton} accessibilityLabel="Previous shell command">
              <Ionicons name="chevron-up" size={17} color={colors.textDim} />
            </TouchableOpacity>
            <TouchableOpacity onPress={() => cycleHistory(1)} disabled={!connected} style={styles.keyButton} accessibilityLabel="Next shell command">
              <Ionicons name="chevron-down" size={17} color={colors.textDim} />
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      ) : (
        <View style={styles.emptyWrap}>
          <Ionicons name="terminal-outline" size={48} color={colors.textDim} />
          <Text variant="sm" style={{ color: colors.textDim, marginTop: spacing.sm, textAlign: 'center' }}>
            {'No shell connected.\nTap the connect button to open an interactive terminal into the session container.'}
          </Text>
        </View>
      )}
    </SafeAreaView>
  );
}
