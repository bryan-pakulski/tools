import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  PanResponder,
  RefreshControl,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { describeSessionLoadError, formatSessionLoadProblem, sessionsApi, type SessionSummary } from '../api/sessions';
import { threadsApi, type ThreadStatus, type ThreadSummary } from '../api/threads';
import { useConnectionStore } from '../store/connection';
import { useTheme } from '../theme/ThemeContext';
import { Text } from './Text';
import { NewSessionSheet } from './NewSessionSheet';
import { NewThreadSheet } from './NewThreadSheet';
import { SafeAreaModal } from './SafeAreaModal';

export type SwipeSessionsDrawerProps = {
  visible: boolean;
  onClose: () => void;
  createRequestToken?: number;
};

const ACTIVE_THREAD_STATUSES: ThreadStatus[] = ['running', 'waiting_peer', 'awaiting_approval'];

function threadStatusLabel(status: ThreadStatus): string {
  return status.replace(/_/g, ' ');
}

export function SwipeSessionsDrawer({ visible, onClose, createRequestToken = 0 }: SwipeSessionsDrawerProps) {
  const { colors } = useTheme();
  const { activeSessionName, setActiveSession, setActiveProviderModel } = useConnectionStore();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [createSessionOpen, setCreateSessionOpen] = useState(false);
  const [createThreadOpen, setCreateThreadOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [switchingName, setSwitchingName] = useState<string | null>(null);

  const swipeResponder = useMemo(
    () =>
      PanResponder.create({
        onMoveShouldSetPanResponder: (_event, gesture) =>
          gesture.dx < -10 && Math.abs(gesture.dx) > Math.abs(gesture.dy) * 1.2,
        onPanResponderRelease: (_event, gesture) => {
          if (gesture.dx < -64) onClose();
        },
      }),
    [onClose],
  );

  const load = useCallback(async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true);
      const currentSession = useConnectionStore.getState().activeSessionName;
      const [sessionResponse, threadResponse] = await Promise.all([
        sessionsApi.list({ timeoutMs: 8_000 }),
        currentSession
          ? threadsApi.list(currentSession, { timeoutMs: 8_000 }).catch(() => null)
          : Promise.resolve(null),
      ]);
      setSessions(sessionResponse.sessions);
      setThreads(threadResponse?.threads || []);
      setLoadError(null);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (!visible) return undefined;
    void load();
    const poll = setInterval(() => void load(false), 3_000);
    return () => clearInterval(poll);
  }, [activeSessionName, load, visible]);

  useEffect(() => {
    if (createRequestToken > 0) setCreateSessionOpen(true);
  }, [createRequestToken]);

  const hasThreadGroup = threads.length > 1;

  const switchSession = async (name: string) => {
    if (switchingName || name === activeSessionName) {
      if (name === activeSessionName) onClose();
      return;
    }
    setSwitchingName(name);
    onClose();
    try {
      const session = sessions.find(item => item.name === name);
      if (!session?.is_loaded) {
        await sessionsApi.load(name, undefined, undefined, { timeoutMs: 30_000 });
      } else {
        await sessionsApi.focus(name, { timeoutMs: 8_000 });
      }
      setActiveSession(name);
    } catch (error) {
      const problem = describeSessionLoadError(error);
      Alert.alert(problem.title, formatSessionLoadProblem(problem));
    } finally {
      setSwitchingName(null);
    }
  };

  const sessionCreated = (session: { name: string; provider: string; model: string }) => {
    setActiveSession(session.name);
    setActiveProviderModel(session.provider, session.model);
    setCreateSessionOpen(false);
    onClose();
  };

  const threadCreated = (thread: { session_name: string }) => {
    setActiveSession(thread.session_name);
    setCreateThreadOpen(false);
    onClose();
  };

  const unloadSession = (session: SessionSummary) => {
    Alert.alert('Unload session?', `Unload “${session.name}” from memory?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Unload',
        onPress: async () => {
          await sessionsApi.unload(session.name);
          if (activeSessionName === session.name) {
            setActiveSession(null);
            setActiveProviderModel(null, null);
            onClose();
          }
          void load();
        },
      },
    ]);
  };

  const deleteSession = (session: SessionSummary) => {
    Alert.alert('Delete session?', `Permanently delete “${session.name}”?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await sessionsApi.delete(session.name);
          void load();
        },
      },
    ]);
  };

  const deleteThread = (thread: ThreadSummary) => {
    if (!activeSessionName || thread.session_name === activeSessionName) return;
    Alert.alert('Delete thread?', `Permanently delete “${thread.title || thread.session_name}” and its conversation?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            const session = sessions.find(item => item.name === thread.session_name);
            if (session?.is_loaded) await sessionsApi.unload(thread.session_name);
            await threadsApi.remove(thread.thread_id, activeSessionName, { timeoutMs: 12_000 });
            void load();
          } catch (cause) {
            Alert.alert('Thread could not be deleted', cause instanceof Error ? cause.message : String(cause));
          }
        },
      },
    ]);
  };

  const sessionStatusColor = (session: SessionSummary) => {
    if (session.is_busy) return colors.accent;
    if (session.is_loaded) return colors.textSoft;
    return colors.textDim;
  };

  const threadStatusColor = (status: ThreadStatus) => {
    if (status === 'error') return colors.error;
    if (status === 'interrupted') return colors.warning;
    if (ACTIVE_THREAD_STATUSES.includes(status)) return colors.accent;
    return colors.textDim;
  };

  const typeIcon = (session: SessionSummary): keyof typeof Ionicons.glyphMap => {
    if (session.session_type === 'container') return 'cube-outline';
    if (session.session_type === 'chat') return 'chatbubble-ellipses-outline';
    return 'folder-open-outline';
  };

  const drawerVisible = visible && !createSessionOpen && !createThreadOpen;

  return (
    <>
      <SafeAreaModal visible={drawerVisible} transparent animationType="fade" onRequestClose={onClose}>
        <View style={styles.overlay}>
          <View
            {...swipeResponder.panHandlers}
            style={[
              styles.drawer,
              {
                backgroundColor: colors.glassStrong,
                borderRightColor: colors.hairline,
                paddingTop: 16,
              },
            ]}
          >
            <View style={[styles.header, { borderBottomColor: colors.hairline }] }>
              <View style={styles.headerCopy}>
                <Text style={[styles.title, { color: colors.text }]}>Threads</Text>
                <Text variant="xs" dim>Agent conversations and sessions</Text>
              </View>
              <View style={styles.headerActions}>
                <TouchableOpacity
                  onPress={() => activeSessionName ? setCreateThreadOpen(true) : setCreateSessionOpen(true)}
                  style={styles.iconButton}
                  accessibilityLabel={activeSessionName ? 'Create thread' : 'Create session'}
                >
                  <Ionicons name="add" size={20} color={colors.textDim} />
                </TouchableOpacity>
                <TouchableOpacity onPress={onClose} style={styles.iconButton} accessibilityLabel="Close threads">
                  <Ionicons name="close" size={20} color={colors.textDim} />
                </TouchableOpacity>
              </View>
            </View>

            <ScrollView
              refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => { setRefreshing(true); void load(); }} />}
              contentContainerStyle={styles.listContent}
            >
              {activeSessionName ? (
                <TouchableOpacity
                  onPress={() => setCreateThreadOpen(true)}
                  activeOpacity={0.72}
                  style={[styles.primaryAction, { backgroundColor: colors.accentSoft }]}
                >
                  <Ionicons name="git-branch-outline" size={18} color={colors.accent} />
                  <Text variant="sm" style={{ color: colors.accent, fontWeight: '700' }}>New thread</Text>
                  <Ionicons name="add" size={18} color={colors.accent} />
                </TouchableOpacity>
              ) : null}

              {hasThreadGroup ? (
                <View style={styles.section}>
                  <SectionTitle label="THREAD GROUP" count={threads.length} />
                  {threads.map(thread => (
                    <TouchableOpacity
                      key={thread.thread_id}
                      onPress={() => void switchSession(thread.session_name)}
                      activeOpacity={0.7}
                      style={[
                        styles.row,
                        { borderBottomColor: colors.hairline },
                        activeSessionName === thread.session_name && { backgroundColor: colors.bgHover },
                      ]}
                    >
                      <View style={styles.typeIconWrap}>
                        <View style={[styles.threadDot, { backgroundColor: threadStatusColor(thread.status) }]} />
                      </View>
                      <View style={styles.rowCopy}>
                        <Text variant="sm" style={styles.rowName} numberOfLines={1}>
                          {thread.title || thread.session_name}
                        </Text>
                        <Text variant="xs" dim numberOfLines={2}>
                          {switchingName === thread.session_name ? 'Opening…' : threadStatusLabel(thread.status)}
                          {thread.current_goal ? ` · ${thread.current_goal}` : ''}
                        </Text>
                      </View>
                      {thread.unread_count > 0 ? (
                        <View style={[styles.unreadBadge, { backgroundColor: colors.accentSoft }] }>
                          <Text variant="xs" style={{ color: colors.accent, fontWeight: '700' }}>{thread.unread_count}</Text>
                        </View>
                      ) : null}
                      {activeSessionName === thread.session_name ? (
                        <Ionicons name="checkmark" size={17} color={colors.accent} />
                      ) : !ACTIVE_THREAD_STATUSES.includes(thread.status) ? (
                        <TouchableOpacity
                          onPress={() => deleteThread(thread)}
                          style={styles.rowAction}
                          accessibilityLabel={`Delete thread ${thread.title || thread.session_name}`}
                        >
                          <Ionicons name="trash-outline" size={17} color={colors.textDim} />
                        </TouchableOpacity>
                      ) : (
                        <Ionicons name="chevron-forward" size={17} color={colors.textDim} />
                      )}
                    </TouchableOpacity>
                  ))}
                </View>
              ) : null}

              <View style={styles.section}>
                <SectionTitle label="SESSIONS" count={sessions.length} />
                {!loading && sessions.length === 0 ? (
                  <View style={styles.empty}>
                    <Text variant="sm" dim>{loadError || 'No saved sessions'}</Text>
                  </View>
                ) : null}
                {loadError ? (
                  <TouchableOpacity onPress={() => void load()} style={styles.retryAction}>
                    <Text variant="sm" style={{ color: colors.accent, fontWeight: '600' }}>Retry</Text>
                  </TouchableOpacity>
                ) : null}
                {sessions.map(session => (
                  <TouchableOpacity
                    key={session.name}
                    onPress={() => void switchSession(session.name)}
                    activeOpacity={0.7}
                    style={[
                      styles.row,
                      { borderBottomColor: colors.hairline },
                      activeSessionName === session.name && { backgroundColor: colors.bgHover },
                    ]}
                  >
                    <View style={styles.typeIconWrap}>
                      <Ionicons name={typeIcon(session)} size={17} color={sessionStatusColor(session)} />
                    </View>
                    <View style={styles.rowCopy}>
                      <Text variant="sm" style={styles.rowName} numberOfLines={1}>{session.name}</Text>
                      <Text variant="xs" dim>
                        {switchingName === session.name ? 'Opening…' : session.is_busy ? 'Working' : session.is_loaded ? 'Loaded' : 'Saved'}
                        {session.session_type ? ` · ${session.session_type}` : ''}
                      </Text>
                    </View>
                    <TouchableOpacity onPress={() => unloadSession(session)} style={styles.rowAction} accessibilityLabel={`Unload ${session.name}`}>
                      <Ionicons name="remove-circle-outline" size={18} color={colors.textDim} />
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => deleteSession(session)} style={styles.rowAction} accessibilityLabel={`Delete ${session.name}`}>
                      <Ionicons name="trash-outline" size={17} color={colors.textDim} />
                    </TouchableOpacity>
                  </TouchableOpacity>
                ))}
              </View>

              <TouchableOpacity
                onPress={() => setCreateSessionOpen(true)}
                activeOpacity={0.72}
                style={[styles.secondaryAction, { borderColor: colors.hairline }]}
              >
                <Ionicons name="add-circle-outline" size={17} color={colors.textDim} />
                <Text variant="sm" dim style={{ fontWeight: '600' }}>New session</Text>
              </TouchableOpacity>
            </ScrollView>
          </View>
          <TouchableOpacity style={styles.backdrop} onPress={onClose} activeOpacity={1} />
        </View>
      </SafeAreaModal>
      <NewSessionSheet
        visible={createSessionOpen}
        onClose={() => setCreateSessionOpen(false)}
        onCreated={sessionCreated}
      />
      <NewThreadSheet
        visible={createThreadOpen}
        parentSessionName={activeSessionName}
        onClose={() => setCreateThreadOpen(false)}
        onCreated={threadCreated}
      />
    </>
  );
}

function SectionTitle({ label, count }: { label: string; count: number }) {
  const { colors } = useTheme();
  return (
    <View style={styles.sectionTitleRow}>
      <Text variant="xs" style={[styles.sectionTitle, { color: colors.textDim }]}>{label}</Text>
      <Text variant="xs" dim>{count}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: { flex: 1, flexDirection: 'row' },
  backdrop: { flex: 1, backgroundColor: 'rgba(5,10,16,0.42)' },
  drawer: {
    width: '88%',
    maxWidth: 380,
    borderRightWidth: StyleSheet.hairlineWidth,
    elevation: 4,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 18,
    shadowOffset: { width: 6, height: 0 },
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 18,
    paddingVertical: 15,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  headerCopy: { flex: 1 },
  headerActions: { flexDirection: 'row', gap: 2 },
  title: { fontSize: 19, fontWeight: '600', letterSpacing: -0.25 },
  iconButton: { width: 40, height: 40, alignItems: 'center', justifyContent: 'center' },
  listContent: { paddingHorizontal: 10, paddingBottom: 24 },
  primaryAction: {
    minHeight: 46,
    marginHorizontal: 4,
    marginTop: 12,
    paddingHorizontal: 13,
    borderRadius: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  secondaryAction: {
    minHeight: 44,
    marginHorizontal: 4,
    marginTop: 18,
    paddingHorizontal: 13,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  section: { marginTop: 20 },
  sectionTitleRow: {
    minHeight: 28,
    paddingHorizontal: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  sectionTitle: { fontFamily: 'monospace', fontWeight: '700', letterSpacing: 1 },
  empty: { paddingHorizontal: 8, paddingVertical: 18, alignItems: 'center' },
  retryAction: { alignSelf: 'center', paddingHorizontal: 12, paddingVertical: 8 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: 58,
    paddingHorizontal: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  typeIconWrap: { width: 28, alignItems: 'flex-start', justifyContent: 'center', marginRight: 8 },
  threadDot: { width: 8, height: 8, borderRadius: 4 },
  rowCopy: { flex: 1, paddingVertical: 8 },
  rowName: { fontWeight: '600' },
  rowAction: { width: 38, height: 44, alignItems: 'center', justifyContent: 'center' },
  unreadBadge: {
    minWidth: 23,
    height: 23,
    paddingHorizontal: 6,
    marginHorizontal: 7,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
