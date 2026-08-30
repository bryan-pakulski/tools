import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import {
  View,
  FlatList,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Clipboard from 'expo-clipboard';
import Markdown, { RenderRules } from 'react-native-markdown-display';
import { openExternalUrl } from '../api/urlSafety';
import { useTheme } from '../theme/ThemeContext';
import { useConnectionStore } from '../store/connection';
import { modesApi, ModeInfo } from '../api/modes';
import { inspectorApi, InspectorVariableGroup } from '../api/inspector';
import { Text, Button, Card, Skeleton, EmptyState, ErrorState } from '../components';
import { BottomSheet } from '../components/BottomSheet';
import { GeneratingIndicator } from '../components/GeneratingIndicator';
import { ArtifactStrip } from '../components/ArtifactStrip';
import { CodeBlock } from '../components/CodeBlock';
import { VisualizationCard } from '../components/VisualizationCard';
import { SubagentActivityPanel } from '../components/SubagentActivityPanel';
import { AttachmentSheet } from '../components/AttachmentSheet';
import type { AttachmentDescriptor } from '../api/attachments';
import { useChatSession, type ChatMessage } from '../hooks/useChatSession';
import { useCommandCompletion, type CompletionItem } from '../hooks/useCommandCompletion';
import { CommandSuggestionBar } from '../components/CommandSuggestionBar';
import { ConflictBanner } from '../components/ConflictBanner';
import { spacing } from '../theme/tokens';

export function ChatScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const activeSessionName = useConnectionStore(state => state.activeSessionName);
  const {
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
  } = useChatSession(activeSessionName);
  const [input, setInput] = useState('');
  const [modeSheetOpen, setModeSheetOpen] = useState(false);
  const [modes, setModes] = useState<ModeInfo[]>([]);
  const [activeMode, setActiveMode] = useState<string | null>(null);
  const [settingsSheetOpen, setSettingsSheetOpen] = useState(false);
  const [varGroups, setVarGroups] = useState<InspectorVariableGroup[]>([]);
  const [varsLoading, setVarsLoading] = useState(false);
  const [attachmentsOpen, setAttachmentsOpen] = useState(false);
  const [selectedAttachments, setSelectedAttachments] = useState<AttachmentDescriptor[]>([]);
  const activeProvider = useConnectionStore(state => state.activeProvider);
  const activeModel = useConnectionStore(state => state.activeModel);
  const yolo = useConnectionStore(state => state.yolo);
  const connection = { activeProvider, activeModel, yolo };
  const flatListRef = useRef<FlatList<ChatMessage>>(null);
  const scrollThrottleRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollEndTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const initialScrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // MUCLI_MOBILE_SCROLL_BOUNDARY_V2: FlatList owns prepend anchoring; do not continuously
  // rewrite offsets from incremental content-size measurements.
  const followOutputRef = useRef(true);
  const initialScrollPendingRef = useRef(true);
  const userScrollActiveRef = useRef(false);
  const momentumScrollRef = useRef(false);
  const lastDistanceFromEndRef = useRef(Number.POSITIVE_INFINITY);
  const streamingRef = useRef(false);
  // MUCLI_SLIDING_WINDOW_V1: guard backward pagination while FlatList's
  // maintainVisibleContentPosition preserves the currently visible message.
  const loadingOlderTriggeredRef = useRef(false);

  // MUCLI_VISUALIZATION_TIMELINE_V2: a WebView gesture explicitly pauses chat
  // following. Releasing the WebView never silently opts the chat back in.
  const onVisualizationInteractionChange = useCallback((active: boolean) => {
    userScrollActiveRef.current = active;
    if (active) followOutputRef.current = false;
    else if (lastDistanceFromEndRef.current <= 64) followOutputRef.current = true;
  }, []);
  const completion = useCommandCompletion();
  // Keep a ref mirror of the streaming state so onContentSizeChange can decide
  // whether to auto-follow without re-creating its callback identity.
  useEffect(() => { streamingRef.current = streaming; }, [streaming]);

  // Follow streaming output only while the user remains parked near the
  // bottom. Coalesced scroll requests avoid fighting FlatList layout.
  // Doubles as the FlatList onContentSizeChange/onLayout handler: RN passes
  // (width, height), which is not `true`, so it degrades to a follow scroll.
  const scrollToBottom = useCallback((force?: boolean | number | unknown) => {
    const isForce = force === true;
    if (isForce) followOutputRef.current = true;
    if (
      !isForce
      && (!followOutputRef.current || userScrollActiveRef.current || momentumScrollRef.current)
    ) return;
    if (scrollThrottleRef.current) return;
    scrollThrottleRef.current = setTimeout(() => {
      scrollThrottleRef.current = null;
      if (
        isForce
        || (followOutputRef.current && !userScrollActiveRef.current && !momentumScrollRef.current)
      ) {
        flatListRef.current?.scrollToEnd({ animated: false });
      }
    }, isForce ? 0 : 32);
  }, []);

  const cancelInitialScroll = useCallback(() => {
    if (initialScrollTimerRef.current) {
      clearTimeout(initialScrollTimerRef.current);
      initialScrollTimerRef.current = null;
    }
    initialScrollPendingRef.current = false;
  }, []);

  const updateFollowFromDistance = useCallback((distanceFromEnd: number) => {
    lastDistanceFromEndRef.current = distanceFromEnd;
    if (userScrollActiveRef.current || momentumScrollRef.current) {
      followOutputRef.current = false;
      return;
    }
    if (distanceFromEnd <= 64) followOutputRef.current = true;
    else if (distanceFromEnd >= 144) followOutputRef.current = false;
  }, []);

  const onChatScroll = useCallback((event: any) => {
    if (!event?.nativeEvent) return;
    const { contentOffset, contentSize, layoutMeasurement } = event.nativeEvent;
    const distanceFromEnd = Math.max(
      0,
      contentSize.height - layoutMeasurement.height - contentOffset.y,
    );
    updateFollowFromDistance(distanceFromEnd);

    // MUCLI_SLIDING_WINDOW_V1: trigger backward pagination when the user
    // scrolls near the top. Only fire once per page — the guard ref prevents
    // re-entry while the request is in flight AND while the scroll adjustment
    // is pending. The guard is NOT cleared here — it's cleared in
    // onChatContentSizeChange after the scroll offset has been applied.
    // Clearing it in .finally() was too early: React re-renders with the
    // prepended messages AFTER the promise resolves, and FlatList fires
    // onScroll with offset~0 (content grew at top) → guard already false →
    // hasMore still true → re-triggers → infinite scroll loop.
    const distanceFromTop = contentOffset.y;
    if (
      distanceFromTop < 120 &&
      hasMore &&
      !loadingOlder &&
      !loadingOlderTriggeredRef.current
    ) {
      loadingOlderTriggeredRef.current = true;
      // FlatList preserves the first visible row while older messages prepend.
      // Clear the request guard only after the page request has settled.
      void loadOlderHistory().finally(() => {
        loadingOlderTriggeredRef.current = false;
      });
    }

  }, [hasMore, loadingOlder, loadOlderHistory, updateFollowFromDistance]);

  const onChatScrollBeginDrag = useCallback(() => {
    cancelInitialScroll();
    if (scrollEndTimerRef.current) {
      clearTimeout(scrollEndTimerRef.current);
      scrollEndTimerRef.current = null;
    }
    userScrollActiveRef.current = true;
    momentumScrollRef.current = false;
    followOutputRef.current = false;
  }, [cancelInitialScroll]);

  const finishUserScroll = useCallback(() => {
    userScrollActiveRef.current = false;
    momentumScrollRef.current = false;
    followOutputRef.current = lastDistanceFromEndRef.current <= 64;
  }, []);

  const onChatScrollEndDrag = useCallback(() => {
    if (scrollEndTimerRef.current) clearTimeout(scrollEndTimerRef.current);
    // Momentum begins immediately after end-drag when present. Delay the
    // no-momentum decision so that event can take ownership first.
    scrollEndTimerRef.current = setTimeout(() => {
      scrollEndTimerRef.current = null;
      if (!momentumScrollRef.current) finishUserScroll();
    }, 80);
  }, [finishUserScroll]);

  const onChatMomentumScrollBegin = useCallback(() => {
    if (scrollEndTimerRef.current) {
      clearTimeout(scrollEndTimerRef.current);
      scrollEndTimerRef.current = null;
    }
    momentumScrollRef.current = true;
    userScrollActiveRef.current = true;
    followOutputRef.current = false;
  }, []);

  const onChatMomentumScrollEnd = useCallback(() => {
    finishUserScroll();
  }, [finishUserScroll]);

  const onChatContentSizeChange = useCallback((_width: number, newHeight: number) => {
    if (!Number.isFinite(newHeight)) return;

    // Debounce the initial jump to the bottom until the current virtualization
    // batch has stopped changing size. A drag cancels this timer permanently.
    if (
      initialScrollPendingRef.current
      && messages.length > 0
      && !userScrollActiveRef.current
      && !momentumScrollRef.current
    ) {
      if (initialScrollTimerRef.current) clearTimeout(initialScrollTimerRef.current);
      initialScrollTimerRef.current = setTimeout(() => {
        initialScrollTimerRef.current = null;
        if (userScrollActiveRef.current || momentumScrollRef.current) return;
        flatListRef.current?.scrollToEnd({ animated: false });
        initialScrollPendingRef.current = false;
      }, 80);
      return;
    }

    if (!streamingRef.current || !followOutputRef.current) return;
    scrollToBottom(false);
  }, [messages.length, scrollToBottom]);

  // MUCLI_SCROLL_LOCK_V1: onChatLayout previously cleared
  // initialScrollPendingRef here — but FlatList fires onLayout before all
  // cells have mounted (virtualization mounts them in batches). Clearing
  // the flag here meant onContentSizeChange's initial-scroll branch was
  // already skipped → the list stayed at the top instead of scrolling to
  // the bottom on session entry. Now we only clear it in
  // onContentSizeChange after the scroll has actually happened.
  const onChatLayout = useCallback(() => {
    // No-op — initial scroll is handled by onContentSizeChange which fires
    // after content has actually been laid out.
  }, []);

  useEffect(() => {
    if (initialScrollTimerRef.current) clearTimeout(initialScrollTimerRef.current);
    initialScrollPendingRef.current = true;
    followOutputRef.current = true;
    userScrollActiveRef.current = false;
    momentumScrollRef.current = false;
    lastDistanceFromEndRef.current = Number.POSITIVE_INFINITY;
    loadingOlderTriggeredRef.current = false;
  }, [activeSessionName]);

  // Clear pending scroll work on unmount.
  useEffect(() => () => {
    if (scrollThrottleRef.current) clearTimeout(scrollThrottleRef.current);
    if (scrollEndTimerRef.current) clearTimeout(scrollEndTimerRef.current);
    if (initialScrollTimerRef.current) clearTimeout(initialScrollTimerRef.current);
  }, []);

  const send = async () => {
    cancelInitialScroll();
    const text = input.trim();
    if ((!text && selectedAttachments.length === 0) || streaming) return;
    followOutputRef.current = true;
    initialScrollPendingRef.current = false;
    const sent = await sendMessage(text, selectedAttachments);
    if (sent) {
      setInput('');
      setSelectedAttachments([]);
      completion.close();
      scrollToBottom(true);
    }
  };

  const onInputChange = (text: string) => {
    setInput(text);
    if (text.startsWith('/')) {
      completion.update(text);
    } else if (completion.visible) {
      completion.close();
    }
  };

  const onAcceptCompletion = (item: CompletionItem) => {
    const newText = item.value + ' ';
    setInput(newText);
    // If the command has subcommands, keep the dropdown open for next level.
    if (item.level === 0) {
      completion.update(newText);
    } else {
      completion.close();
    }
  };

  const loadModes = useCallback(async () => {
    try {
      const res = await modesApi.list();
      setModes(res.modes);
      setActiveMode(res.current);
    } catch { /* ignore */ }
  }, []);

  const selectMode = async (name: string) => {
    try {
      await modesApi.set(name);
      setActiveMode(name);
      setModeSheetOpen(false);
    } catch { /* hook will surface session errors */ }
  };

  const loadVariables = useCallback(async () => {
    setVarsLoading(true);
    try {
      const res = await inspectorApi.getVariables();
      setVarGroups(res.groups);
    } catch { /* ignore */ }
    setVarsLoading(false);
  }, []);

  const setVariable = async (key: string, value: unknown) => {
    try { await inspectorApi.setVariable(key, value); } catch { /* ignore */ }
  };

  const copyMessage = useCallback((text: string) => {
    Clipboard.setStringAsync(text);
  }, []);

  // Messages render at full length — CodeBlock handles long code via
  // internal ScrollView with maxHeight, and Markdown handles long text
  // natively. No message-level truncation needed.

  // Safe link handler: only http(s) URLs reach the OS. Everything else is
  // blocked (with an alert) instead of being handed to Linking.
  const handleMarkdownLinkPress = useCallback((url: string) => {
    void openExternalUrl(url);
    return false;
  }, []);

  const markdownRules = useMemo<RenderRules>(
    () => ({
      fence: (node) => {
        const code = node.content;
        // sourceInfo (fence language) is attached by tokensToAST but missing
        // from the library's ASTNode type declaration.
        const lang = ((node as { sourceInfo?: string }).sourceInfo || '').trim();
        return (
          <CodeBlock
            key={node.key}
            code={code}
            language={lang}
            colors={colors}
          />
        );
      },
      code_block: (node) => {
        const code = node.content;
        return (
          <CodeBlock
            key={node.key}
            code={code}
            colors={colors}
          />
        );
      },
      code_inline: (node) => {
        return (
          <Text key={node.key} style={{ color: colors.syntax.keyword, fontFamily: 'monospace', fontSize: 13, backgroundColor: colors.bgHover, borderRadius: 4, paddingHorizontal: 4 }}>
            {node.content}
          </Text>
        );
      },
    }),
    [colors],
  );

  const memoizedMarkdownStyles = useMemo(() => markdownStyles(colors), [colors]);

  const renderMessage = useCallback(({ item }: { item: ChatMessage }) => {
    if (item.role === 'visualization' && item.artifact && activeSessionName) {
      return (
        <VisualizationCard
          artifact={item.artifact}
          sessionName={activeSessionName}
          onInteractionChange={onVisualizationInteractionChange}
        />
      );
    }

    if (item.role === 'subagent_panel' && item.subagents?.length) {
      return <SubagentActivityPanel agents={item.subagents} />;
    }

    // Collapsible heading for all interim agent output.
    if (item.role === 'collapse') {
      const count = item.collapseCount || (item.childTurns?.length || 0);
      return (
        <View
          // MUCLI_INTERIM_VISUAL_STYLE_V1: interim output uses an accent-tinted surface.
          style={[
            styles.interimGroup,
            { backgroundColor: colors.accentSoft, borderColor: colors.border },
          ]}
        >
          <TouchableOpacity
            onPress={() => setMessages(current => current.map(m =>
              m.id === item.id ? { ...m, collapseOpen: !m.collapseOpen } : m,
            ))}
            style={[styles.collapseHeader, { backgroundColor: colors.bgLift, borderColor: colors.border }]}
            activeOpacity={0.7}
          >
            <View style={[styles.interimAccent, { backgroundColor: colors.accent }]} />
            <Text style={{ fontSize: 12, color: colors.textDim }}>
              {item.collapseOpen ? '▾' : '▸'}
            </Text>
            <Text style={{ fontSize: 13, color: colors.textSoft, fontFamily: 'monospace' }}>
              {count} interim update{count !== 1 ? 's' : ''}
              {item.collapseElapsed ? ` · ${item.collapseElapsed}` : ''}
              {item.collapseTokens ? ` · ${item.collapseTokens}` : ''}
            </Text>
          </TouchableOpacity>
          {item.collapseOpen ? (
            <View style={styles.interimBody}>
              {item.childTurns?.map(child => {
                if (child.role !== 'assistant') return null;
                return (
                  <View key={child.id} style={[styles.msgRow, { justifyContent: 'flex-start' }]}>
                    <View
                      style={[
                        styles.msgBubble,
                        styles.interimMessage,
                        { backgroundColor: colors.bgLift, borderColor: colors.border, maxWidth: '100%' },
                      ]}
                    >
                      {child.streaming ? (
                        <Text style={{ color: colors.text, fontSize: 15, lineHeight: 23 }}>
                          {child.text}
                        </Text>
                      ) : (
                        <Markdown style={memoizedMarkdownStyles} rules={markdownRules} onLinkPress={handleMarkdownLinkPress}>
                          {child.text}
                        </Markdown>
                      )}
                      {child.text.length > 0 ? (
                        <TouchableOpacity
                          onPress={() => copyMessage(child.text)}
                          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                          style={styles.copyButton}
                        >
                          <Ionicons name="copy-outline" size={14} color={colors.textDim} />
                        </TouchableOpacity>
                      ) : null}
                    </View>
                  </View>
                );
              })}
            </View>
          ) : null}
        </View>
      );
    }

    const isUser = item.role === 'user';
    const isAssistant = item.role === 'assistant';

    // Full text — no truncation. CodeBlock scrolls internally; Markdown
    // wraps naturally. Copy button preserves the raw text either way.
    const displayText = item.text;

    return (
      <View style={[
        styles.msgRow,
        isUser ? { justifyContent: 'flex-end' } : { justifyContent: 'flex-start' },
      ]}>
        <View
          style={[
            styles.msgBubble,
            isUser
              ? { backgroundColor: colors.bgHover, maxWidth: '84%' }
              : { backgroundColor: 'transparent', maxWidth: '100%', paddingHorizontal: 0 },
          ]}
        >
          {isUser ? (
            <Text style={{ color: colors.text }}>{displayText}</Text>
          ) : item.streaming ? (
            <Text style={{ color: colors.text, fontSize: 15, lineHeight: 23 }}>
              {displayText}
            </Text>
          ) : (
            <Markdown
              style={memoizedMarkdownStyles}
              rules={markdownRules}
              onLinkPress={handleMarkdownLinkPress}
            >
              {displayText}
            </Markdown>
          )}
          {isUser && item.attachments && item.attachments.length > 0 ? (
            <View style={styles.messageAttachments}>
              {item.attachments.map(attachment => (
                <View key={attachment.attachment_id} style={[styles.messageAttachment, { backgroundColor: colors.bgLift }]}>
                  <Ionicons name="document-outline" size={14} color={colors.textDim} />
                  <Text variant="xs" numberOfLines={1} style={{ color: colors.textSoft, maxWidth: 210 }}>{attachment.name}</Text>
                </View>
              ))}
            </View>
          ) : null}
          {isAssistant && item.text.length > 0 && (
            <TouchableOpacity
              onPress={() => copyMessage(item.text)}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              style={styles.copyButton}
            >
              <Ionicons name="copy-outline" size={14} color={colors.textDim} />
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  }, [activeSessionName, colors, copyMessage, markdownRules, memoizedMarkdownStyles, onVisualizationInteractionChange]);

  return (
    <SafeAreaView edges={['bottom']} style={{ flex: 1, backgroundColor: colors.bg }}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={insets.bottom}
      >
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={item => item.id}
          renderItem={renderMessage}
          initialNumToRender={20}
          maxToRenderPerBatch={10}
          updateCellsBatchingPeriod={32}
          windowSize={9}
          removeClippedSubviews={false}
          maintainVisibleContentPosition={{ minIndexForVisible: 0, autoscrollToTopThreshold: 100 }}
          bounces={false}
          alwaysBounceVertical={false}
          overScrollMode="never"
          contentInsetAdjustmentBehavior="never"
          contentContainerStyle={[
            styles.messageList,
            messages.length === 0 ? styles.messageListEmpty : null,
          ]}
          keyboardShouldPersistTaps="always"
          keyboardDismissMode="none"
          onScroll={onChatScroll}
          onScrollBeginDrag={onChatScrollBeginDrag}
          onScrollEndDrag={onChatScrollEndDrag}
          onMomentumScrollBegin={onChatMomentumScrollBegin}
          onMomentumScrollEnd={onChatMomentumScrollEnd}
          scrollEventThrottle={16}
          onContentSizeChange={scrollToBottom}
          onLayout={scrollToBottom}
          ListHeaderComponent={
            <View>
              {loadingOlder ? (
                <View style={styles.loadingOlderIndicator}>
                  <GeneratingIndicator label="Loading older messages" />
                </View>
              ) : null}
              {error && messages.length > 0 ? (
                <View style={[styles.inlineError, { backgroundColor: colors.bgHover }]}>
                  <Ionicons name="warning-outline" size={15} color={colors.error} />
                  <Text variant="xs" style={{ color: colors.textSoft, flex: 1 }}>{error}</Text>
                  <TouchableOpacity onPress={retry} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
                    <Text variant="xs" style={{ color: colors.accent, fontWeight: '600' }}>Retry</Text>
                  </TouchableOpacity>
                </View>
              ) : null}
            </View>
          }
          ListEmptyComponent={
            historyLoading ? (
              <View style={styles.historyLoading}>
                <Skeleton height={18} style={{ marginBottom: 10, width: '72%' }} />
                <Skeleton height={18} style={{ marginBottom: 10, width: '88%' }} />
                <Skeleton height={18} style={{ width: '56%' }} />
              </View>
            ) : error ? (
              <ErrorState message={error} onRetry={retry} />
            ) : (
              <EmptyState
                icon="sparkles-outline"
                title="What should we build?"
                message="Ask MuCLI to inspect, explain, debug, or change your workspace."
                actionLabel={activeMode ? `Mode: ${activeMode}` : 'Select Mode'}
                onAction={() => { loadModes(); setModeSheetOpen(true); }}
              />
            )
          }
          ListFooterComponent={
            <View style={styles.listFooter}>
              {streaming && waitingForFirstToken ? (
                <GeneratingIndicator label={sseConnected ? activityLabel : 'Reconnecting to session'} />
              ) : (
                <View style={styles.listEndMarker} />
              )}
            </View>
          }
        />
        <ConflictBanner style={{ marginHorizontal: spacing.base, marginBottom: spacing.sm }} />
        <ArtifactStrip sessionName={activeSessionName} refreshKey={artifactRevision} />
        <CommandSuggestionBar
          visible={completion.visible}
          items={completion.items}
          selectedIdx={completion.selectedIdx}
          onSelect={onAcceptCompletion}
        />
        <Composer
          input={input}
          setInput={onInputChange}
          onSend={send}
          onStop={stop}
          streaming={streaming}
          colors={colors}
          insets={insets}
          onModePress={() => { loadModes(); setModeSheetOpen(true); }}
          onSettingsPress={() => { loadVariables(); setSettingsSheetOpen(true); }}
          onAttachmentsPress={() => setAttachmentsOpen(true)}
          selectedAttachments={selectedAttachments}
          onRemoveAttachment={(attachmentId) => setSelectedAttachments(current => current.filter(item => item.attachment_id !== attachmentId))}
        />
        <AttachmentSheet
          visible={attachmentsOpen}
          sessionName={activeSessionName || ''}
          selected={selectedAttachments}
          onSelectedChange={setSelectedAttachments}
          onClose={() => setAttachmentsOpen(false)}
        />
        <ModeBottomSheet
          visible={modeSheetOpen}
          onClose={() => setModeSheetOpen(false)}
          modes={modes}
          activeMode={activeMode}
          onSelect={selectMode}
          colors={colors}
        />
        <SettingsBottomSheet
          visible={settingsSheetOpen}
          onClose={() => setSettingsSheetOpen(false)}
          varGroups={varGroups}
          varsLoading={varsLoading}
          connection={connection}
          onSetVariable={setVariable}
          colors={colors}
        />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

interface ComposerProps {
  input: string;
  setInput: (s: string) => void;
  onSend: () => void;
  onStop: () => void;
  streaming: boolean;
  colors: any;
  insets: { bottom: number };
  onModePress: () => void;
  onSettingsPress: () => void;
  onAttachmentsPress: () => void;
  selectedAttachments: AttachmentDescriptor[];
  onRemoveAttachment: (attachmentId: string) => void;
}

function Composer({ input, setInput, onSend, onStop, streaming, colors, insets, onModePress, onAttachmentsPress, selectedAttachments, onRemoveAttachment }: ComposerProps) {
  return (
    <View>
      {selectedAttachments.length > 0 ? (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.selectedAttachments}>
          {selectedAttachments.map(item => (
            <View key={item.attachment_id} style={[styles.selectedAttachment, { backgroundColor: colors.bgHover }]}>
              <Text variant="xs" numberOfLines={1} style={{ maxWidth: 180 }}>{item.name}</Text>
              <TouchableOpacity onPress={() => onRemoveAttachment(item.attachment_id)}>
                <Ionicons name="close" size={15} color={colors.textDim} />
              </TouchableOpacity>
            </View>
          ))}
        </ScrollView>
      ) : null}
      <View
      style={[
        styles.composer,
        {
          backgroundColor: colors.bgLift,
          borderColor: colors.border,
          marginBottom: Math.max(insets.bottom, 8),
        },
      ]}
    >
      <TouchableOpacity
        onPress={onAttachmentsPress}
        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        style={[styles.composerIconButton, { backgroundColor: selectedAttachments.length ? colors.accentSoft : colors.bgHover }]}
      >
        <Ionicons name="attach" size={19} color={selectedAttachments.length ? colors.accent : colors.textDim} />
      </TouchableOpacity>
      <TouchableOpacity
        onPress={onModePress}
        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        style={[styles.composerIconButton, { backgroundColor: colors.bgHover }]}
      >
        <Ionicons name="options-outline" size={18} color={colors.textDim} />
      </TouchableOpacity>
      <TextInput
        value={input}
        onChangeText={setInput}
        placeholder="Message MuCLI"
        placeholderTextColor={colors.textDim}
        multiline
        style={[styles.input, { color: colors.text }]}
        // Keep focus and the draft keyboard open while a turn is running.
        // Sending is still gated by the composer action and hook busy state.
      />
      {streaming ? (
        <TouchableOpacity
          onPress={onStop}
          style={[styles.sendBtn, { backgroundColor: colors.bgHover }]}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Ionicons name="stop" size={17} color={colors.error} />
        </TouchableOpacity>
      ) : (
        <TouchableOpacity
          onPress={onSend}
          disabled={!input.trim() && selectedAttachments.length === 0}
          style={[styles.sendBtn, { backgroundColor: (input.trim() || selectedAttachments.length) ? colors.accent : colors.bgHover }]}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Ionicons name="arrow-up" size={19} color={(input.trim() || selectedAttachments.length) ? colors.accentText : colors.textDim} />
        </TouchableOpacity>
      )}
      </View>
    </View>
  );
}

interface ModeBottomSheetProps {
  visible: boolean;
  onClose: () => void;
  modes: ModeInfo[];
  activeMode: string | null;
  onSelect: (name: string) => void;
  colors: any;
}

function ModeBottomSheet({ visible, onClose, modes, activeMode, onSelect, colors }: ModeBottomSheetProps) {
  return (
    <BottomSheet visible={visible} onClose={onClose}>
      <View style={{ padding: spacing.sm }}>
        <Text variant="lg" style={{ marginBottom: spacing.base }}>Select Mode</Text>
        {modes.map(m => (
          <TouchableOpacity
            key={m.name}
            onPress={() => onSelect(m.name)}
            disabled={m.disabled}
            style={{ minHeight: 44, paddingVertical: 12 }}
          >
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
              <View style={{ flex: 1 }}>
                <Text variant="base" style={{ fontWeight: '500', color: m.disabled ? colors.textDim : colors.text }}>
                  {m.display_name}
                </Text>
                {m.description && (
                  <Text variant="xs" style={{ color: colors.textDim, marginTop: 2 }}>{m.description}</Text>
                )}
              </View>
              {activeMode === m.name && <Ionicons name="checkmark" size={20} color={colors.accent} />}
            </View>
          </TouchableOpacity>
        ))}
      </View>
    </BottomSheet>
  );
}

function markdownStyles(colors: any) {
  return {
    body: { color: colors.text, fontSize: 15, lineHeight: 23 },
    paragraph: { marginTop: 0, marginBottom: 9 },
    heading1: { fontSize: 21, fontWeight: '700' as const, marginTop: 12, marginBottom: 8 },
    heading2: { fontSize: 19, fontWeight: '700' as const, marginTop: 10, marginBottom: 6 },
    heading3: { fontSize: 17, fontWeight: '600' as const, marginTop: 8, marginBottom: 4 },
    code_inline: { backgroundColor: colors.bgHover, borderRadius: 4, paddingHorizontal: 4 },
    fence: { backgroundColor: colors.bgHover, borderRadius: 10, padding: 12, marginTop: 8, marginBottom: 8 },
    code_block: { backgroundColor: colors.bgHover, borderRadius: 10, padding: 12, marginTop: 8, marginBottom: 8 },
    link: { color: colors.accent, textDecorationLine: 'underline' as const },
    blockquote: { borderLeftWidth: 2, borderLeftColor: colors.borderStrong, paddingLeft: 12, marginLeft: 0, marginTop: 8, marginBottom: 8 },
    list_item: { marginTop: 3, marginBottom: 3 },
  };
}

const styles = StyleSheet.create({
  messageList: { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 16 },
  messageListEmpty: { flexGrow: 1 },
  listFooter: { minHeight: 10, paddingTop: 2 },
  listEndMarker: { height: 8 },
  msgRow: { flexDirection: 'row', marginBottom: 14 },
  msgBubble: { borderRadius: 18, paddingHorizontal: 14, paddingVertical: 10 },
  interimGroup: {
    marginVertical: 7,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 13,
    padding: 4,
    overflow: 'hidden',
  },
  interimAccent: { width: 3, height: 22, borderRadius: 2 },
  interimBody: { paddingHorizontal: 5, paddingTop: 5 },
  interimMessage: {
    paddingHorizontal: 11,
    paddingVertical: 9,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 11,
  },
  collapseHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    marginVertical: 0,
  },
  copyButton: { alignSelf: 'flex-start', minWidth: 28, minHeight: 28, alignItems: 'center', justifyContent: 'center', marginTop: 1 },
  messageAttachments: { flexDirection: 'row', flexWrap: 'wrap', gap: 5, marginTop: 7 },
  messageAttachment: { flexDirection: 'row', alignItems: 'center', gap: 5, borderRadius: 999, paddingHorizontal: 8, paddingVertical: 5 },
  selectedAttachments: { paddingHorizontal: 12, paddingTop: 6, gap: 6 },
  selectedAttachment: { flexDirection: 'row', alignItems: 'center', gap: 6, borderRadius: 999, paddingHorizontal: 9, paddingVertical: 6 },
  codeBlock: { borderRadius: 10, padding: 12, marginVertical: 5 },
  codeBlockHeader: { flexDirection: 'row', justifyContent: 'flex-end', marginBottom: 3 },
  inlineError: { flexDirection: 'row', alignItems: 'center', gap: 8, borderRadius: 12, paddingHorizontal: 12, paddingVertical: 9, marginBottom: 14 },
  historyLoading: { paddingTop: 22, paddingHorizontal: 4 },
  loadingOlderIndicator: { paddingVertical: 10, alignItems: 'center' },
  composer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 5,
    marginHorizontal: 10,
    marginTop: 6,
    padding: 5,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 24,
  },
  composerIconButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 2,
  },
  input: { flex: 1, borderWidth: 0, paddingHorizontal: 8, paddingVertical: 9, maxHeight: 120, minHeight: 40 },
  sendBtn: { width: 38, height: 38, borderRadius: 19, justifyContent: 'center', alignItems: 'center' },
  settingsRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 8, minHeight: 44 },
  settingsSection: { marginTop: 12, marginBottom: 4 },
  settingsLabel: { fontSize: 12, color: '#94a3b8', marginBottom: 4 },
  varValue: { fontSize: 12, color: '#94a3b8' },
});

// Session settings BottomSheet — provider/model info + variables
interface SettingsBottomSheetProps {
  visible: boolean;
  onClose: () => void;
  varGroups: InspectorVariableGroup[];
  varsLoading: boolean;
  connection: { activeProvider: string | null; activeModel: string | null; yolo: boolean };
  onSetVariable: (key: string, value: unknown) => void;
  colors: any;
}

function SettingsBottomSheet({ visible, onClose, varGroups, varsLoading, connection, colors }: SettingsBottomSheetProps) {
  return (
    <BottomSheet visible={visible} onClose={onClose} title="Session Settings">
      <View style={styles.settingsSection}>
        <Text variant="sm" style={{ fontWeight: '600', marginBottom: 8 }}>Current Session</Text>
        <View style={styles.settingsRow}>
          <Text variant="xs" style={{ color: colors.textDim }}>Provider</Text>
          <Text variant="sm">{connection.activeProvider || '—'}</Text>
        </View>
        <View style={styles.settingsRow}>
          <Text variant="xs" style={{ color: colors.textDim }}>Model</Text>
          <Text variant="sm">{connection.activeModel || '—'}</Text>
        </View>
        <View style={styles.settingsRow}>
          <Text variant="xs" style={{ color: colors.textDim }}>YOLO</Text>
          <Text variant="sm">{connection.yolo ? 'On' : 'Off'}</Text>
        </View>
      </View>

      <View style={styles.settingsSection}>
        <Text variant="sm" style={{ fontWeight: '600', marginBottom: 8 }}>Variables</Text>
        {varsLoading ? (
          <Text variant="xs" style={{ color: colors.textDim }}>Loading…</Text>
        ) : varGroups.length === 0 ? (
          <Text variant="xs" style={{ color: colors.textDim }}>No variables configured</Text>
        ) : (
          varGroups.map(group => (
            <View key={group.name} style={{ marginBottom: 12 }}>
              <Text variant="xs" style={{ fontWeight: '500', marginBottom: 4 }}>{group.name}</Text>
              {group.variables.map(v => (
                <View key={v.key} style={styles.settingsRow}>
                  <View style={{ flex: 1 }}>
                    <Text variant="xs" style={{ color: colors.text }}>{v.key}</Text>
                    <Text variant="xs" style={styles.varValue}>{v.help}</Text>
                  </View>
                  <Text variant="xs" style={{ color: v.is_default ? colors.textDim : colors.accent }}>
                    {String(v.value)}
                  </Text>
                </View>
              ))}
            </View>
          ))
        )}
      </View>
    </BottomSheet>
  );
}
