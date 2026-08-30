import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Clipboard from 'expo-clipboard';
import Markdown from 'react-native-markdown-display';
import { openExternalUrl } from '../api/urlSafety';
import { useTheme } from '../theme/ThemeContext';
import { useConnectionStore } from '../store/connection';
import { modesApi, ModeInfo } from '../api/modes';
import { inspectorApi, InspectorVariableGroup } from '../api/inspector';
import { Text, Skeleton, EmptyState, ErrorState } from '../components';
import { ModernBottomSheet } from '../components/ModernBottomSheet';
import { GeneratingIndicator } from '../components/GeneratingIndicator';
import { ArtifactStrip } from '../components/ArtifactStrip';
import { CodeBlock } from '../components/CodeBlock';
import { VisualizationCard } from '../components/VisualizationCard';
import { AttachmentSheet } from '../components/AttachmentSheet';
import type { AttachmentDescriptor } from '../api/attachments';
import { useChatSession, type ChatMessage } from '../hooks/useChatSession';
import { useCommandCompletion, type CompletionItem } from '../hooks/useCommandCompletion';
import { CommandSuggestionBar } from '../components/CommandSuggestionBar';
import { SubagentActivityPanel } from '../components/SubagentActivityPanel';
import {
  ConversationStageRail,
  conversationStages,
} from '../components/ConversationStageRail';

/**
 * Product presentation for the mobile conversation.
 *
 * Data/session behaviour remains owned by useChatSession. This component only
 * brings mobile presentation in line with the reviewed web UI: flat transcript,
 * compact interim disclosures and one glass composer pane with utilities outside.
 */
export function ChatScreenProduct() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const activeSessionName = useConnectionStore(state => state.activeSessionName);
  const activeProvider = useConnectionStore(state => state.activeProvider);
  const activeModel = useConnectionStore(state => state.activeModel);
  const yolo = useConnectionStore(state => state.yolo);
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
  // Round-51 UI: artifacts/visualizations strip is toggleable to reduce clutter.
  const [artifactsVisible, setArtifactsVisible] = useState(true);
  const [selectedAttachments, setSelectedAttachments] = useState<AttachmentDescriptor[]>([]);
  const [workedById, setWorkedById] = useState<Record<string, number>>({});
  const [activeMessageIndex, setActiveMessageIndex] = useState(0);

  const flatListRef = useRef<FlatList<ChatMessage>>(null);
  const loadingOlderRef = useRef(false);
  const checkpointRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingCheckpointIndexRef = useRef<number | null>(null);
  const checkpointJumpAttemptsRef = useRef(0);
  const olderCheckpointAnchorRef = useRef<string | null | undefined>(undefined);
  const followOutputRef = useRef(true);
  const visualizationActiveRef = useRef(false);
  const previousStreamingRef = useRef(false);
  const turnStartedAtRef = useRef<number | null>(null);
  const completion = useCommandCompletion();
  const viewabilityRef = useRef(({ viewableItems }: { viewableItems: Array<{ index: number | null }> }) => {
    const indexes = viewableItems
      .map(item => item.index)
      .filter((index): index is number => index !== null)
      .sort((left, right) => left - right);
    if (indexes.length > 0) {
      const pendingIndex = pendingCheckpointIndexRef.current;
      if (pendingIndex !== null && indexes.includes(pendingIndex)) {
        setActiveMessageIndex(pendingIndex);
        pendingCheckpointIndexRef.current = null;
        checkpointJumpAttemptsRef.current = 0;
        if (checkpointRetryTimerRef.current) clearTimeout(checkpointRetryTimerRef.current);
        checkpointRetryTimerRef.current = null;
      } else {
        setActiveMessageIndex(indexes[0]);
      }
    }
  });
  // FlatList requires viewabilityConfig identity to remain stable. Recreating
  // it when the active marker changes can stop viewability callbacks entirely.
  const stageViewabilityConfigRef = useRef({ itemVisiblePercentThreshold: 20 });

  const jumpToCheckpoint = useCallback((messageIndex: number) => {
    followOutputRef.current = false;
    pendingCheckpointIndexRef.current = messageIndex;
    checkpointJumpAttemptsRef.current = 0;
    setActiveMessageIndex(messageIndex);
    flatListRef.current?.scrollToIndex({
      index: messageIndex,
      animated: false,
      viewPosition: 0,
    });
  }, []);

  const onCheckpointScrollFailed = useCallback(({
    index,
    averageItemLength,
  }: {
    index: number;
    averageItemLength: number;
  }) => {
    if (pendingCheckpointIndexRef.current !== index) return;
    checkpointJumpAttemptsRef.current += 1;
    flatListRef.current?.scrollToOffset({
      offset: Math.max(0, index * Math.max(1, averageItemLength)),
      animated: false,
    });
    if (checkpointRetryTimerRef.current) clearTimeout(checkpointRetryTimerRef.current);
    if (checkpointJumpAttemptsRef.current >= 6) return;
    checkpointRetryTimerRef.current = setTimeout(() => {
      if (pendingCheckpointIndexRef.current !== index) return;
      flatListRef.current?.scrollToIndex({
        index,
        animated: false,
        viewPosition: 0,
      });
    }, 96);
  }, []);

  const loadOlderCheckpoints = useCallback(() => {
    if (loadingOlder) return;
    const stages = conversationStages(messages);
    olderCheckpointAnchorRef.current = stages[0]?.key ?? null;
    void loadOlderHistory().then(checkpointCount => {
      if (checkpointCount <= 0) olderCheckpointAnchorRef.current = undefined;
    });
  }, [loadOlderHistory, loadingOlder, messages]);

  useEffect(() => {
    const anchorKey = olderCheckpointAnchorRef.current;
    if (anchorKey === undefined) return;
    const stages = conversationStages(messages);
    const anchorIndex = anchorKey === null
      ? stages.length
      : stages.findIndex(stage => stage.key === anchorKey);
    if (anchorIndex <= 0) return;
    const target = stages[anchorIndex - 1];
    olderCheckpointAnchorRef.current = undefined;
    jumpToCheckpoint(target.messageIndex);
  }, [jumpToCheckpoint, messages]);

  useEffect(() => () => {
    if (checkpointRetryTimerRef.current) clearTimeout(checkpointRetryTimerRef.current);
  }, []);

  const copyMessage = useCallback((text: string) => {
    void Clipboard.setStringAsync(text);
  }, []);

  const loadModes = useCallback(async () => {
    try {
      const response = await modesApi.list();
      setModes(response.modes || []);
      setActiveMode(response.current || 'default');
    } catch {
      // Session transport errors remain owned by the connection/session layer.
    }
  }, []);

  const loadVariables = useCallback(async () => {
    setVarsLoading(true);
    try {
      const response = await inspectorApi.getVariables();
      setVarGroups(response.groups || []);
    } catch {
      setVarGroups([]);
    } finally {
      setVarsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadModes();
  }, [activeSessionName, loadModes]);

  useEffect(() => {
    // Wall-clock breadcrumb parity with web. Historical timings are not
    // fabricated; only turns observed by this mobile client receive a value.
    const wasStreaming = previousStreamingRef.current;
    if (!wasStreaming && streaming) turnStartedAtRef.current = Date.now();
    if (wasStreaming && !streaming && turnStartedAtRef.current) {
      const completed = [...messages].reverse().find(message =>
        message.role === 'assistant' && !message.streaming && message.text.trim().length > 0,
      );
      if (completed) {
        const elapsed = Math.max(0, Date.now() - turnStartedAtRef.current);
        setWorkedById(current => ({ ...current, [completed.id]: elapsed }));
      }
      turnStartedAtRef.current = null;
    }
    previousStreamingRef.current = streaming;
  }, [messages, streaming]);

  useEffect(() => {
    setWorkedById({});
    previousStreamingRef.current = false;
    turnStartedAtRef.current = null;
    pendingCheckpointIndexRef.current = null;
    checkpointJumpAttemptsRef.current = 0;
    olderCheckpointAnchorRef.current = undefined;
    if (checkpointRetryTimerRef.current) clearTimeout(checkpointRetryTimerRef.current);
    checkpointRetryTimerRef.current = null;
  }, [activeSessionName]);

  const send = useCallback(async () => {
    const text = input.trim();
    if ((!text && selectedAttachments.length === 0) || streaming) return;
    followOutputRef.current = true;
    const sent = await sendMessage(text, selectedAttachments);
    if (!sent) return;
    setInput('');
    setSelectedAttachments([]);
    completion.close();
    requestAnimationFrame(() => flatListRef.current?.scrollToEnd({ animated: false }));
  }, [completion, input, selectedAttachments, sendMessage, streaming]);

  const selectMode = useCallback(async (name: string) => {
    try {
      await modesApi.set(name);
      setActiveMode(name);
      setModeSheetOpen(false);
    } catch {
      // Existing session error presentation remains authoritative.
    }
  }, []);

  const onInputChange = useCallback((text: string) => {
    setInput(text);
    if (text.startsWith('/')) completion.update(text);
    else if (completion.visible) completion.close();
  }, [completion]);

  const onAcceptCompletion = useCallback((item: CompletionItem) => {
    const next = `${item.value} `;
    setInput(next);
    if (item.level === 0) completion.update(next);
    else completion.close();
  }, [completion]);

  const onVisualizationInteractionChange = useCallback((active: boolean) => {
    visualizationActiveRef.current = active;
    if (active) followOutputRef.current = false;
  }, []);

  // Safe link handler: only http(s) URLs reach the OS. Everything else is
  // blocked (with an alert) instead of being handed to Linking.
  const handleMarkdownLinkPress = useCallback((url: string) => {
    void openExternalUrl(url);
    return false;
  }, []);

  const markdownRules = useMemo(() => ({
    fence: (node: any) => (
      <CodeBlock key={node.key} code={node.content} language={(node.sourceInfo || '').trim()} colors={colors} />
    ),
    code_block: (node: any) => (
      <CodeBlock key={node.key} code={node.content} colors={colors} />
    ),
    code_inline: (node: any) => (
      <Text
        key={node.key}
        style={{
          color: colors.syntax.keyword,
          fontFamily: 'monospace',
          fontSize: 13,
          backgroundColor: colors.bgHover,
          borderRadius: 3,
          paddingHorizontal: 4,
        }}
      >
        {node.content}
      </Text>
    ),
  }), [colors]);
  const mdStyles = useMemo(() => markdownStyles(colors), [colors]);

  const renderAssistantBody = useCallback((message: ChatMessage, compact = false) => {
    if (message.streaming) {
      return (
        <Text style={{ color: compact ? colors.textSoft : colors.text, fontSize: compact ? 13 : 15, lineHeight: compact ? 19 : 23 }}>
          {message.text}
        </Text>
      );
    }
    return (
      <Markdown style={compact ? compactMarkdownStyles(colors) : mdStyles} rules={markdownRules} onLinkPress={handleMarkdownLinkPress}>
        {message.text}
      </Markdown>
    );
  }, [colors, markdownRules, mdStyles]);

  const renderMessage = useCallback(({ item }: { item: ChatMessage }) => {
    if (item.role === 'visualization' && item.artifact && activeSessionName) {
      return (
        <View style={styles.visualizationWrap}>
          <VisualizationCard
            artifact={item.artifact}
            sessionName={activeSessionName}
            onInteractionChange={onVisualizationInteractionChange}
          />
        </View>
      );
    }

    if (item.role === 'subagent_panel' && item.subagents?.length) {
      return (
        <View style={styles.visualizationWrap}>
          <SubagentActivityPanel agents={item.subagents} />
        </View>
      );
    }

    if (item.role === 'collapse') {
      const count = item.collapseCount || item.childTurns?.length || 0;
      return (
        <View style={styles.interimGroup}>
          <TouchableOpacity
            onPress={() => setMessages(current => current.map(message =>
              message.id === item.id ? { ...message, collapseOpen: !message.collapseOpen } : message,
            ))}
            style={styles.interimHeader}
            activeOpacity={0.7}
          >
            <Ionicons
              name={item.collapseOpen ? 'chevron-down' : 'chevron-forward'}
              size={13}
              color={colors.textDim}
            />
            <Text variant="xs" style={{ color: colors.textDim }}>
              {count} interim update{count !== 1 ? 's' : ''}
              {item.collapseElapsed ? ` · ${item.collapseElapsed}` : ''}
              {item.collapseTokens ? ` · ${item.collapseTokens}` : ''}
            </Text>
          </TouchableOpacity>
          {item.collapseOpen ? (
            <View style={[styles.interimBody, { borderLeftColor: colors.hairline }]}>
              {item.childTurns?.map(child => {
                if (child.role !== 'assistant') return null;
                return (
                  <View key={child.id} style={styles.interimMessage}>
                    {renderAssistantBody(child, true)}
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
    const workedMs = workedById[item.id];

    return (
      <MessageHandoff phase={isAssistant ? item.handoff : undefined}>
        <View style={[styles.msgRow, isUser ? styles.userRow : styles.assistantRow]}>
        <View
          style={[
            isUser ? styles.userMessage : styles.assistantMessage,
            isUser && { borderRightColor: colors.accent },
          ]}
        >
          {isUser ? (
            <Text style={{ color: colors.textSoft, fontSize: 15, lineHeight: 22 }}>{item.text}</Text>
          ) : (
            renderAssistantBody(item)
          )}

          {isUser && item.attachments?.length ? (
            <View style={styles.messageAttachments}>
              {item.attachments.map(attachment => (
                <View key={attachment.attachment_id} style={[styles.messageAttachment, { borderColor: colors.hairline }]}>
                  <Ionicons name="document-outline" size={13} color={colors.textDim} />
                  <Text variant="xs" numberOfLines={1} style={{ color: colors.textSoft, maxWidth: 210 }}>{attachment.name}</Text>
                </View>
              ))}
            </View>
          ) : null}

          {isAssistant && item.text.length > 0 ? (
            <View style={styles.assistantMeta}>
              {workedMs ? (
                <Text variant="xs" style={{ color: colors.textDim }}>{formatWorkedDuration(workedMs)}</Text>
              ) : <View />}
              <TouchableOpacity
                onPress={() => copyMessage(item.text)}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                style={styles.copyButton}
              >
                <Ionicons name="copy-outline" size={13} color={colors.textDim} />
              </TouchableOpacity>
            </View>
          ) : null}
        </View>
        </View>
      </MessageHandoff>
    );
  }, [activeSessionName, colors, copyMessage, onVisualizationInteractionChange, renderAssistantBody, setMessages, workedById]);

  const onScroll = useCallback((event: any) => {
    const native = event?.nativeEvent;
    if (!native) return;
    const distanceFromEnd = Math.max(0, native.contentSize.height - native.layoutMeasurement.height - native.contentOffset.y);
    if (native.contentOffset.y < 100 && hasMore && !loadingOlder && !loadingOlderRef.current) {
      loadingOlderRef.current = true;
      void loadOlderHistory().finally(() => { loadingOlderRef.current = false; });
    }
    if (!visualizationActiveRef.current) followOutputRef.current = distanceFromEnd <= 100;
  }, [hasMore, loadOlderHistory, loadingOlder]);

  return (
    <SafeAreaView edges={['bottom']} style={[styles.safeArea, { backgroundColor: colors.bg }]}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={insets.bottom}
      >
        {/* Round-44 F13: match ChatScreen's list tuning so both chat
            surfaces behave identically at scale — bounded initial/batch
            render + window size keep long loaded histories responsive. */}
        <View style={styles.timeline}>
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
          contentContainerStyle={[styles.messageList, messages.length === 0 && styles.messageListEmpty]}
          maintainVisibleContentPosition={{ minIndexForVisible: 0 }}
          keyboardShouldPersistTaps="always"
          keyboardDismissMode="none"
          onScroll={onScroll}
          onViewableItemsChanged={viewabilityRef.current}
          viewabilityConfig={stageViewabilityConfigRef.current}
          onScrollToIndexFailed={onCheckpointScrollFailed}
          scrollEventThrottle={32}
          onScrollBeginDrag={() => { followOutputRef.current = false; }}
          onContentSizeChange={() => {
            if (streaming && followOutputRef.current && !visualizationActiveRef.current) {
              flatListRef.current?.scrollToEnd({ animated: false });
            }
          }}
          ListHeaderComponent={
            <View>
              {loadingOlder ? (
                <View style={styles.loadingOlderIndicator}>
                  <GeneratingIndicator label="Loading older messages" />
                </View>
              ) : null}
              {error && messages.length > 0 ? (
                <View style={[styles.inlineError, { borderBottomColor: colors.hairline }]}>
                  <Ionicons name="warning-outline" size={15} color={colors.error} />
                  <Text variant="xs" style={{ color: colors.textSoft, flex: 1 }}>{error}</Text>
                  <TouchableOpacity onPress={retry}>
                    <Text variant="xs" style={{ color: colors.accent, fontWeight: '600' }}>Retry</Text>
                  </TouchableOpacity>
                </View>
              ) : null}
            </View>
          }
          ListEmptyComponent={
            historyLoading ? (
              <View style={styles.historyLoading}>
                <GeneratingIndicator label="Loading conversation history" />
                <Skeleton height={16} style={{ marginBottom: 9, width: '72%' }} />
                <Skeleton height={16} style={{ marginBottom: 9, width: '88%' }} />
                <Skeleton height={16} style={{ width: '56%' }} />
              </View>
            ) : error ? (
              <ErrorState message={error} onRetry={retry} />
            ) : (
              <EmptyState
                icon="sparkles-outline"
                title="What are you working on?"
                message="Ask MuCLI to inspect, explain, debug, or change your workspace."
                actionLabel={activeMode ? `Mode: ${activeMode}` : 'Select mode'}
                onAction={() => { void loadModes(); setModeSheetOpen(true); }}
              />
            )
          }
          ListFooterComponent={
            <View style={styles.listFooter}>
              {streaming && waitingForFirstToken ? (
                <GeneratingIndicator label={sseConnected ? activityLabel : 'Reconnecting to session'} />
              ) : null}
            </View>
          }
        />
        <ConversationStageRail
          messages={messages}
          activeMessageIndex={activeMessageIndex}
          hasMoreHistory={hasMore}
          loadingHistory={loadingOlder}
          onLoadOlder={loadOlderCheckpoints}
          onJump={jumpToCheckpoint}
        />
        </View>

        {artifactsVisible ? (
          <ArtifactStrip sessionName={activeSessionName} refreshKey={artifactRevision} />
        ) : null}
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
          activeMode={activeMode || 'default'}
          onModePress={() => { void loadModes(); setModeSheetOpen(true); }}
          onSettingsPress={() => { void loadVariables(); setSettingsSheetOpen(true); }}
          onAttachmentsPress={() => setAttachmentsOpen(true)}
          selectedAttachments={selectedAttachments}
          onRemoveAttachment={attachmentId => setSelectedAttachments(current => current.filter(item => item.attachment_id !== attachmentId))}
          bottomInset={insets.bottom}
          artifactsVisible={artifactsVisible}
          onToggleArtifacts={() => setArtifactsVisible(current => !current)}
        />

        <AttachmentSheet
          visible={attachmentsOpen}
          sessionName={activeSessionName || ''}
          selected={selectedAttachments}
          onSelectedChange={setSelectedAttachments}
          onClose={() => setAttachmentsOpen(false)}
        />
        <ModeSheet
          visible={modeSheetOpen}
          onClose={() => setModeSheetOpen(false)}
          modes={modes}
          activeMode={activeMode}
          onSelect={selectMode}
        />
        <SessionSettingsSheet
          visible={settingsSheetOpen}
          onClose={() => setSettingsSheetOpen(false)}
          groups={varGroups}
          loading={varsLoading}
          provider={activeProvider}
          model={activeModel}
          yolo={yolo}
        />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function Composer({
  input,
  setInput,
  onSend,
  onStop,
  streaming,
  activeMode,
  onModePress,
  onSettingsPress,
  onAttachmentsPress,
  selectedAttachments,
  onRemoveAttachment,
  bottomInset,
  artifactsVisible,
  onToggleArtifacts,
}: {
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  streaming: boolean;
  activeMode: string;
  onModePress: () => void;
  onSettingsPress: () => void;
  onAttachmentsPress: () => void;
  selectedAttachments: AttachmentDescriptor[];
  onRemoveAttachment: (attachmentId: string) => void;
  bottomInset: number;
  artifactsVisible: boolean;
  onToggleArtifacts: () => void;
}) {
  const { colors } = useTheme();
  const canSend = input.trim().length > 0 || selectedAttachments.length > 0;
  return (
    <View style={[styles.composerArea, { paddingBottom: Math.max(bottomInset, 8) }]}>
      {selectedAttachments.length > 0 ? (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.selectedAttachments}>
          {selectedAttachments.map(item => (
            <View key={item.attachment_id} style={[styles.selectedAttachment, { borderColor: colors.hairline }]}>
              <Text variant="xs" numberOfLines={1} style={{ maxWidth: 180, color: colors.textSoft }}>{item.name}</Text>
              <TouchableOpacity onPress={() => onRemoveAttachment(item.attachment_id)}>
                <Ionicons name="close" size={14} color={colors.textDim} />
              </TouchableOpacity>
            </View>
          ))}
        </ScrollView>
      ) : null}

      <View style={styles.composerUtilities}>
        <TouchableOpacity onPress={onModePress} style={styles.utilityButton}>
          <Text variant="xs" style={{ color: colors.textDim, textTransform: 'capitalize' }}>{activeMode}</Text>
          <Ionicons name="chevron-down" size={13} color={colors.textDim} />
        </TouchableOpacity>
        <TouchableOpacity
          onPress={onToggleArtifacts}
          style={styles.utilityIconButton}
          accessibilityLabel={artifactsVisible ? 'Hide artifacts panel' : 'Show artifacts panel'}
        >
          <Ionicons
            name={artifactsVisible ? 'eye' : 'eye-off'}
            size={16}
            color={artifactsVisible ? colors.accent : colors.textDim}
          />
        </TouchableOpacity>
        <TouchableOpacity onPress={onSettingsPress} style={styles.utilityIconButton} accessibilityLabel="Session settings">
          <Ionicons name="settings-outline" size={16} color={colors.textDim} />
        </TouchableOpacity>
      </View>

      <View style={[styles.composerPane, { backgroundColor: colors.glassStrong, borderColor: colors.hairline }]}>
        <TouchableOpacity
          onPress={onAttachmentsPress}
          style={styles.composerIconButton}
          accessibilityLabel="Attach files"
        >
          <Ionicons name="attach" size={19} color={selectedAttachments.length ? colors.accent : colors.textDim} />
        </TouchableOpacity>
        <TextInput
          value={input}
          onChangeText={setInput}
          placeholder="Message MuCLI"
          placeholderTextColor={colors.textDim}
          multiline
          style={[styles.input, { color: colors.text }]}
        />
        {streaming ? (
          <TouchableOpacity onPress={onStop} style={styles.sendButton} accessibilityLabel="Stop generation">
            <Ionicons name="stop" size={16} color={colors.error} />
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            onPress={onSend}
            disabled={!canSend}
            style={[styles.sendButton, canSend && { backgroundColor: colors.accentStrong }]}
            accessibilityLabel="Send message"
          >
            <Ionicons name="arrow-up" size={18} color={canSend ? colors.accentText : colors.textDim} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

function ModeSheet({
  visible,
  onClose,
  modes,
  activeMode,
  onSelect,
}: {
  visible: boolean;
  onClose: () => void;
  modes: ModeInfo[];
  activeMode: string | null;
  onSelect: (name: string) => void;
}) {
  const { colors } = useTheme();
  return (
    <ModernBottomSheet visible={visible} onClose={onClose} title="Mode">
      <View style={[styles.optionList, { borderTopColor: colors.hairline }]}>
        {modes.map(mode => (
          <TouchableOpacity
            key={mode.name}
            onPress={() => onSelect(mode.name)}
            disabled={mode.disabled}
            style={[styles.optionRow, { borderBottomColor: colors.hairline }, mode.disabled && styles.disabledOption]}
          >
            <View style={styles.optionCopy}>
              <Text variant="sm" style={{ color: colors.text, fontWeight: '600' }}>{mode.display_name}</Text>
              {mode.description ? <Text variant="xs" dim>{mode.description}</Text> : null}
            </View>
            {activeMode === mode.name ? <Ionicons name="checkmark" size={18} color={colors.accent} /> : null}
          </TouchableOpacity>
        ))}
      </View>
    </ModernBottomSheet>
  );
}

function SessionSettingsSheet({
  visible,
  onClose,
  groups,
  loading,
  provider,
  model,
  yolo,
}: {
  visible: boolean;
  onClose: () => void;
  groups: InspectorVariableGroup[];
  loading: boolean;
  provider: string | null;
  model: string | null;
  yolo: boolean;
}) {
  const { colors } = useTheme();
  return (
    <ModernBottomSheet visible={visible} onClose={onClose} title="Session settings">
      <View style={[styles.settingsSummary, { borderTopColor: colors.hairline }]}>
        <SettingLine label="Provider" value={provider || '—'} />
        <SettingLine label="Model" value={model || '—'} />
        <SettingLine label="Auto-approve" value={yolo ? 'On' : 'Off'} />
      </View>
      <Text variant="xs" style={[styles.settingsSectionLabel, { color: colors.textDim }]}>Variables</Text>
      {loading ? (
        <Text variant="xs" dim style={styles.settingsLoading}>Loading…</Text>
      ) : groups.length === 0 ? (
        <Text variant="xs" dim style={styles.settingsLoading}>No variables configured</Text>
      ) : (
        groups.map(group => (
          <View key={group.name} style={[styles.settingsGroup, { borderTopColor: colors.hairline }]}>
            <Text variant="sm" style={styles.settingsGroupTitle}>{group.name}</Text>
            {group.variables.map(variable => (
              <View key={variable.key} style={[styles.settingsVariable, { borderBottomColor: colors.hairline }]}>
                <View style={styles.settingsVariableCopy}>
                  <Text variant="xs" style={{ color: colors.textSoft }}>{variable.key}</Text>
                  {variable.help ? <Text variant="xs" dim numberOfLines={2}>{variable.help}</Text> : null}
                </View>
                <Text variant="xs" style={{ color: variable.is_default ? colors.textDim : colors.accent }} numberOfLines={1}>
                  {String(variable.value)}
                </Text>
              </View>
            ))}
          </View>
        ))
      )}
    </ModernBottomSheet>
  );
}

function SettingLine({ label, value }: { label: string; value: string }) {
  const { colors } = useTheme();
  return (
    <View style={[styles.settingLine, { borderBottomColor: colors.hairline }]}>
      <Text variant="xs" dim>{label}</Text>
      <Text variant="sm" style={{ color: colors.textSoft }} numberOfLines={1}>{value}</Text>
    </View>
  );
}

function MessageHandoff({
  phase,
  children,
}: {
  phase?: ChatMessage['handoff'];
  children: React.ReactNode;
}) {
  const opacity = useRef(new Animated.Value(phase === 'entering' ? 0 : 1)).current;
  const translateY = useRef(new Animated.Value(phase === 'entering' ? 5 : 0)).current;

  useEffect(() => {
    const leaving = phase === 'leaving';
    const entering = phase === 'entering';
    if (!leaving && !entering) {
      opacity.setValue(1);
      translateY.setValue(0);
      return;
    }
    Animated.parallel([
      Animated.timing(opacity, {
        toValue: leaving ? 0 : 1,
        duration: 240,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }),
      Animated.timing(translateY, {
        toValue: leaving ? -3 : 0,
        duration: 240,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }),
    ]).start();
  }, [opacity, phase, translateY]);

  return <Animated.View style={{ opacity, transform: [{ translateY }] }}>{children}</Animated.View>;
}

function formatWorkedDuration(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return minutes > 0 ? `Worked for ${minutes}m ${remainder}s` : `Worked for ${remainder}s`;
}

function markdownStyles(colors: any) {
  return {
    body: { color: colors.text, fontSize: 15, lineHeight: 23 },
    paragraph: { marginTop: 0, marginBottom: 8 },
    heading1: { fontSize: 20, fontWeight: '700' as const, marginTop: 12, marginBottom: 7 },
    heading2: { fontSize: 18, fontWeight: '700' as const, marginTop: 10, marginBottom: 6 },
    heading3: { fontSize: 16, fontWeight: '600' as const, marginTop: 8, marginBottom: 4 },
    code_inline: { backgroundColor: colors.bgHover, borderRadius: 3, paddingHorizontal: 4 },
    fence: { backgroundColor: colors.glass, borderRadius: 9, padding: 11, marginTop: 7, marginBottom: 7 },
    code_block: { backgroundColor: colors.glass, borderRadius: 9, padding: 11, marginTop: 7, marginBottom: 7 },
    link: { color: colors.accent, textDecorationLine: 'underline' as const },
    blockquote: { borderLeftWidth: 1, borderLeftColor: colors.hairline, paddingLeft: 11, marginLeft: 0, marginTop: 7, marginBottom: 7 },
    list_item: { marginTop: 2, marginBottom: 2 },
  };
}

function compactMarkdownStyles(colors: any) {
  return {
    ...markdownStyles(colors),
    body: { color: colors.textSoft, fontSize: 13, lineHeight: 19 },
    paragraph: { marginTop: 0, marginBottom: 5 },
    heading1: { fontSize: 15, fontWeight: '600' as const, marginTop: 6, marginBottom: 4 },
    heading2: { fontSize: 14, fontWeight: '600' as const, marginTop: 6, marginBottom: 4 },
    heading3: { fontSize: 13, fontWeight: '600' as const, marginTop: 5, marginBottom: 3 },
  };
}

const styles = StyleSheet.create({
  safeArea: { flex: 1 },
  flex: { flex: 1 },
  timeline: { flex: 1, position: 'relative' },
  messageList: { paddingLeft: 18, paddingRight: 46, paddingTop: 24, paddingBottom: 12 },
  messageListEmpty: { flexGrow: 1 },
  msgRow: { flexDirection: 'row', marginBottom: 17 },
  userRow: { justifyContent: 'flex-end' },
  assistantRow: { justifyContent: 'flex-start' },
  userMessage: { maxWidth: '86%', paddingVertical: 3, paddingRight: 12, paddingLeft: 2, borderRightWidth: 1 },
  assistantMessage: { width: '100%' },
  assistantMeta: { minHeight: 26, marginTop: 5, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  copyButton: { width: 30, height: 26, alignItems: 'center', justifyContent: 'center' },
  visualizationWrap: { marginBottom: 15 },
  interimGroup: { marginTop: -1, marginBottom: 12 },
  interimHeader: { minHeight: 28, flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 3 },
  interimBody: { marginLeft: 5, paddingLeft: 11, borderLeftWidth: StyleSheet.hairlineWidth },
  interimMessage: { paddingVertical: 4, marginBottom: 6 },
  messageAttachments: { flexDirection: 'row', flexWrap: 'wrap', gap: 5, marginTop: 7 },
  messageAttachment: { flexDirection: 'row', alignItems: 'center', gap: 5, borderWidth: StyleSheet.hairlineWidth, borderRadius: 5, paddingHorizontal: 7, paddingVertical: 4 },
  inlineError: { minHeight: 42, flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 8, marginBottom: 12, borderBottomWidth: StyleSheet.hairlineWidth },
  historyLoading: { paddingTop: 22, paddingHorizontal: 4 },
  loadingOlderIndicator: { paddingVertical: 8, alignItems: 'center' },
  listFooter: { minHeight: 10, paddingTop: 2 },
  composerArea: { paddingHorizontal: 12, paddingTop: 4 },
  composerUtilities: { minHeight: 30, flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-end', gap: 4, paddingHorizontal: 2, marginBottom: 8 },
  utilityButton: { minHeight: 30, flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 7 },
  utilityIconButton: { width: 34, height: 30, alignItems: 'center', justifyContent: 'center' },
  composerPane: { flexDirection: 'row', alignItems: 'flex-end', gap: 3, padding: 5, borderWidth: StyleSheet.hairlineWidth, borderRadius: 14 },
  composerIconButton: { width: 36, height: 36, alignItems: 'center', justifyContent: 'center', marginBottom: 1 },
  input: { flex: 1, borderWidth: 0, paddingHorizontal: 7, paddingVertical: 8, maxHeight: 130, minHeight: 40, fontSize: 15, lineHeight: 22 },
  sendButton: { width: 36, height: 36, borderRadius: 8, justifyContent: 'center', alignItems: 'center', marginBottom: 1 },
  selectedAttachments: { paddingHorizontal: 2, paddingBottom: 7, gap: 6 },
  selectedAttachment: { flexDirection: 'row', alignItems: 'center', gap: 6, borderWidth: StyleSheet.hairlineWidth, borderRadius: 5, paddingHorizontal: 8, paddingVertical: 5 },
  optionList: { borderTopWidth: StyleSheet.hairlineWidth },
  optionRow: { minHeight: 58, flexDirection: 'row', alignItems: 'center', paddingVertical: 9, borderBottomWidth: StyleSheet.hairlineWidth },
  disabledOption: { opacity: 0.35 },
  optionCopy: { flex: 1, paddingRight: 12 },
  settingsSummary: { borderTopWidth: StyleSheet.hairlineWidth },
  settingLine: { minHeight: 48, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 16, borderBottomWidth: StyleSheet.hairlineWidth },
  settingsSectionLabel: { marginTop: 22, marginBottom: 8, fontWeight: '600' },
  settingsLoading: { paddingVertical: 14 },
  settingsGroup: { marginTop: 12, borderTopWidth: StyleSheet.hairlineWidth },
  settingsGroupTitle: { paddingVertical: 10, fontWeight: '600' },
  settingsVariable: { minHeight: 54, flexDirection: 'row', alignItems: 'center', gap: 12, borderBottomWidth: StyleSheet.hairlineWidth, paddingVertical: 8 },
  settingsVariableCopy: { flex: 1 },
});
