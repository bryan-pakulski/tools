import React, { useEffect, useMemo, useRef } from 'react';
import { ScrollView, StyleSheet, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../theme/ThemeContext';
import type { ChatMessage } from '../hooks/useChatSession';

interface Props {
  messages: ChatMessage[];
  activeMessageIndex: number;
  onJump: (messageIndex: number) => void;
  /** Older turns exist on the server but are not loaded yet (paginated history). */
  hasMoreHistory?: boolean;
  /** Loading flag while the older page is being fetched. */
  loadingHistory?: boolean;
  /** Press the indicator to load the older page immediately. */
  onLoadOlder?: () => void;
}

export interface ConversationStage {
  key: string;
  message: ChatMessage;
  messageIndex: number;
}

export function conversationStages(messages: ChatMessage[]): ConversationStage[] {
  const seen = new Set<string>();
  const stages: ConversationStage[] = [];
  messages.forEach((message, messageIndex) => {
    if (message.role !== 'user') return;
    const key = Number.isInteger(message.historyIndex)
      ? `history:${message.historyIndex}`
      : `live:${message.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    stages.push({ key, message, messageIndex });
  });
  return stages;
}

export function ConversationStageRail({
  messages,
  activeMessageIndex,
  onJump,
  hasMoreHistory = false,
  loadingHistory = false,
  onLoadOlder,
}: Props) {
  const { colors } = useTheme();
  const scrollRef = useRef<ScrollView>(null);
  const stages = useMemo(() => conversationStages(messages), [messages]);
  let activeStage = 0;
  for (let index = 0; index < stages.length; index += 1) {
    if (stages[index].messageIndex <= activeMessageIndex) activeStage = index;
    else break;
  }

  useEffect(() => {
    scrollRef.current?.scrollTo({ y: Math.max(0, activeStage * 24 - 72), animated: true });
  }, [activeStage]);

  if (stages.length < 2 && !hasMoreHistory) return null;
  return (
    <View pointerEvents="box-none" style={styles.rail} accessibilityLabel="Conversation stages">
      <ScrollView
        ref={scrollRef}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[styles.list, { backgroundColor: colors.bgLift, borderColor: colors.hairline }]}
      >
        {hasMoreHistory ? (
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityLabel="History continues above — tap to load older messages"
            disabled={loadingHistory}
            hitSlop={{ top: 4, bottom: 4, left: 8, right: 8 }}
            onPress={() => onLoadOlder?.()}
            style={styles.marker}
          >
            <View style={styles.moreBadge}>
              <Ionicons
                name={loadingHistory ? 'hourglass-outline' : 'chevron-up'}
                size={13}
                color={colors.accent}
              />
            </View>
          </TouchableOpacity>
        ) : null}
        {stages.map((stage, index) => {
          const active = index === activeStage;
          const label = stage.message.text.replace(/\s+/g, ' ').trim() || 'Attachment';
          return (
            <TouchableOpacity
              key={stage.key}
              accessibilityRole="button"
              accessibilityLabel={`Jump to prompt ${index + 1}: ${label.slice(0, 80)}`}
              hitSlop={{ top: 4, bottom: 4, left: 8, right: 8 }}
              onPress={() => onJump(stage.messageIndex)}
              style={styles.marker}
            >
              <View style={[
                styles.dot,
                { backgroundColor: active ? colors.accent : colors.borderStrong },
                active && styles.activeDot,
              ]} />
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  rail: { position: 'absolute', zIndex: 20, right: 3, top: 12, bottom: 12, width: 30 },
  list: { paddingVertical: 7, alignItems: 'center', borderWidth: StyleSheet.hairlineWidth, borderRadius: 15 },
  marker: { width: 28, height: 24, alignItems: 'center', justifyContent: 'center' },
  dot: { width: 5, height: 5, borderRadius: 4 },
  activeDot: { width: 9, height: 9 },
  // Round-51: "more history above" indicator at the top of the rail.
  moreBadge: {
    width: 20,
    height: 20,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 3,
    backgroundColor: 'transparent',
    borderWidth: StyleSheet.hairlineWidth,
  },
});
