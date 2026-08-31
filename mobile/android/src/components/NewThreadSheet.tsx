import React, { useEffect, useState } from 'react';
import { StyleSheet, TextInput, View } from 'react-native';
import { threadsApi, type CreateThreadResponse } from '../api/threads';
import { useTheme } from '../theme/ThemeContext';
import { Button } from './Button';
import { ModernBottomSheet } from './ModernBottomSheet';
import { Text } from './Text';

export type NewThreadSheetProps = {
  visible: boolean;
  parentSessionName: string | null;
  onClose: () => void;
  onCreated: (thread: CreateThreadResponse) => void;
};

export function NewThreadSheet({
  visible,
  parentSessionName,
  onClose,
  onCreated,
}: NewThreadSheetProps) {
  const { colors } = useTheme();
  const [title, setTitle] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!visible) return;
    setTitle('');
    setCreating(false);
    setError(null);
  }, [visible]);

  const create = async () => {
    const cleanTitle = title.trim();
    if (!parentSessionName || !cleanTitle || creating) return;
    setCreating(true);
    setError(null);
    try {
      const response = await threadsApi.create({
        parentSessionName,
        title: cleanTitle,
        activate: true,
      }, { timeoutMs: 30_000 });
      onCreated(response);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setCreating(false);
    }
  };

  return (
    <ModernBottomSheet visible={visible} onClose={onClose} title="New thread">
      <View style={styles.content}>
        <Text variant="sm" dim style={styles.intro}>
          Start an independent agent conversation that shares the environment from{' '}
          <Text variant="sm" style={styles.parentName}>{parentSessionName || 'the current session'}</Text>.
          Chat history and working memory start clean.
        </Text>
        <Text variant="xs" style={[styles.label, { color: colors.textDim }]}>THREAD TITLE</Text>
        <TextInput
          testID="thread-title-input"
          value={title}
          onChangeText={setTitle}
          placeholder="Investigate the parser"
          placeholderTextColor={colors.textDim}
          maxLength={120}
          autoCapitalize="sentences"
          autoCorrect
          returnKeyType="done"
          onSubmitEditing={() => void create()}
          style={[
            styles.input,
            {
              color: colors.text,
              backgroundColor: colors.bgLift,
              borderColor: colors.hairline,
            },
          ]}
        />
        <Text variant="xs" dim style={styles.help}>
          The new thread inherits provider, mode, workspace, and container access.
        </Text>
        {error ? <Text variant="xs" style={[styles.error, { color: colors.error }]}>{error}</Text> : null}
        <View style={styles.actions}>
          <Button title="Cancel" variant="ghost" onPress={onClose} disabled={creating} style={styles.action} />
          <Button
            title="Create thread"
            onPress={() => void create()}
            disabled={!parentSessionName || !title.trim()}
            loading={creating}
            style={styles.action}
          />
        </View>
      </View>
    </ModernBottomSheet>
  );
}

const styles = StyleSheet.create({
  content: { paddingTop: 10, paddingBottom: 8 },
  intro: { lineHeight: 21, marginBottom: 22 },
  parentName: { fontWeight: '600' },
  label: { fontFamily: 'monospace', fontWeight: '700', letterSpacing: 1, marginBottom: 7 },
  input: {
    minHeight: 48,
    paddingHorizontal: 13,
    paddingVertical: 10,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 9,
    fontSize: 16,
  },
  help: { lineHeight: 18, marginTop: 8 },
  error: { lineHeight: 18, marginTop: 12 },
  actions: { flexDirection: 'row', gap: 10, marginTop: 24 },
  action: { flex: 1 },
});
