import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Alert, AppState, StyleSheet, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { ArtifactDescriptor, artifactsApi } from '../api/artifacts';
import { openExternalUrl } from '../api/urlSafety';
import { useConnectionStore } from '../store/connection';
import { useTheme } from '../theme/ThemeContext';
import { ModernBottomSheet } from './ModernBottomSheet';
import { Text } from './Text';

export function ArtifactsSheet({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const { colors } = useTheme();
  const sessionName = useConnectionStore(state => state.activeSessionName);
  const [items, setItems] = useState<ArtifactDescriptor[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!sessionName) return setItems([]);
    setLoading(true);
    try {
      const response = await artifactsApi.list(sessionName);
      setItems(response.artifacts || []);
    } catch (error) {
      Alert.alert('Could not load artifacts', String(error));
    } finally {
      setLoading(false);
    }
  }, [sessionName]);

  useEffect(() => {
    if (!visible) return undefined;
    void load();
    const appState = AppState.addEventListener('change', state => {
      if (state === 'active') void load();
    });
    return () => appState.remove();
  }, [load, visible]);

  const remove = (artifact: ArtifactDescriptor) => {
    if (!sessionName) return;
    Alert.alert('Delete artifact?', artifact.name, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await artifactsApi.remove(sessionName, artifact.artifact_id);
          load();
        },
      },
    ]);
  };

  return (
    <ModernBottomSheet visible={visible} onClose={onClose} title="Artifacts">
      {loading ? <ActivityIndicator color={colors.accent} style={styles.loader} /> : null}
      {!loading && items.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="document-attach-outline" size={28} color={colors.textDim} />
          <Text variant="base" style={styles.emptyTitle}>No artifacts yet</Text>
          <Text variant="sm" dim style={styles.emptyBody}>The agent can publish reports, archives, images, and other deliverables with upload_artifact.</Text>
        </View>
      ) : items.map(item => (
        <View key={item.artifact_id} style={[styles.row, { borderBottomColor: colors.border }]}>
          <TouchableOpacity
            style={styles.open}
            onPress={() => sessionName && openExternalUrl(artifactsApi.downloadUrl(sessionName, item.artifact_id))}
          >
            <View style={[styles.icon, { backgroundColor: colors.bgHover }]}>
              <Ionicons name="document-outline" size={20} color={colors.accent} />
            </View>
            <View style={styles.copy}>
              <Text variant="sm" style={styles.name} numberOfLines={1}>{item.name}</Text>
              <Text variant="xs" dim>{formatBytes(item.size)} · {item.mime_type}</Text>
            </View>
            <Ionicons name="download-outline" size={19} color={colors.textDim} />
          </TouchableOpacity>
          <TouchableOpacity onPress={() => remove(item)} style={styles.delete}>
            <Ionicons name="trash-outline" size={18} color={colors.error} />
          </TouchableOpacity>
        </View>
      ))}
    </ModernBottomSheet>
  );
}

function formatBytes(value: number): string {
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

const styles = StyleSheet.create({
  loader: { marginVertical: 24 },
  empty: { alignItems: 'center', paddingHorizontal: 24, paddingVertical: 30 },
  emptyTitle: { fontWeight: '700', marginTop: 12 },
  emptyBody: { textAlign: 'center', marginTop: 5, lineHeight: 20 },
  row: { minHeight: 68, flexDirection: 'row', alignItems: 'center', borderBottomWidth: StyleSheet.hairlineWidth },
  open: { flex: 1, flexDirection: 'row', alignItems: 'center', paddingVertical: 9 },
  icon: { width: 42, height: 42, borderRadius: 13, alignItems: 'center', justifyContent: 'center' },
  copy: { flex: 1, marginHorizontal: 12 },
  name: { fontWeight: '600' },
  delete: { width: 42, height: 48, alignItems: 'center', justifyContent: 'center' },
});
