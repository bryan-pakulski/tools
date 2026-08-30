import React, { useCallback, useEffect, useState } from 'react';
import { AppState, ScrollView, StyleSheet, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { ArtifactDescriptor, artifactsApi } from '../api/artifacts';
import { openExternalUrl } from '../api/urlSafety';
import { useTheme } from '../theme/ThemeContext';
import { Text } from './Text';

export function ArtifactStrip({
  sessionName,
  refreshKey = 0,
}: {
  sessionName: string | null;
  refreshKey?: number;
}) {
  const { colors } = useTheme();
  const [artifacts, setArtifacts] = useState<ArtifactDescriptor[]>([]);

  const load = useCallback(async () => {
    if (!sessionName) {
      setArtifacts([]);
      return;
    }
    try {
      const response = await artifactsApi.list(sessionName);
      const next = response.artifacts || [];
      setArtifacts(current => {
        const unchanged = current.length === next.length
          && current.every((item, index) => item.artifact_id === next[index]?.artifact_id
            && item.size === next[index]?.size);
        return unchanged ? current : next;
      });
    } catch {
      // Chat history remains useful when artifact refresh is temporarily unavailable.
    }
  }, [sessionName]);

  useEffect(() => {
    void load();
    if (!sessionName) return undefined;
    const timer = setInterval(() => { void load(); }, 5000);
    const appState = AppState.addEventListener('change', state => {
      if (state === 'active') void load();
    });
    return () => {
      clearInterval(timer);
      appState.remove();
    };
  }, [load, sessionName, refreshKey]);

  const visualizations = artifacts.filter(a => a.kind === 'visualization');
  const files = artifacts.filter(a => a.kind !== 'visualization');

  if (!sessionName || artifacts.length === 0) return null;

  return (
    <View style={styles.wrap}>
      {visualizations.length > 0 && (
        <View style={styles.section}>
          <View style={styles.heading}>
            <Text variant="xs" dim style={styles.label}>VISUALIZATIONS</Text>
            <Text variant="xs" dim>{visualizations.length}</Text>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.row}>
            {visualizations.map(artifact => {
              const isViz = artifact.kind === 'visualization';
              return (
                <TouchableOpacity
                  key={artifact.artifact_id}
                  onPress={() => openExternalUrl(artifactsApi.viewUrl(sessionName, artifact.artifact_id))}
                  style={[styles.chip, { backgroundColor: colors.bgHover, borderColor: colors.border }]}
                >
                  <Ionicons name="stats-chart-outline" size={16} color={colors.accent} />
                  <View style={styles.copy}>
                    <Text variant="xs" style={styles.name} numberOfLines={1}>
                      {artifact.title || artifact.name}
                    </Text>
                    <Text variant="xs" dim>{formatBytes(artifact.size)}</Text>
                  </View>
                  <Ionicons name="open-outline" size={17} color={colors.textDim} />
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </View>
      )}

      {files.length > 0 && (
        <View style={styles.section}>
          <View style={styles.heading}>
            <Text variant="xs" dim style={styles.label}>FILES</Text>
            <Text variant="xs" dim>{files.length}</Text>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.row}>
            {files.map(artifact => (
              <TouchableOpacity
                key={artifact.artifact_id}
                onPress={() => openExternalUrl(artifactsApi.downloadUrl(sessionName, artifact.artifact_id))}
                style={[styles.chip, { backgroundColor: colors.bgHover, borderColor: colors.border }]}
              >
                <Ionicons name="document-attach-outline" size={16} color={colors.accent} />
                <View style={styles.copy}>
                  <Text variant="xs" style={styles.name} numberOfLines={1}>{artifact.name}</Text>
                  <Text variant="xs" dim>{formatBytes(artifact.size)}</Text>
                </View>
                <Ionicons name="arrow-down-circle-outline" size={17} color={colors.textDim} />
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      )}
    </View>
  );
}

function formatBytes(value: number): string {
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

const styles = StyleSheet.create({
  wrap: { paddingTop: 8, paddingBottom: 4 },
  section: { paddingBottom: 4 },
  heading: { paddingHorizontal: 16, marginBottom: 7, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  label: { fontWeight: '700', letterSpacing: 0.8 },
  row: { paddingHorizontal: 16, gap: 8 },
  chip: { width: 210, minHeight: 54, borderWidth: StyleSheet.hairlineWidth, borderRadius: 14, paddingHorizontal: 11, flexDirection: 'row', alignItems: 'center', gap: 9 },
  copy: { flex: 1 },
  name: { fontWeight: '600' },
});