import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  AppState,
  FlatList,
  Modal,
  RefreshControl,
  StyleSheet,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { SafeAreaModal } from '../components/SafeAreaModal';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import { WebView } from 'react-native-webview';
import { ArtifactDescriptor, artifactsApi } from '../api/artifacts';
import { openExternalUrl } from '../api/urlSafety';
import { attachmentsApi, type AttachmentDescriptor, type PickedDocument } from '../api/attachments';
import { useConnectionStore } from '../store/connection';
import { useTheme } from '../theme/ThemeContext';
import { Text } from '../components/Text';
import { EmptyState } from '../components/EmptyState';

type UnifiedItem =
  | { kind: 'visualization'; artifact: ArtifactDescriptor }
  | { kind: 'model'; artifact: ArtifactDescriptor }
  | { kind: 'upload'; attachment: AttachmentDescriptor };

function formatBytes(value: number): string {
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

export function ArtifactsScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const sessionName = useConnectionStore(state => state.activeSessionName);

  const [artifacts, setArtifacts] = useState<ArtifactDescriptor[]>([]);
  const [uploads, setUploads] = useState<AttachmentDescriptor[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [viewingArtifact, setViewingArtifact] = useState<ArtifactDescriptor | null>(null);

  // Stable exact URL the artifact WebView is allowed to load (nav gate).
  const artifactUri = viewingArtifact
    ? artifactsApi.viewUrl(sessionName as string, viewingArtifact.artifact_id)
    : '';

  const load = useCallback(async () => {
    if (!sessionName) {
      setArtifacts([]);
      setUploads([]);
      return;
    }
    try {
      const [artRes, attRes] = await Promise.all([
        artifactsApi.list(sessionName),
        attachmentsApi.list(sessionName),
      ]);
      setArtifacts(artRes.artifacts || []);
      setUploads(attRes.attachments || []);
    } catch {
      // Keep previous data on transient error.
    }
  }, [sessionName]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const appState = AppState.addEventListener('change', state => {
      if (state === 'active') void load();
    });
    return () => appState.remove();
  }, [load]);

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  const removeArtifact = (artifact: ArtifactDescriptor) => {
    if (!sessionName) return;
    Alert.alert('Delete artifact?', artifact.name, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await artifactsApi.remove(sessionName, artifact.artifact_id);
          void load();
        },
      },
    ]);
  };

  const removeUpload = (item: AttachmentDescriptor) => {
    if (!sessionName) return;
    Alert.alert('Delete upload?', item.name, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await attachmentsApi.remove(sessionName, item.attachment_id);
          void load();
        },
      },
    ]);
  };

  const pickFiles = async () => {
    if (!sessionName) return;
    const result = await DocumentPicker.getDocumentAsync({
      type: '*/*',
      multiple: true,
      copyToCacheDirectory: true,
    });
    if (result.canceled) return;
    setUploading(true);
    try {
      for (const asset of result.assets) {
        await attachmentsApi.upload(sessionName, asset as PickedDocument);
      }
      await load();
    } catch (error) {
      Alert.alert('Upload failed', String(error));
    } finally {
      setUploading(false);
    }
  };

  // Build unified list: visualizations first, then model artifacts, then uploads
  const visualizations = artifacts.filter(a => a.kind === 'visualization');
  const modelArtifacts = artifacts.filter(a => a.kind !== 'visualization');
  const totalCount = visualizations.length + modelArtifacts.length + uploads.length;

  const sections: { title: string; subtitle: string; count: number }[] = [
    { title: 'Visualizations', subtitle: 'Interactive HTML published by the agent', count: visualizations.length },
    { title: 'Model artifacts', subtitle: 'Files published by the agent', count: modelArtifacts.length },
    { title: 'File uploads', subtitle: 'Your uploaded source files for this session', count: uploads.length },
  ];

  const items: (UnifiedItem | { kind: 'header'; title: string; subtitle: string; count: number })[] = [
    ...visualizations.map(a => ({ kind: 'visualization' as const, artifact: a })),
    ...modelArtifacts.map(a => ({ kind: 'model' as const, artifact: a })),
    ...uploads.map(a => ({ kind: 'upload' as const, attachment: a })),
  ];

  const renderItem = ({ item }: { item: UnifiedItem | { kind: 'header'; title: string; subtitle: string; count: number } }) => {
    if (item.kind === 'header') {
      return (
        <View style={[styles.sectionHeader, { borderBottomColor: colors.border }]}>
          <View style={{ flex: 1 }}>
            <Text variant="xs" style={{ fontWeight: '700', letterSpacing: 0.8, color: colors.textDim }}>
              {item.title.toUpperCase()}
            </Text>
            <Text variant="xs" dim style={{ marginTop: 2 }}>
              {item.subtitle}
            </Text>
          </View>
          <View style={[styles.countBadge, { backgroundColor: colors.bgHover }]}>
            <Text variant="xs" style={{ fontWeight: '700', color: colors.textDim }}>{item.count}</Text>
          </View>
        </View>
      );
    }

    if (item.kind === 'visualization') {
      const a = item.artifact;
      return (
        <TouchableOpacity
          onPress={() => setViewingArtifact(a)}
          style={[styles.row, { borderBottomColor: colors.border }]}
          activeOpacity={0.7}
        >
          <View style={[styles.rowIcon, { backgroundColor: colors.accentSoft }]}>
            <Ionicons name="analytics-outline" size={20} color={colors.accent} />
          </View>
          <View style={styles.rowCopy}>
            <Text variant="sm" style={{ fontWeight: '600' }} numberOfLines={1}>
              {a.title || a.name}
            </Text>
            <Text variant="xs" dim numberOfLines={1}>
              {formatBytes(a.size)} · visualization
            </Text>
          </View>
          <TouchableOpacity
            onPress={() => sessionName && openExternalUrl(artifactsApi.downloadUrl(sessionName, a.artifact_id))}
            style={styles.rowAction}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            accessibilityRole="button"
            accessibilityLabel="Download visualization"
          >
            <Ionicons name="download-outline" size={18} color={colors.textDim} />
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => removeArtifact(a)}
            style={styles.rowAction}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="trash-outline" size={18} color={colors.error} />
          </TouchableOpacity>
        </TouchableOpacity>
      );
    }

    if (item.kind === 'model') {
      const a = item.artifact;
      return (
        <TouchableOpacity
          onPress={() => sessionName && openExternalUrl(artifactsApi.downloadUrl(sessionName, a.artifact_id))}
          style={[styles.row, { borderBottomColor: colors.border }]}
          activeOpacity={0.7}
        >
          <View style={[styles.rowIcon, { backgroundColor: colors.accentSoft }]}>
            <Ionicons name="document-attach-outline" size={20} color={colors.accent} />
          </View>
          <View style={styles.rowCopy}>
            <Text variant="sm" style={{ fontWeight: '600' }} numberOfLines={1}>
              {a.title || a.name}
            </Text>
            <Text variant="xs" dim numberOfLines={1}>
              {formatBytes(a.size)} · {a.mime_type || 'file'}
            </Text>
          </View>
          <TouchableOpacity
            onPress={() => sessionName && openExternalUrl(artifactsApi.downloadUrl(sessionName, a.artifact_id))}
            style={styles.rowAction}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="download-outline" size={18} color={colors.textDim} />
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => removeArtifact(a)}
            style={styles.rowAction}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="trash-outline" size={18} color={colors.error} />
          </TouchableOpacity>
        </TouchableOpacity>
      );
    }

    // upload
    const u = item.attachment;
    return (
      <TouchableOpacity
        onPress={() => sessionName && openExternalUrl(attachmentsApi.downloadUrl(sessionName, u.attachment_id))}
        style={[styles.row, { borderBottomColor: colors.border }]}
        activeOpacity={0.7}
      >
        <View style={[styles.rowIcon, { backgroundColor: colors.bgHover }]}>
          <Ionicons name="cloud-upload-outline" size={20} color={colors.textDim} />
        </View>
        <View style={styles.rowCopy}>
          <Text variant="sm" style={{ fontWeight: '600' }} numberOfLines={1}>
            {u.name}
          </Text>
          <Text variant="xs" dim numberOfLines={1}>
            {formatBytes(u.size)} · {u.mime_type || 'file'}
          </Text>
        </View>
        <TouchableOpacity
          onPress={() => sessionName && openExternalUrl(attachmentsApi.downloadUrl(sessionName, u.attachment_id))}
          style={styles.rowAction}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Ionicons name="download-outline" size={18} color={colors.textDim} />
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => removeUpload(u)}
          style={styles.rowAction}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Ionicons name="trash-outline" size={18} color={colors.error} />
        </TouchableOpacity>
      </TouchableOpacity>
    );
  };

  // Build flat list with section headers interleaved
  type FlatItem = UnifiedItem | { kind: 'header'; title: string; subtitle: string; count: number };
  const flatData: FlatItem[] = [];
  let sectionIdx = 0;
  if (visualizations.length > 0) {
    flatData.push({ kind: 'header', ...sections[0] });
    visualizations.forEach(a => flatData.push({ kind: 'visualization', artifact: a }));
    sectionIdx++;
  }
  if (modelArtifacts.length > 0) {
    flatData.push({ kind: 'header', ...sections[1] });
    modelArtifacts.forEach(a => flatData.push({ kind: 'model', artifact: a }));
    sectionIdx++;
  }
  if (uploads.length > 0) {
    flatData.push({ kind: 'header', ...sections[2] });
    uploads.forEach(a => flatData.push({ kind: 'upload', attachment: a }));
    sectionIdx++;
  }

  return (
    <SafeAreaView edges={['top']} style={{ flex: 1, backgroundColor: colors.bg }}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: colors.border, paddingTop: insets.top + 4 }]}>
        <Text variant="lg" style={{ fontWeight: '700' }}>Session files</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity
            onPress={pickFiles}
            disabled={uploading || !sessionName}
            style={[styles.uploadBtn, { backgroundColor: colors.accent, opacity: uploading || !sessionName ? 0.5 : 1 }]}
          >
            {uploading ? (
              <ActivityIndicator color={colors.accentText} size="small" />
            ) : (
              <Ionicons name="cloud-upload-outline" size={18} color={colors.accentText} />
            )}
            <Text variant="xs" style={{ color: colors.accentText, fontWeight: '700', marginLeft: 6 }}>
              {uploading ? 'Uploading' : 'Upload'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* List */}
      <FlatList
        data={flatData}
        keyExtractor={(item, index) => {
          if (item.kind === 'header') return `header-${item.title}`;
          if (item.kind === 'visualization' || item.kind === 'model') return `art-${item.artifact.artifact_id}`;
          return `att-${item.attachment.attachment_id}`;
        }}
        renderItem={renderItem}
        contentContainerStyle={{ paddingBottom: 40 }}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.accent} />}
        ListEmptyComponent={
          totalCount === 0 ? (
            <EmptyState
              icon="file-tray-outline"
              title="No files yet"
              message="Agent outputs, visualizations, and your uploads will appear here in one place."
            />
          ) : null
        }
      />

      {/* In-app visualization viewer */}
      <SafeAreaModal
        visible={viewingArtifact !== null}
        animationType="fade"
        presentationStyle="fullScreen"
        statusBarTranslucent
        onRequestClose={() => setViewingArtifact(null)}
      >
        <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
          <View style={[styles.visHeader, { borderBottomColor: colors.border }]}>
            <View style={{ flex: 1, minWidth: 0 }}>
              <Text variant="xs" dim style={{ fontWeight: '700', letterSpacing: 0.8 }}>
                VISUALIZATION
              </Text>
              <Text numberOfLines={1} style={{ fontSize: 14, fontWeight: '600', color: colors.text }}>
                {viewingArtifact?.title || viewingArtifact?.name || ''}
              </Text>
            </View>
            {sessionName && viewingArtifact && (
              <TouchableOpacity
                onPress={() => openExternalUrl(artifactsApi.downloadUrl(sessionName, viewingArtifact.artifact_id))}
                style={[styles.iconButton, { backgroundColor: colors.bgHover }]}
                accessibilityRole="button"
                accessibilityLabel="Download visualization"
              >
                <Ionicons name="download-outline" size={18} color={colors.textDim} />
              </TouchableOpacity>
            )}
            <TouchableOpacity
              onPress={() => setViewingArtifact(null)}
              style={[styles.closeButton, { backgroundColor: colors.bgHover }]}
              accessibilityRole="button"
              accessibilityLabel="Close visualization"
            >
              <Ionicons name="close" size={24} color={colors.text} />
            </TouchableOpacity>
          </View>
          {sessionName && viewingArtifact && (
            <WebView
              source={{ uri: artifactUri }}
              style={{ flex: 1, backgroundColor: '#ffffff' }}
              originWhitelist={['http://*', 'https://*']}
              javaScriptEnabled
              domStorageEnabled
              cacheEnabled={false}
              // Same hardening as VisualizationCard's InteractiveWebView: an
              // artifact is untrusted active content — confine it to the exact
              // backend URL, no cookies/file access, no mixed content.
              incognito
              sharedCookiesEnabled={false}
              thirdPartyCookiesEnabled={false}
              allowFileAccess={false}
              allowUniversalAccessFromFileURLs={false}
              javaScriptCanOpenWindowsAutomatically={false}
              setSupportMultipleWindows={false}
              mixedContentMode="never"
              scrollEnabled
              nestedScrollEnabled
              showsVerticalScrollIndicator
              showsHorizontalScrollIndicator
              onShouldStartLoadWithRequest={(request) => {
                if (request.url === artifactUri || request.url === 'about:blank') return true;
                // Any other navigation attempt goes to the OS browser instead
                // (http/https only — see openExternalUrl).
                openExternalUrl(request.url);
                return false;
              }}
            />
          )}
        </SafeAreaView>
      </SafeAreaModal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingBottom: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  headerActions: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  uploadBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  countBadge: {
    minWidth: 26,
    height: 24,
    borderRadius: 12,
    paddingHorizontal: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: 64,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  rowIcon: {
    width: 42,
    height: 42,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rowCopy: {
    flex: 1,
    marginLeft: 12,
  },
  rowAction: {
    width: 40,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  visHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  iconButton: {
    width: 36,
    height: 36,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeButton: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
});