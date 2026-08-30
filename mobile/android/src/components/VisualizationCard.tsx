import React, { useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Modal,
  StyleSheet,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { SafeAreaModal } from './SafeAreaModal';
import { Ionicons } from '@expo/vector-icons';
import { WebView } from 'react-native-webview';
import type { ArtifactDescriptor } from '../api/artifacts';
import { artifactsApi } from '../api/artifacts';
import { openExternalUrl } from '../api/urlSafety';
import type { ThemeColors } from '../theme/tokens';
import { useTheme } from '../theme/ThemeContext';
import { Text } from './Text';

const MUCLI_MOBILE_VISUALIZATION_CONTROLS_V1 = true;

interface Props {
  artifact: ArtifactDescriptor;
  sessionName: string;
  onInteractionChange?: (active: boolean) => void;
}

interface InteractiveWebViewProps {
  uri: string;
  colors: ThemeColors;
  onOpenExternal: (target?: string) => void;
  onInteractionChange?: (active: boolean) => void;
}

const SCROLL_CONTAINMENT_SCRIPT = `
(() => {
  const apply = () => {
    document.documentElement.style.overscrollBehavior = 'contain';
    if (document.body) document.body.style.overscrollBehavior = 'contain';
  };
  apply();
  document.addEventListener('DOMContentLoaded', apply, { once: true });
})();
true;
`;

function InteractiveWebView({
  uri,
  colors,
  onOpenExternal,
  onInteractionChange,
}: InteractiveWebViewProps) {
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setLoading(true);
    setFailed(false);
  }, [uri]);

  const finishInteraction = () => onInteractionChange?.(false);

  return (
    <View
      style={styles.webviewHost}
      onTouchStart={() => onInteractionChange?.(true)}
      onTouchEnd={finishInteraction}
      onTouchCancel={finishInteraction}
    >
      {failed ? (
        <View style={styles.fallback}>
          <Ionicons name="warning-outline" size={20} color={colors.error} />
          <Text variant="sm" dim style={styles.fallbackText}>Inline preview failed.</Text>
          <TouchableOpacity onPress={() => onOpenExternal()}>
            <Text variant="sm" style={{ color: colors.accent, fontWeight: '600' }}>
              Open in browser
            </Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <WebView
            source={{ uri }}
            style={styles.webview}
            originWhitelist={['http://*', 'https://*']}
            javaScriptEnabled
            domStorageEnabled
            cacheEnabled={false}
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
            overScrollMode="content"
            injectedJavaScriptBeforeContentLoaded={SCROLL_CONTAINMENT_SCRIPT}
            onLoadEnd={() => setLoading(false)}
            onError={() => {
              setLoading(false);
              setFailed(true);
              finishInteraction();
            }}
            onHttpError={() => {
              setLoading(false);
              setFailed(true);
              finishInteraction();
            }}
            onShouldStartLoadWithRequest={(request) => {
              if (request.url === uri || request.url === 'about:blank') return true;
              onOpenExternal(request.url);
              return false;
            }}
          />
          {loading ? (
            <View style={[styles.loading, { backgroundColor: colors.bgLift }]}>
              <ActivityIndicator color={colors.accent} />
            </View>
          ) : null}
        </>
      )}
    </View>
  );
}

function VisualizationCardImpl({
  artifact,
  sessionName,
  onInteractionChange,
}: Props) {
  const { colors, isDark } = useTheme();
  const { height: windowHeight } = useWindowDimensions();
  const [expanded, setExpanded] = useState(false);
  const [fullScreen, setFullScreen] = useState(false);
  const uri = useMemo(
    () => `${artifactsApi.viewUrl(sessionName, artifact.artifact_id)}?mucli_theme=${isDark ? 'dark' : 'light'}`,
    [artifact.artifact_id, isDark, sessionName],
  );
  const frameHeight = Math.max(
    220,
    Math.min(Number(artifact.height) || 480, Math.min(620, windowHeight * 0.62)),
  );

  const openExternal = (target: string = uri) => {
    void openExternalUrl(target);
  };

  const hidePreview = () => {
    onInteractionChange?.(false);
    setExpanded(false);
  };

  const openFullScreen = () => {
    onInteractionChange?.(false);
    setFullScreen(true);
  };

  return (
    <>
      <View style={[styles.card, { borderColor: colors.border, backgroundColor: colors.bgLift }]}>
        <View style={styles.header}>
          <View style={styles.titleWrap}>
            <Text variant="xs" dim style={styles.eyebrow}>VISUALIZATION</Text>
            <Text numberOfLines={1} style={[styles.title, { color: colors.text }]}>
              {artifact.title || artifact.name}
            </Text>
          </View>

          <View style={styles.headerActions}>
            <TouchableOpacity
              onPress={openFullScreen}
              style={[styles.iconButton, { backgroundColor: colors.bgHover }]}
              accessibilityRole="button"
              accessibilityLabel="Open visualization full screen"
            >
              <Ionicons name="expand-outline" size={17} color={colors.textDim} />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => openExternal()}
              style={[styles.iconButton, { backgroundColor: colors.bgHover }]}
              accessibilityRole="button"
              accessibilityLabel="Open visualization in browser"
            >
              <Ionicons name="open-outline" size={16} color={colors.textDim} />
            </TouchableOpacity>
          </View>
        </View>

        {!expanded ? (
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityLabel="Show interactive visualization preview"
            onPress={() => setExpanded(true)}
            style={[styles.previewPrompt, { borderColor: colors.border }]}
          >
            <Ionicons name="eye-outline" size={18} color={colors.accent} />
            <Text variant="sm" style={{ color: colors.accent, fontWeight: '600' }}>
              Show interactive preview
            </Text>
          </TouchableOpacity>
        ) : (
          <>
            <View style={[styles.previewActions, { borderColor: colors.border }]}>
              <TouchableOpacity
                onPress={hidePreview}
                style={styles.textAction}
                accessibilityRole="button"
                accessibilityLabel="Hide visualization preview"
              >
                <Ionicons name="eye-off-outline" size={16} color={colors.textDim} />
                <Text variant="xs" style={{ color: colors.textSoft, fontWeight: '600' }}>
                  Hide
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                onPress={openFullScreen}
                style={styles.textAction}
                accessibilityRole="button"
                accessibilityLabel="Open visualization full screen"
              >
                <Ionicons name="expand-outline" size={16} color={colors.textDim} />
                <Text variant="xs" style={{ color: colors.textSoft, fontWeight: '600' }}>
                  Full screen
                </Text>
              </TouchableOpacity>
            </View>
            <View style={[styles.frame, { height: frameHeight, borderColor: colors.border }]}>
              <InteractiveWebView
                uri={uri}
                colors={colors}
                onOpenExternal={openExternal}
                onInteractionChange={onInteractionChange}
              />
            </View>
          </>
        )}
      </View>

      <SafeAreaModal
        visible={fullScreen}
        animationType="fade"
        presentationStyle="fullScreen"
        statusBarTranslucent
        onRequestClose={() => setFullScreen(false)}
      >
        <SafeAreaView style={[styles.fullScreenRoot, { backgroundColor: colors.bg }]}>
          <View style={[styles.fullScreenHeader, { borderColor: colors.border }]}>
            <View style={styles.fullScreenTitleWrap}>
              <Text variant="xs" dim style={styles.eyebrow}>VISUALIZATION</Text>
              <Text numberOfLines={1} style={[styles.fullScreenTitle, { color: colors.text }]}>
                {artifact.title || artifact.name}
              </Text>
            </View>
            <TouchableOpacity
              onPress={() => openExternal()}
              style={[styles.iconButton, { backgroundColor: colors.bgHover }]}
              accessibilityRole="button"
              accessibilityLabel="Open visualization in browser"
            >
              <Ionicons name="open-outline" size={17} color={colors.textDim} />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => setFullScreen(false)}
              style={[styles.closeButton, { backgroundColor: colors.bgHover }]}
              accessibilityRole="button"
              accessibilityLabel="Exit full-screen visualization"
            >
              <Ionicons name="close" size={24} color={colors.text} />
            </TouchableOpacity>
          </View>
          <View style={styles.fullScreenFrame}>
            <InteractiveWebView
              uri={uri}
              colors={colors}
              onOpenExternal={openExternal}
            />
          </View>
        </SafeAreaView>
      </SafeAreaModal>
    </>
  );
}

export const VisualizationCard = React.memo(
  VisualizationCardImpl,
  (previous, next) => (
    previous.sessionName === next.sessionName
    && previous.artifact.artifact_id === next.artifact.artifact_id
    && previous.artifact.title === next.artifact.title
    && previous.artifact.name === next.artifact.name
    && previous.artifact.height === next.artifact.height
    && previous.onInteractionChange === next.onInteractionChange
  ),
);

const styles = StyleSheet.create({
  card: {
    marginHorizontal: 16,
    marginVertical: 10,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 14,
    overflow: 'hidden',
  },
  header: {
    minHeight: 54,
    paddingHorizontal: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  titleWrap: { flex: 1, minWidth: 0 },
  eyebrow: { fontWeight: '700', letterSpacing: 0.8, marginBottom: 2 },
  title: { fontSize: 14, fontWeight: '600' },
  headerActions: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  iconButton: {
    width: 34,
    height: 34,
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
  previewPrompt: {
    minHeight: 52,
    borderTopWidth: StyleSheet.hairlineWidth,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  previewActions: {
    minHeight: 42,
    paddingHorizontal: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 4,
  },
  textAction: {
    minHeight: 34,
    paddingHorizontal: 10,
    borderRadius: 9,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  frame: {
    borderTopWidth: StyleSheet.hairlineWidth,
    position: 'relative',
  },
  webviewHost: { flex: 1, position: 'relative' },
  webview: { flex: 1, backgroundColor: '#ffffff' },
  loading: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  fallback: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  fallbackText: { marginTop: 8, marginBottom: 10 },
  fullScreenRoot: { flex: 1 },
  fullScreenHeader: {
    minHeight: 60,
    paddingHorizontal: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  fullScreenTitleWrap: { flex: 1, minWidth: 0 },
  fullScreenTitle: { fontSize: 15, fontWeight: '700' },
  fullScreenFrame: { flex: 1 },
});
