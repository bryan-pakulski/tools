import React, { useCallback, useEffect, useState } from 'react';
import { Alert, StyleSheet, Switch, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../theme/ThemeContext';
import { queueIfOffline, useConnectionStore } from '../store/connection';
import { inspectorApi } from '../api/inspector';
import { AdvancedSettingsSheet } from './AdvancedSettingsSheet';
import { ModernBottomSheet } from './ModernBottomSheet';
import { Text } from './Text';

export type ModernHeaderProps = {
  onOpenSessions: () => void;
  onOpenWork: () => void;
  onOpenWorkspace: () => void;
  onOpenTraces: () => void;
  onOpenConnection: () => void;
  onOpenModes: () => void;
  onOpenProviders: () => void;
  onOpenArtifacts: () => void;
  onOpenContainers: () => void;
};

export function ModernHeader({
  onOpenSessions,
  onOpenWork,
  onOpenWorkspace,
  onOpenTraces,
  onOpenConnection,
  onOpenModes,
  onOpenProviders,
  onOpenArtifacts,
  onOpenContainers,
}: ModernHeaderProps) {
  const insets = useSafeAreaInsets();
  const { colors, isDark, toggleTheme } = useTheme();
  const {
    activeSessionName,
    activeProvider,
    activeModel,
    isConnected,
    yolo,
    setYolo,
    pendingMutations,
    lastReplay,
  } = useConnectionStore();
  const [menuOpen, setMenuOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [yoloSyncing, setYoloSyncing] = useState(false);

  const refreshYolo = useCallback(async () => {
    if (!isConnected || !activeSessionName) return;
    try {
      const response = await inspectorApi.getVariables(activeSessionName);
      for (const group of response.groups || []) {
        const variable = (group.variables || []).find(item => item.key === 'yolo');
        if (variable) {
          setYolo(Boolean(variable.value));
          return;
        }
      }
    } catch {
      // Preserve the last known value during a transient disconnect.
    }
  }, [activeSessionName, isConnected, setYolo]);

  useEffect(() => {
    void refreshYolo();
  }, [refreshYolo, menuOpen]);

  const updateYolo = useCallback(async (next: boolean) => {
    if (!activeSessionName || yoloSyncing) return;
    // G5 (§3.6): offline toggles join the outbound queue and replay with
    // If-Match on reconnect — the optimistic local value stays visible.
    // F11: carry the last-known session revision for CAS protection.
    if (await queueIfOffline('set_variable', { key: 'yolo', value: next }, {
      sessionName: activeSessionName,
      ifMatch: useConnectionStore.getState().sessionRevision ?? undefined,
    })) {
      setYolo(next);
      return;
    }
    const previous = yolo;
    setYolo(next);
    setYoloSyncing(true);
    try {
      const response = await inspectorApi.setVariable('yolo', next, activeSessionName);
      setYolo(Boolean(response.value));
    } catch (error) {
      setYolo(previous);
      Alert.alert('Could not update auto-approve', String(error));
    } finally {
      setYoloSyncing(false);
    }
  }, [activeSessionName, isConnected, setYolo, yolo, yoloSyncing]);

  const sessionTitle = activeSessionName || 'New session';
  const sessionMeta = [activeProvider, activeModel].filter(Boolean).join(' · ') || (isConnected ? 'Connected' : 'Connect to MuCLI');

  // G5 (§3.6): user-visible badge for queued offline mutations, plus a
  // conflict notice when a replay dropped stale state (409 from If-Match).
  const queueBadge = pendingMutations > 0 ? ` · ${pendingMutations} queued` : '';
  useEffect(() => {
    if (lastReplay && lastReplay.conflicts > 0) {
      Alert.alert(
        'Changes applied elsewhere',
        `${lastReplay.conflicts} queued change${lastReplay.conflicts === 1 ? '' : 's'} conflicted with newer session state and was dropped.`,
      );
    }
  }, [lastReplay]);

  const openFromMenu = (action: () => void) => {
    setMenuOpen(false);
    action();
  };

  return (
    <>
      <View
        style={[
          styles.container,
          {
            backgroundColor: colors.glass,
            borderBottomColor: colors.hairline,
            paddingTop: insets.top + 3,
          },
        ]}
      >
        <TouchableOpacity
          accessibilityRole="button"
          accessibilityLabel="Open sessions"
          onPress={onOpenSessions}
          style={styles.iconButton}
        >
          <Text style={[styles.brandMark, { color: colors.textSoft }]}>μ</Text>
        </TouchableOpacity>

        <TouchableOpacity
          accessibilityRole="button"
          accessibilityLabel={isConnected ? 'Current session' : 'Configure MuCLI connection'}
          disabled={isConnected}
          onPress={onOpenConnection}
          activeOpacity={0.72}
          style={styles.titleBlock}
        >
          <View style={styles.titleRow}>
            <Text style={[styles.title, { color: colors.text }]} numberOfLines={1}>
              {sessionTitle}
            </Text>
            <View style={[styles.statusDot, { backgroundColor: isConnected ? colors.textDim : colors.error }]} />
          </View>
          <Text style={[styles.subtitle, { color: isConnected ? colors.textDim : colors.error }]} numberOfLines={1}>
            {sessionMeta}{queueBadge}
          </Text>
        </TouchableOpacity>

        <View style={styles.headerActions}>
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityLabel="Open engineering work"
            disabled={!isConnected}
            onPress={onOpenWork}
            style={[styles.iconButton, !isConnected && styles.disabledAction]}
          >
            <Ionicons name="briefcase-outline" size={19} color={colors.textDim} />
          </TouchableOpacity>
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityLabel="Open Trace Analyzer"
            disabled={!activeSessionName}
            onPress={onOpenTraces}
            style={[styles.iconButton, !activeSessionName && styles.disabledAction]}
          >
            <Ionicons name="analytics-outline" size={19} color={colors.textDim} />
          </TouchableOpacity>
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityLabel="Open settings"
            onPress={() => setMenuOpen(true)}
            style={styles.iconButton}
          >
            <Ionicons name="settings-outline" size={20} color={colors.textDim} />
          </TouchableOpacity>
        </View>
      </View>

      <ModernBottomSheet visible={menuOpen} onClose={() => setMenuOpen(false)} title="Settings">
        {!isConnected && (
          <TouchableOpacity
            onPress={() => openFromMenu(onOpenConnection)}
            style={[styles.connectionBanner, { borderBottomColor: colors.hairline }]}
          >
            <Ionicons name="wifi-outline" size={20} color={colors.accent} />
            <View style={styles.menuCopy}>
              <Text variant="base" style={{ color: colors.text, fontWeight: '600' }}>Connect to MuCLI</Text>
              <Text variant="xs" style={{ color: colors.textDim }}>Configure a reachable GUI server</Text>
            </View>
            <Ionicons name="chevron-forward" size={17} color={colors.textDim} />
          </TouchableOpacity>
        )}

        <SettingsSection title="Session">
          <MenuRow icon="options-outline" label="Mode" detail="Choose the active agent strategy" onPress={() => openFromMenu(onOpenModes)} />
          <MenuRow icon="server-outline" label="Provider and model" detail={[activeProvider, activeModel].filter(Boolean).join(' · ') || 'Not selected'} onPress={() => openFromMenu(onOpenProviders)} />
          <MenuRow icon="wifi-outline" label="Connection" detail={isConnected ? 'Connected to MuCLI' : 'Not connected'} onPress={() => openFromMenu(onOpenConnection)} />
          <MenuRow
            icon="document-attach-outline"
            label="Session files"
            detail={activeSessionName ? 'Visualizations, model artifacts, and uploads' : 'Load a session to view files'}
            onPress={() => openFromMenu(onOpenArtifacts)}
          />
        </SettingsSection>

        <SettingsSection title="Runtime">
          <MenuRow
            icon="briefcase-outline"
            label="Engineering work"
            detail="Autonomous jobs, approvals, verification and review"
            onPress={() => openFromMenu(onOpenWork)}
          />
          <ToggleRow
            icon="flash-outline"
            label="Auto-approve writes"
            detail={yoloSyncing ? 'Updating session…' : 'Apply to this running session'}
            value={yolo}
            onValueChange={updateYolo}
            disabled={!isConnected || !activeSessionName || yoloSyncing}
          />
          <MenuRow icon="grid-outline" label="Workspace tools" detail="Context, workflows, and runtime controls" onPress={() => openFromMenu(onOpenWorkspace)} />
          <MenuRow
            icon="cube-outline"
            label="Containers"
            detail="Create, inspect, start, stop, and remove containers"
            onPress={() => openFromMenu(onOpenContainers)}
          />
        </SettingsSection>

        <SettingsSection title="Configuration">
          <MenuRow
            icon="options-outline"
            label="Session variables"
            detail={activeSessionName ? 'Grouped runtime overrides for this session' : 'Load a session to edit variables'}
            onPress={() => openFromMenu(() => setAdvancedOpen(true))}
          />
        </SettingsSection>

        <SettingsSection title="Appearance">
          <ToggleRow
            icon={isDark ? 'moon-outline' : 'sunny-outline'}
            label="Dark appearance"
            detail="Use the alternate colour scheme"
            value={isDark}
            onValueChange={() => toggleTheme()}
          />
        </SettingsSection>
      </ModernBottomSheet>
      <AdvancedSettingsSheet visible={advancedOpen} onClose={() => setAdvancedOpen(false)} />
    </>
  );
}

function SettingsSection({ title, children }: { title: string; children: React.ReactNode }) {
  const { colors } = useTheme();
  return (
    <View style={styles.section}>
      <Text variant="xs" style={[styles.sectionTitle, { color: colors.textDim }]}>{title}</Text>
      <View style={[styles.sectionBody, { borderTopColor: colors.hairline }]}>{children}</View>
    </View>
  );
}

type MenuRowProps = {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  detail: string;
  onPress: () => void;
};

function MenuRow({ icon, label, detail, onPress }: MenuRowProps) {
  const { colors } = useTheme();
  return (
    <TouchableOpacity
      onPress={onPress}
      activeOpacity={0.68}
      style={[styles.menuRow, { borderBottomColor: colors.hairline }]}
    >
      <View style={styles.menuIcon}>
        <Ionicons name={icon} size={19} color={colors.textDim} />
      </View>
      <View style={styles.menuCopy}>
        <Text variant="sm" style={{ color: colors.text, fontWeight: '600' }}>{label}</Text>
        <Text variant="xs" dim numberOfLines={2}>{detail}</Text>
      </View>
      <Ionicons name="chevron-forward" size={17} color={colors.textDim} />
    </TouchableOpacity>
  );
}

type ToggleRowProps = Omit<MenuRowProps, 'onPress'> & {
  value: boolean;
  onValueChange: (value: boolean) => void;
  disabled?: boolean;
};

function ToggleRow({ icon, label, detail, value, onValueChange, disabled = false }: ToggleRowProps) {
  const { colors } = useTheme();
  return (
    <View style={[styles.menuRow, { borderBottomColor: colors.hairline }]}>
      <View style={styles.menuIcon}>
        <Ionicons name={icon} size={19} color={colors.textDim} />
      </View>
      <View style={styles.menuCopy}>
        <Text variant="sm" style={{ color: colors.text, fontWeight: '600' }}>{label}</Text>
        <Text variant="xs" dim>{detail}</Text>
      </View>
      <Switch
        value={value}
        onValueChange={onValueChange}
        disabled={disabled}
        trackColor={{ false: colors.borderStrong, true: colors.accent }}
        thumbColor={colors.glassStrong}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    minHeight: 68,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingBottom: 9,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  headerActions: { flexDirection: 'row', alignItems: 'center', gap: 1 },
  iconButton: { width: 42, height: 42, alignItems: 'center', justifyContent: 'center' },
  disabledAction: { opacity: 0.32 },
  brandMark: {
    fontFamily: 'serif',
    fontSize: 30,
    lineHeight: 36,
    fontWeight: '400',
    textAlign: 'center',
    includeFontPadding: false,
  },
  titleBlock: { flex: 1, alignItems: 'flex-start', paddingHorizontal: 8 },
  titleRow: { maxWidth: '100%', flexDirection: 'row', alignItems: 'center', gap: 7 },
  title: { maxWidth: '92%', fontSize: 14.5, lineHeight: 19, fontWeight: '600', letterSpacing: -0.15 },
  subtitle: { maxWidth: '100%', marginTop: 1, fontSize: 10.5, lineHeight: 14 },
  statusDot: { width: 5, height: 5, borderRadius: 3 },
  connectionBanner: {
    minHeight: 58,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 4,
    borderBottomWidth: StyleSheet.hairlineWidth,
    marginBottom: 18,
  },
  section: { marginBottom: 21 },
  sectionTitle: { fontWeight: '600', marginBottom: 7, marginLeft: 2 },
  sectionBody: { borderTopWidth: StyleSheet.hairlineWidth },
  menuRow: {
    minHeight: 58,
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 2,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  menuIcon: { width: 28, alignItems: 'flex-start', justifyContent: 'center' },
  menuCopy: { flex: 1, marginHorizontal: 8 },
});
