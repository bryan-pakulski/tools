import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../theme/ThemeContext';
import { useConnectionStore } from '../store/connection';
import { Text, Button } from './';

/**
 * Round-31 F40: visible banner when offline-queue items are parked as
 * CONFLICTED (their If-Match lost the race — the server session moved on).
 * The replay halts on the first conflict, so these block everything queued
 * behind them until the user re-queues (retry, optionally against the
 * freshest captured revision) or discards them permanently.
 */
export function ConflictBanner({ style }: { style?: ViewStyle }) {
  const { colors, spacing, radii } = useTheme();
  const conflictIds = useConnectionStore(state => state.conflictIds);
  const requeueConflict = useConnectionStore(state => state.requeueConflict);
  const discardConflict = useConnectionStore(state => state.discardConflict);
  const replayPending = useConnectionStore(state => state.replayPending);
  // Round-32b F11: single-flight guard — double-tap or overlapping
  // requeue/discard must not run two load/modify/persist cycles (stale
  // writes resurrect discarded items). While true, BOTH buttons disable
  // (including accessibilityState for screen readers).
  const [actionInProgress, setActionInProgress] = React.useState(false);

  if (conflictIds.length === 0) return null;
  const count = conflictIds.length;

  const handleRequeue = () => {
    if (actionInProgress) return;
    setActionInProgress(true);
    void (async () => {
      try {
        // Round-32b F11: read the FRESHEST revision at call time, not the
        // render-captured hook value — a state probe may have landed since
        // this component rendered.
        const freshRevision = useConnectionStore.getState().sessionRevision;
        if (freshRevision === null) {
          // No fresh token available: keep the item PARKED (do not strip
          // its CAS guard and do not retry unguarded). Nothing changes;
          // the banner stays visible for the next state sync.
          return;
        }
        for (const id of conflictIds) {
          await requeueConflict(id, freshRevision);
        }
        await replayPending();
      } finally {
        setActionInProgress(false);
      }
    })();
  };

  const handleDiscard = () => {
    if (actionInProgress) return;
    setActionInProgress(true);
    void (async () => {
      try {
        for (const id of conflictIds) {
          await discardConflict(id);
        }
      } finally {
        setActionInProgress(false);
      }
    })();
  };

  return (
    <View
      style={[
        styles.banner,
        {
          backgroundColor: colors.errorBg,
          borderColor: colors.error,
          borderRadius: radii.lg,
          padding: spacing.base,
          gap: spacing.sm,
        },
        style,
      ]}
      accessibilityRole="alert"
      accessibilityLabel={`${count} offline ${count === 1 ? 'change' : 'changes'} in conflict`}
    >
      <View style={styles.row}>
        <Ionicons name="warning-outline" size={16} color={colors.error} />
        <Text variant="sm" style={{ color: colors.text, flex: 1 }}>
          {count === 1
            ? '1 offline change conflicts with the current session.'
            : `${count} offline changes conflict with the current session.`}
        </Text>
      </View>
      <Text variant="xs" dim>
        The session moved on while you were away. Re-queue to retry against the
        latest state, or discard to drop the change permanently.
      </Text>
      <View style={[styles.row, { gap: spacing.sm }]}>
        <Button
          title="Requeue"
          variant="secondary"
          onPress={handleRequeue}
          disabled={actionInProgress}
          style={styles.button}
        />
        <Button
          title="Discard"
          variant="danger"
          onPress={handleDiscard}
          disabled={actionInProgress}
          style={styles.button}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    borderWidth: StyleSheet.hairlineWidth,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  button: {
    flex: 1,
  },
});