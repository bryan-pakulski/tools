import React, { useState, useCallback } from 'react';
import { View, ActivityIndicator, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useTheme } from '../theme/ThemeContext';
import { refreshPendingMutations, useConnectionStore } from '../store/connection';
import { Text, Input, Button, Card } from '../components';
import { checkHealth } from '../api/client';
import { validateBaseUrl } from '../api/urlSafety';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { spacing } from '../theme/tokens';

export function ConnectionScreen() {
  const { colors } = useTheme();
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { baseUrl, isConnected, setBaseUrl, setConnected, loadFromStorage } = useConnectionStore();
  const [url, setUrl] = useState(baseUrl);
  const [testing, setTesting] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useFocusEffect(
    useCallback(() => {
      if (!loaded) {
        loadFromStorage().then(() => {
          setUrl(useConnectionStore.getState().baseUrl);
          setLoaded(true);
          // G5 (§3.6): hydrate the queued-mutations badge from storage.
          void refreshPendingMutations();
        });
      }
    }, [loaded, loadFromStorage]),
  );

  const testConnection = async () => {
    if (testing) return;
    const validated = validateBaseUrl(url);
    if (!validated.ok) {
      Alert.alert('Invalid server URL', validated.error);
      return;
    }
    const candidate = validated.url;

    // MUCLI_MOBILE_RECONNECT_YOLO_V1: transactional host test. Do not replace
    // the persisted working host until the candidate has answered healthz.
    setTesting(true);
    const MAX_ATTEMPTS = 3;
    const BACKOFF_MS = 1_500;
    let reachable = false;
    try {
      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        if (await checkHealth(candidate, { timeoutMs: 5_000 })) {
          reachable = true;
          break;
        }
        if (attempt < MAX_ATTEMPTS) {
          await new Promise(resolve => setTimeout(resolve, BACKOFF_MS * attempt));
        }
      }

      if (!reachable) {
        Alert.alert(
          'Connection failed',
          `Could not reach ${candidate}. The previous saved host was not changed.`,
        );
        return;
      }

      setBaseUrl(candidate);
      setConnected(true);
      // Round-19 F44: replay fired only from autoReconnect — an explicit
      // user reconnection left queued offline mutations stranded until
      // some later reconnect happened. Fire-and-report here too; the
      // outcome surfaces through the queue badge on the next screen.
      useConnectionStore
        .getState()
        .replayPending()
        .catch(() => {
          // Badge refresh inside replayPending already recorded the
          // failure; never block navigation on it.
        });
      if (navigation.canGoBack()) navigation.goBack();
      else navigation.navigate('Chat');
    } finally {
      setTesting(false);
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <View style={{ flex: 1, padding: spacing.base }}>
        <Text variant="lg" style={{ marginBottom: spacing.base }}>
          Connection Settings
        </Text>
        <Card style={{ marginBottom: spacing.base }}>
          <Text variant="sm" style={{ color: colors.textDim, marginBottom: spacing.xs }}>
            Server URL
          </Text>
          <Input
            value={url}
            onChangeText={setUrl}
            placeholder="http://localhost:30311"
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
          <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: spacing.sm, gap: 6 }}>
            <View style={{
              width: 8, height: 8, borderRadius: 4,
              backgroundColor: isConnected ? colors.success : colors.error,
            }} />
            <Text variant="xs" style={{ color: colors.textDim }}>
              {isConnected ? 'Connected' : 'Not connected'}
            </Text>
          </View>
        </Card>
        <Button title={testing ? 'Testing…' : 'Test Connection'} onPress={testConnection} disabled={testing} />
        {testing && <ActivityIndicator color={colors.accent} style={{ marginTop: spacing.sm }} />}
      </View>
    </SafeAreaView>
  );
}
