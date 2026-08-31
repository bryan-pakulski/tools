import React, { useCallback, useEffect, useState } from 'react';
import { NavigationContainer, DarkTheme, DefaultTheme, NavigationContainerRef } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Pressable, Text, View } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import type { WorkspaceCategoryId } from './workspace';

import { ConnectionPrompt } from '../components/ConnectionPrompt';
import { ContainerManagerSheet } from '../components/ContainerManagerSheet';
import { EdgeSwipeView } from '../components/EdgeSwipeView';
import { ModeDrawer } from '../components/ModeDrawer';
import { ModernHeader } from '../components/ModernHeader';
import { SessionStartPrompt } from '../components/SessionStartPrompt';
import { SwipeSessionsDrawer } from '../components/SwipeSessionsDrawer';
import { sessionsApi } from '../api/sessions';
import { useConnectionStore } from '../store/connection';

import { ChatScreenProduct } from '../screens/ChatScreenProduct';
import { WorkspaceScreen } from '../screens/WorkspaceScreen';
import { WorkspaceCategoryScreen } from '../screens/WorkspaceCategoryScreen';
import { WorkScreen } from '../screens/WorkScreen';
import { JobDetailScreen } from '../screens/JobDetailScreen';
import { JobAnalysisScreen } from '../screens/JobAnalysisScreen';
import { MemoryScreen } from '../screens/MemoryScreen';
import { FilesScreen } from '../screens/FilesScreen';
import { SkillsScreen } from '../screens/SkillsScreen';
import { AudioScreen } from '../screens/AudioScreen';
import { SessionTraceScreenV2 } from '../screens/SessionTraceScreenV2';
import { ProvidersScreen } from '../screens/ProvidersScreen';
import { ConnectionScreen } from '../screens/ConnectionScreen';
import { ModesScreen } from '../screens/ModesScreen';
import { PromptsScreen } from '../screens/PromptsScreen';
import { SystemPromptsScreen } from '../screens/SystemPromptsScreen';
import { TeacherScreen } from '../screens/TeacherScreen';
import { FeatureExplorerScreen } from '../screens/FeatureExplorerScreen';
import { ResearchScreen } from '../screens/ResearchScreen';
import { SecurityScreen } from '../screens/SecurityScreen';
import { LoopScreen } from '../screens/LoopScreen';
import { DebugScreen } from '../screens/DebugScreen';
import { HistoryScreen } from '../screens/HistoryScreen';
import { ThreadsScreen } from '../screens/ThreadsScreen';
import { ShellScreen } from '../screens/ShellScreen';
import { ArtifactsScreen } from '../screens/ArtifactsScreen';

export type RootStackParamList = {
  Chat: undefined;
  Work: undefined;
  JobDetail: { jobId: string };
  JobAnalysis: { jobId: string };
  Workspace: undefined;
  WorkspaceCategory: { categoryId: WorkspaceCategoryId; title: string };
  Teacher: undefined;
  Feature: undefined;
  Research: undefined;
  Security: undefined;
  Loop: undefined;
  Debug: undefined;
  History: undefined;
  Threads: undefined;
  SystemPrompts: undefined;
  Memory: undefined;
  Files: undefined;
  Skills: undefined;
  Audio: undefined;
  Traces: undefined;
  Providers: undefined;
  Connection: undefined;
  Modes: undefined;
  Prompts: undefined;
  Shell: undefined;
  Artifacts: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();
const navRef = React.createRef<NavigationContainerRef<RootStackParamList>>();

const PANEL_SCREENS: {
  name: Exclude<keyof RootStackParamList, 'Chat' | 'Work' | 'JobDetail' | 'JobAnalysis' | 'Workspace' | 'WorkspaceCategory'>;
  title: string;
  component: React.ComponentType;
}[] = [
  { name: 'Teacher', title: 'Teacher', component: TeacherScreen },
  { name: 'Feature', title: 'Feature plans', component: FeatureExplorerScreen },
  { name: 'Research', title: 'Research', component: ResearchScreen },
  { name: 'Security', title: 'Security', component: SecurityScreen },
  { name: 'Loop', title: 'Loop', component: LoopScreen },
  { name: 'Debug', title: 'Debug', component: DebugScreen },
  { name: 'History', title: 'History', component: HistoryScreen },
  { name: 'Threads', title: 'Agent conversations', component: ThreadsScreen },
  { name: 'SystemPrompts', title: 'System prompts', component: SystemPromptsScreen },
  { name: 'Memory', title: 'Memory Center', component: MemoryScreen },
  { name: 'Files', title: 'Files', component: FilesScreen },
  { name: 'Skills', title: 'Skills', component: SkillsScreen },
  { name: 'Audio', title: 'Audio', component: AudioScreen },
  { name: 'Traces', title: 'Session trace', component: SessionTraceScreenV2 },
  { name: 'Providers', title: 'Providers', component: ProvidersScreen },
  { name: 'Connection', title: 'Connection', component: ConnectionScreen },
  { name: 'Modes', title: 'Modes', component: ModesScreen },
  { name: 'Prompts', title: 'Pending prompts', component: PromptsScreen },
  { name: 'Shell', title: 'Shell', component: ShellScreen },
  { name: 'Artifacts', title: 'Session files', component: ArtifactsScreen },
];

function ChatScreenWithChrome() {
  const isConnected = useConnectionStore(state => state.isConnected);
  const baseUrl = useConnectionStore(state => state.baseUrl);
  const activeSessionName = useConnectionStore(state => state.activeSessionName);
  const setActiveSession = useConnectionStore(state => state.setActiveSession);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [modeOpen, setModeOpen] = useState(false);
  const [containersOpen, setContainersOpen] = useState(false);
  const [createRequestToken, setCreateRequestToken] = useState(0);

  const openSessions = useCallback(() => setSessionsOpen(true), []);
  const openMode = useCallback(() => {
    if (activeSessionName) setModeOpen(true);
  }, [activeSessionName]);
  const createSession = useCallback(() => {
    setSessionsOpen(false);
    setCreateRequestToken(value => value + 1);
  }, []);

  useEffect(() => {
    if (!isConnected) return;
    const controller = new AbortController();
    sessionsApi.list({ signal: controller.signal, timeoutMs: 8_000 })
      .then(response => {
        if (controller.signal.aborted) return;
        const selected = useConnectionStore.getState().activeSessionName;
        if (selected && response.loaded.includes(selected)) return;
        const current = response.current && response.loaded.includes(response.current)
          ? response.current
          : response.loaded[0] || null;
        setActiveSession(current);
      })
      .catch(() => {
        // The connection screen owns transport errors.
      });
    return () => controller.abort();
  }, [baseUrl, isConnected, setActiveSession]);

  return (
    <EdgeSwipeView onSwipeFromLeft={openSessions} onSwipeFromRight={openMode}>
      <View style={{ flex: 1, backgroundColor: 'transparent' }}>
        <ModernHeader
          onOpenSessions={openSessions}
          onOpenWork={() => isConnected ? navRef.current?.navigate('Work') : navRef.current?.navigate('Connection')}
          onOpenWorkspace={() => activeSessionName ? navRef.current?.navigate('Workspace') : openSessions()}
          onOpenTraces={() => activeSessionName ? navRef.current?.navigate('Traces') : openSessions()}
          onOpenThreads={() => activeSessionName ? navRef.current?.navigate('Threads') : openSessions()}
          onOpenConnection={() => navRef.current?.navigate('Connection')}
          onOpenModes={() => activeSessionName ? navRef.current?.navigate('Modes') : openSessions()}
          onOpenProviders={() => navRef.current?.navigate('Providers')}
          onOpenArtifacts={() => activeSessionName ? navRef.current?.navigate('Artifacts') : openSessions()}
          onOpenContainers={() => setContainersOpen(true)}
        />
        {!isConnected ? (
          <ConnectionPrompt onConnect={() => navRef.current?.navigate('Connection')} />
        ) : activeSessionName ? (
          <ChatScreenProduct />
        ) : (
          <SessionStartPrompt
            onLoadSession={openSessions}
            onCreateSession={createSession}
            onManageContainers={() => setContainersOpen(true)}
          />
        )}
        <SwipeSessionsDrawer
          visible={sessionsOpen}
          onClose={() => setSessionsOpen(false)}
          createRequestToken={createRequestToken}
        />
        <ContainerManagerSheet visible={containersOpen} onClose={() => setContainersOpen(false)} />
        <ModeDrawer
          visible={Boolean(activeSessionName) && modeOpen}
          onClose={() => setModeOpen(false)}
          onOpenModes={() => navRef.current?.navigate('Modes')}
        />
      </View>
    </EdgeSwipeView>
  );
}

export function AppNavigator() {
  const { colors, isDark } = useTheme();
  const baseTheme = isDark ? DarkTheme : DefaultTheme;
  const navTheme = {
    ...baseTheme,
    colors: {
      ...baseTheme.colors,
      background: 'transparent',
      card: colors.glassStrong,
      border: colors.hairline,
      text: colors.text,
      primary: colors.accent,
    },
  };

  return (
    <NavigationContainer ref={navRef} theme={navTheme}>
      <Stack.Navigator
        screenOptions={{
          animation: 'slide_from_right',
          contentStyle: { backgroundColor: colors.bg },
          headerStyle: { backgroundColor: colors.glassStrong },
          headerTintColor: colors.text,
          headerShadowVisible: false,
          headerBackTitle: '',
          headerTitleStyle: { fontSize: 16, fontWeight: '600' },
        }}
      >
        <Stack.Screen name="Chat" component={ChatScreenWithChrome} options={{ headerShown: false }} />
        <Stack.Screen name="Work" component={WorkScreen} options={{ title: 'Engineering work' }} />
        <Stack.Screen
          name="JobDetail"
          component={JobDetailScreen}
          options={({ navigation, route }) => ({
            title: 'Job review',
            headerRight: () => (
              <Pressable onPress={() => navigation.navigate('JobAnalysis', { jobId: route.params.jobId })} hitSlop={10}>
                <Text style={{ color: colors.textSoft, fontSize: 12, fontWeight: '600' }}>Analyze</Text>
              </Pressable>
            ),
          })}
        />
        <Stack.Screen name="JobAnalysis" component={JobAnalysisScreen} options={{ title: 'Job performance' }} />
        <Stack.Screen name="Workspace" component={WorkspaceScreen} options={{ title: 'Workspace' }} />
        <Stack.Screen
          name="WorkspaceCategory"
          component={WorkspaceCategoryScreen}
          options={({ route }: { route: { params: RootStackParamList['WorkspaceCategory'] } }) => ({ title: route.params.title })}
        />
        {PANEL_SCREENS.map(({ name, title, component: Comp }) => (
          <Stack.Screen key={name} name={name} component={Comp} options={{ title }} />
        ))}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
