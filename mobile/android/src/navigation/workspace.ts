import { Ionicons } from '@expo/vector-icons';

export type WorkspaceScreenName =
  | 'Modes'
  | 'Teacher'
  | 'Feature'
  | 'Research'
  | 'Security'
  | 'Loop'
  | 'Debug'
  | 'Memory'
  | 'Files'
  | 'Skills'
  | 'SystemPrompts'
  | 'History'
  | 'Threads'
  | 'Prompts'
  | 'Traces'
  | 'Audio'
  | 'Providers'
  | 'Connection'
  | 'Shell';

export type WorkspaceCategoryId = 'workflows' | 'context' | 'review' | 'runtime';

export type WorkspaceItem = {
  screen: WorkspaceScreenName;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
};

export type WorkspaceCategory = {
  id: WorkspaceCategoryId;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
  items: readonly WorkspaceItem[];
};

export const WORKSPACE_CATEGORIES: readonly WorkspaceCategory[] = [
  {
    id: 'workflows',
    title: 'Create & run',
    description: 'Choose how MuCLI approaches the task.',
    icon: 'flash-outline',
    items: [
      { screen: 'Modes', title: 'Modes', description: 'Select the active agent strategy.', icon: 'options-outline' },
      { screen: 'Teacher', title: 'Teacher', description: 'Build structured, source-grounded lessons.', icon: 'school-outline' },
      { screen: 'Feature', title: 'Feature plans', description: 'Break larger changes into approvable stages.', icon: 'layers-outline' },
      { screen: 'Research', title: 'Research', description: 'Investigate a topic without editing files.', icon: 'telescope-outline' },
      { screen: 'Security', title: 'Security', description: 'Run a verified security review workflow.', icon: 'shield-checkmark-outline' },
      { screen: 'Loop', title: 'Loop', description: 'Manage long-running autonomous work.', icon: 'repeat-outline' },
      { screen: 'Debug', title: 'Debug', description: 'Work through reproduce, locate, and fix.', icon: 'bug-outline' },
    ],
  },
  {
    id: 'context',
    title: 'Context',
    description: 'Control what MuCLI can see and remember.',
    icon: 'folder-open-outline',
    items: [
      { screen: 'Memory', title: 'Memory', description: 'Inspect saved context and scratchpad state.', icon: 'layers-outline' },
      { screen: 'Files', title: 'Files', description: 'Browse attached workspace files.', icon: 'folder-outline' },
      { screen: 'Skills', title: 'Skills', description: 'Review and manage agent extensions.', icon: 'sparkles-outline' },
      { screen: 'SystemPrompts', title: 'System prompts', description: 'Inspect the active instruction layers.', icon: 'document-text-outline' },
    ],
  },
  {
    id: 'review',
    title: 'Review',
    description: 'See decisions, activity, and pending input.',
    icon: 'pulse-outline',
    items: [
      { screen: 'Threads', title: 'Agent threads', description: 'Review peer conversations, ownership, and coordination activity.', icon: 'git-branch-outline' },
      { screen: 'History', title: 'History', description: 'Review earlier conversation turns.', icon: 'time-outline' },
      { screen: 'Prompts', title: 'Pending prompts', description: 'Answer requests waiting for your input.', icon: 'chatbox-ellipses-outline' },
      { screen: 'Traces', title: 'Traces', description: 'Inspect run telemetry and context growth.', icon: 'analytics-outline' },
      { screen: 'Audio', title: 'Audio', description: 'Manage audio input and transcription.', icon: 'mic-outline' },
    ],
  },
  {
    id: 'runtime',
    title: 'Runtime',
    description: 'Configure the model and server connection.',
    icon: 'settings-outline',
    items: [
      { screen: 'Providers', title: 'Providers', description: 'Choose a provider and model.', icon: 'server-outline' },
      { screen: 'Connection', title: 'Connection', description: 'Configure and test the MuCLI server.', icon: 'wifi-outline' },
      { screen: 'Shell', title: 'Shell', description: 'Open an interactive terminal into the session container.', icon: 'terminal-outline' },
    ],
  },
] as const;

export function getWorkspaceCategory(id: WorkspaceCategoryId): WorkspaceCategory {
  return WORKSPACE_CATEGORIES.find(category => category.id === id) ?? WORKSPACE_CATEGORIES[0];
}
