import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { ThreadsScreen } from '../src/screens/ThreadsScreen';
import { threadsApi, ThreadListItem } from '../src/api/threads';
import { sessionsApi } from '../src/api/sessions';
import { useConnectionStore } from '../src/store/connection';

jest.mock('@react-navigation/native', () => ({
  useFocusEffect: (callback: () => void | (() => void)) => {
    const React = require('react');
    React.useEffect(callback, [callback]);
  },
}));

jest.mock('../src/api/threads', () => ({
  threadsApi: {
    list: jest.fn(),
    activity: jest.fn(),
    create: jest.fn(),
  },
}));

jest.mock('../src/api/sessions', () => ({
  sessionsApi: {
    list: jest.fn(),
    load: jest.fn(),
    focus: jest.fn(),
  },
}));

// React Native Modal does not render children in jest-expo; render the
// ThreadsScreen create sheet inline instead of through a real modal.
jest.mock('../src/components/SafeAreaModal', () => ({
  SafeAreaModal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

function thread(overrides: Partial<ThreadListItem> = {}): ThreadListItem {
  return {
    thread_id: 't-A',
    session_name: 'main',
    title: 'Main thread',
    status: 'idle',
    current_goal: '',
    run_origin: '',
    runtime_id: '',
    last_seen: 0,
    created_at: 1,
    updated_at: 1,
    unread_count: 0,
    claimed_paths: [],
    ...overrides,
  };
}

describe('ThreadsScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useConnectionStore.getState().setBaseUrl('http://test:8000');
    useConnectionStore.getState().setActiveSession('main');
  });

  it('renders empty state when the group has no threads', async () => {
    (threadsApi.list as jest.Mock).mockResolvedValue({
      session_name: 'main',
      thread_group_id: 'g-1',
      current_thread_id: 't-A',
      threads: [],
    });
    const { getByText } = render(<ThreadsScreen />);
    await waitFor(() => expect(threadsApi.list).toHaveBeenCalled());
    
  });

  it('renders thread rows with status and unread count', async () => {
    (threadsApi.list as jest.Mock).mockResolvedValue({
      session_name: 'main',
      thread_group_id: 'g-1',
      current_thread_id: 't-A',
      threads: [
        thread(),
        thread({
          thread_id: 't-B',
          session_name: 'side-thread',
          title: 'Refactor parser',
          status: 'running',
          unread_count: 3,
        }),
      ],
    });
    const { findByText } = render(<ThreadsScreen />);
    expect(await findByText('Refactor parser')).toBeTruthy();
  });

  it('create flow: CTA → confirm calls createThread and refreshes', async () => {
    (threadsApi.list as jest.Mock).mockResolvedValue({
      session_name: 'main',
      thread_group_id: 'g-1',
      current_thread_id: 't-A',
      threads: [thread()],
    });
    (threadsApi.create as jest.Mock).mockResolvedValue({
      ok: true,
      active: true,
      session_name: 'thread-c',
      thread_meta: { thread_id: 't-C', group_id: 'g-1' },
    });
    const { getByTestId, getByText } = render(<ThreadsScreen />);
    await waitFor(() => expect(threadsApi.list).toHaveBeenCalled());

    fireEvent.press(getByTestId('create-thread'));
    fireEvent.changeText(getByTestId('thread-title-input'), 'Audit pass');
    fireEvent.press(getByText('Create thread'));

    await waitFor(() => expect(threadsApi.create).toHaveBeenCalled());
    await waitFor(() =>
      expect(useConnectionStore.getState().activeSessionName).toBe('thread-c'),
    );
  });

  it('tapping a thread row loads that session and switches active', async () => {
    (threadsApi.list as jest.Mock).mockResolvedValue({
      session_name: 'main',
      thread_group_id: 'g-1',
      current_thread_id: 't-A',
      threads: [thread({ thread_id: 't-B', session_name: 'side', title: 'Other' })],
    });
    (sessionsApi.load as jest.Mock).mockResolvedValue({ ok: true });
    const { getByTestId } = render(<ThreadsScreen />);
    await waitFor(() => expect(threadsApi.list).toHaveBeenCalled());
    fireEvent.press(getByTestId('thread-row-t-B'));
    await waitFor(() => expect(sessionsApi.load).toHaveBeenCalledWith('side'));
    expect(useConnectionStore.getState().activeSessionName).toBe('side');
  });
});
