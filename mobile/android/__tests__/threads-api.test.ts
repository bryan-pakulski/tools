import { threadsApi, ThreadSummary } from '../src/api/threads';
import { useConnectionStore } from '../src/store/connection';

// Mock fetch
global.fetch = jest.fn();

function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => 'application/json' },
    json: () => Promise.resolve(body),
  };
}

const SAMPLE_LIST = {
  session_name: 'main',
  thread_group_id: 'g-1',
  current_thread_id: 't-1',
  threads: [
    {
      thread_id: 't-1',
      session_name: 'main',
      title: 'Main thread',
      status: 'running',
      current_goal: 'shipping feature',
      run_origin: '',
      runtime_id: 'r-1',
      last_seen: 1,
      created_at: 1,
      updated_at: 2,
      unread_count: 0,
      claimed_paths: [],
    },
    {
      thread_id: 't-2',
      session_name: 'thread-b',
      title: 'UI work',
      status: 'waiting_peer',
      current_goal: '',
      run_origin: '',
      runtime_id: '',
      last_seen: 0,
      created_at: 1,
      updated_at: 3,
      unread_count: 2,
      claimed_paths: ['src/api.ts'],
    },
  ],
};

describe('threadsApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useConnectionStore.getState().setBaseUrl('http://test:8000');
    useConnectionStore.getState().setActiveSession('main');
  });

  describe('list', () => {
    it('GETs /api/threads with auto-injected session_name', async () => {
      (fetch as jest.Mock).mockResolvedValue(jsonResponse(SAMPLE_LIST));
      const res = await threadsApi.list('main');
      expect(fetch).toHaveBeenCalledWith(
        'http://test:8000/api/threads?session_name=main',
        expect.objectContaining({ method: 'GET' }),
      );
      expect(res.thread_group_id).toBe('g-1');
      expect(res.threads).toHaveLength(2);
      expect(res.threads[1].unread_count).toBe(2);
    });

    it('passes explicit session_name over the injected one', async () => {
      (fetch as jest.Mock).mockResolvedValue(jsonResponse(SAMPLE_LIST));
      await threadsApi.list('other');
      expect(fetch).toHaveBeenCalledWith(
        'http://test:8000/api/threads?session_name=other',
        expect.objectContaining({ method: 'GET' }),
      );
    });
  });

  describe('activity', () => {
    it('GETs /api/threads/activity with after_id + limit params', async () => {
      (fetch as jest.Mock).mockResolvedValue(
        jsonResponse({
          thread_group_id: 'g-1',
          events: [
            {
              event_id: 7,
              kind: 'message',
              actor_thread_id: 't-1',
              target_thread_id: 't-2',
              message_id: 'm-1',
              conflict_id: '',
              payload: { body: 'need src/api.ts' },
              created_at: 123,
              actor_title: 'Main thread',
              target_title: 'UI work',
            },
          ],
          last_event_id: 7,
        }),
      );
      const res = await threadsApi.activity('main', { afterId: 3, limit: 50 });
      expect(fetch).toHaveBeenCalledWith(
        'http://test:8000/api/threads/activity?session_name=main&after_id=3&limit=50',
        expect.objectContaining({ method: 'GET' }),
      );
      expect(res.events[0].payload.body).toBe('need src/api.ts');
    });
  });

  describe('create', () => {
    it('POSTs /api/threads with parent + title and returns meta', async () => {
      (fetch as jest.Mock).mockResolvedValue(
        jsonResponse({
          ok: true,
          active: true,
          session_name: 'thread-c',
          thread_meta: { thread_id: 't-3', group_id: 'g-1', title: 'Docs sweep' },
        }),
      );
      const res = await threadsApi.create({ parentSessionName: 'main', title: 'Docs sweep' });
      expect(fetch).toHaveBeenCalledWith(
        'http://test:8000/api/threads?session_name=main',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            parent_session_name: 'main',
            title: 'Docs sweep',
            name: undefined,
            activate: true,
          }),
        }),
      );
      expect(res.thread_meta.thread_id).toBe('t-3');
    });

    it('surfaces 409 name clash as rejected fetch (ApiError path)', async () => {
      (fetch as jest.Mock).mockResolvedValue(
        jsonResponse({ detail: 'session name already exists' }, 409),
      );
      await expect(threadsApi.create({ parentSessionName: 'main', title: 'x', name: 'main' })).rejects.toThrow();
    });
  });

  describe('remove', () => {
    it('DELETEs a peer thread in the current group', async () => {
      (fetch as jest.Mock).mockResolvedValue(
        jsonResponse({ ok: true, thread_id: 't-2', deleted_session_name: 'thread-b' }),
      );
      const res = await threadsApi.remove('t-2', 'main');
      expect(fetch).toHaveBeenCalledWith(
        'http://test:8000/api/threads/t-2?session_name=main',
        expect.objectContaining({ method: 'DELETE' }),
      );
      expect(res.deleted_session_name).toBe('thread-b');
    });
  });
});
