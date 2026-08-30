import { ApiError, baseUrl, api, checkHealth } from '../src/api/client';
import { useConnectionStore } from '../src/store/connection';

// Mock fetch
global.fetch = jest.fn();

describe('API client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useConnectionStore.getState().setBaseUrl('http://test:8000');
    useConnectionStore.getState().setActiveSession(null);
  });

  describe('baseUrl', () => {
    it('returns the current baseUrl from store', () => {
      expect(baseUrl()).toBe('http://test:8000');
    });
  });

  describe('api.get', () => {
    it('calls fetch with correct URL', async () => {
      (fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ ok: true }),
      });

      const result = await api.get('/api/test');
      expect(fetch).toHaveBeenCalledWith(
        'http://test:8000/api/test',
        expect.objectContaining({ method: 'GET' }),
      );
      expect(result).toEqual({ ok: true });
    });

    it('appends session_name when active session set', async () => {
      useConnectionStore.getState().setActiveSession('mysession');
      (fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ ok: true }),
      });

      await api.get('/api/test');
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('session_name=mysession'),
        expect.any(Object),
      );
    });

    it('throws ApiError on non-ok response', async () => {
      (fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 404,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ detail: 'Not found' }),
      });

      await expect(api.get('/api/missing')).rejects.toThrow(ApiError);
    });
  });

  describe('api.post', () => {
    it('sends JSON body', async () => {
      (fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ ok: true }),
      });

      await api.post('/api/test', { foo: 'bar' });
      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ foo: 'bar' }),
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    });

    it('scopes mutations to the active session', async () => {
      useConnectionStore.getState().setActiveSession('container-feature');
      (fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ ok: true }),
      });

      await api.post('/api/modes/feature');
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('session_name=container-feature'),
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  describe('checkHealth', () => {
    it('returns true on successful healthz', async () => {
      (fetch as jest.Mock).mockResolvedValue({ ok: true });
      const result = await checkHealth('http://test:8000');
      expect(result).toBe(true);
      // checkHealth passes an AbortSignal for its 5s timeout — assert URL and
      // method, with the signal covered by expect.objectContaining.
      expect(fetch).toHaveBeenCalledWith(
        'http://test:8000/healthz',
        expect.objectContaining({ method: 'GET' }),
      );
    });

    it('returns false on fetch error', async () => {
      (fetch as jest.Mock).mockRejectedValue(new Error('Network error'));
      const result = await checkHealth('http://test:8000');
      expect(result).toBe(false);
    });
  });
});
