import { conversationStages } from '../src/components/ConversationStageRail';
import { fetchCheckpointHistory, type ChatMessage } from '../src/hooks/useChatSession';
import { sessionsApi, type SessionHistoryResponse } from '../src/api/sessions';

const user = (id: string, historyIndex?: number): ChatMessage => ({
  id,
  role: 'user',
  text: id,
  historyIndex,
});

describe('conversationStages', () => {
  it('uses one checkpoint per durable user turn and preserves list indexes', () => {
    const messages: ChatMessage[] = [
      user('history-4-0', 4),
      user('history-4-1', 4),
      { id: 'assistant-5', role: 'assistant', text: 'answer' },
      user('history-8-0', 8),
    ];

    expect(conversationStages(messages).map(stage => ({
      key: stage.key,
      messageIndex: stage.messageIndex,
    }))).toEqual([
      { key: 'history:4', messageIndex: 0 },
      { key: 'history:8', messageIndex: 3 },
    ]);
  });

  it('keeps separate live user prompts as separate checkpoints', () => {
    expect(conversationStages([user('live-a'), user('live-b')]).map(stage => stage.key))
      .toEqual(['live:live-a', 'live:live-b']);
  });
});

describe('fetchCheckpointHistory', () => {
  afterEach(() => jest.restoreAllMocks());

  it('keeps scanning raw pages until five user checkpoints are available', async () => {
    const response = (
      startIndex: number,
      turns: SessionHistoryResponse['turns'],
    ): SessionHistoryResponse => ({
      name: 'large-session',
      turns,
      start_index: startIndex,
      has_more: startIndex > 0,
      window_end: startIndex + turns.length,
      total_turns: 200,
    });
    const getHistory = jest.spyOn(sessionsApi, 'getHistory')
      .mockResolvedValueOnce(response(120, [
        { index: 120, role: 'assistant', parts: [] },
        { index: 121, role: 'tool', parts: [] },
      ]))
      .mockResolvedValueOnce(response(80, [
        { index: 80, role: 'user', parts: [] },
        { index: 88, role: 'user', parts: [] },
        { index: 96, role: 'user', parts: [] },
        { index: 104, role: 'user', parts: [] },
        { index: 112, role: 'user', parts: [] },
      ]));

    const result = await fetchCheckpointHistory('large-session', { beforeIndex: 200 });

    expect(getHistory).toHaveBeenCalledTimes(2);
    expect(getHistory.mock.calls[0][1]).toMatchObject({
      beforeIndex: 200,
      checkpointCount: 5,
    });
    expect(getHistory.mock.calls[1][1]).toMatchObject({
      beforeIndex: 120,
      checkpointCount: 5,
    });
    expect(result.turns.map(turn => turn.index)).toEqual([80, 88, 96, 104, 112, 120, 121]);
    expect(result.start_index).toBe(80);
  });
});
