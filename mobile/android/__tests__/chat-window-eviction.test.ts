import { applyHistoryWindowEviction } from '../src/hooks/useChatSession';
import type { ChatMessage } from '../src/hooks/useChatSession';

const historyMessage = (turnIndex: number, partIndex = 0): ChatMessage => ({
  id: `history-${turnIndex}-${partIndex}`,
  role: 'assistant',
  text: `history message ${turnIndex}-${partIndex}`,
  streaming: false,
  origin: 'history',
});

const liveMessage = (text: string): ChatMessage => ({
  id: `live-${text}`,
  role: 'assistant',
  text,
  streaming: true,
  origin: 'stream',
});

describe('applyHistoryWindowEviction (Round-45 F8/F9: whole-turn eviction)', () => {
  it('keeps everything when history is within budget and reports the forward boundary', () => {
    const messages = [
      historyMessage(0),
      historyMessage(1),
      historyMessage(2),
    ];
    const result = applyHistoryWindowEviction(messages, { maxHistoryMessages: 5 });
    expect(result.messages).toBe(messages);
    expect(result.forwardCursor).toBe(3); // index after newest held turn (2) + 1
  });

  it('evicts NEWEST WHOLE turns beyond budget, never splitting a turn', () => {
    // Turn 3 has THREE parts — the budget boundary (2 kept) falls before it.
    // Turns 0-2 each keep the count at/below budget (kept whole); turn 3
    // crosses the budget BEFORE being added, so it is evicted whole.
    const messages = [
      historyMessage(0),
      historyMessage(1),
      historyMessage(2),
      historyMessage(3, 0),
      historyMessage(3, 1),
      historyMessage(3, 2),
      liveMessage('streaming now'),
    ];
    const result = applyHistoryWindowEviction(messages, { maxHistoryMessages: 2 });
    // Turns 0-1 kept (count reaches budget); turn 2 would cross it → evicted
    // whole along with turn 3. The budget is a hard cap for same-size turns.
    expect(result.messages.map(m => m.id)).toEqual([
      'history-0-0',
      'history-1-0',
      'live-streaming now',
    ]);
    // Forward cursor = index after newest wholly-kept turn (1) + 1 = 2.
    expect(result.forwardCursor).toBe(2);
  });

  it('keeps a turn whole when its group crosses the budget (no partial eviction)', () => {
    const messages = [
      historyMessage(0),
      historyMessage(1, 0),
      historyMessage(1, 1),
      historyMessage(1, 2),
      historyMessage(2),
    ];
    const result = applyHistoryWindowEviction(messages, { maxHistoryMessages: 2 });
    // Turn 1 has 3 parts: adding it would cross the budget (1+3=4>2), so it
    // is evicted WHOLE together with turn 2 — no partial turn, no orphans.
    // The whole-turn rule means the kept count can end below budget.
    expect(result.messages.map(m => m.id)).toEqual(['history-0-0']);
    expect(result.forwardCursor).toBe(1); // after turn 0
  });

  it('always survives live messages and unparseable history ids', () => {
    const weird = { ...historyMessage(0), id: 'history-unparseable' };
    const messages = [historyMessage(0), weird, historyMessage(2), liveMessage('x')];
    const result = applyHistoryWindowEviction(messages, { maxHistoryMessages: 1 });
    const ids = result.messages.map(m => m.id);
    // Unparseable ids can't be turn-grouped → always kept whole.
    expect(ids).toContain('history-unparseable');
    expect(ids).toContain('live-x');
    // Turn 0 keeps the count at budget; turn 2 crosses it → evicted whole.
    expect(ids).not.toContain('history-2-0');
    expect(result.forwardCursor).toBe(1); // after turn 0
  });

  it('handles an empty array', () => {
    const result = applyHistoryWindowEviction([], { maxHistoryMessages: 10 });
    expect(result.messages).toEqual([]);
    expect(result.forwardCursor).toBeNull();
  });
});