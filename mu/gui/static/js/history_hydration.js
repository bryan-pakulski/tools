/* MuCLI persisted-history hydration correctness.
 *
 * chat.loadHistory() rebuilds the authoritative transcript, then calls
 * _groupIntermediateTurns(slot, previousTurns) so collapsed-group open state can
 * be preserved from the pre-hydration UI. The core grouping helper historically
 * treated previousTurns as the transcript to group, which replaced the freshly
 * rebuilt history with the old browser slot (usually empty after a reload).
 *
 * Keep the previous array only as UI-state input. Always group the current
 * slot.turns, then restore collapse-open state from the previous rendering.
 */
(function () {
    'use strict';

    function install() {
        if (!window.Alpine) return;
        const chat = Alpine.store('chat');
        if (!chat || chat.__authoritativeHistoryGroupingInstalled) return;
        if (typeof chat._groupIntermediateTurns !== 'function') return;

        chat.__authoritativeHistoryGroupingInstalled = true;
        const coreGroup = chat._groupIntermediateTurns.bind(chat);

        chat._groupIntermediateTurns = function (slot, previousTurns = slot.turns) {
            const previous = Array.isArray(previousTurns) ? previousTurns : [];

            // Normal end-of-turn grouping passes the current slot (or omits the
            // second argument). Preserve the original fast path unchanged.
            // Hydration invariant: when previousTurns !== slot.turns the caller
            // is grouping a stale pre-hydration array and we must re-group the
            // freshly rebuilt slot.turns below instead.
            if (previousTurns !== slot.turns) {
                // fall through to the rebuild path below
            } else if (previousTurns === slot.turns) {
                return coreGroup(slot, previousTurns);
            }

            const openByGroup = new Map();
            const openByUser = new Map();
            for (const turn of previous) {
                if (!turn || turn.role !== 'collapse') continue;
                if (turn.groupKey) openByGroup.set(turn.groupKey, Boolean(turn.open));
                if (turn.live && turn.userId) openByUser.set(turn.userId, Boolean(turn.open));
            }

            // Critical invariant: slot.turns is the newly rebuilt durable
            // transcript. Group that transcript, never the pre-hydration array.
            const result = coreGroup(slot, slot.turns);

            // The core helper used its second argument for open-state lookup as
            // well as content. Restore only that UI state from the old rendering.
            for (const turn of slot.turns || []) {
                if (!turn || turn.role !== 'collapse') continue;
                if (turn.groupKey && openByGroup.has(turn.groupKey)) {
                    turn.open = openByGroup.get(turn.groupKey);
                } else if (turn.userId && openByUser.has(turn.userId)) {
                    turn.open = openByUser.get(turn.userId);
                }
            }
            return result;
        };
    }

    document.addEventListener('alpine:init', () => queueMicrotask(install));
})();
