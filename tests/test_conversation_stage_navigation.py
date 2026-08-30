from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_web_conversation_stage_rail_tracks_and_jumps_to_user_turns():
    template = read("mu/gui/templates/fragments/chat.html")
    script = read("mu/gui/static/js/app.js")
    styles = read("mu/gui/static/css/product.css")

    assert 'class="conversation-stage-rail"' in template
    assert ':data-turn-id="t.id"' in template
    assert 'const key = Number.isInteger(turn.historyIndex)' in script
    assert "`live:${turn.id}`" in script
    assert "_updateActiveStage(el)" in script
    assert "box.getBoundingClientRect().top <= readingLine" in script
    assert "containerRect.height * 0.32" in script
    assert "jumpToStage(id)" in script
    assert "_turnBox(wrapper)" in script
    assert "display:contents" in script
    assert "const box = this._turnBox(node)" in script
    assert ':aria-current="stage.id === $store.chat.activeStageId' in template
    assert 'class="conversation-stage-marker conversation-stage-link"' in template
    assert 'class="conversation-stage-dot"' in template
    assert "$store.chat.current().hasMoreTurns" in template
    assert '@click="$store.chat.loadOlder(null, { navigate: true })"' in template
    assert "History continues below" not in template
    assert "conversation-stage-chevron down" not in template
    assert "loadNewer" not in script
    assert "node.scrollIntoView({ block: 'start'" in script
    assert "behavior: 'smooth'" not in script
    assert "conversation-stage-label" not in template
    assert "overflow-y: auto" in styles


def test_web_history_loads_checkpoint_batches_without_forward_arrow_paging():
    script = read("mu/gui/static/js/app.js")

    assert "WEB_HISTORY_PAGE_TURNS = 200" in script
    assert "WEB_HISTORY_CHECKPOINT_BATCH = 5" in script
    assert "WEB_HISTORY_CHECKPOINT_SCAN_PAGES = 6" in script
    assert "limit_turns: String(WEB_HISTORY_PAGE_TURNS)" in script
    assert "checkpoint_count: String(remaining)" in script
    assert "checkpointIndexes.size < WEB_HISTORY_CHECKPOINT_BATCH" in script
    assert "async loadOlder(name, { navigate = false } = {})" in script
    assert "dst.turns = this._timelineRange(merged, newStart, combinedEnd)" in script


def test_web_history_paging_preserves_content_and_layout_stability():
    script = read("mu/gui/static/js/app.js")
    template = read("mu/gui/templates/fragments/chat.html")
    styles = read("mu/gui/static/css/product.css")
    index = read("mu/gui/templates/index.html")

    assert "const turns = flatten(slot.turns);" in script
    assert "el.scrollTop + node.getBoundingClientRect().top - anchor.top" in script
    assert "enhanceHistoryRange(newStart, beforeIndex)" in script
    assert ':data-history-index="t.historyIndex ?? null"' in template
    assert "scrollbar-gutter: stable" in styles
    assert "overflow-anchor: none" in styles
    assert ".chat-history-loading {\n    position: absolute;" in styles
    assert '<link id="mucli-conversation-css"' in index
    assert index.index('/static/css/refinement.css') < index.index('/static/css/conversation.css')
    assert index.index('/static/css/conversation.css') < index.index('/static/js/product.js')


def test_mobile_conversation_stage_rail_uses_virtualized_list_indexes():
    screen = read("mobile/android/src/screens/ChatScreenProduct.tsx")
    rail = read("mobile/android/src/components/ConversationStageRail.tsx")
    hook = read("mobile/android/src/hooks/useChatSession.ts")
    api = read("mobile/android/src/api/sessions.ts")

    assert "<ConversationStageRail" in screen
    assert "onViewableItemsChanged={viewabilityRef.current}" in screen
    assert "viewabilityConfig={stageViewabilityConfigRef.current}" in screen
    assert "pendingCheckpointIndexRef.current = messageIndex" in screen
    assert "animated: false" in screen
    assert "onScrollToIndexFailed={onCheckpointScrollFailed}" in screen
    assert "checkpointJumpAttemptsRef.current >= 6" in screen
    assert "conversationStages(messages)" in rail
    assert "message.role !== 'user'" in rail
    assert "`history:${message.historyIndex}`" in rail
    assert "<ScrollView" in rail
    assert "accessibilityLabel={`Jump to prompt" in rail
    assert "chevron-up" in rail
    assert "chevron-down" not in rail
    assert "onLoadOlder={loadOlderCheckpoints}" in screen
    assert "const target = stages[anchorIndex - 1]" in screen

    assert "MOBILE_HISTORY_CHECKPOINT_BATCH = 5" in hook
    assert "MOBILE_HISTORY_CHECKPOINT_SCAN_PAGES = 6" in hook
    assert "checkpointCount: remaining" in hook
    assert "checkpointIndexes.size >= MOBILE_HISTORY_CHECKPOINT_BATCH" in hook
    assert "loadNewerHistory" not in hook
    assert "checkpoint_count: options?.checkpointCount" in api
