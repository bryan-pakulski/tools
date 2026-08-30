"""Regression guards for stable live conversation surfaces on web/mobile."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def function_block(source: str, name: str, next_name: str) -> str:
    return source.split(f"        {name}(", 1)[1].split(f"        {next_name}(", 1)[0]


def event_block(source: str, kind: str, next_kind: str) -> str:
    pattern = rf"if \(kind === '{re.escape(kind)}'\) \{{(.*?)if \(kind === '{re.escape(next_kind)}'\)"
    match = re.search(pattern, source, re.DOTALL)
    assert match, f"event block {kind!r} not found"
    return match.group(1)


def test_web_waits_for_successor_text_before_folding_previous_response():
    source = read("mu/gui/static/js/app.js")

    assert "_beginAssistantHandoff(slot, t, name)" in source
    assert "if (startsSuccessor && t.text)" in source
    assert "}, 260);" in source
    assert "this._foldLiveInterim(slot);" not in function_block(
        source, "addToolCall", "addToolResult"
    )
    assert "this._foldLiveInterim(slot);" not in function_block(
        source, "addThinking", "addInfo"
    )


def test_web_artifacts_and_subagents_are_top_level_timeline_anchors():
    source = read("mu/gui/static/js/app.js")
    template = read("mu/gui/templates/fragments/chat.html")

    assert 'turn.role === "visualization"' in source
    assert 'turn.role === "subagent_panel"' in source
    assert "_pushCollapsedSegments" in source
    assert "ct.role === 'visualization'" not in template
    assert "ct.role === 'subagent_panel'" not in template
    assert "t.role === 'visualization'" in template
    assert "t.role === 'subagent_panel'" in template


def test_web_subagent_surface_lingers_and_exposes_live_progress():
    source = read("mu/gui/static/js/app.js")
    template = read("mu/gui/templates/fragments/chat.html")

    assert "_schedulePanelDismiss" in source
    assert "subagentElapsed(a, $store.chat.clock)" in template
    assert "sap-progress" in template
    assert "a.iter" in template and "a.max_iter" in template
    assert "a.context_pct" in template
    for field in ("context_pct: ev.context_pct", "iter: ev.iter", "max_iter: ev.max_iter", "tokens_in: ev.tokens_in"):
        assert field in source


def test_web_subagent_cards_have_titles_iteration_progress_and_activity_drilldown():
    source = read("mu/gui/static/js/app.js")
    template = read("mu/gui/templates/fragments/chat.html")
    css = read("mu/gui/static/css/app.css")

    assert "subagentTitle(a)" in template
    assert "Iteration progress" in template
    assert "sap-progress-fill" in template
    assert "Action timeline" in template
    assert "toggleSubagentDetails" in source
    assert "_mergeSubagentActions" in source
    assert "d=\" + a.depth" not in template
    assert "sap-agent-mark" not in template
    assert "height: 10px" in css
    task_css = css.split(".sap-task", 1)[1].split("}", 1)[0]
    assert "text-overflow: ellipsis" not in task_css


def test_mobile_handoff_does_not_fold_on_tool_or_thinking_activity():
    source = read("mobile/android/src/hooks/useChatSession.ts")

    assert "scheduleAssistantHandoff" in source
    assert "handoff: 'leaving'" in source
    # Round-44 F5: deltas coalesce before flush; the handoff pin survives with
    # the flush-side variable name (`first`), preserving identical behavior.
    assert "handoff: first ? 'entering'" in source
    assert "foldLiveInterim(current, turnId)" in source
    assert "foldLiveInterim" not in event_block(source, "thinking_delta", "tool_call")
    assert "foldLiveInterim" not in event_block(source, "tool_call", "tool_result")


def test_mobile_visualizations_never_render_inside_interim_disclosures():
    hook = read("mobile/android/src/hooks/useChatSession.ts")
    product = read("mobile/android/src/screens/ChatScreenProduct.tsx")
    legacy = read("mobile/android/src/screens/ChatScreen.tsx")

    assert "isTimelineAnchor" in hook
    assert "appendCollapsedSegments" in hook
    assert "child.role === 'visualization'" not in product
    assert "child.role === 'visualization'" not in legacy
    assert "item.role === 'visualization'" in product


def test_mobile_has_first_class_live_subagent_surface():
    hook = read("mobile/android/src/hooks/useChatSession.ts")
    screen = read("mobile/android/src/screens/ChatScreenProduct.tsx")
    panel = read("mobile/android/src/components/SubagentActivityPanel.tsx")

    for kind in ("subagent_start", "subagent_progress", "subagent_end", "subagent_snapshot"):
        assert kind in hook
    assert "subagents?: LiveSubagent[]" in hook
    assert "<SubagentActivityPanel agents={item.subagents}" in screen
    assert "item.role === 'subagent_panel'" in screen
    assert "Iteration" in panel
    assert "Context" in panel
    assert "elapsedAt" in panel
    assert "tool_count" in panel and "tokens_in" in panel
    assert "taskTitle(agent)" in panel
    assert "Iteration progress" in panel
    assert "agentMark" not in panel
    assert "ACTION TIMELINE" in panel
    assert "agent.actions.map" in panel
    assert "Collapse subagent history" in panel
    assert "setPanelOpen(false)" in panel


def test_mobile_input_required_banner_is_centered_and_prominent():
    source = read("mobile/android/src/components/PromptHost.tsx")
    style = source.split("pendingBanner: {", 1)[1].split("},", 1)[0]

    assert "top: '50%'" in style
    assert "transform: [{ translateY: -38 }]" in style
    assert "bottom:" not in style
    assert "backgroundColor: colors.glassStrong" in source
    assert "borderColor: colors.accent" in source


def test_handoff_animations_exist_on_both_clients():
    css = read("mu/gui/static/css/conversation.css")
    mobile = read("mobile/android/src/screens/ChatScreenProduct.tsx")

    assert "interim-handoff-in" in css
    assert "interim-handoff-out" in css
    assert "prefers-reduced-motion" in css
    assert "function MessageHandoff" in mobile
    assert "Animated.timing" in mobile
