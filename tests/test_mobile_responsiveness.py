from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_mobile_requests_are_finite_and_cancellable():
    source = read("mobile/android/src/api/client.ts")
    assert "DEFAULT_REQUEST_TIMEOUT_MS = 12_000" in source
    assert "const controller = new AbortController()" in source
    assert "Request timed out after" in source


def test_mobile_sse_reconnects_after_transport_drop():
    # Spec (test_mobile_streaming_perf Fix 1): pollingInterval must be 0 —
    # push-based SSE with reconnection owned by the explicit backoff loop.
    source = read("mobile/android/src/api/sse.ts")
    assert "DEFAULT_RECONNECT_DELAY_MS = 2_500" in source
    assert "pollingInterval: 0" in source
    assert "SSE push is the primary transport" in source


def test_mobile_history_is_bounded_and_stale_requests_abort():
    hook = read("mobile/android/src/hooks/useChatSession.ts")
    router = read("mu/gui/routers/sessions.py")
    assert "MOBILE_HISTORY_TURN_LIMIT = 80" in hook
    assert "MOBILE_HISTORY_CHECKPOINT_BATCH = 5" in hook
    assert "MOBILE_HISTORY_CHECKPOINT_SCAN_PAGES = 6" in hook
    assert "historyAbortRef.current?.abort()" in hook
    assert "stateAbortRef.current?.abort()" in hook
    assert "limit_turns: Optional[int]" in router
    assert "artifact_limit: Optional[int]" in router


def test_pending_prompt_does_not_automatically_cover_navigator():
    source = read("mobile/android/src/components/PromptHost.tsx")
    assert "if (!activeSessionName) return false" in source
    assert "Input required" in source
    assert "if (!reviewOpen)" in source
    assert "setInterval(recoverPending" not in source


def test_visualization_webviews_are_lazy_mounted():
    source = read("mobile/android/src/components/VisualizationCard.tsx")
    assert "const [expanded, setExpanded]" in source
    assert "Show interactive preview" in source
    assert "!expanded ?" in source


def test_connection_success_returns_to_chat():
    source = read("mobile/android/src/screens/ConnectionScreen.tsx")
    assert "timeoutMs: 5_000" in source
    assert "navigation.goBack()" in source
    assert "Alert.alert('Connected'" not in source
