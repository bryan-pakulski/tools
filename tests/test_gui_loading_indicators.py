from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_web_history_spinner_tracks_request_generation_and_always_clears():
    script = read("mu/gui/static/js/app.js")
    template = read("mu/gui/templates/fragments/chat.html")

    assert "historyLoading: false" in script
    assert "slot.historyLoading = true" in script
    assert "if (slot.historyFetchGen === fetchGen) slot.historyLoading = false" in script
    assert 'x-show="$store.chat.historyLoading"' in template
    assert "Loading conversation history…" in template
    assert "Refreshing conversation…" in template
    assert 'class="ui-loading-spinner tiny"' in template


def test_data_heavy_web_regions_have_visible_loading_motion():
    sources = "\n".join(read(path) for path in (
        "mu/gui/templates/fragments/files_panel.html",
        "mu/gui/templates/fragments/artifacts_panel.html",
        "mu/gui/templates/fragments/history_panel.html",
        "mu/gui/templates/trace.html",
    ))
    css = read("mu/gui/static/css/app.css")

    assert sources.count("ui-loading-region") >= 6
    assert ".ui-loading-region::before" in css
    assert ".work-loading::before" in css
    assert "animation: container-create-spin" in css


def test_mobile_large_history_loading_combines_spinner_and_skeletons():
    screen = read("mobile/android/src/screens/ChatScreenProduct.tsx")
    loading_branch = screen[screen.index("historyLoading ? ("):]

    assert '<GeneratingIndicator label="Loading conversation history" />' in loading_branch
    assert "<Skeleton" in loading_branch
