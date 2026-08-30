from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_shared_popout_positioner_covers_all_anchored_list_surfaces():
    script = read("mu/gui/static/js/floating_popouts.js")
    for selector in (
        ".cmd-complete",
        ".trace-info-popout",
        ".manager-overflow-menu",
        ".mode-popout",
    ):
        assert selector in script
    assert "const placeAbove = above > below" in script
    assert "layer.dataset.placement = placeAbove ? 'top' : 'bottom'" in script
    assert "window.visualViewport" in script
    assert "layer.style.maxHeight" in script
    assert "document.addEventListener('scroll', schedule" in script


def test_shared_popout_assets_load_on_shell_and_trace_pages():
    base = read("mu/gui/templates/base.html")
    trace = read("mu/gui/templates/trace.html")
    for source in (base, trace):
        assert '/static/css/floating_popouts.css' in source
        assert '/static/js/floating_popouts.js' in source


def test_product_popouts_do_not_invent_offscreen_minimum_height():
    script = read("mu/gui/static/js/product.js")
    assert "Math.max(48" in script
    assert "Math.max(170" not in script
    assert "window.visualViewport" in script
    assert "link.href = '/static/css/popouts.css'" not in script
