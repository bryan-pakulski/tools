"""Regression guards for the production web presentation layer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "mu" / "gui" / "templates" / "base.html"
INDEX = ROOT / "mu" / "gui" / "templates" / "index.html"
INSPECTOR = ROOT / "mu" / "gui" / "templates" / "fragments" / "inspector.html"
PANEL_TABS = ROOT / "mu" / "gui" / "templates" / "fragments" / "panel_tabs.html"
FILES_PANEL = ROOT / "mu" / "gui" / "templates" / "fragments" / "files_panel.html"
CONTAINERS = ROOT / "mu" / "gui" / "templates" / "containers.html"
PRODUCT_CSS = ROOT / "mu" / "gui" / "static" / "css" / "product.css"
CRYSTAL_CSS = ROOT / "mu" / "gui" / "static" / "css" / "crystal.css"
CLARITY_CSS = ROOT / "mu" / "gui" / "static" / "css" / "clarity.css"
REFINEMENT_CSS = ROOT / "mu" / "gui" / "static" / "css" / "refinement.css"
POPOUT_CSS = ROOT / "mu" / "gui" / "static" / "css" / "popouts.css"
ROUTE_CSS = ROOT / "mu" / "gui" / "static" / "css" / "route-product.css"
TRACE_CSS = ROOT / "mu" / "gui" / "static" / "css" / "trace.css"
CONTAINER_CRYSTAL_CSS = ROOT / "mu" / "gui" / "static" / "css" / "containers-crystal.css"
PRODUCT_JS = ROOT / "mu" / "gui" / "static" / "js" / "product.js"
WEB_SHELL_JS = ROOT / "mu" / "gui" / "static" / "js" / "web_shell.js"


def test_product_assets_are_loaded_by_main_web_shell():
    text = INDEX.read_text(encoding="utf-8")
    for asset in (
        '/static/css/product.css', '/static/css/crystal.css',
        '/static/css/clarity.css', '/static/css/refinement.css',
        '/static/js/product.js', '/static/js/web_shell.js',
    ):
        assert asset in text
    assert 'class="app product-app"' in text
    assert 'class="panel-stage"' in text
    assert 'fragments/panel_tabs.html' in text


def test_left_navigation_groups_threads_and_sessions():
    """Sidebar contract: Threads section = current session's thread-group roster
    (hidden for singleton groups); Sessions section = the FULL sessions list
    (main session always visible) with hover unload/delete row actions."""
    text = INDEX.read_text(encoding="utf-8")
    sidebar = text.split('<aside class="sidebar product-sidebar"', 1)[1].split('</aside>', 1)[0]
    assert '<span>Threads</span>' in sidebar
    assert '<span>Sessions</span>' in sidebar
    # Threads scoped to the group roster and hidden when it is a singleton.
    assert '$store.threads.groupRoster.length > 1' in sidebar
    assert 'x-for="t in $store.threads.groupRoster"' in sidebar
    # Sessions section iterates the full store list, not a roster-filtered view.
    assert 'x-for="s in $store.sessions.list"' in sidebar
    assert 'otherSessions' not in sidebar
    # Row actions: switch on row click, unload + confirm-guarded delete.
    assert '$store.threads.switchTo(t)' in sidebar
    assert '$store.sessions.switchTo(s.name)' in sidebar
    assert '$store.sessions.unload(s.name)' in sidebar
    assert '$store.sessions.remove(s.name)' in sidebar
    assert '$store.confirm.ask(' in sidebar
    assert 'New thread' in sidebar
    assert '+ new session' in sidebar
    assert '<span>Work</span>' not in sidebar
    assert '<span>Workspace</span>' not in sidebar
    assert '$store.mode.setView(m.name)' not in sidebar
    assert '$store.mode.setView(v.name)' not in sidebar


def test_header_has_trace_action_and_centered_settings_cog():
    text = INDEX.read_text(encoding="utf-8")
    assert "'/trace?session=' + encodeURIComponent($store.chat.currentName || '')" in text
    assert 'aria-label="Open Trace Analyzer"' in text
    assert 'class="product-icon-button settings-cog-button"' in text
    assert '<circle cx="12" cy="12" r="3"></circle>' in text
    assert 'M12.22 2h-.44' in text


def test_product_shell_keeps_core_navigation_contracts():
    text = INDEX.read_text(encoding="utf-8")
    assert '$store.sessions.switchTo(s.name)' in text
    assert '$store.inspector.openDrawer()' in text
    assert '$store.yolo.toggle()' in text
    assert "$store.mode.setView('memory'); $store.layout.panelOpen = true" in text
    assert 'aria-label="Toggle workspace panel"' in text
    assert 'href="/containers"' in text


def test_rhs_panel_uses_exactly_two_dropdowns_not_scrollable_tabs():
    text = PANEL_TABS.read_text(encoding="utf-8")
    css = REFINEMENT_CSS.read_text(encoding="utf-8")
    assert 'class="panel-view-select panel-mode-select"' in text
    assert 'class="panel-view-select panel-tool-select"' in text
    assert 'Select mode' in text
    assert 'Select tool' in text
    assert text.count('<select ') == 2
    assert 'panel-tabs-strip' not in text
    assert 'class="panel-tab"' not in text
    assert '$event.target.value && $store.mode.setView($event.target.value)' in text
    assert '$store.layout.panelOpen = false' in text
    assert '--panel-tabs-h: 64px' in css
    assert '.panel-selectors' in css
    assert '.panel-selector-label' in css
    assert 'border-bottom: 1px solid var(--hairline)' in css
    assert 'background-color: transparent' in css
    assert 'inset: var(--panel-tabs-h) 0 0 0 !important' in css


def test_product_css_covers_primary_web_surfaces():
    product = PRODUCT_CSS.read_text(encoding="utf-8")
    combined = ''.join(path.read_text(encoding="utf-8") for path in (
        PRODUCT_CSS, CRYSTAL_CSS, CLARITY_CSS, REFINEMENT_CSS, POPOUT_CSS,
    ))
    for selector in (
        '.product-header', '.product-sidebar', '.chat-history', '.msg.user',
        '.composer', '.panel-stage', '.panel-select-nav', '.inspector',
        '.welcome-entry', '.prompt-body .options label',
        '.composer-mode-popout', '.composer-settings-popout',
    ):
        assert selector in combined
    assert 'backdrop-filter' in combined
    assert '@media (max-width: 760px)' in combined
    assert '@media (prefers-reduced-motion: reduce)' in combined
    assert 'padding: 0 !important;' in product


def test_alpine_sunrise_is_atmospheric_not_css_scenery():
    css = CLARITY_CSS.read_text(encoding="utf-8")
    route = ROUTE_CSS.read_text(encoding="utf-8")
    for token in (
        '--sky-blue', '--glacier-blue', '--sunrise-pink', '--sunrise-peach',
        '--alpine-green', '--snow-glow',
    ):
        assert token in css
        assert token in route
    assert 'radial-gradient' in css
    assert 'radial-gradient' in route
    assert 'clip-path' not in css
    assert 'clip-path' not in route
    assert '--mountain-near' not in css
    assert '--mountain-far' not in css
    assert '--mountain-near' not in route
    assert '--mountain-far' not in route
    assert '.product-app::after' not in css
    assert 'body::after' not in route


def test_product_javascript_positions_overlays_and_is_presentation_only():
    js = PRODUCT_JS.read_text(encoding="utf-8")
    assert 'installPresentationStylesheet' in js
    assert "installPresentationStylesheet('mucli-popouts-css', '/static/css/popouts.css')" in js
    assert 'polishWelcomeCopy' in js
    assert 'focusComposer' in js
    assert 'preserveOverlayGeometry' in js
    assert 'refineComposerGeometry' in js
    assert 'positionFloatingLayer' in js
    assert 'floatingLayerProfile' in js
    assert 'installComposerFloatingLayers' in js
    assert 'installPanelTransitions' in js
    assert "dataset.placement = preferAbove ? 'top' : 'bottom'" in js
    assert "dataset.alignment = profile.align" in js
    assert "toolbar.style.removeProperty('bottom')" in js
    assert "toolbar.style.bottom = '96px'" not in js
    assert "composer.style.removeProperty('padding-top')" in js
    assert "main.style.flexDirection = 'row'" in js
    assert "node.style.position = 'fixed'" in js
    assert "event.key.toLowerCase() === 'k'" in js
    assert 'fetch(' not in js
    assert 'Alpine.store(' not in js


def test_composer_is_single_pane_with_controls_outside_typing_surface():
    css = REFINEMENT_CSS.read_text(encoding="utf-8")
    assert '.composer::before' in css
    assert 'content: none !important' in css
    assert '.composer-toolbar {' in css
    assert 'position: static !important' in css
    assert 'margin: 0 0 16px' in css
    assert 'justify-content: flex-end' in css
    assert '.composer form {' in css
    assert 'background: color-mix(in srgb, var(--glass-strong) 72%, transparent) !important' in css
    assert '.composer .hint {' in css
    assert 'min-height: 0 !important' in css
    assert 'background: transparent !important' in css


def test_composer_feature_and_provider_popouts_fit_content_cleanly():
    css = POPOUT_CSS.read_text(encoding="utf-8")
    js = PRODUCT_JS.read_text(encoding="utf-8")
    assert 'min-width: min(310px, calc(100vw - 20px))' in css
    assert 'max-width: min(420px, calc(100vw - 20px))' in css
    assert '.composer-mode-popout .mode-desc' in css
    assert 'max-width: 44ch' in css
    assert 'min-width: min(460px, calc(100vw - 20px))' in css
    assert 'max-width: min(590px, calc(100vw - 20px))' in css
    assert '.composer-settings-popout .settings-header' in css
    assert 'position: sticky' in css
    assert 'grid-template-columns: minmax(110px, 142px) minmax(0, 1fr)' in css
    assert '.composer-settings-popout .settings-var-select' in css
    assert "return { minWidth: 460, maxWidth: 590, align: 'end' }" in js
    assert "return { minWidth: 310, maxWidth: 420, align: 'start' }" in js
    assert 'layer.scrollWidth' in js
    assert "layer.style.width = ''" in js
    assert "layer.style.maxHeight = ''" in js


def test_session_history_hydrates_after_authoritative_focus():
    js = WEB_SHELL_JS.read_text(encoding="utf-8")
    assert 'sessions.load = async function' in js
    assert 'sessions.switchTo = async function' in js
    assert 'slot.historyHydrated' in js
    assert 'chat.focus(current)' in js
    assert 'chat.loadHistory(current, { force: true })' in js
    assert 'chat.loadHistory(name, { force: true })' in js
    assert 'slot.pendingReload = true' in js
    assert 'fetch(' not in js


def test_embedded_file_editor_and_codemirror_dependencies_are_removed():
    base = BASE.read_text(encoding="utf-8")
    files = FILES_PANEL.read_text(encoding="utf-8")
    for token in ('codemirror.min.css', 'codemirror.min.js', 'codemirror-simple.js', 'codemirror-modes.min.js'):
        assert token not in base
    for path in (
        ROOT / 'mu/gui/static/vendor/codemirror.min.css',
        ROOT / 'mu/gui/static/vendor/codemirror.min.js',
        ROOT / 'mu/gui/static/vendor/codemirror-simple.js',
        ROOT / 'mu/gui/static/vendor/codemirror-modes.min.js',
    ):
        assert not path.exists()
    assert 'CodeMirror' not in files
    assert 'files-editor-pane' not in files
    assert 'files-cm-host' not in files
    assert 'saveCurrent' not in files
    assert 'reloadFile' not in files
    assert 'filesBrowserPanel()' in files
    assert 'Browse the attached workspace' in files


def test_choice_picker_is_flat_not_card_based():
    css = CRYSTAL_CSS.read_text(encoding="utf-8")
    compact = css.replace(' ', '')
    assert '.prompt-body .options label' in css
    assert 'border-bottom:1pxsolidvar(--hairline)' in compact
    assert 'appearance: none' in css


def test_settings_drawer_is_spacious_clean_and_consistent():
    clarity = CLARITY_CSS.read_text(encoding="utf-8")
    refinement = REFINEMENT_CSS.read_text(encoding="utf-8")
    template = INSPECTOR.read_text(encoding="utf-8")
    assert 'width: min(1040px, 94vw)' in clarity
    assert 'grid-template-columns: 176px minmax(0, 1fr)' in clarity
    assert 'aria-orientation="vertical"' in template
    assert 'class="settings-pane"' in template
    assert 'class="settings-pane-head"' in template
    assert 'Configuration' in template
    assert 'Manage skills' in template
    assert '.settings-pane .var-group-head' in refinement
    assert '.settings-pane .var-row' in refinement
    assert 'border-top: 1px solid color-mix' in refinement
    assert 'background: transparent !important' in refinement
    assert '.settings-pane .var-row > input' in refinement
    assert 'border-bottom: 1px solid var(--hairline)' in refinement


def test_trace_analyzer_uses_floating_glass_panes_and_themed_charts():
    css = TRACE_CSS.read_text(encoding="utf-8")
    route = ROUTE_CSS.read_text(encoding="utf-8")
    assert '@import url("/static/css/route-product.css")' in css
    assert '.trace-section {' in css
    assert 'border-radius: 22px' in css
    assert 'background: color-mix(in srgb, var(--glass) 82%, transparent)' in css
    assert 'backdrop-filter: blur(30px) saturate(108%)' in css
    assert '.trace-canvas-wrap {' in css
    assert 'border-radius: 16px' in css
    assert 'filter: saturate(.44) contrast(.96) brightness(.98)' in css
    assert 'html[data-theme="light"] .trace-canvas' in css
    assert '.trace-card {' in css
    assert 'background: transparent' in css
    assert '.trace-info-popout' in css
    assert '--accent:#8ba9c6' in route
    assert '#79c2cb' not in route
    assert '#e0a040' not in route
    assert '.trace-mu{color:var(--text-soft)!important}' in route


def test_container_management_uses_current_product_surface_without_breaking_ids():
    template = CONTAINERS.read_text(encoding="utf-8")
    assert '/static/css/route-product.css' in template
    assert '/static/css/containers-crystal.css' in template
    for dom_id in (
        'container-list', 'template-list', 'create-environment-button',
        'create-modal', 'environment-name', 'shell-modal', 'shell-command',
    ):
        assert f'id="{dom_id}"' in template

    css = CONTAINER_CRYSTAL_CSS.read_text(encoding="utf-8")
    assert '.manager-card{display:grid' in css
    assert 'border-bottom:1px solid var(--hairline)' in css
    assert '.manager-monitor-grid' in css
    assert '.manager-overflow-menu' in css and 'var(--glass-strong)' in css
