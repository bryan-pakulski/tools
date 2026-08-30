/* Shared viewport-aware placement for anchored popup lists outside the two
 * composer menus handled by product.js. Works on Alpine and dynamically
 * generated vanilla-DOM surfaces. */
(function () {
    const MARGIN = 10;
    const GAP = 8;
    const SELECTOR = '.cmd-complete,.trace-info-popout,.manager-overflow-menu,.mode-popout';
    let frame = 0;

    function visible(node) {
        if (!node || node.hidden) return false;
        const style = getComputedStyle(node);
        return style.display !== 'none' && style.visibility !== 'hidden';
    }

    function anchorFor(layer) {
        if (layer.classList.contains('cmd-complete')) {
            return layer.closest('.composer-input-wrap')?.querySelector('textarea');
        }
        if (layer.classList.contains('trace-info-popout')) {
            return layer.parentElement?.querySelector('.trace-info-btn');
        }
        if (layer.classList.contains('manager-overflow-menu')) {
            return layer.closest('.manager-overflow')?.querySelector('summary');
        }
        if (layer.classList.contains('mode-popout')) {
            return layer.parentElement?.querySelector('.mode-pill');
        }
        return null;
    }

    function profile(layer) {
        if (layer.classList.contains('cmd-complete')) return { min: 240, max: 720, align: 'stretch' };
        if (layer.classList.contains('trace-info-popout')) return { min: 280, max: 420, align: 'start' };
        if (layer.classList.contains('manager-overflow-menu')) return { min: 178, max: 240, align: 'end' };
        return { min: 280, max: 440, align: 'end' };
    }

    function clamp(value, minimum, maximum) {
        return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
    }

    function place(layer) {
        if (!visible(layer)) return;
        const anchor = anchorFor(layer);
        if (!anchor) return;
        layer.dataset.viewportFloating = 'true';
        layer.style.visibility = 'hidden';
        layer.style.width = '';
        layer.style.maxHeight = '';

        const anchorRect = anchor.getBoundingClientRect();
        const measured = layer.getBoundingClientRect();
        const settings = profile(layer);
        const viewport = window.visualViewport;
        const viewportLeft = viewport ? viewport.offsetLeft : 0;
        const viewportTop = viewport ? viewport.offsetTop : 0;
        const viewportWidth = Math.max(1, viewport ? viewport.width : document.documentElement.clientWidth);
        const viewportHeight = Math.max(1, viewport ? viewport.height : document.documentElement.clientHeight);
        const viewportRight = viewportLeft + viewportWidth;
        const viewportBottom = viewportTop + viewportHeight;
        const widthLimit = Math.max(120, viewportWidth - MARGIN * 2);
        const desiredWidth = settings.align === 'stretch'
            ? Math.max(anchorRect.width, layer.scrollWidth)
            : Math.max(measured.width, layer.scrollWidth, settings.min);
        const width = clamp(desiredWidth, Math.min(settings.min, widthLimit), Math.min(settings.max, widthLimit));
        const naturalHeight = Math.max(measured.height, layer.scrollHeight);
        const above = Math.max(0, anchorRect.top - viewportTop - MARGIN - GAP);
        const below = Math.max(0, viewportBottom - anchorRect.bottom - MARGIN - GAP);
        const placeAbove = above > below;
        const available = Math.max(40, placeAbove ? above : below);
        const height = Math.min(naturalHeight, available);

        let left = settings.align === 'end' ? anchorRect.right - width : anchorRect.left;
        left = clamp(left, viewportLeft + MARGIN, viewportRight - width - MARGIN);
        let top = placeAbove ? anchorRect.top - GAP - height : anchorRect.bottom + GAP;
        top = clamp(top, viewportTop + MARGIN, viewportBottom - height - MARGIN);

        layer.style.left = `${Math.round(left)}px`;
        layer.style.top = `${Math.round(top)}px`;
        layer.style.width = `${Math.round(width)}px`;
        layer.style.maxHeight = `${Math.round(available)}px`;
        layer.style.visibility = '';
        layer.dataset.placement = placeAbove ? 'top' : 'bottom';
    }

    function reposition() {
        frame = 0;
        document.querySelectorAll(SELECTOR).forEach(place);
    }

    function schedule() {
        if (frame) return;
        frame = requestAnimationFrame(reposition);
    }

    document.addEventListener('DOMContentLoaded', () => {
        schedule();
        new MutationObserver(schedule).observe(document.body, {
            subtree: true,
            childList: true,
        });
    });
    document.addEventListener('click', schedule, true);
    document.addEventListener('input', schedule, true);
    document.addEventListener('scroll', schedule, { passive: true, capture: true });
    window.addEventListener('resize', schedule, { passive: true });
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', schedule, { passive: true });
        window.visualViewport.addEventListener('scroll', schedule, { passive: true });
    }
})();
