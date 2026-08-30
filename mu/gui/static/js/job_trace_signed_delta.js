(() => {
    'use strict';

    /*
     * Legacy filename, new responsibility: Job Trace is an individual-task
     * breakdown only. Remove the old comparison controls before they can be
     * used, discard stale ?compare= links, and make a direct analyzer visit
     * immediately open the newest available job instead of an empty shell.
     *
     * job_trace.js starts its async job-list request before this deferred script
     * runs. Its captured compare DOM references remain harmless after removal;
     * clearing the query parameter ensures the old comparison fetch path never
     * runs while the list is loading.
     */
    // Shared signed-delta formatter for comparison metrics. Exposed on
    // window so job_trace.js (loaded before this deferred script) and any
    // later renderers share one formatting contract:
    // delta = primary - reference, signed, unit-aware.
    window.formatSigned = function formatSigned(value, unit) {
        if (value == null || Number.isNaN(Number(value))) return '\u2014';
        const num = Number(value);
        const sign = num > 0 ? '+' : num < 0 ? '\u2212' : '';
        const abs = Math.abs(num);
        // delta = primary - reference; the caller renders it next to the
        // primary/reference pair so the signed direction is unambiguous.
        const text = unit === 'seconds'
            ? `${abs.toFixed(2)}s`
            : `${abs.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
        return `${sign}${text}`;
    };
    // Documented contract: a positive delta means primary - reference > 0.
    window.JT_DELTA_SEMANTICS = 'primary - reference';

    document.getElementById('jt-compare-select')?.remove();
    document.getElementById('jt-compare-section')?.remove();

    const initialUrl = new URL(window.location.href);
    if (initialUrl.searchParams.has('compare')) {
        initialUrl.searchParams.delete('compare');
        history.replaceState(null, '', initialUrl);
    }

    const jobSelect = document.getElementById('jt-job-select');
    if (!jobSelect) return;

    let autoLoadStarted = false;

    function requestedJob() {
        return String(new URL(window.location.href).searchParams.get('job') || '').trim();
    }

    function autoLoadFirstJob() {
        if (requestedJob() || autoLoadStarted) return true;
        const first = Array.from(jobSelect.options).find(option => String(option.value || '').trim());
        if (!first) return false;
        autoLoadStarted = true;
        jobSelect.value = first.value;
        jobSelect.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
    }

    // The core analyzer populates the select asynchronously. MutationObserver
    // handles the normal path; the short timer is a defensive fallback for
    // browsers that coalesce option mutations during innerHTML replacement.
    const observer = new MutationObserver(() => {
        if (!requestedJob() && !autoLoadStarted && jobSelect.options.length > 1) {
            setTimeout(() => {
                if (autoLoadFirstJob()) observer.disconnect();
            }, 0);
        }
    });
    observer.observe(jobSelect, { childList: true });

    let checks = 0;
    const timer = window.setInterval(() => {
        checks += 1;
        if (autoLoadFirstJob() || checks >= 100) {
            clearInterval(timer);
            observer.disconnect();
        }
    }, 50);
})();
