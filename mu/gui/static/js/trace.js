/* Trace Analyzer — Alpine component + custom <canvas> renderers.
 *
 * No charting library: the server returns structured series/snapshot
 * (`/api/traces` + `/api/traces/<id>`), and this file draws every chart with
 * vanilla 2D canvas. Mirrors the codebase's bindCanvas idiom from the memory
 * panel. Clicking an iter in any chart/table scrolls the conversation view to
 * that iteration and highlights it.
 *
 * Each panel has a ⓘ info button (see trace.html) whose popout explains what the
 * panel shows, how to read it, and which harness decision it informs.
 */

// Shared palette — within-panel colors are distinct; cross-panel reuse is
// intentional only for semantically-identical signals (e.g. red = error/bad
// everywhere). Kept in one place so the charts read consistently.
const PALETTE = {
    actual: "#e0a040",        // amber — real prompt tokens
    est: "#b0aca0",           // accent-ish — harness estimate (dashed)
    drift: "#e07aa0",         // pink — tokenizer drift (the headline line)
    tokIn: "#e0a040",
    tokOut: "#7ec96b",        // green
    tokCached: "#5b9bd0",     // blue
    tokReasoning: "#c08ae0",  // purple
    attribution: {
        system: "#8f7ac8", user: "#5b9bd0", assistant: "#7ec96b",
        tool_calls: "#d0b05b", tool_results: "#e08b40",
        files_images: "#54b6b0", other: "#8b8b8b", tool_schemas: "#c06b8a",
    },
    taskMem: "#7ec96b",
    scratchpad: "#c98a4b",    // amber-brown (distinct from taskMem)
    subBar: "#5b9bd0",
    read: "#5b9bd0",
    redundant: "#d9534f",
    ok: "#5b9bd0",
    err: "#d9534f",
    cache: "#7ec96b",
    stuck: "#d9534f",
    stall: "#e0a040",
    limit: "rgba(220,100,100,0.55)",
    compaction: "rgba(120,180,220,0.7)",
    warnBand: "rgba(220,160,60,0.12)",
    heat: [224, 160, 64],      // amber ramp for the layer heat strip
    driftPos: [91, 155, 208],  // teal — positive drift (est > actual)
    driftNeg: [217, 83, 79],   // red — negative drift (est < actual)
    status: {                  // memory by_status colors
        active: "#7ec96b", stale: "#e0a040", done: "#5b9bd0",
        pending: "#c08ae0", pruned: "#8b8b8b",
    },
    statusDefault: "#8b8b8b",
};

const READ_TOOLS = new Set([
    "read_file", "get_chunk", "list_dir", "search_for_string",
    "search_references", "retrieve_relevant_context", "get_workspace_details",
]);

function traceApp() {
    return {
        runs: [],
        runId: "",
        summary: null,
        series: { context: [], layers_stacked: {}, drift: [], tokens: [],
                  latency: [],
                  tool_histogram: [], compaction_timeline: [], nudge_timeline: [],
                  nudge_efficacy: [], redundant_reads: [], subagent_timeline: [],
                  memory_series: [], context_attribution: [], top_context_spikes: [],
                  efficiency: [] },
        iters: [],
        tools: [],
        snapshot: { grid: [], drift_strip: [], xs: [], compaction_cols: [],
                    context_actual: [], context_est: [],
                    meta: { layers: [], labels: {}, layer_vmax: {}, n: 0 } },
        loading: false,
        // Round-48 F16: opt-in "all runs" — bounded by default (newest 5).
        allRuns: false,
        selectedIter: null,
        selectedTool: null,
        driftLog: false,        // signed-log y scale for the drift chart
        infoOpen: null,         // open info popout key
        hoverRef: null, hoverIdx: null,
        session: "",            // session scope (?session=) — traces are per-session
        runBounds: [],           // per-run global [start,end] for the combined view
        theme: "dark",          // current data-theme (for the toggle button glyph)
        expandedTools: {},      // toolKey -> true: that tool's result preview is expanded
        expandedCards: {},      // cardKey -> true: all tool previews in that turn expanded
        VIEW_CAP: 100,          // max iters drawn across the plot width (X pan window)
        xStart: 0,              // first iter index visible in the X pan window
        xCount: 100,            // iters currently visible (≤ VIEW_CAP, or totalIters if fewer)
        readHover: null,        // {x, y, label} read-map hover tooltip state

        async init() {
            // Session scope: the analyzer is opened from the chat with
            // ?session=<name> and shows a COMBINED view of every run in that
            // session (all of it, chronologically, run boundaries marked),
            // ignoring every other session. No param → fall back to a global
            // per-run picker.
            const qs = new URLSearchParams(window.location.search);
            this.session = (qs.get("session") || "").trim();
            this.theme = document.documentElement.getAttribute("data-theme") || "dark";
            // Re-render canvases on resize (debounced) — _setupCanvas reads
            // clientWidth at draw time, so a re-render picks up the new CSS size.
            this._resizeHandler = () => {
                clearTimeout(this._resizeTimer);
                this._resizeTimer = setTimeout(() => this.renderAll(), 150);
            };
            window.addEventListener("resize", this._resizeHandler);
            if (this.session) {
                await this.loadSession();
            } else {
                await this.loadRuns();
            }
        },

        async loadSession() {
            // Combined multi-run view: every run in the session merged into
            // one series/summary/snapshot, with run_bounds for boundary markers.
            // Round-47 F15 + Round-48 F16: default fetch is bounded to the
            // NEWEST 5 runs; this.allRuns (user opt-in) widens to every run
            // in the session — the analyzer must not lose access to older
            // runs just because the default payload is bounded.
            this.loading = true;
            this.selectedIter = null;
            this.selectedTool = null;
            try {
                const limit = this.allRuns ? 0 : 5;
                const r = await fetch(
                    "/api/traces/session/" + encodeURIComponent(this.session) + "?cols=256&limit=" + limit
                );
                if (!r.ok) { this.summary = null; return; }
                const d = await r.json();
                this.summary = d.summary;
                this.series = d.series;
                this.iters = d.iters || [];
                this.tools = d.tools || [];
                this.snapshot = d.snapshot || this.snapshot;
                this.runBounds = d.run_bounds || [];
                this.runId = d.run_id || "";
                this._resetWindow();
                this.$nextTick(() => {
                    this.attachHovers();
                    this.renderAll();
                });
            } finally {
                this.loading = false;
            }
        },

        async loadRuns() {
            // No-session fallback: global per-run picker. Auto-loads newest.
            try {
                const url = "/api/traces" + (this.session
                    ? "?session=" + encodeURIComponent(this.session) : "");
                const r = await fetch(url);
                this.runs = await r.json();
            } catch (e) {
                this.runs = [];
            }
            // Auto-load the newest run so the analyzer opens already
            // visualizing data instead of waiting on a pick.
            if (this.runs.length && !this.runId) {
                this.runId = this.runs[0].run_id;
                this.loadRun();
            }
        },

        async clearRun() {
            if (!this.runId || this.session) return;
            if (!window.confirm("Clear this trace run? This deletes the JSONL file.")) return;
            try {
                const r = await fetch("/api/traces/" + encodeURIComponent(this.runId), {
                    method: "DELETE",
                });
                if (!r.ok) { window.alert("Clear failed: " + r.status); return; }
                this.summary = null;
                this.series = {};
                this.iters = [];
                this.tools = [];
                this.snapshot = {};
                this.runId = "";
                await this.loadRuns();
            } catch (e) {
                window.alert("Clear failed: " + e);
            }
        },

        async loadRun() {
            if (!this.runId) { this.summary = null; return; }
            this.loading = true;
            this.selectedIter = null;
            this.selectedTool = null;
            this.runBounds = [];   // single-run view has no run boundaries
            try {
                const r = await fetch("/api/traces/" + encodeURIComponent(this.runId) + "?cols=256");
                const d = await r.json();
                this.summary = d.summary;
                this.series = d.series;
                this.iters = d.iters || [];
                this.tools = d.tools || [];
                this.snapshot = d.snapshot || this.snapshot;
                this._resetWindow();
                this.$nextTick(() => {
                    this.attachHovers();
                    this.renderAll();
                });
            } finally {
                this.loading = false;
            }
        },

        toggleInfo(key) {
            this.infoOpen = (this.infoOpen === key) ? null : key;
        },

        // Mirror the chat's $store.theme.toggle() (app.js): flip data-theme and
        // persist to localStorage. app.css themes via html[data-theme=...], so
        // setting the attribute is enough — no stylesheet swapping needed here
        // (that's only for the chat's hljs code blocks).
        toggleTheme() {
            this.theme = (this.theme === "dark") ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", this.theme);
            try { localStorage.setItem("mucli-theme", this.theme); } catch (e) {}
            this.renderAll();   // re-render canvases against the new theme vars
        },

        // ---- helpers ---------------------------------------------------
        fmtNum(v) { v = Number(v || 0); return v >= 1000 ? v.toLocaleString() : String(v); },
        fmtMs(v) { v = Number(v || 0); if (v >= 60000) return (v / 60000).toFixed(1) + "m"; if (v >= 1000) return (v / 1000).toFixed(1) + "s"; return Math.round(v) + "ms"; },
        fmtCost(v) { v = Number(v || 0); return v ? "$" + (v < 0.01 ? v.toFixed(6) : v.toFixed(4)) : "—"; },
        fmtStatus(r) { return r.iters > 0 ? "done" : "—"; },
        attributionLabel(key) {
            return ({system: "system", user: "user", assistant: "assistant",
                tool_calls: "tool calls", tool_results: "tool results",
                files_images: "files/images", other: "other",
                tool_schemas: "tool schemas"})[key] || key || "other";
        },
        toolHistSorted() { return [...(this.series.tool_histogram || [])].sort((a, b) => b.count - a.count); },
        maxToolCount() { return Math.max(1, ...this.toolHistSorted().map(h => h.count)); },

        // ---- X pan window (cap ~VIEW_CAP iters visible; pan the rest) ------
        // All per-iter series + the heat grid are indexed identically (one
        // point per iter, aligned with `iters`), so a single [xStart, xStart+
        // xCount) window pans every chart together. When total iters ≤
        // VIEW_CAP the window covers everything and the scrollbar is hidden.
        totalIters() { return (this.iters || []).length; },
        _winStart(total) {
            const cnt = Math.min(this.xCount, total);
            return Math.max(0, Math.min(this.xStart, Math.max(0, total - cnt)));
        },
        _winCount(total) { return Math.min(this.xCount, total); },
        // The shared [xStart, xStart+xCount) window is positional over `iters`.
        // Full-length per-iter series (context, tokens, …) have one entry per
        // iter, so a positional slice lands exactly on the visible window.
        // But two series are NOT one-per-iter:
        //   • efficiency — only iters that made tool calls (sparse), and
        //   • the heat strip's snapshot.xs — downsampled to `cols` columns
        //     server-side, so its length ≠ n on long runs.
        // Slicing those positionally clamps to their own shorter length, so
        // they stop panning (freeze) and desync from the other charts. Window
        // them by ITER NUMBER instead: keep the entries whose `iter` falls in
        // the current window's iter range, so they scroll in lockstep.
        _iterBounds() {
            const all = this.iters || [];
            const n = all.length;
            if (!n) return [-Infinity, Infinity];
            const s = this._winStart(n), c = this._winCount(n);
            const lo = all[s] != null ? all[s].iter : -Infinity;
            const hi = all[Math.min(n - 1, s + c - 1)] != null
                ? all[Math.min(n - 1, s + c - 1)].iter : Infinity;
            return [lo, hi];
        },
        _view(arr) {
            if (!arr || !arr.length) return [];
            const n = (this.iters || []).length;
            if (arr.length === n) {            // full per-iter series: positional == iter window
                const s = this._winStart(n);
                return arr.slice(s, s + this._winCount(n));
            }
            const [lo, hi] = this._iterBounds();   // sparse/downsampled: window by iter
            return arr.filter(d => d.iter != null && d.iter >= lo && d.iter <= hi);
        },
        _resetWindow() {
            const total = (this.iters || []).length;
            this.xCount = Math.min(this.VIEW_CAP, total);
            // Start at the latest iterations (RHS): the most recent work is the
            // default point of interest. When the whole run fits in the window
            // (total ≤ xCount) xStart is 0 and covers everything.
            this.xStart = Math.max(0, total - this.xCount);
            this._updatePanCursors();
        },
        xScrollMax() { return Math.max(0, this.totalIters() - this.xCount); },
        // Pannable when the run exceeds the visible window. Dragging any chart
        // shifts the shared [xStart, xStart+xCount) window (see _attachPan); all
        // charts redraw together because they all slice through _view.
        _pannable() { return this.totalIters() > this.xCount; },
        _updatePanCursors() {
            const pannable = this._pannable();
            const cur = pannable ? "grab" : "default";
            document.querySelectorAll(".trace-canvas").forEach(cv => { cv.style.cursor = cur; });
        },
        _attachPan(cv) {
            if (cv._panAttached) return;
            cv._panAttached = true;
            cv.addEventListener("mousedown", (e) => this._onPanDown(e, cv));
        },
        _onPanDown(e, cv) {
            if (!this._pannable()) return;        // nothing to pan — let click through
            e.preventDefault();
            this._dragCV = cv;
            this._dragStartClientX = e.clientX;
            this._dragStartXStart = this.xStart;
            this._dragMoved = false;
            this.hoverRef = null; this.hoverIdx = null; this.readHover = null;  // drop hover chip while panning
            cv.style.cursor = "grabbing";
            // Track on the window so the drag continues even when the cursor
            // leaves the canvas; remove on mouseup to avoid leaks.
            const move = (ev) => this._onPanMove(ev);
            const up = (ev) => {
                window.removeEventListener("mousemove", move);
                window.removeEventListener("mouseup", up);
                this._onPanUp(ev);
            };
            window.addEventListener("mousemove", move);
            window.addEventListener("mouseup", up);
        },
        _onPanMove(ev) {
            if (!this._dragCV) return;
            const cv = this._dragCV;
            const dx = ev.clientX - this._dragStartClientX;
            if (Math.abs(dx) > 4) this._dragMoved = true;   // distinguish drag from click
            // pixels per visible iter — uniform across charts (the window spans
            // xCount iters over the canvas width).
            const ppi = Math.max(1, (cv.clientWidth || 1) / Math.max(1, this.xCount));
            const iters = Math.round(dx / ppi);             // drag right → show earlier iters
            const maxStart = this.xScrollMax();
            const ns = Math.max(0, Math.min(this._dragStartXStart - iters, maxStart));
            if (ns !== this.xStart) { this.xStart = ns; this.renderAll(); }
        },
        _onPanUp() {
            this._dragCV = null;
            this._updatePanCursors();
        },
        // Label for the scrollbar: "iters A–B of N" in actual iter numbers.
        xScrollLabel() {
            const all = this.iters || [];
            const n = all.length;
            if (!n) return "";
            const s = this._winStart(n), c = this._winCount(n);
            const a = all[s] ? all[s].iter : "?";
            const b = all[Math.min(n - 1, s + c - 1)] ? all[Math.min(n - 1, s + c - 1)].iter : "?";
            return "iters " + a + "–" + b + " / " + n;
        },
        // Scrollbar thumb: width = window's share of the run, left = its
        // position within the scrollable range. Bound via :style so Alpine
        // re-renders it reactively on every xStart change.
        thumbStyle() {
            const total = this.totalIters();
            if (!total) return { display: "none" };
            const wPct = (this.xCount / total) * 100;
            const max = this.xScrollMax();
            const lPct = max > 0 ? (this.xStart / max) * (100 - wPct) : 0;
            return { width: wPct + "%", left: lPct + "%" };
        },
        // Drag the thumb to pan (same shared xStart as chart drag).
        onScrollThumbDown(e) {
            if (!this._pannable()) return;
            e.preventDefault(); e.stopPropagation();
            const track = this.$refs.xscrollTrack;
            const rect = track.getBoundingClientRect();
            const total = this.totalIters(), max = this.xScrollMax();
            const thumbW = (this.xCount / total) * rect.width;
            const usable = Math.max(1, rect.width - thumbW);   // px the thumb can travel
            const start = this.xStart, startClientX = e.clientX;
            const move = (ev) => {
                const dx = ev.clientX - startClientX;
                const ns = Math.max(0, Math.min(start + Math.round((dx / usable) * max), max));
                if (ns !== this.xStart) { this.xStart = ns; this.renderAll(); }
            };
            const up = () => {
                window.removeEventListener("mousemove", move);
                window.removeEventListener("mouseup", up);
            };
            window.addEventListener("mousemove", move);
            window.addEventListener("mouseup", up);
        },
        // Click the track (not the thumb) to jump — centers the window on
        // the click, like a native scrollbar.
        onScrollTrackClick(e) {
            if (!this._pannable()) return;
            if (e.target.classList && e.target.classList.contains("trace-xscroll-thumb")) return;
            const track = this.$refs.xscrollTrack;
            const rect = track.getBoundingClientRect();
            const frac = (e.clientX - rect.left) / Math.max(1, rect.width);  // 0..1
            const center = Math.round(frac * this.totalIters());
            const max = this.xScrollMax();
            const ns = Math.max(0, Math.min(center - Math.floor(this.xCount / 2), max));
            if (ns !== this.xStart) { this.xStart = ns; this.renderAll(); }
        },
        // Horizontal wheel / trackpad scroll over any chart pans the shared
        // window too — trackpad horizontal swipe (deltaX) or shift+wheel.
        _attachWheel(cv) {
            if (cv._wheelAttached) return;
            cv._wheelAttached = true;
            cv.addEventListener("wheel", (e) => {
                if (!this._pannable()) return;
                const dx = e.deltaX || (e.shiftKey ? e.deltaY : 0);
                if (!dx) return;
                e.preventDefault();
                const step = Math.max(1, Math.round(this.xCount * 0.1));
                const max = this.xScrollMax();
                const ns = Math.max(0, Math.min(this.xStart + (dx > 0 ? step : -step), max));
                if (ns !== this.xStart) { this.xStart = ns; this.renderAll(); }
            }, { passive: false });
        },
        errorCodes(h) { return Object.entries(h.error_codes || {}); },
        selectedToolSeries() {
            const h = (this.series.tool_histogram || []).find(h => h.name === this.selectedTool);
            return (h && h.latency_series) || [];
        },
        toolsForIter(it) { return (this.tools || []).filter(t => t.iter === it); },
        // Active subagent children at the selected iter, else all seen this run.
        subagentChildren() {
            const tl = this.series.subagent_timeline || [];
            if (this.selectedIter !== null) {
                const e = tl.find(s => s.iter === this.selectedIter);
                if (e && e.children && e.children.length) return e.children;
            }
            const seen = {}; const out = [];
            for (const s of tl) for (const c of (s.children || [])) {
                const k = c.task_id || (c.task_id === 0 ? 0 : (c.status + (out.length)));
                if (!seen[k]) { seen[k] = 1; out.push(c); }
            }
            return out;
        },

        selectIter(it) {
            this.selectedIter = it;
            // Pan the X window to include the selected iter so chart clicks
            // on off-screen iters bring them into view.
            const idx = (this.iters || []).findIndex(x => x.iter === it);
            if (idx >= 0 && (idx < this._winStart(this.iters.length) || idx >= this._winStart(this.iters.length) + this.xCount)) {
                const maxStart = this.xScrollMax();
                this.xStart = Math.max(0, Math.min(idx - Math.floor(this.xCount / 2), maxStart));
            }
            this.$nextTick(() => {
                // The conversation view groups iters into turn cards, so the
                // exact `iter-<it>` id may not exist (the card is keyed by its
                // first iter). Fall back to the card whose [min,max] range
                // contains the iter so chart clicks still scroll to the right turn.
                let el = document.getElementById("iter-" + it);
                if (!el) {
                    const cards = document.querySelectorAll("[data-iter-min]");
                    for (const c of cards) {
                        const lo = +c.getAttribute("data-iter-min");
                        const hi = +c.getAttribute("data-iter-max");
                        if (it >= lo && it <= hi) { el = c; break; }
                    }
                }
                if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
                this.renderAll();
            });
        },

        // Group iters into assistant "turn cards" with run-separator banners
        // spliced in at each run boundary (combined session view). A card
        // starts at each iter where the model produced text (the turn's
        // "voice") and absorbs the following tool-only iters until the next
        // text iter — so prose sits at the top of the card and the tool calls
        // it then made hang below it. Leading tool-only iters (the model
        // acting before first speaking) form an "acting only" card with no
        // voice. Returns a list of either {sep:true, run_id, idx, model,
        // status, iters} or {card:true, ...turnCardFields}.
        turnCards() {
            const out = [];
            let prevRun = null;
            const bounds = this.runBounds || [];
            const idxOf = (rid) => {
                const i = bounds.findIndex(b => b.run_id === rid);
                return i >= 0 ? i + 1 : null;
            };
            let cur = null;
            const flush = () => { if (cur) { out.push({ card: true, ...cur }); cur = null; } };
            for (const it of (this.iters || [])) {
                const rid = it.run_id || "";
                if (rid !== prevRun) {
                    flush();
                    const b = bounds.find(x => x.run_id === rid);
                    out.push({
                        sep: true, run_id: rid, idx: idxOf(rid),
                        model: b ? b.model : "", status: b ? b.status : "",
                        iters: b ? b.iters : null,
                    });
                    prevRun = rid;
                    cur = null;
                }
                const tools = this.toolsForIter(it.iter).map(t => ({ ...t, _iter: it.iter }));
                if (it.has_text) {
                    // A speaking iter closes the previous card and opens a new
                    // one with itself as the voice (not yet pushed, so later
                    // tool-only iters absorb into it).
                    flush();
                    cur = { iters: [it], voice: it, toolList: tools.slice() };
                } else {
                    if (!cur) cur = { iters: [it], voice: null, toolList: [] };
                    else cur.iters.push(it);
                    cur.toolList.push(...tools);
                }
            }
            flush();
            // Badge the last turn card as the final output.
            for (let i = out.length - 1; i >= 0; i--) {
                if (out[i].card) { out[i].final = true; break; }
            }
            return out;
        },

        // ---- turn-card helpers (called from the template) ----------------
        cardKey(c) { return "c" + c.iters[0].iter; },
        cardLabel(c) {
            const a = c.iters[0].iter, b = c.iters[c.iters.length - 1].iter;
            return a === b ? "iter " + a : "iters " + a + "–" + b;
        },
        cardWall(c) { return c.iters.reduce((s, it) => s + (it.wall_ms || 0), 0); },
        cardTokens(c) {
            return c.iters.reduce((s, it) => s + ((it.tokens || {}).in || 0), 0);
        },
        cardMaxDrift(c) {
            let m = 0;
            for (const it of c.iters) { const d = Math.abs(it.context ? it.context.drift_pct : 0); if (d > m) m = d; }
            return Math.round(m);
        },
        cardHasCompaction(c) { return c.iters.some(it => it.compaction); },
        toolKey(c, ti) { return c.iters[0].iter + ":" + ti; },
        toolExpanded(c, ti) {
            return !!this.expandedTools[this.toolKey(c, ti)] || !!this.expandedCards[this.cardKey(c)];
        },
        toggleTool(c, ti) {
            const k = this.toolKey(c, ti);
            this.expandedTools[k] = !this.expandedTools[k];
            // reactive: reassign so Alpine notices the object-key change
            this.expandedTools = { ...this.expandedTools };
        },
        toggleCardTools(c) {
            const k = this.cardKey(c);
            this.expandedCards[k] = !this.expandedCards[k];
            this.expandedCards = { ...this.expandedCards };
        },

        // Draw run-boundary dividers on a per-iter chart. `data` is the series
        // (each point carries `iter`); we map each run's start_iter to its x.
        _runMarks(ctx, w, h, pad, data) {
            if (!this.runBounds || this.runBounds.length < 2 || !data.length) return;
            const t = this._theme();
            const n = data.length;
            ctx.font = "9px " + (getComputedStyle(document.body).fontFamily);
            for (let bi = 1; bi < this.runBounds.length; bi++) {
                const startIter = this.runBounds[bi].start_iter;
                const idx = data.findIndex(d => d.iter === startIter);
                if (idx < 0) continue;
                const x = this._iterX(idx, n, pad, w);
                ctx.strokeStyle = "rgba(176,172,160,0.22)";
                ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
                ctx.setLineDash([]);
                // tiny run number tick at the top so boundaries are countable
                ctx.fillStyle = t.dim; ctx.textAlign = "center"; ctx.textBaseline = "top";
                const label = "run " + (bi + 1);
                const lx = Math.min(Math.max(pad.l + 14, x), w - pad.r - 14);
                ctx.fillText(label, lx, pad.t + 1);
            }
        },
        selectTool(name) {
            this.selectedTool = (this.selectedTool === name) ? null : name;
            this.$nextTick(() => this.renderTool());
        },

        // ---- canvas theme ---------------------------------------------
        _theme() {
            const css = getComputedStyle(document.documentElement);
            const pick = (v, fb) => (css.getPropertyValue(v).trim() || fb);
            return {
                text: pick("--text", "#d4d0c8"),
                dim: pick("--text-dim", "#6b6860"),
                border: pick("--border-strong", "rgba(255,255,255,0.10)"),
                accent: pick("--accent", "#b0aca0"),
                bg: pick("--bg", "#1a1814"),
            };
        },

        _setupCanvas(ref) {
            const c = this.$refs[ref];
            if (!c) return null;
            const dpr = window.devicePixelRatio || 1;
            // Logical (CSS-pixel) height is FIXED per canvas from its original
            // HTML height attribute — cache it on the first call. We set
            // c.height = h*dpr below, so re-reading c.height on a later render
            // would read back the already-scaled value and compound by dpr
            // *every* render. Since mousemove triggers a full renderAll, the
            // canvas would balloon exponentially on hover (the "zooms in a
            // huge amount" bug). Width comes from clientWidth (stable CSS
            // pixels), so it never compounds.
            if (!c._logicalH) {
                c._logicalH = parseInt(c.getAttribute("height")) || c.height || 200;
            }
            const w = c.clientWidth || parseInt(c.getAttribute("width")) || 1200;
            const h = c._logicalH;
            c.width = w * dpr;
            c.height = h * dpr;
            const ctx = c.getContext("2d");
            ctx.scale(dpr, dpr);
            return { ctx, w, h };
        },

        _drawAxes(ctx, w, h, pad, ys, opts = {}) {
            const t = this._theme();
            ctx.strokeStyle = t.border; ctx.fillStyle = t.dim;
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.lineWidth = 1;
            const unit = opts.unit || "";
            // y gridlines + labels
            ctx.textAlign = "right"; ctx.textBaseline = "middle";
            for (let i = 0; i <= 4; i++) {
                const yv = ys.max - (ys.max - ys.min) * (i / 4);
                const yy = pad.t + (h - pad.t - pad.b) * (i / 4);
                ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
                ctx.fillText(this.fmtNum(Math.round(yv)), pad.l - 4, yy);
            }
            // y-axis unit caption (top-left of plot)
            if (unit) {
                ctx.fillStyle = t.dim; ctx.textAlign = "left"; ctx.textBaseline = "top";
                ctx.fillText(unit, pad.l + 2, 2);
                ctx.fillStyle = t.dim;
            }
            // x labels — denser when there are many iters
            const xs = opts.xs || [];
            ctx.textAlign = "center"; ctx.textBaseline = "top";
            if (xs.length) {
                const plotW = w - pad.l - pad.r;
                const want = xs.length > 60 ? 7 : (xs.length > 20 ? 5 : 3);
                const labels = [];
                for (let k = 0; k < want; k++) {
                    const idx = Math.round((k / Math.max(1, want - 1)) * (xs.length - 1));
                    if (!labels.includes(idx)) labels.push(idx);
                }
                for (const idx of labels) {
                    const xx = pad.l + (xs.length === 1 ? plotW / 2 : (idx / (xs.length - 1)) * plotW);
                    ctx.fillText("iter " + xs[idx], xx, h - pad.b + 4);
                }
            }
        },

        _iterX(i, n, pad, w) {
            if (n <= 1) return pad.l + (w - pad.l - pad.r) / 2;
            return pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
        },

        _empty(ctx, w, h, msg) {
            const t = this._theme();
            ctx.fillStyle = t.dim; ctx.font = "11px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            ctx.fillText(msg, 8, 12);
        },

        // ---- hover (line charts) --------------------------------------
        attachHovers() {
            if (this._hoverAttached) return;
            this._hoverAttached = true;
            const cfgs = [
                { ref: "ctxCanvas", pad: { l: 56, r: 12 }, key: "context",
                  label: it => `${this.fmtNum(it.actual)} actual · ${this.fmtNum(it.total_est)} est tok${it.drift_ratio ? " · drift ×" + it.drift_ratio : ""}` },
                { ref: "attrCanvas", pad: { l: 56, r: 12 }, key: "context_attribution",
                  label: it => `${this.fmtNum(it.total)} request · ${(it.delta >= 0 ? "+" : "") + this.fmtNum(it.delta)} · ${this.attributionLabel(it.growth_source)}` },
                { ref: "driftCanvas", pad: { l: 56, r: 12 }, key: "drift",
                  label: it => `${it.drift_pct}% drift` },
                { ref: "tokCanvas", pad: { l: 56, r: 12 }, key: "tokens",
                  label: it => `in ${this.fmtNum(it.in)} · out ${this.fmtNum(it.out)} tok` },
                { ref: "latCanvas", pad: { l: 56, r: 12 }, key: "latency",
                  label: it => `${this.fmtMs(it.wall_ms)} wall` },
                { ref: "effCanvas", pad: { l: 56, r: 12 }, key: "efficiency",
                  label: it => `${this.fmtNum(it.raw_tokens)} raw · ${this.fmtNum(it.injected_tokens)} inj · ${this.fmtNum((it.raw_tokens || 0) - (it.injected_tokens || 0))} saved tok` },
            ];
            for (const c of cfgs) {
                const cv = this.$refs[c.ref]; if (!cv) continue;
                cv._cfg = c;
                cv.addEventListener("mousemove", (e) => this._onHoverMove(e, cv));
                cv.addEventListener("mouseleave", () => {
                    if (this.hoverRef === c.ref) { this.hoverRef = null; this.hoverIdx = null; this.renderAll(); }
                });
                cv.addEventListener("click", (e) => {
                    // A drag on the chart pans the shared window (see _onPanDown);
                    // swallow the trailing click so a pan doesn't also jump the
                    // conversation view to an arbitrary iter.
                    if (this._dragMoved) { this._dragMoved = false; return; }
                    const idx = this._hoverIdxAt(e, cv); if (idx == null) return;
                    const s = this._view(this.series[c.key] || []); if (s[idx]) this.selectIter(s[idx].iter);
                });
            }
            // Read-state map uses a 2D (row + col) hover instead of the x-only
            // line-chart hover: the popup shows the path under the cursor and,
            // when the cursor is on a cell, the iter + read/redundant state.
            const rc = this.$refs.readCanvas;
            if (rc && !rc._readHoverAttached) {
                rc._readHoverAttached = true;
                rc.addEventListener("mousemove", (e) => this._onReadHover(e, rc));
                rc.addEventListener("mouseleave", () => {
                    if (this.hoverRef === "readCanvas") { this.hoverRef = null; this.readHover = null; this.renderAll(); }
                });
            }
            // Drag-to-pan: every per-iter chart shares one [xStart, xStart+
            // xCount) window, so a single drag handler per canvas pans them all
            // together (each slices through _view, all redrawn by renderAll).
            document.querySelectorAll(".trace-canvas").forEach(cv => {
                this._attachPan(cv);
                this._attachWheel(cv);
            });
        },
        _hoverIdxAt(e, cv) {
            const c = cv._cfg; const rect = cv.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const s = this._view(this.series[c.key] || []); const n = s.length; if (!n) return null;
            const w = cv.clientWidth || cv.width;
            const plotL = c.pad.l, plotR = w - c.pad.r;
            if (x < plotL || x > plotR) return null;
            const frac = (x - plotL) / Math.max(1, plotR - plotL);
            return Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
        },
        _onHoverMove(e, cv) {
            if (this._dragCV) return;                  // dragging pans; don't chase hover
            const idx = this._hoverIdxAt(e, cv); if (idx == null) return;
            this.hoverRef = cv._cfg.ref; this.hoverIdx = idx; this.renderAll();
        },
        _drawHover(ctx, w, h, pad, n, idx, label) {
            if (this.hoverRef !== this._curRef) return;
            const t = this._theme();
            const x = this._iterX(idx, n, pad, w);
            ctx.strokeStyle = t.accent; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
            ctx.setLineDash([]);
            // chip
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            const tw = ctx.measureText(label).width + 10;
            const cx = Math.min(Math.max(pad.l + 2, x + 6), w - pad.r - tw - 2);
            ctx.fillStyle = t.bg; ctx.strokeStyle = t.border;
            ctx.fillRect(cx, pad.t + 2, tw, 16);
            ctx.strokeRect(cx, pad.t + 2, tw, 16);
            ctx.fillStyle = t.text; ctx.textAlign = "left"; ctx.textBaseline = "middle";
            ctx.fillText(label, cx + 5, pad.t + 10);
        },

        // ---- renderers -------------------------------------------------
        renderAll() {
            this.renderContext();
            this.renderAttribution();
            this.renderHeatStrip();
            this.renderDrift();
            this.renderReads();
            this.renderSub();
            this.renderMem();
            this.renderTokens();
            this.renderLatency();
            this.renderTool();
            this.renderEfficiency();
        },

        renderAttribution() {
            const s = this._setupCanvas("attrCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            this._curRef = "attrCanvas";
            const dataAll = this.series.context_attribution || [];
            const data = this._view(dataAll);
            if (!data.length) { this._empty(ctx, w, h, "no request attribution data"); return; }
            const keys = ["system", "user", "assistant", "tool_calls", "tool_results", "files_images", "other", "tool_schemas"];
            const pad = { l: 56, r: 12, t: 38, b: 20 };
            const n = data.length;
            const max = Math.max(1, ...dataAll.map(d => d.total || keys.reduce((sum, key) => sum + (d[key] || 0), 0))) * 1.08;
            const plotH = h - pad.t - pad.b;
            this._drawAxes(ctx, w, h, pad, { min: 0, max }, { xs: data.map(d => d.iter), unit: "tokens" });
            const slotW = (w - pad.l - pad.r) / Math.max(1, n);
            const barW = Math.max(1, Math.min(18, slotW * 0.8));
            data.forEach((point, i) => {
                const x = this._iterX(i, n, pad, w) - barW / 2;
                let yBottom = h - pad.b;
                for (const key of keys) {
                    const value = point[key] || 0;
                    const height = value / max * plotH;
                    if (height > 0) {
                        ctx.fillStyle = PALETTE.attribution[key];
                        ctx.fillRect(x, yBottom - height, barW, height);
                        yBottom -= height;
                    }
                }
                if ((point.delta || 0) > 0) {
                    const top = h - pad.b - ((point.total || 0) / max * plotH);
                    ctx.fillStyle = point.delta > max * 0.1 ? PALETTE.err : "rgba(220,160,60,0.85)";
                    ctx.beginPath(); ctx.moveTo(x + barW / 2, Math.max(pad.t + 2, top - 7));
                    ctx.lineTo(x + barW / 2 - 3, Math.max(pad.t + 8, top - 1));
                    ctx.lineTo(x + barW / 2 + 3, Math.max(pad.t + 8, top - 1));
                    ctx.closePath(); ctx.fill();
                }
            });
            this._runMarks(ctx, w, h, pad, data);
            ctx.font = "9px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            let lx = pad.l + 2, ly = 3;
            for (const key of keys) {
                const label = this.attributionLabel(key);
                const width = ctx.measureText(label).width + 17;
                if (lx + width > w - pad.r) { lx = pad.l + 2; ly += 13; }
                ctx.fillStyle = PALETTE.attribution[key]; ctx.fillRect(lx, ly + 2, 7, 7);
                ctx.fillStyle = t.dim; ctx.fillText(label, lx + 10, ly);
                lx += width;
            }
            if (this.hoverRef === "attrCanvas" && this.hoverIdx != null) {
                const point = data[this.hoverIdx];
                this._drawHover(ctx, w, h, pad, n, this.hoverIdx,
                    this.fmtNum(point.total) + " request · " + (point.delta >= 0 ? "+" : "") + this.fmtNum(point.delta) + " · " + this.attributionLabel(point.growth_source));
            }
            this._selectedMark(ctx, w, h, pad, n, data);
        },

        renderContext() {
            const s = this._setupCanvas("ctxCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            const pad = { l: 56, r: 12, t: 14, b: 20 };
            ctx.clearRect(0, 0, w, h);
            this._curRef = "ctxCanvas";
            const ctxAll = this.series.context || [];
            const ctxData = this._view(ctxAll);
            if (!ctxData.length) { this._empty(ctx, w, h, "no context data"); return; }
            // The solid line is the provider's *reported* prompt size
            // (`actual`) — the ground truth, which can never exceed the model's
            // context window. The old code plotted the drift-corrected
            // `real_est` (cl100k × effective_drift_ratio, ×2.5 for Ollama) as
            // "real", which is a *conservative compaction guardrail*, not a
            // measurement: whenever the cl100k estimate overshot the
            // drift-corrected ceiling (emergency compaction territory) it
            // extrapolated to ~2× the raw window — an impossible "real prompt
            // size" that read as "reporting double the max context size."
            // `actual` for frontier providers is the true full prompt; for
            // Ollama it is the non-cached prompt delta (near-zero in a warm
            // loop = cache hit), shown honestly here. The drift-corrected
            // estimate is still surfaced in the hover tooltip + the dedicated
            // drift chart. `total_est` is the raw cl100k estimate.
            // y-scale is from the FULL series so it stays stable as you pan.
            const maxAll = ctxAll.flatMap(d => [d.actual, d.total_est]);
            const max = Math.max(1, ...maxAll) * 1.08;
            this._drawAxes(ctx, w, h, pad, { min: 0, max }, { xs: ctxData.map(d => d.iter), unit: "tokens" });

            const n = ctxData.length;
            const plotH = h - pad.t - pad.b;

            // actual (solid, provider-reported) + total_est (dashed, cl100k)
            const line = (key, color, dash) => {
                ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash || []);
                ctx.beginPath();
                for (let i = 0; i < n; i++) {
                    const x = this._iterX(i, n, pad, w);
                    const y = pad.t + plotH - ((ctxData[i][key] || 0) / max) * plotH;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke(); ctx.setLineDash([]);
            };
            line("actual", PALETTE.actual, []);
            line("total_est", PALETTE.est, [4, 3]);

            this._compactionMarks(ctx, w, h, pad, n);
            this._runMarks(ctx, w, h, pad, ctxData);

            // legend
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            ctx.fillStyle = PALETTE.actual; ctx.fillText("● actual (provider)", w - pad.r - 168, 4);
            ctx.fillStyle = PALETTE.est; ctx.fillText("┄ cl100k est", w - pad.r - 80, 4);

            // hover + selected
            if (this.hoverRef === "ctxCanvas" && this.hoverIdx != null) {
                const it = ctxData[this.hoverIdx];
                const dr = it.drift_ratio ? (" · drift ×" + it.drift_ratio) : "";
                this._drawHover(ctx, w, h, pad, n, this.hoverIdx,
                    this.fmtNum(it.actual) + " actual · " + this.fmtNum(it.total_est) + " est · "
                    + this.fmtNum(it.real_est) + " real_est" + dr);
            }
            this._selectedMark(ctx, w, h, pad, n, ctxData);
        },

        renderHeatStrip() {
            const s = this._setupCanvas("heatCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            const snap = this.snapshot || {};
            const meta = snap.meta || {};
            const layers = meta.layers || [];
            const labels = meta.labels || {};
            const xsAll = snap.xs || [];
            // The snapshot is downsampled to `cols` columns server-side, so
            // xsAll.length ≠ n on long runs. Window by iter number (not
            // position) so the strip pans in lockstep with the other charts
            // instead of clamping to its own shorter length and freezing.
            const [lo, hi] = this._iterBounds();
            const keep = [];
            for (let ci = 0; ci < xsAll.length; ci++) {
                const it = xsAll[ci];
                if (it != null && it >= lo && it <= hi) keep.push(ci);
            }
            const xs = keep.map(ci => xsAll[ci]);
            const gridAll = snap.grid || [];
            const grid = gridAll.map(row => keep.map(ci => (row || [])[ci] || 0));
            const driftStrip = keep.map(ci => (snap.drift_strip || [])[ci] || 128);
            const compCols = (snap.compaction_cols || []).map(ci => keep.indexOf(ci)).filter(ci => ci >= 0);
            if (!grid.length || !layers.length || !xs.length) { this._empty(ctx, w, h, "no layer data"); return; }

            const driftH = 14, gap = 3;
            const pad = { l: 96, r: 12, t: 6, b: 16 };
            const cols = xs.length;
            const cellW = (w - pad.l - pad.r) / Math.max(1, cols);
            const rowsH = h - pad.t - pad.b - driftH - gap;
            const rowH = rowsH / layers.length;

            // layer heat rows
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textBaseline = "middle";
            layers.forEach((key, ri) => {
                const y = pad.t + ri * rowH;
                ctx.fillStyle = t.dim; ctx.textAlign = "right";
                ctx.fillText(labels[key] || key, pad.l - 6, y + rowH / 2);
                const row = grid[ri] || [];
                for (let ci = 0; ci < cols; ci++) {
                    const v = row[ci] || 0;
                    const x = pad.l + ci * cellW;
                    if (v > 0) {
                        const a = 0.06 + (v / 255) * 0.9;
                        ctx.fillStyle = `rgba(${PALETTE.heat[0]},${PALETTE.heat[1]},${PALETTE.heat[2]},${a})`;
                        ctx.fillRect(x, y + 0.5, Math.max(0.6, cellW - 0.4), rowH - 1);
                    }
                }
            });
            // drift strip
            const dy = pad.t + rowsH + gap;
            ctx.fillStyle = t.dim; ctx.textAlign = "right"; ctx.textBaseline = "middle";
            ctx.fillText("drift", pad.l - 6, dy + driftH / 2);
            for (let ci = 0; ci < cols; ci++) {
                const v = driftStrip[ci] || 128;
                const x = pad.l + ci * cellW;
                let fill;
                if (v >= 128) {
                    const a = (v - 128) / 128;
                    fill = `rgba(${PALETTE.driftPos[0]},${PALETTE.driftPos[1]},${PALETTE.driftPos[2]},${a * 0.9})`;
                } else {
                    const a = (128 - v) / 128;
                    fill = `rgba(${PALETTE.driftNeg[0]},${PALETTE.driftNeg[1]},${PALETTE.driftNeg[2]},${a * 0.9})`;
                }
                ctx.fillStyle = fill;
                ctx.fillRect(x, dy, Math.max(0.6, cellW - 0.4), driftH);
            }
            // compaction column markers
            ctx.fillStyle = PALETTE.compaction;
            for (const ci of compCols) {
                ctx.fillRect(pad.l + ci * cellW + cellW / 2 - 1, pad.t, 2, rowsH + gap + driftH);
            }
            // selected-iter column highlight
            if (this.selectedIter !== null) {
                const ci = xs.indexOf(this.selectedIter);
                if (ci >= 0) {
                    ctx.strokeStyle = t.accent; ctx.lineWidth = 1;
                    ctx.strokeRect(pad.l + ci * cellW, pad.t - 1, cellW, rowsH + gap + driftH);
                }
            }
            // x labels
            ctx.fillStyle = t.dim; ctx.textAlign = "center"; ctx.textBaseline = "top";
            const want = cols > 60 ? 7 : (cols > 20 ? 5 : 3);
            for (let k = 0; k < want; k++) {
                const ci = Math.round((k / Math.max(1, want - 1)) * (cols - 1));
                ctx.fillText("iter " + xs[ci], pad.l + ci * cellW + cellW / 2, h - pad.b + 2);
            }
        },

        _selectedMark(ctx, w, h, pad, n, data) {
            if (this.selectedIter === null) return;
            const idx = data.findIndex(d => d.iter === this.selectedIter);
            if (idx < 0) return;
            const t = this._theme();
            const x = this._iterX(idx, n, pad, w);
            ctx.strokeStyle = t.accent; ctx.setLineDash([2, 2]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
            ctx.setLineDash([]);
        },

        renderDrift() {
            const s = this._setupCanvas("driftCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            const pad = { l: 56, r: 12, t: 14, b: 20 };
            ctx.clearRect(0, 0, w, h);
            this._curRef = "driftCanvas";
            const driftAll = this.series.drift || [];
            const drift = this._view(driftAll);
            if (!drift.length) { this._empty(ctx, w, h, "no drift data"); return; }
            const n = drift.length;
            const plotH = h - pad.t - pad.b;

            // raw values (window); clamp p99 from the FULL series so the
            // ±warn band + y-scale stay stable as you pan.
            const raw = drift.map(d => d.drift_pct);
            const absAll = driftAll.map(d => Math.abs(d.drift_pct)).sort((a, b) => a - b);
            const p99 = absAll[Math.min(absAll.length - 1, Math.floor(absAll.length * 0.99))];
            // Clamp the linear axis so the ±15% warn band stays visible; values
            // beyond the clamp get an off-chart cap triangle. (drift_pct is now
            // gated to reliable full-prompt readings — Ollama's near-zero cached
            // delta is zeroed instead of producing ±2000% spikes — but a cold
            // cache can still show a large real divergence, so keep the clamp.)
            const CLAMP = Math.min(200, Math.max(15, p99 || 15));
            const useLog = this.driftLog;
            // y transform: linear within ±CLAMP (capped), or signed-log
            const toY = (v) => {
                let nv;
                if (useLog) {
                    const s = Math.sign(v); const m = Math.log10(1 + Math.abs(v));
                    nv = s * m;
                } else {
                    nv = Math.max(-CLAMP, Math.min(CLAMP, v));
                }
                return pad.t + plotH / 2 - (nv / yScale) * (plotH / 2);
            };
            const yScale = useLog ? Math.log10(1 + CLAMP) : CLAMP;

            // ±15% warn band (only the in-range portion)
            const y15 = toY(15), yN15 = toY(-15);
            ctx.fillStyle = PALETTE.warnBand;
            ctx.fillRect(pad.l, Math.min(y15, yN15), w - pad.l - pad.r, Math.abs(y15 - yN15));
            // zero line
            ctx.strokeStyle = t.border; ctx.setLineDash([3, 3]);
            const y0 = toY(0);
            ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(w - pad.r, y0); ctx.stroke();
            ctx.setLineDash([]);

            // y labels
            ctx.fillStyle = t.dim; ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "right"; ctx.textBaseline = "middle";
            const ticks = useLog ? [-CLAMP, -15, 0, 15, CLAMP] : [-CLAMP, -15, 0, 15, CLAMP];
            for (const v of ticks) ctx.fillText((v > 0 ? "+" : "") + Math.round(v) + "%", pad.l - 4, toY(v));
            // unit caption
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            ctx.fillText(useLog ? "drift % (signed log)" : "drift %", pad.l + 2, 2);
            // x labels
            ctx.textAlign = "center"; ctx.textBaseline = "top";
            const want = n > 60 ? 7 : (n > 20 ? 5 : 3);
            for (let k = 0; k < want; k++) {
                const idx = Math.round((k / Math.max(1, want - 1)) * (n - 1));
                ctx.fillText("iter " + drift[idx].iter, this._iterX(idx, n, pad, w), h - pad.b + 4);
            }

            // drift line (with off-chart caps when linear & clamped)
            ctx.strokeStyle = PALETTE.drift; ctx.lineWidth = 1.8;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {
                const x = this._iterX(i, n, pad, w);
                const y = toY(raw[i]);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
            // cap triangles for clamped values
            if (!useLog) {
                ctx.fillStyle = PALETTE.drift;
                for (let i = 0; i < n; i++) {
                    if (Math.abs(raw[i]) > CLAMP) {
                        const x = this._iterX(i, n, pad, w);
                        const top = raw[i] > 0;
                        const yy = top ? pad.t + 3 : h - pad.b - 3;
                        ctx.beginPath();
                        ctx.moveTo(x, yy);
                        ctx.lineTo(x - 3, yy + (top ? 6 : -6));
                        ctx.lineTo(x + 3, yy + (top ? 6 : -6));
                        ctx.closePath(); ctx.fill();
                    }
                }
            }

            this._runMarks(ctx, w, h, pad, drift);

            // legend
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "right"; ctx.textBaseline = "top";
            ctx.fillStyle = PALETTE.drift; ctx.fillText("● drift", w - pad.r - 2, 2);
            ctx.textAlign = "left";
            ctx.fillStyle = "rgba(220,160,60,0.9)"; ctx.fillRect(pad.l + 2, h - pad.b - 14, 8, 8);
            ctx.fillStyle = t.dim; ctx.fillText("±15% warn", pad.l + 12, h - pad.b - 16);

            // selected + hover
            if (this.hoverRef === "driftCanvas" && this.hoverIdx != null) {
                const it = drift[this.hoverIdx];
                this._drawHover(ctx, w, h, pad, n, this.hoverIdx, it.drift_pct + "% drift");
            }
            this._selectedMark(ctx, w, h, pad, n, drift);
        },

        _compactionMarks(ctx, w, h, pad, n) {
            const comps = this.series.compaction_timeline || [];
            if (!comps.length || !n) return;
            const all = this.series.context || [];
            const start = this._winStart(all.length);
            ctx.fillStyle = PALETTE.compaction;
            for (const c of comps) {
                const idx = all.findIndex(d => d.iter === c.iter);
                if (idx < 0) continue;
                const local = idx - start;   // window-local index
                if (local < 0 || local >= n) continue;
                const x = this._iterX(local, n, pad, w);
                ctx.fillRect(x - 1, pad.t, 2, h - pad.t - pad.b);
            }
        },

        renderReads() {
            const s = this._setupCanvas("readCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            const itersAll = this.iters || [];
            const start = this._winStart(itersAll.length);
            const iters = this._view(itersAll);
            const n = iters.length;
            // Only genuine read tools, keyed by tool + path (a path can be read
            // by more than one tool). Writes carry paths too — exclude them.
            const tools = (this.tools || []).filter(t => READ_TOOLS.has(t.name) && t.path);
            if (!n || !tools.length) { this._empty(ctx, w, h, "no read-tool calls with a path"); return; }
            // unique (tool,path) rows in first-seen order
            const rows = []; const seen = {};
            for (const tool of tools) {
                const k = tool.name + "|" + tool.path;
                if (!seen[k]) { seen[k] = 1; rows.push({ name: tool.name, path: tool.path }); }
            }
            // No Y-axis filename labels: the concern is *duplicate* reads (red
            // cells), which the grid makes visible without per-row text. The
            // full path + read/redundant state pops up on hover (see _onReadHover).
            const pad = { l: 12, r: 12, t: 10, b: 18 };
            const plotH = h - pad.t - pad.b;
            const cellW = (w - pad.l - pad.r) / Math.max(1, n);
            // cap rows to what fits; floor rowH at 6px; note overflow
            let rowH = Math.min(16, plotH / Math.max(1, rows.length));
            rowH = Math.max(6, rowH);
            const maxRows = Math.floor(plotH / rowH);
            const shownRows = rows.slice(0, maxRows);
            const overflow = rows.length - shownRows.length;

            const redundant = new Set((this.series.redundant_reads || []).map(r => r.iter + "|" + r.path));
            const byRow = {};
            for (const tool of tools) {
                const k = tool.name + "|" + tool.path;
                (byRow[k] = byRow[k] || {})[tool.iter] = tool;
            }
            // publish geometry for the hover handler (row/col hit testing)
            const cv = this.$refs.readCanvas;
            if (cv) cv._geom = { pad, cellW, rowH, n, rows: shownRows, iters, byRow, redundant, w, h };

            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textBaseline = "middle";
            shownRows.forEach((r, ri) => {
                const y = pad.t + ri * rowH;
                const k = r.name + "|" + r.path;
                const cells = byRow[k] || {};
                for (let i = 0; i < n; i++) {
                    const it = iters[i].iter;
                    if (cells[it] !== undefined) {
                        const x = pad.l + i * cellW;
                        const isRed = redundant.has(it + "|" + r.path);
                        ctx.fillStyle = isRed ? PALETTE.redundant : PALETTE.read;
                        ctx.globalAlpha = isRed ? 0.85 : 0.6;
                        ctx.fillRect(x + 0.5, y + 1, Math.max(1.2, cellW - 1), rowH - 2);
                        ctx.globalAlpha = 1;
                    }
                }
            });
            if (overflow > 0) {
                ctx.fillStyle = t.dim; ctx.textAlign = "right"; ctx.textBaseline = "top";
                ctx.fillText("+" + overflow + " more paths (hover the grid to identify a row)", w - pad.r - 2, 2);
            }
            // x labels
            ctx.fillStyle = t.dim; ctx.textAlign = "center"; ctx.textBaseline = "top";
            const want = n > 60 ? 7 : (n > 20 ? 5 : 3);
            for (let k = 0; k < want; k++) {
                const idx = Math.round((k / Math.max(1, want - 1)) * (n - 1));
                ctx.fillText("iter " + iters[idx].iter, pad.l + idx * cellW + cellW / 2, h - pad.b + 2);
            }
            this._runMarks(ctx, w, h, pad, iters);
            // legend
            ctx.textAlign = "left"; ctx.textBaseline = "middle";
            ctx.fillStyle = PALETTE.read; ctx.globalAlpha = 0.6; ctx.fillRect(8, 8, 9, 9); ctx.globalAlpha = 1;
            ctx.fillStyle = t.dim; ctx.fillText("read", 20, 13);
            ctx.fillStyle = PALETTE.redundant; ctx.fillRect(54, 8, 9, 9);
            ctx.fillStyle = t.dim; ctx.fillText("redundant re-read", 66, 13);

            // hover tooltip (filename popup on mouse-over)
            if (this.hoverRef === "readCanvas" && this.readHover && this.readHover.label) {
                const { x, y, label } = this.readHover;
                ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
                const tw = ctx.measureText(label).width + 10;
                let cx = x + 10, cy = y - 18;
                if (cx + tw > w - pad.r) cx = x - tw - 10;
                if (cx < pad.l) cx = pad.l;
                if (cy < pad.t) cy = y + 14;
                ctx.fillStyle = t.bg; ctx.strokeStyle = t.border;
                ctx.fillRect(cx, cy, tw, 16); ctx.strokeRect(cx, cy, tw, 16);
                ctx.fillStyle = t.text; ctx.textAlign = "left"; ctx.textBaseline = "middle";
                ctx.fillText(label, cx + 5, cy + 8);
            }
        },

        _onReadHover(e, cv) {
            if (this._dragCV) return;                  // dragging pans; don't chase hover
            const g = cv._geom; if (!g) return;
            const rect = cv.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;
            const plotR = g.w - g.pad.r, plotB = g.h - g.pad.b;
            if (y < g.pad.t || y > plotB || x < g.pad.l || x > plotR) {
                if (this.hoverRef === "readCanvas") { this.readHover = null; this.hoverRef = null; this.renderAll(); }
                return;
            }
            const col = Math.max(0, Math.min(g.n - 1, Math.floor((x - g.pad.l) / Math.max(0.6, g.cellW))));
            const row = Math.floor((y - g.pad.t) / g.rowH);
            if (row < 0 || row >= g.rows.length) {
                if (this.hoverRef === "readCanvas") { this.readHover = null; this.hoverRef = null; this.renderAll(); }
                return;
            }
            const r = g.rows[row];
            const it = g.iters[col] ? g.iters[col].iter : null;
            const isRed = it != null && g.redundant.has(it + "|" + r.path);
            const cell = it != null ? (g.byRow[r.name + "|" + r.path] || {})[it] : undefined;
            let label = r.name + " " + r.path;
            if (it != null) label += " · iter " + it + (isRed ? " · redundant re-read" : (cell ? " · read" : ""));
            this.hoverRef = "readCanvas";
            this.readHover = { x, y, label };
            this.renderAll();
        },

        renderSub() {
            const s = this._setupCanvas("subCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            const saAll = this.series.subagent_timeline || [];
            const sa = this._view(saAll);
            const n = sa.length;
            const anyActive = saAll.some(d => d.active > 0 || d.stuck || d.stall);
            if (!n || !anyActive) { this._empty(ctx, w, h, "no subagents spawned this run"); return; }
            const pad = { l: 56, r: 12, t: 14, b: 18 };
            const maxActive = Math.max(1, ...saAll.map(d => d.active));
            const plotH = h - pad.t - pad.b;
            this._drawAxes(ctx, w, h, pad, { min: 0, max: maxActive }, { xs: sa.map(d => d.iter), unit: "active" });
            ctx.fillStyle = PALETTE.subBar;
            const barW = Math.max(2, (w - pad.l - pad.r) / n - 2);
            for (let i = 0; i < n; i++) {
                const d = sa[i];
                if (d.active <= 0) continue;
                const x = this._iterX(i, n, pad, w) - barW / 2;
                const bh = (d.active / maxActive) * plotH;
                ctx.fillStyle = PALETTE.subBar;
                ctx.fillRect(x, pad.t + plotH - bh, barW, bh);
                // stuck + stall as two stacked caps so both are visible
                if (d.stuck) {
                    ctx.fillStyle = PALETTE.stuck;
                    ctx.fillRect(x, pad.t + plotH - bh - 3, barW, 3);
                }
                if (d.stall) {
                    ctx.fillStyle = PALETTE.stall;
                    ctx.fillRect(x, pad.t + plotH - bh - (d.stuck ? 6 : 3), barW, 3);
                }
            }
            this._runMarks(ctx, w, h, pad, sa);
            // legend
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            ctx.fillStyle = PALETTE.subBar; ctx.fillText("● active", w - pad.r - 130, 2);
            ctx.fillStyle = PALETTE.stuck; ctx.fillText("■ stuck", w - pad.r - 70, 2);
            ctx.fillStyle = PALETTE.stall; ctx.fillText("■ stall", w - pad.r - 24, 2);
        },

        renderMem() {
            const s = this._setupCanvas("memCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            const memAll = this.series.memory_series || [];
            const mem = this._view(memAll);
            const n = mem.length;
            if (!n) { this._empty(ctx, w, h, "no memory data"); return; }
            const pad = { l: 56, r: 12, t: 14, b: 18 };
            const counts = mem.map(d => Math.max(d.task_memory_count, d.scratchpad_count));
            // include by_status stack peaks in the y max so the stack doesn't overflow;
            // status keys from the FULL run so the legend lists every status seen.
            const statusKeys = this._statusKeys(memAll);
            const stackPeaks = memAll.map(d => statusKeys.reduce((a, k) => a + ((d.by_status || {})[k] || 0), 0));
            const max = Math.max(1, ...counts, ...stackPeaks) * 1.1;
            const plotH = h - pad.t - pad.b;
            this._drawAxes(ctx, w, h, pad, { min: 0, max }, { xs: mem.map(d => d.iter), unit: "count" });

            // stacked by_status area (faint, behind the count lines)
            const stackTop = new Array(n).fill(0);
            for (const k of statusKeys) {
                const vals = mem.map(d => (d.by_status || {})[k] || 0);
                const col = PALETTE.status[k] || PALETTE.statusDefault;
                ctx.fillStyle = col; ctx.globalAlpha = 0.22;
                ctx.beginPath();
                for (let i = 0; i < n; i++) {
                    const x = this._iterX(i, n, pad, w);
                    const y = pad.t + plotH - ((stackTop[i] + vals[i]) / max) * plotH;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                for (let i = n - 1; i >= 0; i--) {
                    const x = this._iterX(i, n, pad, w);
                    const y = pad.t + plotH - (stackTop[i] / max) * plotH;
                    ctx.lineTo(x, y);
                }
                ctx.closePath(); ctx.fill();
                for (let i = 0; i < n; i++) stackTop[i] += vals[i];
                ctx.globalAlpha = 1;
            }

            // count lines (scratchpad dashed to match its legend glyph)
            const line = (key, color, dash) => {
                ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash || []);
                ctx.beginPath();
                for (let i = 0; i < n; i++) {
                    const x = this._iterX(i, n, pad, w);
                    const y = pad.t + plotH - (mem[i][key] / max) * plotH;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke(); ctx.setLineDash([]);
            };
            line("task_memory_count", PALETTE.taskMem, []);
            line("scratchpad_count", PALETTE.scratchpad, [4, 3]);

            this._runMarks(ctx, w, h, pad, mem);

            // legend (counts + status)
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            let lx = w - pad.r - 150;
            ctx.fillStyle = PALETTE.taskMem; ctx.fillText("● task_memory", lx, 2); lx += 78;
            ctx.fillStyle = PALETTE.scratchpad; ctx.fillText("┄ scratchpad", lx, 2);
            // status legend along the bottom-left
            let sx = pad.l + 2; const sy = h - pad.b - 12;
            for (const k of statusKeys) {
                const col = PALETTE.status[k] || PALETTE.statusDefault;
                ctx.globalAlpha = 0.5; ctx.fillStyle = col; ctx.fillRect(sx, sy, 8, 8); ctx.globalAlpha = 1;
                ctx.fillStyle = t.dim; ctx.fillText(k, sx + 11, sy - 1);
                sx += 11 + ctx.measureText(k).width + 12;
            }
        },

        _statusKeys(mem) {
            const s = {}; for (const d of mem) for (const k of Object.keys(d.by_status || {})) s[k] = 1;
            // stable, conventional order
            const order = ["active", "pending", "stale", "done", "pruned"];
            return order.filter(k => s[k]).concat(Object.keys(s).filter(k => !order.includes(k)).sort());
        },

        renderTokens() {
            const s = this._setupCanvas("tokCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            this._curRef = "tokCanvas";
            const tkAll = this.series.tokens || [];
            const tk = this._view(tkAll);
            const n = tk.length;
            if (!n) { this._empty(ctx, w, h, "no token data"); return; }
            const pad = { l: 56, r: 12, t: 14, b: 20 };
            const max = Math.max(1, ...tkAll.map(d => Math.max(d.in, d.out, d.cached, d.reasoning))) * 1.1;
            const plotH = h - pad.t - pad.b;
            this._drawAxes(ctx, w, h, pad, { min: 0, max }, { xs: tk.map(d => d.iter), unit: "tokens" });
            const line = (key, color) => {
                ctx.strokeStyle = color; ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (let i = 0; i < n; i++) {
                    const x = this._iterX(i, n, pad, w);
                    const y = pad.t + plotH - ((tk[i][key] || 0) / max) * plotH;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
            };
            line("in", PALETTE.tokIn);
            line("out", PALETTE.tokOut);
            line("cached", PALETTE.tokCached);
            line("reasoning", PALETTE.tokReasoning);
            this._runMarks(ctx, w, h, pad, tk);
            // stacked vertical legend (4 entries × 32px collided)
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            const lg = [["in", PALETTE.tokIn], ["out", PALETTE.tokOut], ["cached", PALETTE.tokCached], ["reasoning", PALETTE.tokReasoning]];
            lg.forEach(([k, c], i) => {
                const y = 2 + i * 12;
                ctx.fillStyle = c; ctx.fillText("● " + k, w - pad.r - 70, y);
            });
            if (this.hoverRef === "tokCanvas" && this.hoverIdx != null) {
                const it = tk[this.hoverIdx];
                this._drawHover(ctx, w, h, pad, n, this.hoverIdx, "in " + this.fmtNum(it.in) + " / out " + this.fmtNum(it.out));
            }
        },

        // Tool-output efficiency: raw tool tokens (amber) vs injected tokens
        // (green). The amber band between the raw line and the green area is
        // tokens kept out of context by storing raw externally + observing.
        effSummary() {
            return (this.summary && this.summary.efficiency) || {};
        },
        // Ranked harness-suspects digest from build_summary (same block the
        // trace tools return). Empty for clean runs -> section stays hidden.
        suspectsList() {
            return (this.summary && this.summary.suspects) || [];
        },
        // Jump from a suspect card to the evidence: map suspect -> panel.
        // Compaction rows live in the compaction timeline; drift in the
        // drift chart; everything else jumps to the first affected
        // iteration recorded on the suspect itself.
        suspectJump(s) {
            const m = /iters? \[([0-9,\s]+)\]/.exec(s.next || "");
            if (m) {
                const first = m[1].split(",")[0].trim();
                if (first) { this.selectIter(parseInt(first, 10)); return; }
            }
            window.scrollTo({ top: 0, behavior: "smooth" });
        },
        renderEfficiency() {
            const s = this._setupCanvas("effCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            this._curRef = "effCanvas";
            const effAll = this.series.efficiency || [];
            const eff = this._view(effAll);
            const n = eff.length;
            if (!n) { this._empty(ctx, w, h, "no tool-output efficiency data"); return; }
            const pad = { l: 56, r: 12, t: 16, b: 20 };
            const max = Math.max(1, ...effAll.map(d => Math.max(d.raw_tokens || 0, d.injected_tokens || 0))) * 1.1;
            const plotH = h - pad.t - pad.b;
            this._drawAxes(ctx, w, h, pad, { min: 0, max }, { xs: eff.map(d => d.iter), unit: "tokens" });
            const x = i => this._iterX(i, n, pad, w);
            const y = v => pad.t + plotH - ((v || 0) / max) * plotH;
            // injected tokens — filled area (what actually entered context)
            ctx.fillStyle = "rgba(126,201,107,0.32)";
            ctx.strokeStyle = PALETTE.tokOut; ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < n; i++) { ctx.lineTo(x(i), y(eff[i].injected_tokens)); }
            ctx.lineTo(x(n - 1), pad.t + plotH); ctx.lineTo(x(0), pad.t + plotH);
            ctx.closePath(); ctx.fill();
            ctx.beginPath();
            for (let i = 0; i < n; i++) { const px = x(i); const py = y(eff[i].injected_tokens); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
            ctx.stroke();
            // raw tokens — line above (what would have entered without observing)
            ctx.strokeStyle = PALETTE.actual; ctx.lineWidth = 1.6; ctx.setLineDash([5, 3]);
            ctx.beginPath();
            for (let i = 0; i < n; i++) { const px = x(i); const py = y(eff[i].raw_tokens); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
            ctx.stroke(); ctx.setLineDash([]);
            this._runMarks(ctx, w, h, pad, eff);
            // legend
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            ctx.fillStyle = PALETTE.actual; ctx.fillText("┄ raw (would-be)", w - pad.r - 150, 2);
            ctx.fillStyle = PALETTE.tokOut; ctx.fillText("● injected (kept)", w - pad.r - 70, 2);
            if (this.hoverRef === "effCanvas" && this.hoverIdx != null) {
                const it = eff[this.hoverIdx];
                const saved = (it.raw_tokens || 0) - (it.injected_tokens || 0);
                this._drawHover(ctx, w, h, pad, n, this.hoverIdx,
                    this.fmtNum(it.raw_tokens) + " raw · " + this.fmtNum(it.injected_tokens) + " inj · " + this.fmtNum(saved) + " saved");
            }
        },

        renderLatency() {
            const s = this._setupCanvas("latCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            this._curRef = "latCanvas";
            const latAll = this.series.latency || [];
            const lat = this._view(latAll);
            const n = lat.length;
            if (!n) { this._empty(ctx, w, h, "no latency data"); return; }
            const pad = { l: 56, r: 12, t: 14, b: 18 };
            const max = Math.max(1, ...latAll.map(d => d.wall_ms)) * 1.1;
            const plotH = h - pad.t - pad.b;
            this._drawAxes(ctx, w, h, pad, { min: 0, max }, { xs: lat.map(d => d.iter), unit: "wall ms" });
            const barW = Math.max(2, (w - pad.l - pad.r) / n - 1);
            for (let i = 0; i < n; i++) {
                const x = this._iterX(i, n, pad, w) - barW / 2;
                const bh = (lat[i].wall_ms / max) * plotH;
                ctx.fillStyle = (this.selectedIter === lat[i].iter) ? PALETTE.actual : "rgba(176,172,160,0.6)";
                ctx.fillRect(x, pad.t + plotH - bh, barW, bh);
            }
            this._runMarks(ctx, w, h, pad, lat);
            if (this.hoverRef === "latCanvas" && this.hoverIdx != null) {
                const it = lat[this.hoverIdx];
                this._drawHover(ctx, w, h, pad, n, this.hoverIdx, this.fmtMs(it.wall_ms) + " wall");
            }
        },

        renderTool() {
            const s = this._setupCanvas("toolCanvas"); if (!s) return;
            const { ctx, w, h } = s; const t = this._theme();
            ctx.clearRect(0, 0, w, h);
            const tsAll = this.selectedToolSeries();
            const ts = this._view(tsAll);
            if (!ts.length) { this._empty(ctx, w, h, this.selectedTool ? "no calls" : "select a tool"); return; }
            const pad = { l: 56, r: 12, t: 14, b: 18 };
            const max = Math.max(1, ...tsAll.map(d => d.latency_ms)) * 1.1;
            const n = ts.length;
            const plotH = h - pad.t - pad.b;
            this._drawAxes(ctx, w, h, pad, { min: 0, max }, { xs: ts.map(d => d.iter), unit: "ms" });
            const barW = Math.max(2, (w - pad.l - pad.r) / n - 2);
            for (let i = 0; i < n; i++) {
                const x = this._iterX(i, n, pad, w) - barW / 2;
                const bh = (ts[i].latency_ms / max) * plotH;
                ctx.fillStyle = ts[i].ok ? PALETTE.ok : PALETTE.err;
                ctx.fillRect(x, pad.t + plotH - bh, barW, Math.max(1, bh));
                if (ts[i].cache_hit) {
                    ctx.fillStyle = PALETTE.cache;
                    ctx.fillRect(x, pad.t + plotH - bh - 2, barW, 2);
                }
            }
            // legend
            ctx.font = "10px " + (getComputedStyle(document.body).fontFamily);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            ctx.fillStyle = PALETTE.ok; ctx.fillText("● ok", w - pad.r - 96, 2);
            ctx.fillStyle = PALETTE.err; ctx.fillText("● error", w - pad.r - 60, 2);
            ctx.fillStyle = PALETTE.cache; ctx.fillText("▬ cache hit", w - pad.r - 24, 2);
        },
    };
}
