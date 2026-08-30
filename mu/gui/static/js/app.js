// μcli GUI — chat-first frontend wired to FastAPI + SSE.
//
// Alpine stores:
//   chat       — turn buffer, history fetch, send
//   sessions   — list/switch/create/delete sessions
//   mode       — agent mode tabs (default/debug/feature/...)
//   prompts    — pending blocking-prompt queue (modal)
//   tokens     — running token meter
//   toast      — ephemeral notifications (error/success/info)
//
// Bootstrap order is in DOMContentLoaded at the bottom.

// MuCLI session type icons — same Ionicons used by the mobile app.
// Kept as local inline SVG so the web GUI has no CDN or icon-font dependency.
const MUCLI_SESSION_TYPE_ICON_NAMES = Object.freeze({
    workspace: "folder-open-outline",
    container: "cube-outline",
    chat: "chatbubble-ellipses-outline",
});

const MUCLI_SESSION_TYPE_ICON_CONTENT = Object.freeze({
    workspace: [
        '<path d="M64,192V120a40,40,0,0,1,40-40h75.89a40,40,0,0,1,22.19,6.72l27.84,18.56A40,40,0,0,0,252.11,112H408a40,40,0,0,1,40,40v40" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32"/>',
        '<path d="M479.9,226.55,463.68,392a40,40,0,0,1-39.93,40H88.25a40,40,0,0,1-39.93-40L32.1,226.55A32,32,0,0,1,64,192h384.1A32,32,0,0,1,479.9,226.55Z" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32"/>',
    ].join(""),
    container: [
        '<path d="M448,341.37V170.61A32,32,0,0,0,432.11,143l-152-88.46a47.94,47.94,0,0,0-48.24,0L79.89,143A32,32,0,0,0,64,170.61V341.37A32,32,0,0,0,79.89,369l152,88.46a48,48,0,0,0,48.24,0l152-88.46A32,32,0,0,0,448,341.37Z" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32"/>',
        '<polyline points="69 153.99 256 263.99 443 153.99" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32"/>',
        '<line x1="256" y1="463.99" x2="256" y2="263.99" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32"/>',
    ].join(""),
    chat: [
        '<path d="M87.48,380c1.2-4.38-1.43-10.47-3.94-14.86A42.63,42.63,0,0,0,81,361.34a199.81,199.81,0,0,1-33-110C47.64,139.09,140.72,48,255.82,48,356.2,48,440,117.54,459.57,209.85A199,199,0,0,1,464,251.49c0,112.41-89.49,204.93-204.59,204.93-18.31,0-43-4.6-56.47-8.37s-26.92-8.77-30.39-10.11a31.14,31.14,0,0,0-11.13-2.07,30.7,30.7,0,0,0-12.08,2.43L81.5,462.78A15.92,15.92,0,0,1,76.84,464a9.61,9.61,0,0,1-9.58-9.74,15.85,15.85,0,0,1,.6-3.29Z" fill="none" stroke="currentColor" stroke-linecap="round" stroke-miterlimit="10" stroke-width="32"/>',
        '<circle cx="160" cy="256" r="32" fill="currentColor"/>',
        '<circle cx="256" cy="256" r="32" fill="currentColor"/>',
        '<circle cx="352" cy="256" r="32" fill="currentColor"/>',
    ].join(""),
});

// The rail represents user prompts (conversation checkpoints), not arbitrary
// provider/tool messages. Scan bounded raw-turn pages until each explicit
// history load has found a useful checkpoint batch.
const WEB_HISTORY_PAGE_TURNS = 200;
const WEB_HISTORY_CHECKPOINT_BATCH = 5;
const WEB_HISTORY_CHECKPOINT_SCAN_PAGES = 6;

window.sessionTypeIconName = function sessionTypeIconName(type) {
    const normalized = type === "container" || type === "chat" ? type : "workspace";
    return MUCLI_SESSION_TYPE_ICON_NAMES[normalized];
};

window.sessionTypeIconMarkup = function sessionTypeIconMarkup(type, className = "session-type-svg") {
    const normalized = type === "container" || type === "chat" ? type : "workspace";
    const safeClass = String(className || "session-type-svg").replace(/[^A-Za-z0-9 _-]/g, "");
    return `<svg class="${safeClass}" viewBox="0 0 512 512" aria-hidden="true" focusable="false">${MUCLI_SESSION_TYPE_ICON_CONTENT[normalized]}</svg>`;
};

// MUCLI_CONTAINER_PERSISTENCE_V1 — actionable session-load errors.
window.describeSessionLoadError = function describeSessionLoadError(payload, fallbackTitle = "Session could not be loaded") {
    const root = payload && typeof payload === "object" ? payload : {};
    const detail = root.detail !== undefined ? root.detail : payload;
    if (detail && typeof detail === "object") {
        const title = String(detail.title || fallbackTitle);
        const message = String(detail.message || "The session could not be loaded.");
        const steps = Array.isArray(detail.resolution_steps) ? detail.resolution_steps : [];
        const technical = String(detail.technical_detail || "").trim();
        const lines = [message];
        if (steps.length) {
            lines.push("", "Resolution:", ...steps.map((step, index) => `${index + 1}. ${step}`));
        }
        if (technical) lines.push("", `Technical detail: ${technical}`);
        return { title, text: lines.join("\n"), code: String(detail.code || "session_load_failed") };
    }
    const text = typeof detail === "string" && detail.trim()
        ? detail.trim()
        : fallbackTitle;
    return { title: fallbackTitle, text, code: "session_load_failed" };
};

document.addEventListener("alpine:init", () => {
    Alpine.store("chat", {
        // Multi-session: each session has its own slot in `chats`. The
        // `currentName` selector flips when the user clicks a different
        // session in the sidebar. Top-level getters (turns, busy, …)
        // proxy to the focused slot, so existing template bindings like
        // `$store.chat.turns` keep working without per-template changes.
        chats: {},
        currentName: null,
        // Connection status (SSE) is a global concern, not per-session.
        connected: null,
        lastOpenAt: 0,
        _renderRaf: 0,
        _scrollRaf: 0,
        _highlightRaf: 0,
        copiedMessageId: null,

        // ---------- per-session slot management ----------------------

        _newSlot() {
            return {
                turns: [],
                busy: false,
                externalActive: false,
                clock: 0,
                // A history reload was requested while a turn was in flight
                // (SSE reconnect or a session_updated from another process).
                // The in-flight assistant text isn't saved server-side yet,
                // so reloading mid-stream would wipe it and leave only the
                // tail deltas in a fresh turn ("output disappeared mid
                // response"). Defer the reload until the turn completes.
                pendingReload: false,
                historyHydrated: false,
                historyLoading: false,
                // A response hand-off waits for replacement text before
                // retiring the previous readable response.  Keeping the
                // timer per session prevents background sessions from
                // disturbing the focused transcript.
                handoffTimer: null,
                // Round-44 F4: streaming deltas coalesce per slot. Text
                // accumulates in pendingDelta and flushes to the turn at a
                // fixed cadence (RAF + 32ms floor) instead of running the
                // full O(history) turn lookup + re-markdown per token.
                pendingDelta: null,
                deltaTimer: null,
                // Round-44 F3: backward pagination cursors. hasMore turns
                // exist above start_index; olderLoading guards one request
                // at a time. hasMoreTurns is exposed for the header button.
                oldestLoadedIndex: null,
                windowEndIndex: null,
                totalHistoryTurns: 0,
                hasMoreTurns: false,
                olderLoading: false,
                activeStageId: null,
                suppressEdgeLoadUntil: 0,
            };
        },
        _slot(name) {
            const key = name || this.currentName || "__default__";
            if (!this.chats[key]) this.chats[key] = this._newSlot();
            return this.chats[key];
        },
        current() { return this._slot(); },
        focus(name) {
            const prev = this.currentName;
            this.currentName = name || null;
            this._slot(name);   // ensure created
            this.followOutput = true;
            // Round-48 F5: the SSE stream is server-filtered to the focused
            // session (r47 F3) — switching focus MUST reconnect with the new
            // filter, or the newly focused session silently receives no
            // live events while the connection looks healthy. Guard: only
            // reconnect on an actual session change.
            if ((prev || null) !== (this.currentName || null)) {
                reconnectWithFocus();
            }
            // Double-RAF: first frame lets Alpine swap the x-for turns to the
            // new session, second frame lets the DOM layout with the new
            // content. A single RAF fires before Alpine renders → scrollHeight
            // is stale → we land mid-content instead of at the bottom.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => this.scroll(true));
            });
        },

        // Back-compat top-level getters: legacy templates read e.g.
        // `$store.chat.turns` / `$store.chat.busy`. Route those to the
        // focused slot so we don't have to rewrite every binding.
        get turns()          { return this._slot().turns; },
        set turns(v)         { this._slot().turns = v; },
        get busy()           { return this._slot().busy; },
        set busy(v)          { this._slot().busy = v; },
        get externalActive() { return this._slot().externalActive; },
        set externalActive(v){ this._slot().externalActive = v; },
        get clock()          { return this._slot().clock; },
        set clock(v)         { this._slot().clock = v; },
        get activeStageId()  { return this._slot().activeStageId; },
        set activeStageId(v) { this._slot().activeStageId = v; },
        get historyLoading() { return this._slot().historyLoading; },
        get stages() {
            const seen = new Set();
            return this.turns
                .filter(turn => {
                    if (turn.role !== "user") return false;
                    const key = Number.isInteger(turn.historyIndex)
                        ? `history:${turn.historyIndex}`
                        : `live:${turn.id}`;
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                })
                .map(turn => ({
                    id: turn.id,
                    historyIndex: turn.historyIndex,
                    text: String(turn.text || "(attachment)")
                        .replace(/\s+/g, " ").trim().slice(0, 72),
                }));
        },

        // ---------- helpers ------------------------------------------

        _id(prefix) {
            return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
        },
        async copyMessage(text, messageId) {
            if (!text || !(await copyToClipboard(text))) return;
            this.copiedMessageId = messageId;
            setTimeout(() => {
                if (this.copiedMessageId === messageId) this.copiedMessageId = null;
            }, 1500);
        },
        _lastTurn(slot) {
            const turns = slot.turns;
            return turns.length ? turns[turns.length - 1] : null;
        },
        // MUCLI_SINGLE_TRACE_LIVE_INTERIM_V1: one authoritative running trace per session.
        _currentTrace(slot) {
            for (let index = slot.turns.length - 1; index >= 0; index -= 1) {
                const turn = slot.turns[index];
                if (turn.role === "trace") return turn;
            }
            return null;
        },
        _activeTrace(slot) {
            let active = null;
            for (let index = slot.turns.length - 1; index >= 0; index -= 1) {
                const turn = slot.turns[index];
                if (turn.role !== "trace" || !turn.running) continue;
                if (active === null) {
                    active = turn;
                    continue;
                }
                // Retire stale duplicate flags left by reconnect/order races.
                turn.running = false;
                turn.elapsed = turn.elapsed || (
                    turn.startedAt
                        ? ((Date.now() - turn.startedAt) / 1000).toFixed(1)
                        : null
                );
            }
            return active;
        },
        isActiveTrace(turn) {
            if (!turn || turn.role !== "trace" || !turn.running) return false;
            const active = this._activeTrace(this.current());
            return Boolean(active && active.id === turn.id);
        },
        _hasStreamingAssistant(slot) {
            return slot.turns.some(turn =>
                turn.role === "assistant" && turn.streaming
            );
        },
        _closeStreamingAssistants(slot, exceptId = null) {
            for (const turn of slot.turns) {
                if (
                    turn.role === "assistant"
                    && turn.streaming
                    && (!exceptId || turn.id !== exceptId)
                ) {
                    turn.streaming = false;
                    turn.html = renderMarkdown(turn.text || "");
                }
            }
        },
        _ensureTrace(slot) {
            let trace = this._activeTrace(slot);
            if (trace) return trace;
            trace = {
                id: this._id("tr"),
                role: "trace",
                events: [],
                open: false,
                running: true,
                startedAt: Date.now(),
                elapsed: null,
            };
            slot.turns.push(trace);
            return trace;
        },
        _ensureBusyTrace(slot) {
            // Poll/hello recovery must not create a second "thinking" row
            // while assistant text is actively streaming.
            if (this._hasStreamingAssistant(slot)) return this._activeTrace(slot);
            return this._ensureTrace(slot);
        },
        _closeTrace(slot) {
            for (const trace of slot.turns) {
                if (trace.role !== "trace" || !trace.running) continue;
                trace.running = false;
                trace.elapsed = trace.elapsed || (
                    trace.startedAt
                        ? ((Date.now() - trace.startedAt) / 1000).toFixed(1)
                        : null
                );
            }
        },
        _isLiveTimelineTurn(turn) {
            return (
                (turn.role === "assistant" && turn.streaming)
                || (turn.role === "trace" && turn.running)
                || (turn.role === "subagent_panel" && turn.running)
            );
        },
        _isTimelineAnchor(turn) {
            return turn && (
                turn.role === "visualization"
                || turn.role === "subagent_panel"
            );
        },
        _collapseSummary(childTurns) {
            let totalElapsed = 0;
            let totalTokens = 0;
            for (const child of childTurns) {
                if (child.elapsed) totalElapsed += parseFloat(child.elapsed) || 0;
                if (child.tokens) totalTokens += Number(child.tokens) || 0;
            }
            return {
                count: childTurns.length,
                elapsed: totalElapsed > 0 ? `${totalElapsed.toFixed(1)}s` : "",
                tokens: totalTokens > 0
                    ? (totalTokens >= 1000
                        ? `${(totalTokens / 1000).toFixed(1)}k`
                        : String(totalTokens))
                    : "",
            };
        },
        _pushCollapsedSegments(target, turns, options) {
            const opts = options || {};
            let segment = [];
            let segmentIndex = 0;
            const flush = () => {
                if (!segment.length) return;
                const childTurns = segment;
                segment = [];
                const groupKey = `${opts.groupKey || "group"}:${segmentIndex++}`;
                const summary = this._collapseSummary(childTurns);
                target.push({
                    id: `${opts.idPrefix || "collapse"}-${segmentIndex}`,
                    role: "collapse",
                    groupKey,
                    userId: opts.userId || "",
                    live: Boolean(opts.live),
                    childTurns,
                    ...summary,
                    open: opts.openByKey && opts.openByKey.has(groupKey)
                        ? Boolean(opts.openByKey.get(groupKey))
                        : Boolean(opts.defaultOpen),
                });
            };
            for (const turn of turns) {
                if (this._isTimelineAnchor(turn)) {
                    flush();
                    target.push(turn);
                } else {
                    segment.push(turn);
                }
            }
            flush();
        },
        _foldLiveInterim(slot, currentId = null) {
            let userIndex = -1;
            for (let index = slot.turns.length - 1; index >= 0; index -= 1) {
                if (slot.turns[index].role === "user") {
                    userIndex = index;
                    break;
                }
            }
            if (userIndex < 0) return;

            const userTurn = slot.turns[userIndex];
            const openByKey = new Map();
            const flatten = (items) => {
                const flattened = [];
                for (const item of items) {
                    if (item.role === "collapse") {
                        if (item.groupKey) openByKey.set(item.groupKey, Boolean(item.open));
                        flattened.push(...flatten(item.childTurns || []));
                    } else {
                        flattened.push(item);
                    }
                }
                return flattened;
            };
            const tail = flatten(slot.turns.slice(userIndex + 1));
            let currentIndex = -1;
            for (let index = tail.length - 1; index >= 0; index -= 1) {
                const turn = tail[index];
                if (
                    turn.role === "assistant"
                    && ((currentId && turn.id === currentId) || (!currentId && turn.streaming))
                ) {
                    currentIndex = index;
                    break;
                }
            }
            // No successor response exists yet. The previous response stays
            // fully readable while tools/thinking continue in the background.
            if (currentIndex < 0) return;

            const regrouped = [];
            this._pushCollapsedSegments(regrouped, tail.slice(0, currentIndex), {
                idPrefix: `live-collapse-${userTurn.id}`,
                groupKey: `live:${userTurn.id}`,
                userId: userTurn.id,
                live: true,
                openByKey,
            });
            regrouped.push(...tail.slice(currentIndex));
            slot.turns.splice(userIndex + 1, slot.turns.length - userIndex - 1, ...regrouped);
        },
        _beginAssistantHandoff(slot, current, name) {
            if (!current || !current.text) return;
            let currentIndex = slot.turns.indexOf(current);
            if (currentIndex < 0) return;
            let hasPrevious = false;
            for (let index = currentIndex - 1; index >= 0; index -= 1) {
                const turn = slot.turns[index];
                if (turn.role === "user") break;
                if (turn.role === "assistant" && turn.text && !turn.streaming) {
                    turn.leaving = true;
                    hasPrevious = true;
                }
            }
            if (!hasPrevious) return;
            current.entering = true;
            if (slot.handoffTimer) clearTimeout(slot.handoffTimer);
            slot.handoffTimer = setTimeout(() => {
                slot.handoffTimer = null;
                current.entering = false;
                for (const turn of slot.turns) turn.leaving = false;
                this._foldLiveInterim(slot, current.id);
                if (!name || name === this.currentName) this.scroll();
            }, 260);
        },

        // ---------- collapse intermediate turns ----------------------
        // Group intermediate assistant + trace turns between a user msg
        // and the final assistant response into a collapsible heading.
        // Visualizations stay inline (trickle up, not contained).
        // Called on turn_complete (finishTurn) and loadHistory.
        _groupIntermediateTurns(slot, previousTurns = slot.turns) {
            const previous = Array.isArray(previousTurns) ? previousTurns : [];
            const previousOpen = new Map();
            const previousLiveOpen = new Map();
            for (const turn of previous) {
                if (turn.role === "collapse" && turn.groupKey) {
                    previousOpen.set(turn.groupKey, Boolean(turn.open));
                }
                if (turn.role === "collapse" && turn.live && turn.userId) {
                    previousLiveOpen.set(turn.userId, Boolean(turn.open));
                }
            }

            const flatten = (items) => {
                const result = [];
                for (const item of items) {
                    if (item.role === "collapse") result.push(...flatten(item.childTurns || []));
                    else result.push(item);
                }
                return result;
            };

            // `previousTurns` is UI-state input only. Hydration and paging
            // replace slot.turns before calling this helper, so grouping the
            // previous array silently discards the page that just arrived.
            const turns = flatten(slot.turns);
            const grouped = [];
            let index = 0;
            while (index < turns.length) {
                const userTurn = turns[index];
                if (userTurn.role !== "user") {
                    grouped.push(userTurn);
                    index += 1;
                    continue;
                }

                grouped.push(userTurn);
                let nextUserIndex = index + 1;
                while (nextUserIndex < turns.length && turns[nextUserIndex].role !== "user") {
                    nextUserIndex += 1;
                }

                const exchange = turns.slice(index + 1, nextUserIndex);
                let finalOffset = -1;
                for (let offset = exchange.length - 1; offset >= 0; offset -= 1) {
                    const candidate = exchange[offset];
                    if (candidate.role === "assistant" && !candidate.streaming) {
                        finalOffset = offset;
                        break;
                    }
                }

                if (finalOffset > 0) {
                    const finalResponse = exchange[finalOffset];
                    const groupKey = JSON.stringify([userTurn.text || "", finalResponse.text || ""]);
                    const openByKey = new Map(previousOpen);
                    // Compatibility with the former one-collapse-per-exchange
                    // shape: carry its disclosure state into every segment.
                    const defaultOpen = (
                        previousOpen.get(groupKey)
                        ?? previousLiveOpen.get(userTurn.id)
                        ?? false
                    );
                    this._pushCollapsedSegments(grouped, exchange.slice(0, finalOffset), {
                        idPrefix: `collapse-${userTurn.id}-${finalResponse.id}`,
                        groupKey,
                        userId: userTurn.id,
                        live: false,
                        openByKey,
                        defaultOpen,
                    });
                    grouped.push(finalResponse, ...exchange.slice(finalOffset + 1));
                } else {
                    grouped.push(...exchange);
                }
                index = nextUserIndex;
            }
            slot.turns = grouped;
        },
        _findById(slot, id) { return slot.turns.find((t) => t.id === id); },
        _lastByRole(slot, role) {
            for (let i = slot.turns.length - 1; i >= 0; i--) {
                if (slot.turns[i].role === role) return slot.turns[i];
            }
            return null;
        },

        // ---------- user + assistant turns ---------------------------

        addUser(text, name, attachments = []) {
            const slot = this._slot(name);
            const previous = this._lastTurn(slot);
            const incomingIds = (attachments || []).map(item => item.attachment_id).join(",");
            const previousIds = (previous && previous.attachments || []).map(item => item.attachment_id).join(",");
            if (previous && previous.role === "user" && previous.text === text && incomingIds === previousIds) return;
            this._closeTrace(slot);
            slot.turns.push({
                id: this._id("u"),
                role: "user",
                text,
                html: renderMarkdown(text),
                streaming: false,
                attachments: attachments || [],
            });
            if (!name || name === this.currentName) this.scroll(true);
        },
        startAssistant(turn_id, name) {
            const slot = this._slot(name);
            const id = turn_id || this._id("a");
            const existing = this._findById(slot, id);
            this._closeStreamingAssistants(slot, id);
            this._closeTrace(slot);
            if (existing && existing.role === "assistant") {
                existing.streaming = true;
            } else {
                slot.turns.push({
                    id,
                    role: "assistant",
                    text: "",
                    html: "",
                    streaming: true,
                });
            }
            if (!name || name === this.currentName) this.scroll();
        },
        // Round-44 F4: streaming deltas coalesce per slot. Every SSE token
        // used to run the O(history) turn lookup, mutate text, and schedule
        // a full re-Markdown of the whole accumulated answer — O(n) per
        // token and quadratic in answer length. Deltas now accumulate in
        // slot.pendingDelta and flush at most once per 32ms window, so the
        // per-token cost is O(1) amortized and re-render cost is bounded by
        // the flush cadence rather than the provider's token rate.
        flushPendingDelta(name) {
            const slot = this._slot(name);
            if (slot.deltaTimer) {
                clearTimeout(slot.deltaTimer);
                slot.deltaTimer = null;
            }
            const pending = slot.pendingDelta;
            if (!pending) return;
            slot.pendingDelta = null;
            let t = this._findById(slot, pending.turnId);
            if (!t || t.role !== "assistant") {
                t = this._lastByRole(slot, "assistant");
                if (!t || !t.streaming) {
                    this.startAssistant(pending.turnId, name);
                    t = this._lastByRole(slot, "assistant");
                }
            }
            if (!t) return;
            this._closeStreamingAssistants(slot, t.id);
            t.streaming = true;
            const startsSuccessor = !t.text;
            t.text += pending.text;
            if (startsSuccessor && t.text) this._beginAssistantHandoff(slot, t, name);
            if (this._renderRaf) cancelAnimationFrame(this._renderRaf);
            const turnRef = t;
            // Round-47 F13: renderMarkdown() over the ENTIRE accumulated
            // text on every 32ms flush is quadratic in response length and
            // blocks the main thread on long replies. While streaming, the
            // rAF only refreshes a cheap plain-text mirror (the template
            // shows .rawText for streaming turns); ONE full markdown render
            // happens at endAssistant().
            this._renderRaf = requestAnimationFrame(() => {
                turnRef.rawText = turnRef.text;
                this._renderRaf = 0;
            });
            if (!name || name === this.currentName) this.scroll();
        },
        appendDelta(turn_id, text, name) {
            if (!text) return;
            const slot = this._slot(name);
            const safeTurnId = turn_id || "__active__";
            const pending = slot.pendingDelta;
            if (pending && pending.turnId === safeTurnId) {
                pending.text += text;
            } else {
                if (pending) this.flushPendingDelta(name);
                slot.pendingDelta = { turnId: safeTurnId, text };
            }
            if (!slot.deltaTimer) {
                slot.deltaTimer = setTimeout(() => {
                    slot.deltaTimer = null;
                    this.flushPendingDelta(name);
                }, 32);
            }
        },
        endAssistant(turn_id, name) {
            const slot = this._slot(name);
            // Round-44 F4: flush coalesced deltas BEFORE finalizing so the
            // full streamed text is present when streaming is retired.
            this.flushPendingDelta(name);
            const t = this._findById(slot, turn_id) || this._lastByRole(slot, "assistant");
            if (!t) return;
            if (this._renderRaf) { cancelAnimationFrame(this._renderRaf); this._renderRaf = 0; }
            if (this._scrollRaf) { cancelAnimationFrame(this._scrollRaf); this._scrollRaf = 0; }
            if (this._highlightRaf) { cancelAnimationFrame(this._highlightRaf); this._highlightRaf = 0; }
            t.streaming = false;
            // Round-47 F13: the full markdown render happens exactly once
            // here (was: every 32ms flush during streaming — quadratic).
            // Clear the plain-text mirror so the template switches back to
            // the rendered HTML.
            t.rawText = null;
            t.html = renderMarkdown(t.text);
            if (!name || name === this.currentName) {
                queueMicrotask(() => requestAnimationFrame(() => {
                    enhanceRenderedTurns([t.id]);
                }));
                // Settle the view after the final reflow — but only if the
                // user is already following at the bottom (see scroll()).
                queueMicrotask(() => this.scroll());
            }
        },

        // ---------- trace events -------------------------------------

        addToolCall(toolName, args, name) {
            const slot = this._slot(name);
            this._closeStreamingAssistants(slot);
            const t = this._ensureTrace(slot);
            t.events.push({
                id: this._id("ev"),
                kind: "tool_call",
                name: toolName || "(unknown)",
                jsonHtml: renderJSON(args),
                at: Date.now(),
            });
            if (!name || name === this.currentName) this.scroll();
        },
        addToolResult(toolName, text, name, cacheKey) {
            const slot = this._slot(name);
            this._closeStreamingAssistants(slot);
            const t = this._ensureTrace(slot);
            t.events.push({
                id: this._id("ev"),
                kind: "tool_result",
                name: toolName || "",
                jsonHtml: renderJSON(text),
                rawText: typeof text === "string" ? text : null,
                cacheKey: cacheKey || "",
                at: Date.now(),
            });
            if (!name || name === this.currentName) this.scroll();
        },
        setLastToolResultCache(toolName, cacheKey, name) {
            const slot = this._slot(name);
            const t = this._ensureTrace(slot);
            // Find the last tool_result event matching toolName and attach cache_key.
            for (let i = t.events.length - 1; i >= 0; i--) {
                const ev = t.events[i];
                if (ev.kind === "tool_result" && (!toolName || ev.name === toolName)) {
                    ev.cacheKey = cacheKey;
                    break;
                }
            }
        },

        // --- Tool result cache popup ----------------------------------
        // Clicking a tool_result trace event that has a cacheKey opens a
        // popup showing the full cached content fetched on demand from the
        // backend cache endpoint. Keeps L5 context lean — the full result
        // is only loaded when the user explicitly clicks.
        cachePopup: { open: false, loading: false, toolName: "", cacheKey: "", result: null, error: "" },
        async openCachePopup(cacheKey, toolName) {
            if (!cacheKey) return;
            this.cachePopup = {
                open: true, loading: true, toolName: toolName || "",
                cacheKey, result: null, error: "",
            };
            try {
                const name = this.currentName || "";
                const resp = await fetch(`/api/sessions/${encodeURIComponent(name)}/cache/${encodeURIComponent(cacheKey)}`);
                if (!resp.ok) {
                    const detail = await resp.text();
                    this.cachePopup.error = `Failed (${resp.status}): ${detail}`;
                    this.cachePopup.loading = false;
                    return;
                }
                const data = await resp.json();
                this.cachePopup.result = data.result;
                this.cachePopup.loading = false;
            } catch (e) {
                this.cachePopup.error = String(e || "Network error");
                this.cachePopup.loading = false;
            }
        },
        closeCachePopup() {
            this.cachePopup.open = false;
            this.cachePopup.result = null;
            this.cachePopup.error = "";
        },
        addThinking(text, name) {
            if (!text) return;
            const slot = this._slot(name);
            this._closeStreamingAssistants(slot);
            const t = this._ensureTrace(slot);
            const last = t.events[t.events.length - 1];
            if (last && last.kind === "thinking") {
                last.text += text;
                return;
            }
            t.events.push({
                id: this._id("ev"),
                kind: "thinking",
                text: String(text),
                at: Date.now(),
            });
            if (!name || name === this.currentName) this.scroll();
        },
        addInfo(text, opts, name) {
            const slot = this._slot(name);
            if (opts && opts.standalone) {
                slot.turns.push({
                    id: this._id("i"),
                    role: "info",
                    text,
                    html: escapeHtml(text),
                });
                if (!name || name === this.currentName) this.scroll();
                return;
            }
            this._closeStreamingAssistants(slot);
            const t = this._ensureTrace(slot);
            t.events.push({
                id: this._id("ev"),
                kind: "info",
                text: String(text),
                at: Date.now(),
            });
            if (!name || name === this.currentName) this.scroll();
        },
        addError(text, name) {
            const slot = this._slot(name);
            this._closeTrace(slot);
            slot.turns.push({
                id: this._id("e"),
                role: "error",
                text,
                html: escapeHtml(text),
            });
            if (!name || name === this.currentName) this.scroll();
        },

        _visualizationTurn(artifact, name) {
            if (!artifact || artifact.kind !== "visualization" || !artifact.artifact_id) return null;
            const sessionName = name || this.currentName || "";
            const artifactId = String(artifact.artifact_id);
            const encodedSession = encodeURIComponent(sessionName);
            const encodedArtifact = encodeURIComponent(artifactId);
            const height = Math.max(180, Math.min(1200, Number(artifact.height) || 480));
            return {
                id: `viz-${artifactId}`,
                role: "visualization",
                artifact: {
                    ...artifact,
                    title: artifact.title || artifact.name || "Visualization",
                    view_url: artifact.view_url ||
                        `/api/sessions/${encodedSession}/artifacts/${encodedArtifact}/view`,
                    download_url: artifact.download_url ||
                        `/api/sessions/${encodedSession}/artifacts/${encodedArtifact}/download`,
                },
                height,
            };
        },
        addVisualization(artifact, name) {
            const slot = this._slot(name);
            const turn = this._visualizationTurn(artifact, name);
            if (!turn) return;
            const historyElement = (!name || name === this.currentName)
                ? document.querySelector(".chat-history")
                : null;
            const shouldFollow = historyElement ? this._atBottom(historyElement) : true;
            const existing = slot.turns.findIndex((item) =>
                item.role === "visualization" &&
                item.artifact &&
                item.artifact.artifact_id === turn.artifact.artifact_id
            );
            if (existing >= 0) {
                slot.turns[existing] = turn;
            } else {
                // MUCLI_VISUALIZATION_TIMELINE_V2: split a live assistant at the
                // artifact boundary so later deltas render after the card.
                let insertAt = slot.turns.length;
                for (let index = slot.turns.length - 1; index >= 0; index--) {
                    const candidate = slot.turns[index];
                    if (candidate.role === "assistant" && candidate.streaming) {
                        candidate.streaming = false;
                        candidate.html = renderMarkdown(candidate.text || "");
                        candidate.id = `${candidate.id}-segment-${turn.artifact.artifact_id}`;
                        insertAt = index + 1;
                        break;
                    }
                }
                slot.turns.splice(insertAt, 0, turn);
            }
            if (!name || name === this.currentName) this.scroll(shouldFollow);
        },

        addCommandResult(result, name) {
            const slot = this._slot(name);
            this._closeTrace(slot);
            const ok = result && result.ok;
            const cmd = (result && result.command) || "/command";
            const msg = (result && result.message) || "";
            const data = (result && result.data) || {};

            let body = "";
            let preformatted = false;
            if (data.stdout || data.stderr) {
                const parts = [];
                if (data.stdout) parts.push(data.stdout.replace(/\n+$/, ""));
                if (data.stderr) parts.push(data.stderr.replace(/\n+$/, ""));
                body = parts.join("\n");
                preformatted = true;
            } else if (data.sessions) {
                const active = data.active || "";
                body = data.sessions.map(s =>
                    (s === active ? "▸ " : "  ") + s
                ).join("\n");
                preformatted = true;
            } else if (data.folders || data.staged_files) {
                const lines = [];
                if (data.folders && data.folders.length) {
                    lines.push(...data.folders.map(f => "  " + f));
                }
                if (data.staged_files && data.staged_files.length) {
                    lines.push(...data.staged_files.map(f => "  " + f));
                }
                body = lines.length ? lines.join("\n") : msg;
                preformatted = lines.length > 0;
            } else if (data.tools) {
                body = data.tools.map(t =>
                    (t.enabled ? "  " : "✗ ") + t.name
                ).join("\n");
                preformatted = true;
            } else if (msg) {
                body = msg;
            }

            slot.turns.push({
                id: this._id("cmd"),
                role: "command",
                ok,
                command: cmd,
                message: msg,
                data,
                body,
                html: body ? renderMarkdown(body) : "",
                isPreformatted: preformatted,
            });
            if (!name || name === this.currentName) this.scroll(true);
        },

        addPromptResolved(record, name) {
            const slot = this._slot(name);
            const cancelled = !!(record && record.cancelled);
            const answer = record && record.answer ? String(record.answer) : "";
            const title = record && record.title ? String(record.title) : "";
            slot.turns.push({
                id: this._id("pr"),
                role: "prompt_resolved",
                cancelled,
                shape: (record && record.shape) || "",
                toolName: (record && record.toolName) || "",
                title,
                titleHtml: renderMarkdownInline(title),
                answer,
                answerHtml: renderMarkdownInline(answer),
                at: Date.now(),
            });
            if (!name || name === this.currentName) this.scroll();
        },

        // ---------- sub-agent status panel ----------------------------
        //
        // One `subagent_panel` turn per session slot, holding an `agents`
        // list upserted by `task_id`. While any agent is running the panel
        // is expanded (live tool / context / token readout); when all are
        // done it collapses to a one-line summary, expandable on click.

        _newAgentRow(task_id) {
            return {
                task_id,
                task: "", title: "", depth: 1, model: "", specialist_key: "", status: "running",
                tool_count: 0, last_tool: null,
                stuck: false, stall: false, repeat_count: 0,
                elapsed: 0, context_pct: 0, iter: 0, max_iter: 0, tokens_in: 0,
                summary: "", kill_reason: null, error: null,
                actions: [], details_open: false,
                observed_at: Date.now(),
            };
        },
        _historySubagentPanel(part) {
            if (!part || !Array.isArray(part.agents) || !part.agents.length) return null;
            const panel = {
                id: `sap-history-${String(part.batch_id || part.agents[0]?.task_id || "batch")}`,
                role: "subagent_panel",
                batch_id: String(part.batch_id || ""),
                agents: [],
                running: false,
                open: false,
                ephemeral: false,
                durable: true,
                dismissTimer: null,
            };
            for (const value of part.agents) {
                if (!value || !value.task_id) continue;
                const row = this._newAgentRow(String(value.task_id));
                const fields = { ...value };
                delete fields.actions;
                this._mergeDefined(row, fields);
                this._mergeSubagentActions(row, { actions: value.actions || [] });
                panel.agents.push(row);
            }
            panel.running = this._panelRunning(panel);
            panel.open = panel.running;
            return panel.agents.length ? panel : null;
        },
        // Each delegation batch owns one stable timeline card. Completed cards
        // remain at their original history location and can be reopened later.
        _findPanel(slot, batchId = "") {
            for (let i = slot.turns.length - 1; i >= 0; i--) {
                const turn = slot.turns[i];
                if (turn.role === "subagent_panel" && (!batchId || turn.batch_id === batchId)) return turn;
            }
            return null;
        },
        _ensurePanel(slot, batchId = "") {
            let p = this._findPanel(slot, batchId);
            if (!p) {
                p = {
                    id: this._id("sap"),
                    role: "subagent_panel",
                    batch_id: batchId || "",
                    agents: [],
                    running: false,
                    open: true,
                    ephemeral: false,
                    dismissTimer: null,
                };
                slot.turns.push(p);
            }
            return p;
        },
        _panelRunning(p) {
            return p.agents.some(a => a.status === "running" || a.status === "stuck" || a.status === "stall");
        },
        _cancelPanelDismiss(p) {
            if (!p) return;
            if (p.dismissTimer) clearTimeout(p.dismissTimer);
            p.dismissTimer = null;
        },
        _schedulePanelDismiss(p, delay = 6000) {
            if (!p || this._panelRunning(p)) return;
            if (p.agents.some(a => a.details_open)) return;
            this._cancelPanelDismiss(p);
            p.running = false;
            p.open = true;
            p.dismissTimer = setTimeout(() => {
                p.dismissTimer = null;
                if (!this._panelRunning(p) && !p.agents.some(a => a.details_open)) {
                    p.open = false;
                }
            }, delay);
        },
        _mergeDefined(target, patch) {
            for (const [key, value] of Object.entries(patch || {})) {
                if (value !== undefined) target[key] = value;
            }
            target.observed_at = Date.now();
            return target;
        },
        _mergeSubagentActions(row, patch) {
            const incoming = [];
            if (Array.isArray(patch && patch.actions)) incoming.push(...patch.actions);
            if (patch && patch.action && typeof patch.action === "object") incoming.push(patch.action);
            for (const raw of incoming) {
                const seq = Number(raw && raw.seq) || 0;
                const tool = String((raw && raw.tool) || "tool");
                if (!seq) continue;
                let action = row.actions.find(item => Number(item.seq) === seq);
                if (!action) {
                    action = { seq, tool, detail: "", status: "running", elapsed: 0, at: Date.now() / 1000 };
                    row.actions.push(action);
                }
                this._mergeDefined(action, {
                    tool,
                    detail: raw.detail !== undefined ? String(raw.detail || "") : undefined,
                    status: raw.status !== undefined ? String(raw.status || "running") : undefined,
                    elapsed: raw.elapsed !== undefined ? Number(raw.elapsed) || 0 : undefined,
                    at: raw.at !== undefined ? Number(raw.at) || action.at : undefined,
                });
            }
            row.actions.sort((left, right) => Number(left.seq) - Number(right.seq));
            if (row.actions.length > 100) row.actions.splice(0, row.actions.length - 100);
        },
        toggleSubagentDetails(panel, agent) {
            if (!panel || !agent) return;
            agent.details_open = !agent.details_open;
            const slot = this._slot(this.currentName);
            if (agent.details_open) {
                panel.open = true;
                this._cancelPanelDismiss(panel);
            } else if (!this._panelRunning(panel)) {
                this._schedulePanelDismiss(panel);
            }
        },
        upsertSubagent(name, agent) {
            const tid = agent && agent.task_id;
            if (!tid) return;
            const slot = this._slot(name);
            const batchId = agent.batch_id || "";
            const terminal = ["done", "killed", "error"].includes(agent.status);
            let p = this._findPanel(slot, batchId);
            if (!p && terminal) return;
            if (!p) p = this._ensurePanel(slot, batchId);
            let row = p.agents.find(a => a.task_id === tid);
            if (!row) {
                row = this._newAgentRow(tid);
                p.agents.push(row);
            }
            this._mergeSubagentActions(row, agent);
            const fields = { ...agent };
            delete fields.action;
            delete fields.actions;
            this._mergeDefined(row, fields);
            p.running = this._panelRunning(p);
            if (p.running) {
                p.open = true;
                this._cancelPanelDismiss(p);
            } else {
                this._schedulePanelDismiss(p);
            }
            if (!name || name === this.currentName) this.scroll();
        },
        replaceSubagentSnapshot(name, children, batchId = "") {
            const slot = this._slot(name);
            const active = (children || []).filter(child => child && child.task_id && (child.status === "running" || child.status === "stuck" || child.status === "stall"));
            let p = this._findPanel(slot, batchId);
            if (!active.length) {
                if (p) this._schedulePanelDismiss(p);
                return;
            }
            if (!p) p = this._ensurePanel(slot, batchId);
            this._cancelPanelDismiss(p);
            for (const c of active) {
                let row = p.agents.find(a => a.task_id === c.task_id);
                if (!row) {
                    row = this._newAgentRow(c.task_id);
                    p.agents.push(row);
                }
                this._mergeDefined(row, {
                    task: c.task || "",
                    title: c.title || "",
                    depth: c.depth || 1,
                    model: c.model || "",
                    specialist_key: c.specialist_key || "",
                    batch_id: c.batch_id || batchId,
                    status: c.status || "running",
                    tool_count: c.tool_count ?? 0,
                    last_tool: c.last_tool ?? null,
                    stuck: !!c.stuck,
                    stall: !!c.stall,
                    repeat_count: c.consecutive_repeats ?? 0,
                    elapsed: c.elapsed ?? 0,
                    context_pct: c.context_pct ?? 0,
                    iter: c.iter ?? 0,
                    max_iter: c.max_iter ?? 0,
                    tokens_in: c.tokens_in ?? 0,
                    actions: Array.isArray(c.actions) ? c.actions : [],
                });
                this._mergeSubagentActions(row, c);
            }
            p.running = true;
            p.open = true;
            if (!name || name === this.currentName) this.scroll();
        },
        finishSubagents(name) {
            const slot = this._slot(name);
            const p = this._lastByRole(slot, "subagent_panel");
            if (!p) return;
            p.running = this._panelRunning(p);
            if (!p.running) this._schedulePanelDismiss(p);
        },

        finishTurn(name) {
            const slot = this._slot(name);
            // Round-44 F4: never drop pending coalesced deltas on turn end.
            this.flushPendingDelta(name);
            if (slot.handoffTimer) {
                clearTimeout(slot.handoffTimer);
                slot.handoffTimer = null;
            }
            for (const turn of slot.turns) {
                turn.entering = false;
                turn.leaving = false;
            }
            this._closeStreamingAssistants(slot);
            this._closeTrace(slot);
            for (const turn of slot.turns) {
                if (turn.role === "subagent_panel" && !this._panelRunning(turn)) {
                    this._schedulePanelDismiss(turn);
                }
            }
            // Group intermediate assistant + trace turns into a collapsible
            // heading so the conversation flow reads:
            //   User → Q, ... <collapsible N responses>, Agent → Final
            this._groupIntermediateTurns(slot);
            // If a history reload was deferred while this turn was streaming
            // (SSE reconnect / session_updated from another process), the
            // turn is now saved server-side — flush it so the view shows the
            // authoritative, complete history.
            if (slot.pendingReload) {
                slot.pendingReload = false;
                this.loadHistory(name, { force: true });
            }
        },

        isBusy(name) {
            return !!(this.chats[name] && this.chats[name].busy);
        },

        // ---------- send + history -----------------------------------

        // Are we parked at (or near) the bottom of the chat feed?
        // Terminal-style autoscroll: only follow new output when the user
        // is already at the bottom. If they scrolled up to read, leave
        // them there — don't yank them back down on every streaming delta.
        // Threshold is generous so minor layout reflows near the tail
        // (markdown re-render, code block resize) don't disable follow.
        followOutput: true,
        _atBottom(el, threshold = 120) {
            return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
        },
        onScroll(event) {
            const el = event && event.currentTarget;
            if (!el) return;
            this.followOutput = this._atBottom(el);
            this._updateActiveStage(el);
            const slot = this._slot();
            // Programmatic stage/page jumps can land at an edge. Do not turn
            // that synthetic scroll event into a second pagination request.
            if (Date.now() < slot.suppressEdgeLoadUntil) return;
            // Round-44 F3: auto-load older turns when the user scrolls near
            // the top. Guarded: one request at a time, only when the cursor
            // says more pages exist. Preserves scroll position — prepended
            // turns push content down, so restore the previous offset after
            // the DOM grows (same double-RAF settle as loadHistory).
            if (el.scrollTop < 120) {
                if (
                    slot.hasMoreTurns
                    && !slot.olderLoading
                    && !slot.busy
                    && slot.oldestLoadedIndex !== null
                    && slot.oldestLoadedIndex > 0
                ) {
                    this.loadOlder();
                }
            }
        },

        _updateActiveStage(el) {
            if (this._stageRaf) return;
            this._stageRaf = requestAnimationFrame(() => {
                this._stageRaf = 0;
                if (!el || !el.isConnected) return;
                // offsetTop is relative to each element's offsetParent, which
                // changes across responsive/product layouts. Compare viewport
                // rectangles instead: the active stage is the newest prompt
                // above the reading line inside the actual scroll viewport.
                const containerRect = el.getBoundingClientRect();
                const readingLine = containerRect.top + Math.min(180, containerRect.height * 0.32);
                let active = null;
                const stages = this.stages;
                const stageIds = new Set(stages.map(stage => stage.id));
                for (const node of el.querySelectorAll('.turn-wrap[data-turn-id]')) {
                    const id = node.dataset.turnId;
                    if (!stageIds.has(id)) continue;
                    const box = this._turnBox(node);
                    if (box.getBoundingClientRect().top <= readingLine) active = id;
                    else break;
                }
                this.activeStageId = active || (stages[0] && stages[0].id) || null;
                queueMicrotask(() => {
                    const selected = document.querySelector('.conversation-stage-link.active');
                    const list = selected?.closest('.conversation-stage-list');
                    if (!selected || !list) return;
                    const selectedTop = selected.offsetTop;
                    const selectedBottom = selectedTop + selected.offsetHeight;
                    if (selectedTop < list.scrollTop) list.scrollTop = selectedTop;
                    else if (selectedBottom > list.scrollTop + list.clientHeight) {
                        list.scrollTop = selectedBottom - list.clientHeight;
                    }
                });
            });
        },

        _turnBox(wrapper) {
            if (!wrapper) return wrapper;
            // .turn-wrap intentionally uses display:contents, so its own
            // rectangle is always 0×0. Geometry and scroll operations must
            // target the concrete element cloned by Alpine's x-if template.
            for (const child of wrapper.children) {
                if (child.tagName === 'TEMPLATE') continue;
                const rect = child.getBoundingClientRect();
                if (rect.width > 0 || rect.height > 0) return child;
            }
            return wrapper;
        },

        _setHistoryScrollTop(el, top) {
            if (!el) return;
            const slot = this._slot();
            slot.suppressEdgeLoadUntil = Date.now() + 180;
            // The base stylesheet historically enabled smooth scrolling for
            // every scrollTop write. Cursor restoration and page navigation
            // must be immediate or subsequent renders interrupt the motion.
            const previousBehavior = el.style.scrollBehavior;
            el.style.scrollBehavior = 'auto';
            el.scrollTop = Math.max(0, top);
            el.style.scrollBehavior = previousBehavior;
        },

        _afterHistoryRender(callback) {
            return new Promise(resolve => {
                queueMicrotask(() => requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        try {
                            callback();
                        } finally {
                            resolve();
                        }
                    });
                }));
            });
        },

        _captureHistoryAnchor(el) {
            if (!el) return null;
            const containerTop = el.getBoundingClientRect().top;
            const wrapper = Array.from(el.querySelectorAll('.turn-wrap[data-turn-id]'))
                .find(item => this._turnBox(item).getBoundingClientRect().bottom >= containerTop);
            const node = this._turnBox(wrapper);
            return {
                id: wrapper?.dataset.turnId || null,
                top: node ? node.getBoundingClientRect().top : containerTop,
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
            };
        },

        _restoreHistoryAnchor(el, anchor, { accountForPrependedHeight = false } = {}) {
            if (!el || !anchor) return;
            const wrapper = anchor.id
                ? Array.from(el.querySelectorAll('.turn-wrap[data-turn-id]'))
                    .find(item => item.dataset.turnId === anchor.id)
                : null;
            const node = this._turnBox(wrapper);
            if (node) {
                // Use the browser's CURRENT scrollTop. Chromium may have
                // partially applied native overflow anchoring before this
                // callback; adding the visual delta is correct either way.
                this._setHistoryScrollTop(
                    el,
                    el.scrollTop + node.getBoundingClientRect().top - anchor.top,
                );
                return;
            }
            const heightDelta = accountForPrependedHeight
                ? Math.max(0, el.scrollHeight - anchor.scrollHeight)
                : 0;
            this._setHistoryScrollTop(
                el,
                Math.min(
                    anchor.scrollTop + heightDelta,
                    Math.max(0, el.scrollHeight - el.clientHeight),
                ),
            );
        },

        _scrollToRenderedTurn(el, id) {
            if (!el || !id) return false;
            const wrapper = Array.from(el.querySelectorAll('.turn-wrap[data-turn-id]'))
                .find(item => item.dataset.turnId === id);
            if (!wrapper) return false;
            const node = this._turnBox(wrapper);
            const slot = this._slot();
            slot.suppressEdgeLoadUntil = Date.now() + 180;
            const previousBehavior = el.style.scrollBehavior;
            el.style.scrollBehavior = 'auto';
            // Let the browser resolve the actual nested scroll container,
            // then apply the timeline's reading offset. This is more robust
            // than reconstructing scrollTop from viewport rectangles across
            // the responsive/product layouts.
            node.scrollIntoView({ block: 'start', inline: 'nearest', behavior: 'auto' });
            el.scrollTop = Math.max(0, el.scrollTop - 28);
            el.style.scrollBehavior = previousBehavior;
            return true;
        },

        jumpToStage(id) {
            const el = document.querySelector('.chat-history');
            if (!this._scrollToRenderedTurn(el, id)) return;
            this.followOutput = false;
            this.activeStageId = id;
            this._updateActiveStage(el);
            // Alpine/MathJax may finish a pending layout pass in the same
            // frame. Re-assert the selected checkpoint once after layout.
            requestAnimationFrame(() => {
                if (this.activeStageId === id) this._scrollToRenderedTurn(el, id);
            });
        },

        _flattenTimeline(items) {
            const flattened = [];
            for (const item of items || []) {
                if (item && item.role === "collapse") {
                    flattened.push(...this._flattenTimeline(item.childTurns || []));
                } else if (item) {
                    flattened.push(item);
                }
            }
            return flattened;
        },

        _timelineRange(items, startIndex, endIndex) {
            return this._flattenTimeline(items).filter(item => (
                !Number.isInteger(item.historyIndex)
                || (item.historyIndex >= startIndex && item.historyIndex < endIndex)
            ));
        },

        async _fetchCheckpointBatch(target, beforeIndex = null) {
            let cursor = Number.isInteger(beforeIndex) ? beforeIndex : null;
            let firstResponse = null;
            let lastResponse = null;
            let turns = [];
            const checkpointIndexes = new Set();

            for (
                let page = 0;
                page < WEB_HISTORY_CHECKPOINT_SCAN_PAGES
                    && checkpointIndexes.size < WEB_HISTORY_CHECKPOINT_BATCH;
                page += 1
            ) {
                const remaining = WEB_HISTORY_CHECKPOINT_BATCH - checkpointIndexes.size;
                const params = new URLSearchParams({
                    limit_turns: String(WEB_HISTORY_PAGE_TURNS),
                    checkpoint_count: String(remaining),
                });
                if (cursor !== null) params.set("before_index", String(cursor));
                if (target) params.set("session_name", target);
                const response = await fetch(`/api/sessions/current/history?${params}`);
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || `history request failed (${response.status})`);
                }
                if (!firstResponse) firstResponse = data;
                lastResponse = data;
                const pageTurns = Array.isArray(data.turns) ? data.turns : [];
                turns = [...pageTurns, ...turns];
                for (const turn of pageTurns) {
                    if (turn.role === "user" && Number.isInteger(turn.index)) {
                        checkpointIndexes.add(turn.index);
                    }
                }
                const nextCursor = data.start_index ?? 0;
                if (!pageTurns.length || nextCursor <= 0 || nextCursor === cursor) {
                    cursor = nextCursor;
                    break;
                }
                cursor = nextCursor;
            }

            const base = firstResponse || lastResponse || {};
            return {
                ...base,
                turns,
                start_index: lastResponse?.start_index ?? base.start_index ?? 0,
                window_end: firstResponse?.window_end ?? base.window_end ?? 0,
            };
        },

        async _historyTurnsToTimeline(historyTurns, sessionKey) {
            const rebuiltTurns = [];
            let historyChunkIndex = 0;
            for (const turn of historyTurns || []) {
                // Markdown/JSON conversion is CPU-heavy. Yield between small
                // batches so navigation, input, interrupts, and SSE stay live.
                if (historyChunkIndex > 0 && historyChunkIndex % 8 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
                historyChunkIndex += 1;
                let traceForTurn = null;
                let partIndex = 0;
                const ensureHistoryTrace = () => {
                    if (traceForTurn) return traceForTurn;
                    traceForTurn = {
                        id: `h-tr-${turn.index}-${partIndex}`,
                        role: "trace",
                        events: [],
                        open: false,
                        running: false,
                        startedAt: 0,
                        elapsed: null,
                        historyIndex: turn.index,
                    };
                    rebuiltTurns.push(traceForTurn);
                    return traceForTurn;
                };

                for (const part of turn.parts || []) {
                    const stablePartIndex = partIndex++;
                    if (
                        part.type === "text"
                        && (turn.role === "user" || turn.role === "assistant")
                    ) {
                        traceForTurn = null;
                        rebuiltTurns.push({
                            id: `h-${turn.index}-${stablePartIndex}`,
                            role: turn.role,
                            text: part.text,
                            html: renderMarkdown(part.text),
                            streaming: false,
                            attachments: [],
                            historyIndex: turn.index,
                        });
                    } else if (part.type === "attachment" && turn.role === "user") {
                        let targetMessage = null;
                        for (let i = rebuiltTurns.length - 1; i >= 0; i -= 1) {
                            const candidate = rebuiltTurns[i];
                            if (candidate.role === "user" && candidate.historyIndex === turn.index) {
                                targetMessage = candidate;
                                break;
                            }
                        }
                        if (!targetMessage) {
                            targetMessage = {
                                id: `h-${turn.index}-${stablePartIndex}`,
                                role: "user",
                                text: "",
                                html: "",
                                streaming: false,
                                attachments: [],
                                historyIndex: turn.index,
                            };
                            rebuiltTurns.push(targetMessage);
                        }
                        if (part.attachment && part.attachment.attachment_id) {
                            targetMessage.attachments.push(part.attachment);
                        }
                    } else if (part.type === "thinking") {
                        ensureHistoryTrace().events.push({
                            id: `h-ev-${turn.index}-${stablePartIndex}`,
                            kind: "thinking",
                            text: String(part.text || ""),
                            at: 0,
                        });
                    } else if (part.type === "tool_call") {
                        ensureHistoryTrace().events.push({
                            id: `h-ev-${turn.index}-${stablePartIndex}`,
                            kind: "tool_call",
                            name: part.tool_name || "(unknown)",
                            jsonHtml: renderJSON(part.tool_args),
                            at: 0,
                        });
                    } else if (part.type === "visualization") {
                        traceForTurn = null;
                        const visualization = this._visualizationTurn(part.artifact, sessionKey);
                        if (visualization && !rebuiltTurns.some(item =>
                            item.role === "visualization"
                            && item.artifact?.artifact_id === visualization.artifact.artifact_id
                        )) {
                            rebuiltTurns.push({ ...visualization, historyIndex: turn.index });
                        }
                    } else if (part.type === "tool_result") {
                        ensureHistoryTrace().events.push({
                            id: `h-ev-${turn.index}-${stablePartIndex}`,
                            kind: "tool_result",
                            name: part.tool_name || "",
                            jsonHtml: renderJSON(part.preview),
                            rawText: typeof part.preview === "string" ? part.preview : null,
                            at: 0,
                        });
                        const visualization = this._visualizationTurn(part.artifact, sessionKey);
                        if (visualization && !rebuiltTurns.some(item =>
                            item.role === "visualization"
                            && item.artifact?.artifact_id === visualization.artifact.artifact_id
                        )) {
                            rebuiltTurns.push({ ...visualization, historyIndex: turn.index });
                        }
                    } else if (part.type === "subagent_panel") {
                        traceForTurn = null;
                        const panel = this._historySubagentPanel(part);
                        if (panel && !rebuiltTurns.some(item =>
                            item.role === "subagent_panel" && item.batch_id === panel.batch_id
                        )) {
                            rebuiltTurns.push({ ...panel, historyIndex: turn.index });
                        }
                    }
                }
            }
            return rebuiltTurns;
        },

        // Fetch enough bounded raw pages to discover five older user prompts,
        // then prepend them as one checkpoint batch. A long tool-heavy agent
        // run can span many raw messages; stopping after one raw page made the
        // first ^ click appear to do nothing.
        async loadOlder(name, { navigate = false } = {}) {
            const target = name || this.currentName;
            const slot = this._slot(target);
            if (
                slot.olderLoading
                || !slot.hasMoreTurns
                || slot.busy
                || slot.oldestLoadedIndex === null
                || slot.oldestLoadedIndex <= 0
            ) return;
            slot.olderLoading = true;
            // Round-45 F2: capture the reload generation AND the cursor at
            // request time. A loadHistory() that completes while this page
            // is in flight bumps historyFetchGen and resets the timeline to
            // the newest tail — a late older-page response must not commit
            // against that replaced timeline or mutate the reset cursors.
            const fetchGen = slot.historyFetchGen || 0;
            const beforeIndex = slot.oldestLoadedIndex;
            try {
                const data = await this._fetchCheckpointBatch(target, beforeIndex);
                const skey = target || data.name || null;
                // Session changed while in flight — drop the page.
                if ((skey || "__default__") !== (target || this.currentName || "__default__")) return;
                const dst = this._slot(skey);
                if (!dst.historyHydrated) return;
                // Round-45 F2: a full reload started/finished while this
                // page was in flight — the timeline it would prepend to no
                // longer exists. Drop the stale page wholesale.
                if ((dst.historyFetchGen || 0) !== fetchGen) return;
                // The cursor moved (another older page landed first) — this
                // response is stale; drop it.
                if (dst.oldestLoadedIndex !== beforeIndex) return;
                const el = document.querySelector(".chat-history");
                const anchor = this._captureHistoryAnchor(el);
                const olderTurns = await this._historyTurnsToTimeline(data.turns, skey);
                // Cooperative yields allow a reload or another page request
                // to win while this page is being converted. Revalidate both
                // guards before mutating cursors or the rendered timeline.
                if ((dst.historyFetchGen || 0) !== fetchGen) return;
                if (dst.oldestLoadedIndex !== beforeIndex) return;
                const previousTurns = dst.turns;
                const newStart = data.start_index ?? beforeIndex;
                const combinedEnd = dst.windowEndIndex ?? beforeIndex;
                const merged = [...olderTurns, ...this._flattenTimeline(previousTurns)];
                const navigationTarget = [...olderTurns].reverse().find(
                    turn => turn.role === 'user',
                );
                // Keep every explicitly loaded checkpoint available so every
                // dot remains a real backward/forward navigation target.
                dst.turns = this._timelineRange(merged, newStart, combinedEnd);
                this._groupIntermediateTurns(dst, previousTurns);
                dst.oldestLoadedIndex = newStart;
                dst.windowEndIndex = combinedEnd;
                dst.totalHistoryTurns = data.total_turns ?? dst.totalHistoryTurns;
                dst.hasMoreTurns = newStart > 0;
                if (!target || target === this.currentName) {
                    await this._afterHistoryRender(() => {
                        if ((dst.historyFetchGen || 0) !== fetchGen) return;
                        if (dst.oldestLoadedIndex !== newStart) return;
                        const historyEl = document.querySelector(".chat-history");
                        if (!historyEl) return;
                        if (!navigate || !this._scrollToRenderedTurn(
                            historyEl,
                            navigationTarget?.id,
                        )) {
                            this._restoreHistoryAnchor(historyEl, anchor, {
                                accountForPrependedHeight: true,
                            });
                        }
                        this._updateActiveStage(historyEl);
                        enhanceHistoryRange(newStart, beforeIndex);
                    });
                }
            } catch (err) {
                console.error("history-older", err);
            } finally {
                slot.olderLoading = false;
            }
        },
        isSlashCommand(text) {
            return /^\/[A-Za-z][\w-]*/.test((text || "").trim());
        },
        canSend(text) {
            const trimmed = (text || "").trim();
            const hasAttachments = Alpine.store("attachments").selectedIds(this.currentName).length > 0;
            if (!trimmed && !hasAttachments) return false;
            return !this.busy || this.isSlashCommand(trimmed);
        },
        scroll(force = false) {
            if (this._scrollRaf) return;
            this._scrollRaf = requestAnimationFrame(() => {
                this._scrollRaf = 0;
                const el = document.querySelector(".chat-history");
                if (!el) return;
                // followOutput is the persistent scroll-lock state, updated
                // by onScroll. force bypasses it (send, session switch, etc.).
                if (!force && !this.followOutput) return;
                el.scrollTop = el.scrollHeight;
            });
        },
        async send(text) {
            text = (text || "").trim();
            const name = this.currentName;
            const attachmentStore = Alpine.store("attachments");
            const selected = attachmentStore.selectedItems(name);
            if (!text && selected.length === 0) return;
            if (!text) text = "Please review the attached document(s).";
            const slot = this._slot(name);
            const isCommand = this.isSlashCommand(text);
            if (isCommand && selected.length) {
                this.addError("Attachments cannot be sent with slash commands.", name);
                return false;
            }
            if (slot.busy && !isCommand) return false;
            this.followOutput = true;   // user sent → re-engage scroll-lock
            this.addUser(text, name, selected);
            // A command can be submitted alongside an active turn. Do not
            // toggle the slot's existing busy state; the server handles the
            // command path independently of normal turn submission.
            if (!isCommand) slot.busy = true;
            try {
                const resp = await fetch("/api/chat/send", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        text,
                        session_name: name,
                        attachment_ids: selected.map(item => item.attachment_id),
                    }),
                });
                if (resp.status === 409) {
                    this.addError("A turn is already in flight.", name);
                    if (!isCommand) slot.busy = false;
                } else if (resp.ok) {
                    if (!isCommand) attachmentStore.clearSelected(name);
                } else if (!resp.ok) {
                    const data = await resp.json().catch(() => ({}));
                    this.addError(data.detail || `send failed (${resp.status})`, name);
                    if (!isCommand) slot.busy = false;
                }
            } catch (err) {
                this.addError(`Network error: ${err}`, name);
                if (!isCommand) slot.busy = false;
            }
            return true;
        },
        async interrupt() {
            const name = this.currentName;
            try {
                const r = await fetch("/api/chat/interrupt", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_name: name }),
                });
                const d = await r.json();
                if (d.ok) {
                    Alpine.store("toast").show("Interrupted — type /continue to resume or enter a new prompt", "info");
                }
            } catch (e) {
                console.error("interrupt", e);
            }
        },
        async loadHistory(name, { force = false } = {}) {
            const target = name || this.currentName;
            // Resolve the slot for the busy check. _slot() creating it is
            // harmless (it just holds the busy/pendingReload flags).
            const slot = this._slot(target);
            // Defer mid-stream reloads: an in-flight assistant turn isn't on
            // the server yet, so `slot.turns = []` + rebuild would drop the
            // streamed text and leave only the tail in a fresh turn. Wait
            // for the turn to complete, then flush via finishTurn.
            if (!force && slot.busy) {
                slot.pendingReload = true;
                return;
            }
            // Round-16 F21: per-slot fetch generation. Reconnect, SSE
            // session_updated, and panel refreshes can each start a history
            // fetch; network reordering can then let a SLOWER older
            // response commit over a newer reload and regress the visible
            // transcript. Each new request bumps the counter; a response
            // whose generation is no longer current is dropped.
            slot.historyFetchGen = (slot.historyFetchGen || 0) + 1;
            const fetchGen = slot.historyFetchGen;
            slot.historyLoading = true;
            let spinnerSlot = slot;
            try {
                const data = await this._fetchCheckpointBatch(target);
                // Round-16 F21: a newer request for this slot started while
                // we were in flight — drop this stale response wholesale.
                if (slot.historyFetchGen !== fetchGen) return;
                // At boot, target may be null (currentName is unset until
                // sessions.load() runs). Re-key by the server's returned
                // name so the history lands where the proxy getters will
                // look once sessions.load syncs currentName.
                const skey = target || data.name || null;
                const dst = this._slot(skey);
                if (dst !== slot) {
                    dst.historyLoading = true;
                    spinnerSlot = dst;
                }
                if (!this.currentName && data.name) this.currentName = data.name;
                // Build the page outside Alpine reactivity and replace it once.
                const rebuiltTurns = await this._historyTurnsToTimeline(data.turns, skey);
                // The cooperative yields above permit a newer refresh to
                // start. Never let this older conversion overwrite it.
                if (slot.historyFetchGen !== fetchGen) return;
                dst.pendingReload = false;
                // Round-44 F3: a full reload holds the newest tail again.
                dst.oldestLoadedIndex = data.start_index ?? null;
                dst.windowEndIndex = data.window_end ?? null;
                dst.totalHistoryTurns = data.total_turns ?? 0;
                dst.hasMoreTurns = (data.start_index ?? 0) > 0;
                const liveHistoryElement = (!name || name === this.currentName)
                    ? document.querySelector(".chat-history")
                    : null;
                const shouldFollowHistory = !dst.historyHydrated
                    || (liveHistoryElement ? this._atBottom(liveHistoryElement) : true);
                const anchor = this._captureHistoryAnchor(liveHistoryElement);
                const previousTurns = dst.turns;
                dst.turns = rebuiltTurns;
                this._groupIntermediateTurns(dst, previousTurns);
                dst.historyHydrated = true;
                if (!name || name === this.currentName) {
                    await this._afterHistoryRender(() => {
                        if (slot.historyFetchGen !== fetchGen) return;
                        const el = document.querySelector(".chat-history");
                        if (!el) return;
                        if (shouldFollowHistory) {
                            this._setHistoryScrollTop(el, el.scrollHeight);
                        } else {
                            this._restoreHistoryAnchor(el, anchor);
                        }
                        this._updateActiveStage(el);
                        enhanceHistoryRange(
                            data.start_index ?? 0,
                            data.window_end ?? data.total_turns ?? 0,
                        );
                    });
                }
            } catch (err) {
                console.error("history", err);
            } finally {
                // A stale request must not hide the spinner owned by a newer
                // refresh for this session.
                if (slot.historyFetchGen === fetchGen) slot.historyLoading = false;
                if (spinnerSlot !== slot && slot.historyFetchGen === fetchGen) {
                    spinnerSlot.historyLoading = false;
                }
            }
        },
    });


    Alpine.store("attachments", {
        itemsBySession: {},
        selectedBySession: {},
        loading: false,
        uploading: false,
        _name(name) { return name || Alpine.store("chat").currentName || ""; },
        items(name) { return this.itemsBySession[this._name(name)] || []; },
        selectedIds(name) { return this.selectedBySession[this._name(name)] || []; },
        selectedItems(name) {
            const ids = new Set(this.selectedIds(name));
            return this.items(name).filter(item => ids.has(item.attachment_id));
        },
        async load(name) {
            const sessionName = this._name(name);
            if (!sessionName) return;
            this.loading = true;
            try {
                const r = await fetch(`/api/sessions/${encodeURIComponent(sessionName)}/attachments`, { cache: "no-store" });
                const data = await r.json().catch(() => ({}));
                if (!r.ok) throw new Error(data.detail || `attachments failed (${r.status})`);
                this.itemsBySession[sessionName] = data.attachments || [];
                const valid = new Set(this.itemsBySession[sessionName].map(item => item.attachment_id));
                this.selectedBySession[sessionName] = this.selectedIds(sessionName).filter(id => valid.has(id));
            } catch (error) {
                Alpine.store("toast").show(String(error), "error");
            } finally {
                this.loading = false;
            }
        },
        toggle(item, name) {
            const sessionName = this._name(name);
            const ids = [...this.selectedIds(sessionName)];
            const index = ids.indexOf(item.attachment_id);
            if (index >= 0) ids.splice(index, 1); else ids.push(item.attachment_id);
            this.selectedBySession[sessionName] = ids;
        },
        removeSelected(attachmentId, name) {
            const sessionName = this._name(name);
            this.selectedBySession[sessionName] = this.selectedIds(sessionName).filter(id => id !== attachmentId);
        },
        clearSelected(name) { this.selectedBySession[this._name(name)] = []; },
        isSelected(item, name) { return this.selectedIds(name).includes(item.attachment_id); },
        async uploadFiles(fileList, name) {
            const sessionName = this._name(name);
            const files = Array.from(fileList || []);
            if (!sessionName || !files.length) return;
            this.uploading = true;
            try {
                const selected = new Set(this.selectedIds(sessionName));
                for (const file of files) {
                    const form = new FormData();
                    form.append("file", file, file.name);
                    const r = await fetch(`/api/sessions/${encodeURIComponent(sessionName)}/attachments`, { method: "POST", body: form });
                    const data = await r.json().catch(() => ({}));
                    if (!r.ok) throw new Error(data.detail || `upload failed (${r.status})`);
                    if (data.attachment?.attachment_id) selected.add(data.attachment.attachment_id);
                }
                await this.load(sessionName);
                this.selectedBySession[sessionName] = Array.from(selected);
            } catch (error) {
                Alpine.store("toast").show(String(error), "error", 6000);
            } finally {
                this.uploading = false;
            }
        },
        async remove(item, name) {
            const sessionName = this._name(name);
            const r = await fetch(`/api/sessions/${encodeURIComponent(sessionName)}/attachments/${encodeURIComponent(item.attachment_id)}`, { method: "DELETE" });
            const data = await r.json().catch(() => ({}));
            if (!r.ok) {
                Alpine.store("toast").show(data.detail || `delete failed (${r.status})`, "error");
                return;
            }
            this.removeSelected(item.attachment_id, sessionName);
            await this.load(sessionName);
        },
    });

    Alpine.store("sessions", {
        list: [],
        current: null,
        loaded: [],     // names currently resident in daemon memory
        busy: [],       // names with a turn in flight (server-side)
        async load() {
            const r = await fetch("/api/sessions");
            const data = await r.json();
            this.list = data.sessions || [];
            this.current = data.current;
            this.loaded = data.loaded || [];
            this.busy = data.busy || [];
            const chat = Alpine.store("chat");
            // Sync server-reported busy into chat slots so the UI shows
            // "generating" after a page refresh mid-turn.
            for (const name of this.busy) {
                const slot = chat._slot(name);
                slot.busy = true;
                // Ensure a live trace exists so the "thinking" header and
                // elapsed timer render immediately on refresh mid-turn.
                chat._ensureBusyTrace(slot);
            }
            // First load — sync the chat store's focus pointer.
            if (!chat.currentName && this.current) chat.focus(this.current);
        },
        isLoaded(name) { return (this.loaded || []).includes(name); },
        isBusy(name) {
            // Server-reported busy OR locally-tracked busy. The local
            // slot flag flips on /send before the server has echoed an
            // event back, so the sidebar dot lights up immediately.
            return (this.busy || []).includes(name) ||
                   Alpine.store("chat").isBusy(name);
        },
        async switchTo(name) {
            // In-memory? Just refocus locally and ask the server to flip
            // its focus pointer. No page reload, so the previously-active
            // session keeps streaming its turn into its own slot.
            if (this.isLoaded(name)) {
                try {
                    const r = await fetch(`/api/sessions/${encodeURIComponent(name)}/focus`, {
                        method: "POST",
                    });
                    if (!r.ok) {
                        const d = await r.json().catch(() => ({}));
                        const problem = describeSessionLoadError(d, `Could not switch sessions (${r.status})`);
                        Alpine.store("toast").show(`${problem.title}
${problem.text}`, "error", 14000);
                        return;
                    }
                } catch (e) {
                    Alpine.store("toast").show(`Could not switch sessions: ${e}`, "error");
                    return;
                }
                this.current = name;
                Alpine.store("chat").focus(name);
                // Ensure history is populated (it may be a session the
                // browser has never rendered before, only resident in
                // the daemon's cache).
                Alpine.store("chat").loadHistory(name);
                // mode.load() refreshes the active mode's panel store
                // via panelModes — no explicit teacher/feature call needed.
                Alpine.store("mode").load();
                // Refresh model selector for the newly-focused session.
                await Alpine.store("inspector").loadCurrentProvider();
                return;
            }
            // Not loaded yet — POST /load (which is idempotent).
            const r = await fetch(`/api/sessions/${encodeURIComponent(name)}/load`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({}),
            });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                const problem = describeSessionLoadError(d, `Load failed (${r.status})`);
                Alpine.store("toast").show(`${problem.title}
${problem.text}`, "error", 16000);
                return;
            }
            await this.load();
            this.current = name;
            Alpine.store("chat").focus(name);
            await Alpine.store("chat").loadHistory(name);
            await Alpine.store("mode").load();
            // Refresh model selector for the newly-loaded session.
            await Alpine.store("inspector").loadCurrentProvider();
        },
        async remove(name) {
            const r = await fetch(`/api/sessions/${encodeURIComponent(name)}`, { method: "DELETE" });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(d.detail || `Delete failed (${r.status})`, "error");
                return;
            }
            Alpine.store("toast").show(`Session '${name}' deleted`, "success");
            await this.load();
        },
        async unload(name) {
            const r = await fetch(`/api/sessions/${encodeURIComponent(name)}/unload`, {
                method: "POST",
            });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(
                    d.detail || `Unload failed (${r.status})`, "error", 6000,
                );
                return false;
            }
            await this.load();
            return true;
        },
        async deactivate() {
            // Detach is intentionally non-blocking: a failed or wedged
            // provider turn must never trap the user in this GUI view.
            const r = await fetch("/api/sessions/active/detach", { method: "POST" });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(d.detail || `Leave failed (${r.status})`, "error");
                return false;
            }
            location.reload();
            return true;
        },
    });

    // ── Toast notifications ──────────────────────────────────
    Alpine.store("toast", {
        messages: [],
        _nextId: 0,
        show(msg, type = "info", ms = 4000) {
            const id = this._nextId++;
            this.messages.push({ id, msg, type });
            if (ms > 0) setTimeout(() => this.dismiss(id), ms);
        },
        dismiss(id) {
            this.messages = this.messages.filter(m => m.id !== id);
        },
    });

    Alpine.store("artifacts", {
        bySession: {},
        loading: false,
        loadedName: null,
        get current() {
            const name = Alpine.store("chat").currentName;
            return (name && this.bySession[name]) || [];
        },
        visualizations() {
            return this.current.filter(a => a.kind === "visualization");
        },
        files() {
            return this.current.filter(a => a.kind !== "visualization");
        },
        async load(name, force = false) {
            const target = name || Alpine.store("chat").currentName;
            if (!target) return;
            if (!force && this.loadedName === target && this.bySession[target]) return;
            this.loading = true;
            try {
                const r = await fetch(`/api/sessions/${encodeURIComponent(target)}/artifacts?_ts=${Date.now()}`, { cache: "no-store" });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    throw new Error(d.detail || `artifact load failed (${r.status})`);
                }
                const d = await r.json();
                this.bySession[target] = d.artifacts || [];
                this.loadedName = target;
            } catch (error) {
                Alpine.store("toast").show(String(error), "error");
            } finally {
                this.loading = false;
            }
        },
        add(artifact, name) {
            if (!artifact || !artifact.artifact_id) return;
            const target = name || Alpine.store("chat").currentName;
            if (!target) return;
            const items = this.bySession[target] || [];
            this.bySession[target] = [artifact, ...items.filter(a => a.artifact_id !== artifact.artifact_id)];
            this.loadedName = target;
        },
        async remove(artifactId, name) {
            const target = name || Alpine.store("chat").currentName;
            if (!target) return;
            const r = await fetch(
                `/api/sessions/${encodeURIComponent(target)}/artifacts/${encodeURIComponent(artifactId)}`,
                { method: "DELETE" },
            );
            if (r.ok) {
                this.bySession[target] = (this.bySession[target] || []).filter(
                    a => a.artifact_id !== artifactId
                );
            }
        },
        formatSize(bytes) {
            const n = Number(bytes) || 0;
            if (n >= 1048576) return `${(n / 1048576).toFixed(1)} MB`;
            if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
            return `${n} B`;
        },
    });

    // ── Slash command completion ───────────────────────────────
    //
    // Multi-level: command → subcommand → dynamic args (sessions,
    // features, files, etc.). Mirrors the TUI's SlashCommandCompleter
    // tree from mu/ui/input.py.

    // ── Shell panel (container mode) ───────────────────────────────────
    //
    // WebSocket-backed interactive shell into the session's attached
    // container. Opens via the tools menu → Shell entry (only visible
    // for container sessions). Uses the same /api/containers/{name}/shell
    // WS endpoint as the containers page modal.

    Alpine.store("shell", {
        // MUCLI_SHELL_QOL_V1
        output: "",
        input: "",
        containerName: null,
        connected: false,
        connecting: false,
        error: null,
        followOutput: true,
        history: [],
        historyIndex: 0,
        historyDraft: "",
        _socket: null,
        _loaded: false,
        _intentional_close: false,
        _completionSeq: 0,

        async load() {
            if (this._loaded) return;
            this._loaded = true;
            if (this._socket) this.disconnect();
            this.output = "";
            this.error = null;
            this.containerName = null;
            this.followOutput = true;

            try {
                const r = await fetch("/api/sessions/active");
                const d = await r.json();
                const container = d.container;
                if (!container || !container.name) {
                    this.error = "No container attached to this session.";
                    return;
                }
                this.containerName = container.name;
                this._loadHistory();
                this.connect();
            } catch (e) {
                this.error = `Failed to resolve container: ${e}`;
            }
        },

        connect() {
            if (!this.containerName) return;
            this._intentional_close = false;
            this.connecting = true;
            this.connected = false;
            this.followOutput = true;
            const proto = location.protocol === "https:" ? "wss:" : "ws:";
            const url = `${proto}//${location.host}/api/containers/${encodeURIComponent(this.containerName)}/shell`;
            const socket = new WebSocket(url);
            this._socket = socket;

            socket.onopen = () => {
                this.connecting = false;
                this.connected = true;
                this.error = null;
                this._appendOutput(`Connected to ${this.containerName}\n`, true);
            };
            socket.onmessage = (event) => {
                const raw = String(event.data);
                if (raw.startsWith("{")) {
                    try {
                        const message = JSON.parse(raw);
                        if (message.type === "shell_completion") {
                            this._applyCompletion(message);
                            return;
                        }
                    } catch (_) {
                        // A command may legitimately print JSON.
                    }
                }
                this._appendOutput(raw.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, ""));
            };
            socket.onclose = () => {
                this.connecting = false;
                this.connected = false;
                if (this._socket === socket) {
                    if (!this._intentional_close) {
                        this._appendOutput("\n[shell disconnected]\n");
                    }
                    this._socket = null;
                }
            };
            socket.onerror = () => {
                this.connecting = false;
                this.connected = false;
                this.error = "Shell connection failed. Is the container running?";
            };
        },

        send(command) {
            const cmd = command !== undefined ? command : this.input;
            if (this._socket && this._socket.readyState === WebSocket.OPEN) {
                this._recordHistory(cmd);
                this.followOutput = true;
                this._appendOutput(`$ ${cmd}\n`, true);
                this._socket.send(cmd + "\n");
                this.input = "";
            }
        },

        complete() {
            if (!this._socket || this._socket.readyState !== WebSocket.OPEN) return;
            const line = this.input;
            this._socket.send(JSON.stringify({
                type: "shell_complete",
                request_id: String(++this._completionSeq),
                line,
                cursor: line.length,
            }));
        },

        historyMove(direction) {
            if (!this.history.length) return;
            if (this.historyIndex === this.history.length) this.historyDraft = this.input;
            this.historyIndex = Math.max(
                0,
                Math.min(this.history.length, this.historyIndex + direction),
            );
            this.input = this.historyIndex === this.history.length
                ? this.historyDraft
                : this.history[this.historyIndex];
        },

        disconnect() {
            this._intentional_close = true;
            if (this._socket) {
                this._socket.close();
                this._socket = null;
            }
            this.connected = false;
            this.connecting = false;
            this._loaded = false;
        },

        clear() {
            this.output = "";
            this.followOutput = true;
        },

        onScroll(event) {
            const el = event && event.currentTarget;
            if (!el) return;
            this.followOutput = el.scrollHeight - el.clientHeight - el.scrollTop < 48;
        },

        jumpToEnd() {
            this.followOutput = true;
            this._scroll(true);
        },

        _historyKey() {
            return `mucli-shell-history:${this.containerName || "default"}`;
        },

        _loadHistory() {
            try {
                const value = JSON.parse(localStorage.getItem(this._historyKey()) || "[]");
                this.history = Array.isArray(value) ? value.map(String).slice(-200) : [];
            } catch (_) {
                this.history = [];
            }
            this.historyIndex = this.history.length;
            this.historyDraft = "";
        },

        _recordHistory(command) {
            const value = String(command || "").trim();
            if (!value) return;
            this.history = this.history.filter((item) => item !== value);
            this.history.push(value);
            this.history = this.history.slice(-200);
            this.historyIndex = this.history.length;
            this.historyDraft = "";
            try {
                localStorage.setItem(this._historyKey(), JSON.stringify(this.history));
            } catch (_) {}
        },

        _applyCompletion(message) {
            if (String(message.source || "") !== this.input) return;
            const start = Math.max(0, Math.min(Number(message.start) || 0, this.input.length));
            const end = Math.max(start, Math.min(Number(message.end) || this.input.length, this.input.length));
            const replacement = String(message.replacement || "");
            const candidates = Array.isArray(message.candidates) ? message.candidates : [];
            if (replacement) {
                this.input = this.input.slice(0, start) + replacement + this.input.slice(end);
                return;
            }
            if (candidates.length > 1) {
                this._appendOutput(`\n${candidates.join("  ")}\n`);
            }
        },

        _appendOutput(text, force = false) {
            if (!text) return;
            this.output += text;
            this._scroll(force);
        },

        _scroll(force = false) {
            if (!force && !this.followOutput) return;
            this.$nextTick(() => {
                const el = document.getElementById("shell-output");
                if (el) el.scrollTop = el.scrollHeight;
            });
        },
    });

    Alpine.store("cmdComplete", {
        commands: [],
        visible: false,
        items: [],
        selectedIdx: 0,
        _dynCache: {},
        _subTree: {
            "/history":       { subs: ["clear", "show"] },
            "/session":       { subs: ["list", "load", "new", "delete"],
                                nested: { new: { subs: ["--type"], nested: { "--type": { subs: ["chat", "workspace", "container"] } } } },
                                dynamic: { load: "sessions", delete: "sessions" } },
            "/workspace":     { subs: ["folder", "file", "clear"],
                                nested: {
                                    folder: { subs: ["remove", "clear"],
                                              dynamic: { remove: "path_dir", "": "path_dir" } },
                                    file:   { subs: ["clear"],
                                              dynamic: { "": "path_file" } },
                                } },
            "/model":         { dynamic: { "": "models" } },
            "/provider":      { subs: ["gemini", "ollama", "openai"] },
            "/ollama":        { subs: ["status", "models", "options", "pull"] },
            "/set":           { subs: ["layer"], dynamic: { "": "variables", layer: "layer_ids" } },
            "/get":           { dynamic: { "": "variables", layer: "layer_ids" },
                                subs: ["layer"] },
            "/unset":         { dynamic: { "": "variables" }, subs: ["--all"] },
            "/mode":          { dynamic: { "": "modes" } },
            "/plan":          { subs: ["on", "off", "toggle"] },
            "/verbose":       { subs: ["on", "off", "toggle"] },
            "/show-thinking": { subs: ["on", "off", "toggle"] },
            "/goal":          { subs: ["set", "clear", "show", "help"] },
            "/research":      { subs: ["status", "sources", "show", "bibliography",
                                       "biblio", "bib", "stats", "topic", "topics", "clear"] },
            "/memory":        { subs: ["status", "list", "clear"],
                                nested: {
                                    list:  { dynamic: { "": "memory_targets" } },
                                    clear: { subs: ["task", "scratchpad", "all"] },
                                } },
            "/tool":          { subs: ["list", "enable", "disable"],
                                dynamic: { enable: "tools", disable: "tools" } },
            "/feature":       { subs: ["list", "show", "new", "load", "delete",
                                       "status", "phases", "exit", "unload"],
                                dynamic: { load: "features", delete: "features",
                                           status: "features", phases: "features" } },
            "/teach":         { subs: ["list", "new", "load", "delete", "exit",
                                       "unload", "status", "next", "grades",
                                       "curriculum", "help"] },
            "/t":             { subs: ["list", "new", "load", "delete", "exit",
                                       "unload", "status", "next", "grades",
                                       "curriculum", "help"] },
            "/stats":         { subs: ["clear"] },
            "/skills":        { dynamic: { "": "skills" } },
            "/docs":          { dynamic: { "": "docs" } },
        },

        async load() {
            try {
                const r = await fetch("/api/chat/commands");
                const data = await r.json();
                this.commands = (data.commands || []).map(c => ({
                    name: c.names[0],
                    aliases: c.names.slice(1),
                    help: c.help,
                }));
            } catch (e) { console.error("cmdComplete.load", e); }
        },

        async _fetchDynamic(kind) {
            if (kind.startsWith("path_")) return null;
            if (this._dynCache[kind]) return this._dynCache[kind];
            try {
                const r = await fetch(`/api/chat/completions?kind=${encodeURIComponent(kind)}`);
                const data = await r.json();
                this._dynCache[kind] = data.items || [];
                return this._dynCache[kind];
            } catch (e) { return []; }
        },

        async _fetchPath(prefix, dirsOnly) {
            const parts = prefix.split("/");
            const dir = parts.length > 1 ? parts.slice(0, -1).join("/") || "/" : ".";
            const partial = parts[parts.length - 1] || "";
            try {
                const r = await fetch(`/api/browse?path=${encodeURIComponent(dir)}`);
                if (!r.ok) return [];
                const data = await r.json();
                return (data.entries || [])
                    .filter(e => dirsOnly ? e.is_dir : true)
                    .filter(e => !partial || e.name.toLowerCase().startsWith(partial.toLowerCase()))
                    .map(e => ({ label: e.name + (e.is_dir ? "/" : ""), value: e.path + (e.is_dir ? "/" : ""), isPath: true }));
            } catch (e) { return []; }
        },

        async _addDynItems(items, kind, query, prefix, level) {
            if (kind.startsWith("path_")) {
                const pathItems = await this._fetchPath(query, kind === "path_dir");
                for (const p of pathItems) {
                    items.push({ label: p.label, desc: "", value: prefix + p.value, level, isPath: true });
                }
            } else {
                const dynItems = await this._fetchDynamic(kind);
                if (dynItems) {
                    for (const d of dynItems) {
                        if (d.toLowerCase().startsWith(query)) {
                            items.push({ label: d, desc: "", value: prefix + d, level });
                        }
                    }
                }
            }
        },

        async update(text) {
            if (!text.startsWith("/")) { this.close(); return; }

            const parts = text.split(/\s+/);
            const cmd = parts[0];

            // Level 0: completing the command name itself (no space typed yet)
            if (parts.length === 1) {
                const q = cmd.toLowerCase();
                this.items = this.commands
                    .filter(c => c.name.toLowerCase().startsWith(q) ||
                                 c.aliases.some(a => a.toLowerCase().startsWith(q)))
                    .map(c => ({ label: c.name, desc: c.help, value: c.name, level: 0 }));
                this.selectedIdx = 0;
                this.visible = this.items.length > 0;
                return;
            }

            const tree = this._subTree[cmd];
            if (!tree) { this.close(); return; }

            // Level 1: completing the subcommand or first arg
            if (parts.length === 2) {
                const q = parts[1].toLowerCase();
                const items = [];

                if (tree.subs) {
                    for (const s of tree.subs) {
                        if (s.toLowerCase().startsWith(q)) {
                            items.push({ label: s, desc: "", value: cmd + " " + s, level: 1 });
                        }
                    }
                }

                if (tree.dynamic && tree.dynamic[""] !== undefined) {
                    await this._addDynItems(items, tree.dynamic[""], q, cmd + " ", 1);
                }

                this.items = items;
                this.selectedIdx = 0;
                this.visible = items.length > 0;
                return;
            }

            const sub = parts[1].toLowerCase();

            // Level 2+: nested subcommand trees (e.g. /workspace folder → remove/clear)
            if (parts.length >= 3 && tree.nested) {
                const nested = tree.nested[sub];
                if (nested) {
                    if (parts.length === 3) {
                        const q = parts[2].toLowerCase();
                        const items = [];
                        if (nested.subs) {
                            for (const s of nested.subs) {
                                if (s.toLowerCase().startsWith(q)) {
                                    items.push({ label: s, desc: "", value: cmd + " " + parts[1] + " " + s, level: 2 });
                                }
                            }
                        }
                        const nDynKey = nested.dynamic && (nested.dynamic[parts[2]] !== undefined ? parts[2] : nested.dynamic[""] !== undefined ? "" : null);
                        if (nDynKey !== null && nested.dynamic) {
                            const kind = nested.dynamic[nDynKey];
                            await this._addDynItems(items, kind, q, cmd + " " + parts[1] + " ", 2);
                        }
                        this.items = items;
                        this.selectedIdx = 0;
                        this.visible = items.length > 0;
                        return;
                    }
                    if (parts.length === 4 && nested.dynamic) {
                        const subSub = parts[2].toLowerCase();
                        const kind = nested.dynamic[subSub];
                        if (kind) {
                            const items = [];
                            await this._addDynItems(items, kind, parts[3].toLowerCase(), cmd + " " + parts[1] + " " + parts[2] + " ", 3);
                            this.items = items;
                            this.selectedIdx = 0;
                            this.visible = items.length > 0;
                            return;
                        }
                    }
                }
            }

            // Dynamic arg after a subcommand (e.g. /session load <name>,
            // /tool enable <name>, /feature load <id>)
            if (parts.length === 3 && tree.dynamic) {
                const kind = tree.dynamic[sub];
                if (kind) {
                    const items = [];
                    await this._addDynItems(items, kind, parts[2].toLowerCase(), cmd + " " + parts[1] + " ", 2);
                    this.items = items;
                    this.selectedIdx = 0;
                    this.visible = items.length > 0;
                    return;
                }
            }

            // No matches at this depth
            this.close();
        },

        close() {
            this.visible = false;
            this.items = [];
            this.selectedIdx = 0;
        },
        moveUp() {
            const len = this.items.length;
            if (len) this.selectedIdx = (this.selectedIdx - 1 + len) % len;
        },
        moveDown() {
            const len = this.items.length;
            if (len) this.selectedIdx = (this.selectedIdx + 1) % len;
        },
        accept(textarea) {
            if (!this.items.length) return;
            const item = this.items[this.selectedIdx];
            const hasMore = item.level === 0 && this._subTree[item.value];
            const isDir = item.isPath && item.value.endsWith("/");
            textarea.value = item.value + (item.isPath ? "" : " ");
            textarea.dispatchEvent(new Event("input"));
            if (hasMore || isDir) {
                this.update(textarea.value);
            } else {
                this.close();
            }
        },
        invalidateCache(kind) {
            if (kind) delete this._dynCache[kind];
            else this._dynCache = {};
        },
    });

    Alpine.store("mode", {
        // `active` is the VIEW — which panel the UI shows.
        // `realMode` is the actual MuCLI backend mode.
        // They can diverge: top picker changes view only,
        // chat composer mode selector changes real mode.
        active: "default",
        realMode: "default",
        modes: [],
        views: [],
        sessionType: "workspace",
        hasExecutionWorkspace: false,
        panelModes: ["teacher", "feature", "research", "security", "loop", "debug", "history", "systemPrompts", "memory", "files", "artifacts", "shell"],
        async load() {
            const r = await fetch("/api/modes");
            const data = await r.json();
            this.modes = data.modes || [];
            this.views = data.views || [];
            this.sessionType = data.session_type || "workspace";
            this.hasExecutionWorkspace = !!data.has_execution_workspace;
            this.realMode = data.current || "default";
            this.active = this.realMode;
            const store = this.panelModes.includes(this.active)
                ? Alpine.store(this.active)
                : null;
            if (store && typeof store.load === "function") store.load();
        },
        async set(name) {
            const r = await fetch(`/api/modes/${name}`, { method: "POST" });
            if (r.ok) {
                this.realMode = name;
                this.active = name;
            } else {
                const d = await r.json().catch(() => ({}));
                Alpine.store("chat").addInfo(d.detail || `mode switch failed (${r.status})`);
            }
            await this.load();
        },
        setView(name) {
            this.active = name;
            const store = this.panelModes.includes(name)
                ? Alpine.store(name)
                : null;
            if (store && typeof store.load === "function") store.load();
        },
    });

    // Shared chrome for the mode operating surfaces.  The native mode stores
    // still own their data and actions; this store only keeps exploration
    // state (lens, query, evidence-guide disclosure) stable across refreshes.
    Alpine.store("modeWorkspace", {
        selected: {},
        queries: {},
        qualityOpen: {},
        _storageKey: "mucli.mode-workspace.v1",
        init() {
            try {
                const saved = JSON.parse(localStorage.getItem(this._storageKey) || "{}");
                this.selected = saved.selected || {};
                this.queries = saved.queries || {};
                this.qualityOpen = saved.qualityOpen || {};
            } catch (_) { /* private browsing / corrupt state */ }
        },
        _persist() {
            try {
                localStorage.setItem(this._storageKey, JSON.stringify({
                    selected: this.selected,
                    queries: this.queries,
                    qualityOpen: this.qualityOpen,
                }));
            } catch (_) { /* exploration state is best-effort */ }
        },
        modeName() {
            return (Alpine.store("mode") || {}).active || "default";
        },
        current() {
            const store = Alpine.store(this.modeName());
            return (store && store.workspace) || null;
        },
        selectedView(modeName) {
            const mode = modeName || this.modeName();
            const workspace = this.current();
            const available = (workspace && workspace.views || []).map(v => v.id);
            const saved = this.selected[mode];
            return available.includes(saved) ? saved : "overview";
        },
        select(viewId) {
            const mode = this.modeName();
            this.selected[mode] = viewId || "overview";
            this._persist();
        },
        shows(...viewIds) {
            const selected = this.selectedView();
            return selected === "overview" || viewIds.includes(selected);
        },
        query(modeName) {
            return this.queries[modeName || this.modeName()] || "";
        },
        setQuery(value) {
            this.queries[this.modeName()] = value || "";
            this._persist();
        },
        matches(...values) {
            const q = this.query().trim().toLowerCase();
            if (!q) return true;
            return values.flat(Infinity).some(value =>
                String(value == null ? "" : value).toLowerCase().includes(q)
            );
        },
        toggleQuality() {
            const mode = this.modeName();
            this.qualityOpen[mode] = !this.qualityOpen[mode];
            this._persist();
        },
        isQualityOpen() {
            return !!this.qualityOpen[this.modeName()];
        },
    });

    Alpine.store("prompts", {
        queue: [],
        // `active` returns the head prompt FOR THE FOCUSED SESSION.
        // Background-session prompts queue up but don't render inline
        // until the user switches to that session. A getter (not a
        // method) so Alpine's proxy tracks dependencies.
        get active() {
            const cur = Alpine.store("chat").currentName;
            for (const item of this.queue) {
                const name = item && item.session_name;
                if (!name || !cur || name === cur) return item;
            }
            return null;
        },
        // List of prompts for the currently-focused session (cards render
        // inline; queue beyond the head shows as "N more pending" badge).
        get visibleQueue() {
            const cur = Alpine.store("chat").currentName;
            return this.queue.filter(item => {
                const name = item && item.session_name;
                return !name || !cur || name === cur;
            });
        },
        push(payload) { this.queue.push(payload); },
        // Remove a specific id from the queue (used after answer/cancel
        // since the head isn't always the right one with cross-session
        // prompts in flight).
        _remove(id) {
            const i = this.queue.findIndex(x => x.id === id);
            if (i >= 0) this.queue.splice(i, 1);
        },
        async answer(id, value) {
            if (!id) return false;
            try {
                const r = await fetch(`/api/prompts/${id}/answer`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(value),
                });
                if (!r.ok) {
                    Alpine.store("chat").addError(
                        `prompt answer failed (${r.status})`
                    );
                    return false;
                }
                this._remove(id);
                return true;
            } catch (e) {
                Alpine.store("chat").addError(`prompt answer failed: ${e}`);
                return false;
            }
        },
        async cancel(id) {
            if (id) {
                try {
                    const r = await fetch(`/api/prompts/${id}/cancel`, { method: "POST" });
                    if (!r.ok) {
                        Alpine.store("chat").addError(`prompt cancel failed (${r.status})`);
                        return false;
                    }
                    this._remove(id);
                    return true;
                } catch (e) {
                    Alpine.store("chat").addError(`prompt cancel failed: ${e}`);
                    return false;
                }
            } else {
                // Edge case (no id): drop the head item to unstick the UI.
                this.queue.shift();
                return true;
            }
        },
    });

    Alpine.store("tokens", { input: 0, output: 0, total: 0, total_cost: 0 });

    Alpine.store("yolo", {
        active: false,
        async load() {
            try {
                const r = await fetch("/api/variables");
                const d = await r.json();
                for (const g of (d.groups || [])) {
                    for (const v of (g.variables || [])) {
                        if (v.key === "yolo") { this.active = !!v.value; return; }
                    }
                }
            } catch (e) { console.error("yolo.load", e); }
        },
        async toggle() {
            const next = !this.active;
            try {
                const r = await fetch("/api/variables/yolo", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ value: next }),
                });
                if (r.ok) {
                    this.active = next;
                    Alpine.store("toast").show(
                        next ? "YOLO ON — auto-approving tools" : "YOLO OFF — tools require approval",
                        next ? "info" : "success",
                    );
                    if (next) this._autoApprovePending();
                }
            } catch (e) { console.error("yolo.toggle", e); }
        },
        _autoApprovePending() {
            const prompts = Alpine.store("prompts");
            const chat = Alpine.store("chat");
            const currentName = chat.currentName;
            if (!currentName) return;
            const pending = prompts.queue.filter(item => {
                if (!item.prompt || item.prompt.shape !== "tool_approval") return false;
                return item.session_name === currentName;
            });
            for (const item of pending) {
                chat.addPromptResolved({
                    shape: "tool_approval",
                    title: "approve tool call",
                    answer: `auto-approved \`${item.prompt.tool_name || "tool"}\` (yolo)`,
                    toolName: item.prompt.tool_name || "",
                    cancelled: false,
                }, item.session_name);
                prompts.answer(item.id, { approved: true, remember: false });
            }
        },
    });

    // ── Skills management ─────────────────────────────────────
    Alpine.store("skills", {
        list: [],
        loaded: false,
        modalOpen: false,
        view: "list",
        editing: null,
        form: { name: "", description: "", trigger: "", body: "", scope: "global" },

        async load() {
            try {
                const r = await fetch("/api/skills");
                if (!r.ok) return;
                const data = await r.json();
                this.list = data.skills || [];
                this.loaded = true;
            } catch (e) { console.error("skills.load", e); }
        },
        enabledCount() {
            return this.list.filter(s => s.enabled).length;
        },
        openModal() {
            this.view = "list";
            this.editing = null;
            this.modalOpen = true;
            this.load();
        },
        closeModal() {
            this.modalOpen = false;
            this.view = "list";
            this.editing = null;
        },
        startCreate() {
            this.form = { name: "", description: "", trigger: "", body: "", scope: "global" };
            this.editing = null;
            this.view = "edit";
        },
        startEdit(skill) {
            this.form = {
                name: skill.name,
                description: skill.description || "",
                trigger: skill.trigger || "",
                body: skill.body || "",
                scope: skill.source.includes("/.mu/skills/") && !skill.source.includes(
                    (typeof process !== "undefined" ? process.env.HOME : "~") + "/.mu/skills/"
                ) ? "workspace" : "global",
            };
            this.editing = skill.name;
            this.view = "edit";
        },
        backToList() {
            this.view = "list";
            this.editing = null;
        },
        async toggle(name) {
            const skill = this.list.find(s => s.name === name);
            if (!skill) return;
            const action = skill.enabled ? "disable" : "enable";
            try {
                const r = await fetch(`/api/skills/${encodeURIComponent(name)}/${action}`, { method: "POST" });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store("toast").show(d.detail || `${action} failed`, "error");
                    return;
                }
                const data = await r.json();
                this.list = data.skills || this.list;
            } catch (e) {
                console.error("skills.toggle", e);
            }
        },
        async save() {
            const f = this.form;
            if (!f.name.trim()) {
                Alpine.store("toast").show("Skill name is required", "error");
                return;
            }
            try {
                const r = await fetch("/api/skills/save", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(f),
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store("toast").show(d.detail || "Save failed", "error");
                    return;
                }
                const data = await r.json();
                this.list = data.skills || this.list;
                Alpine.store("toast").show(
                    `Skill '${f.name}' saved (${f.scope})`, "success",
                );
                this.view = "list";
                this.editing = null;
            } catch (e) {
                console.error("skills.save", e);
                Alpine.store("toast").show("Save failed", "error");
            }
        },
        async deleteSkill(name) {
            try {
                const r = await fetch(`/api/skills/${encodeURIComponent(name)}`, { method: "DELETE" });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store("toast").show(d.detail || "Delete failed", "error");
                    return;
                }
                const data = await r.json();
                this.list = data.skills || this.list;
                Alpine.store("toast").show(`Skill '${name}' deleted`, "info");
                if (this.editing === name) {
                    this.view = "list";
                    this.editing = null;
                }
            } catch (e) {
                console.error("skills.delete", e);
                Alpine.store("toast").show("Delete failed", "error");
            }
        },
        async reload() {
            try {
                const r = await fetch("/api/skills/reload", { method: "POST" });
                if (!r.ok) return;
                const data = await r.json();
                this.list = data.skills || [];
                Alpine.store("toast").show(`${this.list.length} skills reloaded`, "success");
            } catch (e) { console.error("skills.reload", e); }
        },
        isBuiltin(skill) {
            return skill.source && !skill.source.includes("/.mu/skills/");
        },
    });

    Alpine.store("theme", {
        // Initialised from the data-theme attribute the pre-paint
        // script in base.html set. Toggling it updates the DOM,
        // localStorage, and (via subscribers) the highlight.js link.
        current: document.documentElement.getAttribute("data-theme") || "dark",
        toggle() {
            this.current = this.current === "dark" ? "light" : "dark";
            try { localStorage.setItem("mucli-theme", this.current); } catch (e) {}
            applyTheme(this.current);
            // Re-highlight existing code blocks after the stylesheet swap.
            queueMicrotask(rehighlightAll);
        },
    });

    Alpine.store("layout", {
        sidebarOpen: true,
        panelOpen: true,
        sidebarWidth: 220,
        panelWidth: 360,
        _dragging: null,

        init() {
            try {
                const saved = JSON.parse(localStorage.getItem("mucli-layout") || "{}");
                if (saved.sidebarOpen === false) this.sidebarOpen = false;
                if (saved.panelOpen === false) this.panelOpen = false;
                if (saved.sidebarWidth > 0) this.sidebarWidth = saved.sidebarWidth;
                if (saved.panelWidth > 0) this.panelWidth = saved.panelWidth;
            } catch (e) {}
        },
        _persist() {
            try {
                localStorage.setItem("mucli-layout", JSON.stringify({
                    sidebarOpen: this.sidebarOpen,
                    panelOpen: this.panelOpen,
                    sidebarWidth: this.sidebarWidth,
                    panelWidth: this.panelWidth,
                }));
            } catch (e) {}
        },
        toggleSidebar() {
            this.sidebarOpen = !this.sidebarOpen;
            this._persist();
        },
        togglePanel() {
            this.panelOpen = !this.panelOpen;
            this._persist();
        },
        startDrag(which, e) {
            e.preventDefault();
            this._dragging = which;
            const app = document.querySelector(".app");
            document.body.classList.add("resizing");
            const self = this;
            const onMove = (me) => {
                if (!app) return;
                if (which === "sidebar") {
                    const w = Math.max(140, Math.min(500, me.clientX));
                    app.style.setProperty("--sidebar-w", w + "px");
                    self._dragVal = w;
                } else if (which === "panel") {
                    const w = Math.max(240, Math.min(700, window.innerWidth - me.clientX));
                    app.style.setProperty("--panel-w", w + "px");
                    self._dragVal = w;
                }
            };
            const onUp = () => {
                document.body.classList.remove("resizing");
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                if (which === "sidebar" && self._dragVal) self.sidebarWidth = self._dragVal;
                if (which === "panel" && self._dragVal) self.panelWidth = self._dragVal;
                self._dragging = null;
                self._dragVal = null;
                self._persist();
            };
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
        },
    });

    Alpine.store("teacher", {
        // Active course payload from /api/teacher/state. Loaded on demand
        // when the user is in teacher mode AND a session is active.
        course: null,
        courses: [],
        active: false,
        loaded: false,
        coursePath: null,    // debugging breadcrumb: where the data was read
        workspace: null,
        openSections: {
            profile: true,
            curriculum: true,
            current: true,
            assignments: true,
            reviews: false,
        },

        async load() {
            try {
                const r = await fetch("/api/teacher/state");
                const d = await r.json();
                this.active = !!d.active;
                this.course = d.course || null;
                this.courses = d.courses || [];
                this.coursePath = d.course_path || null;
                this.workspace = d.workspace || null;
                this.loaded = true;
                if (window.__mucliTeacherDebug) {
                    console.log("teacher.load", {
                        active_course_id: d.active_course_id,
                        raw_teacher_state_present: d.raw_teacher_state_present,
                        registry_size: d.registry_size,
                        course_path: d.course_path,
                        course: d.course,
                        courses: d.courses,
                    });
                }
            } catch (e) {
                console.error("teacher.load", e);
            }
        },
        toggleSection(name) {
            this.openSections[name] = !this.openSections[name];
        },

        // ---- view helpers ----
        statusGlyph(status) {
            switch ((status || "").toLowerCase()) {
                case "completed":     return "✓";
                case "graded":        return "✓";
                case "lecturing":
                case "presenting":
                case "assigned":      return "◐";
                case "remediating":   return "⚠";
                case "pending":       return "○";
                default:              return "·";
            }
        },
        // Diagnostic: flip this in the browser console to log every
        // teacher state refresh: `window.__mucliTeacherDebug = true`.
        currentLesson() {
            if (!this.course || !this.course.current_lesson_id) return null;
            return (this.course.lessons || []).find(
                l => l.lesson_id === this.course.current_lesson_id
            ) || null;
        },
        currentModule() {
            if (!this.course || !this.course.current_module_id) return null;
            return (this.course.modules || []).find(
                m => m.module_id === this.course.current_module_id
            ) || null;
        },
        gradedAssignments() {
            if (!this.course) return [];
            return (this.course.assignments || []).filter(a => a.grade);
        },
        allAssignments() {
            if (!this.course) return [];
            return (this.course.assignments || []).filter(a =>
                Alpine.store("modeWorkspace").matches(
                    a.assignment_id, a.kind, a.status, a.prompt, a.rubric,
                    a.grade && a.grade.feedback
                )
            );
        },
        scheduledReviews() {
            if (!this.course) return [];
            return (this.course.scheduled_reviews || []).filter(r =>
                Alpine.store("modeWorkspace").matches(
                    r.source_lesson_id, r.source_lesson_title, r.status, r.notes
                )
            );
        },
        filteredModules() {
            if (!this.course) return [];
            const ws = Alpine.store("modeWorkspace");
            return (this.course.modules || []).map(module => {
                if (ws.matches(module.title, module.goal, module.module_id)) return module;
                const lessons = (module.lessons || []).filter(lesson => ws.matches(
                    lesson.title, lesson.lesson_id, lesson.concept_brief,
                    lesson.learning_objectives, lesson.lecture_gaps
                ));
                return { ...module, lessons };
            }).filter(module => module.lessons.length || ws.matches(
                module.title, module.goal, module.module_id
            ));
        },
        // Map learner_profile keys → which fields are array-of-tags
        // versus solo-text. Surfacing both shapes in one helper keeps
        // the template lean.
        profileTagFields() {
            return ["strengths", "gaps", "goals", "modality",
                    "background", "anchors", "stumbling_blocks"];
        },
        profileTextFields() {
            return ["pace", "jargon_tolerance", "motivation",
                    "personality", "notes"];
        },
        // Per-module completion: how many lessons inside this module
        // have status==='completed'? Used for the curriculum progress bar.
        moduleProgress(m) {
            if (!m || !m.lessons || !m.lessons.length) {
                return { done: 0, total: 0, pct: 0 };
            }
            const done = m.lessons.filter(l => (l.status || "") === "completed").length;
            const total = m.lessons.length;
            return { done, total, pct: total ? Math.round((done * 100) / total) : 0 };
        },
        // ISO-ish timestamp from a unix float. Used for learner-profile
        // recorded_at + course updated_at.
        formatTimestamp(unix) {
            if (!unix || typeof unix !== "number") return "";
            try {
                return new Date(unix * 1000).toLocaleString();
            } catch (e) {
                return "";
            }
        },
        async switchCourse(courseId) {
            // Sends a /teach load <id> via chat so the agent's command
            // dispatch handles it (keeping registry/state in sync).
            if (!courseId) return;
            await Alpine.store("chat").send(`/teach load ${courseId}`);
        },
    });

    Alpine.store("feature", {
        // Active feature plan summary from /api/feature/state. Same load
        // triggers as the teacher store: page boot, mode flip, turn_complete
        // SSE, session_updated SSE.
        plan: null,
        features: [],
        active: false,
        loaded: false,
        metadataPath: null,
        workspace: null,
        openSections: {
            events: false,
            reviews: false,
        },
        // Per-phase collapse state. Defaults open for the in-progress
        // phase, closed otherwise; the load() hook sets initial values.
        openPhases: {},
        // The task whose drill-down is currently expanded (one at a time).
        expandedTaskId: null,
        // Drag-and-drop transfer state.
        dragTaskId: null,
        // Search/filter query for the feature list.
        searchQuery: '',
        showFeatureBrowser: false,
        previewMode: false,
        previewArchived: false,
        previewingFeatureId: null,
        planIdentity: null,
        sessionName: null,
        loadRevision: 0,
        navigationRevision: 0,

        _currentSessionName() {
            const chat = Alpine.store("chat");
            return String((chat && chat.currentName) || "");
        },
        _applyPlan(plan, { preview = false, archived = false } = {}) {
            const nextIdentity = String((plan && plan.feature_id) || "");
            if (nextIdentity !== this.planIdentity) {
                // Phase/task ids start at one for most feature plans. Reset
                // view-local expansion state when the plan changes so Alpine
                // cannot carry a similarly-numbered task from another feature
                // into the newly-rendered detail view.
                this.openPhases = {};
                this.expandedTaskId = null;
                this.dragTaskId = null;
                this.planIdentity = nextIdentity || null;
            }
            this.plan = plan || null;
            this.previewMode = !!preview;
            this.previewArchived = !!(preview && archived);

            const phases = (this.plan && this.plan.phase_columns) || [];
            phases.forEach((phase, index) => {
                const key = String(phase.id);
                if (this.openPhases[key] !== undefined) return;
                const status = (phase.status || "").toLowerCase();
                this.openPhases[key] = status === "in_progress"
                    || status === "blocked"
                    || (!!preview && index === 0);
            });
        },
        async load({ forcePlan = false, resetView = false } = {}) {
            const currentSession = this._currentSessionName();
            const sessionChanged = this.sessionName !== null
                && this.sessionName !== currentSession;
            if (sessionChanged || resetView) {
                this.navigationRevision += 1;
                this.showFeatureBrowser = false;
                this.previewMode = false;
                this.previewArchived = false;
                this.previewingFeatureId = null;
                this._applyPlan(null);
            }
            this.sessionName = currentSession;
            if (forcePlan) {
                this.navigationRevision += 1;
                this.previewMode = false;
                this.previewArchived = false;
                this.previewingFeatureId = null;
            }

            const loadId = ++this.loadRevision;
            const navigationAtStart = this.navigationRevision;
            try {
                const r = await fetch("/api/feature/state");
                const d = await r.json();
                if (loadId !== this.loadRevision) return;
                this.active = !!d.active;
                this.features = d.features || [];
                const navigationIsStable = navigationAtStart === this.navigationRevision;
                if (forcePlan || (
                    navigationIsStable
                    && !this.previewMode
                    && !this.previewingFeatureId
                )) {
                    this.metadataPath = d.metadata_path || null;
                    this.workspace = d.workspace || null;
                    this._applyPlan(d.plan || null);
                }
                this.loaded = true;
            } catch (e) {
                console.error("feature.load", e);
            }
        },
        toggleSection(name) {
            this.openSections[name] = !this.openSections[name];
        },
        togglePhase(id) {
            const key = String(id);
            this.openPhases[key] = !this.openPhases[key];
        },
        isPhaseOpen(id) {
            return !!this.openPhases[String(id)];
        },
        toggleTask(id) {
            this.expandedTaskId = this.expandedTaskId === id ? null : id;
        },
        isTaskExpanded(id) {
            return this.expandedTaskId === id;
        },
        isBrowserView() {
            return this.showFeatureBrowser || !this.plan;
        },
        isDetailView() {
            return !!this.plan && !this.showFeatureBrowser;
        },
        activePlanAvailable() {
            if (!this.plan || this.previewMode) return false;
            const id = String(this.plan.feature_id || "");
            return (this.features || []).some(feature =>
                String(feature.feature_id || "") === id && feature.is_active
            );
        },
        planRenderKey() {
            return String((this.plan && this.plan.feature_id) || "feature");
        },
        phaseRenderKey(phase) {
            return `${this.planRenderKey()}:phase:${phase && phase.id}`;
        },
        taskRenderKey(phase, task) {
            return `${this.phaseRenderKey(phase)}:task:${task && task.id}`;
        },

        // ---- view helpers ----
        statusGlyph(status) {
            switch ((status || "").toLowerCase()) {
                case "completed":   return "✓";
                case "in_progress": return "◐";
                case "blocked":     return "⚠";
                case "archived":    return "✕";
                case "pending":
                case "not_started": return "○";
                default:            return "·";
            }
        },
        formatTimestamp(unix) {
            if (!unix || typeof unix !== "number") return "";
            try { return new Date(unix * 1000).toLocaleString(); }
            catch (e) { return ""; }
        },
        // The agent's "current" task is the in_progress one (if any), else
        // the next actionable. summarize_feature_plan ships this as
        // `execution.next_task`; surface both via one accessor.
        currentTask() {
            if (!this.plan) return null;
            const exec = this.plan.execution || {};
            if (exec.next_task) return exec.next_task;
            const active = this.plan.active_tasks || [];
            return active.length ? active[0] : null;
        },
        phaseColumns() {
            const phases = (this.plan && this.plan.phase_columns) || [];
            const ws = Alpine.store("modeWorkspace");
            return phases.map(phase => {
                if (ws.matches(phase.title, phase.goal, phase.status)) return phase;
                const tasks = (phase.tasks || []).filter(task => ws.matches(
                    task.title, task.status, task.objectives, task.action_points,
                    task.exit_criteria, task.blocked_reason, task.notes
                ));
                return { ...phase, tasks };
            }).filter(phase => phase.tasks.length || ws.matches(
                phase.title, phase.goal, phase.status
            ));
        },
        progressPct() {
            if (!this.plan || !this.plan.task_count) return 0;
            return Math.round(
                (this.tasksCompletedCount() /
                    (this.plan.task_count || 1)) * 100
            );
        },
        // The backend's summary uses `tasks_completed` as a bool; derive
        // the count from the task list so the progress bar makes sense.
        tasksCompletedCount() {
            if (!this.plan) return 0;
            return (this.plan.phases || []).filter(
                t => (t.status || "").toLowerCase() === "completed"
            ).length;
        },
        recentEvents(limit = 5) {
            if (!this.plan || !this.plan.event_log) return [];
            return this.plan.event_log.slice().reverse().filter(ev =>
                Alpine.store("modeWorkspace").matches(
                    ev.kind, ev.entity, ev.entity_id, ev.actor, ev.payload
                )
            ).slice(0, limit);
        },
        reviews() {
            if (!this.plan) return [];
            return (this.plan.review_records || []).filter(review =>
                Alpine.store("modeWorkspace").matches(
                    review.task_id, review.summary, review.limitations, review.issues
                )
            );
        },
        // ---- mutating actions ----
        async transitionTask(taskId, toStatus) {
            if (!taskId || !toStatus) return;
            try {
                const r = await fetch(
                    `/api/feature/tasks/${taskId}/transition`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ to_status: toStatus }),
                    }
                );
                if (!r.ok) {
                    const data = await r.json().catch(() => ({}));
                    Alpine.store("chat").addInfo(
                        `feature: cannot move task #${taskId} to ${toStatus} — ${data.detail || r.status}`
                    );
                }
            } catch (e) {
                console.error("feature.transitionTask", e);
            } finally {
                await this.load();
            }
        },
        async toggleExitCriterion(taskId, idx) {
            if (!taskId && taskId !== 0) return;
            try {
                const r = await fetch(
                    `/api/feature/tasks/${taskId}/exit-criteria/${idx}/toggle`,
                    { method: "POST" }
                );
                if (!r.ok) {
                    const data = await r.json().catch(() => ({}));
                    console.warn("toggleExitCriterion", r.status, data.detail);
                }
            } catch (e) {
                console.error("feature.toggleExitCriterion", e);
            } finally {
                await this.load();
            }
        },
        isVerified(task, criterion) {
            return (task.verified_exit_criteria || []).includes(criterion);
        },
        showArchived: false,
        filteredFeatures() {
            const q = (this.searchQuery || '').toLowerCase().trim();
            const all = this.features || [];
            const visible = all.filter(f => this.showArchived ? f.archived : !f.archived);
            if (!q) return visible;
            return visible.filter(f =>
                (f.feature_id || '').toLowerCase().includes(q) ||
                (f.feature_name || '').toLowerCase().includes(q) ||
                (f.status || '').toLowerCase().includes(q)
            );
        },
        archivedCount() {
            return (this.features || []).filter(f => f.archived).length;
        },
        currentCount() {
            return (this.features || []).filter(f => !f.archived).length;
        },
        openFeatureBrowser() {
            this.navigationRevision += 1;
            this.previewingFeatureId = null;
            this.showFeatureBrowser = true;
        },
        closeFeatureBrowser() {
            if (this.activePlanAvailable()) this.showFeatureBrowser = false;
        },
        async _action(featureId, path, method, verb) {
            if (!featureId) return false;
            try {
                const r = await fetch(`/api/feature/${encodeURIComponent(featureId)}${path}`, { method });
                if (!r.ok) {
                    const data = await r.json().catch(() => ({}));
                    Alpine.store("toast").show(data.detail || `${verb} failed (${r.status})`, "error");
                    return false;
                }
                Alpine.store("toast").show(`Feature '${featureId}': ${verb.toLowerCase()} complete`, "success");
                await this.load({ forcePlan: true });
                return true;
            } catch (e) {
                console.error(`feature.${verb}`, e);
                Alpine.store("toast").show(`${verb} failed — network error`, "error");
                return false;
            }
        },
        async deleteFeature(id)    { return this._action(id, "",           "DELETE", "Delete");    },
        async unloadFeature(id)    { return this._action(id, "/unload",    "POST",   "Unload");    },
        async loadFeature(id)      {
            if (!id) return;
            // If a loop session is active, warn the user before switching features
            if (Alpine.store('loop').loopActive) {
                let confirmed = false;
                await new Promise(resolve => {
                    Alpine.store('confirm').ask(
                        'A session is running. Switching features will pause it. Continue?',
                        null,
                        () => { confirmed = true; resolve(); },
                        { danger: true }
                    );
                    // Fallback: if confirm popover is dismissed without action (cancel),
                    // resolve immediately so we don't hang.
                    const watch = setInterval(() => {
                        if (!Alpine.store('confirm').open) {
                            clearInterval(watch);
                            resolve();
                        }
                    }, 150);
                });
                if (!confirmed) return;
            }
            return this._action(id, "/load", "POST", "Load");
        },
        async archiveFeature(id)   { return this._action(id, "/archive",   "POST",   "Archive");   },
        async unarchiveFeature(id) { return this._action(id, "/unarchive", "POST",   "Unarchive"); },
        async approveFeature(id)   { return this._action(id, "/approve",   "POST",   "Approve");   },
        async switchFeature(featureId) {
            if (!featureId) return;
            const loaded = await this.loadFeature(featureId);
            if (loaded) this.showFeatureBrowser = false;
        },
        async previewFeature(id) {
            if (!id) return;
            const navigationId = ++this.navigationRevision;
            this.previewingFeatureId = id;
            try {
                const r = await fetch(`/api/feature/${encodeURIComponent(id)}/preview`);
                if (!r.ok) {
                    const data = await r.json().catch(() => ({}));
                    Alpine.store("toast").show(data.detail || `Preview failed (${r.status})`, "error");
                    return;
                }
                const d = await r.json();
                if (navigationId !== this.navigationRevision) return;
                this.active = !!d.active;
                this.features = d.features || this.features;
                this.metadataPath = d.metadata_path || null;
                this.workspace = d.workspace || this.workspace;
                const item = (this.features || []).find(feature =>
                    String(feature.feature_id || "") === String(id)
                );
                this._applyPlan(d.plan || null, {
                    preview: true,
                    archived: !!(item && item.archived),
                });
                // A preview is a peer view, not another layer beneath the
                // feature browser. Closing the browser here guarantees that
                // only one Mode OS surface is mounted at a time.
                this.showFeatureBrowser = false;
            } catch (e) {
                console.error("feature.previewFeature", e);
                Alpine.store("toast").show("Preview failed — network error", "error");
            } finally {
                if (navigationId === this.navigationRevision) {
                    this.previewingFeatureId = null;
                }
            }
        },
        async exitPreview() {
            this.navigationRevision += 1;
            this.previewMode = false;
            this.previewArchived = false;
            this.previewingFeatureId = null;
            this.showFeatureBrowser = true;
            await this.load({ forcePlan: true });
        },

        showCreateModal: false,
        createForm: { name: '', request: '', directory: '', phases: [] },
        openCreateModal() {
            this.showCreateModal = true;
            this.createForm = { name: '', request: '', directory: '', phases: [] };
        },
        closeCreateModal() {
            this.showCreateModal = false;
            this.createForm = { name: '', request: '', directory: '', phases: [] };
        },
        addPhase() {
            if (!this.createForm) return;
            this.createForm.phases.push({ title: '', goal: '', tasks: [] });
        },
        removePhase(idx) {
            if (!this.createForm) return;
            this.createForm.phases.splice(idx, 1);
        },
        addTask(phaseIdx) {
            if (!this.createForm) return;
            this.createForm.phases[phaseIdx].tasks.push({
                title: '', objectives: [''], exit_criteria: [''],
            });
        },
        removeTask(phaseIdx, taskIdx) {
            if (!this.createForm) return;
            this.createForm.phases[phaseIdx].tasks.splice(taskIdx, 1);
        },
        addListItem(phaseIdx, taskIdx, field) {
            if (!this.createForm) return;
            this.createForm.phases[phaseIdx].tasks[taskIdx][field].push('');
        },
        removeListItem(phaseIdx, taskIdx, field, itemIdx) {
            if (!this.createForm) return;
            this.createForm.phases[phaseIdx].tasks[taskIdx][field].splice(itemIdx, 1);
        },
        async submitCreateFeature() {
            const form = this.createForm;
            if (!form || !(form.name || '').trim()) {
                Alpine.store('toast').show('Feature name is required', 'error');
                return;
            }
            const payload = {
                feature_name: form.name.trim(),
                feature_request: (form.request || '').trim(),
                directory: (form.directory || '').trim(),
                phases: (form.phases || []).filter(p => (p.title || '').trim()).map(p => ({
                    title: p.title.trim(),
                    goal: (p.goal || '').trim(),
                    tasks: (p.tasks || []).filter(t => (t.title || '').trim()).map(t => ({
                        title: t.title.trim(),
                        objectives: (t.objectives || []).map(o => o.trim()).filter(Boolean),
                        exit_criteria: (t.exit_criteria || []).map(c => c.trim()).filter(Boolean),
                    })),
                })),
            };
            try {
                const r = await fetch('/api/feature/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store('toast').show(d.detail || `Create failed (${r.status})`, 'error');
                    return;
                }
                const d = await r.json();
                Alpine.store('toast').show(`Feature '${d.feature_id}' created`, 'success');
                this.closeCreateModal();
                await this.load();
            } catch (e) {
                Alpine.store('toast').show('Create failed — network error', 'error');
            }
        },
    });

    Alpine.store("research", {
        sources: [],
        sourceCount: 0,
        bibliography: "",
        currentTopic: "general",
        topics: [],
        topicFilter: "",
        findings: [],
        findingCount: 0,
        active: false,
        loaded: false,
        workspace: null,
        openSections: {
            sources: true,
            bibliography: false,
            findings: true,
        },
        // Client-side filter: selected source types (empty = show all).
        typeFilter: [],
        // Client-side credibility threshold slider (0 = no filter).
        credibilityMin: 0,
        // Which source row is expanded for detail.
        expandedSourceId: null,

        async load() {
            try {
                const r = await fetch("/api/research/state");
                const d = await r.json();
                this.active = !!d.active;
                this.sources = d.sources || [];
                this.sourceCount = d.source_count || 0;
                this.bibliography = d.bibliography || "";
                this.currentTopic = d.current_topic || "general";
                this.topics = d.topics || [];
                this.findings = d.findings || [];
                this.findingCount = d.finding_count || 0;
                this.workspace = d.workspace || null;
                this.loaded = true;
            } catch (e) {
                console.error("research.load", e);
            }
        },
        toggleSection(name) {
            this.openSections[name] = !this.openSections[name];
        },
        toggleType(type) {
            const i = this.typeFilter.indexOf(type);
            if (i >= 0) this.typeFilter.splice(i, 1);
            else this.typeFilter.push(type);
        },
        isTypeActive(type) {
            return this.typeFilter.length === 0 || this.typeFilter.includes(type);
        },
        toggleSource(id) {
            this.expandedSourceId = this.expandedSourceId === id ? null : id;
        },
        isSourceExpanded(id) {
            return this.expandedSourceId === id;
        },
        filteredSources() {
            return this.sources.filter(s => {
                if (this.topicFilter && !(s.topic || "general").toLowerCase().includes(this.topicFilter.toLowerCase())) return false;
                if (this.typeFilter.length && !this.typeFilter.includes(s.source_type)) return false;
                if (this.credibilityMin > 0 && (s.credibility_score || 0) < this.credibilityMin) return false;
                if (!Alpine.store("modeWorkspace").matches(
                    s.title, s.url, s.source_type, s.authors, Object.keys(s.metadata || {})
                )) return false;
                return true;
            });
        },
        // Sources grouped by topic for the grouped-sources view.
        // Returns [{ topic, sources }] in first-seen order.
        groupedSources() {
            const filtered = this.filteredSources();
            const order = [];
            const buckets = {};
            for (const s of filtered) {
                const t = s.topic || "general";
                if (!(t in buckets)) {
                    buckets[t] = [];
                    order.push(t);
                }
                buckets[t].push(s);
            }
            return order.map(t => ({ topic: t, sources: buckets[t] }));
        },
        sourceCountForTopic(topic) {
            return this.sources.filter(s => (s.topic || "general") === topic).length;
        },
        filteredFindings() {
            return this.findings.filter(f => Alpine.store("modeWorkspace").matches(
                f.content, f.tags, f.source
            ));
        },
        sourceTypes() {
            const types = new Set(this.sources.map(s => s.source_type));
            return [...types].sort();
        },
        credibilityPct(score) {
            return Math.round((score || 0) * 100);
        },
        formatTimestamp(unix) {
            if (!unix || typeof unix !== "number") return "";
            try { return new Date(unix * 1000).toLocaleString(); }
            catch (e) { return ""; }
        },
        async copyBibliography() {
            try {
                await navigator.clipboard.writeText(this.bibliography);
            } catch (e) {
                console.warn("clipboard write failed", e);
            }
        },
    });

    Alpine.store("security", {
        report: null,
        findings: [],
        summary: null,
        active: false,
        loaded: false,
        workspace: null,
        openSections: {
            findings: true,
            stats: false,
        },
        severityFilter: [],
        expandedFindingId: null,

        async load() {
            try {
                const r = await fetch("/api/security/state");
                const d = await r.json();
                this.active = !!d.active;
                this.report = d.report || null;
                this.findings = d.findings || [];
                this.summary = d.summary || null;
                this.workspace = d.workspace || null;
                this.loaded = true;
            } catch (e) {
                console.error("security.load", e);
            }
        },
        toggleSection(name) {
            this.openSections[name] = !this.openSections[name];
        },
        toggleSeverity(sev) {
            const i = this.severityFilter.indexOf(sev);
            if (i >= 0) this.severityFilter.splice(i, 1);
            else this.severityFilter.push(sev);
        },
        isSeverityActive(sev) {
            return this.severityFilter.length === 0 || this.severityFilter.includes(sev);
        },
        toggleFinding(id) {
            this.expandedFindingId = this.expandedFindingId === id ? null : id;
        },
        isFindingExpanded(id) {
            return this.expandedFindingId === id;
        },
        filteredFindings() {
            return this.findings.filter(f => {
                if (this.severityFilter.length && !this.severityFilter.includes(f.severity)) return false;
                return Alpine.store("modeWorkspace").matches(
                    f.title, f.summary, f.vulnerability_class, f.affected_paths,
                    f.exploit_path, f.references, f.status
                );
            });
        },
        severities() {
            const s = new Set(this.findings.map(f => f.severity));
            const order = ["critical", "high", "medium", "low", "info"];
            return order.filter(sev => s.has(sev));
        },
        statusGlyph(status) {
            switch ((status || "").toLowerCase()) {
                case "approved":              return "✓";
                case "exploit_verified":
                case "remediation_verified":  return "◐";
                case "refuted":               return "✕";
                case "proof_attached":
                case "remediation_attached":  return "◔";
                case "new":                   return "○";
                default:                      return "·";
            }
        },
        severityColor(sev) {
            switch ((sev || "").toLowerCase()) {
                case "critical": return "var(--err)";
                case "high":     return "var(--risk-high)";
                case "medium":   return "var(--risk-medium)";
                case "low":      return "var(--risk-low)";
                case "info":     return "var(--text-dimmer)";
                default:         return "var(--text-dim)";
            }
        },
        formatTimestamp(unix) {
            if (!unix || typeof unix !== "number") return "";
            try { return new Date(unix * 1000).toLocaleString(); }
            catch (e) { return ""; }
        },
        async approveFinding(findingId) {
            try {
                const r = await fetch(
                    `/api/security/findings/${findingId}/approve`,
                    { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }
                );
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store("chat").addInfo(`security: cannot approve ${findingId} — ${d.detail || r.status}`);
                }
            } catch (e) {
                console.error("security.approve", e);
            } finally {
                await this.load();
            }
        },
        async refuteFinding(findingId, reason) {
            try {
                const r = await fetch(
                    `/api/security/findings/${findingId}/refute`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ reason: reason || "" }),
                    }
                );
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store("chat").addInfo(`security: cannot refute ${findingId} — ${d.detail || r.status}`);
                }
            } catch (e) {
                console.error("security.refute", e);
            } finally {
                await this.load();
            }
        },
    });

    Alpine.store("loop", {
        loopGoal: "",
        loopActive: false,
        loopFeatures: [],
        backlog: [],
        memory: [],
        active: false,
        loaded: false,
        workspace: null,
        openSections: { backlog: true, features: false, memory: false },

        async load() {
            try {
                const r = await fetch("/api/loop/state");
                const d = await r.json();
                this.active = !!d.active;
                this.loopGoal = d.loop_goal || "";
                this.loopActive = !!d.loop_active;
                this.loopFeatures = d.loop_features || [];
                this.backlog = d.backlog || [];
                this.memory = d.memory || [];
                this.workspace = d.workspace || null;
                this.loaded = true;
            } catch (e) { console.error("loop.load", e); }
        },
        toggleSection(name) { this.openSections[name] = !this.openSections[name]; },
        statusGlyph(status) {
            switch ((status || "").toLowerCase()) {
                case "completed": return "✓";
                case "in_progress": return "◐";
                case "blocked": return "⚠";
                default: return "○";
            }
        },
        formatTimestamp(unix) {
            if (!unix || typeof unix !== "number") return "";
            try { return new Date(unix * 1000).toLocaleString(); }
            catch (e) { return ""; }
        },

        expandedItemId: null,
        toggleItem(id) { this.expandedItemId = this.expandedItemId === id ? null : id; },
        isItemExpanded(id) { return this.expandedItemId === id; },
        filteredBacklog() {
            return this.backlog.filter(item => Alpine.store("modeWorkspace").matches(
                item.content, item.tags, item.source, item.status
            ));
        },
        filteredMemory() {
            return this.memory.filter(item => Alpine.store("modeWorkspace").matches(
                item.content, item.tags, item.source, item.kind
            ));
        },
        async setActive(active) {
            try {
                const r = await fetch('/api/loop/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ active: !!active, goal: this.loopGoal || '' }),
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store('toast').show(d.detail || `Loop control failed (${r.status})`, 'error');
                    return;
                }
                await this.load();
                Alpine.store('toast').show(active ? 'Loop resumed' : 'Loop paused', 'success');
            } catch (e) {
                Alpine.store('toast').show('Loop control failed — network error', 'error');
            }
        },

        showAddItem: false,
        newItemContent: '',
        newItemStatus: 'pending',
        openAddItem() {
            this.showAddItem = true;
            this.newItemContent = '';
            this.newItemStatus = 'pending';
        },
        closeAddItem() { this.showAddItem = false; },
        async addBacklogItem() {
            const content = (this.newItemContent || '').trim();
            if (!content) return;
            try {
                const r = await fetch('/api/loop/backlog', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content, status: this.newItemStatus }),
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store('toast').show(d.detail || `Add failed (${r.status})`, 'error');
                    return;
                }
                Alpine.store('toast').show('Backlog item added', 'success');
                this.closeAddItem();
                await this.load();
            } catch (e) {
                Alpine.store('toast').show('Add failed — network error', 'error');
            }
        },
    });

    Alpine.store("debug", {
        debugTarget: "",
        hypotheses: [],
        suspects: [],
        notes: [],
        findings: [],
        scratchpadCount: 0,
        active: false,
        loaded: false,
        workspace: null,
        openSections: { hypotheses: true, suspects: true, notes: true, findings: false },
        expandedHypothesisId: null,

        async load() {
            try {
                const r = await fetch("/api/debug/state");
                const d = await r.json();
                this.active = !!d.active;
                this.debugTarget = d.debug_target || "";
                this.hypotheses = d.hypotheses || [];
                this.suspects = d.suspects || [];
                this.notes = d.notes || [];
                this.findings = d.findings || [];
                this.scratchpadCount = d.scratchpad_count || 0;
                this.workspace = d.workspace || null;
                this.loaded = true;
            } catch (e) { console.error("debug.load", e); }
        },
        toggleSection(name) { this.openSections[name] = !this.openSections[name]; },
        toggleHypothesis(id) {
            this.expandedHypothesisId = this.expandedHypothesisId === id ? null : id;
        },
        isHypothesisExpanded(id) {
            return this.expandedHypothesisId === id;
        },
        filtered(items) {
            return (items || []).filter(item => Alpine.store("modeWorkspace").matches(
                item.content, item.tags, item.source, item.kind, item.status
            ));
        },
        statusGlyph(status) {
            switch ((status || "").toLowerCase()) {
                case "confirmed": return "✓";
                case "supported": return "◐";
                case "disproved": return "✕";
                case "untested":  return "○";
                default:          return "·";
            }
        },
        formatTimestamp(unix) {
            if (!unix || typeof unix !== "number") return "";
            try { return new Date(unix * 1000).toLocaleString(); }
            catch (e) { return ""; }
        },
    });

    // Context Observatory — current fingerprint + provider-call evolution.
    // One horizontal band per layer (L0..L5); each band gets a fixed hue
    // (so you can tell layers apart) and each cell's *brightness* encodes
    // how often that slice of the layer's text has changed between
    // snapshots (hash-based change frequency). Two render modes share one
    // backend payload (int grid 0=empty / 1..255=1+heat):
    //   heat  — hue=layer, lightness ∝ change frequency (default)
    //   layer — solid per-layer hue, uniform lightness
    // Refreshed each turn by refreshActivePanel() and each iteration by
    // the context_snapshot SSE event (pre_provider_call hook).
    Alpine.store("memory", {
        active: false,
        loaded: false,
        tab: "knowledge",     // durable Memory Ledger | live context observatory
        resolution: 128,
        view: "timeline",    // timeline | stream | churn | fingerprint
        cols: 128,
        rows: 128,
        layers: [],
        regions: [],       // layers plus the explicit available-capacity band
        grid: [],
        totalTokens: 0,
        contextLimit: 0,
        freeTokens: 0,
        fillPct: 0,
        _canvas: null,
        _timelineCanvas: null,
        _timelineObserver: null,
        _renderPending: false,
        timeline: { points: [], summary: {}, selectedId: null },
        timelineHover: { visible: false, x: 0, y: 0, point: null, layer: null },
        cellHover: { visible: false, x: 0, y: 0, row: 0, col: 0, index: 0, cellCount: 0, layer: "", hue: 0, heat: 0, changeCount: 0, tokens: 0, max: 0, chars: 0, content: "", loading: false },
        _hoverTimer: null,
        durable: {
            items: [],
            stats: {},
            query: "",
            scope: "auto",
            kind: "observation",
            newStatement: "",
            loading: false,
            error: "",
            selected: null,
            detail: null,
            recall: null,
        },

        _sessionParam() {
            const chat = Alpine.store("chat");
            const name = chat && chat.currentName ? chat.currentName : "";
            return name ? "session_name=" + encodeURIComponent(name) : "";
        },

        async loadDurable() {
            this.durable.loading = true;
            this.durable.error = "";
            try {
                const params = new URLSearchParams();
                const chat = Alpine.store("chat");
                if (chat && chat.currentName) params.set("session_name", chat.currentName);
                if (this.durable.query) params.set("q", this.durable.query);
                params.set("limit", "200");
                const r = await fetch("/api/v1/memories?" + params.toString());
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail || "memory load failed (" + r.status + ")");
                this.durable.items = d.memories || [];
                this.durable.stats = d.stats || {};
            } catch (e) {
                this.durable.error = String(e || "memory load failed");
            } finally {
                this.durable.loading = false;
            }
        },

        async addMemory() {
            const statement = (this.durable.newStatement || "").trim();
            if (!statement) return;
            this.durable.error = "";
            try {
                const session = this._sessionParam();
                const r = await fetch("/api/v1/memories" + (session ? "?" + session : ""), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        statement: statement,
                        scope: this.durable.scope,
                        kind: this.durable.kind,
                    }),
                });
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail || "memory save failed (" + r.status + ")");
                this.durable.newStatement = "";
                await this.loadDurable();
            } catch (e) {
                this.durable.error = String(e || "memory save failed");
            }
        },

        async openMemory(memory) {
            this.durable.selected = memory;
            this.durable.detail = { loading: true };
            try {
                const session = this._sessionParam();
                const r = await fetch(
                    "/api/v1/memories/" + encodeURIComponent(memory.id)
                    + (session ? "?" + session : "")
                );
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail || "memory detail failed (" + r.status + ")");
                this.durable.detail = d;
            } catch (e) {
                this.durable.detail = { error: String(e || "memory detail failed") };
            }
        },

        async editMemory(memory) {
            const statement = window.prompt("Edit durable memory", memory.statement || "");
            if (statement == null || !statement.trim() || statement.trim() === memory.statement) return;
            try {
                const session = this._sessionParam();
                const r = await fetch(
                    "/api/v1/memories/" + encodeURIComponent(memory.id)
                    + (session ? "?" + session : ""),
                    {
                        method: "PATCH",
                        headers: {
                            "Content-Type": "application/json",
                            "If-Match": memory.etag || "",
                        },
                        body: JSON.stringify({
                            changes: { statement: statement.trim() },
                            reason: "Memory Center edit",
                        }),
                    }
                );
                const d = await r.json();
                if (!r.ok) throw new Error(
                    typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail)
                );
                await this.loadDurable();
            } catch (e) {
                this.durable.error = String(e || "memory edit failed");
            }
        },

        async memoryAction(memory, action) {
            if (action === "forget" && !window.confirm(
                "Forget permanently? Memory content, revisions and search indexes will be purged."
            )) return;
            try {
                const session = this._sessionParam();
                const r = await fetch(
                    "/api/v1/memories/" + encodeURIComponent(memory.id) + "/actions"
                    + (session ? "?" + session : ""),
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "If-Match": memory.etag || "",
                        },
                        body: JSON.stringify({ action: action }),
                    }
                );
                const d = await r.json();
                if (!r.ok) throw new Error(
                    d.detail || "memory " + action + " failed (" + r.status + ")"
                );
                this.durable.detail = null;
                this.durable.selected = null;
                await this.loadDurable();
            } catch (e) {
                this.durable.error = String(e || "memory " + action + " failed");
            }
        },

        async loadRecall() {
            try {
                const session = this._sessionParam();
                const r = await fetch(
                    "/api/v1/memory-recalls/last" + (session ? "?" + session : "")
                );
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail || "no recall receipt yet");
                this.durable.recall = d.receipt;
            } catch (e) {
                this.durable.error = String(e || "recall receipt unavailable");
            }
        },

        bindCanvas(canvas) {
            this._canvas = canvas;
            this.render();
        },

        bindTimelineCanvas(canvas) {
            this._timelineCanvas = canvas;
            if (window.ResizeObserver) {
                if (this._timelineObserver) this._timelineObserver.disconnect();
                this._timelineObserver = new ResizeObserver(() => this._scheduleRender());
                this._timelineObserver.observe(canvas);
            }
            this._scheduleRender();
        },

        async load() {
            await Promise.all([this.loadDurable(), this.loadContext()]);
        },

        async loadContext() {
            const res = this.resolution || 128;
            try {
                const session = this._sessionParam();
                const suffix = session ? "&" + session : "";
                const [stateResponse, timelineResponse] = await Promise.all([
                    fetch("/api/memory/state?cols=" + res + "&rows=" + res + suffix),
                    fetch("/api/memory/timeline?limit=360" + suffix),
                ]);
                const d = await stateResponse.json();
                if (!stateResponse.ok) throw new Error(d.detail || "context state unavailable");
                const history = await timelineResponse.json();
                if (!timelineResponse.ok) throw new Error(history.detail || "context timeline unavailable");
                this._apply(d);
                this.timeline.points = history.points || [];
                this.timeline.summary = history.summary || {};
                if (!this.timeline.selectedId && this.timeline.points.length) {
                    this.timeline.selectedId = this.timeline.points[this.timeline.points.length - 1].id;
                }
                this.loaded = true;
                this._scheduleRender();
            } catch (e) { console.error("memory.load", e); }
        },

        // Switch render mode without a re-fetch — same int grid, different
        // lightness mapping. Triggers a redraw.
        setView(v) {
            if (!["timeline", "stream", "churn", "fingerprint"].includes(v)) return;
            this.view = v;
            this._scheduleRender();
        },

        // Apply a snapshot (from REST load() or a live SSE event) and
        // schedule a single render. Coalesces rapid iteration bursts via
        // requestAnimationFrame so we never draw more than once per frame.
        applySnapshot(snap) {
            this._apply(snap);
            if (snap.timeline_point && snap.timeline_point.id) {
                this.mergeTimelinePoint(snap.timeline_point);
            }
            this._scheduleRender();
        },

        mergeTimelinePoint(point) {
            const points = this.timeline.points || [];
            const index = points.findIndex(item => item.id === point.id);
            if (index >= 0) points.splice(index, 1, point);
            else points.push(point);
            if (points.length > 360) points.splice(0, points.length - 360);
            this.timeline.points = points;
            this.timeline.selectedId = point.id;
            this.recomputeTimelineSummary();
        },

        recomputeTimelineSummary() {
            const points = this.timeline.points || [];
            if (!points.length) {
                this.timeline.summary = { samples: 0, net_delta: 0, compactions: 0 };
                return;
            }
            const churn = {};
            for (const point of points) {
                for (const layer of (point.layers || [])) {
                    churn[layer.id] = (churn[layer.id] || 0) + (layer.changed_chunks || 0);
                }
            }
            let hottest = null;
            for (const id of Object.keys(churn)) {
                if (!hottest || churn[id] > churn[hottest]) hottest = id;
            }
            this.timeline.summary = {
                samples: points.length,
                first_tokens: points[0].total_tokens || 0,
                last_tokens: points[points.length - 1].total_tokens || 0,
                net_delta: (points[points.length - 1].total_tokens || 0) - (points[0].total_tokens || 0),
                peak_tokens: Math.max(...points.map(p => p.total_tokens || 0)),
                peak_fill_pct: Math.max(...points.map(p => p.fill_pct || 0)),
                compactions: points.filter(p => p.compaction).length,
                hottest_layer: hottest && churn[hottest] ? hottest : null,
                hottest_layer_changes: hottest ? churn[hottest] : 0,
                max_churn_score: Math.max(...points.map(p => p.churn_score || 0)),
            };
        },

        selectedTimelinePoint() {
            const points = this.timeline.points || [];
            return points.find(point => point.id === this.timeline.selectedId)
                || points[points.length - 1]
                || null;
        },

        selectTimelinePoint(point) {
            if (point) this.timeline.selectedId = point.id;
        },

        _apply(d) {
            this.active = !!d.active;
            this.cols = d.cols || this.resolution;
            this.rows = d.rows || this.resolution;
            this.layers = d.layers || [];
            this.regions = d.regions || this.layers;
            this.grid = d.grid || [];
            this.totalTokens = d.total_tokens || 0;
            this.contextLimit = d.context_limit || 0;
            this.freeTokens = d.free_tokens || 0;
            this.fillPct = d.fill_pct || 0;
        },

        _scheduleRender() {
            if (this._renderPending) return;
            this._renderPending = true;
            const run = () => {
                this._renderPending = false;
                this.render();
                this.renderTimeline();
            };
            (window.requestAnimationFrame || setTimeout)(run);
        },

        // row index → layer hue. Built once per render from the legend
        // (each region carries row_start/row_end + hue). Regions include the
        // neutral FREE band, so unallocated capacity is rendered rather than
        // being mistaken for a tiny layer or omitted from the map.
        _rowHueMap() {
            const m = new Array(this.rows);
            for (const l of this.regions) {
                const hue = (typeof l.hue === "number") ? l.hue : 0;
                const end = Math.min(l.row_end || 0, this.rows);
                for (let i = l.row_start || 0; i < end; i++) m[i] = hue;
            }
            return m;
        },

        _contextPalette() {
            const light = document.documentElement.getAttribute("data-theme") === "light";
            const styles = getComputedStyle(document.documentElement);
            const css = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
            return {
                light,
                text: css("--text", light ? "#171a20" : "#f2f4f7"),
                textDim: css("--text-dim", light ? "#667080" : "#8b94a3"),
                plotStart: light ? "rgba(89, 109, 255, .055)" : "rgba(99, 113, 255, .055)",
                plotEnd: light ? "rgba(83, 99, 120, .012)" : "rgba(5, 8, 18, .02)",
                grid: light ? "rgba(17, 24, 39, .105)" : "rgba(148, 163, 184, .10)",
                selection: light ? "rgba(23, 26, 32, .68)" : "rgba(255, 255, 255, .78)",
                churnTop: light ? "rgba(198, 51, 108, .27)" : "rgba(245, 92, 146, .48)",
                churnBottom: light ? "rgba(89, 109, 255, .025)" : "rgba(99, 102, 241, .03)",
                churnLine: light ? "rgba(179, 39, 94, .88)" : "rgba(248, 113, 163, .92)",
                compaction: light ? "rgba(16, 132, 111, .76)" : "rgba(45, 212, 191, .72)",
            };
        },

        render() {
            const canvas = this._canvas;
            if (!canvas || !this.grid || !this.grid.length) return;
            const ctx = canvas.getContext("2d");
            const cols = this.cols;
            const rows = this.rows;
            if (canvas.width !== cols || canvas.height !== rows) {
                canvas.width = cols;
                canvas.height = rows;
            }
            ctx.clearRect(0, 0, cols, rows);
            const rowHue = this._rowHueMap();
            const heat = this.view !== "layer";
            const palette = this._contextPalette();
            for (let ri = 0; ri < rows; ri++) {
                const row = this.grid[ri];
                if (!row) continue;
                const hue = rowHue[ri];
                if (hue == null) continue;       // outside any band → transparent
                for (let ci = 0; ci < cols; ci++) {
                    const v = row[ci];
                    if (!v) {
                        // Empty space is still space: a column with no text
                        // renders as a dim, desaturated layer hue so the
                        // band's full extent stays visible.
                        ctx.fillStyle = palette.light
                            ? `hsl(${hue},22%,96%)`
                            : `hsl(${hue},30%,11%)`;
                    } else {
                        // v is 1..255: 1 = present & stable, 255 = churning.
                        const t = (v - 1) / 254;     // 0..1 change frequency
                        const cellLight = palette.light
                            ? (heat ? 91 - 48 * t : 52)
                            : (heat ? 16 + 44 * t : 42);
                        const saturation = palette.light ? 62 : 68;
                        ctx.fillStyle = `hsl(${hue},${saturation}%,${cellLight.toFixed(1)}%)`;
                    }
                    ctx.fillRect(ci, ri, 1, 1);
                }
            }
        },

        _timelineGeometry(canvas) {
            const rect = canvas.getBoundingClientRect();
            const width = Math.max(280, Math.round(rect.width || 0));
            const height = Math.max(220, Math.round(rect.height || 0));
            const dpr = Math.min(2, window.devicePixelRatio || 1);
            if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
                canvas.width = Math.round(width * dpr);
                canvas.height = Math.round(height * dpr);
            }
            const ctx = canvas.getContext("2d");
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            return {
                ctx, width, height,
                left: 48, right: 12, top: 14, bottom: 28,
                plotWidth: width - 60, plotHeight: height - 42,
            };
        },

        _visibleTimelinePoints(width) {
            const points = this.timeline.points || [];
            const maxPoints = Math.max(16, Math.floor((width - 60) / 7));
            return points.slice(-maxPoints);
        },

        _timelineBase(g, points) {
            const { ctx, width, height, left, top, plotWidth, plotHeight } = g;
            const palette = this._contextPalette();
            ctx.clearRect(0, 0, width, height);
            const bg = ctx.createLinearGradient(0, top, 0, top + plotHeight);
            bg.addColorStop(0, palette.plotStart);
            bg.addColorStop(1, palette.plotEnd);
            ctx.fillStyle = bg;
            ctx.fillRect(left, top, plotWidth, plotHeight);
            ctx.strokeStyle = palette.grid;
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = top + plotHeight * i / 4 + .5;
                ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotWidth, y); ctx.stroke();
            }
            ctx.fillStyle = palette.textDim;
            ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.textAlign = "left";
            if (points.length) {
                ctx.fillText("call " + points[0].id, left, height - 8);
                const end = "call " + points[points.length - 1].id;
                ctx.textAlign = "right";
                ctx.fillText(end, left + plotWidth, height - 8);
            }
        },

        renderTimeline() {
            const canvas = this._timelineCanvas;
            if (!canvas || this.view === "fingerprint") return;
            const g = this._timelineGeometry(canvas);
            const points = this._visibleTimelinePoints(g.width);
            this._timelineBase(g, points);
            if (!points.length) {
                g.ctx.fillStyle = this._contextPalette().textDim;
                g.ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
                g.ctx.textAlign = "center";
                g.ctx.fillText("waiting for the first provider call", g.left + g.plotWidth / 2, g.top + g.plotHeight / 2);
                return;
            }
            if (this.view === "stream") this._drawContextStream(g, points);
            else if (this.view === "churn") this._drawContextChurn(g, points);
            else this._drawContextHeatmap(g, points);
            this._drawTimelineSelection(g, points);
        },

        _drawContextHeatmap(g, points) {
            const { ctx, left, top, plotWidth, plotHeight } = g;
            const palette = this._contextPalette();
            const rows = this.layers.length || 8;
            const rowHeight = plotHeight / rows;
            const cellWidth = plotWidth / points.length;
            const fallback = [
                { id: "L0", hue: 210 },
                { id: "L1B", hue: 168 },
                { id: "L2", hue: 280 }, { id: "L3", hue: 25 },
                { id: "L4B", hue: 50 }, { id: "L5", hue: 358 },
            ];
            const legend = this.layers.length ? this.layers : fallback;
            for (let row = 0; row < legend.length; row++) {
                const layerMeta = legend[row];
                const y = top + row * rowHeight;
                ctx.fillStyle = palette.textDim;
                ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
                ctx.textAlign = "right";
                ctx.fillText(layerMeta.id, left - 8, y + rowHeight * .62);
                for (let col = 0; col < points.length; col++) {
                    const pointLayer = (points[col].layers || []).find(item => item.id === layerMeta.id) || {};
                    const ratio = Number(pointLayer.change_ratio || 0);
                    const occupied = Number(pointLayer.tokens || 0) > 0;
                    const alpha = pointLayer.changed
                        ? Math.min(.96, (palette.light ? .38 : .28) + ratio * (palette.light ? .58 : .68))
                        : (occupied ? (palette.light ? .18 : .105) : (palette.light ? .055 : .035));
                    const inset = Math.min(1.5, cellWidth * .14);
                    const lightness = palette.light
                        ? (pointLayer.changed ? 45 : 68)
                        : (pointLayer.changed ? 63 : 48);
                    ctx.fillStyle = `hsla(${layerMeta.hue || 0}, 82%, ${lightness}%, ${alpha})`;
                    ctx.fillRect(
                        left + col * cellWidth + inset,
                        y + 1.5,
                        Math.max(1, cellWidth - inset * 2),
                        Math.max(1, rowHeight - 3),
                    );
                }
            }
        },

        _drawContextStream(g, points) {
            const { ctx, left, top, plotWidth, plotHeight } = g;
            const palette = this._contextPalette();
            const layerIds = ["L0", "L1B", "L2", "L3", "L4B", "L5"];
            const hueById = {};
            for (const layer of this.layers) hueById[layer.id] = layer.hue;
            const peak = Math.max(1, ...points.map(point => point.total_tokens || 0));
            const xAt = index => left + (points.length === 1 ? plotWidth : index * plotWidth / (points.length - 1));
            const cumulative = points.map(() => 0);
            for (const layerId of layerIds) {
                const lower = cumulative.slice();
                const upper = points.map((point, index) => {
                    const layer = (point.layers || []).find(item => item.id === layerId);
                    cumulative[index] += layer ? (layer.tokens || 0) : 0;
                    return cumulative[index];
                });
                ctx.beginPath();
                for (let i = 0; i < points.length; i++) {
                    const y = top + plotHeight - upper[i] / peak * plotHeight;
                    if (i === 0) ctx.moveTo(xAt(i), y); else ctx.lineTo(xAt(i), y);
                }
                for (let i = points.length - 1; i >= 0; i--) {
                    const y = top + plotHeight - lower[i] / peak * plotHeight;
                    ctx.lineTo(xAt(i), y);
                }
                ctx.closePath();
                const hue = hueById[layerId] == null ? 210 : hueById[layerId];
                const fill = ctx.createLinearGradient(0, top, 0, top + plotHeight);
                fill.addColorStop(0, palette.light
                    ? `hsla(${hue}, 72%, 46%, .66)`
                    : `hsla(${hue}, 78%, 62%, .72)`);
                fill.addColorStop(1, palette.light
                    ? `hsla(${hue}, 66%, 72%, .25)`
                    : `hsla(${hue}, 72%, 42%, .30)`);
                ctx.fillStyle = fill;
                ctx.fill();
            }
            ctx.fillStyle = palette.textDim;
            ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.textAlign = "left";
            ctx.fillText(peak.toLocaleString() + " tok peak", left + 6, top + 13);
        },

        _drawContextChurn(g, points) {
            const { ctx, left, top, plotWidth, plotHeight } = g;
            const palette = this._contextPalette();
            const maxChurn = Math.max(10, ...points.map(point => point.churn_score || 0));
            const xAt = index => left + (points.length === 1 ? plotWidth : index * plotWidth / (points.length - 1));
            const yAt = value => top + plotHeight - (value / maxChurn) * plotHeight;
            const gradient = ctx.createLinearGradient(0, top, 0, top + plotHeight);
            gradient.addColorStop(0, palette.churnTop);
            gradient.addColorStop(1, palette.churnBottom);
            ctx.beginPath();
            ctx.moveTo(xAt(0), top + plotHeight);
            points.forEach((point, index) => ctx.lineTo(xAt(index), yAt(point.churn_score || 0)));
            ctx.lineTo(xAt(points.length - 1), top + plotHeight);
            ctx.closePath();
            ctx.fillStyle = gradient;
            ctx.fill();
            ctx.beginPath();
            points.forEach((point, index) => {
                const x = xAt(index), y = yAt(point.churn_score || 0);
                if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.strokeStyle = palette.churnLine;
            ctx.lineWidth = 2;
            ctx.stroke();
            for (let index = 0; index < points.length; index++) {
                if (!points[index].compaction) continue;
                const x = xAt(index);
                ctx.strokeStyle = palette.compaction;
                ctx.setLineDash([3, 4]);
                ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotHeight); ctx.stroke();
                ctx.setLineDash([]);
            }
            ctx.fillStyle = palette.textDim;
            ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.textAlign = "left";
            ctx.fillText(maxChurn.toFixed(1) + "% peak churn", left + 6, top + 13);
        },

        _drawTimelineSelection(g, points) {
            const index = points.findIndex(point => point.id === this.timeline.selectedId);
            if (index < 0) return;
            const cellWidth = g.plotWidth / points.length;
            const x = g.left + (index + .5) * cellWidth;
            g.ctx.strokeStyle = this._contextPalette().selection;
            g.ctx.lineWidth = 1;
            g.ctx.beginPath(); g.ctx.moveTo(x, g.top); g.ctx.lineTo(x, g.top + g.plotHeight); g.ctx.stroke();
        },

        hoverTimeline(event) {
            const canvas = this._timelineCanvas;
            if (!canvas || !(this.timeline.points || []).length) return;
            const rect = canvas.getBoundingClientRect();
            const width = Math.max(280, rect.width);
            const points = this._visibleTimelinePoints(width);
            const plotLeft = 48;
            const plotWidth = width - 60;
            const localX = event.clientX - rect.left;
            if (localX < plotLeft || localX > plotLeft + plotWidth) return this.clearTimelineHover();
            const index = Math.min(points.length - 1, Math.max(0, Math.floor((localX - plotLeft) / plotWidth * points.length)));
            const point = points[index];
            let layer = null;
            if (this.view === "timeline") {
                const localY = event.clientY - rect.top;
                const row = Math.floor((localY - 14) / ((Math.max(220, rect.height) - 42) / 8));
                if (row >= 0 && row < 8) layer = (point.layers || [])[row] || null;
            }
            this.timelineHover = {
                visible: true,
                x: Math.min(width - 210, Math.max(8, localX + 12)),
                y: Math.max(8, event.clientY - rect.top + 12),
                point,
                layer,
            };
        },

        selectTimelineHover() {
            if (this.timelineHover.point) this.selectTimelinePoint(this.timelineHover.point);
        },

        clearTimelineHover() { this.timelineHover.visible = false; },

        hoverCell(event) {
            const canvas = this._canvas;
            if (!canvas || !this.grid.length) return;
            const rect = canvas.getBoundingClientRect();
            const col = Math.min(this.cols - 1, Math.max(0, Math.floor((event.clientX - rect.left) / rect.width * this.cols)));
            const row = Math.min(this.rows - 1, Math.max(0, Math.floor((event.clientY - rect.top) / rect.height * this.rows)));
            const region = (this.regions || []).find(r => row >= r.row_start && row < r.row_end);
            if (!region) return this.clearHover();
            const index = (row - region.row_start) * this.cols + col;
            const key = `${region.id}:${row}:${col}:${this.cols}:${this.rows}`;
            if (this.cellHover.key === key) return;
            if (this._hoverTimer) clearTimeout(this._hoverTimer);
            this.cellHover = { visible: true, key, x: event.clientX - rect.left + 12, y: event.clientY - rect.top + 12, row, col, index, cellCount: (region.row_end - region.row_start) * this.cols, layer: region.id, hue: region.hue || 0, heat: (this.grid[row] || [])[col] || 0, changeCount: region.change_count || 0, tokens: region.tokens || 0, max: region.max || 0, chars: 0, content: "", loading: region.id !== "FREE" };
            if (region.id === "FREE") return;
            this._hoverTimer = setTimeout(async () => {
                try {
                    const session = this._sessionParam();
                    const r = await fetch(`/api/memory/cell?layer=${encodeURIComponent(region.id)}&cols=${this.cols}&rows=${this.rows}&row=${row}&col=${col}${session ? "&" + session : ""}`);
                    const d = await r.json();
                    if (this.cellHover.key === key) Object.assign(this.cellHover, { content: d.content || "", chars: d.chars || 0, loading: false });
                } catch (e) { if (this.cellHover.key === key) this.cellHover.loading = false; }
            }, 120);
        },

        clearHover() {
            if (this._hoverTimer) clearTimeout(this._hoverTimer);
            this._hoverTimer = null;
            this.cellHover.visible = false;
        },

        // ---- Layer-content modal ----
        // Clicking a layer in the legend opens an overlay with the actual
        // text the harness injects for that layer, fetched on demand from
        // /api/memory/content. The body is plain preformatted text so it
        // selects + copies cleanly (the model's real view, not a re-render).
        layerModal: {
            open: false,
            id: "",
            name: "",
            hue: 0,
            content: "",
            tokens: 0,
            chars: 0,
            loading: false,
            error: "",
            copied: false,
        },

        async openLayer(l) {
            // Seed the header immediately from the legend entry so the modal
            // isn't blank while the content fetch is in flight.
            this.layerModal = {
                open: true,
                id: l.id || "",
                name: l.name || l.id || "",
                hue: (typeof l.hue === "number") ? l.hue : 0,
                content: "",
                tokens: l.tokens || 0,
                chars: 0,
                loading: true,
                error: "",
                copied: false,
            };
            try {
                const session = this._sessionParam();
                const r = await fetch(
                    `/api/memory/content?layer=${encodeURIComponent(l.id || "")}${session ? "&" + session : ""}`
                );
                const d = await r.json();
                this.layerModal.content = d.content || "";
                this.layerModal.tokens = (d.tokens != null) ? d.tokens : this.layerModal.tokens;
                this.layerModal.chars = d.chars || 0;
                if (d.name) this.layerModal.name = d.name;
                if (typeof d.hue === "number") this.layerModal.hue = d.hue;
                this.layerModal.error = d.error || "";
            } catch (e) {
                this.layerModal.error = String(e || "fetch failed");
            } finally {
                this.layerModal.loading = false;
            }
        },

        closeLayer() { this.layerModal.open = false; },

        async copyLayer() {
            const text = this.layerModal.content || "";
            if (!text) return;
            // Prefer the async Clipboard API; fall back to a hidden textarea
            // + execCommand for non-secure contexts (http) where it's gated.
            try {
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    await navigator.clipboard.writeText(text);
                } else {
                    const ta = document.createElement("textarea");
                    ta.value = text;
                    ta.style.position = "fixed";
                    ta.style.opacity = "0";
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand("copy");
                    document.body.removeChild(ta);
                }
                this.layerModal.copied = true;
                setTimeout(() => { this.layerModal.copied = false; }, 1500);
            } catch (e) { /* clipboard blocked — selection still works manually */ }
        },
    });

    Alpine.store("confirm", {
        // Small inline confirm-popover anchored near the click coords.
        // Replaces window.confirm() so we don't pop a native modal that
        // blocks the page; just a quiet "are you sure" bubble next to
        // the action button.
        open: false,
        x: 0,
        y: 0,
        message: "",
        danger: false,
        _onConfirm: null,

        ask(message, event, onConfirm, opts) {
            this.message = String(message || "");
            this.danger = !!(opts && opts.danger);
            this._onConfirm = typeof onConfirm === "function" ? onConfirm : null;
            // Anchor to the click coords; clamp inside the viewport so
            // edge-of-screen clicks don't push the popover off-screen.
            const w = window.innerWidth, h = window.innerHeight;
            const px = Math.min((event && event.clientX) || 0, w - 240);
            const py = Math.min((event && event.clientY) || 0, h - 100);
            this.x = Math.max(8, px);
            this.y = Math.max(8, py);
            this.open = true;
        },
        confirm() {
            const cb = this._onConfirm;
            this._onConfirm = null;
            this.open = false;
            if (cb) {
                try { cb(); } catch (e) { console.error("confirm cb", e); }
            }
        },
        cancel() {
            this._onConfirm = null;
            this.open = false;
        },
    });

    Alpine.store("fileBrowser", {
        open: false,
        mode: "folder",       // "folder" or "file"
        path: "",
        parent: "",
        entries: [],
        loading: false,
        _onSelect: null,

        async show(mode, onSelect, startPath) {
            this.mode = mode || "folder";
            this._onSelect = typeof onSelect === "function" ? onSelect : null;
            this.open = true;
            await this.navigate(startPath || "~");
        },
        async navigate(path) {
            this.loading = true;
            try {
                const r = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    console.warn("browse failed:", d.detail || r.status);
                    return;
                }
                const d = await r.json();
                this.path = d.path;
                this.parent = d.parent;
                this.entries = d.entries || [];
            } catch (e) {
                console.error("fileBrowser.navigate", e);
            } finally {
                this.loading = false;
            }
        },
        up() {
            if (this.parent && this.parent !== this.path) this.navigate(this.parent);
        },
        select(entry) {
            if (entry.is_dir && this.mode === "file") {
                this.navigate(entry.path);
                return;
            }
            if (entry.is_dir && this.mode === "folder") {
                this.navigate(entry.path);
                return;
            }
            // File selected in file mode
            if (!entry.is_dir && this.mode === "file") {
                const cb = this._onSelect;
                this._onSelect = null;
                this.open = false;
                if (cb) cb(entry.path);
            }
        },
        selectCurrent() {
            if (this.mode !== "folder") return;
            const cb = this._onSelect;
            this._onSelect = null;
            this.open = false;
            if (cb) cb(this.path);
        },
        cancel() {
            this._onSelect = null;
            this.open = false;
        },
        formatSize(bytes) {
            if (bytes === null || bytes === undefined) return "";
            if (bytes < 1024) return bytes + " B";
            if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " K";
            return (bytes / 1048576).toFixed(1) + " M";
        },

        // --- create folder ---
        creatingFolder: false,
        newFolderName: "",

        startCreateFolder() {
            this.creatingFolder = true;
            this.newFolderName = "";
        },
        async confirmCreateFolder() {
            const name = (this.newFolderName || "").trim();
            if (!name) { this.cancelCreateFolder(); return; }
            try {
                const r = await fetch("/api/browse/mkdir", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: this.path, name }),
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    alert(d.detail || `create folder failed (${r.status})`);
                    return;
                }
                this.creatingFolder = false;
                this.newFolderName = "";
                await this.navigate(this.path);
            } catch (e) {
                console.error("fileBrowser.confirmCreateFolder", e);
            }
        },
        cancelCreateFolder() {
            this.creatingFolder = false;
            this.newFolderName = "";
        },
    });

    Alpine.store("inspector", {
        open: false,
        tab: "workspace",   // workspace | memory | stats | settings
        // ----- workspace
        workspace: { folders: [], cwd: "", newPath: "" },
        async loadWorkspace() {
            const r = await fetch("/api/workspace");
            const d = await r.json();
            this.workspace.folders = d.folders || [];
            this.workspace.cwd = d.cwd || "";
        },
        async addWorkspace() {
            const path = (this.workspace.newPath || "").trim();
            if (!path) return;
            const r = await fetch("/api/workspace", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path }),
            });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                alert(d.detail || `add failed (${r.status})`);
                return;
            }
            this.workspace.newPath = "";
            await this.loadWorkspace();
        },
        async removeWorkspace(p) {
            await fetch(`/api/workspace?path=${encodeURIComponent(p)}`, { method: "DELETE" });
            await this.loadWorkspace();
        },

        // ----- memory
        memory: { taskMemory: [], scratchpad: [], query: "", results: null },
        async loadMemory() {
            const r = await fetch("/api/memory");
            const d = await r.json();
            this.memory.taskMemory = d.task_memory || [];
            this.memory.scratchpad = d.scratchpad || [];
            this.memory.results = null;
        },
        async searchMemory() {
            const q = (this.memory.query || "").trim();
            if (!q) { this.memory.results = null; return; }
            const r = await fetch(`/api/memory/search?q=${encodeURIComponent(q)}`);
            const d = await r.json();
            this.memory.results = d.results || [];
        },
        async deleteMemory(id) {
            await fetch(`/api/memory/${id}`, { method: "DELETE" });
            await this.loadMemory();
            if (this.memory.query) await this.searchMemory();
        },
        async clearMemory() {
            await fetch("/api/memory/clear", { method: "POST" });
            await this.loadMemory();
        },

        // ----- stats
        stats: null,
        async loadStats() {
            const r = await fetch("/api/stats");
            this.stats = await r.json();
        },

        // ----- variables (settings)
        // The server returns {groups:[{name, variables:[...]}]} so the
        // UI can render each category as a collapsible submenu.
        variables: { groups: [] },
        varFilter: "",
        // group name → bool. Collapsed by default; user toggles per group.
        openGroups: {},
        // Default model for spawned sub-agents ("" = inherit parent model).
        // Populated from the variables list so the picker stays in sync
        // with /set and the inspector settings tab.
        subagentModel: "",
        // Ollama local/cloud first-class toggle. `ollamaMode` is the UI
        // selection ("local" | "cloud"); `ollamaApiKey` is the password
        // field value (empty until the user types — the real key is never
        // echoed back by the API); `ollamaKeySet` reflects whether a key
        // is already stored (from the variable entry's `is_set`).
        ollamaMode: "local",
        ollamaApiKey: "",
        ollamaKeySet: false,

        async loadVariables() {
            const r = await fetch("/api/variables");
            const d = await r.json();
            this.variables.groups = d.groups || [];
            // Seed any new groups as collapsed; preserve the user's
            // existing per-group toggle state across refreshes.
            for (const g of this.variables.groups) {
                if (!(g.name in this.openGroups)) this.openGroups[g.name] = false;
            }
            // Reflect the current subagent_model value in the picker.
            this.subagentModel = this._readVariable("subagent_model") || "";
        },
        _readVariable(key) {
            for (const g of this.variables.groups || []) {
                for (const v of g.variables || []) {
                    if (v.key === key) return v.value;
                }
            }
            return null;
        },
        _readVariableEntry(key) {
            for (const g of this.variables.groups || []) {
                for (const v of g.variables || []) {
                    if (v.key === key) return v;
                }
            }
            return null;
        },
        async onSubagentModelChange() {
            // Empty = inherit parent model → reset to the schema default ("").
            const chosen = this.subagentModel;
            const ok = chosen
                ? await this.setVariable("subagent_model", chosen)
                : await this.resetVariable("subagent_model");
            if (ok) {
                Alpine.store("toast").show(
                    chosen
                        ? `subagent model → ${chosen}`
                        : "subagent model → inherit parent",
                    "success",
                );
            }
        },
        toggleGroup(name) {
            this.openGroups[name] = !this.openGroups[name];
        },
        isGroupOpen(name) {
            // While the filter is active, auto-expand any group that has
            // matching entries so the user can actually see the results.
            const q = (this.varFilter || "").trim().toLowerCase();
            if (q) {
                const group = this.variables.groups.find(g => g.name === name);
                if (!group) return false;
                return group.variables.some(v => v.key.toLowerCase().includes(q));
            }
            return !!this.openGroups[name];
        },
        async setVariable(key, value) {
            const r = await fetch(`/api/variables/${encodeURIComponent(key)}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ value }),
            });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(
                    d.detail || `set failed (${r.status})`,
                    "error",
                );
                return false;
            }
            await this.loadVariables();
            return true;
        },
        async resetVariable(key) {
            const r = await fetch(`/api/variables/${encodeURIComponent(key)}`, { method: "DELETE" });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(
                    d.detail || `reset failed (${r.status})`,
                    "error",
                );
                return false;
            }
            await this.loadVariables();
            return true;
        },

        // ----- provider/model switching
        providers: [],
        models: [],
        currentProvider: "",
        currentModel: "",
        async loadProviders() {
            const r = await fetch("/api/providers");
            const d = await r.json();
            this.providers = d.providers || [];
            await this.loadCurrentProvider();
        },
        async loadCurrentProvider() {
            const r = await fetch("/api/providers/current");
            const d = await r.json();
            this.currentProvider = d.provider || "";
            this.currentModel = d.model || "";
            this.ollamaKeySet = !!d.ollama_api_key_set;
            if (this.currentProvider === "ollama") {
                await this._loadOllamaState();
            }
            if (this.currentProvider) {
                await this.loadModels(this.currentProvider);
            }
        },
        async _loadOllamaState() {
            // Ensure the variables list is fresh so we can read the stored
            // ollama_mode / ollama_host / ollama_api_key entries.
            if (!this.variables.groups.length) {
                try { await this.loadVariables(); } catch (_) {}
            }
            const host = this._readVariable("ollama_host") || "";
            const modeVar = this._readVariable("ollama_mode") || "";
            const keyEntry = this._readVariableEntry("ollama_api_key");
            // The endpoint additionally knows about OLLAMA_API_KEY inherited
            // by mucli. Keep that usable without ever returning the secret to
            // the browser; the password field remains visually masked.
            this.ollamaKeySet = this.ollamaKeySet || !!(keyEntry && keyEntry.is_set);
            this.ollamaApiKey = "";
            // Infer the toggle position: an ollama.com host → cloud;
            // else an explicit "cloud" mode → cloud; else "local" (the
            // "auto" legacy default maps to local in the UI).
            if (host.includes("ollama.com")) {
                this.ollamaMode = "cloud";
            } else if (modeVar === "cloud") {
                this.ollamaMode = "cloud";
            } else {
                this.ollamaMode = "local";
            }
        },
        _ollamaDiscoveryParams() {
            // Belt-and-suspenders: pass the chosen mode + key to the
            // discovery endpoint so the dropdown matches the target the
            // running provider will use (the backend also falls back to
            // the active session's variables).
            if (this.currentProvider !== "ollama") return "";
            const parts = [];
            parts.push(`ollama_mode=${encodeURIComponent(this.ollamaMode)}`);
            if (this.ollamaApiKey) {
                parts.push(`ollama_api_key=${encodeURIComponent(this.ollamaApiKey)}`);
            }
            return `?${parts.join("&")}`;
        },
        async loadModels(provider) {
            const suffix = provider === "ollama" ? this._ollamaDiscoveryParams() : "";
            const r = await fetch(
                `/api/providers/${encodeURIComponent(provider)}/models${suffix}`,
            );
            const d = await r.json();
            this.models = d.models || [];
        },
        async onProviderChange() {
            // provider dropdown changed — reload models for new provider
            this.currentModel = "";
            this.models = [];
            if (this.currentProvider === "ollama") {
                await this._loadOllamaState();
            }
            if (this.currentProvider) {
                await this.loadModels(this.currentProvider);
            }
        },
        async onOllamaModeChange(mode) {
            this.ollamaMode = mode;
            // Cloud pins the host to ollama.com; local clears the host
            // override so OLLAMA_HOST env / localhost applies (a custom
            // daemon set via the inspector `ollama_host` field is
            // preserved until the user explicitly picks a mode).
            await this.setVariable("ollama_mode", mode);
            await this.setVariable(
                "ollama_host",
                mode === "cloud" ? "https://ollama.com" : "",
            );
            this.currentModel = "";
            this.models = [];
            // Cloud with no key can't list models — wait for the user to
            // enter one (onOllamaKeyChange reloads).
            if (mode === "cloud" && !this.ollamaApiKey && !this.ollamaKeySet) {
                Alpine.store("toast").show(
                    "cloud selected — enter an API key to list models.",
                    "info",
                );
                return;
            }
            await this.loadModels("ollama");
        },
        async onOllamaKeyChange() {
            // The field is empty on load (the API never echoes the key),
            // so only act when the user actually typed something. An
            // empty submit clears the stored key.
            const value = this.ollamaApiKey || "";
            await this.setVariable("ollama_api_key", value);
            this.ollamaKeySet = value !== "";
            this.currentModel = "";
            this.models = [];
            if (this.ollamaMode === "cloud" && !value) {
                Alpine.store("toast").show(
                    "API key cleared — cloud needs a key to list models.",
                    "info",
                );
                return;
            }
            await this.loadModels("ollama");
        },
        // The active provider's model list, with `extra` (the currently
        // bound value) guaranteed to be present + first. A loaded session's
        // model sometimes isn't in the fetched list (e.g. an elided
        // `:latest` tag, or a transient listing failure) — without this the
        // <select> can't match its bound value and falls back to the "—"
        // placeholder even though a model IS loaded.
        modelOptions(extra) {
            const out = [];
            const seen = new Set();
            const add = (m) => {
                if (m && !seen.has(m)) { seen.add(m); out.push(m); }
            };
            add(extra);
            for (const m of this.models || []) add(m);
            return out;
        },
        async onModelChange() {
            if (!this.currentProvider || !this.currentModel) return;
            const body = {
                provider: this.currentProvider,
                model: this.currentModel,
            };
            // For ollama, carry the chosen mode + key (and the derived
            // host) so the hot-swapped provider starts on the right
            // endpoint with the right auth, and they persist for reload.
            if (this.currentProvider === "ollama") {
                body.ollama_mode = this.ollamaMode;
                body.ollama_host =
                    this.ollamaMode === "cloud" ? "https://ollama.com" : "";
                if (this.ollamaApiKey) body.ollama_api_key = this.ollamaApiKey;
            }
            const r = await fetch("/api/providers/switch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(
                    d.detail || `switch failed (${r.status})`,
                    "error",
                );
                return;
            }
            // Persist the switched model so the picker stays in sync and
            // a reload restores it (the backend already persisted it).
            this.currentModel = (await r.json().catch(() => ({}))).model || this.currentModel;
            Alpine.store("toast").show(
                `model → ${this.currentModel} (${this.currentProvider})`,
                "success",
            );
        },

        // ----- shell
        async openDrawer() {
            // The header control is settings-first; workspace, memory, and
            // stats remain available as tabs in the centered modal.
            this.tab = "settings";
            this.open = true;
            await this.refreshActiveTab();
        },
        close() { this.open = false; },
        async setTab(name) {
            this.tab = name;
            await this.refreshActiveTab();
        },
        async refreshActiveTab() {
            switch (this.tab) {
                case "workspace": await this.loadWorkspace(); break;
                case "memory":    await this.loadMemory(); break;
                case "stats":     await this.loadStats(); break;
                case "settings":  await this.loadVariables(); break;
            }
        },
        filteredVariables() {
            // Returns the same {groups: [...]} shape the template
            // expects. When a filter is active, each group's variables
            // are narrowed to matching keys, and empty groups drop out.
            const q = (this.varFilter || "").trim().toLowerCase();
            if (!q) return this.variables.groups;
            return this.variables.groups
                .map(g => ({
                    name: g.name,
                    variables: g.variables.filter(v => v.key.toLowerCase().includes(q)),
                }))
                .filter(g => g.variables.length > 0);
        },
    });

    // ── TTS (text-to-speech) ─────────────────────────────────────
    Alpine.store("tts", {
        enabled: false,
        playing: false,
        loading: false,
        currentMessageId: null,
        _audio: null,
        _voice: "",

        async load() {
            try {
                const r = await fetch("/api/variables");
                const d = await r.json();
                for (const g of (d.groups || [])) {
                    for (const v of (g.variables || [])) {
                        if (v.key === "tts_enabled") { this.enabled = !!v.value; }
                        if (v.key === "tts_voice") { this._voice = v.value || ""; }
                    }
                }
            } catch (e) { /* non-fatal */ }
        },

        async speak(text, messageId) {
            if (this.playing && this.currentMessageId === messageId) {
                this.stop();
                return;
            }
            this.stop();
            this.loading = true;
            this.currentMessageId = messageId;
            try {
                const resp = await fetch("/api/audio/tts", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text, voice: this._voice || undefined }),
                });
                if (!resp.ok) {
                    const d = await resp.json().catch(() => ({}));
                    Alpine.store("toast").show(d.detail || `TTS failed (${resp.status})`, "error");
                    this.loading = false;
                    this.currentMessageId = null;
                    return;
                }
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                this._audio = new Audio(url);
                this._audio.onended = () => { this._reset(); };
                this._audio.onerror = () => {
                    Alpine.store("toast").show("Audio playback failed", "error");
                    this._reset();
                };
                this.loading = false;
                this.playing = true;
                await this._audio.play();
            } catch (e) {
                Alpine.store("toast").show(`TTS error: ${e}`, "error");
                this._reset();
            }
        },

        stop() {
            if (this._audio) {
                this._audio.pause();
                this._audio.currentTime = 0;
                URL.revokeObjectURL(this._audio.src);
                this._audio = null;
            }
            this._reset();
        },

        _reset() {
            this.playing = false;
            this.loading = false;
            this.currentMessageId = null;
        },

        isSpeaking(messageId) {
            return this.playing && this.currentMessageId === messageId;
        },

        isLoading(messageId) {
            return this.loading && this.currentMessageId === messageId;
        },
    });

    // ── STT (speech-to-text) ──────────────────────────────────────
    Alpine.store("stt", {
        enabled: false,
        recording: false,
        loading: false,
        _mediaRecorder: null,
        _chunks: [],

        async load() {
            try {
                const r = await fetch("/api/variables");
                const d = await r.json();
                for (const g of (d.groups || [])) {
                    for (const v of (g.variables || [])) {
                        if (v.key === "stt_enabled") { this.enabled = !!v.value; }
                    }
                }
            } catch (e) { /* non-fatal */ }
        },

        async toggleRecording() {
            if (this.recording) {
                this.stopRecording();
                return;
            }
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                Alpine.store("toast").show("Microphone not supported in this browser", "error");
                return;
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this._chunks = [];
                this._mediaRecorder = new MediaRecorder(stream);
                this._mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) this._chunks.push(e.data);
                };
                this._mediaRecorder.onstop = () => {
                    this._sendForTranscription();
                    stream.getTracks().forEach(t => t.stop());
                };
                this._mediaRecorder.start();
                this.recording = true;
            } catch (e) {
                Alpine.store("toast").show(`Microphone access denied: ${e}`, "error");
            }
        },

        stopRecording() {
            if (this._mediaRecorder && this._mediaRecorder.state !== "inactive") {
                this._mediaRecorder.stop();
            }
            this.recording = false;
        },

        async _sendForTranscription() {
            if (this._chunks.length === 0) return;
            this.loading = true;
            try {
                const blob = new Blob(this._chunks, { type: this._mediaRecorder.mimeType || "audio/webm" });
                const formData = new FormData();
                formData.append("audio", blob, "recording.webm");
                const resp = await fetch("/api/audio/stt", {
                    method: "POST",
                    body: formData,
                });
                if (!resp.ok) {
                    const d = await resp.json().catch(() => ({}));
                    Alpine.store("toast").show(d.detail || `STT failed (${resp.status})`, "error");
                    this.loading = false;
                    return;
                }
                const data = await resp.json();
                const text = (data.text || "").trim();
                if (text) {
                    const ta = document.querySelector(".composer textarea");
                    if (ta) {
                        const prefix = ta.value && !ta.value.endsWith(" ") && !ta.value.endsWith("\n") ? " " : "";
                        ta.value += prefix + text;
                        ta.dispatchEvent(new Event("input"));
                        ta.focus();
                    }
                }
            } catch (e) {
                Alpine.store("toast").show(`STT error: ${e}`, "error");
            } finally {
                this.loading = false;
                this._chunks = [];
            }
        },
    });

    // ── Files: workspace explorer + editor ─────────────────────
    //
    // Holds tree state + open-file state for the "Files" view panel. The
    // CodeMirror instance itself is owned by the `filesPanel()` component in
    // the panel template (it needs $refs); this store owns the data + the
    // /api/files endpoints. The component calls `setEditorValue()` after
    // openFile and reads `pendingSaveContent()` on save so CM stays the
    // source of truth for what's typed while the store tracks dirty state.
    Alpine.store("files", {
        roots: [],
        selectedPath: "",
        content: "",
        originalContent: "",
        dirty: false,
        readOnly: false,
        readOnlyWhy: "",
        mtime: null,
        loading: false,
        error: "",
        expanded: {},
        // Inline prompt state for new-file / new-folder / rename. Each is a
        // small object the template renders as an input; null = no prompt.
        newPrompt: null,        // { isDir: false, parent: "<path>" }
        renameTarget: null,     // { from: "<path>" }
        confirmDelete: null,    // { path: "<path>", isDir: false }

        async load() {
            await this.refreshTree();
        },

        async refreshTree() {
            this.loading = true;
            this.error = "";
            try {
                const r = await fetch("/api/files/tree");
                if (r.status === 409) {
                    // No workspace — leave roots empty; the panel shows the
                    // empty state with an "attach folder" affordance.
                    this.roots = [];
                    return;
                }
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    this.error = d.detail || `tree failed (${r.status})`;
                    return;
                }
                const d = await r.json();
                this.roots = d.roots || [];
            } catch (e) {
                this.error = String(e);
            } finally {
                this.loading = false;
            }
        },

        async expand(path) {
            if (this.expanded[path]) {
                this.expanded[path] = false;
                this._collapseInTree(path);
                return;
            }
            this.expanded[path] = true;
            // Fetch one level of children for this dir (lazy expand).
            try {
                const r = await fetch(`/api/files/tree?path=${encodeURIComponent(path)}`);
                if (!r.ok) return;
                const d = await r.json();
                this._mergeChildren(path, d.entries || []);
            } catch (e) {
                console.error("files.expand", e);
            }
        },

        // Recursively clear the `children` of a collapsed dir inside the
        // cached tree so a re-expand re-fetches fresh state.
        _collapseInTree(path) {
            const walk = (nodes) => {
                for (const n of nodes || []) {
                    if (n.path === path) { n.children = null; return; }
                    if (n.children) walk(n.children);
                }
            };
            walk(this.roots);
        },

        _mergeChildren(path, entries) {
            const walk = (nodes) => {
                for (const n of nodes || []) {
                    if (n.path === path) { n.children = entries; return true; }
                    if (n.children && walk(n.children)) return true;
                }
                return false;
            };
            walk(this.roots);
        },

        async openFile(path) {
            this.selectedPath = path;
            this.loading = true;
            this.error = "";
            try {
                const r = await fetch(`/api/files/read?path=${encodeURIComponent(path)}`);
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    this.error = d.detail || `read failed (${r.status})`;
                    return;
                }
                const d = await r.json();
                this.content = d.content || "";
                this.originalContent = this.content;
                this.readOnly = !!d.readonly;
                this.readOnlyWhy = d.why || "";
                this.mtime = d.mtime ?? null;
                this.dirty = false;
                // Signal the component to push the new value into CodeMirror.
                this._contentVersion = (this._contentVersion || 0) + 1;
            } catch (e) {
                this.error = String(e);
            } finally {
                this.loading = false;
            }
        },

        closeFile() {
            this.selectedPath = "";
            this.content = "";
            this.originalContent = "";
            this.dirty = false;
            this.readOnly = false;
            this.readOnlyWhy = "";
            this.mtime = null;
            this._contentVersion = (this._contentVersion || 0) + 1;
        },

        // Called by the component on save — it passes CM's current value.
        async save(cmValue) {
            if (!this.selectedPath || this.readOnly) return;
            const path = this.selectedPath;
            try {
                const r = await fetch("/api/files/save", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        path,
                        content: cmValue,
                        expected_mtime: this.mtime,
                    }),
                });
                if (r.status === 409) {
                    Alpine.store("toast").show(
                        "File changed on disk — reload to see the new version.",
                        "warn", 6000,
                    );
                    return;
                }
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    Alpine.store("toast").show(d.detail || `save failed (${r.status})`, "error");
                    return;
                }
                const d = await r.json();
                this.originalContent = cmValue;
                this.content = cmValue;
                this.mtime = d.mtime ?? this.mtime;
                this.dirty = false;
                Alpine.store("toast").show("saved", "success", 2000);
            } catch (e) {
                Alpine.store("toast").show(String(e), "error");
            }
        },

        // New file / folder prompt. `parent` is the dir to create in
        // (defaults to the first root).
        startNew(isDir) {
            const parent = this._defaultParent();
            this.newPrompt = { isDir: !!isDir, parent, name: "" };
        },
        cancelNew() { this.newPrompt = null; },
        async confirmNew() {
            if (!this.newPrompt) return;
            const name = (this.newPrompt.name || "").trim();
            if (!name) { this.newPrompt = null; return; }
            const path = this.newPrompt.parent
                ? this.newPrompt.parent + "/" + name
                : name;
            const isDir = this.newPrompt.isDir;
            this.newPrompt = null;
            const r = await fetch("/api/files/create", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path, is_dir: isDir }),
            });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(d.detail || "create failed", "error");
                return;
            }
            await this._afterFsChange(this.newPrompt ? null : path);
            if (!isDir) await this.openFile(path);
        },

        startRename(path) {
            this.renameTarget = { from: path, to: path };
        },
        cancelRename() { this.renameTarget = null; },
        async confirmRename() {
            if (!this.renameTarget) return;
            const to = (this.renameTarget.to || "").trim();
            const from = this.renameTarget.from;
            this.renameTarget = null;
            if (!to || to === from) return;
            const r = await fetch("/api/files/rename", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ from, to }),
            });
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(d.detail || "rename failed", "error");
                return;
            }
            // If the open file was the one renamed, follow it to the new path.
            if (this.selectedPath === from) {
                this.selectedPath = to;
                this.originalContent = this.content;
            }
            await this._afterFsChange(to);
        },

        askDelete(path, isDir, ev) {
            this.confirmDelete = { path, isDir };
            Alpine.store("confirm").ask(
                isDir ? `Delete directory '${path}'?` : `Delete file '${path}'?`,
                ev, () => this.confirmDeleteGo(), { danger: true },
            );
        },
        async confirmDeleteGo() {
            const t = this.confirmDelete;
            this.confirmDelete = null;
            if (!t) return;
            const recursive = t.isDir;
            const r = await fetch(
                `/api/files?path=${encodeURIComponent(t.path)}` +
                (recursive ? "&recursive=true" : ""),
                { method: "DELETE" },
            );
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                Alpine.store("toast").show(d.detail || "delete failed", "error");
                return;
            }
            if (this.selectedPath === t.path ||
                (this.selectedPath && this.selectedPath.startsWith(t.path + "/"))) {
                this.closeFile();
            }
            await this._afterFsChange();
        },

        _defaultParent() {
            if (this.selectedPath) {
                // If a dir is selected, create in it; if a file, in its dir.
                return this.selectedPath;
            }
            return this.roots[0] ? this.roots[0].path : "";
        },

        async _afterFsChange(focusPath) {
            // Refresh the tree, then try to re-expand the relevant dir so the
            // new/renamed entry is visible. Cheap: just refetch the top tree.
            await this.refreshTree();
        },
    });

    // ── History search ───────────────────────────────────────────
    Alpine.store("history", {
        query: "",
        role: "",
        tool_name: "",
        results: [],
        loading: false,
        hasMore: false,
        totalMatches: 0,
        loaded: false,
        searched: false,
        error: "",
        openSections: { results: true },

        toggleSection(name) {
            this.openSections[name] = !this.openSections[name];
        },

        async search() {
            const q = (this.query || "").trim();
            if (!q) { return; }
            this.loading = true;
            this.error = "";
            try {
                const params = new URLSearchParams({ query: q, max_results: 20 });
                if (this.role) { params.set("role", this.role); }
                if (this.tool_name) { params.set("tool_name", this.tool_name); }
                const r = await fetch(`/api/chat/history/search?${params}`);
                if (!r.ok) {
                    this.error = `Search failed (${r.status})`;
                    this.results = [];
                    this.loading = false;
                    return;
                }
                const data = await r.json();
                this.results = data.results || [];
                this.totalMatches = data.total_matches || 0;
                this.hasMore = !!data.has_more;
                this.loaded = true;
                this.searched = true;
            } catch (e) {
                this.error = `Search error: ${e}`;
                this.results = [];
            } finally {
                this.loading = false;
            }
        },

        clearResults() {
            this.results = [];
            this.totalMatches = 0;
            this.hasMore = false;
        },
    });

    // ── System Prompts editor ──────────────────────────────────────────
    Alpine.store("systemPrompts", {
        items: [],
        selected: null,        // currently-selected prompt name
        text: "",              // editor textarea content
        rawFile: null,         // raw file content (frontmatter + body)
        source: "",            // "file" | "hardcoded" | "override"
        path: null,            // file path on disk
        version: null,
        chars: 0,
        validation: [],
        loading: false,
        saving: false,
        error: "",
        dirty: false,
        openSections: { list: true, editor: true },

        toggleSection(name) {
            this.openSections[name] = !this.openSections[name];
        },

        async load() {
            this.loading = true;
            this.error = "";
            try {
                const r = await fetch("/api/system-prompts");
                if (!r.ok) { this.error = `Failed to load (${r.status})`; return; }
                const data = await r.json();
                this.items = data.items || [];
                // Auto-select first item if nothing selected
                if (!this.selected && this.items.length > 0) {
                    await this.select(this.items[0].name);
                }
            } catch (e) {
                this.error = `Load error: ${e}`;
            } finally {
                this.loading = false;
            }
        },

        async select(name) {
            this.selected = name;
            this.loading = true;
            this.error = "";
            this.dirty = false;
            try {
                const r = await fetch(`/api/system-prompts/${encodeURIComponent(name)}`);
                if (!r.ok) { this.error = `Failed to load prompt (${r.status})`; return; }
                const data = await r.json();
                this.text = data.text || "";
                this.rawFile = data.raw_file || null;
                this.source = data.source || "";
                this.path = data.path || null;
                this.version = data.version ?? null;
                this.chars = data.chars || 0;
                this.validation = data.validation || [];
            } catch (e) {
                this.error = `Select error: ${e}`;
            } finally {
                this.loading = false;
            }
        },

        markDirty() {
            this.dirty = true;
            this.chars = (this.text || "").length;
        },

        async save() {
            if (!this.selected) return;
            this.saving = true;
            this.error = "";
            try {
                const r = await fetch(`/api/system-prompts/${encodeURIComponent(this.selected)}`, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: this.text }),
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    this.error = d.detail || `Save failed (${r.status})`;
                    return;
                }
                const data = await r.json();
                this.validation = data.validation || [];
                this.path = data.path || this.path;
                this.dirty = false;
                // Refresh the list to pick up updated source/version
                await this.load();
                // Re-select to refresh metadata
                await this.select(this.selected);
            } catch (e) {
                this.error = `Save error: ${e}`;
            } finally {
                this.saving = false;
            }
        },

        async reload() {
            this.error = "";
            try {
                await fetch("/api/system-prompts/reload", { method: "POST" });
                if (this.selected) { await this.select(this.selected); }
            } catch (e) {
                this.error = `Reload error: ${e}`;
            }
        },

        async init(name) {
            this.error = "";
            try {
                const body = { force: false };
                if (name) { body.names = [name]; }
                const r = await fetch("/api/system-prompts/init", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    this.error = d.detail || `Init failed (${r.status})`;
                    return;
                }
                const data = await r.json();
                // Refresh list and re-select
                await this.load();
                if (this.selected) { await this.select(this.selected); }
            } catch (e) {
                this.error = `Init error: ${e}`;
            }
        },

        async reset() {
            if (!this.selected) return;
            this.error = "";
            try {
                const r = await fetch(`/api/system-prompts/${encodeURIComponent(this.selected)}/reset`, {
                    method: "POST",
                });
                if (!r.ok) {
                    const d = await r.json().catch(() => ({}));
                    this.error = d.detail || `Reset failed (${r.status})`;
                    return;
                }
                await this.load();
                if (this.selected) { await this.select(this.selected); }
            } catch (e) {
                this.error = `Reset error: ${e}`;
            }
        },

        sourceLabel() {
            if (this.source === "override") return "runtime override";
            if (this.source === "file") return "file override";
            if (this.source === "hardcoded") return "hardcoded fallback";
            return this.source || "—";
        },

        hasValidationIssues() {
            return this.validation && this.validation.length > 0;
        },
    });
});

function escapeHtml(s) {
    return String(s || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
    if (typeof marked === "undefined") return escapeHtml(text);
    try {
        const html = marked.parse(text, { breaks: true, gfm: true });
        // x-html content is not compiled by Alpine, so a delegated listener
        // handles these controls after each streamed markdown re-render.
        return html.replace(
            /<pre><code([^>]*)>/g,
            '<pre class="code-block"><button class="code-copy-btn" type="button" title="Copy code" aria-label="Copy code"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg></button><code$1>'
        );
    } catch {
        return escapeHtml(text);
    }
}

async function copyToClipboard(text) {
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
        } else {
            const textarea = document.createElement("textarea");
            textarea.value = text;
            textarea.style.cssText = "position:fixed;opacity:0";
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand("copy");
            textarea.remove();
        }
        return true;
    } catch (_) {
        return false;
    }
}

document.addEventListener("click", async (event) => {
    const button = event.target.closest(".code-copy-btn");
    if (!button) return;
    const code = button.parentElement && button.parentElement.querySelector("code");
    if (!code || !(await copyToClipboard(code.textContent || ""))) return;
    button.classList.add("copied");
    button.title = "Copied";
    setTimeout(() => {
        button.classList.remove("copied");
        button.title = "Copy code";
    }, 1500);
});

// Alias for clarity inside the modal — same rendering, just named so
// the call site reads intent ("block markdown for the title/desc").
function renderMarkdownBlock(text) { return renderMarkdown(text); }

// Inline-only render: no <p> wrapper, no block elements. Used for
// option labels where backtick `code` should render but we don't want
// paragraph layout.
function renderMarkdownInline(text) {
    if (typeof marked === "undefined") return escapeHtml(text);
    try {
        return marked.parseInline(String(text || ""), { gfm: true });
    } catch {
        return escapeHtml(text);
    }
}

function highlightAll() {
    if (typeof hljs === "undefined") return;
    document.querySelectorAll("pre code").forEach((el) => {
        if (!el.dataset.highlighted) {
            try {
                hljs.highlightElement(el);
                el.dataset.highlighted = "1";
            } catch {}
        }
    });
}

function highlightInScope(selector) {
    if (typeof hljs === "undefined") return;
    document.querySelectorAll(`${selector} pre code`).forEach((el) => {
        if (!el.dataset.highlighted) {
            try {
                hljs.highlightElement(el);
                el.dataset.highlighted = "1";
            } catch {}
        }
    });
}

function enhanceHistoryRoots(roots) {
    const safeRoots = Array.from(roots || []).filter(Boolean);
    if (!safeRoots.length) return Promise.resolve();

    if (typeof hljs !== "undefined") {
        const codeNodes = new Set();
        for (const root of safeRoots) {
            root.querySelectorAll("pre code").forEach(node => codeNodes.add(node));
        }
        for (const node of codeNodes) {
            if (node.dataset.highlighted) continue;
            try {
                hljs.highlightElement(node);
                node.dataset.highlighted = "1";
            } catch {}
        }
    }

    const mathNodes = new Set();
    for (const root of safeRoots) {
        if (root.matches(".msg.assistant .body")) mathNodes.add(root);
        root.querySelectorAll(".msg.assistant .body").forEach(node => mathNodes.add(node));
    }
    return typesetMath(Array.from(mathNodes));
}

function enhanceRenderedTurns(turnIds) {
    const ids = new Set(Array.from(turnIds || [], value => String(value)));
    if (!ids.size) return Promise.resolve();
    const roots = Array.from(
        document.querySelectorAll(".chat-history .turn-wrap[data-turn-id]"),
    ).filter(node => ids.has(String(node.dataset.turnId || "")));
    return enhanceHistoryRoots(roots);
}

function enhanceHistoryRange(startIndex, endIndex) {
    const start = Number(startIndex);
    const end = Number(endIndex);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        return Promise.resolve();
    }
    const roots = Array.from(
        document.querySelectorAll(".chat-history .turn-wrap[data-history-index]"),
    ).filter(node => {
        const index = Number(node.dataset.historyIndex);
        return Number.isInteger(index) && index >= start && index < end;
    });
    return enhanceHistoryRoots(roots);
}

// --- MathJax math typesetting ---------------------------------------------
// MathJax is loaded (vendored) in base.html with startup.typeset=false, so
// the only place math gets typeset is where we ask it to. We scope each
// call to a subtree (the freshly-rendered assistant turn, the modal, the
// reloaded history) rather than re-scanning the whole document — that
// keeps streaming responsive and avoids re-typesetting stable turns.
//
// `skipHtmlTags:['pre','code',...]` in the MathJax config keeps `$` inside
// code blocks from being mistaken for math delimiters, so this is safe to
// run over a turn that mixes prose math and fenced code.
let _mathjaxBusy = Promise.resolve();
function typesetMath(elements) {
    const mj = window.MathJax;
    if (!mj || typeof mj.typesetPromise !== "function") return Promise.resolve();
    if (!elements || !elements.length) return Promise.resolve();
    // Serialize calls so a streaming delta doesn't kick off overlapping
    // typesets on the same node (MathJax chokes if a node is mid-typeset
    // when another typeset starts on it).
    _mathjaxBusy = _mathjaxBusy.then(() => {
        try {
            return mj.typesetPromise(elements);
        } catch (e) {
            console.warn("typesetMath", e);
        }
    });
    return _mathjaxBusy;
}
// Convenience: typeset everything inside a selector. Used after a full
// history reload or modal mount where we don't have a node ref handy.
function typesetMathInScope(selector) {
    if (typeof window.MathJax === "undefined") return;
    const nodes = document.querySelectorAll(selector);
    if (nodes.length) typesetMath(Array.from(nodes));
}

function bootSSE() {
    let hasConnectedBefore = false;
    let source = null;
    let reconnectTimer = null;
    let pingTimer = null;
    // Round-16 F22: exponential backoff with jitter for SSE reconnects.
    // The fixed 2s retry made every open browser reconnect in synchrony
    // after an outage and stampede the server with reconciliation fetches.
    let reconnectAttempt = 0;
    let stabilityTimer = null;
    // Round-48 F5: set while connect() deliberately replaces the stream
    // (focus change) so onerror doesn't schedule a retry loop.
    let reconnectIntentional = false;
    const RECONNECT_BASE_MS = 1000;
    const RECONNECT_MAX_MS = 30000;
    // Round-17 F27: the ladder resets only after the connection has STAYED
    // open for this long — a proxy that completes the handshake then drops
    // the stream would otherwise keep resetting to base forever.
    const RECONNECT_STABLE_MS = 10000;

    // Round-48 F5: focus-driven SSE reconnection. The stream is
    // server-filtered to the focused session (r47 F3) — switching focus
    // must replace the stream with the new filter. Deliberate close: the
    // backoff ladder is NOT advanced (the onerror retry path treats an
    // intentional close as a failure otherwise).
    function reconnectWithFocus() {
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        reconnectIntentional = true;
        try {
            connect();
        } finally {
            reconnectIntentional = false;
        }
        reconnectAttempt = 0;
    }

    function connect() {
        if (source) { try { source.close(); } catch {} source = null; }
        // Connection-alive watchdog. The server only emits a `ping` event
        // when its event queue goes idle for 15s — during a busy agent run
        // (constant tool calls / deltas) it NEVER pings. So the watchdog
        // must be reset on ANY server activity (message OR ping), not just
        // ping, or a long busy turn falsely trips "reconnecting…" every 45s.
        function bumpWatchdog() {
            if (pingTimer) clearTimeout(pingTimer);
            pingTimer = setTimeout(() => {
                // No activity at all in 45s — connection is actually dead.
                if (source) source.dispatchEvent(new Event("error"));
            }, 45000);
        }
        // Round-47 F3: when a tab is focused on one session, ask the bus to
        // filter server-side — the tab no longer parses/routes every other
        // session's events (at scale, background loops would tax every tab).
        const focused = Alpine.store("chat").currentName;
        source = new EventSource("/api/events" + (focused ? ("?session=" + encodeURIComponent(focused)) : ""));
        source.onopen = () => {
            const chat = Alpine.store("chat");
            chat.connected = true;
            chat.lastOpenAt = Date.now();
            // Round-16 F22 / Round-17 F27: reset the backoff ladder only
            // after the connection STAYS open for RECONNECT_STABLE_MS —
            // a flapping proxy that handshakes then drops must not keep
            // resetting the ladder to base. onerror cancels the pending
            // reset.
            if (stabilityTimer) clearTimeout(stabilityTimer);
            stabilityTimer = setTimeout(() => {
                stabilityTimer = null;
                reconnectAttempt = 0;
            }, RECONNECT_STABLE_MS);
            bumpWatchdog();
            // On reconnect (not initial boot), re-sync state in case we
            // missed events while the connection was down.
            if (hasConnectedBefore) {
                chat.loadHistory();
                Alpine.store("sessions").load();
                refreshActivePanel();
            }
            hasConnectedBefore = true;
        };
        source.onmessage = (ev) => {
            // Any message means the connection is alive — push the
            // watchdog out, even before we try to parse the payload.
            bumpWatchdog();
            let data;
            try { data = JSON.parse(ev.data); } catch { return; }
            if (data && data.kind === "hello") {
                const busyNames = data.busy || [];
                const chat = Alpine.store("chat");
                for (const name of busyNames) {
                    const slot = chat._slot(name);
                    slot.busy = true;
                    // A turn is in flight on the server but the live trace
                    // may not exist yet (history reload built only completed
                    // traces). Ensure a running trace so the "thinking" header
                    // + elapsed show immediately instead of waiting for the
                    // next tool event.
                    chat._ensureBusyTrace(slot);
                }
                return;
            }
            routeEvent(data);
        };
        source.addEventListener("ping", bumpWatchdog);
        source.onerror = () => {
            const chat = Alpine.store("chat");
            chat.connected = false;
            // Close the dead source and schedule a fresh connection.
            // EventSource auto-retries, but it can silently give up
            // after repeated failures or if the server returns a
            // non-retryable HTTP status. Taking control ourselves
            // guarantees recovery.
            try { source.close(); } catch {}
            source = null;
            if (pingTimer) { clearTimeout(pingTimer); pingTimer = null; }
            // Round-17 F27: an error before the stability window elapsed
            // cancels the pending ladder reset — flapping connections
            // keep climbing the backoff instead of resetting to base.
            if (stabilityTimer) { clearTimeout(stabilityTimer); stabilityTimer = null; }
            if (reconnectTimer) clearTimeout(reconnectTimer);
            // Round-16 F22: exponential backoff + jitter, capped, so a
            // server outage doesn't synchronize every open browser into a
            // reconnect stampede. Reset to base on stable open (onopen).
            reconnectAttempt = Math.min(reconnectAttempt + 1, 8);
            const backoffMs = Math.min(
                RECONNECT_BASE_MS * Math.pow(2, reconnectAttempt - 1),
                RECONNECT_MAX_MS
            );
            const jitteredMs = Math.round(
                backoffMs * (0.5 + Math.random() * 0.5)
            );
            reconnectTimer = setTimeout(connect, jitteredMs);
        };
    }
    connect();
}

// Refresh the panel store for whichever mode is currently active.
// Called after turn_complete / session_updated so the side-panel
// reflects state the agent just changed. Quiet no-op when the active
// mode has no panel store.
function refreshActivePanel() {
    const mode = Alpine.store("mode");
    if (!mode || !mode.panelModes || !mode.panelModes.includes(mode.active)) return;
    const store = Alpine.store(mode.active);
    if (store && typeof store.load === "function") store.load();
}

function routeEvent(ev) {
    const chat = Alpine.store("chat");
    const prompts = Alpine.store("prompts");
    const tokens = Alpine.store("tokens");
    // Every event carries `session_name` (stamped by WebUI._publish on
    // the server). Route into the matching per-session slot so events
    // from a background session don't clobber the focused one.
    const name = ev.session_name || null;
    const slot = chat._slot(name);
    const isFocused = !name || name === chat.currentName;
    switch (ev.kind) {
        case "user_message": break;  // optimistic local turn; history refresh reconciles other clients
        case "attachment_created":
        case "attachment_deleted":
            if (isFocused) Alpine.store("attachments").load(name);
            break;
        case "context_snapshot": {
            // Live Context Observatory push from the pre_provider_call hook —
            // one per iteration. Only act when the Memory view is the
            // active panel for the focused session, so background turns
            // and other views pay nothing.
            const mode = Alpine.store("mode");
            if (isFocused && mode && mode.active === "memory") {
                Alpine.store("memory").applySnapshot(ev);
            }
            break;
        }
        case "memory_updated": {
            const mode = Alpine.store("mode");
            if (isFocused && mode && mode.active === "memory") {
                Alpine.store("memory").loadDurable();
            }
            break;
        }
        case "mode_changed": {
            if (isFocused) {
                const mode = Alpine.store("mode");
                mode.realMode = ev.mode || "default";
                mode.active = mode.realMode;
                mode.sessionType = ev.session_type || mode.sessionType;
                mode.load();
            }
            break;
        }
        case "assistant_start": chat.startAssistant(ev.turn_id, name); break;
        case "assistant_delta": chat.appendDelta(ev.turn_id, ev.text || "", name); break;
        case "external_write_lost":
            // Round-48 F9: an external surface's write was overwritten by a
            // local save before the watcher could adopt it. Surface the
            // conflict visibly — silent edit loss was the r48 finding.
            try {
                const chatStore = Alpine.store("chat");
                chatStore.pushInfo(`⚠ External edit to ${ev.name || "session"} (rev ${ev.lost_revision}) was overwritten before it could be loaded — reload the session to reconcile.`);
            } catch (e) { console.warn("external_write_lost notify failed", e); }
            break;
        case "assistant_end":
            chat.endAssistant(ev.turn_id, name);
            // Keep busy=true until turn_complete (more tool calls may follow).
            break;
        case "tool_call": chat.addToolCall(ev.tool_name, ev.tool_args, name); break;
        case "subagent_start":
            chat.upsertSubagent(name, {
                task_id: ev.task_id, task: ev.task || "", depth: ev.depth || 1,
                title: ev.title || "",
                model: ev.model || "", status: "running",
                specialist_key: ev.specialist_key || "",
                iter: ev.iter, max_iter: ev.max_iter,
                batch_id: ev.batch_id || "",
            });
            break;
        case "subagent_progress":
            // Merge live fields onto the matching agent row.
            chat.upsertSubagent(name, {
                task_id: ev.task_id,
                task: ev.task,
                title: ev.title,
                depth: ev.depth,
                model: ev.model,
                specialist_key: ev.specialist_key,
                tool_count: ev.tool_count,
                last_tool: ev.last_tool,
                status: ev.status,
                stuck: ev.stuck,
                stall: ev.stall,
                repeat_count: ev.repeat_count,
                elapsed: ev.elapsed,
                context_pct: ev.context_pct,
                iter: ev.iter,
                max_iter: ev.max_iter,
                tokens_in: ev.tokens_in,
                action: ev.action,
                batch_id: ev.batch_id || "",
            });
            break;
        case "subagent_end":
            chat.upsertSubagent(name, {
                task_id: ev.task_id,
                task: ev.task,
                title: ev.title,
                depth: ev.depth,
                model: ev.model,
                specialist_key: ev.specialist_key,
                status: ev.status || "done",
                summary: ev.summary || "",
                kill_reason: ev.kill_reason || null,
                error: ev.error || null,
                elapsed: ev.elapsed,
                tokens_in: ev.tokens && ev.tokens.in ? ev.tokens.in : undefined,
                batch_id: ev.batch_id || "",
                artifact: ev.artifact || null,
            });
            chat.finishSubagents(name);
            break;
        case "subagent_snapshot":
            // Authoritative reconciliation from the registry each parent iter.
            chat.replaceSubagentSnapshot(name, ev.children || [], ev.batch_id || "");
            break;
        case "thinking_delta": chat.addThinking(ev.text || "", name); break;
        case "tool_result":
            chat.addToolResult(ev.tool_name || "", ev.text || "", name);
            break;
        case "tool_result_cache":
            chat.setLastToolResultCache(ev.tool_name || "", ev.cache_key || "", name);
            break;
        case "artifact_created":
            Alpine.store("artifacts").add(ev.artifact, name);
            chat.addVisualization(ev.artifact, name);
            break;
        case "info": chat.addInfo(ev.text || "", null, name); break;
        case "error":
            chat.addError(ev.text || "", name);
            slot.busy = false;
            chat.finishTurn(name);
            break;
        case "status_start": break;  // spinner status; trace dots cover it
        case "status_end": break;
        case "prompt":
            // Prompts are session-scoped; the queue carries name so the
            // inline card only renders in its own session's chat.
            prompts.push({ id: ev.id, prompt: ev.prompt, session_name: name });
            break;
        case "prompt_resolved":
        case "prompt_cancelled": break;
        case "turn_complete":
            // A terminal error is carried on turn_complete as well as the
            // immediate error event so reconnecting clients cannot remain
            // stuck in a busy state or lose the failure detail.
            if (ev.result && ev.result.status === "error" && ev.result.error) {
                const previousError = chat._lastByRole(slot, "error");
                if (!previousError || previousError.text !== String(ev.result.error)) {
                    chat.addError(String(ev.result.error), name);
                }
            }
            // Tokens are global (current session's totals) — only update
            // the meter when the focused session is the one that completed.
            if (isFocused && ev.result && ev.result.tokens) {
                Object.assign(tokens, ev.result.tokens);
            }
            slot.busy = false;
            chat.finishTurn(name);
            if (ev.result && ev.result.status === "interrupted") {
                chat.addInfo(
                    "Execution paused. Type `/continue` to resume, or enter a new prompt.",
                    null, name
                );
            }
            // Refresh the active panel store so kanban/curriculum/etc.
            // reflect whatever the just-finished turn changed.
            if (isFocused) refreshActivePanel();
            break;
        case "history_refresh":
            chat.loadHistory(name, { force: true });
            if (isFocused) {
                Alpine.store("artifacts").load(name, true);
                refreshActivePanel();
            }
            break;
        case "command_result":
            chat.addCommandResult(ev.result, name);
            slot.busy = false;
            chat.finishTurn(name);
            Alpine.store("cmdComplete").invalidateCache();
            // Slash commands like /feature new mutate the session's
            // panel state without triggering a model turn, so the
            // turn_complete hook never fires. Refresh here too.
            if (isFocused) {
                if (ev.result && ev.result.data && ev.result.data.current_mode) {
                    // `/mode <name>` mutates the session directly rather than
                    // passing through POST /api/modes, so it has no separate
                    // mode_changed event. Reload the execution boundary and
                    // the matching Mode OS workspace immediately.
                    Alpine.store("mode").load();
                }
                refreshActivePanel();
                Alpine.store("yolo").load();
                Alpine.store("skills").load();
            }
            break;
        case "diff":
            chat.addInfo(`diff proposed: ${ev.filename}`, null, name);
            break;
        case "session_updated":
            slot.externalActive = true;
            // The server only reloads its in-memory copy when the session
            // ISN'T busy (watcher defers mid-turn). When it deferred
            // (reloaded=false), we must not reload either — doing so would
            // wipe an in-flight assistant turn that isn't saved yet.
            // loadHistory() also guards this (defers when slot.busy), but
            // skip the fetch entirely and say so in the banner.
            if (ev.reloaded === false) {
                chat.addInfo(
                    "↻ another mucli process updated this session — " +
                    "history will refresh when the current turn finishes",
                    { standalone: true }, name
                );
            } else {
                chat.addInfo(
                    "↻ another mucli process updated this session — reloading history",
                    { standalone: true }, name
                );
                chat.loadHistory(name);
            }
            if (isFocused) refreshActivePanel();
            break;
    }
}

// ---------- json rendering helper -----------------------------------------
//
// Tool args and tool results often arrive as nested objects or JSON-encoded
// strings. Always emit a multi-line pretty-print + highlight.js classes
// so users can read the structure instead of squinting at a stringified blob.

function renderJSON(val) {
    if (val === undefined || val === null) return null;

    let pretty;
    if (typeof val === "string") {
        const trimmed = val.trim();
        // If the string itself parses as JSON, reformat it pretty.
        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            try {
                pretty = JSON.stringify(JSON.parse(trimmed), null, 2);
            } catch {
                pretty = val;
            }
        } else {
            pretty = val;
        }
    } else {
        try { pretty = JSON.stringify(val, null, 2); }
        catch { pretty = String(val); }
    }

    // Try highlight.js for json; if anything fails fall back to plain text.
    if (typeof hljs !== "undefined") {
        try {
            return hljs.highlight(pretty, {
                language: "json",
                ignoreIllegals: true,
            }).value;
        } catch {}
    }
    return escapeHtml(pretty);
}

function summarizeTrace(t, _tick) {
    // `_tick` ($store.chat.clock) is read by the template binding so Alpine
    // re-evaluates this expression every 500ms while the focused slot is
    // busy — that's what makes the running elapsed counter actually tick.
    // Without it, the elapsed only updates when a trace event is pushed.
    if (!t || !t.events) return "trace";
    const counts = {};
    for (const ev of t.events) counts[ev.kind] = (counts[ev.kind] || 0) + 1;
    const parts = [];
    if (counts.tool_call) parts.push(`${counts.tool_call} tool${counts.tool_call > 1 ? "s" : ""}`);
    if (counts.tool_result) parts.push(`${counts.tool_result} result${counts.tool_result > 1 ? "s" : ""}`);
    if (counts.thinking) parts.push(`${counts.thinking} think`);
    if (counts.info) parts.push(`${counts.info} note${counts.info > 1 ? "s" : ""}`);
    const label = parts.join(" · ") || (t.running ? "thinking" : "trace");
    const dur = t.elapsed
        ? `${t.elapsed}s`
        : (t.running
            ? `${((Date.now() - t.startedAt) / 1000).toFixed(1)}s`
            : null);
    return dur ? `${label} · ${dur}` : label;
}

function _fmtTok(n) {
    n = Number(n) || 0;
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "k";
    return String(n);
}

function subagentElapsed(a, _tick) {
    const elapsed = Number(a && a.elapsed) || 0;
    if (!a || !["running", "stuck", "stall"].includes(a.status)) return elapsed;
    const observedAt = Number(a.observed_at) || Date.now();
    return elapsed + Math.max(0, Date.now() - observedAt) / 1000;
}

function subagentTitle(a) {
    const title = String((a && a.title) || "").replace(/\s+/g, " ").trim();
    if (title) return title;
    const specialist = String((a && a.specialist_key) || "").replace(/[_-]+/g, " ").trim();
    if (specialist) return specialist.replace(/\b\w/g, c => c.toUpperCase()) + " task";
    return "Delegated task";
}

function subagentToolLabel(tool) {
    const value = String(tool || "").trim();
    const labels = {
        apply_patch: "Edit files",
        bash: "Run command",
        get_chunk: "Read source chunk",
        list_dir: "Inspect directory",
        read_file: "Read file",
        search_for_string: "Search code",
        spawn_agent: "Delegate task",
        web_search: "Search the web",
    };
    if (labels[value]) return labels[value];
    if (!value) return "Working";
    return value.replace(/[_-]+/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function subagentActionStatus(action) {
    if (!action) return "";
    if (action.status === "error") return "Failed";
    if (action.status === "done") return "Completed";
    return "In progress";
}

function summarizeSubagentPanel(p, _tick) {
    if (!p || !p.agents || !p.agents.length) return "subagents";
    const n = p.agents.length;
    const parts = [`${n} subagent${n > 1 ? "s" : ""}`];
    if (p.running) {
        const active = p.agents.filter(a => a.status === "running" || a.status === "stuck").length;
        parts.push(active === n ? "running" : `${active} running`);
    } else {
        const done = p.agents.filter(a => a.status === "done").length;
        const killed = p.agents.filter(a => a.status === "killed").length;
        const errored = p.agents.filter(a => a.status === "error").length;
        const bits = [];
        if (done) bits.push("done");
        if (killed) bits.push("killed");
        if (errored) bits.push("errored");
        parts.push(bits.join("/") || "done");
    }
    const elapsed = Math.max(0, ...p.agents.map(a => subagentElapsed(a, _tick)));
    if (elapsed) parts.push(`${elapsed.toFixed(1)}s`);
    const tok = p.agents.reduce((s, a) => s + (a.tokens_in || 0), 0);
    if (tok) parts.push(`${_fmtTok(tok)} tok`);
    return parts.join(" · ");
}

function subagentStatusText(a) {
    switch (a.status) {
        case "running": return a.last_tool ? subagentToolLabel(a.last_tool) : "Starting";
        case "stuck":   return `Stuck${a.repeat_count ? ` · ${a.repeat_count} repeats` : ""}`;
        case "killed":  return `Stopped${a.kill_reason ? ` · ${a.kill_reason}` : ""}`;
        case "error":   return "Failed";
        case "done":    return "Completed";
        default:        return a.status;
    }
}

function eventLabel(ev) {
    switch (ev.kind) {
        case "tool_call":   return "→ tool";
        case "tool_result": return "← result";
        case "thinking":    return "thinking";
        case "info":        return "info";
        default:            return ev.kind;
    }
}

// ---------- prompt modal component ---------------------------------------
//
// Defined at module scope (not inside the modal template) because the
// modal HTML lives inside a `<template x-if>`. The browser treats
// `<template>` content as inert — scripts inside it never execute.
// Putting promptModal() here guarantees it's defined when Alpine
// evaluates `x-data="promptModal()"` on the modal mount.

function promptModal() {
    return {
        // Local mirror of the active prompt — sync() keeps it fresh.
        prompt: {},
        id: null,
        shape: "",
        description: "",
        multi: false,

        // Per-shape local form state.
        value: null,
        text: "",
        otherText: "",
        answers: {},
        remember: false,

        sync(active) {
            // x-effect calls this whenever the bound item changes.
            // The inline prompt cards iterate $store.prompts.queue, so
            // each card's `active` is its own queue entry (not the
            // head). Reset local form state only when the id changes
            // so re-renders don't clobber what the user has typed.
            if (!active) {
                this.id = null;
                this.prompt = {};
                this.shape = "";
                this.description = "";
                this.multi = false;
                return;
            }
            if (active.id === this.id) return;
            const p = active.prompt || {};
            this.id = active.id;
            this.prompt = p;
            this.shape = p.shape || "";
            this.description = String(p.description || p.message || "");
            this.multi = !!p.multi_select;
            this.value = this.multi ? [] : null;
            this.text = String(p.default || "");
            this.otherText = "";
            this.answers = {};
            this.remember = false;
            // Apply syntax highlighting to any code fences in the
            // freshly-rendered title/description/options.
            queueMicrotask(() => highlightInScope(".modal"));
            queueMicrotask(() => typesetMathInScope(".modal"));
        },

        title() {
            switch (this.shape) {
                case "choice":
                case "choices":       return this.prompt.question || "choose";
                case "input":         return "input";
                case "confirm":       return "confirm";
                case "quiz":          return "quiz";
                case "tool_approval": return "approve tool call";
                case "diff":          return "review change";
                default:              return "prompt";
            }
        },
        // Title and description are model-authored. They commonly carry
        // markdown (code fences, inline `code`, lists, etc.) — render
        // them via marked so the user sees a parseable layout instead
        // of a single line of plaintext.
        titleHtml() { return renderMarkdownBlock(this.title()); },
        descriptionHtml() { return renderMarkdownBlock(this.description); },

        okLabel() {
            switch (this.shape) {
                case "tool_approval": return "approve";
                case "confirm":       return "yes";
                case "diff":          return "approve";
                default:              return "ok";
            }
        },
        optLabel(o) { return typeof o === "string" ? o : (o.label || o.name || JSON.stringify(o)); },
        // Inline-only markdown for option labels (so `code` renders but
        // a paragraph wrapper doesn't get inserted).
        optLabelHtml(o) { return renderMarkdownInline(this.optLabel(o)); },
        optValue(o) { return typeof o === "string" ? o : (o.value !== undefined ? o.value : (o.id || o.label)); },
        isChoiceSelected(value) {
            return this.multi
                ? Array.isArray(this.value) && this.value.includes(value)
                : this.value === value;
        },
        setChoice(value, checked) {
            if (!this.multi) {
                this.value = checked ? value : null;
                return;
            }
            const selected = Array.isArray(this.value) ? [...this.value] : [];
            const index = selected.indexOf(value);
            if (checked && index < 0) selected.push(value);
            if (!checked && index >= 0) selected.splice(index, 1);
            this.value = selected;
        },

        // Quiz options: the TUI submits the option STRING (q.options[idx]),
        // so the GUI does the same — graders expect the literal option
        // text, not an index or an id. Falls back gracefully for object
        // options.
        quizOptionValue(o) {
            if (typeof o === "string") return o;
            if (o == null) return "";
            if (o.value !== undefined) return o.value;
            if (o.label !== undefined) return o.label;
            return String(o);
        },
        quizOptionLabel(o) { return this.optLabel(o); },

        // Expose the inline-markdown helper inside the modal scope so
        // x-html template expressions can call it (e.g. for question
        // prompts that contain inline `code`).
        renderMarkdownInline(text) { return renderMarkdownInline(text); },
        formatArgs(args) {
            if (args === undefined || args === null) return "(no arguments)";
            try { return JSON.stringify(args, null, 2); }
            catch { return String(args); }
        },

        async submit() {
            if (!this.id) return;
            let payload;
            switch (this.shape) {
                case "choice":
                case "choices": {
                    const selected = Array.isArray(this.value)
                        ? this.value
                        : (this.value !== null && this.value !== undefined ? [this.value] : []);
                    const hasOther = selected.includes("__other__");
                    const real = selected.filter(v => v !== "__other__");
                    // `choice` is the tool-facing multi/single picker and
                    // returns all selected labels. `choices` is the runtime
                    // recovery picker (`prompt_choices`) and expects one
                    // scalar `value`; returning `selected` there silently
                    // caused it to fall back to retry every time.
                    payload = this.shape === "choices"
                        ? { value: real[0] || (hasOther ? this.otherText : "") }
                        : {
                            selected: real,
                            other_text: hasOther ? this.otherText : "",
                        };
                    break;
                }
                case "input":         payload = { value: this.text }; break;
                case "confirm":       payload = { value: true }; break;
                case "quiz":          payload = { answers: this.answers }; break;
                case "tool_approval": payload = { approved: true, remember: this.remember }; break;
                case "diff":          payload = { approved: true }; break;
                default:              payload = { value: this.value };
            }
            const answered = await Alpine.store("prompts").answer(this.id, payload);
            // Keep the prompt card in place if the POST did not reach the
            // server; otherwise the blocked agent thread has no way to
            // receive a recovery/retry/abort decision.
            if (!answered) return;
            Alpine.store("chat").addPromptResolved({
                shape: this.shape,
                title: this.title(),
                answer: this._summarizeAnswer(payload),
                toolName: this.prompt.tool_name || "",
                cancelled: false,
            });
        },
        async cancel() {
            if (this.id) {
                const cancelled = await Alpine.store("prompts").cancel(this.id);
                if (!cancelled) return;
                Alpine.store("chat").addPromptResolved({
                    shape: this.shape,
                    title: this.title(),
                    answer: "",
                    toolName: this.prompt.tool_name || "",
                    cancelled: true,
                });
            } else {
                Alpine.store("prompts").queue.shift();
            }
        },

        _summarizeAnswer(payload) {
            switch (this.shape) {
                case "choice":
                case "choices": {
                    const sel = (payload && payload.selected) || [];
                    const other = (payload && payload.other_text) || "";
                    const parts = sel.slice();
                    if (other) parts.push(other);
                    return parts.length ? parts.join(", ") : "(none)";
                }
                case "input":         return String((payload && payload.value) || "");
                case "confirm":       return "yes";
                case "quiz": {
                    const a = (payload && payload.answers) || {};
                    const entries = Object.entries(a);
                    if (!entries.length) return "(no answers)";
                    // Show the answers themselves, not just the count.
                    return entries.map(([k, v]) => `${k}: ${v}`).join("; ");
                }
                case "tool_approval": {
                    const tool = this.prompt.tool_name ? ` \`${this.prompt.tool_name}\`` : "";
                    const remember = payload && payload.remember ? " (remembered)" : "";
                    return `approved${tool}${remember}`;
                }
                case "diff":          return "approved";
                default:              return String((payload && payload.value) || "");
            }
        },
    };
}

function autoSize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

// ---------- theme ----------------------------------------------------------
//
// The early-load script in base.html sets data-theme before first paint.
// These helpers run after DOM ready to (a) sync the highlight.js stylesheet
// to the current theme and (b) flip the theme on user toggle.

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const dark = document.getElementById("hljs-dark");
    const light = document.getElementById("hljs-light");
    if (dark)  dark.disabled  = (theme === "light");
    if (light) light.disabled = (theme === "dark");
    document.querySelectorAll("iframe.visualization-frame").forEach((frame) => {
        try {
            frame.contentWindow?.postMessage({ type: "mucli-theme", theme }, "*");
        } catch (e) { /* Sandboxed frames may be between navigations. */ }
    });
    // Canvas pixels do not inherit CSS variables. Redraw observability
    // visualisations after a theme switch so labels, grids, fills and the
    // current fingerprint use the matching contrast palette immediately.
    requestAnimationFrame(() => {
        try {
            const memory = window.Alpine && Alpine.store("memory");
            if (memory) memory._scheduleRender();
        } catch (e) { /* Alpine may not be initialised during first paint. */ }
    });
}

function visualizationThemeUrl(value, theme) {
    if (!value) return "";
    try {
        const url = new URL(value, window.location.origin);
        url.searchParams.set("mucli_theme", theme === "light" ? "light" : "dark");
        return url.origin === window.location.origin
            ? `${url.pathname}${url.search}${url.hash}`
            : url.toString();
    } catch (e) {
        return value;
    }
}

function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") || "dark";
    const next = cur === "dark" ? "light" : "dark";
    try { localStorage.setItem("mucli-theme", next); } catch (e) {}
    applyTheme(next);
}

// Re-highlight code blocks after theme flips so colors apply
// consistently to anything already rendered.
function rehighlightAll() {
    document.querySelectorAll("pre code").forEach((el) => {
        delete el.dataset.highlighted;
    });
    highlightAll();
}

document.addEventListener("DOMContentLoaded", () => {
    applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
    Alpine.store("layout").init();
    bootSSE();
    Alpine.store("chat").loadHistory();
    Alpine.store("sessions").load();
    Alpine.store("tts").load();
    Alpine.store("stt").load();
    // mode.load() preloads the active mode's panel store via panelModes —
    // the panel populates the instant the user lands in a panel mode,
    // no extra mode-flip round-trip needed.
    Alpine.store("mode").load();
    // Todos are a cross-mode visualisation, not loop-mode-only state. Load
    // them independently so the header field is populated in research,
    // teacher, feature, and every other view.
    Alpine.store("loop").load();
    Alpine.store("yolo").load();
    Alpine.store("skills").load();
    Alpine.store("cmdComplete").load();
    Alpine.store("inspector").loadProviders();
    setInterval(() => Alpine.store("sessions").load(), 5000);
    setInterval(() => Alpine.store("loop").load(), 5000);
    // Live clock: bump while ANY session's turn is in flight so the
    // running trace header re-renders its elapsed time. (One global tick
    // is enough — we re-render every slot's clock; backgrounded ones
    // aren't visible but the cost is negligible.)
    setInterval(() => {
        const chat = Alpine.store("chat");
        let anyBusy = false;
        for (const key of Object.keys(chat.chats)) {
            if (chat.chats[key].busy) {
                chat.chats[key].clock = Date.now();
                anyBusy = true;
            }
        }
        // Touching `clock` (the focused-slot getter) is harmless if not busy.
        if (anyBusy) { /* re-render already triggered above */ }
    }, 500);
});

/* ==== Model pricing settings (moved from product.js: runtime store + API calls are not presentation-layer) ==== */
    document.addEventListener('alpine:init', () => {
        Alpine.store('pricingSettings', {
            loaded: false,
            loading: false,
            saving: false,
            dirty: false,
            error: '',
            provider: 'all',
            version: '',
            currency: 'USD',
            unit: 'per_million_tokens',
            models: [],
            configPath: '',
            activeConfigPath: '',
            defaultConfigPath: '',
            usingOverride: false,

            _applyCatalog(data) {
                const catalog = data && typeof data === 'object' ? data : {};
                this.version = String(catalog.version || 'custom');
                this.currency = String(catalog.currency || 'USD');
                this.unit = String(catalog.unit || 'per_million_tokens');
                this.models = Array.isArray(catalog.models)
                    ? catalog.models.map(row => ({ ...row }))
                    : [];
                this.configPath = String(catalog.config_path || '');
                this.activeConfigPath = String(catalog.active_config_path || '');
                this.defaultConfigPath = String(catalog.default_config_path || '');
                this.usingOverride = !!catalog.using_override;
                this.loaded = true;
                this.dirty = false;
                this.error = '';
            },

            async load(force = false) {
                if (this.loaded && !force) return;
                this.loading = true;
                this.error = '';
                try {
                    const response = await fetch('/api/providers/pricing', { cache: 'no-store' });
                    const data = await response.json().catch(() => ({}));
                    if (!response.ok) throw new Error(data.detail || `pricing load failed (${response.status})`);
                    this._applyCatalog(data);
                } catch (error) {
                    this.error = String(error instanceof Error ? error.message : error);
                } finally {
                    this.loading = false;
                }
            },

            filteredModels() {
                const provider = String(this.provider || 'all').toLowerCase();
                if (provider === 'all') return this.models;
                return this.models.filter(row => String(row.provider || '').toLowerCase() === provider);
            },

            providerCount(provider) {
                const target = String(provider || '').toLowerCase();
                return this.models.filter(row => String(row.provider || '').toLowerCase() === target).length;
            },

            displayRate(value) {
                return value === null || value === undefined ? '' : String(value);
            },

            setRate(row, field, raw) {
                if (!row || !field) return;
                const text = String(raw ?? '').trim();
                if (!text) {
                    row[field] = null;
                    this.dirty = true;
                    return;
                }
                const value = Number(text);
                if (!Number.isFinite(value) || value < 0) {
                    this.error = `${field} must be a non-negative number`;
                    return;
                }
                row[field] = value;
                this.error = '';
                this.dirty = true;
            },

            setBilling(row, value) {
                if (!row) return;
                row.billing = String(value || 'unknown');
                this.dirty = true;
            },

            sourceLabel() {
                if (!this.loaded) return 'Pricing registry not loaded';
                if (this.usingOverride) return `Operator override · ${this.activeConfigPath || this.configPath}`;
                return `Packaged defaults · ${this.activeConfigPath || this.defaultConfigPath}`;
            },

            async save() {
                if (this.saving) return;
                this.saving = true;
                this.error = '';
                try {
                    const response = await fetch('/api/providers/pricing', {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            version: this.version || 'operator',
                            currency: this.currency || 'USD',
                            unit: this.unit || 'per_million_tokens',
                            models: this.models.map(row => ({ ...row })),
                        }),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (!response.ok) throw new Error(data.detail || `pricing save failed (${response.status})`);
                    this._applyCatalog(data);
                    Alpine.store('toast').show('Model pricing saved', 'success');
                } catch (error) {
                    this.error = String(error instanceof Error ? error.message : error);
                    Alpine.store('toast').show(this.error, 'error', 7000);
                } finally {
                    this.saving = false;
                }
            },

            async reset() {
                this.saving = true;
                this.error = '';
                try {
                    const response = await fetch('/api/providers/pricing/reset', { method: 'POST' });
                    const data = await response.json().catch(() => ({}));
                    if (!response.ok) throw new Error(data.detail || `pricing reset failed (${response.status})`);
                    this._applyCatalog(data);
                    Alpine.store('toast').show('Model pricing reset to packaged defaults', 'success');
                } catch (error) {
                    this.error = String(error instanceof Error ? error.message : error);
                    Alpine.store('toast').show(this.error, 'error', 7000);
                } finally {
                    this.saving = false;
                }
            },
        });
    });


    function installPricingSettings() {
        const tabs = document.querySelector('.inspector-tabs');
        const body = document.querySelector('.inspector-body');
        if (!tabs || !body || document.getElementById('pricing-settings-pane')) return;

        const tab = document.createElement('button');
        tab.type = 'button';
        tab.setAttribute(':class', "{ active: $store.inspector.tab === 'pricing' }");
        tab.setAttribute('@click', "$store.inspector.setTab('pricing'); $store.pricingSettings.load()");
        tab.setAttribute('role', 'tab');
        tab.textContent = 'pricing';
        const settingsTab = Array.from(tabs.querySelectorAll('button')).find(
            button => button.textContent.trim().toLowerCase() === 'settings'
        );
        tabs.insertBefore(tab, settingsTab || null);

        const pane = document.createElement('section');
        pane.id = 'pricing-settings-pane';
        pane.className = 'settings-pane pricing-settings-pane';
        pane.setAttribute('x-show', "$store.inspector.tab === 'pricing'");
        pane.innerHTML = `
            <div class="pricing-settings-head">
                <div>
                    <h3 class="pricing-settings-title">Model pricing</h3>
                    <p class="pricing-settings-copy">Edit the per-million token rates MuCLI uses for cost accounting. OpenAI, Gemini, and Ollama Cloud all use this registry. Local Ollama remains $0 attributable provider/API cost; host compute is intentionally separate.</p>
                </div>
                <a class="pricing-settings-advanced" href="/static/model_costs.html">Advanced registry</a>
            </div>

            <div class="pricing-settings-meta" x-show="$store.pricingSettings.loaded">
                <span class="pricing-settings-source" x-text="$store.pricingSettings.sourceLabel()" :title="$store.pricingSettings.activeConfigPath || $store.pricingSettings.configPath"></span>
                <span class="pricing-settings-state"
                      :class="{ 'is-override': $store.pricingSettings.usingOverride }"
                      x-text="$store.pricingSettings.usingOverride ? 'override active' : 'defaults'"></span>
            </div>

            <div class="pricing-provider-filter" x-show="$store.pricingSettings.loaded">
                <button type="button" :class="{ active: $store.pricingSettings.provider === 'all' }" @click="$store.pricingSettings.provider = 'all'">all</button>
                <button type="button" :class="{ active: $store.pricingSettings.provider === 'openai' }" @click="$store.pricingSettings.provider = 'openai'">OpenAI <span x-text="$store.pricingSettings.providerCount('openai')"></span></button>
                <button type="button" :class="{ active: $store.pricingSettings.provider === 'gemini' }" @click="$store.pricingSettings.provider = 'gemini'">Gemini <span x-text="$store.pricingSettings.providerCount('gemini')"></span></button>
                <button type="button" :class="{ active: $store.pricingSettings.provider === 'ollama' }" @click="$store.pricingSettings.provider = 'ollama'">Ollama <span x-text="$store.pricingSettings.providerCount('ollama')"></span></button>
                <button type="button" @click="$store.pricingSettings.load(true)">refresh</button>
            </div>

            <div class="pricing-settings-error" x-show="$store.pricingSettings.error" x-text="$store.pricingSettings.error"></div>
            <div class="pricing-settings-empty" x-show="$store.pricingSettings.loading">Loading model pricing…</div>

            <div class="pricing-model-list" x-show="$store.pricingSettings.loaded && !$store.pricingSettings.loading">
                <template x-for="row in $store.pricingSettings.filteredModels()" :key="row.provider + ':' + row.key">
                    <article class="pricing-model-row">
                        <div class="pricing-model-head">
                            <span class="pricing-provider-badge" x-text="row.provider"></span>
                            <span class="pricing-model-name" x-text="row.key" :title="row.key"></span>
                            <select class="pricing-billing-select" :value="row.billing" @change="$store.pricingSettings.setBilling(row, $event.target.value)">
                                <option value="token">token priced</option>
                                <option value="estimated_token">estimated token</option>
                                <option value="local">local / $0 API</option>
                                <option value="unknown">unpriced</option>
                            </select>
                        </div>
                        <div class="pricing-rate-grid">
                            <label class="pricing-rate-field">
                                <span>Input / 1M</span>
                                <input type="number" min="0" step="0.001"
                                       :disabled="row.billing === 'local' || row.billing === 'unknown'"
                                       :value="$store.pricingSettings.displayRate(row.input_per_million)"
                                       @input="$store.pricingSettings.setRate(row, 'input_per_million', $event.target.value)">
                            </label>
                            <label class="pricing-rate-field">
                                <span>Cached input / 1M</span>
                                <input type="number" min="0" step="0.001"
                                       :disabled="row.billing === 'local' || row.billing === 'unknown'"
                                       :value="$store.pricingSettings.displayRate(row.cached_input_per_million)"
                                       @input="$store.pricingSettings.setRate(row, 'cached_input_per_million', $event.target.value)">
                            </label>
                            <label class="pricing-rate-field">
                                <span>Output / 1M</span>
                                <input type="number" min="0" step="0.001"
                                       :disabled="row.billing === 'local' || row.billing === 'unknown'"
                                       :value="$store.pricingSettings.displayRate(row.output_per_million)"
                                       @input="$store.pricingSettings.setRate(row, 'output_per_million', $event.target.value)">
                            </label>
                        </div>
                        <p class="pricing-model-note" x-show="row.billing === 'local'">Local Ollama provider/API cost is recorded as $0. Host GPU/CPU economics are not included.</p>
                        <p class="pricing-model-note" x-show="row.billing === 'unknown'">This model remains unpriced until you select token pricing and enter rates.</p>
                        <p class="pricing-model-note" x-show="row.notes && row.billing !== 'local' && row.billing !== 'unknown'" x-text="row.notes"></p>
                    </article>
                </template>
            </div>

            <div class="pricing-settings-empty" x-show="$store.pricingSettings.loaded && !$store.pricingSettings.loading && $store.pricingSettings.filteredModels().length === 0">No pricing rows for this provider.</div>

            <div class="pricing-settings-actions" x-show="$store.pricingSettings.loaded">
                <span class="pricing-settings-dirty" x-text="$store.pricingSettings.dirty ? 'Unsaved pricing changes' : ($store.pricingSettings.usingOverride ? 'Operator pricing override active' : 'Using packaged defaults')"></span>
                <div class="pricing-settings-buttons">
                    <button type="button" @click="$store.confirm.ask('Reset all model pricing to packaged defaults?', $event, () => $store.pricingSettings.reset(), {danger:true})">reset</button>
                    <button type="button" class="primary" :disabled="!$store.pricingSettings.dirty || $store.pricingSettings.saving" @click="$store.pricingSettings.save()" x-text="$store.pricingSettings.saving ? 'saving…' : 'save pricing'"></button>
                </div>
            </div>
        `;
        body.appendChild(pane);

        if (window.Alpine && typeof Alpine.initTree === 'function') {
            Alpine.initTree(tab);
            Alpine.initTree(pane);
        }
    }


document.addEventListener('DOMContentLoaded', () => {
    if (typeof installPricingSettings === 'function') installPricingSettings();
});
