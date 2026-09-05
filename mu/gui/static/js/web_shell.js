/* MuCLI web shell reliability glue.
 *
 * Keeps session hydration authoritative after the session focus is known,
 * preserves live transcripts across reconnects, and adds lightweight wall-clock
 * timing for completed model turns without changing backend/API semantics.
 */
(function () {
    function installStylesheet(id, href) {
        if (document.getElementById(id)) return;
        const link = document.createElement('link');
        link.id = id;
        link.rel = 'stylesheet';
        link.href = href;
        document.head.appendChild(link);
    }

    installStylesheet('mucli-conversation-css', '/static/css/conversation.css');
    installStylesheet('mucli-pricing-rows-css', '/static/css/pricing_rows.css');

    function hasLiveTranscript(slot) {
        return (slot.turns || []).some(turn => (
            (turn.role === 'assistant' && turn.streaming)
            || (turn.role === 'trace' && turn.running && (turn.events || []).length > 0)
            || (turn.role === 'subagent_panel' && turn.running)
        ));
    }

    function lastFinalAssistant(slot) {
        for (let index = slot.turns.length - 1; index >= 0; index -= 1) {
            const turn = slot.turns[index];
            if (turn.role === 'user') break;
            if (turn.role === 'assistant' && !turn.streaming && String(turn.text || '').trim()) {
                return turn;
            }
        }
        return null;
    }

    function workedFingerprintMap(slot) {
        const map = new Map();
        let userText = '';
        for (const turn of slot.turns || []) {
            if (turn.role === 'user') userText = String(turn.text || '');
            if (turn.role === 'assistant' && turn.workedMs && !turn.streaming) {
                map.set(`${userText}\u0000${String(turn.text || '')}`, Number(turn.workedMs));
            }
        }
        return map;
    }

    function restoreWorkedFingerprints(slot, map) {
        if (!map || !map.size) return;
        let userText = '';
        for (const turn of slot.turns || []) {
            if (turn.role === 'user') userText = String(turn.text || '');
            if (turn.role === 'assistant' && !turn.streaming) {
                const value = map.get(`${userText}\u0000${String(turn.text || '')}`);
                if (value) turn.workedMs = value;
            }
        }
    }

    function formatWorkedDuration(ms) {
        const totalSeconds = Math.max(0, Math.round(Number(ms || 0) / 1000));
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        return minutes > 0
            ? `Worked for ${minutes}m ${seconds}s`
            : `Worked for ${seconds}s`;
    }

    function decorateWorkedDurations(chat, name) {
        const target = name || chat.currentName;
        if (!target || target !== chat.currentName) return;
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if (target !== chat.currentName) return;
            const slot = chat._slot(target);
            const assistants = (slot.turns || []).filter(turn => turn.role === 'assistant' && !turn.streaming);
            const nodes = Array.from(document.querySelectorAll('.chat-history > .turn-wrap > .msg.assistant'));
            assistants.forEach((turn, index) => {
                const node = nodes[index];
                if (!node) return;
                let crumb = node.querySelector(':scope > .turn-worked-breadcrumb');
                if (!turn.workedMs) {
                    if (crumb) crumb.remove();
                    return;
                }
                if (!crumb) {
                    crumb = document.createElement('div');
                    crumb.className = 'turn-worked-breadcrumb';
                    node.appendChild(crumb);
                }
                crumb.textContent = formatWorkedDuration(turn.workedMs);
            });
        }));
    }

    function nullableRate(raw, label) {
        const text = String(raw ?? '').trim();
        if (!text) return null;
        const value = Number(text);
        if (!Number.isFinite(value) || value < 0) {
            throw new Error(`${label} must be a non-negative number`);
        }
        return value;
    }

    function capabilityList(raw, fallback = []) {
        const values = String(raw ?? '').split(',').map(value => (
            value.trim().toLowerCase().replaceAll(' ', '_')
        )).filter(Boolean);
        return values.length ? [...new Set(values)] : [...fallback];
    }

    function installPricingRowMethods(pricing) {
        if (!pricing || pricing.__rowCreationInstalled) return;
        pricing.__rowCreationInstalled = true;
        pricing.addModelRow = function addModelRow(payload) {
            const raw = payload && typeof payload === 'object' ? payload : {};
            const provider = String(raw.provider || '').trim().toLowerCase();
            const key = String(raw.key || '').trim();
            const billing = String(raw.billing || 'token').trim().toLowerCase();
            if (!['openai', 'gemini', 'ollama'].includes(provider)) {
                throw new Error('Choose OpenAI, Gemini, or Ollama');
            }
            if (!key) throw new Error('Model name is required');
            if (!['token', 'estimated_token', 'local', 'unknown'].includes(billing)) {
                throw new Error('Choose a valid billing mode');
            }
            const duplicate = (this.models || []).some(row => (
                String(row.provider || '').toLowerCase() === provider
                && String(row.key || '').toLowerCase() === key.toLowerCase()
            ));
            if (duplicate) throw new Error(`Pricing row already exists for ${provider}/${key}`);

            let inputRate = nullableRate(raw.input_per_million, 'Input rate');
            let cachedRate = nullableRate(raw.cached_input_per_million, 'Cached input rate');
            let outputRate = nullableRate(raw.output_per_million, 'Output rate');
            if (billing === 'local' || billing === 'unknown') {
                inputRate = null;
                cachedRate = null;
                outputRate = null;
            } else if (inputRate === null || outputRate === null) {
                throw new Error('Input and output rates are required for token-priced models');
            }

            const row = {
                provider,
                key,
                billing,
                aliases: [],
                input_modalities: capabilityList(raw.input_modalities, ['text']),
                output_modalities: capabilityList(raw.output_modalities, ['text']),
                capabilities: capabilityList(raw.capabilities),
                input_per_million: inputRate,
                cached_input_per_million: cachedRate,
                output_per_million: outputRate,
                estimated_total_per_million: null,
                context_window: null,
                long_context_cutoff: null,
                long_input_per_million: null,
                long_cached_input_per_million: null,
                long_output_per_million: null,
                role: '',
                notes: 'Operator-added pricing row.',
                source: 'operator settings',
            };
            this.models = [...(this.models || []), row].sort((a, b) => {
                const providerOrder = { openai: 0, gemini: 1, ollama: 2 };
                const providerDelta = (providerOrder[a.provider] ?? 9) - (providerOrder[b.provider] ?? 9);
                return providerDelta || String(a.key || '').localeCompare(String(b.key || ''));
            });
            this.provider = provider;
            this.dirty = true;
            this.error = '';
            Alpine.store('toast').show(`Added pricing row for ${provider}/${key} — save pricing to persist it`, 'success', 5000);
            return row;
        };
    }

    function installPricingRowCreator() {
        const pane = document.getElementById('pricing-settings-pane');
        if (!pane || pane.querySelector('.pricing-add-row')) return;
        const filter = pane.querySelector('.pricing-provider-filter');
        const list = pane.querySelector('.pricing-model-list');
        if (!filter || !list) return;

        const addButton = document.createElement('button');
        addButton.type = 'button';
        addButton.className = 'pricing-add-model-button';
        addButton.textContent = '+ model';
        filter.appendChild(addButton);

        const creator = document.createElement('form');
        creator.className = 'pricing-add-row';
        creator.hidden = true;
        creator.innerHTML = `
            <div class="pricing-add-row-head">
                <span class="pricing-add-row-title">Add model pricing</span>
                <button type="button" class="pricing-add-cancel" aria-label="Cancel adding model">×</button>
            </div>
            <div class="pricing-add-row-grid">
                <label class="pricing-add-field">
                    <span>Provider</span>
                    <select name="provider">
                        <option value="openai">OpenAI</option>
                        <option value="gemini">Gemini</option>
                        <option value="ollama">Ollama</option>
                    </select>
                </label>
                <label class="pricing-add-field">
                    <span>Model</span>
                    <input name="key" type="text" autocomplete="off" placeholder="model-name or model:cloud" required>
                </label>
                <label class="pricing-add-field">
                    <span>Billing</span>
                    <select name="billing">
                        <option value="token">token priced</option>
                        <option value="estimated_token">estimated token</option>
                        <option value="local">local / $0 API</option>
                        <option value="unknown">unpriced</option>
                    </select>
                </label>
                <label class="pricing-add-field">
                    <span>Native inputs</span>
                    <input name="input_modalities" type="text" value="text" placeholder="text, image, audio">
                </label>
                <label class="pricing-add-field">
                    <span>Outputs</span>
                    <input name="output_modalities" type="text" value="text" placeholder="text">
                </label>
                <label class="pricing-add-field">
                    <span>Capabilities</span>
                    <input name="capabilities" type="text" placeholder="reasoning, tool_calling">
                </label>
            </div>
            <div class="pricing-add-row-rates">
                <label class="pricing-add-field">
                    <span>Input / 1M</span>
                    <input name="input_per_million" type="number" min="0" step="0.001" placeholder="0.000">
                </label>
                <label class="pricing-add-field">
                    <span>Cached input / 1M</span>
                    <input name="cached_input_per_million" type="number" min="0" step="0.001" placeholder="optional">
                </label>
                <label class="pricing-add-field">
                    <span>Output / 1M</span>
                    <input name="output_per_million" type="number" min="0" step="0.001" placeholder="0.000">
                </label>
            </div>
            <div class="pricing-add-row-error" hidden></div>
            <div class="pricing-add-row-actions">
                <button type="button" class="pricing-add-cancel">cancel</button>
                <button type="submit" class="primary">add pricing row</button>
            </div>
        `;
        list.parentNode.insertBefore(creator, list);

        const providerInput = creator.elements.provider;
        const billingInput = creator.elements.billing;
        const keyInput = creator.elements.key;
        const rateInputs = [
            creator.elements.input_per_million,
            creator.elements.cached_input_per_million,
            creator.elements.output_per_million,
        ];
        const errorNode = creator.querySelector('.pricing-add-row-error');

        function pricingStore() {
            return window.Alpine ? Alpine.store('pricingSettings') : null;
        }

        function syncRateAvailability() {
            const disabled = billingInput.value === 'local' || billingInput.value === 'unknown';
            rateInputs.forEach(input => {
                input.disabled = disabled;
                if (disabled) input.value = '';
            });
        }

        function closeCreator() {
            creator.hidden = true;
            creator.reset();
            errorNode.hidden = true;
            errorNode.textContent = '';
            syncRateAvailability();
        }

        addButton.addEventListener('click', () => {
            const pricing = pricingStore();
            const selectedProvider = pricing && ['openai', 'gemini', 'ollama'].includes(pricing.provider)
                ? pricing.provider
                : 'openai';
            providerInput.value = selectedProvider;
            billingInput.value = selectedProvider === 'ollama' ? 'estimated_token' : 'token';
            syncRateAvailability();
            errorNode.hidden = true;
            errorNode.textContent = '';
            creator.hidden = false;
            keyInput.focus();
        });

        creator.querySelectorAll('.pricing-add-cancel').forEach(button => {
            button.addEventListener('click', closeCreator);
        });
        billingInput.addEventListener('change', syncRateAvailability);
        creator.addEventListener('submit', event => {
            event.preventDefault();
            const pricing = pricingStore();
            if (!pricing || typeof pricing.addModelRow !== 'function') return;
            try {
                pricing.addModelRow({
                    provider: providerInput.value,
                    key: keyInput.value,
                    billing: billingInput.value,
                    input_per_million: creator.elements.input_per_million.value,
                    cached_input_per_million: creator.elements.cached_input_per_million.value,
                    output_per_million: creator.elements.output_per_million.value,
                    input_modalities: creator.elements.input_modalities.value,
                    output_modalities: creator.elements.output_modalities.value,
                    capabilities: creator.elements.capabilities.value,
                });
                closeCreator();
            } catch (error) {
                errorNode.textContent = String(error instanceof Error ? error.message : error);
                errorNode.hidden = false;
            }
        });
        syncRateAvailability();
    }

    document.addEventListener('alpine:init', () => {
        queueMicrotask(() => {
            const sessions = Alpine.store('sessions');
            const chat = Alpine.store('chat');
            const layout = Alpine.store('layout');
            const pricing = Alpine.store('pricingSettings');
            installPricingRowMethods(pricing);
            if (!sessions || !chat || sessions.__webShellHydrationInstalled) return;
            sessions.__webShellHydrationInstalled = true;

            /*
             * The core bootstrap historically calls chat.loadHistory() before
             * sessions.load(). Suppress that provisional unscoped request: it can
             * hydrate an empty/default slot and then make the real session look
             * permanently empty. The sessions wrapper below performs the first
             * authoritative hydration after /api/sessions has resolved focus.
             */
            const coreLoadHistory = chat.loadHistory.bind(chat);
            chat.loadHistory = async function (name, options = {}) {
                const target = name || this.currentName || sessions.current || null;
                if (!target) return;

                const slot = this._slot(target);
                const worked = workedFingerprintMap(slot);
                const wasBusy = !!slot.busy;

                // A page refresh during a running turn starts with only an empty
                // busy-trace placeholder. It is safe (and important) to hydrate the
                // durable transcript beneath that placeholder. Once real live deltas
                // exist, never replace them mid-turn.
                if (wasBusy && hasLiveTranscript(slot)) {
                    slot.pendingReload = true;
                    return;
                }

                const result = await coreLoadHistory(target, {
                    ...options,
                    force: true,
                });

                const hydrated = this._slot(target);
                restoreWorkedFingerprints(hydrated, worked);
                if (hydrated.historyHydrated) hydrated.__authoritativeHydrated = true;

                if (wasBusy) {
                    hydrated.busy = true;
                    this._ensureBusyTrace(hydrated);
                }
                decorateWorkedDurations(this, target);
                return result;
            };

            const originalLoad = sessions.load.bind(sessions);
            sessions.load = async function (...args) {
                const result = await originalLoad(...args);
                const current = this.current || chat.currentName;
                if (current) {
                    if (chat.currentName !== current) chat.focus(current);
                    const slot = chat._slot(current);
                    if (!slot.__authoritativeHydrated || !slot.historyHydrated) {
                        await chat.loadHistory(current, { force: true });
                    }
                    decorateWorkedDurations(chat, current);
                }
                return result;
            };

            const originalSwitchTo = sessions.switchTo.bind(sessions);
            sessions.switchTo = async function (name) {
                const result = await originalSwitchTo(name);
                if (this.current !== name) return result;
                const slot = chat._slot(name);
                slot.__authoritativeHydrated = false;
                await chat.loadHistory(name, { force: true });
                decorateWorkedDurations(chat, name);
                return result;
            };

            const originalFocus = chat.focus.bind(chat);
            chat.focus = function (name) {
                const result = originalFocus(name);
                decorateWorkedDurations(this, name);
                return result;
            };

            const originalSend = chat.send.bind(chat);
            chat.send = async function (text) {
                const name = this.currentName;
                const slot = this._slot(name);
                const trimmed = String(text || '').trim();
                const track = !!name && !this.isSlashCommand(trimmed) && this.canSend(trimmed);
                if (track && !slot.busy) slot.__turnStartedAt = Date.now();
                const result = await originalSend(text);
                if (track && !slot.busy) delete slot.__turnStartedAt;
                return result;
            };

            const originalAddCommandResult = chat.addCommandResult.bind(chat);
            chat.addCommandResult = function (result, name) {
                this._slot(name).__skipWorkedFinish = true;
                return originalAddCommandResult(result, name);
            };

            const originalFinishTurn = chat.finishTurn.bind(chat);
            chat.finishTurn = function (name) {
                const slot = this._slot(name);
                const startedAt = Number(slot.__turnStartedAt || 0);
                const skipWorked = !!slot.__skipWorkedFinish;
                slot.__skipWorkedFinish = false;

                const result = originalFinishTurn(name);

                if (startedAt && !skipWorked) {
                    const finalResponse = lastFinalAssistant(slot);
                    if (finalResponse) {
                        finalResponse.workedMs = Math.max(0, Date.now() - startedAt);
                        delete slot.__turnStartedAt;
                    } else if (!slot.busy) {
                        // Terminal error/interruption with no completed assistant
                        // response is not a fully completed turn.
                        delete slot.__turnStartedAt;
                    }
                }
                decorateWorkedDurations(this, name);
                return result;
            };

            // The RHS now uses compact selectors. Keep a useful desktop default
            // without over-allocating space to navigation chrome.
            if (layout && window.innerWidth >= 1100 && Number(layout.panelWidth || 0) < 500) {
                layout.panelWidth = 520;
            }
        });
    });

    document.addEventListener('DOMContentLoaded', () => {
        // product.js installs the pricing pane first (it is loaded before this
        // reliability layer), so the row creator can augment it without owning
        // a second pricing UI or API path.
        installPricingRowCreator();
    });
})();
