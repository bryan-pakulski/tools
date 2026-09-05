(() => {
    'use strict';

    const state = { catalog: null, models: [], dirty: false };
    const $ = id => document.getElementById(id);

    function esc(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
    }
    function nullable(value) {
        const text = String(value ?? '').trim();
        if (!text) return null;
        const number = Number(text);
        return Number.isFinite(number) ? number : null;
    }
    function input(value, field, type = 'text', extra = '') {
        const display = value == null ? '' : String(value);
        return `<input data-field="${field}" type="${type}" value="${esc(display)}" ${extra}>`;
    }
    function markDirty() {
        state.dirty = true;
        $('mc-dirty').hidden = false;
        $('mc-status').textContent = '';
    }
    function syncThemeIcon() {
        const dark = (document.documentElement.getAttribute('data-theme') || 'dark') === 'dark';
        document.querySelectorAll('.mc-theme-dark').forEach(node => { node.style.display = dark ? '' : 'none'; });
        document.querySelectorAll('.mc-theme-light').forEach(node => { node.style.display = dark ? 'none' : ''; });
    }
    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        try { localStorage.setItem('mucli-theme', next); } catch (_) {}
        syncThemeIcon();
    }

    function billingSelect(value) {
        const options = [
            ['token', 'Token rates'],
            ['estimated_token', 'Estimated input/output'],
            ['local', 'Local / $0 API'],
            ['unknown', 'Unpriced'],
        ];
        return `<select data-field="billing">${options.map(([key, label]) => `<option value="${key}" ${key === value ? 'selected' : ''}>${label}</option>`).join('')}</select>`;
    }

    function render() {
        const needle = String($('mc-search').value || '').trim().toLowerCase();
        const rows = state.models.map((item, index) => {
            const search = `${item.provider} ${item.key} ${(item.aliases || []).join(' ')} ${(item.input_modalities || []).join(' ')} ${(item.output_modalities || []).join(' ')} ${(item.capabilities || []).join(' ')} ${item.role || ''} ${item.notes || ''}`.toLowerCase();
            const hidden = needle && !search.includes(needle) ? ' hidden' : '';
            return `<tr data-index="${index}" data-search="${esc(search)}"${hidden}>
                <td>${input(item.provider, 'provider')}</td>
                <td class="mc-model-cell">${input(item.key, 'key')}</td>
                <td>${input((item.input_modalities || ['text']).join(', '), 'input_modalities')}</td>
                <td>${input((item.output_modalities || ['text']).join(', '), 'output_modalities')}</td>
                <td>${input((item.capabilities || []).join(', '), 'capabilities')}</td>
                <td>${billingSelect(item.billing || 'token')}</td>
                <td>${input(item.input_per_million, 'input_per_million', 'number', 'min="0" step="0.001"')}</td>
                <td>${input(item.cached_input_per_million, 'cached_input_per_million', 'number', 'min="0" step="0.001"')}</td>
                <td>${input(item.output_per_million, 'output_per_million', 'number', 'min="0" step="0.001"')}</td>
                <td>${input(item.context_window, 'context_window', 'number', 'min="1" step="1"')}</td>
                <td>${input(item.long_context_cutoff, 'long_context_cutoff', 'number', 'min="1" step="1"')}</td>
                <td>${input(item.long_input_per_million, 'long_input_per_million', 'number', 'min="0" step="0.001"')}</td>
                <td>${input(item.long_cached_input_per_million, 'long_cached_input_per_million', 'number', 'min="0" step="0.001"')}</td>
                <td>${input(item.long_output_per_million, 'long_output_per_million', 'number', 'min="0" step="0.001"')}</td>
                <td>${input((item.aliases || []).join(', '), 'aliases')}</td>
                <td class="mc-notes-cell">
                    ${input(item.role || '', 'role')}
                    <textarea data-field="notes" rows="2" placeholder="Notes">${esc(item.notes || '')}</textarea>
                </td>
                <td><button class="mc-row-remove" data-remove title="Remove model" aria-label="Remove model">×</button></td>
            </tr>`;
        }).join('');
        $('mc-models').innerHTML = rows || '<tr><td colspan="17" class="mc-table-empty">No configured models.</td></tr>';
        $('mc-count').textContent = `${state.models.length} model${state.models.length === 1 ? '' : 's'}`;
        wireRows();
    }

    function updateField(index, field, raw) {
        const item = state.models[index];
        if (!item) return;
        if (['aliases', 'input_modalities', 'output_modalities', 'capabilities'].includes(field)) {
            item[field] = String(raw || '').split(',').map(value => {
                const clean = value.trim();
                return field === 'aliases' ? clean : clean.toLowerCase().replaceAll(' ', '_');
            }).filter(Boolean);
        } else if ([
            'input_per_million', 'cached_input_per_million', 'output_per_million',
            'estimated_total_per_million', 'context_window', 'long_context_cutoff',
            'long_input_per_million', 'long_cached_input_per_million', 'long_output_per_million',
        ].includes(field)) {
            item[field] = nullable(raw);
        } else {
            item[field] = String(raw ?? '');
        }
        markDirty();
    }

    function wireRows() {
        $('mc-models').querySelectorAll('tr[data-index]').forEach(row => {
            const index = Number(row.dataset.index);
            row.querySelectorAll('[data-field]').forEach(control => {
                const eventName = control.tagName === 'SELECT' ? 'change' : 'input';
                control.addEventListener(eventName, () => updateField(index, control.dataset.field, control.value));
            });
            row.querySelector('[data-remove]')?.addEventListener('click', () => {
                state.models.splice(index, 1);
                markDirty();
                render();
            });
        });
    }

    function addModel() {
        state.models.unshift({
            provider: 'ollama', key: '', billing: 'unknown', aliases: [],
            input_modalities: ['text'], output_modalities: ['text'], capabilities: [],
            input_per_million: null, cached_input_per_million: null, output_per_million: null,
            estimated_total_per_million: null, context_window: null,
            long_context_cutoff: null, long_input_per_million: null,
            long_cached_input_per_million: null, long_output_per_million: null,
            role: '', notes: '', source: 'operator configuration',
        });
        markDirty();
        render();
        $('mc-models').querySelector('input[data-field="key"]')?.focus();
    }

    async function load() {
        const response = await fetch('/api/providers/pricing');
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        state.catalog = data;
        state.models = JSON.parse(JSON.stringify(data.models || []));
        state.dirty = false;
        $('mc-dirty').hidden = true;
        $('mc-version').textContent = `Registry ${data.version}`;
        $('mc-config-path').textContent = data.using_override
            ? `override · ${data.active_config_path}`
            : `defaults · edit ${data.config_path}`;
        render();
    }

    async function save() {
        const invalid = state.models.find(item => !String(item.provider || '').trim() || !String(item.key || '').trim());
        if (invalid) {
            $('mc-status').textContent = 'Provider and model are required on every row.';
            return;
        }
        const invalidEstimate = state.models.find(item => {
            if (item.billing !== 'estimated_token') return false;
            const split = item.input_per_million != null || item.output_per_million != null;
            return split
                ? item.input_per_million == null || item.output_per_million == null
                : item.estimated_total_per_million == null;
        });
        if (invalidEstimate) {
            $('mc-status').textContent = 'Estimated pricing requires both input and output rates.';
            return;
        }
        $('mc-save').disabled = true;
        $('mc-status').textContent = 'Saving…';
        try {
            const response = await fetch('/api/providers/pricing', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    version: new Date().toISOString(),
                    currency: state.catalog?.currency || 'USD',
                    unit: state.catalog?.unit || 'per_million_tokens',
                    models: state.models,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            state.catalog = data;
            state.models = JSON.parse(JSON.stringify(data.models || []));
            state.dirty = false;
            $('mc-dirty').hidden = true;
            $('mc-version').textContent = `Registry ${data.version}`;
            $('mc-config-path').textContent = `override · ${data.active_config_path}`;
            $('mc-status').textContent = 'Saved.';
            render();
        } catch (error) {
            $('mc-status').textContent = `Save failed: ${error.message}`;
        } finally {
            $('mc-save').disabled = false;
        }
    }

    async function reset() {
        if (!window.confirm('Remove the user pricing override and return to packaged defaults?')) return;
        $('mc-status').textContent = 'Resetting…';
        const response = await fetch('/api/providers/pricing/reset', { method: 'POST' });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            $('mc-status').textContent = `Reset failed: ${data.detail || response.status}`;
            return;
        }
        state.catalog = data;
        state.models = JSON.parse(JSON.stringify(data.models || []));
        state.dirty = false;
        $('mc-dirty').hidden = true;
        $('mc-version').textContent = `Registry ${data.version}`;
        $('mc-config-path').textContent = `defaults · edit ${data.config_path}`;
        $('mc-status').textContent = 'Using packaged defaults.';
        render();
    }

    async function init() {
        syncThemeIcon();
        $('mc-search').addEventListener('input', render);
        $('mc-theme').addEventListener('click', toggleTheme);
        $('mc-add').addEventListener('click', addModel);
        $('mc-save').addEventListener('click', save);
        $('mc-reset').addEventListener('click', reset);
        window.addEventListener('beforeunload', event => {
            if (!state.dirty) return;
            event.preventDefault();
            event.returnValue = '';
        });
        try {
            await load();
            $('mc-loading').hidden = true;
            $('mc-main').hidden = false;
        } catch (error) {
            $('mc-loading').hidden = true;
            $('mc-error').hidden = false;
            $('mc-error').textContent = `Could not load model pricing: ${error.message}`;
        }
    }

    void init();
})();
