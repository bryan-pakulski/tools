from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_pricing_is_first_class_in_settings():
    # The pricingSettings runtime store and its API calls live in app.js
    # (the runtime layer); product.js keeps presentation only — the
    # presentation-only contract is pinned by test_web_product_ui.
    product = (ROOT / "mu/gui/static/js/product.js").read_text(encoding="utf-8")
    app = (ROOT / "mu/gui/static/js/app.js").read_text(encoding="utf-8")
    shell = (ROOT / "mu/gui/static/js/web_shell.js").read_text(encoding="utf-8")
    css = (ROOT / "mu/gui/static/css/pricing_settings.css").read_text(encoding="utf-8")
    rows_css = (ROOT / "mu/gui/static/css/pricing_rows.css").read_text(encoding="utf-8")
    providers = (ROOT / "mu/gui/routers/providers.py").read_text(encoding="utf-8")

    assert "Alpine.store('pricingSettings'" in app
    assert "installPricingSettings" in app
    assert "tab.textContent = 'pricing'" in app
    assert "Model pricing" in app
    assert "OpenAI" in app
    assert "Gemini" in app
    assert "Ollama" in app
    assert "Input / 1M" in app
    assert "Cached input / 1M" in app
    assert "Output / 1M" in app
    assert "token priced" in app
    assert "estimated token" in app
    assert "local / $0 API" in app
    assert "unpriced" in app
    assert "Advanced registry" in app
    # Presentation layer stays free of runtime concerns.
    assert "fetch(" not in product
    assert "Alpine.store(" not in product

    assert "fetch('/api/providers/pricing'" in app
    assert "method: 'PUT'" in app
    assert "fetch('/api/providers/pricing/reset'" in app
    assert "method: 'POST'" in app

    assert "addModelRow" in shell
    assert "add pricing row" in shell
    assert "+ model" in shell
    assert "Pricing row already exists" in shell
    assert "Input and output rates are required" in shell
    assert "Operator-added pricing row." in shell
    assert "pricing_rows.css" in shell

    assert '@router.get("/pricing")' in providers
    assert '@router.put("/pricing")' in providers
    assert '@router.post("/pricing/reset")' in providers

    assert ".pricing-rate-grid" in css
    assert ".pricing-provider-filter" in css
    assert ".pricing-settings-actions" in css
    assert ".pricing-add-row" in rows_css
    assert ".pricing-add-model-button" in rows_css


def test_pricing_settings_preserve_local_ollama_cost_semantics():
    app = (ROOT / "mu/gui/static/js/app.js").read_text(encoding="utf-8")
    pricing = (ROOT / "utils/model_pricing.py").read_text(encoding="utf-8")

    assert "Local Ollama remains $0 attributable provider/API cost" in app
    assert '"billing": "local"' in pricing
    assert '"api_cost_usd": 0.0' in pricing
    assert "host compute is excluded" in pricing
