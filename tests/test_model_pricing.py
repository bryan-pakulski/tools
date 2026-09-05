from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mu.jobs import JobService, JobSpec, JobStore
from mu.jobs.runner import SessionJobRunner
from utils.model_pricing import (
    PRICING_VERSION,
    estimate_model_cost,
    pricing_catalog,
    pricing_config_path,
    reset_pricing_config,
    resolve_model_capabilities,
    resolve_token_pricing,
    save_pricing_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_openai_cached_tokens_are_not_double_charged():
    estimate = estimate_model_cost(
        provider="openai",
        model_name="gpt-5.6-terra",
        input_tokens=200_000,
        cached_tokens=40_000,
        output_tokens=20_000,
    )
    assert estimate["pricing_key"] == "gpt-5.6-terra"
    assert estimate["long_context_tier"] is False
    assert estimate["api_cost_usd"] == pytest.approx(0.71)
    assert estimate["usage"]["uncached_input"] == 160_000
    assert estimate["usage"]["cached_input"] == 40_000
    assert estimate["cost_components"]["input_usd"] == pytest.approx(0.41)
    assert estimate["cost_components"]["output_usd"] == pytest.approx(0.30)


def test_openai_long_context_tier_and_specific_alias_win():
    long = estimate_model_cost(
        provider="openai",
        model_name="gpt-5.6-sol-2026-08-01",
        input_tokens=300_000,
        cached_tokens=100_000,
        output_tokens=50_000,
    )
    assert long["long_context_tier"] is True
    assert long["rates"]["input_per_million"] == 10.0
    assert long["rates"]["output_per_million"] == 45.0
    assert long["api_cost_usd"] == pytest.approx(4.35)

    mini = resolve_token_pricing("openai", "gpt-5.4-mini")
    assert mini is not None
    assert mini.key == "gpt-5.4-mini"
    assert mini.input_per_million == pytest.approx(0.75)


def test_gemini_pro_high_tier_and_reasoning_not_double_charged():
    estimate = estimate_model_cost(
        provider="gemini",
        model_name="models/gemini-3.1-pro-preview-customtools",
        input_tokens=250_000,
        cached_tokens=50_000,
        output_tokens=20_000,
        reasoning_tokens=10_000,
    )
    assert estimate["pricing_key"] == "gemini-3.1-pro-preview"
    assert estimate["long_context_tier"] is True
    assert estimate["api_cost_usd"] == pytest.approx(1.18)
    assert estimate["usage"]["reasoning"] == 10_000


def test_ollama_local_zero_api_cost_and_glm_cloud_uses_split_estimate():
    local = estimate_model_cost(
        provider="ollama",
        model_name="qwen3-coder-next:latest",
        input_tokens=50_000,
        output_tokens=2_000,
        endpoint="http://localhost:11434",
    )
    assert local["billing"] == "local"
    assert local["api_cost_usd"] == 0.0
    assert local["catalog"]["context_window"] == 256_000

    cloud = estimate_model_cost(
        provider="ollama",
        model_name="glm-5.2:cloud",
        input_tokens=50_000,
        output_tokens=2_000,
        endpoint="https://ollama.com",
    )
    assert cloud["billing"] == "estimated_token"
    assert cloud["source"] == "configured_estimate"
    assert cloud["rate_shape"] == "input_output"
    assert cloud["rates"]["input_per_million"] == pytest.approx(1.40)
    assert cloud["rates"]["output_per_million"] == pytest.approx(4.40)
    assert cloud["cost_components"]["input_usd"] == pytest.approx(0.0700)
    assert cloud["cost_components"]["output_usd"] == pytest.approx(0.0088)
    assert cloud["cost_components"]["total_usd"] == pytest.approx(0.0788)
    assert cloud["api_cost_usd"] == pytest.approx(0.0788)
    assert cloud["catalog"]["context_window"] == 976_000


def test_unknown_model_is_unpriced_not_free():
    estimate = estimate_model_cost(
        provider="openai",
        model_name="future-model-not-in-map",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert estimate["billing"] == "unknown"
    assert estimate["source"] == "unpriced"
    assert estimate["api_cost_usd"] is None


def test_public_catalog_is_versioned_unified_and_exposes_config_paths():
    catalog = pricing_catalog()
    assert catalog["version"] == PRICING_VERSION
    assert {item["provider"] for item in catalog["models"]} == {"openai", "gemini", "ollama"}
    assert catalog["ollama"]
    assert catalog["config_path"].endswith("model_pricing.json")
    assert catalog["active_config_path"]
    glm = next(item for item in catalog["models"] if item["key"] == "glm-5.2:cloud")
    assert glm["input_per_million"] == pytest.approx(1.4)
    assert glm["output_per_million"] == pytest.approx(4.4)
    assert glm["estimated_total_per_million"] is None

    glm_53 = next(
        item for item in catalog["models"] if item["key"] == "glm-5.3-flash:cloud"
    )
    assert glm_53["input_modalities"] == ["text", "image"]
    assert glm_53["output_modalities"] == ["text"]
    assert "vision" in glm_53["capabilities"]
    assert glm_53["input_per_million"] == pytest.approx(0.15)
    assert glm_53["cached_input_per_million"] == pytest.approx(0.03)
    assert glm_53["output_per_million"] == pytest.approx(0.50)


def test_capability_resolution_is_conservative_for_unknown_models():
    gemini = resolve_model_capabilities("gemini", "gemini-3.6-flash")
    assert gemini["input_modalities"] == [
        "text", "image", "audio", "video", "document"
    ]

    unknown = resolve_model_capabilities("openai", "future-unknown-model")
    assert unknown["matched"] is False
    assert unknown["input_modalities"] == ["text"]
    assert unknown["capabilities"] == []


def test_legacy_override_payload_inherits_packaged_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("MUCLI_HOME", str(tmp_path / "legacy-capability-home"))
    rows = [dict(item) for item in pricing_catalog()["models"]]
    glm = next(item for item in rows if item["key"] == "glm-5.3-flash:cloud")
    glm.pop("input_modalities")
    glm.pop("output_modalities")
    glm.pop("capabilities")

    saved = save_pricing_config(
        {
            "version": "legacy-client",
            "currency": "USD",
            "unit": "per_million_tokens",
            "models": rows,
        }
    )
    migrated = next(
        item for item in saved["models"] if item["key"] == "glm-5.3-flash:cloud"
    )
    assert migrated["input_modalities"] == ["text", "image"]
    assert "vision" in migrated["capabilities"]


def test_operator_override_is_live_and_can_be_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("MUCLI_HOME", str(tmp_path / "mucli-home"))
    base = pricing_catalog()
    assert base["using_override"] is False
    rows = [dict(item) for item in base["models"]]
    glm = next(item for item in rows if item["key"] == "glm-5.2:cloud")
    glm["input_per_million"] = 2.0
    glm["output_per_million"] = 5.0

    saved = save_pricing_config({
        "version": "operator-test",
        "currency": "USD",
        "unit": "per_million_tokens",
        "models": rows,
    })
    assert pricing_config_path().exists()
    assert saved["using_override"] is True
    assert saved["version"] == "operator-test"

    estimate = estimate_model_cost(
        provider="ollama",
        model_name="glm-5.2:cloud",
        input_tokens=50_000,
        output_tokens=2_000,
        endpoint="https://ollama.com",
    )
    assert estimate["pricing_version"] == "operator-test"
    assert estimate["config_source"] == "user"
    assert estimate["rates"]["input_per_million"] == pytest.approx(2.0)
    assert estimate["rates"]["output_per_million"] == pytest.approx(5.0)
    assert estimate["cost_components"]["input_usd"] == pytest.approx(0.10)
    assert estimate["cost_components"]["output_usd"] == pytest.approx(0.01)
    assert estimate["api_cost_usd"] == pytest.approx(0.11)

    reset = reset_pricing_config()
    assert reset["using_override"] is False
    assert not pricing_config_path().exists()


def test_legacy_blended_estimate_override_remains_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("MUCLI_HOME", str(tmp_path / "legacy-home"))
    rows = [dict(item) for item in pricing_catalog()["models"]]
    glm = next(item for item in rows if item["key"] == "glm-5.2:cloud")
    glm["input_per_million"] = None
    glm["output_per_million"] = None
    glm["estimated_total_per_million"] = 2.0
    save_pricing_config({
        "version": "legacy-blended",
        "currency": "USD",
        "unit": "per_million_tokens",
        "models": rows,
    })

    estimate = estimate_model_cost(
        provider="ollama",
        model_name="glm-5.2:cloud",
        input_tokens=50_000,
        output_tokens=2_000,
        endpoint="https://ollama.com",
    )
    assert estimate["rate_shape"] == "legacy_blended"
    assert estimate["cost_components"]["input_usd"] == pytest.approx(0.10)
    assert estimate["cost_components"]["output_usd"] == pytest.approx(0.004)
    assert estimate["api_cost_usd"] == pytest.approx(0.104)


def test_cost_registry_is_editable_in_gui_and_exposed_to_all_control_planes():
    providers_router = (ROOT / "mu/gui/routers/providers.py").read_text(encoding="utf-8")
    work = (ROOT / "mu/gui/templates/work.html").read_text(encoding="utf-8")
    html = (ROOT / "mu/gui/static/model_costs.html").read_text(encoding="utf-8")
    script = (ROOT / "mu/gui/static/js/model_costs.js").read_text(encoding="utf-8")
    css = (ROOT / "mu/gui/static/css/model_costs.css").read_text(encoding="utf-8")
    work_semantics = (ROOT / "mu/gui/static/js/work_analysis_link.js").read_text(encoding="utf-8")
    commands = (ROOT / "mu/commands/__init__.py").read_text(encoding="utf-8")
    costs = (ROOT / "mu/commands/costs.py").read_text(encoding="utf-8")
    mobile = (ROOT / "mobile/android/src/api/providers.ts").read_text(encoding="utf-8")
    config = (ROOT / "config/model_pricing.json").read_text(encoding="utf-8")

    assert '@router.get("/pricing")' in providers_router
    assert '@router.put("/pricing")' in providers_router
    assert '@router.post("/pricing/reset")' in providers_router
    assert 'href="/static/model_costs.html"' in work
    assert 'Pricing registry' in html
    assert 'Input / 1M' in html
    assert 'Output / 1M' in html
    assert 'Native inputs' in html
    assert 'Capabilities' in html
    assert 'Blended est. / 1M total' not in html
    assert 'Quick estimator' not in html
    assert "fetch('/api/providers/pricing', {" in script
    assert "method: 'PUT'" in script
    assert "/api/providers/pricing/reset" in script
    assert 'Estimated input/output' in script
    assert 'input_modalities' in script
    assert 'output_modalities' in script
    assert 'capabilities' in script
    assert 'product-header' in html
    assert '.mc-table' in css
    assert 'estimated_token' in config
    assert '"glm-5.2:cloud"' in config
    assert '"glm-5.3-flash:cloud"' in config
    assert "value.textContent = 'Unpriced'" in work_semantics
    assert "value.textContent = '$0.00 API'" in work_semantics
    assert 'components.input_usd' in work_semantics
    assert 'components.output_usd' in work_semantics
    assert 'from . import costs' in commands
    assert '"/costs"' in costs and '"/pricing"' in costs
    assert 'estimated input and output independently' in costs.lower()
    assert 'estimated_total_per_million' in mobile
    assert "pricing: () => api.get<ModelPricingCatalog>('/api/providers/pricing')" in mobile


class _Manager:
    def __init__(self):
        self.token_counts = {
            "input": 10,
            "output": 20,
            "total": 30,
            "cached": 0,
            "reasoning": 0,
            "total_cost": 99.0,
        }


class _Session:
    def __init__(self):
        self.session_manager = _Manager()
        self.variables = {}
        self.provider = SimpleNamespace(
            name="openai",
            model_name="gpt-5.4-mini",
            BASE_URL="https://api.openai.com/v1",
        )


def test_durable_job_usage_result_persists_tokens_and_pricing_provenance(tmp_path):
    service = JobService(JobStore(str(tmp_path / "jobs.sqlite3")))
    job = service.create(JobSpec(
        title="Costed job",
        execution={
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "agent_mode": "default",
            "session_type": "workspace",
        },
    ))
    session = _Session()
    runner = SessionJobRunner(service, build_session_fn=lambda *a, **k: session, base_args=SimpleNamespace())
    before = runner._token_snapshot(session)
    session.session_manager.token_counts.update({
        "input": 1_000_010,
        "output": 100_020,
        "total": 1_100_030,
        "cached": 200_000,
        "reasoning": 25_000,
        "total_cost": 1234.0,
    })

    cost, result = runner._usage_result(job, session, before, {"status": "completed"})

    assert cost == pytest.approx(1.065)
    assert result["tokens"] == {
        "input": 1_000_000,
        "output": 100_000,
        "total": 1_100_000,
        "cached": 200_000,
        "reasoning": 25_000,
    }
    record = result["cost"]
    assert record["pricing_version"] == PRICING_VERSION
    assert record["pricing_key"] == "gpt-5.4-mini"
    assert record["source"] == "pricing_map"
    assert record["api_cost_usd"] == pytest.approx(1.065)
    assert record["cost_components"]["input_usd"] == pytest.approx(0.615)
    assert record["cost_components"]["output_usd"] == pytest.approx(0.45)
    assert record["legacy_loop_cost_usd"] == pytest.approx(1135.0)
    assert session.session_manager.token_counts["total_cost"] == pytest.approx(100.065)
