from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "mu" / "gui" / "templates" / "work.html"
WORK_POLISH = ROOT / "mu" / "gui" / "static" / "css" / "work_polish.css"
WORK_ANALYSIS_LINK = ROOT / "mu" / "gui" / "static" / "js" / "work_analysis_link.js"
ROUTE_THEME = ROOT / "mu" / "gui" / "static" / "js" / "route_theme.js"
TRACE_HTML = ROOT / "mu" / "gui" / "static" / "job_trace.html"
TRACE_JS = ROOT / "mu" / "gui" / "static" / "js" / "job_trace.js"
TRACE_SINGLE_TASK = ROOT / "mu" / "gui" / "static" / "js" / "job_trace_signed_delta.js"
MODEL_HTML = ROOT / "mu" / "gui" / "static" / "model_costs.html"
RUNNER = ROOT / "mu" / "jobs" / "runner.py"
PRICING = ROOT / "config" / "model_pricing.json"


def test_engineering_work_uses_primary_mucli_chrome_and_theme_switcher():
    html = WORK.read_text(encoding="utf-8")
    css = WORK_POLISH.read_text(encoding="utf-8")
    theme = ROUTE_THEME.read_text(encoding="utf-8")

    assert '/static/css/product.css' in html
    assert '/static/css/clarity.css' in html
    assert '/static/css/work_polish.css' in html
    assert '/static/js/route_theme.js' in html
    assert 'work-header product-header' in html
    assert 'data-route-theme' in html
    assert 'product-icon-button work-header-icon' in html
    assert 'title="Analyze jobs"' in html
    assert 'title="Model pricing"' in html
    assert 'title="Manage jobs"' in html
    assert 'work-pane-head-controls' in html
    assert 'id="work-refresh"' in html
    assert 'work-manage-button' not in html
    assert '.work-header.product-header' in css
    assert '.work-board-refresh' in css
    assert "localStorage.setItem('mucli-theme'" in theme


def test_job_management_is_a_full_viewport_modal_not_document_flow():
    html = WORK.read_text(encoding="utf-8")
    css = WORK_POLISH.read_text(encoding="utf-8")

    assert 'id="work-management-backdrop"' in html
    assert 'class="work-management-pane" role="dialog" aria-modal="true"' in html
    assert '.work-product-app > .work-management-backdrop' in css
    assert 'position: fixed;' in css
    assert 'height: 100dvh;' in css
    assert 'body.work-management-open { overflow: hidden; }' in css
    assert '.work-product-app > .work-management-backdrop[hidden]' in css


def test_job_trace_and_model_pricing_share_product_visual_language():
    trace = TRACE_HTML.read_text(encoding="utf-8")
    pricing = MODEL_HTML.read_text(encoding="utf-8")

    for html in (trace, pricing):
        assert '/static/css/product.css' in html
        assert '/static/css/clarity.css' in html
        assert 'product-header' in html
    assert 'id="jt-theme"' in trace
    assert 'id="mc-theme"' in pricing
    assert 'Pricing registry' in pricing
    assert 'Input / 1M' in pricing
    assert 'Output / 1M' in pricing
    assert 'Native inputs' in pricing
    assert 'Capabilities' in pricing
    assert 'Blended est. / 1M total' not in pricing
    assert 'Quick estimator' not in pricing


def test_job_trace_focuses_on_selected_individual_task_without_comparison_ui():
    work_link = WORK_ANALYSIS_LINK.read_text(encoding="utf-8")
    single_task = TRACE_SINGLE_TASK.read_text(encoding="utf-8")

    assert 'syncHeaderAnalysisLink' in work_link
    assert '/static/job_trace.html?job=${encodeURIComponent(jobId)}' in work_link
    assert "document.getElementById('jt-compare-select')?.remove()" in single_task
    assert "document.getElementById('jt-compare-section')?.remove()" in single_task
    assert "searchParams.delete('compare')" in single_task
    assert 'autoLoadFirstJob' in single_task
    assert "jobSelect.dispatchEvent(new Event('change'" in single_task


def test_job_analyzer_explains_state_residence_and_links_full_harness_trace():
    html = TRACE_HTML.read_text(encoding="utf-8")
    script = TRACE_JS.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert 'Agent harness trace' in html
    assert 'state residence is separated from actual execution' in html
    assert 'Lifecycle drill-down' in html
    assert 'id="jt-intervals"' in html
    assert 'runtime_trace' in script
    assert 'interpretation' in script
    assert 'entry_event' in script
    assert 'exit_event' in script
    assert 'args.trace = True' in runner


def test_glm_cloud_default_estimate_uses_separate_input_and_output_rates():
    config = json.loads(PRICING.read_text(encoding="utf-8"))
    glm = next(row for row in config["models"] if row["provider"] == "ollama" and row["key"] == "glm-5.2:cloud")

    assert glm["billing"] == "estimated_token"
    assert glm["input_per_million"] == 1.4
    assert glm["output_per_million"] == 4.4
    assert glm["estimated_total_per_million"] is None
    assert 'separate measured input and output tokens' in glm["notes"].lower()
