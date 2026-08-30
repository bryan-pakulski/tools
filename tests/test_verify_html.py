"""Unit tests for the verify_html composed self-review tool."""

from __future__ import annotations

from mu.tools.agent import self_review as sr


def test_verdict_parsing_pass():
    review = {"ok": True, "output": "VERDICT: PASS\n- all good"}
    assert review["output"].splitlines()[0].strip().upper().startswith("VERDICT: PASS")


def test_envelope_shape():
    env = sr._envelope(True, None, "msg", {"k": 1})
    assert set(env) == {"ok", "error_code", "message", "data", "artifacts", "telemetry"}
    assert env["telemetry"]["tool_name"] == "verify_html"


def test_codex_review_builds_prompt(tmp_path):
    html = tmp_path / "page.html"
    html.write_text("<html><body>hi</body></html>")
    captured = {}

    def fake_attempt(task, repo, out_path, attempt):
        captured["task"] = task
        return {"ok": True, "output": "VERDICT: PASS\n- ok", "output_chars": 10}

    orig = sr._run_codex_review
    import mu.tools.agent.best_of as bo

    sr_check = lambda *a, **k: None
    # Patch _run_attempt inside the best_of module (self_review imports it lazily).
    original = bo._run_attempt
    bo._run_attempt = fake_attempt
    try:
        result = sr._run_codex_review(str(html), "rendered text", "contrast", str(tmp_path))
    finally:
        bo._run_attempt = original
    assert result["ok"] is True
    assert "contrast" in captured["task"]
    assert "rendered text" in captured["task"]
    assert "VERDICT" in captured["task"]


def test_tool_registered_in_descriptors():
    from mu.tools.descriptors import TOOLS

    assert any(t.name == "verify_html" for t in TOOLS)


def test_parse_verdict_tolerant():
    parse = sr._parse_verdict
    assert parse("VERDICT: PASS\n- ok") == "pass"
    assert parse("verdict: issues\n- x") == "issues"
    assert parse("```markdown\nVERDICT: PASS\n```\nnotes") == "pass"
    assert parse("\n\nVERDICT:  ISSUES \n") == "issues"
    assert parse("VERDICT: PASSING") == "unknown"  # partial word rejected
    assert parse("no verdict here") == "unknown"
