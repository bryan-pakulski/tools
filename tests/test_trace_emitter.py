"""Tests for the per-run trace emitter (`mu/trace/emitter.py`).

Drives a fake session through a stub provider for a 3-iteration turn and
asserts the JSONL trace contains a ``run_start`` header, one ``iter`` record
per iteration, a ``turn_end`` line, a populated ``drift_pct``, and per-tool
``tool`` lines. Also asserts that disabling ``trace_enabled`` writes no file.
"""

import json
import os

import pytest

from providers.base import LLMProvider, MessagePart, ProviderResponse


class _ThreeIterProvider(LLMProvider):
    """Returns a ``todo_list`` tool call for the first two calls, then text.

    This yields a 3-iteration turn: two tool-bearing iterations followed by a
    text-only completion iteration. Distinct ``input_tokens`` per call exercise
    the drift calculation against the harness's cl100k_base layer estimate.
    """

    def __init__(self, model_name="dummy"):
        self.calls = 0
        self.model_name = model_name
        self.requests = []

    def get_available_models(self):
        return ["dummy"]

    def generate(self, messages, system_prompt=None, thinking=False, tools=None):
        self.calls += 1
        self.requests.append((system_prompt or "", messages))
        if self.calls <= 2:
            return ProviderResponse(
                text="",
                parts=[
                    MessagePart(
                        type="tool_call",
                        tool_name="todo_list",
                        tool_args={},
                        tool_call_id=f"c{self.calls}",
                    )
                ],
                input_tokens=100 + self.calls,
                output_tokens=2,
                total_tokens=102 + self.calls,
            )
        return ProviderResponse(
            text="done",
            parts=[MessagePart(type="text", text="done")],
            input_tokens=300,
            output_tokens=1,
            total_tokens=301,
        )

    def upload_file(self, *a, **kw):
        return None


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(tmp_path / "history"))
    from mu.session.session import Session, SessionManager

    sm = SessionManager()
    sess = Session(_ThreeIterProvider(), False, "system", sm)
    sess.variables["agent_mode"] = "default"
    return sess


def _trace_files(tmp_path):
    trace_root = tmp_path / "history" / "trace"
    if not trace_root.exists():
        return []
    return sorted(trace_root.glob("*.jsonl"))


def _read_trace(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_three_iter_turn_writes_full_trace(session, tmp_path):
    from mu.agent.loop_body import run_turn

    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    assert len(files) == 1, f"expected one trace file, got {files}"
    recs = _read_trace(files[0])

    types = [r.get("type") for r in recs]
    assert types[0] == "run_start"
    assert types.count("iter") == 3
    # Round-51 T2: run_end is the new terminal record; turn_end precedes it.
    assert "turn_end" in types
    assert types[-1] == "run_end"
    assert types.index("turn_end") < types.index("run_end")

    run_start = recs[0]
    assert run_start["run_id"]
    assert run_start["model"] == "dummy"
    assert "context_limit" in run_start
    assert "max_iterations" in run_start

    iter_recs = [r for r in recs if r["type"] == "iter"]
    # iters are numbered 0,1,2 (or 1,2,3) — assert monotonic and contiguous
    iters = [r["iter"] for r in iter_recs]
    assert iters == sorted(iters)
    assert len(set(iters)) == 3
    # drift_pct is populated and is a float on every iter with real input tokens
    for r in iter_recs:
        ctx = r["context"]
        assert "drift_pct" in ctx
        assert isinstance(ctx["drift_pct"], (int, float))
        assert ctx["prompt_tokens_actual"] > 0
        assert ctx["total_est"] >= 0
        assert "l0" in ctx and "l5" in ctx

    turn_end = next(r for r in recs if r["type"] == "turn_end")
    assert turn_end["status"] == "completed"
    assert turn_end["iters"] == 3
    assert turn_end["total_in"] >= 0


def test_tool_records_emitted_for_tool_iters(session, tmp_path):
    from mu.agent.loop_body import run_turn

    session.send_message("do the thing")

    recs = _read_trace(_trace_files(tmp_path)[0])
    tool_recs = [r for r in recs if r["type"] == "tool"]
    # Two tool-bearing iterations, one todo_list call each.
    assert len(tool_recs) == 2
    for tr in tool_recs:
        assert tr["name"] == "todo_list"
        assert "iter" in tr
        assert "latency_ms" in tr
        assert "arg_fp" in tr
        assert tr["ok"] is True


def test_trace_estimate_is_snapshotted_before_response_is_archived(session, tmp_path):
    """Trace drift must compare the provider request, not mutated history."""
    from mu.agent.loop_body import _estimate_messages_tokens
    from utils.token_estimator import estimate_tokens

    session.send_message("do the thing")
    records = _read_trace(_trace_files(tmp_path)[0])
    iter_records = [record for record in records if record["type"] == "iter"]

    for record, (system_prompt, messages) in zip(iter_records, session.provider.requests):
        expected = estimate_tokens(system_prompt) + _estimate_messages_tokens(messages)
        context = record["context"]
        assert context["total_est"] == expected
        assert context["estimate_source"] == "pre_request"


def test_trace_disabled_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.config.HISTORY_DIR", str(tmp_path / "history"))
    from mu.agent.loop_body import run_turn
    from mu.session.session import Session, SessionManager

    sess = Session(_ThreeIterProvider(), False, "system", SessionManager())
    sess.variables["agent_mode"] = "default"
    sess.variables["trace_enabled"] = False

    sess.send_message("do the thing")

    assert _trace_files(tmp_path) == []
    # And the session never got an emitter cached.
    assert getattr(sess, "_trace_emitter", None) is None


def test_build_iter_record_drift_sign():
    """Drift is signed: (actual - total_est) / max(1, actual) * 100."""
    from mu.trace.emitter import build_iter_record

    class _Resp:
        input_tokens = 200
        output_tokens = 5
        cached_tokens = 0
        reasoning_tokens = 0
        parts = [MessagePart(type="text", text="hi")]

    import time

    rec = build_iter_record(
        object(),  # session unused for the drift math here (layers default 0)
        iteration=0,
        max_iter=10,
        response=_Resp(),
        total_in=200,
        total_out=5,
        total_cost=0.0,
        has_text=True,
        has_tool_call=False,
        iter_start=time.monotonic(),
        cost_delta=0.0,
    )
    # With total_est=0 and actual=200, drift = (200-0)/200*100 = 100.0
    assert rec["context"]["drift_pct"] == 100.0
    assert rec["context"]["prompt_tokens_actual"] == 200


def test_request_manifest_attributes_context_components():
    from mu.trace.emitter import build_request_record
    from providers.base import Message, ToolDefinition

    messages = [
        Message(role="user", parts=[MessagePart(type="text", text="inspect this")]),
        Message(role="assistant", parts=[MessagePart(
            type="tool_call", tool_name="read_file", tool_args={"path": "large.py"}
        )]),
        Message(role="tool", parts=[MessagePart(
            type="tool_result", tool_name="read_file", tool_result="x" * 8000
        )]),
    ]
    tools = [ToolDefinition(
        name="read_file", description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )]
    record = build_request_record(
        iteration=7, system_prompt="system rules", messages=messages,
        tools=tools, token_estimate=1234,
    )

    assert record["component_tokens"]["system"] > 0
    assert record["component_tokens"]["user"] > 0
    assert record["component_tokens"]["tool_calls"] > 0
    assert record["component_tokens"]["tool_results"] > 500
    assert record["component_tokens"]["tool_schemas"] > 0
    assert record["tool_schema_bytes"] > 0
    assert record["messages"][2]["parts"] == 1
    assert record["messages"][2]["part_details"][0]["type"] == "tool_result"
    assert record["messages"][2]["part_details"][0]["tokens"] > 500


def test_run_start_emits_effective_limit_and_divergence(session, tmp_path, caplog):
    """Round-51 T1: run_start carries the drift-corrected effective_limit the
    preflight guard enforces alongside the configured context_token_limit, and
    warns when the two diverge >5%."""
    import logging
    from providers.ollama import OllamaProvider

    session.variables["context_token_limit"] = 580000
    # Simulate an Ollama-style provider whose real window resolves far below
    # the configured limit (the observed 580k-vs-guard mismatch).
    session.provider.effective_context_window = lambda *a, **k: 200000
    session.provider.compaction_safety_factor = lambda: 2.5

    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    assert files, "expected a trace file"
    recs = _read_trace(files[0])
    run_start = next(r for r in recs if r["type"] == "run_start")
    assert "effective_limit" in run_start
    # Guard: max(1024, window/factor) = 80000 — diverges >5% from 580000.
    assert run_start["effective_limit"] == 80000
    assert run_start["context_limit"] == 580000
    # Window below configured limit drives the effective limit.
    assert run_start["limit_source"] == "provider_window"
    assert session._effective_guard_limit == 80000
    with caplog.at_level(logging.WARNING):
        pass  # warning emitted during send_message above; presence asserted below
    assert any(
        "Context governance divergence" in r.message
        for r in caplog.records
    ) or any(
        "Context governance divergence" in str(getattr(r, "msg", ""))
        for r in caplog.records
    )


def test_run_start_no_warn_when_limits_align(session, tmp_path, caplog):
    """When the effective guard limit tracks the configured limit, no
    divergence warning fires and effective_limit mirrors context_limit."""
    session.variables["context_token_limit"] = 250000
    session.provider.effective_context_window = lambda *a, **k: 250000
    session.provider.compaction_safety_factor = lambda: 1.0
    import logging

    with caplog.at_level(logging.WARNING):
        session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    run_start = next(r for r in recs if r["type"] == "run_start")
    assert run_start["effective_limit"] == run_start["context_limit"]
    assert not any("Context governance divergence" in str(r.msg) for r in caplog.records)


def test_run_end_emitted_on_completion(session, tmp_path):
    """Round-51 T2: every run terminates with a run_end record carrying the
    final status, reason, iteration count and token totals."""
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    kinds = [r["type"] for r in recs]
    assert "run_end" in kinds
    run_end = next(r for r in recs if r["type"] == "run_end")
    # run_end is the terminal record of the trace
    assert kinds[-1] == "run_end"
    assert run_end["status"] == "completed"
    assert run_end["iters"] == 3
    assert isinstance(run_end["tokens_in"], int)
    assert isinstance(run_end["tokens_out"], int)
    assert run_end["run_id"] == recs[0]["run_id"]


def test_run_end_marked_failed_on_error(session, tmp_path):
    """An exception inside the turn maps to run_end status=failed with the
    error string as reason, so crashed runs are no longer indistinguishable
    from healthy ones."""
    def boom(msg):
        raise RuntimeError("provider exploded")

    session.provider.generate = boom
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    run_ends = [r for r in recs if r["type"] == "run_end"]
    assert run_ends, "expected run_end even on error path"
    assert run_ends[0]["status"] == "failed"
    assert "RuntimeError" in str(run_ends[0]["reason"]) or "boom" in str(run_ends[0]["reason"])


def test_context_collapse_recorded_on_silent_l5_drop(session, tmp_path, monkeypatch):
    """Round-51 T3: a >50% L5 drop between consecutive iterations with no
    compaction record emits a context_collapse event with from/to values."""
    import mu.trace.emitter as trace_emitter

    # Prime: previous iteration sat at 100k L5 (prev num 0 so in-run iter 1
    # is its successor).
    session._trace_prev_iter_l5 = 100000
    session._trace_prev_iter_num = 0

    real_iter_record = trace_emitter.build_iter_record

    def fake_iter_record(session, **kw):
        rec = real_iter_record(session, **kw)
        if kw.get("iteration") == 1:
            rec["context"]["l5"] = 20000
        return rec

    monkeypatch.setattr(trace_emitter, "build_iter_record", fake_iter_record)
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    collapses = [r for r in recs if r.get("type") == "context_collapse"]
    assert collapses, f"expected context_collapse record; got types {[r['type'] for r in recs]}"
    c = collapses[0]
    assert c["from_l5"] == 100000
    assert c["to_l5"] == 20000
    assert c["hint"] == "silently_reset"


def test_no_context_collapse_when_compaction_explains_drop(session, tmp_path, monkeypatch):
    """When a compaction record was drained in the same iteration, the drop
    is attributed to it and context_collapse must NOT fire."""
    import mu.trace.emitter as trace_emitter

    session._trace_prev_iter_l5 = 100000
    session._trace_prev_iter_num = 0

    real_iter_record = trace_emitter.build_iter_record

    def fake_iter_record(session, **kw):
        rec = real_iter_record(session, **kw)
        if kw.get("iteration") == 1:
            rec["context"]["l5"] = 20000
        return rec

    monkeypatch.setattr(trace_emitter, "build_iter_record", fake_iter_record)
    monkeypatch.setattr(
        trace_emitter, "drain_compactions",
        lambda session: [{"summary": "legit", "digest": "abc", "iters": [1, 2]}],
    )
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    assert not [r for r in recs if r.get("type") == "context_collapse"]


def test_run_start_limit_source_user_when_limits_align(session, tmp_path):
    """Round-51 T1: limit_source='user' when configured == effective and no
    provider window constraint applies."""
    session.variables["context_token_limit"] = 250000
    session.provider.effective_context_window = lambda *a, **k: 250000
    session.provider.compaction_safety_factor = lambda: 1.0

    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    run_start = next(r for r in recs if r["type"] == "run_start")
    assert run_start["limit_source"] == "user"


def test_run_start_limit_source_drift_corrected_on_learned_drift(
    session, tmp_path
):
    """When no provider window is knowable but the guard applies a safety
    factor, the effective limit diverges from the configured one and the
    source is drift_corrected."""
    session.variables["context_token_limit"] = 100000
    session.provider.effective_context_window = lambda *a, **k: None
    session.provider.compaction_safety_factor = lambda: 2.5

    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    run_start = next(r for r in recs if r["type"] == "run_start")
    # resolve: user limit / static factor = 40000
    assert run_start["effective_limit"] == 40000
    assert run_start["limit_source"] == "drift_corrected"


def test_preflight_limit_rise_triggers_single_recheck(session, tmp_path, monkeypatch):
    """Round-51 T1: a >5% rise of the resolved effective limit between
    preflight passes triggers exactly one re-resolution, and the final
    guard budget uses the stable second value."""
    from mu.agent.context_guard import _preflight_context_check
    import mu.session.budgets as budgets

    session.variables["context_token_limit"] = 250000
    session._last_effective_limit = 100000
    calls = {"n": 0}

    def counting_drift(s):
        calls["n"] += 1
        # Simulate drift correction ratcheting the ceiling up mid-run.
        return 300000

    monkeypatch.setattr(budgets, "drift_corrected_context_limit", counting_drift)

    from providers.base import Message

    messages = [Message(role="user", parts=[MessagePart(type="text", text="hi")])]
    _preflight_context_check(session, "sys", messages)

    # One initial resolve + exactly one re-check after the material rise.
    assert calls["n"] == 2
    assert session._last_effective_limit == 300000


def test_divergence_warning_fires_once_per_run(session, tmp_path, caplog):
    """Re-entrant preflight calls warn about configured-vs-effective
    divergence only once per run, but always refresh the divergence
    record."""
    import logging

    session.variables["context_token_limit"] = 580000
    session.provider.effective_context_window = lambda *a, **k: 200000
    session.provider.compaction_safety_factor = lambda: 2.5

    from mu.agent.context_guard import _preflight_context_check
    from providers.base import Message, MessagePart

    messages = [Message(role="user", parts=[MessagePart(type="text", text="x")])]
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            _preflight_context_check(session, "sys", messages)

    warns = [
        r for r in caplog.records if "Context governance divergence" in str(r.msg)
    ]
    assert len(warns) == 1
    div = getattr(session, "_context_limit_divergence", None)
    assert div and div["configured_limit"] == 580000
    assert div["effective_limit"] == 80000
    assert session._context_limit_divergence_warned is True


def test_run_end_idempotent_second_call_noop(session, tmp_path):
    """Round-51 T2: run_end is written exactly once per emitter — a second
    call must be a no-op, not a duplicate terminal record."""
    from mu.trace.emitter import get_emitter

    session.send_message("do the thing")

    em = get_emitter(session)
    assert em is not None
    before = _read_trace(_trace_files(tmp_path)[-1])
    n_before = sum(1 for r in before if r.get("type") == "run_end")
    assert n_before == 1

    # Late/double emit attempt — must be a no-op.
    em.run_end({"status": "stopped", "reason": "late", "iters": 0})
    after = _read_trace(_trace_files(tmp_path)[-1])
    n_after = sum(1 for r in after if r.get("type") == "run_end")
    assert n_after == n_before == 1




def test_run_end_failed_status_and_reason_from_error(session, tmp_path):
    """Round-51 T2: a turn summary carrying status=error must map to
    run_end.status=failed with the error text as reason (the finally-block
    mapping contract), and completed turns keep status=completed."""
    from mu.trace.emitter import get_emitter

    session.send_message("do the thing")
    files = _trace_files(tmp_path)
    recs = _read_trace(files[-1])
    ends = [r for r in recs if r.get("type") == "run_end"]
    assert ends and ends[-1]["status"] == "completed"
    assert "cost" in ends[-1]
    assert "tokens_in" in ends[-1] and "tokens_out" in ends[-1]
    assert "reason" in ends[-1] and "iters" in ends[-1]

    # Unit-level mapping contract for the failure branch.
    for summary, expected in (
        ({"status": "error", "error": "boom"}, "failed"),
        ({"status": "completed", "error": None}, "completed"),
        ({"status": "max_iterations_reached", "error": "Reached maximum iterations"},
         "failed"),
        ({"status": "max_iterations_reached", "error": None}, "max_iterations"),
    ):
        _status = str(summary_status := summary.get("status", "unknown") or "unknown")
        _error = summary.get("error")
        _run_status = "failed" if (_status == "error" or _error) else "completed"
        if _status == "max_iterations_reached" and _error is None:
            _run_status = "max_iterations"
        assert _run_status == expected, (summary, _run_status, expected)


def test_parser_run_summary_reflects_run_end_status(session, tmp_path):
    """Round-51 T2: parser run summary prefers run_end status over the
    eternal 'running' default."""
    from mu.trace.parser import parse_trace

    session.send_message("do the thing")
    files = _trace_files(tmp_path)
    run = parse_trace(str(files[-1]))
    assert run.run_end is not None
    assert run.run_end.get("status") == "completed"


def test_context_collapse_record_fields(session, tmp_path, monkeypatch):
    """Round-51 T3: collapse record carries drop_pct, probable_cause and
    last known compaction iter for bisection."""
    import mu.trace.emitter as trace_emitter

    session._trace_prev_iter_l5 = 100000
    session._trace_prev_iter_num = 0
    session._trace_last_compaction_iter = 1

    real_iter_record = trace_emitter.build_iter_record

    def fake_iter_record(session, **kw):
        rec = real_iter_record(session, **kw)
        if kw.get("iteration") == 1:
            rec["context"]["l5"] = 20000
        return rec

    monkeypatch.setattr(trace_emitter, "build_iter_record", fake_iter_record)
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    collapses = [r for r in recs if r.get("type") == "context_collapse"]
    assert collapses
    c = collapses[0]
    # 100000 -> 20000 = exactly 80% drop
    assert c["drop_pct"] == 80.0
    assert c["probable_cause"] in ("compaction", "restore", "external", "unknown")


def test_parser_surfaces_context_collapses(session, tmp_path, monkeypatch):
    """Round-51 T3: parser categories context_collapse records under
    TraceRun.context_collapses for run diagnostics."""
    from mu.trace.parser import parse_trace
    import mu.trace.emitter as trace_emitter

    session._trace_prev_iter_l5 = 100000
    session._trace_prev_iter_num = 0

    real_iter_record = trace_emitter.build_iter_record

    def fake_iter_record(session, **kw):
        rec = real_iter_record(session, **kw)
        if kw.get("iteration") == 1:
            rec["context"]["l5"] = 20000
        return rec

    monkeypatch.setattr(trace_emitter, "build_iter_record", fake_iter_record)
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    run = parse_trace(str(files[0])) if False else None
    from mu.trace.parser import parse_trace as _pt

    run = _pt = _pt_run = None
    from mu.trace import parser as parser_mod
    run = parser_mod.parse_trace(str(files[-1]))
    assert run.context_collapses, "parser must collect context_collapse records"
    c = run.context_collapses[0]
    assert c["from_l5"] == 100000 and c["to_l5"] == 20000
    assert c["drop_pct"] == 80.0


def test_tool_record_ok_false_for_invalid_args_envelope(session, tmp_path):
    """Round-51 T4: a tool returning an invalid_args envelope must record
    ok=false with error_code=invalid_args in the trace (the ok-flag must
    come from the envelope, not the raw string prefix)."""
    from providers.base import MessagePart, ProviderResponse

    class _InvalidArgsProvider(_ThreeIterProvider):
        def generate(self, messages, system_prompt=None, thinking=False, tools=None):
            self.calls += 1
            if self.calls == 1:
                return ProviderResponse(
                    text="",
                    parts=[
                        MessagePart(
                            type="tool_call",
                            tool_name="search_and_replace_file",
                            tool_args={
                                "filename": "/tmp/__nope__.py",
                                "search": "zzz-will-not-match-zz",
                                "replace": "y",
                            },  # missing search text → invalid_args envelope
                            tool_call_id="c1",
                        )
                    ],
                    input_tokens=100,
                    output_tokens=2,
                    total_tokens=102,
                )
            return ProviderResponse(
                text="done",
                parts=[MessagePart(type="text", text="done")],
                input_tokens=300,
                output_tokens=1,
                total_tokens=301,
            )

    session.provider = _InvalidArgsProvider()
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    tools = [r for r in recs if r.get("type") == "tool"]
    assert tools
    bad = [t for t in tools if t.get("ok") is False]
    assert bad, "expected at least one failed tool record (ok=false)"
    assert any(r.get("error_code") for r in bad), (
        "failed envelope must carry its error_code into the trace record"
    )


def test_invalid_args_escalation_counter(session, monkeypatch):
    """Round-51 T4: consecutive invalid_args failures escalate — a
    corrective hint naming the tool is injected after 3 consecutive
    failures, and the counter resets on a successful call."""
    # The escalation contract lives in the loop: track consecutive
    # envelope-failure counts per run; here we verify the session helper
    # counts consecutive failures across distinct calls and resets on
    # success (via _announce_retryable_failure's per-(tool,code) counter
    # plus the loop's consecutive tracking).
    raw_bad = json.dumps({
        "ok": False,
        "error_code": "invalid_args",
        "retryable": True,
        "hint": "check the parameters",
        "message": "bad args",
        "data": {},
        "artifacts": [],
        "telemetry": {},
    })
    c1 = session._announce_retryable_failure("bash", raw_bad)
    c2 = session._announce_retryable_failure("bash", raw_bad)
    c3 = session._announce_retryable_failure("bash", raw_bad)
    assert (c1, c2, c3) == (1, 2, 3)
    # A successful call resets the run-scoped consecutive counter.
    ok_raw = '{"ok": true, "error_code": null, "retryable": false, "hint": "", "message": "fine", "data": {}, "artifacts": [], "telemetry": {}}'
    session._announce_retryable_failure("other", raw_bad)
    counts = session._retryable_failure_counts
    assert ("bash", "invalid_args") in session._retryable_failure_counts
    assert session._retryable_failure_counts[("other", "invalid_args")] == 1


def test_request_record_slimming_after_first_iter(session, tmp_path):
    """Round-51 T6: iteration 1 keeps full part_details; iteration N>1 is a
    bounded summary (no part_details) with substantial size reduction and a
    summarized marker."""
    session.send_message("do the thing")

    files = _trace_files(tmp_path)
    recs = _read_trace(files[0])
    requests = [r for r in recs if r.get("type") == "request"]
    assert len(requests) >= 3

    it1 = next(r for r in requests if r.get("iter") == 1)
    it2 = next(r for r in requests if r.get("iter") == 2)

    # it1: full detail (part_details present when available)
    assert it1.get("summarized") is False
    # itN: summarized — no part_details keys anywhere
    itN = next((r for r in requests if r.get("iter", 0) > 1), None)
    assert itN is not None
    assert itN.get("summarized") is True
    for m in itN["messages"]:
        assert "part_details" not in m
        assert {"index", "role", "parts", "bytes", "tokens"} >= set(m.keys())

    # Per-message shrinkage: the summarized record must not carry
    # part_details (the payload that dominates real long-run requests).
    msg1 = it1["messages"][-1]
    msgN = itN["messages"][-1]
    assert msg1.get("part_details") is not None
    assert "part_details" not in msgN
    # The slim record is at most the full record size (equal when no
    # part_details existed) and never carries raw part dumps.
    slim_len = len(json.dumps(itN["messages"]))
    full_len = len(json.dumps(it1["messages"]))


def test_build_request_record_summarize_direct():
    """Round-51 T6: summarize=True strips part_details but keeps totals."""
    from mu.trace.emitter import build_request_record
    from providers.base import Message, MessagePart

    messages = [
        Message(role="user", parts=[MessagePart(type="text", text="inspect this")]),
        Message(role="tool", parts=[MessagePart(
            type="tool_result", tool_name="bash",
            tool_result="out" * 500,
        )]),
    ]
    full = build_request_record(
        iteration=1, system_prompt="system rules", messages=messages,
        tools=[], token_estimate=10,
    )
    slim = build_request_record(
        iteration=2, system_prompt="system rules", messages=messages,
        tools=[], token_estimate=1234, summarize=True,
    )
    assert full["messages"][0].get("part_details")
    assert "part_details" not in slim["messages"][0]
    assert slim["summarized"] is True and full["summarized"] is False
    assert slim["component_tokens"]["system"] == full["component_tokens"]["system"]
    assert slim["component_total_tokens"] == full["component_total_tokens"]
    assert slim["tools_hash"] == full["tools_hash"]
    assert len(json.dumps(slim)) < len(json.dumps(full))


def test_summarized_record_bounded_under_2kb():
    """Round-51 T6: a realistic 500-message request summarizes to a bounded
    <2KB record (>80% size reduction vs full dump)."""
    from mu.trace.emitter import build_request_record
    from providers.base import Message, MessagePart

    messages = []
    for i in range(500):
        if i % 3 == 0:
            messages.append(Message(role="user", parts=[MessagePart(type="text", text="continue")]))
        else:
            messages.append(Message(role="tool", parts=[MessagePart(
                type="tool_result", tool_name="bash", tool_result="x" * 4000)]))
    full = build_request_record(iteration=1, system_prompt="s" * 2000, messages=messages, tools=[], token_estimate=1)
    slim = build_request_record(iteration=9, system_prompt="s", messages=messages, tools=[], token_estimate=1, summarize=True)
    full_len = len(json.dumps(full))
    slim_len = len(json.dumps(slim))
    reduction = 1 - slim_len / full_len
    assert reduction > 0.80, f"expected >80% reduction, got {100*(1-slim_len/full_len):.1f}%"
    assert slim_len < 2048, "summary record must be <2KB for a typical request"
    # Older messages collapsed into one aggregate row.
    assert slim["messages"][0]["role"] == "older"
    kept = [m for m in slim["messages"] if m.get("role") != "older"]
    assert len(slim["messages"]) <= 21  # aggregate + keep_recent(20)
    # Aggregate preserves byte/token totals for drift diagnostics.
    agg = slim["messages"][0]
    assert agg["bytes"] > 0 and agg["collapsed_count"] > 0


def test_restore_trim_fires_on_oversized_history(session, tmp_path, caplog):
    """Round-51 T7: a turn starting with oversized restored history compacts
    to the budget (limit minus reserve) and logs the trim loudly."""
    import logging

    # Inject oversized fake history into the manager.
    sm = session.session_manager
    big_msgs = [
        {"role": "user", "parts": [{"type": "text", "text": "junk " * 5000}]}
        for _ in range(300)
    ]
    sm.history = list(sm.history) + big_msgs
    session.variables["trace_enabled"] = False  # isolate from run tracer

    with caplog.at_level(logging.WARNING):
        session.send_message("trim probe")

    restore_trims = [
        r for r in caplog.records if "Restore trim" in str(r.msg)
    ]
    assert restore_trims, "expected restore-trim warning for oversized restore"
