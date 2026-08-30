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
    assert types[-1] == "turn_end"

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

    turn_end = recs[-1]
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
