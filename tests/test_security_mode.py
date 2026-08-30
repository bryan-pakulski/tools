"""Tests for the security audit engine.

The anti-hallucination contract is the entire point — every layer must
refuse to advance a finding without a real, executed PoC and a real,
executed post-patch re-verification.

Coverage:
  * Data model round-trip (save → load preserves all fields).
  * Mode + prompt registration in AGENTIC_MODES.
  * `verify_security_proof` actually runs the command and matches markers.
  * Missing markers leave proof.verified=False with diagnostics.
  * `attach_remediation` refuses when proof isn't verified yet.
  * `verify_remediation` flips back to False if PoC still triggers post-"patch".
  * `approve_finding` refuses unless BOTH flags are True.
  * Plan-mode blocks `verify_security_proof` / `verify_remediation`.
  * The tool handlers wire all of the above through `ToolExecutionContext`.
"""

import json
import os

import pytest

import tempfile

from mu.security.engine import (
    SEVERITY_LEVELS,
    SecurityFinding,
    SecurityProof,
    SecurityRemediation,
    SecurityReport,
    add_finding,
    approve_finding,
    attach_proof,
    attach_remediation,
    create_security_report,
    load_report,
    refute_finding,
    save_report,
    summarize_report,
    verify_proof,
    verify_remediation,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def report(workspace):
    class _FC:
        folders = [str(workspace)]

    return create_security_report(
        title="audit-1",
        summary="initial audit",
        folder_context=_FC(),
        session_id="",
    )


# ============================================================ data model


def test_create_report_scaffolds_directories(report, workspace):
    assert os.path.isdir(report.directory)
    assert os.path.isdir(os.path.join(report.directory, "findings"))
    assert os.path.isdir(os.path.join(report.directory, "proofs"))
    assert os.path.isdir(os.path.join(report.directory, "patches"))
    assert os.path.isfile(os.path.join(report.directory, "report.json"))
    assert os.path.isfile(os.path.join(report.directory, "README.md"))


def test_finding_starts_in_new_status(report):
    f = add_finding(
        report,
        title="xss in /search",
        vulnerability_class="xss",
        severity="high",
        affected_paths=["app/views/search.py"],
        exploit_path="user query reflected without escaping",
    )
    assert f.status == "new"
    assert f.proof is None
    assert f.remediation is None
    assert f.severity == "high"


def test_invalid_severity_falls_back_to_medium(report):
    f = add_finding(report, title="x", severity="apocalyptic")
    assert f.severity == "medium"


def test_attach_proof_requires_command_and_markers(report):
    f = add_finding(report, title="x", severity="low")
    with pytest.raises(ValueError):
        attach_proof(f, command="", expected_markers=["X"])
    with pytest.raises(ValueError):
        attach_proof(f, command="echo hi", expected_markers=[])


def test_save_then_load_roundtrips_findings(report):
    f = add_finding(
        report, title="path traversal", severity="critical", vulnerability_class="path_traversal"
    )
    attach_proof(
        f,
        command="echo MARKER-OK",
        expected_markers=["MARKER-OK"],
        description="trivial proof",
    )
    save_report("", report)
    loaded = load_report(report.metadata_path)
    assert loaded is not None
    assert loaded.scan_id == report.scan_id
    assert len(loaded.findings) == 1
    g = loaded.findings[0]
    assert g.title == "path traversal"
    assert g.severity == "critical"
    assert g.vulnerability_class == "path_traversal"
    assert g.proof is not None
    assert g.proof.expected_markers == ["MARKER-OK"]


# ============================================================ verification


def test_verify_proof_succeeds_when_markers_present(report, workspace):
    f = add_finding(report, title="x", severity="medium")
    attach_proof(
        f, command="echo PWNED-7421", expected_markers=["PWNED-7421"]
    )
    verify_proof(f, cwd=str(workspace), report=report)
    assert f.proof.verified is True
    assert f.status == "exploit_verified"
    assert "PWNED-7421" in f.proof.captured_stdout
    # Evidence persisted under proofs/<finding_id>/.
    ev = os.path.join(report.directory, "proofs", f.finding_id, "stdout.txt")
    assert os.path.isfile(ev)
    with open(ev, encoding="utf-8") as fh:
        assert "PWNED-7421" in fh.read()


def test_verify_proof_fails_when_markers_missing(report, workspace):
    f = add_finding(report, title="x", severity="medium")
    attach_proof(
        f,
        command="echo something_else",
        expected_markers=["EXPECTED-MARKER-NEVER-PRESENT"],
    )
    verify_proof(f, cwd=str(workspace), report=report)
    assert f.proof.verified is False
    assert f.status == "proof_attached"
    assert "missing expected marker" in f.proof.notes


def test_verify_proof_records_exit_code(report, workspace):
    f = add_finding(report, title="x", severity="medium")
    attach_proof(f, command="echo OK && exit 7", expected_markers=["OK"])
    verify_proof(f, cwd=str(workspace), report=report)
    assert f.proof.exit_code == 7
    # Markers still matched, so exploit_verified despite non-zero exit.
    assert f.proof.verified is True


# ============================================================ remediation gating


def test_attach_remediation_refuses_when_proof_unverified(report):
    f = add_finding(report, title="x", severity="medium")
    attach_proof(f, command="echo MARKER", expected_markers=["MARKER"])
    # Proof attached but not verified yet.
    with pytest.raises(ValueError) as exc:
        attach_remediation(f, description="d", patch_diff="--- a\n+++ b\n")
    assert "proof is verified" in str(exc.value) or "proof" in str(exc.value).lower()


def test_verify_remediation_flips_when_exploit_still_triggers(report, workspace):
    """Simulate a "patch that doesn't actually fix anything" — the PoC
    still triggers, so remediation must be marked unverified."""
    f = add_finding(report, title="x", severity="critical")
    attach_proof(f, command="echo STILL-VULN", expected_markers=["STILL-VULN"])
    verify_proof(f, cwd=str(workspace), report=report)
    assert f.proof.verified is True

    attach_remediation(
        f, description="cosmetic only", patch_diff="--- noop\n+++ noop\n"
    )
    verify_remediation(f, cwd=str(workspace), report=report)
    assert f.remediation.verified is False
    assert f.status == "remediation_attached"
    assert "still present" in f.remediation.notes


def test_verify_remediation_succeeds_when_exploit_no_longer_triggers(
    report, workspace
):
    """The PoC depends on a sentinel file. Verify the proof while the file
    exists, then 'apply the patch' (delete the file), and re-verify. The
    exploit must now fail to trigger."""
    sentinel = workspace / "trigger.txt"
    sentinel.write_text("PWNED-XYZ\n", encoding="utf-8")

    f = add_finding(report, title="x", severity="high")
    attach_proof(
        f,
        command="cat trigger.txt 2>/dev/null || true",
        expected_markers=["PWNED-XYZ"],
    )
    verify_proof(f, cwd=str(workspace), report=report)
    assert f.proof.verified is True

    attach_remediation(
        f, description="remove the sentinel", patch_diff="--- a\n+++ b\n"
    )
    sentinel.unlink()  # "apply the patch"
    verify_remediation(f, cwd=str(workspace), report=report)
    assert f.remediation.verified is True
    assert f.status == "remediation_verified"


# ============================================================ approval gating


def test_approve_refuses_unverified_proof(report):
    f = add_finding(report, title="x", severity="medium")
    with pytest.raises(ValueError):
        approve_finding(f)


def test_approve_refuses_when_only_proof_verified(report, workspace):
    f = add_finding(report, title="x", severity="medium")
    attach_proof(f, command="echo OK", expected_markers=["OK"])
    verify_proof(f, cwd=str(workspace), report=report)
    with pytest.raises(ValueError) as exc:
        approve_finding(f)
    assert "remediation" in str(exc.value).lower()


def test_approve_succeeds_only_when_both_verified(report, workspace):
    sentinel = workspace / "trig.txt"
    sentinel.write_text("PWN", encoding="utf-8")
    f = add_finding(report, title="x", severity="high")
    attach_proof(
        f, command="cat trig.txt 2>/dev/null || true", expected_markers=["PWN"]
    )
    verify_proof(f, cwd=str(workspace), report=report)
    attach_remediation(f, description="d", patch_diff="--- a\n+++ b\n")
    sentinel.unlink()
    verify_remediation(f, cwd=str(workspace), report=report)
    approve_finding(f)
    assert f.status == "approved"


def test_refute_finding_records_reason(report):
    f = add_finding(report, title="false alarm", severity="low")
    attach_proof(f, command="echo no", expected_markers=["never-matches"])
    refute_finding(f, reason="could not construct working exploit")
    assert f.status == "refuted"
    assert "could not construct" in f.proof.notes


# ============================================================ mode prompt + tools


def test_security_mode_registered_in_agentic_modes():
    from utils.config import AGENTIC_MODES, AGENT_MODE_METADATA

    assert "security" in AGENTIC_MODES
    body = AGENTIC_MODES["security"]
    # The anti-hallucination contract MUST appear in the prompt.
    assert "verify_security_proof" in body
    assert "verify_remediation" in body
    assert "approve_security_finding" in body
    # Mode is exposed to the /mode UI selector.
    assert "security" in AGENT_MODE_METADATA


def test_security_mode_in_mode_choices_for_autocomplete():
    from mu.ui.input import MODE_CHOICES

    assert "security" in MODE_CHOICES


def test_plan_mode_blocks_security_verification_tools():
    """`verify_security_proof` and `verify_remediation` actually execute
    shell commands against the workspace. Plan mode must block them."""
    from mu.agent.plan_mode import WRITE_TOOLS

    assert "verify_security_proof" in WRITE_TOOLS
    assert "verify_remediation" in WRITE_TOOLS
    # Pure record-keeping tools are NOT blocked — model can plan.
    assert "add_security_finding" not in WRITE_TOOLS
    assert "approve_security_finding" not in WRITE_TOOLS


def test_all_security_tools_registered_in_tools_module():
    from mu.tools._dispatcher import TOOL_HANDLERS
    from mu.tools.descriptors import TOOLS

    expected = {
        "create_security_report",
        "add_security_finding",
        "attach_security_proof",
        "verify_security_proof",
        "attach_remediation_patch",
        "verify_remediation",
        "approve_security_finding",
        "refute_security_finding",
        "get_security_state",
    }
    names = {t.name for t in TOOLS}
    assert expected.issubset(names)
    for n in expected:
        assert n in TOOL_HANDLERS, f"{n} missing from TOOL_HANDLERS"


# ============================================================ end-to-end via handlers


class _FakeContext:
    """Minimal ToolExecutionContext look-alike."""

    def __init__(self, workspace_path, session_obj):
        class _FC:
            folders = [str(workspace_path)]
        self.folder_context = _FC()
        self.ui = None
        self.variables = {}
        self.invocation_source = "test"
        self.session = session_obj


class _FakeSession:
    """Minimal Session look-alike — only the attrs the handlers read."""

    def __init__(self):
        class _SM:
            current_session_name = ""
        self.session_manager = _SM()
        self._active_security_report = None
        self._active_security_session_id = ""


def test_handlers_drive_full_audit_lifecycle(workspace):
    """Walk a finding from create → add → proof attach → verify → patch
    attach → verify remediation → approve — all through the tool handlers.
    Smoke test for the integration boundary."""
    from mu.tools.security.handlers import (
        _handle_add_security_finding,
        _handle_approve_security_finding,
        _handle_attach_remediation_patch,
        _handle_attach_security_proof,
        _handle_create_security_report,
        _handle_get_security_state,
        _handle_verify_remediation,
        _handle_verify_security_proof,
    )

    sentinel = workspace / "p.txt"
    sentinel.write_text("PWNED-E2E", encoding="utf-8")
    session = _FakeSession()
    ctx = _FakeContext(workspace, session)

    result = json.loads(
        _handle_create_security_report({"title": "e2e"}, ctx)
    )
    assert result["findings_total"] == 0

    result = json.loads(
        _handle_add_security_finding(
            {"title": "sentinel read", "severity": "high"}, ctx
        )
    )
    fid = result["finding_id"]
    assert result["status"] == "new"

    result = json.loads(
        _handle_attach_security_proof(
            {
                "finding_id": fid,
                "command": "cat p.txt 2>/dev/null || true",
                "expected_markers": ["PWNED-E2E"],
            },
            ctx,
        )
    )
    assert result["status"] == "proof_attached"

    result = json.loads(_handle_verify_security_proof({"finding_id": fid}, ctx))
    assert result["proof_verified"] is True
    assert result["status"] == "exploit_verified"

    result = json.loads(
        _handle_attach_remediation_patch(
            {
                "finding_id": fid,
                "description": "remove sentinel",
                "patch_diff": "--- a\n+++ b\n",
            },
            ctx,
        )
    )
    assert result["status"] == "remediation_attached"

    sentinel.unlink()
    result = json.loads(_handle_verify_remediation({"finding_id": fid}, ctx))
    assert result["remediation_verified"] is True
    assert result["status"] == "remediation_verified"

    result = json.loads(_handle_approve_security_finding({"finding_id": fid}, ctx))
    assert result["status"] == "approved"

    state = json.loads(_handle_get_security_state({}, ctx))
    assert state["findings_total"] == 1
    assert state["by_status"].get("approved") == 1
    assert state["approved"][0]["finding_id"] == fid


def test_handlers_refuse_approval_without_dual_verification(workspace):
    from mu.tools.security.handlers import (
        _handle_add_security_finding,
        _handle_approve_security_finding,
        _handle_attach_security_proof,
        _handle_create_security_report,
        _handle_verify_security_proof,
    )

    session = _FakeSession()
    ctx = _FakeContext(workspace, session)

    json.loads(_handle_create_security_report({"title": "e2e"}, ctx))
    fid = json.loads(
        _handle_add_security_finding(
            {"title": "x", "severity": "medium"}, ctx
        )
    )["finding_id"]
    json.loads(
        _handle_attach_security_proof(
            {
                "finding_id": fid,
                "command": "echo OK",
                "expected_markers": ["OK"],
            },
            ctx,
        )
    )
    json.loads(_handle_verify_security_proof({"finding_id": fid}, ctx))

    # Proof verified but no remediation — approval must refuse.
    result = json.loads(_handle_approve_security_finding({"finding_id": fid}, ctx))
    assert "error" in result
    assert "remediation" in result["error"].lower()


def test_garbage_proof_loads_as_none_and_cannot_verify():
    """Round-29 F2: a corrupt/hand-edited report whose proof dict
    carries no usable fields must deserialize to proof=None (not a
    default proof), so verify_proof refuses — a vacuous empty-marker
    check would 'verify' a finding with zero evidence."""
    import json as _json

    report = SecurityReport(scan_id="r29-garbage", title="t")
    report.directory = tempfile.mkdtemp()
    save_report("s", report)
    data = _json.load(open(report.metadata_path))
    data["findings"] = [
        {"finding_id": "f001", "title": "x", "proof": {"garbage": 1}}
    ]
    _json.dump(data, open(report.metadata_path, "w"))
    loaded = load_report(report.metadata_path)
    assert loaded is not None and loaded.findings[0].proof is None
    with pytest.raises(ValueError, match="attach a proof"):
        verify_proof(loaded.findings[0], cwd=tempfile.gettempdir())


def test_empty_marker_proof_refused():
    """Round-29 F2: an empty expected_markers list vacuously matches
    (`not missing` with zero markers) — verify_proof must refuse."""
    from mu.security.engine import SecurityProof, SecurityFinding

    finding = SecurityFinding(
        finding_id="f001",
        title="x",
        proof=SecurityProof(kind="command", command="echo hi",
                            expected_markers=[]),
    )
    with pytest.raises(ValueError, match="vacuous"):
        verify_proof(finding, cwd=tempfile.gettempdir())
