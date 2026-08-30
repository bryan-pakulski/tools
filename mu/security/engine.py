"""Security scan engine — mirrors `mu/feature/engine.py` for security audits.

The mode's anti-hallucination contract:

  * Every claimed vulnerability is provisional until its PoC actually runs.
  * `verify_security_proof` executes the PoC and checks that the model's
    declared `expected_markers` literally appear in the captured output.
    If they don't, `proof_verified=False` and the model must revise.
  * Every patch is provisional until `verify_remediation` re-runs the PoC
    post-fix and confirms the markers are no longer present (the exploit
    no longer triggers).
  * `approve_security_finding` refuses unless both flags are True.

Persistence layout (per-scan directory under `documentation/`):

  documentation/security_scan_<id>/
    report.json           # SecurityReport state
    README.md             # human index
    findings/<finding_id>.md
    proofs/<finding_id>/  # captured stdout/stderr from PoC runs
    patches/<finding_id>.diff
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.config import HISTORY_DIR


# ----------------------------------------------------------------- status sets

FINDING_STATUSES = {
    "new",                  # finding posted, no proof yet
    "proof_attached",       # proof recipe stored, not executed
    "exploit_verified",     # PoC ran and markers matched
    "remediation_attached", # patch stored, not yet re-verified
    "remediation_verified", # post-patch run failed to trigger markers
    "approved",             # final; only reachable from remediation_verified
    "refuted",              # proof failed to verify after retries — abandoned
}


SEVERITY_LEVELS = ("info", "low", "medium", "high", "critical")


# ----------------------------------------------------------------- data model


@dataclass
class SecurityProof:
    """Executable recipe that demonstrates a vulnerability."""

    kind: str = "command"           # "command" | "script" | "http"
    command: str = ""               # shell command run from workspace root
    expected_markers: list[str] = field(default_factory=list)
    description: str = ""
    captured_stdout: str = ""
    captured_stderr: str = ""
    exit_code: int | None = None
    verified: bool = False
    verified_at: float | None = None
    last_run_at: float | None = None
    last_run_evidence_path: str = ""
    notes: str = ""


@dataclass
class SecurityRemediation:
    """Proposed fix + post-patch re-verification."""

    description: str = ""
    patch_diff: str = ""             # unified diff the user should apply
    patch_applied_path: str = ""     # where the diff is stored on disk
    post_patch_stdout: str = ""
    post_patch_stderr: str = ""
    post_patch_exit_code: int | None = None
    verified: bool = False
    verified_at: float | None = None
    notes: str = ""


@dataclass
class SecurityFinding:
    finding_id: str
    title: str
    summary: str = ""
    vulnerability_class: str = ""    # sql_injection, xss, path_traversal, etc.
    severity: str = "medium"
    affected_paths: list[str] = field(default_factory=list)
    exploit_path: str = ""           # human-readable trace
    references: list[str] = field(default_factory=list)
    status: str = "new"
    proof: SecurityProof | None = None
    remediation: SecurityRemediation | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class SecurityReport:
    scan_id: str
    title: str
    summary: str = ""
    workspace_root: str = ""
    directory: str = ""              # documentation/security_scan_<scan_id>
    metadata_path: str = ""          # absolute path to report.json
    status: str = "scanning"         # scanning | findings_pending | complete
    findings: list[SecurityFinding] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---------------- lookup helpers
    def find(self, finding_id: str) -> SecurityFinding | None:
        for f in self.findings:
            if f.finding_id == finding_id:
                return f
        return None


# ------------------------------------------------------------------ slug / paths


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug or "scan"


def _workspace_root(folder_context) -> str:
    if (
        folder_context
        and getattr(folder_context, "folders", None)
        and folder_context.folders
    ):
        return os.path.abspath(folder_context.folders[0])
    return os.getcwd()


def _scan_directory(workspace_root: str, scan_id: str) -> str:
    return os.path.join(workspace_root, "documentation", f"security_scan_{scan_id}")


def _resolve_metadata_path(
    *,
    directory: str,
    session_id: str | None,
    scan_id: str,
) -> str:
    if directory:
        return os.path.join(directory, "report.json")
    base = (
        os.path.join(HISTORY_DIR, "sessions", session_id, "security")
        if session_id
        else HISTORY_DIR
    )
    return os.path.join(base, f"security_scan_{scan_id}.json")


# ------------------------------------------------------------------ persistence


def save_report(session_id: str, report: SecurityReport) -> SecurityReport:
    report.metadata_path = _resolve_metadata_path(
        directory=report.directory,
        session_id=session_id,
        scan_id=report.scan_id,
    )
    report.updated_at = time.time()
    os.makedirs(os.path.dirname(report.metadata_path), exist_ok=True)
    with open(report.metadata_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2, default=str)
    return report


def _proof_from_dict(data: dict) -> SecurityProof | None:
    """Round-28 F2: tolerate unknown/stale keys from newer or corrupt
    reports instead of TypeError-crashing the whole load.

    Round-29 F2: a dict that carries no usable proof (no command AND no
    expected markers) deserializes to None — a default-constructed
    proof would sail through verify_proof's empty-marker check and
    verify a finding with zero evidence.
    """
    valid = {f for f in SecurityProof.__dataclass_fields__}
    try:
        proof = SecurityProof(**{k: v for k, v in data.items() if k in valid})
    except (TypeError, ValueError):
        return None
    if not str(proof.command or "").strip():
        return None
    if not [m for m in (proof.expected_markers or []) if str(m).strip()]:
        return None
    return proof


def _remediation_from_dict(data: dict) -> SecurityRemediation | None:
    try:
        valid = {f for f in SecurityRemediation.__dataclass_fields__}
        return SecurityRemediation(**{k: v for k, v in data.items() if k in valid})
    except (TypeError, ValueError):
        return None


def load_report(metadata_path: str) -> SecurityReport | None:
    if not metadata_path or not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    findings: list[SecurityFinding] = []
    for raw in data.get("findings", []) or []:
        proof = raw.get("proof")
        remediation = raw.get("remediation")
        findings.append(
            SecurityFinding(
                finding_id=raw.get("finding_id", ""),
                title=raw.get("title", ""),
                summary=raw.get("summary", ""),
                vulnerability_class=raw.get("vulnerability_class", ""),
                severity=raw.get("severity", "medium"),
                affected_paths=list(raw.get("affected_paths", []) or []),
                exploit_path=raw.get("exploit_path", ""),
                references=list(raw.get("references", []) or []),
                status=raw.get("status", "new"),
                proof=(
                    _proof_from_dict(proof) if isinstance(proof, dict) else None
                ),
                remediation=(
                    _remediation_from_dict(remediation)
                    if isinstance(remediation, dict)
                    else None
                ),
                created_at=float(raw.get("created_at", time.time())),
                updated_at=float(raw.get("updated_at", time.time())),
            )
        )
    return SecurityReport(
        scan_id=data.get("scan_id", ""),
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        workspace_root=data.get("workspace_root", ""),
        directory=data.get("directory", ""),
        metadata_path=data.get("metadata_path", metadata_path),
        status=data.get("status", "scanning"),
        findings=findings,
        created_at=float(data.get("created_at", time.time())),
        updated_at=float(data.get("updated_at", time.time())),
    )


# ------------------------------------------------------------------ scaffolding


def _initialize_scan_docs(report: SecurityReport) -> None:
    os.makedirs(report.directory, exist_ok=True)
    os.makedirs(os.path.join(report.directory, "findings"), exist_ok=True)
    os.makedirs(os.path.join(report.directory, "proofs"), exist_ok=True)
    os.makedirs(os.path.join(report.directory, "patches"), exist_ok=True)
    readme_path = os.path.join(report.directory, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# Security Scan Report\n\n"
                f"- **Scan ID:** `{report.scan_id}`\n"
                f"- **Title:** {report.title}\n"
                f"- **Workspace:** `{report.workspace_root}`\n\n"
                "Findings live under `findings/`. Every approved finding has a "
                "verified PoC under `proofs/<finding_id>/` and a verified patch "
                "under `patches/<finding_id>.diff`.\n\n"
                "**No finding is approved without both verifications.**\n"
            )


def create_security_report(
    *,
    title: str,
    summary: str = "",
    folder_context: Any = None,
    session_id: str | None = None,
    scan_id: str | None = None,
) -> SecurityReport:
    sid = _slugify(scan_id or f"{title}_{uuid.uuid4().hex[:6]}")
    workspace = _workspace_root(folder_context)
    directory = _scan_directory(workspace, sid)
    report = SecurityReport(
        scan_id=sid,
        title=title.strip() or "Untitled security scan",
        summary=summary.strip(),
        workspace_root=workspace,
        directory=directory,
        status="scanning",
    )
    _initialize_scan_docs(report)
    return save_report(session_id or "", report)


# ------------------------------------------------------------------ finding ops


def _allocate_finding_id(report: SecurityReport) -> str:
    """Round-28 F3: collision-free even after findings are removed —
    allocate one past the highest existing numeric id, not len()+1
    (len() re-allocates a removed id and breaks evidence paths)."""
    highest = 0
    for finding in report.findings:
        match = re.fullmatch(r"f(\d+)", finding.finding_id or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"f{highest + 1:03d}"


def add_finding(
    report: SecurityReport,
    *,
    title: str,
    summary: str = "",
    vulnerability_class: str = "",
    severity: str = "medium",
    affected_paths: list[str] | None = None,
    exploit_path: str = "",
    references: list[str] | None = None,
) -> SecurityFinding:
    sev = severity if severity in SEVERITY_LEVELS else "medium"
    finding = SecurityFinding(
        finding_id=_allocate_finding_id(report),
        title=title.strip() or "(untitled)",
        summary=summary.strip(),
        vulnerability_class=str(vulnerability_class or "").strip(),
        severity=sev,
        affected_paths=list(affected_paths or []),
        exploit_path=exploit_path.strip(),
        references=list(references or []),
        status="new",
    )
    report.findings.append(finding)
    report.updated_at = time.time()
    return finding


def attach_proof(
    finding: SecurityFinding,
    *,
    command: str,
    expected_markers: list[str],
    description: str = "",
    kind: str = "command",
) -> SecurityProof:
    proof = SecurityProof(
        kind=str(kind or "command"),
        command=str(command or "").strip(),
        expected_markers=[str(m) for m in (expected_markers or []) if str(m).strip()],
        description=str(description or "").strip(),
    )
    if not proof.command:
        raise ValueError("proof command must be a non-empty string")
    if not proof.expected_markers:
        raise ValueError("proof must declare at least one expected_marker")
    finding.proof = proof
    finding.status = "proof_attached"
    finding.updated_at = time.time()
    return proof


def attach_remediation(
    finding: SecurityFinding,
    *,
    description: str,
    patch_diff: str,
    patch_applied_path: str = "",
) -> SecurityRemediation:
    if finding.proof is None or not finding.proof.verified:
        raise ValueError(
            "remediation cannot be attached before the proof is verified — "
            "the exploit must be demonstrably real first"
        )
    rem = SecurityRemediation(
        description=str(description or "").strip(),
        patch_diff=str(patch_diff or "").strip(),
        patch_applied_path=str(patch_applied_path or "").strip(),
    )
    if not rem.patch_diff:
        raise ValueError("remediation patch_diff must be a non-empty unified diff")
    finding.remediation = rem
    finding.status = "remediation_attached"
    finding.updated_at = time.time()
    return rem


# ------------------------------------------------------------------ execution


def _markers_present(text: str, markers: list[str]) -> tuple[bool, list[str]]:
    missing = [m for m in markers if m not in text]
    return (not missing, missing)


def run_proof_command(
    proof: SecurityProof,
    *,
    cwd: str,
    timeout: float = 60.0,
) -> tuple[int, str, str]:
    """Execute the PoC. Returns (exit_code, stdout, stderr).

    Runs through `/bin/bash -lc` so shell pipelines / redirects work.
    Wall-clock bounded by `timeout`; on timeout the partial output is
    still returned and exit_code is -1.
    """
    if not cwd or not os.path.isdir(cwd):
        cwd = os.getcwd()
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", proof.command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired as e:
        return -1, (e.stdout or "") if isinstance(e.stdout, str) else "", (
            f"timed out after {timeout}s"
        )


def verify_proof(
    finding: SecurityFinding,
    *,
    cwd: str,
    report: SecurityReport | None = None,
    timeout: float = 60.0,
) -> SecurityProof:
    """Run the PoC and decide if the exploit actually triggered."""
    if finding.proof is None:
        raise ValueError("attach a proof before verifying")
    proof = finding.proof
    # Round-29 F2: an EMPTY marker list vacuously matches (`not missing`
    # is True with zero markers) — refuse instead of verifying nothing.
    if not [m for m in (proof.expected_markers or []) if str(m).strip()]:
        raise ValueError(
            "proof has no expected markers — verification would be vacuous"
        )
    exit_code, stdout, stderr = run_proof_command(proof, cwd=cwd, timeout=timeout)
    combined = stdout + "\n" + stderr
    matched, missing = _markers_present(combined, proof.expected_markers)
    proof.captured_stdout = stdout
    proof.captured_stderr = stderr
    proof.exit_code = exit_code
    proof.last_run_at = time.time()
    proof.verified = matched
    if matched:
        proof.verified_at = time.time()
        finding.status = "exploit_verified"
    else:
        proof.notes = (
            f"missing expected marker(s) in output: {missing!r}\n"
            f"exit_code={exit_code}"
        )
        finding.status = "proof_attached"
    finding.updated_at = time.time()
    if report is not None:
        _persist_proof_evidence(report, finding)
    return proof


def verify_remediation(
    finding: SecurityFinding,
    *,
    cwd: str,
    report: SecurityReport | None = None,
    timeout: float = 60.0,
) -> SecurityRemediation:
    """Re-run the proof; the markers must NOT appear post-patch."""
    if finding.remediation is None:
        raise ValueError("attach a remediation before verifying it")
    if finding.proof is None or not finding.proof.verified:
        raise ValueError(
            "cannot verify remediation: the original proof was never verified, "
            "so 'no longer triggering' is meaningless"
        )
    rem = finding.remediation
    exit_code, stdout, stderr = run_proof_command(
        finding.proof, cwd=cwd, timeout=timeout
    )
    combined = stdout + "\n" + stderr
    still_present, _ = _markers_present(combined, finding.proof.expected_markers)
    rem.post_patch_stdout = stdout
    rem.post_patch_stderr = stderr
    rem.post_patch_exit_code = exit_code
    rem.verified = not still_present
    if rem.verified:
        rem.verified_at = time.time()
        finding.status = "remediation_verified"
    else:
        rem.notes = (
            "exploit markers still present after patch — remediation does not "
            "actually fix the vulnerability"
        )
        finding.status = "remediation_attached"
    finding.updated_at = time.time()
    if report is not None:
        _persist_remediation_evidence(report, finding)
    return rem


def approve_finding(finding: SecurityFinding) -> SecurityFinding:
    """Move a finding to 'approved'. Hard-gated on dual verification."""
    if finding.proof is None or not finding.proof.verified:
        raise ValueError("cannot approve: proof has not been verified")
    if finding.remediation is None or not finding.remediation.verified:
        raise ValueError("cannot approve: remediation has not been verified")
    finding.status = "approved"
    finding.updated_at = time.time()
    return finding


def refute_finding(finding: SecurityFinding, *, reason: str = "") -> SecurityFinding:
    """Abandon a finding (e.g. PoC couldn't be made to trigger after
    multiple revisions). The model should call this rather than silently
    moving on, so the audit trail records that the hypothesis was tested
    and rejected."""
    finding.status = "refuted"
    if finding.proof is not None:
        finding.proof.notes = (
            (finding.proof.notes or "")
            + (f"\nrefuted: {reason}" if reason else "\nrefuted")
        ).strip()
    finding.updated_at = time.time()
    return finding


# ------------------------------------------------------------------ evidence dirs


def _persist_proof_evidence(report: SecurityReport, finding: SecurityFinding) -> None:
    if not report.directory or finding.proof is None:
        return
    proof_dir = os.path.join(report.directory, "proofs", finding.finding_id)
    os.makedirs(proof_dir, exist_ok=True)
    stdout_path = os.path.join(proof_dir, "stdout.txt")
    stderr_path = os.path.join(proof_dir, "stderr.txt")
    meta_path = os.path.join(proof_dir, "meta.json")
    with open(stdout_path, "w", encoding="utf-8") as fh:
        fh.write(finding.proof.captured_stdout or "")
    with open(stderr_path, "w", encoding="utf-8") as fh:
        fh.write(finding.proof.captured_stderr or "")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "command": finding.proof.command,
                "expected_markers": finding.proof.expected_markers,
                "exit_code": finding.proof.exit_code,
                "verified": finding.proof.verified,
                "verified_at": finding.proof.verified_at,
            },
            fh,
            indent=2,
        )
    finding.proof.last_run_evidence_path = proof_dir


def _persist_remediation_evidence(
    report: SecurityReport, finding: SecurityFinding
) -> None:
    if not report.directory or finding.remediation is None:
        return
    patch_path = os.path.join(
        report.directory, "patches", f"{finding.finding_id}.diff"
    )
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    with open(patch_path, "w", encoding="utf-8") as fh:
        fh.write(finding.remediation.patch_diff or "")
    finding.remediation.patch_applied_path = patch_path


# ------------------------------------------------------------------ summary


def summarize_report(report: SecurityReport) -> dict:
    by_status: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for f in report.findings:
        by_status[f.status] = by_status.get(f.status, 0) + 1
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
    return {
        "scan_id": report.scan_id,
        "title": report.title,
        "status": report.status,
        "directory": report.directory,
        "metadata_path": report.metadata_path,
        "findings_total": len(report.findings),
        "by_status": by_status,
        "by_severity": by_severity,
        "approved": [
            {"finding_id": f.finding_id, "title": f.title, "severity": f.severity}
            for f in report.findings
            if f.status == "approved"
        ],
    }


__all__ = [
    "FINDING_STATUSES",
    "SEVERITY_LEVELS",
    "SecurityProof",
    "SecurityRemediation",
    "SecurityFinding",
    "SecurityReport",
    "add_finding",
    "approve_finding",
    "attach_proof",
    "attach_remediation",
    "create_security_report",
    "load_report",
    "refute_finding",
    "run_proof_command",
    "save_report",
    "summarize_report",
    "verify_proof",
    "verify_remediation",
]
