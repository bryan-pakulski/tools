"""Terminal-Bench installed-agent adapter for mucli.

Wraps mucli as a Terminal-Bench agent: the harness installs mucli into the
task container, feeds it the task instruction headlessly, and mucli works in
``/app`` until done. Grading stays with the task's own verifier scripts —
mucli's JSONL traces are additionally captured into the agent logs dir for
post-run harness analysis (tokens, wall-clock, compactions).

Usage (TB 0.2.x):
    tb run --dataset terminal-bench-core==0.1.1 \
        --agent-import-path bench.tb_mucli_agent.MucliTBAgent \
        --task-id hello-world --model <provider/model>

Env:
    MUCLI_BENCH_REPO  optional; repo dir to copy into the container for
                      offline installs (default: pip install -e from git).

Provider keys are forwarded verbatim (OPENAI_API_KEY / GEMINI_API_KEY /
OLLAMA_HOST) so mucli picks whichever provider the benchmark run targets.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

# Where mucli's own repo lives on the host (bench/ sits inside it).
_MUCLI_REPO_HOST = Path(__file__).resolve().parent.parent

_FORWARD_KEYS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OLLAMA_HOST",
    "OLLAMA_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "MUCLI_BENCH_PROVIDER",
    "MUCLI_BENCH_MODEL",
)


class MucliAgent(AbstractInstalledAgent):
    """Run mucli headlessly inside the TB task container."""

    @staticmethod
    def name() -> str:
        return "mucli"

    def __init__(self, model_name: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name or os.environ.get("MUCLI_BENCH_MODEL", "")
        self._version = kwargs.get("version", "bench-1")

    # -- repo delivery ----------------------------------------------------

    def _build_repo_tarball(self) -> Path:
        """Create a tarball of mucli's tracked files (git archive, ~2 MB)."""
        import subprocess
        import tempfile

        out = Path(tempfile.mkdtemp(prefix="mucli-bench-")) / "mucli-src.tar.gz"
        subprocess.run(
            ["git", "archive", "--format=tar.gz", "-o", str(out), "HEAD"],
            cwd=str(_MUCLI_REPO_HOST),
            check=True,
        )
        return out

    def perform_task(self, instruction, session, logging_dir=None):
        """Copy the mucli repo tarball into the container before install.

        The base implementation only copies the setup script; mucli's source
        must also land at /mucli-src (extracted) so the offline install works
        without a registry round-trip.
        """
        tarball = self._build_repo_tarball()
        session.copy_to_container(
            paths=tarball,
            container_dir="/tmp",
            container_filename="mucli-src.tar.gz",
        )
        session.container.exec_run(
            ["sh", "-c", "mkdir -p /mucli-src && tar -xzf /tmp/mucli-src.tar.gz -C /mucli-src"]
        )
        return super().perform_task(instruction, session, logging_dir)

    # -- AbstractInstalledAgent contract ---------------------------------

    @property
    def _env(self) -> dict[str, str]:
        env = {key: os.environ[key] for key in _FORWARD_KEYS if os.environ.get(key)}
        if self._model_name:
            env.setdefault("MUCLI_BENCH_MODEL", self._model_name)
        return env

    @property
    def _install_agent_script_path(self) -> os.PathLike:
        # Template must live next to this file for _get_templated_script_path.
        expected = Path(__file__).parent / "mucli-setup.sh.j2"
        if not expected.exists():
            nested = Path(__file__).parent / "agent_templates" / "mucli-setup.sh.j2"
            if nested.exists():
                expected.write_text(nested.read_text())
        return self._get_templated_script_path("mucli-setup.sh.j2")

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped = shlex.quote(instruction)
        model_part = (
            f" --provider-model {shlex.quote(self._model_name)}"
            if self._model_name
            else ""
        )
        return [
            TerminalCommand(
                command=(
                    "mucli bench-run "
                    f"--instruction {escaped}{model_part} "
                    "--working-dir /app"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]


# The setup script template is rendered from the agent directory.
MUCLI_SETUP_TEMPLATE = """#!/bin/bash
set -euo pipefail

# mucli TB agent install: minimal python deps + mucli source into /opt/mucli.
mkdir -p /opt/mucli
apt-get update -qq && apt-get install -y -qq git python3-pip >/dev/null 2>&1 || true

python3 -m pip install --quiet --no-input \
    "openai>=1.59" "fastapi" "uvicorn" "pydantic" "rich" "prompt_toolkit" \
    "httpx" "python-dotenv" "psutil" || true

# Copy the host-mounted mucli checkout if present (git archive tarballs have
# no .git dir — check for mucli.py instead), else clone from origin.
if [ -f /mucli-src/mucli.py ]; then
    cp -r /mucli-src/. /opt/mucli/
else
    echo "MUCLI_SOURCE_MISSING: mount the mucli repo as /mucli-src" >&2
    exit 1
fi

cd /opt/mucli
python3 -m pip install --quiet -e . --no-deps || true
cat > /usr/local/bin/mucli <<'EOF'
#!/bin/sh
exec python3 /opt/mucli/mucli.py "$@"
EOF
chmod +x /usr/local/bin/mucli

cat > /usr/local/bin/mucli-bench-wrapper <<'EOF'
#!/bin/bash
# Non-interactive one-shot run: send the instruction, wait for completion.
cd "$MUCLI_WORKING_DIR"
mucli --session "tb-$$" --headless "$MUCLI_TASK_INSTRUCTION"
EOF
chmod +x /usr/local/bin/mucli-bench-wrapper
echo "mucli install OK"
"""


_TEMPLATE_HERE = Path(__file__).parent / "mucli-setup.sh.j2"
if not _TEMPLATE_HERE.exists():
    _TEMPLATE_HERE.write_text(MUCLI_SETUP_TEMPLATE)



def write_setup_template(target_dir: Path):
    """Materialize the jinja template where AbstractInstalledAgent expects it."""
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "mucli-setup.sh.j2"
    path.write_text(MUCLI_SETUP_TEMPLATE)
    return path