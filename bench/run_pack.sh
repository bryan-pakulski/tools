#!/usr/bin/env bash
# MuCLI benchmark pack runner — Terminal-Bench core tasks via the MuCLI agent.
#
# Examples:
#   bash bench/run_pack.sh --smoke
#   bash bench/run_pack.sh --task fix-git --attempts 3
#   MODEL=ollama/glm-5.3-flash bash bench/run_pack.sh --prepare
#   MODEL=openai/gpt-5 bash bench/run_pack.sh --attempts 3 --run-label gpt5
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root

TB_PY="${TB_PY:-$HOME/.venvs/tb/bin/tb}"
[ -x "$TB_PY" ] || { echo "tb not found at $TB_PY (set TB_PY=...)" >&2; exit 1; }

DATASET="terminal-bench-core==0.1.1"
SOURCE_DATASET_PATH="${TB_DATASET_PATH:-$HOME/.cache/terminal-bench/terminal-bench-core/0.1.1}"
PREPARED_DATASET_PATH="${TB_PREPARED_DATASET_PATH:-bench/artifacts/tb-prepared/terminal-bench-core/0.1.1}"
AGENT_PATH="bench.tb_mucli_agent:MucliAgent"
MODEL="${MODEL:-}"
SETUP_TIMEOUT_SEC="${TB_SETUP_TIMEOUT_SEC:-180}"
OUTER_CLEANUP_MARGIN_SEC="${TB_OUTER_CLEANUP_MARGIN_SEC:-30}"
ATTEMPTS="${TB_ATTEMPTS:-1}"
RUN_LABEL="${TB_RUN_LABEL:-}"
SELECTION="full"
SINGLE_TASK=""
PREPARE=0

usage() {
  cat >&2 <<EOF
usage: $0 [--smoke | --task TASK] [--attempts N] [--prepare] [--run-label LABEL]

  --smoke          run hello-world only
  --task TASK      run one task from the pinned dataset
  --attempts N     repeated trials per task (default: ${TB_ATTEMPTS:-1})
  --prepare        build/reuse immutable task images before starting the run
  --run-label TEXT record a comparison label in provenance.json
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke)
      [ "$SELECTION" = "full" ] || { usage; exit 2; }
      SELECTION="smoke"
      shift
      ;;
    --task)
      [ "$SELECTION" = "full" ] && [ "$#" -ge 2 ] || { usage; exit 2; }
      SELECTION="task"
      SINGLE_TASK="$2"
      shift 2
      ;;
    --attempts)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      ATTEMPTS="$2"
      shift 2
      ;;
    --prepare)
      PREPARE=1
      shift
      ;;
    --run-label)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      RUN_LABEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

case "$ATTEMPTS" in
  ''|*[!0-9]*) echo "--attempts must be a positive integer" >&2; exit 2 ;;
esac
[ "$ATTEMPTS" -gt 0 ] || { echo "--attempts must be positive" >&2; exit 2; }

case "$SELECTION" in
  full)
    TASKS=(hello-world processing-pipeline fix-git git-multibranch
           nginx-request-logging cron-broken-network sqlite-db-truncate
           train-fasttext fix-permissions build-tcc-qemu)
    ;;
  smoke)
    TASKS=(hello-world)
    ;;
  task)
    TASKS=("$SINGLE_TASK")
    ;;
esac

if [ ! -d "$SOURCE_DATASET_PATH" ]; then
  echo "cached dataset not found at $SOURCE_DATASET_PATH" >&2
  echo "download it first: $TB_PY datasets download --dataset $DATASET" >&2
  exit 2
fi

# Network dependency installation must not consume native task time. Validate
# the complete offline payload before any task container is launched.
WHEELHOUSE="bench/artifacts/mucli-wheelhouse.tar.gz"
if ! python3 - "$WHEELHOUSE" <<'PY'
import hashlib
import json
import sys
import tarfile

try:
    with tarfile.open(sys.argv[1], "r:gz") as archive:
        manifest = json.load(archive.extractfile("wheelhouse/manifest.json"))
except (OSError, KeyError, TypeError, json.JSONDecodeError, tarfile.TarError):
    raise SystemExit(1)

versions = str(manifest.get("python_versions", "")).split()
if manifest.get("format") != 2 or not {"3.12", "3.13", "3.14"}.issubset(versions):
    raise SystemExit(1)

excluded = ("kokoro", "faster-whisper", "soundfile", "playwright", "pytest", "black")
lines = []
for line in open("requirements.txt", encoding="utf-8"):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    if line.lower().startswith(excluded):
        continue
    lines.append(line if line.endswith("\n") else line + "\n")
expected = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
if manifest.get("requirements_sha256") != expected:
    raise SystemExit(1)
PY
then
  echo "wheelhouse missing or incompatible: run 'bash bench/build_wheelhouse.sh'" >&2
  exit 2
fi

if [ "$PREPARE" -eq 1 ]; then
  python3 bench/prepare_tb.py \
    --source "$SOURCE_DATASET_PATH" \
    --output "$PREPARED_DATASET_PATH" \
    "${TASKS[@]}"
fi

PREPARED_MANIFEST=""
if python3 bench/prepare_tb.py \
    --check-only \
    --source "$SOURCE_DATASET_PATH" \
    --output "$PREPARED_DATASET_PATH" \
    "${TASKS[@]}" >/dev/null 2>&1; then
  DATASET_PATH="$PREPARED_DATASET_PATH"
  BUILD_ARGS=(--no-rebuild --no-cleanup)
  PREPARED_MANIFEST="$PREPARED_DATASET_PATH/prepare-manifest.json"
  echo "using immutable prebuilt task images: $DATASET_PATH"
else
  DATASET_PATH="$SOURCE_DATASET_PATH"
  BUILD_ARGS=(--rebuild --no-cleanup)
  echo "using cached source dataset: $DATASET_PATH"
  echo "task images are not fully prepared; add --prepare for reproducible fast startup" >&2
fi
DATASET_ARGS=(--dataset-path "$DATASET_PATH")

EXTRA=()
if [ -n "$MODEL" ]; then
  EXTRA+=(--model "$MODEL")
fi

mkdir -p bench/results
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="bench/results/run-$STAMP"
mkdir -p "$OUT"

PROVENANCE_ARGS=()
if [ -n "$PREPARED_MANIFEST" ]; then
  PROVENANCE_ARGS+=(--prepared-manifest "$PREPARED_MANIFEST")
fi
python3 bench/write_tb_provenance.py \
  --output "$OUT/provenance.json" \
  --repo . \
  --dataset "$DATASET_PATH" \
  --model "$MODEL" \
  --attempts "$ATTEMPTS" \
  --tb "$TB_PY" \
  --setup-allowance-seconds "$SETUP_TIMEOUT_SEC" \
  --outer-cleanup-margin-seconds "$OUTER_CLEANUP_MARGIN_SEC" \
  --run-label "$RUN_LABEL" \
  "${PROVENANCE_ARGS[@]}" \
  "${TASKS[@]}"

echo "== MuCLI benchmark pack -> $OUT =="
echo "   attempts per task: $ATTEMPTS; execution timing excludes agent setup"
harness_failed=0
for task in "${TASKS[@]}"; do
  echo "-- $task"
  task_config="$DATASET_PATH/$task/task.yaml"
  if ! execution_timeout=$(python3 - "$task_config" <<'PY'
import sys
from pathlib import Path

from bench.tb_support import read_task_execution_timeout

try:
    print(read_task_execution_timeout(Path(sys.argv[1])))
except ValueError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(2)
PY
  ); then
    harness_failed=1
    continue
  fi
  outer_timeout=$(awk \
    -v execution="$execution_timeout" \
    -v setup="$SETUP_TIMEOUT_SEC" \
    -v cleanup="$OUTER_CLEANUP_MARGIN_SEC" \
    'BEGIN { print execution + setup + cleanup }')
  echo "   execution budget: ${execution_timeout}s; setup allowance: ${SETUP_TIMEOUT_SEC}s"
  if ! "$TB_PY" run \
    "${DATASET_ARGS[@]}" \
    --agent-import-path "$AGENT_PATH" \
    --task-id "$task" \
    --output-path "$OUT/$task" \
    --n-concurrent 1 \
    --n-attempts "$ATTEMPTS" \
    --global-agent-timeout-sec "$outer_timeout" \
    --agent-kwarg "execution_timeout_sec=$execution_timeout" \
    --agent-kwarg "setup_timeout_sec=$SETUP_TIMEOUT_SEC" \
    --agent-kwarg "benchmark_prompt=verify-v1" \
    "${BUILD_ARGS[@]}" \
    "${EXTRA[@]}"; then
    echo "   (harness failed for $task — see $OUT/$task)" >&2
    harness_failed=1
  fi
done

summary_failed=0
python3 bench/summarize_tb.py "$OUT" "${TASKS[@]}" || summary_failed=1
if [ "$harness_failed" -ne 0 ] || [ "$summary_failed" -ne 0 ]; then
  exit 1
fi
