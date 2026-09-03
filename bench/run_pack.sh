#!/usr/bin/env bash
# mucli benchmark pack runner — Terminal-Bench core tasks via the mucli agent.
# Usage:
#   bash bench/run_pack.sh [--smoke]      # smoke = hello-world only
#   MODEL=openai/gpt-4.1 bash bench/run_pack.sh
#
# Prereqs (one-time): uv venv with TB >=3.13 + docker running:
#   uv venv --python 3.13 ~/.venvs/tb && uv pip install --python ~/.venvs/tb/bin/python terminal-bench
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

TB_PY="${TB_PY:-$HOME/.venvs/tb/bin/tb}"
[ -x "$TB_PY" ] || { echo "tb not found at $TB_PY (set TB_PY=...)"; exit 1; }

DATASET="terminal-bench-core==0.1.1"
AGENT_PATH="bench.tb_mucli_agent:MucliAgent"
MODEL="${MODEL:-}"
EXTRA=()
if [ -n "$MODEL" ]; then EXTRA+=(--model "$MODEL"); fi

if [ "${1:-}" = "--smoke" ]; then
  TASKS=(hello-world)
else
  TASKS=(hello-world processing-pipeline fix-git git-multibranch
         nginx-request-logging cron-broken-network sqlite-db-truncate
         train-fasttext fix-permissions build-tcc-qemu)
fi

mkdir -p bench/results
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="bench/results/run-$STAMP"
mkdir -p "$OUT"

echo "== mucli bench pack → $OUT =="
for task in "${TASKS[@]}"; do
  echo "-- $task"
  "$TB_PY" run \
    --dataset "$DATASET" \
    --agent-import-path "$AGENT_PATH" \
    --task-id "$task" \
    --output-path "$OUT/$task" \
    "${EXTRA[@]}" || echo "   (task $task failed — see $OUT/$task)"
done

python3 - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows = []
for d in sorted(out.iterdir()):
    for res in d.glob("**/results.json"):
        data = json.loads(res.read_text())
        rows.append((d.name, data.get("is_resolved"), data.get("duration_sec")))
    if not list(d.glob("**/results.json")):
        rows.append((d.name, None, None))
print("\n== PACK RESULTS ==")
for name, ok, dur in rows:
    print(f"  {name:<28} {'PASS' if ok else 'FAIL' if ok is False else 'NO-RESULT'} {dur if dur else ''}")
scored = [1.0 if ok else 0.0 for _, ok, _ in rows if ok is not None]
if rows:
    print(f"pack score: {sum(1 for _,ok,_ in rows if ok)}/{len(rows)}")
PY