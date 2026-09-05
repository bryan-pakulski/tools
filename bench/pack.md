# MuCLI Harness Benchmark Pack

Small, time-gated task list to measure **harness + model effectiveness**.
The runnable pack currently contains ten pinned Terminal-Bench core tasks.
SWE-bench and MuCLI-native tasks below remain design candidates rather than
implemented runners. Terminal-Bench supplies verifier outcomes and time gates;
the MuCLI adapter adds exact source fingerprints and token-bearing JSONL traces.

---

## A. Why these suites

| Suite | What it measures | Infra | Time-gating | Verdict |
|---|---|---|---|---|
| **Terminal-Bench** (laude-institute/terminal-bench) | Full terminal-agent loop w/ verifier scripts | Docker, free, MIT | ✅ per-task `timeout_s` | **Core** — same shape as mucli |
| **SWE-bench Lite/Verified** | Real-issue repo patches graded by FAIL_TO_PASS tests | Docker + HF dataset, free | Harness-level | **Core** (4 instances) |
| Aider polyglot | Single-file edit format, 225 Exercism | No docker, free | Whole-run only | Cross-check only — not agent-shaped |
| LiveCodeBench | Competitive programming | No docker, free | Judge limits only | Measures model, not harness — skip |
| SWE-Lancer (OpenAI) | Real Upwork tasks, $-graded | ~23 GB docker | Runner-set | Too heavy for routine runs |
| InterCode | Old terminal bash tasks | Docker | No | Dormant since 2023 — skip |

## B. Current Terminal-Bench pack (10 tasks, all time-gated)

The task IDs and native timeout gates are pinned in `bench/tb_suite.yaml` and
`bench/run_pack.sh`. The selection covers file operations, shell repair, Git,
service configuration, data recovery, model training, and a QEMU build task.

`hello-world`, `processing-pipeline`, `fix-git`, `git-multibranch`,
`nginx-request-logging`, `cron-broken-network`, `sqlite-db-truncate`,
`train-fasttext`, `fix-permissions`, and `build-tcc-qemu`.

The score printed by the runner is resolved trials / total trials. Harness
failures and missing result files make the runner exit non-zero; ordinary task
failures are valid benchmark outcomes and remain in the score.

## C. Running Terminal-Bench

One-time setup from the repository root:

```bash
uv venv --python 3.13 ~/.venvs/tb
uv pip install --python ~/.venvs/tb/bin/python terminal-bench
bash bench/build_wheelhouse.sh
```

The wheelhouse contains compatible manylinux wheels for Python 3.10–3.14, so
dependency installation is offline and reproducible. The runner rejects stale
wheelhouses and a task container whose Python minor is absent from the payload.
Rebuild it whenever `requirements.txt` changes.

For stable comparison runs, prebuild the selected task images once. The
prepared dataset rewrites only the Compose image reference to a
content-addressed local tag and records the task-input hash plus Docker image ID
in `prepare-manifest.json`. Later runs validate both before using
`--no-rebuild --no-cleanup`:

```bash
MODEL=ollama/glm-5.3-flash bash bench/run_pack.sh --smoke --prepare
MODEL=ollama/glm-5.3-flash bash bench/run_pack.sh --prepare
```

Image construction happens before the benchmark output directory and timing
window are created. The second command can be expensive on its first run,
especially for `train-fasttext` and `build-tcc-qemu`; subsequent `--prepare`
runs reuse matching image IDs.

Run a cheap end-to-end check, then the full pack:

```bash
MODEL=ollama/glm-5.3-flash bash bench/run_pack.sh --smoke
MODEL=ollama/glm-5.3-flash bash bench/run_pack.sh --task fix-git
MODEL=openai/gpt-5 bash bench/run_pack.sh --attempts 3 --run-label mucli-gpt5
```

`--attempts N` asks Terminal-Bench for repeated trials of every selected task.
Use the same prepared task-image manifest, task list, model settings, prompt,
and attempt count for each harness being compared. A single attempt is useful
for smoke testing but is not a reliable harness comparison.

The runner uses the cached pinned dataset at
`~/.cache/terminal-bench/terminal-bench-core/0.1.1`. Override it with
`TB_DATASET_PATH=/path/to/tasks`. If it is missing, download the pinned dataset
with `~/.venvs/tb/bin/tb datasets download --dataset terminal-bench-core==0.1.1`.
Results land under `bench/results/`.

For direct TB use, pass both timing controls explicitly (360 is `hello-world`'s
native budget, while 570 adds the 180-second setup cap and a 30-second outer
cleanup margin):

```bash
~/.venvs/tb/bin/tb run \
  --dataset-path ~/.cache/terminal-bench/terminal-bench-core/0.1.1 \
  --agent-import-path bench.tb_mucli_agent:MucliAgent \
  --task-id hello-world \
  --global-agent-timeout-sec 570 \
  --agent-kwarg execution_timeout_sec=360 \
  --agent-kwarg setup_timeout_sec=180 \
  --agent-kwarg benchmark_prompt=verify-v1 \
  --model openai/gpt-5
```

The adapter benchmarks modified tracked files as well as committed files.
Untracked and ignored files are excluded so local credentials cannot be copied
into task containers. Each trial records `source.json`, MuCLI traces, token
totals, and `agent-logs/mucli-execution.json`. The pack summary reports the
command-only execution time and labels setup as excluded. Source transfer,
Python dependency installation, import preflight, and task-image building are
therefore never reported as MuCLI execution time. TB's raw
`agent_started_at`/`agent_ended_at` timestamps remain setup-inclusive because
TB 0.2.18 starts its timer before invoking the installed-agent adapter.

The adapter runs MuCLI with the explicit `terminal-bench` tool profile, which
keeps direct workspace, shell, task, result, and context tools while omitting
unrelated schemas. It appends the versioned `verify-v1` instruction: verify all
requirements directly, exercise service changes end to end, preserve exact
bytes when restoring version-control content, and stop once acceptance checks
pass. Both choices are recorded in each run's `provenance.json` alongside the
tracked-worktree fingerprint, model, Terminal-Bench version, wheelhouse hash,
attempt count, timing policy, and prepared Docker image IDs.

If the native execution limit expires, the adapter interrupts MuCLI and uses
narrowly scoped TERM/KILL fallbacks before the verifier starts. Completed
iteration tokens from the partial trace are retained in the result instead of
being reported as zero.

## D. Future suite candidates

### SWE-bench Lite (4)

| # | Instance | Gate | Oracle | Discriminates |
|---|----------|------|--------|---------------|
| 5 | `django__django-11039` | 15 min | FAIL_TO_PASS pass, PASS_TO_PASS intact | Precise edit, large repo |
| 6 | `sympy__sympy-13480` | 15 min | same | Math reasoning + TDD fix |
| 7 | `astropy__astropy-12907` | 15 min | same | Corner-case bug |
| 8 | `matplotlib__matplotlib-23964` | 20 min | same | Multi-file reasoning |

Partial: 1.0 full pass; 0.5 F2P pass with P2P regression. Verify ids exist in
`princeton-nlp/SWE-bench_Lite` before first run [unverified].

### mucli-native (3) — run in mucli container mode

| # | Task | Gate | Oracle | Discriminates |
|---|------|------|--------|---------------|
| 9 | Seeded failing-test fix | 10 min | `pytest` exit 0 in container | Test-driven repair loop |
| 10 | Multi-file rename refactor | 15 min | `pytest tests/` green AND `grep -rc old_symbol src/ | wc -l` = 0 | Cross-file edit discipline |
| 11 | Long-context fix (3 files, ~40k chars) | 15 min | pytest green | Context budget management — mucli's differentiator |

Partial: 1.0 all green; 0.5 ≥ half the seeded tests pass (10) / green with leftovers (10–11).

### Bash/sysadmin (2)

| # | Task | Gate | Oracle | Discriminates |
|---|------|------|--------|---------------|
| 12 | Log-triage across rotated logs | 8 min | summary file matches expected greps/jq | Pipeline fluency |
| 13 | Service setup + verifier | 12 min | verifier: process running, config valid | Env config, docs reading |

---

### SWE-bench
```bash
pip install swe-bench
python -m swebench.harness.run_evaluation \
  --predictions_path bench/predictions.jsonl \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --run_id mucli-pack --max_workers 4
```
mucli emits `{"instance_id", "model_patch"}` per instance (patch = `git diff`
inside container). Grading is docker-only, $0 API on re-runs.

## E. What this discriminates

- **Harness quality** (context management, tool reliability): tasks 9–11 + trace
  metrics (compactions, watchdog nudges, drift) vs outcome
- **Model capability**: 5–8 + TB 2–3
- **Loop discipline under time pressure**: every task's gate + partial credit
- **Cost-effectiveness**: score ÷ tokens
