# mucli Harness Benchmark Pack

Small, time-gated task list to measure **harness + model effectiveness**.
Runners: Terminal-Bench (`tb`) + SWE-bench grading + mucli-native container tasks.
Scoring: per-task 0 / 0.5 / 1 → pack score = mean × 13. Wall-clock + token cost
come free from mucli JSONL traces.

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

## B. The 13-task pack (all time-gated)

### Terminal-Bench (4)

| # | Task | Gate | Pass oracle | Discriminates | Partial |
|---|------|------|-------------|---------------|---------|
| 1 | TB `hello-world` | 5 min | TB verifier exit 0 | Baseline loop: read → act → exit | 0.5 if pass >2× gate |
| 2 | TB `train-fasttext` | 20 min | verifier: model file + accuracy ≥ bar | Long-horizon tool chaining | 0.5 model saved, below bar |
| 3 | TB `build-tcc` | 15 min | verifier exit 0 | Build/debug unfamiliar repo | 0.5 compiles, tests fail |
| 4 | TB sysadmin task (`fix-permissions`-class) | 10 min | verifier exit 0 | Shell fluency, no code edit | 0.5 most checks pass |

> Verify current task ids in the TB registry before pinning; ids above are the
> canonical set from the Terminal-Bench 1.x/2.0 public suite [unverified per-release].

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

## C. Running

### Terminal-Bench
```bash
pip install terminal-bench
tb run --task-id hello-world --agent-import-path bench.mucli_agent.MucliAgent
tb run --suite bench/tb_suite.yaml --agent-import-path bench.mucli_agent.MucliAgent
```
Custom agent: implement the TB agent interface (class receives container handle +
task prompt; returns when done) — one Python file wrapping `mucli.py` subprocess.
[unverified exact signature — check TB docs at install time]

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

### Cost per full pack run
13 tasks × 30–80k input tokens ≈ 0.5–1.5 M tokens ≈ **$5–20** on frontier
models [unverified]; mucli-native tasks cheapest. SWE-bench re-grades free.

### Scoring
`pack_score = Σ per-task credit / 13`; also report wall-clock per task and
token cost (mucli traces: `~/.mucli/trace/*.jsonl` already carry tokens + wall_ms
per iteration — free instrumentation).

---

## D. What this discriminates

- **Harness quality** (context management, tool reliability): tasks 9–11 + trace
  metrics (compactions, watchdog nudges, drift) vs outcome
- **Model capability**: 5–8 + TB 2–3
- **Loop discipline under time pressure**: every task's gate + partial credit
- **Cost-effectiveness**: score ÷ tokens