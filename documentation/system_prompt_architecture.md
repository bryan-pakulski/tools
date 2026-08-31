# System Prompt Architecture

This documents every component that contributes to the model's system
prompt in mucli, the file-based externalization layer that lets users
override the base and per-mode prompts from disk, and the refinement
pass applied to the base + default prompts.

Companion code: `mu/prompts/` (loader), `mu/commands/prompts.py`
(`/prompts`), `mu/commands/_prompt_flags.py` (CLI flags),
`mu/gui/routers/system_prompts.py` (GUI editor API),
`mu/prompts/templates/` (refined templates). Tests: `tests/test_prompts.py`.

---

## 1. Catalog of prompt components

Every string that lands in the system prompt, with its location, char
count, and responsibility. Char counts are the source literal lengths
(hardcoded fallbacks); file/runtime overrides may differ.

| Component | Location | Chars | Responsibility |
|-----------|----------|------:|----------------|
| `--system` CLI default | `mucli.py:1216` | ~603 | Non-agentic base instruction (the `session.system_instruction` prefix). Used as-is when agentic mode is OFF; prepended to the agentic base when ON. |
| `AGENTIC_SYSTEM_BASE` | `utils/config.py:367` | 11 432 | Agentic base: caveman grammar, full TOOL SURFACE taxonomy (incl. self-management tools), sub-agent guidance, 13 GENERAL RULES, and a SELF-MANAGEMENT block (rule 7) instructing the agent to own its context — prune the todo ledger, promote durable/drop ephemeral, supersede-don't-sibling, reconcile on staleness signals from `context_status` (stale_memory_count/stale_todos/memory_pressure_pct), watch fill, self-trigger `checkpoint_progress`, recognize its own stall, write its own handoffs. Appended as `workspace_context` whenever agentic mode is ON. |
| `AGENTIC_MODES["default"]` | `utils/config.py:437` | 2 412 | Collation-aware default coding workflow (clarify → recall → semantic retrieval → plan → parallel collect → act → verify → save → summarize). |
| `AGENTIC_MODES["debug"]` | `utils/config.py:459` | 3 150 | Debugging workflow with scratchpad-tagging protocol (hypothesis/suspect/repro/bisect), recall-first, repro→locate→hypothesize→bisect→fix→verify→persist. |
| `AGENTIC_MODES["feature"]` | `utils/config.py:488` | 3 021 | Feature Task Engine workflow: plan→per-task loop→review, anchored to the feature-task engine tools, mandatory memory+scratchpad, blockers. |
| `AGENTIC_MODES["research"]` | `utils/config.py:523` | 3 180 | Research workflow: recall, parallel multi-source search, semantic retrieval for codebase, delegation, citation requirements + source credibility table, anti-detection. |
| `AGENTIC_MODES["loop"]` | `utils/config.py:569` | 2 637 | Long-horizon autonomous loop: goal lock, user-visible backlog, per-increment re-orient→gather→act→verify→reflect, memory discipline, timeline updates. |
| `AGENTIC_MODES["security"]` | `utils/config.py:611` | 4 356 | Security audit engine: anti-hallucination contract (PoC + patch both verified), discovery→per-finding proof-and-patch→final report, refutation of failed hypotheses. |
| `AGENTIC_MODES["teacher"]` | `utils/config.py:657` | 14 483 | One-on-one tutor: personalization contract, chat-based teaching (watcher records), curriculum + per-lesson loop, dual-presentation artifacts, spaced review. |
| `AGENT_MODE_METADATA` | `utils/config.py:814` | — | Real-mode registry (display_name, description, documentation path, and optional specialist `tool_phases`). Mode tool phases are added to the effective provider schema automatically when lazy tool exposure is enabled. The remaining metadata drives `/mode`, the GUI modes list, and the splash banner. The read-only view panels (history, memory, systemPrompts) are NOT here — they live in `GUI_VIEW_PANELS` and are surfaced through the GUI Tools menu, not as settable agent modes. |
| `GUI_VIEW_PANELS` | `utils/config.py` | — | GUI-only view-panel registry (history, memory, systemPrompts). Read-only; never agent modes — `POST /api/modes/{name}` rejects them and they never appear in `/mode`, the splash banner, or `--mode-prompt`. Surfaced as the `views` array in `GET /api/modes`. |
| `NUDGE_EMPTY_RESPONSE` | `utils/config.py:788` | 161 | Injected as a user message when the model returns no text after tool calls (`loop_body.py:845`). |
| Feature-mode dynamic block | `loop_body.py:475-489` | ~1 100 | Appended to `base_system_prompt` when `active_mode == "feature"`: FEATURE MODE SYSTEM PROMPT instructions (staged engine, exit criteria, one task at a time). |
| Loop-mode dynamic block | `loop_body.py:490-503` | ~700 | Appended when `active_mode == "loop"`: LOOP MODE SYSTEM PROMPT + the locked `loop_goal`. |
| Teacher learner-profile block | `loop_body.py:504-507` | variable | Appended when `active_mode == "teacher"` via `_render_learner_profile_block(session)` — the auto-injected LEARNER PROFILE. |
| Resumption block | `loop_body.py:531` | variable | Appended for resumed sessions with in-flight teacher/feature state. |
| L0–L5 hierarchical context | `mu/session/context.py:94` | variable | `inject_hierarchical_context` assembles L0 time prelude, L1B skills, L2 conversation summary, L3 active goal, L4/L4B retrieval, L5 history. |
| L3 memory snapshot | `loop_body.py:655-663` | variable | Persisted working-memory snapshot + eviction notices (per turn). |
| L3 scratchpad snapshot | `loop_body.py:678-690` | variable | Turn scratchpad snapshot + eviction notices (per turn). |

### Assembly order (one turn)

`loop_body.py` builds the prompt in this order:

1. `base_system_prompt = session.system_instruction` (the `--system` value).
2. Append feature/loop/teacher dynamic block (`loop_body.py:474-507`).
3. Append `workspace_context` =
   `{agentic_system_base}\n\n### CURRENT STRATEGY MODE: {MODE}\n{mode_instruction}`
   plus an `ACTIVE TOOL REGISTRIES` receipt describing the effective provider
   schema (`loop_body.py:457`) — only when agentic.
4. Append resumption block (`loop_body.py:531`).
5. `inject_hierarchical_context` → L0–L5 (`loop_body.py:532`).
6. Append L3 memory + scratchpad snapshots + eviction notices
   (`loop_body.py:643-690`).
7. `retry.py` reuses the assembled prompt across retries (no re-injection).

`agentic_system_base` and `mode_instruction` are resolved with the
priority ladder in §2.

### Per-iteration rebuild (not frozen at turn start)

Steps 5–6 are re-run **every iteration within a turn**, so L2 (conversation
summary) and L3 (active goal / memory / scratchpad) reflect mid-turn updates
(auto-compaction rewriting the summary, tools mutating `feature_state` /
the scratchpad) instead of being frozen at their turn-start value — the
long-horizon amnesia bug. To keep this cheap:

- L1B (skills) is built **once per turn**
  (`session._turn_skills_block`,
  `loop_body.py:541-544`) and passed to `inject_hierarchical_context` as
  `cached_skills=`, so each iteration's rebuild skips
  the skills-tree walk.
- L2 and L3 are always reassembled from in-memory state.

A periodic **L2 progress checkpoint** (`HistoryMixin.force_progress_checkpoint`,
gated by `progress_checkpoint_every`) folds recent history into the structured
summary without advancing the compaction anchor, so long turns that never
hit the compaction budget still get a fresh Progress / Open-items picture.

---

## 2. File-based externalization

### Resolution priority (highest first)

1. **Runtime `/set` override** — session variables
   `agentic_system_base_override` and `agentic_mode_prompt_<mode>`.
   Set via `/set`, `--mode-prompt`, or the GUI. Highest priority; wins
   over files and hardcoded.
2. **File override** — `$MUCLI_HOME/prompts/{base,<mode>}.md`
   (`MUCLI_HOME` defaults to `~/.mucli/`, `utils/config.py:13`).
3. **Hardcoded fallback** — `AGENTIC_SYSTEM_BASE` / `AGENTIC_MODES` in
   `utils/config.py`.

Layers 2–3 live in `mu.prompts`; layer 1 is applied at the call site
(`mu/agent/loop_body.py:438-454` and `utils/runtime_metrics.py:222-242`)
via `session.variables.get(key) or PromptLibrary.get_*()`, so a non-empty
runtime override always wins and an empty-string override falls through
to the file.

### File format

```markdown
---
name: default
version: 2
description: Collation-aware default coding workflow.
---
<prompt body — exactly what the model sees>
```

Frontmatter is optional but recommended. Parsed with the same YAML
already required by `mu.skills`, with a manual fallback when PyYAML is
absent. `version` is an integer surfaced by `/prompts` and the GUI for
per-session version tracking.

### Loader (`mu/prompts/__init__.py`)

- `get_base() -> str` / `get_mode(mode) -> str` — resolved text
  (file > hardcoded). Used by the loop.
- `get_resolved(name) -> ResolvedPrompt` — full resolution with
  `source` (`file`|`hardcoded`), `path`, `version`, `chars`.
- `resolved_snapshot(session=None) -> dict` — per-prompt summary for
  `/prompts` and the GUI; layers in the runtime override when a session
  is supplied.
- `reload()` — drop the mtime-keyed cache so edits apply next turn.
- `init_templates(names, force=False)` — write template files to
  `$MUCLI_HOME/prompts/`. Bundled refined templates for `base`/`default`;
  other modes seeded verbatim from the hardcoded fallback so
  `/prompts init <mode>` externalizes the current prompt for editing.
- `write_override(name, text, version=None)` — persist a file (GUI PUT).
- `read_override_raw(name)` — raw file content (frontmatter + body).
- `validate(name, text) -> list[str]` — critical-anchor drift detection
  (see §4).

### CLI integration (`mucli.py` + `mu/commands/_prompt_flags.py`)

- `--system-file <path>` — load the non-agentic base instruction from a
  file (overrides `--system`). `-` reads stdin.
- `--mode-prompt NAME=PATH` — install a runtime override for `base` or a
  mode (repeatable). `NAME` ∈ {base, default, debug, feature, research,
  loop, security, teacher}. Sets the same session variables as
  `/set`, so it sits at priority 1. (history/memory/systemPrompts are GUI
  view panels, not prompt-overridable modes.)

### Slash command (`/prompts`, `mu/commands/prompts.py`)

```
/prompts                 — list every prompt + current source/version/chars
/prompts reload          — clear the cache (pick up file edits)
/prompts init [name|all] — write template files to $MUCLI_HOME/prompts/
/prompts show <name>     — print the currently-effective prompt text
/prompts validate [name] — check critical tool-surface anchors are present
/prompts edit <name>     — open the override file in $EDITOR (seeds it first)
```

### GUI editor API (`mu/gui/routers/system_prompts.py`, prefix `/api/system-prompts`)

Distinct from `/api/prompts` (the prompt-response store that unblocks the
agent thread for `ask_user_choice`/approval). Endpoints:

- `GET  /api/system-prompts` — list resolution snapshot.
- `GET  /api/system-prompts/{name}` — current text, source, version,
  chars, validation warnings, raw file content.
- `PUT  /api/system-prompts/{name}` — write a file override (body in
  payload); returns validation warnings.
- `POST /api/system-prompts/reload` — clear cache.
- `POST /api/system-prompts/init` — seed template files.
- `POST /api/system-prompts/{name}/reset` — delete the file override so
  the hardcoded fallback takes over.

The GUI editor panel is built on this API (`mu/gui/templates/fragments/
system_prompts_panel.html`): list → select → edit textarea → PUT →
reload. It is surfaced as a view-only panel in the GUI Tools menu (not an
agent mode).

### Hot-reload

Editing a file on disk does not affect the current turn (the loop reads
the prompt once per turn at `loop_body.py:438`). Run `/prompts reload`
(or call `POST /api/system-prompts/reload`) to drop the cache; the next
turn picks up the edit. The cache is mtime-keyed, so an external edit +
reload is sufficient — no restart.

### Backward compatibility

- No file present → behavior is byte-identical to before: the hardcoded
  constants are the fallback, and the existing
  `agentic_system_base_override` / `agentic_mode_prompt_<mode>` runtime
  overrides keep working unchanged.
- `tests/test_mode_sota_patterns.py` asserts substrings against the
  Python constants `AGENTIC_SYSTEM_BASE` / `AGENTIC_MODES[...]`. Those
  constants are untouched (still the fallback), so every existing test
  passes without modification — verified (145 prompt/mode/security/gui
  tests + 182 session/context/loop/provider tests).
- New code paths import `mu.prompts` lazily inside the assembly block, so
  a failure to read the prompts dir degrades gracefully to the hardcoded
  fallback (the `or` in `loop_body.py:449-454`).

### Edge cases

- Unknown mode name in `--mode-prompt` → `SystemExit` with the valid set
  (`_prompt_flags.apply_prompt_flags`).
- Empty-string runtime override → falls through to file/hardcoded (the
  `or`).
- Corrupt frontmatter → parsed as empty meta, body still loaded
  (`_split_frontmatter` swallows YAML errors).
- Unreadable file → logged, falls back to hardcoded (`_resolve_file`).
- `MUCLI_HOME` override → respected (same env var already used for
  `~/.mucli/sessions/`).

---

## 3. Refinement pass (base + default)

The bundled templates `mu/prompts/templates/base.md` and
`default.md` are refined versions: shorter than the originals, with
redundancy removed and wording made deterministic, while preserving every
critical tool-surface / workflow anchor the harness depends on.

| Prompt | Original (chars) | Refined (chars) | Ratio | Anchors preserved |
|--------|-----------------:|----------------:|------:|-------------------|
| base   | 11 432 | 6 032 | 0.53 | bash, read_file, apply_diff, search_and_replace_file, search_for_string, retrieve_relevant_context, spawn_agent, todo_write, save_memory, save_scratchpad, flush, plan mode, parallel/concurrent |
| default | 2 412 | 2 126 | 0.88 | search_memory, retrieve_relevant_context, bash, verif, todo_write, parallel/concurrent, spawn_agent, save_memory |

Refinement approach:
- **Caveman grammar section (base):** collapsed seven example bullets into
  five without dropping a rule; kept the "thinking tokens: caveman too"
  instruction (the highest-value budget saver).
- **Tool surface (base):** kept verbatim — it is the critical taxonomy
  `test_mode_sota_patterns.py` pins, and trimming it would lose
  capability visibility.
- **General rules (base):** tightened prose per rule (e.g. rule 3's
  diff-format spec compressed from 5 lines to one flowing sentence)
  without dropping any of the 13 rules or their conditions.
- **Default workflow:** condensed each of the 8 steps' prose; kept every
  tool name and the parallel/verify/memory anchors.

Both refined templates pass `validate()` clean (zero missing anchors) —
verified in `tests/test_prompts.py::test_validate_passes_for_refined_templates`
and `test_refined_{base,default}_no_longer_than_original`.

The other five modes are externalizable (the loader supports them; `/prompts init <mode>`
seeds them verbatim from the hardcoded fallback) but their refinement is
intentionally deferred — each is large (security 4 356, teacher 14 483)
and pinned by `test_mode_sota_patterns.py` substrings, so refining them
is a follow-up with its own validation pass rather than a rushed edit.

---

## 4. Drift detection (`validate`)

`mu.prompts.validate(name, text)` checks the resolved prompt still names
the critical anchors for its kind. The required substrings mirror
`tests/test_mode_sota_patterns.py` so a hand-edited file that drops a
frontier feature (e.g. `spawn_agent`, `plan mode`, `retrieve_relevant_context`)
is surfaced as a warning rather than a silent regression. Tuples mean
any-of (e.g. `("parallel", "concurrent")`).

Surfaces:
- `/prompts validate [name]` — CLI, prints per-prompt OK / missing anchors.
- `GET /api/system-prompts/{name}` and `PUT ...` — returns `validation`
  warnings so the GUI editor can show them inline.
- `tests/test_prompts.py::test_validate_passes_for_hardcoded_constants` —
  guards the shipped hardcoded prompts against drift.

`teacher` has no pinned anchors (free-form content) and validates clean by
design. (`history` was removed from `AGENTIC_MODES` — it is now a GUI view
panel, not a prompt-overridable mode.)

---

## 5. What is implemented vs. documented

**Implemented and tested:**
- `mu.prompts` loader with the full priority ladder, mtime cache, reload,
  init, write, read, validate, resolved snapshot.
- Loop + runtime-metrics wiring (`loop_body.py`, `runtime_metrics.py`).
- CLI flags `--system-file`, `--mode-prompt`.
- `/prompts` slash command (list / reload / init / show / validate / edit).
- GUI API `/api/system-prompts` (list / get / put / reload / init / reset).
- Refined `base.md` + `default.md` templates (≤ original, anchors
  preserved, validate clean).
- 22 unit tests + E2E assembly verification + 327 regression tests green.

**Documented / follow-up:**
- GUI editor panel — implemented (`mu/gui/templates/fragments/
  system_prompts_panel.html` + `Alpine.store("systemPrompts")`), surfaced
  as a view-only panel in the GUI Tools menu (not an agent mode).
- Refinement of the remaining 5 modes (debug/feature/research/loop/
  security/teacher) — mechanism supports it; refinement deferred
  to a dedicated pass with per-mode validation.
- Per-session prompt-version persistence in the saved session file
  (currently version is surfaced live via `/prompts` and the API; writing
  it into `session.json` for across-restart auditing is a small follow-up
  in `mu/session/manager.py`).
