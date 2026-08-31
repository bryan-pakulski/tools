# Session class — per-turn state container + agent-loop entry.
import os
import json
import time
import glob
import re
import random
import shutil
import traceback
import hashlib
import threading
import uuid
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from typing import Optional

from mu.agent.approval import build_approval_prompt, collect_approval_plans, ApprovalPlan
from mu.agent.collation import CollationBuffer
from mu.feature.engine import refresh_and_persist_feature_plan, summarize_feature_plan
from mu.memory.stores import ScratchpadStore, TaskMemoryStore
from mu.retrieval.index import SemanticCodeIndex
from mu.retrieval.index import RETRIEVAL_INDEX as _RETRIEVAL_INDEX
from mu.workspace.folder_context import FolderContext
from providers.base import LLMProvider, Message, MessagePart, FileReference, ImageData
from mu.tools._dispatcher import execute_tool
from mu.tools._envelope import infer_tool_error_code
from mu.tools.descriptors import TOOLS, COLLATED_TOOLS
# Importing `mu.tools` triggers `@tool`-decorator registrations
# (every tool in mu/tools/<group>/handlers.py) which mirror into
# `mu.tools.descriptors.TOOLS` / `TOOL_DESCRIPTORS` /
# `mu.tools._dispatcher.TOOL_HANDLERS` so the Session loop sees them.
import mu.tools  # noqa: F401
from utils.logger import logger
from utils.helpers import get_safe_mime_type, display_image_in_terminal
from utils.runtime_metrics import build_live_status_line
from utils.config import (
    calculate_cost,
    AGENTIC_SYSTEM_BASE,
    AGENTIC_MODES,
    DEFAULT_VARIABLES,
    NUDGE_EMPTY_RESPONSE,
    validate_and_cast,
)


# Shared helpers live in `mu/session/helpers.py` (extracted to break the
# circular-import cycle with `mu/agent/loop_body.py` and
# `mu/session/manager.py`). Re-exported here so `mucli` and tests that
# import these names from `mu.session.session` keep working.
from mu.session.helpers import (
    _HookAbort,
    _hook_abort_envelope,
    _safe_feature_path_prefix,
    _sanitize_for_log,
    _shorten_tool_args,
    _slugify_feature_id,
    derive_feature_state_status,
)
from mu.session.history import HistoryMixin


# `SessionManager` lives in `mu/session/manager.py`. Re-exported here
# so `from mu.session.session import SessionManager` keeps working.
from mu.session.manager import SessionManager  # noqa: E402, F401


class Session:
    def __init__(
        self,
        provider: LLMProvider,
        thinking: bool,
        system_instruction: str,
        session_manager: SessionManager,
        ui=None,
        debug: bool = False,
    ):
        logger.info("Initializing Session object")
        self.provider = provider
        self.thinking = thinking
        self.system_instruction = system_instruction
        self.session_manager = session_manager
        self.ui = ui
        self.debug = debug
        self.variables = session_manager.variables
        self.agentic = True
        # Dual-registry contract — see run_turn in mu/agent/loop_body.py:
        # staged_files = ephemeral provider-native payloads for THIS turn
        # (image_input bytes / file_refs), cleared after the turn.
        # staged_attachments = durable attachment descriptors that rehydrate
        # as a tool-notice text part. add_file writes to both on purpose.
        self.staged_files = []
        self.staged_attachments = []
        self.disabled_tools = []  # list of tool names strings
        self.disabled_skills: list[str] = []  # names of skills to suppress
        self.retrieval_index = _RETRIEVAL_INDEX
        self._pending_retrieved_context = ""
        self._pending_user_text = ""
        # Per-turn dedup for skill auto-expansion banners: which skills we
        # already announced for the current `_pending_user_text`, so a
        # re-assembly of the system prompt (retry / re-inject) doesn't
        # double-print. Reset when the user text changes. Only the
        # `announce=True` path (real turn assembly) touches this — the
        # `/memory` size-measurement path leaves it alone.
        self._skills_announced_text: str = ""
        self._skills_announced: set[str] = set()
        # One-shot system-prompt briefings queued by load/switch commands.
        # Drained at the top of every agent turn so the model knows it
        # just resumed an in-flight course / feature / session and can
        # re-orient without the user re-explaining state. See
        # `queue_resumption_briefing` below + the drain site in
        # `mu.agent.loop_body`.
        self._pending_resumption_briefings: list[str] = []
        self.paused_execution_text: str | None = None
        # Flips to True when `raise_blocker` fires inside the agentic
        # loop. The loop-mode watchdog reads it to know the agent
        # paused intentionally — otherwise it would keep prodding the
        # model with "continue!" and burn tokens in a wedge loop.
        self._loop_blocker_raised: bool = False
        # Flips to True when a hook returns HookResult(action="abort")
        # at any fire site. The agentic loop checks this at its
        # iteration boundary and exits cleanly with status
        # "hook_aborted". `_hook_abort_reason` carries the payload from
        # the aborting hook for the turn-response error field.
        self._hook_abort_requested: bool = False
        self._hook_abort_reason: str | None = None
        # Reactive overflow recovery (Claude Code Tier 5): counts compact-and-
        # retry attempts on a "prompt too long" provider error so we never loop
        # compact-and-fail, while still letting a later iteration in the same
        # turn recover from a fresh overflow. Capped at
        # _MAX_OVERFLOW_RECOVERIES_PER_TURN (in mu.agent.loop_body). Reset per
        # turn in run_turn + _collect_turn_response.
        self._overflow_recoveries_this_turn: int = 0
        # Async sub-agent orchestrator state. `_subagent_cancelled` is the
        # cooperative kill flag (read at the top of each run_turn iteration,
        # mirroring `_hook_abort_requested`); `_subagent_kill_reason` carries
        # the reason ("killed_by_parent" | "runtime_exceeded" | ...).
        # `_subagent_lifecycle` is set per-child by spawn_agent so the child's
        # own loop can feed tool calls to its SubagentLifecycleManager.
        # `_subagent_registry` is the per-parent control plane for async
        # sub-agent runs (task_id -> running child). All absent / default for
        # single-agent sessions.
        self._subagent_cancelled: bool = False
        self._subagent_kill_reason: str | None = None
        self._subagent_lifecycle = None
        from mu.agent.registry import SubagentRegistry

        self._subagent_registry = SubagentRegistry()
        # Per-session usage tracker. Populated by the pre_tool /
        # post_tool hooks in `mu/agent/usage_tracker.py`. Surfaced via
        # `/stats`. Reset via `/stats clear`.
        import time as _time_mod

        self.tool_stats: dict = {
            "session_started_at": _time_mod.time(),
            "first_call_at": None,
            "last_call_at": None,
            "tools": {},  # name → {count, success, failed, total_ms, last_used_at, last_args}
            "skills": {},  # name → {invocations, last_used_at}
            "approvals": {"approved": 0, "denied": 0},
            "errors": {},  # error_code → count
        }
        from mu.tools.shell.background import BackgroundTaskRegistry
        self.background_tasks = BackgroundTaskRegistry()

        self.sync_runtime_state()
        if self.folder_context.folders:
            if self.ui:
                self.ui.show_info(
                    f"Restored folder context: {', '.join(self.folder_context.folders)}"
                )
            logger.info(f"Restored folder context: {self.folder_context.folders}")
            # Deliberately NOT calling os.chdir() here: the process cwd is
            # global, and GUI/container mode runs multiple sessions in one
            # process (one thread each) — a chdir would race sessions with
            # different workspaces. Tool handlers resolve their working
            # directory per-call via default_working_directory(), which
            # already prefers folder_context.folders[0].
            if self.ui:
                self.ui.show_info(
                    f"Workspace root: {self.folder_context.folders[0]}"
                )

    def stage_attachment_ids(self, attachment_ids, *, replace=True):
        """Validate registry IDs and stage descriptor-only history parts."""
        registry = getattr(self, "attachment_registry", None)
        if registry is None:
            raise ValueError("attachment registry is unavailable for this session")
        staged = []
        seen = set()
        for raw_id in attachment_ids or []:
            attachment_id = str(raw_id or "").strip()
            if not attachment_id or attachment_id in seen:
                continue
            descriptor = registry.get(attachment_id)
            if descriptor is None or registry.resolve_path(attachment_id) is None:
                raise ValueError(f"attachment not found: {attachment_id}")
            seen.add(attachment_id)
            staged.append(dict(descriptor))
        current = [] if replace else list(getattr(self, "staged_attachments", []) or [])
        by_id = {
            str(part.get("attachment", {}).get("attachment_id") or ""): part
            for part in current
            if isinstance(part, dict)
        }
        for descriptor in staged:
            by_id[descriptor["attachment_id"]] = {
                "type": "attachment",
                "attachment": descriptor,
            }
        self.staged_attachments = [part for key, part in by_id.items() if key]
        return staged

    def add_file(self, file_path):
        file_path = file_path.strip("'\"")
        file_path = os.path.expanduser(file_path)

        if not os.path.exists(file_path):
            if self.ui:
                self.ui.show_error(f"Error: File '{file_path}' not found.")
            return

        safe_mime = get_safe_mime_type(file_path)

        # The durable attachment registry is the canonical path for user inputs.
        # Keep the old provider-native staging below as a compatibility fallback
        # for sessions created before runtime-state initialization.
        registry = getattr(self, "attachment_registry", None)
        if registry is not None:
            try:
                descriptor = registry.add(
                    os.path.basename(file_path), file_path, safe_mime
                )
                self.stage_attachment_ids([descriptor["attachment_id"]], replace=False)
                if self.ui:
                    self.ui.show_info(
                        f"Attached {descriptor['name']} ({descriptor['attachment_id'][:8]})"
                    )
                # Fall through to the provider-native staging below so
                # staged_files mirrors the legacy contract: images become
                # image_input dicts with the full source path, other files
                # keep flowing through provider.upload_file.
            except Exception as e:
                if self.ui:
                    self.ui.show_error(f"Attachment failed: {e}")
                return None

        # Images route through the vision path (image_input + raw bytes), not
        # the file-ref path — provider.upload_file for OpenAI/Ollama returns a
        # local path that becomes a plain "[File: ...]" text stub, which
        # vision-capable models can't actually look at.
        if safe_mime.startswith("image/"):
            try:
                with open(file_path, "rb") as fh:
                    raw = fh.read()
            except OSError as e:
                if self.ui:
                    self.ui.show_error(f"Could not read image: {e}")
                return
            import base64 as _b64
            self.staged_files.append(
                {
                    "type": "image_input",
                    "image": {
                        "data_b64": _b64.b64encode(raw).decode("ascii"),
                        "mime_type": safe_mime,
                        "source": file_path,
                    },
                }
            )
            if self.ui:
                size_kb = max(1, len(raw) // 1024)
                self.ui.show_info(
                    f"Staged image: {os.path.basename(file_path)} ({safe_mime}, {size_kb} KB)"
                )
            return

        if self.ui:
            self.ui.show_info(f"Uploading {file_path} as {safe_mime}...")

        try:
            file_ref = self.provider.upload_file(file_path, safe_mime)
            if file_ref:
                self.staged_files.append(
                    {
                        "type": "file",
                        "file_ref": {
                            "uri": file_ref.uri,
                            "mime_type": file_ref.mime_type,
                            "display_name": file_ref.display_name,
                        },
                    }
                )
                if self.ui:
                    self.ui.show_info("Upload complete.")
        except Exception as e:
            if self.ui:
                self.ui.show_error(f"Upload failed: {e}")

    def clear_files(self):
        self.staged_files = []
        self.staged_attachments = []
        if self.ui:
            self.ui.show_info("Staged files cleared.")

    def sync_runtime_state(self):
        self.folder_context = self.session_manager.folder_context
        self.collation_buffer = self.session_manager.collation_buffer
        self.task_memory = self.session_manager.task_memory
        self.turn_scratchpad = self.session_manager.turn_scratchpad
        self.tool_result_cache = self.session_manager.tool_result_cache
        self.tool_stats = self.session_manager.tool_stats
        self.feature_state = self.session_manager.get_feature_state()
        self.variables = self.session_manager.variables
        self.thread_meta = self.session_manager.thread_meta
        thread_session_name = self.session_manager.current_session_name
        self._thread_coordination_required = bool(thread_session_name)
        if not self._thread_coordination_required:
            # Ephemeral/test Sessions without a durable identity are not
            # threads yet, so there is no journal boundary to enforce.
            self.thread_coordinator = None
            self._thread_coordination_error = ""
        else:
            try:
                from mu.threads.coordinator import ThreadCoordinator

                publish = None
                if self.ui is not None:
                    publish = getattr(self.ui, "_publish", None) or getattr(
                        self.ui, "publish", None
                    )

                def _publish_thread_event(event):
                    if publish is not None:
                        publish(
                            {
                                **event,
                                "session_name": thread_session_name,
                            }
                        )

                self.thread_coordinator = ThreadCoordinator(
                    self.thread_meta.group_id,
                    publish=_publish_thread_event if publish is not None else None,
                )
                self.thread_coordinator.register_thread(
                    self.thread_meta, thread_session_name
                )
                self._thread_coordination_error = ""
            except Exception as exc:
                self.thread_coordinator = None
                self._thread_coordination_error = str(exc)
                logger.debug("thread coordinator initialization failed", exc_info=True)
        if not getattr(self, "_thread_runtime_id", None):
            self._thread_runtime_id = f"runtime-{os.getpid()}-{uuid.uuid4().hex}"
        # Artifacts are intentionally self-managed outside session.json. Build
        # the registry lazily so old sessions load without migrations.
        try:
            from mu.artifact import ArtifactRegistry
            from utils.config import HISTORY_DIR

            session_dir = os.path.join(
                HISTORY_DIR, "sessions", self.session_manager.current_session_name
            )
            self.artifact_registry = ArtifactRegistry(session_dir)
        except Exception:
            self.artifact_registry = None
        # MUCLI_SUBAGENT_DURABLE_RESULTS_V1: bind after every session load/reload.
        try:
            self._subagent_registry.bind_parent(self)
        except Exception:
            # Defensive: best-effort path must not break the caller.
            logger.debug("Suppressed exception", exc_info=True)
        try:
            from mu.attachment import AttachmentRegistry
            from utils.config import HISTORY_DIR

            session_dir = os.path.join(
                HISTORY_DIR, "sessions", self.session_manager.current_session_name
            )
            self.attachment_registry = AttachmentRegistry(session_dir)
        except Exception:
            self.attachment_registry = None

        try:
            from mu.container.registry import ContainerRegistry

            self.container_ref = next(
                (
                    ref
                    for ref in ContainerRegistry().list_containers()
                    if self.session_manager.current_session_name in ref.attached_sessions
                ),
                None,
            )
        except Exception:
            self.container_ref = None
        setattr(
            self.folder_context,
            "feature_metadata_dir",
            self.session_manager.get_feature_metadata_root(),
        )
        setattr(
            self.folder_context,
            "feature_metadata_index",
            self.session_manager.get_feature_metadata_index(),
        )

    def get_durable_memory_service(self):
        """Shared Memory Ledger used by TUI, web, mobile and agent loop."""

        return self.session_manager.get_durable_memory_service()

    def _derive_feature_state_status(self, feature_plan: dict | None) -> str:
        return derive_feature_state_status(feature_plan)

    def _set_feature_state(
        self,
        *,
        feature_plan: dict | None = None,
        status: str | None = None,
        blocker: dict | None = None,
    ):
        current = self.session_manager.get_feature_state() or {}
        current_plan = current.get("feature_plan")
        plan_summary = feature_plan if isinstance(feature_plan, dict) else current_plan
        next_phase = (
            plan_summary.get("next_phase")
            if isinstance(plan_summary, dict)
            else current.get("next_phase")
        )
        state = {
            "type": "feature",
            "status": status or derive_feature_state_status(plan_summary),
            "feature_id": (
                plan_summary.get("feature_id")
                if isinstance(plan_summary, dict)
                else current.get("feature_id")
            ),
            "feature_name": (
                plan_summary.get("feature_name")
                if isinstance(plan_summary, dict)
                else current.get("feature_name")
            ),
            "directory": (
                plan_summary.get("directory")
                if isinstance(plan_summary, dict)
                else current.get("directory")
            ),
            "metadata_path": (
                plan_summary.get("metadata_path")
                if isinstance(plan_summary, dict)
                else current.get("metadata_path")
            ),
            "next_phase": next_phase,
            "feature_plan": plan_summary,
            "blocker": blocker,
            "updated_at": time.time(),
        }
        previous_feature_id = str(current.get("feature_id", "") or "").strip()
        new_feature_id = str(state.get("feature_id", "") or "").strip()
        same_feature = previous_feature_id and previous_feature_id == new_feature_id
        state["started_at"] = (
            float(current.get("started_at", time.time()) or time.time())
            if same_feature
            else time.time()
        )
        state["start_tokens"] = (
            int(
                current.get(
                    "start_tokens",
                    self.session_manager.token_counts.get("total", 0),
                )
                or 0
            )
            if same_feature
            else int(self.session_manager.token_counts.get("total", 0) or 0)
        )
        self.session_manager.set_feature_state(state, self.folder_context)
        self.sync_runtime_state()

    def _refresh_feature_state(
        self, metadata_path: str, *, status: str | None = None
    ):
        try:
            plan = refresh_and_persist_feature_plan(
                self.session_manager.current_session_name,
                metadata_path=metadata_path,
            )
            self._set_feature_state(
                feature_plan=summarize_feature_plan(plan),
                status=status,
            )
        except (FileNotFoundError, OSError, ValueError):
            return

    def _sync_feature_state_for_tool(
        self,
        tool_name: str,
        tool_args: dict,
        raw_result,
        structured_result,
    ):
        """Feature-state writer. Body moved to
        `mu/session/tools_glue.py:sync_feature_state_for_tool`."""
        from mu.session.tools_glue import sync_feature_state_for_tool

        return sync_feature_state_for_tool(
            self,
            tool_name,
            tool_args,
            raw_result,
            structured_result,
        )

    # Message helpers (`_build_messages_from_history`,
    # `_summarize_message_parts`,
    # `_clip_preview`, `_prepare_runtime_history`) moved to
    # `mu/session/messages.py`. Forwarders preserve the bound-method
    # interface for the agent loop and tests.

    def _build_messages_from_history(
        self, recent_history_dicts, new_user_message_dict
    ) -> list[Message]:
        from mu.session.messages import build_messages_from_history
        from mu.session.budgets import resolve_tool_result_floor

        floor = resolve_tool_result_floor(self)
        return build_messages_from_history(
            recent_history_dicts, new_user_message_dict,
            tool_result_floor=floor,
        )

    def _summarize_message_parts(self, msg_dict: dict) -> str:
        from mu.session.messages import summarize_message_parts

        return summarize_message_parts(
            msg_dict, provider=getattr(self, "provider", None)
        )

    # Budget helpers (`_resolve_context_limit`, `_resolve_response_reserve`,
    # `_compaction_token_budget`) moved to `mu/session/budgets.py`. These
    # forwarders preserve the bound-method interface so existing call sites
    # don't need to thread a `session` parameter around.

    def _resolve_context_limit(self) -> int:
        from mu.session.budgets import resolve_context_limit

        return resolve_context_limit(self)

    def _resolve_response_reserve(self) -> int:
        from mu.session.budgets import resolve_response_reserve

        return resolve_response_reserve(self)

    def _compaction_token_budget(self) -> int:
        from mu.session.budgets import compaction_token_budget

        return compaction_token_budget(self)

    def _prepare_runtime_history(
        self, turn_start_index: int | None = None
    ) -> list[dict]:
        """History slicing + tool-window compression. Body moved to
        `mu/session/messages.py:prepare_runtime_history`."""
        from mu.session.messages import prepare_runtime_history

        return prepare_runtime_history(
            self, turn_start_index, provider=getattr(self, "provider", None)
        )

    def _inject_conversation_summary(self, system_prompt: str) -> str:
        summary = str(
            getattr(self.session_manager, "conversation_summary", "") or ""
        ).strip()
        if not summary:
            return system_prompt
        # Prepend the pinned session_goal as a preamble to the L2 summary.
        # This ensures the goal survives compaction even if L3 is small
        # or the session_goal variable is cleared at end of turn.
        session_goal = str(self.variables.get("session_goal", "") or "").strip()
        preamble = ""
        if session_goal:
            preamble = f"[Active Goal: {session_goal}]\n\n"
        return (
            f"{system_prompt}\n\n"
            "A rolling summary of older conversation history is available below. "
            "Use it for long-term continuity before re-reading or re-deriving prior work.\n"
            f"{preamble}{summary}"
        )

    def _build_active_goal_context(self) -> str:
        sections = []
        # session_goal is the mode-agnostic, top-level pinned ask. It
        # renders FIRST so it survives every compaction and reminds the
        # model what the user originally wanted across long runs.
        session_goal = str(self.variables.get("session_goal", "") or "").strip()
        if session_goal:
            sections.append(f"- session_goal (pinned): {session_goal}")
            sections.append(
                f"- Active Goal: {session_goal}"
            )
            sections.append(
                "- session_goal_policy: every meaningful action should advance "
                "this goal. If a sub-task drifts off, pause and re-anchor. "
                "Use /goal clear when the user signals the goal has shifted."
            )
        loop_goal = str(self.variables.get("loop_goal", "") or "").strip()
        if loop_goal and str(self.variables.get("agent_mode", "default")).lower() == "loop":
            sections.append(f"- loop_goal: {loop_goal}")
            sections.append(
                "- loop_memory_policy: persist durable findings with save_memory and in-flight steps with save_scratchpad."
            )
        feature_state = self.session_manager.get_feature_state()
        if isinstance(feature_state, dict):
            feature_id = str(feature_state.get("feature_id", "") or "").strip()
            status = str(feature_state.get("status", "idle") or "idle")
            next_task = feature_state.get("next_task")
            if isinstance(next_task, dict):
                next_task_text = str(
                    next_task.get("title")
                    or next_task.get("task")
                    or next_task.get("name")
                    or ""
                ).strip()
            else:
                next_task_text = str(next_task or "").strip()
            phase = feature_state.get("next_phase")
            phase_title = (
                str((phase or {}).get("title", "")).strip()
                if isinstance(phase, dict)
                else ""
            )
            sections.append(f"- feature_id: {feature_id or 'n/a'}")
            sections.append(f"- status: {status}")
            if phase_title:
                sections.append(f"- active_phase: {phase_title}")
            if next_task_text:
                sections.append(f"- next_task: {next_task_text}")

        return "\n".join(sections).strip()

    def _ensure_session_goal_persistence(self) -> None:
        """Mirror the live `session_goal` variable into task_memory once
        per goal value so compaction can never erase the user's original
        top-level ask. Mode-agnostic — runs every turn for every mode.

        Idempotent: searches existing memory for the goal text first and
        only writes if absent. The variable is always the source of
        truth for L3 rendering; the memory entry is a durable audit
        trace and a recovery hatch if the variable ever gets cleared
        accidentally.

        Lifecycle: saves with kind='goal', status='active'. Stores
        the entry ID on ``self._active_goal_memory_id`` so the strip
        / clear path can mark it 'done' (audit trail retained).
        On goal shift (new text while old is active), marks the old
        entry 'done' before creating the new one.
        """
        session_goal = str(self.variables.get("session_goal", "") or "").strip()
        if not session_goal:
            return
        # Check if this exact goal text is already persisted as an active
        # goal entry — if so, just record the ID and return (idempotent).
        existing = self.task_memory.search("session goal", limit=12)
        for entry in existing:
            if (
                session_goal in str(entry.content or "")
                and getattr(entry, "status", "active") == "active"
                and getattr(entry, "kind", "") == "goal"
            ):
                self._active_goal_memory_id = entry.id
                return
        # Goal shift: mark the previous active goal entry as done before
        # creating the new one. The old entry stays in the store for
        # audit trail — it's just deprioritized.
        old_id = getattr(self, "_active_goal_memory_id", None)
        if old_id is not None:
            old_entry = self.task_memory.get_entry(old_id)
            if old_entry is not None and getattr(old_entry, "status", "active") == "active":
                self.task_memory.update_status(old_id, "done")
        # Save new goal entry with lifecycle metadata.
        entry = self.task_memory.save(
            f"Locked session goal: {session_goal}",
            tags=["goal", "session-goal", "locked"],
            source="session_goal",
            kind="goal",
            status="active",
        )
        self._active_goal_memory_id = entry.id

    def _ensure_loop_goal_persistence(self) -> None:
        if str(self.variables.get("agent_mode", "default")).lower() != "loop":
            return
        loop_goal = str(self.variables.get("loop_goal", "") or "").strip()
        if not loop_goal:
            return
        existing = self.task_memory.search("loop goal", limit=12)
        if any(loop_goal in str(entry.content or "") for entry in existing):
            return
        self.task_memory.save(
            f"Locked loop goal: {loop_goal}",
            tags=["loop", "goal", "locked"],
            source="loop_mode",
        )
        self.turn_scratchpad.save(
            f"Current loop goal: {loop_goal}",
            tags=["loop", "goal"],
            source="loop_mode",
        )

    # ── Loop state management ────────────────────────────────────────

    def get_loop_state(self) -> dict:
        """Return the current loop mode state dict."""
        loop_features_raw = self.variables.get("loop_features", "")
        try:
            loop_features = json.loads(loop_features_raw) if loop_features_raw else []
        except (json.JSONDecodeError, TypeError):
            loop_features = []
        return {
            "goal": self.variables.get("loop_goal", ""),
            "active": self.variables.get("loop_active", False),
            "features": loop_features,
        }

    def start_loop(self, goal: str) -> None:
        """Activate loop mode with the given long-horizon goal."""
        self.variables["loop_goal"] = goal
        self.variables["loop_active"] = True
        self.variables["loop_features"] = json.dumps([])
        self.variables["agent_mode"] = "loop"
        self._ensure_loop_goal_persistence()
        self.session_manager.save_history(self.folder_context)

    def stop_loop(self) -> None:
        """Deactivate loop mode."""
        self.variables["loop_active"] = False
        self.session_manager.save_history(self.folder_context)

    def add_loop_feature(self, feature_id: str) -> None:
        """Record a feature created during this loop session."""
        loop_features_raw = self.variables.get("loop_features", "")
        try:
            loop_features = json.loads(loop_features_raw) if loop_features_raw else []
        except (json.JSONDecodeError, TypeError):
            loop_features = []
        loop_features.append({
            "id": feature_id,
            "timestamp": datetime.now().isoformat(),
        })
        self.variables["loop_features"] = json.dumps(loop_features)
        self.save_history()

    def get_loop_features(self) -> list:
        """Return list of feature dicts created during this loop."""
        state = self.get_loop_state()
        return state.get("features", [])

    # ── End loop state management ─────────────────────────────────────

    def _build_retrieved_workspace_context(self, query: str) -> str:
        if not self.folder_context or not self.folder_context.folders:
            return ""
        request = str(query or "").strip()
        if not request:
            return ""
        top_k = max(1, int(self.variables.get("retrieval_top_k", 5) or 5))
        char_budget = max(
            1, int(self.variables.get("retrieval_context_char_limit", 10000) or 10000)
        )
        self.retrieval_index.refresh_incremental(self.folder_context)
        payload = self.retrieval_index.retrieve(request, top_k=top_k, filters={})
        lines = []
        used = 0
        for item in payload.get("results", []):
            snippet = str(item.get("snippet", "") or "").strip()
            entry = (
                f"- {item.get('path')} (score={item.get('score')})\n"
                f"{snippet}\n"
            )
            if used + len(entry) > char_budget and lines:
                break
            lines.append(entry)
            used += len(entry)
        if not lines:
            return ""
        return "".join(lines).strip()

    def _build_context_files_block(self) -> str:
        """LAYER 1A — load context files (AGENTS.md, CLAUDE.md, MUCLI.md,
        .mu/CONTEXT.md) from workspace folders and the global ~/.mu/CONTEXT.md.

        Whole-file-or-skip: files that fit the remaining budget are included
        in full; files that exceed the remaining budget are skipped with a
        marker. No truncation — middle content is never lost.

        Discovery order per folder (first match wins):
          1. AGENTS.md   2. CLAUDE.md   3. MUCLI.md   4. .mu/CONTEXT.md

        Global ~/.mu/CONTEXT.md is loaded first (broadest scope), then each
        attached workspace folder (increasingly specific). Identical files
        are deduplicated by content hash.
        """
        import hashlib
        import os

        raw_budget = self.variables.get("context_files_max_chars", 8000)
        try:
            budget = max(0, int(raw_budget)) if raw_budget is not None else 8000
        except (TypeError, ValueError):
            budget = 8000
        if budget == 0:
            return ""

        candidate_names = ["AGENTS.md", "CLAUDE.md", "MUCLI.md", ".mu/CONTEXT.md"]
        seen_hashes: set[str] = set()
        blocks: list[str] = []
        remaining = budget

        # Build search paths: global first, then each workspace folder.
        search_paths: list[tuple[str, str]] = []
        global_ctx = os.path.expanduser("~/.mu/CONTEXT.md")
        if os.path.isfile(global_ctx):
            search_paths.append(("~/.mu/CONTEXT.md", global_ctx))
        folders = (
            list(self.folder_context.folders)
            if self.folder_context and self.folder_context.folders
            else []
        )
        for folder in folders:
            folder_path = str(folder)
            for name in candidate_names:
                candidate = os.path.join(folder_path, name)
                if os.path.isfile(candidate):
                    search_paths.append((name, candidate))
                    break  # first match per folder wins

        for label, filepath in search_paths:
            if remaining <= 0:
                break
            try:
                with open(filepath, "r", errors="replace") as f:
                    content_text = f.read()
            except Exception:
                continue
            if not content_text.strip():
                continue
            content_hash = hashlib.md5(content_text.encode("utf-8", errors="replace")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            file_len = len(content_text)
            if file_len > remaining:
                over = file_len - remaining
                blocks.append(f"## {label}\n[skipped: {file_len} chars, {over} over remaining budget]")
                # Don't reduce remaining — skipped files don't consume budget.
                # But stop if we've skipped — remaining stays, next file might fit.
                continue
            blocks.append(f"## {label}\n{content_text}")
            remaining -= file_len

        if not blocks:
            return ""
        return "\n\n".join(blocks)

    def _build_skills_block(self, *, announce: bool = False) -> str:
        """LAYER 1B — render the installed skills (from `mu/skills/`,
        `~/.mu/skills/`, and `<workspace>/.mu/skills/`) into a labelled
        system-prompt block. Capped by `skills_max_chars` (default 6144).
        Mode is controlled by `skills_mode` (`"compact"` default).

        When `announce` is True (the real per-turn system-prompt assembly
        path), every skill whose `trigger` regex matches the current
        user message prints the same `🎯 SKILL ACTIVE` banner the
        `invoke_skill` tool uses — so trigger-regex auto-expansion is
        visible, not silent. The `/memory` size-measurement path calls
        this with `announce=False` (the default) and never banners.
        """
        try:
            from mu.skills import (
                announce_skill,
                discover_skills,
                render_skills_block,
            )
        except ImportError:
            return ""
        raw = self.variables.get("skills_max_chars", 6144)
        try:
            budget = max(0, int(raw)) if raw is not None else 6144
        except (TypeError, ValueError):
            budget = 6144
        if budget == 0:
            return ""
        folders = (
            list(self.folder_context.folders)
            if self.folder_context and self.folder_context.folders
            else []
        )
        skills = discover_skills(folders)
        disabled = set(getattr(self, "disabled_skills", []) or [])
        if disabled:
            skills = [s for s in skills if s.name not in disabled]
        mode = str(self.variables.get("skills_mode", "compact") or "compact").lower()
        if mode not in {"compact", "full"}:
            mode = "compact"
        user_text = str(getattr(self, "_pending_user_text", "") or "")

        # Trigger-regex auto-expansion is a compact-mode concept (in "full"
        # mode every body is already inlined, so there's no hidden
        # activation to surface). Announce matched skills once per turn.
        if announce and mode == "compact" and user_text:
            try:
                self._announce_auto_expanded_skills(skills, user_text, announce_skill)
            except Exception:
                logger.debug("skill auto-expand banner failed", exc_info=True)

        # Auto-expanded bodies get a fractional slice of the L1B budget by
        # default: a multi-KB skill body inlined on trigger-match crowds the
        # name+description index out and ships content the model can pull
        # on demand via `invoke_skill`. 0.4 caps a body at 40% of L1B —
        # enough head + front-matter to act on, tail dropped deliberately
        # (context-optimisation lever). Set 1.0 to restore legacy inline-all.
        body_budget_scale = 0.4
        return render_skills_block(
            skills,
            budget=budget,
            user_text=user_text,
            mode=mode,
            budget_scale=body_budget_scale,
        )

    def _announce_auto_expanded_skills(self, skills, user_text, announce_skill) -> None:
        """Print the activation banner for each skill whose trigger regex
        matches `user_text`, deduped per turn so a re-injected system
        prompt doesn't double-print. Also bumps a per-skill
        `auto_expansions` counter on `tool_stats` so `/stats` can audit
        trigger effectiveness alongside explicit `invoke_skill` calls."""
        if user_text != self._skills_announced_text:
            self._skills_announced_text = user_text
            self._skills_announced = set()
        ui = getattr(self, "ui", None)
        stats = getattr(self, "tool_stats", None)
        from mu.skills import match_trigger

        for skill in skills:
            if not match_trigger(skill, user_text):
                continue
            if skill.name in self._skills_announced:
                continue
            self._skills_announced.add(skill.name)
            announce_skill(ui, skill.name, via="trigger")
            # Defensive, optional stats bump — never let accounting break
            # the turn.
            if isinstance(stats, dict):
                try:
                    sk_bucket = stats.setdefault(
                        "skills", {}
                    ).setdefault(
                        skill.name,
                        {"invocations": 0, "auto_expansions": 0, "last_used_at": None},
                    )
                    sk_bucket.setdefault("auto_expansions", 0)
                    sk_bucket.setdefault("invocations", 0)
                    sk_bucket.setdefault("last_used_at", None)
                    sk_bucket["auto_expansions"] = int(sk_bucket["auto_expansions"]) + 1
                except Exception:
                    # Defensive: best-effort path must not break the caller.
                    logger.debug("Suppressed exception", exc_info=True)

    def _inject_hierarchical_context(
        self,
        system_prompt: str,
        *,
        cached_skills: str | None = None,
        cached_context_files: str | None = None,
    ) -> str:
        """Layered system-prompt assembly. Body moved to
        `mu/session/context.py:inject_hierarchical_context`.

        ``cached_skills`` forwards per-turn-cached L1B text so the agent
        loop can rebuild L2 / L3 fresh every iteration without re-reading
        the skills tree from disk each time. ``cached_context_files``
        forwards per-turn-cached L1A context-files text for the same reason.
        """
        from mu.session.context import inject_hierarchical_context

        return inject_hierarchical_context(
            self,
            system_prompt,
            cached_skills=cached_skills,
            cached_context_files=cached_context_files,
        )

    def queue_resumption_briefing(self, briefing: str) -> None:
        """Add a one-shot resumption note to the next agent turn.

        Used by /teach load, /feature load, and session-switch paths to
        tell the agent it just resumed in-flight state: which course /
        feature is active, where the user was last, what's pending. The
        briefing flushes into the next turn's system prompt and then
        clears — it never accumulates.
        """
        text = (briefing or "").strip()
        if not text:
            return
        if not hasattr(self, "_pending_resumption_briefings"):
            self._pending_resumption_briefings = []
        self._pending_resumption_briefings.append(text)

    def _drain_resumption_briefings(self) -> str:
        """Drain queued resumption briefings into a formatted block for
        the system prompt. Returns an empty string if none are queued."""
        briefings = getattr(self, "_pending_resumption_briefings", None) or []
        if not briefings:
            return ""
        self._pending_resumption_briefings = []
        body = "\n\n".join(briefings)
        return (
            "## RESUMPTION CONTEXT\n"
            "You just resumed in-flight work. Orient against this state "
            "before responding — do NOT ask the user to re-explain.\n\n"
            f"{body}"
        )

    def _render_tool_result(self, result) -> str:
        if isinstance(result, dict):
            summary = result.get("summary")
            if summary:
                return str(summary)
            return json.dumps(result, indent=2, sort_keys=True)
        if isinstance(result, list):
            return json.dumps(result, indent=2, sort_keys=True)
        return str(result)

    def _clip_preview(self, text: str, limit: int = 240) -> str:
        from mu.session.messages import clip_preview

        return clip_preview(text, limit)

    def _parse_search_results(self, raw_result: str) -> dict:
        matches = []
        file_counts = defaultdict(int)
        for line in str(raw_result).splitlines():
            if " -> " not in line or ":" not in line:
                continue
            path_and_line, snippet = line.split(" -> ", 1)
            try:
                path, line_no = path_and_line.rsplit(":", 1)
                line_no = int(line_no)
            except ValueError:
                continue
            file_counts[path] += 1
            if len(matches) < 8:
                matches.append(
                    {
                        "path": path,
                        "line": line_no,
                        "snippet": self._clip_preview(snippet, 160),
                    }
                )
        return {
            "match_count": sum(file_counts.values()),
            "file_count": len(file_counts),
            "files": sorted(file_counts.keys())[:8],
            "matches": matches,
        }

    def _parse_workspace_details(self, raw_result: str) -> dict:
        folders = []
        tracked_files = []
        section = None
        for line in str(raw_result).splitlines():
            stripped = line.strip()
            if stripped == "Workspace Folders:":
                section = "folders"
                continue
            if stripped == "Tracked Files:":
                section = "files"
                continue
            if stripped.startswith("- "):
                value = stripped[2:]
                if section == "folders":
                    folders.append(value)
                elif section == "files":
                    tracked_files.append(value)
        return {
            "folders": folders,
            "folder_count": len(folders),
            "tracked_file_count": len(tracked_files),
            "tracked_files_preview": tracked_files[:10],
        }

    def _parse_list_dir(self, raw_result: str, path: str) -> dict:
        entries = [
            line.strip() for line in str(raw_result).splitlines() if line.strip()
        ]
        return {
            "path": path or ".",
            "entry_count": len(entries),
            "entries_preview": entries[:20],
        }

    def _parse_json_result(self, raw_result: str) -> dict:
        try:
            parsed = json.loads(str(raw_result))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except (TypeError, json.JSONDecodeError):
            return {"preview": self._clip_preview(raw_result, 260)}

    def _unwrap_tool_envelope(self, raw_result):
        parsed = self._parse_json_result(raw_result)
        required = {"ok", "error_code", "message", "data", "artifacts", "telemetry"}
        if not isinstance(parsed, dict) or not required.issubset(parsed.keys()):
            return None, raw_result
        message = parsed.get("message", "")
        data = parsed.get("data")
        if isinstance(message, str) and message.strip():
            return parsed, message
        if isinstance(data, str):
            return parsed, data
        return parsed, raw_result

    def _build_feature_mode_prompt(self, text: str) -> str:
        base_instruction = (
            "FEATURE MODE DIRECTIVE: use the feature-task engine for this request. First call create_feature to create canonical session-managed feature metadata, then create_phases, then create_task for each planned ticket. "
            "Legacy fallback: create_feature_task is allowed only when a single-call bootstrap is explicitly requested. "
            "Do not create alternate planning documents and do not begin code implementation until the user has reviewed and approved the plan. "
            "After approval, call get_current_task/get_tasks at the start of every implementation turn, work on only the next incomplete task, and keep task state synchronized via tool calls only. "
            "Use get_execution_state to identify the next pending phase/task, use block_task when work cannot continue without user input, and use resume_task after the user provides unblock context. "
            "Use update_task_status/approve_feature_task/get_tasks/get_current_task exclusively to read or change task status. "
            "Every task must define explicit EXIT CRITERIA, and you may set update_task_status(..., status='completed') only after all exit criteria for that task are demonstrably met and verified in the current codebase/tests. "
            "As each criterion is satisfied, call update_task_status with cumulative verified_exit_criteria so progress is visible in the task UI. "
            "In review mode, use review_all_completed_tasks first, then review_completed_tasks with categorized issues (bug/risk/enhancement), propose_task_diff for proposed fixes, decide_task_diff for user approvals/denials, and archive_task once tasks become archive-ready. "
            "Harness execution model: progress one task at a time, validate, then move to the next task. Never batch multiple tasks in one step. "
            "For investigation-heavy turns, gather read-only context first, use save_scratchpad for temporary phase notes, and call flush before acting on the collected context. "
            "Memory discipline is mandatory: use save_memory for durable facts/decisions that must survive long loops; use save_scratchpad for short-lived hypotheses and in-flight notes each turn; query memory/scratchpad before re-reading large context. "
            "If you become blocked because you need a user decision or missing context, call raise_blocker with a precise summary, what you tried, and the exact input you need so the harness can pause and ask the user for help. "
            "Do not pause after progress reports. Unless blocked or waiting on explicit approval/decision, immediately continue to the next actionable implementation step in the same run without asking the user to 'continue'. "
            "Never move to the next task until the current task's exit criteria are fully satisfied and the task is marked completed via update_task_status. "
            "When all tasks are complete, perform a review pass over the tasks and code changes together. If review fails, move failing tasks back to in_progress and continue implementing. If review succeeds, call approve_feature_task with review_status completed before you report success. "
            "In every turn response, clearly identify: current task, evidence gathered, changes made, verification result, and the immediate next step.\n\n"
        )
        return base_instruction + text

    def _build_loop_mode_prompt(self, text: str) -> str:
        loop_goal = str(
            self.variables.get("loop_goal")
            or text
            or ""
        ).strip()
        if loop_goal and not str(self.variables.get("loop_goal", "")).strip():
            self.variables["loop_goal"] = loop_goal
        base_instruction = (
            "LOOP MODE DIRECTIVE: You are executing a long-horizon loop with a locked mission. "
            "Maintain a self-directed backlog, keep exactly one active task, and continuously run plan -> execute -> verify -> re-plan cycles. "
            "Persist durable decisions using save_memory and short-term plans using save_scratchpad. "
            "After each increment, provide a timeline update with: objective, actions, evidence, decision, and next step. "
            "When blocked on user input/credentials/environment constraints, call raise_blocker with explicit unblock requirements. "
            "Do not stop unless user explicitly asks to stop loop mode.\n\n"
            f"LOCKED LOOP GOAL:\n{loop_goal}\n\n"
            "INCREMENT REQUEST:\n"
        )
        return base_instruction + text

    def _feature_doc_tool_violation(self, tool_name: str, tool_args: dict) -> str | None:
        if str(self.variables.get("agent_mode", "default")).lower() != "feature":
            return None
        feature_state = self.session_manager.get_feature_state()
        if not isinstance(feature_state, dict):
            return None
        feature_dir = str(feature_state.get("directory", "") or "").strip()
        if not feature_dir:
            return None
        if tool_name not in {"read_file", "get_chunk", "write_file", "apply_diff"}:
            return None

        arg_key = "file" if tool_name == "get_chunk" else "filename"
        target = str(tool_args.get(arg_key, "") or "").strip()
        if not target:
            return None

        if os.path.isabs(target):
            candidate = os.path.abspath(target)
        elif self.folder_context.folders:
            candidate = os.path.abspath(os.path.join(self.folder_context.folders[0], target))
        else:
            candidate = os.path.abspath(target)

        feature_root = os.path.abspath(feature_dir)
        if not candidate.startswith(feature_root + os.sep):
            return None

        filename = os.path.basename(candidate)
        if filename == "feature_plan.json":
            return (
                "Feature status files are managed by the feature-task engine. "
                f"Do not use {tool_name} on '{filename}'. "
                "Use get_tasks/get_current_task/update_task_status/approve_feature_task instead."
            )
        return None

    def _build_structured_tool_result(
        self,
        tool_name: str,
        tool_args: dict,
        raw_result,
        *,
        execution_source: str = "session",
        cache_key: Optional[str] = None,
    ):
        """Structured-envelope builder. Body moved to
        `mu/session/tools_glue.py:build_structured_tool_result`."""
        from mu.session.tools_glue import build_structured_tool_result

        return build_structured_tool_result(
            self,
            tool_name,
            tool_args,
            raw_result,
            execution_source=execution_source,
            cache_key=cache_key,
        )

    def _record_hook_abort(self, point: str, abort_result) -> None:
        """Stamp the abort flag + reason. Called whenever a hook returns
        `HookResult(action="abort")` at any fire site. The agentic loop
        reads `_hook_abort_requested` at its iteration boundary and
        exits cleanly with status `"hook_aborted"`."""
        reason = abort_result.payload
        if reason is None:
            reason = "Hook requested abort"
        reason_str = str(reason)
        # First abort wins — don't let a later abort clobber the original
        # cause within the same turn.
        if not self._hook_abort_requested:
            self._hook_abort_requested = True
            self._hook_abort_reason = reason_str
            logger.info(f"Hook abort at {point}: {reason_str}")
            if self.ui is not None:
                try:
                    self.ui.show_info(f"⏹  Hook abort ({point}): {reason_str}")
                except Exception:
                    # Defensive: best-effort path must not break the caller.
                    logger.debug("Suppressed exception", exc_info=True)

    def _execute_tool_with_memory(
        self,
        tool_name: str,
        tool_args: dict,
        *,
        invocation_source: str = "session",
    ):
        """Hook-fire dispatch around the canonical executor. Body moved
        to `mu/session/tools_glue.py:execute_tool_with_memory`."""
        from mu.session.tools_glue import execute_tool_with_memory

        return execute_tool_with_memory(
            self,
            tool_name,
            tool_args,
            invocation_source=invocation_source,
        )

    def _prompt_tool_choice(
        self, prompt_text: str, choices: list[str], default: str
    ) -> str:
        if self.ui and hasattr(self.ui, "prompt_choices"):
            return self.ui.prompt_choices(prompt_text, choices=choices, default=default)
        return default

    def _confirm_retry(self) -> bool:
        if self.ui and hasattr(self.ui, "confirm"):
            return self.ui.confirm(
                "An error occurred during the LLM call. Would you like to retry?",
                default=True,
            )
        # No CLI UI available (server mode) — auto-retry
        return True

    def _provider_error_recovery_choice(self) -> str:
        if self.ui and hasattr(self.ui, "prompt_choices"):
            return self._prompt_tool_choice(
                "Provider call failed. Choose recovery strategy:",
                choices=["retry", "rollback_retry", "abort"],
                default="retry",
            )
        if self._confirm_retry():
            error_msg = str(getattr(self, '_last_provider_error', '') or '').lower()
            status = self._extract_http_status_code(error_msg)
            is_4xx = bool(status is not None and 400 <= status < 500)
            if is_4xx:
                return "rollback_retry"
            # In non-interactive flows, avoid infinite retry loops for errors that
            # are not classified as transient/retryable.
            if not self._is_transient_provider_error(RuntimeError(error_msg)):
                return "abort"
            return "retry"
        return "abort"

    def _announce_retryable_failure(self, tool_name: str, raw_result) -> int:
        """If `raw_result` is a structured failure envelope with `retryable=True`,
        surface its `hint` on the live UI so the human can see what the agent
        saw. Also tracks repeat retryable failures of the same (tool, error_code)
        and escalates to an error banner on the third strike.

        Returns the current retryable-failure count for this (tool_name,
        error_code) pair, or 0 if the result is not a retryable failure.
        The caller (loop_body) uses the return value to inject a corrective
        message into the conversation when the count exceeds a threshold,
        breaking retryable-failure loops that pattern-based loop detection
        cannot catch (different tool args each time → different fingerprints).
        """
        if not bool(self.variables.get("reflective_retry_enabled", True)):
            return 0
        envelope = None
        if isinstance(raw_result, dict):
            envelope = raw_result
        elif isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
                if isinstance(parsed, dict):
                    envelope = parsed
            except (ValueError, TypeError):
                envelope = None
        if not envelope:
            return 0
        if envelope.get("ok") is not False:
            return 0
        if not envelope.get("retryable"):
            return 0
        error_code = str(envelope.get("error_code") or "unknown")
        hint = str(envelope.get("hint") or "").strip()
        if not hint:
            return 0

        # Track repeats — `_retryable_failure_counts` is a dict keyed by
        # (tool_name, error_code) -> count. Reset each turn (cleared in
        # `_collect_turn_response`).
        if not hasattr(self, "_retryable_failure_counts"):
            self._retryable_failure_counts = {}
        key = (tool_name, error_code)
        self._retryable_failure_counts[key] = self._retryable_failure_counts.get(key, 0) + 1
        count = self._retryable_failure_counts[key]

        # Expose the latest retryable error_code so the agent loop can feed
        # a synthetic `retryable~<error_code>` fingerprint into loop
        # detection (R8, FM-6). Retryable-failure storms use different
        # args each call, so their normal tool fingerprints never repeat
        # and pattern-based loop detection cannot catch them. A dedicated
        # retryable-fingerprint history lane closes that gap.
        self._last_retryable_error_code = error_code

        if self.ui:
            if count >= 3:
                self.ui.show_error(
                    f"🔁 {tool_name} has hit {error_code} {count}x this turn. "
                    f"Hint stays the same: {hint[:160]}"
                )
            else:
                self.ui.show_info(
                    f"  [retryable {error_code}] {hint[:200]}"
                )

        return count

    # Retry helpers moved to `mu/agent/retry.py`. Static-method
    # forwarders preserve the `Session._is_transient_provider_error`
    # interface used by `_HookAbort` handling and tests.

    @staticmethod
    def _is_transient_provider_error(error: Exception) -> bool:
        from mu.agent.retry import is_transient_provider_error

        return is_transient_provider_error(error)

    @staticmethod
    def _extract_http_status_code(message: str) -> int | None:
        from mu.agent.retry import extract_http_status_code

        return extract_http_status_code(message)

    @staticmethod
    def _is_retryable_http_status(status_code: int) -> bool:
        from mu.agent.retry import _RETRYABLE_HTTP_STATUS

        return status_code in _RETRYABLE_HTTP_STATUS

    # Loop-detection helpers moved to `mu/agent/loop_detection.py`.
    # Static-method forwarders preserve the `Session.<method>`
    # call sites used by the iteration loop and tests.

    @staticmethod
    def _coarse_tool_args(tool_args, tool_name=""):
        from mu.agent.loop_detection import coarse_tool_args

        return coarse_tool_args(tool_args, tool_name)

    @staticmethod
    def _tool_call_fingerprint(tool_name: str, tool_args, *, pattern_only: bool = False) -> str:
        from mu.agent.loop_detection import tool_call_fingerprint

        return tool_call_fingerprint(tool_name, tool_args, pattern_only=pattern_only)

    @staticmethod
    def _track_tool_for_loop_detection(tool_name: str, tool_args) -> bool:
        from mu.agent.loop_detection import track_tool_for_loop_detection

        return track_tool_for_loop_detection(tool_name, tool_args)

    @staticmethod
    def _is_repeated_tool_sequence(
        sequence_history: list[str], repeat_threshold: int = 3
    ) -> bool:
        from mu.agent.loop_detection import is_repeated_tool_sequence

        return is_repeated_tool_sequence(sequence_history, repeat_threshold)

    @staticmethod
    def _is_periodic_sequence(
        sequence_history: list[str],
        *,
        max_period: int = 6,
        min_repeats: int = 2,
    ) -> bool:
        from mu.agent.loop_detection import is_periodic_sequence

        return is_periodic_sequence(
            sequence_history, max_period=max_period, min_repeats=min_repeats
        )

    def _provider_generate_with_retry(
        self,
        *,
        messages,
        system_prompt,
        thinking,
        tools,
    ):
        """Call the provider with exponential-backoff retry on transient
        failures. Body moved to `mu/agent/retry.py:provider_generate_with_retry`."""
        from mu.agent.retry import provider_generate_with_retry

        return provider_generate_with_retry(
            self,
            messages=messages,
            system_prompt=system_prompt,
            thinking=thinking,
            tools=tools,
        )

    def _request_tool_approval(
        self,
        *,
        approval_plan: ApprovalPlan,
        display_args: dict,
        count_info: str,
    ) -> tuple[str, str | None]:
        prompt_text, choices, default = build_approval_prompt(
            approval_plan,
            display_args=display_args,
            count_info=count_info,
        )

        if self.ui and hasattr(self.ui, "request_tool_approval"):
            decision = self.ui.request_tool_approval(
                tool_name=approval_plan.tool_name,
                tool_args=approval_plan.tool_args,
                display_args=display_args,
                count_info=count_info,
                can_approve=approval_plan.can_approve,
                modifications=[mod.to_payload() for mod in approval_plan.modifications],
                preview_error=approval_plan.preview_error,
                error_code=approval_plan.error_code,
                approval_policy=approval_plan.approval_policy,
                prompt_text=prompt_text,
                choices=choices,
                default=default,
            )
            if isinstance(decision, dict):
                return (
                    "y" if decision.get("approved") else "n",
                    decision.get("reason"),
                )
            return decision

        choice = self._prompt_tool_choice(prompt_text, choices, default)
        reason = None
        if choice == "e" and self.ui and hasattr(self.ui, "prompt"):
            reason = self.ui.prompt("Provide an explanation to the model")
        return choice, reason

    def _collect_turn_response(
        self,
        start_index: int,
        *,
        status: str,
        total_in: int,
        total_out: int,
        total_cost: float,
        error: str | None = None,
    ) -> dict:
        # Reset the compaction watermark so a future turn starts fresh.
        self._compaction_watermark = 0
        # Reset the once-per-turn proactive-compaction flag so the next turn's
        # turn-start roll + auto-compaction hook can fire again.
        self._compacted_this_turn = False
        # Reset the reactive-overflow-recovery counter for the next turn.
        self._overflow_recoveries_this_turn = 0
        # Reset per-turn retry counters so the next turn isn't penalised for
        # failures the previous turn already escalated on.
        if hasattr(self, "_retryable_failure_counts"):
            self._retryable_failure_counts.clear()
        history_delta = self.session_manager.history[start_index:]
        assistant_messages = []
        assistant_text_parts = []
        tool_calls = []
        tool_results = []

        for message in history_delta:
            role = message.get("role")
            if role == "assistant":
                assistant_messages.append(message)
            for part in message.get("parts", []):
                part_type = part.get("type")
                if role == "assistant" and part_type == "text":
                    assistant_text_parts.append(part.get("text", ""))
                elif part_type == "tool_call":
                    tool_calls.append(
                        {
                            "tool_name": part.get("tool_name"),
                            "tool_args": part.get("tool_args", {}),
                        }
                    )
                elif part_type == "tool_result":
                    tool_results.append(
                        {
                            "tool_name": part.get("tool_name"),
                            "tool_result": part.get("tool_result"),
                        }
                    )

        # Model-managed memory promotion is automatic and non-blocking. The
        # model decides what deserves save_memory; the harness commits the
        # eligible, non-secret records and surfaces a compact receipt instead
        # of interrupting the normal conversation with approval prompts.
        durable_writes = list(getattr(self, "_turn_durable_writes", []) or [])
        if self.variables.get("durable_memory_enabled", True):
            try:
                service = self.get_durable_memory_service()
                promoted = service.capture_task_entries(self)
                seen_write_ids = {item.id for item in durable_writes}
                durable_writes.extend(
                    item for item in promoted if item.id not in seen_write_ids
                )
                self._last_durable_writes = [item.to_dict() for item in durable_writes]
                if durable_writes:
                    # Phase-6 r22 F2: this capture still runs inside the
                    # armed turn window — plain save_history would LWW a
                    # concurrent surface write. Route through turn CAS.
                    self.session_manager.save_history_turn(self.folder_context)
                    if (
                        self.ui
                        and self.variables.get("durable_memory_show_receipts", True)
                    ):
                        self.ui.show_info(
                            f"Memory · stored {len(durable_writes)} cross-session "
                            f"record{'s' if len(durable_writes) != 1 else ''}"
                        )
            except Exception:
                logger.debug("durable memory capture failed", exc_info=True)

        response = {
            "ok": error is None and status not in {"error"},
            "status": status,
            "error": error,
            "session_name": self.session_manager.current_session_name,
            "assistant_text": "\n\n".join(
                [text for text in assistant_text_parts if str(text).strip()]
            ),
            "assistant_messages": assistant_messages,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "history_delta": history_delta,
            "tokens": {
                "input": total_in,
                "output": total_out,
                "total": total_in + total_out,
                "estimated_cost": total_cost,
            },
            "session_totals": dict(self.session_manager.token_counts),
            "memory": {
                "recall_receipt_id": str(
                    getattr(
                        getattr(self, "_last_durable_recall_receipt", None),
                        "id",
                        "",
                    )
                    or ""
                ),
                "recalled": len(
                    getattr(
                        getattr(self, "_last_durable_recall_receipt", None),
                        "included",
                        [],
                    )
                    or []
                ),
                "stored": len(durable_writes),
                "writes": [item.to_dict() for item in durable_writes],
            },
        }
        # Stash a compact turn summary for the run tracer. The `send_message`
        # finally block reads this and emits the `turn_end` line + flushes the
        # trace file. Kept small — full content lives in session.json.
        try:
            self._trace_turn_summary = {
                "status": status,
                "total_in": int(total_in or 0),
                "total_out": int(total_out or 0),
                "total_cost": float(total_cost or 0.0),
                "tool_calls": len(tool_calls),
                "tool_results": len(tool_results),
                "error": error,
                "session_totals": dict(self.session_manager.token_counts),
                "memory": dict(response.get("memory", {})),
            }
        except Exception:  # noqa: BLE001 — telemetry must never break a turn
            pass
        return response

    def send_message(self, text, *, origin="user"):
        """Body moved to `mu/agent/loop_body.py:run_turn`.

        Wraps the turn in a `finally` that strips the pinned
        `session_goal`. Rationale: a pinned goal grounds the model
        for the duration of a SINGLE multi-iteration turn (where it
        survives the L2 conversation-summary compaction); once that
        turn finishes, the goal would otherwise bias every
        subsequent unrelated request. End-of-turn clearing keeps
        the goal load-bearing exactly where it should be — within
        the turn — and out of the way afterwards.
        """
        from mu.agent.loop_body import run_turn

        # Phase-6 F47: arm turn-scoped CAS for the whole turn. Every
        # save_history_turn during the run verifies the on-disk revision
        # against the turn baseline; a concurrent surface write (GUI /
        # mobile / CLI) is detected and merged instead of clobbered.
        # Disarmed in the finally below — no path leaks the armed state.
        try:
            self.session_manager.begin_turn_cas()
        except Exception:
            logger.debug("begin_turn_cas failed", exc_info=True)
        # Phase-6 r21 F4: bind this Session as the manager's runtime
        # owner so a mid-turn winner reload (save_history_turn conflict
        # path) can re-sync Session aliases to the adopted objects.
        try:
            self.session_manager._bind_runtime_owner(self)
        except Exception:
            logger.debug("bind_runtime_owner failed", exc_info=True)
        coordinator = getattr(self, "thread_coordinator", None)
        turn_id = f"turn-{uuid.uuid4().hex}"
        self._thread_turn_id = turn_id
        self._thread_run_origin = str(origin or "user")[:40]
        lease_stop = threading.Event()
        lease_thread = None
        lease_acquired = False
        turn_started = False
        try:
            if coordinator is not None:
                if not coordinator.heartbeat(
                    self.thread_meta.thread_id,
                    self._thread_runtime_id,
                    ttl=120.0,
                ):
                    raise RuntimeError(
                        "This thread already has an active turn in another runtime."
                    )
                lease_acquired = True
                coordinator.set_status(
                    self.thread_meta.thread_id,
                    "running",
                    goal=str(text or "")[:500],
                    run_origin=self._thread_run_origin,
                    runtime_id=self._thread_runtime_id,
                )

                def _renew_thread_lease():
                    while not lease_stop.wait(30.0):
                        try:
                            if not coordinator.heartbeat(
                                self.thread_meta.thread_id,
                                self._thread_runtime_id,
                                ttl=120.0,
                            ):
                                return
                        except Exception:
                            logger.debug(
                                "thread execution lease renewal failed",
                                exc_info=True,
                            )
                            return

                lease_thread = threading.Thread(
                    target=_renew_thread_lease,
                    name="mucli-thread-lease",
                    daemon=True,
                )
                lease_thread.start()
            turn_started = True
            if self._thread_run_origin == "user":
                # Preserve the historical call shape for embedders/tests that
                # monkeypatch the two-argument loop entry point.
                return run_turn(self, text)
            return run_turn(self, text, origin=self._thread_run_origin)
        finally:
            lease_stop.set()
            if lease_thread is not None:
                lease_thread.join(timeout=1.0)
            # Phase-6 r22 F3: persistence-bearing turn cleanup (goal
            # strip saves) runs while turn CAS is STILL ARMED so its
            # save merges against a concurrent surface write instead of
            # clobbering it after the protective window closed.
            if turn_started:
                self._strip_session_goal_after_turn()
            try:
                self.session_manager.end_turn_cas()
            except Exception:
                logger.debug("end_turn_cas failed", exc_info=True)
            # Lazy LAYER 3B: clear the orchestrator role once no children are
            # active so a session that spawned sub-agents earlier doesn't keep
            # emitting ORCHESTRATOR guidance on unrelated future turns. The
            # role is re-stamped on the next spawn_agent call.
            try:
                if not self._subagent_registry.has_active():
                    if str(self.variables.get("session_role", "") or "") == "parent":
                        self.variables["session_role"] = ""
            except Exception:
                # Defensive: best-effort path must not break the caller.
                logger.debug("Suppressed exception", exc_info=True)
            # Clean up turn-prompt protection now that the turn is over.
            # The turn's starting prompt was protected during the active
            # turn; after the turn, unprotect it if it's not otherwise
            # worthy of long-term protection (e.g. short/slash commands).
            turn_idx = getattr(self, "_current_turn_start_index", None)
            if turn_idx is not None:
                self.session_manager._cleanup_protected(turn_idx)
                self._current_turn_start_index = None
            # Cross-surface phase 2 (G6): if the CLI-side watcher deferred a
            # reload because this turn was executing, apply it now — the turn
            # boundary is the one safe point (own state already saved).
            try:
                sync = getattr(self, "_surface_sync", None)
                if sync is not None:
                    sync.apply_pending()
            except Exception:
                logger.debug("surface sync apply_pending failed", exc_info=True)
            try:
                if coordinator is not None and lease_acquired:
                    coordinator.release_paths(
                        self.thread_meta.thread_id, turn_id=turn_id
                    )
                    coordinator.set_status(
                        self.thread_meta.thread_id,
                        "idle",
                        runtime_id=self._thread_runtime_id,
                    )
                    coordinator.release_execution_lease(
                        self.thread_meta.thread_id, self._thread_runtime_id
                    )
            except Exception:
                logger.debug("thread turn cleanup failed", exc_info=True)
            self._thread_turn_id = None
            # Run tracer: emit the turn_end line and flush+close the trace
            # file. Runs on every exit path (normal completion, hook abort,
            # sub-agent kill, exception) — telemetry must never be lost and
            # must never propagate a failure into the turn.
            try:
                from mu.trace.emitter import get_emitter

                _em = get_emitter(self)
                if _em is not None and not _em._closed:
                    _summary = getattr(self, "_trace_turn_summary", {}) or {}
                    # Efficiency metrics (spec #12): compression, cache rates,
                    # retrieval rate, tool-output share. Folded into turn_end.
                    _eff = {}
                    try:
                        from mu.session.efficiency_metrics import (
                            collect_efficiency_metrics,
                        )

                        _eff = collect_efficiency_metrics(
                            self,
                            tool_calls_this_turn=_summary.get("tool_calls", 0),
                            retrieval_calls_this_turn=int(
                                getattr(self, "_eff_retrievals", 0)
                            ),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    _em.turn_end(
                        {
                            "status": _summary.get("status", "unknown"),
                            "total_in": _summary.get("total_in", 0),
                            "total_out": _summary.get("total_out", 0),
                            "total_cost": _summary.get("total_cost", 0.0),
                            "tool_calls": _summary.get("tool_calls", 0),
                            "tool_results": _summary.get("tool_results", 0),
                            "error": _summary.get("error"),
                            "session_totals": _summary.get("session_totals", {}),
                            "memory": _summary.get("memory", {}),
                            "iters": _em.iter_count,
                            "efficiency": _eff,
                        }
                    )
                    # Round-51 T2: terminal run record on every exit path
                    # (normal completion, max-iterations, exception, stop).
                    # Previously runs were truncated mid-flight with no
                    # run_end record at all — the parser reported status
                    # 'running' forever. Guard is run-level (session attr
                    # keyed by run id) so a rebuilt emitter instance cannot
                    # double-terminate the same run.
                    _run_id = getattr(_em, "run_id", None)
                    if getattr(self, "_run_end_emitted_for", None) == _run_id:
                        _status = None  # already terminated this run
                    else:
                        self._run_end_emitted_for = _run_id
                        _status = str(
                            _summary.get("status", "unknown") or "unknown"
                        )
                    _error = _summary.get("error")
                    _run_status = (
                        "failed" if (_status == "error" or _error) else "completed"
                    )
                    if _status == "max_iterations_reached":
                        # Round-51 T2: the loop's own terminal status is
                        # authoritative — preserve it verbatim so the
                        # parser/trace.html can distinguish "did real work
                        # then ran out of budget" from a clean completion.
                        _run_status = "max_iterations"
                    _reason = (
                        str(_error)[:280]
                        if _error
                        else (_status if _status not in ("unknown", "") else "turn_complete")
                    )
                    _em.run_end(
                        {
                            "status": _run_status,
                            "reason": _reason,
                            "iters": _em.iter_count,
                            "tokens_in": _summary.get("total_in", 0),
                            "tokens_out": _summary.get("total_out", 0),
                            "cost": _summary.get("total_cost", 0.0),
                        }
                    )
            except Exception:  # noqa: BLE001 — telemetry must never break a turn
                pass

    def _session_goal_is_sticky(self) -> bool:
        """Should the pinned session_goal survive the end of this turn?

        True when the user opted in via ``session_goal_sticky`` or the
        active mode is a long-horizon mode (loop / feature) that defaults
        to keeping the goal across turn boundaries. Explicit opt-out
        (``session_goal_sticky=False`` set by the user) is honored even
        in long-horizon modes. Uses ``session_goal_sticky_explicit`` to
        tell a user override apart from the schema default (mirrors
        ``show_thinking_explicit``).
        """
        if bool(self.variables.get("session_goal_sticky_explicit", False)):
            return bool(self.variables.get("session_goal_sticky", False))
        mode = str(self.variables.get("agent_mode", "default") or "default").lower()
        return mode in ("loop", "feature")

    def _strip_session_goal_after_turn(self) -> None:
        """Clear `session_goal` at the end of every agent turn — unless
        the goal is *sticky*, in which case it persists across turns in
        L3 until the user clears it (/goal clear) or sets a new goal.

        Sticky when: ``session_goal_sticky`` is set True, OR the active
        mode is a long-horizon mode (loop / feature) that defaults to
        keeping the goal across turn boundaries. Long-horizon multi-turn
        work needs the goal to survive turn boundaries so the model
        doesn't lose the thread between turns; conversational default
        use strips it so a pinned goal can't bias an unrelated next
        request.

        The variable is the only thing that gets reset — the durable
        `task_memory` audit entry (saved by
        `_ensure_session_goal_persistence`) stays as history. So
        `/memory search` can still surface the original ask, but L3
        no longer renders it after the turn completes (when non-sticky).

        Lifecycle: marks the active goal memory entry (tracked via
        ``self._active_goal_memory_id``) as ``status='done'`` so
        future searches don't resurface it unless explicitly requested
        via ``include_all=True``.

        Safe to call even when no goal was set — it's a no-op then.
        """
        current = str(self.variables.get("session_goal", "") or "").strip()
        if not current:
            return
        if self._session_goal_is_sticky():
            # Keep the goal pinned across turns. Don't mark the memory
            # entry done — it's still the active objective.
            return
        self.variables["session_goal"] = ""
        # Mark the goal memory entry as done (audit trail retained).
        goal_id = getattr(self, "_active_goal_memory_id", None)
        if goal_id is not None:
            entry = self.task_memory.get_entry(goal_id)
            if entry is not None and getattr(entry, "status", "active") == "active":
                self.task_memory.update_status(goal_id, "done")
            self._active_goal_memory_id = None
        # Clear the explicit flag so future modes' defaults apply
        # cleanly on the next turn. (See
        # `show_thinking_explicit` for the same pattern.)
        # Phase-6 r23 F1: this runs INSIDE the armed turn window (the
        # finally calls it before end_turn_cas) — plain save_history
        # would LWW-clobber a concurrent surface write; route through
        # turn CAS so it merges.
        try:
            self.session_manager.save_history_turn(self.folder_context)
        except Exception:
            # Defensive: best-effort path must not break the caller.
            logger.debug("Suppressed exception", exc_info=True)
        # Tell the UI so the user has a clear breadcrumb that the
        # pin auto-released — handy when /goal status surprises them.
        ui = getattr(self, "ui", None)
        if ui is not None and hasattr(ui, "show_info"):
            try:
                ui.show_info(
                    f"🎯 Session goal cleared (was: {current!r}). "
                    "Use /goal <text> to pin a new one."
                )
            except Exception:
                # Defensive: best-effort path must not break the caller.
                logger.debug("Suppressed exception", exc_info=True)

    def shutdown(self) -> None:
        """Release resources held for the lifetime of the session.

        Cancels any still-running async sub-agents and reaps background
        bash tasks. Called from the REPL's top-level ``finally`` on exit
        so no child threads or subprocesses outlive the session. Safe to
        call more than once.
        """
        try:
            self._subagent_registry.shutdown()
        except Exception:
            logger.debug("subagent registry shutdown failed", exc_info=True)
        try:
            self.background_tasks.shutdown()
        except Exception:
            logger.debug("background task shutdown failed", exc_info=True)
