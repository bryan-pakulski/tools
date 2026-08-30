"""History summarization and token-budget rolling.

`HistoryMixin` factors the rolling-summary algorithm out of `SessionManager`.
The methods operate on three instance attributes the host class is
expected to provide:

  * `history`              — list[dict] of message dicts
  * `summary_anchor`       — int index; everything < anchor is summarized
  * `conversation_summary` — str rolling summary

The mixin is a plain class with no `__init__`; consumers either inherit
or compose. `SessionManager` inherits.

Algorithm — `roll_history_summary_to_token_budget`:
  1. Estimate runtime tokens for messages[anchor:]
  2. If under budget, return False
  3. Try `roll_history_summary(keep_recent=...)` — moves a block of older
     messages into `conversation_summary`, advancing `anchor`.
  4. If no rolling possible, call `_degrade_oldest_runtime_payload` —
     truncates the oldest oversized text or tool_result part to a fixed
     character cap, returning True if it changed anything.
  5. Repeat up to `max_passes` times.

Token estimate: `len(text) / 4` per part field (chars→tokens approximation).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from providers.base import LLMProvider, Message, MessagePart
from utils.logger import logger

from .helpers import _shorten_tool_args


# Canonical section order for structured (###-headed) conversation summaries.
# Used by `_merge_structured_summary` (append-by-section) and
# `_clip_conversation_summary` (section-aware trimming). The order also
# encodes trim priority: Task and Open items are the most load-bearing and
# are trimmed last; Progress is the bulkiest/most-summarizable and is
# trimmed first.
_SUMMARY_CANONICAL_ORDER = [
    "Task",
    "Progress",
    "Key decisions",
    "Current state",
    "Open items",
]

# Sections trimmed first (in this order) when the summary exceeds its char
# budget. Task and Open items are intentionally excluded — they are the
# shortest, most load-bearing sections and should survive as long as
# possible. See R2 in documentation/harness-investigation.md (FM-3).
_SUMMARY_TRIM_ORDER = ["Progress", "Key decisions", "Current state"]

# Portion-based compaction: the max rendered-characters of history one
# `roll_history_summary` LLM call summarizes at a time (~6k tokens). Mirrors
# the `_OVERFLOW_CHUNK_CHARS = 12_000` chunk size in `mu/session/messages.py`;
# doubled here because a conversation segment mixes small text turns with
# tool results and benefits from a little more per-call context. The budget
# loop (`roll_history_summary_to_token_budget`) calls `roll_history_summary`
# repeatedly, so the whole history is still summarized — just one bounded
# portion per LLM call instead of all at once (Claude Code "context collapse").
# ``None`` disables segmentation (legacy whole-block behavior).
_COMPACTION_SEGMENT_CHARS = 24_000

# How much of each tool result the LLM summarizer is allowed to see when
# rendering history for a compaction summary (`_render_entries_for_llm`).
# Was 300 — so little that the summarizer couldn't choose what mattered in
# a big tool result. 4000 chars (~1k tokens) is enough for the model to
# pick out the load-bearing paths/identifiers/findings, and the full
# original stays recallable via the ``[cache:KEY]`` tag (tool-result sidecar),
# so this is not lossy — it's "summarizer sees enough to decide".
_COMPACT_RENDER_TOOL_RESULT_CHARS = 4000

# Above this size, `_summarize_payload_via_llm` routes an oversized payload
# through the chunked summarizer (`_chunk_summarize_text`) so the *whole*
# payload is summarized across bounded chunks instead of just its head.
# Below it, the whole payload goes to one generate() call (no first-8k
# slice) — ~32k chars (~8k tokens) is a comfortable single summarization
# call, so only genuinely huge tool results (100k+) pay for chunking.
_PAYLOAD_SUMMARIZE_CHAR_THRESHOLD = 32_000


def _parse_summary_sections(text: str) -> Dict[str, str]:
    """Split a structured (``###``-headed) summary into ``{header: body}``.

    Module-level helper shared by `_merge_structured_summary` (append by
    section) and `_clip_conversation_summary` (section-aware trim) so the
    two paths agree on what counts as a section.
    """
    sections: Dict[str, str] = {}
    current_header: Optional[str] = None
    current_lines: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("###"):
            if current_header is not None:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = line.strip().lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_header is not None:
        sections[current_header] = "\n".join(current_lines).strip()
    return sections


def _reassemble_summary_sections(sections: Dict[str, str]) -> str:
    """Reassemble a ``{header: body}`` map into a single ``###``-headed
    summary in canonical section order, dropping empty sections."""
    out_lines: List[str] = []
    for header in _SUMMARY_CANONICAL_ORDER:
        body = sections.get(header, "").strip()
        if body:
            out_lines.append(f"### {header}")
            out_lines.append(body)
            out_lines.append("")
    # Append any non-canonical sections that exist (future-proofing).
    for header, body in sections.items():
        if header in _SUMMARY_CANONICAL_ORDER:
            continue
        body = body.strip()
        if body:
            out_lines.append(f"### {header}")
            out_lines.append(body)
            out_lines.append("")
    return "\n".join(out_lines).strip()


class HistoryMixin:
    """History-summarization methods. Host must supply `history`,
    `summary_anchor`, and `conversation_summary` as instance attributes.
    """

    # --------------------------------------------------------- summarization

    def _summarize_history_batch(self, entries: List[Dict[str, Any]]) -> str:
        lines = [self._summarize_history_message(entry) for entry in entries]
        return "\n".join(line for line in lines if line)

    def _summarize_history_message(self, entry: Dict[str, Any]) -> str:
        role = str(entry.get("role", "message"))
        parts: List[str] = []
        for part in entry.get("parts", []):
            part_type = part.get("type")
            if part_type == "text":
                text = str(part.get("text", "")).strip().replace("\n", " ")
                if text:
                    parts.append(text[:140])
            elif part_type == "tool_call":
                parts.append(
                    f"tool_call:{part.get('tool_name')} "
                    f"args={_shorten_tool_args(part.get('tool_args', {}))}"
                )
            elif part_type == "tool_result":
                result = str(part.get("tool_result", "")).strip().replace("\n", " ")
                if len(result) > 140:
                    result = f"{result[:137]}..."
                if result:
                    parts.append(
                        f"tool_result:{part.get('tool_name', 'tool')} => {result}"
                    )
            elif part_type == "file":
                file_ref = part.get("file_ref", {})
                parts.append(
                    f"file:{file_ref.get('display_name') or file_ref.get('uri') or 'unknown'}"
                )

        if not parts:
            return f"- {role}: [no serializable content]"
        return f"- {role}: " + " | ".join(parts)

    # ------------------------------------------------- LLM summary generation

    _LLM_SUMMARY_SYSTEM_PROMPT = (
        "You are a conversation summarizer for an AI coding agent. "
        "Summarize the following conversation segment so the agent can "
        "continue its work with the essential context preserved. You "
        "decide what matters — prioritize information the agent needs to "
        "keep working and drop what is no longer relevant.\n\n"
        "Suggested sections (use whichever apply; you may add, omit, or "
        "reorganize them to fit what actually matters in this segment). "
        "Start each section you use with ###:\n\n"
        "### Task\nThe user's original request or goal. Quote it if short, "
        "paraphrase if long. Never omit.\n\n"
        "### Progress\nWhat has been done so far. List concrete actions, "
        "file changes, tool calls, and their outcomes.\n\n"
        "### Key decisions\nArchitectural or design choices made, with "
        "rationale. Include any rejected approaches and why.\n\n"
        "### Current state\nFiles modified, tests run and their results, "
        "current errors or blockers. Include file paths and function "
        "names verbatim.\n\n"
        "### Open items\nWhat still needs to be done. Be specific — "
        "list the next actionable steps.\n\n"
        "Rules:\n"
        "- Preserve file paths, function names, error messages, and "
        "identifiers VERBATIM.\n"
        "- When a tool result is summarized, KEEP its [cache:KEY] tag so "
        "the full original result can be recalled later — do not drop it.\n"
        "- Tool results appear as compact `[action: ...]` records preserving "
        "the tool, args, ok/fail status, modified files, error_code, and "
        "cache_key. Preserve the modified_files and any unresolved error_code "
        "from these records; the full raw output is recoverable via recall "
        "and does NOT need to be re-summarized.\n"
        "- Do NOT add commentary outside the sections you choose.\n"
        "- Be concise but complete. Target 200-600 words.\n"
    )

    def _generate_llm_summary(
        self,
        provider: Optional[LLMProvider],
        entries: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Generate a structured LLM summary of conversation entries.

        Calls the provider with a summarization prompt and the
        conversation segment rendered as readable text. Returns the
        model's structured summary text, or None on any failure
        (caller falls back to _summarize_history_batch).

        Cost: one non-streaming provider call, no tools, ~1000 output
        tokens. The investment is worth it — mechanical truncation
        loses semantic meaning and causes agents to re-read files
        they already explored (the compaction-loop failure mode).
        """
        if provider is None or not entries:
            return None

        try:
            # Render the conversation segment as readable text for the
            # summarizer. Use _render_entries_for_llm with a generous
            # per-tool-result budget + [cache:KEY] tags so the model can
            # choose what matters and mark recallable results.
            rendered = self._render_entries_for_llm(entries)
            if not rendered.strip():
                return None

            # Optional compact-focus steering (Claude Code `/compact <focus>`
            # style): when set, the summary emphasizes the given focus.
            focus = ""
            focus_val = getattr(self, "_compact_focus", None)
            if focus_val:
                focus = f"Focus for this summary: {focus_val}\n\n"

            user_prompt = (
                f"{focus}Summarize this conversation segment:\n\n"
                f"{rendered}"
            )

            messages = [
                Message(
                    role="user",
                    parts=[MessagePart(type="text", text=user_prompt)],
                ),
            ]

            response = provider.generate(
                messages=messages,
                system_prompt=self._LLM_SUMMARY_SYSTEM_PROMPT,
                thinking=False,
                tools=None,
            )

            summary_text = str(response.text or "").strip()
            if not summary_text:
                return None

            # The summary scaffold is now suggested (not mandated), so a
            # freeform summary with no ### headers is valid — the model
            # chose its own structure. `_merge_structured_summary` falls
            # back to append when no ### sections are present, and
            # `_clip_conversation_summary` bounds growth. Only empty text
            # is rejected (handled above).
            return summary_text

        except Exception as exc:
            logger.warning(
                "LLM summary generation failed: %s — falling back to "
                "mechanical summarization.",
                exc,
            )
            return None

    def _render_entries_for_llm(
        self, entries: List[Dict[str, Any]]
    ) -> str:
        """Render conversation entries as readable text for the LLM
        summarizer.

        Unlike _summarize_history_message (which truncates to 140
        chars), this preserves much more content — up to 500 chars per
        text part and ``_COMPACT_RENDER_TOOL_RESULT_CHARS`` (4000) per
        tool result — so the LLM has enough context to choose what
        matters. Tool results also carry their ``[cache:KEY]`` tag (the
        full original is recallable via ``recall`` / ``lookup_by_locator``
        in ``mu/session/tool_cache.py``), so the summarizer can mark a
        compacted result as recallable instead of losing it to truncation.
        """
        lines: List[str] = []
        for entry in entries:
            role = str(entry.get("role", "message"))
            parts_text: List[str] = []
            for part in entry.get("parts", []):
                part_type = part.get("type")
                if part_type == "text":
                    text = str(part.get("text", "")).strip().replace("\n", " ")
                    if text:
                        parts_text.append(text[:500])
                elif part_type == "tool_call":
                    parts_text.append(
                        f"tool_call:{part.get('tool_name')} "
                        f"args={_shorten_tool_args(part.get('tool_args', {}))}"
                    )
                elif part_type == "tool_result":
                    # Action record (spec #4/#5): when the part carries a
                    # cache_key or a structured envelope, render a compact
                    # one-line record preserving the decision, outcome,
                    # modified_files, error_code, and cache_key — far less
                    # for the summarizer to process than a 4000-char clip,
                    # and more faithful (no truncation of the middle). The
                    # full raw is recoverable via recall(cache_key).
                    from mu.session.action_record import (
                        is_action_record_eligible,
                        render_action_record,
                    )

                    if is_action_record_eligible(part):
                        parts_text.append(render_action_record(part))
                    else:
                        result = str(part.get("tool_result", "")).strip().replace("\n", " ")
                        if len(result) > _COMPACT_RENDER_TOOL_RESULT_CHARS:
                            result = result[: _COMPACT_RENDER_TOOL_RESULT_CHARS - 3] + "..."
                        if result:
                            # cache_key tag: the full tool result lives in the
                            # tool-result sidecar cache and can be recalled by key,
                            # so the summarizer need not preserve every byte — it
                            # just marks the result as recallable.
                            cache_key = part.get("cache_key")
                            cache_tag = f"[cache:{cache_key}] " if cache_key else ""
                            parts_text.append(
                                f"tool_result:{part.get('tool_name', 'tool')} => "
                                f"{cache_tag}{result}"
                            )
                elif part_type == "file":
                    file_ref = part.get("file_ref", {})
                    parts_text.append(
                        f"file:{file_ref.get('display_name') or file_ref.get('uri') or 'unknown'}"
                    )
            if parts_text:
                lines.append(f"- {role}: " + " | ".join(parts_text))
        return "\n".join(lines)

    def _merge_structured_summary(self, new_summary: str) -> None:
        """Merge a new LLM-generated structured summary into the existing
        ``conversation_summary`` by section.

        Both the existing and new summaries use ``###`` header sections
        (Task, Progress, Key decisions, Current state, Open items).
        Instead of blind-appending (which creates duplicate headers and
        grows unbounded), this method:

          * Parses the existing summary into ``{section_title: content}``.
          * Parses the new summary the same way.
          * For each section: appends new content after the existing
            content, separated by a blank line.
          * Sections in the new summary that don't exist in the old one
            are added outright.
          * Sections in the old summary not touched by the new one are
            preserved unchanged.

        After merge, ``_clip_conversation_summary`` is called with the
        larger 12000-char budget (set in Task 3) to prevent unbounded
        growth while preserving much more context than the old 4000 cap.
        """
        if not new_summary:
            return

        # If existing summary doesn't have ### sections, fall back to
        # blind append — it's a legacy mechanical summary.
        if self.conversation_summary and "###" not in self.conversation_summary:
            self.conversation_summary = (
                f"{self.conversation_summary}\n\n{new_summary}".strip()
            )
            self._clip_conversation_summary()
            return

        existing = _parse_summary_sections(self.conversation_summary or "")
        incoming = _parse_summary_sections(new_summary)

        # Merge: append new content to existing sections, add new sections.
        for header, body in incoming.items():
            if not body or body == "None":
                continue
            if header in existing:
                existing_body = existing[header]
                if existing_body and existing_body != "None":
                    existing[header] = f"{existing_body}\n\n{body}"
                else:
                    existing[header] = body
            else:
                existing[header] = body

        self.conversation_summary = _reassemble_summary_sections(existing)
        self._clip_conversation_summary()

    def _clip_conversation_summary(self, limit: int = 24_000) -> None:
        """Trim the rolling summary to ``limit`` chars, section-aware.

        Old behavior: keep the LAST ``limit`` chars. Because the canonical
        section order is ``Task → Progress → … → Open items``, keep-newest
        clipped from the FRONT — destroying the **Task** section (the
        original user request) first, the single most load-bearing section.

        New behavior (R2 / FM-3): parse the summary into ``###`` sections
        and trim in priority order ``Progress → Key decisions → Current
        state`` (the bulkiest/most-summarizable first), preserving **Task**
        and **Open items** unless the summary still exceeds the budget with
        only those two left. Trimming takes from the FRONT of a section's
        body (oldest progress first) so the newest content in each section
        survives. Falls back to keep-newest for legacy/unstructured
        summaries that have no ``###`` headers.
        """
        if len(self.conversation_summary) <= limit:
            return

        sections = _parse_summary_sections(self.conversation_summary)
        if not sections:
            # Legacy/unstructured summary — keep-newest fallback. Subtract
            # the marker overhead so the final summary respects the limit
            # instead of exceeding it by the marker length.
            marker = f"[conversation_summary_truncated_to_last_{limit}_chars]"
            keep = max(0, limit - len(marker) - 1)
            clipped = self.conversation_summary[-keep:].lstrip() if keep else ""
            newline_index = clipped.find("\n")
            if newline_index > 0:
                clipped = clipped[newline_index + 1 :]
            self.conversation_summary = f"{marker}\n{clipped}".strip()
            return

        # Trim sections in priority order until under budget. Trim from the
        # FRONT of a section's body (oldest content first) so the newest
        # content survives. We re-measure after every cut because the
        # ``[..._trimmed]`` marker and reassembly change the byte count —
        # a single-pass ``excess`` calculation does not converge.
        for section in _SUMMARY_TRIM_ORDER:
            while len(self.conversation_summary) > limit and sections.get(section):
                body = sections[section]
                excess = len(self.conversation_summary) - limit
                # Cut a bit more than `excess` to absorb the trim marker and
                # reassembly overhead so the loop converges in few passes.
                cut = excess + 64
                if cut >= len(body):
                    sections[section] = ""
                else:
                    trimmed = body[cut:].lstrip()
                    sections[section] = (
                        f"[{section}_trimmed_to_fit_budget]\n{trimmed}"
                        if trimmed
                        else ""
                    )
                self.conversation_summary = _reassemble_summary_sections(sections)

        if len(self.conversation_summary) <= limit:
            return

        # Still over budget — Task/Open items alone (plus any non-canonical
        # sections) are too big. Last resort: keep-newest across the whole
        # reassembled summary so we at least stay within the budget.
        clipped = self.conversation_summary[-limit:].lstrip()
        newline_index = clipped.find("\n")
        if newline_index > 0:
            clipped = clipped[newline_index + 1 :]
        self.conversation_summary = (
            f"[conversation_summary_truncated_to_last_{limit}_chars]\n{clipped}"
        ).strip()

    # ----------------------------------------------------- token estimation

    def _active_model(self) -> str:
        """The model whose tokenizer should drive history-token estimates.

        Reads `provider_config['model']` populated by Session at startup
        and refreshed on `/model` / `/provider`. Empty string is fine —
        the estimator falls back to a general-purpose encoder."""
        cfg = getattr(self, "provider_config", None) or {}
        return str(cfg.get("model") or "")

    def _estimate_tokens_from_text(self, text: Any) -> int:
        # Delegate to the shared tiktoken-backed estimator. Driving
        # every token-budget decision through the same function means
        # /memory layer counts, the splash banner, and the compactor
        # trim trigger all agree.
        from utils.token_estimator import estimate_tokens

        return estimate_tokens(text, self._active_model())

    def _estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        role = str(message.get("role", "") or "")
        total = 3 + self._estimate_tokens_from_text(role)
        for part in message.get("parts", []):
            part_type = str(part.get("type", "") or "")
            total += self._estimate_tokens_from_text(part_type)
            if part_type == "text":
                total += self._estimate_tokens_from_text(part.get("text", ""))
            elif part_type == "tool_call":
                total += self._estimate_tokens_from_text(part.get("tool_name", ""))
                total += self._estimate_tokens_from_text(
                    json.dumps(part.get("tool_args", {}), default=str)
                )
            elif part_type == "tool_result":
                total += self._estimate_tokens_from_text(part.get("tool_name", ""))
                total += self._estimate_tokens_from_text(
                    json.dumps(part.get("tool_result", ""), default=str)
                )
            elif part_type == "file":
                file_ref = part.get("file_ref", {}) or {}
                total += self._estimate_tokens_from_text(
                    file_ref.get("display_name") or file_ref.get("uri") or ""
                )
        return total

    def estimate_runtime_history_tokens(
        self, start_index: Optional[int] = None
    ) -> int:
        start = (
            self.summary_anchor if start_index is None else max(0, int(start_index))
        )
        total = sum(
            self._estimate_message_tokens(message) for message in self.history[start:]
        )
        # prepare_runtime_history() re-injects protected messages below the
        # anchor (plus a preserved-context marker) into every provider
        # request. Include them here so budget/compaction decisions see the
        # TRUE request size — otherwise compaction can declare the history
        # under budget while up to _PROTECTED_CAP verbatim messages ride
        # along uncounted.
        protected = getattr(self, "protected_indices", set())
        below = [idx for idx in protected if idx < start]
        if below:
            total += sum(
                self._estimate_message_tokens(self.history[idx])
                for idx in below
                if idx < len(self.history)
            )
            total += 60  # preserved-context marker envelope
        return total

    # ------------------------------------------------------ rolling summary

    def roll_history_summary(
        self,
        keep_recent: int,
        provider: Optional[LLMProvider] = None,
        *,
        max_segment_chars: Optional[int] = None,
    ) -> bool:
        keep_recent = max(1, int(keep_recent or 1))
        if self.summary_anchor > len(self.history):
            self.summary_anchor = 0
        unsummarized_count = len(self.history) - self.summary_anchor
        if unsummarized_count <= keep_recent:
            return False

        target_anchor = len(self.history) - keep_recent
        # Advance target to the next 'user' boundary so we don't split a
        # mid-turn assistant/tool group.
        for idx in range(target_anchor, len(self.history)):
            if self.history[idx].get("role") == "user":
                target_anchor = idx
                break

        # R3 / FM-8: never advance the anchor past the oldest protected
        # tool-result of the active turn. This keeps the last K tool
        # results verbatim in the unsummarized tail even under emergency
        # compaction with a tiny keep_recent. Budget is instead reclaimed
        # by `_degrade_oldest_runtime_payload` degrading OLDER content.
        floor = getattr(self, "_tool_result_floor", 0) or 0
        if floor > 0:
            turn_start = getattr(self, "_active_turn_start_index", None)
            floor_indices = self.tool_result_floor_indices(turn_start, int(floor))
            if floor_indices:
                oldest_floor = min(floor_indices)
                if target_anchor > oldest_floor:
                    target_anchor = oldest_floor

        if target_anchor <= self.summary_anchor:
            return False

        # Portion-based compaction (Claude Code "context collapse" style):
        # when ``max_segment_chars`` is set, summarize only the OLDEST
        # bounded segment in this call rather than the whole pre-target
        # block. The budget loop (`roll_history_summary_to_token_budget`)
        # calls us repeatedly, advancing segment-by-segment, so no single
        # LLM summarization call ever ingests the entire history — each
        # call is small, cheap, and lets the model focus on one portion.
        # ``None`` (the default) reproduces the legacy whole-block behavior
        # so existing call sites are unaffected.
        end = target_anchor
        if max_segment_chars and max_segment_chars > 0:
            seg_end = self.summary_anchor
            acc = 0
            while seg_end < target_anchor:
                msg_len = len(
                    self._render_entries_for_llm([self.history[seg_end]])
                )
                # Always include at least one entry (a single oversized
                # message becomes a one-entry segment; truly huge payloads
                # are reclaimed by `_degrade_oldest_runtime_payload`).
                if seg_end > self.summary_anchor and acc + msg_len > max_segment_chars:
                    break
                acc += msg_len
                seg_end += 1
            end = min(seg_end, target_anchor)
            if end <= self.summary_anchor:
                return False

        # Exclude protected messages from summarization — they stay
        # verbatim in L5 even after the anchor advances past them.
        # This preserves important context (initial user request, key
        # decisions) through compaction without losing recent history.
        protected = getattr(self, "protected_indices", set())
        entries_to_summarize = [
            msg
            for idx, msg in enumerate(self.history[self.summary_anchor : end])
            if (self.summary_anchor + idx) not in protected
        ]

        # LLM-generated structured summary (Claude Code / Pi style).
        # Falls back to mechanical truncation on any failure.
        summary_batch = self._generate_llm_summary(provider, entries_to_summarize)
        if summary_batch is None:
            summary_batch = self._summarize_history_batch(entries_to_summarize)
        else:
            # Verbatim tail: even a good LLM summary can drop exact
            # identifiers (file paths, error strings, command text). Append
            # the bounded mechanical lines so the compacted L2 keeps
            # searchable original tokens; _clip_conversation_summary bounds
            # growth.
            mechanical = self._summarize_history_batch(entries_to_summarize)
            if mechanical:
                summary_batch = f"{summary_batch}\n\n[verbatim segment]\n{mechanical}"

        # Record which summarizer path produced this batch, so the run tracer
        # can flag the catastrophically-lossy mechanical fallback (140-char
        # truncation per part) — a silent long-horizon state-loss mode.
        if "### Task" in (summary_batch or "") or "### Progress" in (summary_batch or ""):
            self._last_summary_mode = "llm"
        elif summary_batch:
            self._last_summary_mode = "mechanical"
        else:
            self._last_summary_mode = "none"

        if not summary_batch:
            self.summary_anchor = end
            return True

        # Merge into existing summary. When the summary is LLM-generated
        # it already has ### Task / ### Progress / ### Key decisions
        # sections; use _merge_structured_summary to merge by section
        # instead of blind-append. For mechanical summaries, fall back
        # to the original header-append behavior.
        if "### Task" in summary_batch or "### Progress" in summary_batch:
            self._merge_structured_summary(summary_batch)
        else:
            header = (
                f"### Summarized conversation through message {end}\n"
                if not self.conversation_summary
                else f"\n### Summarized conversation through message {end}\n"
            )
            self.conversation_summary = (
                f"{self.conversation_summary}{header}{summary_batch}".strip()
            )
            self._clip_conversation_summary()
        self.summary_anchor = end
        return True

    def roll_history_summary_to_token_budget(
        self,
        token_budget: int,
        *,
        keep_recent: int = 12,
        max_passes: int = 8,
        provider: Optional[LLMProvider] = None,
        max_segment_chars: Optional[int] = _COMPACTION_SEGMENT_CHARS,
    ) -> bool:
        token_budget = max(1, int(token_budget or 1))
        # Run-tracer instrumentation: record one compaction event per call so
        # the trace can correlate compactions with context growth / drift. The
        # `kind` (turn_start | auto_hook | emergency_preflight) and iteration
        # are stashed by the caller via _pending_compaction_kind/iter. The log
        # is drained by the trace emitter at the post-response seam.
        _before_len = len(self.history)
        _before_anchor = self.summary_anchor
        try:
            _before_tokens = self.estimate_runtime_history_tokens()
        except Exception:  # noqa: BLE001
            _before_tokens = 0
        changed = False
        for _ in range(max(1, int(max_passes or 1))):
            if self.estimate_runtime_history_tokens() <= token_budget:
                break
            if self.roll_history_summary(
                keep_recent=keep_recent,
                provider=provider,
                max_segment_chars=max_segment_chars,
            ):
                changed = True
                continue
            if self._degrade_oldest_runtime_payload(provider=provider):
                changed = True
                continue
            break
        if changed:
            try:
                try:
                    _after_tokens = self.estimate_runtime_history_tokens()
                except Exception:  # noqa: BLE001
                    _after_tokens = 0
                log = getattr(self, "_compaction_log", None)
                if log is None:
                    log = []
                    self._compaction_log = log
                log.append(
                    {
                        "kind": getattr(self, "_pending_compaction_kind", "auto"),
                        "iter": getattr(self, "_pending_compaction_iter", 0),
                        "tokens_before": int(_before_tokens),
                        "tokens_after": int(_after_tokens),
                        "tokens_saved": int(_before_tokens - _after_tokens),
                        "msgs_before": _before_len,
                        "msgs_after": len(self.history),
                        "anchor_before": _before_anchor,
                        "anchor_after": self.summary_anchor,
                        "anchor_delta": self.summary_anchor - _before_anchor,
                        "summarizer": getattr(self, "_last_summary_mode", "unknown"),
                        "keep_recent": keep_recent,
                        "budget": token_budget,
                    }
                )
            except Exception:  # noqa: BLE001 — tracer must never break compaction
                pass
        return changed

    def _degrade_oldest_runtime_payload(
        self,
        max_chars: int = 16_000,
        provider: Optional[LLMProvider] = None,
    ) -> bool:
        """Fallback budget guard: summarize or clip the oldest oversized
        unsummarized part. Returns True if a change was made.

        When a provider is available, calls provider.generate to produce
        an LLM summary of the oversized payload instead of destructively
        truncating it. This preserves semantic meaning — the model can
        still understand what was in the payload even after budget
        pressure.

        When provider is None or the LLM call fails, falls back to
        expanded mechanical truncation (16000 chars, was 8000).
        """
        if self.summary_anchor > len(self.history):
            self.summary_anchor = 0
        # R3 / FM-8: never degrade a protected active-turn tool result.
        floor = getattr(self, "_tool_result_floor", 0) or 0
        floor_indices = (
            self.tool_result_floor_indices(
                getattr(self, "_active_turn_start_index", None), int(floor)
            )
            if floor > 0
            else set()
        )
        for msg_idx, message in enumerate(self.history[self.summary_anchor :], start=self.summary_anchor):
            if msg_idx in floor_indices:
                continue
            parts = message.get("parts", []) or []
            for part in parts:
                p_type = part.get("type")
                if p_type == "text":
                    value = str(part.get("text", "") or "")
                    if len(value) > max_chars:
                        # Try LLM summarization first.
                        if provider is not None:
                            try:
                                summary = self._summarize_payload_via_llm(
                                    provider, value
                                )
                                if summary and len(summary) < len(value):
                                    part["text"] = summary
                                    return True
                            except Exception as exc:
                                logger.warning(
                                    "LLM payload summarization failed: %s "
                                    "— falling back to mechanical truncation.",
                                    exc,
                                )
                        # Mechanical fallback: expanded from 8000 to 16000.
                        part["text"] = (
                            value[:max_chars].rstrip()
                            + f"\n[truncated_to_{max_chars}_chars_for_context_budget]"
                        )
                        return True
                elif p_type == "tool_result":
                    raw = part.get("tool_result", "")
                    serialized = (
                        json.dumps(raw, default=str)
                        if not isinstance(raw, str)
                        else raw
                    )
                    if len(serialized) > max_chars:
                        # Try LLM summarization first.
                        if provider is not None:
                            try:
                                summary = self._summarize_payload_via_llm(
                                    provider, serialized
                                )
                                if summary and len(summary) < len(serialized):
                                    part["tool_result"] = summary
                                    return True
                            except Exception as exc:
                                logger.warning(
                                    "LLM payload summarization failed: %s "
                                    "— falling back to mechanical truncation.",
                                    exc,
                                )
                        # Mechanical fallback: expanded from 8000 to 16000.
                        clipped = (
                            serialized[:max_chars].rstrip()
                            + f"\n[truncated_to_{max_chars}_chars_for_context_budget]"
                        )
                        part["tool_result"] = clipped
                        return True
        return False

    def _summarize_payload_via_llm(
        self,
        provider: Optional[LLMProvider],
        payload: str,
    ) -> Optional[str]:
        """Generate an LLM summary of an oversized history payload.

        Returns the summary text (which should be shorter than the
        original), or None on any failure (caller falls back to
        mechanical truncation).

        Payloads above ``_PAYLOAD_SUMMARIZE_CHAR_THRESHOLD`` are routed
        through the chunked summarizer (``_chunk_summarize_text`` in
        ``mu/session/messages.py``) so the *whole* payload is summarized
        across bounded chunks — not just its first 8k. Smaller payloads
        are summarized in a single call (no slice), so the model sees the
        full content and can choose what matters.
        """
        if provider is None or not payload:
            return None

        try:
            # Large payload: chunk-summarize the whole thing.
            if len(payload) > _PAYLOAD_SUMMARIZE_CHAR_THRESHOLD:
                from mu.session.messages import _chunk_summarize_text

                summary = _chunk_summarize_text(
                    provider, payload, budget_tokens=4_096
                ).strip()
                if summary and len(summary) < len(payload):
                    return summary
                return None

            system_prompt = (
                "You are a context summarizer for an AI coding agent. "
                "Summarize the following content into a concise but "
                "information-preserving summary. Keep file paths, function "
                "names, error messages, and key findings VERBATIM. "
                "Be concise but complete. Target 200-500 words.\n\n"
                "Output ONLY the summary, no headers or commentary."
            )

            messages = [
                Message(
                    role="user",
                    parts=[MessagePart(type="text", text=payload)],
                ),
            ]

            response = provider.generate(
                messages=messages,
                system_prompt=system_prompt,
                thinking=False,
                tools=None,
            )

            summary = str(response.text or "").strip()
            if not summary or len(summary) >= len(payload):
                return None
            return summary

        except Exception:
            return None

    # --------------------------------------------------- periodic L2 checkpoint

    def force_progress_checkpoint(
        self,
        provider: Optional[LLMProvider] = None,
        *,
        min_new_entries: int = 6,
    ) -> bool:
        """Refresh L2 mid-turn without compacting.

        On long turns that never reach the compaction token budget,
        ``conversation_summary`` (L2) stays frozen at its turn-start value
        while the model racks up real progress in L5 — so the model keeps
        re-deriving context it already gathered (the long-horizon stall).
        A checkpoint folds *recent* history into the structured L2 summary
        (Progress / Key decisions / Current state / Open items) so the
        model sees an up-to-date picture the next iteration.

        Unlike ``roll_history_summary`` this does **not** advance
        ``summary_anchor`` — entries stay verbatim in L5; only L2 is
        enriched. ``_checkpoint_anchor`` tracks how far we've already
        checkpointed so repeated checkpoints only summarize work done
        since the last one (not the whole turn again), bounding cost.

        Returns True if L2 was updated.
        """
        if not self.history:
            return False
        try:
            base = int(self.summary_anchor or 0)
        except Exception:
            base = 0
        try:
            start = max(base, int(getattr(self, "_checkpoint_anchor", 0) or 0))
        except Exception:
            start = base
        end = len(self.history)
        # Not enough new work since the last checkpoint to justify a
        # provider call. 6 ≈ a couple of tool rounds.
        if end - start < max(1, int(min_new_entries or 1)):
            return False
        entries = [
            msg
            for idx, msg in enumerate(self.history[start:end], start=start)
            if idx not in getattr(self, "protected_indices", set())
        ]
        if not entries:
            return False

        summary_batch = self._generate_llm_summary(provider, entries)
        if summary_batch and (
            "### Task" in summary_batch or "### Progress" in summary_batch
        ):
            self._merge_structured_summary(summary_batch)
        elif summary_batch:
            # Legacy/unstructured fallback — append under a dated header so
            # it's distinguishable from compaction-driven sections.
            header = (
                f"\n### Progress checkpoint (messages {start}-{end})\n"
                if self.conversation_summary
                else f"### Progress checkpoint (messages {start}-{end})\n"
            )
            self.conversation_summary = (
                f"{self.conversation_summary}{header}{summary_batch}".strip()
            )
            self._clip_conversation_summary()
        else:
            # LLM summary unavailable (no provider) — mechanical snapshot.
            mech = self._summarize_history_batch(entries)
            if not mech:
                return False
            header = (
                f"\n### Progress checkpoint (messages {start}-{end})\n"
                if self.conversation_summary
                else f"### Progress checkpoint (messages {start}-{end})\n"
            )
            self.conversation_summary = (
                f"{self.conversation_summary}{header}{mech}".strip()
            )
            self._clip_conversation_summary()

        self._checkpoint_anchor = end
        return True


__all__ = ["HistoryMixin"]
