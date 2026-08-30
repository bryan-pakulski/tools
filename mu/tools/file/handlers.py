"""File mutator `@tool` handlers.

`write_file` / `apply_diff` / `search_and_replace_file`. The descriptors retain
normal host-workspace approval metadata; container workers force YOLO execution
so these tools run autonomously within the Docker boundary.

Each tool is a positional-signature body (publicly callable, easy to
test) plus a thin `_<name>_tool(args, context)` wrapper that carries
the `@tool` registration.
"""

from __future__ import annotations

import bisect
import difflib
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict

from mu.tools import tool
from mu.tools._bounds import check_bounds as _check_bounds
from mu.tools.capabilities import session_type_from_context
from utils.logger import logger


# ============================================================== write_file

def write_file(
    filename: str,
    content: str,
    folder_context,
    *,
    session_type: str = "workspace",
) -> str:
    """Creates or overwrites a file with the provided content."""
    if not _check_bounds(filename, folder_context, session_type=session_type):
        logger.warning(f"write_file: Access denied or path ignored: {filename}")
        return f"Error: Access denied or path ignored. '{filename}'"

    try:
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        if folder_context and hasattr(folder_context, "track_file"):
            folder_context.track_file(filename)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        logger.error(f"write_file: Error writing to {filename}: {e}")
        return f"Error writing file: {e}"


@tool(
    name="write_file",
    description="Creates or overwrites a file with the provided content.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Path to the file."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["filename", "content"],
    },
    requires_approval=True,
    execution_kind="mutate",
    preview_policy="required",
)
def _write_file_tool(args: Dict[str, Any], context) -> str:
    return write_file(
        args.get("filename", ""),
        args.get("content", ""),
        context.folder_context,
        session_type=session_type_from_context(context),
    )


# ============================================================== apply_diff

def _sanitize_diff(diff: str, filename: str) -> str:
    """Cleans up common LLM diff issues (chatter markers, missing
    headers, malformed hunk counts) so the system `patch` utility can
    consume it."""
    lines = diff.splitlines()
    if not lines:
        return diff

    # Chatter stripping must only apply OUTSIDE hunks: unified-diff content
    # lines may legitimately contain '***' or '```' (markdown separators,
    # fence examples) and stripping them would corrupt the patch.
    in_hunk = False
    cleaned_lines = []
    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
        elif line.startswith("--- ") or line.startswith("+++ "):
            in_hunk = False
        if not in_hunk:
            trimmed = line.strip()
            if trimmed.startswith("***") or trimmed.endswith("***"):
                continue
            if trimmed.startswith("```"):
                continue
        cleaned_lines.append(line)
    lines = cleaned_lines

    first_meaningful_line = -1
    for i, line in enumerate(lines):
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            first_meaningful_line = i
            break
    if first_meaningful_line != -1:
        lines = lines[first_meaningful_line:]

    sanitized = []
    hunk_lines = []

    def flush_hunk(hlines):
        if not hlines:
            return
        if hlines[0].startswith("@@"):
            header = hlines[0].strip()
            match = re.match(r"^@@ -(\d+),?\d* \+(\d+),?\d* @@(.*)$", header)

            start_old, start_new = "1", "1"
            tail = ""
            if match:
                start_old, start_new, tail = match.groups()

            count_old = 0
            count_new = 0
            content_lines = []
            for hl in hlines[1:]:
                if not hl:
                    continue
                if hl.startswith("-"):
                    count_old += 1
                    content_lines.append(hl)
                elif hl.startswith("+"):
                    count_new += 1
                    content_lines.append(hl)
                elif hl.startswith(" ") or hl.startswith("\t"):
                    count_old += 1
                    count_new += 1
                    content_lines.append(hl)
                else:
                    trimmed_hl = hl.strip()
                    if trimmed_hl:
                        count_old += 1
                        count_new += 1
                        content_lines.append(" " + hl)
                    else:
                        count_old += 1
                        count_new += 1
                        content_lines.append(" ")

            if not content_lines and header == "@@":
                return

            new_header = (
                f"@@ -{start_old},{count_old} +{start_new},{count_new} @@{tail}"
            )
            sanitized.append(new_header)
            sanitized.extend(content_lines)
            return
        sanitized.extend(hlines)

    if not any(l.startswith("--- ") for l in lines[:3]):
        sanitized.append(f"--- {filename}")
        sanitized.append(f"+++ {filename}")

    for line in lines:
        if line.startswith("---") or line.startswith("+++"):
            flush_hunk(hunk_lines)
            hunk_lines = []
            sanitized.append(line)
        elif line.startswith("@@"):
            flush_hunk(hunk_lines)
            hunk_lines = [line]
        elif line.startswith("+") or line.startswith("-") or line.startswith(" "):
            if hunk_lines:
                hunk_lines.append(line)
            else:
                trimmed = line.strip()
                if trimmed:
                    sanitized.append(" " + line)
        else:
            if hunk_lines:
                hunk_lines.append(" " + line)

    flush_hunk(hunk_lines)

    return "\n".join(sanitized) + "\n"


def apply_diff(filename: str, diff: str, folder_context, *, session_type: str = "workspace") -> str:
    """Applies a unified diff to a file."""
    if not _check_bounds(filename, folder_context, session_type=session_type):
        logger.warning(f"apply_diff: Access denied or path ignored: {filename}")
        return f"Error: Access denied or path ignored. '{filename}'"

    if not os.path.exists(filename):
        return f"Error: File '{filename}' does not exist. Cannot apply diff."

    try:
        diff = _sanitize_diff(diff, filename)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_diff:
            tmp_diff.write(diff)
            tmp_diff_path = tmp_diff.name

        try:
            result = subprocess.run(
                ["patch", "-u", "-l", "-F3", filename, "-i", tmp_diff_path],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            logger.error("apply_diff: 'patch' utility not found.")
            return "Error: 'patch' utility not found on system. Please install it to apply diffs."
        finally:
            # Temp cleanup in finally (codex round-8 F6): exceptions
            # previously leaked the temp diff (patch content) in /tmp.
            try:
                os.unlink(tmp_diff_path)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"apply_diff: Exception for {filename}: {e}")
        return f"Error applying diff: {e}"

    if result.returncode == 0:
        if folder_context and hasattr(folder_context, "track_file"):
            folder_context.track_file(filename)
        return f"Successfully applied diff to {filename}"
    logger.error(
        f"apply_diff: Patch error for {filename}: {result.stderr or result.stdout}"
    )
    return (
        f"Error applying diff via 'patch': {result.stderr or result.stdout}"
    )


@tool(
    name="apply_diff",
    description=(
        "Applies a unified diff to a file. This is a FALLBACK method. Use "
        "search_and_replace_file as the PRIMARY method for targeted code "
        "changes. Use apply_diff only for complex multi-file changes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Path to the file to modify.",
            },
            "diff": {
                "type": "string",
                "description": (
                    "The unified diff content to apply. MUST follow standard "
                    "unified diff format: --- filename, +++ filename, "
                    "@@ -L,C +L,C @@ headers, and +/-/space line markers."
                ),
            },
            "proposal_id": {
                "type": "string",
                "description": (
                    "Review-mode only: id of the approved diff proposal this "
                    "diff matches. Required when feature review mode is active."
                ),
            },
        },
        "required": ["filename", "diff"],
    },
    requires_approval=True,
    execution_kind="mutate",
    preview_policy="required",
)
def _apply_diff_tool(args: Dict[str, Any], context) -> str:
    return apply_diff(
        args.get("filename", ""),
        args.get("diff", ""),
        context.folder_context,
        session_type=session_type_from_context(context),
    )


# ===================================================== search_and_replace_file

def search_and_replace_file(
    filename: str,
    search: str,
    replace: str,
    expected_count: int = None,
    normalize_whitespace: bool = False,
    dry_run: bool = False,
    folder_context=None,
    session_type: str = "workspace",
) -> str:
    """Search and replace text in a file using exact string matching.

    Returns a JSON string carrying success/matches_found/match_locations/
    modified/preview/diff/error fields. See docstring on the @tool
    registration below for the agent-facing semantics."""
    if not _check_bounds(filename, folder_context, session_type=session_type):
        logger.warning(
            f"search_and_replace_file: Access denied or path ignored: {filename}"
        )
        return json.dumps(
            {"success": False, "error": f"Access denied or path ignored. '{filename}'"}
        )

    if not search:
        return json.dumps({"success": False, "error": "Search string cannot be empty"})

    if not os.path.exists(filename):
        return json.dumps(
            {"success": False, "error": f"File '{filename}' does not exist"}
        )

    try:
        with open(filename, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"File '{filename}' appears to be binary. "
                            "Search and replace not supported."
                        ),
                    }
                )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error reading file: {str(e)}"})

    # Capture identity before read; re-check before write so a concurrent
    # modification between read and write is not silently lost.
    try:
        _stat_before = os.stat(filename)
        _mtime_before = _stat_before.st_mtime_ns
        _size_before = _stat_before.st_size
    except OSError as e:
        return json.dumps({"success": False, "error": f"Error statting file: {str(e)}"})
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return json.dumps(
            {"success": False, "error": f"File '{filename}' is not UTF-8 encoded"}
        )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error reading file: {str(e)}"})

    matches = []
    lines = content.split("\n")

    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line) + 1)

    def _line_col_for_offset(offset: int) -> tuple[int, int]:
        line_index = bisect.bisect_right(line_starts, offset) - 1
        line_num = max(1, line_index + 1)
        column = offset - line_starts[line_index] + 1
        return line_num, column

    if normalize_whitespace:
        tokens = [token for token in re.split(r"(\s+)", search) if token != ""]
        pattern_parts = []
        for token in tokens:
            if token.isspace():
                pattern_parts.append(r"\s+")
            else:
                pattern_parts.append(re.escape(token))
        regex = re.compile("".join(pattern_parts))
        regex_matches = list(regex.finditer(content))
        located_matches = [(m.start(), m.end()) for m in regex_matches]
    else:
        located_matches = []
        start_pos = 0
        while True:
            pos = content.find(search, start_pos)
            if pos == -1:
                break
            located_matches.append((pos, pos + len(search)))
            start_pos = pos + 1

    for start_offset, _ in located_matches:
        line_num, column = _line_col_for_offset(start_offset)

        if 1 <= line_num <= len(lines):
            context_parts = []
            if line_num > 1:
                before_line = lines[line_num - 2].strip()
                if len(before_line) > 80:
                    before_line = before_line[:77] + "..."
                context_parts.append(f"  {before_line}")
            match_line = lines[line_num - 1].strip()
            if len(match_line) > 80:
                match_line = match_line[:77] + "..."
            context_parts.append(f"> {match_line}")
            if line_num < len(lines):
                after_line = lines[line_num].strip()
                if len(after_line) > 80:
                    after_line = after_line[:77] + "..."
                context_parts.append(f"  {after_line}")
            context_line = "\n".join(context_parts)
        else:
            context_line = ""

        matches.append({"line": line_num, "column": column, "context": context_line})

    if len(matches) == 0:
        return json.dumps(
            {
                "success": False,
                "matches_found": 0,
                "error": (
                    "No matches found for search string. Make sure the search "
                    "string matches exactly, including whitespace."
                ),
                "search_length": len(search),
            }
        )

    if expected_count is not None and len(matches) != expected_count:
        return json.dumps(
            {
                "success": False,
                "matches_found": len(matches),
                "expected_count": expected_count,
                "match_locations": matches,
                "error": (
                    f"Expected {expected_count} matches but found {len(matches)}. "
                    "Use expected_count to disambiguate or provide more specific "
                    "search string."
                ),
            }
        )

    if normalize_whitespace:
        if not located_matches:
            new_content = content
        else:
            rebuilt = []
            cursor = 0
            for start_offset, end_offset in located_matches:
                rebuilt.append(content[cursor:start_offset])
                rebuilt.append(replace)
                cursor = end_offset
            rebuilt.append(content[cursor:])
            new_content = "".join(rebuilt)
    else:
        new_content = content.replace(search, replace)

    diff_iter = difflib.unified_diff(
        content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    preview = "".join(diff_iter)
    diff_output = f"DIFF:\n{preview}" if preview else "DIFF:\n(no changes)"

    if dry_run:
        return json.dumps(
            {
                "success": True,
                "matches_found": len(matches),
                "match_locations": matches,
                "modified": False,
                "preview": preview,
                "diff": diff_output,
                "dry_run": True,
            }
        )

    # Concurrent-edit guard: if the file changed since our read, refuse
    # instead of silently clobbering the interleaved modification.
    try:
        _stat_now = os.stat(filename)
        if (
            _stat_now.st_mtime_ns != _mtime_before
            or _stat_now.st_size != _size_before
        ):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"File '{filename}' changed since it was read "
                        "(concurrent modification). Re-read the file and retry."
                    ),
                    "concurrent_modification": True,
                }
            )
    except OSError as e:
        return json.dumps({"success": False, "error": f"Error statting file: {str(e)}"})

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(new_content)
        if folder_context and hasattr(folder_context, "track_file"):
            folder_context.track_file(filename)
        return json.dumps(
            {
                "success": True,
                "matches_found": len(matches),
                "match_locations": matches,
                "modified": True,
                "preview": preview,
                "diff": diff_output,
            }
        )
    except Exception as e:
        logger.error(f"search_and_replace_file: Error writing {filename}: {e}")
        return json.dumps({"success": False, "error": f"Error writing file: {str(e)}"})


@tool(
    name="search_and_replace_file",
    description=(
        "Search and replace text in a file using exact string matching. "
        "This is the PRIMARY method for targeted code modifications. Use "
        "apply_diff (unified diff) only as a fallback for complex multi-file "
        "changes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Path to the file."},
            "search": {
                "type": "string",
                "description": (
                    "The exact text to search for in the file. Must match "
                    "exactly including whitespace."
                ),
            },
            "replace": {
                "type": "string",
                "description": "The text to replace the search match with.",
            },
            "expected_count": {
                "type": "integer",
                "description": (
                    "Optional expected number of matches. If provided and "
                    "count differs, operation fails (safety check for "
                    "disambiguation)."
                ),
            },
            "normalize_whitespace": {
                "type": "boolean",
                "description": (
                    "If True, normalize whitespace in search pattern "
                    "(collapse multiple spaces, trim leading/trailing)."
                ),
                "default": False,
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "If True, return preview of changes without modifying "
                    "the file."
                ),
                "default": False,
            },
        },
        "required": ["filename", "search", "replace"],
    },
    requires_approval=True,
    execution_kind="mutate",
    preview_policy="required",
)
def _search_and_replace_file_tool(args: Dict[str, Any], context) -> str:
    return search_and_replace_file(
        args.get("filename", ""),
        args.get("search", ""),
        args.get("replace", ""),
        args.get("expected_count"),
        args.get("normalize_whitespace", False),
        args.get("dry_run", False),
        context.folder_context,
        session_type=session_type_from_context(context),
    )
