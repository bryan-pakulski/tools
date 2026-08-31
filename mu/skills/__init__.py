"""Skills system — declarative agent extensions discovered from
`SKILL.md` files with YAML frontmatter.

Discovery roots (later wins on name collision):
  1. `<repo>/mu/skills/<name>/SKILL.md`              — built-in
  2. `~/.mu/skills/<name>/SKILL.md`                   — per-user
  3. `<each workspace folder>/.mu/skills/<name>/SKILL.md` — per-project

Frontmatter schema:

    ---
    name: commit-message
    description: Format a git commit message from staged diff context.
    trigger: \\b(commit|git\\s+message)\\b
    ---

`trigger` is a regex (case-insensitive) tested against the latest user
message; a match auto-expands the skill body inline for that turn.

v2 design:
  * compact-by-default: only name + description go into
    the system prompt under `# AVAILABLE SKILLS`;
  * `# AUTO-EXPANDED SKILLS` carries full bodies for skills whose
    trigger matched the latest user message;
  * model can call `invoke_skill(name)` to expand any other skill;
  * both activation paths print a visible `🎯 SKILL ACTIVE` banner via
    `announce_skill` (auto-expansion tags itself `· trigger`) and bump
    a per-skill counter in `/stats`;
  * users list / inspect / reload / enable / disable via `/skills`.

Budget knob: `skills_max_chars` (default 6144).
Mode knob: `skills_mode` (`"compact"` default, `"full"` for v1 behavior).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover — yaml is a runtime dependency
    yaml = None  # type: ignore


logger = logging.getLogger(__name__)


_MAX_TRIGGER_LEN = 512
_CATASTROPHIC_PATTERNS = ("(.+)+", "(.*)+", "(.+)*", "(.*)*", "(\\w+)+")


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source: str  # absolute path to SKILL.md it was loaded from
    trigger: Optional[str] = None
    trigger_regex: Optional["re.Pattern[str]"] = None
    mtime: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)


def _builtin_skills_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _user_skills_dir() -> str:
    return os.path.expanduser("~/.mu/skills")


def _compile_trigger(trigger: Optional[str], source: str) -> Optional["re.Pattern[str]"]:
    if not trigger:
        return None
    if len(trigger) > _MAX_TRIGGER_LEN:
        logger.warning(
            "skill %s: trigger pattern exceeds %d chars, ignoring",
            source,
            _MAX_TRIGGER_LEN,
        )
        return None
    for bad in _CATASTROPHIC_PATTERNS:
        if bad in trigger:
            logger.warning(
                "skill %s: trigger pattern %r contains catastrophic-backtracking shape %r, ignoring",
                source,
                trigger,
                bad,
            )
            return None
    try:
        return re.compile(trigger, re.IGNORECASE)
    except re.error as exc:
        logger.warning("skill %s: trigger regex %r failed to compile: %s", source, trigger, exc)
        return None


def match_trigger(skill: Skill, user_text: Optional[str]) -> bool:
    """Return True if `skill`'s compiled trigger regex matches `user_text`."""
    if skill.trigger_regex is None or not user_text:
        return False
    try:
        return bool(skill.trigger_regex.search(user_text))
    except Exception:  # pragma: no cover — defensive against runtime regex bugs
        return False


def _parse_skill_md(path: str) -> Optional[Skill]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return None
    if not raw.lstrip().startswith("---"):
        return None
    stripped = raw.lstrip()
    after_first = stripped[3:]
    end_idx = after_first.find("\n---")
    if end_idx == -1:
        return None
    fm_text = after_first[:end_idx]
    body = after_first[end_idx + 4 :].strip("\n").strip()

    meta: Dict[str, object] = {}
    if yaml is not None:
        try:
            loaded = yaml.safe_load(fm_text) or {}
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = {}
    if not meta:
        for line in fm_text.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("'").strip('"')

    name = str(meta.get("name") or "").strip()
    if not name:
        # Fall back to the directory name so a skill is still discoverable.
        name = os.path.basename(os.path.dirname(path)) or "skill"
    description = str(meta.get("description") or "").strip()
    trigger_val = meta.get("trigger")
    trigger = str(trigger_val).strip() if trigger_val else None
    trigger_regex = _compile_trigger(trigger, path)

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    extra = {
        k: v for k, v in meta.items() if k not in {"name", "description", "trigger"}
    }
    return Skill(
        name=name,
        description=description,
        body=body,
        source=path,
        trigger=trigger,
        trigger_regex=trigger_regex,
        mtime=mtime,
        extra=extra,
    )


def _scan_dir(root: str) -> List[Skill]:
    out: List[Skill] = []
    if not root or not os.path.isdir(root):
        return out
    for entry in sorted(os.listdir(root)):
        skill_path = os.path.join(root, entry, "SKILL.md")
        if os.path.isfile(skill_path):
            parsed = _parse_skill_md(skill_path)
            if parsed is not None:
                out.append(parsed)
    return out


# Discovery cache: keep last result for a short TTL so multiple per-turn
# lookups don't re-walk the same dirs. Invalidated by mtime change of any
# root or by `clear_skill_cache()`.
_CACHE_TTL_SECS = 5.0
_cache: Dict[str, object] = {"key": None, "skills": [], "ts": 0.0}


def clear_skill_cache() -> None:
    """Force the next `discover_skills` call to rescan from disk."""
    _cache["key"] = None
    _cache["skills"] = []
    _cache["ts"] = 0.0


def _cache_key(roots: List[str]) -> Tuple[Tuple[str, float], ...]:
    out: List[Tuple[str, float]] = []
    for root in roots:
        if root and os.path.isdir(root):
            try:
                out.append((root, os.path.getmtime(root)))
            except OSError:
                out.append((root, 0.0))
        else:
            out.append((root, -1.0))
    return tuple(out)


def discover_skills(workspace_folders: Optional[List[str]] = None) -> List[Skill]:
    """Discover all installed skills. Later sources override earlier on
    name collision so workspace-level skills can shadow built-ins."""
    roots: List[str] = [_builtin_skills_dir(), _user_skills_dir()]
    if workspace_folders:
        for folder in workspace_folders:
            roots.append(os.path.join(folder, ".mu", "skills"))

    key = _cache_key(roots)
    now = time.monotonic()
    if (
        _cache["key"] == key
        and isinstance(_cache.get("skills"), list)
        and (now - float(_cache.get("ts", 0.0) or 0.0)) < _CACHE_TTL_SECS
    ):
        return list(_cache["skills"])  # type: ignore[arg-type]

    skills_by_name: Dict[str, Skill] = {}
    for root in roots:
        for skill in _scan_dir(root):
            skills_by_name[skill.name] = skill

    skills = sorted(skills_by_name.values(), key=lambda s: s.name.lower())
    _cache["key"] = key
    _cache["skills"] = skills
    _cache["ts"] = now
    return skills


def get_skill(name: str, workspace_folders: Optional[List[str]] = None) -> Optional[Skill]:
    """Find a single skill by name from the same discovery roots."""
    if not name:
        return None
    needle = name.strip().lower()
    for skill in discover_skills(workspace_folders):
        if skill.name.lower() == needle:
            return skill
    return None


def _index_line(skill: Skill) -> str:
    return f"## SKILL: {skill.name}\n{skill.description}".strip()


def announce_skill(ui: Any, skill_name: str, via: Optional[str] = None) -> None:
    """Print a visible banner when a skill becomes active.

    Used for both activation paths so the user sees the same highlight
    regardless of how the skill entered context:
      * `invoke_skill` tool call (via=None — banner text unchanged)
      * trigger-regex auto-expansion (via="trigger")

    Renders the static styling via Rich markup but escapes the dynamic
    skill name (and `via` tag) so a bracketed name can never derail the
    markup parser. Falls back to a plain print when no Rich console is
    attached (headless / non-TTY).
    """
    from rich.markup import escape as _esc

    label = skill_name or "?"
    tag = f" · {_esc(via)}" if via else ""
    body = (
        f"[bold black on yellow] 🎯 SKILL ACTIVE: {_esc(label)}{tag} "
        f"[/bold black on yellow]"
    )
    console = getattr(ui, "console", None) if ui is not None else None
    if console is not None:
        try:
            from rich.text import Text as _Text

            console.print(_Text.from_markup(body))
            return
        except Exception:
            pass
    try:
        print(f"[SKILL ACTIVE: {label}{' · ' + via if via else ''}]")
    except Exception:
        pass


def render_skills_expanded(skill: Skill) -> str:
    """Format a single skill's full body for inclusion in context."""
    header = f"## SKILL: {skill.name}\n{skill.description}".strip()
    if skill.body:
        return f"{header}\n\n{skill.body}".strip()
    return header


def _render_full(skills: List[Skill], budget: int) -> str:
    """v1-style rendering: name + description + body for every skill, up to budget."""
    if not skills or budget <= 0:
        return ""
    lines: List[str] = ["# AVAILABLE SKILLS"]
    used = len(lines[0]) + 2
    rendered = 0
    for skill in skills:
        block = render_skills_expanded(skill)
        if used + len(block) + 2 > budget:
            break
        lines.append(block)
        used += len(block) + 2
        rendered += 1
    remaining = len(skills) - rendered
    if remaining > 0:
        lines.append(f"\n... and {remaining} more skill(s) not shown (budget reached).")
    return "\n".join(lines).strip()


def _render_compact(
    skills: List[Skill], budget: int, user_text: Optional[str], budget_scale: float = 1.0
) -> str:
    if not skills or budget <= 0:
        return ""

    expanded: List[Skill] = []
    indexed: List[Skill] = []
    for skill in skills:
        if match_trigger(skill, user_text):
            expanded.append(skill)
        else:
            indexed.append(skill)

    # Cap the per-body budget so a single auto-expanded skill cannot crowd
    # the name+description index out of LAYER 1B. Oversized bodies drop-tail
    # at the cap; `budget_scale` tunes (clamped to [0.1, 1.0], 1.0 = legacy).
    cap = int(budget * min(max(budget_scale, 0.1), 1.0))

    out: List[str] = []
    used = 0

    # Auto-expanded skills get budget priority so a triggered skill is
    # never dropped in favor of an inert index line.
    if expanded:
        header = "# AUTO-EXPANDED SKILLS"
        out.append(header)
        used += len(header) + 2
        for skill in expanded:
            block = render_skills_expanded(skill)
            if len(block) > cap:
                trimmed = block[: cap - 3].rstrip() + "..."
                block = f"{trimmed}\n\n[body capped: invoke via `invoke_skill {skill.name}` for the remainder]"
            if used + len(block) + 2 > budget:
                break
            out.append(block)
            used += len(block) + 2

    if indexed:
        header = "# AVAILABLE SKILLS"
        out.append(header)
        used += len(header) + 2
        rendered = 0
        for skill in indexed:
            block = _index_line(skill)
            if used + len(block) + 2 > budget:
                break
            out.append(block)
            used += len(block) + 2
            rendered += 1
        remaining = len(indexed) - rendered
        if remaining > 0:
            out.append(f"\n... and {remaining} more skill(s) not shown (budget reached).")

    return "\n".join(out).strip()


def render_skills_block(
    skills: List[Skill],
    budget: int = 6144,
    *,
    user_text: Optional[str] = None,
    mode: str = "compact",
    budget_scale: float = 1.0,
) -> str:
    """Render the discovered skills as a system-prompt block.

    `mode="compact"` (default) emits a name+description index and inlines
    only those skills whose trigger matches `user_text`. `mode="full"`
    inlines every skill's body up to the budget (v1 behavior).
    `budget_scale` caps each auto-expanded body at
    `budget * budget_scale` chars (clamped to [0.1, 1.0]) so one large
    triggered skill cannot crowd the index out of the block; the tail is
    recoverable on demand via `invoke_skill`.
    """
    if mode == "full":
        return _render_full(skills, budget)
    return _render_compact(skills, budget, user_text, budget_scale)


__all__ = [
    "Skill",
    "announce_skill",
    "discover_skills",
    "get_skill",
    "match_trigger",
    "render_skills_block",
    "render_skills_expanded",
    "clear_skill_cache",
]
