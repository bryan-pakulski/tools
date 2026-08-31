from dataclasses import dataclass

from mu.tools._envelope import infer_tool_error_code
from mu.tools.descriptors import (
    get_modifications,
    tool_approval_policy,
    tool_requires_approval,
)


@dataclass(frozen=True)
class ModificationPreview:
    filename: str
    original_content: str
    modified_content: str
    preview_error: str | None = None
    error_code: str | None = None

    @property
    def can_render_diff(self) -> bool:
        return (
            bool(self.filename)
            and self.original_content is not None
            and self.modified_content is not None
            and not self.preview_error
        )

    def to_payload(self) -> dict:
        return {
            "filename": self.filename,
            "original_content": self.original_content,
            "modified_content": self.modified_content,
            "preview_error": self.preview_error,
            "error_code": self.error_code,
            "can_render_diff": self.can_render_diff,
        }


@dataclass(frozen=True)
class ApprovalPlan:
    tool_name: str
    tool_args: dict
    requires_approval: bool
    can_approve: bool
    modifications: list[ModificationPreview]
    preview_error: str | None = None
    error_code: str | None = None
    approval_policy: str = "default"

    def to_payload(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "requires_approval": self.requires_approval,
            "can_approve": self.can_approve,
            "preview_error": self.preview_error,
            "error_code": self.error_code,
            "modifications": [mod.to_payload() for mod in self.modifications],
            "approval_policy": self.approval_policy,
        }


def _normalize_modification(
    tool_name: str,
    original_content: str,
    modified_content: str,
    filename: str,
) -> ModificationPreview:
    preview_error = None
    if modified_content and str(modified_content).startswith("ERROR:"):
        preview_error = str(modified_content)
    return ModificationPreview(
        filename=filename,
        original_content=original_content,
        modified_content=modified_content,
        preview_error=preview_error,
        error_code=infer_tool_error_code(tool_name, preview_error or modified_content),
    )


def build_approval_plan(
    tool_name: str,
    tool_args: dict,
    folder_context,
    *,
    strict_mode: bool = False,
    yolo: bool = False,
) -> ApprovalPlan:
    approval_policy = tool_approval_policy(tool_name, tool_args)
    requires_approval = approval_policy == "always_human" or (
        False if yolo else (strict_mode or tool_requires_approval(tool_name, tool_args))
    )
    if not requires_approval:
        return ApprovalPlan(
            tool_name=tool_name,
            tool_args=tool_args,
            requires_approval=False,
            can_approve=True,
            modifications=[],
            approval_policy=approval_policy,
        )

    normalized_modifications = [
        _normalize_modification(tool_name, original, modified, filename)
        for original, modified, filename in get_modifications(
            tool_name, tool_args, folder_context
        )
    ]
    preview_failure = next(
        (mod for mod in normalized_modifications if mod.preview_error),
        None,
    )

    return ApprovalPlan(
        tool_name=tool_name,
        tool_args=tool_args,
        requires_approval=True,
        can_approve=preview_failure is None,
        modifications=normalized_modifications,
        preview_error=preview_failure.preview_error if preview_failure else None,
        error_code=preview_failure.error_code if preview_failure else None,
        approval_policy=approval_policy,
    )


def collect_approval_plans(
    tool_calls: list,
    folder_context,
    *,
    strict_mode: bool = False,
    yolo: bool = False,
) -> dict[int, ApprovalPlan]:
    plans = {}
    for index, call in enumerate(tool_calls):
        plan = build_approval_plan(
            call.tool_name,
            call.tool_args,
            folder_context,
            strict_mode=strict_mode,
            yolo=yolo,
        )
        if plan.requires_approval:
            plans[index] = plan
    return plans


def build_approval_prompt(
    approval_plan: ApprovalPlan,
    *,
    display_args: dict,
    count_info: str,
) -> tuple[str, list[str], str]:
    if approval_plan.can_approve:
        prompt_text = (
            f"\n[bold yellow]Permission Required[/bold yellow] for tool: "
            f"[cyan]{approval_plan.tool_name}[/cyan]{count_info}\nArgs: {display_args}\nAllow?"
        )
        return prompt_text, ["y", "n", "e"], "y"

    prompt_text = (
        f"\n[bold red]Diff Failed[/bold red] for tool: "
        f"[cyan]{approval_plan.tool_name}[/cyan]{count_info}\nArgs: {display_args}\nReject or Explain?"
    )
    return prompt_text, ["n", "e"], "n"
