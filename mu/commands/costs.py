"""TUI view of the configurable model-pricing registry."""

from __future__ import annotations

from typing import Any

from utils.model_pricing import pricing_catalog

from . import CommandResult, command


def _money(value) -> str:
    return "—" if value is None else f"${float(value):.3f}"


def _emit(session: Any, text: str, allow_prompt: bool) -> None:
    if not allow_prompt:
        return
    ui = getattr(session, "ui", None)
    if ui is not None and hasattr(ui, "show_info"):
        ui.show_info(text)


@command(
    "/costs",
    "/pricing",
    help="Show MuCLI's configurable model pricing registry. Optional: /costs openai|gemini|ollama",
)
def costs_cmd(session: Any, args: str, *, allow_prompt: bool = True) -> CommandResult:
    catalog = pricing_catalog()
    wanted = str(args or "").strip().lower()
    if wanted and wanted not in {"openai", "gemini", "ollama"}:
        return CommandResult(ok=False, message="Usage: /costs [openai|gemini|ollama]")

    lines = [
        f"Model pricing registry · {catalog['version']}",
        f"Active config: {catalog.get('active_config_path') or catalog.get('config_path')}",
        "Rates are USD / 1M tokens and are telemetry estimates, not invoices.",
        "",
    ]
    for provider in ("openai", "gemini", "ollama"):
        if wanted and wanted != provider:
            continue
        rows = [item for item in catalog["models"] if item["provider"] == provider]
        if not rows:
            continue
        lines.append(provider.upper())
        for item in rows:
            billing = str(item.get("billing") or "unknown")
            if billing == "local":
                rates = "$0 provider/API · host compute excluded"
            elif billing == "estimated_token":
                if item.get("input_per_million") is not None and item.get("output_per_million") is not None:
                    rates = (
                        f"~{_money(item.get('input_per_million'))} in / "
                        f"~{_money(item.get('output_per_million'))} out"
                    )
                    if item.get("cached_input_per_million") is not None:
                        rates += f" / ~{_money(item.get('cached_input_per_million'))} cached"
                else:
                    rates = f"~{_money(item.get('estimated_total_per_million'))} / 1M total · legacy blended"
            elif billing == "token":
                rates = (
                    f"{_money(item.get('input_per_million'))} in / "
                    f"{_money(item.get('cached_input_per_million'))} cached / "
                    f"{_money(item.get('output_per_million'))} out"
                )
            else:
                rates = "unpriced"
            tier = ""
            if item.get("long_context_cutoff"):
                tier = f" · high tier >{int(item['long_context_cutoff']):,} input"
            inputs = ",".join(item.get("input_modalities") or ["text"])
            capabilities = ",".join(item.get("capabilities") or [])
            capability_note = f" · inputs: {inputs}"
            if capabilities:
                capability_note += f" · features: {capabilities}"
            lines.append(
                f"  {item['key']:<30} {rates}{tier}{capability_note}"
            )
        lines.append("")

    lines.extend([
        "Estimated input and output independently from measured token counts.",
        f"GUI editor: /static/model_costs.html",
        f"Override file: {catalog.get('config_path')}",
    ])
    body = "\n".join(lines)
    _emit(session, body, allow_prompt)
    return CommandResult(ok=True, message="Model pricing registry generated.", data=catalog)
