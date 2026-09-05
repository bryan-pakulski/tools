"""Configurable model pricing registry for MuCLI.

The packaged defaults live in ``config/model_pricing.json``. Operators can
replace them without changing Python by writing ``$MUCLI_HOME/model_pricing.json``
(default ``~/.mucli/model_pricing.json``), and the GUI uses the same file.

Every lookup reloads the small JSON registry so manual config edits take effect
without restarting MuCLI. Historical job attempts persist the pricing version,
key and applied rates at execution time.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


USD_PER_MILLION = 1_000_000.0
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_pricing.json"


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    key: str
    input_per_million: Optional[float] = None
    cached_input_per_million: Optional[float] = None
    output_per_million: Optional[float] = None
    billing: str = "token"  # token | estimated_token | local | unknown
    # Kept for backward compatibility with older operator overrides. New
    # estimated-token rows should prefer separate input/output rates.
    estimated_total_per_million: Optional[float] = None
    aliases: tuple[str, ...] = ()
    # Provider/API-level modalities, not just capabilities claimed by the
    # underlying weights. Runtime attachment routing consumes this field, so
    # an Ollama model is only marked for media shapes that /api/chat accepts.
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    capabilities: tuple[str, ...] = ()
    context_window: Optional[int] = None
    long_context_cutoff: Optional[int] = None
    long_input_per_million: Optional[float] = None
    long_cached_input_per_million: Optional[float] = None
    long_output_per_million: Optional[float] = None
    role: str = ""
    notes: str = ""
    source: str = ""

    def public_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["input_modalities"] = list(self.input_modalities)
        value["output_modalities"] = list(self.output_modalities)
        value["capabilities"] = list(self.capabilities)
        return value


@dataclass(frozen=True)
class ModelCatalogEntry:
    """Backward-compatible catalog view for older Ollama callers."""

    provider: str
    key: str
    aliases: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    capabilities: tuple[str, ...] = ()
    context_window: Optional[int] = None
    role: str = ""
    local_size: str = ""
    usage_tier: str = ""
    notes: str = ""
    source: str = ""

    def public_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["input_modalities"] = list(self.input_modalities)
        value["output_modalities"] = list(self.output_modalities)
        value["capabilities"] = list(self.capabilities)
        return value


def pricing_config_path() -> Path:
    root = Path(os.path.expanduser(os.environ.get("MUCLI_HOME", "~/.mucli")))
    return root / "model_pricing.json"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"Pricing config must be a JSON object: {path}")
    return value


def _normalise_model_row(value: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(value or {})
    provider = str(row.get("provider") or "").strip().lower()
    key = str(row.get("key") or "").strip()
    if not provider or not key:
        raise ValueError("Every pricing row requires provider and key")
    billing = str(row.get("billing") or "token").strip().lower()
    if billing not in {"token", "estimated_token", "local", "unknown"}:
        raise ValueError(f"Unsupported billing mode for {provider}/{key}: {billing}")

    def nullable_float(name: str) -> Optional[float]:
        raw = row.get(name)
        if raw in (None, ""):
            return None
        number = float(raw)
        if number < 0:
            raise ValueError(f"{name} cannot be negative for {provider}/{key}")
        return number

    def nullable_int(name: str) -> Optional[int]:
        raw = row.get(name)
        if raw in (None, ""):
            return None
        number = int(raw)
        if number <= 0:
            raise ValueError(f"{name} must be positive for {provider}/{key}")
        return number

    aliases = row.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [part.strip() for part in aliases.split(",") if part.strip()]
    if not isinstance(aliases, list):
        raise ValueError(f"aliases must be an array for {provider}/{key}")

    def string_list(name: str, default: list[str]) -> list[str]:
        raw = row.get(name, default)
        if isinstance(raw, str):
            raw = [part.strip() for part in raw.split(",") if part.strip()]
        if not isinstance(raw, list):
            raise ValueError(f"{name} must be an array for {provider}/{key}")
        normalized: list[str] = []
        for item in raw:
            label = str(item or "").strip().lower().replace(" ", "_")
            if label and label not in normalized:
                normalized.append(label)
        if name in {"input_modalities", "output_modalities"} and not normalized:
            raise ValueError(f"{name} cannot be empty for {provider}/{key}")
        return normalized

    return {
        "provider": provider,
        "key": key,
        "billing": billing,
        "aliases": [str(item).strip() for item in aliases if str(item).strip()],
        # Missing metadata intentionally degrades to text-only. This makes
        # custom/unknown models safe by default instead of optimistically
        # sending binary payloads to APIs that may reject them.
        "input_modalities": string_list("input_modalities", ["text"]),
        "output_modalities": string_list("output_modalities", ["text"]),
        "capabilities": string_list("capabilities", []),
        "input_per_million": nullable_float("input_per_million"),
        "cached_input_per_million": nullable_float("cached_input_per_million"),
        "output_per_million": nullable_float("output_per_million"),
        "estimated_total_per_million": nullable_float("estimated_total_per_million"),
        "context_window": nullable_int("context_window"),
        "long_context_cutoff": nullable_int("long_context_cutoff"),
        "long_input_per_million": nullable_float("long_input_per_million"),
        "long_cached_input_per_million": nullable_float("long_cached_input_per_million"),
        "long_output_per_million": nullable_float("long_output_per_million"),
        "role": str(row.get("role") or ""),
        "notes": str(row.get("notes") or ""),
        "source": str(row.get("source") or ""),
    }


def validate_pricing_config(value: Dict[str, Any]) -> Dict[str, Any]:
    models = value.get("models")
    if not isinstance(models, list):
        raise ValueError("Pricing config requires a models array")
    normalized = [_normalise_model_row(item) for item in models if isinstance(item, dict)]
    seen: set[tuple[str, str]] = set()
    for item in normalized:
        identity = (item["provider"], item["key"].lower())
        if identity in seen:
            raise ValueError(f"Duplicate model pricing row: {item['provider']}/{item['key']}")
        seen.add(identity)
        if item["billing"] == "estimated_token":
            split = item["input_per_million"] is not None or item["output_per_million"] is not None
            if split and (item["input_per_million"] is None or item["output_per_million"] is None):
                raise ValueError(
                    f"estimated_token requires both input_per_million and output_per_million for {item['provider']}/{item['key']}"
                )
            if not split and item["estimated_total_per_million"] is None:
                raise ValueError(
                    f"estimated_token requires input/output rates or a legacy blended rate for {item['provider']}/{item['key']}"
                )
    return {
        "version": str(value.get("version") or "custom"),
        "currency": str(value.get("currency") or "USD").upper(),
        "unit": str(value.get("unit") or "per_million_tokens"),
        "models": normalized,
    }


def _default_registry() -> Dict[str, Any]:
    return validate_pricing_config(_read_json(DEFAULT_CONFIG_PATH))


def _inherit_packaged_capabilities(value: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill new capability fields in pre-feature operator overrides.

    Pricing overrides intentionally replace packaged rates. Capability fields
    were added later, so treating an absent field as an explicit text-only
    choice would silently disable vision for every existing installation.
    Explicit fields in the override always win.
    """
    packaged = _read_json(DEFAULT_CONFIG_PATH).get("models") or []
    by_name: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in packaged:
        if not isinstance(row, dict):
            continue
        provider = str(row.get("provider") or "").strip().lower()
        names = [row.get("key"), *(row.get("aliases") or [])]
        for name in names:
            normalized = str(name or "").strip().lower().replace("models/", "")
            if provider and normalized:
                by_name[(provider, normalized)] = row

    merged = dict(value)
    merged_rows = []
    for raw in value.get("models") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        provider = str(row.get("provider") or "").strip().lower()
        key = str(row.get("key") or "").strip().lower().replace("models/", "")
        default = by_name.get((provider, key))
        if default:
            for field in ("input_modalities", "output_modalities", "capabilities"):
                if field not in row and field in default:
                    row[field] = default[field]
        merged_rows.append(row)
    merged["models"] = merged_rows
    return merged


def _registry() -> tuple[Dict[str, Any], Path, bool]:
    override = pricing_config_path()
    path = override if override.exists() else DEFAULT_CONFIG_PATH
    try:
        raw = _read_json(path)
        if path == override:
            raw = _inherit_packaged_capabilities(raw)
        return validate_pricing_config(raw), path, path == override
    except Exception:
        if path != DEFAULT_CONFIG_PATH:
            # A bad operator override must not make all cost accounting unusable.
            # The GUI exposes the active/default paths so the invalid file can be fixed.
            return _default_registry(), DEFAULT_CONFIG_PATH, False
        raise


def _item(value: Dict[str, Any]) -> ModelPricing:
    return ModelPricing(
        provider=value["provider"],
        key=value["key"],
        input_per_million=value.get("input_per_million"),
        cached_input_per_million=value.get("cached_input_per_million"),
        output_per_million=value.get("output_per_million"),
        billing=value.get("billing", "token"),
        estimated_total_per_million=value.get("estimated_total_per_million"),
        aliases=tuple(value.get("aliases") or []),
        input_modalities=tuple(value.get("input_modalities") or ["text"]),
        output_modalities=tuple(value.get("output_modalities") or ["text"]),
        capabilities=tuple(value.get("capabilities") or []),
        context_window=value.get("context_window"),
        long_context_cutoff=value.get("long_context_cutoff"),
        long_input_per_million=value.get("long_input_per_million"),
        long_cached_input_per_million=value.get("long_cached_input_per_million"),
        long_output_per_million=value.get("long_output_per_million"),
        role=value.get("role", ""),
        notes=value.get("notes", ""),
        source=value.get("source", ""),
    )


def _configured_items() -> tuple[ModelPricing, ...]:
    registry, _, _ = _registry()
    return tuple(_item(value) for value in registry["models"])


_DEFAULT = _default_registry()
PRICING_VERSION = str(_DEFAULT["version"])
# Backward-compatible exported constants. Runtime lookup is dynamic and does
# not rely on these constants, so GUI/manual config edits take effect live.
PRICING: tuple[ModelPricing, ...] = tuple(
    _item(value) for value in _DEFAULT["models"] if value["provider"] != "ollama"
)
OLLAMA_CATALOG: tuple[ModelCatalogEntry, ...] = tuple(
    ModelCatalogEntry(
        provider="ollama",
        key=value["key"],
        aliases=tuple(value.get("aliases") or []),
        input_modalities=tuple(value.get("input_modalities") or ["text"]),
        output_modalities=tuple(value.get("output_modalities") or ["text"]),
        capabilities=tuple(value.get("capabilities") or []),
        context_window=value.get("context_window"),
        role=value.get("role", ""),
        notes=value.get("notes", ""),
        source=value.get("source", ""),
    )
    for value in _DEFAULT["models"]
    if value["provider"] == "ollama"
)


def save_pricing_config(value: Dict[str, Any]) -> Dict[str, Any]:
    # Older GUI/mobile clients do not know the capability columns yet. Treat
    # their omitted fields as a schema migration, while preserving any fields
    # an updated operator explicitly supplied.
    normalized = validate_pricing_config(_inherit_packaged_capabilities(value))
    target = pricing_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="model-pricing-", suffix=".json", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(normalized, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return pricing_catalog()


def reset_pricing_config() -> Dict[str, Any]:
    target = pricing_config_path()
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    return pricing_catalog()


def _normalise(value: str) -> str:
    return str(value or "").strip().lower().replace("models/", "")


def infer_provider(model_name: str) -> str:
    name = _normalise(model_name)
    if name.startswith("gemini-"):
        return "gemini"
    if name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return ""


def _matches(model_name: str, candidate: str) -> bool:
    model = _normalise(model_name)
    key = _normalise(candidate)
    if not model or not key:
        return False
    if model == key:
        return True
    return model.startswith(key + "-") or model.startswith(key + ":")


def _resolve(provider: str, model_name: str) -> Optional[ModelPricing]:
    provider_name = _normalise(provider) or infer_provider(model_name)
    matches: list[tuple[int, ModelPricing]] = []
    for item in _configured_items():
        if provider_name and item.provider != provider_name:
            continue
        for alias in (item.key, *item.aliases):
            if _matches(model_name, alias):
                matches.append((len(alias), item))
                break
    return max(matches, key=lambda pair: pair[0])[1] if matches else None


def resolve_token_pricing(provider: str, model_name: str) -> Optional[ModelPricing]:
    item = _resolve(provider, model_name)
    if item is None or item.billing not in {"token", "estimated_token"}:
        return None
    return item


def resolve_model_capabilities(provider: str, model_name: str) -> Dict[str, Any]:
    """Return conservative, runtime-usable capability metadata.

    Unknown models are text-only until an operator adds an explicit registry
    row. This result is deliberately separate from pricing resolution because
    local and unpriced models still need a safe input-modality decision.
    """
    item = _resolve(provider, model_name)
    if item is None:
        return {
            "provider": _normalise(provider) or infer_provider(model_name),
            "model": str(model_name or ""),
            "matched": False,
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "capabilities": [],
        }
    return {
        "provider": item.provider,
        "model": str(model_name or item.key),
        "key": item.key,
        "matched": True,
        "input_modalities": list(item.input_modalities),
        "output_modalities": list(item.output_modalities),
        "capabilities": list(item.capabilities),
    }


def input_modality_for_mime(mime_type: str, filename: str = "") -> str:
    """Map an upload MIME type to the registry's input-modality vocabulary."""
    mime = str(mime_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    name = str(filename or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    document_mimes = {
        "application/pdf",
        "application/json",
        "application/rtf",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    document_suffixes = (
        ".csv", ".doc", ".docx", ".json", ".md", ".pdf", ".ppt",
        ".pptx", ".rtf", ".txt", ".xls", ".xlsx",
    )
    if mime.startswith("text/") or mime in document_mimes or name.endswith(document_suffixes):
        return "document"
    return "file"


def resolve_ollama_catalog(model_name: str) -> Optional[ModelCatalogEntry]:
    item = _resolve("ollama", model_name)
    if item is None:
        return None
    return ModelCatalogEntry(
        provider="ollama",
        key=item.key,
        aliases=item.aliases,
        input_modalities=item.input_modalities,
        output_modalities=item.output_modalities,
        capabilities=item.capabilities,
        context_window=item.context_window,
        role=item.role,
        notes=item.notes,
        source=item.source,
    )


def ollama_billing_mode(*, model_name: str, mode: str = "", endpoint: str = "") -> str:
    model = _normalise(model_name)
    selected_mode = _normalise(mode)
    host = _normalise(endpoint)
    if selected_mode == "cloud" or model.endswith(":cloud") or "ollama.com" in host:
        return "cloud"
    if selected_mode == "local":
        return "local"
    if host:
        return "cloud" if "ollama.com" in host else "local"
    return "local"


def _token_cost(
    pricing: ModelPricing,
    *,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
) -> tuple[float, Dict[str, Any], Dict[str, float]]:
    high = bool(pricing.long_context_cutoff and input_tokens > pricing.long_context_cutoff)
    input_rate = pricing.long_input_per_million if high and pricing.long_input_per_million is not None else pricing.input_per_million
    cached_rate = pricing.long_cached_input_per_million if high and pricing.long_cached_input_per_million is not None else pricing.cached_input_per_million
    output_rate = pricing.long_output_per_million if high and pricing.long_output_per_million is not None else pricing.output_per_million
    effective_cached_rate = cached_rate if cached_rate is not None else input_rate
    uncached = max(0, input_tokens - cached_tokens)

    uncached_input_usd = (uncached / USD_PER_MILLION * input_rate) if input_rate is not None else 0.0
    cached_input_usd = (cached_tokens / USD_PER_MILLION * effective_cached_rate) if effective_cached_rate is not None else 0.0
    output_usd = (output_tokens / USD_PER_MILLION * output_rate) if output_rate is not None else 0.0
    input_usd = uncached_input_usd + cached_input_usd
    cost = input_usd + output_usd

    return cost, {
        "input_per_million": input_rate,
        "cached_input_per_million": effective_cached_rate,
        "output_per_million": output_rate,
    }, {
        "input_usd": input_usd,
        "uncached_input_usd": uncached_input_usd,
        "cached_input_usd": cached_input_usd,
        "output_usd": output_usd,
        "total_usd": cost,
    }


def _legacy_blended_cost(
    pricing: ModelPricing,
    *,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, Dict[str, Any], Dict[str, float]]:
    rate = float(pricing.estimated_total_per_million or 0.0)
    input_usd = input_tokens / USD_PER_MILLION * rate
    output_usd = output_tokens / USD_PER_MILLION * rate
    total = input_usd + output_usd
    return total, {"estimated_total_per_million": rate}, {
        "input_usd": input_usd,
        "uncached_input_usd": input_usd,
        "cached_input_usd": 0.0,
        "output_usd": output_usd,
        "total_usd": total,
    }


def estimate_model_cost(
    *,
    provider: str,
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    provider_reported_cost: Optional[float] = None,
    ollama_mode: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    provider_name = _normalise(provider) or infer_provider(model_name) or "unknown"
    model = str(model_name or "")
    in_tokens = max(0, int(input_tokens or 0))
    out_tokens = max(0, int(output_tokens or 0))
    cache_tokens = max(0, min(in_tokens, int(cached_tokens or 0)))
    reasoning = max(0, int(reasoning_tokens or 0))
    registry, source_path, using_override = _registry()

    base: Dict[str, Any] = {
        "pricing_version": str(registry["version"]),
        "provider": provider_name,
        "model": model,
        "usage": {
            "input": in_tokens,
            "uncached_input": max(0, in_tokens - cache_tokens),
            "cached_input": cache_tokens,
            "output": out_tokens,
            "reasoning": reasoning,
        },
        "config_source": "user" if using_override else "default",
        "config_path": str(source_path),
    }

    if provider_reported_cost is not None:
        base.update({
            "pricing_key": "provider-reported",
            "billing": "provider_reported",
            "source": "provider_reported",
            "api_cost_usd": max(0.0, float(provider_reported_cost)),
            "rates": {},
        })
        return base

    pricing = _resolve(provider_name, model)

    if provider_name == "ollama":
        location = ollama_billing_mode(model_name=model, mode=ollama_mode, endpoint=endpoint)
        if location == "local":
            base.update({
                "pricing_key": pricing.key if pricing else model,
                "billing": "local",
                "source": "local",
                "api_cost_usd": 0.0,
                "rates": {},
                "cost_components": {
                    "input_usd": 0.0,
                    "uncached_input_usd": 0.0,
                    "cached_input_usd": 0.0,
                    "output_usd": 0.0,
                    "total_usd": 0.0,
                },
                "note": "Local Ollama has $0 attributable provider/API cost; host compute is excluded.",
            })
            if pricing:
                base["catalog"] = pricing.public_dict()
            return base
        if pricing and pricing.billing == "estimated_token":
            if pricing.input_per_million is not None and pricing.output_per_million is not None:
                cost, rates, components = _token_cost(
                    pricing,
                    input_tokens=in_tokens,
                    cached_tokens=cache_tokens,
                    output_tokens=out_tokens,
                )
                base.update({
                    "pricing_key": pricing.key,
                    "billing": "estimated_token",
                    "source": "configured_estimate",
                    "api_cost_usd": cost,
                    "rates": rates,
                    "cost_components": components,
                    "rate_shape": "input_output",
                    "note": pricing.notes or "Configured input/output provider estimate.",
                    "catalog": pricing.public_dict(),
                })
                return base
            if pricing.estimated_total_per_million is not None:
                cost, rates, components = _legacy_blended_cost(
                    pricing,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
                base.update({
                    "pricing_key": pricing.key,
                    "billing": "estimated_token",
                    "source": "configured_estimate",
                    "api_cost_usd": cost,
                    "rates": rates,
                    "cost_components": components,
                    "rate_shape": "legacy_blended",
                    "note": pricing.notes or "Configured legacy blended provider estimate.",
                    "catalog": pricing.public_dict(),
                })
                return base
        if pricing and pricing.billing == "token":
            cost, rates, components = _token_cost(
                pricing,
                input_tokens=in_tokens,
                cached_tokens=cache_tokens,
                output_tokens=out_tokens,
            )
            base.update({
                "pricing_key": pricing.key,
                "billing": "token",
                "source": "pricing_map",
                "api_cost_usd": cost,
                "rates": rates,
                "cost_components": components,
                "catalog": pricing.public_dict(),
            })
            return base
        base.update({
            "pricing_key": pricing.key if pricing else model,
            "billing": "unknown",
            "source": "unpriced",
            "api_cost_usd": None,
            "rates": {},
            "note": "No configured Ollama Cloud estimate matched this model.",
        })
        if pricing:
            base["catalog"] = pricing.public_dict()
        return base

    if pricing is None:
        base.update({
            "pricing_key": "",
            "billing": "unknown",
            "source": "unpriced",
            "api_cost_usd": None,
            "rates": {},
            "note": "No pricing registry entry matched this model; cost remains unknown rather than $0.",
        })
        return base

    if pricing.billing == "estimated_token":
        if pricing.input_per_million is not None and pricing.output_per_million is not None:
            cost, rates, components = _token_cost(
                pricing,
                input_tokens=in_tokens,
                cached_tokens=cache_tokens,
                output_tokens=out_tokens,
            )
            base.update({
                "pricing_key": pricing.key,
                "billing": "estimated_token",
                "source": "configured_estimate",
                "api_cost_usd": cost,
                "rates": rates,
                "cost_components": components,
                "rate_shape": "input_output",
                "long_context_tier": bool(pricing.long_context_cutoff and in_tokens > pricing.long_context_cutoff),
            })
            return base
        if pricing.estimated_total_per_million is not None:
            cost, rates, components = _legacy_blended_cost(
                pricing,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
            base.update({
                "pricing_key": pricing.key,
                "billing": "estimated_token",
                "source": "configured_estimate",
                "api_cost_usd": cost,
                "rates": rates,
                "cost_components": components,
                "rate_shape": "legacy_blended",
                "long_context_tier": False,
            })
            return base

    cost, rates, components = _token_cost(
        pricing,
        input_tokens=in_tokens,
        cached_tokens=cache_tokens,
        output_tokens=out_tokens,
    )
    high = bool(pricing.long_context_cutoff and in_tokens > pricing.long_context_cutoff)
    base.update({
        "pricing_key": pricing.key,
        "billing": pricing.billing,
        "source": "pricing_map",
        "api_cost_usd": cost,
        "long_context_tier": high,
        "rates": rates,
        "cost_components": components,
        "source_url": pricing.source,
    })
    return base


def calculate_model_cost(**kwargs: Any) -> Optional[float]:
    value = estimate_model_cost(**kwargs).get("api_cost_usd")
    return None if value is None else float(value)


def pricing_catalog() -> Dict[str, Any]:
    registry, active_path, using_override = _registry()
    models = [_item(value).public_dict() for value in registry["models"]]
    return {
        "version": registry["version"],
        "currency": registry["currency"],
        "unit": registry["unit"],
        "models": models,
        # Compatibility for existing clients while they move to the unified table.
        "ollama": [item for item in models if item["provider"] == "ollama"],
        "config_path": str(pricing_config_path()),
        "active_config_path": str(active_path),
        "default_config_path": str(DEFAULT_CONFIG_PATH),
        "using_override": using_override,
        "provider_notes": {
            "openai": "Configured token-rate estimate.",
            "gemini": "Configured token-rate estimate; output may include thinking tokens.",
            "ollama_local": "$0 attributable provider/API cost; host compute is separate.",
            "ollama_cloud": "Uses separate configured input/output estimates when present; legacy blended overrides remain supported.",
        },
    }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ModelCatalogEntry",
    "ModelPricing",
    "OLLAMA_CATALOG",
    "PRICING",
    "PRICING_VERSION",
    "calculate_model_cost",
    "estimate_model_cost",
    "infer_provider",
    "ollama_billing_mode",
    "pricing_catalog",
    "pricing_config_path",
    "reset_pricing_config",
    "resolve_model_capabilities",
    "resolve_ollama_catalog",
    "resolve_token_pricing",
    "save_pricing_config",
    "input_modality_for_mime",
    "validate_pricing_config",
]
