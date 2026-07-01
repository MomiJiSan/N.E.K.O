from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from .vision_schema import redact_sensitive_text


class VlmProvider(Protocol):
    def analyze(self, preparation: Mapping[str, Any]) -> "VlmProviderResult":
        ...


@dataclass(frozen=True)
class VlmProviderResult:
    status: str
    reason: str = ""
    scene: Mapping[str, Any] | None = None
    objects: list[Mapping[str, Any]] = field(default_factory=list)
    ui: list[Mapping[str, Any]] = field(default_factory=list)
    insights: list[Mapping[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    model_calls: list[Mapping[str, Any]] = field(default_factory=list)
    external_call_executed: bool = False
    raw_payload: Mapping[str, Any] | None = None


class DisabledVlmProvider:
    def analyze(self, _preparation: Mapping[str, Any]) -> VlmProviderResult:
        return VlmProviderResult(
            status="skipped",
            reason="provider_disabled",
            external_call_executed=False,
        )


def run_vlm_provider(provider: VlmProvider, preparation: Mapping[str, Any]) -> VlmProviderResult:
    if preparation.get("status") != "prepared":
        return VlmProviderResult(
            status="skipped",
            reason=str(preparation.get("reason") or "input_not_prepared"),
            external_call_executed=False,
        )
    try:
        result = provider.analyze(preparation)
    except Exception as exc:
        return VlmProviderResult(
            status="failed",
            reason=redact_sensitive_text(str(exc)),
            external_call_executed=False,
        )
    return result if isinstance(result, VlmProviderResult) else VlmProviderResult(status="failed", reason="invalid_provider_result")


def apply_vlm_provider_result(vision: Mapping[str, Any], result: VlmProviderResult) -> dict[str, Any]:
    payload = deepcopy(dict(vision))
    diagnostics = payload.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        payload["diagnostics"] = diagnostics
    diagnostics["vlm_provider"] = {
        "status": str(result.status or "unknown"),
        "reason": redact_sensitive_text(result.reason),
        "external_call_executed": bool(result.external_call_executed),
    }

    if result.status in {"merged", "completed", "ok"}:
        _merge_result_payload(payload, result)

    privacy = payload.setdefault("privacy", {})
    if not isinstance(privacy, dict):
        privacy = {}
        payload["privacy"] = privacy
    privacy["stores_raw_image"] = False
    privacy["external_model_calls"] = bool(result.external_call_executed)
    payload["model_calls"] = _safe_model_calls(result.model_calls) if result.external_call_executed else []
    return payload


def _merge_result_payload(payload: dict[str, Any], result: VlmProviderResult) -> None:
    if isinstance(result.scene, Mapping):
        scene = _safe_mapping(result.scene)
        if scene:
            payload["scene"] = scene
    if result.objects:
        payload["objects"] = [_safe_mapping(item) for item in result.objects if isinstance(item, Mapping)]
    if result.ui:
        payload["ui"] = [_safe_mapping(item) for item in result.ui if isinstance(item, Mapping)]
    if result.insights:
        payload["insights"] = [_safe_mapping(item) for item in result.insights if isinstance(item, Mapping)]
    if result.suggestions:
        payload["suggestions"] = [redact_sensitive_text(item) for item in result.suggestions if str(item).strip()]


def _safe_model_calls(model_calls: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    safe_calls = []
    for call in model_calls:
        if not isinstance(call, Mapping):
            continue
        safe_calls.append(
            {
                key: value
                for key, value in _safe_mapping(call).items()
                if key in {"provider", "model", "status", "latency_ms", "tokens"}
            }
        )
    return safe_calls


def _safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if _is_raw_key(key_text):
            continue
        safe_value = _safe_value(item)
        if safe_value is not None:
            safe[key_text] = safe_value
    return safe


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, (list, tuple)):
        return [item for item in (_safe_value(item) for item in value) if item is not None]
    if isinstance(value, Mapping):
        return _safe_mapping(value)
    return None


def _is_raw_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(term in normalized for term in ("base64", "bytes", "data_url", "image", "path", "pixels", "raw"))
