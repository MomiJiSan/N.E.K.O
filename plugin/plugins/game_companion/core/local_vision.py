from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .vision_schema import redact_sensitive_text

VisionBackendFn = Callable[[Path, str], dict[str, Any]]
RAW_OUTPUT_KEY_TERMS = (
    "base64",
    "bytes",
    "data_url",
    "image",
    "path",
    "pixels",
    "raw",
)


@dataclass(frozen=True)
class LocalVisionBackend:
    classifier: VisionBackendFn | None = None
    detector: VisionBackendFn | None = None


_DEFAULT_BACKEND: LocalVisionBackend | None = None


def set_default_local_vision_backend(backend: LocalVisionBackend | None) -> None:
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend


def get_default_local_vision_backend() -> LocalVisionBackend | None:
    return _DEFAULT_BACKEND


def reset_default_local_vision_backend() -> None:
    set_default_local_vision_backend(None)


def analyze_local_vision(
    image_path: str | Path,
    *,
    profile_id: str,
    backend: LocalVisionBackend | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    active_backend = backend or get_default_local_vision_backend() or LocalVisionBackend()
    classifier_payload = _run_backend(active_backend.classifier, path, profile_id)
    detector_payload = _run_backend(active_backend.detector, path, profile_id)
    scene = _scene_from_classifier(classifier_payload)
    objects = _list_from_detector(detector_payload, "objects")
    ui = _list_from_detector(detector_payload, "ui")
    classifier_status = _backend_status(classifier_payload, active_backend.classifier, kind="classifier")
    detector_status = _backend_status(detector_payload, active_backend.detector, kind="detector")
    return {
        "available": classifier_status["status"] == "ready" or detector_status["status"] == "ready",
        "profile_id": profile_id,
        "scene": scene,
        "objects": objects,
        "ui": ui,
        "diagnostics": {
            "classifier": classifier_status,
            "detector": detector_status,
        },
    }


async def analyze_local_vision_async(
    image_path: str | Path,
    *,
    profile_id: str,
    backend: LocalVisionBackend | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(analyze_local_vision, image_path, profile_id=profile_id, backend=backend)


def _run_backend(
    backend_fn: VisionBackendFn | None,
    image_path: Path,
    profile_id: str,
) -> dict[str, Any] | None:
    if backend_fn is None:
        return None
    try:
        return _sanitize_backend_payload(backend_fn(image_path, profile_id))
    except Exception as exc:
        return {
            "error": redact_sensitive_text(str(exc)),
            "error_type": type(exc).__name__,
        }


def _scene_from_classifier(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"label": "unknown", "confidence": 0.0}
    label = str(payload.get("label") or payload.get("scene") or "unknown")
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    scene = {"label": label, "confidence": confidence}
    all_scores = payload.get("all_scores")
    if isinstance(all_scores, dict):
        scene["all_scores"] = {
            str(key): _safe_json_scalar(score)
            for key, score in all_scores.items()
            if _safe_json_scalar(score) is not None
        }
    latency_ms = payload.get("latency_ms")
    if latency_ms is not None:
        scene["latency_ms"] = latency_ms
    model_name = payload.get("model_name")
    if model_name:
        scene["model_name"] = str(model_name)
    return scene


def _list_from_detector(payload: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [_sanitize_detection_item(item) for item in value if isinstance(item, dict)]


def _backend_status(payload: dict[str, Any] | None, backend_fn: VisionBackendFn | None, *, kind: str) -> dict[str, Any]:
    if backend_fn is None:
        return {"status": "skipped", "reason": "not_configured"}
    if isinstance(payload, dict) and payload.get("error"):
        status = {"status": "failed", "error": payload.get("error")}
        if payload.get("error_type"):
            status["error_type"] = payload.get("error_type")
        return status
    if not _is_valid_backend_payload(payload, kind=kind):
        return {"status": "failed", "reason": "invalid_payload"}
    return {"status": "ready"}


def _is_valid_backend_payload(payload: dict[str, Any] | None, *, kind: str) -> bool:
    if not isinstance(payload, dict):
        return False
    if kind == "classifier":
        return bool(payload.get("label") or payload.get("scene"))
    if kind == "detector":
        return isinstance(payload.get("objects"), list) or isinstance(payload.get("ui"), list)
    return bool(payload)


def _sanitize_backend_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return payload
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        if _is_raw_output_key(key_text):
            continue
        sanitized_value = _safe_json_value(value)
        if sanitized_value is not None:
            sanitized[key_text] = sanitized_value
    return sanitized


def _sanitize_detection_item(item: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_backend_payload(item)
    return sanitized if isinstance(sanitized, dict) else {}


def _is_raw_output_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(term in normalized for term in RAW_OUTPUT_KEY_TERMS)


def _safe_json_value(value: Any) -> Any:
    scalar = _safe_json_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            safe_item = _safe_json_value(item)
            if safe_item is not None:
                result.append(safe_item)
        return result
    if isinstance(value, dict):
        return _sanitize_backend_payload(value)
    return None


def _safe_json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return None
