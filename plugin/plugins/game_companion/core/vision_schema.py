from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Mapping

VISION_SCHEMA_VERSION = 1
SOURCE_CONTEXT_ALLOWED_KEYS = {
    "type",
    "profile_id",
    "video_path",
    "ordinal",
    "frame_index",
    "timestamp_seconds",
    "expected_layout",
    "label",
    "tags",
    "note",
}
RAW_SOURCE_KEY_TERMS = frozenset({"base64", "bytes", "data_url", "image", "path", "pixels", "raw"})
WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]+")
DATA_URL_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+")
LONG_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b")


@dataclass(frozen=True)
class VisionFrameAnalysis:
    profile_id: str
    source: dict[str, Any] | None
    frame: dict[str, Any]
    scene: dict[str, Any]
    text: list[dict[str, Any]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    ui: list[dict[str, Any]] = field(default_factory=list)
    game_state: dict[str, Any] = field(default_factory=dict)
    insights: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    model_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        privacy = {
            "stores_raw_image": False,
            "external_model_calls": False,
        }
        privacy.update(deepcopy(self.privacy))
        privacy["stores_raw_image"] = False
        privacy["external_model_calls"] = False
        model_calls = [] if not privacy["external_model_calls"] else deepcopy(self.model_calls)
        return {
            "schema_version": VISION_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "source": sanitize_source_payload(self.source),
            "frame": deepcopy(self.frame),
            "scene": deepcopy(self.scene),
            "text": deepcopy(self.text),
            "objects": deepcopy(self.objects),
            "ui": deepcopy(self.ui),
            "game_state": deepcopy(self.game_state),
            "insights": deepcopy(self.insights),
            "suggestions": list(self.suggestions),
            "confidence": float(self.confidence),
            "diagnostics": deepcopy(self.diagnostics),
            "privacy": privacy,
            "model_calls": model_calls,
        }


def build_frame_metadata(
    *,
    image_path: str | Path,
    width: int,
    height: int,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": round(float(width) / float(height), 4) if height else None,
        "content_hash": content_hash_for_file(image_path),
    }
    if quality is not None:
        frame["quality"] = deepcopy(quality)
    return frame


def content_hash_for_file(image_path: str | Path) -> str:
    digest = sha256()
    with Path(image_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def source_with_origin(source: dict[str, Any], source_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    enriched = sanitize_source_payload(source) or {}
    if isinstance(source_context, Mapping) and source_context.get("type"):
        origin = _sanitize_source_context(source_context)
        if origin:
            enriched["origin"] = origin
    return enriched


def sanitize_source_payload(source: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(source, Mapping):
        return None
    sanitized: dict[str, Any] = {}
    for key, value in source.items():
        key_text = str(key)
        if _is_raw_source_key(key_text):
            if key_text == "video_path" and isinstance(value, str):
                sanitized[key_text] = redact_sensitive_text(value)
            continue
        safe_value = _json_safe_source_value(value)
        if safe_value is not None:
            sanitized[key_text] = safe_value
    return sanitized


def _sanitize_source_context(source_context: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in SOURCE_CONTEXT_ALLOWED_KEYS:
        if key not in source_context:
            continue
        value = source_context.get(key)
        safe_value = _json_safe_source_value(value)
        if safe_value is not None:
            sanitized[key] = safe_value
    if "type" not in sanitized:
        return {}
    return sanitized


def _json_safe_source_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, (list, tuple)):
        items = [_json_safe_source_value(item) for item in value]
        return [item for item in items if item is not None]
    if isinstance(value, Mapping):
        return sanitize_source_payload(value)
    return None


def _is_raw_source_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized == "video_path":
        return True
    return any(term in normalized for term in RAW_SOURCE_KEY_TERMS)


def redact_sensitive_text(value: str) -> str:
    text = DATA_URL_RE.sub("[redacted_image_data]", str(value))
    text = WINDOWS_PATH_RE.sub("[redacted_path]", text)
    text = LONG_BASE64_RE.sub("[redacted_base64]", text)
    if len(text) > 280:
        return f"{text[:280]}..."
    return text


def error_vision_payload(profile_id: str, code: str, message: str) -> dict[str, Any]:
    return VisionFrameAnalysis(
        profile_id=profile_id,
        source=None,
        frame={},
        scene={"label": "error", "confidence": 0.0},
        diagnostics={
            "warnings": [{"code": code, "message": message}],
            "error": {"code": code, "message": message},
        },
    ).to_dict()
