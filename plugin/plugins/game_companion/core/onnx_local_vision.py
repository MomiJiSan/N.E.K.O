from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .local_vision import VisionBackendFn
from .vision_schema import redact_sensitive_text

SessionFactory = Callable[[Path], Any]

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class OnnxClassifierConfig:
    model_path: str | Path
    labels: Sequence[str]
    model_name: str = "onnx_screen_classifier"
    input_size: tuple[int, int] = (224, 224)
    normalize_imagenet: bool = True


def create_onnx_classifier_backend(
    config: OnnxClassifierConfig,
    *,
    session_factory: SessionFactory | None = None,
) -> VisionBackendFn:
    classifier = OnnxClassifierBackend(config, session_factory=session_factory)
    return classifier.classify


def load_onnx_classifier_config(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> OnnxClassifierConfig | None:
    if not bool(config.get("enabled")):
        return None
    model_path_value = str(config.get("model_path") or "").strip()
    if not model_path_value:
        return None
    labels = _labels_from_config(config, base_dir=base_dir)
    if not labels:
        return None
    return OnnxClassifierConfig(
        model_path=_resolve_path(model_path_value, base_dir),
        labels=labels,
        model_name=str(config.get("model_name") or "onnx_screen_classifier"),
        input_size=_input_size_from_config(config.get("input_size")),
        normalize_imagenet=bool(config.get("normalize_imagenet", True)),
    )


class OnnxClassifierBackend:
    def __init__(
        self,
        config: OnnxClassifierConfig,
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._config = config
        self._labels = tuple(str(label) for label in config.labels if str(label).strip())
        self._model_path = Path(config.model_path).expanduser()
        self._input_size = (
            max(1, int(config.input_size[0])),
            max(1, int(config.input_size[1])),
        )
        self._session_factory = session_factory or _default_session_factory
        self._session: Any | None = None
        self._input_name = ""

    def classify(self, image_path: Path, _profile_id: str) -> dict[str, Any]:
        if not self._labels:
            return {"error": "onnx_classifier_labels_missing"}
        if not self._model_path.is_file():
            return {"error": "onnx_classifier_model_not_found"}
        try:
            session = self._session or self._session_factory(self._model_path)
            self._session = session
            input_name = self._input_name or _session_input_name(session)
            self._input_name = input_name
            tensor = self._preprocess(image_path)
            started_at = time.perf_counter()
            outputs = session.run(None, {input_name: tensor})
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            logits = _first_output_vector(outputs)
            if logits.size != len(self._labels):
                return {
                    "error": f"logits_label_mismatch: logits={logits.size}, labels={len(self._labels)}",
                    "model_name": self._config.model_name,
                }
            scores = _softmax(logits)
            top_index = int(np.argmax(scores))
            label = self._labels[top_index]
            confidence = float(scores[top_index])
            return {
                "label": label,
                "confidence": round(max(0.0, min(confidence, 1.0)), 4),
                "all_scores": {
                    self._labels[index]: round(float(score), 6)
                    for index, score in enumerate(scores)
                },
                "latency_ms": round(max(0.0, latency_ms), 3),
                "model_name": self._config.model_name,
                "provider": "onnxruntime",
            }
        except Exception as exc:
            return {
                "error": redact_sensitive_text(str(exc)),
                "error_type": type(exc).__name__,
                "model_name": self._config.model_name,
            }

    def _preprocess(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            resized = image.convert("RGB").resize(self._input_size, resampling)
            array = np.asarray(resized, dtype=np.float32) / 255.0
        if self._config.normalize_imagenet:
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array.astype(np.float32, copy=False), axis=0)


def _default_session_factory(model_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed") from exc
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _labels_from_config(config: Mapping[str, Any], *, base_dir: str | Path) -> tuple[str, ...]:
    inline = config.get("labels")
    if isinstance(inline, list):
        labels = tuple(str(label).strip() for label in inline if str(label).strip())
        if labels:
            return labels
    labels_path_value = str(config.get("labels_path") or "").strip()
    if not labels_path_value:
        return ()
    labels_path = _resolve_path(labels_path_value, base_dir)
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    raw_labels = data.get("labels") if isinstance(data, Mapping) else data
    if not isinstance(raw_labels, list):
        return ()
    return tuple(str(label).strip() for label in raw_labels if str(label).strip())


def _resolve_path(path_value: str, base_dir: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path(base_dir).expanduser() / path


def _input_size_from_config(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (max(1, int(value[0])), max(1, int(value[1])))
        except (TypeError, ValueError):
            pass
    return (224, 224)


def _session_input_name(session: Any) -> str:
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("onnx session has no inputs")
    return str(inputs[0].name)


def _first_output_vector(outputs: Any) -> np.ndarray:
    if not isinstance(outputs, (list, tuple)) or not outputs:
        raise ValueError("onnx session returned no outputs")
    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim == 2:
        logits = logits[0]
    if logits.ndim != 1 or logits.size <= 0:
        raise ValueError("onnx classifier output must be a non-empty vector")
    return logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = np.sum(exp)
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(logits, dtype=np.float32)
    return exp / total
