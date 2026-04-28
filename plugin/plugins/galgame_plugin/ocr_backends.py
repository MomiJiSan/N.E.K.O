from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from .rapidocr_support import inspect_rapidocr_installation, load_rapidocr_runtime
from .reader import normalize_text
from .tesseract_support import inspect_tesseract_installation, resolve_tesseract_path


_CJK_CHAR_RE_PATTERN = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
_KANA_CHAR_RE_PATTERN = r"[\u3040-\u30ff]"
_OCR_PREPARE_UPSCALE_SOURCE_LONG_EDGE = 900
_OCR_PREPARE_TARGET_LONG_EDGE = 1400
_OCR_PREPARE_MAX_LONG_EDGE = 1600
_RAPIDOCR_RUNTIME_CACHE_LOCK = threading.Lock()
_RAPIDOCR_RUNTIME_CACHE: dict[tuple[str, str, str, str, str], Any] = {}
_RAPIDOCR_RUNTIME_CACHE_MAX_ENTRIES = 2


def _ocr_reader_compat_symbol(name: str, fallback: Any) -> Any:
    module = sys.modules.get("plugin.plugins.galgame_plugin.ocr_reader")
    if module is None:
        return fallback
    return getattr(module, name, fallback)


class OcrBackend(Protocol):
    def is_available(self) -> bool: ...

    def extract_text(self, image: Any) -> str: ...


@dataclass(slots=True)
class OcrTextBox:
    text: str
    left: float
    top: float
    right: float
    bottom: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "text": self.text,
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }


@dataclass(slots=True)
class _RapidOcrToken:
    text: str
    score: float
    left: float
    right: float
    top: float
    bottom: float
    height: float


def _score_ocr_text(text: str) -> tuple[float, int, int]:
    import re

    normalized = normalize_text(text)
    if not normalized:
        return (-1.0, 0, 0)
    cjk_count = len(re.findall(_CJK_CHAR_RE_PATTERN, normalized))
    kana_count = len(re.findall(_KANA_CHAR_RE_PATTERN, normalized))
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", normalized)
    isolated_ascii_tokens = sum(1 for token in ascii_tokens if len(token) == 1)
    multi_char_ascii_tokens = sum(1 for token in ascii_tokens if len(token) > 1)
    significant_chars = sum(1 for ch in normalized if not ch.isspace())
    score = (
        (cjk_count * 5.0)
        + (kana_count * 4.0)
        + (multi_char_ascii_tokens * 1.5)
        + (significant_chars * 0.2)
        - (isolated_ascii_tokens * 2.0)
    )
    return (score, cjk_count + kana_count, significant_chars)


def _significant_char_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def _prepare_ocr_image(image: Any, *, apply_filters: bool = True) -> Any:
    from PIL import Image, ImageFilter, ImageOps

    resampling = getattr(Image, "Resampling", Image)
    prepared = image.convert("L")
    prepared = ImageOps.autocontrast(prepared)
    long_edge = max(prepared.width, prepared.height, 1)
    scale = 1.0
    if long_edge < _OCR_PREPARE_UPSCALE_SOURCE_LONG_EDGE:
        scale = min(2.0, _OCR_PREPARE_TARGET_LONG_EDGE / float(long_edge))
    elif long_edge > _OCR_PREPARE_MAX_LONG_EDGE:
        scale = _OCR_PREPARE_MAX_LONG_EDGE / float(long_edge)
    if abs(scale - 1.0) > 0.01:
        prepared = prepared.resize(
            (
                max(int(round(prepared.width * scale)), 1),
                max(int(round(prepared.height * scale)), 1),
            ),
            resampling.LANCZOS,
        )
    if apply_filters:
        prepared = prepared.filter(ImageFilter.MedianFilter(size=3))
        prepared = prepared.filter(ImageFilter.SHARPEN)
    return prepared


def _rapidocr_points(box: Any) -> list[tuple[float, float]]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, (list, tuple)):
        return []
    points: list[tuple[float, float]] = []
    for point in box:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return points


def _should_insert_ascii_space(previous_text: str, next_text: str) -> bool:
    if not previous_text or not next_text:
        return False
    left = previous_text[-1]
    right = next_text[0]
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()


def _join_ocr_segments(parts: list[str]) -> str:
    rendered = ""
    for part in parts:
        normalized = normalize_text(str(part or "")).replace("\n", " ").strip()
        if not normalized:
            continue
        if not rendered:
            rendered = normalized
            continue
        if _should_insert_ascii_space(rendered, normalized):
            rendered += " "
        rendered += normalized
    return rendered


def _rapidocr_tokens_from_output(raw_output: Any) -> list[_RapidOcrToken]:
    payload = raw_output[0] if isinstance(raw_output, tuple) and raw_output else raw_output
    if not isinstance(payload, list):
        return []
    tokens: list[_RapidOcrToken] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, score = item[0], item[1], item[2]
        normalized = normalize_text(str(text or "")).strip()
        if not normalized:
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        if score_value < 0.45:
            continue
        points = _rapidocr_points(box)
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        top = min(ys)
        bottom = max(ys)
        tokens.append(
            _RapidOcrToken(
                text=normalized,
                score=score_value,
                left=min(xs),
                right=max(xs),
                top=top,
                bottom=bottom,
                height=max(bottom - top, 1.0),
            )
        )
    return tokens


def _rapidocr_lines_from_output(raw_output: Any) -> list[tuple[str, float, OcrTextBox]]:
    tokens = _rapidocr_tokens_from_output(raw_output)
    if not tokens:
        return []
    tokens.sort(key=lambda token: (token.top, token.left))
    lines: list[list[_RapidOcrToken]] = []
    for token in tokens:
        token_center = (token.top + token.bottom) / 2.0
        placed = False
        for line in lines:
            line_top = min(item.top for item in line)
            line_bottom = max(item.bottom for item in line)
            line_center = (line_top + line_bottom) / 2.0
            threshold = max((line_bottom - line_top) * 0.6, token.height * 0.6, 12.0)
            if abs(token_center - line_center) <= threshold:
                line.append(token)
                placed = True
                break
        if not placed:
            lines.append([token])
    rendered_lines: list[str] = []
    line_results: list[tuple[str, float, OcrTextBox]] = []
    lines.sort(key=lambda line: (min(item.top for item in line), min(item.left for item in line)))
    for line in lines:
        line.sort(key=lambda item: item.left)
        text = _join_ocr_segments([item.text for item in line])
        if not text:
            continue
        rendered_lines.append(text)
        line_results.append(
            (
                text,
                sum(item.score for item in line) / len(line),
                OcrTextBox(
                    text=text,
                    left=min(item.left for item in line),
                    top=min(item.top for item in line),
                    right=max(item.right for item in line),
                    bottom=max(item.bottom for item in line),
                ),
            )
        )
    text = "\n".join(line for line in rendered_lines if line)
    normalized = normalize_text(text)
    if not normalized:
        return []
    scores = [score for _, score, _ in line_results]
    average_score = (sum(scores) / len(scores)) if scores else 0.0
    if _significant_char_count(normalized) < 4 and average_score < 0.55:
        return []
    return line_results


def _rapidocr_text_from_output(raw_output: Any) -> str:
    lines = _rapidocr_lines_from_output(raw_output)
    if not lines:
        return ""
    return "\n".join(text for text, _score, _box in lines)


class TesseractOcrBackend:
    def __init__(
        self,
        *,
        tesseract_path: str = "",
        install_target_dir_raw: str = "",
        languages: str = "",
    ) -> None:
        self._tesseract_path = tesseract_path
        self._install_target_dir_raw = install_target_dir_raw
        self._languages = languages

    def is_available(self) -> bool:
        path = resolve_tesseract_path(
            self._tesseract_path,
            install_target_dir_raw=self._install_target_dir_raw,
        )
        if not path:
            return False
        inspection = inspect_tesseract_installation(
            configured_path=self._tesseract_path,
            install_target_dir_raw=self._install_target_dir_raw,
            languages=self._languages,
        )
        return bool(inspection.get("installed"))

    def extract_text(self, image: Any) -> str:
        import pytesseract

        path = resolve_tesseract_path(
            self._tesseract_path,
            install_target_dir_raw=self._install_target_dir_raw,
        )
        if path:
            pytesseract.pytesseract.tesseract_cmd = path
        lang = self._languages
        config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
        prepared = _prepare_ocr_image(image)

        best_text = ""
        best_score = (-1.0, 0, 0)
        for candidate in (image, prepared):
            text = pytesseract.image_to_string(candidate, lang=lang, config=config).strip()
            score = _score_ocr_text(text)
            if score > best_score:
                best_text = text
                best_score = score
        return best_text


class RapidOcrBackend:
    def __init__(
        self,
        *,
        install_target_dir_raw: str,
        engine_type: str,
        lang_type: str,
        model_type: str,
        ocr_version: str,
    ) -> None:
        self._install_target_dir_raw = install_target_dir_raw
        self._engine_type = engine_type
        self._lang_type = lang_type
        self._model_type = model_type
        self._ocr_version = ocr_version
        self._runtime = None
        self._runtime_lock = threading.Lock()
        self._warmup_started = False
        self._warmup_completed = False
        self._warmup_error = ""

    def is_available(self) -> bool:
        inspect_fn = _ocr_reader_compat_symbol(
            "inspect_rapidocr_installation",
            inspect_rapidocr_installation,
        )
        inspection = inspect_fn(
            install_target_dir_raw=self._install_target_dir_raw,
            engine_type=self._engine_type,
            lang_type=self._lang_type,
            model_type=self._model_type,
            ocr_version=self._ocr_version,
        )
        return bool(inspection.get("installed"))

    def _ensure_runtime(self) -> Any:
        if self._runtime is None:
            with self._runtime_lock:
                if self._runtime is None:
                    key = (
                        str(self._install_target_dir_raw or ""),
                        str(self._engine_type or ""),
                        str(self._lang_type or ""),
                        str(self._model_type or ""),
                        str(self._ocr_version or ""),
                    )
                    with _RAPIDOCR_RUNTIME_CACHE_LOCK:
                        runtime = _RAPIDOCR_RUNTIME_CACHE.get(key)
                        if runtime is None:
                            load_fn = _ocr_reader_compat_symbol(
                                "load_rapidocr_runtime",
                                load_rapidocr_runtime,
                            )
                            runtime, _metadata = load_fn(
                                install_target_dir_raw=self._install_target_dir_raw,
                                engine_type=self._engine_type,
                                lang_type=self._lang_type,
                                model_type=self._model_type,
                                ocr_version=self._ocr_version,
                                force_reload=False,
                            )
                            _RAPIDOCR_RUNTIME_CACHE[key] = runtime
                            while len(_RAPIDOCR_RUNTIME_CACHE) > _RAPIDOCR_RUNTIME_CACHE_MAX_ENTRIES:
                                old_key = next(iter(_RAPIDOCR_RUNTIME_CACHE))
                                _RAPIDOCR_RUNTIME_CACHE.pop(old_key, None)
                        self._runtime = runtime
        return self._runtime

    def warmup_async(self, logger: Any | None = None) -> None:
        if self._warmup_started or self._warmup_completed:
            return
        self._warmup_started = True

        def _warmup() -> None:
            try:
                import numpy as np
                from PIL import Image

                runtime = self._ensure_runtime()
                runtime(np.asarray(Image.new("RGB", (96, 32), "white")))
                self._warmup_completed = True
            except Exception as exc:
                self._warmup_error = str(exc)
                if logger is not None:
                    try:
                        logger.debug("ocr_reader RapidOCR warmup skipped/failed: %s", exc)
                    except Exception:
                        pass

        threading.Thread(target=_warmup, name="galgame-rapidocr-warmup", daemon=True).start()

    def extract_text(self, image: Any) -> str:
        import numpy as np

        runtime = self._ensure_runtime()
        prepared = _prepare_ocr_image(image, apply_filters=False).convert("RGB")
        output = runtime(np.asarray(prepared))
        return _rapidocr_text_from_output(output)

    def extract_text_with_boxes(self, image: Any) -> tuple[str, list[OcrTextBox]]:
        import numpy as np

        runtime = self._ensure_runtime()
        prepared = _prepare_ocr_image(image, apply_filters=False).convert("RGB")
        output = runtime(np.asarray(prepared))
        lines = _rapidocr_lines_from_output(output)
        if not lines:
            return "", []
        scale_x = prepared.width / max(float(getattr(image, "width", prepared.width)), 1.0)
        scale_y = prepared.height / max(float(getattr(image, "height", prepared.height)), 1.0)
        boxes = [
            OcrTextBox(
                text=box.text,
                left=box.left / scale_x,
                top=box.top / scale_y,
                right=box.right / scale_x,
                bottom=box.bottom / scale_y,
            )
            for _text, _score, box in lines
        ]
        return "\n".join(text for text, _score, _box in lines), boxes
