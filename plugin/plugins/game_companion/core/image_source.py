from __future__ import annotations

from dataclasses import dataclass
import base64
import binascii
from io import BytesIO
from pathlib import Path
import tempfile
from typing import Any

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - exercised only when Pillow is absent.
    Image = None  # type: ignore[assignment]

    class UnidentifiedImageError(Exception):
        pass


@dataclass(frozen=True)
class ImageMetadata:
    path: str
    width: int
    height: int
    mode: str
    format: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "image_path",
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "format": self.format,
        }


class ImageSourceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def read_image_metadata(image_path: str | Path) -> ImageMetadata:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise ImageSourceError("image_not_found", f"image file was not found: {path}")

    if Image is None:
        raise ImageSourceError("image_decode_failed", "Pillow is required to decode image files")

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            image_format = image.format
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageSourceError("image_decode_failed", f"image file could not be decoded: {path}") from exc

    return ImageMetadata(
        path=str(path.resolve()),
        width=width,
        height=height,
        mode=mode,
        format=image_format,
    )


def image_data_url_to_temp_file(data_url: str, *, suffix: str = ".png") -> Path:
    header, separator, payload = str(data_url or "").partition(",")
    if not separator or "base64" not in header.lower():
        raise ImageSourceError("image_decode_failed", "image_data_url must be a base64 data URL")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageSourceError("image_decode_failed", "image_data_url could not be decoded") from exc

    if Image is None:
        raise ImageSourceError("image_decode_failed", "Pillow is required to decode image files")
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            image_format = (image.format or "PNG").lower()
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageSourceError("image_decode_failed", "image_data_url is not a decodable image") from exc

    ext = f".{image_format}" if image_format in {"png", "jpg", "jpeg", "webp", "bmp"} else suffix
    tmp = tempfile.NamedTemporaryFile(prefix="game_companion_frame_", suffix=ext, delete=False)
    try:
        tmp.write(raw)
        return Path(tmp.name)
    finally:
        tmp.close()
