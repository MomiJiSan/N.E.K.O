"""Persistent content-addressed CG cache owned by the dedicated plugin."""

from __future__ import annotations

import asyncio
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Awaitable, Callable, Mapping
import uuid

from PIL import Image


ImageProvider = Callable[[dict[str, Any]], Awaitable[bytes]]


class CgCacheError(ValueError):
    pass


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return dict(default)
    return value if isinstance(value, dict) else dict(default)


def _validate_png(payload: bytes) -> tuple[int, int]:
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise CgCacheError("generated asset must be a PNG")
    width, height = struct.unpack(">II", payload[16:24])
    if (width, height) != (1920, 1080):
        raise CgCacheError("generated asset must be 1920x1080")
    return width, height


def _validate_webp(payload: bytes) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            image_format = image.format
            width, height = image.size
    except Exception as error:
        raise CgCacheError("generated asset must be a valid WebP") from error
    if image_format != "WEBP" or (width, height) != (1920, 1080):
        raise CgCacheError("generated asset must be a 1920x1080 WebP")
    return width, height


class CgCache:
    def __init__(
        self,
        static_ui_root: Path,
        *,
        state_root: Path | None = None,
        max_bytes: int = 1024 * 1024 * 1024,
        negative_ttl_seconds: float = 600.0,
        provider: ImageProvider | None = None,
    ):
        self.static_ui_root = static_ui_root
        self.cg_root = static_ui_root / "cg"
        self.state_root = state_root or static_ui_root.parent / "cg_cache"
        self.metadata_root = self.state_root / "metadata"
        self.recipe_root = self.state_root / "recipes"
        self.index_path = self.state_root / "cache_index.json"
        self.max_bytes = max(1, int(max_bytes))
        self.negative_ttl_seconds = max(1.0, float(negative_ttl_seconds))
        self.provider = provider
        self._singleflight_guard = asyncio.Lock()
        self._singleflight: dict[str, asyncio.Lock] = {}
        self._negative_until: dict[str, float] = {}

    def prepare(self, source_index: Path) -> None:
        self.cg_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.recipe_root.mkdir(parents=True, exist_ok=True)
        target_index = self.static_ui_root / "index.html"
        if source_index.is_file():
            target_index.parent.mkdir(parents=True, exist_ok=True)
            if not target_index.exists() or source_index.read_bytes() != target_index.read_bytes():
                shutil.copy2(source_index, target_index)
        if not self.index_path.exists():
            _atomic_write(self.index_path, b'{"assets":{}}')

    async def issue_recipe(self, recipe: Mapping[str, Any]) -> str:
        normalized = {str(key): value for key, value in recipe.items()}
        digest = await asyncio.to_thread(lambda: hashlib.sha256(_canonical_bytes(normalized)).hexdigest())
        generation_key = f"sha256:{digest}"
        await asyncio.to_thread(
            _atomic_write,
            self.recipe_root / f"{digest}.json",
            _canonical_bytes(normalized),
        )
        return generation_key

    async def lookup(self, generation_key: str) -> dict[str, Any] | None:
        digest = self._key_digest(generation_key)
        return await asyncio.to_thread(self._lookup_sync, digest, True)

    async def recipe(self, generation_key: str) -> dict[str, Any]:
        digest = self._key_digest(generation_key)
        recipe = await asyncio.to_thread(
            _read_json,
            self.recipe_root / f"{digest}.json",
            {},
        )
        if not recipe:
            raise CgCacheError("generation key was not issued by resolve_interaction")
        return recipe

    async def ensure(self, generation_key: str, *, enabled: bool) -> dict[str, Any]:
        digest = self._key_digest(generation_key)
        cached = await asyncio.to_thread(self._lookup_sync, digest, True)
        if cached is not None:
            return cached
        recipe_path = self.recipe_root / f"{digest}.json"
        if not await asyncio.to_thread(recipe_path.is_file):
            raise CgCacheError("generation key was not issued by resolve_interaction")
        if not enabled:
            return {"status": "disabled", "generation_key": generation_key}
        if self.provider is None:
            return {"status": "provider_unavailable", "generation_key": generation_key}

        now = time.monotonic()
        if self._negative_until.get(digest, 0.0) > now:
            return {"status": "negative_cache", "generation_key": generation_key}
        lock = await self._lock_for(digest)
        try:
            async with lock:
                cached = await asyncio.to_thread(self._lookup_sync, digest, True)
                if cached is not None:
                    return cached
                recipe = await asyncio.to_thread(_read_json, recipe_path, {})
                try:
                    payload = await self.provider(recipe)
                    return await self.commit_png(generation_key, payload)
                except (CgCacheError, OSError, RuntimeError, TimeoutError, ValueError, TypeError):
                    self._negative_until[digest] = time.monotonic() + self.negative_ttl_seconds
                    return {"status": "generation_failed", "generation_key": generation_key}
        finally:
            async with self._singleflight_guard:
                if self._singleflight.get(digest) is lock and not lock.locked():
                    self._singleflight.pop(digest, None)

    async def commit_png(self, generation_key: str, payload: bytes) -> dict[str, Any]:
        digest = self._key_digest(generation_key)
        if not isinstance(payload, bytes) or len(payload) > 20 * 1024 * 1024:
            raise CgCacheError("generated asset size is invalid")
        return await asyncio.to_thread(self._commit_png_sync, digest, payload)

    async def commit_webp(self, generation_key: str, payload: bytes) -> dict[str, Any]:
        digest = self._key_digest(generation_key)
        if not isinstance(payload, bytes) or len(payload) > 20 * 1024 * 1024:
            raise CgCacheError("generated asset size is invalid")
        return await asyncio.to_thread(self._commit_webp_sync, digest, payload)

    async def _lock_for(self, digest: str) -> asyncio.Lock:
        async with self._singleflight_guard:
            return self._singleflight.setdefault(digest, asyncio.Lock())

    @staticmethod
    def _key_digest(generation_key: str) -> str:
        prefix = "sha256:"
        if not isinstance(generation_key, str) or not generation_key.startswith(prefix):
            raise CgCacheError("invalid generation key")
        digest = generation_key[len(prefix):].lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise CgCacheError("invalid generation key")
        return digest

    def _lookup_sync(self, recipe_digest: str, touch: bool) -> dict[str, Any] | None:
        recipe_meta = _read_json(self.metadata_root / f"{recipe_digest}.json", {})
        asset_id = recipe_meta.get("asset_id")
        relative_path = recipe_meta.get("relative_path")
        sha256 = recipe_meta.get("sha256")
        if not all(isinstance(value, str) and value for value in (asset_id, relative_path, sha256)):
            return None
        asset_path = self.static_ui_root / str(relative_path)
        try:
            asset_path.resolve().relative_to(self.static_ui_root.resolve())
        except ValueError:
            return None
        if not asset_path.is_file():
            return None
        if touch:
            index = _read_json(self.index_path, {"assets": {}})
            assets = index.setdefault("assets", {})
            if isinstance(assets, dict) and isinstance(assets.get(asset_id), dict):
                assets[asset_id]["last_access"] = time.time()
                _atomic_write(self.index_path, _canonical_bytes(index))
        return {
            "status": "ready",
            "asset_id": asset_id,
            "sha256": sha256,
            "relative_url": f"/plugin/from_the_heart/ui/{str(relative_path).replace(os.sep, '/')}",
        }

    def _commit_png_sync(self, recipe_digest: str, payload: bytes) -> dict[str, Any]:
        return self._commit_asset_sync(
            recipe_digest,
            payload,
            extension="png",
            mime="image/png",
            dimensions=_validate_png(payload),
        )

    def _commit_webp_sync(self, recipe_digest: str, payload: bytes) -> dict[str, Any]:
        return self._commit_asset_sync(
            recipe_digest,
            payload,
            extension="webp",
            mime="image/webp",
            dimensions=_validate_webp(payload),
        )

    def _commit_asset_sync(
        self,
        recipe_digest: str,
        payload: bytes,
        *,
        extension: str,
        mime: str,
        dimensions: tuple[int, int],
    ) -> dict[str, Any]:
        recipe_path = self.recipe_root / f"{recipe_digest}.json"
        if not recipe_path.is_file():
            raise CgCacheError("generation key was not issued by resolve_interaction")
        width, height = dimensions
        asset_id = hashlib.sha256(payload).hexdigest()
        relative_path = Path("cg") / asset_id[:2] / f"{asset_id}.{extension}"
        asset_path = self.static_ui_root / relative_path
        if not asset_path.exists():
            _atomic_write(asset_path, payload)

        metadata = {
            "asset_id": asset_id,
            "sha256": asset_id,
            "relative_path": str(relative_path),
            "width": width,
            "height": height,
            "mime": mime,
            "recipe_sha256": recipe_digest,
            "created_at": time.time(),
        }
        _atomic_write(self.metadata_root / f"{recipe_digest}.json", _canonical_bytes(metadata))

        index = _read_json(self.index_path, {"assets": {}})
        assets = index.setdefault("assets", {})
        if not isinstance(assets, dict):
            assets = {}
            index["assets"] = assets
        assets[asset_id] = {
            "relative_path": str(relative_path),
            "size": len(payload),
            "last_access": time.time(),
        }
        self._evict_sync(assets, protected_asset_id=asset_id)
        _atomic_write(self.index_path, _canonical_bytes(index))
        return {
            "status": "ready",
            "asset_id": asset_id,
            "sha256": asset_id,
            "relative_url": f"/plugin/from_the_heart/ui/{str(relative_path).replace(os.sep, '/')}",
        }

    def _evict_sync(self, assets: dict[str, Any], *, protected_asset_id: str) -> None:
        def total_size() -> int:
            return sum(
                int(item.get("size", 0))
                for item in assets.values()
                if isinstance(item, dict)
            )

        candidates = sorted(
            (
                (str(asset_id), item)
                for asset_id, item in assets.items()
                if asset_id != protected_asset_id and isinstance(item, dict)
            ),
            key=lambda pair: float(pair[1].get("last_access", 0.0)),
        )
        for asset_id, item in candidates:
            if total_size() <= self.max_bytes:
                break
            relative_path = item.get("relative_path")
            if isinstance(relative_path, str):
                path = self.static_ui_root / relative_path
                try:
                    path.resolve().relative_to(self.static_ui_root.resolve())
                    path.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
            assets.pop(asset_id, None)


__all__ = ["CgCache", "CgCacheError", "ImageProvider"]
