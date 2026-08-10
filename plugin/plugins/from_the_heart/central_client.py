"""Async, fail-closed proxy from the local plugin to the central CG service."""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx

from .cg_cache import CgCache


class CentralClientError(ValueError):
    pass


def _validate_origin(origin: str) -> str:
    normalized = origin.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CentralClientError("central service URL is invalid")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise CentralClientError("central service URL must be an origin")
    if parsed.scheme == "http":
        host = parsed.hostname.lower()
        if host != "localhost":
            try:
                if not ipaddress.ip_address(host).is_loopback:
                    raise CentralClientError("non-loopback central service must use HTTPS")
            except ValueError as error:
                raise CentralClientError("non-loopback central service must use HTTPS") from error
    return normalized


class CentralCgClient:
    def __init__(
        self,
        origin: str,
        token: str,
        cache: CgCache,
        *,
        request_timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.origin = _validate_origin(origin)
        self.token = token.strip()
        if not self.token:
            raise CentralClientError("central service token is required")
        self.cache = cache
        self.request_timeout_seconds = max(0.5, float(request_timeout_seconds))
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self.transport = transport

    async def resolve_recipe(
        self,
        generation_key: str,
        recipe: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "game_id": recipe.get("game_id"),
            "node_id": recipe.get("node_id"),
            "node_contract_version": recipe.get("node_contract_version"),
            "base_asset_sha256": recipe.get("base_asset_sha256"),
            "visual_variant_key": recipe.get("visual_variant_key"),
            "generation_key": generation_key,
        }
        response = await self._json("POST", "/v1/cg/resolve", json_payload=payload)
        return await self._mirror_if_ready(generation_key, response)

    async def ensure(self, generation_key: str, *, timeout_seconds: float) -> dict[str, Any]:
        cached = await self.cache.lookup(generation_key)
        if cached is not None:
            return cached
        recipe = await self.cache.recipe(generation_key)
        state = await self.resolve_recipe(generation_key, recipe)
        if state.get("status") == "ready":
            return state
        deadline = asyncio.get_running_loop().time() + max(0.5, float(timeout_seconds))
        while state.get("status") in {"queued", "generating"}:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return {"status": "pending", "generation_key": generation_key}
            await asyncio.sleep(min(self.poll_interval_seconds, remaining))
            state = await self._json(
                "GET",
                f"/v1/cg/recipes/{generation_key}",
            )
            state = await self._mirror_if_ready(generation_key, state)
        return state

    async def _mirror_if_ready(
        self,
        generation_key: str,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        if state.get("generation_key") != generation_key:
            raise CentralClientError("central generation key mismatch")
        if state.get("status") != "ready":
            return dict(state)
        asset = state.get("asset")
        if not isinstance(asset, Mapping):
            raise CentralClientError("central asset descriptor is missing")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or asset.get("sha256") != asset_id:
            raise CentralClientError("central asset hash is invalid")
        if asset.get("mime") != "image/webp" or (asset.get("width"), asset.get("height")) != (1920, 1080):
            raise CentralClientError("central asset profile is invalid")
        download_path = asset.get("download_path")
        expected_path = f"/v1/cg/assets/{asset_id}"
        if download_path != expected_path:
            raise CentralClientError("central asset path is invalid")
        payload = await self._bytes("GET", expected_path)
        mirrored = await self.cache.commit_webp(generation_key, payload)
        if mirrored.get("asset_id") != asset_id:
            raise CentralClientError("central asset content hash mismatch")
        return mirrored

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.origin,
            headers={"Authorization": f"Bearer {self.token}"},
            follow_redirects=False,
            timeout=self.request_timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.request(method, path, json=json_payload)
        if response.is_redirect:
            raise CentralClientError("central service redirects are disabled")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise CentralClientError("central response must be an object")
        return payload

    async def _bytes(self, method: str, path: str) -> bytes:
        async with httpx.AsyncClient(
            base_url=self.origin,
            headers={"Authorization": f"Bearer {self.token}"},
            follow_redirects=False,
            timeout=self.request_timeout_seconds,
            transport=self.transport,
        ) as client:
            async with client.stream(method, path) as response:
                if response.is_redirect:
                    raise CentralClientError("central service redirects are disabled")
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if content_type.split(";", 1)[0].strip().lower() != "image/webp":
                    raise CentralClientError("central asset MIME is invalid")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise CentralClientError("central asset is too large")
                    chunks.append(chunk)
        return b"".join(chunks)


__all__ = ["CentralCgClient", "CentralClientError"]
