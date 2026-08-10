"""Authoritative recipe normalization and central cache state transitions."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Mapping

from ..contracts import ContractError, ContractRepository
from .storage import CentralRepository, CentralStorageError, FilesystemObjectStore, RecipeRecord


class CentralServiceError(ValueError):
    pass


def canonical_generation_key(recipe: Mapping[str, Any]) -> str:
    payload = json.dumps(
        recipe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class CentralCgService:
    def __init__(
        self,
        contracts: ContractRepository,
        repository: CentralRepository,
        objects: FilesystemObjectStore,
        *,
        lease_seconds: float = 180.0,
        negative_ttl_seconds: float = 600.0,
    ):
        self.contracts = contracts
        self.repository = repository
        self.objects = objects
        self.lease_seconds = max(1.0, float(lease_seconds))
        self.negative_ttl_seconds = max(1.0, float(negative_ttl_seconds))

    async def prepare(self) -> None:
        await self.repository.prepare()
        await self.objects.prepare()

    async def resolve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            contract = self.contracts.get(str(payload.get("node_id") or ""))
            if str(payload.get("game_id") or "") != "from_the_heart":
                raise CentralServiceError("unsupported game id")
            if str(payload.get("node_contract_version") or "") != contract.version:
                raise CentralServiceError("node contract version mismatch")
            if str(payload.get("base_asset_sha256") or "").lower() != contract.base_asset_sha256:
                raise CentralServiceError("base asset sha256 mismatch")
            variant_key = str(payload.get("visual_variant_key") or "")
            recipe = contract.generation_recipe(variant_key)
        except ContractError as error:
            raise CentralServiceError(str(error)) from error

        generation_key = canonical_generation_key(recipe)
        submitted_key = payload.get("generation_key")
        if submitted_key is not None and submitted_key != generation_key:
            raise CentralServiceError("client generation key mismatch")
        record = await self.repository.resolve(generation_key, recipe)
        return await self._public_record(record)

    async def status(self, generation_key: str) -> dict[str, Any]:
        record = await self.repository.get(generation_key)
        if record is None:
            raise CentralServiceError("unknown generation key")
        return await self._public_record(record)

    async def claim(self, generation_key: str, worker_id: str) -> dict[str, Any]:
        if not worker_id.strip() or len(worker_id) > 128:
            raise CentralServiceError("worker id is invalid")
        try:
            record = await self.repository.claim(
                generation_key,
                worker_id=worker_id.strip(),
                lease_seconds=self.lease_seconds,
            )
        except CentralStorageError as error:
            raise CentralServiceError(str(error)) from error
        owns_lease = record.lease_owner == worker_id.strip()
        return {
            "generation_key": record.generation_key,
            "status": record.status if record.status != "generating" or owns_lease else "busy",
            "recipe": record.recipe if owns_lease else None,
            "lease_token": record.lease_token if owns_lease else None,
            "lease_expires_at": record.lease_expires_at,
        }

    async def complete(
        self,
        generation_key: str,
        *,
        lease_token: str,
        payload: bytes,
    ) -> dict[str, Any]:
        if not lease_token:
            raise CentralServiceError("lease token is required")
        try:
            asset = await self.objects.put_webp(payload)
            record = await self.repository.complete(
                generation_key,
                lease_token=lease_token,
                asset=asset,
            )
        except CentralStorageError as error:
            raise CentralServiceError(str(error)) from error
        return await self._public_record(record)

    async def fail(self, generation_key: str, *, lease_token: str) -> dict[str, Any]:
        try:
            record = await self.repository.fail(
                generation_key,
                lease_token=lease_token,
                negative_ttl_seconds=self.negative_ttl_seconds,
            )
        except CentralStorageError as error:
            raise CentralServiceError(str(error)) from error
        return await self._public_record(record)

    async def asset_path(self, asset_id: str):
        if len(asset_id) != 64 or any(char not in "0123456789abcdef" for char in asset_id):
            raise CentralServiceError("asset id is invalid")
        asset = await self.repository.asset(asset_id)
        if asset is None:
            raise CentralServiceError("asset not found")
        path = self.objects.path_for(str(asset["storage_key"]))
        if not await asyncio.to_thread(path.is_file):
            raise CentralServiceError("asset object is missing")
        return path

    async def _public_record(self, record: RecipeRecord) -> dict[str, Any]:
        result: dict[str, Any] = {
            "generation_key": record.generation_key,
            "status": record.status,
            "asset": None,
            "failure_until": record.failure_until,
        }
        if record.status != "ready" or record.asset_id is None:
            return result
        asset = await self.repository.asset(record.asset_id)
        if asset is None:
            raise CentralServiceError("ready recipe has no asset metadata")
        result["asset"] = {
            "asset_id": asset["asset_id"],
            "sha256": asset["sha256"],
            "mime": asset["mime"],
            "width": asset["width"],
            "height": asset["height"],
            "size": asset["size"],
            "download_path": f"/v1/cg/assets/{asset['asset_id']}",
        }
        return result


__all__ = ["CentralCgService", "CentralServiceError", "canonical_generation_key"]
