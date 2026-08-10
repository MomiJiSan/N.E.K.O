"""Database-controlled recipe state and content-addressed WebP storage."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping
import uuid

from PIL import Image


class CentralStorageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecipeRecord:
    generation_key: str
    status: str
    recipe: dict[str, Any]
    asset_id: str | None
    failure_until: float | None
    lease_token: str | None
    lease_owner: str | None
    lease_expires_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CentralRepository:
    """SQLite reference backend; uniqueness and leases remain server-owned."""

    def __init__(self, database_path: Path):
        self.database_path = database_path

    async def prepare(self) -> None:
        await asyncio.to_thread(self._prepare_sync)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _prepare_sync(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS cg_assets (
                    asset_id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    storage_key TEXT NOT NULL,
                    mime TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cg_recipes (
                    generation_key TEXT PRIMARY KEY,
                    recipe_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    asset_id TEXT,
                    failure_until REAL,
                    lease_token TEXT,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES cg_assets(asset_id)
                );
                """
            )

    async def resolve(self, generation_key: str, recipe: Mapping[str, Any]) -> RecipeRecord:
        return await asyncio.to_thread(self._resolve_sync, generation_key, dict(recipe))

    def _resolve_sync(self, generation_key: str, recipe: dict[str, Any]) -> RecipeRecord:
        recipe_json = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cg_recipes WHERE generation_key = ?",
                (generation_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO cg_recipes
                    (generation_key, recipe_json, status, created_at, updated_at)
                    VALUES (?, ?, 'queued', ?, ?)""",
                    (generation_key, recipe_json, now, now),
                )
            elif str(row["recipe_json"]) != recipe_json:
                raise CentralStorageError("generation key recipe mismatch")
            elif row["status"] == "failed" and float(row["failure_until"] or 0.0) <= now:
                connection.execute(
                    """UPDATE cg_recipes SET status = 'queued', failure_until = NULL,
                    lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL,
                    updated_at = ? WHERE generation_key = ?""",
                    (now, generation_key),
                )
            connection.commit()
        return self._get_sync(generation_key)

    async def get(self, generation_key: str) -> RecipeRecord | None:
        return await asyncio.to_thread(self._get_optional_sync, generation_key)

    def _get_optional_sync(self, generation_key: str) -> RecipeRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cg_recipes WHERE generation_key = ?",
                (generation_key,),
            ).fetchone()
        return self._record(row) if row is not None else None

    def _get_sync(self, generation_key: str) -> RecipeRecord:
        record = self._get_optional_sync(generation_key)
        if record is None:
            raise CentralStorageError("unknown generation key")
        return record

    async def claim(
        self,
        generation_key: str,
        *,
        worker_id: str,
        lease_seconds: float,
    ) -> RecipeRecord:
        return await asyncio.to_thread(
            self._claim_sync,
            generation_key,
            worker_id,
            lease_seconds,
        )

    def _claim_sync(
        self,
        generation_key: str,
        worker_id: str,
        lease_seconds: float,
    ) -> RecipeRecord:
        now = time.time()
        token = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cg_recipes WHERE generation_key = ?",
                (generation_key,),
            ).fetchone()
            if row is None:
                raise CentralStorageError("unknown generation key")
            status = str(row["status"])
            expired_lease = status == "generating" and float(row["lease_expires_at"] or 0.0) <= now
            retryable_failure = status == "failed" and float(row["failure_until"] or 0.0) <= now
            if status == "queued" or expired_lease or retryable_failure:
                connection.execute(
                    """UPDATE cg_recipes SET status = 'generating', lease_token = ?,
                    lease_owner = ?, lease_expires_at = ?, failure_until = NULL,
                    updated_at = ? WHERE generation_key = ?""",
                    (token, worker_id, now + max(1.0, lease_seconds), now, generation_key),
                )
            connection.commit()
        return self._get_sync(generation_key)

    async def complete(
        self,
        generation_key: str,
        *,
        lease_token: str,
        asset: Mapping[str, Any],
    ) -> RecipeRecord:
        return await asyncio.to_thread(
            self._complete_sync,
            generation_key,
            lease_token,
            dict(asset),
        )

    def _complete_sync(
        self,
        generation_key: str,
        lease_token: str,
        asset: dict[str, Any],
    ) -> RecipeRecord:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, lease_token, lease_expires_at FROM cg_recipes WHERE generation_key = ?",
                (generation_key,),
            ).fetchone()
            if row is None or row["status"] != "generating":
                raise CentralStorageError("generation is not claimed")
            if row["lease_token"] != lease_token or float(row["lease_expires_at"] or 0.0) <= now:
                raise CentralStorageError("generation lease is invalid or expired")
            connection.execute(
                """INSERT OR IGNORE INTO cg_assets
                (asset_id, sha256, storage_key, mime, width, height, size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    asset["asset_id"],
                    asset["sha256"],
                    asset["storage_key"],
                    asset["mime"],
                    asset["width"],
                    asset["height"],
                    asset["size"],
                    now,
                ),
            )
            connection.execute(
                """UPDATE cg_recipes SET status = 'ready', asset_id = ?,
                lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL,
                updated_at = ? WHERE generation_key = ?""",
                (asset["asset_id"], now, generation_key),
            )
            connection.commit()
        return self._get_sync(generation_key)

    async def fail(
        self,
        generation_key: str,
        *,
        lease_token: str,
        negative_ttl_seconds: float,
    ) -> RecipeRecord:
        return await asyncio.to_thread(
            self._fail_sync,
            generation_key,
            lease_token,
            negative_ttl_seconds,
        )

    def _fail_sync(
        self,
        generation_key: str,
        lease_token: str,
        negative_ttl_seconds: float,
    ) -> RecipeRecord:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE cg_recipes SET status = 'failed', failure_until = ?,
                lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL,
                updated_at = ? WHERE generation_key = ? AND status = 'generating'
                AND lease_token = ?""",
                (now + max(1.0, negative_ttl_seconds), now, generation_key, lease_token),
            )
            if cursor.rowcount != 1:
                raise CentralStorageError("generation lease is invalid")
            connection.commit()
        return self._get_sync(generation_key)

    async def asset(self, asset_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._asset_sync, asset_id)

    def _asset_sync(self, asset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cg_assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _record(row: sqlite3.Row) -> RecipeRecord:
        recipe = json.loads(str(row["recipe_json"]))
        if not isinstance(recipe, dict):
            raise CentralStorageError("stored recipe is invalid")
        return RecipeRecord(
            generation_key=str(row["generation_key"]),
            status=str(row["status"]),
            recipe=recipe,
            asset_id=str(row["asset_id"]) if row["asset_id"] else None,
            failure_until=float(row["failure_until"]) if row["failure_until"] else None,
            lease_token=str(row["lease_token"]) if row["lease_token"] else None,
            lease_owner=str(row["lease_owner"]) if row["lease_owner"] else None,
            lease_expires_at=float(row["lease_expires_at"]) if row["lease_expires_at"] else None,
        )


class FilesystemObjectStore:
    """Content-addressed reference object store with atomic writes."""

    def __init__(self, root: Path, *, max_bytes: int = 20 * 1024 * 1024):
        self.root = root
        self.max_bytes = max(1, int(max_bytes))

    async def prepare(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def put_webp(self, payload: bytes) -> dict[str, Any]:
        return await asyncio.to_thread(self._put_webp_sync, payload)

    def _put_webp_sync(self, payload: bytes) -> dict[str, Any]:
        if not isinstance(payload, bytes) or not payload or len(payload) > self.max_bytes:
            raise CentralStorageError("asset size is invalid")
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                image_format = image.format
                width, height = image.size
        except Exception as error:
            raise CentralStorageError("asset is not a valid image") from error
        if image_format != "WEBP" or (width, height) != (1920, 1080):
            raise CentralStorageError("asset must be a 1920x1080 WebP")
        asset_id = hashlib.sha256(payload).hexdigest()
        storage_key = f"cg/{asset_id[:2]}/{asset_id}.webp"
        target = self.path_for(storage_key)
        target_is_valid = False
        if target.exists():
            try:
                target_is_valid = hashlib.sha256(target.read_bytes()).hexdigest() == asset_id
            except OSError:
                target_is_valid = False
        if not target_is_valid:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.parent / f".{uuid.uuid4().hex[:16]}.tmp"
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        return {
            "asset_id": asset_id,
            "sha256": asset_id,
            "storage_key": storage_key,
            "mime": "image/webp",
            "width": width,
            "height": height,
            "size": len(payload),
        }

    def path_for(self, storage_key: str) -> Path:
        path = self.root / storage_key
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise CentralStorageError("object storage path is invalid") from error
        return path


__all__ = [
    "CentralRepository",
    "CentralStorageError",
    "FilesystemObjectStore",
    "RecipeRecord",
]
