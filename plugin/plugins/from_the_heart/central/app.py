"""FastAPI entry point for the deployable central CG cache service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from ..contracts import ContractRepository
from .service import CentralCgService, CentralServiceError
from .storage import CentralRepository, FilesystemObjectStore


@dataclass(frozen=True, slots=True)
class CentralSettings:
    database_path: Path
    object_root: Path
    client_token: str
    worker_token: str
    lease_seconds: float = 180.0
    negative_ttl_seconds: float = 600.0

    @classmethod
    def from_env(cls) -> "CentralSettings":
        data_root = Path(os.getenv("FROM_THE_HEART_CENTRAL_DATA", "data/from_the_heart_cg"))
        return cls(
            database_path=Path(
                os.getenv("FROM_THE_HEART_CENTRAL_DB", str(data_root / "central.sqlite3"))
            ),
            object_root=Path(
                os.getenv("FROM_THE_HEART_CENTRAL_OBJECT_ROOT", str(data_root / "objects"))
            ),
            client_token=os.getenv("FROM_THE_HEART_CENTRAL_CLIENT_TOKEN", ""),
            worker_token=os.getenv("FROM_THE_HEART_CENTRAL_WORKER_TOKEN", ""),
            lease_seconds=float(os.getenv("FROM_THE_HEART_CENTRAL_LEASE_SECONDS", "180")),
            negative_ttl_seconds=float(
                os.getenv("FROM_THE_HEART_CENTRAL_FAILED_TTL_SECONDS", "600")
            ),
        )


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    node_id: str
    node_contract_version: str
    base_asset_sha256: str
    visual_variant_key: str
    generation_key: str | None = None


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str


def create_app(settings: CentralSettings | None = None) -> FastAPI:
    resolved = settings or CentralSettings.from_env()
    plugin_root = Path(__file__).resolve().parents[1]
    service = CentralCgService(
        ContractRepository(plugin_root / "contracts"),
        CentralRepository(resolved.database_path),
        FilesystemObjectStore(resolved.object_root),
        lease_seconds=resolved.lease_seconds,
        negative_ttl_seconds=resolved.negative_ttl_seconds,
    )
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.prepare()
        yield

    app = FastAPI(
        title="From the Heart Central CG Cache",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.cg_service = service

    def require_token(authorization: str | None, expected: str) -> None:
        if not expected or authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid service token")

    @app.post("/v1/cg/resolve")
    async def resolve(
        payload: ResolveRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        require_token(authorization, resolved.client_token)
        try:
            return await service.resolve(payload.model_dump())
        except CentralServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/cg/recipes/{generation_key}")
    async def status(
        generation_key: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        require_token(authorization, resolved.client_token)
        try:
            return await service.status(generation_key)
        except CentralServiceError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/v1/cg/jobs/{generation_key}/claim")
    async def claim(
        generation_key: str,
        payload: ClaimRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        require_token(authorization, resolved.worker_token)
        try:
            return await service.claim(generation_key, payload.worker_id)
        except CentralServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.put("/v1/cg/jobs/{generation_key}/asset")
    async def complete(
        generation_key: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_lease_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        require_token(authorization, resolved.worker_token)
        try:
            payload = await request.body()
            return await service.complete(
                generation_key,
                lease_token=x_lease_token or "",
                payload=payload,
            )
        except CentralServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/cg/jobs/{generation_key}/fail")
    async def fail(
        generation_key: str,
        authorization: str | None = Header(default=None),
        x_lease_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        require_token(authorization, resolved.worker_token)
        try:
            return await service.fail(generation_key, lease_token=x_lease_token or "")
        except CentralServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/cg/assets/{asset_id}")
    async def asset(
        asset_id: str,
        authorization: str | None = Header(default=None),
    ) -> FileResponse:
        require_token(authorization, resolved.client_token)
        try:
            path = await service.asset_path(asset_id)
        except CentralServiceError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path, media_type="image/webp", filename=f"{asset_id}.webp")

    return app


app = create_app()


__all__ = ["CentralSettings", "app", "create_app"]
