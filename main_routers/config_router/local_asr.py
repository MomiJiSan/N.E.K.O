"""Local ASR preparation lives outside the formal WebSocket session start."""
from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from main_logic.asr_client import (
    get_asr_core_capabilities,
    local_service,
    reload_control,
)
from main_routers.system_router._shared import _validate_local_mutation_request
from utils.config_manager import get_config_manager

from ._shared import router


class PrepareLocalAsrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    enabled: Literal[True]
    provider: Literal["local_streaming"]

    @field_validator("enabled", mode="before")
    @classmethod
    def explicitly_enabled(cls, value):
        if type(value) is not bool or not value:
            raise ValueError("Independent ASR must be explicitly enabled")
        return value


class ReloadLocalAsrRequest(PrepareLocalAsrRequest):
    candidate_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class EmptyReloadControl(BaseModel):
    model_config = ConfigDict(extra="forbid")


_PUBLIC_FIELDS = frozenset({
    "request_id", "operation_id", "lease_id", "state", "ready", "error_code", "code",
    "worker_instance", "instance_id", "timings", "expires_in_seconds", "released", "legacy_eager",
    "candidate_id", "active_changed", "retirement_pending", "late_cancel", "rolled_back",
    "route_generation", "previous_worker_instance", "candidate_worker_instance", "budget",
    "enabled", "candidates",
})


def _response(value: dict) -> dict:
    # URLs, credentials, executable paths and subprocess diagnostics stay local.
    return {"success": True, **{key: value[key] for key in _PUBLIC_FIELDS if key in value}}


def _failure(error: Exception):
    code = getattr(error, "code", "ASR_LOCAL_PREPARE_FAILED")
    if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code):
        code = "ASR_LOCAL_PREPARE_FAILED"
    return JSONResponse({"success": False, "code": code}, status_code=409)


async def _core_allows_independent_asr() -> bool:
    config = await get_config_manager().aget_model_api_config("realtime")
    core = str(config.get("api_type") or "").strip().lower()
    capability = get_asr_core_capabilities(core)
    return bool(capability and capability.supports_independent_asr)


@router.post("/local-asr/prepare")
async def prepare_local_asr(request: Request, payload: PrepareLocalAsrRequest):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        if not await _core_allows_independent_asr():
            return JSONResponse({"success": False, "code": "ASR_INDEPENDENT_UNAVAILABLE"}, status_code=409)
        return _response(await local_service.prepare(str(payload.request_id)))
    except Exception as error:
        return _failure(error)


@router.get("/local-asr/operations/{operation_id}")
async def local_asr_operation(request: Request, operation_id: UUID):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        return _response(await local_service.status(str(operation_id)))
    except Exception as error:
        return _failure(error)


@router.delete("/local-asr/prepare/{request_id}")
async def cancel_local_asr_prepare(request: Request, request_id: UUID):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        await local_service.cancel_request(str(request_id))
        return {"success": True, "released": True}
    except Exception as error:
        return _failure(error)


@router.get("/local-asr/candidates")
async def local_asr_candidates(request: Request):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        return _response(await reload_control.get_control().candidates())
    except Exception as error:
        return _failure(error)


@router.post("/local-asr/reload")
async def reload_local_asr(request: Request, payload: ReloadLocalAsrRequest):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        if not await _core_allows_independent_asr():
            return JSONResponse({"success": False, "code": "ASR_INDEPENDENT_UNAVAILABLE"}, status_code=409)
        return _response(await reload_control.get_control().reload(str(payload.request_id), payload.candidate_id))
    except Exception as error:
        return _failure(error)


@router.get("/local-asr/reloads/{operation_id}")
async def local_asr_reload_operation(request: Request, operation_id: UUID):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        return _response(await reload_control.get_control().status(str(operation_id)))
    except Exception as error:
        return _failure(error)


@router.post("/local-asr/reloads/{operation_id}/{action}")
async def local_asr_reload_action(request: Request, operation_id: UUID,
                                  action: Literal["finalize", "rollback"], payload: EmptyReloadControl):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        return _response(await reload_control.get_control().action(str(operation_id), action))
    except Exception as error:
        return _failure(error)


@router.delete("/local-asr/reloads/{operation_id}")
async def cancel_local_asr_reload(request: Request, operation_id: UUID):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        return _response(await reload_control.get_control().action(str(operation_id), "cancel"))
    except Exception as error:
        return _failure(error)


@router.delete("/local-asr/reload-requests/{request_id}")
async def cancel_local_asr_reload_request(request: Request, request_id: UUID):
    denied = _validate_local_mutation_request(request)
    if denied is not None:
        return denied
    try:
        return _response(await reload_control.get_control().cancel_request(str(request_id)))
    except Exception as error:
        return _failure(error)


async def _close_local_service() -> None:
    await local_service.close()


router.on_startup.append(local_service.startup)
router.on_shutdown.append(_close_local_service)
