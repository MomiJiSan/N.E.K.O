"""
插件管理路由
"""
import secrets
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict
from starlette.responses import JSONResponse

from config import AUTOSTART_CSRF_TOKEN

from plugin.logging_config import get_logger
from plugin.server.application.plugins import (
    PluginLifecycleService,
    PluginQueryService,
    PluginRegistryService,
)
from plugin.server.domain.errors import ServerDomainError
from plugin.server.infrastructure.auth import require_admin
from plugin.server.infrastructure.error_mapping import raise_http_from_domain
from plugin.server.lifecycle import ensure_plugin_messaging_started
from plugin.server.local_app_bridge.errors import LocalAppBridgeError
from plugin.server.local_app_bridge.runtime import launch_registered_local_app
from utils.host_origin_guard import is_http_browser_origin_allowed

router = APIRouter()
logger = get_logger("server.routes.plugins")
query_service = PluginQueryService()
lifecycle_service = PluginLifecycleService()
registry_service = PluginRegistryService()

_LOCAL_APP_CSRF_HEADER = "X-CSRF-Token"


class LocalAppLaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_id: str


def _require_trusted_product_ui(request: Request) -> None:
    # Native/no-Origin callers use the in-process launch API. The HTTP surface
    # exists only for the N.E.K.O product UI and must not become a generic
    # loopback process launcher.
    if not (request.headers.get("origin") or request.headers.get("referer")):
        raise HTTPException(
            status_code=403,
            detail={"code": "local_app_ui_origin_required"},
        )
    if not is_http_browser_origin_allowed(request.scope):
        raise HTTPException(
            status_code=403,
            detail={"code": "local_app_ui_origin_forbidden"},
        )


@router.get("/local-app/ui-token")
async def local_app_ui_token(request: Request) -> JSONResponse:
    _require_trusted_product_ui(request)
    return JSONResponse(
        {"token": AUTOSTART_CSRF_TOKEN},
        headers={"Cache-Control": "no-store"},
    )


@router.post("/local-app/launch")
async def launch_local_app_endpoint(
    request: Request,
    payload: LocalAppLaunchRequest,
    _: str = require_admin,
) -> dict[str, object]:
    _require_trusted_product_ui(request)
    provided_token = request.headers.get(_LOCAL_APP_CSRF_HEADER, "")
    if not AUTOSTART_CSRF_TOKEN or not secrets.compare_digest(
        provided_token, AUTOSTART_CSRF_TOKEN
    ):
        raise HTTPException(
            status_code=403,
            detail={"code": "local_app_ui_csrf_failed"},
        )
    try:
        result = await launch_registered_local_app(payload.app_id)
    except LocalAppBridgeError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code},
        ) from exc
    except (LookupError, OSError, RuntimeError):
        logger.warning(
            "local app launch unavailable: app_id={}, error=safe_redacted",
            payload.app_id,
        )
        raise HTTPException(
            status_code=503,
            detail={"code": "local_app_unavailable"},
        ) from None
    return {"success": True, "app_id": result.app_id}


@router.get("/plugin/status")
async def plugin_status(plugin_id: Optional[str] = Query(default=None)) -> dict[str, object]:
    try:
        return await query_service.get_plugin_status(plugin_id)
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)

@router.get("/plugins")
async def list_plugins(locale: Optional[str] = Query(default=None)) -> dict[str, object]:
    try:
        return await query_service.list_plugins(locale=locale)
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)


@router.post("/plugin/{plugin_id}/start")
async def start_plugin_endpoint(plugin_id: str, _: str = require_admin) -> dict[str, object]:
    try:
        await ensure_plugin_messaging_started()
        return await lifecycle_service.start_plugin(plugin_id, persist_user_intent=True)
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)


@router.post("/plugin/{plugin_id}/refresh")
async def refresh_plugin_endpoint(plugin_id: str, _: str = require_admin) -> dict[str, object]:
    try:
        return await registry_service.refresh_plugin(plugin_id)
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)


@router.post("/plugin/{plugin_id}/stop")
async def stop_plugin_endpoint(plugin_id: str, _: str = require_admin) -> dict[str, object]:
    try:
        return await lifecycle_service.stop_plugin(plugin_id, persist_user_intent=True)
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)


@router.delete("/plugin/{plugin_id}")
async def delete_plugin_endpoint(plugin_id: str, _: str = require_admin) -> dict[str, object]:
    try:
        return await lifecycle_service.delete_plugin(plugin_id)
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)


@router.post("/plugins/refresh")
async def refresh_plugins_endpoint(_: str = require_admin) -> dict[str, object]:
    try:
        return await registry_service.refresh_registry()
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)


@router.post("/plugin/{plugin_id}/reload")
async def reload_plugin_endpoint(plugin_id: str, _: str = require_admin) -> dict[str, object]:
    try:
        return await lifecycle_service.reload_plugin(plugin_id)
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)


@router.post("/plugins/reload")
async def reload_all_plugins_endpoint(_: str = require_admin) -> dict[str, object]:
    """
    重载所有插件
    
    停止所有运行中的插件，然后重新加载。
    用于前端全局重载按钮。
    """
    try:
        return await lifecycle_service.reload_all_plugins()
    except ServerDomainError as error:
        raise_http_from_domain(error, logger=logger)
