from __future__ import annotations

from collections.abc import Mapping

from plugin.sdk.local_app import (
    TrustedLocalAppPluginContext,
    trusted_local_app_operation,
)
from plugin.sdk.plugin import NekoPluginBase, custom_event


class TrustedLocalAppFixturePlugin(NekoPluginBase):
    @trusted_local_app_operation("knowledge_dungeon.bootstrap")
    async def bootstrap(
        self,
        context: TrustedLocalAppPluginContext,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "context": {
                "app_id": context.app_id,
                "client_id": context.client_id,
                "session_id": context.session_id,
                "scope": context.scope,
                "operation": context.operation,
            },
            "payload": dict(payload),
        }

    @custom_event(event_type="fixture", id="stacked_trusted")
    @trusted_local_app_operation("private.stacked")
    async def stacked_trusted(
        self,
        context: TrustedLocalAppPluginContext,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        return {"context": context.operation, "payload": dict(payload)}
