"""Dedicated, game-initiated bridge for NEKO: From the Heart."""

from __future__ import annotations

import asyncio
from typing import Any

from plugin.sdk.plugin import Err, NekoPluginBase, Ok, lifecycle, neko_plugin, plugin_entry

from .cg_cache import CgCache, CgCacheError
from .central_client import CentralCgClient, CentralClientError
from .contracts import ContractError, ContractRepository
from .service import InteractionService, RuntimeSettings


@neko_plugin
class FromTheHeartPlugin(NekoPluginBase):
    name = "from_the_heart"
    passive = True

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.settings = RuntimeSettings.from_env()
        self.contracts = ContractRepository(self.config_dir / "contracts")
        self.cg_cache = CgCache(
            self.data_path("static_ui"),
            state_root=self.data_path("cg_cache"),
            max_bytes=self.settings.cg_cache_max_bytes,
            negative_ttl_seconds=self.settings.failed_generation_ttl_seconds,
            provider=None,
        )
        self.central_cg = None
        if self.settings.central_service_url and self.settings.central_service_token:
            try:
                self.central_cg = CentralCgClient(
                    self.settings.central_service_url,
                    self.settings.central_service_token,
                    self.cg_cache,
                )
            except CentralClientError:
                self.central_cg = None
        self.interactions = InteractionService(
            self.contracts,
            self.cg_cache,
            settings=self.settings,
            central=self.central_cg,
        )

    @lifecycle(id="startup")
    async def startup(self, **_: Any):
        await asyncio.to_thread(
            self.cg_cache.prepare,
            self.config_dir / "static" / "index.html",
        )
        self._register_writable_static_ui()
        return Ok({
            "status": "ready",
            "dialogue_enabled": self.settings.dialogue_enabled,
            "dynamic_cg_enabled": self.settings.dynamic_cg_enabled,
        })

    @lifecycle(id="shutdown")
    async def shutdown(self, **_: Any):
        return Ok({"status": "stopped"})

    def _register_writable_static_ui(self) -> None:
        self._static_ui_config = {
            "enabled": True,
            "directory": str(self.data_path("static_ui")),
            "index_file": "index.html",
            "cache_control": "public, max-age=31536000, immutable",
            "plugin_id": self.plugin_id,
        }
        self._notify_static_ui_registered(self._static_ui_config)

    @plugin_entry(
        id="resolve_interaction",
        name="Resolve From the Heart interaction",
        description="Resolve one bounded game dialogue slot without changing story state.",
        input_schema={
            "type": "object",
            "required": [
                "protocol_version",
                "game_id",
                "game_version",
                "node_id",
                "node_contract_version",
                "interaction_id",
                "base_asset_sha256",
                "player_text",
            ],
            "properties": {
                "protocol_version": {"type": "string"},
                "game_id": {"type": "string"},
                "game_version": {"type": "string"},
                "node_id": {"type": "string"},
                "node_contract_version": {"type": "string"},
                "interaction_id": {"type": "string"},
                "base_asset_sha256": {"type": "string"},
                "player_text": {"type": "string", "maxLength": 200},
                "safe_context": {"type": "object"},
            },
            "additionalProperties": False,
        },
    )
    async def resolve_interaction(self, **kwargs: Any):
        try:
            return Ok(await self.interactions.resolve(kwargs))
        except ContractError as error:
            return Err({"code": error.code, "message": str(error)})
        except (CgCacheError, OSError, RuntimeError, ValueError, TypeError) as error:
            return Err({"code": "INTERACTION_FAILED", "message": str(error)})

    @plugin_entry(
        id="ensure_cg",
        name="Ensure From the Heart CG",
        description="Resolve or generate a previously issued content-addressed CG recipe.",
        input_schema={
            "type": "object",
            "required": [
                "protocol_version",
                "interaction_id",
                "node_id",
                "node_contract_version",
                "generation_key",
            ],
            "properties": {
                "protocol_version": {"type": "string"},
                "interaction_id": {"type": "string"},
                "node_id": {"type": "string"},
                "node_contract_version": {"type": "string"},
                "generation_key": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )
    async def ensure_cg(self, **kwargs: Any):
        try:
            return Ok(await self.interactions.ensure_cg(kwargs))
        except ContractError as error:
            return Err({"code": error.code, "message": str(error)})
        except (CgCacheError, OSError, RuntimeError, ValueError, TypeError) as error:
            return Err({"code": "CG_FAILED", "message": str(error)})


__all__ = ["FromTheHeartPlugin"]
