"""Application service for bounded game interactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from typing import Any, Mapping

from .cg_cache import CgCache
from .central_client import CentralCgClient
from .contracts import ContractError, ContractRepository, validate_request
from .dialogue import DialogueCandidate, DialogueGenerator
from .policy import apply_policy


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    dialogue_enabled: bool = True
    dynamic_cg_enabled: bool = False
    ai_timeout_seconds: float = 2.5
    cg_cache_max_bytes: int = 1024 * 1024 * 1024
    failed_generation_ttl_seconds: float = 600.0
    cg_generation_timeout_seconds: float = 120.0
    central_service_url: str = ""
    central_service_token: str = ""

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        def number(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default

        return cls(
            dialogue_enabled=cls._env_bool("NEKO_FROM_THE_HEART_DIALOGUE_ENABLED", True),
            dynamic_cg_enabled=cls._env_bool("NEKO_FROM_THE_HEART_DYNAMIC_CG_ENABLED", False),
            ai_timeout_seconds=max(0.5, number("NEKO_FROM_THE_HEART_AI_TIMEOUT_SECONDS", 2.5)),
            cg_cache_max_bytes=max(
                1,
                int(number("NEKO_FROM_THE_HEART_CG_CACHE_MAX_BYTES", 1024 * 1024 * 1024)),
            ),
            failed_generation_ttl_seconds=max(
                1.0,
                number("NEKO_FROM_THE_HEART_FAILED_GENERATION_TTL_SECONDS", 600.0),
            ),
            cg_generation_timeout_seconds=max(
                1.0,
                number("NEKO_FROM_THE_HEART_CG_GENERATION_TIMEOUT_SECONDS", 120.0),
            ),
            central_service_url=os.getenv("NEKO_FROM_THE_HEART_CENTRAL_CG_URL", "").strip(),
            central_service_token=os.getenv(
                "NEKO_FROM_THE_HEART_CENTRAL_CG_TOKEN",
                "",
            ).strip(),
        )


class InteractionService:
    def __init__(
        self,
        contracts: ContractRepository,
        cache: CgCache,
        *,
        settings: RuntimeSettings,
        dialogue: DialogueGenerator | None = None,
        central: CentralCgClient | None = None,
    ):
        self.contracts = contracts
        self.cache = cache
        self.settings = settings
        self.dialogue = dialogue or DialogueGenerator(timeout_seconds=settings.ai_timeout_seconds)
        self.central = central

    async def resolve(self, args: Mapping[str, Any]) -> dict[str, Any]:
        node_id = str(args.get("node_id") or "")
        contract = self.contracts.get(node_id)
        player_text = validate_request(contract, args)

        exact_intent = contract.exact_intent(player_text)
        candidate: DialogueCandidate | None = None
        if exact_intent is not None:
            candidate = DialogueCandidate(
                intent_key=exact_intent,
                requested_extra_dish="none",
                lobster_stance="neutral",
                social_key="neutral",
                reply_text="",
            )
        elif self.settings.dialogue_enabled:
            try:
                candidate = await self.dialogue.generate(contract, player_text)
            except Exception:
                # Never log raw player text. The caller receives a deterministic fallback.
                candidate = None

        decision = apply_policy(contract, player_text=player_text, candidate=candidate)
        asset: dict[str, Any] = {
            "status": "builtin" if decision.asset_id != contract.base_asset_id else "fallback",
            "asset_id": decision.asset_id,
            "sha256": decision.asset_sha256,
            "relative_url": None,
        }
        generation: dict[str, Any] = {
            "recommended": False,
            "generation_key": None,
            "reason": "disabled" if not self.settings.dynamic_cg_enabled else "not_required",
        }

        if decision.dynamic_cg_candidate and self.settings.dynamic_cg_enabled:
            recipe = contract.generation_recipe(decision.visual_variant_key)
            generation_key = await self.cache.issue_recipe(recipe)
            cached = await self.cache.lookup(generation_key)
            if cached is not None:
                asset = cached
                generation = {
                    "recommended": False,
                    "generation_key": generation_key,
                    "reason": "cache_hit",
                }
            else:
                generation = {
                    "recommended": self.central is not None,
                    "generation_key": generation_key,
                    "reason": "cache_miss" if self.central is not None else "central_unconfigured",
                }
                if self.central is not None:
                    try:
                        central_state = await asyncio.wait_for(
                            self.central.resolve_recipe(generation_key, recipe),
                            timeout=min(2.0, self.settings.ai_timeout_seconds),
                        )
                        if central_state.get("status") == "ready":
                            asset = central_state
                            generation = {
                                "recommended": False,
                                "generation_key": generation_key,
                                "reason": "central_cache_hit",
                            }
                        elif central_state.get("status") == "failed":
                            generation = {
                                "recommended": False,
                                "generation_key": generation_key,
                                "reason": "central_negative_cache",
                            }
                    except Exception:
                        generation = {
                            "recommended": False,
                            "generation_key": generation_key,
                            "reason": "central_unavailable",
                        }

        return {
            "protocol_version": "1.0",
            "game_id": "from_the_heart",
            "game_version": str(args.get("game_version")),
            "node_id": contract.node_id,
            "node_contract_version": contract.version,
            "base_asset_sha256": contract.base_asset_sha256,
            "interaction_id": str(args.get("interaction_id")),
            "accepted": candidate is not None,
            "intent_key": decision.intent_key,
            "semantic": {
                "requested_extra_dish": decision.requested_extra_dish,
                "lobster_stance": decision.lobster_stance,
                "social_key": decision.social_key,
            },
            "reply_text": decision.reply_text,
            "reaction_key": decision.reaction_key,
            "local_facts": decision.local_facts,
            "visual_signature": decision.visual_signature,
            "visual_variant_key": decision.visual_variant_key,
            "asset": asset,
            "generation": generation,
            "fallback_used": decision.used_fallback,
        }

    async def ensure_cg(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if str(args.get("protocol_version") or "") != "1.0":
            raise ContractError("PROTOCOL_MISMATCH", "unsupported protocol version")
        contract = self.contracts.get(str(args.get("node_id") or ""))
        if str(args.get("node_contract_version") or "") != contract.version:
            raise ContractError("CONTRACT_VERSION_MISMATCH", "node contract version mismatch")
        interaction_id = str(args.get("interaction_id") or "").strip()
        if not interaction_id or len(interaction_id) > 128:
            raise ContractError("INVALID_INTERACTION_ID", "interaction id is invalid")
        generation_key = args.get("generation_key")
        if not isinstance(generation_key, str):
            raise ContractError("INVALID_GENERATION_KEY", "generation key is required")
        if not self.settings.dynamic_cg_enabled:
            asset = {"status": "disabled", "generation_key": generation_key}
        elif self.central is None:
            asset = {"status": "central_unconfigured", "generation_key": generation_key}
        else:
            try:
                asset = await self.central.ensure(
                    generation_key,
                    timeout_seconds=self.settings.cg_generation_timeout_seconds,
                )
            except Exception:
                asset = {"status": "central_unavailable", "generation_key": generation_key}
        return {
            "protocol_version": "1.0",
            "game_id": "from_the_heart",
            "node_id": contract.node_id,
            "node_contract_version": contract.version,
            "interaction_id": interaction_id,
            "generation_key": generation_key,
            "asset": asset,
        }


__all__ = ["InteractionService", "RuntimeSettings"]
