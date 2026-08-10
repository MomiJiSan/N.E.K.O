from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from plugin.plugins.from_the_heart.central.app import CentralSettings, create_app
from plugin.plugins.from_the_heart.central.service import CentralCgService, CentralServiceError
from plugin.plugins.from_the_heart.central.storage import CentralRepository, FilesystemObjectStore
from plugin.plugins.from_the_heart.central_client import CentralCgClient
from plugin.plugins.from_the_heart.cg_cache import CgCache
from plugin.plugins.from_the_heart.contracts import ContractRepository


PLUGIN_DIR = Path(__file__).resolve().parents[3] / "plugins" / "from_the_heart"
NODE_ID = "ch2.restaurant.favorite_food"
BASE_SHA = "01ce8fd74cd99fad908a8a3d7af9021fbc6735c22ebe3bd28d375e36d52d581d"


def resolve_payload(**overrides):
    payload = {
        "game_id": "from_the_heart",
        "node_id": NODE_ID,
        "node_contract_version": "1.1.0",
        "base_asset_sha256": BASE_SHA,
        "visual_variant_key": "restaurant_extra_crab_playful",
    }
    payload.update(overrides)
    return payload


def webp_asset() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1920, 1080), color=(32, 96, 160)).save(
        buffer,
        format="WEBP",
        quality=70,
    )
    return buffer.getvalue()


async def make_service(tmp_path: Path) -> CentralCgService:
    service = CentralCgService(
        ContractRepository(PLUGIN_DIR / "contracts"),
        CentralRepository(tmp_path / "central.sqlite3"),
        FilesystemObjectStore(tmp_path / "objects"),
        lease_seconds=60,
        negative_ttl_seconds=60,
    )
    await service.prepare()
    return service


@pytest.mark.asyncio
async def test_central_service_recomputes_recipe_key_and_deduplicates_concurrency(tmp_path):
    service = await make_service(tmp_path)
    first, second = await asyncio.gather(
        service.resolve(resolve_payload()),
        service.resolve(resolve_payload()),
    )
    assert first["generation_key"] == second["generation_key"]
    assert first["status"] == second["status"] == "queued"

    with pytest.raises(CentralServiceError, match="client generation key mismatch"):
        await service.resolve(resolve_payload(generation_key="sha256:" + "f" * 64))
    with pytest.raises(CentralServiceError, match="unknown visual variant"):
        await service.resolve(resolve_payload(visual_variant_key="player_supplied_prompt"))


@pytest.mark.asyncio
async def test_worker_lease_is_private_and_ready_asset_is_content_addressed(tmp_path):
    service = await make_service(tmp_path)
    queued = await service.resolve(resolve_payload())
    generation_key = queued["generation_key"]
    claimed = await service.claim(generation_key, "worker-a")
    assert claimed["status"] == "generating"
    assert claimed["recipe"]["visual_variant_key"] == "restaurant_extra_crab_playful"
    assert claimed["recipe"]["visual_signature"]["table_extra"] == "crab_plate"

    competing = await service.claim(generation_key, "worker-b")
    assert competing["status"] == "busy"
    assert competing["lease_token"] is None
    assert competing["recipe"] is None

    ready = await service.complete(
        generation_key,
        lease_token=claimed["lease_token"],
        payload=webp_asset(),
    )
    assert ready["status"] == "ready"
    assert ready["asset"]["asset_id"] == ready["asset"]["sha256"]
    assert ready["asset"]["download_path"].endswith(ready["asset"]["asset_id"])
    assert (await service.asset_path(ready["asset"]["asset_id"])).is_file()

    repeated = await service.resolve(resolve_payload())
    assert repeated["asset"] == ready["asset"]


def test_central_http_api_separates_client_and_worker_authority(tmp_path):
    app = create_app(
        CentralSettings(
            database_path=tmp_path / "central.sqlite3",
            object_root=tmp_path / "objects",
            client_token="client-secret",
            worker_token="worker-secret",
        )
    )
    with TestClient(app) as client:
        assert client.post("/v1/cg/resolve", json=resolve_payload()).status_code == 401
        created = client.post(
            "/v1/cg/resolve",
            json=resolve_payload(),
            headers={"Authorization": "Bearer client-secret"},
        )
        assert created.status_code == 200
        generation_key = created.json()["generation_key"]

        claim = client.post(
            f"/v1/cg/jobs/{generation_key}/claim",
            json={"worker_id": "worker-a"},
            headers={"Authorization": "Bearer worker-secret"},
        )
        assert claim.status_code == 200
        lease_token = claim.json()["lease_token"]

        uploaded = client.put(
            f"/v1/cg/jobs/{generation_key}/asset",
            content=webp_asset(),
            headers={
                "Authorization": "Bearer worker-secret",
                "X-Lease-Token": lease_token,
                "Content-Type": "image/webp",
            },
        )
        assert uploaded.status_code == 200
        asset = uploaded.json()["asset"]

        downloaded = client.get(
            asset["download_path"],
            headers={"Authorization": "Bearer client-secret"},
        )
        assert downloaded.status_code == 200
        assert downloaded.headers["content-type"].startswith("image/webp")
        assert downloaded.content == webp_asset()


@pytest.mark.asyncio
async def test_local_neko_client_mirrors_ready_central_webp(tmp_path):
    app = create_app(
        CentralSettings(
            database_path=tmp_path / "central.sqlite3",
            object_root=tmp_path / "objects",
            client_token="client-secret",
            worker_token="worker-secret",
        )
    )
    await app.state.cg_service.prepare()
    contract = ContractRepository(PLUGIN_DIR / "contracts").get(NODE_ID)
    recipe = contract.generation_recipe("restaurant_extra_crab_playful")

    source_index = tmp_path / "index.html"
    source_index.write_text("ready", encoding="utf-8")
    local_cache = CgCache(tmp_path / "local-static")
    local_cache.prepare(source_index)
    generation_key = await local_cache.issue_recipe(recipe)

    queued = await app.state.cg_service.resolve(resolve_payload(generation_key=generation_key))
    claimed = await app.state.cg_service.claim(queued["generation_key"], "worker-a")
    await app.state.cg_service.complete(
        queued["generation_key"],
        lease_token=claimed["lease_token"],
        payload=webp_asset(),
    )

    central = CentralCgClient(
        "http://127.0.0.1",
        "client-secret",
        local_cache,
        transport=httpx.ASGITransport(app=app),
    )
    mirrored = await central.resolve_recipe(generation_key, recipe)
    assert mirrored["status"] == "ready"
    assert mirrored["relative_url"].endswith(".webp")
    assert await local_cache.lookup(generation_key) == mirrored
