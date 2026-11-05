import pytest

from main_logic.asr_client import local_service as local
from tests.unit.test_asr_local_service import Service, clean_env, managed  # noqa: F401


@pytest.fixture(autouse=True)
def clear_reload_env(monkeypatch):
    for name in ("NEKO_ASR_CATALOG", "NEKO_ASR_STATE_DIR", "NEKO_ASR_MEMORY_BUDGET_MB"):
        monkeypatch.delenv(name, raising=False)


def configure(monkeypatch):
    for name in local._MANAGED_FIELDS:
        monkeypatch.setenv(name, "C:\\trusted")
    monkeypatch.setenv("NEKO_ASR_CATALOG", "C:\\catalog.json")
    monkeypatch.setenv("NEKO_ASR_STATE_DIR", "C:\\state")


def test_catalog_config_requires_managed_absolute_pair(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_CATALOG", "C:\\catalog.json")
    assert not local.configuration_is_available()
    configure(monkeypatch)
    config = local.configuration()
    assert config.catalog == "C:\\catalog.json" and config.memory_budget_mb == 2048
    monkeypatch.setenv("NEKO_ASR_MEMORY_BUDGET_MB", "1536")
    assert local.configuration().memory_budget_mb == 1536


@pytest.mark.parametrize("name,value", [("NEKO_ASR_CATALOG", "relative"), ("NEKO_ASR_STATE_DIR", ""),
                                       ("NEKO_ASR_MEMORY_BUDGET_MB", "0"), ("NEKO_ASR_MEMORY_BUDGET_MB", "no"),
                                       ("NEKO_ASR_MEMORY_BUDGET_MB", "1048577")])
def test_invalid_reload_config_fails_closed(monkeypatch, name, value):
    configure(monkeypatch)
    monkeypatch.setenv(name, value)
    assert not local.configuration_is_available()


@pytest.mark.asyncio
async def test_reload_cli_and_control_endpoint_lifetime(monkeypatch):
    from dataclasses import replace
    from uuid import uuid4
    service = Service()
    manager = managed(service, monkeypatch)
    manager.config = replace(manager.config, catalog="C:\\catalog.json", state_dir="C:\\state", memory_budget_mb=1536)
    with pytest.raises(local.LocalServiceError, match="NOT_PREPARED"):
        manager.control_endpoint()
    value = await manager.prepare(str(uuid4()))
    args, _ = service.spawn_args
    assert args[args.index("--catalog") + 1] == "C:\\catalog.json"
    assert args[args.index("--state-dir") + 1] == "C:\\state"
    assert args[args.index("--memory-budget-mb") + 1] == "1536"
    endpoint = manager.control_endpoint()
    await manager.cancel(value["lease_id"])
    assert manager.control_endpoint() == endpoint
    await manager.close()
    with pytest.raises(local.LocalServiceError, match="MANAGER_CLOSED"):
        manager.control_endpoint()


@pytest.mark.asyncio
async def test_manager_close_owns_reload_control(monkeypatch):
    from unittest.mock import AsyncMock
    service = Service()
    manager = managed(service, monkeypatch)
    control = manager._reload_control = AsyncMock()
    await manager.close()
    control.close.assert_awaited_once()
