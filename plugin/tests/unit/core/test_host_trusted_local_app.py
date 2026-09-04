from __future__ import annotations

import asyncio

import pytest

from plugin._types.exceptions import PluginExecutionError
from plugin.core.host import PluginProcessHost


pytestmark = pytest.mark.plugin_unit


@pytest.mark.asyncio
async def test_host_only_invocation_crosses_process_with_verified_context(
    tmp_path,
) -> None:
    config_path = tmp_path / "plugin.toml"
    config_path.write_text("[plugin]\nname='trusted-local-app'\n", encoding="utf-8")
    host = PluginProcessHost(
        plugin_id="trusted_local_app_fixture",
        entry_point=(
            "tests.fixtures.plugin_test_trusted_local_app_fixture:"
            "TrustedLocalAppFixturePlugin"
        ),
        config_path=config_path,
    )
    context = {
        "app_id": "knowledge_dungeon",
        "client_id": "electron-1",
        "session_id": "session-1",
        "scope": "study_companion:dungeon",
        "operation": "knowledge_dungeon.bootstrap",
    }

    try:
        await host.start(
            message_target_queue=asyncio.Queue(),
            startup_timeout=5.0,
            startup_failure="fail",
        )
        result = await host.trigger_trusted_local_app(
            context=context,
            operation="knowledge_dungeon.bootstrap",
            payload={"request_id": "request-1"},
            timeout=2.0,
        )
        assert result == {
            "context": context,
            "payload": {"request_id": "request-1"},
        }
        aliased = await host.trigger_trusted_local_app(
            context=context,
            operation="private.stacked",
            payload={"request_id": "request-2"},
            timeout=2.0,
        )
        assert aliased == {
            "context": "knowledge_dungeon.bootstrap",
            "payload": {"request_id": "request-2"},
        }

        # A trusted handler is not a normal entry even when its Python method
        # name is known to the caller.
        with pytest.raises(PluginExecutionError, match="not found"):
            await host.trigger("bootstrap", {}, timeout=2.0)
        with pytest.raises(PluginExecutionError, match="not found"):
            await host.trigger_custom_event(
                "fixture", "stacked_trusted", {}, timeout=2.0
            )
    finally:
        await host.shutdown(timeout=1.0)
