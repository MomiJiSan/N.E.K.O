from __future__ import annotations

import pytest

from plugin.sdk.local_app import (
    TrustedLocalAppPluginContext,
    collect_trusted_local_app_operations,
    trusted_local_app_operation,
)


pytestmark = pytest.mark.plugin_unit


def test_trusted_context_rejects_forged_or_extra_fields() -> None:
    valid = {
        "app_id": "app",
        "client_id": "client",
        "session_id": "session",
        "scope": "scope",
        "operation": "operation",
    }
    assert TrustedLocalAppPluginContext.from_mapping(valid).app_id == "app"
    with pytest.raises(ValueError, match="fields"):
        TrustedLocalAppPluginContext.from_mapping({**valid, "admin": "true"})
    with pytest.raises(ValueError, match="app_id"):
        TrustedLocalAppPluginContext.from_mapping({**valid, "app_id": "bad app"})


def test_decorator_requires_async_handler_and_collector_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="valid identifier"):
        trusted_local_app_operation("operation")

    with pytest.raises(TypeError, match="must be async"):

        @trusted_local_app_operation("demo.operation")
        def _sync_handler() -> None:
            return None

    class _Duplicate:
        @trusted_local_app_operation("demo.operation")
        async def first(self) -> None:
            return None

        @trusted_local_app_operation("demo.operation")
        async def second(self) -> None:
            return None

    with pytest.raises(ValueError, match="duplicate"):
        collect_trusted_local_app_operations(_Duplicate())


def test_collector_only_returns_dedicated_handlers() -> None:
    class _Adapter:
        async def ordinary(self) -> None:
            return None

        @trusted_local_app_operation("demo.operation")
        async def trusted(self) -> None:
            return None

    adapter = _Adapter()
    assert collect_trusted_local_app_operations(adapter) == {
        "demo.operation": adapter.trusted
    }
