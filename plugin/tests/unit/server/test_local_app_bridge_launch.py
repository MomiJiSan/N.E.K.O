from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from plugin.server.local_app_bridge.contracts import LaunchMaterial, LocalAppPolicy
from plugin.server.local_app_bridge.launch import launch_local_app
from plugin.server.local_app_bridge.runtime import (
    LocalAppBridgeRuntime,
    LocalAppInstallation,
    LocalAppPluginRegistration,
    LocalAppPluginTarget,
    _derive_persistent_client_id,
)
from plugin.server.local_app_bridge import runtime as runtime_module


pytestmark = pytest.mark.plugin_unit


def _registration(app_id: str = "knowledge_dungeon") -> LocalAppPluginRegistration:
    scope = "study_companion:dungeon"
    operation = "knowledge_dungeon.bootstrap"
    return LocalAppPluginRegistration(
        policy=LocalAppPolicy(
            app_id=app_id,
            allowed_operations={scope: frozenset({operation})},
        ),
        targets=(
            LocalAppPluginTarget(
                scope=scope,
                operation=operation,
                plugin_id="study_companion",
                plugin_operation=operation,
            ),
        ),
    )


class _FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


@pytest.mark.asyncio
async def test_host_launch_uses_only_frozen_installation_and_manifest_scope(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "knowledge-dungeon.exe"
    executable.write_bytes(b"test executable placeholder")
    calls: list[dict[str, object]] = []
    process = _FakeProcess()

    async def client_id_provider(app_id: str) -> str:
        assert app_id == "knowledge_dungeon"
        return "lac_stable"

    async def process_launcher(
        selected_executable: Path,
        *,
        args: tuple[str, ...],
        material: LaunchMaterial,
    ) -> _FakeProcess:
        calls.append(
            {
                "executable": selected_executable,
                "args": args,
                "material": material,
            }
        )
        return process

    runtime = LocalAppBridgeRuntime(
        client_id_provider=client_id_provider,
        process_launcher=process_launcher,
    )
    runtime.register_plugin_app(_registration())
    runtime.configure_installations(
        (
            LocalAppInstallation(
                app_id="knowledge_dungeon",
                title="Knowledge Dungeon",
                executable=executable,
                args=("--product-mode",),
            ),
        )
    )
    origin = await runtime.start()

    result = await runtime.launch_registered_app("knowledge_dungeon")

    assert result.app_id == "knowledge_dungeon"
    assert result.client_id == "lac_stable"
    assert result.process_id == 4321
    assert len(calls) == 1
    call = calls[0]
    assert call["executable"] == executable.resolve()
    assert call["args"] == ("--product-mode",)
    material = call["material"]
    assert isinstance(material, LaunchMaterial)
    assert material.bridge_origin == origin
    assert material.app_id == "knowledge_dungeon"
    assert material.client_id == "lac_stable"
    service = runtime._service
    assert service is not None
    paired = await service.pair(
        launch_code=material.launch_code,
        app_id=material.app_id,
        client_id=material.client_id,
    )
    assert paired.granted_scopes == frozenset({"study_companion:dungeon"})

    await runtime.close()
    assert process.terminated is True


@pytest.mark.asyncio
async def test_host_launch_requires_both_manifest_registration_and_installation(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "registered.exe"
    executable.write_bytes(b"placeholder")

    async def client_id_provider(_app_id: str) -> str:
        return "lac_stable"

    async def should_not_launch(**_kwargs: object) -> _FakeProcess:
        raise AssertionError("unregistered app must not launch")

    runtime = LocalAppBridgeRuntime(
        client_id_provider=client_id_provider,
        process_launcher=should_not_launch,
    )
    with pytest.raises(ValueError, match="manifest-authorized"):
        runtime.configure_installations(
            (
                LocalAppInstallation(
                    app_id="knowledge_dungeon",
                    title="Knowledge Dungeon",
                    executable=executable,
                ),
            )
        )

    runtime = LocalAppBridgeRuntime(
        client_id_provider=client_id_provider,
        process_launcher=should_not_launch,
    )
    runtime.register_plugin_app(_registration())
    await runtime.start()
    with pytest.raises(LookupError, match="registered and installed"):
        await runtime.launch_registered_app("knowledge_dungeon")
    await runtime.close()


@pytest.mark.asyncio
async def test_installation_registration_freezes_when_listener_starts(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "registered.exe"
    executable.write_bytes(b"placeholder")
    installation = LocalAppInstallation(
        app_id="knowledge_dungeon",
        title="Knowledge Dungeon",
        executable=executable,
    )
    runtime = LocalAppBridgeRuntime()
    runtime.register_plugin_app(_registration())
    runtime.configure_installations((installation,))
    await runtime.start()

    with pytest.raises(RuntimeError, match="frozen"):
        runtime.configure_installations((installation,))

    await runtime.close()


@pytest.mark.parametrize(
    "args",
    [
        ("bad\nargument",),
        ("x" * 1025,),
        tuple(str(index) for index in range(33)),
    ],
)
def test_installation_rejects_unbounded_or_control_arguments(
    tmp_path: Path, args: tuple[str, ...]
) -> None:
    executable = tmp_path / "registered.exe"
    executable.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="argument"):
        LocalAppInstallation(
            app_id="knowledge_dungeon",
            title="Knowledge Dungeon",
            executable=executable,
            args=args,
        )


@pytest.mark.asyncio
async def test_persistent_client_id_is_deterministic_and_app_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ConfigManager:
        @staticmethod
        def ensure_cloudsave_client_credentials() -> tuple[str, str]:
            return "host-client-id", "host-client-proof"

    monkeypatch.setattr(runtime_module, "get_config_manager", lambda: _ConfigManager())

    first = await _derive_persistent_client_id("knowledge_dungeon")
    restarted = await _derive_persistent_client_id("knowledge_dungeon")
    different_app = await _derive_persistent_client_id("another_app")

    assert first == restarted
    assert first.startswith("lac_") and len(first) == 68
    assert first != different_app
    assert "host-client-id" not in first
    assert "host-client-proof" not in first


@pytest.mark.asyncio
async def test_default_launcher_delivers_material_only_through_stdin(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "material.json"
    script = (
        "import pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.readline())"
    )
    material = LaunchMaterial(
        bridge_origin="http://127.0.0.1:49123",
        app_id="knowledge_dungeon",
        client_id="lac_stable",
        launch_code="one-time-secret",
    )

    process = await launch_local_app(
        sys.executable,
        args=("-c", script, str(output_path)),
        material=material,
    )
    assert await asyncio.wait_for(process.wait(), timeout=5.0) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == material.to_dict()
