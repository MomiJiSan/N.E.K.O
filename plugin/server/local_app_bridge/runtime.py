"""Explicit lifecycle and provider-neutral plugin target registration."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from plugin.server.application.plugins.trusted_local_app_dispatch import (
    TrustedLocalAppPluginDispatch,
)
from plugin.server.local_app_bridge.contracts import (
    LaunchMaterial,
    LocalAppPolicy,
    require_identifier,
)
from plugin.server.local_app_bridge.http_server import LocalAppBridgeHttpServer
from plugin.server.local_app_bridge.launch import launch_local_app
from plugin.server.local_app_bridge.service import LocalAppBridgeService
from utils.config_manager import get_config_manager


_LOCAL_APP_CLIENT_ID_DOMAIN = b"local-app-client-v1\0"
MAX_LAUNCHED_LOCAL_APP_PROCESSES = 16


async def _derive_persistent_client_id(app_id: str) -> str:
    """Derive a stable, app-scoped identity without exposing the host ID."""

    normalized_app_id = require_identifier(app_id, "app_id")

    def derive() -> str:
        client_id, client_proof = (
            get_config_manager().ensure_cloudsave_client_credentials()
        )
        if not isinstance(client_id, str) or not client_id:
            raise RuntimeError("host client identity is unavailable")
        if not isinstance(client_proof, str) or not client_proof:
            raise RuntimeError("host client proof is unavailable")
        message = (
            _LOCAL_APP_CLIENT_ID_DOMAIN
            + normalized_app_id.encode("utf-8")
            + b"\0"
            + client_id.encode("utf-8")
        )
        digest = hmac.new(
            client_proof.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()
        return f"lac_{digest}"

    return require_identifier(await asyncio.to_thread(derive), "client_id")


@dataclass(frozen=True, slots=True)
class LocalAppPluginTarget:
    scope: str
    operation: str
    plugin_id: str
    plugin_operation: str
    timeout: float = 30.0

    def __post_init__(self) -> None:
        require_identifier(self.scope, "scope")
        require_identifier(self.operation, "operation")
        require_identifier(self.plugin_id, "plugin_id")
        require_identifier(self.plugin_operation, "plugin_operation")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(float(self.timeout))
            or self.timeout <= 0
        ):
            raise ValueError("target timeout must be a positive finite number")


@dataclass(frozen=True, slots=True)
class LocalAppPluginRegistration:
    policy: LocalAppPolicy
    targets: tuple[LocalAppPluginTarget, ...]

    def __post_init__(self) -> None:
        target_keys = {(target.scope, target.operation) for target in self.targets}
        policy_keys = {
            (scope, operation)
            for scope, operations in self.policy.allowed_operations.items()
            for operation in operations
        }
        if not target_keys or len(target_keys) != len(self.targets):
            raise ValueError("local app targets must be non-empty and unique")
        if target_keys != policy_keys:
            raise ValueError("local app targets must exactly match the app policy")


@dataclass(frozen=True, slots=True)
class LocalAppInstallation:
    """Host-owned launch target, deliberately separate from plugin metadata."""

    app_id: str
    title: str
    executable: Path
    args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.app_id, "app_id")
        if (
            not isinstance(self.title, str)
            or not self.title.strip()
            or len(self.title.strip()) > 128
            or any(ord(character) < 32 for character in self.title)
        ):
            raise ValueError("local app title is invalid")
        executable = Path(self.executable).expanduser().resolve(strict=True)
        if not executable.is_file():
            raise ValueError("local app executable must be a regular file")
        if not isinstance(self.args, tuple) or len(self.args) > 32:
            raise ValueError("local app arguments exceed the bounded contract")
        normalized_args: list[str] = []
        for argument in self.args:
            if (
                not isinstance(argument, str)
                or len(argument) > 1024
                or any(ord(character) < 32 for character in argument)
            ):
                raise ValueError("local app argument is invalid")
            normalized_args.append(argument)
        object.__setattr__(self, "executable", executable)
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "args", tuple(normalized_args))


@dataclass(frozen=True, slots=True)
class LocalAppLaunchResult:
    app_id: str
    client_id: str
    process_id: int


@dataclass(frozen=True, slots=True)
class LocalAppDescriptor:
    app_id: str
    title: str
    available: bool


ClientIdProvider = Callable[[str], Awaitable[str]]
ProcessLauncher = Callable[..., Awaitable[asyncio.subprocess.Process]]


class LocalAppBridgeRuntime:
    """Own one listener per plugin-host lifecycle and retain registrations."""

    def __init__(
        self,
        *,
        client_id_provider: ClientIdProvider | None = None,
        process_launcher: ProcessLauncher | None = None,
    ) -> None:
        self._registrations: dict[str, LocalAppPluginRegistration] = {}
        self._installations: dict[str, LocalAppInstallation] = {}
        self._service: LocalAppBridgeService | None = None
        self._server: LocalAppBridgeHttpServer | None = None
        self._client_id_provider = client_id_provider or _derive_persistent_client_id
        self._process_launcher = process_launcher or launch_local_app
        self._launched_processes: set[asyncio.subprocess.Process] = set()
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._server is not None

    @property
    def origin(self) -> str:
        server = self._server
        if server is None:
            raise RuntimeError("local app bridge is not running")
        return server.origin

    def register_plugin_app(self, registration: LocalAppPluginRegistration) -> None:
        if self._service is not None or self._server is not None:
            raise RuntimeError("local app registrations are frozen while running")
        app_id = registration.policy.app_id
        if app_id in self._registrations:
            raise ValueError(f"app already registered: {app_id}")
        self._registrations[app_id] = registration

    def configure_plugin_apps(
        self, registrations: tuple[LocalAppPluginRegistration, ...]
    ) -> None:
        """Replace host-owned manifest registrations before listener startup."""
        if self._service is not None or self._server is not None:
            raise RuntimeError("local app registrations are frozen while running")
        configured: dict[str, LocalAppPluginRegistration] = {}
        for registration in registrations:
            app_id = registration.policy.app_id
            if app_id in configured:
                raise ValueError(f"duplicate app registration: {app_id}")
            configured[app_id] = registration
        self._registrations = configured

    def configure_installations(
        self, installations: Sequence[LocalAppInstallation]
    ) -> None:
        """Freeze host-owned executable locations before listener startup."""
        if self._service is not None or self._server is not None:
            raise RuntimeError("local app installations are frozen while running")
        configured: dict[str, LocalAppInstallation] = {}
        for installation in installations:
            if installation.app_id not in self._registrations:
                raise ValueError(
                    f"local app installation is not manifest-authorized: {installation.app_id}"
                )
            if installation.app_id in configured:
                raise ValueError(
                    f"duplicate local app installation: {installation.app_id}"
                )
            configured[installation.app_id] = installation
        self._installations = configured

    def authorized_app_ids(self) -> frozenset[str]:
        return frozenset(self._registrations)

    def describe_plugin_app(
        self, plugin_id: str, *, fallback_title: str
    ) -> LocalAppDescriptor | None:
        """Return the frontend-safe subset of one authorized declaration."""
        if not isinstance(plugin_id, str) or not plugin_id:
            return None
        for app_id, registration in self._registrations.items():
            if not all(
                target.plugin_id == plugin_id for target in registration.targets
            ):
                continue
            installation = self._installations.get(app_id)
            title = installation.title if installation is not None else fallback_title
            return LocalAppDescriptor(
                app_id=app_id,
                title=str(title).strip() or app_id,
                available=installation is not None and self.is_running,
            )
        return None

    async def start(self) -> str:
        async with self._lock:
            if self._server is not None:
                return self._server.origin
            service = LocalAppBridgeService()
            for registration in self._registrations.values():
                self._install_registration(service, registration)
            server = LocalAppBridgeHttpServer(service)
            # Publish the service before the listener await so a synchronous
            # registration scheduled during bind cannot be missed.
            self._service = service
            try:
                origin = await server.start()
            except BaseException:
                self._service = None
                await server.close()
                raise
            self._server = server
            return origin

    async def close(self) -> None:
        async with self._lock:
            server = self._server
            self._server = None
            self._service = None
            if server is not None:
                await server.close()
            processes = tuple(self._launched_processes)
            self._launched_processes.clear()
            if processes:
                await asyncio.gather(
                    *(self._stop_process(process) for process in processes),
                    return_exceptions=True,
                )

    async def launch_registered_app(self, app_id: str) -> LocalAppLaunchResult:
        """Launch one host-installed app; callers cannot supply executable data."""
        normalized_app_id = require_identifier(app_id, "app_id")
        async with self._lock:
            service = self._service
            server = self._server
            if service is None or server is None:
                raise RuntimeError("local app bridge is not running")
            registration = self._registrations.get(normalized_app_id)
            installation = self._installations.get(normalized_app_id)
            if registration is None or installation is None:
                raise LookupError("local app is not registered and installed")
            self._launched_processes = {
                launched
                for launched in self._launched_processes
                if launched.returncode is None
            }
            if len(self._launched_processes) >= MAX_LAUNCHED_LOCAL_APP_PROCESSES:
                raise RuntimeError("local app process capacity is reached")
            client_id = require_identifier(
                await self._client_id_provider(normalized_app_id), "client_id"
            )
            material = await service.issue_launch_material(
                bridge_origin=server.origin,
                app_id=normalized_app_id,
                client_id=client_id,
                requested_scopes=frozenset(registration.policy.allowed_operations),
            )
            process = await self._process_launcher(
                installation.executable,
                args=installation.args,
                material=material,
            )
            process_id = process.pid
            if not isinstance(process_id, int) or process_id <= 0:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                raise RuntimeError("local app process did not provide a process id")
            if process.returncode is None:
                self._launched_processes.add(process)
            return LocalAppLaunchResult(
                app_id=normalized_app_id,
                client_id=client_id,
                process_id=process_id,
            )

    async def issue_launch_material(
        self,
        *,
        app_id: str,
        client_id: str,
        requested_scopes: frozenset[str],
    ) -> LaunchMaterial:
        service = self._service
        server = self._server
        if service is None or server is None:
            raise RuntimeError("local app bridge is not running")
        return await service.issue_launch_material(
            bridge_origin=server.origin,
            app_id=app_id,
            client_id=client_id,
            requested_scopes=requested_scopes,
        )

    @staticmethod
    def _install_registration(
        service: LocalAppBridgeService,
        registration: LocalAppPluginRegistration,
    ) -> None:
        service.register_app(registration.policy)
        plugin_dispatch = TrustedLocalAppPluginDispatch()
        for target in registration.targets:

            async def handler(context, payload, *, _target=target):
                return await plugin_dispatch.invoke(
                    plugin_id=_target.plugin_id,
                    plugin_operation=_target.plugin_operation,
                    context={
                        "app_id": context.identity.app_id,
                        "client_id": context.identity.client_id,
                        "session_id": context.identity.session_id,
                        "scope": context.scope,
                        "operation": context.operation,
                    },
                    payload=payload,
                    timeout=float(_target.timeout),
                )

            service.register_operation(
                app_id=registration.policy.app_id,
                scope=target.scope,
                operation=target.operation,
                handler=handler,
            )

    @staticmethod
    async def _stop_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
            return
        except TimeoutError:
            pass
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


_runtime = LocalAppBridgeRuntime()


def get_local_app_bridge_runtime() -> LocalAppBridgeRuntime:
    return _runtime


async def start_local_app_bridge() -> str:
    return await _runtime.start()


async def stop_local_app_bridge() -> None:
    await _runtime.close()


async def launch_registered_local_app(app_id: str) -> LocalAppLaunchResult:
    """Host-only callable entry point used by trusted product UI assembly."""
    return await _runtime.launch_registered_app(app_id)
