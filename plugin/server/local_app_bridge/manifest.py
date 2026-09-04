"""Strict host-side loading of local-app registrations from plugin manifests."""

from __future__ import annotations

import asyncio
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from plugin.core.state import state
from plugin.server.local_app_bridge.contracts import (
    LocalAppPolicy,
    require_identifier,
)
from plugin.server.local_app_bridge.runtime import (
    LocalAppPluginRegistration,
    LocalAppPluginTarget,
    get_local_app_bridge_runtime,
)

MAX_MANIFEST_APPS = 64
MAX_OPERATIONS_PER_APP = 32
_CANONICAL_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


@dataclass(frozen=True, slots=True)
class LocalAppManifestIssue:
    plugin_id: str
    code: str


class _InvalidManifest(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _registry_snapshot() -> dict[str, dict[str, object]]:
    with state.acquire_plugins_read_lock():
        return {
            plugin_id: dict(metadata)
            for plugin_id, metadata in state.plugins.items()
            if isinstance(plugin_id, str) and isinstance(metadata, dict)
        }


def _parse_registration(
    *, plugin_id: str, config_path: Path
) -> LocalAppPluginRegistration | None:
    try:
        with config_path.open("rb") as stream:
            manifest = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _InvalidManifest("manifest_unreadable") from exc
    plugin = manifest.get("plugin")
    if not isinstance(plugin, dict):
        raise _InvalidManifest("plugin_table_invalid")
    raw_local_app = plugin.get("local_app")
    if raw_local_app is None:
        return None
    if not isinstance(raw_local_app, dict) or set(raw_local_app) != {
        "app_id",
        "scope",
        "operations",
    }:
        raise _InvalidManifest("local_app_fields_invalid")
    try:
        app_id = require_identifier(raw_local_app["app_id"], "app_id")
        scope = require_identifier(raw_local_app["scope"], "scope")
    except Exception as exc:
        raise _InvalidManifest("local_app_identity_invalid") from exc
    operations = raw_local_app["operations"]
    if (
        not isinstance(operations, dict)
        or not operations
        or len(operations) > MAX_OPERATIONS_PER_APP
    ):
        raise _InvalidManifest("local_app_operations_invalid")
    targets: list[LocalAppPluginTarget] = []
    try:
        for external_operation, plugin_operation in operations.items():
            if (
                not isinstance(external_operation, str)
                or _CANONICAL_OPERATION_RE.fullmatch(external_operation) is None
                or not isinstance(plugin_operation, str)
                or _CANONICAL_OPERATION_RE.fullmatch(plugin_operation) is None
            ):
                raise ValueError("operation is not canonical")
            targets.append(
                LocalAppPluginTarget(
                    scope=scope,
                    operation=require_identifier(
                        external_operation, "external_operation"
                    ),
                    plugin_id=plugin_id,
                    plugin_operation=require_identifier(
                        plugin_operation, "plugin_operation"
                    ),
                )
            )
    except Exception as exc:
        raise _InvalidManifest("local_app_operations_invalid") from exc
    return LocalAppPluginRegistration(
        policy=LocalAppPolicy(
            app_id=app_id,
            allowed_operations={
                scope: frozenset(target.operation for target in targets)
            },
        ),
        targets=tuple(targets),
    )


def _discover_registrations(
    snapshot: dict[str, dict[str, object]],
) -> tuple[tuple[LocalAppPluginRegistration, ...], tuple[LocalAppManifestIssue, ...]]:
    by_app: dict[str, tuple[str, LocalAppPluginRegistration]] = {}
    blocked_apps: set[str] = set()
    issues: list[LocalAppManifestIssue] = []
    for plugin_id in sorted(snapshot):
        metadata = snapshot[plugin_id]
        if metadata.get("runtime_enabled") is not True:
            continue
        raw_path = metadata.get("config_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        try:
            registration = _parse_registration(
                plugin_id=plugin_id,
                config_path=Path(raw_path).resolve(strict=False),
            )
        except _InvalidManifest as exc:
            issues.append(LocalAppManifestIssue(plugin_id, exc.code))
            continue
        if registration is None:
            continue
        app_id = registration.policy.app_id
        if app_id in blocked_apps:
            issues.append(LocalAppManifestIssue(plugin_id, "duplicate_app_id"))
            continue
        previous = by_app.pop(app_id, None)
        if previous is not None:
            previous_plugin_id, _registration = previous
            blocked_apps.add(app_id)
            issues.append(LocalAppManifestIssue(previous_plugin_id, "duplicate_app_id"))
            issues.append(LocalAppManifestIssue(plugin_id, "duplicate_app_id"))
            continue
        if len(by_app) >= MAX_MANIFEST_APPS:
            issues.append(LocalAppManifestIssue(plugin_id, "app_capacity_reached"))
            continue
        by_app[app_id] = (plugin_id, registration)
    registrations = tuple(item[1] for item in by_app.values())
    return registrations, tuple(issues)


async def configure_local_app_bridge_from_registry() -> tuple[
    LocalAppManifestIssue, ...
]:
    """Load immutable declarations and freeze them before the listener starts."""
    snapshot = await asyncio.to_thread(_registry_snapshot)
    registrations, issues = await asyncio.to_thread(_discover_registrations, snapshot)
    get_local_app_bridge_runtime().configure_plugin_apps(registrations)
    return issues
