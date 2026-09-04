"""Load host-private local-app launch targets before the bridge binds."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from plugin.server.local_app_bridge.errors import LocalAppBridgeError
from plugin.server.local_app_bridge.runtime import (
    LocalAppInstallation,
    get_local_app_bridge_runtime,
)

LOCAL_APP_INSTALLATIONS_FILE_ENV = "NEKO_LOCAL_APP_INSTALLATIONS_FILE"
MAX_INSTALLATIONS_FILE_BYTES = 64 * 1024
MAX_INSTALLATIONS = 64


@dataclass(frozen=True, slots=True)
class LocalAppInstallationIssue:
    code: str


class _InvalidInstallationConfig(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidInstallationConfig("duplicate_json_key")
        result[key] = value
    return result


def _load_installations(path: Path) -> tuple[LocalAppInstallation, ...]:
    if not path.is_absolute() or not path.is_file():
        raise _InvalidInstallationConfig("installations_file_invalid")
    try:
        encoded = path.read_bytes()
        if len(encoded) > MAX_INSTALLATIONS_FILE_BYTES:
            raise _InvalidInstallationConfig("installations_file_too_large")
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except _InvalidInstallationConfig:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _InvalidInstallationConfig("installations_file_unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "installations"}:
        raise _InvalidInstallationConfig("installations_fields_invalid")
    if payload["version"] != 1:
        raise _InvalidInstallationConfig("installations_version_unsupported")
    raw_installations = payload["installations"]
    if (
        not isinstance(raw_installations, list)
        or len(raw_installations) > MAX_INSTALLATIONS
    ):
        raise _InvalidInstallationConfig("installations_list_invalid")
    installations: list[LocalAppInstallation] = []
    try:
        for raw in raw_installations:
            if not isinstance(raw, dict) or set(raw) != {
                "app_id",
                "title",
                "executable",
                "args",
            }:
                raise _InvalidInstallationConfig("installation_fields_invalid")
            executable = raw["executable"]
            args = raw["args"]
            if not isinstance(executable, str) or not Path(executable).is_absolute():
                raise _InvalidInstallationConfig("installation_executable_invalid")
            if not isinstance(args, list):
                raise _InvalidInstallationConfig("installation_args_invalid")
            installations.append(
                LocalAppInstallation(
                    app_id=raw["app_id"],
                    title=raw["title"],
                    executable=Path(executable),
                    args=tuple(args),
                )
            )
    except _InvalidInstallationConfig:
        raise
    except (LocalAppBridgeError, OSError, TypeError, ValueError) as exc:
        raise _InvalidInstallationConfig("installation_invalid") from exc
    return tuple(installations)


async def configure_local_app_installations_from_host() -> tuple[
    LocalAppInstallationIssue, ...
]:
    """Apply one host-owned file; invalid input disables launch, not plugins."""
    runtime = get_local_app_bridge_runtime()
    raw_path = os.getenv(LOCAL_APP_INSTALLATIONS_FILE_ENV, "").strip()
    if not raw_path:
        runtime.configure_installations(())
        return ()
    try:
        installations = await asyncio.to_thread(_load_installations, Path(raw_path))
        runtime.configure_installations(installations)
    except (
        _InvalidInstallationConfig,
        LocalAppBridgeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        runtime.configure_installations(())
        code = (
            str(exc)
            if isinstance(exc, _InvalidInstallationConfig)
            else "installations_configuration_invalid"
        )
        return (LocalAppInstallationIssue(code=code),)
    return ()
