from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tomllib
from typing import Literal
import zipfile

from plugin.neko_plugin_cli.public import inspect_package


InstallAction = Literal["install", "upgrade", "override_builtin", "blocked"]
PackageType = Literal["plugin", "bundle"]


@dataclass(frozen=True, slots=True)
class PluginInstallPlan:
    action: InstallAction
    package_type: PackageType
    package_id: str
    plugin_id: str
    directory_name: str
    current_version: str
    target_version: str
    confirmation_token: str
    reason: str
    legacy_plugin_ids: tuple[str, ...]
    installed_package_id: str = ""
    current_source: str = ""
    target_source: str = ""


def confirmation_token(*, package_path: Path, target_dir: Path) -> str:
    digest = hashlib.sha256()
    with package_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    digest.update(b"\0")
    digest.update(str(target_dir.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update((target_dir / "plugin.toml").read_bytes())
    return digest.hexdigest()


def build_install_plan(
    *,
    package_path: Path,
    plugins_root: Path,
    builtin_plugins_root: Path | None = None,
) -> PluginInstallPlan:
    package_path = package_path.expanduser().resolve()
    plugins_root = plugins_root.expanduser().resolve()
    inspected = inspect_package(package_path)

    if inspected.package_type == "bundle":
        conflicts = _bundle_conflicts(
            inspected.plugins,
            plugins_root,
            builtin_plugins_root=builtin_plugins_root,
        )
        return PluginInstallPlan(
            action="blocked" if conflicts else "install",
            package_type="bundle",
            package_id=inspected.package_id,
            plugin_id=inspected.package_id,
            directory_name="",
            current_version="",
            target_version=inspected.version,
            confirmation_token="",
            reason="bundle_conflict" if conflicts else "",
            legacy_plugin_ids=(),
            target_source="user",
        )

    if len(inspected.plugins) != 1:
        return _blocked(
            inspected.package_id,
            inspected.package_id,
            inspected.version,
            reason="invalid_plugin_count",
        )

    packaged_plugin = inspected.plugins[0]
    plugin_id = packaged_plugin.plugin_id
    directory_name = Path(packaged_plugin.archive_path).name
    packaged_manifest = _read_packaged_plugin_manifest(
        package_path,
        archive_path=packaged_plugin.archive_path,
    )
    target_version = _plugin_text(packaged_manifest, "version") or inspected.version
    previous_ids = _previous_ids(packaged_manifest)
    installed = _installed_plugins(plugins_root)

    legacy_ids = tuple(sorted(previous_id for previous_id in previous_ids if previous_id in installed))
    if legacy_ids:
        return PluginInstallPlan(
            action="blocked",
            package_type="plugin",
            package_id=inspected.package_id,
            plugin_id=plugin_id,
            directory_name=directory_name,
            current_version="",
            target_version=target_version,
            confirmation_token="",
            reason="legacy_plugin_present",
            legacy_plugin_ids=legacy_ids,
        )

    target_dir = plugins_root / directory_name
    matching = installed.get(plugin_id, [])
    if target_dir.exists():
        target_manifest = _read_manifest(target_dir / "plugin.toml")
        if _plugin_text(target_manifest, "id") != plugin_id:
            return _blocked(
                inspected.package_id,
                plugin_id,
                target_version,
                reason="directory_identity_conflict",
                directory_name=directory_name,
            )
    if len(matching) > 1 or (matching and matching[0].resolve() != target_dir.resolve()):
        return _blocked(
            inspected.package_id,
            plugin_id,
            target_version,
            reason="multiple_installations",
            directory_name=directory_name,
        )
    if not target_dir.exists():
        builtin_matches = _matching_builtins(
            plugin_id=plugin_id,
            builtin_plugins_root=builtin_plugins_root,
        )
        if len(builtin_matches) > 1:
            return _blocked(
                inspected.package_id,
                plugin_id,
                target_version,
                reason="multiple_builtin_sources",
                directory_name=directory_name,
            )
        if builtin_matches:
            return _plan_builtin_override(
                package_path=package_path,
                archive_path=packaged_plugin.archive_path,
                package_id=inspected.package_id,
                plugin_id=plugin_id,
                directory_name=directory_name,
                target_dir=target_dir,
                target_version=target_version,
                previous_ids=previous_ids,
                packaged_manifest=packaged_manifest,
                builtin_dir=builtin_matches[0],
            )
        return PluginInstallPlan(
            action="install",
            package_type="plugin",
            package_id=inspected.package_id,
            plugin_id=plugin_id,
            directory_name=directory_name,
            current_version="",
            target_version=target_version,
            confirmation_token="",
            reason="",
            legacy_plugin_ids=(),
            target_source="user",
        )

    current_manifest = _read_manifest(target_dir / "plugin.toml")
    return PluginInstallPlan(
        action="upgrade",
        package_type="plugin",
        package_id=inspected.package_id,
        plugin_id=plugin_id,
        directory_name=directory_name,
        current_version=_plugin_text(current_manifest, "version"),
        target_version=target_version,
        confirmation_token=confirmation_token(package_path=package_path, target_dir=target_dir),
        reason="",
        legacy_plugin_ids=(),
        current_source="user",
        target_source="user",
    )


def _matching_builtins(*, plugin_id: str, builtin_plugins_root: Path | None) -> list[Path]:
    if builtin_plugins_root is None:
        return []
    return _installed_plugins(builtin_plugins_root.expanduser().resolve()).get(plugin_id, [])


def _plan_builtin_override(
    *,
    package_path: Path,
    archive_path: str,
    package_id: str,
    plugin_id: str,
    directory_name: str,
    target_dir: Path,
    target_version: str,
    previous_ids: tuple[str, ...],
    packaged_manifest: dict[str, object],
    builtin_dir: Path,
) -> PluginInstallPlan:
    """Build the fail-closed plan for a user copy shadowing a builtin plugin."""

    builtin_manifest_path = builtin_dir / "plugin.toml"
    builtin_manifest = _read_manifest(builtin_manifest_path)
    current_version = _plugin_text(builtin_manifest, "version")
    invalid_reason = ""
    if package_id != plugin_id or directory_name != plugin_id or builtin_dir.name != plugin_id:
        invalid_reason = "override_identity_mismatch"
    elif previous_ids:
        invalid_reason = "override_previous_ids_not_supported"
    elif not _packaged_entry_is_valid(
        package_path=package_path,
        archive_path=archive_path,
        plugin_id=plugin_id,
        manifest=packaged_manifest,
    ):
        invalid_reason = "override_entry_missing"
    elif _plugin_text(builtin_manifest, "id") != plugin_id:
        invalid_reason = "override_builtin_identity_mismatch"

    if invalid_reason:
        return PluginInstallPlan(
            action="blocked",
            package_type="plugin",
            package_id=package_id,
            plugin_id=plugin_id,
            directory_name=directory_name,
            current_version=current_version,
            target_version=target_version,
            confirmation_token="",
            reason=invalid_reason,
            legacy_plugin_ids=(),
            current_source="builtin",
            target_source="market",
        )

    return PluginInstallPlan(
        action="override_builtin",
        package_type="plugin",
        package_id=package_id,
        plugin_id=plugin_id,
        directory_name=directory_name,
        current_version=current_version,
        target_version=target_version,
        confirmation_token=_builtin_override_confirmation_token(
            package_path=package_path,
            builtin_manifest_path=builtin_manifest_path,
            target_dir=target_dir,
            current_version=current_version,
            target_version=target_version,
        ),
        reason="",
        legacy_plugin_ids=(),
        installed_package_id=package_id,
        current_source="builtin",
        target_source="market",
    )


def _builtin_override_confirmation_token(
    *,
    package_path: Path,
    builtin_manifest_path: Path,
    target_dir: Path,
    current_version: str,
    target_version: str,
) -> str:
    """Bind confirmation to the package and the exact effective builtin source."""

    digest = hashlib.sha256()
    with package_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    for value in (
        hashlib.sha256(builtin_manifest_path.read_bytes()).hexdigest(),
        "builtin",
        str(target_dir.resolve()),
        current_version,
        target_version,
    ):
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _packaged_entry_is_valid(
    *,
    package_path: Path,
    archive_path: str,
    plugin_id: str,
    manifest: dict[str, object],
) -> bool:
    entry = _plugin_text(manifest, "entry")
    if ":" not in entry:
        return False
    module_name, attribute = (part.strip() for part in entry.rsplit(":", 1))
    if not module_name or not attribute:
        return False

    canonical_prefix = f"plugin.plugins.{plugin_id}"
    if module_name == canonical_prefix:
        relative_module = ""
    elif module_name.startswith(canonical_prefix + "."):
        relative_module = module_name[len(canonical_prefix) + 1 :]
    else:
        # Third-party packages may use a local module path. Keep it inside the
        # package root and require a concrete Python module to exist.
        relative_module = module_name
    if any(part in {"", ".", ".."} for part in relative_module.split(".")) and relative_module:
        return False

    base = archive_path.rstrip("/")
    candidates = (
        f"{base}/__init__.py"
        if not relative_module
        else f"{base}/{relative_module.replace('.', '/')}.py",
        f"{base}/{relative_module.replace('.', '/')}/__init__.py" if relative_module else "",
    )
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
    return any(candidate and candidate in names for candidate in candidates)


def _blocked(
    package_id: str,
    plugin_id: str,
    target_version: str,
    *,
    reason: str,
    directory_name: str = "",
) -> PluginInstallPlan:
    return PluginInstallPlan(
        action="blocked",
        package_type="plugin",
        package_id=package_id,
        plugin_id=plugin_id,
        directory_name=directory_name,
        current_version="",
        target_version=target_version,
        confirmation_token="",
        reason=reason,
        legacy_plugin_ids=(),
    )


def _bundle_conflicts(
    plugins: list[object],
    plugins_root: Path,
    *,
    builtin_plugins_root: Path | None = None,
) -> bool:
    installed = _installed_plugins(plugins_root)
    builtin_installed = (
        _installed_plugins(builtin_plugins_root.expanduser().resolve())
        if builtin_plugins_root is not None
        else {}
    )
    for packaged in plugins:
        plugin_id = getattr(packaged, "plugin_id", "")
        archive_path = getattr(packaged, "archive_path", "")
        directory_name = Path(archive_path).name
        if (
            plugin_id in installed
            or plugin_id in builtin_installed
            or (plugins_root / directory_name).exists()
            or (
                builtin_plugins_root is not None
                and (builtin_plugins_root / directory_name).exists()
            )
        ):
            return True
    return False


def _installed_plugins(plugins_root: Path) -> dict[str, list[Path]]:
    installed: dict[str, list[Path]] = {}
    if not plugins_root.is_dir():
        return installed
    for manifest_path in plugins_root.glob("*/plugin.toml"):
        if manifest_path.parent.name.startswith("."):
            continue
        manifest = _read_manifest(manifest_path)
        plugin_id = _plugin_text(manifest, "id")
        if plugin_id:
            installed.setdefault(plugin_id, []).append(manifest_path.parent)
    return installed


def _read_packaged_plugin_manifest(package_path: Path, *, archive_path: str) -> dict[str, object]:
    member_name = f"{archive_path.rstrip('/')}/plugin.toml"
    with zipfile.ZipFile(package_path) as archive:
        return tomllib.loads(archive.read(member_name).decode("utf-8"))


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _plugin_table(manifest: dict[str, object]) -> dict[str, object]:
    value = manifest.get("plugin")
    return value if isinstance(value, dict) else {}


def _plugin_text(manifest: dict[str, object], key: str) -> str:
    value = _plugin_table(manifest).get(key)
    return value.strip() if isinstance(value, str) else ""


def _previous_ids(manifest: dict[str, object]) -> tuple[str, ...]:
    value = _plugin_table(manifest).get("previous_ids")
    if not isinstance(value, list):
        return ()
    return tuple(sorted({item.strip() for item in value if isinstance(item, str) and item.strip()}))
