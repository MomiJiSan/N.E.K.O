from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path

import pytest

from plugin.core.state import state
from plugin.server import install_registry


@pytest.fixture(autouse=True)
def isolated_install_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with state.acquire_plugins_read_lock():
        plugins_backup = copy.deepcopy(state.plugins)
    registry_backup = dict(install_registry._install_plugin_registry)
    hooks = install_registry._tutorial_migration_hooks
    hooks_backup = list(hooks) if isinstance(hooks, list) else {
        key: list(value) for key, value in hooks.items()
    }
    monkeypatch.setattr(install_registry, "_install_plugin_registry", {})
    monkeypatch.setattr(install_registry, "_tutorial_migration_hooks", {})
    with state.acquire_plugins_write_lock():
        state.plugins.clear()
    try:
        yield
    finally:
        with state.acquire_plugins_write_lock():
            state.plugins.clear()
            state.plugins.update(plugins_backup)
        install_registry._install_plugin_registry = registry_backup
        install_registry._tutorial_migration_hooks = hooks_backup


def _write_manifest(
    root: Path,
    plugin_id: str,
    *,
    declaration: str | None,
) -> Path:
    root.mkdir(parents=True)
    manifest = root / "plugin.toml"
    text = (
        "[plugin]\n"
        f'id = "{plugin_id}"\n'
        f'name = "{plugin_id}"\n'
        'version = "1.0.0"\n'
        'type = "plugin"\n'
        f'entry = "plugin.plugins.{plugin_id}:Plugin"\n'
    )
    if declaration is not None:
        text += "\n" + declaration.strip() + "\n"
    manifest.write_text(text, encoding="utf-8")
    return manifest


def _select_plugin(
    plugin_id: str,
    manifest: Path,
    *,
    entries: tuple[str, ...],
    source: str = "user",
    load_state: str | None = None,
) -> None:
    meta: dict[str, object] = {
        "id": plugin_id,
        "config_path": str(manifest),
        "entries_preview": [{"id": entry_id} for entry_id in entries],
        "effective_source": source,
    }
    if load_state is not None:
        meta["runtime_load_state"] = load_state
    with state.acquire_plugins_write_lock():
        state.plugins[plugin_id] = meta


_STUDY_DECLARATION = """
[plugin.install]
enabled = true
ui_i18n_dir = "i18n"
tutorial_enabled = true

[plugin.install.kinds.rapidocr_models]
entry_id = "study_download_rapidocr_models"
label = "RapidOCR Models"
queued_message = "RapidOCR model download queued"
entry_timeout = 600.0
"""

_GALGAME_DECLARATION = """
[plugin.install]
enabled = true
ui_i18n_dir = "i18n/ui"
tutorial_enabled = true

[plugin.install.kinds.textractor]
entry_id = "galgame_install_textractor"
label = "Textractor"
queued_message = "Textractor install queued"
entry_timeout = 600.0

[plugin.install.kinds.rapidocr_models]
entry_id = "galgame_download_rapidocr_models"
label = "RapidOCR Models"
queued_message = "RapidOCR model download queued"
entry_timeout = 600.0
"""


@pytest.mark.parametrize(
    ("plugin_id", "declaration", "entries", "expected_kinds", "i18n_relative"),
    [
        (
            "study_companion",
            _STUDY_DECLARATION,
            ("study_download_rapidocr_models",),
            {"rapidocr_models"},
            Path("i18n"),
        ),
        (
            "galgame_plugin",
            _GALGAME_DECLARATION,
            ("galgame_install_textractor", "galgame_download_rapidocr_models"),
            {"textractor", "rapidocr_models"},
            Path("i18n/ui"),
        ),
    ],
)
def test_explicit_install_declaration_uses_selected_registry_source(
    tmp_path: Path,
    plugin_id: str,
    declaration: str,
    entries: tuple[str, ...],
    expected_kinds: set[str],
    i18n_relative: Path,
) -> None:
    plugin_root = tmp_path / "market" / plugin_id
    manifest = _write_manifest(plugin_root, plugin_id, declaration=declaration)
    (plugin_root / i18n_relative).mkdir(parents=True)
    _select_plugin(plugin_id, manifest, entries=entries, source="user")

    registration = install_registry.get_install_plugin_registration(plugin_id)

    assert registration is not None
    assert registration.plugin_id == plugin_id
    assert set(registration.install_kinds) == expected_kinds
    assert registration.ui_i18n_dir == (plugin_root / i18n_relative).resolve()
    assert registration.tutorial_enabled is True
    assert {spec.entry_timeout for spec in registration.install_kinds.values()} == {600.0}


def test_explicit_disabled_declaration_does_not_fall_back_to_legacy(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "study_companion",
        "study_companion",
        declaration="[plugin.install]\nenabled = false",
    )
    _select_plugin(
        "study_companion",
        manifest,
        entries=("study_download_rapidocr_models",),
    )

    assert install_registry.get_install_plugin_registration("study_companion") is None


@pytest.mark.parametrize(
    ("plugin_id", "entries", "expected_kinds"),
    [
        (
            "study_companion",
            ("study_download_rapidocr_models",),
            {"rapidocr_models"},
        ),
        (
            "galgame_plugin",
            ("galgame_install_textractor", "galgame_download_rapidocr_models"),
            {"textractor", "rapidocr_models"},
        ),
    ],
)
def test_legacy_install_declaration_is_limited_to_externalized_plugins(
    tmp_path: Path,
    plugin_id: str,
    entries: tuple[str, ...],
    expected_kinds: set[str],
) -> None:
    plugin_root = tmp_path / plugin_id
    manifest = _write_manifest(plugin_root, plugin_id, declaration=None)
    i18n = plugin_root / ("i18n/ui" if plugin_id == "galgame_plugin" else "i18n")
    i18n.mkdir(parents=True)
    _select_plugin(plugin_id, manifest, entries=entries)

    registration = install_registry.get_install_plugin_registration(plugin_id)

    assert registration is not None
    assert set(registration.install_kinds) == expected_kinds
    assert "tesseract" not in registration.install_kinds
    assert registration.ui_i18n_dir == i18n.resolve()


def test_invalid_explicit_entry_fails_closed_without_dynamic_fallback(
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "study_companion"
    manifest = _write_manifest(
        plugin_root,
        "study_companion",
        declaration=_STUDY_DECLARATION,
    )
    (plugin_root / "i18n").mkdir()
    _select_plugin("study_companion", manifest, entries=("different_entry",))
    install_registry.register_install_plugin(
        "study_companion",
        install_kinds={},
        ui_i18n_dir=plugin_root / "i18n",
    )

    with pytest.raises(ValueError, match="study_download_rapidocr_models"):
        install_registry.get_install_plugin_registration("study_companion")


def test_invalid_explicit_i18n_escape_fails_closed(tmp_path: Path) -> None:
    plugin_root = tmp_path / "galgame_plugin"
    declaration = _GALGAME_DECLARATION.replace(
        'ui_i18n_dir = "i18n/ui"',
        'ui_i18n_dir = "../outside"',
    )
    manifest = _write_manifest(plugin_root, "galgame_plugin", declaration=declaration)
    _select_plugin(
        "galgame_plugin",
        manifest,
        entries=("galgame_install_textractor", "galgame_download_rapidocr_models"),
    )

    with pytest.raises(ValueError, match="ui_i18n_dir"):
        install_registry.get_install_plugin_registration("galgame_plugin")


def test_failed_selected_runtime_does_not_expose_install_api(tmp_path: Path) -> None:
    plugin_root = tmp_path / "study_companion"
    manifest = _write_manifest(plugin_root, "study_companion", declaration=_STUDY_DECLARATION)
    (plugin_root / "i18n").mkdir()
    _select_plugin(
        "study_companion",
        manifest,
        entries=("study_download_rapidocr_models",),
        load_state="failed",
    )

    assert install_registry.get_install_plugin_registration("study_companion") is None


def test_dynamic_registration_remains_available_for_selected_third_party_plugin(
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "third_party"
    manifest = _write_manifest(plugin_root, "third_party", declaration=None)
    _select_plugin("third_party", manifest, entries=("third_party_install",))
    install_registry.register_install_plugin(
        "third_party",
        install_kinds={
            "models": install_registry.InstallKindRegistration(
                entry_id="third_party_install",
                label="Models",
                queued_message="Models queued",
            )
        },
        tutorial_enabled=True,
    )

    registration = install_registry.get_install_plugin_registration("third_party")

    assert registration is not None
    assert set(registration.install_kinds) == {"models"}
    assert registration.tutorial_enabled is True


def test_stale_dynamic_registration_is_hidden_when_plugin_is_not_selected() -> None:
    install_registry.register_install_plugin(
        "third_party",
        install_kinds={},
    )

    assert install_registry.get_install_plugin_registration("third_party") is None


def test_selected_source_change_during_parse_retries_complete_read_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first" / "study_companion"
    second_root = tmp_path / "second" / "study_companion"
    first_manifest = _write_manifest(
        first_root,
        "study_companion",
        declaration=_STUDY_DECLARATION,
    )
    second_manifest = _write_manifest(
        second_root,
        "study_companion",
        declaration=_STUDY_DECLARATION,
    )
    (first_root / "i18n").mkdir()
    (second_root / "i18n").mkdir()
    _select_plugin(
        "study_companion",
        first_manifest,
        entries=("study_download_rapidocr_models",),
        source="builtin",
    )
    original = install_registry._registration_for_selected_source
    calls = 0

    def switch_after_first_parse(plugin_id: str, selected):
        nonlocal calls
        calls += 1
        registration = original(plugin_id, selected)
        if calls == 1:
            _select_plugin(
                "study_companion",
                second_manifest,
                entries=("study_download_rapidocr_models",),
                source="user",
            )
        return registration

    monkeypatch.setattr(
        install_registry,
        "_registration_for_selected_source",
        switch_after_first_parse,
    )

    registration = install_registry.get_install_plugin_registration("study_companion")

    assert calls == 2
    assert registration is not None
    assert registration.config_path == second_manifest.resolve()
    assert registration.effective_source == "user"
    assert registration.ui_i18n_dir == (second_root / "i18n").resolve()


def test_selected_source_continuously_changing_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [
        tmp_path / "first" / "study_companion",
        tmp_path / "second" / "study_companion",
    ]
    manifests = [
        _write_manifest(root, "study_companion", declaration=_STUDY_DECLARATION)
        for root in roots
    ]
    for root in roots:
        (root / "i18n").mkdir()
    _select_plugin(
        "study_companion",
        manifests[0],
        entries=("study_download_rapidocr_models",),
        source="builtin",
    )
    original = install_registry._registration_for_selected_source
    calls = 0

    def keep_switching(plugin_id: str, selected):
        nonlocal calls
        registration = original(plugin_id, selected)
        calls += 1
        next_index = calls % 2
        _select_plugin(
            "study_companion",
            manifests[next_index],
            entries=("study_download_rapidocr_models",),
            source="user" if next_index else "builtin",
        )
        return registration

    monkeypatch.setattr(
        install_registry,
        "_registration_for_selected_source",
        keep_switching,
    )

    assert install_registry.get_install_plugin_registration("study_companion") is None
    assert calls == 2


def test_tutorial_migration_hooks_for_normalizes_plugin_id(tmp_path: Path) -> None:
    def migrate(_store_path: Path) -> None:
        pass

    install_registry.register_tutorial_migration_hook(
        migrate,
        plugin_id="study_companion",
    )

    assert install_registry.tutorial_migration_hooks_for(" study_companion ") == [migrate]
