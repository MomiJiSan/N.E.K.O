from dataclasses import replace

import pytest

from main_logic.asr_client import local_service as local
from tests.unit.test_asr_local_service import Service, clean_env, managed  # noqa: F401


def executable_config(monkeypatch):
    for name, value in {
        "NEKO_ASR_SERVICE_EXE": "C:\\package\\bin\\asr\\neko-asr.exe",
        "NEKO_ASR_SERVICE_ROOT": "C:\\package\\bin\\asr",
        "NEKO_ASR_MODEL_DIR": "C:\\package\\models\\asr\\paraformer",
        "NEKO_ASR_SMOKE_WAV": "C:\\package\\models\\asr\\smoke.wav",
    }.items():
        monkeypatch.setenv(name, value)


def test_executable_probe_is_syntax_only(monkeypatch):
    executable_config(monkeypatch)
    monkeypatch.setattr(local.Path, "is_file", lambda *_: pytest.fail("probe accessed filesystem"))
    config = local.configuration()
    assert config.executable.endswith("neko-asr.exe")
    assert config.python == ""
    assert local.configuration_is_available()


def test_executable_and_python_are_mutually_exclusive(monkeypatch):
    executable_config(monkeypatch)
    monkeypatch.setenv("NEKO_ASR_SERVICE_PYTHON", "C:\\python.exe")
    assert not local.configuration_is_available()


@pytest.mark.parametrize("name", ["NEKO_ASR_SERVICE_EXE", "NEKO_ASR_SERVICE_ROOT", "NEKO_ASR_MODEL_DIR", "NEKO_ASR_SMOKE_WAV"])
@pytest.mark.parametrize("value", ["", "relative"])
def test_incomplete_executable_config_does_not_fall_back(monkeypatch, name, value):
    executable_config(monkeypatch)
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("NEKO_ASR_URL", local.DEFAULT_URL)
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 64)
    assert not local.configuration_is_available()


def test_external_endpoint_still_supported_without_managed_fields(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_URL", local.DEFAULT_URL)
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 64)
    assert isinstance(local.configuration(), local.Endpoint)


def test_executable_validates_binary_and_root_without_python_source(tmp_path):
    exe = tmp_path / "neko-asr.exe"
    exe.write_bytes(b"fixture-not-executed")
    config = local.ManagedConfig("", str(tmp_path), str(tmp_path / "model"), str(tmp_path / "smoke"), executable=str(exe))
    local.LocalServiceManager._validate_files(config)
    with pytest.raises(local.LocalServiceError, match="CONFIG_INVALID"):
        local.LocalServiceManager._validate_files(replace(config, executable=str(tmp_path / "missing.exe")))
    with pytest.raises(local.LocalServiceError, match="CONFIG_INVALID"):
        local.LocalServiceManager._validate_files(replace(config, python="C:\\python.exe"))


@pytest.mark.asyncio
async def test_frozen_service_uses_direct_argv_and_drops_interpreter_paths(monkeypatch):
    service = Service()
    manager = managed(service, monkeypatch)
    manager.config = replace(manager.config, python="", executable="C:\\package path\\asr\\neko-asr.exe")
    monkeypatch.setenv("PYTHONPATH", "C:\\development-only")
    monkeypatch.setenv("PYTHONHOME", "C:\\development-only")
    await manager._start()
    args, kwargs = service.spawn_args
    assert args[0] == manager.config.executable
    assert args[1] == "--lazy" and "-m" not in args
    assert kwargs["cwd"] == manager.config.root
    assert "PYTHONPATH" not in kwargs["env"] and "PYTHONHOME" not in kwargs["env"]
    assert len(kwargs["env"]["NEKO_ASR_TOKEN"]) == 64
    await manager.close()
