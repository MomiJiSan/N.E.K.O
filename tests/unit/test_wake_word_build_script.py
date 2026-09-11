"""Execute build preflight guards without permitting cloning or compilation."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_wake_word_runtime.ps1"
PACKAGING_FLAGS = ("SHERPA_ONNX_SPLIT_PYTHON_PACKAGE", "SHERPA_ONNX_IS_FOR_PYPI")
POWERSHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


@pytest.fixture(params=POWERSHELLS or [None], ids=lambda path: Path(path).stem if path else "no-powershell")
def preflight(request, tmp_path):
    if request.param is None:
        pytest.skip("PowerShell is required to execute Windows build preflight")
    marker = tmp_path / "external-command-called.txt"
    wrapper = tmp_path / "preflight.ps1"
    wrapper.write_text(
        "param([string]$BuildScript, [string]$OutputPath, [string]$PythonPath)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "function global:git {\n"
        "  [IO.File]::AppendAllText($env:NEKO_BUILD_TEST_MARKER, 'git')\n"
        "  throw 'EXTERNAL_COMMAND_BLOCKED'\n"
        "}\n"
        "function global:uv {\n"
        "  [IO.File]::AppendAllText($env:NEKO_BUILD_TEST_MARKER, 'uv')\n"
        "  throw 'EXTERNAL_COMMAND_BLOCKED'\n"
        "}\n"
        "try {\n"
        "  & $BuildScript -Python $PythonPath -OutputDirectory $OutputPath\n"
        "} catch { Write-Output $_.Exception.Message; exit 91 }\n",
        encoding="utf-8",
    )

    def run(output, inherited=None):
        env = {key: value for key, value in os.environ.items() if key not in PACKAGING_FLAGS}
        env.update(inherited or {})
        env["NEKO_BUILD_TEST_MARKER"] = str(marker)
        result = subprocess.run(
            [request.param, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(wrapper), "-BuildScript", str(BUILD_SCRIPT),
             "-OutputPath", str(output), "-PythonPath", sys.executable],
            env=env, capture_output=True, text=True, timeout=20,
        )
        return result, marker

    return run


@pytest.mark.parametrize("flag", PACKAGING_FLAGS)
@pytest.mark.parametrize("value", ["0", "1"])
def test_inherited_packaging_flags_fail_before_clone(preflight, tmp_path, flag, value):
    output = tmp_path / "new output"
    result, marker = preflight(output, {flag: value})
    assert result.returncode == 91
    assert f"Unset {flag}" in result.stdout
    assert not marker.exists(), "Preflight must fail before any external command"
    assert not output.exists()


@pytest.mark.parametrize("kind", ["nonempty_directory", "file"])
def test_existing_output_is_preserved_and_rejected_before_clone(preflight, tmp_path, kind):
    output = tmp_path / "existing output"
    if kind == "nonempty_directory":
        output.mkdir()
        retained = output / "build-manifest.json"
    else:
        retained = output
    retained.write_bytes(b"existing build evidence")
    result, marker = preflight(output)
    assert result.returncode == 91
    assert "fresh empty output directory" in result.stdout
    assert not marker.exists(), "Preflight must fail before any external command"
    assert retained.read_bytes() == b"existing build evidence"


def test_clean_preflight_reaches_only_the_blocked_clone(preflight, tmp_path):
    output = tmp_path / "clean output"
    output.mkdir()
    result, marker = preflight(output)
    assert result.returncode == 91
    assert "EXTERNAL_COMMAND_BLOCKED" in result.stdout
    assert marker.read_text() == "git"
    assert list(output.iterdir()) == []
