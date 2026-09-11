param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$wakeSourceCommit = '11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf'
$wakeOutput = [IO.Path]::GetFullPath($OutputDirectory)
$wakeSource = Join-Path $wakeOutput 'sherpa-onnx'
$wakePatch = Join-Path $PSScriptRoot 'patches/sherpa-onnx-kws-timestamps.patch'
if (Test-Path -LiteralPath $wakeSource) {
    throw "Use a fresh output directory; source already exists: $wakeSource"
}
New-Item -ItemType Directory -Force -Path $wakeOutput | Out-Null
git clone --depth 1 --branch v1.13.8 https://github.com/k2-fsa/sherpa-onnx.git $wakeSource
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
$wakeActualCommit = git -C $wakeSource rev-parse HEAD
if ($wakeActualCommit -ne $wakeSourceCommit) { throw 'Upstream tag commit changed' }
git -C $wakeSource apply --check $wakePatch
if ($LASTEXITCODE -ne 0) { throw 'Patch check failed' }
git -C $wakeSource apply $wakePatch
if ($LASTEXITCODE -ne 0) { throw 'Patch failed' }
$wakeSavedCmake = $env:SHERPA_ONNX_CMAKE_ARGS
try {
    $env:SHERPA_ONNX_CMAKE_ARGS = '-DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_TTS=ON -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON'
    $wakePythonPath = [IO.Path]::GetFullPath($Python).Replace('\', '/')
    $env:SHERPA_ONNX_CMAKE_ARGS += ' -DPython_EXECUTABLE="' + $wakePythonPath + '" -DPYTHON_EXECUTABLE="' + $wakePythonPath + '"'
    Push-Location $wakeSource
    try {
        New-Item -ItemType Directory -Force -Path 'build/sherpa_onnx/bin' | Out-Null
        uv run --no-project --python $Python --with setuptools==83.0.0 --with wheel==0.48.0 python setup.py build --build-temp b
        if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
        uv run --no-project --python $Python --with setuptools==83.0.0 --with wheel==0.48.0 python setup.py bdist_wheel --skip-build
        if ($LASTEXITCODE -ne 0) { throw 'Wheel build failed' }
        Get-ChildItem -LiteralPath (Join-Path $wakeSource 'dist') -Filter '*.whl' | Get-FileHash -Algorithm SHA256
    } finally { Pop-Location }
} finally { $env:SHERPA_ONNX_CMAKE_ARGS = $wakeSavedCmake }
