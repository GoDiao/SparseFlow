[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12",
    [string]$VenvName = ".venv-runtime",
    [string]$UvExe = "",
    [string]$TorchVersion = "2.9.1+cpu",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cpu"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cacheRoot = Join-Path $projectRoot ".cache"
$toolsRoot = Join-Path $projectRoot ".tools"
$venvRoot = Join-Path $projectRoot $VenvName

$env:UV_CACHE_DIR = Join-Path $cacheRoot "uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $toolsRoot "python"
$env:XDG_CACHE_HOME = $cacheRoot
$env:HF_HOME = Join-Path $cacheRoot "huggingface"
$env:TORCH_HOME = Join-Path $cacheRoot "torch"
$env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip"
$env:TEMP = Join-Path $cacheRoot "tmp"
$env:TMP = $env:TEMP
$env:SPARSEFLOW_NATIVE_CACHE = Join-Path $cacheRoot "native\int8_vnni_windows"
$env:PYTHONUTF8 = "1"

@(
    $env:UV_CACHE_DIR,
    $env:UV_PYTHON_INSTALL_DIR,
    $env:HF_HOME,
    $env:TORCH_HOME,
    $env:PIP_CACHE_DIR,
    $env:TEMP,
    $env:SPARSEFLOW_NATIVE_CACHE
) | ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null }

if (-not $UvExe) {
    $localUv = Join-Path $toolsRoot "uv\uv.exe"
    if (Test-Path -LiteralPath $localUv) {
        $UvExe = $localUv
    } else {
        $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCommand) {
            $UvExe = $uvCommand.Source
        }
    }
}
if (-not $UvExe -or -not (Test-Path -LiteralPath $UvExe)) {
    & (Join-Path $PSScriptRoot "bootstrap_uv_windows.ps1")
    if ($LASTEXITCODE -ne 0) { throw "uv bootstrap failed with exit code $LASTEXITCODE" }
    $UvExe = Join-Path $toolsRoot "uv\uv.exe"
}

Write-Host "Project root: $projectRoot"
Write-Host "uv cache:     $env:UV_CACHE_DIR"
Write-Host "Python root:  $env:UV_PYTHON_INSTALL_DIR"
Write-Host "Environment:  $venvRoot"

& $UvExe python install $PythonVersion --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw "uv python install failed with exit code $LASTEXITCODE" }

$pythonExe = Join-Path $venvRoot "Scripts\python.exe"
if (Test-Path -LiteralPath $pythonExe) {
    Write-Host "Reusing existing environment: $venvRoot"
} else {
    & $UvExe venv --python $PythonVersion $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit code $LASTEXITCODE" }
}

& $UvExe pip install --python $pythonExe "torch==$TorchVersion" --index $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "CPU-only PyTorch installation failed with exit code $LASTEXITCODE" }

Push-Location $projectRoot
try {
    & $UvExe pip install --python $pythonExe -e ".[runtime]"
    if ($LASTEXITCODE -ne 0) { throw "SparseFlow runtime installation failed with exit code $LASTEXITCODE" }

    & $pythonExe -c "import torch, transformers; print(f'torch={torch.__version__} cpu_only={not torch.cuda.is_available()} transformers={transformers.__version__}')"
    if ($LASTEXITCODE -ne 0) { throw "Runtime import verification failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "SparseFlow runtime environment is ready."
Write-Host "Open a new PowerShell and run:"
Write-Host "  . .\scripts\use_runtime_windows.ps1"
