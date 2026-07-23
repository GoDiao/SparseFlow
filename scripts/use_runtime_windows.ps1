$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cacheRoot = Join-Path $projectRoot ".cache"

$env:UV_CACHE_DIR = Join-Path $cacheRoot "uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".tools\python"
$env:XDG_CACHE_HOME = $cacheRoot
$env:HF_HOME = Join-Path $cacheRoot "huggingface"
$env:TORCH_HOME = Join-Path $cacheRoot "torch"
$env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip"
$env:TEMP = Join-Path $cacheRoot "tmp"
$env:TMP = $env:TEMP
$env:SPARSEFLOW_NATIVE_CACHE = Join-Path $cacheRoot "native\int8_vnni_windows"
$env:PYTHONPATH = Join-Path $projectRoot "src"

@(
    $env:UV_CACHE_DIR,
    $env:UV_PYTHON_INSTALL_DIR,
    $env:HF_HOME,
    $env:TORCH_HOME,
    $env:PIP_CACHE_DIR,
    $env:TEMP,
    $env:SPARSEFLOW_NATIVE_CACHE
) | ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null }

$activate = Join-Path $projectRoot ".venv-runtime\Scripts\Activate.ps1"
if (-not (Test-Path -LiteralPath $activate)) {
    throw "Missing .venv-runtime. Run .\scripts\setup_windows.ps1 first."
}
. $activate
Set-Location $projectRoot
Write-Host "SparseFlow runtime active: $projectRoot"
