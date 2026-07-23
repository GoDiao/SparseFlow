[CmdletBinding()]
param(
    [string]$Version = "0.11.31",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$toolsDir = Join-Path $projectRoot ".tools\uv"
$cacheDir = Join-Path $projectRoot ".cache\tmp\uv-bootstrap"
$uvExe = Join-Path $toolsDir "uv.exe"

if ((Test-Path -LiteralPath $uvExe) -and -not $Force) {
    Write-Host "uv already available: $uvExe"
    & $uvExe --version
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force $toolsDir, $cacheDir | Out-Null
$artifact = "uv-x86_64-pc-windows-msvc.zip"
$baseUri = "https://github.com/astral-sh/uv/releases/download/$Version"
$archive = Join-Path $cacheDir $artifact
$checksum = "$archive.sha256"

Write-Host "Downloading uv $Version to the project drive..."
Invoke-WebRequest "$baseUri/$artifact" -OutFile $archive
Invoke-WebRequest "$baseUri/$artifact.sha256" -OutFile $checksum

$expected = ((Get-Content -LiteralPath $checksum -Raw).Trim() -split "\s+")[0].ToLowerInvariant()
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $expected -or $actual -ne $expected) {
    throw "uv archive checksum mismatch: expected=$expected actual=$actual"
}

$extractDir = Join-Path $cacheDir "extract"
if (Test-Path -LiteralPath $extractDir) {
    Remove-Item -LiteralPath $extractDir -Recurse -Force
}
Expand-Archive -LiteralPath $archive -DestinationPath $extractDir -Force
$downloadedUv = Get-ChildItem -LiteralPath $extractDir -Filter "uv.exe" -File -Recurse |
    Select-Object -First 1
if (-not $downloadedUv) {
    throw "uv.exe was not found in $artifact"
}
Copy-Item -LiteralPath $downloadedUv.FullName -Destination $uvExe -Force

Write-Host "uv installed: $uvExe"
& $uvExe --version
exit $LASTEXITCODE
