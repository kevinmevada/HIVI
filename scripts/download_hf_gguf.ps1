# Download a Hugging Face repo/file with a visible progress bar (run in YOUR VS Code terminal).
# Single file:
#   .\scripts\download_hf_gguf.ps1 -Repo "unsloth/Qwen3.8-27B-GGUF" -Include "Qwen3.8-27B-UD-Q5_K_M.gguf" -OutDir "Models\..."
# Full repo (e.g. Colibrì container):
#   .\scripts\download_hf_gguf.ps1 -Repo "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64" -OutDir "Models\..."

param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [string]$Include = "",
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$SourceNote = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $Root $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ""
Write-Host "========== HIVI HF DOWNLOAD ==========" -ForegroundColor Cyan
Write-Host "Repo:    $Repo"
Write-Host ("Include: {0}" -f ($(if ($Include) { $Include } else { "(entire repo)" })))
Write-Host "Out:     $OutDir"
Write-Host "Progress bar appears below (keep this terminal focused)." -ForegroundColor Green
Write-Host ""

# USB may be unplugged — never inherit stale D:\HIVI HF paths
$HfHome = Join-Path $Root "Models\.hf_cache"
New-Item -ItemType Directory -Force -Path $HfHome | Out-Null
$env:HF_HOME = $HfHome
$env:HUGGINGFACE_HUB_CACHE = Join-Path $HfHome "hub"
$env:HF_HUB_CACHE = $env:HUGGINGFACE_HUB_CACHE
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "0"
$env:PYTHONUNBUFFERED = "1"
$env:TQDM_MININTERVAL = "0.2"

Write-Host "HF_HOME: $env:HF_HOME (local; Xet off)" -ForegroundColor DarkGray
Write-Host ""

if ($Include) {
    uv run hf download $Repo --include $Include --local-dir $OutDir
} else {
    uv run hf download $Repo --local-dir $OutDir
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Download failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

$src = Join-Path $OutDir "SOURCE.txt"
if (-not (Test-Path $src)) {
    @"
Model repository: https://huggingface.co/$Repo
File include: $(if ($Include) { $Include } else { "(entire repo)" })
Download date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
$SourceNote
"@ | Set-Content $src -Encoding utf8
}

Write-Host ""
Write-Host "========== DOWNLOAD COMPLETE ==========" -ForegroundColor Green
Get-ChildItem $OutDir -File | Select-Object Name, @{N = 'GB'; E = { [math]::Round($_.Length / 1GB, 2) } } | Format-Table -AutoSize
$shardBytes = (Get-ChildItem $OutDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
Write-Host ("Total size: {0:N2} GB" -f ($shardBytes / 1GB))
Write-Host "Folder: $OutDir"
