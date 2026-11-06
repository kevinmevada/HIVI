# HIVI project start — always runs P0 USB inspection first (read-only).

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "HIVI project: $Root" -ForegroundColor Green
Write-Host "First step: P0 USB inspection" -ForegroundColor Green

& (Join-Path $PSScriptRoot "p0_usb_inspect.ps1")

$venvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "Virtual env activated (.venv)" -ForegroundColor Green
} else {
    Write-Host "No .venv found — run: uv sync" -ForegroundColor Yellow
}

Write-Host "Checklist report: data\p0_checklist_latest.txt" -ForegroundColor Green
Write-Host "Ready. Next P0: SHA256, load/inference tests." -ForegroundColor Green
