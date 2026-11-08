# HIVI End-to-End Project Room Validation (live :8080 + :8888).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Missing $py" -ForegroundColor Red
    exit 1
}

Write-Host "Running project-room E2E against live inference + Hindsight..." -ForegroundColor Cyan
& $py -m pytest tests/e2e/test_project_room_live.py -q --tb=short -s
exit $LASTEXITCODE
