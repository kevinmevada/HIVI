# HIVI P1.0 audit launcher (read-only). Shows live progress.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "HIVI P1.0 — READ-ONLY audit" -ForegroundColor Cyan
Write-Host "Does not modify model files. Reports -> data\P1_0\" -ForegroundColor Green
Write-Host "SHA256 of large GGUFs can take a long time — progress prints below." -ForegroundColor Yellow
Write-Host ""
uv run python scripts/p1_0/run.py
Write-Host ""
Write-Host "Done. Open: data\P1_0\P1_0_FINAL_CERTIFICATION.md" -ForegroundColor Green
