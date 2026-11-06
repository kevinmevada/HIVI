# Launch HIVI chat UI (model selector → Colibrì or llama.cpp)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "Stopping prior HIVI / coli / llama-server..."
try {
  & $py (Join-Path $env:USERPROFILE "Downloads\colibri-main\colibri-main\c\coli") stop 2>$null
} catch {}
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
  $_.CommandLine -match 'interfaces\\hivi-chat\\server.py|llama-server|coli (web|serve)'
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
foreach ($p in @(7860, 8080, 8000)) {
  Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.OwningProcess -gt 0) { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
  }
}
Start-Sleep 2

Set-Location (Join-Path $root "interfaces\hivi-chat")
Write-Host "HIVI UI → http://127.0.0.1:7860"
Start-Process "http://127.0.0.1:7860"
& $py server.py --host 127.0.0.1 --port 7860
