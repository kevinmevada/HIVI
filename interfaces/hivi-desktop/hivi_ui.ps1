# Launch HIVI as a desktop app window — never a browser tab.
#   .\scripts\hivi_ui.ps1              opens the HIVI app; it starts HIVI if needed and
#                                      stops what it started when you close the window
#   .\scripts\hivi_ui.ps1 -ServerOnly  runs the server in this console (logs here), no window;
#                                      open the HIVI app separately and it connects to it
# Stops only what HIVI itself started: a previous HIVI control server and the
# backend recorded in data\run\hivi_backend.json, plus the embedding / reranker
# side services in data\run\hivi_aux_*.json. Other llama-server / Colibrì
# processes (e.g. one you started for something else) are left alone.
param([switch]$ServerOnly)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$uiUrl = "http://127.0.0.1:7860"

if (-not $ServerOnly) {
    $pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
    if (-not (Test-Path $pyw)) { $pyw = $py }
    $app = Join-Path $root "interfaces\hivi-desktop\hivi_app.py"
    Start-Process -FilePath $pyw -ArgumentList "`"$app`"" -WorkingDirectory $root
    Write-Host "Opening the HIVI app window..."
    return
}

# If HIVI is already listening, reuse it. The WMI process lookup below can be
# unavailable in restricted shells; blindly starting a second server then
# makes Uvicorn raise an address-in-use exception (often shown as a debugger
# breakpoint dialog in IDEs).
try {
    $running = Invoke-RestMethod -Uri "$uiUrl/api/status" -TimeoutSec 2
    if ($running.backend -and $null -ne $running.status) {
        Write-Host "HIVI is already running → $uiUrl"
        return
    }
}
catch {}

if (Test-NetConnection -ComputerName "127.0.0.1" -Port 7860 -InformationLevel Quiet -WarningAction SilentlyContinue) {
    Write-Error "Port 7860 is already in use by another service. Stop it or configure HIVI to use another port."
}

Write-Host "Stopping a previous HIVI control server / HIVI-owned backends (if any)..."
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match 'interfaces\\hivi-chat\\server\.py'
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$dataRoot = if ($env:HIVI_DATA_ROOT) { $env:HIVI_DATA_ROOT } else { Join-Path $root "data" }
$runDir = Join-Path $dataRoot "run"
$pidFiles = @(Get-ChildItem $runDir -File -Filter "hivi_backend.json" -ErrorAction SilentlyContinue) +
@(Get-ChildItem $runDir -File -Filter "hivi_aux_*.json" -ErrorAction SilentlyContinue)
foreach ($pidFile in $pidFiles) {
    try {
        $rec = Get-Content $pidFile.FullName -Raw | ConvertFrom-Json
        $proc = if ($rec.pid) { Get-Process -Id $rec.pid -ErrorAction SilentlyContinue } else { $null }
        # Guard against PID reuse after a reboot: only stop it if it still looks like our backend.
        if ($proc -and $proc.ProcessName -match 'llama-server|python') {
            & taskkill /PID $rec.pid /T /F 2>$null | Out-Null
            $what = if ($rec.engine) { $rec.engine } else { $rec.kind }
            Write-Host "  stopped HIVI process PID $($rec.pid) ($what)"
        }
    }
    catch {}
    Remove-Item $pidFile.FullName -Force -ErrorAction SilentlyContinue
}
Start-Sleep 2

Set-Location (Join-Path $root "interfaces\hivi-chat")
Write-Host "HIVI server → $uiUrl  (open the HIVI app to use it)"
& $py server.py --host 127.0.0.1 --port 7860