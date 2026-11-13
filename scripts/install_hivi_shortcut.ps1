# Create a Windows shortcut that has HIVI's icon instead of Python's icon.
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_hivi_shortcut.ps1
param(
    [switch]$NoDesktop
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$app = Join-Path $root "interfaces\hivi-desktop\hivi_app.py"
$icon = Join-Path $root "interfaces\hivi-desktop\hivi.ico"
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) {
    $pyw = Join-Path $root ".venv\Scripts\python.exe"
}
if (-not (Test-Path $pyw)) {
    $pyw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
}
if (-not $pyw) {
    $pyw = (Get-Command python.exe -ErrorAction Stop).Source
}

foreach ($path in @($app, $icon)) {
    if (-not (Test-Path $path)) {
        throw "HIVI file not found: $path"
    }
}

$shell = New-Object -ComObject WScript.Shell
$locations = @()
if (-not $NoDesktop) {
    $locations += Join-Path ([Environment]::GetFolderPath("Desktop")) "HIVI.lnk"
}
$startMenu = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs\HIVI.lnk"
$locations += $startMenu

foreach ($shortcutPath in $locations) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $pyw
    $shortcut.Arguments = "`"$app`""
    $shortcut.WorkingDirectory = $root
    $shortcut.IconLocation = "$icon,0"
    $shortcut.Description = "HIVI local AI workbench"
    $shortcut.Save()
    Write-Host "Created $shortcutPath"
}

Write-Host ""
Write-Host "Pin the HIVI shortcut (not the running Python window) to the taskbar."
