# Fetch third-party inference engines into src/hivi/inference/ (not tracked by git).
#
#   .\scripts\fetch_backends.ps1                    # Colibrì + llama.cpp at the pinned refs
#   .\scripts\fetch_backends.ps1 -Only colibri
#   .\scripts\fetch_backends.ps1 -LlamaRef b11454   # any tag / branch / commit
#   .\scripts\fetch_backends.ps1 -WithAirLLM        # optional, unused by the runtime today
#
# Pins live in docs/BACKENDS.md. Update both together.
param(
    [ValidateSet("all", "colibri", "llama")]
    [string]$Only = "all",
    [string]$ColibriRef = "v2.1.0",
    [string]$LlamaRef = "master",
    [switch]$WithAirLLM,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "src\hivi\inference"

function Get-Backend([string]$Name, [string]$Url, [string]$Ref, [string]$Dir) {
    $target = Join-Path $Dest $Dir
    if (Test-Path $target) {
        if (-not $Force) {
            Write-Host "skip  $Name (exists: $target; use -Force to replace)" -ForegroundColor DarkGray
            return
        }
        Remove-Item -Recurse -Force $target
    }
    Write-Host "fetch $Name @ $Ref" -ForegroundColor Cyan
    git clone --quiet --depth 1 --branch $Ref $Url $target 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Not a branch/tag name: clone, then check out the exact commit.
        git clone --quiet $Url $target
        if ($LASTEXITCODE -ne 0) { throw "clone failed: $Url" }
        git -C $target checkout --quiet $Ref
        if ($LASTEXITCODE -ne 0) { throw "ref not found in ${Name}: $Ref" }
    }
    $sha = (git -C $target rev-parse --short HEAD).Trim()
    Write-Host "      $Name at $sha" -ForegroundColor Green
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
if ($Only -in @("all", "colibri")) {
    Get-Backend "Colibri" "https://github.com/JustVugg/colibri.git" $ColibriRef "colibri-LLM-main"
}
if ($Only -in @("all", "llama")) {
    Get-Backend "llama.cpp" "https://github.com/ggml-org/llama.cpp.git" $LlamaRef "llama.cpp-master"
}
if ($WithAirLLM) {
    Get-Backend "AirLLM" "https://github.com/lyogavin/airllm.git" "main" "airllm-main"
}
Write-Host ""
Write-Host "Next: build llama-server (or set HIVI_LLAMA_SERVER to an existing binary)." -ForegroundColor Yellow
Write-Host "      Colibri: see src\hivi\inference\colibri-LLM-main\c (or set HIVI_COLI)." -ForegroundColor Yellow
