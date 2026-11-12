# Find the fastest llama.cpp thread count for a model on this PC.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\bench_threads.ps1 [-Model <path.gguf>]
# Stop the model in HIVI first so the benchmark has the RAM and GPU to itself.
param([string]$Model = "")
$root = Split-Path -Parent $PSScriptRoot
$bench = (Get-ChildItem $root -Recurse -Filter llama-bench.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if (-not $bench) { Write-Error "llama-bench.exe not found under $root"; exit 1 }
if (-not $Model) {
  $Model = (Get-ChildItem "$root\Models" -Recurse -Filter *HauhauCS*.gguf -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
}
if (-not $Model) { Write-Error "Pass -Model <path to .gguf>"; exit 1 }
Write-Host "Model: $Model"
Write-Host "Testing generation speed (tokens/s) at different thread counts. Takes a few minutes."
& $bench -m $Model -t 6,8,12,16 -p 0 -n 64 -r 2 --fit on
Write-Host ""
Write-Host "Pick the -t with the highest t/s, then make it permanent:"
Write-Host '  [Environment]::SetEnvironmentVariable("HIVI_THREADS","8","User")   # use your best number'
Write-Host "and restart HIVI."
