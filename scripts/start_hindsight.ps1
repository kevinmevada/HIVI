# Start Hindsight API for HIVI (pg0 + shared llama.cpp at :8080/v1).
# Does NOT start a second inference process.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$hf = Join-Path $Root "Models\.hf_cache"
New-Item -ItemType Directory -Force -Path $hf, (Join-Path $hf "hub"), (Join-Path $Root "data\hindsight") | Out-Null

# Never inherit stale USB D:\ HF paths
$env:HF_HOME = $hf
$env:HUGGINGFACE_HUB_CACHE = Join-Path $hf "hub"
$env:HF_HUB_CACHE = $env:HUGGINGFACE_HUB_CACHE
$env:HF_HUB_DISABLE_XET = "1"

$env:HINDSIGHT_HOME = Join-Path $Root "data\hindsight"
$env:HINDSIGHT_API_LLM_PROVIDER = "openai"
$env:HINDSIGHT_API_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
$env:HINDSIGHT_API_LLM_API_KEY = "hivi-local"
# Auto-resolve model id from shared /v1/models unless an explicit non-sentinel is set.
# "local" is treated as unresolved (llama.cpp/Colibrì rarely advertise that id).
$explicitModel = $env:HINDSIGHT_API_LLM_MODEL
$needsResolve = [string]::IsNullOrWhiteSpace($explicitModel) -or ($explicitModel -eq "local")
if ($needsResolve) {
    try {
        $modelsJson = Invoke-RestMethod -Uri "$($env:HINDSIGHT_API_LLM_BASE_URL)/models" -TimeoutSec 5
        $first = $null
        if ($modelsJson.data -and $modelsJson.data.Count -gt 0) {
            $first = $modelsJson.data[0].id
        } elseif ($modelsJson.models -and $modelsJson.models.Count -gt 0) {
            $first = $modelsJson.models[0].name
        }
        if ($first) {
            $env:HINDSIGHT_API_LLM_MODEL = [string]$first
        }
    } catch {
        Write-Host "WARN: could not query $($env:HINDSIGHT_API_LLM_BASE_URL)/models — $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
if ([string]::IsNullOrWhiteSpace($env:HINDSIGHT_API_LLM_MODEL)) {
    $env:HINDSIGHT_API_LLM_MODEL = "local"
}
$env:HINDSIGHT_API_LLM_MAX_CONCURRENT = "1"
$env:HINDSIGHT_API_LLM_TIMEOUT = "300"
# Colibrì (and some local OpenAI-compat servers) reject response_format grammars.
# Soft path: schema in prompt + parse JSON (no grammar enforcement).
$env:HINDSIGHT_API_LLM_OPENAI_COMPATIBLE_JSON_MODE = "false"
$env:HINDSIGHT_API_LLM_STRICT_SCHEMA = "false"
# Colibrì qwen36 ctx=4096: disable auto-consolidation (prompts overflow 4k) and
# cap retain completion so extraction cannot runaway to 64k default.
$env:HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION = "false"
# Completion budget must exceed chunk size (Hindsight validates this).
$env:HINDSIGHT_API_RETAIN_CHUNK_SIZE = "600"
$env:HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS = "2048"
$env:HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS = "256"
$env:HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS = "512"
$env:HINDSIGHT_API_RETAIN_MAX_CONCURRENT = "1"
$env:HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT = "1"
$env:HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU = "true"
$env:HINDSIGHT_API_EMBEDDINGS_LOCAL_MAX_CONCURRENT = "1"
# Keep embeddings/reranker off the coding GPU
$env:CUDA_VISIBLE_DEVICES = ""

$exe = Join-Path $Root ".venv-hindsight\Scripts\hindsight-api.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Missing $exe — create with: uv venv .venv-hindsight && uv pip install --python .venv-hindsight hindsight-api" -ForegroundColor Red
    exit 1
}

Write-Host "Hindsight → http://127.0.0.1:8888" -ForegroundColor Cyan
Write-Host "LLM       → $($env:HINDSIGHT_API_LLM_BASE_URL) model=$($env:HINDSIGHT_API_LLM_MODEL)" -ForegroundColor DarkGray
Write-Host "HF_HOME   → $env:HF_HOME" -ForegroundColor DarkGray
& $exe --host 127.0.0.1 --port 8888 --log-level info
