# Smoke test for the HIVI model router and retrieval side services.
# Start the UI first (scripts\hivi_ui.ps1), then run this in a second window.
#
#   .\scripts\test_model_router.ps1            # embeddings, rerank, fast + heavy chat
#   .\scripts\test_model_router.ps1 -Review    # also swap to the review model (slow)
param(
  [string]$Base = "http://127.0.0.1:7860",
  [switch]$Review
)
$ErrorActionPreference = "Stop"

function Post($path, $body) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-WebRequest -Uri "$Base$path" -Method Post -ContentType "application/json" `
       -Body ($body | ConvertTo-Json -Depth 8) -TimeoutSec 1800 -UseBasicParsing
  $sw.Stop()
  [pscustomobject]@{ Response = $r; Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1) }
}

function Chat($role) {
  Write-Host "`n== $role ==" -ForegroundColor Cyan
  $res = Post "/v1/chat/completions" @{
    model = $role
    messages = @(@{ role = "user"; content = "Reply with exactly: OK" })
    max_tokens = 16
    stream = $false
  }
  $json = $res.Response.Content | ConvertFrom-Json
  Write-Host ("model: {0}  via: {1}  time: {2}s" -f `
    $res.Response.Headers["X-HIVI-Model"], $res.Response.Headers["X-HIVI-Model-Via"], $res.Seconds)
  Write-Host ("reply: " + $json.choices[0].message.content)
}

Write-Host "== roles ==" -ForegroundColor Cyan
$roles = Invoke-RestMethod "$Base/api/roles"
$roles.roles.PSObject.Properties | ForEach-Object {
  $m = if ($_.Value.model) { $_.Value.model.name } else { "NOT FOUND" }
  "{0,-22} {1,-12} {2}" -f $_.Name, $_.Value.alias, $m
}

Write-Host "`n== embeddings ==" -ForegroundColor Cyan
$e = Post "/v1/embeddings" @{ model = "embedding"; input = @("HIVI local AI", "job application form") }
$ej = $e.Response.Content | ConvertFrom-Json
Write-Host ("vectors: {0}  dims: {1}  time: {2}s" -f $ej.data.Count, $ej.data[0].embedding.Count, $e.Seconds)

Write-Host "`n== rerank ==" -ForegroundColor Cyan
$r = Post "/v1/rerank" @{
  model = "reranker"
  query = "How do I start the HIVI UI?"
  documents = @("Run scripts\hivi_ui.ps1 to launch the UI.", "Bananas are yellow.")
}
$rj = $r.Response.Content | ConvertFrom-Json
$rj.results | ForEach-Object { "doc {0}: score {1:N3}" -f $_.index, $_.relevance_score }
Write-Host "time: $($r.Seconds)s"

Chat "researcher"   # Qwen3.5-9B (swap)
Chat "main-chef"    # Qwen3.6-35B-A3B (swap)
Chat "developer"    # same model, no swap: should be fast
if ($Review) { Chat "security-auditor" }  # Bolt 74B (swap)

Write-Host "`n== side services ==" -ForegroundColor Cyan
Invoke-RestMethod "$Base/api/aux" | ConvertTo-Json -Depth 4
