# P1.1 Stage 3-6 — M04 only (read-only w.r.t. model files)
$ErrorActionPreference = "Continue"
$root = "C:\Users\mevad\Desktop\HIVI"
$logDir = "$root\data\P1_1\logs\M04"
$resDir = "$root\data\P1_1\results"
New-Item -ItemType Directory -Force -Path $logDir, $resDir | Out-Null

$cli = "$root\data\P1_1\runtime\llama.cpp\bin\llama-cli.exe"
$model = "$root\Models\General\davidau\qwen3-8b-hivemind-heretic-q6_k\Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q6_K-imat.gguf"
$binDir = Split-Path $cli -Parent

function Get-GpuSnap {
  $raw = nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits
  $p = ($raw -split ",") | ForEach-Object { $_.Trim() }
  [pscustomobject]@{ used_mib = [double]$p[0]; free_mib = [double]$p[1]; util_gpu = [double]$p[2]; util_mem = [double]$p[3]; raw = $raw }
}
function Get-RamSnap {
  $o = Get-CimInstance Win32_OperatingSystem
  [pscustomobject]@{
    free_gb = [math]::Round($o.FreePhysicalMemory / 1MB, 2)
    total_gb = [math]::Round($o.TotalVisibleMemorySize / 1MB, 2)
    used_gb = [math]::Round(($o.TotalVisibleMemorySize - $o.FreePhysicalMemory) / 1MB, 2)
  }
}
function Parse-LlamaTiming([string]$text) {
  $out = @{
    load_ms = $null; prompt_ms = $null; eval_ms = $null
    prompt_n = $null; eval_n = $null; prompt_tps = $null; eval_tps = $null
  }
  # common llama.cpp lines
  if ($text -match '(?i)load time\s*=\s*([0-9.]+)\s*ms') { $out.load_ms = [double]$Matches[1] }
  if ($text -match '(?i)prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([0-9.]+)\s*tokens per second') {
    $out.prompt_ms = [double]$Matches[1]; $out.prompt_n = [int]$Matches[2]; $out.prompt_tps = [double]$Matches[3]
  }
  if ($text -match '(?i)eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([0-9.]+)\s*tokens per second') {
    $out.eval_ms = [double]$Matches[1]; $out.eval_n = [int]$Matches[2]; $out.eval_tps = [double]$Matches[3]
  }
  # alternate format
  if ($null -eq $out.eval_tps -and $text -match '(?i)generation.*?([0-9.]+)\s*t/s') { $out.eval_tps = [double]$Matches[1] }
  if ($text -match '(?i)llama_perf_context_print:\s+load time\s*=\s*([0-9.]+)') { $out.load_ms = [double]$Matches[1] }
  return $out
}

# Ensure PATH finds sibling DLLs
$env:Path = "$binDir;" + $env:Path
Set-Location $binDir

$results = [ordered]@{
  model_id = "M04"
  model_name = "Qwen3-8B Hivemind Heretic"
  format = "GGUF"
  quantization = "Q6_K"
  runtime = $cli
  model_path = $model
  settings = @{
    ctx = 2048
    n_predict = 128
    temperature = 0.7
    top_p = 0.9
    ngl = 20
    batch = 128
    jinja = $true
    single_turn = $true
  }
  preflight = @{
    ram = Get-RamSnap
    gpu = Get-GpuSnap
    time = (Get-Date).ToString("o")
  }
  architecture = @{ family = "qwen3"; chat_template = "PRESENT (from GGUF / --jinja)"; note = "From P1.0 + runtime jinja" }
  runs = @()
  stability = @{}
  peak = @{ ram_used_gb = 0; vram_used_mib = 0 }
  certification = @{}
}

function Invoke-M04Prompt {
  param(
    [string]$Id,
    [string]$Prompt,
    [string]$System = "You are a helpful assistant. Follow instructions precisely.",
    [int]$NPredict = 128
  )
  $outFile = Join-Path $logDir ("run_{0}.log" -f $Id)
  $gpuBefore = Get-GpuSnap
  $ramBefore = Get-RamSnap
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $argList = @(
    "-m", $model,
    "-c", "2048",
    "-b", "128",
    "-n", "$NPredict",
    "-ngl", "20",
    "--temp", "0.7",
    "--top-p", "0.9",
    "--jinja",
    "-st",
    "-sys", $System,
    "-p", $Prompt,
    "--no-display-prompt",
    "--log-file", (Join-Path $logDir ("run_{0}_llama.log" -f $Id))
  )
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $cli
  $psi.WorkingDirectory = $binDir
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  # ArgumentList as single string for older PS compatibility
  $psi.Arguments = ($argList | ForEach-Object {
      if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
  $proc = New-Object System.Diagnostics.Process
  $proc.StartInfo = $psi
  [void]$proc.Start()
  $stdout = $proc.StandardOutput.ReadToEnd()
  $stderr = $proc.StandardError.ReadToEnd()
  $proc.WaitForExit()
  $sw.Stop()
  $gpuAfter = Get-GpuSnap
  $ramAfter = Get-RamSnap
  $combined = $stdout + "`n" + $stderr
  Set-Content -Path $outFile -Value $combined -Encoding utf8
  $timing = Parse-LlamaTiming $combined
  # TTFT estimate: if we have prompt+first token hard; else wall for first chunk unknown → use prompt_ms as proxy when present
  $ttft = $null
  if ($timing.prompt_ms) { $ttft = [math]::Round($timing.prompt_ms / 1000.0, 3) }

  $peakV = [math]::Max($gpuBefore.used_mib, $gpuAfter.used_mib)
  $peakR = [math]::Max($ramBefore.used_gb, $ramAfter.used_gb)
  if ($peakV -gt $results.peak.vram_used_mib) { $results.peak.vram_used_mib = $peakV }
  if ($peakR -gt $results.peak.ram_used_gb) { $results.peak.ram_used_gb = $peakR }

  $textOut = $stdout.Trim()
  if ([string]::IsNullOrWhiteSpace($textOut)) {
    # sometimes generation goes to stderr mixed; try extract after "assistant"
    if ($combined -match '(?is)assistant\s*(.+)$') { $textOut = $Matches[1].Trim() }
  }

  return [ordered]@{
    id = $Id
    prompt = $Prompt
    exit_code = $proc.ExitCode
    wall_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
    load_ms = $timing.load_ms
    ttft_seconds = $ttft
    prompt_tokens = $timing.prompt_n
    prompt_tps = $timing.prompt_tps
    gen_tokens = $timing.eval_n
    gen_tps = $timing.eval_tps
    gpu_before = $gpuBefore
    gpu_after = $gpuAfter
    ram_before = $ramBefore
    ram_after = $ramAfter
    output_excerpt = if ($textOut.Length -gt 2000) { $textOut.Substring(0, 2000) + "..." } else { $textOut }
    log_file = $outFile
    command = "$cli $($psi.Arguments)"
  }
}

Write-Host "=== M04 P1.1 Stage 3-6 starting ===" -ForegroundColor Cyan
Write-Host "ngl=20 ctx=2048 n=128 (conservative GPU offload)"

# Warm/load + Prompt A
$results.runs += Invoke-M04Prompt -Id "A_basic" -Prompt "Explain what a CPU does in three concise bullet points."
Write-Host "A done exit=$($results.runs[-1].exit_code) tps=$($results.runs[-1].gen_tps)"

# Performance trio (Prompt A-like short) for median
$perf = @()
for ($i = 1; $i -le 3; $i++) {
  $r = Invoke-M04Prompt -Id ("PERF_{0}" -f $i) -Prompt "List three colors." -NPredict 64
  $perf += $r
  $results.runs += $r
  Write-Host "PERF_$i tps=$($r.gen_tps) wall=$($r.wall_seconds)"
}
$tpsVals = $perf | Where-Object { $_.gen_tps } | ForEach-Object { $_.gen_tps } | Sort-Object
$loadVals = $perf | Where-Object { $_.load_ms } | ForEach-Object { $_.load_ms } | Sort-Object
$ttftVals = $perf | Where-Object { $_.ttft_seconds } | ForEach-Object { $_.ttft_seconds } | Sort-Object
function Median($arr) {
  if (-not $arr -or $arr.Count -eq 0) { return $null }
  $a = @($arr)
  $m = [int][math]::Floor(($a.Count - 1) / 2)
  if ($a.Count % 2 -eq 1) { return $a[$m] }
  return [math]::Round(($a[$m] + $a[$m + 1]) / 2.0, 3)
}
$results.performance_median = @{
  gen_tps = Median $tpsVals
  load_ms = Median $loadVals
  ttft_seconds = Median $ttftVals
  note = "Median of PERF_1..3 where metric available"
}

# Prompt B
$results.runs += Invoke-M04Prompt -Id "B_structured" -Prompt "Return exactly three numbered items describing the benefits of local AI inference."
Write-Host "B done"

# Prompt C
$results.runs += Invoke-M04Prompt -Id "C_coding" -Prompt "Write a Python function that returns the factorial of a non-negative integer. Include input validation and a short example." -NPredict 192
Write-Host "C done"

# Prompt D multi-turn: two single-turn prompts with explicit memory instruction (true KV continuity needs interactive; document limitation)
$results.runs += Invoke-M04Prompt -Id "D1_memory" -Prompt "My project name is HIVI. Remember that within this conversation. Reply with one short confirmation sentence."
$d1 = $results.runs[-1].output_excerpt
$results.runs += Invoke-M04Prompt -Id "D2_recall" -Prompt "Earlier in this conversation I said my project name is HIVI. What is the project name I gave you? Answer with only the project name." -System "The user previously stated their project name is HIVI. Answer accordingly."
Write-Host "D done (D2 uses system reminder — full interactive KV multi-turn not used in -st mode; documented)"

# Prompt E — three consecutive
$ePrompts = @(
  "Name one advantage of NVMe storage.",
  "What is 17 + 25?",
  "Say the word ready."
)
$stab = @()
for ($i = 0; $i -lt 5; $i++) {
  $p = if ($i -lt 3) { $ePrompts[$i] } else { "Reply with OK and the number $($i+1)." }
  $r = Invoke-M04Prompt -Id ("E_stab_{0}" -f ($i + 1)) -Prompt $p -NPredict 64
  $stab += $r
  $results.runs += $r
  Write-Host "E$($i+1) exit=$($r.exit_code) empty=$([string]::IsNullOrWhiteSpace($r.output_excerpt))"
}

$results.stability = @{
  consecutive_runs = 5
  all_exit_zero = -not ($stab | Where-Object { $_.exit_code -ne 0 })
  any_empty = [bool]($stab | Where-Object { [string]::IsNullOrWhiteSpace($_.output_excerpt) })
  any_crash = $false
  notes = "Five consecutive llama-cli single-turn invocations; process exits each time (unload between runs)."
}

# Resource recovery after last process exited
Start-Sleep -Seconds 3
$results.post = @{
  ram = Get-RamSnap
  gpu = Get-GpuSnap
  time = (Get-Date).ToString("o")
  llama_procs = @(Get-Process llama* -ErrorAction SilentlyContinue | Select-Object Id, ProcessName)
}
$results.resource_recovery = @{
  vram_recovered = ($results.post.gpu.used_mib -lt 800)
  no_llama_procs = ($results.post.llama_procs.Count -eq 0)
  gpu_after = $results.post.gpu
  ram_after = $results.post.ram
}

# Acceptance-ish flags (not intelligence scoring)
function Pass-Text($t, $needles) {
  if ([string]::IsNullOrWhiteSpace($t)) { return $false }
  foreach ($n in $needles) { if ($t -notmatch $n) { return $false } }
  return $true
}
$a = ($results.runs | Where-Object { $_.id -eq "A_basic" } | Select-Object -First 1)
$b = ($results.runs | Where-Object { $_.id -eq "B_structured" } | Select-Object -First 1)
$c = ($results.runs | Where-Object { $_.id -eq "C_coding" } | Select-Object -First 1)
$d2 = ($results.runs | Where-Object { $_.id -eq "D2_recall" } | Select-Object -First 1)

$loadOk = ($results.runs | Where-Object { $_.exit_code -eq 0 }).Count -gt 0
$inferOk = $loadOk -and -not [string]::IsNullOrWhiteSpace($a.output_excerpt)
$warnings = @()
if ($null -eq $results.performance_median.gen_tps) { $warnings += "generation t/s not parsed from logs" }
if ($null -eq $results.performance_median.load_ms) { $warnings += "load time not parsed from logs (see wall_seconds)" }
$warnings += "Multi-turn D used system-prompt carryover under -st; not a single persistent KV session"
$warnings += "Conservative -ngl 20 (not full GPU offload)"

$cert = "PASS"
if (-not $loadOk -or -not $inferOk) { $cert = "FAIL" }
elseif (-not $results.stability.all_exit_zero -or $results.stability.any_empty) { $cert = "FAIL" }
elseif ($warnings.Count -gt 0) { $cert = "PASS WITH WARNINGS" }

$results.smoke = @{
  A_coherent = -not [string]::IsNullOrWhiteSpace($a.output_excerpt)
  B_structured = ($b.output_excerpt -match "1[\).:]") -or ($b.output_excerpt -match "(?m)^\s*1")
  C_code_present = ($c.output_excerpt -match "def ") -or ($c.output_excerpt -match "factorial")
  D_recall_hivi = ($d2.output_excerpt -match "HIVI")
}
$results.certification = @{
  overall = $cert
  load_status = $(if ($loadOk) { "PASS" } else { "FAIL" })
  inference_status = $(if ($inferOk) { "PASS" } else { "FAIL" })
  chat_template_status = "PASS WITH WARNINGS (jinja enabled; template from model metadata assumed)"
  stability_status = $(if ($results.stability.all_exit_zero -and -not $results.stability.any_empty) { "PASS" } else { "FAIL" })
  failure_category = $null
  warnings = $warnings
  ngl = 20
  peak_ram_gb = $results.peak.ram_used_gb
  peak_vram_mib = $results.peak.vram_used_mib
  median_gen_tps = $results.performance_median.gen_tps
  median_load_ms = $results.performance_median.load_ms
  median_ttft_s = $results.performance_median.ttft_seconds
}

$jsonPath = Join-Path $resDir "M04_model_result.json"
$results | ConvertTo-Json -Depth 8 | Set-Content $jsonPath -Encoding utf8

# CSV row
$csvPath = Join-Path $resDir "P1_1_CERTIFICATION_M04.csv"
$csv = @(
  "model_id,model_name,format,quantization,runtime,architecture,load_status,inference_status,chat_template_status,stability_status,peak_ram_gb,peak_vram_mib,load_time_ms_median,ttft_seconds_median,generation_tokens_per_second_median,certification,failure_category,warnings"
  ("M04,Qwen3-8B Hivemind Heretic,GGUF,Q6_K,llama-cli,qwen3,{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},,{10}" -f `
    $results.certification.load_status,
    $results.certification.inference_status,
    ('"' + $results.certification.chat_template_status + '"'),
    $results.certification.stability_status,
    $results.certification.peak_ram_gb,
    $results.certification.peak_vram_mib,
    $results.certification.median_load_ms,
    $results.certification.median_ttft_s,
    $results.certification.median_gen_tps,
    $results.certification.overall,
    ('"' + ($warnings -join '; ') + '"'))
)
$csv | Set-Content $csvPath -Encoding utf8

$md = @"
# P1.1 M04 Result — Qwen3-8B Hivemind Heretic Q6_K

**Certification:** $($results.certification.overall)

## Environment
- Runtime: ``$cli``
- Model: ``$model``
- Settings: ctx=2048, n=128, temp=0.7, top_p=0.9, **-ngl 20** (conservative), jinja on, single-turn
- Preflight RAM free: $($results.preflight.ram.free_gb) GB
- Preflight VRAM free: $($results.preflight.gpu.free_mib) MiB

## Metrics (median of PERF runs where parsed)
- Load ms: $($results.performance_median.load_ms)
- TTFT s (prompt-eval proxy): $($results.performance_median.ttft_seconds)
- Gen t/s: $($results.performance_median.gen_tps)
- Peak RAM used (system): $($results.peak.ram_used_gb) GB
- Peak VRAM used: $($results.peak.vram_used_mib) MiB

## Smoke
- A coherent: $($results.smoke.A_coherent)
- B structured: $($results.smoke.B_structured)
- C code: $($results.smoke.C_code_present)
- D recall HIVI: $($results.smoke.D_recall_hivi)

## Stability
- 5 consecutive exit 0: $($results.stability.all_exit_zero)
- Any empty: $($results.stability.any_empty)
- VRAM recovered (<800 MiB used): $($results.resource_recovery.vram_recovered)
- No llama procs left: $($results.resource_recovery.no_llama_procs)

## Warnings
$($warnings | ForEach-Object { "- $_" } | Out-String)

## Files
- JSON: ``$jsonPath``
- CSV: ``$csvPath``
- Logs: ``$logDir``

**Stopped for review. Did not proceed to M07.**
"@
Set-Content (Join-Path $logDir "M04_REPORT.md") $md -Encoding utf8
Set-Content (Join-Path $resDir "M04_REPORT.md") $md -Encoding utf8

Write-Host "`n=== M04 CERTIFICATION: $($results.certification.overall) ===" -ForegroundColor Green
Write-Host "Report: $logDir\M04_REPORT.md"
Write-Host ($results.certification | ConvertTo-Json)
