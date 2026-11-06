# HIVI P0 — checklist report (read-only)
# USB? PC Models library? USB Models? Ollama?

$ErrorActionPreference = "Continue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pcModelsRoot = Join-Path $projectRoot "Models"
$usbRoot = "D:\HIVI\Models"
$legacyRoot = "D:\HIVI\HIVI_MODELS"
$reportDir = Join-Path $projectRoot "data"
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$reportPath = Join-Path $reportDir "p0_checklist_latest.txt"
$archivePath = Join-Path $reportDir ("p0_checklist_{0}.txt" -f $stamp)
# Buffer in memory — avoids Add-Content lock fights when folderOpen + terminal both run start.ps1
$script:ReportLines = New-Object System.Collections.Generic.List[string]

function Write-Line {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
    [void]$script:ReportLines.Add($Text)
}

function Save-HiviChecklistReport {
    param([string]$LatestPath, [string]$ArchivePath)
    $body = ($script:ReportLines -join [Environment]::NewLine) + [Environment]::NewLine
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    # Unique archive first (never contended across concurrent start.ps1 runs)
    [System.IO.File]::WriteAllText($ArchivePath, $body, $utf8)
    # Pointer file always updated (tiny, rarely locked)
    $pointer = Join-Path (Split-Path $LatestPath -Parent) "p0_checklist_LATEST.path"
    [System.IO.File]::WriteAllText($pointer, $ArchivePath + [Environment]::NewLine, $utf8)

    # Try to refresh latest.txt; if locked (folderOpen + terminal race), keep archive+pointer
    $tmp = "$LatestPath.$PID.tmp"
    [System.IO.File]::WriteAllText($tmp, $body, $utf8)
    $ok = $false
    for ($i = 0; $i -lt 10; $i++) {
        try {
            if (Test-Path -LiteralPath $LatestPath) {
                # Rename locked/stale latest out of the way when possible
                $bak = "$LatestPath.stale_$PID"
                Rename-Item -LiteralPath $LatestPath -NewName (Split-Path $bak -Leaf) -ErrorAction SilentlyContinue
                Remove-Item -LiteralPath $bak -Force -ErrorAction SilentlyContinue
            }
            Move-Item -LiteralPath $tmp -Destination $LatestPath -Force -ErrorAction Stop
            $ok = $true
            break
        } catch {
            Start-Sleep -Milliseconds (200 * ($i + 1))
        }
    }
    if (-not $ok) {
        Write-Host ("Note: {0} is locked by another process — using archive instead." -f $LatestPath) -ForegroundColor Yellow
        Write-Host ("Newest report: {0}" -f $ArchivePath) -ForegroundColor Green
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

function Find-HiviModelPackages {
    param([string]$Root)
    $weightExt = @(".gguf", ".safetensors", ".bin", ".onnx", ".h5", ".pt", ".pth")
    $found = @()
    if (-not (Test-Path $Root)) { return $found }

    $dirs = Get-ChildItem $Root -Directory -Recurse -ErrorAction SilentlyContinue
    foreach ($dir in $dirs) {
        $files = Get-ChildItem $dir.FullName -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notmatch '^\.' -and $_.DirectoryName -notmatch '\\\.cache\\' }
        # only files directly in this dir (not scanning .cache noise as packages)
        $files = Get-ChildItem -LiteralPath $dir.FullName -File -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne ".gitignore" -and $_.Name -ne "CACHEDIR.TAG" -and $_.Extension -ne ".metadata" }
        if (-not $files) { continue }

        $hasWeight = $false
        $hasSource = $false
        $hasSha = $false
        $bytes = 0L
        $weightNames = @()
        foreach ($f in $files) {
            $bytes += $f.Length
            $ext = $f.Extension.ToLowerInvariant()
            if ($weightExt -contains $ext) {
                $hasWeight = $true
                $weightNames += $f.Name
            }
            if ($f.Name -eq "SOURCE.txt") { $hasSource = $true }
            if ($f.Name -eq "SHA256SUMS.txt") { $hasSha = $true }
        }
        if ($hasWeight -or ($hasSource -and $hasSha)) {
            $rel = $dir.FullName.Substring($Root.Length).TrimStart("\")
            $found += [pscustomobject]@{
                Name        = $dir.Name
                Path        = $dir.FullName
                Rel         = $rel
                SizeBytes   = $bytes
                SizeGB      = [math]::Round($bytes / 1GB, 2)
                HasSOURCE   = $hasSource
                HasSHA      = $hasSha
                HasWeights  = $hasWeight
                WeightFiles = ($weightNames -join ", ")
                LastWrite   = ($files | Measure-Object LastWriteTime -Maximum).Maximum
            }
        }
    }
    return $found
}

function Write-ModelList {
    param([object[]]$Models)
    if (-not $Models -or $Models.Count -eq 0) {
        Write-Line "   Count: 0 — no model packages found" "Yellow"
        return
    }
    Write-Line ("   Count: {0}" -f $Models.Count) "Green"
    $i = 1
    foreach ($m in $Models) {
        Write-Line ("   {0}. {1}" -f $i, $m.Name)
        Write-Line ("      Path: {0}" -f $m.Rel)
        if ($m.WeightFiles) {
            Write-Line ("      File: {0}" -f $m.WeightFiles)
        }
        Write-Line ("      Size: {0} GB | SOURCE: {1} | SHA256: {2} | Weights: {3}" -f `
            $m.SizeGB, `
            ($(if ($m.HasSOURCE) { "YES" } else { "NO" })), `
            ($(if ($m.HasSHA) { "YES" } else { "NO" })), `
            ($(if ($m.HasWeights) { "YES" } else { "NO" })))
        Write-Line ("      Last write: {0}" -f $m.LastWrite)
        $i++
    }
}

Write-Line ""
Write-Line "========== HIVI P0 CHECKLIST REPORT ==========" "Cyan"
Write-Line ("Generated: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Line ""

# --- 1. USB connected ---
$vol = Get-Volume -ErrorAction SilentlyContinue | Where-Object { $_.DriveLetter -eq "D" }
$dReady = Test-Path "D:\"
$usbConnected = $false
$usbLabel = "(none)"
$usbFs = "(none)"
$usbHealth = "(none)"
$usbSizeGiB = $null
$usbFreeGiB = $null

if ($dReady -and $vol) {
    $usbConnected = $true
    $usbLabel = if ($vol.FileSystemLabel) { $vol.FileSystemLabel } else { "(unlabeled)" }
    $usbFs = $vol.FileSystem
    $usbHealth = $vol.HealthStatus
    $usbSizeGiB = [math]::Round($vol.Size / 1GB, 2)
    $usbFreeGiB = [math]::Round($vol.SizeRemaining / 1GB, 2)
} elseif ($dReady) {
    $usbConnected = $true
    $usbLabel = "(volume info unavailable)"
}

Write-Line "1. USB CONNECTED?" "Cyan"
if ($usbConnected) {
    Write-Line "   [YES] Drive D: is present" "Green"
    Write-Line ("   Label: {0}" -f $usbLabel)
    Write-Line ("   Filesystem: {0} | Health: {1}" -f $usbFs, $usbHealth)
    if ($null -ne $usbSizeGiB) {
        Write-Line ("   Size: {0} GiB | Free: {1} GiB" -f $usbSizeGiB, $usbFreeGiB)
    }
    Write-Line ("   HIVI volume label? {0}" -f ($(if ($usbLabel -eq "HIVI") { "YES" } else { "NO ($usbLabel)" })))
} else {
    Write-Line "   [NO] Drive D: not found — plug in the HIVI USB" "Yellow"
}
Write-Line ""

# --- 2. PC HIVI Models library (project Models\) ---
Write-Line "2. MODELS ON PC (HIVI library — Models\)" "Cyan"
Write-Line ("   Root: {0}" -f $pcModelsRoot)
$pcLibModels = @()
if (-not (Test-Path $pcModelsRoot)) {
    Write-Line "   [NO library] Path missing" "Yellow"
} else {
    $pcLibModels = @(Find-HiviModelPackages -Root $pcModelsRoot)
    Write-ModelList -Models $pcLibModels
}
Write-Line ""

# --- 3. Models on USB ---
Write-Line "3. MODELS ON USB (D:\HIVI\Models)" "Cyan"
$usbModels = @()
if (-not $usbConnected) {
    Write-Line "   Skipped — USB not connected" "Yellow"
} elseif (-not (Test-Path $usbRoot)) {
    Write-Line "   [NO library] Path missing: $usbRoot" "Yellow"
} else {
    $usbModels = @(Find-HiviModelPackages -Root $usbRoot)
    Write-ModelList -Models $usbModels
}
if (Test-Path $legacyRoot) {
    Write-Line ("   Legacy folder still present: {0}" -f $legacyRoot) "Yellow"
} else {
    Write-Line "   Legacy D:\HIVI\HIVI_MODELS: absent"
}
Write-Line ""

# --- 4. Models on PC (Ollama) ---
Write-Line "4. MODELS ON PC (Ollama — separate runtime store)" "Cyan"
$ollamaModels = @()
$ollamaOk = [bool](Get-Command ollama -ErrorAction SilentlyContinue)

if (-not $ollamaOk) {
    Write-Line "   [NO] ollama not on PATH" "Yellow"
} else {
    $raw = ollama list 2>&1 | Out-String
    $lines = $raw -split "`r?`n" | Where-Object { $_.Trim() -ne "" }
    foreach ($line in $lines) {
        if ($line -match '^\s*NAME\s+') { continue }
        if ($line -match '^\s*(\S+)\s+(\S+)\s+(.+?)\s{2,}(.+)$') {
            $ollamaModels += [pscustomobject]@{
                Name = $Matches[1]
                Id = $Matches[2]
                Size = $Matches[3].Trim()
                Modified = $Matches[4].Trim()
            }
        }
    }
    if ($ollamaModels.Count -eq 0) {
        Write-Line "   Count: 0 — no Ollama models listed" "Yellow"
    } else {
        Write-Line ("   Count: {0}" -f $ollamaModels.Count) "Green"
        $i = 1
        foreach ($m in $ollamaModels) {
            Write-Line ("   {0}. {1}" -f $i, $m.Name)
            Write-Line ("      Size: {0} | Modified: {1} | Id: {2}" -f $m.Size, $m.Modified, $m.Id)
            $i++
        }
    }
}
Write-Line ""

# --- 5. Summary ---
Write-Line "5. CHECKLIST SUMMARY" "Cyan"
Write-Line ("   [ {0} ] USB connected (D:)" -f ($(if ($usbConnected) { "YES" } else { "NO " })))
Write-Line ("   [ {0} ] HIVI label on D:" -f ($(if ($usbLabel -eq "HIVI") { "YES" } else { "NO " })))
Write-Line ("   [ {0} ] PC HIVI Models\ present" -f ($(if (Test-Path $pcModelsRoot) { "YES" } else { "NO " })))
Write-Line ("   [ {0} ] PC library models found ({1})" -f `
    ($(if ($pcLibModels.Count -gt 0) { "YES" } else { "NO " })), $pcLibModels.Count)
Write-Line ("   [ {0} ] USB Models folder present" -f ($(if (Test-Path $usbRoot) { "YES" } else { "NO " })))
Write-Line ("   [ {0} ] USB model packages found ({1})" -f `
    ($(if ($usbModels.Count -gt 0) { "YES" } else { "NO " })), $usbModels.Count)
Write-Line ("   [ {0} ] Ollama available" -f ($(if ($ollamaOk) { "YES" } else { "NO " })))
Write-Line ("   [ {0} ] Ollama models found ({1})" -f `
    ($(if ($ollamaModels.Count -gt 0) { "YES" } else { "NO " })), $ollamaModels.Count)
Write-Line ""
Write-Line "Not checked yet (later P0): SHA256 verify, load test, inference, RAM/VRAM." "DarkGray"
Write-Line ("Report saved: {0}" -f $reportPath) "DarkGray"
Write-Line "========== END CHECKLIST ==========" "Cyan"
Write-Line ""

Save-HiviChecklistReport -LatestPath $reportPath -ArchivePath $archivePath
Write-Host ("Checklist archive: {0}" -f $archivePath) -ForegroundColor DarkGray
