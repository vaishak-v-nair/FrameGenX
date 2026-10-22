# Detached continuation: after the 15-prompt ablation completes, finish the
# full 30-prompt set (resume skips completed rows), then render results.
# Launched hidden by the build session; safe to kill any time — resume covers it.
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $repoRoot 'ablation.log'
$fullLog = Join-Path $repoRoot 'full30.log'
while (!(Test-Path $log) -or !(Select-String -Path $log -Pattern "ABLATION COMPLETE" -Quiet)) {
    Start-Sleep -Seconds 120
}
Set-Location -LiteralPath $repoRoot
"=== continuing to full 30 ($(Get-Date)) ===" | Out-File -Append -Encoding utf8 $fullLog
$failed = $false
foreach ($c in @("baseline", "rag", "rag_lora", "full")) {
    & venv\Scripts\python.exe benchmark.py --config $c --limit 0 >> $fullLog 2>&1
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}
& venv\Scripts\python.exe scripts\make_results.py >> $fullLog 2>&1
if ($LASTEXITCODE -ne 0) { $failed = $true }
if ($failed) {
    "=== FULL30 INCOMPLETE: review errors and rerun ===" | Out-File -Append -Encoding utf8 $fullLog
    exit 1
}
"=== FULL30 COMPLETE ($(Get-Date)) ===" | Out-File -Append -Encoding utf8 $fullLog
