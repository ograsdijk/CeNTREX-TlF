param([Parameter(Mandatory=$true)][string]$ProgressPath)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = 'RC GPU SPARSE BENCHMARK - progress only'
Clear-Host
Write-Host 'RC GPU SPARSE COMPARISON' -ForegroundColor Cyan
Write-Host 'Float64/complex128 | batches up to 4096 | three-hour cap including preparation'
Write-Host 'Production RC runs independently. Closing this window does not stop either run.'
function Read-GpuJson([string]$Path) {
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader = [System.IO.StreamReader]::new($stream)
    try { return ($reader.ReadToEnd() | ConvertFrom-Json) }
    finally { $reader.Dispose() }
}
while ($true) {
    try { $state = Read-GpuJson $ProgressPath }
    catch { Start-Sleep -Seconds 2; continue }
    $detail = ('{0} jobs recorded | elapsed {1:N1} min | budget remaining {2:N1} min | failed {3} | timeouts {4}' -f $state.completed_jobs,($state.elapsed_s/60),($state.remaining_budget_s/60),$state.failed_jobs,$state.timed_out_jobs)
    $operation = $state.stage
    if ($null -ne $state.current_job) {
        $job = $state.current_job
        $operation = ('{0} | {1} | batch={2} | substeps/quarter={3}' -f $job.kind,$job.method,$job.batch,$job.substeps)
    }
    Write-Progress -Id 1 -Activity $state.stage -Status $detail -CurrentOperation $operation
    if ($state.status -ne 'running') {
        Write-Progress -Id 1 -Activity 'GPU suite' -Completed
        Write-Host ('Suite status: {0}. Report: {1}' -f $state.status,(Join-Path (Split-Path $ProgressPath) 'REPORT.md'))
        break
    }
    if (-not (Get-Process -Id $state.runner_pid -ErrorAction SilentlyContinue)) {
        Write-Host 'Runner ended; inspect runner.stderr.log and REPORT.md.' -ForegroundColor Yellow
        break
    }
    Start-Sleep -Seconds 3
}
Read-Host 'Press Enter to close'
