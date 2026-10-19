param([Parameter(Mandatory=$true)][string]$ProgressPath)
$ErrorActionPreference='Stop'
$ProgressPreference='Continue'
$Host.UI.RawUI.WindowTitle='RC GPU benchmarks - progress only'
Clear-Host
Write-Host 'RC GPU: integrator and optimization benchmarks' -ForegroundColor Cyan
Write-Host 'Current experiment and numerical settings appear below.'
Write-Host 'Closing this monitor does not stop the benchmark or production simulation.'
while ($true) {
    try {
        $stream=[System.IO.File]::Open($ProgressPath,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
        $reader=[System.IO.StreamReader]::new($stream)
        try {$state=$reader.ReadToEnd() | ConvertFrom-Json} finally {$reader.Dispose()}
    } catch {Start-Sleep -Seconds 2;continue}
    $detail=('{0} configurations complete | elapsed {1:N1} min | budget left {2:N1} min | failed {3}' -f $state.completed_jobs,($state.elapsed_s/60),($state.remaining_budget_s/60),$state.failed_jobs)
    $operation=$state.stage
    if ($null -ne $state.current_job) {
        $job=$state.current_job
        $operation=('{0} | {1} | {2} | batch {3} | interval {4} us | dt {5} ns' -f $job.integrator,$job.variant,$job.fixture,$job.batch,$job.span_us,$job.dt_ns)
    }
    Write-Progress -Id 1 -Activity $state.stage -Status $detail -CurrentOperation $operation
    if ($state.status -ne 'running') {
        Write-Progress -Id 1 -Activity 'GPU benchmark' -Completed
        Write-Host ('Status: {0}. Report: {1}' -f $state.status,(Join-Path (Split-Path $ProgressPath) 'REPORT.md'))
        break
    }
    if (-not (Get-Process -Id $state.runner_pid -ErrorAction SilentlyContinue)) {Write-Host 'Runner exited; inspect logs.';break}
    Start-Sleep -Seconds 3
}
Read-Host 'Press Enter to close'
