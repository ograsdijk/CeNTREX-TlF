param([Parameter(Mandatory=$true)][string]$ProgressPath)
$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='RC and R2 GPU experiments - progress only'
Clear-Host
Write-Host 'RC + R(2): GPU kernel, integrator, waveform and transfer tests' -ForegroundColor Cyan
Write-Host 'No time cap. Closing this monitor does not stop the benchmarks.'
while ($true) {
    try {$state=Get-Content -LiteralPath $ProgressPath -Raw | ConvertFrom-Json} catch {Start-Sleep -Seconds 2;continue}
    $detail=('{0} jobs complete | elapsed {1:N1} min | failed jobs {2}' -f $state.completed_jobs,($state.elapsed_s/60),$state.failed_jobs)
    $operation=$state.stage
    if ($null -ne $state.current_job) {
        $j=$state.current_job
        $operation=('{0}: {1}, batch {2}, step {3} ns, threads {4}, tile {5}, wave table {6}' -f $j.system,$j.integrator,$j.batch,$j.dt_ns,$j.threads,$j.tile,$j.waveform)
    }
    Write-Progress -Id 1 -Activity $state.stage -Status $detail -CurrentOperation $operation
    if ($state.status -ne 'running') {Write-Progress -Id 1 -Activity 'GPU tests' -Completed;Write-Host $state.status;break}
    if (-not (Get-Process -Id $state.runner_pid -ErrorAction SilentlyContinue)) {Write-Host 'Runner exited; inspect results and logs.';break}
    Start-Sleep -Seconds 3
}
Read-Host 'Press Enter to close'
