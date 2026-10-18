param(
    [string]$Deadline = '2026-10-12T02:55:49+00:00',
    [string]$Output = 'C:\Users\ogras\Documents\GitHub\CeNTREX-TlF\benchmarks\rc_gpu_sparse_results\v1_20261011_235549'
)
$ErrorActionPreference = 'Stop'
$taskGpuRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskGpuRoot
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:RAYON_NUM_THREADS = '12'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:CUPY_CACHE_DIR = Join-Path $PSScriptRoot '.cupy_cache'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$taskGpuDeadline = [DateTimeOffset]::Parse($Deadline)
$taskGpuRemaining = ($taskGpuDeadline - [DateTimeOffset]::UtcNow).TotalMilliseconds
if ($taskGpuRemaining -le 0) { throw 'The authorized benchmark deadline has passed.' }
$taskGpuPython = Join-Path $PSScriptRoot '.venv-gpu-bench\Scripts\python.exe'
$taskGpuScript = Join-Path $PSScriptRoot 'run_rc_gpu_sparse_suite.py'
$taskGpuProcess = Start-Process -FilePath $taskGpuPython -ArgumentList @('-u',$taskGpuScript,'--output',$Output,'--deadline',$Deadline) -WorkingDirectory $taskGpuRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Output 'runner.stdout.log') -RedirectStandardError (Join-Path $Output 'runner.stderr.log') -PassThru
$taskGpuProgress = Join-Path $Output 'progress.json'
while (-not $taskGpuProcess.HasExited -and -not (Test-Path -LiteralPath $taskGpuProgress)) {
    Start-Sleep -Seconds 1
    $taskGpuProcess.Refresh()
    if ([DateTimeOffset]::UtcNow -ge $taskGpuDeadline) { break }
}
if (Test-Path -LiteralPath $taskGpuProgress) {
    Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'rc_gpu_sparse_progress.ps1'),'-ProgressPath',$taskGpuProgress) -WindowStyle Normal | Out-Null
}
$taskGpuRemaining = [Math]::Max(1,($taskGpuDeadline - [DateTimeOffset]::UtcNow).TotalMilliseconds)
if (-not $taskGpuProcess.WaitForExit([int]$taskGpuRemaining)) {
    # This root was created above; only its benchmark descendants are terminated.
    & taskkill.exe /PID $taskGpuProcess.Id /T /F
    'External deadline watchdog stopped the benchmark tree.' | Set-Content -LiteralPath (Join-Path $Output 'deadline-watchdog.log')
}
