# Stop the campaign cleanly-enough: watchdog first, then supervisor, then coordinator/workers.
# Source/branch jobs resume from verified savepoints; non-resumable probes replay as whole jobs.
$root = Split-Path -Parent $PSScriptRoot
$rootPattern = [regex]::Escape($root) + '(?=[\\/\s"]|$)'
foreach ($pat in "campaign_watchdog.py","handoff_recorded_queue.py","research_supervisor.py","resume_recorded_research.py","experiments.rl_runner") {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match "python" -and $_.CommandLine -match $rootPattern -and $_.CommandLine -match [regex]::Escape($pat) } |
    ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null; Write-Output "stopped $pat pid $($_.ProcessId)" }
}
