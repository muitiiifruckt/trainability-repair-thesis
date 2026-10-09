# Stop the campaign cleanly-enough: watchdog first, then supervisor, then coordinator/workers.
# Work is resumable from savepoints written every 5000 steps (loses at most the current partial branches).
$root = Split-Path -Parent $PSScriptRoot
foreach ($pat in "campaign_watchdog.py","research_supervisor.py","experiments.rl_runner") {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match "python" -and $_.CommandLine -match [regex]::Escape($pat) } |
    ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null; Write-Output "stopped $pat pid $($_.ProcessId)" }
}
