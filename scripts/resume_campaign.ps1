# Resume the minatar-repair-20261007 campaign (safe to run repeatedly; the supervisor lock prevents duplicates).
# Usage (from project root):  powershell -ExecutionPolicy Bypass -File scripts\resume_campaign.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "-u","scripts\research_supervisor.py","--workers","2","--interval","30" `
  -WindowStyle Hidden -RedirectStandardOutput "runs\supervisor.stdout.log" -RedirectStandardError "runs\supervisor.stderr.log"
Write-Output "supervisor started; check runs\minatar-repair-20261007\supervisor.json and progress.json"
