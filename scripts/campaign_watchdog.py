"""External watchdog: restart research_supervisor.py if it exited with attention_required/crashed.

Does not touch experiments/ or configs/. Read-only on campaign state apart from launching the supervisor.
Stops after 3 restarts for the same last_error, so a real defect is surfaced instead of looped.
"""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"runs/minatar-repair-20261007"
sys.path.insert(0, str(ROOT/"scripts"))
from research_supervisor import alive  # noqa: E402

def read(p):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}

restarts, last = 0, None
while True:
    sup = read(FOLDER/"supervisor.json"); man = read(FOLDER/"manifest.json")
    if man.get("jobs", {}).get("initial_program", {}).get("status") == "complete":
        print("campaign complete"); break
    if not alive(sup.get("supervisor_pid")):
        sig = sup.get("last_error", "")
        restarts = restarts + 1 if sig == last else 1; last = sig
        if restarts > 3:
            print("same failure 3 times; attention needed:", sig); break
        subprocess.Popen([str(ROOT/".venv/Scripts/python.exe"), "-u", "scripts/research_supervisor.py", "--workers", "2"],
                         cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW, stdin=subprocess.DEVNULL)
        print(time.strftime("%H:%M:%S"), "supervisor restarted", sig[:80], flush=True)
        time.sleep(120)
    time.sleep(60)
