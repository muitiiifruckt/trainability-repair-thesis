"""Keep one authorized local research coordinator alive; never change its seeds.

Normal completion stops supervision. Repeated identical infrastructure errors
leave the queue intact and request attention instead of an endless retry loop.
The Codex heartbeat handles diagnosis; this helper only resumes existing code.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def alive(pid):
    if not pid:
        return False
    from ctypes import wintypes
    library = ctypes.WinDLL("kernel32",use_last_error=True)
    library.OpenProcess.argtypes = (wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    library.OpenProcess.restype = wintypes.HANDLE
    library.GetExitCodeProcess.argtypes = (wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
    library.GetExitCodeProcess.restype = wintypes.BOOL
    library.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = library.OpenProcess(0x1000,False,int(pid))
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(library.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value == 259
    finally:
        library.CloseHandle(handle)


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError,json.JSONDecodeError):
        return {}


def write(path,value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path); return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def supervise(config_path,interval=30,workers=2):
    config = read(config_path)
    folder = (ROOT/config["output_root"]).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Research output must stay in workspace")
    folder.mkdir(parents=True,exist_ok=True)
    status_path = folder/"supervisor.json"
    import msvcrt
    lock = (folder/"supervisor.lock").open("a+b")
    lock.write(b"0"); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state = read(status_path)
    state.update(status="monitoring",supervisor_pid=__import__("os").getpid(),worker_limit=workers)
    child = None
    try:
        while True:
            manifest = read(folder/"manifest.json")
            if manifest.get("jobs",{}).get("initial_program",{}).get("status") == "complete":
                state.update(status="complete",updated_at=time.time()); write(status_path,state); return
            coordinator = manifest.get("worker_pid")
            if alive(coordinator) or (child is not None and child.poll() is None):
                state.update(status="monitoring",worker_pid=coordinator,updated_at=time.time())
                write(status_path,state); time.sleep(interval); continue
            # A killed parent can leave active source workers. Wait for their
            # leases to clear before starting another coordinator on that queue.
            leases = [read(path) for path in (folder/"workers").glob("*/*/worker_lease.json")]
            if any(alive(lease.get("pid")) for lease in leases):
                state.update(status="waiting_for_existing_workers",updated_at=time.time())
                write(status_path,state); time.sleep(interval); continue
            error = manifest.get("worker_error",{})
            signature = (error.get("type","")+":"+error.get("message",""))
            if "MemoryError" in signature:
                state["worker_limit"] = max(1,min(state["worker_limit"],2))
                if state.get("last_error") == signature:
                    state["worker_limit"] = 1
            if signature and signature == state.get("last_error"):
                state["same_error_restarts"] = state.get("same_error_restarts",0)+1
            else:
                state["last_error"] = signature
                state["same_error_restarts"] = 0
            if state["same_error_restarts"] >= 3:
                state.update(status="attention_required",updated_at=time.time()); write(status_path,state); return
            with (folder/"supervised-worker.log").open("ab",buffering=0) as log:
                child = subprocess.Popen([sys.executable,"-u","-m","experiments.rl_runner","auto",
                    "--resume","--workers",str(state["worker_limit"]),"--config",str(config_path)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(status="restarted",worker_pid=child.pid,updated_at=time.time(),
                         restarts=state.get("restarts",0)+1)
            write(status_path,state); time.sleep(interval)
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/research_program.json")
    parser.add_argument("--interval",type=int,default=30)
    parser.add_argument("--workers",type=int,choices=(1,2,3,4),default=2)
    args = parser.parse_args()
    supervise(args.config.resolve(),args.interval,args.workers)
