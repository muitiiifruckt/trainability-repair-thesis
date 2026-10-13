"""Keep one authorized local research coordinator alive; never change its seeds.

Normal completion stops supervision. Repeated identical infrastructure errors
leave the queue intact and request attention instead of an endless retry loop.
The Codex heartbeat handles diagnosis; this helper only resumes existing code.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
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


def process_identity(pid):
    """Query image and creation time; no process is stopped or modified."""
    from ctypes import wintypes
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    library.OpenProcess.restype = wintypes.HANDLE
    library.CloseHandle.argtypes = (wintypes.HANDLE,)
    library.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
    library.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    handle = library.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        length = wintypes.DWORD(len(buffer))
        creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not library.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            return None
        if not library.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return {"image_name": Path(buffer.value).name.lower(), "created_at": ticks / 10_000_000 - 11_644_473_600}
    finally:
        library.CloseHandle(handle)


def research_owner_alive(pid, recorded_at=None):
    if not alive(pid):
        return False
    if not isinstance(recorded_at, (int, float)) or recorded_at <= 0:
        return True  # unknown ownership remains conservative
    identity = process_identity(pid)
    # A newer non-Python process cannot own an old Python lease. Python or
    # unreadable identity remains conservative, including possible clock skew.
    return not (identity and identity["image_name"] not in ("python.exe", "pythonw.exe")
                and identity["created_at"] > recorded_at + 2)


def quarantine_reused_leases_locked(folder):
    """Caller must already own worker.lock; preserve all branch data."""
    folder = folder.resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Lease recovery must stay inside workspace")
    manifest = read(folder / "manifest.json")
    if research_owner_alive(manifest.get("worker_pid"), manifest.get("updated_at")):
        raise RuntimeError("A healthy coordinator still owns the queue")
    leases = []
    for path in folder.glob("workers/*/*/worker_lease.json"):
        try:
            lease = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(lease, dict):
                raise ValueError("Lease must be an object")
        except (OSError, ValueError) as error:
            raise RuntimeError("Cannot verify existing worker lease") from error
        leases.append((path, lease))
    if any(research_owner_alive(v.get("pid"), v.get("started_at")) for _, v in leases):
        raise RuntimeError("Healthy or unverifiable child leases remain")
    archived = []
    for path, lease in leases:
        if not alive(lease.get("pid")):
            continue  # frozen child already handles ordinary dead leases
        identity = process_identity(lease["pid"])
        if (not identity or identity["image_name"] in ("python.exe", "pythonw.exe")
                or identity["created_at"] <= lease.get("started_at", float("inf")) + 2):
            raise RuntimeError("Lease reuse proof changed; preserve the file")
        relative = path.resolve().relative_to(folder)
        destination = (folder / "recovery/reused_pid_leases" / relative).with_name(f"lease-{time.time_ns()}.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.resolve().is_relative_to(folder / "recovery"):
            raise ValueError("Lease quarantine escaped campaign")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.replace(destination)
        record = {"old_path": str(relative), "archive": str(destination.relative_to(folder)),
                  "pid": lease["pid"], "lease_started_at": lease["started_at"],
                  "current_identity": identity, "lease_sha256": digest, "time": time.time()}
        write(destination.with_suffix(".audit.json"), record)
        archived.append(record)
    return archived


def quarantine_reused_leases(folder):
    """Remove only leases proved to belong to a newer non-Python process."""
    folder = folder.resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Lease recovery must stay inside workspace")
    if not list(folder.glob("workers/*/*/worker_lease.json")):
        return []
    import msvcrt
    with (folder / "worker.lock").open("r+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            return quarantine_reused_leases_locked(folder)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError,json.JSONDecodeError):
        return {}


def write(path,value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
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
            if research_owner_alive(coordinator, manifest.get("updated_at")) or (child is not None and child.poll() is None):
                state.update(status="monitoring",worker_pid=coordinator,updated_at=time.time())
                write(status_path,state); time.sleep(interval); continue
            # A killed parent can leave active source workers. Wait for their
            # leases to clear before starting another coordinator on that queue.
            leases = [read(path) for path in (folder/"workers").glob("*/*/worker_lease.json")]
            if any(research_owner_alive(lease.get("pid"), lease.get("started_at")) for lease in leases):
                state.update(status="waiting_for_existing_workers",updated_at=time.time())
                write(status_path,state); time.sleep(interval); continue
            try:
                archived = quarantine_reused_leases(folder)
            except (OSError, RuntimeError) as error:
                state.update(status="waiting_for_owner_verification", ownership_error=str(error), updated_at=time.time())
                write(status_path, state); time.sleep(interval); continue
            if archived:
                state.setdefault("reused_pid_leases_quarantined", []).extend(archived)
            error = manifest.get("worker_error",{})
            signature = (error.get("type","")+":"+error.get("message",""))
            if "MemoryError" in signature:
                state["worker_limit"] = max(1,min(state["worker_limit"],2))
                if state.get("last_error") == signature:
                    state["worker_limit"] = 1
            if signature.strip(":") and signature == state.get("last_error"):
                state["same_error_restarts"] = state.get("same_error_restarts",0)+1
            else:
                state["last_error"] = signature
                state["same_error_restarts"] = 0
            if state["same_error_restarts"] >= 3:
                state.update(status="attention_required",updated_at=time.time()); write(status_path,state); return
            with (folder/"supervised-worker.log").open("ab",buffering=0) as log:
                # Replay already recorded expansion before recomputing a gate:
                # partial imported checkpoints otherwise turn it to incomplete.
                # The wrapper owns worker.lock and calls unchanged frozen methods.
                child = subprocess.Popen([sys.executable,"-u",str(ROOT/"scripts/resume_recorded_research.py"),
                    "--workers",str(state["worker_limit"]),"--config",str(config_path)],cwd=ROOT,
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
