"""Одноразовый handoff после сохранения текущей механистической серии.

Не прерывает незавершённый probe или живых child writers. Останавливает
только сверенные PID данной кампании; новый supervisor запускается скрыто.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_supervisor import alive
from resume_recorded_research import read, rows, write, make_plan, coverage


def safe_boundary(config, manifest, progress, checkpoints, active_leases):
    if active_leases or progress.get("stage") != "mechanisms" or progress.get("event") != "mechanisms_complete":
        return False
    games = set(config["screen"]["games"] + config["adaptive"]["development_extension_games"])
    expected = [f"mechanisms:{c['checkpoint_id']}:{r}" for c in checkpoints
                if c["source_mode"] == "natural" and c["game"] in games for r in config["mechanisms"]["seeds"]]
    jobs = manifest.get("jobs", {})
    return bool(expected) and all(jobs.get(k, {}).get("status") == "complete" for k in expected) and jobs.get(progress.get("job_id"), {}).get("status") == "complete"


def terminate(pid):
    from ctypes import wintypes
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    library.OpenProcess.restype = wintypes.HANDLE
    library.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    library.TerminateProcess.restype = wintypes.BOOL
    library.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = library.OpenProcess(0x1001, False, pid)
    if not handle:
        if not alive(pid):
            return
        raise OSError(ctypes.get_last_error(), "Cannot open owned research process")
    try:
        if not library.TerminateProcess(handle, 0):
            raise OSError(ctypes.get_last_error(), "Cannot stop owned research process")
    finally:
        library.CloseHandle(handle)
    for _ in range(100):
        if not alive(pid):
            return
        time.sleep(.1)
    raise RuntimeError("Research process did not exit")


def handoff(config_path, coordinator_pid, supervisor_pid, workers):
    config = read(config_path)
    folder = (ROOT / config["output_root"]).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Output outside workspace")
    status_file = folder / "recorded_handoff.json"
    started = time.time()
    while time.time() - started < 3600:
        manifest, progress = read(folder / "manifest.json"), read(folder / "progress.json")
        supervisor = read(folder / "supervisor.json")
        if manifest.get("worker_pid") != coordinator_pid or supervisor.get("supervisor_pid") != supervisor_pid:
            write(status_file, {"status": "owner_changed_no_action", "time": time.time()})
            return
        checkpoints = rows(folder / "checkpoints.jsonl")
        plan = make_plan(config, manifest, checkpoints, read(folder / "dependency_manifest.json")["code"])
        if plan is None or all(coverage(plan, op, rows(folder / "outcomes.jsonl"))["complete"] for op in plan["operations"]):
            write(status_file, {"status": "no_pending_recorded_expansion", "time": time.time()})
            return
        active_leases = []
        for path in folder.glob("workers/*/*/worker_lease.json"):
            try:
                lease = read(path)
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError):
                # An unreadable lease is uncertainty, never permission to stop.
                active_leases.append({"lease_temporarily_unreadable": str(path)})
                continue
            if alive(lease.get("pid")):
                active_leases.append(lease)
        ready = safe_boundary(config, manifest, progress, checkpoints, active_leases)
        if ready or (not alive(coordinator_pid) and not active_leases):
            snapshot = {"status": "handoff_at_saved_boundary", "time": time.time(),
                "coordinator_pid": coordinator_pid, "supervisor_pid": supervisor_pid,
                "progress": progress, "decision_sha256": plan["decision_sha256"]}
            write(status_file, snapshot)
            # Supervisor first, so it cannot launch the old route during handoff.
            terminate(supervisor_pid)
            terminate(coordinator_pid)
            with (ROOT / "runs/supervisor.stdout.log").open("ab", buffering=0) as stdout, (ROOT / "runs/supervisor.stderr.log").open("ab", buffering=0) as stderr:
                child = subprocess.Popen([str(ROOT / ".venv/Scripts/python.exe"), "-u", "scripts/research_supervisor.py",
                    "--workers", str(workers), "--config", str(config_path)], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            snapshot.update(status="replacement_supervisor_started", launched_pid=child.pid, completed_at=time.time())
            write(status_file, snapshot)
            return
        if progress.get("stage") not in ("mechanisms", "parallel_source", "resume_recorded_resolution"):
            # Never kill a learner on an unexpected route; heartbeat diagnoses it.
            write(status_file, {"status": "boundary_missed_no_termination", "time": time.time(), "progress": progress})
            return
        time.sleep(.25)
    write(status_file, {"status": "timed_out_no_termination", "time": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_program.json")
    parser.add_argument("--coordinator-pid", type=int, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    handoff(args.config.resolve(), args.coordinator_pid, args.supervisor_pid, args.workers)
