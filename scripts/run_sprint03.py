"""Detached, sequential S6 pilots then reviewed scientific queues; fails closed."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "runs/causal-trainability/sprint03_controller_20261109"


def state(**row):
    CONTROL.mkdir(parents=True, exist_ok=True)
    row.update(pid=os.getpid(), updated=time.time())
    temporary = CONTROL / "state.tmp"
    temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
    temporary.replace(CONTROL / "state.json")


def run(label, arguments):
    if (CONTROL / "STOP").exists():
        raise RuntimeError("STOP requested")
    expected = json.loads((CONTROL / "code_freeze.json").read_text(encoding="utf-8"))
    if expected != fingerprint():
        raise RuntimeError("Code/protocol changed since controller freeze")
    state(status="running", stage=label, command=arguments)
    with (CONTROL / f"{label}.log").open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m"] + arguments, cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def fingerprint():
    paths = list((ROOT / "causal_trainability").glob("*.py"))
    paths += list((ROOT / "scripts").glob("*sprint03*.py"))
    paths += [ROOT / "scripts/bench_probe_v3_overhead.py"]
    paths += [ROOT / "notes" / name for name in ("sprint03_live.md", "two_state_v2.md", "ppo_probe_v3.md")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait-pid", type=int)
    args = p.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    freeze = CONTROL / "code_freeze.json"
    if freeze.exists():
        if json.loads(freeze.read_text(encoding="utf-8")) != fingerprint():
            raise RuntimeError("Existing controller has a different code freeze")
    else:
        freeze.write_text(json.dumps(fingerprint(), indent=2), encoding="utf-8")
    # An OS handle refers to this process instance even if the numeric PID is reused.
    if args.wait_pid and os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, args.wait_pid)
        if handle:
            state(status="waiting", stage="previous_timing_pilot", wait_pid=args.wait_pid)
            try:
                while kernel.WaitForSingleObject(handle, 1000) == 258:
                    if (CONTROL / "STOP").exists():
                        return
            finally:
                kernel.CloseHandle(handle)
    # No concurrent training within this controller: timing pilots require idle CPU.
    root = "runs/causal-trainability"
    specs = [
        ("s5", ["scripts.bench_sprint03_two_state", "--cadence", "30", "--out", f"{root}/s03_s5_overhead_clean",
                "--report", "research/results/two-state-v2/overhead_clean.json"],
         "research/results/two-state-v2/overhead_clean.json", "passes",
         [["causal_trainability.two_state_v2"]]),
        ("probe", ["scripts.bench_probe_v3_overhead", "--isolated", "--mode", "trained", "--output", "research/results/ppo-probe-v3/overhead_clean.json"],
         "research/results/ppo-probe-v3/overhead_clean.json", "passed",
         [["causal_trainability.probe_v3", "--queue", "--workers", "1"], ["causal_trainability.probe_v3", "--summarize"]]),
        ("probe_frozen", ["scripts.bench_probe_v3_overhead", "--isolated", "--mode", "frozen", "--output", "research/results/ppo-probe-v3/overhead_frozen_clean.json"],
         "research/results/ppo-probe-v3/overhead_frozen_clean.json", "passed", []),
        ("live", ["scripts.bench_sprint03_live", "--out", f"{root}/s03_live_overhead_clean", "--pairs", "3"],
         f"{root}/s03_live_overhead_clean/timing.json", "passed",
         [["causal_trainability.sprint03_live", "--queue", "--workers", "2", "--out", f"{root}/sprint03_live_20261109"],
          ["causal_trainability.sprint03_live", "--summarize", "--out", f"{root}/sprint03_live_20261109"]]),
    ]
    # All pilots finish before the first scientific outcome; each stage has its own gate.
    outcomes = {}
    for name, benchmark, report, key, commands in specs:
        try:
            if not (ROOT / report).exists():
                run(f"timing_{name}", benchmark)
            result = json.loads((ROOT / report).read_text(encoding="utf-8"))
            frozen = json.loads((CONTROL / "code_freeze.json").read_text(encoding="utf-8"))
            expected_code = {k.replace("\\", "/"): v for k, v in frozen.items() if k.replace("\\", "/").startswith("causal_trainability/")}
            if result["revision"]["files"] != expected_code:
                raise RuntimeError("Timing report does not match frozen scientific code")
            outcomes[name] = {"passed": result[key], "report": report,
                              "sha256": hashlib.sha256((ROOT / report).read_bytes()).hexdigest()}
        except Exception as exc:
            outcomes[name] = {"passed": False, "error": repr(exc)}
        (CONTROL / "gates.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    for name, _, _, _, commands in specs:
        if not outcomes[name]["passed"]:
            continue
        if name == "probe" and not outcomes["probe_frozen"]["passed"]:
            continue
        for index, command in enumerate(commands):
            run(f"science_{name}_{index}", command)
    state(status="finished", stage="all_authorized_stages", gates=outcomes,
          remaining="substantive shift requires supervisor agreement; failed timing gates remain blocked")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        state(status="failed", error=repr(exc))
        raise
