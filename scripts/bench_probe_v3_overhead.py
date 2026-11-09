"""Fresh synthetic S6 calibration, no aged scientific outcomes.

Run alone: python -m scripts.bench_probe_v3_overhead --isolated --mode trained --output OUT.json
Repeat with --mode frozen and another output. Both reports must pass.
The caller must ensure all other experiment workers are idle.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path
from statistics import mean, median

import torch

from causal_trainability import probe, probe_v3 as v3
from causal_trainability.ppo_minatar import ActorCritic, CONFIG, seed_of
from causal_trainability.telemetry import Telemetry, code_revision


def memory_peak():
    if platform.system() != "Windows":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    if not get_memory(current_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize


def state_digest(model, optimizer):
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(model.state_dict())
    visit(optimizer.state_dict())
    return digest.hexdigest()


def timed(state, cfg, states, targets, reference, spec, enabled, mode):
    rng_before = torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory(prefix="probe_v3_overhead_") as folder:
        started = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            model = probe.ProbeNet(cfg, state, spec["outputs"])
        groups = model.layers()
        if mode == "frozen":
            for parameter in groups["conv"] + groups["fc"]:
                parameter.requires_grad_(False)
        parameters = groups["head"] if mode == "frozen" else [p for group in groups.values() for p in group]
        optimizer = torch.optim.Adam(parameters, lr=spec["lr"], eps=spec["adam_eps"])
        log = Telemetry(folder, "overhead_fresh", model.layers(), optimizer, meta={"kind": "probe_v3_overhead"},
                        enabled=enabled, every=spec["telemetry_every"])
        try:
            v3.fit(model, optimizer, states, targets, spec, seed_of("probe-v3-overhead", "order"), log, reference)
        finally:
            log.close()
        seconds = time.perf_counter() - started  # includes telemetry flush, snapshot writes and artifact hashing
        digest = state_digest(model, optimizer)
    return {"seconds": seconds, "weights_adam_sha256": digest,
            "rng_unchanged": torch.equal(rng_before, torch.get_rng_state()), "peak_working_set_bytes": memory_peak()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated", action="store_true", help="Caller asserts all other experiment workers are idle")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--states", type=int, default=1800)
    parser.add_argument("--mode", choices=("trained", "frozen"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists")
    if not args.isolated:
        raise SystemExit("Run only after the coordinator verifies all other workers are idle; pass --isolated")
    if args.pairs < 3 or args.steps < 2000 or args.steps % 400:
        raise SystemExit("Need >=3 pairs, >=2000 steps divisible by 400")
    torch.set_num_threads(1)
    cfg = dict(CONFIG, input_channels=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_of("probe-v3-overhead", "fresh"))
        state = ActorCritic(cfg).state_dict()
    generator = torch.Generator().manual_seed(seed_of("probe-v3-overhead", "synthetic"))
    states = (torch.rand(args.states, 4, 10, 10, generator=generator) < .03).to(torch.uint8)
    targets = torch.randn(args.states, v3.SPEC["outputs"], generator=generator)
    targets = (targets - targets.mean(0)) / targets.std(0, unbiased=False)
    # Five L3 points in 2000 updates are denser than the five points in 8000 scientific updates.
    spec = dict(v3.SPEC, steps=args.steps, late_start=args.steps // 2,
                snapshot_steps=[0, args.steps // 8, args.steps // 2, 3 * args.steps // 4, args.steps])
    warmup_spec = dict(spec, steps=400, late_start=200, snapshot_steps=[0, 200, 400])
    warmup = {str(flag): timed(state, cfg, states, targets, states[:256], warmup_spec, flag, args.mode)
              for flag in (False, True)}
    pairs = []
    for i in range(args.pairs):
        order = (True, False) if i % 2 == 0 else (False, True)
        values = {flag: timed(state, cfg, states, targets, states[:256], spec, flag, args.mode) for flag in order}
        pairs.append({"order": list(order), "on": values[True], "off": values[False],
                      "overhead": values[True]["seconds"] / values[False]["seconds"] - 1,
                      "bit_equal": values[True]["weights_adam_sha256"] == values[False]["weights_adam_sha256"]})
        print(json.dumps(pairs[-1]), flush=True)
    avg, med = mean(row["overhead"] for row in pairs), median(row["overhead"] for row in pairs)
    all_runs = list(warmup.values()) + [row[flag] for row in pairs for flag in ("on", "off")]
    peaks = [row["peak_working_set_bytes"] for row in all_runs]
    memory_ok = all(peak is not None and peak < 1_000_000_000 for peak in peaks)
    passed = (avg <= .05 and med <= .05 and all(row["bit_equal"] for row in pairs)
              and all(row["rng_unchanged"] for row in all_runs) and memory_ok)
    result = {"schema_version": 1, "kind": "fresh_synthetic_calibration_not_scientific_aged_outcomes",
              "isolation_asserted_by_caller": args.isolated, "mode": args.mode, "spec": spec, "config": cfg,
              "states_sha256": probe.digest(states), "targets_sha256": probe.digest(targets),
              "warmup": warmup, "pairs": pairs, "mean_overhead": avg, "median_overhead": med,
              "memory_below_1GB": memory_ok, "passed": passed, "revision": code_revision(),
              "protocol_sha256": v3.sha256(v3.ROOT / "notes/ppo_probe_v3.md"),
              "benchmark_sha256": v3.sha256(__file__), "threads": torch.get_num_threads(),
              "platform": platform.platform(),
              "limitation": "Synthetic sparse states; cadence and five snapshots conservatively match full geometry, but no aged paths tested"}
    v3.atomic_json(args.output, result)
    print("passed:", passed, "mean:", avg, "median:", med, flush=True)


if __name__ == "__main__":
    main()
