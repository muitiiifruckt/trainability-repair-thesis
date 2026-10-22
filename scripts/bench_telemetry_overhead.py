"""Telemetry overhead of the PPO stand (sprint 02, T2; docs/observability_spec.md rule 3: <= 5%).

Interleaved pairs of identical runs with telemetry on and off, one process at a time, on a
free machine. Reports the wall time of each run and the median overhead.

Usage: .venv\\Scripts\\python.exe -m scripts.bench_telemetry_overhead --steps 200000 --pairs 3 --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path
from statistics import median

import torch

from causal_trainability.ppo_minatar import CONFIG, PPO, resolve
from causal_trainability.telemetry import code_revision


def timed(cfg, telemetry):
    with tempfile.TemporaryDirectory() as folder:
        return PPO(dict(cfg), folder).train(telemetry_enabled=telemetry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists")
    torch.set_num_threads(1)
    cfg = resolve(dict(CONFIG, total_steps=args.steps))
    pairs = []
    for i in range(args.pairs):
        order = (True, False) if i % 2 == 0 else (False, True)  # alternate to cancel drift
        times = {flag: timed(cfg, flag) for flag in order}
        pairs.append({"on": times[True], "off": times[False], "overhead": times[True] / times[False] - 1})
        print(pairs[-1], flush=True)
    result = {"steps": args.steps, "config": cfg, "pairs": pairs,
              "median_overhead": median(p["overhead"] for p in pairs),
              "torch_threads": torch.get_num_threads(), "platform": platform.platform(),
              "processor": platform.processor(), "revision": code_revision()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("median overhead", round(result["median_overhead"] * 100, 2), "%")


if __name__ == "__main__":
    main()
