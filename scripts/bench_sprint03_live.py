"""Warm timing-only pilot on the actual S3/S4 start, before scientific branches."""
import argparse
import copy
import json
import time
from pathlib import Path
from statistics import mean, median

import torch

from causal_trainability import deficit, hybrids
from causal_trainability.sprint03_live import SETTINGS, atomic_json, intervene
from causal_trainability.telemetry import ROOT, code_revision


def run(out, arm, enabled, steps):
    source = ROOT / "runs/causal-trainability/ppo_pilot/breakout_cnn_s0"
    chain = ROOT / "runs/causal-trainability/deficit_B/breakout_cnn_s0/actions_chain"
    base = hybrids.load(source, "age1.00")["config"]
    fresh = hybrids.load(source, "age0.00")
    state, metadata = deficit.chain_end(chain)
    ppo, _, _ = deficit.prepared(base, state, state["optimizer"], "actions", 5, steps,
        ("s03-overhead", arm), "overhead", out, "overhead_pilot", f"overhead_{arm}",
        telemetry_every=SETTINGS["telemetry_every"],
        checkpoint_fractions=[0.0], snapshot_fractions=[0., .1, .25, .5, .75, 1.])
    intervene(ppo, fresh["theta0"], arm, metadata["rounds"][-1]["perm"], ppo.cfg["shift"]["perm"])
    started = time.perf_counter()
    ppo.train(telemetry_enabled=enabled)
    seconds = time.perf_counter() - started
    return seconds, copy.deepcopy(ppo.model.state_dict()), copy.deepcopy(ppo.optimizer.state_dict())


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--pairs", type=int, default=3)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit("Timing output already exists")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    results = {}
    for arm in ("continue", "heads_reset"):
        run(args.out / arm / "warmup", arm, True, 10_240)
        pairs = []
        for pair in range(args.pairs):
            order = (True, False) if pair % 2 == 0 else (False, True)
            states = {flag: run(args.out / arm / f"pair{pair}_{flag}", arm, flag, args.steps) for flag in order}
            bit_equal = same(states[True][1:], states[False][1:])
            if not bit_equal:
                raise ValueError("Telemetry changes weights/Adam")
            row = {"on": states[True][0], "off": states[False][0],
                   "overhead": states[True][0] / states[False][0] - 1, "bit_equal": bit_equal}
            pairs.append(row)
            print(arm, pair, json.dumps(row), flush=True)
        overheads = [r["overhead"] for r in pairs]
        results[arm] = {"pairs": pairs, "mean_overhead": mean(overheads), "median_overhead": median(overheads),
                        "passed": mean(overheads) <= .05 and median(overheads) <= .05}
    atomic_json(args.out / "timing.json", {"steps": args.steps, "telemetry_every": SETTINGS["telemetry_every"], "torch_threads": 1,
        "revision": code_revision(), "cells": results, "passed": all(r["passed"] for r in results.values())})


if __name__ == "__main__":
    main()
