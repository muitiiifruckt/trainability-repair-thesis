"""Timing-only S5 observability pilot on fresh synthetic agents, never aged sources.

Use before freezing the package. No reward values or policy outcomes are exported.
Full wall time includes background writer close and snapshot serialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from causal_trainability import two_state as ts
from causal_trainability import two_state_v2 as v2
from causal_trainability.telemetry import ROOT, Telemetry, TelemetryStore, code_revision


def digest(agent, optimizer):
    digest_ = hashlib.sha256()
    for name, p in agent.named_parameters():
        digest_.update(name.encode())
        digest_.update(p.detach().numpy().tobytes())
        for key, value in sorted(optimizer.state.get(p, {}).items()):
            digest_.update(key.encode())
            digest_.update(value.numpy().tobytes() if torch.is_tensor(value) else repr(value).encode())
    return digest_.hexdigest()


def timed(folder, fixture, cfg, enabled, cells):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    store = TelemetryStore(folder, "pilot") if enabled else None
    digests = []
    try:
        for cell in range(cells):
            agent, optimizer = ts.build(fixture, None, cfg)
            telemetry = Telemetry(folder, f"cell{cell}", agent.layers(), optimizer,
                meta={"kind": "two_state_v2_overhead_pilot", "cell": cell},
                every=cfg["telemetry_every"], store=store) if enabled else None
            v2.trajectory(agent, optimizer, ts.NEW_GOAL, cfg["iterations"],
                ts.seed_of("s5-overhead", cell), cfg, telemetry=telemetry, snapshot_every=10)
            digests.append(digest(agent, optimizer))
    finally:
        if store is not None:
            store.close()
    return time.perf_counter() - started, digests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cadence", type=int, default=5)
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.cadence < 1 or args.cadence > 30 or args.cells < 1 or args.pairs < 3:
        raise ValueError("Need cadence in 1..30, positive cells, at least three pairs")
    if args.out.exists() or args.report.exists():
        raise ValueError("Fresh pilot root and report required; do not overwrite pilots")
    torch.set_num_threads(1)
    cfg = dict(v2.CONFIG, telemetry_every=args.cadence)
    revision = code_revision()  # warm metadata lookup outside measured cells
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ts.seed_of("s5-overhead", "fresh-fixture"))
        fixture = ts.Agent(cfg["hidden"])
    args.out.mkdir(parents=True)
    rows = []
    # Each mode gets its own unreported warmup; no scientific rewards are read.
    for enabled in (False, True):
        timed(args.out / f"warmup_{enabled}", fixture, cfg, enabled, 3)
    for pair in range(args.pairs):
        measurements = {}
        signatures = {}
        for enabled in ((True, False) if pair % 2 == 0 else (False, True)):
            measurements[enabled], signatures[enabled] = timed(
                args.out / f"pair{pair}_{enabled}", fixture, cfg, enabled, args.cells)
        equal = signatures[True] == signatures[False]
        rows.append(dict(on=measurements[True], off=measurements[False],
                         overhead=measurements[True] / measurements[False] - 1,
                         bit_equal=equal, weights_adam_sha256=signatures[True]))
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "weights_adam_sha256"}), flush=True)
    values = [r["overhead"] for r in rows]
    mean, median = statistics.mean(values), statistics.median(values)
    report = dict(schema_version=1, kind="timing_only_fresh_synthetic", config=cfg,
        cells_per_pair=args.cells, pairs=rows, mean_overhead=mean, median_overhead=median,
        passes=all(r["bit_equal"] for r in rows) and mean <= .05 and median <= .05,
        torch_threads=torch.get_num_threads(), platform=platform.platform(), revision=revision,
        script_sha256=v2.sha(__file__), output_root=str(args.out), includes_writer_close_and_snapshots=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    v2.atomic_json(args.report, report)
    v2.atomic_json(args.out / "report.json", report)
    print("PASS" if report["passes"] else "BLOCKED", flush=True)


if __name__ == "__main__":
    main()
