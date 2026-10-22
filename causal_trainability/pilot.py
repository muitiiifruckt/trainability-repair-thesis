"""T2 pilot queue: 2 games x 2 architectures x 2 seeds, one run at a time.

Breakout and the `cnn` architecture are for development; Space Invaders and `cnn_ln`
are held out, so on them only base learning is checked (no repair outcomes).
Finished runs (result.json present) are skipped, so the queue can be restarted.

Run:  .venv\\Scripts\\python.exe -m causal_trainability.pilot [--steps 1000000]
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from causal_trainability.telemetry import ROOT

GAMES = ("breakout", "space_invaders")
ARCHS = ("cnn", "cnn_ln")
SEEDS = (0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--out", default=str(ROOT / "runs" / "causal-trainability" / "ppo_pilot"))
    args = parser.parse_args()
    out = Path(args.out)
    for game, arch, seed in itertools.product(GAMES, ARCHS, SEEDS):
        folder = out / f"{game}_{arch}_s{seed}"
        if (folder / "result.json").exists():
            print(f"skip {folder.name}", flush=True)
            continue
        if folder.exists():  # an interrupted run restarts from scratch, without duplicated rows
            shutil.rmtree(folder)
        started = time.time()
        subprocess.run([sys.executable, "-m", "causal_trainability.ppo_minatar", "--game", game, "--arch", arch,
                        "--seed", str(seed), "--steps", str(args.steps), "--out", str(folder)], cwd=ROOT, check=True)
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        print(f"{folder.name}: eval {result['eval_return']:.2f} vs random {result['random_return']:.2f}, "
              f"{result['steps_per_sec']:.0f} steps/s, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
