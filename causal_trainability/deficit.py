"""Live adaptation deficit after changed conditions (sprint 02, part 2, T5/T7; notes/ppo_deficit.md).

For one source history the aged agent (checkpoint age 1.00: weights and Adam) and the fresh
agent (theta0 of the same history, empty Adam) both learn the same shifted task:

* `channels` - the observation channels are permuted (they swap their meaning);
* `actions`  - the actions are permuted.

A permutation has no fixed points and is drawn per (game, history, kind, round), so a fresh
agent faces a task exactly as hard as the original one.  Environment and PPO seeds depend on
(history, kind, round, repeat) only, so the aged and fresh branches of a repeat are paired.

Run one branch:  .venv\\Scripts\\python.exe -m causal_trainability.deficit --source DIR --arm aged --shift actions --repeat 0
Run the queue:   .venv\\Scripts\\python.exe -m causal_trainability.deficit --queue
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from causal_trainability import hybrids
from causal_trainability.ppo_minatar import PPO, VecMinAtar, evaluate, seed_of
from causal_trainability.telemetry import ROOT, code_revision, config_hash

BRANCH = dict(steps=500_000, windows=10, eval_episodes=30, round=1)
SHIFTS = ("channels", "actions")
ARMS = ("aged", "fresh")


def derangement(n, key):
    """A permutation of range(n) without fixed points, determined by `key`."""
    rng = np.random.RandomState(seed_of("shift", *key))
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return [int(i) for i in perm]


def shift_spec(cfg, kind, round_):
    n = cfg["num_actions"] if kind == "actions" else cfg["input_channels"]
    return {"kind": kind, "round": round_, "perm": derangement(n, (cfg["game"], cfg["seed"], kind, round_))}


def auc(curve, steps, windows):
    """Mean over equal windows of env steps of the mean return of episodes finished in each window."""
    means = []
    for w in range(windows):
        lo, hi = w * steps / windows, (w + 1) * steps / windows
        returns = [r for row in curve if lo < row["env_steps"] <= hi for r in row["returns"]]
        means.append(float(np.mean(returns)) if returns else None)
    present = [m for m in means if m is not None]
    return (float(np.mean(present)) if present else None), means


def branch(source, arm, kind, repeat, out, settings=BRANCH):
    fresh, aged = hybrids.load(source, "age0.00"), hybrids.load(source, "age1.00")
    base = aged["config"]
    shift = shift_spec(base, kind, settings["round"])
    history = f"{base['game']}_{base['arch']}_s{base['seed']}"
    key = (history, kind, settings["round"], repeat)
    cfg = dict(base, total_steps=settings["steps"], shift=shift, checkpoint_fractions=[0.0],
               snapshot_fractions=[0.0, 0.5, 1.0], run_kind="deficit_branch",
               run_name=f"deficit_{history}_{kind}_r{settings['round']}_{arm}_rep{repeat}")
    with torch.random.fork_rng(devices=[]):  # construction must not depend on or move the global RNG
        ppo = PPO(cfg, out)
    start = aged if arm == "aged" else fresh
    ppo.model.load_state_dict(start["model"])
    ppo.theta0 = start["theta0"]
    if arm == "aged":
        ppo.optimizer.load_state_dict(aged["optimizer"])  # fresh keeps the empty Adam made by PPO()
    seeds = {"env": seed_of("deficit-env", *key), "ppo": seed_of("deficit-ppo", *key),
             "eval_policy": seed_of("deficit-eval-policy", *key)}
    ppo.envs = VecMinAtar(cfg, seeds["env"])
    ppo.obs = torch.tensor(ppo.envs.observations())
    ppo.generator = torch.Generator().manual_seed(seeds["ppo"])
    eval_seeds = [seed_of("deficit-eval", history, kind, i) for i in range(settings["eval_episodes"])]
    policy = lambda: torch.Generator().manual_seed(seeds["eval_policy"])
    start_return = evaluate(ppo.model, cfg, eval_seeds, policy())
    seconds = ppo.train()
    area, windows = auc(ppo.curve, settings["steps"], settings["windows"])
    result = {"history": history, "arm": arm, "shift": shift, "repeat": repeat, "settings": settings,
              "seeds": dict(seeds, eval_episodes=eval_seeds),
              "minimal_action_set": [int(a) for a in ppo.envs.envs[0].minimal_action_set()],
              "config": cfg, "config_hash": config_hash(cfg), "revision": code_revision(),
              "sources": {tag: hybrids.checkpoint_row(source, tag) for tag in ("age0.00", "age1.00")},
              "start_return": start_return, "auc": area, "auc_windows": windows,
              "final_return": evaluate(ppo.model, cfg, eval_seeds, policy()),
              "random_return": evaluate(ppo.model, cfg, eval_seeds),
              "episodes": sum(len(row["returns"]) for row in ppo.curve), "wall_time_sec": seconds,
              "curve": ppo.curve}
    (Path(out) / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def queue(sources, out, workers, settings=BRANCH, repeats=(0, 1)):
    jobs = []
    for source, kind, arm, repeat in itertools.product(sources, SHIFTS, ARMS, repeats):
        folder = Path(out) / Path(source).name / f"{kind}_r{settings['round']}" / f"{arm}_rep{repeat}"
        if (folder / "result.json").exists():
            continue
        if folder.exists():  # an interrupted branch restarts from scratch
            shutil.rmtree(folder)
        jobs.append((source, kind, arm, repeat, folder))

    def run(job):
        source, kind, arm, repeat, folder = job
        started = time.time()
        subprocess.run([sys.executable, "-m", "causal_trainability.deficit", "--source", str(source), "--arm", arm,
                        "--shift", kind, "--repeat", str(repeat), "--out", str(folder)], cwd=ROOT, check=True)
        r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        return f"{folder.relative_to(out)}: auc {r['auc']:.2f}, start {r['start_return']:.2f}, final {r['final_return']:.2f}, {time.time() - started:.0f}s"

    print(f"{len(jobs)} branches to run", flush=True)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for line in pool.map(run, jobs):
            print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--pilot", default=str(ROOT / "runs" / "causal-trainability" / "ppo_pilot"))
    parser.add_argument("--histories", nargs="+", default=[f"breakout_cnn_s{s}" for s in range(6)])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--source")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--shift", choices=SHIFTS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--out", default=str(ROOT / "runs" / "causal-trainability" / "deficit_A"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.queue:
        revision = code_revision()
        if revision["dirty_causal_trainability"]:
            raise SystemExit("causal_trainability has uncommitted changes; commit before the queue")
        missing = [h for h in args.histories if not (Path(args.pilot) / h / "result.json").exists()]
        if missing:
            raise SystemExit(f"Unfinished sources: {missing}")
        queue([Path(args.pilot) / h for h in args.histories], Path(args.out), args.workers)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        branch(Path(args.source), args.arm, args.shift, args.repeat, Path(args.out))


if __name__ == "__main__":
    # Run through the importable module so pickled classes are not recorded as __main__.*
    import causal_trainability.deficit as module
    module.main()
