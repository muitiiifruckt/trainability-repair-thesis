"""Live adaptation deficit after changed conditions (sprint 02, part 2, T5/T7; notes/ppo_deficit.md).

For one source history the aged agent (checkpoint age 1.00: weights and Adam) and the fresh
agent (theta0 of the same history, empty Adam) both learn the same shifted task:

* `channels` - the observation channels are permuted (they swap their meaning);
* `actions`  - the actions are permuted.

The mapping of round r is the mapping of round r - 1 composed with a permutation without
fixed points, drawn per (game, history, kind, r): every channel or output changes its meaning
in every round, and a fresh agent faces a task exactly as hard as the original one.
Environment and PPO seeds depend on (history, kind, round, repeat) only, so the aged and fresh
branches of a repeat are paired.

Stage A: both arms on round 1.  Stage B: the aged agent first passes rounds 1..4 (a chain,
weights and Adam carried over), then both arms learn round 5.

Run the queue:  .venv\\Scripts\\python.exe -m causal_trainability.deficit --queue [--stage B]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from minatar import Environment

from causal_trainability import hybrids
from causal_trainability.ppo_minatar import PPO, VecMinAtar, evaluate, seed_of
from causal_trainability.telemetry import ROOT, code_revision, config_hash

BRANCH = dict(steps=500_000, windows=10, eval_episodes=30, round=1)
STAGES = {"A": dict(BRANCH), "B": dict(BRANCH, round=5)}
CHAIN = dict(rounds=4, steps=250_000, windows=5)
SHIFTS = ("channels", "actions")
ARMS = ("aged", "fresh")


def derangement(n, key):
    """A permutation of range(n) without fixed points, determined by `key`."""
    rng = np.random.RandomState(seed_of("shift", *key))
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return [int(i) for i in perm]


def function_of(mapping, kind, game):
    """What a mapping does: the channel order, or which outputs move (other actions act as no-ops)."""
    if kind == "channels":
        return tuple(mapping)
    moves = [a for a in Environment(game).minimal_action_set() if a != 0]
    return tuple(mapping.index(a) for a in moves)


def shift_spec(cfg, kind, round_):
    """Mapping of round `round_`: m_r = m_{r-1}[d_r] with d_r without fixed points, so nothing keeps its
    meaning from round r - 1; d_r is redrawn until m_r also acts unlike every earlier m_0..m_{r-1}
    (m_0 is the original task).  Round 1 takes the first draw, as in stage A."""
    n = cfg["num_actions"] if kind == "actions" else cfg["input_channels"]
    history = [list(range(n))]
    for r in range(1, round_ + 1):
        rng = np.random.RandomState(seed_of("shift", cfg["game"], cfg["seed"], kind, r))
        seen = {function_of(m, kind, cfg["game"]) for m in history}
        previous = function_of(history[-1], kind, cfg["game"])
        while True:
            d = rng.permutation(n)
            if (d == np.arange(n)).any():
                continue
            mapping = [history[-1][int(i)] for i in d]
            acts = function_of(mapping, kind, cfg["game"])
            if acts not in seen and all(a != b for a, b in zip(acts, previous)):
                break
        history.append(mapping)
    return {"kind": kind, "round": round_, "perm": history[-1]}


def auc(curve, steps, windows):
    """Mean over equal windows of env steps of the mean return of episodes finished in each window."""
    means = []
    for w in range(windows):
        lo, hi = w * steps / windows, (w + 1) * steps / windows
        returns = [r for row in curve if lo < row["env_steps"] <= hi for r in row["returns"]]
        means.append(float(np.mean(returns)) if returns else None)
    present = [m for m in means if m is not None]
    return (float(np.mean(present)) if present else None), means


def history_of(cfg):
    return f"{cfg['game']}_{cfg['arch']}_s{cfg['seed']}"


def prepared(base, state, adam, kind, round_, steps, key, seed_prefix, out, run_kind, run_name, **extra):
    """A PPO learner on the shifted task of `round_`, started from `state` (and `adam`, if given)."""
    cfg = dict(base, total_steps=steps, shift=shift_spec(base, kind, round_), run_kind=run_kind, run_name=run_name,
               **extra)
    with torch.random.fork_rng(devices=[]):  # construction must not depend on or move the global RNG
        ppo = PPO(cfg, out)
    ppo.model.load_state_dict(state["model"])
    ppo.theta0 = state["theta0"]
    if adam is not None:
        ppo.optimizer.load_state_dict(adam)  # otherwise the empty Adam made by PPO() stays
    seeds = {"env": seed_of(f"{seed_prefix}-env", *key), "ppo": seed_of(f"{seed_prefix}-ppo", *key)}
    ppo.envs = VecMinAtar(cfg, seeds["env"])
    ppo.obs = torch.tensor(ppo.envs.observations())
    ppo.generator = torch.Generator().manual_seed(seeds["ppo"])
    return ppo, cfg, seeds


def chain(source, kind, out, settings=CHAIN):
    """Aged agent through rounds 1..rounds; the end state is saved with its sha256 in chain.json."""
    aged = hybrids.load(source, "age1.00")
    base, out = aged["config"], Path(out)
    history = history_of(base)
    state = {"model": aged["model"], "optimizer": aged["optimizer"], "theta0": aged["theta0"]}
    rounds = []
    for r in range(1, settings["rounds"] + 1):
        ppo, cfg, seeds = prepared(base, state, state["optimizer"], kind, r, settings["steps"], (history, kind, r),
                                   "deficit-chain", out / f"round{r}", "deficit_chain", f"chain_{history}_{kind}_round{r}",
                                   checkpoint_fractions=[], snapshot_fractions=[0.0, 1.0])
        seconds = ppo.train()
        area, windows = auc(ppo.curve, settings["steps"], settings["windows"])
        rounds.append({"round": r, "perm": cfg["shift"]["perm"], "seeds": seeds, "auc": area, "auc_windows": windows,
                       "wall_time_sec": seconds, "config_hash": config_hash(cfg)})
        state = {"model": copy.deepcopy(ppo.model.state_dict()), "optimizer": copy.deepcopy(ppo.optimizer.state_dict()),
                 "theta0": state["theta0"]}
    path = out / "chain_end.pt"
    torch.save(state, path)
    summary = {"history": history, "kind": kind, "settings": settings, "rounds": rounds,
               "source": hybrids.checkpoint_row(source, "age1.00"), "revision": code_revision(),
               "end": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (out / "chain.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def chain_end(folder):
    folder = Path(folder)
    summary = json.loads((folder / "chain.json").read_text(encoding="utf-8"))
    path = folder / summary["end"]["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["end"]["sha256"]:
        raise ValueError(f"Chain end checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False), summary


def branch(source, arm, kind, repeat, out, settings=BRANCH, chain_dir=None):
    fresh, aged = hybrids.load(source, "age0.00"), hybrids.load(source, "age1.00")
    base = aged["config"]
    history = history_of(base)
    key = (history, kind, settings["round"], repeat)
    chained = None
    if arm == "aged" and settings["round"] > 1:
        if chain_dir is None:
            raise ValueError("An aged branch after round 1 must start from the end of its chain")
        aged, chained = chain_end(chain_dir)
    start = aged if arm == "aged" else fresh
    ppo, cfg, seeds = prepared(base, start, aged["optimizer"] if arm == "aged" else None, kind, settings["round"],
                               settings["steps"], key, "deficit", out, "deficit_branch",
                               f"deficit_{history}_{kind}_r{settings['round']}_{arm}_rep{repeat}",
                               checkpoint_fractions=[0.0], snapshot_fractions=[0.0, 0.5, 1.0])
    seeds["eval_policy"] = seed_of("deficit-eval-policy", *key)
    eval_seeds = [seed_of("deficit-eval", history, kind, i) for i in range(settings["eval_episodes"])]
    policy = lambda: torch.Generator().manual_seed(seeds["eval_policy"])
    start_return = evaluate(ppo.model, cfg, eval_seeds, policy())
    seconds = ppo.train()
    area, windows = auc(ppo.curve, settings["steps"], settings["windows"])
    result = {"history": history, "arm": arm, "shift": cfg["shift"], "repeat": repeat, "settings": settings,
              "seeds": dict(seeds, eval_episodes=eval_seeds),
              "minimal_action_set": [int(a) for a in ppo.envs.envs[0].minimal_action_set()],
              "config": cfg, "config_hash": config_hash(cfg), "revision": code_revision(),
              "sources": {tag: hybrids.checkpoint_row(source, tag) for tag in ("age0.00", "age1.00")},
              "chain": None if chained is None else {"end": chained["end"], "rounds": chained["rounds"]},
              "start_return": start_return, "auc": area, "auc_windows": windows,
              "final_return": evaluate(ppo.model, cfg, eval_seeds, policy()),
              "random_return": evaluate(ppo.model, cfg, eval_seeds),
              "episodes": sum(len(row["returns"]) for row in ppo.curve), "wall_time_sec": seconds,
              "curve": ppo.curve}
    (Path(out) / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def fresh_folder(folder, done):
    """Skip finished work; an interrupted job restarts from scratch."""
    if (folder / done).exists():
        return False
    if folder.exists():
        shutil.rmtree(folder)
    return True


def queue(sources, out, workers, stage="A", repeats=(0, 1)):
    settings = STAGES[stage]
    command = [sys.executable, "-m", "causal_trainability.deficit", "--stage", stage]

    def chain_job(job):
        source, kind, folder = job
        subprocess.run(command + ["--source", str(source), "--shift", kind, "--chain-out", str(folder)], cwd=ROOT, check=True)
        s = json.loads((folder / "chain.json").read_text(encoding="utf-8"))
        return f"{folder.relative_to(out)}: round AUC {[round(r['auc'], 2) for r in s['rounds']]}"

    def branch_job(job):
        source, kind, arm, repeat, folder = job
        started = time.time()
        extra = ["--chain-dir", str(Path(out) / Path(source).name / f"{kind}_chain")] if stage == "B" else []
        subprocess.run(command + ["--source", str(source), "--arm", arm, "--shift", kind, "--repeat", str(repeat),
                                  "--out", str(folder)] + extra, cwd=ROOT, check=True)
        r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        return f"{folder.relative_to(out)}: auc {r['auc']:.2f}, start {r['start_return']:.2f}, final {r['final_return']:.2f}, {time.time() - started:.0f}s"

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        if stage == "B":
            chains = [(s, k, Path(out) / Path(s).name / f"{k}_chain") for s, k in itertools.product(sources, SHIFTS)]
            chains = [c for c in chains if fresh_folder(c[2], "chain.json")]
            print(f"{len(chains)} chains to run", flush=True)
            for line in pool.map(chain_job, chains):
                print(line, flush=True)
        jobs = [(s, k, a, rep, Path(out) / Path(s).name / f"{k}_r{settings['round']}" / f"{a}_rep{rep}")
                for s, k, a, rep in itertools.product(sources, SHIFTS, ARMS, repeats)]
        jobs = [j for j in jobs if fresh_folder(j[4], "result.json")]
        print(f"{len(jobs)} branches to run", flush=True)
        for line in pool.map(branch_job, jobs):
            print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--stage", choices=sorted(STAGES), default="A")
    parser.add_argument("--pilot", default=str(ROOT / "runs" / "causal-trainability" / "ppo_pilot"))
    parser.add_argument("--histories", nargs="+", default=[f"breakout_cnn_s{s}" for s in range(6)])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--source")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--shift", choices=SHIFTS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--chain-out")
    parser.add_argument("--chain-dir")
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.queue:
        revision = code_revision()
        if revision["dirty_causal_trainability"]:
            raise SystemExit("causal_trainability has uncommitted changes; commit before the queue")
        missing = [h for h in args.histories if not (Path(args.pilot) / h / "result.json").exists()]
        if missing:
            raise SystemExit(f"Unfinished sources: {missing}")
        out = Path(args.out or ROOT / "runs" / "causal-trainability" / f"deficit_{args.stage}")
        queue([Path(args.pilot) / h for h in args.histories], out, args.workers, args.stage)
    elif args.chain_out:
        Path(args.chain_out).mkdir(parents=True, exist_ok=True)
        chain(Path(args.source), args.shift, Path(args.chain_out))
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        branch(Path(args.source), args.arm, args.shift, args.repeat, Path(args.out), STAGES[args.stage], args.chain_dir)


if __name__ == "__main__":
    # Run through the importable module so pickled classes are not recorded as __main__.*
    import causal_trainability.deficit as module
    module.main()
