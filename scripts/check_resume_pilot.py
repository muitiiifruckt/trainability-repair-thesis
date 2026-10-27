"""Does training resumed from a real pilot checkpoint end bit for bit where the pilot ended? (sprint 03, S2)

Restores each checkpoint of one pilot run (weights, Adam, environments, sampling generator, rollout),
continues to the full budget without any change of conditions and compares the final weights,
Adam state and step counter with the pilot's 100% L3 snapshot.

Usage: .venv\\Scripts\\python.exe -m scripts.check_resume_pilot --run breakout_cnn_s0 --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from causal_trainability import hybrids
from causal_trainability.ppo_minatar import PPO
from causal_trainability.telemetry import ROOT, code_revision

PILOT = ROOT / "runs" / "causal-trainability" / "ppo_pilot"


def check(folder, cfg, tag, saved):
    with tempfile.TemporaryDirectory() as scratch:
        ppo = PPO(dict(cfg), scratch)
        ppo.restore(hybrids.load(folder, tag))
        ppo.train(resume=True)
    weights = all(torch.equal(t, saved["model"][k]) for k, t in ppo.model.state_dict().items())
    state = ppo.optimizer.state_dict()["state"]
    adam = all(torch.equal(state[i][key], saved["optimizer"]["state"][i][key])
               for i in state for key in ("exp_avg", "exp_avg_sq", "step"))
    return {"checkpoint": hybrids.checkpoint_row(folder, tag), "weights_equal": weights, "adam_equal": adam,
            "global_step": ppo.global_step, "iterations": ppo.iterations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="breakout_cnn_s0")
    parser.add_argument("--tags", nargs="+", default=["age0.00", "age0.25"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = PILOT / args.run
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    final = max(folder.glob("snapshots/*.pt"), key=lambda p: int(p.stem.rsplit("_u", 1)[1]))
    saved = torch.load(final, map_location="cpu", weights_only=False)
    result = {"run": args.run, "snapshot": final.name, "shift": None,
              "resumed": [check(folder, cfg, tag, saved) for tag in args.tags], "revision": code_revision()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in result["resumed"]:
        print(r["checkpoint"]["tag"], r["weights_equal"], r["adam_equal"], r["global_step"])


if __name__ == "__main__":
    main()
