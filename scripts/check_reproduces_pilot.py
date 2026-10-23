"""Does the current trainer reproduce a pilot run bit for bit? (sprint 02, part 2, T5)

Retrains one pilot source with the current code into a temporary folder and compares the
final weights and Adam state with the pilot's 100% L3 snapshot.

Usage: .venv\\Scripts\\python.exe -m scripts.check_reproduces_pilot --run breakout_cnn_s0 --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from causal_trainability.ppo_minatar import PPO
from causal_trainability.telemetry import ROOT, code_revision

PILOT = ROOT / "runs" / "causal-trainability" / "ppo_pilot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="breakout_cnn_s0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = PILOT / args.run
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    final = max(folder.glob("snapshots/*.pt"), key=lambda p: int(p.stem.rsplit("_u", 1)[1]))
    saved = torch.load(final, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as scratch:
        ppo = PPO(cfg, scratch)
        ppo.train()
    weights = all(torch.equal(t, saved["model"][k]) for k, t in ppo.model.state_dict().items())
    state = ppo.optimizer.state_dict()["state"]
    adam = all(torch.equal(state[i][key], saved["optimizer"]["state"][i][key])
               for i in state for key in ("exp_avg", "exp_avg_sq", "step"))
    result = {"run": args.run, "snapshot": final.name, "weights_equal": weights, "adam_equal": adam,
              "revision": code_revision()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["weights_equal"], result["adam_equal"])


if __name__ == "__main__":
    main()
