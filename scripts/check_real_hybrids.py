"""T3 checks on real pilot checkpoints: the all-aged hybrid reproduces the checkpoint.

Weights and every Adam state bit for bit, Adam state bound to the hybrid's parameters,
GAE advantages/returns rebuilt from the stored rollout. Also counts truncated episodes in
each rollout: with none, the critic re-bootstrap of timeouts is not exercised.

Usage: .venv\\Scripts\\python.exe -m scripts.check_real_hybrids --output OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from causal_trainability import hybrids
from causal_trainability.telemetry import ROOT, code_revision

PILOT = ROOT / "runs" / "causal-trainability" / "ppo_pilot"


def check(folder, tag):
    fresh, aged = hybrids.load(folder, "age0.00"), hybrids.load(folder, tag)
    model, optimizer, batch = hybrids.assemble(fresh, aged, dict.fromkeys(hybrids.PARTS, "aged"))
    saved = aged["optimizer"]["state"]
    return {"run": folder.name, "tag": tag,
            "weights_equal": all(torch.equal(t, aged["model"][k]) for k, t in model.state_dict().items()),
            "adam_equal": all(torch.equal(optimizer.state[p][key], saved[i][key])
                              for i, p in enumerate(model.parameters()) for key in ("exp_avg", "exp_avg_sq", "step")),
            "adam_bound_to_hybrid": all(p in optimizer.state for p in model.parameters()),
            "max_abs_advantage_diff": float((batch["advantages"] - aged["rollout"]["advantages"]).abs().max()),
            "max_abs_return_diff": float((batch["returns"] - aged["rollout"]["returns"]).abs().max()),
            "truncated_episodes_in_rollout": len(aged["rollout"]["timeouts"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [check(folder, tag) for folder in sorted(p for p in PILOT.iterdir() if (p / "result.json").exists())
            for tag in ("age0.25", "age1.00")]
    for row in rows:
        print(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows, "revision": code_revision()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
