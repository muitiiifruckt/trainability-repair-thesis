"""Continuation-LR controls with unchanged source histories and paired tapes.

This extends the earlier supervised semantics control, never labels it as RL.
The 100-step run is the exact prefix of the 500-step run; actual compute counts
include both executions. Incremental results make the sweep resumable.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments import controlled_probe as probe


def run(output: Path, continuation_lrs=(.00025, .001, .004), source_lr=.001,
        horizons=(100, 500), seeds=3, repeats=2, ages=(100, 600)):
    torch.set_num_threads(1)
    config = dict(continuation_lrs=list(continuation_lrs), source_lr=source_lr,
                  horizons=list(horizons), seeds=seeds, repeats=repeats, ages=list(ages),
                  source_sha256=hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest())
    result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "kind": "synthetic_supervised_LR_control_not_RL", "config": config,
        "branches": [], "completed_checkpoints": [], "actual_optimizer_updates": 0}
    if result["config"] != config:
        raise ValueError("Synthetic sweep configuration or source changed")
    initial_updates = result["actual_optimizer_updates"]
    started = time.monotonic()
    for seed in range(seeds):
        for scenario in probe.SCENARIOS:
            old_x, old_y, x, y, test_x, test_y = probe.data(seed, scenario)
            torch.manual_seed(seed)
            model = probe.Model()
            initial = copy.deepcopy(model.state_dict())
            optimizer = probe.new_optimizer(model, source_lr)
            source_generator = torch.Generator().manual_seed(30000+seed)
            if all(f"{scenario}:seed{seed}:age{age}" in result["completed_checkpoints"] for age in ages):
                continue
            for age in range(1, max(ages)+1):
                indices = torch.randint(len(old_x), (64,), generator=source_generator)
                probe.train_step(model, optimizer, old_x[indices], old_y[indices])
                result["actual_optimizer_updates"] += 1
                identifier = f"{scenario}:seed{seed}:age{age}"
                if age not in ages or identifier in result["completed_checkpoints"]:
                    continue
                checkpoint = probe.snapshot(model, optimizer, initial, source_generator)
                for repeat in range(repeats):
                    generator = torch.Generator().manual_seed(40000+seed*100+repeat)
                    tape = [torch.randint(len(x), (64,), generator=generator) for _ in range(max(horizons))]
                    tape_hash = hashlib.sha256(torch.stack(tape).numpy().tobytes()).hexdigest()
                    for lr in continuation_lrs:
                        adapted = copy.deepcopy(checkpoint)
                        for group in adapted["optimizer"]["param_groups"]:
                            group["lr"] = lr
                        for horizon in horizons:
                            for action in probe.ACTIONS:
                                branch = probe.run_branch(adapted, action, (x,y,test_x,test_y), tape[:horizon])
                                result["branches"].append(dict(branch, checkpoint_id=identifier,
                                    scenario=scenario, training_seed=seed, source_updates=age,
                                    continuation_repeat=repeat, continuation_lr=lr, horizon=horizon,
                                    tape_sha256=tape_hash))
                                result["actual_optimizer_updates"] += branch["training_updates"]
                result["completed_checkpoints"].append(identifier)
                result["wall_time_sec"] = result.get("wall_time_sec", 0)+time.monotonic()-started
                started = time.monotonic()
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
                temporary.replace(output)
                print(f"synthetic LR: {identifier}; {len(result['branches'])} branches", flush=True)
    result["updates_this_invocation"] = result["actual_optimizer_updates"]-initial_updates
    return result


if __name__ == "__main__":
    run(Path(probe.ROOT / "research/results/synthetic_lr_sweep.json"))
