"""Small CPU control for repair semantics; this is supervised regression, not RL.

No downloaded implementation is executed. Branches get identical future inputs,
teacher targets and minibatch order within a continuation repeat.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = (
    "continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset",
    "t_reset", "t_reset_equivalent_schedule",
)
SCENARIOS = ("unchanged", "target_sign_flip", "target_scale_drop", "input_shift")


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.head = nn.Linear(32, 1)
        self.double()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def new_optimizer(model: Model, lr: float = 0.001, eps: float = 1e-8) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=lr, eps=eps, foreach=False, fused=False)


def train_step(model: Model, optimizer: torch.optim.Adam, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def snapshot(model: Model, optimizer: torch.optim.Adam, initial: dict, generator: torch.Generator) -> dict:
    return {
        "weights": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "initial_weights": copy.deepcopy(initial),
        "training_batch_rng": generator.get_state().clone(),
    }


def restore(checkpoint: dict) -> tuple[Model, torch.optim.Adam]:
    model = Model()
    model.load_state_dict(copy.deepcopy(checkpoint["weights"]))
    optimizer = new_optimizer(model)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    return model, optimizer


def apply_action(model: Model, optimizer: torch.optim.Adam, checkpoint: dict, action: str) -> None:
    if action not in ACTIONS:
        raise ValueError(action)
    if action in {"head_reset", "head_and_optimizer_reset"}:
        # Keep Parameter identities. Replacing a module would invalidate optimizer references.
        with torch.no_grad():
            model.head.weight.copy_(checkpoint["initial_weights"]["head.weight"])
            model.head.bias.copy_(checkpoint["initial_weights"]["head.bias"])
    if action in {"optimizer_reset", "head_and_optimizer_reset"}:
        # Clear m, v and their shared timestep. Hyperparameters are deliberately retained.
        optimizer.state.clear()
    elif action == "t_reset":
        for state in optimizer.state.values():
            state["step"].zero_()


def equivalent_schedule(optimizer: torch.optim.Adam, local_next_step: int, base_groups: list[dict]) -> None:
    """Match a t-reset update without modifying moments or their global clock.

    Exact algebraic equivalence includes epsilon rescaling. The check in tests
    covers multiple subsequent steps, not just the first update norm.
    """
    for group, base in zip(optimizer.param_groups, base_groups, strict=True):
        steps = {int(optimizer.state[p]["step"].item()) for p in group["params"]}
        if len(steps) != 1:
            raise ValueError("Control requires a common clock across every parameter in a group")
        global_next_step = steps.pop() + 1
        b1, b2 = group["betas"]
        local_b1, local_b2 = 1 - b1 ** local_next_step, 1 - b2 ** local_next_step
        global_b1, global_b2 = 1 - b1 ** global_next_step, 1 - b2 ** global_next_step
        group["lr"] = base["lr"] * (math.sqrt(local_b2) / local_b1) / (math.sqrt(global_b2) / global_b1)
        group["eps"] = base["eps"] * math.sqrt(local_b2 / global_b2)


def teacher(x: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(415)
    w1 = torch.randn(8, 16, generator=generator, dtype=torch.float64) / math.sqrt(8)
    w2 = torch.randn(16, 1, generator=generator, dtype=torch.float64) / math.sqrt(16)
    return torch.tanh(x @ w1) @ w2


def data(seed: int, scenario: str, count: int = 512) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20000 + seed)
    old_x = torch.randn(count, 8, generator=generator, dtype=torch.float64)
    future_x = torch.randn(count, 8, generator=generator, dtype=torch.float64)
    test_x = torch.randn(count, 8, generator=generator, dtype=torch.float64)
    if scenario == "input_shift":
        future_x += 1.5
        test_x += 1.5
    old_y = teacher(old_x) * (30.0 if scenario == "target_scale_drop" else 1.0)
    future_y = teacher(future_x)
    test_y = teacher(test_x)
    if scenario == "target_sign_flip":
        future_y = -future_y
        test_y = -test_y
    return old_x, old_y, future_x, future_y, test_x, test_y


@torch.no_grad()
def evaluate(model: Model, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(model(x), y))


def diagnostics(model: Model, optimizer: torch.optim.Adam, x: torch.Tensor, y: torch.Tensor) -> dict:
    params = list(model.parameters())
    gradients = torch.autograd.grad(F.mse_loss(model(x), y), params)
    g = torch.cat([v.detach().flatten() for v in gradients])
    m = torch.cat([optimizer.state[p]["exp_avg"].flatten() for p in params])
    v = torch.cat([optimizer.state[p]["exp_avg_sq"].flatten() for p in params])
    w = torch.cat([p.detach().flatten() for p in params])
    denominator = float(g.norm() * m.norm())
    with torch.no_grad():
        features = model.features(x)
        singular = torch.linalg.svdvals(features - features.mean(dim=0))
        if float(singular.sum()) > 0:
            probabilities = singular / singular.sum()
            nonzero = probabilities > 0
            rank = float(torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum()))
        else:
            rank = 0.0
        activities = features.abs().mean(dim=0)
        normalized = activities / (activities.mean() + 1e-12)
    return {
        "pre_future_training_loss": evaluate(model, x, y),
        "weight_norm": float(w.norm()), "gradient_norm": float(g.norm()),
        "m_norm": float(m.norm()), "sqrt_v_norm": float(v.sqrt().norm()),
        "gradient_m_cosine": float(torch.dot(g, m)) / denominator if denominator else None,
        "feature_entropy_rank_centered": rank,
        "dormant_fraction_threshold_0_001": float((normalized <= 0.001).double().mean()),
        "optimizer_age": int(optimizer.state[params[0]]["step"].item()),
        "diagnostic_inputs": len(x), "diagnostic_backward_passes": 1,
    }


def run_branch(checkpoint: dict, action: str, future: tuple, batches: list[torch.Tensor]) -> dict:
    x, y, test_x, test_y = future
    model, optimizer = restore(checkpoint)
    base_groups = [{"lr": group["lr"], "eps": group["eps"]} for group in optimizer.param_groups]
    pre = evaluate(model, x, y)
    apply_action(model, optimizer, checkpoint, action)
    immediate = evaluate(model, x, y)
    record_steps = {0, 1, 5, 10, 20, 50, len(batches)}
    curve = [{"updates": 0, "training_loss": immediate, "heldout_loss": evaluate(model, test_x, test_y)}]
    first_update_norm = None
    failed = None
    for step, indices in enumerate(batches, 1):
        if action == "t_reset_equivalent_schedule":
            equivalent_schedule(optimizer, step, base_groups)
        before = torch.cat([p.detach().flatten().clone() for p in model.parameters()]) if step == 1 else None
        loss = train_step(model, optimizer, x[indices], y[indices])
        if not math.isfinite(loss) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            failed = f"nonfinite_at_update_{step}"
            break
        if step == 1:
            after = torch.cat([p.detach().flatten() for p in model.parameters()])
            first_update_norm = float((after - before).norm())
        if step in record_steps:
            curve.append({"updates": step, "training_loss": evaluate(model, x, y), "heldout_loss": evaluate(model, test_x, test_y)})
    return {
        "action": action, "loss_pre": pre, "loss_immediate": immediate,
        "loss_final": evaluate(model, x, y) if failed is None else None,
        "heldout_loss_final": evaluate(model, test_x, test_y) if failed is None else None,
        "first_update_norm": first_update_norm, "failure": failed, "curve": curve,
        "training_updates": len(batches) if failed is None else step,
    }


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ages = sorted(set(int(value) for value in args.ages.split(",")))
    if min(ages) <= 0 or args.seeds < 1 or args.repeats < 1 or args.updates < 1:
        raise ValueError("Positive ages, seeds, repeats and updates required")
    started = time.perf_counter()
    records, checkpoint_records = [], []
    for seed in range(args.seeds):
        for scenario in SCENARIOS:
            old_x, old_y, x, y, test_x, test_y = data(seed, scenario)
            torch.manual_seed(seed)
            model = Model()
            initial = copy.deepcopy(model.state_dict())
            optimizer = new_optimizer(model, args.lr)
            source_generator = torch.Generator().manual_seed(30000 + seed)
            for age in range(1, max(ages) + 1):
                indices = torch.randint(len(old_x), (64,), generator=source_generator)
                train_step(model, optimizer, old_x[indices], old_y[indices])
                if age not in ages:
                    continue
                checkpoint_id = f"{scenario}:seed{seed}:age{age}"
                checkpoint = snapshot(model, optimizer, initial, source_generator)
                checkpoint_records.append({
                    "checkpoint_id": checkpoint_id, "scenario": scenario, "training_seed": seed, "source_updates": age,
                    "pre_source_loss": evaluate(model, old_x, old_y),
                    "diagnostics": diagnostics(model, optimizer, x, y),
                })
                for repeat in range(args.repeats):
                    generator = torch.Generator().manual_seed(40000 + seed * 100 + repeat)
                    batches = [torch.randint(len(x), (64,), generator=generator) for _ in range(args.updates)]
                    for action in ACTIONS:
                        branch = run_branch(checkpoint, action, (x, y, test_x, test_y), batches)
                        records.append(dict(branch, checkpoint_id=checkpoint_id, scenario=scenario, training_seed=seed,
                                            source_updates=age, continuation_repeat=repeat))
                print(f"completed {checkpoint_id}", flush=True)
    return {
        "kind": "synthetic_supervised_semantics_control_not_RL", "date": "2026-10-05",
        "config": {"ages": ages, "seeds": args.seeds, "repeats": args.repeats, "future_updates": args.updates,
                   "batch_size": 64, "source_samples": 512, "future_training_samples": 512, "heldout_samples": 512,
                   "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
                   "dtype": "float64", "device": "cpu", "threads": 1, "actions": ACTIONS, "scenarios": SCENARIOS},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "wall_time_sec": time.perf_counter() - started,
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "checkpoints": checkpoint_records, "branches": records,
        "limitations": ["Artificial teacher shifts; no RL environment or return", "No learned selector or held-out environments",
                        "Diagnostics see available future labeled task; these are not generally available in live RL",
                        "Head reset returns to initial head, preserving current encoder and Parameter identities",
                        "Single fixed learning rate; not a tuned comparison of optimizer algorithms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ages", default="100,600")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "results" / "controlled_probe.json")
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(payload['checkpoints'])} checkpoints and {len(payload['branches'])} branches to {args.output}")
    print("Failures:", dict(Counter(r["failure"] for r in payload["branches"] if r["failure"])))


if __name__ == "__main__":
    main()
