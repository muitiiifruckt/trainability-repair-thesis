"""Fresh/aged hybrids from PPO checkpoints (sprint 02, T3; docs/topic_card.md).

A hybrid H(params, adam, data, targets) takes each part from the fresh checkpoint
(age 0.00: theta0, empty Adam, the first rollout) or from an aged one of the same
history.  Targets are rebuilt with GAE from the chosen critic on the chosen data, so
data and target construction can be crossed independently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from causal_trainability.ppo_minatar import ActorCritic

PARTS = ("params", "adam", "data", "targets")


def load(folder, tag):
    """Load a checkpoint and verify it against checkpoints.jsonl."""
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    row = next(r for r in rows if r["tag"] == tag)
    path = folder / row["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def model_from(state, cfg):
    with torch.random.fork_rng():  # construction must not consume the caller's global RNG
        model = ActorCritic(cfg)
    model.load_state_dict(state)
    return model


def targets(critic, rollout, cfg):
    """GAE advantages and returns of `rollout` under `critic`'s value function."""
    steps, envs = rollout["shape"]
    with torch.no_grad():
        _, values = critic(rollout["obs"])
        _, next_value = critic(rollout["next_obs"])
    values = values.reshape(steps, envs)
    rewards = rollout["raw_rewards"].reshape(steps, envs).clone()
    if rollout["timeouts"]:  # re-bootstrap truncated episodes with this critic
        with torch.no_grad():
            _, boot = critic(torch.stack([obs for _, _, obs in rollout["timeouts"]]))
        for (t, i, _), value in zip(rollout["timeouts"], boot):
            rewards[t, i] += cfg["gamma"] * value
    dones = rollout["dones"].reshape(steps, envs)
    advantages = torch.zeros_like(rewards)
    last = torch.zeros(envs)
    for t in reversed(range(steps)):
        following = next_value if t == steps - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + cfg["gamma"] * following * nonterminal - values[t]
        last = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterminal * last
        advantages[t] = last
    return advantages.flatten(), (advantages + values).flatten(), values.flatten()


def assemble(fresh, aged, choice):
    """Return (model, optimizer, batch) of the hybrid; `choice` maps each part to 'fresh' or 'aged'."""
    if set(choice) != set(PARTS) or not set(choice.values()) <= {"fresh", "aged"}:
        raise ValueError(f"Bad hybrid choice {choice}")
    cfg = aged["config"]
    if fresh["config"] != cfg or any(not torch.equal(fresh["theta0"][k], aged["theta0"][k]) for k in aged["theta0"]):
        raise ValueError("Fresh and aged checkpoints come from different histories")
    source = {"fresh": fresh, "aged": aged}
    model = model_from(source[choice["params"]]["model"], cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])
    if choice["adam"] == "aged":
        optimizer.load_state_dict(aged["optimizer"])  # binds the saved state to this model's parameters
    rollout = source[choice["data"]]["rollout"]
    critic = model_from(source[choice["targets"]]["model"], cfg)
    advantages, returns, values = targets(critic, rollout, cfg)
    batch = {"obs": rollout["obs"], "actions": rollout["actions"], "logprobs": rollout["logprobs"],
             "advantages": advantages, "returns": returns, "values": values}
    return model, optimizer, batch
