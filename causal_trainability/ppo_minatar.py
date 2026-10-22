"""PPO on MinAtar for the topic-card line (sprint 02, T2).

CleanRL-style PPO with vectorised native MinAtar environments and two architectures:
`cnn` (conv 16 -> fc 128, as the MinAtar baseline) and `cnn_ln` (the same with
LayerNorm before every nonlinearity).  Telemetry is always on (per layer, every
iteration); L3 snapshots and full checkpoints (weights, theta0, Adam, envs, RNG and
the last rollout) are written at fixed fractions of the budget so that fresh/aged
hybrids can be assembled later (T3).

Run:  .venv\\Scripts\\python.exe -m causal_trainability.ppo_minatar --game breakout --arch cnn --seed 0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from minatar import Environment
from torch import nn

from causal_trainability.telemetry import ROOT, Telemetry, code_revision, config_hash

CONFIG = dict(game="breakout", arch="cnn", seed=0, total_steps=1_000_000, num_envs=8, num_steps=128,
              lr=2.5e-4, adam_eps=1e-8, gamma=0.99, gae_lambda=0.95, update_epochs=4, num_minibatches=4,
              clip=0.1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, sticky_action_prob=0.1,
              difficulty_ramping=True, episode_cap=10_000, input_channels=None, conv_channels=16,
              hidden=128, num_actions=6, reference_size=256, eval_episodes=30, telemetry_every=4,
              snapshot_fractions=[0.0, 0.25, 0.5, 0.75, 1.0], checkpoint_fractions=[0.0, 0.25, 1.0])
ARCHS = ("cnn", "cnn_ln")


def seed_of(*keys):
    return int(hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:8], 16)


def native_channels(game):
    """Native observation channels; padding to 10 would leave never-trained conv weights."""
    return int(Environment(game).state_shape()[2])


def resolve(cfg):
    return cfg if cfg["input_channels"] else dict(cfg, input_channels=native_channels(cfg["game"]))


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg["arch"] not in ARCHS:
            raise ValueError(f"Unknown arch {cfg['arch']}")
        c, h = cfg["conv_channels"], cfg["hidden"]
        self.layer_norm = cfg["arch"] == "cnn_ln"
        self.conv = nn.Conv2d(cfg["input_channels"], c, 3)
        self.fc = nn.Linear(c * 8 * 8, h)
        self.actor = nn.Linear(h, cfg["num_actions"])
        self.critic = nn.Linear(h, 1)
        if self.layer_norm:
            self.conv_norm = nn.LayerNorm([c, 8, 8])
            self.fc_norm = nn.LayerNorm(h)
        for layer, gain in ((self.conv, math.sqrt(2)), (self.fc, math.sqrt(2)), (self.actor, 0.01), (self.critic, 1.0)):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)

    def features(self, obs):
        x = self.conv(obs.float())
        x = torch.relu(self.conv_norm(x) if self.layer_norm else x).flatten(1)
        conv_out = x
        x = self.fc(x)
        return torch.relu(self.fc_norm(x) if self.layer_norm else x), conv_out

    def forward(self, obs):
        hidden, _ = self.features(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def layers(self):
        groups = {"conv": list(self.conv.parameters()), "fc": list(self.fc.parameters()),
                  "actor": list(self.actor.parameters()), "critic": list(self.critic.parameters())}
        if self.layer_norm:
            groups["conv"] += list(self.conv_norm.parameters())
            groups["fc"] += list(self.fc_norm.parameters())
        return groups


def observation(env, channels):
    native = np.asarray(env.state())
    obs = np.zeros((channels, 10, 10), dtype=np.uint8)
    obs[:native.shape[2]] = native.transpose(2, 0, 1)
    return obs


class VecMinAtar:
    """Native MinAtar environments stepped in lock-step; observations padded to 10 channels."""

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.envs = []
        for i in range(cfg["num_envs"]):
            env = Environment(cfg["game"], sticky_action_prob=cfg["sticky_action_prob"],
                              difficulty_ramping=cfg["difficulty_ramping"])
            env.seed(seed_of("env", cfg["game"], seed, i))
            env.reset()
            self.envs.append(env)
        self.returns = np.zeros(cfg["num_envs"])
        self.lengths = np.zeros(cfg["num_envs"], dtype=np.int64)
        self.finished = []  # (return, length, truncated) of completed episodes

    def observe(self, env):
        return observation(env, self.cfg["input_channels"])

    def observations(self):
        return np.stack([self.observe(env) for env in self.envs])

    def step(self, actions):
        """Returns rewards, dones and, for truncated episodes, the observation before reset."""
        rewards = np.zeros(len(self.envs), dtype=np.float32)
        dones = np.zeros(len(self.envs), dtype=bool)
        truncated_obs = {}
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            reward, terminal = env.act(int(action))
            rewards[i] = reward
            self.returns[i] += reward
            self.lengths[i] += 1
            truncated = not terminal and self.lengths[i] >= self.cfg["episode_cap"]
            if terminal or truncated:
                if truncated:
                    truncated_obs[i] = self.observe(env)
                self.finished.append((float(self.returns[i]), int(self.lengths[i]), bool(truncated)))
                self.returns[i], self.lengths[i] = 0.0, 0
                env.reset()
                dones[i] = True
        return rewards, dones, truncated_obs


def reference_batch(cfg):
    """Fixed observations from a uniform random policy; the same for every run of a game."""
    probe = dict(cfg, num_envs=8)
    envs = VecMinAtar(probe, seed_of("reference", cfg["game"]))
    rng = np.random.RandomState(seed_of("reference-actions", cfg["game"]))
    rows = []
    while len(rows) * probe["num_envs"] < cfg["reference_size"]:
        rows.append(envs.observations())
        envs.step(rng.randint(cfg["num_actions"], size=probe["num_envs"]))
    return torch.tensor(np.concatenate(rows)[: cfg["reference_size"]])


def effective_rank(features):
    singular = torch.linalg.svdvals(features.double() - features.double().mean(0))
    if float(singular.sum()) == 0.0:
        return 0.0
    p = singular / singular.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def representation_metrics(model, reference):
    with torch.no_grad():
        hidden, conv = model.features(reference)
    return {"conv": {"dead_fraction": float((conv <= 0).all(0).float().mean())},
            "fc": {"dead_fraction": float((hidden <= 0).all(0).float().mean()),
                   "effective_rank": effective_rank(hidden)}}


def evaluate(model, cfg, seeds, generator=None):
    """Mean return over seeded episodes; sampling policy if a generator is given, else uniform random."""
    returns = []
    for seed in seeds:
        env = Environment(cfg["game"], sticky_action_prob=cfg["sticky_action_prob"],
                          difficulty_ramping=cfg["difficulty_ramping"])
        env.seed(int(seed))
        env.reset()
        rng = np.random.RandomState(seed)
        total = 0.0
        for _ in range(cfg["episode_cap"]):
            if generator is None:
                action = rng.randint(cfg["num_actions"])
            else:
                with torch.no_grad():
                    logits, _ = model(torch.tensor(observation(env, cfg["input_channels"])[None]))
                action = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator))
            reward, done = env.act(action)
            total += reward
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


class PPO:
    def __init__(self, cfg, out):
        cfg = resolve(cfg)
        self.cfg, self.out = cfg, Path(out)
        torch.manual_seed(seed_of("init", cfg["game"], cfg["arch"], cfg["seed"]))
        self.model = ActorCritic(cfg)
        self.theta0 = copy.deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])
        self.envs = VecMinAtar(cfg, cfg["seed"])
        self.generator = torch.Generator().manual_seed(seed_of("ppo", cfg["game"], cfg["arch"], cfg["seed"]))
        self.obs = torch.tensor(self.envs.observations())
        self.global_step, self.iteration = 0, 0
        self.batch_size = cfg["num_envs"] * cfg["num_steps"]
        self.iterations = cfg["total_steps"] // self.batch_size
        self.last_rollout = None

    def rollout(self):
        cfg, n, t_max = self.cfg, self.cfg["num_envs"], self.cfg["num_steps"]
        obs = torch.zeros((t_max, n, cfg["input_channels"], 10, 10), dtype=torch.uint8)
        actions = torch.zeros((t_max, n), dtype=torch.long)
        logprobs, rewards, raw_rewards, dones, values = (torch.zeros((t_max, n)) for _ in range(5))
        timeouts = []  # (t, env, observation before reset) for re-bootstrapping with another critic
        for t in range(t_max):
            obs[t] = self.obs
            with torch.no_grad():
                logits, value = self.model(self.obs)
            probs = torch.softmax(logits, -1)
            action = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
            actions[t], values[t] = action, value
            logprobs[t] = torch.log_softmax(logits, -1).gather(1, action[:, None]).squeeze(1)
            reward, done, truncated = self.envs.step(action.numpy())
            reward = torch.tensor(reward)
            raw_rewards[t] = reward
            if truncated:  # bootstrap timeouts instead of treating them as terminal
                with torch.no_grad():
                    _, boot = self.model(torch.tensor(np.stack(list(truncated.values()))))
                for j, i in enumerate(truncated):
                    reward[i] += cfg["gamma"] * boot[j]
                    timeouts.append((t, i, torch.tensor(truncated[i])))
            rewards[t], dones[t] = reward, torch.tensor(done, dtype=torch.float32)
            self.obs = torch.tensor(self.envs.observations())
            self.global_step += n
        with torch.no_grad():
            _, next_value = self.model(self.obs)
        advantages = torch.zeros_like(rewards)
        last = torch.zeros(n)
        for t in reversed(range(t_max)):
            following = next_value if t == t_max - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg["gamma"] * following * nonterminal - values[t]
            last = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterminal * last
            advantages[t] = last
        returns = advantages + values
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        self.last_rollout = {"obs": flat(obs), "actions": flat(actions), "logprobs": flat(logprobs),
                             "advantages": flat(advantages), "returns": flat(returns), "values": flat(values),
                             "rewards": flat(rewards), "raw_rewards": flat(raw_rewards), "timeouts": timeouts,
                             "dones": flat(dones), "next_obs": self.obs.clone(), "shape": (t_max, n)}
        return self.last_rollout

    def learn(self, batch):
        cfg, size = self.cfg, self.batch_size
        minibatch = size // cfg["num_minibatches"]
        stats = {"approx_kl": [], "clipfrac": [], "value_loss": [], "policy_loss": [], "entropy": [],
                 "grad_norm_preclip": []}
        for _ in range(cfg["update_epochs"]):
            order = torch.randperm(size, generator=self.generator)
            for start in range(0, size, minibatch):
                idx = order[start:start + minibatch]
                logits, value = self.model(batch["obs"][idx])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, batch["actions"][idx][:, None]).squeeze(1)
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                log_ratio = logp - batch["logprobs"][idx]
                ratio = log_ratio.exp()
                adv = batch["advantages"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])).mean()
                value_loss = 0.5 * ((value - batch["returns"][idx]) ** 2).mean()
                loss = policy_loss - cfg["ent_coef"] * entropy + cfg["vf_coef"] * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                stats["grad_norm_preclip"].append(float(nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                                  cfg["max_grad_norm"])))
                self.optimizer.step()
                with torch.no_grad():
                    stats["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
                    stats["clipfrac"].append(float(((ratio - 1).abs() > cfg["clip"]).float().mean()))
                    stats["value_loss"].append(float(value_loss))
                    stats["policy_loss"].append(float(policy_loss))
                    stats["entropy"].append(float(entropy))
        with torch.no_grad():
            logits, value = self.model(batch["obs"])
        scale = {"logit_abs_mean": float(logits.abs().mean()), "logit_std": float(logits.std()),
                 "value_mean": float(value.mean()), "value_std": float(value.std()),
                 "advantage_raw_std": float(batch["advantages"].std())}
        variance = float(batch["returns"].var())
        explained = 1 - float((batch["returns"] - batch["values"]).var()) / variance if variance > 0 else None
        return {k: float(np.mean(v)) for k, v in stats.items()} | {"explained_variance": explained} | scale

    def checkpoint(self, tag):
        folder = self.out / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{tag}.pt"
        torch.save({"config": self.cfg, "model": self.model.state_dict(), "theta0": self.theta0,
                    "optimizer": self.optimizer.state_dict(), "envs": copy.deepcopy(self.envs),
                    "obs": self.obs.clone(), "generator": self.generator.get_state(),
                    "global_step": self.global_step, "iteration": self.iteration,
                    "rollout": self.last_rollout}, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with (self.out / "checkpoints.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"tag": tag, "path": path.relative_to(self.out).as_posix(), "sha256": digest,
                                     "global_step": self.global_step, "iteration": self.iteration}) + "\n")
        return path

    def train(self, telemetry_enabled=True):
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        reference = reference_batch(cfg)
        run_id = f"ppo_{cfg['game']}_{cfg['arch']}_s{cfg['seed']}"
        telemetry = Telemetry(self.out, run_id, self.model.layers(), self.optimizer, every=cfg["telemetry_every"],
                              enabled=telemetry_enabled,
                              meta={"kind": "ppo_source", "game": cfg["game"], "arch": cfg["arch"],
                                    "seed": cfg["seed"], "config_hash": config_hash(cfg)})
        snapshot_at = {round(f * self.iterations) for f in cfg["snapshot_fractions"]}
        # A checkpoint is taken after the rollout and before the update of its iteration, so its
        # weights, Adam state, data and critic all belong to the same moment (fresh = age 0.00).
        checkpoint_at = {}
        for f in cfg["checkpoint_fractions"]:
            checkpoint_at.setdefault(max(1, round(f * self.iterations)), []).append(f)
        if 0 in snapshot_at:
            telemetry.snapshot(0, self.model, self.optimizer)
        started = time.time()
        for self.iteration in range(1, self.iterations + 1):
            seen = len(self.envs.finished)
            batch = self.rollout()
            for fraction in checkpoint_at.get(self.iteration, []):
                self.checkpoint(f"age{fraction:.2f}")
            stats = self.learn(batch)
            episodes = self.envs.finished[seen:]
            stats.update(env_steps=self.global_step, steps_per_sec=self.global_step / (time.time() - started),
                         episodes=len(episodes),
                         episode_return=float(np.mean([e[0] for e in episodes])) if episodes else None)
            if telemetry.enabled and (self.iteration % telemetry.every == 0 or self.iteration == self.iterations):
                telemetry.step(self.iteration, force=True, metrics=stats,
                               layer_extra=representation_metrics(self.model, reference))
            if self.iteration in snapshot_at:
                telemetry.snapshot(self.iteration, self.model, self.optimizer)
        telemetry.close()
        return time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="breakout")
    parser.add_argument("--arch", default="cnn", choices=ARCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=CONFIG["total_steps"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = resolve(dict(CONFIG, game=args.game, arch=args.arch, seed=args.seed, total_steps=args.steps))
    out = Path(args.out or ROOT / "runs" / "causal-trainability" / "ppo_pilot" / f"{args.game}_{args.arch}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    ppo = PPO(cfg, out)
    seconds = ppo.train()
    eval_seeds = [seed_of("eval", cfg["game"], i) for i in range(cfg["eval_episodes"])]
    result = {"config": cfg, "config_hash": config_hash(cfg), "revision": code_revision(),
              "wall_time_sec": seconds, "env_steps": ppo.global_step,
              "steps_per_sec": ppo.global_step / seconds,
              "eval_return": evaluate(ppo.model, cfg, eval_seeds, torch.Generator().manual_seed(seed_of("eval-policy"))),
              "random_return": evaluate(ppo.model, cfg, eval_seeds),
              "last_training_returns": [e[0] for e in ppo.envs.finished[-50:]]}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("env_steps", "wall_time_sec", "steps_per_sec", "eval_return", "random_return")}))


if __name__ == "__main__":
    # Run through the importable module so pickled classes are not recorded as __main__.*
    import causal_trainability.ppo_minatar as module
    module.main()
