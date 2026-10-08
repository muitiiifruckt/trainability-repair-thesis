"""Small CPU DQN core with explicit checkpoint and one-shot repair semantics.

The MinAtar environment is a dependency; no downloaded learner is executed.
Observations are channel-first binary tensors, padded to ten channels. Replay
stores raw rewards and genuine terminal flags: our episode cap is truncation.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Import the dependency before learner work begins. On the observed Windows
# runtime, late MinAtar rendering-dependency imports after BLAS work caused a
# native pyarrow/pandas heap fault. No rendering is used by this core.
from minatar import Environment


@dataclass
class DQNConfig:
    input_channels: int = 10
    conv_channels: int = 16
    hidden_size: int = 128
    num_actions: int = 6
    lr: float = 2.5e-4
    batch_size: int = 32
    replay_capacity: int = 100000
    warmup: int = 5000
    gamma: float = 0.99
    target_period: int = 1000
    epsilon_decay: int = 100000
    epsilon_final: float = 0.1
    grad_clip: float = 10.0
    episode_cap: int = 10000
    replay_ratio: float = 1.0
    reward_scale: float = 1.0
    optimizer: str = "adam"
    dtype: str = "float32"
    threads: int = 1
    adam_eps: float = 1e-8
    adam_betas: tuple[float, float] = (0.9, 0.999)
    sticky_action_prob: float = 0.1
    difficulty_ramping: bool = True

    def validate(self) -> None:
        for key in ("input_channels", "conv_channels", "hidden_size", "num_actions",
                    "batch_size", "replay_capacity", "target_period", "epsilon_decay",
                    "episode_cap", "threads"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.replay_capacity < max(self.batch_size, self.warmup):
            raise ValueError("Replay capacity must cover warmup and a training batch")
        if self.warmup < 0 or self.replay_ratio < 0:
            raise ValueError("Warmup and replay ratio must be nonnegative")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_final <= 1:
            raise ValueError("gamma and epsilon_final must lie in [0, 1]")
        if not 0 <= self.sticky_action_prob <= 1:
            raise ValueError("sticky_action_prob must lie in [0, 1]")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("CPU dtype must be float32 or float64")
        if self.lr <= 0 or self.adam_eps <= 0 or self.grad_clip < 0:
            raise ValueError("lr/epsilon must be positive and clipping nonnegative")
        if not all(0 <= beta < 1 for beta in self.adam_betas):
            raise ValueError("Adam betas must lie in [0, 1)")
        if not math.isfinite(self.reward_scale) or not math.isfinite(self.replay_ratio):
            raise ValueError("Reward scale and replay ratio must be finite")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


def _stream_seed(seed: int, stream: int) -> int:
    return int(np.random.SeedSequence([int(seed), stream]).generate_state(1)[0])


class QNetwork(nn.Module):
    def __init__(self, config: DQNConfig, injected: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(config.input_channels, config.conv_channels, 3)
        self.fc_hidden = nn.Linear(config.conv_channels * 8 * 8, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.num_actions)
        self.injected = injected
        if injected:
            self.injection_head = nn.Linear(config.hidden_size, config.num_actions)
            self.injection_reference = nn.Linear(config.hidden_size, config.num_actions)
            self.head.requires_grad_(False)
            self.injection_reference.requires_grad_(False)
        self.to(device="cpu", dtype=config.torch_dtype)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cpu", dtype=self.conv.weight.dtype)
        return F.relu(self.fc_hidden(F.relu(self.conv(x)).flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        result = self.head(features)
        if self.injected:
            result = result + (self.injection_head(features) - self.injection_reference(features))
        return result


class Replay:
    def __init__(self, config: DQNConfig):
        self.capacity = config.replay_capacity
        self.dtype = config.torch_dtype
        shape = (self.capacity, config.input_channels, 10, 10)
        self.obs = np.zeros(shape, dtype=np.uint8)
        self.next_obs = np.zeros(shape, dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminals = np.zeros(self.capacity, dtype=np.bool_)
        self.episode_ids = np.zeros(self.capacity, dtype=np.int64)
        self.transition_ids = np.zeros(self.capacity, dtype=np.int64)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, terminal, episode_id, transition_id):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminals[self.ptr] = terminal
        self.episode_ids[self.ptr] = episode_id
        self.transition_ids[self.ptr] = transition_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ordered_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size, dtype=np.int64)
        return np.concatenate((np.arange(self.ptr, self.capacity), np.arange(self.ptr))).astype(np.int64)

    def sample_indices(self, rng, count: int) -> np.ndarray:
        if self.size == 0 or count < 1:
            raise ValueError("Cannot sample an empty replay or nonpositive count")
        return np.asarray(rng.choice(self.size, size=count, replace=count > self.size), dtype=np.int64)

    def batch(self, indices) -> dict[str, torch.Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("Replay indices must be a valid one-dimensional array")
        return {
            "obs": torch.tensor(self.obs[indices], dtype=self.dtype),
            "next_obs": torch.tensor(self.next_obs[indices], dtype=self.dtype),
            "actions": torch.tensor(self.actions[indices], dtype=torch.long),
            "rewards": torch.tensor(self.rewards[indices], dtype=self.dtype),
            "terminals": torch.tensor(self.terminals[indices], dtype=torch.bool),
        }

    def snapshot(self) -> dict:
        # Only occupied slots are copied. Full capacity is reconstructed on restore.
        result = {name: getattr(self, name)[:self.size].copy() for name in
                  ("obs", "next_obs", "actions", "rewards", "terminals", "episode_ids", "transition_ids")}
        for value in result.values():
            value.setflags(write=False)
        return dict(result, ptr=self.ptr, size=self.size, capacity=self.capacity)

    @classmethod
    def restore(cls, snapshot: dict, config: DQNConfig) -> "Replay":
        replay = cls(config)
        if snapshot["capacity"] != replay.capacity:
            raise ValueError("Replay snapshot capacity does not match config")
        replay.size, replay.ptr = int(snapshot["size"]), int(snapshot["ptr"])
        for name in ("obs", "next_obs", "actions", "rewards", "terminals", "episode_ids", "transition_ids"):
            getattr(replay, name)[:replay.size] = snapshot[name]
        return replay


@torch.no_grad()
def td_targets(target: nn.Module, batch: dict, gamma: float, reward_scale: float = 1.0) -> torch.Tensor:
    """Plain DQN targets. Genuine terminal rows never call the next-state critic."""
    result = batch["rewards"].reshape(-1) * reward_scale
    alive = ~batch["terminals"].reshape(-1).bool()
    if alive.any():
        result = result.clone()
        result[alive] += gamma * target(batch["next_obs"][alive]).max(dim=1).values
    return result.detach()


def equivalent_adam_schedule(optimizer, local_next_step: int, base_groups: list[dict]) -> None:
    """Exact t-only reset control with epsilon outside sqrt and common group clocks."""
    if not isinstance(optimizer, torch.optim.Adam) or local_next_step < 1:
        raise ValueError("The schedule control requires Adam and positive local time")
    for group, base in zip(optimizer.param_groups, base_groups, strict=True):
        steps = {int(optimizer.state[p].get("step", torch.tensor(0)).item()) for p in group["params"]}
        if len(steps) != 1:
            raise ValueError("Control requires a common clock across a parameter group")
        global_next_step = steps.pop() + 1
        b1, b2 = group["betas"]
        local_b1, local_b2 = 1 - b1 ** local_next_step, 1 - b2 ** local_next_step
        global_b1, global_b2 = 1 - b1 ** global_next_step, 1 - b2 ** global_next_step
        group["lr"] = base["lr"] * (math.sqrt(local_b2) / local_b1) / (math.sqrt(global_b2) / global_b1)
        group["eps"] = base["eps"] * math.sqrt(local_b2 / global_b2)


def _optimizer(model: QNetwork, config: DQNConfig):
    # Frozen original head remains bound after injection; frozen reference is excluded.
    base_params = [p for name, p in model.named_parameters() if not name.startswith("injection_")]
    if config.optimizer.lower() == "adam":
        result = torch.optim.Adam(base_params, lr=config.lr, betas=config.adam_betas,
                                  eps=config.adam_eps, weight_decay=0, amsgrad=False,
                                  foreach=False, fused=False)
    elif config.optimizer.lower() == "rmsprop":
        result = torch.optim.RMSprop(base_params, lr=config.lr, alpha=0.95, eps=0.01, centered=True)
    elif config.optimizer.lower() in ("sgd", "sgd_momentum"):
        result = torch.optim.SGD(base_params, lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer {config.optimizer}")
    if model.injected:
        result.add_param_group({"params": list(model.injection_head.parameters())})
    return result


class TrainingState:
    @classmethod
    def create(cls, game: str, seed: int, config: DQNConfig | None = None) -> "TrainingState":
        state = cls()
        state.config = copy.deepcopy(config or DQNConfig())
        state.config.validate()
        torch.set_num_threads(state.config.threads)
        state.game, state.seed = game, int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_stream_seed(seed, 0))
            state.q = QNetwork(state.config)
        state.target = copy.deepcopy(state.q).requires_grad_(False)
        state.optimizer = _optimizer(state.q, state.config)
        state.initial_weights = copy.deepcopy(state.q.state_dict())
        state.replay = Replay(state.config)
        state.env = Environment(game, sticky_action_prob=state.config.sticky_action_prob,
                                difficulty_ramping=state.config.difficulty_ramping)
        state.env.seed(_stream_seed(seed, 1))
        state.env.reset()
        if state.env.num_actions() != state.config.num_actions:
            raise ValueError("Configured action count must match the native environment")
        state.exploration_rng = np.random.RandomState(_stream_seed(seed, 2))
        state.replay_rng = np.random.RandomState(_stream_seed(seed, 3))
        state.environment_steps = state.gradient_updates = state.last_target_update = 0
        state.episode_id = state.episode_length = 0
        state.episode_return = 0.0
        state.episode_returns = []
        state.episode_lengths = []
        state.episode_truncated = []
        state._update_credit = 0.0
        state._schedule_control = None
        state.last_sample_indices = None
        state.repair_history = []
        state.observation = state._observation()
        return state

    @property
    def environment(self):
        return self.env

    def _observation(self) -> np.ndarray:
        # Native state may alias environment storage; take ownership before act/reset.
        native = np.asarray(self.env.state()).copy()
        if native.shape[:2] != (10, 10) or native.shape[2] > self.config.input_channels:
            raise ValueError(f"Unsupported native observation shape {native.shape}")
        result = np.zeros((self.config.input_channels, 10, 10), dtype=np.uint8)
        result[:native.shape[2]] = native.transpose(2, 0, 1)
        return result

    def exploration_epsilon(self) -> float:
        progress = min(1.0, max(0, self.environment_steps - self.config.warmup) / self.config.epsilon_decay)
        return 1.0 - (1.0 - self.config.epsilon_final) * progress

    def _check_bindings(self) -> None:
        optimizer_params = [p for group in self.optimizer.param_groups for p in group["params"]]
        if len({id(p) for p in optimizer_params}) != len(optimizer_params):
            raise RuntimeError("A parameter is bound to the optimizer more than once")
        bound = {id(p) for p in optimizer_params}
        if any(id(p) not in bound for p in self.q.parameters() if p.requires_grad):
            raise RuntimeError("A trainable parameter is missing from the optimizer")

    def learn_batch(self, batch: dict, fixed_targets: torch.Tensor | None = None,
                    allow_target_sync: bool = True) -> dict:
        self._check_bindings()
        targets = td_targets(self.target, batch, self.config.gamma, self.config.reward_scale) if fixed_targets is None \
            else fixed_targets.detach().to(device="cpu", dtype=self.config.torch_dtype).reshape(-1)
        predictions = self.q(batch["obs"]).gather(1, batch["actions"].reshape(-1, 1)).squeeze(1)
        if targets.shape != predictions.shape:
            raise ValueError("TD targets must have one scalar per transition")
        loss = F.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for p in self.q.parameters() if p.requires_grad]
        gradient_norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in params if p.grad is not None]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Nonfinite DQN gradient")
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip, error_if_nonfinite=True)
        if self._schedule_control is not None:
            equivalent_adam_schedule(self.optimizer, self._schedule_control["updates"] + 1,
                                     self._schedule_control["base_groups"])
        before = torch.cat([p.detach().flatten().clone() for p in params])
        self.optimizer.step()
        self.gradient_updates += 1
        if self._schedule_control is not None:
            self._schedule_control["updates"] += 1
        if not all(torch.isfinite(p).all() for p in self.q.parameters()):
            raise FloatingPointError("Nonfinite DQN parameters")
        update_norm = float((torch.cat([p.detach().flatten() for p in params]) - before).norm())
        self.optimizer.zero_grad(set_to_none=True)
        synced = False
        if allow_target_sync and self.gradient_updates - self.last_target_update >= self.config.target_period:
            # Structure can change during injection, so recreate the frozen target.
            self.target = copy.deepcopy(self.q).requires_grad_(False)
            self.last_target_update = self.gradient_updates
            synced = True
        return {"loss": float(loss.detach()), "gradient_norm": float(gradient_norm),
                "update_norm": update_norm, "target_synced": synced}

    def step(self) -> dict:
        epsilon = self.exploration_epsilon()
        if self.environment_steps < self.config.warmup or self.exploration_rng.rand() < epsilon:
            action = int(self.exploration_rng.randint(self.config.num_actions))
        else:
            with torch.no_grad():
                obs = torch.tensor(self.observation[None], dtype=self.config.torch_dtype)
                action = int(self.q(obs).argmax(dim=1).item())
        raw_reward, terminal = self.env.act(action)
        next_obs = self._observation()
        self.environment_steps += 1
        self.episode_length += 1
        self.episode_return += float(raw_reward)
        truncated = self.episode_length >= self.config.episode_cap and not bool(terminal)
        self.replay.add(self.observation, next_obs, action, raw_reward, bool(terminal),
                        self.episode_id, self.environment_steps)
        self.observation = next_obs
        updates = []
        if self.replay.size >= max(self.config.warmup, self.config.batch_size):
            self._update_credit += self.config.replay_ratio
            while self._update_credit + 1e-12 >= 1:
                indices = self.replay.sample_indices(self.replay_rng, self.config.batch_size)
                self.last_sample_indices = indices.copy()
                updates.append(self.learn_batch(self.replay.batch(indices)))
                self._update_credit -= 1
        if terminal or truncated:
            self.episode_returns.append(self.episode_return)
            self.episode_lengths.append(self.episode_length)
            self.episode_truncated.append(bool(truncated))
            self.episode_id += 1
            self.episode_return = 0.0
            self.episode_length = 0
            self.env.reset()
            self.observation = self._observation()
        return {"raw_reward": float(raw_reward), "reward": float(raw_reward), "action": action,
                "terminal": bool(terminal), "truncated": bool(truncated), "epsilon": epsilon,
                "loss": float(np.mean([item["loss"] for item in updates])) if updates else None,
                "updates": len(updates), "environment_steps": self.environment_steps,
                "gradient_updates": self.gradient_updates,
                "target_synced": any(item["target_synced"] for item in updates)}

    def snapshot(self) -> dict:
        """Owned snapshot, treated as immutable; no tensor/storage aliases to the learner.

        Replay arrays are write-protected. Python dicts/tensors are conventionally
        immutable, not a security boundary; restore always makes fresh copies.
        """
        return {
            "schema_version": 1, "config": asdict(self.config), "game": self.game, "seed": self.seed,
            "weights": copy.deepcopy(self.q.state_dict()), "q_injected": self.q.injected,
            "target": copy.deepcopy(self.target.state_dict()), "target_injected": self.target.injected,
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "initial_weights": copy.deepcopy(self.initial_weights), "replay": self.replay.snapshot(),
            "environment": copy.deepcopy(self.env), "observation": self.observation.copy(),
            "exploration_rng": copy.deepcopy(self.exploration_rng.get_state()),
            "replay_rng": copy.deepcopy(self.replay_rng.get_state()),
            "environment_steps": self.environment_steps, "gradient_updates": self.gradient_updates,
            "last_target_update": self.last_target_update, "episode_id": self.episode_id,
            "episode_length": self.episode_length, "episode_return": self.episode_return,
            "episode_returns": self.episode_returns.copy(), "episode_lengths": self.episode_lengths.copy(),
            "episode_truncated": self.episode_truncated.copy(), "update_credit": self._update_credit,
            "schedule_control": copy.deepcopy(self._schedule_control),
            "repair_history": copy.deepcopy(self.repair_history),
            "last_sample_indices": None if self.last_sample_indices is None else self.last_sample_indices.copy(),
        }

    @classmethod
    def restore(cls, checkpoint: dict, continuation_seed: int | None = None) -> "TrainingState":
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema")
        state = cls()
        state.config = DQNConfig(**copy.deepcopy(checkpoint["config"]))
        state.config.validate()
        torch.set_num_threads(state.config.threads)
        state.game, state.seed = checkpoint["game"], checkpoint["seed"]
        # Construction consumes no caller RNG; all initialized tensors are overwritten.
        with torch.random.fork_rng(devices=[]):
            state.q = QNetwork(state.config, checkpoint.get("q_injected", False))
            state.target = QNetwork(state.config, checkpoint.get("target_injected", False))
        state.q.load_state_dict(copy.deepcopy(checkpoint["weights"]))
        state.target.load_state_dict(copy.deepcopy(checkpoint["target"]))
        state.target.requires_grad_(False)
        state.optimizer = _optimizer(state.q, state.config)
        state.optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        state.initial_weights = copy.deepcopy(checkpoint["initial_weights"])
        state.replay = Replay.restore(checkpoint["replay"], state.config)
        state.env = copy.deepcopy(checkpoint["environment"])
        state.observation = checkpoint["observation"].copy()
        state.exploration_rng = np.random.RandomState()
        state.replay_rng = np.random.RandomState()
        state.exploration_rng.set_state(copy.deepcopy(checkpoint["exploration_rng"]))
        state.replay_rng.set_state(copy.deepcopy(checkpoint["replay_rng"]))
        if continuation_seed is not None:
            state.env.seed(_stream_seed(continuation_seed, 1))
            state.exploration_rng.seed(_stream_seed(continuation_seed, 2))
            state.replay_rng.seed(_stream_seed(continuation_seed, 3))
        for name in ("environment_steps", "gradient_updates", "last_target_update", "episode_id",
                     "episode_length", "episode_return", "episode_returns", "episode_lengths", "episode_truncated"):
            setattr(state, name, copy.deepcopy(checkpoint[name]))
        state._update_credit = checkpoint["update_credit"]
        state._schedule_control = copy.deepcopy(checkpoint["schedule_control"])
        state.repair_history = copy.deepcopy(checkpoint["repair_history"])
        state.last_sample_indices = copy.deepcopy(checkpoint["last_sample_indices"])
        state._check_bindings()
        return state

    def apply_repair(self, action: str, repair_seed: int = 0) -> dict:
        return apply_repair(self, action, seed=repair_seed)

    @torch.no_grad()
    def evaluate(self, seeds) -> dict:
        returns, lengths, truncated = [], [], []
        for seed in seeds:
            env = Environment(self.game, sticky_action_prob=self.config.sticky_action_prob,
                              difficulty_ramping=self.config.difficulty_ramping)
            env.seed(int(seed))
            env.reset()
            total, done = 0.0, False
            for length in range(1, self.config.episode_cap + 1):
                native = np.asarray(env.state()).copy()
                observation = np.zeros((self.config.input_channels, 10, 10), dtype=np.uint8)
                observation[:native.shape[2]] = native.transpose(2, 0, 1)
                action = int(self.q(torch.tensor(observation[None], dtype=self.config.torch_dtype)).argmax(1))
                reward, done = env.act(action)
                total += float(reward)
                if done:
                    break
            returns.append(total)
            lengths.append(length)
            truncated.append(not bool(done))
        if not returns:
            raise ValueError("Evaluation needs at least one seed")
        return {"mean_return": float(np.mean(returns)), "returns": returns, "lengths": lengths,
                "truncated": truncated, "evaluation_environment_steps": int(sum(lengths)),
                "greedy": True, "episode_cap": self.config.episode_cap}

    def diagnostics(self, batch_size: int = 1024, seed: int = 0) -> dict:
        if self.replay.size == 0:
            return {"diagnostic_inputs": 0, "diagnostic_backward_passes": 0,
                    "environment_steps": self.environment_steps, "gradient_updates": self.gradient_updates}
        rng = np.random.RandomState(seed)
        indices = self.replay.sample_indices(rng, min(batch_size, self.replay.size))
        batch = self.replay.batch(indices)
        targets = td_targets(self.target, batch, self.config.gamma, self.config.reward_scale)
        predictions = self.q(batch["obs"]).gather(1, batch["actions"][:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predictions, targets)
        params = [p for p in self.q.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(loss, params)
        g = torch.cat([value.detach().flatten() for value in gradients])
        m = torch.cat([self.optimizer.state.get(p, {}).get("exp_avg", torch.zeros_like(p)).flatten() for p in params])
        v = torch.cat([self.optimizer.state.get(p, {}).get("exp_avg_sq", torch.zeros_like(p)).flatten() for p in params])
        denominator = float(g.norm() * m.norm())
        with torch.no_grad():
            features = self.q.features(batch["obs"])
            singular = torch.linalg.svdvals(features - features.mean(0))
            probabilities = singular / singular.sum() if float(singular.sum()) else singular
            nonzero = probabilities > 0
            rank = float(torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())) if nonzero.any() else 0.0
            activity = features.abs().mean(0)
            normalized = activity / (activity.mean() + 1e-12)
            output = self.q(batch["obs"])
        optimizer_steps = [int(self.optimizer.state.get(p, {}).get("step", torch.tensor(0)).item()) for p in params]
        return {
            "pre_td_huber": float(loss.detach()), "pre_td_mse": float(F.mse_loss(predictions, targets).detach()),
            "weight_norm": float(torch.cat([p.detach().flatten() for p in params]).norm()),
            "gradient_norm": float(g.norm()), "m_norm": float(m.norm()), "sqrt_v_norm": float(v.sqrt().norm()),
            "gradient_m_cosine": float(torch.dot(g, m)) / denominator if denominator else None,
            "feature_entropy_rank_centered": rank,
            "dormant_fraction_threshold_0_001": float((normalized <= 0.001).float().mean()),
            "q_abs_max": float(output.abs().max()), "target_abs_max": float(targets.abs().max()),
            "environment_steps": self.environment_steps, "gradient_updates": self.gradient_updates,
            "target_age_updates": self.gradient_updates - self.last_target_update,
            "optimizer_age_min": min(optimizer_steps), "optimizer_age_max": max(optimizer_steps),
            "recent_return_mean": float(np.mean(self.episode_returns[-20:])) if self.episode_returns else None,
            "replay_size": self.replay.size, "diagnostic_inputs": len(indices),
            "diagnostic_backward_passes": 1, "diagnostic_environment_steps": 0,
        }


def apply_repair(state: TrainingState, action: str, seed: int = 0) -> dict:
    aliases = {"head_reset_optimizer_reset": "head_and_optimizer_reset", "full_reset": "optimizer_reset",
               "parameter_repair": "head_reset", "parameter_and_optimizer": "head_and_optimizer_reset",
               "plasticity_injection": "injection"}
    action = aliases.get(action, action)
    allowed = {"continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset", "fresh_head_reset",
               "fresh_head_and_optimizer_reset", "shrink_perturb", "t_reset", "t_reset_equivalent_schedule", "injection"}
    if action not in allowed:
        raise ValueError(f"Unknown repair {action}")
    if action == "shrink_perturb" and state.q.injected:
        raise ValueError("Shrink-and-perturb is defined only for the original network")
    if action in {"t_reset", "t_reset_equivalent_schedule"} and not isinstance(state.optimizer, torch.optim.Adam):
        raise ValueError("Timestep controls require Adam")
    touched = []
    if action in {"head_reset", "head_and_optimizer_reset"}:
        with torch.no_grad():
            state.q.head.weight.copy_(state.initial_weights["head.weight"])
            state.q.head.bias.copy_(state.initial_weights["head.bias"])
        touched.append("online_head")
    if action in {"fresh_head_reset", "fresh_head_and_optimizer_reset", "shrink_perturb", "injection"}:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_stream_seed(seed, 4))
            if action == "shrink_perturb":
                fresh = QNetwork(state.config)
                with torch.no_grad():
                    for name, parameter in state.q.named_parameters():
                        if parameter.requires_grad:
                            parameter.mul_(0.99).add_(fresh.state_dict()[name], alpha=0.01)
                touched.append("online_weights_shrink_0.99_perturb_0.01")
            else:
                fresh_head = nn.Linear(state.config.hidden_size, state.config.num_actions).to(dtype=state.config.torch_dtype)
                if action == "injection":
                    if state.q.injected:
                        raise ValueError("The head-only injection can be applied once")
                    if state._schedule_control is not None:
                        raise ValueError("Injection cannot be combined with the common-clock schedule control")
                    state.q.head.requires_grad_(False)
                    state.q.injection_head = fresh_head
                    state.q.injection_reference = copy.deepcopy(fresh_head).requires_grad_(False)
                    state.q.injected = True
                    state.optimizer.add_param_group({"params": list(fresh_head.parameters()),
                                                     "lr": state.optimizer.param_groups[0]["lr"]})
                    touched.extend(["online_head_frozen", "new_head_residual_and_frozen_reference", "new_optimizer_group"])
                else:
                    with torch.no_grad():
                        state.q.head.weight.copy_(fresh_head.weight)
                        state.q.head.bias.copy_(fresh_head.bias)
                    touched.append("online_head_fresh_sample")
    if action in {"optimizer_reset", "head_and_optimizer_reset", "fresh_head_and_optimizer_reset"}:
        state.optimizer.state.clear()
        state._schedule_control = None
        touched.append("optimizer_memory")
    elif action == "t_reset":
        for value in state.optimizer.state.values():
            value["step"].zero_()
        state._schedule_control = None
        touched.append("optimizer_clock")
    elif action == "t_reset_equivalent_schedule":
        state._schedule_control = {"updates": 0, "base_groups": [
            {"lr": group["lr"], "eps": group["eps"]} for group in state.optimizer.param_groups]}
        touched.append("learning_rate_epsilon_schedule")
    state._check_bindings()
    receipt = {"action": action, "seed": int(seed), "touched": touched,
               "environment_steps": state.environment_steps, "gradient_updates": state.gradient_updates,
               "target_updated": False}
    state.repair_history.append(copy.deepcopy(receipt))
    return receipt


# Compatibility name for callers that spell out the replay role.
ReplayBuffer = Replay
