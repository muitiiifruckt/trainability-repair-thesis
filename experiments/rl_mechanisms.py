"""Conditional learner probes on recorded MinAtar transitions.

Targets, datasets and minibatch tapes are fixed before any intervention. These
outcomes measure fitting of checkpoint bootstrap targets, not the correctness of
those targets or improvement in live RL. No collector or target sync runs here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rl_core import TrainingState, apply_repair, td_targets


MAIN_ACTIONS = ("continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset")
MECHANISM_ACTIONS = MAIN_ACTIONS + ("t_reset", "t_reset_equivalent_schedule", "fresh_network_reference")
DEFAULT_LR_GRID = (6.25e-5, 1.25e-4, 2.5e-4, 5e-4, 1e-3)
DEFAULT_BUDGETS = {
    "baseline_updates": 2000,
    "panel_updates": 500,
    "lr_updates": 500,
    "curve_points": (100, 500, 2000),
    "recent_window": 5000,
    "old_window": 5000,
    "heldout_min": 256,
    "heldout_max": 4096,
    "fit_eval_max": 4096,
    "evaluation_batch_size": 256,
}


class InsufficientProbeData(ValueError):
    """A probe cannot obtain its specified episode-disjoint replay split."""


@contextmanager
def _preserve_random_state():
    # restore() may restore source RNGs. Keep this read-only probe from advancing
    # or replacing the caller's global streams; probe sampling uses a local RNG.
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _mapping_hash(value: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(value):
        h.update(str(key).encode())
        item = value[key]
        if isinstance(item, torch.Tensor):
            h.update(_tensor_hash(item).encode())
        elif isinstance(item, Mapping):
            h.update(_mapping_hash(item).encode())
        else:
            h.update(json.dumps(_json_safe(item), sort_keys=True).encode())
    return h.hexdigest()


def _checkpoint_dict(checkpoint: Mapping | str | Path) -> dict:
    if isinstance(checkpoint, (str, Path)):
        # rl_core snapshots contain the native environment and NumPy arrays.
        # Paths accepted here are trusted locally generated experiment artifacts,
        # never author downloads or externally supplied pickle checkpoints.
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a snapshot mapping or a saved snapshot path")
    return dict(checkpoint)


def _budgets(overrides: Mapping | None = None) -> dict:
    result = dict(DEFAULT_BUDGETS)
    if overrides:
        unknown = set(overrides) - set(result)
        if unknown:
            raise ValueError(f"Unknown probe budgets: {sorted(unknown)}")
        result.update(overrides)
    for name in result:
        if name == "curve_points":
            result[name] = tuple(sorted({int(x) for x in result[name]}))
            if any(x < 0 for x in result[name]):
                raise ValueError("curve_points must be nonnegative")
        else:
            value = result[name]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            result[name] = int(value)
            minimum = 0 if name.endswith("_updates") else 1
            if result[name] < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
    if result["heldout_max"] < result["heldout_min"]:
        raise ValueError("heldout_max must be >= heldout_min")
    if result["recent_window"] != result["old_window"]:
        raise ValueError("The data factorial requires equal-sized recent and old windows")
    return result


def _sample_subset(indices: torch.Tensor, maximum: int, rng: torch.Generator) -> torch.Tensor:
    if len(indices) <= maximum:
        return indices.clone()
    return indices[torch.randperm(len(indices), generator=rng)[:maximum]].clone()


@dataclass
class FixedTDProbe:
    source: Any
    snapshot: dict
    budgets: dict
    seed: int
    train_indices: torch.Tensor
    recent_indices: torch.Tensor
    old_indices: torch.Tensor
    heldout_indices: torch.Tensor
    fit_indices: torch.Tensor
    targets: dict[str, torch.Tensor]
    metadata: dict


def prepare_fixed_td(checkpoint: Mapping | str | Path, seed: int = 0,
                     budgets: Mapping | None = None) -> FixedTDProbe:
    """Cache both target functions before repairs, with an episode-disjoint split.

    All episodes represented in the last recent_window + old_window transitions
    are excluded from heldout. Older replay rows are never silently randomized
    into a same-episode validation split. This is a temporal holdout, not IID.
    """
    settings = _budgets(budgets)
    snapshot = _checkpoint_dict(checkpoint)
    with _preserve_random_state():
        source = TrainingState.restore(snapshot)
        ordered = torch.as_tensor(source.replay.ordered_indices(), dtype=torch.long).cpu()
        window = settings["recent_window"] + settings["old_window"]
        if len(ordered) < window:
            raise InsufficientProbeData(f"Need {window} recent/old transitions; replay has {len(ordered)}")
        train_indices = ordered[-window:].clone()
        recent_indices = ordered[-settings["recent_window"]:].clone()
        old_indices = ordered[-window:-settings["recent_window"]].clone()
        episodes = torch.as_tensor(source.replay.episode_ids, dtype=torch.long).cpu()
        if bool((episodes[ordered] < 0).any()):
            raise InsufficientProbeData("Replay lacks valid episode IDs; no random-row fallback is allowed")
        excluded_episodes = torch.unique(episodes[train_indices])
        older = ordered[:-window]
        heldout = older[~torch.isin(episodes[older], excluded_episodes)]
        if len(heldout) < settings["heldout_min"]:
            raise InsufficientProbeData(
                f"Only {len(heldout)} transitions from episodes absent from the last {window}; "
                f"need {settings['heldout_min']}"
            )
        rng = torch.Generator().manual_seed(int(seed))
        heldout = _sample_subset(heldout, settings["heldout_max"], rng)
        fit = _sample_subset(train_indices, settings["fit_eval_max"], rng)
        capacity = int(source.replay.obs.shape[0])
        targets = {}
        source.q.eval()
        source.target.eval()
        with torch.no_grad():
            for teacher_name, model in (("old_target", source.target), ("refreshed_online", source.q)):
                values = torch.full((capacity,), float("nan"), dtype=next(model.parameters()).dtype)
                for indices in ordered.split(settings["evaluation_batch_size"]):
                    batch = source.replay.batch(indices)
                    y = td_targets(model, batch, source.config.gamma, source.config.reward_scale)
                    values[indices] = y.detach().cpu().reshape(-1)
                if not bool(torch.isfinite(values[ordered]).all()):
                    raise InsufficientProbeData(f"Nonfinite pre-repair TD targets: {teacher_name}")
                targets[teacher_name] = values
        ids = torch.as_tensor(source.replay.transition_ids, dtype=torch.long).cpu()
        discrepancy = targets["old_target"][ordered] - targets["refreshed_online"][ordered]
        metadata = {
            "split_kind": "temporal_episode_disjoint",
            "training_transitions": len(train_indices),
            "recent_transitions": len(recent_indices),
            "old_transitions": len(old_indices),
            "heldout_transitions": len(heldout),
            "fit_evaluation_transitions": len(fit),
            "training_episode_ids": torch.unique(episodes[train_indices]).tolist(),
            "heldout_episode_ids": torch.unique(episodes[heldout]).tolist(),
            "excluded_recent_episode_ids": excluded_episodes.tolist(),
            "train_transition_ids_hash": _tensor_hash(ids[train_indices]),
            "recent_transition_ids_hash": _tensor_hash(ids[recent_indices]),
            "old_transition_ids_hash": _tensor_hash(ids[old_indices]),
            "heldout_transition_ids_hash": _tensor_hash(ids[heldout]),
            "fit_transition_ids_hash": _tensor_hash(ids[fit]),
            "target_hashes": {name: _tensor_hash(values[ordered]) for name, values in targets.items()},
            "target_discrepancy_rms": float(discrepancy.square().mean().sqrt()),
            "target_functions_identical": bool(torch.equal(targets["old_target"][ordered], targets["refreshed_online"][ordered])),
            "target_semantics": "raw_reward * reward_scale + gamma * (1-terminal) * max_a Q_pre_repair(next_obs,a)",
            "teacher_privilege": False,
            "cache_forward_examples": 2 * len(ordered),
            "source_weights_hash": _mapping_hash(source.q.state_dict()),
            "source_target_hash": _mapping_hash(source.target.state_dict()),
        }
    return FixedTDProbe(source, snapshot, settings, int(seed), train_indices, recent_indices,
                        old_indices, heldout, fit, targets, metadata)


def make_training_tape(indices: torch.Tensor, updates: int, batch_size: int, seed: int) -> torch.Tensor:
    """Materialize physical replay indices; each repair consumes this same tape."""
    rng = torch.Generator().manual_seed(int(seed))
    positions = torch.randint(len(indices), (int(updates), int(batch_size)), generator=rng)
    return indices[positions].clone()


def _loss_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(predictions).all() and torch.isfinite(targets).all())
    return {
        "status": "ok" if finite else "nonfinite",
        "huber": float(F.smooth_l1_loss(predictions, targets)) if finite else None,
        "mse": float(F.mse_loss(predictions, targets)) if finite else None,
        "n": len(targets),
    }


def evaluate_fixed_td(state: Any, probe: FixedTDProbe, fit_indices: torch.Tensor | None = None) -> dict:
    """Evaluate each learner under BOTH pre-repair teachers on the common holdout."""
    fit_indices = probe.fit_indices if fit_indices is None else fit_indices
    result = {"forward_examples": 0}
    previous_mode = state.q.training
    state.q.eval()
    try:
        with torch.no_grad():
            for name, indices in (("fit", fit_indices), ("heldout", probe.heldout_indices)):
                pieces = []
                for batch_indices in indices.split(probe.budgets["evaluation_batch_size"]):
                    batch = probe.source.replay.batch(batch_indices)
                    pieces.append(state.q(batch["obs"]).gather(1, batch["actions"].long().reshape(-1, 1)).reshape(-1).cpu())
                predictions = torch.cat(pieces)
                result[name] = {teacher: _loss_metrics(predictions, y[indices]) for teacher, y in probe.targets.items()}
                result["forward_examples"] += len(indices)
    finally:
        state.q.train(previous_mode)
    return result


def _optimizer_clocks(state: Any) -> dict:
    values = [int(s["step"].item()) for s in state.optimizer.state.values() if "step" in s]
    return {"minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "initialized_parameters": len(values)}


def _run_fixed_branch(probe: FixedTDProbe, action: str, tape: torch.Tensor,
                      teacher: str = "old_target", lr: float | None = None,
                      curve_points: tuple[int, ...] = (), fit_indices: torch.Tensor | None = None) -> tuple[dict, Any]:
    state = TrainingState.restore(probe.snapshot)
    if action == "fresh_network_reference":
        state.q.load_state_dict(copy.deepcopy(state.initial_weights))
        state.optimizer.state.clear()
    else:
        apply_repair(state, action, seed=probe.seed)
    if lr is not None:
        for group in state.optimizer.param_groups:
            group["lr"] = float(lr)
    state.q.train()
    target_before = _mapping_hash(state.target.state_dict())
    optimizer_start = _optimizer_clocks(state)
    points = sorted({0, len(tape)} | {p for p in curve_points if 0 <= p <= len(tape)})
    curves = {"0": evaluate_fixed_td(state, probe, fit_indices)}
    evaluation_examples = curves["0"]["forward_examples"]
    completed, attempted, status, failure = 0, 0, "ok", None
    gradient_start = state.gradient_updates
    for step, indices in enumerate(tape, 1):
        batch = probe.source.replay.batch(indices)
        attempted = step
        try:
            outcome = state.learn_batch(batch, fixed_targets=probe.targets[teacher][indices], allow_target_sync=False)
        except FloatingPointError as exc:
            status, failure = "nonfinite", str(exc)
            completed = state.gradient_updates - gradient_start
            curves[str(completed)] = evaluate_fixed_td(state, probe, fit_indices)
            evaluation_examples += curves[str(completed)]["forward_examples"]
            break
        completed = state.gradient_updates - gradient_start
        loss = outcome.get("loss") if isinstance(outcome, dict) else outcome
        if loss is not None and not math.isfinite(float(loss)):
            status = "nonfinite"
        if step in points or status != "ok":
            curves[str(step)] = evaluate_fixed_td(state, probe, fit_indices)
            evaluation_examples += curves[str(step)]["forward_examples"]
        if status != "ok":
            break
    if _mapping_hash(state.target.state_dict()) != target_before:
        raise RuntimeError("Frozen TD probe unexpectedly changed the target network")
    final = curves[str(completed)]
    if any(m["status"] != "ok" for split in ("fit", "heldout") for m in final[split].values()):
        status = "nonfinite"
    result = {
        "action": action, "training_teacher": teacher, "learning_rate_override": lr,
        "status": status, "failure": failure, "requested_updates": len(tape),
        "total_updates": completed, "attempted_updates": attempted,
        "environment_interactions": 0, "tape_hash": _tensor_hash(tape), "curves": curves,
        "final": final, "optimizer_clock_before": optimizer_start,
        "optimizer_clock_after": _optimizer_clocks(state),
        "weights_hash_after": _mapping_hash(state.q.state_dict()),
        "target_hash_after": target_before, "evaluation_forward_examples": evaluation_examples,
    }
    return _json_safe(result), state


def _attach_common_gain(result: dict, before: dict) -> None:
    result["heldout_gain_from_common_pre_repair"] = {}
    result["heldout_gain_from_own_post_repair"] = {}
    for teacher in before["heldout"]:
        final = result["final"]["heldout"][teacher]
        own_start = result["curves"]["0"]["heldout"][teacher]
        for field, start in (("heldout_gain_from_common_pre_repair", before["heldout"][teacher]),
                             ("heldout_gain_from_own_post_repair", own_start)):
            result[field][teacher] = {
                metric: start[metric] - final[metric] if start[metric] is not None and final[metric] is not None else None
                for metric in ("huber", "mse")
            }


def _skip_result(reason: str, seed: int, settings: dict) -> dict:
    return {"status": "skipped", "reason": reason, "seed": int(seed), "config": _json_safe(settings),
            "total_updates": 0, "environment_interactions": 0,
            "outcomes": None, "limitations": ["No episode-overlapping or random-row fallback was used."]}


def run_mechanisms(checkpoint: Mapping | str | Path, seed: int = 0,
                   budgets: Mapping | None = None, lr_grid=DEFAULT_LR_GRID) -> dict:
    """Run frozen-TD repairs, LR sweep and a crossed data/target panel.

    Baseline curves default to 100/500/2000 updates. LR and data panels default
    to 500 updates. Counts include every intervention, with no hidden collector.
    """
    started = time.perf_counter()
    settings = _budgets(budgets)
    rates = tuple(float(lr) for lr in lr_grid)
    if any(not math.isfinite(lr) or lr <= 0 for lr in rates):
        raise ValueError("Every learning rate must be finite and positive")
    with _preserve_random_state():
        try:
            probe = prepare_fixed_td(checkpoint, seed, settings)
        except InsufficientProbeData as exc:
            return _skip_result(str(exc), seed, settings)
        batch_size = int(probe.source.config.batch_size)
        max_updates = max(settings["baseline_updates"], settings["lr_updates"])
        tape = make_training_tape(probe.train_indices, max_updates, batch_size, seed + 101)
        before = evaluate_fixed_td(probe.source, probe)
        baseline = []
        for action in MECHANISM_ACTIONS:
            result, _ = _run_fixed_branch(probe, action, tape[:settings["baseline_updates"]],
                                         curve_points=settings["curve_points"])
            baseline.append(result)
        sweep = []
        if settings["lr_updates"]:
            for rate in rates:
                for action in MAIN_ACTIONS:
                    result, _ = _run_fixed_branch(probe, action, tape[:settings["lr_updates"]], lr=rate)
                    sweep.append(result)
        panel = []
        # A common position tape maps to each equally-sized dataset. Within a
        # dataset, old/refreshed teacher arms see byte-identical physical rows.
        position_tape = make_training_tape(torch.arange(settings["recent_window"]),
                                          settings["panel_updates"], batch_size, seed + 211)
        rng = torch.Generator().manual_seed(seed + 307)
        for dataset, indices in (("recent", probe.recent_indices), ("old", probe.old_indices)):
            panel_tape = indices[position_tape]
            fit = _sample_subset(indices, settings["fit_eval_max"], rng)
            for teacher in probe.targets:
                result, _ = _run_fixed_branch(probe, "continue", panel_tape, teacher=teacher,
                                             curve_points=(100, 500), fit_indices=fit)
                result["dataset"] = dataset
                result["position_tape_hash"] = _tensor_hash(position_tape)
                panel.append(result)
        all_branches = baseline + sweep + panel
        for branch in all_branches:
            _attach_common_gain(branch, before)
        result = {
            "status": "ok" if all(x["status"] == "ok" for x in all_branches) else "nonfinite",
            "seed": int(seed), "config": settings, "model_config": probe.snapshot.get("config", {}),
            "lr_grid": rates, "split": probe.metadata, "pre_repair": before,
            "baseline": baseline, "lr_sweep": sweep, "data_target_panel": panel,
            "total_updates": sum(x["total_updates"] for x in all_branches),
            "attempted_updates": sum(x["attempted_updates"] for x in all_branches),
            "environment_interactions": 0,
            "forward_examples": probe.metadata["cache_forward_examples"] + before["forward_examples"] +
                                sum(x["evaluation_forward_examples"] for x in all_branches),
            "training_examples": sum(x["total_updates"] for x in all_branches) * batch_size,
            "wall_seconds": time.perf_counter() - started,
            "limitations": [
                "Bootstrap refresh is an intervention, not privileged ground truth.",
                "Replay windows differ in time; the older window is not guaranteed better coverage.",
                "Fitting/generalization of frozen targets does not establish live RL utility.",
                "fresh_network_reference reuses source initial weights; it is not a new initializer draw.",
            ],
        }
    return _json_safe(result)


def choose_by_probe(checkpoint: Mapping | str | Path, updates_per_action: int = 500,
                    seed: int = 0, budgets: Mapping | None = None) -> dict:
    """Rank four repairs by common heldout OLD-target Huber loss.

    This returns a decision only. The caller must restore the original repaired
    checkpoint for independent live continuation, charging all probe updates.
    Probe weights and replay/target state are never returned as a continuation.
    """
    if isinstance(updates_per_action, bool) or int(updates_per_action) != updates_per_action or updates_per_action <= 0:
        raise ValueError("updates_per_action must be a positive integer")
    settings = _budgets(budgets)
    started = time.perf_counter()
    with _preserve_random_state():
        try:
            probe = prepare_fixed_td(checkpoint, seed, settings)
        except InsufficientProbeData as exc:
            result = _skip_result(str(exc), seed, settings)
            result.update(chosen_action=None, scores={}, updates_per_action=int(updates_per_action))
            return result
        tape = make_training_tape(probe.train_indices, int(updates_per_action),
                                  int(probe.source.config.batch_size), seed + 101)
        branches = []
        for action in MAIN_ACTIONS:
            result, _ = _run_fixed_branch(probe, action, tape)
            branches.append(result)
        scores = {x["action"]: x["final"]["heldout"]["old_target"]["huber"]
                  if x["status"] == "ok" and x["total_updates"] == int(updates_per_action) else None
                  for x in branches}
        eligible = [x["action"] for x in branches if x["status"] == "ok" and scores[x["action"]] is not None]
        chosen = min(eligible, key=lambda a: scores[a]) if eligible else None
        result = {
            "status": "ok" if chosen is not None else "nonfinite", "chosen_action": chosen,
            "scores": scores, "seed": int(seed), "updates_per_action": int(updates_per_action),
            "ranking_metric": "episode_disjoint_heldout_old_target_huber",
            "tie_break": "predeclared_action_order", "total_updates": sum(x["total_updates"] for x in branches),
            "attempted_updates": sum(x["attempted_updates"] for x in branches),
            "environment_interactions": 0, "tape_hash": _tensor_hash(tape),
            "split": probe.metadata, "branches": branches,
            "forward_examples": probe.metadata["cache_forward_examples"] + sum(x["evaluation_forward_examples"] for x in branches),
            "training_examples": sum(x["total_updates"] for x in branches) * int(probe.source.config.batch_size),
            "restore_original_repaired_checkpoint": True,
            "reuse_probe_weights": False, "wall_seconds": time.perf_counter() - started,
            "limitations": ["Heldout frozen-TD fit can rank differently from independent live return."],
        }
    return _json_safe(result)
