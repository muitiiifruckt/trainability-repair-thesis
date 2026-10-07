"""Validity tests for recorded-transition probes, rather than RL performance."""
import copy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rl_core import DQNConfig, TrainingState
from experiments.rl_mechanisms import (
    InsufficientProbeData, MAIN_ACTIONS, _run_fixed_branch, choose_by_probe,
    evaluate_fixed_td, make_training_tape, prepare_fixed_td, run_mechanisms,
)


SMALL_BUDGETS = {
    "baseline_updates": 5, "panel_updates": 3, "lr_updates": 3,
    "curve_points": (1, 3, 5), "recent_window": 4, "old_window": 4,
    "heldout_min": 4, "heldout_max": 8, "fit_eval_max": 8,
    "evaluation_batch_size": 4,
}


class FrozenTDMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        config = DQNConfig(conv_channels=2, hidden_size=8, replay_capacity=64,
                           warmup=8, batch_size=4, dtype="float64", target_period=2,
                           reward_scale=0.25, adam_eps=0.01)
        self.state = TrainingState.create("breakout", seed=19, config=config)
        rng = np.random.RandomState(29)
        for i in range(40):
            obs = (rng.rand(10, 10, 10) < 0.1).astype(np.uint8)
            next_obs = (rng.rand(10, 10, 10) < 0.1).astype(np.uint8)
            self.state.replay.add(obs, next_obs, i % 6, float(i % 3 - 1),
                                  i % 4 == 3, i // 4, 100 + i)
        for step in range(8):
            batch = self.state.replay.batch(np.arange(step, step + 4))
            self.state.learn_batch(batch, allow_target_sync=False)
        self.snapshot = self.state.snapshot()

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def probe(self, seed=31):
        return prepare_fixed_td(self.snapshot, seed=seed, budgets=SMALL_BUDGETS)

    def test_episode_disjoint_holdout_and_ordered_windows(self):
        probe = self.probe()
        episodes = torch.as_tensor(probe.source.replay.episode_ids)
        training_episodes = set(episodes[probe.train_indices].tolist())
        heldout_episodes = set(episodes[probe.heldout_indices].tolist())
        self.assertFalse(training_episodes & heldout_episodes)
        torch.testing.assert_close(probe.recent_indices, torch.arange(36, 40))
        torch.testing.assert_close(probe.old_indices, torch.arange(32, 36))
        self.assertTrue(set(probe.fit_indices.tolist()) <= set(probe.train_indices.tolist()))
        self.assertEqual(probe.metadata["split_kind"], "temporal_episode_disjoint")

    def test_wraparound_windows_use_transition_chronology(self):
        for i in range(40, 80):
            obs = np.zeros((10, 10, 10), dtype=np.uint8)
            self.state.replay.add(obs, obs, 0, 0, i % 4 == 3, i // 4, 100 + i)
        probe = prepare_fixed_td(self.state.snapshot(), seed=31, budgets=SMALL_BUDGETS)
        ids = torch.as_tensor(probe.source.replay.transition_ids)
        torch.testing.assert_close(ids[probe.recent_indices], torch.arange(176, 180))
        torch.testing.assert_close(ids[probe.old_indices], torch.arange(172, 176))

    def test_target_cache_uses_raw_rewards_and_pre_repair_models(self):
        probe = self.probe()
        indices = torch.arange(40)
        batch = probe.source.replay.batch(indices)
        with torch.no_grad():
            for name, model in (("old_target", probe.source.target),
                                ("refreshed_online", probe.source.q)):
                expected = batch["rewards"] * self.state.config.reward_scale
                alive = ~batch["terminals"]
                expected[alive] += self.state.config.gamma * model(batch["next_obs"][alive]).max(1).values
                torch.testing.assert_close(probe.targets[name][indices], expected, rtol=0, atol=0)
        self.assertFalse(probe.metadata["target_functions_identical"])
        old = {k: v.clone() for k, v in probe.targets.items()}
        with torch.no_grad():
            for p in probe.source.q.parameters():
                p.add_(5)
            for p in probe.source.target.parameters():
                p.sub_(3)
        for name in old:
            torch.testing.assert_close(probe.targets[name], old[name], equal_nan=True, rtol=0, atol=0)

    def test_fixed_tape_no_collector_no_sync_and_no_snapshot_mutation(self):
        original = {name: copy.deepcopy(self.snapshot[name]) for name in
                    ("weights", "target", "optimizer", "replay", "initial_weights")}
        probe = self.probe()
        tape = make_training_tape(probe.train_indices, 6, 4, 53)
        first, a = _run_fixed_branch(probe, "continue", tape, curve_points=(1, 3, 6))
        second, b = _run_fixed_branch(probe, "continue", tape, curve_points=(1, 3, 6))
        self.assertEqual(first["tape_hash"], second["tape_hash"])
        self.assertEqual(first["curves"], second["curves"])
        for pa, pb in zip(a.q.parameters(), b.q.parameters(), strict=True):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        self.assertEqual(a.environment_steps, self.snapshot["environment_steps"])
        self.assertEqual(a.last_target_update, self.snapshot["last_target_update"])
        self.assertEqual(a.replay.size, self.snapshot["replay"]["size"])
        self.assertEqual(first["environment_interactions"], 0)
        self.assertEqual(first["total_updates"], 6)
        for name in original:
            self.assert_nested_equal(self.snapshot[name], original[name])
        self.assert_nested_equal(a.target.state_dict(), self.snapshot["target"])

    def test_t_reset_matches_exact_lr_epsilon_schedule(self):
        probe = self.probe()
        tape = make_training_tape(probe.train_indices, 6, 4, 57)
        reset_result, reset = _run_fixed_branch(probe, "t_reset", tape)
        control_result, control = _run_fixed_branch(probe, "t_reset_equivalent_schedule", tape)
        for a, b in zip(reset.q.parameters(), control.q.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=2e-11)
        self.assertEqual(reset_result["optimizer_clock_after"]["minimum"], 6)
        self.assertEqual(control_result["optimizer_clock_after"]["minimum"], 14)
        self.assertAlmostEqual(reset_result["final"]["heldout"]["old_target"]["huber"],
                               control_result["final"]["heldout"]["old_target"]["huber"], places=12)

    def test_factorial_shared_tapes_costs_and_strict_json(self):
        result = run_mechanisms(self.snapshot, seed=31, budgets=SMALL_BUDGETS, lr_grid=(1e-4, 3e-4))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_updates"], 7 * 5 + 4 * 2 * 3 + 4 * 3)
        self.assertEqual(result["environment_interactions"], 0)
        self.assertEqual(len({x["tape_hash"] for x in result["baseline"]}), 1)
        self.assertEqual(len({x["tape_hash"] for x in result["lr_sweep"]}), 1)
        panel = result["data_target_panel"]
        self.assertEqual(len({x["position_tape_hash"] for x in panel}), 1)
        for dataset in ("recent", "old"):
            arms = [x for x in panel if x["dataset"] == dataset]
            self.assertEqual(len({x["tape_hash"] for x in arms}), 1)
            self.assertEqual({x["training_teacher"] for x in arms}, {"old_target", "refreshed_online"})
            self.assertEqual(arms[0]["curves"]["0"], arms[1]["curves"]["0"])
        for arm in result["baseline"]:
            self.assertEqual(set(arm["curves"]), {"0", "1", "3", "5"})
            self.assertIn("heldout_gain_from_common_pre_repair", arm)
        json.dumps(result, allow_nan=False)

    def test_probe_chooser_counts_all_actions_and_preserves_global_rng(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        result = choose_by_probe(self.snapshot, updates_per_action=3, seed=37, budgets=SMALL_BUDGETS)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_updates"], 12)
        self.assertEqual(result["environment_interactions"], 0)
        self.assertEqual(set(result["scores"]), set(MAIN_ACTIONS))
        self.assertEqual(result["chosen_action"], min(MAIN_ACTIONS, key=lambda a: result["scores"][a]))
        self.assertFalse(result["reuse_probe_weights"])
        self.assertTrue(result["restore_original_repaired_checkpoint"])
        self.assertEqual(random.getstate(), py_before)
        self.assert_nested_equal(np.random.get_state(), np_before)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_before, rtol=0, atol=0)
        json.dumps(result, allow_nan=False)

    def test_insufficient_episode_holdout_explicitly_skips(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["replay"]["episode_ids"] = np.zeros(40, dtype=np.int64)
        with self.assertRaises(InsufficientProbeData):
            prepare_fixed_td(snapshot, seed=31, budgets=SMALL_BUDGETS)
        result = run_mechanisms(snapshot, seed=31, budgets=SMALL_BUDGETS, lr_grid=())
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["total_updates"], 0)
        chooser = choose_by_probe(snapshot, updates_per_action=3, seed=37, budgets=SMALL_BUDGETS)
        self.assertIsNone(chooser["chosen_action"])
        json.dumps(chooser, allow_nan=False)

    def test_no_hidden_collector_and_nonfinite_failures_are_serializable(self):
        with patch.object(TrainingState, "step", side_effect=AssertionError("collector must never run")):
            result = choose_by_probe(self.snapshot, updates_per_action=2, seed=37, budgets=SMALL_BUDGETS)
            self.assertEqual(result["status"], "ok")
        with patch.object(TrainingState, "learn_batch", side_effect=FloatingPointError("test failure")):
            result = choose_by_probe(self.snapshot, updates_per_action=2, seed=37, budgets=SMALL_BUDGETS)
            self.assertEqual(result["status"], "nonfinite")
            self.assertIsNone(result["chosen_action"])
            self.assertEqual(result["attempted_updates"], 4)
            self.assertEqual(result["total_updates"], 0)
            json.dumps(result, allow_nan=False)

    def test_heldout_metrics_not_averaged_with_training_fit(self):
        probe = self.probe()
        metrics = evaluate_fixed_td(probe.source, probe)
        self.assertEqual(metrics["fit"]["old_target"]["n"], len(probe.fit_indices))
        self.assertEqual(metrics["heldout"]["old_target"]["n"], len(probe.heldout_indices))
        self.assertNotIn("loss", metrics)


if __name__ == "__main__":
    unittest.main()
