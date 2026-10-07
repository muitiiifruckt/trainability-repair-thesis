"""Checkpoint, intervention and target semantics for the local MinAtar learner."""
import copy
import unittest

import numpy as np
import torch

from experiments.rl_core import DQNConfig, Replay, TrainingState, td_targets


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **changes):
        values = dict(conv_channels=2, hidden_size=8, replay_capacity=32, warmup=8,
                      batch_size=4, target_period=3, epsilon_decay=20, episode_cap=7,
                      dtype="float64")
        values.update(changes)
        return DQNConfig(**values)

    def state(self, **changes):
        return TrainingState.create("breakout", 17, self.config(**changes))

    def warmed(self, **changes):
        state = self.state(**changes)
        for _ in range(12):
            state.step()
        return state

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, np.random.RandomState):
            self.assert_tree_equal(a.get_state(), b.get_state())
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_tree_equal(left, right)
        elif hasattr(a, "__dict__"):
            self.assertEqual(type(a), type(b))
            self.assert_tree_equal(vars(a), vars(b))
        else:
            self.assertEqual(a, b)

    def assert_networks_equal(self, a, b, atol=0):
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        for name in a.state_dict():
            torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], rtol=0, atol=atol)

    def test_full_checkpoint_reproduces_environment_replay_and_learning_trace(self):
        original = self.warmed()
        checkpoint = original.snapshot()
        restored = TrainingState.restore(checkpoint)
        self.assertIs(restored.env.random, restored.env.env.random)
        for _ in range(25):
            self.assert_tree_equal(original.step(), restored.step())
            self.assert_tree_equal(original.last_sample_indices, restored.last_sample_indices)
            self.assert_tree_equal(original.env.state(), restored.env.state())
            self.assert_networks_equal(original.q, restored.q)
            self.assert_networks_equal(original.target, restored.target)
        self.assert_tree_equal(original.snapshot(), restored.snapshot())
        # Ring wrap happened; chronological IDs survive a restore.
        ids = original.replay.transition_ids[original.replay.ordered_indices()]
        np.testing.assert_array_equal(ids, np.arange(original.environment_steps - original.replay.size + 1,
                                                     original.environment_steps + 1))

    def test_snapshot_and_forks_own_independent_storage(self):
        state = self.warmed()
        checkpoint = state.snapshot()
        a, b = TrainingState.restore(checkpoint), TrainingState.restore(checkpoint)
        self.assertFalse(checkpoint["replay"]["obs"].flags.writeable)
        self.assertFalse(np.shares_memory(a.replay.obs, b.replay.obs))
        self.assertNotEqual(a.q.head.weight.data_ptr(), b.q.head.weight.data_ptr())
        a_param, b_param = next(a.q.parameters()), next(b.q.parameters())
        self.assertNotEqual(a.optimizer.state[a_param]["exp_avg"].data_ptr(),
                            b.optimizer.state[b_param]["exp_avg"].data_ptr())
        with torch.no_grad():
            a.q.head.weight.add_(3)
        a.optimizer.state[a_param]["exp_avg"].zero_()
        a.replay.obs.fill(0)
        a.env.act(4)
        self.assert_tree_equal(state.snapshot(), checkpoint)
        self.assert_tree_equal(b.snapshot(), checkpoint)

    def test_seeded_forks_change_rngs_but_preserve_current_world_and_clocks(self):
        checkpoint = self.warmed().snapshot()
        a = TrainingState.restore(checkpoint, continuation_seed=99)
        b = TrainingState.restore(checkpoint, continuation_seed=99)
        c = TrainingState.restore(checkpoint, continuation_seed=100)
        self.assert_tree_equal(a.observation, checkpoint["observation"])
        self.assertEqual(a.environment_steps, checkpoint["environment_steps"])
        self.assertEqual(a.env.last_action, checkpoint["environment"].last_action)
        self.assertFalse(np.array_equal(a.exploration_rng.get_state()[1], c.exploration_rng.get_state()[1]))
        self.assertFalse(np.array_equal(a.replay_rng.get_state()[1], c.replay_rng.get_state()[1]))
        for _ in range(8):
            self.assert_tree_equal(a.step(), b.step())

    def test_optimizer_reset_preserves_weights_targets_replay_and_schedules(self):
        state = self.warmed()
        checkpoint = state.snapshot()
        epsilon = state.exploration_epsilon()
        state.apply_repair("optimizer_reset")
        self.assertEqual(len(state.optimizer.state), 0)
        self.assert_tree_equal(state.q.state_dict(), checkpoint["weights"])
        self.assert_tree_equal(state.target.state_dict(), checkpoint["target"])
        self.assert_tree_equal(state.replay.snapshot(), checkpoint["replay"])
        self.assertEqual(state.exploration_epsilon(), epsilon)
        self.assertEqual(state.gradient_updates, checkpoint["gradient_updates"])
        self.assertEqual(state.last_target_update, checkpoint["last_target_update"])

    def test_head_reset_preserves_parameter_bindings_optimizer_and_target(self):
        state = self.warmed()
        original_ids = [id(p) for p in state.q.parameters()]
        old_optimizer = copy.deepcopy(state.optimizer.state_dict())
        target = copy.deepcopy(state.target.state_dict())
        encoder = state.q.conv.weight.detach().clone()
        state.apply_repair("head_reset")
        self.assertEqual(original_ids, [id(p) for p in state.q.parameters()])
        torch.testing.assert_close(state.q.head.weight, state.initial_weights["head.weight"], rtol=0, atol=0)
        torch.testing.assert_close(state.q.conv.weight, encoder, rtol=0, atol=0)
        self.assert_tree_equal(state.optimizer.state_dict(), old_optimizer)
        self.assert_tree_equal(state.target.state_dict(), target)
        state._check_bindings()

    def test_full_reset_matches_fresh_adam_and_preserves_global_target_phase(self):
        checkpoint = self.warmed().snapshot()
        a, b = TrainingState.restore(checkpoint), TrainingState.restore(checkpoint)
        a.apply_repair("optimizer_reset")
        b.optimizer = torch.optim.Adam(b.q.parameters(), lr=b.config.lr,
                                       betas=b.config.adam_betas, eps=b.config.adam_eps,
                                       foreach=False, fused=False)
        batch = a.replay.batch(np.arange(4))
        remaining = a.config.target_period - (a.gradient_updates - a.last_target_update)
        for update in range(1, remaining + 1):
            result = a.learn_batch(batch)
            b.learn_batch(batch)
            self.assert_networks_equal(a.q, b.q)
            self.assertEqual(result["target_synced"], update == remaining)
        self.assertEqual(a.last_target_update, checkpoint["last_target_update"] + a.config.target_period)

    def test_t_reset_matches_lr_epsilon_control_in_float64_through_target_sync(self):
        checkpoint = self.warmed(adam_eps=0.01).snapshot()
        a, b = TrainingState.restore(checkpoint), TrainingState.restore(checkpoint)
        a.apply_repair("t_reset")
        b.apply_repair("t_reset_equivalent_schedule")
        rng = np.random.RandomState(211)
        for update in range(1, 10):
            indices = a.replay.sample_indices(rng, 4)
            batch = a.replay.batch(indices)
            fixed = td_targets(a.target, batch, a.config.gamma) * (1 + update % 3)
            a.learn_batch(batch, fixed_targets=fixed)
            b.learn_batch(batch, fixed_targets=fixed)
            self.assert_networks_equal(a.q, b.q, atol=2e-12)
            self.assert_networks_equal(a.target, b.target, atol=2e-12)
            self.assertEqual(a.last_target_update, b.last_target_update)
        a_step = int(a.optimizer.state[next(a.q.parameters())]["step"].item())
        b_step = int(b.optimizer.state[next(b.q.parameters())]["step"].item())
        self.assertEqual(a_step, 9)
        self.assertEqual(b_step, 9 + int(checkpoint["optimizer"]["state"][0]["step"].item()))

    def test_terminal_mask_and_raw_rewards_with_truncation_bootstrap(self):
        class ConstantCritic(torch.nn.Module):
            def forward(self, x):
                return torch.full((len(x), 6), 2.0, dtype=x.dtype)
        state = self.state(episode_cap=1, replay_ratio=0, reward_scale=3)
        result = state.step()
        self.assertTrue(result["truncated"])
        self.assertFalse(state.replay.terminals[0])
        self.assertEqual(state.replay.rewards[0], result["raw_reward"])
        batch = state.replay.batch([0])
        expected = result["raw_reward"] * 3 + state.config.gamma * 2
        torch.testing.assert_close(td_targets(ConstantCritic(), batch, state.config.gamma, 3),
                                   torch.tensor([expected], dtype=torch.float64))
        batch["terminals"].fill_(True)
        batch["next_obs"].fill_(float("nan"))
        class MustNotRun(torch.nn.Module):
            def forward(self, x):
                raise AssertionError("Terminal critic should never run")
        torch.testing.assert_close(td_targets(MustNotRun(), batch, state.config.gamma, 3), batch["rewards"] * 3)

    def test_evaluation_and_diagnostics_do_not_mutate_training_or_global_rng(self):
        state = self.warmed()
        checkpoint = state.snapshot()
        torch_rng = torch.random.get_rng_state().clone()
        numpy_rng = copy.deepcopy(np.random.get_state())
        evaluation = state.evaluate([42, 43])
        diagnostics = state.diagnostics(batch_size=8, seed=44)
        self.assertEqual(len(evaluation["returns"]), 2)
        self.assertEqual(diagnostics["diagnostic_environment_steps"], 0)
        self.assertEqual(diagnostics["diagnostic_inputs"], 8)
        self.assert_tree_equal(state.snapshot(), checkpoint)
        self.assert_tree_equal(torch.random.get_rng_state(), torch_rng)
        self.assert_tree_equal(np.random.get_state(), numpy_rng)

    def test_restore_and_repairs_preserve_callers_global_torch_rng(self):
        checkpoint = self.warmed().snapshot()
        rng = torch.random.get_rng_state().clone()
        state = TrainingState.restore(checkpoint)
        state.apply_repair("fresh_head_reset", repair_seed=81)
        self.assert_tree_equal(torch.random.get_rng_state(), rng)
        one = state.q.head.weight.detach().clone()
        other = TrainingState.restore(checkpoint)
        other.apply_repair("fresh_head_reset", repair_seed=81)
        torch.testing.assert_close(other.q.head.weight, one, rtol=0, atol=0)

    def test_fractional_and_high_replay_ratios_count_actual_updates(self):
        slow = self.state(replay_ratio=0.5, warmup=4)
        fast = self.state(replay_ratio=2, warmup=4)
        for _ in range(7):
            slow.step()
            fast.step()
        self.assertEqual(slow.gradient_updates, 2)
        self.assertEqual(fast.gradient_updates, 8)
        clone = TrainingState.restore(slow.snapshot())
        self.assert_tree_equal(slow.step(), clone.step())

    def test_function_preserving_injection_has_fresh_optimizer_group_and_restores(self):
        state = self.warmed()
        batch = state.replay.batch(np.arange(4))
        before = state.q(batch["obs"]).detach()
        target = copy.deepcopy(state.target.state_dict())
        state.apply_repair("injection", repair_seed=9)
        torch.testing.assert_close(state.q(batch["obs"]), before, rtol=0, atol=1e-15)
        self.assertFalse(state.q.head.weight.requires_grad)
        self.assertTrue(state.q.injection_head.weight.requires_grad)
        self.assertFalse(state.q.injection_reference.weight.requires_grad)
        self.assertEqual(len(state.optimizer.param_groups), 2)
        self.assertEqual(len(state.optimizer.state[state.q.injection_head.weight]), 0)
        self.assert_tree_equal(state.target.state_dict(), target)
        clone = TrainingState.restore(state.snapshot())
        for _ in range(4):
            state.learn_batch(batch)
            clone.learn_batch(batch)
            self.assert_networks_equal(state.q, clone.q)
            self.assert_networks_equal(state.target, clone.target)
        self.assertTrue(state.target.injected)

    def test_replay_batch_returns_owned_tensors_and_preserves_chronology(self):
        replay = Replay(self.config(replay_capacity=8, warmup=4))
        obs = np.ones((10, 10, 10), dtype=np.uint8)
        for i in range(13):
            replay.add(obs, obs, i % 6, i, False, i // 3, i)
        np.testing.assert_array_equal(replay.transition_ids[replay.ordered_indices()], np.arange(5, 13))
        batch = replay.batch([0, 1])
        batch["obs"].zero_()
        self.assertEqual(int(replay.obs[0].sum()), 1000)
        self.assertEqual(batch["obs"].dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
