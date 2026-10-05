"""Scientific validity checks for branch cloning and Adam reset controls."""
import copy
import unittest

import torch

from experiments.controlled_probe import Model, apply_action, equivalent_schedule, new_optimizer, restore, snapshot, train_step


class BranchSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(13)
        self.model = Model()
        self.initial = copy.deepcopy(self.model.state_dict())
        self.optimizer = new_optimizer(self.model, eps=0.01)
        self.x = torch.randn(16, 8, dtype=torch.float64)
        self.y = torch.randn(16, 1, dtype=torch.float64)
        for _ in range(12):
            train_step(self.model, self.optimizer, self.x, self.y)
        self.checkpoint = snapshot(self.model, self.optimizer, self.initial, torch.Generator())

    def assert_models_equal(self, first, second, atol=0):
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=atol)

    def test_clone_has_independent_parameter_and_moment_storage(self):
        clone, opt = restore(self.checkpoint)
        first = next(clone.parameters())
        original = next(self.model.parameters())
        self.assertNotEqual(first.data_ptr(), original.data_ptr())
        self.assertNotEqual(opt.state[first]["exp_avg"].data_ptr(), self.optimizer.state[original]["exp_avg"].data_ptr())
        opt.state[first]["exp_avg"].zero_()
        self.assertGreater(float(self.optimizer.state[original]["exp_avg"].abs().sum()), 0)

    def test_optimizer_reset_preserves_predictions(self):
        clone, opt = restore(self.checkpoint)
        before = clone(self.x).detach().clone()
        apply_action(clone, opt, self.checkpoint, "optimizer_reset")
        torch.testing.assert_close(clone(self.x), before, rtol=0, atol=0)
        self.assertEqual(len(opt.state), 0)

    def test_head_reset_keeps_optimizer_parameter_bindings(self):
        clone, opt = restore(self.checkpoint)
        identities = {id(p) for group in opt.param_groups for p in group["params"]}
        apply_action(clone, opt, self.checkpoint, "head_reset")
        self.assertEqual(identities, {id(p) for p in clone.parameters()})
        torch.testing.assert_close(clone.head.weight, self.initial["head.weight"], rtol=0, atol=0)

    def test_continue_clones_follow_identical_trajectory(self):
        a, oa = restore(self.checkpoint)
        b, ob = restore(self.checkpoint)
        for _ in range(5):
            train_step(a, oa, self.x, self.y)
            train_step(b, ob, self.x, self.y)
        self.assert_models_equal(a, b)

    def test_full_reset_matches_fresh_optimizer(self):
        a, oa = restore(self.checkpoint)
        b, _ = restore(self.checkpoint)
        apply_action(a, oa, self.checkpoint, "optimizer_reset")
        ob = new_optimizer(b, eps=0.01)
        for _ in range(5):
            train_step(a, oa, self.x, self.y)
            train_step(b, ob, self.x, self.y)
        self.assert_models_equal(a, b)

    def test_t_reset_matches_learning_rate_and_epsilon_schedule(self):
        a, oa = restore(self.checkpoint)
        b, ob = restore(self.checkpoint)
        base = [{"lr": g["lr"], "eps": g["eps"]} for g in ob.param_groups]
        apply_action(a, oa, self.checkpoint, "t_reset")
        for step in range(1, 11):
            equivalent_schedule(ob, step, base)
            train_step(a, oa, self.x, self.y * (step % 3 + 1))
            train_step(b, ob, self.x, self.y * (step % 3 + 1))
            self.assert_models_equal(a, b, atol=2e-12)
        self.assertEqual(int(oa.state[next(a.parameters())]["step"]), 10)
        self.assertEqual(int(ob.state[next(b.parameters())]["step"]), 22)


if __name__ == "__main__":
    unittest.main()
