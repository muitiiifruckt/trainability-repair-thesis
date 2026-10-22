import copy
import tempfile
import unittest

import torch

from causal_trainability import two_state as ts
from causal_trainability.telemetry import Telemetry, TelemetryStore


class TwoStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = dict(ts.CONFIG)

    def test_telemetry_leaves_training_bit_identical(self):
        agent = ts.make_agent(0, self.cfg)
        twin = copy.deepcopy(agent)
        opt, twin_opt = ts.make_optimizer(agent, self.cfg), ts.make_optimizer(twin, self.cfg)
        with tempfile.TemporaryDirectory() as folder:
            store = TelemetryStore(folder, "probe")
            telemetry = Telemetry(folder, "probe", agent.layers(), opt, meta={"kind": "test"}, every=1, store=store)
            curve_on, _ = ts.train(agent, opt, ts.NEW_GOAL, 30, torch.Generator().manual_seed(7), self.cfg,
                                   telemetry=telemetry)
            store.close()
        curve_off, _ = ts.train(twin, twin_opt, ts.NEW_GOAL, 30, torch.Generator().manual_seed(7), self.cfg)
        self.assertEqual(curve_on, curve_off)
        for a, b in zip(agent.parameters(), twin.parameters()):
            self.assertTrue(torch.equal(a, b))
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(opt.state[a][key], twin_opt.state[b][key]))

    def test_exact_return_of_always_switch_policy(self):
        agent = ts.make_agent(0, self.cfg)
        with torch.no_grad():
            agent.body.weight.zero_()
            agent.body.bias.fill_(1.0)
            agent.actor.weight.zero_()
            agent.actor.bias.copy_(torch.tensor([-50.0, 50.0]))
        # Always switching from s1 reaches s2 at t=0 and leaves it next step: reward on even steps.
        expected = sum(self.cfg["gamma"] ** t for t in range(0, self.cfg["horizon"], 2))
        self.assertAlmostEqual(ts.exact_return(agent, ts.NEW_GOAL, self.cfg), expected, places=6)

    def test_norm_matching_keeps_dead_units_dead(self):
        fresh = ts.make_agent(1, self.cfg)
        dead = copy.deepcopy(fresh)
        with torch.no_grad():
            dead.body.bias.copy_(-(dead.body.weight.max(dim=1).values.clamp_min(0.0)) - 1.0)
        self.assertEqual(ts.dead_fraction(dead), 1.0)
        self.assertEqual(ts.dead_fraction(ts.matched_norm(dead, fresh)), 1.0)

    def test_on_policy_iteration_has_unit_importance_weight(self):
        agent = ts.make_agent(2, self.cfg)
        batch = ts.sample_batch(agent, ts.NEW_GOAL, torch.Generator().manual_seed(3), self.cfg)
        with torch.no_grad():
            logits, _ = agent(batch["states"])
            anchor = torch.log_softmax(logits, -1).gather(1, batch["actions"][:, None]).squeeze(1)
        self.assertTrue(torch.allclose(anchor, batch["behaviour_logp"]))


if __name__ == "__main__":
    unittest.main()
