import copy
import tempfile
import unittest

import torch

from causal_trainability import two_state as ts
from causal_trainability import two_state_v2 as v2
from causal_trainability.telemetry import Telemetry, TelemetryStore


class FeedbackControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = dict(v2.CONFIG)

    def test_first_data_and_update_identical_in_live_and_frozen(self):
        fresh = ts.make_agent(0, self.cfg)
        aged, opt = ts.build(fresh, None, self.cfg)
        batch = ts.sample_batch(aged, ts.OLD_GOAL, torch.Generator().manual_seed(7), self.cfg)
        ts.ppo_iteration(aged, opt, batch, self.cfg)
        for repair in v2.MENU:
            live, live_opt = v2.repaired(fresh, aged, opt, repair, self.cfg)
            frozen, frozen_opt = v2.repaired(fresh, aged, opt, repair, self.cfg)
            behavior = copy.deepcopy(frozen)
            live_batch = ts.sample_batch(live, ts.NEW_GOAL, torch.Generator().manual_seed(11), self.cfg)
            frozen_batch = ts.sample_batch(behavior, ts.NEW_GOAL, torch.Generator().manual_seed(11), self.cfg)
            for key in live_batch:
                self.assertTrue(torch.equal(live_batch[key], frozen_batch[key]), key)
            live_curve, _ = v2.trajectory(live, live_opt, ts.NEW_GOAL, 1, 11, self.cfg)
            frozen_curve, _ = v2.trajectory(frozen, frozen_opt, ts.NEW_GOAL, 1, 11, self.cfg, frozen=behavior)
            self.assertEqual(live_curve, frozen_curve)
            for p, q in zip(live.parameters(), frozen.parameters()):
                self.assertTrue(torch.equal(p, q))
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(live_opt.state[p][key], frozen_opt.state[q][key]))

    def test_telemetry_does_not_change_learning(self):
        fresh = ts.make_agent(1, self.cfg)
        a, ao = ts.build(fresh, None, self.cfg)
        b, bo = ts.build(fresh, None, self.cfg)
        with tempfile.TemporaryDirectory() as folder:
            store = TelemetryStore(folder, "test")
            telemetry = Telemetry(folder, "test", a.layers(), ao, meta={}, store=store)
            ca, _ = v2.trajectory(a, ao, ts.NEW_GOAL, 10, 12, self.cfg, telemetry, snapshot_every=5)
            store.close()
        cb, _ = v2.trajectory(b, bo, ts.NEW_GOAL, 10, 12, self.cfg)
        self.assertEqual(ca, cb)
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p, q))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(ao.state[p][key], bo.state[q][key]))

    def panel(self, opposite=False):
        histories = []
        for h in range(8):
            rows = []
            for mode in v2.MODES:
                for repair in v2.MENU:
                    best = "optimizer_reset" if opposite and mode == "frozen" else "continue"
                    for repeat in range(4):
                        rows.append(dict(mode=mode, repair=repair, repeat=repeat,
                                         final_return=2.0 if repair == best else 0.0))
            histories.append(dict(history=h, source_converged=True, rows=rows))
        return histories

    def test_equal_decision_is_not_counterexample(self):
        result = v2.summarize(self.panel(), self.cfg)
        self.assertEqual(result["verdict"], "not_demonstrated")
        self.assertEqual(result["heldout_regret"]["mean"], 0)

    def test_opposite_decision_passes_and_test_repeats_cannot_change_choices(self):
        panel = self.panel(opposite=True)
        result = v2.summarize(panel, self.cfg)
        self.assertEqual(result["verdict"], "feedback_decision_change")
        for history in panel:
            for row in history["rows"]:
                if row["mode"] == "live" and row["repeat"] >= 2:
                    row["final_return"] = -row["final_return"]
        altered = v2.summarize(panel, self.cfg)
        self.assertEqual([x["choices"] for x in result["per_history"]],
                         [x["choices"] for x in altered["per_history"]])
        self.assertEqual(altered["verdict"], "not_demonstrated")

    def test_incomplete_and_duplicate_panels_rejected(self):
        self.assertEqual(v2.summarize(self.panel()[:4], self.cfg)["verdict"], "incomplete")
        panel = self.panel()
        panel[0]["rows"].append(panel[0]["rows"][0])
        with self.assertRaises(ValueError):
            v2.summarize(panel, self.cfg)


if __name__ == "__main__":
    unittest.main()
