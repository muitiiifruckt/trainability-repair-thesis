import math
import tempfile
import unittest
from pathlib import Path

import torch

from causal_trainability import hybrids, lab
from causal_trainability.telemetry import TelemetryStore
from tests.test_causal_ppo import train


class LabTests(unittest.TestCase):
    def test_fresh_batches_continue_theta0_without_touching_global_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            train(folder, False)
            fresh = hybrids.load(folder, "age0.00")
            state = torch.random.get_rng_state()
            batches = lab.fresh_batches(fresh, 3)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
            self.assertIs(batches[0], fresh["rollout"])
            self.assertFalse(torch.equal(batches[1]["obs"], batches[0]["obs"]))
            # Same saved state -> same continuation.
            again = lab.fresh_batches(fresh, 3)
            self.assertTrue(torch.equal(again[2]["obs"], batches[2]["obs"]))
            self.assertTrue(torch.equal(again[2]["actions"], batches[2]["actions"]))

    def test_lab_runs_on_all_hybrids(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as out:
            train(folder, False)
            store = TelemetryStore(out, "lab_test")
            config = dict(lab.LAB, batches=2)
            history, results, digests = lab.run_history(folder, Path(out), store, config)
            self.assertEqual(len(digests), 2)
            store.close()
            self.assertEqual(set(results), set(lab.HYBRIDS))
            for combo in lab.HYBRIDS:
                for kind in lab.MEASURES:
                    self.assertEqual(len(results[combo][kind]), 2)
                    for row in results[combo][kind]:
                        self.assertTrue(all(math.isfinite(v) for point in row["curve"] for v in point.values()))
            # The start of a measurement depends on the parameters only, not on the Adam state.
            for b in range(2):
                for kind in lab.MEASURES:
                    self.assertEqual(results["FFFF"][kind][b]["start"], results["FAFF"][kind][b]["start"])
                    self.assertEqual(results["AFFF"][kind][b]["start"], results["AAFF"][kind][b]["start"])
            # The value loss splits exactly into bias and variance parts.
            for row in results["AAFF"]["value"]:
                for point in row["curve"]:
                    self.assertAlmostEqual(point["objective"], point["bias_part"] + point["variance_part"], places=5)
            # Fresh params on theta0 data: importance weights are 1 and nothing is clipped.
            self.assertEqual(results["FFFF"]["policy"][0]["is_clipped_fraction"], 0.0)
            summary, verdict = lab.summarize({history: results}, config)
            self.assertEqual(set(verdict), {"value_params", "value_adam", "policy_params", "policy_adam"})
            self.assertTrue((Path(out) / "snapshots" / "lab_test.pt").exists())
            self.assertTrue((Path(out) / "telemetry" / "lab_test.jsonl").stat().st_size > 0)
            # Lab telemetry rows carry the reference-batch metrics too.
            import json
            row = json.loads((Path(out) / "telemetry" / "lab_test.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("dead_fraction", row["layers"]["fc"])
            self.assertIn("effective_rank", row["layers"]["fc"])


if __name__ == "__main__":
    unittest.main()
