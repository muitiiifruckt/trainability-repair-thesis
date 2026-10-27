import math
import tempfile
import unittest
from pathlib import Path

import torch

from causal_trainability import deficit, hybrids, probe
from tests.test_causal_ppo import train

SMALL = dict(probe.PROBE, data_envs=2, states_per_env=32, outputs=4, steps=20, batch=16, log_every=10, seeds=2,
             eval_chunk=16)


def chains(source, folder):
    settings = dict(deficit.CHAIN, rounds=1, steps=64, windows=1)
    for kind in deficit.SHIFTS:
        deficit.chain(Path(source), kind, Path(folder) / f"{kind}_chain", settings)


class ProbeTests(unittest.TestCase):
    def test_probe_starts_equal_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as runs, \
                tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            train(source, False)
            chains(source, runs)
            first = probe.run_history(Path(source), Path(runs), Path(a), SMALL)
            second = probe.run_history(Path(source), Path(runs), Path(b), SMALL)
            self.assertEqual(len(first["targets_sha256"]), SMALL["seeds"])
            for arm in probe.ARMS:
                for run, again in zip(first["runs"][arm], second["runs"][arm]):
                    # Zero head and standardised targets: every arm starts from loss 1 exactly.
                    self.assertAlmostEqual(run["curve"][0]["loss"], 1.0, places=5)
                    self.assertEqual(run["auc_log"], again["auc_log"])
                    self.assertTrue(math.isfinite(run["auc_log"]))
            # Arms differ only in where the body comes from.
            self.assertNotEqual(first["runs"]["fresh"][0]["auc_log"], first["runs"]["aged_1M"][0]["auc_log"])
            self.assertTrue((Path(a) / "telemetry" / f"probe_{first['history']}.jsonl").stat().st_size > 0)

    def test_bodies_come_from_their_sources(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as runs:
            train(source, False)
            chains(source, runs)
            cfg, starts, origin = probe.bodies(Path(source), Path(runs))
            aged = hybrids.load(source, "age1.00")
            end, _ = deficit.chain_end(Path(runs) / "actions_chain")
            for arm, expected in (("fresh", aged["theta0"]), ("aged_1M", aged["model"]),
                                  ("aged_actions_chain", end["model"])):
                net = probe.ProbeNet(cfg, starts[arm], 4)
                for name in ("conv.weight", "fc.weight", "fc.bias"):
                    self.assertTrue(torch.equal(net.body.state_dict()[name], expected[name]))
                self.assertEqual(float(net.head.weight.detach().abs().sum()), 0.0)
            self.assertIn("chain_end", origin["aged_actions_chain"])


class SummaryTests(unittest.TestCase):
    def results(self, differences, noise=0.01):
        rows = []
        for h, d in enumerate(differences):
            runs = {"fresh": [], probe.PRIMARY: []}
            for s in range(3):
                base = -1.0 + 0.001 * s
                runs["fresh"].append({"auc_log": base, "final_log": base})
                value = base + d + noise * (s - 1)
                runs[probe.PRIMARY].append({"auc_log": value, "final_log": value})
            rows.append({"history": f"h{h}", "arms": ["fresh", probe.PRIMARY], "runs": runs})
        return rows

    def test_verdicts(self):
        self.assertEqual(probe.summarize(self.results([0.5, 0.6, 0.4, 0.55, 0.45, 0.5]))["verdict"], "worse")
        self.assertEqual(probe.summarize(self.results([0.0, 0.01, -0.01, 0.02, -0.02, 0.0]))["verdict"],
                         "practically_not_worse")
        self.assertEqual(probe.summarize(self.results([0.5, -0.4, 0.3, -0.2, 0.6, -0.5]))["verdict"], "undetermined")
        # A difference below the within-history spread is not "worse", however consistent.
        self.assertNotEqual(probe.summarize(self.results([0.05] * 5 + [0.06], noise=0.2))["verdict"], "worse")
        with self.assertRaises(SystemExit):
            probe.summarize(self.results([0.5] * 5))


if __name__ == "__main__":
    unittest.main()
