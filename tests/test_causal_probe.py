import math
import tempfile
import unittest
from pathlib import Path

import torch

from causal_trainability import deficit, hybrids, probe
from tests.test_causal_ppo import train

SMALL = dict(probe.PROBE, data_envs=2, states_per_env=32, outputs=4, steps=20, batch=16, log_every=10,
             snapshot_steps=[0, 10, 20], seeds=2, eval_chunk=16)


def chains(source, folder):
    settings = dict(deficit.CHAIN, rounds=1, steps=64, windows=1)
    for kind in deficit.SHIFTS:
        deficit.chain(Path(source), kind, Path(folder) / f"{kind}_chain", settings)


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.source, cls.runs = root / "source", root / "runs"
        train(cls.source, False)
        chains(cls.source, cls.runs)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_probe_starts_equal_is_deterministic_and_telemetry_free(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = probe.run_history(self.source, self.runs, Path(a), SMALL)
            silent = probe.run_history(self.source, self.runs, Path(b), SMALL, telemetry=False)
            names = [probe.run_name(arm, mode) for arm, mode in probe.RUNS]
            self.assertEqual(sorted(first["runs"]), sorted(names))
            for name in names:
                for run, again in zip(first["runs"][name], silent["runs"][name]):
                    # Zero head and standardised labels: every run starts from loss 1 exactly.
                    self.assertAlmostEqual(run["curve"][0]["loss"], 1.0, places=5)
                    self.assertEqual(run["auc_log"], again["auc_log"])  # telemetry does not touch training
                    self.assertTrue(math.isfinite(run["auc_log"]))
                    moved = run["distance_moved"]
                    if name.endswith("_frozen"):
                        self.assertEqual((moved["conv"], moved["fc"]), (0.0, 0.0))
                    else:
                        self.assertGreater(min(moved["conv"], moved["fc"]), 0.0)
                    self.assertGreater(moved["head"], 0.0)
            self.assertTrue((Path(a) / "telemetry" / f"probe_{first['history']}.jsonl").stat().st_size > 0)
            self.assertEqual(len(first["seeds"]["targets"]), SMALL["seeds"])
            # Scale-matched bodies have the fresh mean feature norm.
            fresh_norm = first["runs"]["fresh"][0]["start"]["feature_norm"]
            for body in probe.SCALED:
                self.assertAlmostEqual(first["runs"][f"{body}_scaled"][0]["start"]["feature_norm"], fresh_norm, places=4)

    def test_bodies_come_from_their_sources(self):
        cfg, starts, origin = probe.bodies(self.source, self.runs)
        aged = hybrids.load(self.source, "age1.00")
        ends = {kind: deficit.chain_end(self.runs / f"{kind}_chain")[0] for kind in deficit.SHIFTS}
        for arm, expected in (("fresh", aged["theta0"]), ("aged_1M", aged["model"]),
                              ("aged_actions_chain", ends["actions"]["model"]),
                              ("aged_channels_chain", ends["channels"]["model"])):
            net = probe.ProbeNet(cfg, starts[arm], 4)
            for name in ("conv.weight", "fc.weight", "fc.bias"):
                self.assertTrue(torch.equal(net.body.state_dict()[name], expected[name]))
            self.assertEqual(float(net.head.weight.detach().abs().sum()), 0.0)
            trained = {id(p) for group in net.layers().values() for p in group}
            old_heads = {id(p) for p in list(net.body.actor.parameters()) + list(net.body.critic.parameters())}
            self.assertFalse(trained & old_heads)  # the old actor and critic are never optimised
        states = probe.collect_states(starts["aged_1M"], cfg, "test", SMALL)
        scaled, c = probe.rescaled(starts["aged_1M"], cfg, states, 2.0, SMALL["eval_chunk"])
        self.assertTrue(torch.allclose(scaled["fc.weight"], starts["aged_1M"]["fc.weight"] * c))
        norm = probe.hidden(probe.ProbeNet(cfg, scaled, 4), states, SMALL["eval_chunk"]).norm(dim=1).mean()
        self.assertAlmostEqual(float(norm), 2.0, places=4)
        self.assertIn("chain_end", origin["aged_channels_chain"])


class SummaryTests(unittest.TestCase):
    def results(self, chain, one_m, noise=0.01, scale=0.3):
        """Per-history auc_log differences of the chain body against fresh and of the 1M body against fresh."""
        rows = []
        for h, (c, m) in enumerate(zip(chain, one_m)):
            runs = {}
            for s in range(3):
                base = -0.3 + 0.001 * s
                values = {"fresh": base, "fresh_frozen": base + scale, "aged_1M": base + m,
                          "aged_actions_chain": base + c + noise * (s - 1)}
                values["aged_1M_scaled"], values["aged_actions_chain_scaled"] = values["aged_1M"], values["aged_actions_chain"]
                for name, v in values.items():
                    runs.setdefault(name, []).append({"seed": s, "auc_log": v, "final_log": v})
            rows.append({"history": f"h{h}", "runs": runs})
        return rows

    def test_verdicts_and_reading(self):
        big = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13]  # r about 0.36
        s = probe.summarize(self.results(big, [0.0] * 6))
        self.assertEqual(s["comparisons"]["chain_vs_fresh"]["verdict"], "worse")
        self.assertEqual(s["comparisons"]["chain_vs_1M"]["verdict"], "worse")
        self.assertEqual(s["reading"], "body_slower_specific_to_chain")
        s = probe.summarize(self.results(big, big))
        self.assertEqual(s["comparisons"]["chain_vs_1M"]["verdict"], "practically_not_worse")
        self.assertEqual(s["reading"], "body_slower_not_specific_to_chain")
        # The chain body is worse than fresh; against 1M the evidence is mixed: specificity is not shown either way.
        s = probe.summarize(self.results(big, [0.2, -0.15, 0.25, -0.2, 0.3, -0.1]))
        self.assertEqual(s["comparisons"]["chain_vs_1M"]["verdict"], "undetermined")
        self.assertEqual(s["reading"], "body_slower_specificity_undetermined")
        small = [0.02, 0.025, 0.021, 0.022, 0.024, 0.023]  # significant but r about 0.08
        s = probe.summarize(self.results(small, [0.0] * 6, noise=0.001))
        self.assertEqual(s["comparisons"]["chain_vs_fresh"]["verdict"], "worse_but_small")
        self.assertEqual(s["reading"], "body_slower_but_small")
        neg = [-x for x in big]
        s = probe.summarize(self.results(neg, [0.0] * 6))
        self.assertEqual(s["comparisons"]["chain_vs_fresh"]["verdict"], "better")
        self.assertEqual(s["reading"], "body_not_slower")
        mixed = [0.3, -0.25, 0.2, -0.1, 0.35, -0.3]
        self.assertEqual(probe.summarize(self.results(mixed, [0.0] * 6))["comparisons"]["chain_vs_fresh"]["verdict"],
                         "undetermined")
        # A difference below the within-history spread is not "worse", however consistent.
        self.assertNotEqual(probe.summarize(self.results([0.10] * 5 + [0.11], [0.0] * 6, noise=0.3))
                            ["comparisons"]["chain_vs_fresh"]["verdict"], "worse")
        with self.assertRaises(SystemExit):
            probe.summarize(self.results(big[:5], [0.0] * 5))


if __name__ == "__main__":
    unittest.main()
