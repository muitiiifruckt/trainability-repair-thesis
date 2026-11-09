import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from causal_trainability import probe, probe_v3 as v3
from causal_trainability.ppo_minatar import ActorCritic, CONFIG
from causal_trainability.telemetry import Telemetry, record_artifact


class ProbeV3Tests(unittest.TestCase):
    def test_late_window_and_head_correction(self):
        curve = [{"step": 0, "loss": 1.0}, {"step": 4, "loss": math.exp(-1)},
                 {"step": 6, "loss": math.exp(-2)}, {"step": 8, "loss": math.exp(-3)}]
        out = v3.outcomes(curve, 4)
        self.assertEqual(out["late_auc_log"], -2.5)
        self.assertEqual(out["late_gain_log"], 2.0)
        rows = []
        for h in range(6):
            runs = {}
            # A raw chain disadvantage of .1 disappears when its frozen disadvantage is removed.
            for body, value in (("fresh", -1.0), ("fresh_frozen", -0.1),
                                ("aged_actions_chain", -.9), ("aged_actions_chain_frozen", 0.0)):
                runs[body] = [{"seed": s, "late_auc_log": value} for s in range(3)]
            rows.append({"history": f"h{h}", "runs": runs})
        self.assertAlmostEqual(v3.compare(rows, "aged_actions_chain", "fresh")["mean_d"], 0.0)
        self.assertAlmostEqual(v3.compare(rows, "aged_actions_chain", "fresh", False)["mean_d"], .1)

    def test_same_scale_factor_applies_to_weights_and_bias_only(self):
        state = {"fc.weight": torch.ones(2, 3), "fc.bias": torch.ones(2), "conv.weight": torch.ones(3)}
        shrunk = v3.scaled_copy(state, .2)
        self.assertTrue(torch.equal(shrunk["fc.weight"], state["fc.weight"] * .2))
        self.assertTrue(torch.equal(shrunk["fc.bias"], state["fc.bias"] * .2))
        self.assertIs(shrunk["conv.weight"], state["conv.weight"])
        self.assertTrue(torch.equal(state["fc.weight"], torch.ones(2, 3)))

    def test_fit_telemetry_preserves_parameters_adam_and_rng(self):
        torch.set_num_threads(1)
        cfg = dict(CONFIG, input_channels=4, conv_channels=2, hidden=8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            state = ActorCritic(cfg).state_dict()
            states = torch.randint(2, (24, 4, 10, 10), dtype=torch.uint8)
            targets = torch.randn(24, 4)
            targets = (targets - targets.mean(0)) / targets.std(0, unbiased=False)
        spec = dict(v3.SPEC, steps=20, late_start=10, log_every=5, telemetry_every=10,
                    snapshot_steps=[0, 10, 20], outputs=4, batch=8, eval_chunk=12)
        results = []
        for enabled in (True, False):
            with tempfile.TemporaryDirectory() as folder:
                with torch.random.fork_rng(devices=[]):
                    model = probe.ProbeNet(cfg, state, 4)
                optimizer = torch.optim.Adam([p for group in model.layers().values() for p in group], lr=.001)
                log = Telemetry(folder, "test", model.layers(), optimizer, every=10, enabled=enabled, meta={})
                rng_before = torch.get_rng_state().clone()
                result = v3.fit(model, optimizer, states, targets, spec, 99, log, states[:8])
                log.close()
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                results.append((copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), result))
                if enabled:
                    artifacts = [json.loads(line) for line in (Path(folder) / "artifacts.jsonl").read_text().splitlines()]
                    self.assertEqual(len([row for row in artifacts if row["kind"] == "snapshot"]), 3)
        for key in results[0][0]:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))
        for i, state_a in results[0][1]["state"].items():
            for key, value in state_a.items():
                self.assertTrue(torch.equal(value, results[1][1]["state"][i][key]))
        self.assertEqual(results[0][2], results[1][2])

    def test_atomic_result_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            v3.atomic_json(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                v3.atomic_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})

    def test_completed_arm_seal_detects_artifact_and_result_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arm = root / "arms" / "test_t0"
            attempt = arm / "attempts" / "000"
            attempt.mkdir(parents=True)
            telemetry = attempt / "telemetry.jsonl"
            telemetry.write_text("{}\n")
            record_artifact(attempt, telemetry, "telemetry")
            snapshot = attempt / "test_u0.pt"
            snapshot.write_bytes(b"snapshot")
            record_artifact(attempt, snapshot, "snapshot")
            row = {"attempt": attempt.relative_to(root).as_posix()}
            v3.atomic_json(arm / "result.json", row)
            spec = {"snapshot_steps": [0]}
            v3.seal_arm(arm, row, spec, True)
            v3.verify_arm(arm, row, spec)
            snapshot.write_bytes(b"corruption")
            with self.assertRaises(ValueError):
                v3.verify_arm(arm, row, spec)
            snapshot.write_bytes(b"snapshot")
            (arm / "result.json").write_text("{}")
            with self.assertRaises(ValueError):
                v3.verify_arm(arm, row, spec)


if __name__ == "__main__":
    unittest.main()
