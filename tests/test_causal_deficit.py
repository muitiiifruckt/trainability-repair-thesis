import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from minatar import Environment

from causal_trainability import deficit, hybrids
from causal_trainability.ppo_minatar import PPO, env_action, observation, resolve
from tests.test_causal_ppo import TINY, train

LONGER = dict(TINY, total_steps=256)  # 8 iterations: the 0.25 checkpoint is taken at iteration 2


def run(cfg, folder):
    torch.set_num_threads(1)
    ppo = PPO(dict(cfg), folder)
    ppo.train(telemetry_enabled=False)
    return ppo


class ResumeAndShiftTests(unittest.TestCase):
    def test_resume_from_checkpoint_is_bit_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            whole = run(LONGER, a)
            resumed = PPO(dict(LONGER), b)
            resumed.restore(hybrids.load(a, "age0.25"))
            resumed.train(telemetry_enabled=False, resume=True)
            for x, y in zip(whole.model.parameters(), resumed.model.parameters()):
                self.assertTrue(torch.equal(x, y))
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    self.assertTrue(torch.equal(whole.optimizer.state[x][key], resumed.optimizer.state[y][key]))
            self.assertEqual(whole.global_step, resumed.global_step)

    def test_identity_shift_changes_nothing(self):
        cfg = resolve(dict(TINY))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
            plain = run(cfg, a)
            channels = run(dict(cfg, shift={"kind": "channels", "perm": list(range(cfg["input_channels"]))}), b)
            actions = run(dict(cfg, shift={"kind": "actions", "perm": list(range(cfg["num_actions"]))}), c)
            for x, y, z in zip(plain.model.parameters(), channels.model.parameters(), actions.model.parameters()):
                self.assertTrue(torch.equal(x, y) and torch.equal(x, z))

    def test_shift_is_applied(self):
        cfg = resolve(dict(TINY))
        perm = deficit.derangement(cfg["input_channels"], ("test",))
        self.assertFalse(any(i == p for i, p in enumerate(perm)))
        self.assertEqual(perm, deficit.derangement(cfg["input_channels"], ("test",)))
        env = Environment(cfg["game"])
        env.seed(3)
        env.reset()
        for _ in range(5):
            env.act(3)
        native = np.asarray(env.state()).transpose(2, 0, 1)
        shifted = observation(env, dict(cfg, shift={"kind": "channels", "perm": perm}))
        self.assertTrue(np.array_equal(shifted[:len(perm)], native[perm]))
        moves = deficit.derangement(cfg["num_actions"], ("test-actions",))
        shifted_cfg = dict(cfg, shift={"kind": "actions", "perm": moves})
        self.assertEqual([env_action(shifted_cfg, a) for a in range(cfg["num_actions"])], moves)


class BranchTests(unittest.TestCase):
    def test_aged_and_fresh_branches_start_right_and_are_paired(self):
        settings = dict(deficit.BRANCH, steps=128, windows=2, eval_episodes=2)
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as out:
            train(source, False)
            fresh, aged = hybrids.load(source, "age0.00"), hybrids.load(source, "age1.00")
            results = {arm: deficit.branch(Path(source), arm, "actions", 0, Path(out) / arm, settings)
                       for arm in deficit.ARMS}
            for arm, start in (("aged", aged), ("fresh", fresh)):
                first = hybrids.load(Path(out) / arm, "age0.00")  # branch checkpoint after its first rollout
                for name, tensor in first["model"].items():
                    self.assertTrue(torch.equal(tensor, start["model"][name]))
                steps = [s["step"] for s in first["optimizer"]["state"].values()]
                if arm == "aged":
                    self.assertTrue(all(torch.equal(s, aged["optimizer"]["state"][0]["step"]) for s in steps))
                else:
                    self.assertEqual(steps, [])
            # Same environment seeds: both arms see the same first observations.
            a, f = (hybrids.load(Path(out) / arm, "age0.00") for arm in ("aged", "fresh"))
            self.assertTrue(torch.equal(a["rollout"]["obs"][: TINY["num_envs"]], f["rollout"]["obs"][: TINY["num_envs"]]))
            for arm, result in results.items():
                saved = json.loads((Path(out) / arm / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["shift"]["kind"], "actions")
                self.assertEqual(len(saved["auc_windows"]), 2)
                self.assertEqual(saved["config"]["run_kind"], "deficit_branch")


class AnalysisTests(unittest.TestCase):
    def write(self, root, deficits, drop=None, perm=(1, 0)):
        for h, d in enumerate(deficits):
            for rep in (0, 1):
                for arm, value in (("aged", 5.0 - d), ("fresh", 5.0)):
                    if drop == (h, rep, arm):
                        continue
                    folder = Path(root) / f"h{h}" / "channels_r1" / f"{arm}_rep{rep}"
                    folder.mkdir(parents=True)
                    row = {"history": f"h{h}", "arm": arm, "repeat": rep,
                           "shift": {"kind": "channels", "perm": list(perm)}, "config": {"run_name": arm, "lr": 1},
                           "revision": {"files": {}}, "seeds": {"env": rep}, "auc": value + 0.01 * rep,
                           "auc_windows": [value], "final_return": value, "start_return": 0.0}
                    (folder / "result.json").write_text(json.dumps(row), encoding="utf-8")

    def test_verdicts(self):
        from scripts.analyze_deficit import analyze
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
            self.write(a, [1.0, 1.1, 0.9, 1.2, 1.0, 0.8])
            self.assertEqual(analyze(a)["shifts"]["channels"]["verdict"], "deficit")
            self.write(b, [0.01, -0.02, 0.0, 0.02, -0.01, 0.01])
            self.assertEqual(analyze(b)["shifts"]["channels"]["verdict"], "practically_none")
            self.write(c, [2.0, -1.5, 1.0, -0.5, 1.5, -1.0])
            self.assertEqual(analyze(c)["shifts"]["channels"]["verdict"], "undetermined")

    def test_incomplete_or_unpaired_data_is_refused(self):
        from scripts.analyze_deficit import analyze
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.write(a, [1.0] * 6, drop=(2, 1, "fresh"))
            with self.assertRaises(SystemExit):
                analyze(a)
            self.write(b, [1.0] * 5)
            with self.assertRaises(SystemExit):
                analyze(b)


if __name__ == "__main__":
    unittest.main()
