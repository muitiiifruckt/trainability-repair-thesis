import copy
import tempfile
import unittest
from pathlib import Path

import torch

from causal_trainability import hybrids
from causal_trainability.ppo_minatar import CONFIG, PPO

TINY = dict(CONFIG, num_envs=2, num_steps=16, num_minibatches=2, update_epochs=1, total_steps=128,
            reference_size=16)


def train(folder, telemetry):
    torch.set_num_threads(1)
    ppo = PPO(dict(TINY), folder)
    ppo.train(telemetry_enabled=telemetry)
    return ppo


class PPOTests(unittest.TestCase):
    def test_telemetry_leaves_training_bit_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            on, off = train(a, True), train(b, False)
            for x, y in zip(on.model.parameters(), off.model.parameters()):
                self.assertTrue(torch.equal(x, y))
            for x, y in zip(on.model.parameters(), off.model.parameters()):
                self.assertTrue(torch.equal(on.optimizer.state[x]["exp_avg_sq"], off.optimizer.state[y]["exp_avg_sq"]))
            self.assertTrue((Path(a) / "telemetry").exists())
            # Closed telemetry and every L3 snapshot carry a sha256 in artifacts.jsonl.
            import hashlib, json
            rows = [json.loads(line) for line in (Path(a) / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["kind"] for r in rows}, {"telemetry", "snapshot"})
            for r in rows:
                self.assertEqual(hashlib.sha256((Path(a) / r["path"]).read_bytes()).hexdigest(), r["sha256"])
            from scripts.build_runs_index import build
            index = build(a)
            self.assertEqual(len(index), 1)
            self.assertTrue(index[0]["artifacts"])

    def test_hybrids(self):
        with tempfile.TemporaryDirectory() as folder:
            train(folder, False)
            fresh, aged = hybrids.load(folder, "age0.00"), hybrids.load(folder, "age1.00")
            all_aged = dict.fromkeys(hybrids.PARTS, "aged")
            model, optimizer, batch = hybrids.assemble(fresh, aged, all_aged)
            # H(aged x 4) reproduces the checkpoint bit for bit (weights and Adam state).
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, aged["model"][name]))
            saved = aged["optimizer"]["state"]
            for i, p in enumerate(model.parameters()):
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    self.assertTrue(torch.equal(optimizer.state[p][key], saved[i][key]))
            # Rebuilt targets reproduce the stored rollout (batched forward passes may differ in the last bits).
            self.assertTrue(torch.allclose(batch["advantages"], aged["rollout"]["advantages"], atol=1e-5))
            # Fresh Adam is empty; crossed parts come from the right sources.
            mixed = dict(params="fresh", adam="aged", data="fresh", targets="aged")
            model, optimizer, batch = hybrids.assemble(fresh, aged, mixed)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, fresh["model"][name]))
            self.assertTrue(torch.equal(batch["obs"], fresh["rollout"]["obs"]))
            first = next(iter(model.parameters()))
            self.assertIn("exp_avg", optimizer.state[first])
            _, empty, _ = hybrids.assemble(fresh, aged, dict(mixed, adam="fresh"))
            self.assertEqual(len(empty.state), 0)
            # A hybrid trains without errors and stays finite.
            logits, value = model(batch["obs"][:8])
            loss = -torch.log_softmax(logits, -1).mean() + ((value - batch["returns"][:8]) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


class CheckpointFromCLITests(unittest.TestCase):
    def test_checkpoint_written_by_subprocess_loads(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, "-m", "causal_trainability.ppo_minatar", "--steps", "2048",
                            "--out", folder], check=True, capture_output=True)
            checkpoint = hybrids.load(folder, "age1.00")
            self.assertEqual(type(checkpoint["envs"]).__module__, "causal_trainability.ppo_minatar")


if __name__ == "__main__":
    unittest.main()
