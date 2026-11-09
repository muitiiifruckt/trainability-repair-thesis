import copy
import tempfile
import unittest
from pathlib import Path

import torch

from causal_trainability.ppo_minatar import PPO
from causal_trainability.sprint03_live import intervene, jobs
from tests.test_causal_ppo import TINY


class InterventionsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ppo = PPO(dict(TINY), self.tmp.name)
        self.ppo.learn(self.ppo.rollout())

    def test_head_reset_keeps_body_adam(self):
        p = self.ppo
        body = copy.deepcopy(p.optimizer.state[p.model.fc.weight])
        intervene(p, p.theta0, "heads_reset", None, None)
        self.assertTrue(torch.equal(p.model.actor.weight, p.theta0["actor.weight"]))
        self.assertNotIn(p.model.actor.weight, p.optimizer.state)
        for key, tensor in body.items():
            self.assertTrue(torch.equal(tensor, p.optimizer.state[p.model.fc.weight][key]))

    def test_inverse_preserves_physical_distribution_and_roundtrip(self):
        p = self.ppo
        old, new = [2, 0, 3, 5, 1, 4], [3, 5, 1, 4, 2, 0]
        model, adam = copy.deepcopy(p.model.state_dict()), copy.deepcopy(p.optimizer.state_dict())
        obs = p.obs
        with torch.no_grad():
            before = p.model(obs)[0].softmax(-1)
        intervene(p, p.theta0, "inverse", old, new)
        with torch.no_grad():
            after = p.model(obs)[0].softmax(-1)
        self.assertTrue(torch.allclose(before[:, torch.argsort(torch.tensor(old))],
                                       after[:, torch.argsort(torch.tensor(new))], atol=1e-7))
        intervene(p, p.theta0, "inverse", new, old)
        for key, value in model.items():
            self.assertTrue(torch.equal(value, p.model.state_dict()[key]))
        for key, state in adam["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, p.optimizer.state_dict()["state"][key][name]))

    def test_inverse_update_equivariance(self):
        p = self.ppo
        q = PPO(dict(TINY), Path(self.tmp.name) / "other")
        q.model.load_state_dict(p.model.state_dict())
        q.optimizer.load_state_dict(copy.deepcopy(p.optimizer.state_dict()))
        old, new = [2, 0, 3, 5, 1, 4], [3, 5, 1, 4, 2, 0]
        batch = p.rollout()
        transformed = copy.deepcopy(batch)
        mapping = torch.tensor([new.index(old[a]) for a in range(6)])
        transformed["actions"] = mapping[batch["actions"]]
        intervene(q, q.theta0, "inverse", old, new)
        q.generator.set_state(p.generator.get_state())
        p.learn(batch)
        q.learn(transformed)
        intervene(q, q.theta0, "inverse", new, old)
        for a, b in zip(p.model.parameters(), q.model.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=2e-7, rtol=1e-5))
            for key in ("exp_avg", "exp_avg_sq", "step"):
                self.assertTrue(torch.allclose(p.optimizer.state[a][key], q.optimizer.state[b][key], atol=2e-7, rtol=1e-5))

    def test_panel_size_and_no_duplicate_jobs(self):
        self.assertEqual(len(jobs()), 108)
        self.assertEqual(len(set(jobs())), 108)


if __name__ == "__main__":
    unittest.main()
