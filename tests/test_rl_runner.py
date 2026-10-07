"""Resumption, atomic outcome deduplication and package accounting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rl_core import TrainingState
from experiments.rl_runner import ROOT, Campaign, MAIN, dump, load_state, rows


class RunnerTests(unittest.TestCase):
    def setUp(self):
        (ROOT/"runs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="runner_test_",dir=ROOT/"runs")
        self.folder = Path(self.temporary.name)
        self.config_path = self.folder/"test_config.json"
        config = json.loads((ROOT/"configs/research_program.json").read_text(encoding="utf-8"))
        config.update(program_id="runner_test",output_root=str(self.folder.relative_to(ROOT)))
        config["learner"].update(conv_channels=2,hidden_size=8,batch_size=4,replay_capacity=32,
                                 warmup=8,target_period=4,episode_cap=7,dtype="float64")
        config["screen"].update(games=["breakout"],seeds=[0],ages=[12,20],minimum_target_age=2,
                                repeats=[0],horizon=8,curve_steps=[0,3,8],evaluation_episodes=1)
        config["resources"].update(updates_per_package=100000,resume_every_steps=2,bootstrap_repeats=10)
        dump(self.config_path,config)
        self.campaign = Campaign(self.config_path,resume=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_source_and_branch_resume_deduplicate_and_keep_target_age(self):
        first = self.campaign.source("breakout",0,[12,20])
        self.assertTrue(all(item["target_update_age"] >= 2 for item in first))
        checkpoint = first[0]
        outcome = self.campaign.branch(checkpoint,"continue",0)
        next_campaign = Campaign(self.config_path,resume=True)
        self.assertEqual(next_campaign.source("breakout",0,[12,20]),first)
        self.assertEqual(next_campaign.branch(checkpoint,"continue",0)["j_final"],outcome["j_final"])
        self.assertEqual(len(rows(self.folder/"outcomes.jsonl")),1)
        self.assertEqual(len(rows(self.folder/"diagnostics.jsonl")),2)
        self.assertEqual(next_campaign.package["updates"],self.campaign.package["updates"])

    def test_interrupted_branch_restores_trace_and_same_seeds(self):
        checkpoint = self.campaign.source("breakout",0,[12])[0]
        original_step = TrainingState.step
        counter = [0]
        def interrupted(state):
            counter[0] += 1
            if counter[0] == 5:
                raise RuntimeError("simulated infrastructure interruption")
            return original_step(state)
        with patch.object(TrainingState,"step",interrupted):
            with self.assertRaisesRegex(RuntimeError,"simulated"):
                self.campaign.branch(checkpoint,"optimizer_reset",0)
        self.assertEqual(rows(self.folder/"outcomes.jsonl"),[])
        resumed = Campaign(self.config_path,resume=True).branch(checkpoint,"optimizer_reset",0)
        with tempfile.TemporaryDirectory(prefix="runner_reference_",dir=ROOT/"runs") as reference:
            reference_config = json.loads(self.config_path.read_text(encoding="utf-8"))
            reference_config["output_root"] = str(Path(reference).relative_to(ROOT))
            reference_path = Path(reference)/"config_reference.json"
            dump(reference_path,reference_config)
            metadata = copy.deepcopy(checkpoint)
            metadata["artifact_path"] = str((self.folder/checkpoint["artifact_path"]).resolve())
            expected = Campaign(reference_path,resume=True).branch(metadata,"optimizer_reset",0)
        for field in ("j_final","j_immediate","evaluation_curve","continuation_seed","repair_seed","budget_gradient_updates"):
            self.assertEqual(resumed[field],expected[field])

    def test_package_rotation_and_restart_preserve_prior_cost(self):
        self.campaign.config["resources"]["updates_per_package"] = 10
        self.campaign.package["updates"] = 9
        self.campaign.persist()
        restarted = Campaign(self.config_path,resume=True)
        self.assertEqual(restarted.package["updates"],9)
        restarted.config["resources"]["updates_per_package"] = 10
        with patch.object(restarted,"report",return_value={}):
            restarted.reserve(2)
        self.assertEqual(restarted.package["updates"],0)
        self.assertEqual(restarted.manifest["packages"][0]["updates"],9)
        with self.assertRaisesRegex(ValueError,"exceeds"):
            restarted.reserve(11)

    def test_checksum_and_config_change_block_reuse(self):
        checkpoint = self.campaign.source("breakout",0,[12])[0]
        broken = dict(checkpoint,sha256="bad")
        with self.assertRaisesRegex(ValueError,"checksum"):
            self.campaign.branch(broken,"continue",0)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["learner"]["lr"] *= 2
        dump(self.config_path,config)
        with self.assertRaisesRegex(ValueError,"Changed campaign config"):
            Campaign(self.config_path,resume=True)


if __name__ == "__main__":
    unittest.main()
