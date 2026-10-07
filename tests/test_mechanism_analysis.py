"""Analysis preserves history grouping and explicit failed fitting outcomes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.mechanism_analysis import _history_stats, _fixed_points, summarize_mechanisms


class MechanismAnalysisTests(unittest.TestCase):
    def test_histories_and_checkpoint_ages_receive_equal_weight(self):
        records = [dict(trajectory_id="a",checkpoint_id="young",value=0) for _ in range(8)]
        records += [dict(trajectory_id="a",checkpoint_id="old",value=10)]
        records += [dict(trajectory_id="b",checkpoint_id="young",value=20)]
        stat = _history_stats(records)
        self.assertEqual(stat["source_histories"],2)
        self.assertEqual(stat["mean"],12.5)
        self.assertEqual(stat["history_min"],5)

    def test_failed_curves_remain_raw_and_never_count_as_success(self):
        job = dict(checkpoint_id="cp",trajectory_id="history",game="breakout",repeat=0,
                   source_mode="natural",model_config={"lr":.00025})
        point = {split:{teacher:{"huber":2.,"mse":4.,"status":"ok"}
                 for teacher in ("old_target","refreshed_online")} for split in ("fit","heldout")}
        branch = dict(action="head_reset",total_updates=1,status="nonfinite",failure="gradient",
                      curves={"1":point},tape_hash="common")
        results = _fixed_points(job,branch,"baseline")
        self.assertEqual(len(results),8)
        self.assertTrue(all(row["value"] is None and row["raw_value"] in (2,4) for row in results))

    def test_empty_data_is_explicit_and_figures_are_generated_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/"input"; run.mkdir()
            config = run/"config.json"
            config.write_text(json.dumps({"mechanisms":{"seeds":[0,1]}}),encoding="utf-8")
            before = config.read_bytes()
            result = summarize_mechanisms(run,root/"output")
            self.assertEqual(result["coverage"]["fixed_probe_files"],0)
            self.assertEqual(result["costs"]["analysis_optimizer_updates"],0)
            self.assertFalse(result["live_return"]["analyzed"])
            self.assertEqual(config.read_bytes(),before)
            self.assertTrue((root/"output/mechanism_summary.json").exists())
            self.assertTrue((root/"output/mechanism_synthetic_lr.png").exists())


if __name__ == "__main__":
    unittest.main()
