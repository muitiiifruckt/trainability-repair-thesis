import copy
import tempfile
import unittest
from pathlib import Path

from scripts.recover_nul_control_files import reconstruct, require_nul


class NulControlRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.config = {"program_id": "p", "main_menu": ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"],
                       "screen": {"horizon": 50000}, "learner": {"replay_ratio": 1}}
        self.checkpoint = {"checkpoint_id": "natural_asterix_seed4_age50000", "trajectory_id": "natural_asterix_seed4",
                           "source_mode": "natural", "sha256": "verified", "gradient_updates": 45500}
        prefix = "branch:natural:natural_asterix_seed4_age50000:"
        self.progress = {"job_id": prefix + "head_and_optimizer_reset:repeat2:h50000", "checkpoint_id": self.checkpoint["checkpoint_id"],
                         "repeat": 2, "horizon": 50000, "completed_steps": 29051, "gradient_updates": 74551,
                         "package_updates": 179051, "package_wall_sec": 1400., "time": 2000.,
                         "event": "branch_complete", "j_final": 1.15}
        self.outcomes = [{"job_id": prefix + action + ":repeat2:h50000", "repeat": 2, "completed_env_steps": 50000,
                          "budget_env_steps": 50000, "budget_gradient_updates": 50000, "trajectory_id": self.checkpoint["trajectory_id"],
                          "source_mode": "natural", "checkpoint_sha256": "verified", "j_final": 1., "failure_reason": None}
                         for action in self.config["main_menu"][:-1]]

    def rebuild(self, outcomes=None, progress=None, ledger=None):
        return reconstruct(self.config, {"time": 600., "code": {}}, self.checkpoint,
                           progress or self.progress, self.outcomes if outcomes is None else outcomes,
                           {"pid": 123, "task_id": "branch:stable"}, ledger or {"accounted_updates": 0})

    def test_only_raw_completed_jobs_restored_and_old_work_retained(self):
        result = self.rebuild()
        self.assertEqual(result["active_package"]["updates"], 179051)
        self.assertEqual(len(result["jobs"]), 3)
        self.assertNotIn(self.progress["job_id"], result["jobs"])
        self.assertEqual(result["scheduler_task_id"], "branch:stable")
        self.assertEqual(result["recovery_provenance"]["unfinished_rule"],
                         "restart only this branch from the verified source checkpoint")

    def test_missing_duplicate_or_other_repeat_cannot_be_completed(self):
        variants = [self.outcomes[:-1], self.outcomes + [self.outcomes[0]]]
        other = copy.deepcopy(self.outcomes)
        other[0]["repeat"] = 3
        variants.append(other)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.rebuild(outcomes=variant)

    def test_partial_nonfinite_or_failed_outcome_cannot_be_completed(self):
        for field, value in [("completed_env_steps", 49999), ("j_final", float("nan")),
                             ("failure_reason", "OSError: lost write"), ("checkpoint_sha256", "different")]:
            variant = copy.deepcopy(self.outcomes)
            variant[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.rebuild(outcomes=variant)

    def test_counter_or_source_clock_disagreement_is_rejected(self):
        for field in ("package_updates", "gradient_updates"):
            variant = dict(self.progress)
            variant[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.rebuild(progress=variant)
        with self.assertRaises(ValueError):
            self.rebuild(ledger={"accounted_updates": 200000})

    def test_valid_file_is_never_replaced_as_nul_damage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_bytes(b"\0" * 211)
            self.assertEqual(require_nul(path)["size"], 211)
            for value in (b"{}", b"", b"\0{}"):
                path.write_bytes(value)
                with self.assertRaises(ValueError):
                    require_nul(path)


if __name__ == "__main__":
    unittest.main()
