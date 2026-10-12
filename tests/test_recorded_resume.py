"""Дешёвые проверки durable decision и безопасного handoff; без torch."""
import unittest
from scripts.resume_recorded_research import make_plan, coverage
from scripts.handoff_recorded_queue import safe_boundary


class RecordedResumeTest(unittest.TestCase):
    def setUp(self):
        self.config = {"screen": {"games": ["breakout"], "seeds": [0], "horizon": 50000},
            "confirm": {"additional_seeds": [1], "additional_repeats": [4]},
            "main_menu": ["continue", "optimizer_reset"],
            "adaptive": {"development_extension_games": []}, "mechanisms": {"seeds": [0]}}
        self.manifest = {"decisions": [{"decision": "increase_resolution", "source_seeds": [1], "repeats": [4]}]}
        self.checkpoints = [{"checkpoint_id": "cp", "game": "breakout", "source_mode": "natural", "nominal_age": 50000}]
        self.plan = make_plan(self.config, self.manifest, self.checkpoints, {})

    def outcome(self, repeat, action):
        return {"source_mode": "natural", "game": "breakout", "training_seed": 1,
            "nominal_age": 50000, "repeat": repeat, "repair_id": action,
            "trajectory_id": "natural_breakout_seed1", "budget_env_steps": 50000,
            "completed_env_steps": 50000, "j_final": 1.0, "failure_reason": None}

    def test_recorded_expansion_survives_new_incomplete_gate(self):
        self.manifest["new_gate"] = "incomplete"
        again = make_plan(self.config, self.manifest, self.checkpoints, {})
        self.assertEqual(again["operations"], self.plan["operations"])
        self.assertEqual(again["operations"][0]["seeds"], [1])
        self.assertEqual(again["operations"][1]["seeds"], [0, 1])
        self.assertEqual(coverage(again, again["operations"][0], [])["missing"], 8)

    def test_partial_outcomes_resume_only_missing_work(self):
        op = self.plan["operations"][0]
        outcomes = [self.outcome(r, a) for r in op["repeats"] for a in self.plan["menu"]]
        self.assertFalse(coverage(self.plan, op, outcomes[:-1])["complete"])
        self.assertEqual(coverage(self.plan, op, outcomes[:-1])["missing"], 1)
        self.assertTrue(coverage(self.plan, op, outcomes)["complete"])
        with self.assertRaises(ValueError):
            coverage(self.plan, op, outcomes + [outcomes[0]])

    def test_numerical_failure_is_finished_not_infinite_retry(self):
        op = self.plan["operations"][0]
        outcomes = [self.outcome(r, a) for r in op["repeats"] for a in self.plan["menu"]]
        outcomes[0].update(failure_reason="FloatingPointError", completed_env_steps=10, j_final=None)
        result = coverage(self.plan, op, outcomes)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["numerical_failures"]), 1)

    def test_stale_complete_event_never_stops_an_active_job(self):
        progress = {"stage": "mechanisms", "event": "mechanisms_complete", "job_id": "mechanisms:cp:0"}
        manifest = {"jobs": {"mechanisms:cp:0": {"status": "running"}}}
        self.assertFalse(safe_boundary(self.config, manifest, progress, self.checkpoints, []))
        manifest["jobs"]["mechanisms:cp:0"]["status"] = "complete"
        self.assertTrue(safe_boundary(self.config, manifest, progress, self.checkpoints, []))
        self.assertFalse(safe_boundary(self.config, manifest, progress, self.checkpoints, [{"pid": 1}]))
        progress["stage"] = "lr_control"
        self.assertFalse(safe_boundary(self.config, manifest, progress, self.checkpoints, []))


if __name__ == "__main__":
    unittest.main()
