"""Дешёвые проверки durable decision и безопасного handoff; без torch."""
import unittest
from scripts.resume_recorded_research import make_plan, coverage, validate_plan, ensure_no_live_child_leases
import copy
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
from scripts import handoff_recorded_queue as handoff_module
from scripts.handoff_recorded_queue import safe_boundary


class RecordedResumeTest(unittest.TestCase):
    def setUp(self):
        self.config = {"screen": {"games": ["breakout"], "seeds": [0], "ages": [50000], "horizon": 50000},
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
            "budget_gradient_updates": 50000, "probe_gradient_updates": 0,
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

    def test_infrastructure_error_is_not_a_numerical_outcome(self):
        op = self.plan["operations"][0]
        row = self.outcome(0, "continue")
        row["failure_reason"] = "MemoryError: saving replay"
        with self.assertRaises(ValueError):
            coverage(self.plan, op, [row])

    def test_saved_plan_cannot_change_operations_or_lose_its_decision(self):
        validate_plan(self.plan, self.config, self.manifest, self.checkpoints)
        altered = copy.deepcopy(self.plan)
        altered["operations"][0]["seeds"] = [0]
        with self.assertRaises(ValueError):
            validate_plan(altered, self.config, self.manifest, self.checkpoints)
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.config, {"decisions": []}, self.checkpoints)

    def test_wrong_gradient_budget_is_not_completed_work(self):
        row = self.outcome(0, "continue")
        row["budget_gradient_updates"] = 100
        with self.assertRaises(ValueError):
            coverage(self.plan, self.plan["operations"][0], [row])

    def test_stale_complete_event_never_stops_an_active_job(self):
        progress = {"stage": "mechanisms", "event": "mechanisms_complete", "job_id": "mechanisms:cp:0"}
        manifest = {"jobs": {"mechanisms:cp:0": {"status": "running"}}}
        self.assertFalse(safe_boundary(self.config, manifest, progress, self.checkpoints, []))
        manifest["jobs"]["mechanisms:cp:0"]["status"] = "complete"
        self.assertTrue(safe_boundary(self.config, manifest, progress, self.checkpoints, []))
        self.assertFalse(safe_boundary(self.config, manifest, progress, self.checkpoints, [{"pid": 1}]))
        progress["stage"] = "lr_control"
        self.assertFalse(safe_boundary(self.config, manifest, progress, self.checkpoints, []))

    def test_live_handoff_is_retired_and_only_records_an_audit(self):
        config = dict(self.config, output_root="runs/nonexistent-guard-unit-test")
        def fake_read(path):
            return {"config.json": config, "manifest.json": {"worker_pid": 100},
                    "progress.json": {"stage": "source"}, "supervisor.json": {"supervisor_pid": 200}}[path.name]
        with patch.object(handoff_module, "read", side_effect=fake_read), patch.object(handoff_module, "alive", return_value=True), patch.object(handoff_module, "write") as write:
            record = handoff_module.handoff(Path("config.json"), 100, 200, 2)
            self.assertEqual(record["process_mutations"], "disabled")
            self.assertTrue(record["coordinator_alive"])
            self.assertEqual(write.call_args[0][0].name, "recorded_handoff_audit.json")
            self.assertFalse(hasattr(handoff_module, "terminate"))

    def test_standalone_resume_never_starts_beside_live_child_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            lease = folder / "workers/source/example/worker_lease.json"
            lease.parent.mkdir(parents=True)
            lease.write_text(json.dumps({"pid": 321}), encoding="utf-8")
            with patch("scripts.research_supervisor.alive", return_value=True):
                with self.assertRaises(RuntimeError):
                    ensure_no_live_child_leases(folder)
            with patch("scripts.research_supervisor.alive", return_value=False):
                ensure_no_live_child_leases(folder)


if __name__ == "__main__":
    unittest.main()
