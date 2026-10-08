"""Logged transfer evaluation: freeze audit, pairing, budget and cluster tests."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.rl_transfer import POLICY_IDS, _bootstrap, protocol_hash, summarize_transfer, validate_frozen_protocol


class FrozenTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.envelope = {
            "schema_version": 1, "frozen_at": "2026-10-07T00:00:00Z",
            "protocol": {"development_games": ["breakout", "asterix"],
                         "reserved_games": ["freeway", "seaquest"],
                         "policy_ids": list(POLICY_IDS),
                         "main_menu": ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"],
                         "harm_margin": 0.0,
                         "fit_audit": {"training_games": ["breakout", "asterix"]}},
        }
        self.envelope["protocol_hash"] = protocol_hash(self.envelope)
        self.protocol_path = self.root / "transfer_protocol.json"
        self.protocol_path.write_text(json.dumps(self.envelope), encoding="utf-8")
        self.rows = self.make_rows()
        self.write_rows()

    def make_rows(self, histories=5, games=("freeway", "seaquest"), budget=50000):
        records = []
        actions = {"continue": "continue", "SBS": "optimizer_reset", "age_return_ridge": "continue",
                   "diagnostic_ridge": "head_reset", "diagnostic_tree": "head_and_optimizer_reset", "chooser": "continue"}
        for game in games:
            for history in range(histories):
                for age in (50, 200):
                    for repeat in (0, 1):
                        base = 20 + 5 * history + age / 100 + repeat + (100 if game == "seaquest" else 0)
                        effects = {"continue": 0, "SBS": 1, "age_return_ridge": 0.5,
                                   "diagnostic_ridge": 1 + history, "diagnostic_tree": history - 1,
                                   "chooser": 0.75}
                        for policy in POLICY_IDS:
                            probe = 2000 if policy == "chooser" else 0
                            records.append({"policy_id": policy, "selected_action": actions[policy],
                                            "phase": "reserved", "game": game,
                                            "trajectory_id": f"{game}_source_{history}", "checkpoint_id": f"age_{age}",
                                            "repeat": repeat, "j_final": base + effects[policy], "failure_reason": None,
                                            "total_budget_gradient_updates": budget,
                                            "probe_gradient_updates": probe, "continuation_gradient_updates": budget - probe,
                                            "budget_env_steps": budget, "continuation_env_steps": budget - probe,
                                            "wall_time_sec": 0.25,
                                            "protocol_hash": self.envelope["protocol_hash"],
                                            "evaluation_started_at": "2026-10-07T00:00:01Z"})
        return records

    def write_rows(self):
        (self.root / "transfer_outcomes.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows), encoding="utf-8")

    def test_hash_timestamp_and_artifact_integrity(self):
        validated = validate_frozen_protocol(self.protocol_path)
        self.assertEqual(validated["protocol_hash"], self.envelope["protocol_hash"])
        altered = copy.deepcopy(self.envelope)
        altered["protocol"]["harm_margin"] = 4.0
        self.protocol_path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            summarize_transfer(self.root)
        artifact = self.root / "selector.json"
        artifact.write_text("frozen selector", encoding="utf-8")
        self.envelope["protocol"]["artifacts"] = [{"path": "selector.json", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]
        self.envelope["protocol_hash"] = protocol_hash(self.envelope)
        self.protocol_path.write_text(json.dumps(self.envelope), encoding="utf-8")
        self.assertEqual(len(validate_frozen_protocol(self.protocol_path)["verified_artifacts"]), 1)
        artifact.write_text("changed selector", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            validate_frozen_protocol(self.protocol_path)

    def test_paired_diagnostic_comparisons_and_input_read_only(self):
        before_protocol = self.protocol_path.read_bytes()
        outcomes = self.root / "transfer_outcomes.jsonl"
        before_outcomes = outcomes.read_bytes()
        result = summarize_transfer(self.root, bootstrap_repeats=200)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["population_oracle_estimated"])
        self.assertFalse(result["policies_fitted_or_selected_by_this_analysis"])
        self.assertEqual(len(result["strata"]), 1)
        comparisons = result["strata"][0]["comparisons"]
        sbs = comparisons["diagnostic_ridge_minus_SBS"]
        age = comparisons["diagnostic_ridge_minus_age_return_ridge"]
        self.assertTrue(sbs["primary_comparison"])
        self.assertEqual(sbs["paired_effect"]["mean"], 2.0)
        self.assertEqual(age["paired_effect"]["mean"], 2.5)
        self.assertEqual(sbs["valid_pairs"], 40)
        self.assertFalse(sbs["paired_effect"]["uncertain"])
        self.assertIsNotNone(sbs["paired_effect"]["ci95"])
        self.assertEqual(outcomes.read_bytes(), before_outcomes)
        self.assertEqual(self.protocol_path.read_bytes(), before_protocol)
        self.assertTrue((self.root / "transfer_summary.json").exists())
        pairs = [json.loads(line) for line in (self.root / "transfer_paired_effects.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["comparison"] == "diagnostic_ridge_minus_SBS" for row in pairs))
        json.dumps(result, allow_nan=False)

    def test_declared_budget_strata_ignore_actual_chooser_continuation_length(self):
        extra = self.make_rows(histories=1, games=("freeway",), budget=10000)
        for row in extra:
            row["trajectory_id"] = "freeway_low_budget_source"
        self.rows.extend(extra)
        self.write_rows()
        result = summarize_transfer(self.root, bootstrap_repeats=30)
        self.assertEqual(len(result["strata"]), 2)
        full = next(s for s in result["strata"] if s["total_budget_gradient_updates"] == 50000)
        self.assertEqual(full["budget_env_steps"], 50000)
        self.assertEqual(full["policies"]["chooser"]["actual_continuation_env_steps"], [48000])
        self.assertEqual(full["policies"]["continue"]["actual_continuation_env_steps"], [50000])
        small = next(s for s in result["strata"] if s["total_budget_gradient_updates"] == 10000)
        stat = small["comparisons"]["diagnostic_ridge_minus_SBS"]["paired_effect"]
        self.assertTrue(stat["uncertain"])
        self.assertIsNone(stat["ci95"])

    def test_phases_and_development_reserved_roles_are_separate(self):
        extra = self.make_rows(histories=2, games=("freeway",))
        for row in extra:
            row.update(game="breakout", trajectory_id=row["trajectory_id"].replace("freeway", "breakout"), phase="development")
        self.rows.extend(extra)
        self.write_rows()
        result = summarize_transfer(self.root, bootstrap_repeats=30)
        self.assertEqual({s["evaluation_split"] for s in result["strata"]}, {"development", "reserved"})
        self.rows[0]["phase"] = "development"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "phase/split"):
            summarize_transfer(self.root)

    def test_failures_and_missing_pairs_remain_explicit(self):
        bad = next(row for row in self.rows if row["policy_id"] == "diagnostic_ridge")
        bad.update(j_final=None, failure_reason="nonfinite")
        reference = next(row for row in self.rows if row["policy_id"] == "SBS")
        self.rows.remove(reference)
        self.write_rows()
        result = summarize_transfer(self.root, bootstrap_repeats=30)
        self.assertEqual(result["status"], "incomplete")
        stratum = result["strata"][0]
        self.assertEqual(stratum["policies"]["diagnostic_ridge"]["failure_reasons"], {"nonfinite": 1})
        comparison = stratum["comparisons"]["diagnostic_ridge_minus_SBS"]
        self.assertEqual(comparison["valid_pairs"], 39)
        self.assertTrue(comparison["success_pairs_only_statistics"])
        self.assertIn("missing_reference", comparison["unavailable_reasons"])

    def test_evaluation_before_freeze_hash_mismatch_duplicate_and_budget_rejected(self):
        self.rows[0]["evaluation_started_at"] = "2026-10-06T23:59:59Z"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "before the protocol"):
            summarize_transfer(self.root)
        self.rows[0]["evaluation_started_at"] = "2026-10-07T00:00:01Z"
        self.rows[0]["protocol_hash"] = "0" * 64
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "different frozen protocol"):
            summarize_transfer(self.root)
        self.rows[0]["protocol_hash"] = self.envelope["protocol_hash"]
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_transfer(self.root)
        self.rows.pop()
        self.rows[0]["probe_gradient_updates"] = 50001
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "exceed"):
            summarize_transfer(self.root)

    def test_cluster_resampling_not_repeat_resampling_or_checkpoint_overweighting(self):
        records = []
        for history, values in (("one", [-10, 10]), ("two", [0, 20])):
            for checkpoint, effect in enumerate(values):
                repeats = 20 if checkpoint == 0 else 1
                for repeat in range(repeats):
                    records.append({"game": "freeway", "trajectory_id": history,
                                    "checkpoint_id": checkpoint, "repeat": repeat, "effect": effect})
        stat = _bootstrap(records, "effect", repeats=100, seed=37)
        self.assertEqual(stat["mean"], 5.0)
        self.assertEqual(stat["source_histories_per_game"], {"freeway": 2})
        duplicates = [dict(row, repeat=row["repeat"] + 100) for row in records]
        again = _bootstrap(records + duplicates, "effect", repeats=100, seed=37)
        self.assertEqual(stat["ci95"], again["ci95"])
        self.assertTrue(stat["uncertain"])

    def test_heldout_game_fit_audit_and_ambiguous_history_are_rejected(self):
        self.envelope["protocol"]["fit_audit"]["training_games"] = ["seaquest"]
        self.envelope["protocol_hash"] = protocol_hash(self.envelope)
        self.protocol_path.write_text(json.dumps(self.envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-development"):
            validate_frozen_protocol(self.protocol_path)
        self.envelope["protocol"]["fit_audit"]["training_games"] = ["breakout"]
        self.envelope["protocol_hash"] = protocol_hash(self.envelope)
        self.protocol_path.write_text(json.dumps(self.envelope), encoding="utf-8")
        for row in self.rows:
            row["protocol_hash"] = self.envelope["protocol_hash"]
        self.rows[-1]["trajectory_id"] = self.rows[0]["trajectory_id"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "multiple games"):
            summarize_transfer(self.root)

    def test_missing_declared_game_blocks_suite_mean(self):
        self.rows = [row for row in self.rows if row["game"] == "freeway"]
        self.write_rows()
        result = summarize_transfer(self.root, bootstrap_repeats=30)
        self.assertEqual(result["status"], "incomplete")
        stratum = result["strata"][0]
        self.assertEqual(stratum["absent_evaluation_games"], ["seaquest"])
        statistics = stratum["comparisons"]["diagnostic_ridge_minus_SBS"]["paired_effect"]
        self.assertIsNone(statistics["mean"])
        self.assertEqual(statistics["observed_games_macro_mean"], 2.0)
        self.assertTrue(statistics["uncertain"])

    def test_empty_results_and_bootstrap_disabled_are_descriptive(self):
        result = summarize_transfer(self.root, bootstrap_repeats=0)
        stat = result["strata"][0]["comparisons"]["diagnostic_ridge_minus_SBS"]["paired_effect"]
        self.assertEqual(stat["mean"], 2.0)
        self.assertIsNone(stat["ci95"])
        self.rows = []
        self.write_rows()
        result = summarize_transfer(self.root, bootstrap_repeats=30)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["strata"], [])


if __name__ == "__main__":
    unittest.main()
