"""Checks for independent selection/evaluation and grouped uncertainty."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rl_analysis import MENU, analyze, build_report, choose_action, fit_selectors, train_selector


class Fixture:
    def __init__(self, root):
        self.root = root
        self.config = {"kind": "synthetic_analysis_fixture_not_RL", "main_menu": list(MENU),
                       "screen": {"games": ["game_a"]}, "adaptive": {"development_extension_games": ["game_b"], "reserved_games": ["reserved"]}}
        self.checkpoints, self.diagnostics, self.outcomes = [], [], []

    def history(self, trajectory, game="game_a", effects=(0, 1, 2, 3), repeat_effects=None, feature=1.0, ages=(100,), mode="natural", budget=50):
        for age in ages:
            identifier = f"{trajectory}_age{age}"
            self.checkpoints.append({"checkpoint_id": identifier, "trajectory_id": trajectory, "game": game,
                                     "nominal_age": age, "environment_steps": age, "training_seed": trajectory,
                                     "source_mode": mode, "j_pre": 10, "random_return": 0, "baseline_learned": True})
            self.diagnostics.append({"checkpoint_id": identifier, "features": {"x": feature, "nullable": None, "post_oracle_score": 999}})
            for repeat in range(4):
                values = repeat_effects[repeat] if repeat_effects is not None else effects
                noise = (-2, 2, -1, 1)[repeat]
                for action, effect in zip(MENU, values, strict=True):
                    self.outcomes.append({"checkpoint_id": identifier, "trajectory_id": trajectory, "game": game,
                                          "source_mode": mode, "repair_id": action, "repeat": repeat, "continuation_seed": 100+repeat,
                                          "budget_env_steps": budget, "budget_gradient_updates": 10,
                                          "j_pre": 10, "j_immediate": 10, "j_final": 10+noise+effect,
                                          "adaptation_auc": 10+noise+effect, "failure_reason": None})

    def save(self):
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        for name, rows in (("checkpoints", self.checkpoints), ("diagnostics", self.diagnostics), ("outcomes", self.outcomes)):
            (self.root / f"{name}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture = Fixture(self.root)

    def test_missing_running_files_are_incomplete(self):
        result = analyze(self.root, bootstrap_repeats=5)
        self.assertEqual(result["primary_gate"], "incomplete")
        self.assertTrue(result["uncertain"])
        self.assertEqual(len(result["missing_files"]), 3)

    def test_report_defers_fitting_before_heterogeneity_gate(self):
        with patch("experiments.rl_analysis.fit_selectors") as fit:
            paths = build_report(self.root, self.root / "report", bootstrap_repeats=5)
            fit.assert_not_called()
        selector_path = next(path for path in paths if path.name == "selector_analysis.json")
        self.assertEqual(json.loads(selector_path.read_text(encoding="utf-8"))["status"], "deferred_before_confirmed_signal")

    def test_paired_noise_cancels_and_history_count_is_not_checkpoint_count(self):
        for index in range(6):
            self.fixture.history(f"history_{index}", effects=(0, 1, 2, 3), ages=(100, 200))
        self.fixture.save()
        result = analyze(self.root, bootstrap_repeats=30)["strata"][0]
        statistic = result["action_statistics"]["head_reset"]["paired_effect"]["game_a"]
        self.assertEqual(statistic["mean"], 2)
        self.assertEqual(statistic["ci95"], [2, 2])
        self.assertEqual(statistic["source_histories"], 6)
        self.assertEqual(statistic["checkpoints"], 12)
        self.assertEqual(result["primary_gate"], "single_repair")
        self.assertEqual(result["parameter_repair_signal"]["status"], "positive")
        crossfit = result["cross_fitted_estimated_winner"]["advantage_vs_sbs"]["game_a"]
        self.assertTrue(crossfit["bootstrap_refits_selection"])
        self.assertEqual(crossfit["ci95"], [0, 0])

    def test_sbs_never_uses_test_source_history(self):
        self.fixture.history("train_a", effects=(0, 10, 0, 0), ages=(100, 200))
        self.fixture.history("test_b", effects=(0, 0, 100, 0), ages=(100, 200))
        self.fixture.save()
        result = analyze(self.root, bootstrap_repeats=10)["strata"][0]
        choices = result["cross_fitted_estimated_winner"]["choices"]
        test_choices = [row for row in choices if row["trajectory_id"] == "test_b"]
        self.assertTrue(test_choices)
        self.assertTrue(all(row["sbs_action"] == "optimizer_reset" for row in test_choices))
        for row in choices:
            self.assertNotIn(row["trajectory_id"], row["sbs_training_trajectory_ids"])
            self.assertFalse(set(map(tuple, row["selection_repeat_ids"])) & set(map(tuple, row["evaluation_repeat_ids"])))

    def test_noisy_maximum_is_not_heterogeneity_signal(self):
        alternating = [(0, 2, -2, 0), (0, -1, 1, 0), (0, 2, -2, 0), (0, -1, 1, 0)]
        for index in range(6):
            self.fixture.history(f"noise_{index}", repeat_effects=alternating)
        self.fixture.save()
        result = analyze(self.root, bootstrap_repeats=30)["strata"][0]
        self.assertNotEqual(result["primary_gate"], "heterogeneous")
        self.assertFalse(result["gate_is_confirmatory"])
        self.assertIn("naive_same_sample_maximum_gap_vs_heldout_history_sbs", result)

    def test_failed_branch_blocks_gate_and_is_not_imputed(self):
        for index in range(6):
            self.fixture.history(f"history_{index}")
        failed = next(row for row in self.fixture.outcomes if row["repair_id"] == "head_reset")
        failed["failure_reason"], failed["j_final"] = "nonfinite_update", None
        self.fixture.save()
        result = analyze(self.root, bootstrap_repeats=10)["strata"][0]
        self.assertEqual(result["primary_gate"], "incomplete")
        self.assertEqual(result["parameter_repair_signal"]["status"], "incomplete")
        self.assertEqual(result["failed_rows_by_reason"], {"nonfinite_update": 1})
        self.assertTrue(result["success_pair_only_statistics"])

    def test_finite_truncated_outcome_is_not_full_budget_success(self):
        for index in range(6):
            self.fixture.history(f"history_{index}")
        truncated = next(row for row in self.fixture.outcomes if row["repair_id"] == "head_reset")
        truncated["completed_env_steps"] = 10  # Declared environment budget is 50.
        self.fixture.save()
        result = analyze(self.root, bootstrap_repeats=10)["strata"][0]
        self.assertEqual(result["primary_gate"], "incomplete")
        self.assertEqual(result["completed_environment_budget_mismatch_rows"], 1)
        self.assertEqual(result["parameter_repair_signal"]["status"], "incomplete")

    def test_source_mode_and_environment_budget_are_not_pooled(self):
        for index in range(3):
            self.fixture.history(f"natural_{index}", mode="natural", budget=50)
            self.fixture.history(f"stress_{index}", mode="stress", budget=20, effects=(0, -10, 0, 0))
        self.fixture.save()
        result = analyze(self.root, bootstrap_repeats=10)
        self.assertEqual(len(result["strata"]), 2)
        effects = {row["source_mode"]: row["action_statistics"]["optimizer_reset"]["paired_effect"]["game_a"]["mean"] for row in result["strata"]}
        self.assertEqual(effects, {"natural": 1, "stress": -10})

    def test_preprocessing_and_hyperparameter_tuning_are_outer_train_only(self):
        for index in range(3):
            self.fixture.history(f"a_{index}", game="game_a", feature=index, ages=(100, 200))
            self.fixture.history(f"b_{index}", game="game_b", feature=1000000+index, ages=(100, 200))
        self.fixture.save()
        result = fit_selectors(self.root, bootstrap_repeats=10)["strata"][0]
        schemes = result["schemes"]
        for scheme in schemes.values():
            for policy in scheme["policies"].values():
                for audit in policy["fold_audits"]:
                    self.assertFalse(set(audit["training_trajectory_ids"]) & set(audit["evaluation_trajectory_ids"]))
        audits = schemes["leave_one_game_out"]["policies"]["diagnostic_ridge"]["fold_audits"]
        heldout_b = next(row for row in audits if row["evaluation_games"] == ["game_b"])
        index = heldout_b["feature_names"].index("x")
        self.assertEqual(heldout_b["scaler_mean"][index], 1)
        self.assertNotIn("post_oracle_score", heldout_b["feature_names"])
        self.assertEqual([row["alpha"] for row in heldout_b["inner_tuning"]], [0.1, 1, 10, 100])

    def test_frozen_deploy_includes_development_extension_but_excludes_reserved(self):
        for index in range(3):
            self.fixture.history(f"a_{index}", game="game_a", feature=index)
            self.fixture.history(f"b_{index}", game="game_b", feature=index+3)
            self.fixture.history(f"reserved_{index}", game="reserved", effects=(0, 1000, -1000, 0), feature=1000000000)
        # Reserved exclusion must override an accidental broad screen whitelist.
        self.fixture.config["screen"]["games"].append("reserved")
        self.fixture.save()
        model = train_selector(self.root, ("natural", 50))
        self.assertEqual(model.metadata["fit_audit"]["training_games"], ["game_a", "game_b"])
        self.assertFalse(model.metadata["heldout_outcomes_used"])
        self.assertTrue(all(not name.startswith("reserved") for name in model.metadata["checkpoint_ids"]))
        self.assertIn(choose_action(model, {"x": 1, "nominal_age": 100, "j_pre": 10}), MENU)

    def test_reserved_transfer_checkpoints_do_not_change_development_gate_or_cv(self):
        for index in range(6):
            self.fixture.history(f"a_{index}")
            self.fixture.history(f"test_{index}", game="reserved", effects=(0, 1000, -1000, 0))
        # Transfer uses its own outcome file; these deliberately lack main-menu rows.
        self.fixture.outcomes = [row for row in self.fixture.outcomes if row["game"] != "reserved"]
        self.fixture.save()
        result = analyze(self.root, bootstrap_repeats=10)
        self.assertEqual(result["primary_gate"], "single_repair")
        self.assertFalse(result["checkpoints_without_outcomes"])
        self.assertEqual(len(result["reserved_checkpoint_ids_excluded_from_development_analysis"]), 6)
        # Even accidental reserved rows in the main outcome file must not train CV.
        self.fixture.history("accidental_test_row", game="reserved", effects=(0, 1e6, -1e6, 0))
        self.fixture.save()
        selectors = fit_selectors(self.root, bootstrap_repeats=5)
        for policy in selectors["strata"][0]["schemes"]["heldout_source_history"]["policies"].values():
            for audit in policy["fold_audits"]:
                self.assertEqual(audit["training_games"], ["game_a"])
                self.assertEqual(audit["evaluation_games"], ["game_a"])


if __name__ == "__main__":
    unittest.main()
