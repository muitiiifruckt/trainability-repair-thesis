"""Read-only statistical analysis of checkpoint repair experiments.

Rows are paired by checkpoint and continuation repeat; full source histories are
the independent resampling/split units. Cross-fitted estimated-winner value is
not a population oracle. Missing/failed branches block progression gates.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MENU = ("continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset")
REQUIRED = ("checkpoints.jsonl", "diagnostics.jsonl", "outcomes.jsonl")
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _jsonl(path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as error:
                    # A concurrently written final line is an incomplete record.
                    if not line.endswith("\n"):
                        break
                    raise ValueError(f"Invalid JSON in {path}:{number}") from error
    return result


def _load(run_dir):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    checkpoints, diagnostics, rows = [_jsonl(run_dir / name) for name in REQUIRED]
    cp = {}
    for item in checkpoints:
        identifier = item["checkpoint_id"]
        if identifier in cp and cp[identifier] != item:
            raise ValueError(f"Conflicting checkpoint metadata: {identifier}")
        cp[identifier] = item
    features = {}
    for item in diagnostics:
        identifier = item["checkpoint_id"]
        if identifier in features and features[identifier] != item.get("features", {}):
            raise ValueError(f"Conflicting pre-repair diagnostics: {identifier}")
        features[identifier] = item.get("features", {})
    progress_path, manifest_path = run_dir / "progress.json", run_dir / "manifest.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    worker = {"progress": progress, "worker_status": manifest.get("worker_status"),
              "worker_pid": manifest.get("worker_pid"), "manifest_updated_at": manifest.get("updated_at"),
              "jobs_by_status": dict(Counter(str(job.get("status", "unspecified")) for job in manifest.get("jobs", {}).values()))}
    return {"config": config, "checkpoints": cp, "features": features, "rows": rows, "worker": worker,
            "menu": tuple(config.get("main_menu", MENU)),
            "missing_files": [name for name in REQUIRED if not (run_dir / name).exists()]}


def _repeat(row):
    return (str(row.get("continuation_seed", "")), str(row.get("repeat", "")))


def _valid(row):
    return not row.get("failure_reason") and _finite(row.get("j_final")) and not _budget_mismatch(row)


def _budget_mismatch(row):
    completed, declared = row.get("completed_env_steps"), row.get("budget_env_steps")
    return _finite(completed) and _finite(declared) and completed != declared


def _reserved_games(data):
    return (set(data["config"].get("adaptive", {}).get("reserved_games", [])) |
            set(data["config"].get("reserved_games", [])))


def _development_games(data):
    config = data["config"]
    games = list(config.get("screen", {}).get("games", config.get("train_games", [])))
    games += list(config.get("adaptive", {}).get("development_extension_games", []))
    return sorted(set(games) - _reserved_games(data))


def _strata(data):
    buckets = defaultdict(list)
    for row in data["rows"]:
        metadata = data["checkpoints"].get(row.get("checkpoint_id"), {})
        if row.get("game", metadata.get("game")) in _reserved_games(data) or row.get("policy_id"):
            continue
        if row.get("repair_id") in data["menu"]:
            mode = row.get("source_mode", metadata.get("source_mode", "unspecified"))
            buckets[(str(mode), row.get("budget_env_steps"))].append(row)
    return buckets


def _panels(data, rows):
    panels = defaultdict(dict)
    for row in rows:
        key = (row["checkpoint_id"], _repeat(row))
        action = row["repair_id"]
        if action in panels[key]:
            raise ValueError(f"Duplicate outcome for {key}, {action}")
        panels[key][action] = row
    return dict(panels)


def _metadata(data, checkpoint_id, panel):
    result = dict(data["checkpoints"].get(checkpoint_id, {}))
    example = next(iter(panel.values()))
    result.setdefault("checkpoint_id", checkpoint_id)
    result.setdefault("trajectory_id", example.get("trajectory_id"))
    result.setdefault("game", example.get("game"))
    if not result.get("trajectory_id") or not result.get("game"):
        raise ValueError(f"Missing game/source history for {checkpoint_id}")
    for row in panel.values():
        if row.get("trajectory_id") != result["trajectory_id"] or row.get("game") != result["game"]:
            raise ValueError(f"Outcome metadata disagree for {checkpoint_id}")
    return result


def _complete(data, panels):
    by_cp = defaultdict(list)
    for (identifier, repeat), panel in panels.items():
        if all(action in panel and _valid(panel[action]) for action in data["menu"]):
            metadata = _metadata(data, identifier, panel)
            by_cp[identifier].append({"metadata": metadata, "repeat": repeat, "panel": panel})
    return dict(by_cp)


def _mean_games(records, field):
    """Equal checkpoint weight within history; equal histories within game."""
    histories = defaultdict(lambda: defaultdict(list))
    for item in records:
        if _finite(item.get(field)):
            histories[item["game"]][item["trajectory_id"]].append(item[field])
    return {game: float(np.mean([np.mean(values) for values in groups.values()]))
            for game, groups in sorted(histories.items()) if groups}


def _bootstrap_paired(records, field, repeats, seed=1729):
    """Nested paired bootstrap: histories, then repeat vectors within checkpoint.

    `records` must contain per-repeat differences (not independently resampled
    action returns). Checkpoints remain nested in their entire source history.
    """
    by_game = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in records:
        if _finite(item.get(field)):
            by_game[item["game"]][item["trajectory_id"]][item["checkpoint_id"]].append(float(item[field]))
    rng = np.random.default_rng(seed)
    result = {}
    for game, histories in sorted(by_game.items()):
        keys = sorted(histories)
        estimate = float(np.mean([np.mean([np.mean(values) for values in histories[key].values()]) for key in keys]))
        draws = []
        for _ in range(repeats):
            sampled = rng.choice(len(keys), len(keys), replace=True)
            means = []
            for index in sampled:
                checkpoint_means = []
                for values in histories[keys[index]].values():
                    array = np.asarray(values)
                    checkpoint_means.append(float(rng.choice(array, len(array), replace=True).mean()))
                means.append(float(np.mean(checkpoint_means)))
            draws.append(float(np.mean(means)))
        interval = [float(value) for value in np.quantile(draws, [0.025, 0.975])] if draws else None
        result[game] = {"mean": estimate, "ci95": interval, "source_histories": len(keys),
                        "checkpoints": sum(len(value) for value in histories.values()),
                        "paired_repeats": sum(len(values) for history in histories.values() for values in history.values()),
                        "ci_unit": "full_source_history_with_nested_paired_continuation_repeats",
                        "small_history_count": len(keys) < 5}
    return result


def _cp_means(data, complete):
    samples = []
    for identifier, repeats in sorted(complete.items()):
        metadata = repeats[0]["metadata"]
        returns = np.asarray([[item["panel"][action]["j_final"] for action in data["menu"]] for item in repeats], dtype=float)
        samples.append({"checkpoint_id": identifier, "trajectory_id": metadata["trajectory_id"],
                        "game": metadata["game"], "metadata": metadata, "repeats": repeats,
                        "returns": returns.mean(axis=0),
                        "effects": (returns - returns[:, [data["menu"].index("continue")]]).mean(axis=0)})
    return samples


def _single_best(samples, menu):
    """Use only supplied training histories, equally weighted within each game."""
    if not samples:
        return None
    scores = []
    for index in range(len(menu)):
        records = [dict(item, effect=float(item["effects"][index])) for item in samples]
        scores.append(float(np.mean(list(_mean_games(records, "effect").values()))))
    return menu[int(np.argmax(scores))]


def _crossfit(data, samples, seed=2718):
    """Per-checkpoint repeat split; SBS excludes the *entire* test history."""
    records, choices, skipped = [], [], []
    rng = np.random.default_rng(seed)
    menu = data["menu"]
    for item in samples:
        training = [other for other in samples if other["trajectory_id"] != item["trajectory_id"]]
        sbs = _single_best(training, menu)
        repeats = item["repeats"]
        if sbs is None or len(repeats) < 2:
            skipped.append(item["checkpoint_id"])
            continue
        order = rng.permutation(len(repeats))
        for fold in (0, 1):
            choose_ids = order[fold::2]
            evaluation_ids = order[1-fold::2]
            chosen_values = np.asarray([[repeats[index]["panel"][action]["j_final"] for action in menu] for index in choose_ids])
            chosen = menu[int(np.argmax(chosen_values.mean(axis=0)))]
            choices.append({"checkpoint_id": item["checkpoint_id"], "trajectory_id": item["trajectory_id"],
                            "game": item["game"], "fold": fold, "chosen_action": chosen, "sbs_action": sbs,
                            "selection_repeat_ids": [list(repeats[index]["repeat"]) for index in choose_ids],
                            "evaluation_repeat_ids": [list(repeats[index]["repeat"]) for index in evaluation_ids],
                            "sbs_training_trajectory_ids": sorted({other["trajectory_id"] for other in training})})
            for index in evaluation_ids:
                panel = repeats[index]["panel"]
                records.append({"checkpoint_id": item["checkpoint_id"], "trajectory_id": item["trajectory_id"], "game": item["game"],
                                "effect_vs_continue": float(panel[chosen]["j_final"] - panel["continue"]["j_final"]),
                                "advantage_vs_sbs": float(panel[chosen]["j_final"] - panel[sbs]["j_final"]),
                                "selected_return": float(panel[chosen]["j_final"]), "repeat": list(repeats[index]["repeat"])})
    return records, choices, skipped


def _bootstrap_crossfit(data, samples, records, repeats, seed=1729):
    """Resample histories and paired vectors, refitting winner and SBS choices.

    Selection and evaluation halves stay disjoint in every draw: repeat vectors
    are sampled *within* their original half, never across that boundary. SBS
    training excludes every bootstrap copy of the original evaluated history.
    These are procedure-conditional exploratory intervals, not simultaneous
    intervals or evidence from an independent confirmation cohort.
    """
    fields = ("advantage_vs_sbs", "effect_vs_continue")
    result = {field: _bootstrap_paired(records, field, 0) for field in fields}
    if not records or not repeats:
        return result
    menu = data["menu"]
    continue_index = menu.index("continue")
    split_rng = np.random.default_rng(2718)
    groups = defaultdict(lambda: defaultdict(list))
    for item in samples:
        if len(item["repeats"]) < 2:
            continue
        order = split_rng.permutation(len(item["repeats"]))
        values = np.asarray([[repeat["panel"][action]["j_final"] for action in menu]
                             for repeat in item["repeats"]], dtype=float)
        groups[item["game"]][item["trajectory_id"]].append(
            dict(item, return_vectors=values, split_order=order))
    rng = np.random.default_rng(seed)
    draws = {field: defaultdict(list) for field in fields}
    for _ in range(repeats):
        bootstrap_samples = []
        for game, histories in sorted(groups.items()):
            history_ids = sorted(histories)
            for copy_number, history_index in enumerate(rng.choice(len(history_ids), len(history_ids), replace=True)):
                origin = history_ids[history_index]
                for item in histories[origin]:
                    vectors = item["return_vectors"]
                    train_vectors = vectors[rng.choice(len(vectors), len(vectors), replace=True)]
                    bootstrap_samples.append(dict(item, origin_trajectory_id=origin,
                        trajectory_id=f"{game}:bootstrap_history_{copy_number}",
                        effects=(train_vectors - train_vectors[:, [continue_index]]).mean(axis=0)))
        # SBS changes with the resampled training histories and their repeats.
        sbs_cache = {}
        draw_records = []
        for item in bootstrap_samples:
            origin = item["origin_trajectory_id"]
            if origin not in sbs_cache:
                training = [other for other in bootstrap_samples if other["origin_trajectory_id"] != origin]
                sbs_cache[origin] = _single_best(training, menu)
            sbs = sbs_cache[origin]
            if sbs is None:
                continue
            vectors, order = item["return_vectors"], item["split_order"]
            fold_records = []
            for fold in (0, 1):
                choose_pool, evaluate_pool = order[fold::2], order[1-fold::2]
                choose_ids = rng.choice(choose_pool, len(choose_pool), replace=True)
                evaluate_ids = rng.choice(evaluate_pool, len(evaluate_pool), replace=True)
                action_index = int(np.argmax(vectors[choose_ids].mean(axis=0)))
                evaluation = vectors[evaluate_ids]
                for vector in evaluation:
                    fold_records.append({"advantage_vs_sbs": float(vector[action_index] - vector[menu.index(sbs)]),
                                         "effect_vs_continue": float(vector[action_index] - vector[continue_index])})
            # Keep checkpoint and source-history weights independent of repeats.
            draw_records.append({"checkpoint_id": item["checkpoint_id"], "game": item["game"],
                                 "trajectory_id": item["trajectory_id"],
                                 **{field: float(np.mean([row[field] for row in fold_records])) for field in fields}})
        for field in fields:
            for game, value in _mean_games(draw_records, field).items():
                draws[field][game].append(value)
    for field in fields:
        for game, stat in result[field].items():
            values = draws[field].get(game, [])
            valid = len(values)
            stat.update({"ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])] if valid else None,
                         "bootstrap_valid_draws": valid, "bootstrap_requested_draws": repeats,
                         "bootstrap_refits_selection": True,
                         "ci_unit": "full_source_history_paired_continuation_vectors_and_refitted_choices"})
            if valid < 0.95 * repeats:
                stat["ci95"] = None
                stat["bootstrap_warning"] = "insufficient_draws_with_outside_history_sbs_training"
    return result


def _qualification(samples):
    known = []
    for item in samples:
        metadata = item["metadata"]
        if isinstance(metadata.get("baseline_learned"), bool):
            known.append(metadata["baseline_learned"])
        elif _finite(metadata.get("random_return")):
            source_return = metadata.get("source_eval_return", metadata.get("j_pre"))
            if _finite(source_return):
                margin = float(metadata.get("learning_margin", 0.0))
                known.append(source_return > metadata["random_return"] + margin)
    return {"known_checkpoints": len(known), "learned_checkpoints": sum(known),
            "unlearned_checkpoints": len(known) - sum(known), "unknown_checkpoints": len(samples) - len(known),
            "rule": "explicit_baseline_learned_or_separate_source_return_above_random_plus_declared_margin"}


def _gate(incomplete, qualification, crossfit_stats, choices, min_repeats, margin=0.0):
    if incomplete:
        return "incomplete"
    if qualification["known_checkpoints"] and not qualification["learned_checkpoints"]:
        return "baseline_unlearned"
    if qualification["unknown_checkpoints"] or min_repeats < 4:
        return "uncertain"
    adequate = [stat for stat in crossfit_stats.values() if stat["source_histories"] >= 5 and stat["ci95"] is not None]
    if len(adequate) != len(crossfit_stats) or not adequate:
        return "uncertain"
    unique = {item["chosen_action"] for item in choices}
    if len(unique) >= 2 and any(stat["ci95"][0] > margin for stat in adequate):
        return "heterogeneous"
    # This is evidence for one repair at this resolution, not proof of no heterogeneity.
    sbs = {item["sbs_action"] for item in choices}
    if len(sbs) == 1 and len(unique) == 1 and unique == sbs and all(stat["ci95"][1] <= margin for stat in adequate):
        return "single_repair"
    return "uncertain"


def _parameter_signal(data, panels, bootstrap_repeats, incomplete):
    contrasts = {}
    for label, action, reference in (("head_vs_continue", "head_reset", "continue"),
                                     ("joint_vs_optimizer", "head_and_optimizer_reset", "optimizer_reset")):
        records = []
        for (identifier, repeat), panel in panels.items():
            if action not in panel or reference not in panel or not _valid(panel[action]) or not _valid(panel[reference]):
                continue
            metadata = _metadata(data, identifier, panel)
            records.append({"checkpoint_id": identifier, "trajectory_id": metadata["trajectory_id"], "game": metadata["game"],
                            "effect": float(panel[action]["j_final"] - panel[reference]["j_final"])})
        contrasts[label] = _bootstrap_paired(records, "effect", bootstrap_repeats)
    margin = float(data["config"].get("parameter_effect_margin", 0.0))
    adequate = [(label, game, stat) for label, games in contrasts.items() for game, stat in games.items()
                if stat["source_histories"] >= 5 and stat["ci95"] is not None]
    positive = [{"contrast": label, "game": game} for label, game, stat in adequate if stat["ci95"][0] > margin]
    if incomplete:
        status = "incomplete"
    elif positive:
        status = "positive"
    elif adequate and len(adequate) == sum(map(len, contrasts.values())) and all(stat["ci95"][1] <= margin for _, _, stat in adequate):
        status = "no_demonstrated_benefit_at_this_budget"
    else:
        status = "uncertain"
    return {"status": status, "positive": status == "positive", "uncertain": status in {"uncertain", "incomplete"},
            "contrasts": contrasts, "positive_contrasts": positive, "margin": margin,
            "interpretation": "conditional_effect_of_declared_weight_intervention; not_a_diagnosis_of_a_broken_parameter_state"}


def analyze(run_dir: Path, bootstrap_repeats=1000) -> dict:
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")
    data = _load(run_dir)
    result = {"kind": data["config"].get("kind", "unspecified"), "run_dir": str(Path(run_dir).resolve()),
              "main_menu": list(data["menu"]), "missing_files": data["missing_files"],
              "outcome_rows": len(data["rows"]), "bootstrap_repeats": bootstrap_repeats,
              "population_oracle_estimated": False, "selection_trained": False, "strata": [],
              "worker_progress": data["worker"], "provisional": True,
              "interpretation": "descriptive_screen_with_cross_fitted_estimated_winner; raw_returns_reported_per_game"}
    referenced = {row.get("checkpoint_id") for row in data["rows"] if not row.get("policy_id")}
    reserved_checkpoint_ids = {identifier for identifier, item in data["checkpoints"].items() if item.get("game") in _reserved_games(data)}
    missing_checkpoints = sorted(set(data["checkpoints"]) - referenced - reserved_checkpoint_ids)
    result["checkpoints_without_outcomes"] = missing_checkpoints
    result["reserved_checkpoint_ids_excluded_from_development_analysis"] = sorted(reserved_checkpoint_ids)
    for (mode, budget), rows in sorted(_strata(data).items(), key=lambda pair: str(pair[0])):
        panels = _panels(data, rows)
        complete = _complete(data, panels)
        samples = _cp_means(data, complete)
        pairs = defaultdict(list)
        failures = Counter(str(row.get("failure_reason")) for row in rows if row.get("failure_reason"))
        nonfinite = sum(not _finite(row.get("j_final")) for row in rows)
        budget_mismatches = sum(_budget_mismatch(row) for row in rows)
        missing_action_rows = sum(len(set(data["menu"]) - set(panel)) for panel in panels.values())
        for (identifier, repeat), panel in panels.items():
            if "continue" not in panel or not _valid(panel["continue"]):
                continue
            metadata = _metadata(data, identifier, panel)
            for action in data["menu"]:
                if action not in panel or not _valid(panel[action]):
                    continue
                row = panel[action]
                effect = float(row["j_final"] - panel["continue"]["j_final"])
                threshold = float(data["config"].get("harm_margin", 0.0))
                pairs[action].append({"checkpoint_id": identifier, "trajectory_id": metadata["trajectory_id"], "game": metadata["game"],
                                      "repeat": list(repeat), "effect": effect, "harm": float(effect < -threshold),
                                      "immediate_change": float(row["j_immediate"] - row["j_pre"]) if _finite(row.get("j_immediate")) and _finite(row.get("j_pre")) else None,
                                      "paired_immediate_effect": float(row["j_immediate"] - panel["continue"]["j_immediate"]) if _finite(row.get("j_immediate")) and _finite(panel["continue"].get("j_immediate")) else None,
                                      "auc_effect": float(row["adaptation_auc"] - panel["continue"]["adaptation_auc"]) if _finite(row.get("adaptation_auc")) and _finite(panel["continue"].get("adaptation_auc")) else None})
        action_stats = {action: {"paired_effect": _bootstrap_paired(records, "effect", bootstrap_repeats),
                                 "harm_fraction": _bootstrap_paired(records, "harm", bootstrap_repeats),
                                 "immediate_change": _bootstrap_paired(records, "immediate_change", bootstrap_repeats),
                                 "paired_immediate_effect": _bootstrap_paired(records, "paired_immediate_effect", bootstrap_repeats),
                                 "auc_effect": _bootstrap_paired(records, "auc_effect", bootstrap_repeats)} for action, records in pairs.items()}
        crossfit, choices, skipped = _crossfit(data, samples)
        cf_bootstrap = _bootstrap_crossfit(data, samples, crossfit, bootstrap_repeats)
        cf_stats = cf_bootstrap["advantage_vs_sbs"]
        naive = []
        for item in samples:
            sbs = _single_best([other for other in samples if other["trajectory_id"] != item["trajectory_id"]], data["menu"])
            if sbs is not None:
                naive.append(dict(item, optimistic_gap=float(max(item["returns"]) - item["returns"][data["menu"].index(sbs)])))
        qualification = _qualification(samples)
        minimum = min((len(value) for value in complete.values()), default=0)
        incomplete = bool(data["missing_files"] or missing_checkpoints or missing_action_rows or failures or nonfinite or budget_mismatches or skipped or not samples or not _finite(budget) or budget <= 0)
        gradient_budgets = [row["budget_gradient_updates"] for row in rows if _finite(row.get("budget_gradient_updates"))]
        gate = _gate(incomplete, qualification, cf_stats, choices, minimum, float(data["config"].get("heterogeneity_margin", 0.0)))
        parameter_signal = _parameter_signal(data, panels, bootstrap_repeats, incomplete)
        result["strata"].append({"source_mode": mode, "budget_env_steps": budget,
                                 "gradient_budget_range": [min(gradient_budgets), max(gradient_budgets)] if gradient_budgets else None,
                                 "gradient_budget_warning": len(set(gradient_budgets)) > 1,
                                 "rows": len(rows), "complete_checkpoints": len(complete), "complete_repeat_panels": sum(map(len, complete.values())),
                                 "source_histories": len({item["trajectory_id"] for item in samples}), "minimum_complete_repeats": minimum,
                                 "missing_action_rows": missing_action_rows, "failed_rows_by_reason": dict(failures), "nonfinite_or_missing_final_rows": nonfinite,
                                 "completed_environment_budget_mismatch_rows": budget_mismatches,
                                 "baseline_qualification": qualification, "action_statistics": action_stats,
                                 "naive_same_sample_maximum_gap_vs_heldout_history_sbs": _mean_games(naive, "optimistic_gap"),
                                 "naive_maximum_caveat": "optimistic_in_expectation; not_a_strict_population_oracle_bound",
                                 "cross_fitted_estimated_winner": {"advantage_vs_sbs": cf_stats,
                                     "effect_vs_continue": cf_bootstrap["effect_vs_continue"],
                                     "choices": choices, "skipped_checkpoints": skipped,
                                     "ci_caveat": "histories_and_paired_repeat_vectors_resampled_with_refitted_choices;_fixed_disjoint_repeat_split;_not_simultaneous_or_independent_confirmation"},
                                 "primary_gate": gate, "gate_is_confirmatory": False,
                                 "parameter_repair_signal": parameter_signal,
                                 "uncertain": gate in {"incomplete", "uncertain"},
                                 "gating_caveat": "screening_gate_only; single_repair_is_menu_and_budget_specific; failure_cases_not_imputed_as_success",
                                 "success_pair_only_statistics": bool(failures or nonfinite or missing_action_rows or budget_mismatches)})
    result["primary_gate"] = "incomplete" if not result["strata"] else (result["strata"][0]["primary_gate"] if len(result["strata"]) == 1 else "multiple_strata_see_per_stratum_gates")
    result["uncertain"] = not result["strata"] or any(item["uncertain"] for item in result["strata"])
    result["parameter_repair_signal"] = result["strata"][0]["parameter_repair_signal"] if len(result["strata"]) == 1 else {"status": "see_per_stratum_signals", "positive": False, "uncertain": True}
    return result


def _allowed_features(data, item, age_only=False):
    metadata = item["metadata"]
    raw = data["features"].get(item["checkpoint_id"], {})
    if age_only:
        return {"nominal_age": metadata.get("nominal_age"),
                "environment_steps": metadata.get("environment_steps"),
                "recent_return": raw.get("recent_return", metadata.get("source_eval_return", metadata.get("j_pre"))),
                "recent_return_slope": raw.get("recent_return_slope")}
    forbidden = ("post", "future", "heldout", "oracle", "winner", "repair", "continuation", "j_final", "j_immediate")
    result = {str(key): value for key, value in raw.items()
              if _finite(value) or value is None
              if not any(word in str(key).lower() for word in forbidden)
              if str(key).lower() not in {"game", "game_id", "environment_id", "training_seed", "trajectory_id"}}
    result.setdefault("nominal_age", metadata.get("nominal_age"))
    result.setdefault("recent_return", metadata.get("source_eval_return", metadata.get("j_pre")))
    return result


def _matrix(data, samples, columns, age_only):
    return np.asarray([[float(_allowed_features(data, item, age_only).get(column))
                        if _finite(_allowed_features(data, item, age_only).get(column)) else np.nan
                        for column in columns] for item in samples], dtype=float)


def _model(model_name, alpha):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    estimator = (DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=41)
                 if model_name == "tree" else Ridge(alpha=alpha))
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("estimator", estimator)])


def _predictor(data, train, test, model_name, age_only, return_model=False):
    from sklearn.model_selection import GroupKFold
    columns = sorted({key for item in train for key in _allowed_features(data, item, age_only)})
    if not columns:
        columns = ["constant_missing_feature"]
    x_train = _matrix(data, train, columns, age_only)
    x_test = _matrix(data, test, columns, age_only)
    y_train = np.asarray([item["effects"] for item in train], dtype=float)
    groups = np.asarray([item["trajectory_id"] for item in train])
    group_count = len(set(groups))
    alpha = 1.0
    tuning = []
    if model_name != "tree" and group_count >= 2:
        folds = GroupKFold(n_splits=min(5, group_count))
        for candidate in RIDGE_ALPHAS:
            errors = []
            for train_ids, dev_ids in folds.split(x_train, y_train, groups):
                fitted = _model(model_name, candidate).fit(x_train[train_ids], y_train[train_ids])
                errors.append(float(np.mean((fitted.predict(x_train[dev_ids]) - y_train[dev_ids]) ** 2)))
            tuning.append({"alpha": candidate, "grouped_development_mse": float(np.mean(errors))})
        alpha = min(tuning, key=lambda item: item["grouped_development_mse"])["alpha"]
    fitted = _model(model_name, alpha).fit(x_train, y_train)
    predictions = fitted.predict(x_test)
    audit = {"feature_names": columns, "alpha": alpha if model_name != "tree" else None,
             "max_depth": 2 if model_name == "tree" else None, "inner_tuning": tuning,
             "imputer_statistics": fitted.named_steps["impute"].statistics_.tolist(),
             "scaler_mean": fitted.named_steps["scale"].mean_.tolist(),
             "training_trajectory_ids": sorted(set(groups)),
             "evaluation_trajectory_ids": sorted({item["trajectory_id"] for item in test}),
             "training_games": sorted({item["game"] for item in train}),
             "evaluation_games": sorted({item["game"] for item in test})}
    return (fitted, audit) if return_model else (predictions, audit)


@dataclass
class TrainedSelector:
    pipeline: object
    feature_names: tuple
    menu: tuple
    metadata: dict


def train_selector(run_dir: Path, stratum, family="ridge", feature_set="diagnostic") -> TrainedSelector:
    """Train a deployable frozen policy using declared development games only.

    `stratum` is (source_mode, budget_env_steps), or a dict with those keys.
    No held-out transfer-game outcome is used for preprocessing, tuning or SBS.
    """
    if family not in {"ridge", "tree"} or feature_set not in {"diagnostic", "age_return"}:
        raise ValueError("family must be ridge/tree and feature_set diagnostic/age_return")
    data = _load(run_dir)
    key = (str(stratum["source_mode"]), stratum["budget_env_steps"]) if isinstance(stratum, dict) else (str(stratum[0]), stratum[1])
    allowed_games = _development_games(data)
    reserved_games = _reserved_games(data)
    if not allowed_games:
        raise ValueError("Deploy training requires an explicit config.screen.games or train_games whitelist")
    rows = [row for row in _strata(data).get(key, []) if row.get("game") in set(allowed_games)]
    complete = _complete(data, _panels(data, rows))
    samples = [item for item in _cp_means(data, complete) if item["checkpoint_id"] in data["features"]]
    if len({item["trajectory_id"] for item in samples}) < 3:
        raise ValueError("Deploy training needs at least three complete development source histories")
    fitted, audit = _predictor(data, samples, samples, family, feature_set == "age_return", return_model=True)
    audit["evaluation_trajectory_ids"] = []
    audit["evaluation_games"] = []
    metadata = {"family": family, "feature_set": feature_set, "source_mode": key[0], "budget_env_steps": key[1],
                "kind": data["config"].get("kind", "unspecified"), "development_game_whitelist": list(allowed_games),
                "excluded_reserved_games": sorted(reserved_games),
                "main_menu": list(data["menu"]), "fit_audit": audit, "checkpoint_ids": [item["checkpoint_id"] for item in samples],
                "single_best_repair": _single_best(samples, data["menu"]),
                "deployment_inputs": "pre_repair_features_plus_nominal_age_and_source_eval_return_as_recent_return",
                "heldout_outcomes_used": False}
    return TrainedSelector(fitted, tuple(audit["feature_names"]), data["menu"], metadata)


def choose_action(model: TrainedSelector, features: dict) -> str:
    """Apply frozen preprocessing/model; missing permitted features are imputed."""
    values = dict(features)
    values.setdefault("recent_return", values.get("source_eval_return", values.get("j_pre")))
    x = np.asarray([[float(values.get(name)) if _finite(values.get(name)) else np.nan for name in model.feature_names]], dtype=float)
    prediction = model.pipeline.predict(x)[0]
    if not np.isfinite(prediction).all():
        raise ValueError("Selector returned nonfinite predicted effects")
    return model.menu[int(np.argmax(prediction))]


def fit_selectors(run_dir: Path, bootstrap_repeats=1000, strata=None) -> dict:
    """Nested grouped CV; final returns of each test history are evaluator-only.

    This compares prespecified policy classes. Choosing the best class after
    viewing these scores still requires an independent confirmation cohort.
    """
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    data = _load(run_dir)
    result = {"kind": data["config"].get("kind", "unspecified"), "strata": [],
              "missing_files": data["missing_files"], "hyperparameters": {"ridge_alpha_grid": list(RIDGE_ALPHAS), "tree_max_depth": 2},
              "interpretation": "nested_grouped_cross_validation_not_a_confirmed_unseen_family_result",
              "excluded_reserved_games": sorted(_reserved_games(data)),
              "feature_precondition": "caller_must_collect_diagnostics_before_repair_without_final_oracle_labels"}
    for (mode, budget), rows in sorted(_strata(data).items(), key=lambda pair: str(pair[0])):
        if strata is not None and (mode, budget) not in set(map(tuple, strata)):
            continue
        panels = _panels(data, rows)
        complete = _complete(data, panels)
        samples = [item for item in _cp_means(data, complete) if item["checkpoint_id"] in data["features"]]
        histories = np.asarray([item["trajectory_id"] for item in samples])
        games = np.asarray([item["game"] for item in samples])
        entry = {"source_mode": mode, "budget_env_steps": budget, "complete_checkpoints_with_diagnostics": len(samples),
                 "source_histories": len(set(histories)), "games": len(set(games)), "schemes": {},
                 "selector_gate": "insufficient_source_histories", "gate_is_confirmatory": False}
        if len(set(histories)) < 3:
            result["strata"].append(entry)
            continue
        schemes = {"heldout_source_history": (GroupKFold(n_splits=min(5, len(set(histories)))), histories)}
        if len(set(games)) >= 2:
            schemes["leave_one_game_out"] = (LeaveOneGroupOut(), games)
        for scheme, (splitter, split_groups) in schemes.items():
            records = defaultdict(list)
            audits = defaultdict(list)
            skipped_folds = []
            for fold, (train_ids, test_ids) in enumerate(splitter.split(np.zeros((len(samples), 1)), groups=split_groups)):
                train = [samples[index] for index in train_ids]
                test = [samples[index] for index in test_ids]
                sbs = _single_best(train, data["menu"])
                if len({item["trajectory_id"] for item in train}) < 2 or sbs is None:
                    skipped_folds.append(fold)
                    continue
                choices = {"single_best_repair": [sbs] * len(test)}
                for policy, model_name, age_only in (("age_return_ridge", "ridge", True), ("diagnostic_ridge", "ridge", False), ("diagnostic_tree", "tree", False)):
                    predictions, audit = _predictor(data, train, test, model_name, age_only)
                    choices[policy] = [data["menu"][int(np.argmax(row))] for row in predictions]
                    audits[policy].append(dict(audit, fold=fold))
                audits["single_best_repair"].append({"fold": fold, "chosen_action": sbs,
                    "training_trajectory_ids": sorted({item["trajectory_id"] for item in train}),
                    "evaluation_trajectory_ids": sorted({item["trajectory_id"] for item in test}),
                    "training_games": sorted({item["game"] for item in train}), "evaluation_games": sorted({item["game"] for item in test})})
                for policy, actions in choices.items():
                    for item, action in zip(test, actions, strict=True):
                        for repeat in item["repeats"]:
                            panel = repeat["panel"]
                            records[policy].append({"checkpoint_id": item["checkpoint_id"], "trajectory_id": item["trajectory_id"], "game": item["game"],
                                                    "chosen_action": action, "fold": fold,
                                                    "effect_vs_continue": float(panel[action]["j_final"] - panel["continue"]["j_final"]),
                                                    "advantage_vs_sbs": float(panel[action]["j_final"] - panel[sbs]["j_final"]),
                                                    "harm": float(panel[action]["j_final"] < panel["continue"]["j_final"] - float(data["config"].get("harm_margin", 0))),
                                                    "repeat": list(repeat["repeat"])})
            policy_stats = {}
            age_lookup = {(item["checkpoint_id"], tuple(item["repeat"])): item for item in records["age_return_ridge"]}
            for policy, policy_records in records.items():
                relative = [dict(item, advantage_vs_age_return=item["effect_vs_continue"] - age_lookup[(item["checkpoint_id"], tuple(item["repeat"]))]["effect_vs_continue"])
                            for item in policy_records if (item["checkpoint_id"], tuple(item["repeat"])) in age_lookup]
                policy_stats[policy] = {"effect_vs_continue": _bootstrap_paired(policy_records, "effect_vs_continue", bootstrap_repeats),
                                        "advantage_vs_sbs": _bootstrap_paired(policy_records, "advantage_vs_sbs", bootstrap_repeats),
                                        "advantage_vs_age_return": _bootstrap_paired(relative, "advantage_vs_age_return", bootstrap_repeats),
                                        "harm_fraction": _bootstrap_paired(policy_records, "harm", bootstrap_repeats),
                                        "choices": list({(item["checkpoint_id"], item["fold"]):
                                                         {key: item[key] for key in ("checkpoint_id", "trajectory_id", "game", "fold", "chosen_action")}
                                                         for item in policy_records}.values()),
                                        "fold_audits": audits[policy]}
            entry["schemes"][scheme] = {"policies": policy_stats, "skipped_folds": skipped_folds,
                "outcome_ci_conditioning": "conditional_on_outer_fold_fitted_policies;_nested_paired_evaluation_uncertainty",
                "cross_game_scale": "raw_effect_targets; per_game_evaluation; no_test_fitted_return_normalization"}
        # Conservative exploratory gate. Do not choose a model class by its test score here.
        source_scheme = entry["schemes"].get("heldout_source_history", {})
        if source_scheme.get("skipped_folds") or len(set(histories)) < 5:
            entry["selector_gate"] = "uncertain"
        else:
            ridge = source_scheme.get("policies", {}).get("diagnostic_ridge", {})
            vs_sbs = ridge.get("advantage_vs_sbs", {})
            vs_age = ridge.get("advantage_vs_age_return", {})
            positive = [game for game, stat in vs_sbs.items() if stat["ci95"] and stat["ci95"][0] > 0
                        and game in vs_age and vs_age[game]["ci95"] and vs_age[game]["ci95"][0] > 0
                        and stat["source_histories"] >= 5]
            entry["selector_gate"] = "promising_grouped_development_signal" if positive else "no_demonstrated_increment_over_age_and_sbs"
        entry["policy_choice_requires_independent_confirmation"] = True
        result["strata"].append(entry)
    return result


def build_report(run_dir: Path, output_dir: Path, bootstrap_repeats=1000) -> list[Path]:
    """Write JSON plus headless figures; `analyze`/`fit_selectors` do not write."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptive = analyze(run_dir, bootstrap_repeats)
    eligible = [(item["source_mode"], item["budget_env_steps"]) for item in descriptive["strata"]
                if item["source_mode"] == "natural" and item["primary_gate"] == "heterogeneous"]
    selectors = (fit_selectors(run_dir, bootstrap_repeats, strata=eligible) if eligible else
                 {"kind": descriptive["kind"], "status": "deferred_before_confirmed_signal", "strata": [],
                  "reason": "no_complete_natural_stratum_with_replicated_heterogeneity_gate",
                  "population_oracle_estimated": False})
    files = []
    for name, payload in (("repair_analysis.json", descriptive), ("selector_analysis.json", selectors)):
        path = output_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        files.append(path)
    for index, stratum in enumerate(descriptive["strata"]):
        games = sorted({game for stats in stratum["action_statistics"].values() for game in stats["paired_effect"]})
        if not games:
            continue
        fig, axes = plt.subplots(len(games), 1, figsize=(9, 2.8 * len(games) + 1), squeeze=False)
        for ax, game in zip(axes.flat, games, strict=True):
            entries = [(action, stat["paired_effect"][game]) for action, stat in stratum["action_statistics"].items() if game in stat["paired_effect"]]
            means = [stat["mean"] for _, stat in entries]
            bounds = [stat["ci95"] or [stat["mean"], stat["mean"]] for _, stat in entries]
            # Bootstrap intervals need not contain the observed point estimate.
            ax.scatter(range(len(entries)), means, color="#2563eb", zorder=3)
            for position, bound in enumerate(bounds):
                ax.plot([position, position], bound, color="#2563eb", linewidth=2)
            ax.axhline(0, color="#4b5563", linewidth=1)
            ax.set_xticks(range(len(entries)), [action.replace("_", "\n") for action, _ in entries])
            ax.set_ylabel("Paired final return effect")
            ax.set_title(game, loc="left")
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"{descriptive['kind']}: {stratum['source_mode']}, budget {stratum['budget_env_steps']} transitions\nGate: {stratum['primary_gate']}")
        fig.text(0.06, 0.015, "95% nested bootstrap intervals: full source histories + paired continuation repeats.\nRaw per-game returns; failed/missing pairs excluded and separately block progression. Not a population oracle.", fontsize=9)
        fig.tight_layout(rect=[0, 0.10, 1, 0.90])
        path = output_dir / f"paired_effects_{index}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    files = build_report(args.run_dir, args.output_dir or args.run_dir / "analysis", args.bootstrap_repeats)
    print("\n".join(str(path.resolve()) for path in files))


if __name__ == "__main__":
    main()
