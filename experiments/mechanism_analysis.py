"""Read saved fixed-TD and synthetic probes; never collect or train an agent.

Loss means give equal weight to repeats within a checkpoint, checkpoints within
a source history, and histories within a game. Temporal holdout fit is a
conditional outcome, not policy return or evidence that a teacher is correct.
PNG figures can be regenerated from the exported JSON with --from-summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAIN = ("continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset")
BASELINE = MAIN + ("t_reset", "t_reset_equivalent_schedule", "fresh_network_reference")
TEACHERS = ("old_target", "refreshed_online")
SCENARIOS = ("unchanged", "target_sign_flip", "target_scale_drop", "input_shift")
GROUP_FIELDS = ("domain", "game", "source_mode", "family", "trial_horizon", "dataset",
                "training_teacher", "evaluation_teacher", "split", "metric", "horizon", "lr", "action")
POLICY = {"minimum_histories": 3, "minimum_repeats": 2, "rl_lr_count": 5,
          "synthetic_lr_count": 3, "relative_descriptive_band": 0.05,
          "float32_curve_relative_tolerance": 1e-5, "float64_curve_relative_tolerance": 1e-10}
LABELS = {"continue": "Continue", "optimizer_reset": "Optimizer reset", "head_reset": "Head reset",
          "head_and_optimizer_reset": "Head + optimizer", "t_reset": "t reset",
          "t_reset_equivalent_schedule": "Equivalent LR + eps", "fresh_network_reference": "Initial network"}
PARAMETER_REFERENCES = {"fresh_head_reset": "head_reset", "fresh_head_and_optimizer_reset": "head_and_optimizer_reset",
                        "shrink_perturb": "continue", "injection": "continue"}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _jsonl(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if not line.endswith("\n"):
                break  # Concurrent writer's unfinished final record.
            raise
    return result


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _history_stats(rows, value="value"):
    """Do not promote rows, ages, or continuation repeats to independent histories."""
    nested = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("trajectory_id") and _finite(row.get(value)):
            nested[row["trajectory_id"]][row["checkpoint_id"]].append(float(row[value]))
    history_means = [{"trajectory_id": history,
                      "mean": float(np.mean([np.mean(values) for values in checkpoints.values()])),
                      "checkpoints": len(checkpoints), "rows": sum(map(len, checkpoints.values()))}
                     for history, checkpoints in sorted(nested.items())]
    values = [row["mean"] for row in history_means]
    return {"mean": float(np.mean(values)) if values else None,
            "history_min": min(values) if values else None, "history_max": max(values) if values else None,
            "source_histories": len(values), "history_means": history_means,
            "valid_rows": sum(row["rows"] for row in history_means),
            "excluded_missing_history_rows": sum(not row.get("trajectory_id") for row in rows),
            "dispersion_semantics": "min/max across source-history means; not a confidence interval"}


def _group(rows, fields=GROUP_FIELDS, value="value"):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field) for field in fields)].append(row)
    return [dict(zip(fields, key), **_history_stats(items, value),
                 recorded_rows=len(items), invalid_rows=sum(not _finite(x.get(value)) for x in items))
            for key, items in sorted(buckets.items(), key=lambda item: repr(item[0]))]


def _branch_valid(branch, horizon):
    return (int(branch.get("total_updates", branch.get("training_updates", 0))) >= horizon
            and not branch.get("failure") and branch.get("status", "ok") == "ok")


def _fixed_points(job, branch, family):
    metadata = {key: job.get(key) for key in ("checkpoint_id", "trajectory_id", "game", "source_mode", "repeat")}
    metadata.update(domain="fixed_td", family=family, trial_horizon=None,
                    dataset=branch.get("dataset", "combined"), action=branch.get("action"),
                    training_teacher=branch.get("training_teacher", "old_target"),
                    lr=branch.get("learning_rate_override") or job.get("model_config", {}).get("lr"),
                    branch_status=branch.get("status", "ok"), failure=branch.get("failure"),
                    tape_hash=branch.get("tape_hash"), job_status=job.get("status"))
    result = []
    for horizon, point in branch.get("curves", {}).items():
        horizon = int(horizon)
        for split in ("fit", "heldout"):
            for teacher in TEACHERS:
                metrics = point.get(split, {}).get(teacher, {})
                for metric in ("huber", "mse"):
                    value = metrics.get(metric)
                    finite_point = metrics.get("status", "ok") == "ok" and _finite(value)
                    # Failure rows stay in raw data, and never become eligible successes.
                    valid = finite_point and _branch_valid(branch, horizon)
                    result.append(dict(metadata, evaluation_teacher=teacher, split=split, metric=metric,
                                       horizon=horizon, value=float(value) if valid else None,
                                       raw_value=float(value) if _finite(value) else None, valid=valid))
    return result


def _synthetic_points(raw):
    result = []
    for branch in raw.get("branches", []):
        scenario = branch.get("scenario", "unspecified")
        seed = branch.get("training_seed")
        metadata = dict(domain="synthetic_supervised", game=scenario, source_mode="supervised_control",
                        trajectory_id=f"{scenario}:seed{seed}" if seed is not None else None,
                        checkpoint_id=branch.get("checkpoint_id"), repeat=branch.get("continuation_repeat"),
                        family="lr_sweep", trial_horizon=branch.get("horizon"), dataset="synthetic_future",
                        action=branch.get("action"), training_teacher="synthetic_teacher",
                        evaluation_teacher="synthetic_teacher", metric="mse",
                        lr=branch.get("continuation_lr"), failure=branch.get("failure"),
                        tape_hash=branch.get("tape_sha256"), branch_status="failed" if branch.get("failure") else "ok")
        for point in branch.get("curve", []):
            horizon = int(point["updates"])
            for split, field in (("fit", "training_loss"), ("heldout", "heldout_loss")):
                value = point.get(field)
                valid = _finite(value) and _branch_valid(branch, horizon)
                result.append(dict(metadata, horizon=horizon, split=split, valid=valid,
                                   value=float(value) if valid else None,
                                   raw_value=float(value) if _finite(value) else None))
    return result


def _paired_contrasts(points):
    fields = tuple(field for field in GROUP_FIELDS if field != "action")
    panels = defaultdict(dict)
    for row in points:
        if row["family"] not in ("baseline", "lr_sweep"):
            continue
        key = (row["checkpoint_id"], row["repeat"]) + tuple(row.get(field) for field in fields)
        if row["action"] in panels[key]:
            raise ValueError(f"Duplicate curve point/action: {key}, {row['action']}")
        panels[key][row["action"]] = row
    contrasts = []
    for panel in panels.values():
        baseline = panel.get("continue")
        for action, row in panel.items():
            if action == "continue":
                continue
            matched = baseline is not None and baseline.get("tape_hash") == row.get("tape_hash") and bool(row.get("tape_hash"))
            valid = matched and _finite(row["value"]) and _finite(baseline["value"])
            item = {key: row.get(key) for key in fields + ("checkpoint_id", "trajectory_id", "repeat")}
            item.update(action=action, value=baseline["value"] - row["value"] if valid else None,
                        matched_tape=matched, failure=row.get("failure"), valid_pair=valid,
                        adverse=bool(valid and row["value"] > baseline["value"]))
            contrasts.append(item)
    return {"sign": "continue loss minus repaired loss; positive is better fitting",
            "raw_pairs": contrasts, "groups": _group(contrasts),
            "adverse_pairs": sum(row["adverse"] for row in contrasts),
            "invalid_or_unmatched_pairs": sum(not row["valid_pair"] for row in contrasts)}


def _equivalence_pair(left, right, metadata, synthetic=False, dtype="float32"):
    result = dict(metadata, status="uncertain", dtype=dtype,
                  observable="saved loss curves; not parameter-vector equality",
                  numerical_tolerance=POLICY["float64_curve_relative_tolerance" if dtype == "float64" else "float32_curve_relative_tolerance"])
    if left is None or right is None:
        return dict(result, reason="missing_arm")
    tape_key = "tape_sha256" if synthetic else "tape_hash"
    if not left.get(tape_key) or left.get(tape_key) != right.get(tape_key):
        return dict(result, reason="unmatched_minibatch_tapes")
    if not _branch_valid(left, metadata["requested_updates"]) or not _branch_valid(right, metadata["requested_updates"]):
        return dict(result, reason="failed_or_incomplete_arm")
    vectors = []
    names = []
    if synthetic:
        maps = [{int(point["updates"]): point for point in branch.get("curve", [])} for branch in (left, right)]
    else:
        maps = [{int(step): point for step, point in branch.get("curves", {}).items()} for branch in (left, right)]
    if set(maps[0]) != set(maps[1]):
        return dict(result, reason="unmatched_curve_horizons", left_horizons=sorted(maps[0]), right_horizons=sorted(maps[1]))
    for index, mapping in enumerate(maps):
        values, keys = [], []
        for step, point in sorted(mapping.items()):
            if synthetic:
                for field in ("training_loss", "heldout_loss"):
                    keys.append(f"{step}:{field}")
                    values.append(point.get(field))
            else:
                for split in ("fit", "heldout"):
                    for teacher in TEACHERS:
                        for metric in ("huber", "mse"):
                            keys.append(f"{step}:{split}:{teacher}:{metric}")
                            values.append(point.get(split, {}).get(teacher, {}).get(metric))
        if not all(_finite(value) for value in values):
            return dict(result, reason="nonfinite_curve_values")
        vectors.append(np.asarray(values, dtype=np.float64))
        if index == 0:
            names = keys
    if not len(vectors[0]):
        return dict(result, reason="no_curve_values")
    difference = vectors[0] - vectors[1]
    scale = max(float(np.linalg.norm(vectors[0])), float(np.linalg.norm(vectors[1])), np.finfo(float).tiny)
    relative = float(np.linalg.norm(difference) / scale)
    result.update(status="numerically_consistent" if relative <= result["numerical_tolerance"] else "numerical_discrepancy",
                  reason="matched_saved_observables", horizons=sorted(maps[0]), entries=len(names),
                  absolute_l2=float(np.linalg.norm(difference)), relative_l2=relative,
                  max_absolute_difference=float(np.max(np.abs(difference))), reference_l2=scale,
                  observable_names=names, left_values=vectors[0].tolist(), right_values=vectors[1].tolist())
    if synthetic:
        a, b = left.get("first_update_norm"), right.get("first_update_norm")
        result["first_update_norm_absolute_difference"] = abs(a - b) if _finite(a) and _finite(b) else None
    return result


def _equivalences(jobs, synthetic):
    records = []
    for job in jobs:
        panel = {branch["action"]: branch for branch in job.get("baseline", [])}
        metadata = {field: job.get(field) for field in ("checkpoint_id", "trajectory_id", "game", "repeat")}
        metadata.update(domain="fixed_td", requested_updates=job.get("config", {}).get("baseline_updates", 2000))
        records.append(_equivalence_pair(panel.get("t_reset"), panel.get("t_reset_equivalent_schedule"), metadata,
                                         dtype=job.get("model_config", {}).get("dtype", "float32")))
    panels = defaultdict(dict)
    for branch in synthetic.get("branches", []):
        key = (branch.get("checkpoint_id"), branch.get("continuation_repeat"), branch.get("continuation_lr"), branch.get("horizon"))
        action = branch.get("action")
        if action in panels[key]:
            raise ValueError(f"Duplicate synthetic branch: {key}, {action}")
        panels[key][action] = branch
    for (checkpoint, repeat, rate, horizon), panel in sorted(panels.items(), key=lambda item: repr(item[0])):
        sample = next(iter(panel.values()))
        metadata = dict(domain="synthetic_supervised", checkpoint_id=checkpoint, repeat=repeat, lr=rate,
                        game=sample.get("scenario"), requested_updates=int(horizon))
        records.append(_equivalence_pair(panel.get("t_reset"), panel.get("t_reset_equivalent_schedule"), metadata,
                                         synthetic=True, dtype="float64"))
    finite = [row for row in records if _finite(row.get("relative_l2"))]
    return {"records": records, "counts": dict(Counter(row["status"] for row in records)),
            "max_relative_l2": max((row["relative_l2"] for row in finite), default=None),
            "max_absolute_l2": max((row["absolute_l2"] for row in finite), default=None),
            "claim_limit": "Matching scalar observables support the algebraic control; they do not prove identical weights or live trajectories."}


def _lr_descriptions(points, jobs, synthetic, missing_jobs=()):
    """In-sample LR adjustment is descriptive, and cannot establish mediation."""
    candidates = defaultdict(list)
    for row in points:
        if row["family"] == "lr_sweep" and row["split"] == "heldout" and row["action"] in MAIN:
            if row["domain"] == "synthetic_supervised" and row["horizon"] != row["trial_horizon"]:
                continue
            if row["domain"] == "fixed_td" and row["horizon"] == 0:
                continue
            candidates[(row["domain"], row["game"], row["source_mode"], row["evaluation_teacher"],
                        row["metric"], row["horizon"])].append(row)
    result = []
    for (domain, game, mode, teacher, metric, horizon), rows in sorted(candidates.items(), key=lambda item: repr(item[0])):
        synthetic_domain = domain == "synthetic_supervised"
        relevant = [job for job in jobs if job.get("game") == game and job.get("source_mode") == mode]
        if synthetic_domain:
            rates = sorted(set(synthetic.get("config", {}).get("continuation_lrs", [])))
            default_rate = synthetic.get("config", {}).get("source_lr")
            expected_repeats = int(synthetic.get("config", {}).get("repeats", 2))
        else:
            grids = {tuple(job.get("lr_grid", [])) for job in relevant}
            rates = sorted(set(next(iter(grids)))) if len(grids) == 1 else []
            defaults = {job.get("model_config", {}).get("lr") for job in relevant}
            default_rate = next(iter(defaults)) if len(defaults) == 1 else None
            expected_repeats = len({job.get("repeat") for job in relevant})
        expected_count = POLICY["synthetic_lr_count" if synthetic_domain else "rl_lr_count"]
        panels = defaultdict(dict)
        for row in rows:
            key = (row["checkpoint_id"], row["repeat"])
            panels[key][(row["action"], row["lr"])] = row
        complete = {key: panel for key, panel in panels.items()
                    if rates and all((action, rate) in panel and _finite(panel[(action, rate)]["value"])
                                     for action in MAIN for rate in rates)
                    and all(panel[(action, rates[0])].get("trajectory_id") for action in MAIN)
                    and all(len({panel[(action, rate)].get("tape_hash") for action in MAIN}) == 1
                            and panel[("continue", rate)].get("tape_hash") for rate in rates)}
        by_cp = defaultdict(list)
        for (checkpoint, repeat), panel in complete.items():
            by_cp[checkpoint].append((repeat, panel))
        eligible_cp = {cp: values for cp, values in by_cp.items() if len(values) >= POLICY["minimum_repeats"]}
        histories = {values[0][1][("continue", rates[0])]["trajectory_id"] for values in eligible_cp.values()} if rates else set()
        reasons = []
        if len(rates) != expected_count:
            reasons.append(f"requires_declared_{expected_count}_rate_grid")
        if default_rate not in rates:
            reasons.append("default_rate_not_in_grid_or_mixed_defaults")
        if len(histories) < POLICY["minimum_histories"]:
            reasons.append("fewer_than_three_independent_source_histories")
        if expected_repeats < POLICY["minimum_repeats"] or len(eligible_cp) != len({row["checkpoint_id"] for row in rows}):
            reasons.append("missing_or_failed_repeated_checkpoint_panels")
        if len(complete) != len(panels):
            reasons.append("incomplete_or_failed_lr_panels")
        relevant_ids = {job.get("checkpoint_id") for job in relevant}
        if not synthetic_domain and any(item["checkpoint_id"] in relevant_ids for item in missing_jobs):
            reasons.append("missing_declared_probe_repeats")
        common = dict(domain=domain, game=game, source_mode=mode, evaluation_teacher=teacher,
                      metric=metric, horizon=horizon, lr_grid=rates, default_lr=default_rate,
                      source_histories=len(histories), complete_repeat_panels=len(complete), recorded_repeat_panels=len(panels),
                      status="uncertain" if reasons else "conditional_descriptive", insufficient_reasons=reasons,
                      relative_descriptive_band=POLICY["relative_descriptive_band"],
                      inference="LR minima are selected on these same heldout outcomes; no causal mediation or prospective validation claim.")
        for baseline, repair in (("continue", "optimizer_reset"), ("head_reset", "head_and_optimizer_reset")):
            estimates = []
            for cp, repeats in eligible_cp.items():
                if default_rate not in rates:
                    continue
                scores = {action: {rate: float(np.mean([panel[(action, rate)]["value"] for _, panel in repeats]))
                                   for rate in rates} for action in (baseline, repair)}
                best = {action: min(rates, key=lambda rate: scores[action][rate]) for action in scores}
                sample = repeats[0][1][(baseline, rates[0])]
                estimates.append(dict(checkpoint_id=cp, trajectory_id=sample["trajectory_id"],
                                      default_baseline=scores[baseline][default_rate], default_repair=scores[repair][default_rate],
                                      tuned_baseline=scores[baseline][best[baseline]], tuned_repair=scores[repair][best[repair]],
                                      baseline_best_lr=best[baseline], repair_best_lr=best[repair]))
            stats = {field: _history_stats(estimates, field) for field in
                     ("default_baseline", "default_repair", "tuned_baseline", "tuned_repair")}
            db, dr, tb, tr = (stats[field]["mean"] for field in stats)
            default_effect = db - dr if db is not None else None
            tuned_effect = tb - tr if tb is not None else None
            classification = "uncertain"
            if not reasons and default_effect is not None:
                if default_effect <= POLICY["relative_descriptive_band"] * db:
                    classification = "no_material_default_lr_advantage"
                elif abs(tuned_effect) <= POLICY["relative_descriptive_band"] * tb:
                    classification = "default_advantage_absent_after_in_sample_lr_adjustment"
                elif tuned_effect > 0:
                    classification = "advantage_remains_after_in_sample_lr_adjustment"
                else:
                    classification = "effect_reverses_after_in_sample_lr_adjustment"
            result.append(dict(common, comparison=f"{repair}_vs_{baseline}", classification=classification,
                               default_paired_loss_advantage=default_effect, tuned_paired_loss_advantage=tuned_effect,
                               checkpoint_scores=estimates, statistics=stats))
    return result


def _factorial(points):
    panels = defaultdict(dict)
    for row in points:
        if row["family"] != "data_target_panel" or row["split"] != "heldout":
            continue
        key = (row["checkpoint_id"], row["repeat"], row["game"], row["source_mode"], row["horizon"],
               row["evaluation_teacher"], row["metric"])
        panels[key][(row["dataset"], row["training_teacher"])] = row
    contrasts = []
    required = {(dataset, teacher) for dataset in ("old", "recent") for teacher in TEACHERS}
    for panel in panels.values():
        sample = next(iter(panel.values()))
        item = {field: sample.get(field) for field in ("domain", "game", "source_mode", "checkpoint_id", "trajectory_id",
                                                      "repeat", "horizon", "evaluation_teacher", "metric")}
        valid = required <= set(panel) and all(_finite(panel[key]["value"]) for key in required)
        effects = {"old_data_advantage_old_teacher": None, "old_data_advantage_refreshed_teacher": None,
                   "refresh_advantage_old_data": None, "refresh_advantage_recent_data": None,
                   "refresh_by_recency_interaction": None}
        if valid:
            old_old = panel[("old", "old_target")]["value"]
            old_new = panel[("old", "refreshed_online")]["value"]
            recent_old = panel[("recent", "old_target")]["value"]
            recent_new = panel[("recent", "refreshed_online")]["value"]
            effects.update(old_data_advantage_old_teacher=recent_old - old_old,
                           old_data_advantage_refreshed_teacher=recent_new - old_new,
                           refresh_advantage_old_data=old_old - old_new,
                           refresh_advantage_recent_data=recent_old - recent_new,
                           refresh_by_recency_interaction=(recent_old - recent_new) - (old_old - old_new))
        for effect, value in effects.items():
            contrasts.append(dict(item, effect=effect, value=value, complete_panel=valid))
    fields = ("domain", "game", "source_mode", "horizon", "evaluation_teacher", "metric", "effect")
    return {"raw_contrasts": contrasts, "groups": _group(contrasts, fields),
            "semantics": "Positive advantage means lower heldout loss; interaction is refresh advantage on recent minus old data. Each contrast keeps its evaluation teacher fixed.",
            "claim_limit": "Temporal windows are not a privileged occupancy repair; refreshed online targets are not ground truth."}


def _load_inputs(run_dir):
    config = _json(run_dir / "config.json", {})
    checkpoints = {row["checkpoint_id"]: row for row in _jsonl(run_dir / "checkpoints.jsonl")}
    jobs, sources, rejected, seen = [], [], [], {}
    for path in sorted((run_dir / "mechanisms").glob("*.json")):
        raw_bytes = path.read_bytes()
        job = json.loads(raw_bytes)
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw_bytes).hexdigest()))
        cp = checkpoints.get(job.get("checkpoint_id"), {})
        for field in ("game", "trajectory_id", "source_mode"):
            if field not in job:
                job[field] = cp.get(field)
            elif cp.get(field) is not None and job[field] != cp[field]:
                raise ValueError(f"Mechanism/checkpoint metadata disagree: {path}, {field}")
        key = (job.get("checkpoint_id"), job.get("repeat"))
        if key in seen:
            raise ValueError(f"Duplicate mechanism checkpoint/repeat: {key}")
        seen[key] = path
        if not job.get("checkpoint_id"):
            rejected.append(dict(path=str(path), reason="missing_checkpoint_id"))
        elif job.get("status") == "skipped":
            rejected.append(dict(path=str(path), checkpoint_id=job["checkpoint_id"], reason=job.get("reason", "skipped")))
        else:
            jobs.append(job)
    synthetic_path = run_dir / "synthetic_lr_sweep.json"
    synthetic = _json(synthetic_path, {})
    if synthetic:
        if not str(synthetic.get("kind", "")).startswith("synthetic_supervised"):
            raise ValueError("Unexpected synthetic sweep kind; live return must not enter fitting analysis")
        sources.append(dict(path=str(synthetic_path.resolve()), sha256=hashlib.sha256(synthetic_path.read_bytes()).hexdigest()))
    expected = [(cp["checkpoint_id"], repeat) for cp in checkpoints.values() if cp.get("source_mode") == "natural"
                for repeat in config.get("mechanisms", {}).get("seeds", [])]
    missing = [dict(checkpoint_id=cp, repeat=repeat) for cp, repeat in expected if (cp, repeat) not in seen]
    return jobs, synthetic, sources, rejected, missing


def _parameter_followups(run_dir):
    outcomes = _jsonl(run_dir / "outcomes.jsonl")
    auxiliary = [row for row in outcomes if row.get("repair_id") in PARAMETER_REFERENCES]
    auxiliary_units = {(row.get("checkpoint_id"), row.get("source_mode"), row.get("repeat"), row.get("budget_env_steps"))
                       for row in auxiliary}
    panel = defaultdict(dict)
    for row in outcomes:
        key = (row.get("checkpoint_id"), row.get("source_mode"), row.get("repeat"), row.get("budget_env_steps"))
        if key not in auxiliary_units or row.get("repair_id") not in set(PARAMETER_REFERENCES) | set(PARAMETER_REFERENCES.values()):
            continue
        if row["repair_id"] in panel[key]:
            if panel[key][row["repair_id"]] == row:
                continue
            raise ValueError(f"Conflicting live parameter outcome: {key}, {row['repair_id']}")
        panel[key][row["repair_id"]] = row
    records, pairs = [], []
    for arms in panel.values():
        for action, row in arms.items():
            valid = (not row.get("failure_reason") and _finite(row.get("j_final"))
                     and row.get("completed_env_steps", row.get("budget_env_steps")) == row.get("budget_env_steps"))
            records.append(dict(domain="online_parameter_followup", checkpoint_id=row.get("checkpoint_id"),
                                trajectory_id=row.get("trajectory_id"), game=row.get("game"), repeat=row.get("repeat"),
                                source_mode=row.get("source_mode"), budget_env_steps=row.get("budget_env_steps"),
                                action=action, value=float(row["j_final"]) if valid else None,
                                raw_j_final=row.get("j_final") if _finite(row.get("j_final")) else None,
                                failure=row.get("failure_reason"), valid=valid))
        for action, reference in PARAMETER_REFERENCES.items():
            row = arms.get(action)
            if row is None:
                continue
            baseline = arms.get(reference)
            reasons = []
            if baseline is None:
                reasons.append("missing_reference")
            if row.get("failure_reason"):
                reasons.append("repair_failure")
            if baseline and baseline.get("failure_reason"):
                reasons.append("reference_failure")
            if not _finite(row.get("j_final")) or baseline is not None and not _finite(baseline.get("j_final")):
                reasons.append("missing_or_nonfinite_final_return")
            for candidate in (row, baseline):
                if candidate and candidate.get("completed_env_steps", candidate.get("budget_env_steps")) != candidate.get("budget_env_steps"):
                    reasons.append("incomplete_budget")
            if baseline:
                for field in ("trajectory_id", "game", "continuation_seed", "checkpoint_sha256", "evaluation_seeds"):
                    if row.get(field) != baseline.get(field):
                        reasons.append(f"unmatched_{field}")
            pairs.append(dict(domain="online_parameter_followup", checkpoint_id=row.get("checkpoint_id"),
                              trajectory_id=row.get("trajectory_id"), game=row.get("game"), source_mode=row.get("source_mode"),
                              repeat=row.get("repeat"), budget_env_steps=row.get("budget_env_steps"), action=action,
                              reference=reference, value=float(row["j_final"]-baseline["j_final"]) if not reasons else None,
                              valid_pair=not reasons, invalid_reasons=sorted(set(reasons)),
                              repair_return=row.get("j_final") if _finite(row.get("j_final")) else None,
                              reference_return=baseline.get("j_final") if baseline and _finite(baseline.get("j_final")) else None))
    group_fields = ("domain", "game", "source_mode", "budget_env_steps", "action", "reference")
    summaries = _group(pairs, group_fields)
    for item in summaries:
        matching = [row for row in pairs if all(row.get(field) == item.get(field) for field in group_fields)]
        cp_repeats = defaultdict(set)
        for row in matching:
            if row["valid_pair"]:
                cp_repeats[row["checkpoint_id"]].add(row["repeat"])
        minimum = min(map(len, cp_repeats.values()), default=0)
        reasons = []
        if item["source_histories"] < 5:
            reasons.append("fewer_than_five_source_histories")
        if minimum < 4 or len(cp_repeats) != len({row["checkpoint_id"] for row in matching}):
            reasons.append("fewer_than_four_complete_repeats_per_checkpoint")
        if any(not row["valid_pair"] for row in matching):
            reasons.append("failed_or_unmatched_pairs")
        item.update(status="uncertain" if reasons else "conditional_descriptive", insufficient_reasons=reasons,
                    minimum_complete_repeats=minimum, adverse_valid_pairs=sum(row["valid_pair"] and row["value"] < 0 for row in matching),
                    failed_or_unmatched_pairs=sum(not row["valid_pair"] for row in matching))
    return {"outcome": "independent_capped_policy_evaluation_raw_return", "raw_outcomes": records,
            "raw_pairs": pairs, "return_statistics": _group(records, ("domain", "game", "source_mode", "budget_env_steps", "action")),
            "paired_return_effects": summaries, "sign": "repair raw return minus reference raw return; positive is better",
            "recorded_auxiliary_rows": len(auxiliary), "failed_or_invalid_recorded_rows": sum(not row["valid"] for row in records),
            "failure_reasons": dict(Counter(str(row.get("failure_reason")) for row in auxiliary if row.get("failure_reason"))),
            "policy": {"minimum_histories": 5, "minimum_complete_repeats_per_checkpoint": 4},
            "limitations": ["Live return is kept separate from frozen-TD and supervised fitting outcomes.",
                            "Injection is a head-only trainable residual with frozen references on a shared encoder; it changes parameterization and optimizer bindings and is not a pure weight intervention.",
                            "Fresh head and shrink-and-perturb effects can interact with retained optimizer moments and target lag.",
                            "Conditional descriptive effects do not demonstrate unseen-environment selector performance."]}


def summarize_mechanisms(run_dir, output_dir):
    """Export auditable loss summaries/figures from completed artifacts only."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    jobs, synthetic, sources, rejected, missing = _load_inputs(run_dir)
    parameter_live = _parameter_followups(run_dir)
    live_path = run_dir / "outcomes.jsonl"
    if live_path.exists():
        sources.append(dict(path=str(live_path.resolve()), sha256=hashlib.sha256(live_path.read_bytes()).hexdigest()))
    points = [point for job in jobs for family in ("baseline", "lr_sweep", "data_target_panel")
              for branch in job.get(family, []) for point in _fixed_points(job, branch, family)]
    synthetic_points = _synthetic_points(synthetic)
    all_points = points + synthetic_points
    fixed_branches = [branch for job in jobs for family in ("baseline", "lr_sweep", "data_target_panel")
                      for branch in job.get(family, [])]
    coverage = []
    for job in jobs:
        branches = {branch["action"]: branch for branch in job.get("baseline", [])}
        missing_curves = [{"action": action, "horizon": horizon} for action in BASELINE for horizon in (100, 500, 2000)
                          if action not in branches or str(horizon) not in branches[action].get("curves", {})]
        coverage.append(dict(checkpoint_id=job["checkpoint_id"], repeat=job.get("repeat"), game=job.get("game"),
                             missing_standard_baseline_curves=missing_curves,
                             recorded_baseline_actions=sorted(branches), lr_grid=job.get("lr_grid", []),
                             recorded_data_target_cells=len(job.get("data_target_panel", []))))
    summary = {
        "schema_version": 1, "kind": "separate_fixed_TD_synthetic_fit_and_parameter_live_return", "run_dir": str(run_dir.resolve()),
        "policy": POLICY, "sources": sources, "raw_points": all_points, "curve_statistics": _group(all_points),
        "paired_action_contrasts": _paired_contrasts(all_points), "t_reset_equivalence": _equivalences(jobs, synthetic),
        "lr_adjustment_descriptions": _lr_descriptions(all_points, jobs, synthetic, missing), "data_target_factorial": _factorial(points),
        "parameter_live_followups": parameter_live,
        "coverage": {"fixed_probe_files": len(jobs), "rejected_or_skipped_files": rejected,
                     "missing_declared_probe_jobs": missing, "jobs": coverage,
                     "recorded_branch_statuses": dict(Counter(branch.get("status", "ok") for branch in fixed_branches)),
                     "failed_fixed_branches": sum(bool(branch.get("failure")) or branch.get("status", "ok") != "ok" for branch in fixed_branches),
                     "synthetic_file_present": bool(synthetic), "synthetic_recorded_branches": len(synthetic.get("branches", [])),
                     "synthetic_failed_branches": sum(bool(branch.get("failure")) for branch in synthetic.get("branches", [])),
                     "synthetic_completed_checkpoints": synthetic.get("completed_checkpoints", []),
                     "synthetic_config": synthetic.get("config", {})},
        "costs": {"fixed_total_optimizer_updates": sum(int(job.get("total_updates", 0)) for job in jobs),
                  "fixed_attempted_optimizer_updates": sum(int(job.get("attempted_updates", 0)) for job in jobs),
                  "fixed_environment_interactions": sum(int(job.get("environment_interactions", 0)) for job in jobs),
                  "synthetic_actual_optimizer_updates_including_sources_and_executed_prefixes": synthetic.get("actual_optimizer_updates"),
                  "synthetic_branch_optimizer_updates": sum(int(branch.get("training_updates", 0)) for branch in synthetic.get("branches", [])),
                  "analysis_optimizer_updates": 0, "analysis_environment_interactions": 0},
        "live_return": {"analyzed": bool(parameter_live["recorded_auxiliary_rows"]), "section": "parameter_live_followups",
                        "reason": "Only separately recorded parameter follow-up policy evaluations enter live return; fit losses cannot substitute for them."},
        "limitations": ["All loss outcomes are conditional on recorded inputs and cached pre-repair teachers.",
                        "Fitting improvements do not identify the underlying failure cause or establish live repair utility.",
                        "In-sample LR minima and empirical best arms are optimistic descriptive comparisons.",
                        "Source histories are grouping units; repeats, source ages, teachers and horizons are not independent replicates.",
                        "Failed/missing arms are retained and block descriptive LR classification; no failure penalty is invented.",
                        "fresh_network_reference uses the source initializer and is not a new initialization draw.",
                        "t-reset curves test numerical consistency of saved observables; scalar losses cannot establish parameter equality."]}
    summary["artifacts"] = {"json": "mechanism_summary.json", "figures": ["mechanism_fixed_td.png", "mechanism_lr_sweep.png",
                              "mechanism_data_target.png", "mechanism_t_equivalence.png", "mechanism_synthetic_lr.png"]}
    if parameter_live["recorded_auxiliary_rows"]:
        summary["artifacts"]["figures"].append("parameter_live_followups.png")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "mechanism_summary.json", summary)
    render_summary(summary, output_dir)
    return summary


def render_summary(summary, output_dir):
    """Rebuild scientific figures exclusively from exported numeric JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = summary.get("curve_statistics", [])
    colors = dict(zip(BASELINE, ("#475569", "#2563eb", "#c2410c", "#047857", "#9333ea", "#db2777", "#a16207")))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10})

    def save(fig, name, title, footer):
        fig.suptitle(title, x=.08, ha="left", fontsize=13)
        fig.text(.08, .015, footer, fontsize=8, va="bottom")
        fig.tight_layout(rect=[.015, .11, 1, .92])
        fig.savefig(output_dir / name, dpi=170)
        plt.close(fig)

    def placeholder(ax, message="No complete finite observations saved"):
        ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", wrap=True)
        ax.set_axis_off()

    def draw_curve(ax, rows, x_field, actions):
        plotted = False
        for action in actions:
            values = sorted((row for row in rows if row["action"] == action and row.get("mean") is not None),
                            key=lambda row: row[x_field])
            if not values:
                continue
            xs, means = [row[x_field] for row in values], [row["mean"] for row in values]
            low = [row["mean"] - row["history_min"] for row in values]
            high = [row["history_max"] - row["mean"] for row in values]
            ax.errorbar(xs, means, yerr=[low, high], marker="o", markersize=3, linewidth=1.2,
                        capsize=2, color=colors.get(action), label=LABELS.get(action, action))
            plotted = True
        if plotted:
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel("Heldout MSE (lower is better)")
            ax.legend(fontsize=7, frameon=False)
        else:
            placeholder(ax)

    games = sorted({row["game"] for row in stats if row["domain"] == "fixed_td" and row.get("game")}) or ["No saved games"]
    fig, axes = plt.subplots(len(games), 2, figsize=(11.5, max(3.5, len(games)*3.1)), squeeze=False)
    for game, axis_row in zip(games, axes):
        for teacher, ax in zip(TEACHERS, axis_row):
            rows = [row for row in stats if row["domain"] == "fixed_td" and row["game"] == game
                    and row["family"] == "baseline" and row["split"] == "heldout" and row["metric"] == "mse"
                    and row["evaluation_teacher"] == teacher]
            draw_curve(ax, rows, "horizon", BASELINE)
            ax.set_title(f"{game} | evaluated against {teacher}", loc="left")
            ax.set_xlabel("Actual learner updates; frozen data and targets")
    save(fig, "mechanism_fixed_td.png", "Fixed-TD fitting after repair",
         "Equal source-history weight; bars show min/max history means, not confidence intervals.\nFit is not policy return. Old and refreshed teachers are both pre-repair bootstrap functions.")

    fig, axes = plt.subplots(len(games), 2, figsize=(11.5, max(3.5, len(games)*3.1)), squeeze=False)
    for game, axis_row in zip(games, axes):
        for teacher, ax in zip(TEACHERS, axis_row):
            rows = [row for row in stats if row["domain"] == "fixed_td" and row["game"] == game
                    and row["family"] == "lr_sweep" and row["split"] == "heldout" and row["metric"] == "mse"
                    and row["horizon"] > 0 and row["evaluation_teacher"] == teacher]
            if rows:
                last = max(row["horizon"] for row in rows)
                rows = [row for row in rows if row["horizon"] == last]
                draw_curve(ax, rows, "lr", MAIN)
                ax.set_xscale("log")
            else:
                placeholder(ax)
            ax.set_title(f"{game} | {teacher}", loc="left")
            ax.set_xlabel("Continuation learning rate (log scale)")
    save(fig, "mechanism_lr_sweep.png", "Learning-rate sensitivity of four repairs",
         "Saved longest LR-panel horizon; equal source-history weight; whiskers are min/max, not CI.\nChoosing minima on these same heldout curves is descriptive and optimistic; it does not establish a causal mechanism.")

    fig, axes = plt.subplots(len(games), 2, figsize=(11.5, max(3.5, len(games)*3.1)), squeeze=False)
    labels = [(dataset, teacher) for dataset in ("old", "recent") for teacher in TEACHERS]
    for game, axis_row in zip(games, axes):
        for teacher, ax in zip(TEACHERS, axis_row):
            rows = [row for row in stats if row["domain"] == "fixed_td" and row["game"] == game
                    and row["family"] == "data_target_panel" and row["split"] == "heldout" and row["metric"] == "mse"
                    and row["horizon"] > 0 and row["evaluation_teacher"] == teacher]
            if rows:
                last = max(row["horizon"] for row in rows)
                rows = [row for row in rows if row["horizon"] == last]
                cells = {(row["dataset"], row["training_teacher"]): row for row in rows if row["mean"] is not None}
                for x, cell in enumerate(labels):
                    if cell in cells:
                        item = cells[cell]
                        ax.bar(x, item["mean"], width=.65, color="#2563eb" if cell[1] == "old_target" else "#047857",
                               yerr=[[item["mean"]-item["history_min"]], [item["history_max"]-item["mean"]]], capsize=2)
                ax.set_xticks(range(4), [f"{dataset}\n{train.replace('_', ' ')}" for dataset, train in labels], fontsize=7)
                ax.set_ylabel("Common heldout MSE")
            else:
                placeholder(ax)
            ax.set_title(f"{game} | evaluation teacher: {teacher}", loc="left")
    save(fig, "mechanism_data_target.png", "Crossed recorded data and cached training teacher",
         "All four cells are evaluated against each common heldout teacher; whiskers are history min/max.\nOlder data and refreshed online targets are interventions, without privileged correctness labels.")

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    records = summary.get("t_reset_equivalence", {}).get("records", [])
    finite = [row for row in records if _finite(row.get("relative_l2"))]
    nonzero = [(index, row["relative_l2"]) for index, row in enumerate(finite) if row["relative_l2"] > 0]
    if nonzero:
        ax.scatter([item[0] for item in nonzero], [item[1] for item in nonzero], s=16, color="#9333ea")
        ax.set_yscale("log")
        ax.set_xlabel("Matched checkpoint × repeat × LR × requested horizon pair")
        ax.set_ylabel("L2(curve difference) / max L2(curves)")
    elif finite:
        placeholder(ax, "All matched saved curve vectors have exactly zero L2 difference")
    else:
        placeholder(ax, "No complete matched t-reset/control curve pairs")
    save(fig, "mechanism_t_equivalence.png", "Numerical t-reset versus equivalent LR + epsilon schedule",
         f"Exact floating-point values are compared without rounding. Complete pairs: {len(finite)}; exact zeros: {len(finite)-len(nonzero)}.\nLoss-vector agreement does not prove identical parameter vectors or greedy-policy trajectories.")

    scenarios = sorted({row["game"] for row in stats if row["domain"] == "synthetic_supervised"}) or ["No synthetic sweep"]
    horizons = sorted({row["trial_horizon"] for row in stats if row["domain"] == "synthetic_supervised" and row["trial_horizon"] is not None}) or [100, 500]
    fig, axes = plt.subplots(len(scenarios), len(horizons), figsize=(5.4*len(horizons), max(3.5, 2.8*len(scenarios))), squeeze=False)
    for scenario, axis_row in zip(scenarios, axes):
        for horizon, ax in zip(horizons, axis_row):
            rows = [row for row in stats if row["domain"] == "synthetic_supervised" and row["game"] == scenario
                    and row["split"] == "heldout" and row["horizon"] == horizon and row["trial_horizon"] == horizon]
            draw_curve(ax, rows, "lr", MAIN + ("t_reset", "t_reset_equivalent_schedule"))
            if rows:
                ax.set_xscale("log")
            ax.set_title(f"{scenario} | {horizon} optimizer updates", loc="left")
            ax.set_xlabel("Continuation learning rate (log scale)")
    save(fig, "mechanism_synthetic_lr.png", "Synthetic supervised continuation-LR controls (not RL)",
         "Equal source-history weight; ages remain grouped within their training seed; whiskers show min/max.\n100-step executions and 500-step executions are counted separately. Failed arms remain in raw JSON and coverage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-summary", type=Path, help="Regenerate figures only from exported numeric JSON")
    args = parser.parse_args()
    if args.from_summary:
        render_summary(_json(args.from_summary), args.output_dir)
    elif args.run_dir:
        summary = summarize_mechanisms(args.run_dir, args.output_dir)
        print(json.dumps({"fixed_probe_files": summary["coverage"]["fixed_probe_files"],
                          "raw_points": len(summary["raw_points"]), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    else:
        parser.error("Supply --run-dir or --from-summary")


if __name__ == "__main__":
    main()
