"""Evaluate logged outcomes of frozen repair policies without running learners.

The input protocol must have been frozen before each evaluation began. Policies
are compared within declared resource strata, using paired outcomes and source
trajectory resampling. This module neither fits policies nor estimates an oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np


POLICY_IDS = ("continue", "SBS", "age_return_ridge", "diagnostic_ridge", "diagnostic_tree", "chooser")
ROW_FIELDS = (
    "policy_id", "selected_action", "phase", "game", "trajectory_id", "repeat", "j_final",
    "failure_reason", "total_budget_gradient_updates", "probe_gradient_updates",
    "continuation_gradient_updates", "wall_time_sec", "protocol_hash", "evaluation_started_at",
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def protocol_hash(envelope: Mapping) -> str:
    """Hash the complete envelope, including freeze time, except its hash field."""
    payload = {key: value for key, value in envelope.items() if key != "protocol_hash"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value, field):
    if _finite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid {field}: {value!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"{field} requires a timezone or Unix timestamp")
        return parsed.timestamp()
    raise ValueError(f"Missing or invalid {field}")


def _strict_json(text, source):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {source}; use null plus failure_reason")
    return json.loads(text, parse_constant=reject_constant)


def validate_frozen_protocol(path: str | Path) -> dict:
    """Verify envelope integrity, declarations and optional artifact hashes."""
    path = Path(path)
    envelope = _strict_json(path.read_text(encoding="utf-8"), path)
    if envelope.get("schema_version") != 1 or not isinstance(envelope.get("protocol"), dict):
        raise ValueError("Frozen transfer protocol requires schema_version=1 and a protocol mapping")
    actual = protocol_hash(envelope)
    if envelope.get("protocol_hash") != actual:
        raise ValueError("Frozen protocol hash mismatch")
    frozen_at = _timestamp(envelope.get("frozen_at"), "frozen_at")
    payload = envelope["protocol"]
    development = payload.get("development_games")
    reserved = payload.get("reserved_games")
    if not isinstance(development, list) or not development or not isinstance(reserved, list) or not reserved:
        raise ValueError("Explicit nonempty development_games and reserved_games are required")
    if any(not isinstance(game, str) or not game for game in development + reserved):
        raise ValueError("Game whitelists must contain nonempty names")
    if len(set(development)) != len(development) or len(set(reserved)) != len(reserved):
        raise ValueError("Game whitelist contains duplicates")
    if set(development) & set(reserved):
        raise ValueError("Development and reserved games must be disjoint")
    policies = payload.get("policy_ids", list(POLICY_IDS))
    if not isinstance(policies, list) or len(set(policies)) != len(policies) or not set(policies) <= set(POLICY_IDS):
        raise ValueError("Invalid or duplicate frozen policy_ids")
    if "continue" not in policies or "SBS" not in policies or "age_return_ridge" not in policies:
        raise ValueError("Frozen comparisons require continue, SBS and age_return_ridge")
    menu = payload.get("main_menu")
    if not isinstance(menu, list) or not menu or len(set(menu)) != len(menu):
        raise ValueError("A nonempty frozen main_menu is required")
    if any(not isinstance(action, str) or not action for action in menu):
        raise ValueError("main_menu actions must be nonempty strings")
    if "continue" not in menu:
        raise ValueError("The frozen menu must include continue")
    margins = payload.get("harm_margin", 0.0)
    values = list(margins.values()) if isinstance(margins, dict) else [margins]
    if any(not _finite(value) or value < 0 for value in values):
        raise ValueError("harm_margin must be finite and nonnegative")
    # When fit audits are included, enforce their development-game provenance.
    def check_training_games(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"training_games", "development_game_whitelist"}:
                    if not isinstance(item, list) or not set(item) <= set(development):
                        raise ValueError("Frozen policy fit audit contains a non-development game")
                check_training_games(item)
        elif isinstance(value, list):
            for item in value:
                check_training_games(item)
    check_training_games(payload)
    verified = []
    for artifact in payload.get("artifacts", []):
        if not isinstance(artifact, dict) or "path" not in artifact or "sha256" not in artifact:
            raise ValueError("Each optional artifact requires path and sha256")
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        h = hashlib.sha256()
        with artifact_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != artifact["sha256"]:
            raise ValueError(f"Frozen artifact hash mismatch: {artifact_path}")
        verified.append(str(artifact_path.resolve()))
    return {"envelope": envelope, "protocol_hash": actual, "frozen_at_epoch": frozen_at,
            "development_games": development, "reserved_games": reserved, "policy_ids": policies,
            "verified_artifacts": verified,
            "validation_scope": "envelope_hash_and_ordering; file_hashes_verified_only_for_explicit_artifacts"}


def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], [f"Missing outcome file: {path.name}"]
    records, warnings = [], []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = _strict_json(line, f"{path}:{line_number}")
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    warnings.append("Ignored an incomplete final outcome line")
                    break
                raise
            if not isinstance(record, dict):
                raise ValueError(f"Outcome row must be a mapping: {path}:{line_number}")
            records.append(record)
    return records, warnings


def _validate_row(row: dict, frozen: dict) -> dict:
    missing = set(ROW_FIELDS) - set(row)
    if missing:
        raise ValueError(f"Transfer row missing fields: {sorted(missing)}")
    if row["protocol_hash"] != frozen["protocol_hash"]:
        raise ValueError("Transfer row uses a different frozen protocol hash")
    started = _timestamp(row["evaluation_started_at"], "evaluation_started_at")
    if started < frozen["frozen_at_epoch"]:
        raise ValueError("Evaluation began before the protocol was frozen")
    for field in ("evaluation_finished_at", "completed_at"):
        if field in row and _timestamp(row[field], field) < started:
            raise ValueError("Evaluation finish precedes evaluation start")
    if row["policy_id"] not in frozen["policy_ids"]:
        raise ValueError(f"Policy absent from frozen policy_ids: {row['policy_id']}")
    for field in ("phase", "game", "trajectory_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"Transfer {field} must be a nonempty string")
    if row["game"] in frozen["development_games"]:
        split = "development"
    elif row["game"] in frozen["reserved_games"]:
        split = "reserved"
    else:
        raise ValueError(f"Undeclared evaluation game: {row['game']}")
    declared_split = row.get("evaluation_split")
    phase_split = row["phase"] if row["phase"] in {"development", "reserved"} else None
    if (declared_split is not None and declared_split != split) or (phase_split is not None and phase_split != split):
        raise ValueError("Evaluation phase/split disagrees with development/reserved game whitelist")
    if row["failure_reason"] is not None and not isinstance(row["failure_reason"], str):
        raise ValueError("failure_reason must be a string or null")
    if row["j_final"] is not None and not _finite(row["j_final"]):
        raise ValueError("j_final must be finite or null; failures are not successful zero returns")
    action = row["selected_action"]
    menu = frozen["envelope"]["protocol"]["main_menu"]
    if action not in menu and not (action is None and row["failure_reason"]):
        raise ValueError("selected_action is outside the frozen menu")
    if row["policy_id"] == "continue" and action != "continue":
        raise ValueError("continue policy must select continue")
    for field in ("total_budget_gradient_updates", "probe_gradient_updates", "continuation_gradient_updates"):
        value = row[field]
        if not _finite(value) or value < 0 or int(value) != value:
            raise ValueError(f"{field} must be a nonnegative integer")
    if row["probe_gradient_updates"] + row["continuation_gradient_updates"] > row["total_budget_gradient_updates"]:
        raise ValueError("Probe plus continuation updates exceed the declared total budget")
    for field in ("budget_env_steps", "continuation_env_steps"):
        if field in row and (not _finite(row[field]) or row[field] < 0 or int(row[field]) != row[field]):
            raise ValueError(f"{field} must be a nonnegative integer when recorded")
    if not _finite(row["wall_time_sec"]) or row["wall_time_sec"] < 0:
        raise ValueError("wall_time_sec must be finite and nonnegative")
    row = dict(row, evaluation_split=split)
    row["_valid"] = not bool(row["failure_reason"]) and _finite(row["j_final"])
    row["_failure"] = row["failure_reason"] or (None if row["_valid"] else "missing_final_return")
    return row


def _stratum(row):
    # budget_env_steps is a declared allowance. Actual chooser continuation steps
    # can be 48k while both methods have a 50k allowance and 50k update allowance.
    return (row["phase"], row["evaluation_split"], str(row.get("source_mode", "unspecified")),
            int(row["total_budget_gradient_updates"]), row.get("budget_env_steps"))


def _panel_key(row):
    checkpoint = row.get("checkpoint_id")
    if checkpoint is None:
        checkpoint = row.get("nominal_age", row.get("checkpoint_age", "one_checkpoint"))
    repeat = json.dumps(row["repeat"], sort_keys=True, separators=(",", ":"))
    return (row["game"], row["trajectory_id"], str(checkpoint), repeat)


def _point(records, field):
    checkpoints = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in records:
        if _finite(row.get(field)):
            checkpoints[row["game"]][row["trajectory_id"]][row.get("checkpoint_id", "one_checkpoint")].append(float(row[field]))
    histories = {game: {history: [float(np.mean(values)) for values in cp.values()]
                        for history, cp in groups.items()}
                 for game, groups in checkpoints.items()}
    per_game = {game: float(np.mean([np.mean(values) for values in groups.values()]))
                for game, groups in histories.items()}
    return (float(np.mean(list(per_game.values()))) if per_game else None), per_game, histories


def _bootstrap(records, field, repeats, seed):
    point, per_game, histories = _point(records, field)
    counts = {game: len(groups) for game, groups in histories.items()}
    enough = bool(counts) and all(count >= 2 for count in counts.values())
    rng = np.random.RandomState(seed)
    draws, game_draws = [], defaultdict(list)
    if repeats and enough:
        values = {game: np.asarray([np.mean(rows) for rows in groups.values()], dtype=float)
                  for game, groups in histories.items()}
        for _ in range(repeats):
            game_values = {}
            for game in sorted(values):
                array = values[game]
                game_values[game] = float(np.mean(array[rng.randint(len(array), size=len(array))]))
                game_draws[game].append(game_values[game])
            draws.append(float(np.mean(list(game_values.values()))))
    return {
        "mean": point, "ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
        "per_game": {game: {"mean": value,
                            "ci95": np.quantile(game_draws[game], [0.025, 0.975]).tolist() if game_draws[game] else None,
                            "source_histories": counts[game], "uncertain": counts[game] < 5}
                     for game, value in sorted(per_game.items())},
        "source_histories_per_game": counts, "valid_paired_rows": len(records),
        "bootstrap_repeats": repeats, "resampling_unit": "whole_source_trajectory_within_fixed_game",
        "weighting": "equal_repeats_and_checkpoints_within_history_then_equal_histories_then_equal_games",
        "uncertain": not enough or any(count < 5 for count in counts.values()),
        "ci_unavailable_reason": None if draws else
                                 ("bootstrap_disabled" if repeats == 0 else "need_two_valid_histories_per_game"),
        "ci_caveat": "conditional_on_frozen_policies_and_successful_pairs; evaluated_games_are_fixed; no refitting or game-population inference",
    }


def _comparison_ids(policies):
    result = []
    for policy in policies:
        if policy != "continue":
            result.append((policy, "continue"))
        if policy not in {"continue", "SBS"}:
            result.append((policy, "SBS"))
        if policy.startswith("diagnostic_"):
            result.append((policy, "age_return_ridge"))
    return result


def _harm_margin(payload, game):
    margin = payload.get("harm_margin", 0.0)
    return float(margin.get(game, 0.0)) if isinstance(margin, dict) else float(margin)


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def summarize_transfer(run_dir: str | Path, bootstrap_repeats: int = 1000) -> dict:
    """Read frozen outcomes; write only two derived reports in run_dir."""
    if isinstance(bootstrap_repeats, bool) or int(bootstrap_repeats) != bootstrap_repeats or bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be a nonnegative integer")
    bootstrap_repeats = int(bootstrap_repeats)
    run_dir = Path(run_dir).resolve()
    frozen = validate_frozen_protocol(run_dir / "transfer_protocol.json")
    raw_rows, warnings = _read_rows(run_dir / "transfer_outcomes.jsonl")
    rows = [_validate_row(row, frozen) for row in raw_rows]
    game_by_history = {}
    for row in rows:
        previous = game_by_history.setdefault(row["trajectory_id"], row["game"])
        if previous != row["game"]:
            raise ValueError("A source trajectory_id belongs to multiple games; use unique source-history IDs")
    strata = defaultdict(list)
    for row in rows:
        strata[_stratum(row)].append(row)
    paired_rows, summaries = [], []
    payload = frozen["envelope"]["protocol"]
    for stratum, stratum_rows in sorted(strata.items(), key=lambda item: str(item[0])):
        phase, split, mode, gradient_budget, env_budget = stratum
        descriptor = {"phase": phase, "evaluation_split": split, "source_mode": mode,
                      "total_budget_gradient_updates": gradient_budget, "budget_env_steps": env_budget}
        expected_games = frozen["reserved_games"] if split == "reserved" else frozen["development_games"]
        absent_games = sorted(set(expected_games) - {row["game"] for row in stratum_rows})
        panels = defaultdict(dict)
        for row in stratum_rows:
            key = _panel_key(row)
            if row["policy_id"] in panels[key]:
                raise ValueError(f"Duplicate frozen policy outcome: {key}, {row['policy_id']}")
            panels[key][row["policy_id"]] = row
        comparisons = {}
        for candidate, reference in _comparison_ids(frozen["policy_ids"]):
            comparison = f"{candidate}_minus_{reference}"
            pairs = []
            for key, panel in sorted(panels.items()):
                left, right = panel.get(candidate), panel.get(reference)
                game, history, checkpoint, repeat = key
                problems = []
                for label, item in (("candidate", left), ("reference", right)):
                    if item is None:
                        problems.append(f"missing_{label}")
                    elif not item["_valid"]:
                        problems.append(f"{label}:{item['_failure']}")
                effect = float(left["j_final"] - right["j_final"]) if not problems else None
                pair = dict(descriptor, protocol_hash=frozen["protocol_hash"], comparison=comparison,
                            candidate_policy=candidate, reference_policy=reference,
                            game=game, trajectory_id=history, checkpoint_id=checkpoint,
                            repeat=json.loads(repeat), candidate_action=left["selected_action"] if left else None,
                            reference_action=right["selected_action"] if right else None,
                            candidate_j_final=left["j_final"] if left else None,
                            reference_j_final=right["j_final"] if right else None,
                            pair_status="ok" if not problems else "unavailable", unavailable_reasons=problems,
                            effect=effect, harm=float(effect < -_harm_margin(payload, game)) if effect is not None else None)
                pairs.append(pair)
            valid = [pair for pair in pairs if pair["pair_status"] == "ok"]
            seed = int(hashlib.sha256((json.dumps(descriptor, sort_keys=True) + comparison).encode()).hexdigest()[:8], 16)
            effect_stats = _bootstrap(valid, "effect", bootstrap_repeats, seed)
            harm_stats = _bootstrap(valid, "harm", bootstrap_repeats, seed)
            for statistics in (effect_stats, harm_stats):
                missing_games = sorted(set(expected_games) - set(statistics["per_game"]))
                statistics["missing_evaluation_games"] = missing_games
                if len(valid) != len(pairs) or missing_games:
                    statistics["uncertain"] = True
                if missing_games:
                    statistics["observed_games_macro_mean"] = statistics["mean"]
                    statistics["mean"] = None
                    statistics["ci95"] = None
                    statistics["ci_unavailable_reason"] = "missing_valid_pairs_for_a_declared_evaluation_game"
                    for game in missing_games:
                        statistics["per_game"][game] = {"mean": None, "ci95": None,
                                                        "source_histories": 0, "uncertain": True}
            comparisons[comparison] = {
                "paired_effect": effect_stats,
                "harm_fraction": harm_stats,
                "requested_pairs": len(pairs), "valid_pairs": len(valid),
                "unavailable_pair_count": len(pairs) - len(valid),
                "unavailable_reasons": dict(Counter(reason for pair in pairs for reason in pair["unavailable_reasons"])),
                "success_pairs_only_statistics": len(pairs) != len(valid),
                "primary_comparison": candidate.startswith("diagnostic_") and reference in {"SBS", "age_return_ridge"},
            }
            paired_rows.extend(pairs)
        policy_counts = {}
        for policy in frozen["policy_ids"]:
            selected = [row for row in stratum_rows if row["policy_id"] == policy]
            policy_counts[policy] = {
                "rows": len(selected), "valid_rows": sum(row["_valid"] for row in selected),
                "failed_rows": sum(not row["_valid"] for row in selected),
                "failure_reasons": dict(Counter(row["_failure"] for row in selected if not row["_valid"])),
                "selected_actions": dict(Counter(str(row["selected_action"]) for row in selected)),
                "probe_gradient_updates_total": sum(row["probe_gradient_updates"] for row in selected),
                "continuation_gradient_updates_total": sum(row["continuation_gradient_updates"] for row in selected),
                "wall_time_sec_total": sum(row["wall_time_sec"] for row in selected),
                "actual_continuation_env_steps": sorted({row["continuation_env_steps"] for row in selected if "continuation_env_steps" in row}),
            }
        summaries.append(dict(descriptor, rows=len(stratum_rows), paired_panels=len(panels),
                              source_histories=len({row["trajectory_id"] for row in stratum_rows}),
                              declared_evaluation_games=expected_games, absent_evaluation_games=absent_games,
                              policies=policy_counts, comparisons=comparisons))
    incomplete = not rows or bool(warnings) or any(pair["pair_status"] != "ok" for pair in paired_rows) or \
                 any(stratum["absent_evaluation_games"] for stratum in summaries)
    result = {
        "schema_version": 1, "status": "incomplete" if incomplete else "complete",
        "run_dir": str(run_dir), "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": frozen["protocol_hash"], "frozen_at": frozen["envelope"]["frozen_at"],
        "protocol_validated": True, "freeze_precedes_all_evaluations": True,
        "validation_scope": frozen["validation_scope"], "verified_artifacts": frozen["verified_artifacts"],
        "development_games": frozen["development_games"], "reserved_games": frozen["reserved_games"],
        "outcome_rows": len(rows), "bootstrap_repeats": bootstrap_repeats, "warnings": warnings,
        "population_oracle_estimated": False, "policies_fitted_or_selected_by_this_analysis": False,
        "strata": summaries, "raw_paired_effects_file": "transfer_paired_effects.jsonl",
        "limitations": [
            "Diagnostic policies are reported separately; this analysis does not pick the best model on reserved outcomes.",
            "Raw returns are reported per game; the macro mean is not normalized across game reward scales.",
            "Fewer than five valid source histories per game is flagged uncertain; fewer than two cannot yield a cluster CI.",
            "Failure/missing pairs remain explicit; success-only paired estimates can be biased.",
            "Timestamp/hash checks audit logged provenance, not an independent proof of absence of training leakage.",
        ],
    }
    pair_path = run_dir / "transfer_paired_effects.jsonl"
    temporary = pair_path.with_name(pair_path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in paired_rows),
                         encoding="utf-8", newline="\n")
    temporary.replace(pair_path)
    _write_json(run_dir / "transfer_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    result = summarize_transfer(args.run_dir, args.bootstrap_repeats)
    print(json.dumps({"status": result["status"], "outcome_rows": result["outcome_rows"],
                      "strata": len(result["strata"]), "protocol_hash": result["protocol_hash"]}))


if __name__ == "__main__":
    main()
