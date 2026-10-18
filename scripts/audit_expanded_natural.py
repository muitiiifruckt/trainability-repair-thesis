"""Строгий stdlib audit полного balanced natural cohort (384 branches).

Learner/checkpoints не загружаются. Возраста и continuation repeats получают
равный вес внутри trajectory_id, затем source histories получают равный вес.
Все оценки описательные; доверительные интервалы не вычисляются.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
GAMES = ("asterix", "breakout")
SEEDS = tuple(range(6))
AGES = (50000, 200000)
MENU = ("continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset")
EFFECTS = (*MENU[1:], "interaction")
STEPS = (0, 5000, 20000, 50000)
COHORTS = {"screen": (0, 1), "confirmation": (2, 3), "pooled": (0, 1, 2, 3)}
SEED_GROUPS = {"old_seeds_0_2": (0, 1, 2), "new_seeds_3_5": (3, 4, 5)}
SELECTION_METRICS = ("winner_vs_continue", "winner_vs_train_sbs", "winner_vs_leave_history_out_sbs")
CANONICAL_RULE = (
    "Полные selected rows сортируются по (game, training_seed, nominal_age, repeat, "
    "MENU.index(repair_id)); json.dumps(rows, ensure_ascii=False, sort_keys=True, "
    "separators=(',', ':'), allow_nan=False), затем один LF; SHA-256 от UTF-8 bytes."
)


def effect(values: dict) -> dict:
    result = {action: values[action] - values["continue"] for action in MENU[1:]}
    result["interaction"] = (values[MENU[3]] - values[MENU[2]]
                             - values[MENU[1]] + values[MENU[0]])
    return result


def average_maps(records: list[dict], key: str, names: tuple) -> dict:
    return {name: mean(record[key][name] for record in records) for name in names}


def choose(scores: dict) -> tuple[str, list[str]]:
    """Точная ничья разрешается первым action в фиксированном MENU, как раньше."""
    winner = max(MENU, key=lambda action: scores[action])
    return winner, [action for action in MENU if scores[action] == scores[winner]]


def summarize_histories(histories: list[dict]) -> dict:
    return {
        "complete_histories": len(histories),
        "trajectory_ids": [record["trajectory_id"] for record in histories],
        "mean_j_final": average_maps(histories, "j_final", MENU),
        "mean_adaptation_auc": average_maps(histories, "adaptation_auc", MENU),
        "mean_effects": average_maps(histories, "effects", EFFECTS),
        "mean_auc_effects": average_maps(histories, "auc_effects", EFFECTS),
        "mean_curve_effects": {
            str(step): {action: mean(record["curve_effects"][str(step)][action]
                                    for record in histories) for action in EFFECTS}
            for step in STEPS
        },
    }


def selection_means(histories: list[dict]) -> dict:
    return {key: mean(record[key] for record in histories) for key in SELECTION_METRICS}


def repeat_split(panels: dict, game: str) -> dict:
    directions = []
    for train_name, test_name in (("screen", "confirmation"), ("confirmation", "screen")):
        contexts = []
        for seed, age in itertools.product(SEEDS, AGES):
            train = {action: mean(panels[(game, seed, age, repeat)]["j_final"][action]
                                  for repeat in COHORTS[train_name]) for action in MENU}
            test = {action: mean(panels[(game, seed, age, repeat)]["j_final"][action]
                                 for repeat in COHORTS[test_name]) for action in MENU}
            selected, tied = choose(train)
            contexts.append({"seed": seed, "age": age, "train": train, "test": test,
                             "selected": selected, "tied_actions": tied})
        sbs_scores = {action: mean(context["train"][action] for context in contexts)
                      for action in MENU}
        sbs, sbs_tied = choose(sbs_scores)
        histories = []
        for seed in SEEDS:
            own = [context for context in contexts if context["seed"] == seed]
            others = [context for context in contexts if context["seed"] != seed]
            loo_scores = {action: mean(context["train"][action] for context in others)
                          for action in MENU}
            loo_sbs, loo_tied = choose(loo_scores)
            histories.append({
                "trajectory_id": f"natural_{game}_seed{seed}", "seed": seed,
                "selected_actions_by_age": {str(c["age"]): c["selected"] for c in own},
                "tied_actions_by_age": {str(c["age"]): c["tied_actions"] for c in own},
                "leave_history_out_sbs": loo_sbs, "leave_history_out_sbs_tied_actions": loo_tied,
                "winner_vs_continue": mean(c["test"][c["selected"]] - c["test"]["continue"] for c in own),
                "winner_vs_train_sbs": mean(c["test"][c["selected"]] - c["test"][sbs] for c in own),
                "winner_vs_leave_history_out_sbs": mean(c["test"][c["selected"]] - c["test"][loo_sbs] for c in own),
            })
        directions.append({
            "train": train_name, "test": test_name,
            "train_sbs": sbs, "train_sbs_tied_actions": sbs_tied,
            "train_sbs_scores": sbs_scores,
            "contexts_with_exact_tie": sum(len(c["tied_actions"]) > 1 for c in contexts),
            "history_records": histories, "mean": selection_means(histories),
            "seed_groups": {name: selection_means([h for h in histories if h["seed"] in seeds])
                            for name, seeds in SEED_GROUPS.items()},
        })
    bidirectional_histories = [{
        "trajectory_id": f"natural_{game}_seed{seed}", "seed": seed,
        **{key: mean(direction["history_records"][seed][key] for direction in directions)
           for key in SELECTION_METRICS},
    } for seed in SEEDS]
    return {
        "directions": directions, "bidirectional_history_records": bidirectional_histories,
        "bidirectional_mean": selection_means(bidirectional_histories),
        "seed_groups": {name: selection_means([h for h in bidirectional_histories if h["seed"] in seeds])
                        for name, seeds in SEED_GROUPS.items()},
        "tie_rule": "Точное равенство train means; первый action в MENU. Tolerance не применяется.",
        "interpretation": (
            "Exploratory outcome-label repeat-split на тех же source histories. Chooser использует "
            "train-half outcomes каждого checkpoint; это не diagnosis и не unseen-history benefit. "
            "Train SBS выбран на всех train-half histories данной игры; leave-history-out SBS "
            "выбран на остальных histories, но chooser по-прежнему видит own-history train labels. "
            "Два направления зависимы и не увеличивают число независимых histories. Без CI."
        ),
    }


def audit_bytes(raw: bytes, source: Path) -> dict:
    """Разбирает единственный снимок source bytes, сохраняя append-only provenance."""
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert len({row["job_id"] for row in rows}) == len(rows), "Duplicate source job_id"
    selected = [row for row in rows if row.get("source_mode") == "natural"
                and row["game"] in GAMES and row["training_seed"] in SEEDS
                and row["nominal_age"] in AGES and row["repeat"] in COHORTS["pooled"]]
    expected = set(itertools.product(GAMES, SEEDS, AGES, COHORTS["pooled"], MENU))
    observed = [(row["game"], row["training_seed"], row["nominal_age"], row["repeat"], row["repair_id"])
                for row in selected]
    assert len(selected) == 384, f"Expected 384 selected branches, got {len(selected)}"
    assert len(set(observed)) == len(observed), "Duplicate canonical branch"
    assert set(observed) == expected, "Canonical coverage differs from 2 games x 6 seeds x 2 ages x 4 repeats x 4 actions"
    selected.sort(key=lambda row: (row["game"], row["training_seed"], row["nominal_age"],
                                   row["repeat"], MENU.index(row["repair_id"])))
    canonical = (json.dumps(selected, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    raw_panels = collections.defaultdict(dict)
    checkpoint_hashes = collections.defaultdict(set)
    for row in selected:
        game, seed, age = row["game"], row["training_seed"], row["nominal_age"]
        assert row["trajectory_id"] == f"natural_{game}_seed{seed}", "Unexpected trajectory_id"
        assert row["checkpoint_id"] == f"natural_{game}_seed{seed}_age{age}", "Unexpected checkpoint_id"
        assert row.get("failure_reason") is None, f"Failed branch: {row['job_id']}"
        assert row["completed_env_steps"] == row["budget_env_steps"] == 50000, "Incomplete env budget"
        assert row["budget_gradient_updates"] == 50000, "Unexpected gradient budget"
        assert row.get("probe_gradient_updates", 0) == 0, "Unexpected probe budget"
        for key in ("j_pre", "j_immediate", "j_final", "adaptation_auc"):
            assert math.isfinite(row[key]), f"Nonfinite {key}"
        checkpoint_hash = row["checkpoint_sha256"]
        assert len(checkpoint_hash) == 64 and all(c in "0123456789abcdef" for c in checkpoint_hash), "Invalid checkpoint SHA"
        checkpoint_hashes[(game, seed, age)].add(checkpoint_hash)
        curve = row["evaluation_curve"]
        assert [point["env_steps"] for point in curve] == list(STEPS), "Unexpected eval schedule"
        for point in curve:
            assert len(point["returns"]) == len(point["evaluation_seeds"]) == 20, "Expected 20 eval episodes"
            assert len(set(point["evaluation_seeds"])) == 20, "Duplicate evaluation seed within point"
            assert all(math.isfinite(value) for value in point["returns"]), "Nonfinite episode return"
            assert math.isfinite(point["mean_return"]), "Nonfinite curve mean"
            assert abs(mean(point["returns"]) - point["mean_return"]) < 1e-12, "Episode mean mismatch"
        assert abs(curve[0]["mean_return"] - row["j_immediate"]) < 1e-12, "Immediate mismatch"
        assert abs(curve[-1]["mean_return"] - row["j_final"]) < 1e-12, "Final mismatch"
        assert row["evaluation_seeds"] == curve[-1]["evaluation_seeds"], "Final eval seeds mismatch"
        auc = sum((right["env_steps"] - left["env_steps"])
                  * (right["mean_return"] + left["mean_return"]) / 2
                  for left, right in zip(curve, curve[1:])) / 50000
        assert abs(auc - row["adaptation_auc"]) < 1e-12, "Normalized trapezoid AUC mismatch"
        raw_panels[(game, seed, age, row["repeat"])][row["repair_id"]] = row
    assert all(len(hashes) == 1 for hashes in checkpoint_hashes.values()), "Checkpoint differs between repeats"
    panels = {}
    for key, actions in sorted(raw_panels.items()):
        assert set(actions) == set(MENU), "Incomplete matched panel"
        base = actions["continue"]
        for row in actions.values():
            assert row["checkpoint_sha256"] == base["checkpoint_sha256"], "Unmatched checkpoint"
            assert row["continuation_seed"] == base["continuation_seed"], "Unmatched continuation seed"
            assert row["evaluation_seeds"] == base["evaluation_seeds"], "Unmatched final eval seeds"
            assert [p["evaluation_seeds"] for p in row["evaluation_curve"]] == [
                p["evaluation_seeds"] for p in base["evaluation_curve"]], "Unmatched curve eval seeds"
        final = {action: actions[action]["j_final"] for action in MENU}
        aucs = {action: actions[action]["adaptation_auc"] for action in MENU}
        panels[key] = {
            "game": key[0], "seed": key[1], "trajectory_id": base["trajectory_id"],
            "age": key[2], "repeat": key[3], "checkpoint_sha256": base["checkpoint_sha256"],
            "j_final": final, "effects": effect(final), "adaptation_auc": aucs, "auc_effects": effect(aucs),
            "curve_effects": {str(step): effect({action: actions[action]["evaluation_curve"][index]["mean_return"]
                                                 for action in MENU}) for index, step in enumerate(STEPS)},
        }
    assert len(panels) == 96, "Expected 96 complete matched panels"
    summaries = {}
    for game in GAMES:
        summaries[game] = {}
        for cohort, repeats in COHORTS.items():
            histories = []
            for seed in SEEDS:
                group = [panels[(game, seed, age, repeat)] for age, repeat in itertools.product(AGES, repeats)]
                histories.append({
                    "trajectory_id": f"natural_{game}_seed{seed}", "seed": seed,
                    "j_final": average_maps(group, "j_final", MENU),
                    "effects": average_maps(group, "effects", EFFECTS),
                    "adaptation_auc": average_maps(group, "adaptation_auc", MENU),
                    "auc_effects": average_maps(group, "auc_effects", EFFECTS),
                    "curve_effects": {str(step): {action: mean(p["curve_effects"][str(step)][action]
                                                             for p in group) for action in EFFECTS} for step in STEPS},
                })
            summaries[game][cohort] = {
                **summarize_histories(histories), "expected_histories": 6, "fully_complete": True,
                "history_records": histories,
                "seed_groups": {name: summarize_histories([h for h in histories if h["seed"] in seeds])
                                for name, seeds in SEED_GROUPS.items()},
            }
    try:
        source_name = str(source.relative_to(ROOT))
    except ValueError:
        source_name = str(source)
    return {
        "schema_version": 1, "audit_kind": "balanced_expanded_natural_384",
        "captured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": source_name, "source_rows": len(rows), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256_rule": "SHA-256 точных bytes одного read_bytes(), включая исходные пробелы и line endings.",
        "source_failure_rows": sum(row.get("failure_reason") is not None for row in rows),
        "source_floating_point_error_rows": sum("FloatingPointError" in str(row.get("failure_reason", "")) for row in rows),
        "selected_rows": len(selected), "excluded_rows": len(rows) - len(selected),
        "selected_cohort_sha256": hashlib.sha256(canonical).hexdigest(), "selected_cohort_sha256_rule": CANONICAL_RULE,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage_counts": dict(collections.Counter(row.get("stage", "unknown") for row in selected)),
        "menu": list(MENU), "cohorts": COHORTS, "ages": AGES, "seeds": SEEDS,
        "complete_panels": len(panels), "checkpoint_count": len(checkpoint_hashes),
        "summaries": summaries, "panel_records": list(panels.values()),
        "crossfit": {game: repeat_split(panels, game) for game in GAMES},
        "estimand": (
            "Paired final raw-return difference vs matched continue; равный вес двух ages и repeats "
            "внутри trajectory_id, затем равный вес шести source histories на игру. "
            "Interaction=joint-head-optimizer+continue. Secondary adaptation_auc: normalized trapezoid "
            "по [0,5000,20000,50000] / 50000, не замена primary final."
        ),
        "checks": [
            "Unique source job_id и exact canonical coverage 384 без duplicate branches",
            "Canonical trajectory/checkpoint identities; selected failures/FloatingPointError отсутствуют",
            "Полные fixed 50k env budgets, 50k gradient budget, probe=0",
            "Общий checkpoint SHA между actions/repeats одного source checkpoint",
            "Общие continuation и все evaluation seeds между четырьмя actions каждого panel",
            "1536 curve means реконструированы из 20 finite episode returns; final/immediate/AUC согласованы",
        ],
        "limitations": (
            "12 source trajectories всего, n=6/game; ages, repeats и repeat-split directions зависимы. "
            "Old/new subsets n=3/game. Только описательные точки, без CI и pooled-game claim. "
            "Repeats 4/5 и прочие rows excluded по заданному cohort; полнота будущей панели 576 не утверждается. "
            "Reserved games и learner/checkpoints не открывались."
        ),
    }


def audit(source: Path) -> dict:
    return audit_bytes(source.read_bytes(), source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/minatar-repair-20261007/outcomes.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Новый immutable JSON; существующий путь запрещён")
    args = parser.parse_args()
    result = audit(args.source.resolve())
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "source_rows": result["source_rows"], "source_sha256": result["source_sha256"],
        "selected_rows": result["selected_rows"], "selected_cohort_sha256": result["selected_cohort_sha256"],
        "mean_effects": {game: {cohort: data["mean_effects"] for cohort, data in cohorts.items()}
                         for game, cohorts in result["summaries"].items()},
        "crossfit_bidirectional_mean": {game: data["bidirectional_mean"] for game, data in result["crossfit"].items()},
    }, ensure_ascii=False))
