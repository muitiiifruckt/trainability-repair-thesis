"""Лёгкий аудит independent repeats; learner и checkpoints не загружаются.

Сохраняет компактный неизменяемый срез завершённых matched panels.
Повторы и возраста усредняются внутри trajectory_id; интервалы не строятся.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
MENU = ("continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset")
COHORTS = {"screen": (0, 1), "confirmation": (2, 3)}


def audit(source: Path) -> dict:
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert len({row["job_id"] for row in rows}) == len(rows), "Duplicate job_id"
    selected = [row for row in rows if row.get("source_mode") == "natural"
                and row["game"] in ("asterix", "breakout")
                and row["training_seed"] in (0, 1, 2)
                and row["nominal_age"] in (50000, 200000)
                and row["repeat"] in (0, 1, 2, 3)
                and row["repair_id"] in MENU]
    panels = collections.defaultdict(dict)
    for row in selected:
        assert row.get("failure_reason") is None, "Failed branch"
        assert row["completed_env_steps"] == row["budget_env_steps"] == 50000
        assert row["budget_gradient_updates"] == 50000
        assert row.get("probe_gradient_updates", 0) == 0
        assert math.isfinite(row["j_final"])
        curve = row["evaluation_curve"]
        assert [point["env_steps"] for point in curve] == [0, 5000, 20000, 50000]
        for point in curve:
            assert len(point["returns"]) == len(point["evaluation_seeds"]) == 20
            assert abs(mean(point["returns"]) - point["mean_return"]) < 1e-12
        assert abs(curve[-1]["mean_return"] - row["j_final"]) < 1e-12
        key = (row["game"], row["trajectory_id"], row["nominal_age"], row["repeat"])
        assert row["repair_id"] not in panels[key]
        panels[key][row["repair_id"]] = row

    complete = {}
    incomplete = []
    for key, actions in sorted(panels.items()):
        if set(actions) != set(MENU):
            incomplete.append({"panel": list(key), "available_actions": sorted(actions)})
            continue
        base = actions["continue"]
        for row in actions.values():
            assert row["continuation_seed"] == base["continuation_seed"]
            assert row["checkpoint_sha256"] == base["checkpoint_sha256"]
            assert row["evaluation_seeds"] == base["evaluation_seeds"]
            assert [p["evaluation_seeds"] for p in row["evaluation_curve"]] == [
                p["evaluation_seeds"] for p in base["evaluation_curve"]]
        final = {action: row["j_final"] for action, row in actions.items()}
        effects = {action: final[action] - final["continue"] for action in MENU[1:]}
        effects["interaction"] = final["head_and_optimizer_reset"] - final["head_reset"] - final["optimizer_reset"] + final["continue"]
        complete[key] = {"game": key[0], "trajectory_id": key[1], "age": key[2],
                         "repeat": key[3], "j_final": final, "effects": effects}

    summaries = {}
    for game in ("asterix", "breakout"):
        summaries[game] = {}
        for cohort, repeats in COHORTS.items():
            histories = []
            for seed in (0, 1, 2):
                trajectory = f"natural_{game}_seed{seed}"
                keys = [(game, trajectory, age, repeat) for age in (50000, 200000) for repeat in repeats]
                if not all(key in complete for key in keys):
                    continue
                group = [complete[key] for key in keys]
                histories.append({"trajectory_id": trajectory, "seed": seed,
                                  "j_final": {action: mean(p["j_final"][action] for p in group) for action in MENU},
                                  "effects": {action: mean(p["effects"][action] for p in group) for action in (*MENU[1:], "interaction")}})
            summaries[game][cohort] = {
                "complete_histories": len(histories), "expected_histories": 3,
                "history_records": histories,
                "mean_effects": {action: mean(h["effects"][action] for h in histories) for action in (*MENU[1:], "interaction")} if histories else {},
                "fully_complete": len(histories) == 3,
            }

    crossfit = {}
    for game in ("asterix", "breakout"):
        if not all(summaries[game][cohort]["fully_complete"] for cohort in COHORTS):
            continue
        directions = []
        for train_name, test_name in (("screen", "confirmation"), ("confirmation", "screen")):
            train_repeats, test_repeats = COHORTS[train_name], COHORTS[test_name]
            contexts = []
            for seed in (0, 1, 2):
                trajectory = f"natural_{game}_seed{seed}"
                for age in (50000, 200000):
                    train = {action: mean(complete[(game, trajectory, age, rep)]["j_final"][action] for rep in train_repeats) for action in MENU}
                    test = {action: mean(complete[(game, trajectory, age, rep)]["j_final"][action] for rep in test_repeats) for action in MENU}
                    contexts.append({"trajectory_id": trajectory, "age": age, "train": train, "test": test,
                                     "selected": max(MENU, key=lambda action: train[action])})
            sbs = max(MENU, key=lambda action: mean(c["train"][action] for c in contexts))
            history_records = []
            for seed in (0, 1, 2):
                trajectory = f"natural_{game}_seed{seed}"
                own = [c for c in contexts if c["trajectory_id"] == trajectory]
                others = [c for c in contexts if c["trajectory_id"] != trajectory]
                loo_sbs = max(MENU, key=lambda action: mean(c["train"][action] for c in others))
                history_records.append({"trajectory_id": trajectory,
                    "selected_actions": [c["selected"] for c in own], "leave_history_out_sbs": loo_sbs,
                    "winner_vs_continue": mean(c["test"][c["selected"]] - c["test"]["continue"] for c in own),
                    "winner_vs_train_sbs": mean(c["test"][c["selected"]] - c["test"][sbs] for c in own),
                    "winner_vs_leave_history_out_sbs": mean(c["test"][c["selected"]] - c["test"][loo_sbs] for c in own)})
            directions.append({"train": train_name, "test": test_name, "train_sbs": sbs,
                "histories": history_records,
                "mean": {key: mean(h[key] for h in history_records) for key in (
                    "winner_vs_continue", "winner_vs_train_sbs", "winner_vs_leave_history_out_sbs")}})
        crossfit[game] = {"directions": directions,
                         "bidirectional_mean": {key: mean(d["mean"][key] for d in directions) for key in directions[0]["mean"]},
                         "interpretation": "Exploratory repeat-split label stability; uses trial outcomes, not a pre-repair diagnostic selector. n=3 training histories; no confidence interval."}

    return {"schema_version": 1, "captured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_rows": len(rows), "selected_rows": len(selected),
            "stage_counts": dict(collections.Counter(row.get("stage", "unknown") for row in selected)),
            "complete_panels": len(complete), "incomplete_panels": incomplete,
            "menu": list(MENU), "cohorts": COHORTS, "summaries": summaries,
            "panel_records": list(complete.values()), "crossfit": crossfit,
            "estimand": "Paired final raw-return difference; equal ages/repeats within source trajectory, then equal histories. Interaction=joint-head-optimizer+continue.",
            "checks": "Unique jobs, finite finals, fixed 50k budgets, matched continuation/checkpoint/evaluation seeds, all curve means reconstructed from 20 returns.",
            "limitations": "Same source histories, new continuation repeats only; n=3 per game. Partial histories excluded; no pooled-game repair claim or confidence intervals."}


def plot(snapshot: dict, path: Path) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert not path.exists(), "Plot is immutable"
    data = snapshot["summaries"]["breakout"]
    assert all(data[cohort]["fully_complete"] for cohort in COHORTS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    colors = ("#2878b5", "#d57c1f", "#2d986f")
    for ax, action, label in zip(axes, MENU[1:], ("Adam reset", "Head reset", "Head + Adam reset")):
        for index in range(3):
            values = [data[cohort]["history_records"][index]["effects"][action] for cohort in COHORTS]
            ax.plot((0, 1), values, marker="o", color=colors[index], label=f"Source seed {index}")
        ax.scatter((0, 1), [data[cohort]["mean_effects"][action] for cohort in COHORTS],
                   marker="D", color="black", s=50, zorder=4, label="Mean of 3 histories")
        ax.axhline(0, color="gray", linewidth=.9, linestyle="--")
        ax.set_xticks((0, 1), ("Screen\nrepeats 0/1", "Confirmation\nrepeats 2/3"))
        ax.set_title(label)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Final return difference vs matched continue")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Breakout: effect signs across independent continuation repeats")
    fig.text(.5, .015, "Same 3 source histories; two ages averaged within history. Descriptive points; no confidence intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/minatar-repair-20261007/outcomes.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = audit(args.source.resolve())
    result["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.plot:
        plot(result, args.plot)
    print(json.dumps({"source_rows": result["source_rows"], "source_sha256": result["source_sha256"],
                      "summaries": result["summaries"], "crossfit": result["crossfit"]}, ensure_ascii=True))
