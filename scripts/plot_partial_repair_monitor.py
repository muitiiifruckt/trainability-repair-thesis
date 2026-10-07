"""Render an immutable monitoring snapshot without importing the RL runtime.

This figure describes one source history. It does not estimate population
uncertainty, choose a winner, or combine unfinished repairs with completed ones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"continue": "Continue", "optimizer_reset": "Reset Adam",
          "head_reset": "Reset head", "head_and_optimizer_reset": "Reset head + Adam"}
COLORS = {"continue": "#2878b5", "optimizer_reset": "#db7c26",
          "head_reset": "#289b69", "head_and_optimizer_reset": "#9c58aa"}


def render(snapshot: Path, checkpoint_id: str, output: Path) -> dict:
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    checkpoint = next(x for x in data["checkpoints"] if x["checkpoint_id"] == checkpoint_id)
    selected = [x for x in data["outcomes"] if x["checkpoint_id"] == checkpoint_id
                and x.get("failure_reason") is None and x.get("j_final") is not None
                and x["completed_env_steps"] == x["budget_env_steps"]
                and x["repair_id"] in LABELS]
    strata = {(x["source_mode"], x["budget_env_steps"]) for x in selected}
    if len(strata) != 1:
        raise ValueError("A partial figure must contain exactly one mode and continuation budget")
    identities = [(x["repair_id"], x["repeat"]) for x in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate repair/repeat in immutable snapshot")
    mode, budget = next(iter(strata))
    fig, (curve_ax, effect_ax) = plt.subplots(1, 2, figsize=(11, 4.8),
                                           gridspec_kw={"width_ratios": [1.6, 1]})
    counts = {}
    for action in LABELS:
        trials = [x for x in selected if x["repair_id"] == action]
        if not trials:
            continue
        counts[action] = len(trials)
        curves = [{p["env_steps"]: p["mean_return"] for p in x["evaluation_curve"]} for x in trials]
        common_steps = sorted(set.intersection(*(set(curve) for curve in curves)))
        for curve in curves:
            steps = sorted(curve)
            curve_ax.plot(np.array(steps)/1000, [curve[x] for x in steps],
                          color=COLORS[action], alpha=.3, linewidth=1.2)
        means = [np.mean([curve[x] for curve in curves]) for x in common_steps]
        curve_ax.plot(np.array(common_steps)/1000, means, marker="o", linewidth=2.2,
                      color=COLORS[action], label=f"{LABELS[action]} (n={len(trials)})")
    baseline = {x["repeat"]: x for x in selected if x["repair_id"] == "continue"}
    pairs = []
    for row in selected:
        if row["repair_id"] != "continue" and row["repeat"] in baseline:
            pairs.append(dict(action=row["repair_id"], repeat=row["repeat"],
                              effect=row["j_final"]-baseline[row["repeat"]]["j_final"]))
    for index, pair in enumerate(pairs):
        effect_ax.bar(index, pair["effect"], color=COLORS[pair["action"]], alpha=.85)
        effect_ax.annotate(f"{pair['effect']:+.2f}", (index, pair["effect"]),
                           xytext=(0, -15 if pair["effect"] < 0 else 6),
                           textcoords="offset points", ha="center", fontsize=10)
    effect_ax.set_xticks(range(len(pairs)),
                        [f"{LABELS[p['action']]}\nrepeat {p['repeat']}" for p in pairs])
    effect_ax.axhline(0, color="#444444", linewidth=1)
    effect_ax.margins(y=.28)
    curve_ax.set(xlabel="Continuation transitions (thousands)", ylabel="Mean raw evaluation return",
                 title="Recovery curves: thin lines are individual repeats")
    effect_ax.set(ylabel="Final return minus matched Continue", title="Paired effects within this history")
    curve_ax.legend(frameon=False, fontsize=9)
    for axis in (curve_ax, effect_ax):
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{checkpoint['game'].title()}, source seed {checkpoint['training_seed']}, "
                 f"age {checkpoint['environment_steps']:,} transitions | {mode}", fontsize=12)
    fig.text(.5, .015, "PRELIMINARY: one training history; 20 evaluation episodes per point. "
             "Unfinished repairs excluded; no population confidence interval.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"checkpoint_id": checkpoint_id, "mode": mode, "budget_env_steps": budget,
            "completed_repeats_per_action": counts, "paired_final_effects": pairs,
            "analysis_optimizer_updates": 0, "figure": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("checkpoint_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.snapshot, args.checkpoint_id, args.output), indent=2))
