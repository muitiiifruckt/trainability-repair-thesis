"""Produce descriptive summaries and a scientific plot; no selector is fitted."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results"
LABELS = {
    "continue": "Continue", "optimizer_reset": "Optimizer reset", "head_reset": "Head reset",
    "head_and_optimizer_reset": "Head + optimizer", "t_reset": "Timestep reset",
}


def main() -> None:
    raw = json.loads((RESULTS / "controlled_probe.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    paired = {}
    for row in raw["branches"]:
        if row["failure"] is not None:
            raise ValueError("Failed branches require an explicit aggregation policy")
        grouped[(row["checkpoint_id"], row["action"])].append(row)
        paired[(row["checkpoint_id"], row["continuation_repeat"], row["action"])] = row
    summaries = []
    mean_loss = {}
    for (checkpoint, action), rows in grouped.items():
        item = {"checkpoint_id": checkpoint, "scenario": rows[0]["scenario"], "source_updates": rows[0]["source_updates"],
                "training_seed": rows[0]["training_seed"], "action": action, "continuation_repeats": len(rows),
                "training_loss_mean": statistics.mean(r["loss_final"] for r in rows),
                "heldout_loss_mean": statistics.mean(r["heldout_loss_final"] for r in rows),
                "loss_immediate": rows[0]["loss_immediate"]}
        summaries.append(item)
        mean_loss[(checkpoint, action)] = item["heldout_loss_mean"]
    max_control_difference = 0.0
    for (checkpoint, repeat, action), row in paired.items():
        if action == "t_reset":
            control = paired[(checkpoint, repeat, "t_reset_equivalent_schedule")]
            max_control_difference = max(max_control_difference, abs(row["heldout_loss_final"] - control["heldout_loss_final"]))
    # Exclude algebraically equivalent action from winners to prevent meaningless duplicate labels.
    counts = Counter()
    for checkpoint in raw["checkpoints"]:
        identifier = checkpoint["checkpoint_id"]
        winner = min(LABELS, key=lambda action: mean_loss[(identifier, action)])
        counts[winner] += 1
    summary = {
        "kind": "descriptive_synthetic_control_not_RL", "checkpoints": len(raw["checkpoints"]),
        "branches": len(raw["branches"]), "wall_time_sec": raw["runtime"]["wall_time_sec"],
        "max_t_reset_vs_equivalent_schedule_heldout_loss_difference": max_control_difference,
        "naive_empirical_winner_counts_excluding_equivalent_duplicate": dict(counts),
        "checkpoint_action_means": summaries,
        "caveat": "Winners use the same two continuation repeats, are descriptive, and do not estimate a population oracle or unseen-environment selection performance.",
    }
    (RESULTS / "controlled_probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scenarios = raw["config"]["scenarios"]
    latest_age = max(raw["config"]["ages"])
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), sharey=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    colors = ["#4b5563", "#2563eb", "#4b5563", "#4b5563", "#4b5563"]
    for ax, scenario in zip(axes.flat, scenarios, strict=True):
        means, lower, upper = [], [], []
        for action in LABELS:
            vals = [row["heldout_loss_mean"] for row in summaries
                    if row["scenario"] == scenario and row["source_updates"] == latest_age and row["action"] == action]
            mean = statistics.mean(vals)
            means.append(mean)
            lower.append(mean - min(vals))
            upper.append(max(vals) - mean)
        ax.bar(range(len(LABELS)), means, color=colors, width=0.7, yerr=[lower, upper], capsize=3)
        ax.set_yscale("log")
        ax.set_ylabel("Held-out MSE (log scale)")
        ax.set_xlabel("Intervention")
        ax.set_title(scenario.replace("_", " ").capitalize(), loc="left", fontsize=12)
        ax.set_xticks(range(len(LABELS)), ["Continue", "Optimizer\nreset", "Head\nreset", "Head +\noptimizer", "Timestep\nreset"], fontsize=9)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Synthetic teacher shifts: repair response after {raw['config']['future_updates']} updates", fontsize=15, x=0.075, ha="left")
    fig.text(0.075, 0.035,
             f"Source: controlled_probe.json · 2026-10-05 · CPU float64 · source checkpoint age {latest_age} updates.\n"
             f"Bars: mean of {raw['config']['seeds']} training-seed means; each averages {raw['config']['repeats']} paired continuation repeats.\n"
             "Whiskers: min–max across training seeds, not confidence intervals. Artificial regression tasks; no RL or selector evaluation.", fontsize=9)
    fig.tight_layout(rect=[0.055, 0.13, 1, 0.94])
    fig.savefig(RESULTS / "controlled_probe.png", dpi=170)
    plt.close(fig)
    print("Synthetic checkpoints:", summary["checkpoints"], "; branches:", summary["branches"])
    print("Runtime seconds:", round(summary["wall_time_sec"], 2))
    print("Empirical winner counts:", dict(counts))
    print("Max schedule-control difference:", max_control_difference)
    for scenario in scenarios:
        values = {action: statistics.mean(row["heldout_loss_mean"] for row in summaries
                  if row["scenario"] == scenario and row["source_updates"] == latest_age and row["action"] == action)
                  for action in LABELS}
        print(scenario, values)


if __name__ == "__main__":
    main()
