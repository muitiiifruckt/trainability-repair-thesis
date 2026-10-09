"""Descriptive source-history means from an immutable monitoring snapshot.

Incomplete histories are excluded. No confidence intervals or winner inference.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(snapshot, output):
    data = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    history = data["summary"]["by_history"]
    games = [game for game, item in data["summary"]["by_game"].items() if item["complete_histories"]]
    actions = ("optimizer_reset", "head_reset", "head_and_optimizer_reset")
    colors = ("#2878b5", "#d97a27", "#24986a")
    fig, axes = plt.subplots(1, len(games), figsize=(6*len(games), 5.3), squeeze=False)
    for axis, game in zip(axes[0], games):
        trials = sorted((name, item) for name, item in history.items()
                        if item["game"] == game and item["complete_pilot_history"])
        for index, (name, item) in enumerate(trials):
            offset = .12*(index-(len(trials)-1)/2)
            seed = int(name.rsplit("seed", 1)[1])
            axis.scatter([item["mean_final_effect"][action] for action in actions],
                         np.arange(3)+offset, color=colors[seed % len(colors)], s=55,
                         label=f"Source seed {seed}", zorder=3)
        if len(trials) > 1:
            means = data["summary"]["by_game"][game]["mean_effect"]
            axis.scatter([means[action] for action in actions], np.arange(3),
                         color="#222222", marker="D", s=35, label="Mean of histories", zorder=4)
        axis.axvline(0, color="#555555", linewidth=1)
        axis.set_yticks(range(3), ["Reset Adam", "Reset head", "Reset head + Adam"])
        axis.invert_yaxis()
        unit = "history" if len(trials) == 1 else "histories"
        axis.set_title(f"{game.title()}: {len(trials)} complete source {unit}")
        axis.set_xlabel("Final raw return minus matched Continue")
        axis.set_ylim(2.5, -.5)
        axis.margins(x=.22)
        axis.grid(axis="x", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    legend = {}
    for axis in axes[0]:
        handles, labels = axis.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(list(legend.values()), list(legend), loc="lower center", bbox_to_anchor=(.5, .075),
               frameon=False, ncol=4, fontsize=9)
    fig.suptitle("Partial online screen: effects differ across histories and games", fontsize=13)
    fig.text(.5, .015, "Each history averages two ages and two continuation repeats. "
             "Incomplete histories excluded; no population confidence intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .18, 1, .93))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.snapshot, args.output)
