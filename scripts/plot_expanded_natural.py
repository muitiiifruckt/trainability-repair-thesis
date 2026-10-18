"""График descriptive screen→confirmation из immutable compact384 JSON."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def plot(source: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert not output.exists(), "Immutable plot output already exists"
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    assert snapshot["audit_kind"] == "balanced_expanded_natural_384", "Unexpected snapshot kind"
    assert snapshot["selected_rows"] == 384 and snapshot["complete_panels"] == 96, "Incomplete cohort"
    games = ("asterix", "breakout")
    cohorts = ("screen", "confirmation")
    actions = ("optimizer_reset", "head_reset", "head_and_optimizer_reset")
    labels = ("Adam reset", "Head reset", "Head + Adam reset")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    markers = ("o", "s", "^", "v", "P", "X")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), sharey="row")
    for row, game in enumerate(games):
        data = snapshot["summaries"][game]
        for cohort in cohorts:
            assert data[cohort]["fully_complete"] and data[cohort]["complete_histories"] == 6
            assert [record["seed"] for record in data[cohort]["history_records"]] == list(range(6))
        for ax, action, label in zip(axes[row], actions, labels):
            for seed in range(6):
                values = [data[cohort]["history_records"][seed]["effects"][action] for cohort in cohorts]
                ax.plot((0, 1), values, marker=markers[seed], markersize=5.5,
                        color=colors[seed], linewidth=1.25, alpha=.87,
                        label=f"Source seed {seed}")
            ax.scatter((0, 1), [data[cohort]["mean_effects"][action] for cohort in cohorts],
                       marker="D", color="black", s=55, zorder=5, label="Среднее 6 histories")
            ax.axhline(0, color="#777777", linewidth=.9, linestyle="--", zorder=0)
            ax.set_xticks((0, 1), ("Screen\nrepeats 0/1", "Confirmation\nrepeats 2/3"))
            ax.set_xlim(-.15, 1.15)
            ax.set_title(label, fontsize=11)
            ax.grid(axis="y", alpha=.2)
        axes[row, 0].set_ylabel(f"{game.title()}\nΔ final return vs matched continue")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945),
               ncol=4, frameon=False, fontsize=9, columnspacing=1.6)
    fig.suptitle("Balanced natural cohort: эффекты ремонта на тех же histories", fontsize=14)
    fig.text(.5, .031, "6 source histories на игру; два ages и repeats усреднены внутри history. Описательные точки; без CI.",
             ha="center", fontsize=9)
    fig.text(.5, .011, "Screen→confirmation: новые continuation repeats тех же histories; не проверка diagnosis или unseen-history transfer.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .865), h_pad=2)
    file_format = output.suffix.lstrip(".").lower()
    assert file_format in ("png", "pdf", "svg"), "Use .png, .pdf or .svg"
    # Exclusive creation сохраняет immutable path даже при конкурентном запуске.
    try:
        with output.open("xb") as stream:
            fig.savefig(stream, format=file_format, dpi=170)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Compact384 JSON от audit_expanded_natural.py")
    parser.add_argument("output", type=Path, help="Новый immutable .png/.pdf/.svg")
    args = parser.parse_args()
    plot(args.source, args.output)
