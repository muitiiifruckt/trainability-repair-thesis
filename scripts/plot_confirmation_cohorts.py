"""Научный график из компактного среза; код обучения не загружается."""
import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"


def plot(source: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert not output.exists(), "Immutable output already exists"
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    assert snapshot["selected_rows"] == 192 and snapshot["complete_panels"] == 48
    games = ("asterix", "breakout")
    cohorts = ("screen", "confirmation")
    actions = ("optimizer_reset", "head_reset", "head_and_optimizer_reset")
    labels = ("Adam reset", "Head reset", "Head + Adam reset")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharey="row")
    colors = ("#2878b5", "#d57c1f", "#2d986f")
    for row, game in enumerate(games):
        data = snapshot["summaries"][game]
        assert all(data[cohort]["fully_complete"] for cohort in cohorts)
        for ax, action, label in zip(axes[row], actions, labels):
            for index in range(3):
                values = [data[cohort]["history_records"][index]["effects"][action] for cohort in cohorts]
                ax.plot((0, 1), values, marker="o", color=colors[index], label=f"Source seed {index}")
            ax.scatter((0, 1), [data[cohort]["mean_effects"][action] for cohort in cohorts],
                       marker="D", color="black", s=45, zorder=4, label="Среднее 3 histories")
            ax.axhline(0, color="gray", linewidth=.9, linestyle="--")
            ax.set_xticks((0, 1), ("Screen\nrepeats 0/1", "Confirmation\nrepeats 2/3"))
            ax.set_title(label, fontsize=11)
            ax.grid(axis="y", alpha=.18)
        axes[row, 0].set_ylabel(f"{game.title()}\nΔ final return vs matched continue")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .944), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Проверка эффекта ремонта в независимых продолжениях", fontsize=14)
    fig.text(.5, .022, "Те же 3 исходные histories на игру; два возраста усреднены внутри history. Без доверительных интервалов.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .885), h_pad=2)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.source, args.output)
