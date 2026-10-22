"""Figure for notes/two_state_counterexample.md: total effect of each repair, three ways.

Usage: .venv\\Scripts\\python.exe scripts/plot_two_state.py RUN_DIR OUTPUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MENU = ("continue", "optimizer_reset", "parameter_reset", "occupancy_restoration")
LABELS = {"continue": "continue", "optimizer_reset": "сброс\nAdam", "parameter_reset": "сброс\nпараметров",
          "occupancy_restoration": "восстановл.\noccupancy"}
TITLES = {"A_occupancy": "A: короткое старение", "B_saturated": "B: насыщение (долгое старение)",
          "C_dead_units": "C: 75% мёртвых нейронов (контроль)", "D_control": "D: контроль без дефицита"}
LAB_OF = {"continue": "AAAA", "optimizer_reset": "AFAA", "parameter_reset": "FAAF", "occupancy_restoration": "AAFA"}
WAYS = (("lab", "лаборатория: 1 итерация на общем батче", "#2a78d6", "o"),
        ("frozen_behaviour", "живое, данные от замороженной политики", "#eb6834", "s"),
        ("live", "живое обучение (150 итераций)", "#1baf7a", "^"))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def per_history(rows, way, repair):
    if way == "lab":
        return [h["lab"][LAB_OF[repair]]["after"] - h["lab"]["AAAA"]["start"] for h in rows]
    out = []
    for h in rows:
        values = [x["total_effect"] for x in h["live"] if x["mode"] == way and x["repair"] == repair]
        out.append(sum(values) / len(values))
    return out


def main(run, output):
    per = json.loads((Path(run) / "per_history.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    width = 0.24
    for ax, (scenario, rows) in zip(axes.flat, per.items()):
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for k, (way, _, color, marker) in enumerate(WAYS):
            for i, repair in enumerate(MENU):
                values = per_history(rows, way, repair)
                x = i + (k - 1) * width
                ax.bar(x, sum(values) / len(values), width * 0.86, color=color, alpha=0.35, edgecolor="none")
                ax.scatter([x] * len(values), values, s=14, color=color, marker=marker, linewidths=0, zorder=3)
        ax.set_xticks(range(len(MENU)), [LABELS[r] for r in MENU])
        ax.set_title(TITLES[scenario], loc="left", color=INK, fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("полный эффект: return после − return\nстарого агента до ремонта")
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=6, label=l) for _, l, c, m in WAYS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Столбик — среднее по 8 историям, точки — истории. Оптимум новой задачи ≈ 6.51.\n"
             "Лабораторный гибрид ремонта: continue = AAAA, сброс Adam = AFAA, сброс параметров = FAAF, "
             "occupancy = AAFA (параметры, Adam, данные, critic).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
