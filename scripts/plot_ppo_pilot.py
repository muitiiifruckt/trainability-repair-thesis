"""Figure and summary for notes/ppo_stand.md: PPO pilot learning curves and per-layer telemetry.

Usage: .venv\\Scripts\\python.exe scripts/plot_ppo_pilot.py PILOT_DIR OUTPUT_PNG SUMMARY_JSON
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMES = {"breakout": "Breakout (разработка)", "space_invaders": "Space Invaders (отложенная)"}
ARCHS = {"cnn": ("cnn", "#2a78d6"), "cnn_ln": ("cnn_ln (отложенная)", "#eb6834")}
SEEDS = {0: "-", 1: "--"}
ROWS = (("episode_return", "return эпизода\n(обучение, сглажено)"),
        ("fc_effective_rank", "effective rank fc\n(reference-батч)"),
        ("fc_dead_fraction", "доля мёртвых\nнейронов fc"),
        ("fc_adam_frac", "доля √v < 10ε\nв Adam, fc"))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def series(folder):
    path = folder / "telemetry" / f"ppo_{folder.name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("metrics")]
    steps = [r["metrics"]["env_steps"] for r in rows]
    returns, smoothed, window = [r["metrics"]["episode_return"] for r in rows], [], []
    for value in returns:  # running mean over the last 10 telemetry rows with finished episodes
        if value is not None:
            window = (window + [value])[-10:]
        smoothed.append(sum(window) / len(window) if window else None)
    return {"steps": steps, "episode_return": smoothed,
            "fc_effective_rank": [r["layers"]["fc"]["effective_rank"] for r in rows],
            "fc_dead_fraction": [r["layers"]["fc"]["dead_fraction"] for r in rows],
            "fc_adam_frac": [r["layers"]["fc"]["adam_frac_sqrt_v_below_10eps"] for r in rows]}


def main(pilot, output, summary_path):
    pilot = Path(pilot)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(ROWS), len(GAMES), figsize=(10.5, 9.5), sharex=True)
    summary = {}
    for col, game in enumerate(GAMES):
        randoms = []
        for arch, (label, color) in ARCHS.items():
            for seed, style in SEEDS.items():
                folder = pilot / f"{game}_{arch}_s{seed}"
                if not (folder / "result.json").exists():
                    continue
                result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                randoms.append(result["random_return"])
                data = series(folder)
                summary[folder.name] = {k: result[k] for k in ("eval_return", "random_return", "steps_per_sec",
                                                               "wall_time_sec", "env_steps")}
                summary[folder.name].update({f"final_{k}": data[k][-1] for k, _ in ROWS})
                for row, (key, _) in enumerate(ROWS):
                    axes[row, col].plot([s / 1e6 for s in data["steps"]], data[key], color=color, linestyle=style,
                                        linewidth=1.6, label=f"{label}, seed {seed}")
        if randoms:
            axes[0, col].axhline(sum(randoms) / len(randoms), color=MUTED, linestyle=":", linewidth=1.2,
                                 label="случайная политика")
        axes[0, col].set_title(GAMES[game], loc="left", color=INK, fontsize=10)
        axes[-1, col].set_xlabel("шаги среды, млн")
    for row, (_, label) in enumerate(ROWS):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.grid(color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
