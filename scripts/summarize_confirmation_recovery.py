"""Повторяемая сводка paired recovery curve/AUC/harm по полному пилоту."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def summarize(source):
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["source_mode"] == "natural" and r["training_seed"] in (0, 1, 2)
            and r["repeat"] in (0, 1, 2, 3) and r["nominal_age"] in (50000, 200000)
            and r["game"] in ("asterix", "breakout")]
    assert len(rows) == 192
    panels = collections.defaultdict(dict)
    for row in rows:
        panels[(row["game"], row["trajectory_id"], row["nominal_age"], row["repeat"])][row["repair_id"]] = row
    assert len(panels) == 48 and all(len(v) == 4 for v in panels.values())
    result = {"source_sha256": hashlib.sha256(raw).hexdigest(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "estimand": "Paired differences; equal ages/repeats within trajectory, then equal source histories. Descriptive harm=negative paired final, no independence claim.",
              "by_game": {}}
    for game in ("asterix", "breakout"):
        records = {}
        for cohort, repeats in (("screen", (0, 1)), ("confirmation", (2, 3)), ("all_repeats", (0, 1, 2, 3))):
            group = [(key, value) for key, value in panels.items() if key[0] == game and key[3] in repeats]
            records[cohort] = {}
            for action in ("optimizer_reset", "head_reset", "head_and_optimizer_reset"):
                histories = []
                for seed in (0, 1, 2):
                    trajectory = f"natural_{game}_seed{seed}"
                    own = [v for k, v in group if k[1] == trajectory]
                    assert len(own) == 2 * len(repeats)
                    histories.append({"trajectory_id": trajectory,
                        "final_effect": mean(v[action]["j_final"] - v["continue"]["j_final"] for v in own),
                        "auc_effect": mean(v[action]["adaptation_auc"] - v["continue"]["adaptation_auc"] for v in own),
                        "curve_effect": {str(step): mean(v[action]["evaluation_curve"][i]["mean_return"] - v["continue"]["evaluation_curve"][i]["mean_return"] for v in own)
                                         for i, step in enumerate((0, 5000, 20000, 50000))}})
                effects = [v[action]["j_final"] - v["continue"]["j_final"] for _, v in group]
                records[cohort][action] = {
                    "history_records": histories,
                    "final_mean": mean(h["final_effect"] for h in histories),
                    "auc_mean": mean(h["auc_effect"] for h in histories),
                    "negative_panels": sum(v < 0 for v in effects),
                    "zero_panels": sum(v == 0 for v in effects), "panel_count": len(effects),
                    "paired_curve_means": {str(step): mean(h["curve_effect"][str(step)] for h in histories) for step in (0, 5000, 20000, 50000)}}
        result["by_game"][game] = records
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/minatar-repair-20261007/outcomes.jsonl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.source)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({game: {cohort: {action: {"auc_mean": value["auc_mean"], "history_auc": [h["auc_effect"] for h in value["history_records"]]}
                                    for action, value in records.items()}
                            for cohort, records in cohorts.items()}
                      for game, cohorts in result["by_game"].items()}, ensure_ascii=True))
