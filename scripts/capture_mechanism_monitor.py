"""Compact read-only audit of completed mechanisms and synthetic LR controls.

Losses are conditional fitting outcomes. No learner imports or policy selection.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(result_dir, run_dir, output):
    summary_path = result_dir / "mechanism_summary.json"
    synthetic_path = result_dir / "synthetic_lr_sweep.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    assert summary["coverage"]["fixed_probe_files"] == 24
    assert not summary["coverage"]["missing_declared_probe_jobs"]
    assert summary["coverage"]["failed_fixed_branches"] == 0
    assert len(synthetic["branches"]) == 1728
    for source in summary["sources"]:
        assert digest(Path(source["path"])) == source["sha256"], source["path"]
    panels = collections.defaultdict(dict)
    for row in synthetic["branches"]:
        assert not row["failure"] and math.isfinite(row["heldout_loss_final"])
        key = (row["scenario"], row["training_seed"], row["source_updates"],
               row["continuation_repeat"], row["continuation_lr"], row["horizon"])
        assert row["action"] not in panels[key]
        panels[key][row["action"]] = row
    assert len(panels) == 288
    contrasts = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, panel in panels.items():
        assert len(panel) == 6
        assert len({row["tape_sha256"] for row in panel.values()}) == 1
        base = panel["continue"]["heldout_loss_final"]
        assert base > 0
        for action, row in panel.items():
            if action == "continue":
                continue
            group = (key[0], key[4], key[5], action)
            contrasts[group][key[1]].append((base - row["heldout_loss_final"]) / base)
    synthetic_effects = []
    for (scenario, lr, horizon, action), histories in sorted(contrasts.items()):
        means = {str(seed): statistics.mean(values) for seed, values in histories.items()}
        assert len(means) == 3
        synthetic_effects.append({"scenario": scenario, "continuation_lr": lr,
                                  "horizon": horizon, "action": action,
                                  "relative_loss_advantage_mean": statistics.mean(means.values()),
                                  "source_seed_means": means,
                                  "normalization": "(continue heldout MSE - repaired MSE) / continue MSE, then ages/repeats within seed"})
    fixed_curves = [row for row in summary["curve_statistics"]
                    if row["domain"] == "fixed_td" and row["family"] == "baseline"
                    and row["evaluation_teacher"] == "old_target"
                    and row["split"] == "heldout" and row["metric"] == "huber"]
    data = {
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kind": "completed_conditional_fitting_controls",
        "capture_script_sha256": digest(Path(__file__)),
        "source_files": [{"path": str(summary_path), "sha256": digest(summary_path)},
                         {"path": str(synthetic_path), "sha256": digest(synthetic_path)}],
        "raw_source_files": summary["sources"],
        "coverage": summary["coverage"],
        "costs": summary["costs"],
        "fixed_td_old_target_heldout_huber": fixed_curves,
        "synthetic_relative_effects": synthetic_effects,
        "lr_adjustment_descriptions": summary["lr_adjustment_descriptions"],
        "t_reset_equivalence": summary["t_reset_equivalence"],
        "limitations": summary["limitations"],
        "progress": json.loads((run_dir / "progress.json").read_text(encoding="utf-8")),
        "source_dependency_manifest": json.loads((run_dir / "dependency_manifest.json").read_text(encoding="utf-8")),
        "classification": "fixed-TD repair fitting benefit absent for joint at 2000 updates; synthetic LR sensitivity present; online mechanisms and selection remain uncertain",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"fixed_jobs": data["coverage"]["fixed_probe_files"],
                      "synthetic_branches": len(synthetic["branches"]),
                      "t_control_counts": data["t_reset_equivalence"]["counts"],
                      "output": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture(args.result_dir, args.run_dir, args.output)
