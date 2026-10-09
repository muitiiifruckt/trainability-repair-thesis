"""Keep absolute loss contrasts and two distinct normalizations separate."""
import argparse
import collections
import hashlib
import json
import statistics as st
from pathlib import Path


def inspect(source, destination):
    raw = source.read_bytes()
    data = json.loads(raw)
    panels = collections.defaultdict(dict)
    for row in data["branches"]:
        key = (row["scenario"], row["training_seed"], row["source_updates"],
               row["continuation_repeat"], row["continuation_lr"], row["horizon"])
        assert not row["failure"] and row["action"] not in panels[key]
        panels[key][row["action"]] = row["heldout_loss_final"]
    groups = collections.defaultdict(list)
    for key, losses in panels.items():
        if key[0] != "target_scale_drop":
            continue
        for baseline, repair in (("continue", "optimizer_reset"),
                                 ("head_reset", "head_and_optimizer_reset")):
            b, a = losses[baseline], losses[repair]
            assert b > 0
            groups[(key[4], key[5], baseline, repair)].append(
                {"seed": key[1], "source_age": key[2], "repeat": key[3],
                 "baseline_loss": b, "repair_loss": a,
                 "absolute_advantage": b-a, "relative_advantage": (b-a)/b})
    records = []
    for (lr, horizon, baseline, repair), rows in sorted(groups.items()):
        seed_rows = {seed: [row for row in rows if row["seed"] == seed]
                     for seed in sorted({row["seed"] for row in rows})}
        assert len(seed_rows) == 3 and all(len(rr) == 4 for rr in seed_rows.values())
        by_seed = {str(seed): {"absolute_advantage": st.mean(row["absolute_advantage"] for row in rr),
                             "baseline_loss": st.mean(row["baseline_loss"] for row in rr),
                             "mean_panel_relative_advantage": st.mean(row["relative_advantage"] for row in rr)}
                   for seed, rr in seed_rows.items()}
        by_age = {str(age): {"mean_panel_relative_advantage": st.mean(
                              row["relative_advantage"] for row in rows if row["source_age"] == age),
                            "positive_pairs": sum(row["absolute_advantage"] > 0 for row in rows
                                                  if row["source_age"] == age), "pairs": 6}
                  for age in sorted({row["source_age"] for row in rows})}
        records.append({"lr": lr, "horizon": horizon, "baseline": baseline, "repair": repair,
                        "absolute_advantage": st.mean(row["absolute_advantage"] for row in rows),
                        "ratio_of_grand_mean_losses": st.mean(row["absolute_advantage"] for row in rows)
                                                     / st.mean(row["baseline_loss"] for row in rows),
                        "mean_panel_ratios": st.mean(row["mean_panel_relative_advantage"]
                                                     for row in by_seed.values()),
                        "by_seed": by_seed, "by_age": by_age})
    result = {"source_sha256": hashlib.sha256(raw).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "relative_sign": "positive means lower repaired heldout MSE",
              "ratio_of_grand_mean_losses": "mean(b-a)/mean(b); baseline-loss-weighted panel ratios",
              "mean_panel_ratios": "mean((b-a)/b); equal ages/repeats within seed, then equal seeds",
              "independent_unit": "source seed, not source ages or continuation repeats",
              "records": records}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"source_sha256": result["source_sha256"], "groups": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    inspect(args.source, args.destination)
