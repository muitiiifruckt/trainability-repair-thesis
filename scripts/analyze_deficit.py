"""Analysis of the live adaptation deficit (sprint 02, part 2, T7; rule in notes/ppo_deficit.md).

Usage: .venv\\Scripts\\python.exe -m scripts.analyze_deficit --runs runs/causal-trainability/deficit_A --output OUT.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

HISTORIES = 6
REPEATS = (0, 1)
ARMS = ("aged", "fresh")
T975_DF5 = 2.570581835636314  # two-sided 95%, 6 histories
T9875_DF5 = 3.163381449748609  # two-sided 97.5% (Bonferroni over the two shifts), 6 histories
DELTA_SHARE = 0.10  # "practically no deficit": 95% upper bound below 10% of the mean fresh AUC


def interval(values, t):
    m, half = mean(values), t * stdev(values) / math.sqrt(len(values))
    return [m - half, m + half]


def pooled_sd(groups):
    """Square root of the mean within-group variance."""
    return math.sqrt(mean(stdev(g) ** 2 for g in groups))


def comparable(config):
    return {k: v for k, v in config.items() if k != "run_name"}


def analyze(runs):
    paths = sorted(Path(runs).glob("*/*/*/result.json"))
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    cells = collections.defaultdict(dict)
    for r in rows:
        cells[(r["shift"]["kind"], r["history"], r["repeat"])][r["arm"]] = r
    shifts = {}
    for kind in sorted({k for k, _, _ in cells}):
        histories = sorted({h for k, h, _ in cells if k == kind})
        if len(histories) != HISTORIES:
            raise SystemExit(f"{kind}: {len(histories)} histories, the rule needs exactly {HISTORIES}")
        per_history = []
        for h in histories:
            pairs = [cells.get((kind, h, rep), {}) for rep in REPEATS]
            if any(set(p) != set(ARMS) for p in pairs):
                raise SystemExit(f"{kind} {h}: incomplete pairs")
            for p in pairs:
                a, f = p["aged"], p["fresh"]
                if (a["shift"]["perm"] != f["shift"]["perm"] or comparable(a["config"]) != comparable(f["config"])
                        or a["revision"]["files"] != f["revision"]["files"] or a["seeds"] != f["seeds"]):
                    raise SystemExit(f"{kind} {h} repeat {a['repeat']}: aged and fresh branches are not paired")
            empty = {arm: [sum(w is None for w in p[arm]["auc_windows"]) for p in pairs] for arm in ARMS}
            if any(n > 1 for v in empty.values() for n in v):
                raise SystemExit(f"{kind} {h}: more than one empty window; see notes/ppo_deficit.md")
            diffs = {key: [p["fresh"][key] - p["aged"][key] for p in pairs] for key in ("auc", "final_return")}
            per_history.append({"history": h, "deficit_auc": mean(diffs["auc"]), "deficit_final": mean(diffs["final_return"]),
                                "paired_auc_differences": diffs["auc"], "empty_windows": empty,
                                **{key: {arm: [p[arm][key] for p in pairs] for arm in ARMS}
                                   for key in ("auc", "final_return", "start_return")},
                                "action_perm": pairs[0]["aged"]["shift"]["perm"]})
        values = [h["deficit_auc"] for h in per_history]
        finals = [h["deficit_final"] for h in per_history]
        fresh_auc = mean(mean(h["auc"]["fresh"]) for h in per_history)
        ci95, ci975 = interval(values, T975_DF5), interval(values, T9875_DF5)
        positive = sum(v > 0 for v in values)
        delta = DELTA_SHARE * fresh_auc
        verdict = ("deficit" if ci975[0] > 0 and positive >= 5 else
                   "practically_none" if ci95[1] < delta else "undetermined")
        shifts[kind] = {
            "verdict": verdict, "mean_deficit_auc": mean(values), "ci95_deficit_auc": ci95,
            "ci975_bonferroni_deficit_auc": ci975, "positive_histories": positive, "delta": delta,
            "mean_fresh_auc": fresh_auc, "mean_aged_auc": mean(mean(h["auc"]["aged"]) for h in per_history),
            "mean_deficit_final": mean(finals), "ci95_deficit_final": interval(finals, T975_DF5),
            "transfer_but_worse_final": sum(h["deficit_auc"] <= 0 and h["deficit_final"] > 0 for h in per_history),
            "noise": {"within_history_sd_auc": {arm: pooled_sd([h["auc"][arm] for h in per_history]) for arm in ARMS},
                      "within_history_sd_paired_difference": pooled_sd([h["paired_auc_differences"] for h in per_history]),
                      "between_history_sd_deficit": stdev(values)},
            "per_history": per_history}
    return {"schema_version": 1, "runs": Path(runs).as_posix(), "branches": len(rows),
            "results_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest(),
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shifts": shifts, "deficit_shown": any(s["verdict"] == "deficit" for s in shifts.values()),
            "rule": "notes/ppo_deficit.md"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; analyses are immutable")
    result = analyze(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for kind, s in result["shifts"].items():
        print(kind, s["verdict"], round(s["mean_deficit_auc"], 3), [round(x, 3) for x in s["ci975_bonferroni_deficit_auc"]],
              s["positive_histories"], "of 6; noise", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["noise"].items()})


if __name__ == "__main__":
    main()
