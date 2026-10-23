"""Exploratory look at stage B of the deficit test (outside the preregistered rule; notes/ppo_deficit.md).

Per history and shift: AUC of each chain round, telemetry along the chain and in the round-5 branches,
how many positions of the round-5 mapping coincide with an earlier round, and the stage A deficit.

Usage: .venv\\Scripts\\python.exe -m scripts.explore_deficit_B --runs runs/causal-trainability/deficit_B
           --stage-a research/results/ppo-deficit/stage_A.json --output OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from causal_trainability.deficit import ARMS, SHIFTS, function_of, shift_spec

TAIL = 0.1  # share of updates averaged at the start and at the end of a run


def telemetry(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    k = max(1, int(len(rows) * TAIL))

    def at(part, key, layer=None):
        get = (lambda r: r["layers"][layer][key]) if layer else (lambda r: r["metrics"][key])
        return mean(get(r) for r in part)

    return {f"{name}_{edge}": at(part, key, layer)
            for edge, part in (("start", rows[:k]), ("end", rows[-k:]))
            for name, key, layer in (("entropy", "entropy", None), ("explained_variance", "explained_variance", None),
                                     ("fc_dead", "dead_fraction", "fc"), ("fc_rank", "effective_rank", "fc"))}


def coincidences(cfg, kind, round_):
    """Positions (channels, or the two moves) whose meaning in round `round_` repeats an earlier round."""
    n = cfg["num_actions"] if kind == "actions" else cfg["input_channels"]
    maps = [list(range(n))] + [shift_spec(cfg, kind, r)["perm"] for r in range(1, round_ + 1)]
    funcs = [function_of(m, kind, cfg["game"]) for m in maps]
    last = funcs[-1]
    return {"any_earlier": sum(any(f[i] == x for f in funcs[:-1]) for i, x in enumerate(last)),
            "original": sum(funcs[0][i] == x for i, x in enumerate(last)), "positions": len(last)}


def explore(runs, stage_a):
    runs = Path(runs)
    a = json.loads(Path(stage_a).read_text(encoding="utf-8"))["shifts"]
    a_deficit = {(kind, h["history"]): h["deficit_auc"] for kind, s in a.items() for h in s["per_history"]}
    rows = []
    for folder in sorted(p for p in runs.iterdir() if p.is_dir()):
        for kind in SHIFTS:
            if not (folder / f"{kind}_chain" / "chain.json").exists():
                continue
            chain = json.loads((folder / f"{kind}_chain" / "chain.json").read_text(encoding="utf-8"))
            branches = {(arm, rep): json.loads(p.read_text(encoding="utf-8"))
                        for arm in ARMS for rep in (0, 1)
                        for p in [folder / f"{kind}_r5" / f"{arm}_rep{rep}" / "result.json"] if p.exists()}
            row = {"history": folder.name, "kind": kind,
                   "chain_auc": [r["auc"] for r in chain["rounds"]],
                   "chain_telemetry": [telemetry(folder / f"{kind}_chain" / f"round{r['round']}" / "telemetry"
                                                 / f"chain_{folder.name}_{kind}_round{r['round']}.jsonl")
                                       for r in chain["rounds"]],
                   "stage_a_deficit_auc": a_deficit.get((kind, folder.name))}
            if len(branches) == 2 * len(ARMS):
                cfg = branches[("fresh", 0)]["config"]
                arm_mean = lambda arm, key: mean(branches[(arm, rep)][key] for rep in (0, 1))
                first = lambda arm: mean(mean(w for w in branches[(arm, rep)]["auc_windows"][:5] if w is not None)
                                         for rep in (0, 1))
                row.update({
                    "auc": {arm: arm_mean(arm, "auc") for arm in ARMS},
                    "final_return": {arm: arm_mean(arm, "final_return") for arm in ARMS},
                    "start_return": {arm: arm_mean(arm, "start_return") for arm in ARMS},
                    "deficit_auc": arm_mean("fresh", "auc") - arm_mean("aged", "auc"),
                    # The first 250k steps of a round-5 branch, comparable with one chain round.
                    "first_250k_auc": {arm: first(arm) for arm in ARMS},
                    "coincidences": coincidences(cfg, kind, 5),
                    "branch_telemetry": {f"{arm}_rep{rep}": telemetry(folder / f"{kind}_r5" / f"{arm}_rep{rep}" / "telemetry"
                                                                      / f"{b['config']['run_name']}.jsonl")
                                         for (arm, rep), b in branches.items()}})
            rows.append(row)
    return {"schema_version": 1, "runs": runs.as_posix(), "exploratory": True, "rule": "notes/ppo_deficit.md",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = explore(args.runs, args.stage_a)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in result["rows"]:
        line = [r["history"], r["kind"], "chain", [round(x, 2) for x in r["chain_auc"]],
                "entropy end", [round(t["entropy_end"], 2) for t in r["chain_telemetry"]]]
        if "deficit_auc" in r:
            line += ["r5 aged/fresh", round(r["auc"]["aged"], 2), round(r["auc"]["fresh"], 2),
                     "deficit B", round(r["deficit_auc"], 2), "A", round(r["stage_a_deficit_auc"], 2),
                     "coinc", f"{r['coincidences']['any_earlier']}/{r['coincidences']['positions']}"]
        print(*line)


if __name__ == "__main__":
    main()
