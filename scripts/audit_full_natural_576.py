"""Stdlib audit of the complete natural panel (576 branches) — sprint 02, T0.

Extends scripts/audit_expanded_natural.py (384 branches) to repeats 0–5. Ages and
repeats get equal weight inside a trajectory_id, then histories get equal weight.
With 6 histories per game a t-interval over histories is reported (AGENTS.md allows
intervals from 5 histories); repeats and ages are not independent units.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.audit_expanded_natural import (AGES, EFFECTS, GAMES, MENU, SEEDS, STEPS, average_maps, choose,
                                            effect, summarize_histories)

ROOT = Path(__file__).resolve().parents[1]
REPEATS = tuple(range(6))
COHORTS = {"r01": (0, 1), "r23": (2, 3), "r45": (4, 5), "pooled": REPEATS}
PAIRS = ("r01", "r23", "r45")
T975_DF5 = 2.570581835636314  # Student t quantile, 6 histories


def interval(values):
    m = mean(values)
    half = T975_DF5 * stdev(values) / math.sqrt(len(values))
    return {"mean": m, "ci95": [m - half, m + half], "per_history": values,
            "positive_histories": sum(v > 0 for v in values)}


def load_panels(raw):
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert len({r["job_id"] for r in rows}) == len(rows), "Duplicate source job_id"
    selected = [r for r in rows if r.get("source_mode") == "natural" and r["game"] in GAMES
                and r["training_seed"] in SEEDS and r["nominal_age"] in AGES and r["repeat"] in REPEATS]
    observed = [(r["game"], r["training_seed"], r["nominal_age"], r["repeat"], r["repair_id"]) for r in selected]
    assert len(selected) == 576, f"Expected 576 branches, got {len(selected)}"
    assert set(observed) == set(itertools.product(GAMES, SEEDS, AGES, REPEATS, MENU)), "Incomplete coverage"
    selected.sort(key=lambda r: (r["game"], r["training_seed"], r["nominal_age"], r["repeat"], MENU.index(r["repair_id"])))
    canonical = (json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                            allow_nan=False) + "\n").encode("utf-8")
    raw_panels, hashes = collections.defaultdict(dict), collections.defaultdict(set)
    for r in selected:
        assert r.get("failure_reason") is None, f"Failed branch {r['job_id']}"
        assert r["completed_env_steps"] == r["budget_env_steps"] == 50000, "Incomplete budget"
        assert all(math.isfinite(r[k]) for k in ("j_pre", "j_immediate", "j_final", "adaptation_auc"))
        assert [p["env_steps"] for p in r["evaluation_curve"]] == list(STEPS), "Unexpected eval schedule"
        hashes[(r["game"], r["training_seed"], r["nominal_age"])].add(r["checkpoint_sha256"])
        raw_panels[(r["game"], r["training_seed"], r["nominal_age"], r["repeat"])][r["repair_id"]] = r
    assert all(len(h) == 1 for h in hashes.values()), "Checkpoint differs between repeats"
    panels = {}
    for key, actions in raw_panels.items():
        base = actions["continue"]
        for r in actions.values():
            assert r["continuation_seed"] == base["continuation_seed"], "Unmatched continuation seed"
            assert r["evaluation_seeds"] == base["evaluation_seeds"], "Unmatched eval seeds"
        final = {a: actions[a]["j_final"] for a in MENU}
        auc = {a: actions[a]["adaptation_auc"] for a in MENU}
        panels[key] = {"j_final": final, "effects": effect(final), "adaptation_auc": auc,
                       "auc_effects": effect(auc), "j_pre": base["j_pre"],
                       "curve_effects": {str(s): effect({a: actions[a]["evaluation_curve"][i]["mean_return"]
                                                         for a in MENU}) for i, s in enumerate(STEPS)}}
    return panels, hashlib.sha256(canonical).hexdigest(), len(rows)


def histories_for(panels, game, repeats):
    out = []
    for seed in SEEDS:
        group = [panels[(game, seed, age, r)] for age, r in itertools.product(AGES, repeats)]
        out.append({"trajectory_id": f"natural_{game}_seed{seed}", "seed": seed,
                    "j_final": average_maps(group, "j_final", MENU), "effects": average_maps(group, "effects", EFFECTS),
                    "adaptation_auc": average_maps(group, "adaptation_auc", MENU),
                    "auc_effects": average_maps(group, "auc_effects", EFFECTS),
                    "curve_effects": {str(s): {a: mean(p["curve_effects"][str(s)][a] for p in group) for a in EFFECTS}
                                      for s in STEPS}})
    return out


def repeat_split(panels, game):
    """Choose per checkpoint on one repeat pair, evaluate on another; all 6 ordered pairs."""
    per_direction = []
    for train, test in itertools.permutations(PAIRS, 2):
        contexts = []
        for seed, age in itertools.product(SEEDS, AGES):
            score = lambda pair: {a: mean(panels[(game, seed, age, r)]["j_final"][a] for r in COHORTS[pair]) for a in MENU}
            tr, te = score(train), score(test)
            contexts.append({"seed": seed, "train": tr, "test": te, "selected": choose(tr)[0]})
        sbs = choose({a: mean(c["train"][a] for c in contexts) for a in MENU})[0]
        per_direction.append({"train": train, "test": test, "train_sbs": sbs, "histories": [
            {"seed": seed,
             "winner_vs_continue": mean(c["test"][c["selected"]] - c["test"]["continue"] for c in contexts if c["seed"] == seed),
             "winner_vs_train_sbs": mean(c["test"][c["selected"]] - c["test"][sbs] for c in contexts if c["seed"] == seed)}
            for seed in SEEDS]})
    result = {}
    for key in ("winner_vs_continue", "winner_vs_train_sbs"):
        values = [mean(d["histories"][i][key] for d in per_direction) for i in range(len(SEEDS))]
        result[key] = interval(values)
    result["directions"] = [{k: d[k] for k in ("train", "test", "train_sbs")} for d in per_direction]
    result["note"] = ("Outcome-label repeat split on the same histories, not a diagnostic selector; "
                      "directions are dependent and do not add histories.")
    return result


def audit(source):
    raw = source.read_bytes()
    panels, cohort_sha, source_rows = load_panels(raw)
    games = {}
    for game in GAMES:
        cohorts = {name: summarize_histories(histories_for(panels, game, reps)) for name, reps in COHORTS.items()}
        pooled = histories_for(panels, game, REPEATS)
        games[game] = {"cohorts": cohorts,
                       "pooled_effect_intervals": {e: interval([h["effects"][e] for h in pooled]) for e in EFFECTS},
                       "pooled_auc_effect_intervals": {e: interval([h["auc_effects"][e] for h in pooled]) for e in EFFECTS},
                       "repeat_split_selection": repeat_split(panels, game)}
    return {"schema_version": 1, "audit_kind": "full_natural_576",
            "source": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
            "source_rows": source_rows, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_cohort_sha256": cohort_sha,
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cohorts": COHORTS, "games": games,
            "estimand": ("Paired final raw-return difference vs matched continue; equal weight of ages and repeats "
                         "inside trajectory_id, then of the 6 histories per game; t-interval over histories."),
            "limitations": ("6 histories per game; repeats and ages are dependent within a history; "
                            "no multiplicity correction; exploratory pilot, reserved games closed.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/minatar-repair-20261007/outcomes.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; audits are immutable")
    result = audit(args.source.resolve())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for game, data in result["games"].items():
        print(game, {e: (round(v["mean"], 3), [round(x, 3) for x in v["ci95"]], v["positive_histories"])
                     for e, v in data["pooled_effect_intervals"].items()})
        sel = data["repeat_split_selection"]
        print("  selection", {k: round(sel[k]["mean"], 3) for k in ("winner_vs_continue", "winner_vs_train_sbs")})


if __name__ == "__main__":
    main()
