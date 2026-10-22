"""Repeat noise and age split on the full natural panel (576 branches) — sprint 02, T0, post-review.

Descriptive. Noise: per checkpoint, SD of each repair effect across the three repeat pairs vs the
spread (max - min) of pair-pooled action means; both averaged over checkpoints. Age split: effect
per history at each age (equal weight of repeats), then mean over the 6 histories.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.audit_expanded_natural import AGES, GAMES, MENU, SEEDS
from scripts.audit_full_natural_576 import COHORTS, PAIRS, REPEATS, ROOT, load_panels


def audit(source):
    raw = source.read_bytes()
    panels, cohort_sha, _ = load_panels(raw)
    games = {}
    for game in GAMES:
        sds, spreads = [], []
        for seed, age in itertools.product(SEEDS, AGES):
            pair = {p: {a: mean(panels[(game, seed, age, r)]["j_final"][a] for r in COHORTS[p]) for a in MENU}
                    for p in PAIRS}
            sds += [stdev(pair[p][a] - pair[p]["continue"] for p in PAIRS) for a in MENU[1:]]
            pooled = {a: mean(pair[p][a] for p in PAIRS) for a in MENU}
            spreads.append(max(pooled.values()) - min(pooled.values()))
        by_age = {str(age): {a: mean(mean(panels[(game, s, age, r)]["effects"][a] for r in REPEATS) for s in SEEDS)
                             for a in MENU[1:]} for age in AGES}
        games[game] = {"mean_sd_of_effect_across_repeat_pairs": mean(sds),
                       "mean_spread_of_action_means_per_checkpoint": mean(spreads),
                       "effects_by_age": by_age}
    return {"schema_version": 1, "audit_kind": "full_natural_576_noise_age",
            "source_sha256": hashlib.sha256(raw).hexdigest(), "selected_cohort_sha256": cohort_sha,
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "games": games, "note": "Computed after the T0 review; descriptive, not preregistered."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/minatar-repair-20261007/outcomes.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; audits are immutable")
    result = audit(args.source.resolve())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for game, g in result["games"].items():
        print(game, round(g["mean_sd_of_effect_across_repeat_pairs"], 3), round(g["mean_spread_of_action_means_per_checkpoint"], 3),
              {age: {a: round(v, 3) for a, v in e.items()} for age, e in g["effects_by_age"].items()})


if __name__ == "__main__":
    main()
