"""Fixed-TD laboratory contrasts vs online repair effects on the same checkpoints — sprint 02, T0.

Descriptive only. Fixed-TD: continue heldout Huber loss minus repaired loss (positive = better
fitting) from mechanism_summary.json, lr 2.5e-4, old_target teacher, averaged over its 2 repeats.
Online: final raw return of the repair minus matched continue, averaged over repeats 0–5.
A checkpoint is (game, seed, age), matched by checkpoint_id only (no checkpoint SHA in the
mechanism summary); a history is equal-weight over its 2 ages. Fixed-TD contrasts with
|value| < TIE are ties (sign 0); at horizon 2000 Adam reset is within 6e-4 of continue on every checkpoint.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean

from scripts.audit_expanded_natural import MENU

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research/results/minatar-repair-20261007"
REPAIRS = MENU[1:]
HORIZONS = (100, 500, 2000)
MAIN = dict(horizon=2000, evaluation_teacher="old_target")
TIE = 1e-3


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def best(scores):
    return max(MENU, key=lambda a: scores[a])


def sign(x, tie=0.0):
    return 0 if abs(x) < tie else (1 if x > 0 else -1)


def agree(lab, live):
    return sum(sign(p, TIE) == sign(q) for p, q in zip(lab, live))


def with_interaction(scores):
    return dict(scores, interaction=scores["head_and_optimizer_reset"] - scores["head_reset"]
                - scores["optimizer_reset"] + scores["continue"])


def fixed_td(summary, horizon, teacher):
    cells = collections.defaultdict(list)
    for r in summary["paired_action_contrasts"]["raw_pairs"]:
        if (r["domain"], r["family"], r["split"], r["metric"]) != ("fixed_td", "baseline", "heldout", "huber"):
            continue
        if r["lr"] != 0.00025 or r["horizon"] != horizon or r["evaluation_teacher"] != teacher:
            continue
        if r["action"] in REPAIRS:
            assert r["valid_pair"] and r["failure"] is None
            cells[(r["checkpoint_id"], r["action"])].append(r["value"])
    out = {}
    for checkpoint in sorted({c for c, _ in cells}):
        assert all(len(cells[(checkpoint, a)]) == 2 for a in REPAIRS), checkpoint
        out[checkpoint] = with_interaction({"continue": 0.0, **{a: mean(cells[(checkpoint, a)]) for a in REPAIRS}})
    return out


def online(outcomes):
    cells = collections.defaultdict(dict)
    for r in outcomes:
        if r.get("source_mode") == "natural" and r["repeat"] in range(6):
            cells[(r["checkpoint_id"], r["repeat"])][r["repair_id"]] = r["j_final"]
    out = {}
    for checkpoint in sorted({c for c, _ in cells}):
        panels = [cells[(checkpoint, k)] for k in range(6)]
        assert all(set(p) == set(MENU) for p in panels), checkpoint
        out[checkpoint] = with_interaction({a: mean(p[a] - p["continue"] for p in panels) for a in MENU})
    return out


def compare(lab, live):
    rows = []
    for checkpoint, scores in lab.items():
        game, seed, age = checkpoint.split("_")[1], int(checkpoint.split("seed")[1].split("_")[0]), int(checkpoint.split("age")[1])
        on = live[checkpoint]
        chosen = best(scores)
        tied = [a for a in MENU if scores[chosen] - scores[a] < TIE]
        rows.append({"checkpoint_id": checkpoint, "game": game, "seed": seed, "age": age, "fixed_td": scores,
                     "online": on, "fixed_td_choice": chosen, "fixed_td_tied_with_choice": tied,
                     "online_best": best(on), "online_effect_of_fixed_td_choice": on[chosen] - on["continue"]})
    games = {}
    for game in sorted({r["game"] for r in rows}):
        g = [r for r in rows if r["game"] == game]
        seeds = sorted({r["seed"] for r in g})
        per_action = {}
        for a in (*REPAIRS, "interaction"):
            x, y = [r["fixed_td"][a] for r in g], [r["online"][a] for r in g]
            hx = [mean(r["fixed_td"][a] for r in g if r["seed"] == s) for s in seeds]
            hy = [mean(r["online"][a] for r in g if r["seed"] == s) for s in seeds]
            per_action[a] = {"sign_agreement_checkpoints": agree(x, y),
                             "fixed_td_positive_checkpoints": sum(sign(p, TIE) > 0 for p in x),
                             "fixed_td_tie_checkpoints": sum(sign(p, TIE) == 0 for p in x),
                             "online_positive_checkpoints": sum(q > 0 for q in y),
                             "spearman_checkpoints": spearman(x, y),
                             "history_fixed_td": hx, "history_online": hy,
                             "sign_agreement_histories": agree(hx, hy)}
        games[game] = {"histories": len(seeds), "checkpoints": len(g), "per_action": per_action,
                       "choice_counts": dict(collections.Counter(r["fixed_td_choice"] for r in g)),
                       "online_best_counts": dict(collections.Counter(r["online_best"] for r in g)),
                       "choice_matches_online_best": sum(r["fixed_td_choice"] == r["online_best"] for r in g),
                       "online_best_within_fixed_td_tie": sum(r["online_best"] in r["fixed_td_tied_with_choice"] for r in g),
                       "history_online_effect_of_fixed_td_choice":
                           [mean(r["online_effect_of_fixed_td_choice"] for r in g if r["seed"] == s) for s in seeds]}
    return rows, games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; audits are immutable")
    summary_raw = (RUN / "mechanism_summary.json").read_bytes()
    outcomes_raw = (ROOT / "runs/minatar-repair-20261007/outcomes.jsonl").read_bytes()
    summary = json.loads(summary_raw)
    live = online(json.loads(line) for line in outcomes_raw.decode("utf-8").splitlines() if line.strip())
    variants = {}
    for horizon, teacher in itertools.product(HORIZONS, ("old_target", "refreshed_online")):
        rows, games = compare(fixed_td(summary, horizon, teacher), live)
        variants[f"h{horizon}_{teacher}"] = {"horizon": horizon, "evaluation_teacher": teacher, "games": games,
                                             **({"checkpoints": rows} if dict(horizon=horizon, evaluation_teacher=teacher) == MAIN else {})}
    result = {"schema_version": 1, "audit_kind": "fixed_td_vs_online",
              "mechanism_summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
              "outcomes_sha256": hashlib.sha256(outcomes_raw).hexdigest(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "main_variant": f"h{MAIN['horizon']}_{MAIN['evaluation_teacher']}", "variants": variants,
              "signs": "fixed_td: continue loss minus repaired loss; online: repair j_final minus continue; positive = repair better",
              "fixed_td_tie": TIE,
              "limitations": ("Descriptive and post hoc; fixed-TD covers asterix seeds 0-3 and breakout seeds 0-4 only; "
                              "ages and repeats are dependent within a history; no intervals; fixed-TD is a "
                              "2000-update fitting outcome on cached data with old-target training, not 50k-step return; "
                              "checkpoints matched by id only.")}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, v in variants.items():
        for game, g in v["games"].items():
            print(name, game, "match", g["choice_matches_online_best"], "/", g["checkpoints"],
                  "within tie", g["online_best_within_fixed_td_tie"], g["choice_counts"],
                  {a: (s["sign_agreement_checkpoints"], round(s["spearman_checkpoints"] or 0, 2))
                   for a, s in g["per_action"].items()})


if __name__ == "__main__":
    main()
