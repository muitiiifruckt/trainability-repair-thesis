"""Descriptive H-int / H-age analysis on completed paired screen outcomes.

Unit of independence = trajectory_id (full source history); repeats and ages are averaged
within history first. Bootstrap resamples histories. Development-only, no multiplicity
correction; intervals are suppressed below 5 histories (a 2-history bootstrap is not informative, see research/debate/log.md).
"""
import json, collections, random, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
A = ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"]
path = ROOT/"runs/minatar-repair-20261007/outcomes.jsonl"
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
cell = collections.defaultdict(dict)  # (game,traj,age,repeat) -> action -> j_final
for r in rows:
    if r.get("failure_reason") or r["repair_id"] not in A: continue
    cell[(r["game"], r["trajectory_id"], r["nominal_age"], r["repeat"])][r["repair_id"]] = r["j_final"]

def contrasts(v):
    c = v["continue"]
    return {"opt": v["optimizer_reset"]-c, "head": v["head_reset"]-c, "joint": v["head_and_optimizer_reset"]-c,
            "interaction": v["head_and_optimizer_reset"]-v["head_reset"]-v["optimizer_reset"]+c}

# history x age -> mean over repeats
hist = collections.defaultdict(lambda: collections.defaultdict(list))
for (g, t, age, rep), v in cell.items():
    if all(a in v for a in A):
        for k, x in contrasts(v).items(): hist[(g, t)][(age, k)].append(x)
def per_hist(g, key, ages=None):
    out = []
    for (gg, t), d in hist.items():
        if gg != g: continue
        vals = [st.mean(x) for (age, k), x in d.items() if k == key and (ages is None or age in ages)]
        if vals: out.append(st.mean(vals))
    return out
def boot(xs, n=2000, seed=0):
    if len(xs) < 5: return "suppressed(<5 histories)"
    rng = random.Random(seed); m = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return [round(m[int(.025*n)], 3), round(m[int(.975*n)], 3)]
res = {"n_complete_cells": sum(all(a in v for a in A) for v in cell.values()), "games": {}}
for g in sorted({k[0] for k in cell}):
    res["games"][g] = {}
    for key in ["opt", "head", "joint", "interaction"]:
        xs = per_hist(g, key)
        res["games"][g][key] = {"n_histories": len(xs), "mean": round(st.mean(xs), 3) if xs else None, "boot95": boot(xs)}
    ages = sorted({a for d in hist.values() for (a, k) in d})
    res["games"][g]["head_by_age"] = {str(a): [round(st.mean(per_hist(g, "head", {a})), 3), len(per_hist(g, "head", {a}))]
                                      for a in ages if per_hist(g, "head", {a})}
out = ROOT/"research/results/parallel/interaction_age.json"
out.write_text(json.dumps(res, indent=1), encoding="utf-8"); print(json.dumps(res, indent=1))
