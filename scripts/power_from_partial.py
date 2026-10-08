"""Rough power estimate from completed paired screen outcomes (descriptive, no RL imports).

Uses between-checkpoint spread of paired effects (action - continue) to estimate how many
independent source histories a one-sample t test needs for 80% power at alpha .05.
Early, tiny sample: the sd is itself very uncertain, so treat output as planning order of magnitude.
"""
import json, collections, statistics as st, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT/"runs/minatar-repair-20261007/outcomes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
d = collections.defaultdict(dict)
for r in rows:
    if r.get("failure_reason"): continue
    d[(r["checkpoint_id"], r["repeat"])][r["repair_id"]] = r["j_final"]
eff = collections.defaultdict(list)
for key, v in d.items():
    if "continue" not in v or len(v) < 4: continue
    for a, x in v.items():
        if a != "continue": eff[a].append(x - v["continue"])
z = 2.8016  # z_.975 + z_.8
out = {"n_complete_pairs": len({k for k, v in d.items() if len(v) == 4}), "actions": {}}
for a, e in eff.items():
    m, s = st.mean(e), (st.stdev(e) if len(e) > 1 else float("nan"))
    n = math.ceil((z * s / m) ** 2) + 1 if m and s == s else None
    out["actions"][a] = {"mean_effect": round(m, 3), "sd": round(s, 3), "n_units": len(e),
                         "histories_for_80pct_power": n}
print(json.dumps(out, indent=1, ensure_ascii=False))
(ROOT/"research/results/parallel/power_from_partial.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
