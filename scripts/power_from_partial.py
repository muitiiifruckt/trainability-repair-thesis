"""Planning power from completed paired screen outcomes (descriptive, no RL imports).

Independent unit = trajectory_id (source history); repeats/ages are averaged inside a history.
Needs >=3 complete histories per game, otherwise reports 'insufficient' (the earlier version used
repeats of one history as units and a z approximation; both were invalid, see research/debate/log.md).
Sample size uses t quantiles via simple search over noncentral-free approximation: n s.t. t_{.975,n-1}+t_{.8,n-1} <= |d|*sqrt(n)/sd.
"""
import json, collections, statistics as st, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
A = ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"]
T975 = {2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,12:2.179,15:2.145,20:2.093,30:2.045}
T80 = {2:1.061,3:0.978,4:0.941,5:0.920,6:0.906,7:0.896,8:0.889,9:0.883,10:0.879,12:0.873,15:0.866,20:0.860,30:0.854}
def q(tab, df):
    k = max(x for x in tab if x <= max(df, 2)); return tab[k]
rows = [json.loads(l) for l in (ROOT/"runs/minatar-repair-20261007/outcomes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cell = collections.defaultdict(dict)
for r in rows:
    if not r.get("failure_reason") and r["repair_id"] in A:
        cell[(r["game"], r["trajectory_id"], r["nominal_age"], r["repeat"])][r["repair_id"]] = r["j_final"]
per = collections.defaultdict(lambda: collections.defaultdict(list))
for (g, t, age, rep), v in cell.items():
    if all(a in v for a in A):
        for a in A[1:]: per[(g, a)][t].append(v[a]-v["continue"])
out = {}
for (g, a), d in sorted(per.items()):
    hs = [st.mean(x) for x in d.values()]; n = len(hs)
    if n < 3: out[f"{g}/{a}"] = {"n_histories": n, "status": "insufficient (<3 histories)"}; continue
    m, s = st.mean(hs), st.stdev(hs)
    need = None
    if m and s:
        for k in range(3, 400):
            if (q(T975, k-1)+q(T80, k-1)) * s / math.sqrt(k) <= abs(m): need = k; break
    out[f"{g}/{a}"] = {"n_histories": n, "mean": round(m,3), "sd_between_histories": round(s,3), "histories_for_80pct_power": need}
print(json.dumps(out, indent=1, ensure_ascii=False))
(ROOT/"research/results/parallel/power_from_partial.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
