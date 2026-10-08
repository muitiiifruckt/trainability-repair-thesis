"""A-cycle 1b: eval-noise floor. SE of final-eval difference from 20 per-episode returns (eval seeds paired? check)."""
import json, collections, statistics as st, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT/"runs/minatar-repair-20261007/outcomes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cell = collections.defaultdict(dict)
for r in rows: cell[(r["trajectory_id"], r["nominal_age"], r["repeat"])][r["repair_id"]] = r
res=[]
for k,v in sorted(cell.items()):
    c=v["continue"]; cf=c["evaluation_curve"][-1]
    for a,r in v.items():
        if a=="continue": continue
        rf=r["evaluation_curve"][-1]
        same_seeds = rf["evaluation_seeds"]==cf["evaluation_seeds"]
        d=[x-y for x,y in zip(rf["returns"],cf["returns"])]
        se_pair=st.stdev(d)/math.sqrt(len(d))
        se_ind=math.sqrt(st.variance(rf["returns"])/20+st.variance(cf["returns"])/20)
        res.append((k,a,round(rf["mean_return"]-cf["mean_return"],2),round(se_pair,2),round(se_ind,2),same_seeds))
        print(k,a,res[-1][2:])
sds=[st.stdev(e["returns"]) for r in rows for e in r["evaluation_curve"][-1:]]
print("episode sd of final eval: mean",round(st.mean(sds),2),"min",round(min(sds),2),"max",round(max(sds),2))
