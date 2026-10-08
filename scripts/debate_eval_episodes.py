"""A round2: eval episodes needed for paired-diff SE < target, from per-episode returns in outcomes.jsonl (read-only).
Only eval noise is reduced; trajectory (training) noise is not."""
import json, collections, statistics as st, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows=[json.loads(l) for l in (ROOT/"runs/minatar-repair-20261007/outcomes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cell=collections.defaultdict(dict)
for r in rows: cell[(r["trajectory_id"],r["nominal_age"],r["repeat"])][r["repair_id"]]=r
sds=[]
for k,v in cell.items():
    c=v["continue"]["evaluation_curve"][-1]["returns"]
    for a,r in v.items():
        if a=="continue": continue
        d=[x-y for x,y in zip(r["evaluation_curve"][-1]["returns"],c)]
        sds.append((k,a,st.stdev(d)))
vals=sorted(s for _,_,s in sds)
q=lambda p: vals[min(len(vals)-1,int(p*len(vals)))]
print("paired-diff sd per episode: min %.2f median %.2f max %.2f"%(vals[0],st.median(vals),vals[-1]))
for tgt in (0.5,0.25):
    for name,s in (("median",st.median(vals)),("max",vals[-1])):
        print("target SE<%.2f, %s sd=%.2f -> n_episodes >= %d"%(tgt,name,s,math.ceil((s/tgt)**2)))
# compare with the 20 now
print("SE with n=20: median %.2f max %.2f"%(st.median(vals)/math.sqrt(20),vals[-1]/math.sqrt(20)))
