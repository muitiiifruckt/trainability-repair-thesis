"""A-cycle 1: immediate/recovery decomposition of repair effects on completed screen outcomes (read-only)."""
import json, collections, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT/"runs/minatar-repair-20261007/outcomes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cell = collections.defaultdict(dict)
for r in rows:
    if r.get("failure_reason"): continue
    cell[(r["trajectory_id"], r["nominal_age"], r["repeat"])][r["repair_id"]] = r
out = []
for k, v in sorted(cell.items()):
    c = v["continue"]
    for a, r in v.items():
        if a == "continue": continue
        total = r["j_final"] - c["j_final"]
        imm = r["j_immediate"] - c["j_immediate"]
        # recovery = growth over budget relative to continue's growth
        rec = (r["j_final"] - r["j_immediate"]) - (c["j_final"] - c["j_immediate"])
        auc = r["adaptation_auc"] - c["adaptation_auc"]
        mid = [e["mean_return"] for e in r["evaluation_curve"]]
        cm = [e["mean_return"] for e in c["evaluation_curve"]]
        out.append(dict(key=k, action=a, total=round(total,3), imm=round(imm,3), rec=round(rec,3), auc=round(auc,3),
                        curve_delta=[round(x-y,2) for x,y in zip(mid,cm)]))
        print(k, a, "total", round(total,2), "imm", round(imm,2), "rec", round(rec,2), "auc", round(auc,2), [round(x-y,2) for x,y in zip(mid,cm)])
# correlation of immediate drop vs final effect for head_reset
hr=[o for o in out if o["action"]=="head_reset"]
print("head_reset: imm vs total", [(o["imm"],o["total"]) for o in hr])
(ROOT/"research/debate/cost_decomp.json").write_text(json.dumps(out, indent=1, default=list), encoding="utf-8")
