"""Descriptive per-game summary of the main screen (paired effects vs continue).

Immediate effect = j_immediate(action) - j_immediate(continue) on the same continuation/evaluation seeds
(j_immediate - j_pre is biased, see research/decisions.md D6). Final effect = j_final(action) - j_final(continue).
Development-only, history is the independent unit; no intervals below 5 histories.
"""
import json, collections, statistics as st, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
src = ROOT/"runs/minatar-repair-20261007/outcomes.jsonl"
raw = src.read_bytes()
rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
A = ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"]
cell = collections.defaultdict(dict)
for r in rows:
    if not r.get("failure_reason") and r["repair_id"] in A:
        cell[(r["game"], r["trajectory_id"], r["nominal_age"], r["repeat"])][r["repair_id"]] = r
out = {"source_rows": len(rows), "source_sha256": hashlib.sha256(raw).hexdigest(), "games": {}}
for g in sorted({k[0] for k in cell}):
    cells = {k: v for k, v in cell.items() if k[0] == g and all(a in v for a in A)}
    if not cells: continue
    G = {"complete_cells": len(cells), "histories": len({k[1] for k in cells}),
         "continue_final_mean": round(st.mean(v["continue"]["j_final"] for v in cells.values()), 2), "by_age": {}}
    for age in sorted({k[2] for k in cells}):
        sub = [v for k, v in cells.items() if k[2] == age]
        A_ = {}
        for a in A[1:]:
            imm = [v[a]["j_immediate"] - v["continue"]["j_immediate"] for v in sub]
            fin = [v[a]["j_final"] - v["continue"]["j_final"] for v in sub]
            A_[a] = {"immediate_mean": round(st.mean(imm), 2), "immediate_negative": f"{sum(x < 0 for x in imm)}/{len(imm)}",
                     "final_mean": round(st.mean(fin), 2), "final_negative": f"{sum(x < 0 for x in fin)}/{len(fin)}"}
        G["by_age"][str(age)] = {"cells": len(sub), "effects": A_}
    hist = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in cells.items():
        for a in A[1:]: hist[k[1]][a].append(v[a]["j_final"] - v["continue"]["j_final"])
    G["final_effect_by_history"] = {h.replace("natural_", ""): {a: round(st.mean(x), 2) for a, x in d.items()} for h, d in sorted(hist.items())}
    G["final_effect_mean"] = {a: round(st.mean(v[a]["j_final"] - v["continue"]["j_final"] for v in cells.values()), 2) for a in A[1:]}
    out["games"][g] = G
dst = ROOT/"research/results/parallel/screen_summary.json"
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
