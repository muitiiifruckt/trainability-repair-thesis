"""[B2] Факты по реальным данным кампании (только чтение runs/minatar-repair-20261007/*.jsonl). 1 поток, numpy.
(1) масштаб эффектов y в единицах sd возвратов эпизодов continue и «потолок» пользы идеального оракула знака (для δ=0.2 sd из prereg);
(2) collinearity clock-признаков (age, epsilon, replay_size) при двух возрастах;
(3) насколько whitelist-признаки кодируют игру (разделимость breakout/asterix);
(4) оценка числа историй/времени до n>=12."""
import json, sys, collections
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/minatar-repair-20261007"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
out, cps, diags = rd(RUN / "outcomes.jsonl"), {r["checkpoint_id"]: r for r in rd(RUN / "checkpoints.jsonl")}, {r["checkpoint_id"]: r for r in rd(RUN / "diagnostics.jsonl")}
cfg = json.loads((RUN / "config.json").read_text(encoding="utf-8"))["learner"]
by = collections.defaultdict(dict)
for r in out: by[(r["checkpoint_id"], r["repeat"])][r["repair_id"]] = r
print(f"outcomes: {len(out)} строк, ячеек (checkpoint, repeat): {len(by)}; историй: {len({r['trajectory_id'] for r in out})}")

# (1) масштаб y
for game in sorted({r["game"] for r in out}):
    ep = [x for r in out if r["game"] == game and r["repair_id"] == "continue" for x in r["evaluation_curve"][-1]["returns"]]
    sd = float(np.std(ep, ddof=1)); print(f"[{game}] sd возвратов эпизода (continue, финальные eval) = {sd:.2f} по {len(ep)} эпизодам; среднее {np.mean(ep):.2f}")
    for act in ("optimizer_reset", "head_reset", "head_and_optimizer_reset"):
        cells = [(k, v[act]["j_final"] - v["continue"]["j_final"]) for k, v in by.items() if cps[k[0]]["game"] == game and act in v and "continue" in v]
        y = np.array([c[1] for c in cells]) / sd
        hist = collections.defaultdict(list)
        for (k, e) in cells: hist[cps[k[0]]["trajectory_id"]].append(e / sd)
        hm = np.array([np.mean(v) for v in hist.values()])
        oracle = float(np.mean(np.maximum(y, 0.0)) - max(float(np.mean(y)), 0.0))
        print(f"  {act:26s}: ячеек {len(y)}, mean y/sd = {y.mean():+.3f}, sd(y/sd) = {y.std(ddof=1):.3f}, "
              f"выигрыш оракула-по-ячейке над лучшей константой = {oracle:.3f} sd; (по историям: {len(hm)} шт., sd {hm.std(ddof=1) if len(hm)>1 else float('nan'):.3f})")
print("  => идеальный оракул знака (знает шумные y) выигрывает сотые доли sd: δ=0.2 sd недостижим даже для оракула.")

# (2) clock collinearity
rows = []
for cid, c in cps.items():
    steps = c["environment_steps"]; eps = 1 - (1 - cfg["epsilon_final"]) * min(1.0, max(0, steps - cfg["warmup"]) / cfg["epsilon_decay"])
    rows.append((c["nominal_age"], steps, eps, diags[cid]["features"]["replay_size"]))
M = np.array(rows, float)
print("\nclock-признаки по 12 checkpoint: ages", sorted(set(M[:, 0])), " eps по возрасту:", {a: sorted(set(np.round(M[M[:, 0] == a, 2], 4))) for a in sorted(set(M[:, 0]))},
      " replay_size:", {a: sorted(set(M[M[:, 0] == a, 3])) for a in sorted(set(M[:, 0]))})
Z = (M[:, [0, 2, 3]] - M[:, [0, 2, 3]].mean(0)); sv = np.linalg.svd(Z, compute_uv=False)
print("  сингулярные числа центрированной матрицы [age, epsilon, replay_size]:", np.round(sv, 4), "-> ранг ≈", int((sv > 1e-3 * sv[0]).sum()))
print("  корреляции (age, eps)= %.4f, (age, replay)= %.4f" % (np.corrcoef(M[:, 0], M[:, 2])[0, 1], np.corrcoef(M[:, 0], M[:, 3])[0, 1]))

# (3) whitelist признаки и игра
def auc(pos, neg): return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
for feat in ("recent_return_mean", "pre_td_huber"):
    for age in (50000, 200000):
        a = [diags[c]["features"][feat] for c, v in cps.items() if v["game"] == "breakout" and v["nominal_age"] == age]
        b = [diags[c]["features"][feat] for c, v in cps.items() if v["game"] == "asterix" and v["nominal_age"] == age]
        print(f"  {feat:20s} age {age:6d}: breakout {np.round(sorted(a), 3)} asterix {np.round(sorted(b), 3)}  AUC(breakout>asterix) = {auc(a, b):.2f}")
