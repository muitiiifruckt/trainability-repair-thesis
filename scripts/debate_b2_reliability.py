"""[B2] Описательная надёжность эффектов repair по repeat (split-half r, ICC(1)) и граница «оракула» — только чтение runs/.../outcomes.jsonl. 1 поток, numpy.
Эффект ячейки (checkpoint, repeat) = (j_final(arm) - j_final(continue)) / sd, sd = sd возвратов эпизодов continue (финальные eval, все ветки этой игры).
Выводит: split-half r и ICC(1) между repeat 0 и 1 по checkpoint; то же после вычитания среднего по возрасту; выигрыш оракула знака над лучшей константой.
Использование: python scripts/debate_b2_reliability.py [game=breakout]   (PYTHONIOENCODING=utf-8 на Windows)"""
import sys, json, collections
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
game = sys.argv[1] if len(sys.argv) > 1 else "breakout"
rows = [json.loads(l) for l in open(ROOT / "runs/minatar-repair-20261007/outcomes.jsonl", encoding="utf-8") if l.strip() and json.loads(l)["game"] == game]
by = collections.defaultdict(dict)
for r in rows: by[(r["checkpoint_id"], r["repeat"])][r["repair_id"]] = r
sd = float(np.std([x for r in rows if r["repair_id"] == "continue" for x in r["evaluation_curve"][-1]["returns"]], ddof=1))
print(f"{game}: строк {len(rows)}, sd возвратов эпизода continue = {sd:.2f}")
for act in ("optimizer_reset", "head_reset", "head_and_optimizer_reset"):
    E = {c: [(by[(c, rep)][act]["j_final"] - by[(c, rep)]["continue"]["j_final"]) / sd for rep in (0, 1)]
         for c in sorted({k[0] for k in by}) if all((c, rep) in by and act in by[(c, rep)] and "continue" in by[(c, rep)] for rep in (0, 1))}
    M = np.array(list(E.values())); n = len(M)
    if n < 3: print(f"  {act}: n_checkpoint={n} (<3)"); continue
    cm = M.mean(1); k = 2
    msb = k * ((cm - M.mean()) ** 2).sum() / (n - 1); msw = ((M - cm[:, None]) ** 2).sum() / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw); r = np.corrcoef(M[:, 0], M[:, 1])[0, 1]
    age = {c: int(c.split("age")[1]) for c in E}; res = {}
    for a in set(age.values()):
        cs = [c for c in E if age[c] == a]; m = np.mean([E[c] for c in cs], axis=0)
        for c in cs: res[c] = np.array(E[c]) - m
    R = np.array(list(res.values())); r_age = np.corrcoef(R[:, 0], R[:, 1])[0, 1]
    y = M.reshape(-1); oracle = float(np.mean(np.maximum(y, 0)) - max(float(y.mean()), 0.0))
    print(f"  {act:26s} n_cp={n}: split-half r={r:+.2f}, ICC(1)={icc:+.2f}, после вычитания возраста r={r_age:+.2f}; "
          f"mean y={y.mean():+.3f} sd, sd(y)={y.std(ddof=1):.3f} sd, выигрыш оракула над константой={oracle:.3f} sd")
