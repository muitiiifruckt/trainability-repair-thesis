"""[B2] Reliability gate (P8): как split-half надёжность y между repeat связана с мощностью F-теста «информативности D» (B2-DGP, n=12/24 историй).
y = сигнал(B2, без шума) + среднее двух repeat с независимым шумом; gate-статистика = corr остатков repeat 0 и 1 после вычитания среднего по ячейке (игра x возраст).
Выводит по уровню шума: средняя gate-r, доля прохождения gate (r>=0.3), мощность F-теста безусловно и среди прошедших gate, ложноположительные при b_D=0.
1 поток, numpy. Использование: python scripts/debate_b2_gate.py [n_hist=12] [n_sims=500]"""
import sys, json, time, os
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import debate_b2_v2_check as V

def run(n_hist, n_sims, noise_mean, b_D, seed, thr=0.3):
    rng = np.random.default_rng(seed); rs, rej = [], []
    scn = dict(b_age=0.3, b_game_age=0, b_L=0, b_D=b_D, sigma_u=0.7, game=0, noise=0.0)
    for _ in range(n_sims):
        X, ysig, h, g = V.simulate_b2(scn, n_hist, rng)
        e = rng.normal(size=(2, len(ysig))) * np.sqrt(2.0) * noise_mean * np.array([1.0, 3.0])[g]     # шум одного repeat, масштаб y по играм как в simulate_b2
        y1, y2 = ysig + e[0], ysig + e[1]; y = 0.5 * (y1 + y2)
        age = X[:, 0]; cell = g * 2 + age.astype(int)
        r1, r2 = y1.copy(), y2.copy()
        for c in np.unique(cell):
            m = cell == c; r1[m] -= r1[m].mean(); r2[m] -= r2[m].mean()
        gate_r = np.corrcoef(r1, r2)[0, 1]
        res = V.compare_F(X, y, h, g, rng, "L2", 199)
        rs.append(gate_r); rej.append(res["positive"])
    rs, rej = np.array(rs), np.array(rej)
    ok = rs >= thr
    return dict(mean_gate_r=round(float(rs.mean()), 3), pass_rate=round(float(ok.mean()), 3), power_all=round(float(rej.mean()), 3),
                power_given_pass=round(float(rej[ok].mean()), 3) if ok.any() else None, power_given_fail=round(float(rej[~ok].mean()), 3) if (~ok).any() else None)

if __name__ == "__main__":
    n_hist = int(sys.argv[1]) if len(sys.argv) > 1 else 12; n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    out = {}
    for noise in (0.5, 0.7, 1.0, 1.4, 2.0):
        for b in (0.0, 0.8):
            t0 = time.time(); r = run(n_hist, n_sims, noise, b, 20261009 + int(noise * 10))
            out[f"noise_mean={noise}|b_D={b}|n={n_hist}"] = dict(r, sec=round(time.time() - t0, 1)); print(f"noise_mean={noise} b_D={b} n={n_hist}", r, flush=True)
    Path(ROOT / f"research/debate/b2_gate_n{n_hist}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
