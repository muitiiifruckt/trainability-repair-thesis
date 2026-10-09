"""[B2] Реальное покрытие "95% interval" из experiments/rl_analysis.py::_bootstrap_paired при малом числе историй.
Только чтение: функция импортируется из живого кода (байт-код не пишется), данных кампании скрипт не трогает.
Модель: эффект ячейки (история h, checkpoint c, repeat r) = mu + u_h + v_hc + e_hcr, 2 checkpoint x 2 repeat на историю (как в screen);
истинное mu = 0. Доля симуляций, в которых интервал накрывает 0.
1 поток, numpy. Использование: python scripts/debate_b2_ci_coverage.py [n_sims=400] [boot=300] [setting,setting] [tag]"""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.rl_analysis import _bootstrap_paired  # noqa: E402  (read-only import of the live analysis code)

def panel(n_hist, tau_h, tau_c, sig_r, rng, n_ckpt=2, n_rep=2):
    recs = []
    for h in range(n_hist):
        uh = rng.normal() * tau_h
        for c in range(n_ckpt):
            vc = rng.normal() * tau_c
            for r in range(n_rep):
                recs.append({"game": "g", "trajectory_id": f"h{h}", "checkpoint_id": f"h{h}c{c}",
                             "effect": uh + vc + rng.normal() * sig_r})
    return recs

def coverage(n_hist, comps, n_sims, boot, seed):
    rng = np.random.default_rng(seed); hit = 0; widths = []
    for _ in range(n_sims):
        recs = panel(n_hist, *comps, rng)
        est = _bootstrap_paired(recs, "effect", boot, seed=int(rng.integers(1, 2**31 - 1)))["g"]
        lo, hi = est["ci95"]; hit += (lo <= 0.0 <= hi); widths.append(hi - lo)
    return hit / n_sims, float(np.mean(widths))

if __name__ == "__main__":
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    boot = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    # (tau_history, tau_checkpoint, sigma_repeat): оценка по 6 ячейкам breakout head_reset: repeat ~0.9, checkpoint ~0.9; история неизвестна
    settings = {"base": (0.5, 0.7, 0.9), "repeat-only": (0.0, 0.0, 1.0), "hist-heavy": (1.0, 0.7, 0.9)}   # (tau_h, tau_c, sigma_r)
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(settings)
    tag = sys.argv[4] if len(sys.argv) > 4 else "all"
    out = {}
    for name, comps in ((k, v) for k, v in settings.items() if k in only):
        for n_hist in (1, 2, 3, 5, 6, 12):
            t0 = time.time()
            cov, width = coverage(n_hist, comps, n_sims, boot, seed=20261009 + n_hist)
            out[f"{name}|n_hist={n_hist}"] = {"coverage_of_true_0": cov, "mean_width": round(width, 3), "n_sims": n_sims, "boot": boot, "sec": round(time.time() - t0, 1)}
            print(name, n_hist, out[f"{name}|n_hist={n_hist}"], flush=True)
    Path(ROOT / f"research/debate/b2_ci_coverage_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
