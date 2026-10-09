"""Dry-run of the pre-registered diagnostic-value procedure (research/debate/prereg_diag_sign.md) on SYNTHETIC data.
Checks the procedure itself: false-positive rate under null/confounded scenarios and power under a known signal.
Usage: python scripts/debate_prereg_dryrun.py [n_histories=12] [n_sims=200]   (1 thread, numpy only)"""
import sys, json, time
sys.dont_write_bytecode = True
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ALPHA_RIDGE, N_PERM, N_BOOT, ALPHA, MIN_HIST = 1.0, 99, 1000, 0.025, 12

def ridge_fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]; P = np.eye(A.shape[1]) * ALPHA_RIDGE; P[0, 0] = 0
    w = np.linalg.solve(A.T @ A + P, A.T @ ytr)
    return np.c_[np.ones(len(Xte)), (Xte - mu) / sd] @ w

def loho_utility(X, y, h):
    """Per-history mean realised effect of the rule 'repair iff predicted effect > 0'. Leave-one-history-out."""
    U = {}
    for g in np.unique(h):
        te = h == g; tr = ~te
        pred = ridge_fit_predict(X[tr], y[tr], X[te]) if X.shape[1] else np.full(te.sum(), y[tr].mean())
        U[g] = np.mean(np.where(pred > 0, y[te], 0.0))
    return np.array([U[g] for g in np.unique(h)])

def procedure(clock, diag, y, h, rng, n_perm=N_PERM, n_boot=N_BOOT):
    """Returns dict(positive, delta, lb, p, best_baseline). clock: (n,k) age/eps/replay; diag: (n,2) pre_td, recent_return."""
    hs = np.unique(h)
    def utilities(diag_):
        return {"age_only": loho_utility(clock[:, :1], y, h), "diag_only": loho_utility(diag_, y, h),
                "all": loho_utility(np.c_[clock[:, :1], diag_], y, h),
                "const": loho_utility(np.zeros((len(y), 0)), y, h), "continue": np.zeros(len(hs))}
    def stat(U, best): return U["all"] - U[best]
    U = utilities(diag)
    best = max(("age_only", "diag_only", "const", "continue"), key=lambda k: U[k].mean())
    d = stat(U, best); delta = d.mean()
    # permutation of diagnostic columns across histories (rows of one history move together), best baseline re-picked
    hist_rows = {g: np.where(h == g)[0] for g in hs}; cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(hs)); dp = diag.copy()
        for g, g2 in zip(hs, hs[perm]):          # same-length blocks (2 ages per history)
            dp[hist_rows[g]] = diag[hist_rows[g2]]
        Up = utilities(dp); bp = max(("age_only", "diag_only", "const", "continue"), key=lambda k: Up[k].mean())
        cnt += stat(Up, bp).mean() >= delta
    p = (cnt + 1) / (n_perm + 1)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]) if p < ALPHA else None
    lb = np.quantile(boots, ALPHA) if boots is not None else None
    positive = (len(hs) >= MIN_HIST) and delta > 0 and p < ALPHA and lb is not None and lb > 0
    return dict(positive=bool(positive), delta=float(delta), p=float(p), lb=None if lb is None else float(lb), best=best)

def simulate(scn, n_hist, rng):
    h = np.repeat(np.arange(n_hist), 2); age = np.tile([0.0, 1.0], n_hist)          # 50k / 200k
    game = (h % 2)                                                                    # two games
    rho = 0.7 if scn["confound"] else 0.0
    diag1 = rho * (age - .5) / .5 + np.sqrt(1 - rho ** 2) * rng.normal(size=len(h))   # pre_td
    diag2 = 0.5 * (age - .5) / .5 + rng.normal(size=len(h))                           # recent return
    u = rng.normal(size=n_hist)[h] * scn["sigma_u"]                                   # history random effect (ICC)
    y = scn["b_age"] * (age - .5) / .5 + scn["b_diag"] * diag1 + u + scn["game"] * (game - .5) + rng.normal(size=len(h))
    return np.c_[age, age * 0, age * 0], np.c_[diag1, diag2], y, h

SCN = {
  "null": dict(b_age=0, b_diag=0, sigma_u=0.7, confound=False, game=0),
  "null_high_ICC": dict(b_age=0, b_diag=0, sigma_u=1.5, confound=False, game=0),
  "age_only_signal": dict(b_age=0.8, b_diag=0, sigma_u=0.7, confound=False, game=0),
  "age_signal_confounded_diag(rho=.7)": dict(b_age=0.8, b_diag=0, sigma_u=0.7, confound=True, game=0),
  "game_offset_only": dict(b_age=0, b_diag=0, sigma_u=0.7, confound=False, game=1.5),
  "diag_signal(b=0.8)": dict(b_age=0.3, b_diag=0.8, sigma_u=0.7, confound=False, game=0),
  "diag_signal_weak(b=0.4)": dict(b_age=0.3, b_diag=0.4, sigma_u=0.7, confound=False, game=0),
}
if __name__ == "__main__":
    n_hist = int(sys.argv[1]) if len(sys.argv) > 1 else 12; n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(SCN)
    out = {}
    for name in only:
        t0 = time.time(); rng = np.random.default_rng(12345); pos = 0; ps = []
        for _ in range(n_sims):
            c, d, y, h = simulate(SCN[name], n_hist, rng); r = procedure(c, d, y, h, rng); pos += r["positive"]; ps.append(r["p"])
        out[name] = dict(n_hist=n_hist, n_sims=n_sims, positive_rate=pos / n_sims, mean_p=float(np.mean(ps)), sec=round(time.time() - t0, 1))
        print(name, out[name], flush=True)
    Path(ROOT / f"research/debate/prereg_dryrun_v1_n{n_hist}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
