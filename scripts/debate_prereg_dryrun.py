"""Dry-run (v2) of the analysis plan research/debate/prereg_diag_sign.md on SYNTHETIC data. Checks the procedure itself.

History: v1 (scripts/debate_prereg_dryrun_v1.py) had a design error: the candidate feature set equalled the union of the baselines,
decision-utility statistic, global z-score, global permutation -> ~zero power at n=12 (research/debate/prereg_dryrun_v1_n12.json).
v2 (this file), designed on synthetic data only (no real outcome data used):
  * two comparisons with the same machinery
      Level 2 (primary):   baseline features [age, pre_td, recent_return] + mechanistic block D  vs  best of 5 baselines
      Level 1 (secondary): [age, pre_td, recent_return]                                         vs  best of {game, age}
  * statistic = relative LOHO squared-error gain  T = (SE_best_baseline - SE_candidate) / SE_best_baseline  (per-history mean SE)
  * game intercepts in every model; features z-scored WITHIN GAME on training histories; y scaled by the training-fold sd per game
    (no heldout normalisation, no centering)
  * null by Freedman-Lane residual block permutation WITHIN GAME (keeps correlation of the tested block with the adjustment features)
  * stratified (by game) bootstrap over histories
  * decision utility is secondary / descriptive
Usage: python scripts/debate_prereg_dryrun.py <n_histories> <n_sims> <comma-separated scenarios|all> <tag> [legacy]
1 thread, numpy only. Output research/debate/prereg_dryrun_v2_n<N>_<tag>[_legacy].json
"""
import sys, json, time
sys.dont_write_bytecode = True
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ALPHA_RIDGE, N_PERM, N_BOOT, ALPHA, MIN_HIST, MIN_PER_GAME = 1.0, 99, 1000, 0.025, 12, 5
DELTAS = (0.05, 0.10, 0.20)                     # relative-gain margins for the negative verdict
# raw X columns: 0 age, 1 pre_td_huber, 2 recent_return_mean, 3.. mechanistic block D (2 columns)
LEVELS = {
    "L2_primary": dict(cand=[0, 1, 2, 3, 4], perm=[3, 4], adj=[0, 1, 2],
                       base={"game": [], "age": [0], "age_return": [0, 2], "loss_return": [1, 2], "base_all": [0, 1, 2]}),
    "L1_secondary": dict(cand=[0, 1, 2], perm=[1, 2], adj=[0], base={"game": [], "age": [0]}),
}


def design(Xtr, gtr, Xte, gte, ng, cols):
    """Game one-hot (unpenalised) + features z-scored within game using TRAINING rows of that game only."""
    Ztr = np.zeros((len(Xtr), len(cols))); Zte = np.zeros((len(Xte), len(cols)))
    if cols:
        for k in range(ng):
            a, b = gtr == k, gte == k
            if a.any():
                mu, sd = Xtr[a][:, cols].mean(0), Xtr[a][:, cols].std(0) + 1e-9
                Ztr[a] = (Xtr[a][:, cols] - mu) / sd; Zte[b] = (Xte[b][:, cols] - mu) / sd
    return np.c_[np.eye(ng)[gtr], Ztr], np.c_[np.eye(ng)[gte], Zte]


def loho(X, y, h, g, ng, cols):
    """Leave-one-history-out. y is scaled by the training-fold sd of y within each game (no centering). Returns (pred, y_scaled)."""
    pred = np.zeros(len(y)); ys = np.zeros(len(y))
    for hh in np.unique(h):
        te = h == hh; tr = ~te
        s = np.array([y[tr & (g == k)].std() + 1e-9 if (tr & (g == k)).any() else 1.0 for k in range(ng)])
        A, B = design(X[tr], g[tr], X[te], g[te], ng, cols)
        pen = np.diag([0.0] * ng + [ALPHA_RIDGE] * len(cols))
        w = np.linalg.solve(A.T @ A + pen + 1e-9 * np.eye(A.shape[1]), A.T @ (y[tr] / s[g[tr]]))
        pred[te] = B @ w; ys[te] = y[te] / s[g[te]]
    return pred, ys


def per_hist(v, h, hs):
    return np.array([v[h == x].mean() for x in hs])


def freedman_lane(X, perm, adj, h, g, rng):
    """Replace columns `perm` by fitted(adj, within game) + residuals block-permuted among histories of the same game."""
    Xp = X.copy()
    for k in np.unique(g):
        rows = np.where(g == k)[0]
        A = np.c_[np.ones(len(rows)), X[rows][:, adj]]
        beta = np.linalg.lstsq(A, X[rows][:, perm], rcond=None)[0]
        fit = A @ beta; res = X[rows][:, perm] - fit
        hs_g = np.unique(h[rows]); moved = hs_g[rng.permutation(len(hs_g))]
        for a, b in zip(hs_g, moved):
            ra, rb = np.where(h[rows] == a)[0], np.where(h[rows] == b)[0]
            Xp[np.ix_(rows[ra], perm)] = fit[ra] + res[rb]
    return Xp


def compare(X, y, h, g_true, rng, level, legacy=False):
    spec = LEVELS[level]
    hs = np.unique(h); g_hist = np.array([g_true[h == x][0] for x in hs])
    g = np.zeros_like(g_true) if legacy else g_true          # legacy = v1-style pooled handling of games
    ng = int(g.max()) + 1
    P = {m: loho(X, y, h, g, ng, c) for m, c in spec["base"].items()}
    ys = P["game"][1]
    SE = {m: per_hist((ys - p[0]) ** 2, h, hs) for m, p in P.items()}
    best = min(SE, key=lambda m: SE[m].mean())
    def cand_SE(Xc):
        p, y2 = loho(Xc, y, h, g, ng, spec["cand"]); return per_hist((y2 - p) ** 2, h, hs), p
    SEc, pc = cand_SE(X)
    T = (SE[best].mean() - SEc.mean()) / SE[best].mean()
    cnt = 0
    for _ in range(N_PERM):
        Xp = freedman_lane(X, spec["perm"], spec["adj"], h, g, rng)
        SEp, _ = cand_SE(Xp)
        bp = min(SE, key=lambda m: SE[m].mean())            # baselines do not depend on the permuted block
        cnt += (SE[bp].mean() - SEp.mean()) / SE[bp].mean() >= T
    p = (cnt + 1) / (N_PERM + 1)
    idx_by_g = [np.where(g_hist == k)[0] for k in np.unique(g_hist)]
    Tb = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by_g])
        Tb[i] = (SE[best][ii].mean() - SEc[ii].mean()) / SE[best][ii].mean()
    lb, ub = float(np.quantile(Tb, ALPHA)), float(np.quantile(Tb, 1 - ALPHA))
    enough = len(hs) >= MIN_HIST and min((g_hist == k).sum() for k in np.unique(g_hist)) >= MIN_PER_GAME
    # secondary descriptive decision utility: chosen = repair iff predicted (scaled) effect > 0
    U = {m: per_hist(np.where(P[m][0] > 0, ys, 0.0), h, hs) for m in P}; U["continue"] = np.zeros(len(hs))
    ub_name = max(U, key=lambda m: U[m].mean()); d = per_hist(np.where(pc > 0, ys, 0.0), h, hs) - U[ub_name]
    u_lb = float(np.quantile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)], ALPHA))
    return dict(positive=bool(enough and T > 0 and p < ALPHA and lb > 0), T=float(T), p=float(p), lb=lb, ub=ub, best=best,
                enough=bool(enough), util_lb_pos=bool(u_lb > 0), ponly=bool(enough and T > 0 and p < ALPHA))


def simulate(s, n_hist, rng):
    h = np.repeat(np.arange(n_hist), 2); age = np.tile([0.0, 1.0], n_hist); g = h % 2; a_z = (age - .5) / .5; n = len(h)
    N = lambda: rng.normal(size=n)
    L = 0.5 * a_z + np.sqrt(0.75) * N()                       # pre-repair loss latent (correlated with age)
    R = 0.5 * a_z + N()                                       # recent return latent
    M = 0.3 * L + N()                                         # mechanistic latent
    pr = s.get("proxy", 0.0)
    D1 = (pr * L + np.sqrt(1 - pr ** 2) * N()) if pr > 0 else (M + 0.5 * N())
    D2 = N()
    u = rng.normal(size=n_hist)[h] * s["sigma_u"]
    noise = rng.standard_t(3, size=n) / np.sqrt(3) if s.get("heavy") else N()
    y = s["b_age"] * a_z + s["b_L"] * L + s["b_M"] * M + u + s["game"] * (g - .5) + noise
    y = y * np.array([1.0, 3.0])[g]                           # games differ in return scale
    sc = lambda off, scale: (np.array(off)[g], np.array(scale)[g])
    oR, sR = sc([0, 1], [1, 3]); oD, sD = sc([0, 3], [1, 2]); sL = np.array([1.0, 1.5])[g]
    X = np.c_[age, sL * L, oR + sR * R, oD + sD * D1, oD + sD * D2]
    return X, y, h, g


SCN = {
    "null": dict(b_age=0, b_L=0, b_M=0, sigma_u=0.7, game=0),
    "null_high_ICC": dict(b_age=0, b_L=0, b_M=0, sigma_u=1.5, game=0),
    "null_heavy_tail": dict(b_age=0, b_L=0, b_M=0, sigma_u=0.7, game=0, heavy=True),
    "age_only_signal": dict(b_age=0.8, b_L=0, b_M=0, sigma_u=0.7, game=0),
    "loss_signal_D_is_proxy": dict(b_age=0.3, b_L=0.8, b_M=0, sigma_u=0.7, game=0, proxy=0.7),
    "game_offset_only": dict(b_age=0, b_L=0, b_M=0, sigma_u=0.7, game=1.5),
    "mech_signal_b0.4": dict(b_age=0.3, b_L=0, b_M=0.4, sigma_u=0.7, game=0),
    "mech_signal_b0.8": dict(b_age=0.3, b_L=0, b_M=0.8, sigma_u=0.7, game=0),
    "mech_signal_b1.2": dict(b_age=0.3, b_L=0, b_M=1.2, sigma_u=0.7, game=0),
}


def summarise(recs, n_sims):
    pos = float(np.mean([r["positive"] for r in recs])); out = dict(positive_rate=pos)
    se = 1.96 * np.sqrt(pos * (1 - pos) / n_sims); out["positive_mc95"] = [round(max(0, pos - se), 3), round(min(1, pos + se), 3)]
    out["mean_T"] = round(float(np.mean([r["T"] for r in recs])), 3); out["mean_p"] = round(float(np.mean([r["p"] for r in recs])), 3)
    out["best_baseline_freq"] = {b: round(float(np.mean([r["best"] == b for r in recs])), 2) for b in sorted({r["best"] for r in recs})}
    out["util_lb_positive_rate"] = float(np.mean([r["util_lb_pos"] for r in recs]))
    out["p_only_rate"] = float(np.mean([r["ponly"] for r in recs]))   # T>0 & permutation p<0.025, without the bootstrap-LB condition
    for d in DELTAS:
        neg = float(np.mean([(not r["positive"]) and r["enough"] and r["ub"] < d for r in recs]))
        out[f"negative_rate_delta{d}"] = neg; out[f"uncertain_rate_delta{d}"] = round(1 - pos - neg, 3)
    return out


if __name__ == "__main__":
    n_hist = int(sys.argv[1]); n_sims = int(sys.argv[2])
    names = list(SCN) if sys.argv[3] == "all" else sys.argv[3].split(",")
    tag = sys.argv[4]; legacy = len(sys.argv) > 5 and sys.argv[5] == "legacy"
    out = {}
    for name in names:
        t0 = time.time(); rng = np.random.default_rng(12345); recs = {lv: [] for lv in LEVELS}
        for _ in range(n_sims):
            X, y, h, g = simulate(SCN[name], n_hist, rng)
            for lv in LEVELS: recs[lv].append(compare(X, y, h, g, rng, lv, legacy))
        out[name] = {lv: summarise(recs[lv], n_sims) for lv in LEVELS}
        out[name].update(n_hist=n_hist, n_sims=n_sims, legacy=legacy, sec=round(time.time() - t0, 1))
        print(name, json.dumps(out[name]), flush=True)
    Path(ROOT / f"research/debate/prereg_dryrun_v2_n{n_hist}_{tag}{'_legacy' if legacy else ''}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
