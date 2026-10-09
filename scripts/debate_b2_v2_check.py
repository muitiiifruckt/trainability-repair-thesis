"""[B2] Независимая векторизованная проверка процедуры v2 из scripts/debate_prereg_dryrun.py (по её docstring/описанию):
game one-hot + признаки z-score внутри игры по обучающим историям, y делится на sd обучающей части внутри игры, ridge alpha=1 только на признаках,
T = относительный выигрыш LOHO-SE кандидата над лучшим baseline in hindsight, нулевое распределение = Freedman-Lane (блок tested = fit(adj)+
остатки, переставленные между историями ОДНОЙ игры), bootstrap по историям внутри игры.
Сценарии DGP берутся у A (A.simulate), чтобы проверить T/best на тех же данных (--selfcheck), плюс собственные сценарии B2 (SCN_B2).
1 поток, numpy. Использование:
  python scripts/debate_b2_v2_check.py --selfcheck
  python scripts/debate_b2_v2_check.py <n_hist> <n_sims> <n_perm> <levels csv> <scenario csv|all> <tag> [seed] [dgp=A|B2]"""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
import os
ALPHA_RIDGE = float(os.environ.get("B2_RIDGE", "1.0"))                  # alpha ridge (в prereg = 1.0; B2_RIDGE для чувствительности)
PRESPEC = os.environ.get("B2_PRESPEC", "") == "1"                       # компаратор задан заранее (L2: base_all, L1: age), без best-of-k in hindsight
NULL_MODE = os.environ.get("B2_NULL", "perm")                           # perm = перестановки блоков остатков между историями игры (как в prereg v2); flip = случайные знаки блоков остатков
ALPHA_D = float(os.environ.get("B2_RIDGE_D", str(ALPHA_RIDGE)))        # отдельный (больший) штраф на тестируемый блок D в кандидате
ALPHA, MIN_HIST, MIN_PER_GAME, N_BOOT = 0.025, 12, 5, 1000
DELTAS = (0.05, 0.10, 0.20)
LEVELS = {  # колонки сырой X: 0 age, 1 pre_td, 2 recent_return, 3-4 механистический блок D
    "L2": dict(cand=[0, 1, 2, 3, 4], perm=[3, 4], adj=[0, 1, 2], base={"game": [], "age": [0], "age_return": [0, 2], "loss_return": [1, 2], "base_all": [0, 1, 2]}),
    "L1": dict(cand=[0, 1, 2], perm=[1, 2], adj=[0], base={"game": [], "age": [0]}),
}

def loho_batch(Xf, y, h, g, ng, pen_cols=None):
    """Xf (P,n,kf) сырые признаки. Возвращает pred (P,n) и ys (n,) — y, поделённый на sd обучающей части своей игры (по фолдам)."""
    P, n, kf = Xf.shape
    pred = np.empty((P, n)); ys = np.empty(n); onehot = np.eye(ng)[g]
    for hh in np.unique(h):
        te = h == hh; tr = ~te
        s = np.array([y[tr & (g == k)].std() + 1e-9 for k in range(ng)])
        ysc = y / s[g]; ys[te] = ysc[te]
        if kf:
            Z = np.empty_like(Xf)
            for k in range(ng):
                m = g == k; mt = m & tr
                mu = Xf[:, mt, :].mean(1, keepdims=True); sd = Xf[:, mt, :].std(1, keepdims=True) + 1e-9
                Z[:, m, :] = (Xf[:, m, :] - mu) / sd
            A = np.concatenate([np.broadcast_to(onehot, (P, n, ng)), Z], -1)
        else:
            A = np.broadcast_to(onehot, (P, n, ng))
        At = A[:, tr, :]
        pen = np.diag([0.0] * ng + (list(pen_cols) if pen_cols is not None else [ALPHA_RIDGE] * kf)) + 1e-9 * np.eye(ng + kf)
        G = np.einsum("pnk,pnl->pkl", At, At) + pen
        b = np.einsum("pnk,n->pk", At, ysc[tr])
        w = np.linalg.solve(G, b[..., None])[..., 0]
        pred[:, te] = np.einsum("pnk,pk->pn", A[:, te, :], w)
    return pred, ys

def hist_mean(vals, h, hs):
    return np.stack([vals[:, h == x].mean(1) for x in hs], 1)

def fl_blocks(X, spec, h, g, hs, n_perm, rng):
    """Freedman-Lane: (n_perm, n, len(perm)) переставленные блоки; строка k истории a получает fit(a,k)+остаток(донор b, k)."""
    perm, adj = spec["perm"], spec["adj"]; n = len(h)
    fit = np.empty((n, len(perm))); res = np.empty((n, len(perm)))
    for k in np.unique(g):
        rows = np.where(g == k)[0]; A = np.c_[np.ones(len(rows)), X[rows][:, adj]]
        beta = np.linalg.lstsq(A, X[rows][:, perm], rcond=None)[0]
        fit[rows] = A @ beta; res[rows] = X[rows][:, perm] - fit[rows]
    rows_of = np.stack([np.where(h == x)[0] for x in hs])               # (nh,2)
    g_hist = np.array([g[h == x][0] for x in hs])
    out = np.empty((n_perm, n, len(perm)))
    for i in range(n_perm):
        if NULL_MODE == "flip":
            sign = rng.choice([-1.0, 1.0], len(hs))
            for kk in range(rows_of.shape[1]):
                out[i, rows_of[:, kk], :] = fit[rows_of[:, kk]] + sign[:, None] * res[rows_of[:, kk]]
            continue
        donor = np.arange(len(hs))
        for k in np.unique(g_hist):
            idx = np.where(g_hist == k)[0]; donor[idx] = rng.permutation(idx)
        for kk in range(rows_of.shape[1]):
            out[i, rows_of[:, kk], :] = fit[rows_of[:, kk]] + res[rows_of[donor, kk]]
    return out

def compare(X, y, h, g, rng, level, n_perm):
    spec = LEVELS[level]; hs = np.unique(h); ng = int(g.max()) + 1; n = len(y)
    g_hist = np.array([g[h == x][0] for x in hs])
    ys_ref = None; SE = {}
    for m, cols in spec["base"].items():
        pred, ys = loho_batch(X[None][:, :, cols], y, h, g, ng); ys_ref = ys
        SE[m] = hist_mean((ys - pred) ** 2, h, hs)[0]
    best = ({"L2": "base_all", "L1": "age"}[level]) if PRESPEC else min(SE, key=lambda m: SE[m].mean())
    Xc = X[None][:, :, spec["cand"]]
    pen_c = [ALPHA_D if c in spec["perm"] else ALPHA_RIDGE for c in spec["cand"]]
    pred, ys = loho_batch(Xc, y, h, g, ng, pen_c); SEc = hist_mean((ys - pred) ** 2, h, hs)[0]
    T = (SE[best].mean() - SEc.mean()) / SE[best].mean()
    # перестановки: блок perm внутри cand заменяется на переставленный
    pos_in_cand = [spec["cand"].index(c) for c in spec["perm"]]
    blocks = fl_blocks(X, spec, h, g, hs, n_perm, rng)
    Xp = np.repeat(Xc, n_perm, 0); Xp[:, :, pos_in_cand] = blocks
    predp, ysp = loho_batch(Xp, y, h, g, ng, pen_c)
    SEp = hist_mean((ysp - predp) ** 2, h, hs)                           # (n_perm,nh)
    Tp = (SE[best].mean() - SEp.mean(1)) / SE[best].mean()
    p = (1 + (Tp >= T).sum()) / (n_perm + 1)
    idx_by_g = [np.where(g_hist == k)[0] for k in np.unique(g_hist)]
    ii = np.concatenate([rng.choice(ix, (N_BOOT, len(ix))) for ix in idx_by_g], 1)
    Tb = (SE[best][ii].mean(1) - SEc[ii].mean(1)) / SE[best][ii].mean(1)
    lb, ub = float(np.quantile(Tb, ALPHA)), float(np.quantile(Tb, 1 - ALPHA))
    enough = len(hs) >= MIN_HIST and min((g_hist == k).sum() for k in np.unique(g_hist)) >= MIN_PER_GAME
    se_means = {m: float(SE[m].mean()) for m in SE}; se_means["cand"] = float(SEc.mean())
    return dict(positive=bool(enough and T > 0 and p < ALPHA and lb > 0), T=float(T), p=float(p), lb=lb, ub=ub, best=best, enough=bool(enough), se=se_means)

def _design_insample(Xf, g, ng, y):
    """Все строки (без LOHO): game one-hot + признаки z-score внутри игры; y делится на sd внутри игры. Xf (P,n,k)."""
    P, n, k = Xf.shape; onehot = np.broadcast_to(np.eye(ng)[g], (P, n, ng))
    Z = np.empty_like(Xf)
    for c in range(ng):
        m = g == c
        Z[:, m, :] = (Xf[:, m, :] - Xf[:, m, :].mean(1, keepdims=True)) / (Xf[:, m, :].std(1, keepdims=True) + 1e-9)
    ys = y.copy()
    for c in range(ng):
        m = g == c; ys[m] = y[m] / (y[m].std() + 1e-9)
    return np.concatenate([onehot, Z], -1), ys

def _rss(A, ys):
    G = np.einsum("pnk,pnl->pkl", A, A) + 1e-9 * np.eye(A.shape[-1]); b = np.einsum("pnk,n->pk", A, ys)
    w = np.linalg.solve(G, b[..., None])[..., 0]
    r = ys[None, :] - np.einsum("pnk,pk->pn", A, w)
    return (r ** 2).sum(1)

def compare_F(X, y, h, g, rng, level, n_perm):
    """Частный F-тест (информация): добавляет ли блок D к [игра, age, ...] — in-sample OLS, нулевое распределение = Freedman-Lane перестановки D."""
    spec = LEVELS[level]; hs = np.unique(h); ng = int(g.max()) + 1; n = len(y); q = len(spec["perm"])
    g_hist = np.array([g[h == x][0] for x in hs])
    A0, ys = _design_insample(X[None][:, :, spec["adj"]], g, ng, y)
    rss0 = _rss(A0, ys)[0]
    blocks = fl_blocks(X, spec, h, g, hs, n_perm, rng)
    Dall = np.concatenate([X[None][:, :, spec["perm"]], blocks], 0)                   # (1+n_perm, n, q), 0 = наблюдаемый
    Zd, _ = _design_insample(Dall, g, ng, y)
    A1 = np.concatenate([np.broadcast_to(A0, (Dall.shape[0],) + A0.shape[1:]), Zd[:, :, ng:]], -1)
    rss1 = _rss(A1, ys); p1 = A1.shape[-1]
    F = ((rss0 - rss1) / q) / (rss1 / (n - p1))
    p = (1 + (F[1:] >= F[0]).sum()) / (n_perm + 1)
    enough = len(hs) >= MIN_HIST and min((g_hist == k).sum() for k in np.unique(g_hist)) >= MIN_PER_GAME
    return dict(positive=bool(enough and p < ALPHA), T=float((rss0 - rss1[0]) / rss0), p=float(p), lb=0.0, ub=0.0, best="F", enough=bool(enough), se={"cand": 1.0, "base": 1.0})

def compare_W(X, y, h, g, rng, level, n_perm):
    """Стьюдентизированный (cluster-robust по историям) Wald-тест блока D после FWL-очистки от [игра, adj]; нулевое распределение = Freedman-Lane.
    Нужен потому, что F-тест с перестановками не устойчив к гетероскедастичности, связанной между D и y (нестабильные истории)."""
    spec = LEVELS[level]; hs = np.unique(h); ng = int(g.max()) + 1; n = len(y); q = len(spec["perm"]); nh = len(hs)
    g_hist = np.array([g[h == x][0] for x in hs])
    A0, ys = _design_insample(X[None][:, :, spec["adj"]], g, ng, y)
    A0 = A0[0]; Pm = A0 @ np.linalg.pinv(A0.T @ A0 + 1e-9 * np.eye(A0.shape[1])) @ A0.T
    yr = ys - Pm @ ys
    blocks = fl_blocks(X, spec, h, g, hs, n_perm, rng)
    Dall = np.concatenate([X[None][:, :, spec["perm"]], blocks], 0)                    # (P,n,q)
    Zd, _ = _design_insample(Dall, g, ng, y); Zd = Zd[:, :, ng:]
    Dr = Zd - np.einsum("ij,pjk->pik", Pm, Zd)                                          # FWL
    G = np.einsum("pnk,pnl->pkl", Dr, Dr) + 1e-9 * np.eye(q)
    b = np.linalg.solve(G, np.einsum("pnk,n->pk", Dr, yr)[..., None])[..., 0]
    e = yr[None, :] - np.einsum("pnk,pk->pn", Dr, b)
    sc = (Dr * e[..., None]).reshape(Dr.shape[0], nh, n // nh, q).sum(2)               # (P,nh,q) вклад кластера-истории
    meat = np.einsum("phk,phl->pkl", sc, sc) * (nh / max(nh - 1, 1))
    Ginv = np.linalg.inv(G); V = Ginv @ meat @ Ginv + 1e-12 * np.eye(q)
    W = np.einsum("pk,pk->p", b, np.linalg.solve(V, b[..., None])[..., 0])
    p = (1 + (W[1:] >= W[0]).sum()) / (n_perm + 1)
    enough = nh >= MIN_HIST and min((g_hist == k).sum() for k in np.unique(g_hist)) >= MIN_PER_GAME
    return dict(positive=bool(enough and p < ALPHA), T=float(W[0]), p=float(p), lb=0.0, ub=0.0, best="W", enough=bool(enough), se={"cand": 1.0, "base": 1.0})

# --- сценарии B2 (реалистичнее): признаки кодируют игру и возраст, эффект y имеет масштаб/смещение по игре, возможно взаимодействие игра x возраст
def simulate_b2(s, n_hist, rng):
    h = np.repeat(np.arange(n_hist), 2); age = np.tile([0.0, 1.0], n_hist); g = (np.arange(n_hist) >= n_hist - n_hist // 2).astype(int)[h]
    a_z = (age - .5) / .5; n = len(h); N = lambda: rng.normal(size=n)
    hist = rng.normal(size=(n_hist, 3))[h]                               # историческая компонента признаков (оба возраста похожи)
    L = 0.5 * a_z + 0.6 * hist[:, 0] + 0.8 * N()
    R = 0.5 * a_z + 0.6 * hist[:, 1] + 0.8 * N()
    Dlat = 0.3 * L + 0.6 * hist[:, 2] + 0.8 * N()
    D1 = s.get("d_game_age", 0.0) * a_z * (2 * g - 1) + Dlat; D2 = N()   # D1 может иметь возрастной тренд разного знака по играм
    u = rng.normal(size=n_hist)[h] * s["sigma_u"]
    sh = np.exp(s.get("hetero", 0.0) * rng.normal(size=n_hist))[h]       # масштаб «нестабильной» истории: растёт и шум признаков D, и шум/случайный эффект y (связанная гетероскедастичность)
    if s.get("hetero", 0.0) > 0:
        D1 = D1 * (sh if s.get("hetero_coupled", True) else 1.0); D2 = D2 * (sh if s.get("hetero_coupled", True) else 1.0); u = u * sh
    y = (s["b_age"] * a_z + s["b_game_age"] * a_z * (2 * g - 1) + s["b_L"] * L + s["b_D"] * Dlat + u + s["game"] * (g - .5) + s.get("noise", 1.0) * N() * (sh if s.get("hetero", 0.0) > 0 else 1.0))
    y = y * np.array([1.0, s.get("scale1", 3.0)])[g]
    X = np.c_[age, np.array([1.0, 1.5])[g] * L, np.array([0, 1.])[g] + np.array([1, 3.])[g] * R,
              np.array([0, 3.])[g] + np.array([1, 2.])[g] * D1, np.array([0, 3.])[g] + np.array([1, 2.])[g] * D2]
    return X, y, h, g

SCN_B2 = {
    "null": dict(b_age=0, b_game_age=0, b_L=0, b_D=0, sigma_u=0.7, game=0),
    "null_game_offset": dict(b_age=0, b_game_age=0, b_L=0, b_D=0, sigma_u=0.7, game=1.5),
    "null_age_x_game": dict(b_age=0, b_game_age=0.8, b_L=0, b_D=0, sigma_u=0.7, game=0, d_game_age=0.8),   # y и D1 имеют возрастной тренд разного знака по играм
    "null_age_x_game_no_Dtrend": dict(b_age=0, b_game_age=0.8, b_L=0, b_D=0, sigma_u=0.7, game=0, d_game_age=0.0),
    "null_loss_signal": dict(b_age=0.3, b_game_age=0, b_L=0.8, b_D=0, sigma_u=0.7, game=0),
    "mech_b0.4": dict(b_age=0.3, b_game_age=0, b_L=0, b_D=0.4, sigma_u=0.7, game=0),
    "mech_b0.8": dict(b_age=0.3, b_game_age=0, b_L=0, b_D=0.8, sigma_u=0.7, game=0),
    "mech_b1.2": dict(b_age=0.3, b_game_age=0, b_L=0, b_D=1.2, sigma_u=0.7, game=0),
    "null_hetero_coupled": dict(b_age=0, b_game_age=0, b_L=0, b_D=0, sigma_u=0.7, game=0, hetero=0.8, hetero_coupled=True),
    "null_hetero_uncoupled": dict(b_age=0, b_game_age=0, b_L=0, b_D=0, sigma_u=0.7, game=0, hetero=0.8, hetero_coupled=False),
    # 4 repeats вместо 2: шум наблюдения y в sqrt(2) раз меньше (sigma_u сохраняется)
    "mech_b0.4_4rep": dict(b_age=0.3, b_game_age=0, b_L=0, b_D=0.4, sigma_u=0.7, game=0, noise=0.7071),
    "mech_b0.8_4rep": dict(b_age=0.3, b_game_age=0, b_L=0, b_D=0.8, sigma_u=0.7, game=0, noise=0.7071),
    "null_4rep": dict(b_age=0, b_game_age=0, b_L=0, b_D=0, sigma_u=0.7, game=0, noise=0.7071),
}

def rank_int_block(X, g, cols):
    """Замена столбцов блока D на нормальные scores рангов внутри игры (устойчивость к «нестабильным» историям с выбросами D)."""
    from scipy.stats import rankdata, norm
    X = X.copy()
    for k in np.unique(g):
        m = g == k
        for c in cols:
            r = rankdata(X[m, c]); X[m, c] = norm.ppf((r - 0.5) / m.sum())
    return X

def selfcheck():
    sys.path.insert(0, str(ROOT / "scripts"))
    import debate_prereg_dryrun as A
    rng = np.random.default_rng(11); worst_T = 0.0; same = 0; tot = 0
    for t in range(12):
        for sc in ("null", "age_only_signal", "mech_signal_b0.8", "game_offset_only"):
            X, y, h, g = A.simulate(A.SCN[sc], 12, rng)
            for lv in LEVELS:
                a = A.compare(X, y, h, g, np.random.default_rng(1), {"L2": "L2_primary", "L1": "L1_secondary"}[lv])
                b = compare(X, y, h, g, np.random.default_rng(1), lv, n_perm=9)
                worst_T = max(worst_T, abs(a["T"] - b["T"])); same += a["best"] == b["best"]; tot += 1
    print(f"selfcheck v2: max|T_A - T_B2| = {worst_T:.2e}; best baseline совпал {same}/{tot}")

if __name__ == "__main__":
    if sys.argv[1] == "--selfcheck":
        selfcheck(); sys.exit()
    n_hist, n_sims, n_perm = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    levels = sys.argv[4].split(","); dgp = sys.argv[8] if len(sys.argv) > 8 else "A"
    cmp_fn = {"F": compare_F, "W": compare_W}.get(os.environ.get("B2_STAT", ""), compare)
    if dgp == "A":
        sys.path.insert(0, str(ROOT / "scripts")); import debate_prereg_dryrun as A; SC, sim = A.SCN, A.simulate
    else:
        SC, sim = SCN_B2, simulate_b2
    names = list(SC) if sys.argv[5] == "all" else sys.argv[5].split(",")
    tag = sys.argv[6]; seed = int(sys.argv[7]) if len(sys.argv) > 7 else 20261009
    out = {}
    for name in names:
        t0 = time.time(); rng = np.random.default_rng(seed); recs = {lv: [] for lv in levels}
        for _ in range(n_sims):
            X, y, h, g = sim(SC[name], n_hist, rng)
            if os.environ.get("B2_RANKD", "") == "1": X = rank_int_block(X, g, [3, 4])      # блок D = столбцы 3,4
            if os.environ.get("B2_RANKY", "") == "1":                                       # y -> нормальные scores рангов внутри игры (только для теста информации)
                from scipy.stats import rankdata, norm
                y = y.copy()
                for k in np.unique(g):
                    m = g == k; y[m] = norm.ppf((rankdata(y[m]) - 0.5) / m.sum())
            for lv in levels: recs[lv].append(cmp_fn(X, y, h, g, rng, lv, n_perm))
        out[name] = {}
        for lv in levels:
            r = recs[lv]; pos = float(np.mean([x["positive"] for x in r])); se = 1.96 * np.sqrt(pos * (1 - pos) / n_sims)
            pop = {m: float(np.mean([x["se"][m] for x in r])) for m in r[0]["se"]}            # популяционная (по симуляциям) LOHO-ошибка каждой модели
            best_name = ({"L2": "base_all", "L1": "age"}[lv]) if PRESPEC else min((m for m in pop if m != "cand"), key=lambda m: pop[m])
            best_pop = pop[best_name]; T_true = 1 - pop["cand"] / best_pop   # истинный относительный выигрыш кандидата (n-1 историй обучения)
            out[name][lv] = dict(positive_rate=pos, mc95=[round(max(0, pos - se), 3), round(min(1, pos + se), 3)], mean_T=round(float(np.mean([x["T"] for x in r])), 3),
                                 T_true=round(float(T_true), 3), best_pop=best_name,
                                 mean_p=round(float(np.mean([x["p"] for x in r])), 3), frac_p_lt_alpha=round(float(np.mean([x["p"] < ALPHA for x in r])), 3),
                                 positive_noLB=round(float(np.mean([x["enough"] and x["T"] > 0 and x["p"] < ALPHA for x in r])), 3),
                                 ub_below_Ttrue=round(float(np.mean([x["ub"] < T_true for x in r])), 3), lb_above_Ttrue=round(float(np.mean([x["lb"] > T_true for x in r])), 3),
                                 **{f"negative_rate_delta{d}": round(float(np.mean([x["enough"] and x["ub"] < d for x in r])), 3) for d in DELTAS})
        out[name].update(n_hist=n_hist, n_sims=n_sims, n_perm=n_perm, dgp=dgp, sec=round(time.time() - t0, 1))
        print(name, json.dumps(out[name]), flush=True)
    Path(ROOT / f"research/debate/b2_v2_check_n{n_hist}_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
