"""Dry-run v3 плана research/debate/prereg_diag_sign.md (§12) на СИНТЕТИКЕ. Проверяет процедуру, не данные кампании.

Основа — scripts/debate_prereg_dryrun.py (v2: LOHO ridge alpha=1, intercept игры, z-score признаков внутри игры по обучающим историям,
y делится на sd обучающей части внутри игры, T = относительный выигрыш LOHO-SE кандидата над компаратором, bootstrap по историям, страты = игра).
Изменения v3 (все из §12; каждое проверяется здесь):
  (a) блок B = {age, pre_td_huber, recent_return_mean, j_pre}; в симуляции есть латентное качество политики Q: j_pre = Q + шум greedy-оценки,
      recent_return зависит от Q слабо, сценарии «сигнал по Q, D — прокси Q» (Q_signal_D_proxy*);
  (b) два уровня вердикта: «информативно» = enough & T>0 & p<0.025;  «полезно» = информативно & нижняя граница бутстрэпа T > 0;
  (c) нулевое распределение — настраиваемая схема (--scheme): d_perm (v2: Freedman-Lane по блоку D, перестановка остатков между историями игры),
      d_flip (знаки остатков D), y_perm (Freedman-Lane по остаткам y, перестановка между историями игры),
      y_flip (Freedman-Lane по остаткам y, СЛУЧАЙНЫЕ ЗНАКИ на историю), + ранги (--rankD/--rankY: нормальные scores внутри игры по обучающим историям fold'а);
      для y-схем LOHO по перестановкам — матричное умножение (гэт-матрица фолда фиксирована), поэтому 999-9999 перестановок дёшевы;
  (d) компаратор задан заранее (L2: base_all = B; L1: age); best-of-5 в v3 не вычисляется;
  (e) R повторов (--rep): шум среднего y имеет дисперсию 2/R (R=2 — как в v2);
  (f) reliability: ICC(1) эффекта по повторам после вычитания средних game x age (одно-измерительный ICC(1), df скорректированы), порог 0.3;
      режим `gate` — рабочие характеристики оценки ICC при заданном истинном ICC; в режиме `main` ICC считается в каждой симуляции;
  (g) ворота negative: negative = enough & не информативно & UB(T) < delta (delta 0.05/0.10/0.20), считаются доли при b=0.8;
  (h) --n_perm любой (в итоговом анализе >= 9999; для знаков при 2^n_hist <= n_perm — полное перебирание).
Использование:
  python scripts/debate_prereg_dryrun_v3.py selfcheck
  python scripts/debate_prereg_dryrun_v3.py main --n_hist 12 --n_sims 1000 --n_perm 999 --scenarios null,null_hetero_coupled --scheme y_flip --tag X
  python scripts/debate_prereg_dryrun_v3.py gate --n_sims 4000 --tag G
Окружение: 1 поток (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1), только numpy+scipy. Результат: research/debate/prereg_dryrun_v3_<tag>.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, argparse, itertools
from functools import lru_cache
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import norm
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ALPHA_RIDGE, ALPHA, N_BOOT, MIN_HIST, MIN_PER_GAME, ICC_THR = 1.0, 0.025, 1000, 12, 5, 0.3
DELTAS = (0.05, 0.10, 0.20)
# столбцы сырой X: 0 age, 1 pre_td_huber, 2 recent_return_mean, 3 j_pre, 4..5 механистический блок D
B_COLS, D_COLS = [0, 1, 2, 3], [4, 5]
LEVELS = {
    "L2": dict(cand=B_COLS + D_COLS, base=B_COLS, perm=D_COLS, nuis=B_COLS),   # первичный: B + D против base_all = B
    "L1": dict(cand=B_COLS, base=[0], perm=[1, 2, 3], nuis=[0]),               # вторичный: B против age
}


# ---------------------------------------------------------------- ранговые нормальные scores (по обучающим данным, тест — через обучающую ЭФР)
@lru_cache(None)
def ns_train(m):
    return norm.ppf((np.arange(m) + 0.5) / m)


@lru_cache(None)
def ns_test(m):
    return norm.ppf((np.arange(m + 1) + 0.5) / (m + 1))


def rank_apply(tr, te):
    """tr (m,q), te (t,q): scores обучающих значений внутри столбцов; тестовые значения отображаются через обучающую выборку."""
    m = tr.shape[0]
    rk = np.argsort(np.argsort(tr, axis=0), axis=0)
    c = (tr[None, :, :] < te[:, None, :]).sum(1)
    return ns_train(m)[rk], ns_test(m)[c]


# ---------------------------------------------------------------- LOHO как линейный сглаживатель
def make_folds(X, g, h, hs, ng, cols, rank_cols=(), alpha=ALPHA_RIDGE):
    """Для каждой истории (fold): (tr, te, H), H (n_te x n_tr) переводит масштабированный y обучающих строк в предсказания тестовых.
    Игра: one-hot без штрафа; признаки z-score внутри игры по ОБУЧАЮЩИМ строкам (как в v2)."""
    folds, k, oh = [], len(cols), np.eye(ng)
    rc = [i for i, c in enumerate(cols) if c in rank_cols]
    for hh in hs:
        te = np.where(h == hh)[0]; tr = np.where(h != hh)[0]
        Ztr = np.zeros((len(tr), k)); Zte = np.zeros((len(te), k))
        if k:
            gtr, gte = g[tr], g[te]
            for kk in range(ng):
                a, b = gtr == kk, gte == kk
                if not a.any():
                    continue
                xa = X[tr][a][:, cols].copy(); xb = X[te][b][:, cols].copy()
                if rc:
                    sa, sb = rank_apply(xa[:, rc], xb[:, rc]); xa[:, rc] = sa; xb[:, rc] = sb
                mu, sd = xa.mean(0), xa.std(0) + 1e-9
                Ztr[a] = (xa - mu) / sd
                if b.any():
                    Zte[b] = (xb - mu) / sd
        A = np.c_[oh[g[tr]], Ztr]; Bm = np.c_[oh[g[te]], Zte]
        G = A.T @ A + np.diag([0.0] * ng + [alpha] * k) + 1e-9 * np.eye(ng + k)
        folds.append((tr, te, Bm @ np.linalg.solve(G, A.T)))
    return folds


def loho_se_multi(models, Y, g, ng, yrank=False):
    """models: {имя: folds}. Y (n,P) — P вариантов y (наблюдаемый или переставленные). Возвращает {имя: SE (n_hist,P)} — средняя по строкам
    истории квадратичная ошибка в масштабированных единицах (y делится на sd обучающей части своей игры, без центрирования; v2) или в rank-scores."""
    names = list(models); nf = len(models[names[0]]); P = Y.shape[1]
    out = {m: np.empty((nf, P)) for m in names}
    for f in range(nf):
        tr, te, _ = models[names[0]][f]
        gtr, gte = g[tr], g[te]; Ytr, Yte = Y[tr], Y[te]
        if yrank:
            Str, Ste = np.empty_like(Ytr), np.empty_like(Yte)
            for kk in range(ng):
                a, b = gtr == kk, gte == kk
                Str[a], Ste[b] = rank_apply(Ytr[a], Yte[b])
        else:
            s = np.empty((ng, P))
            for kk in range(ng):
                a = gtr == kk
                s[kk] = Ytr[a].std(0) + 1e-9 if a.any() else 1.0
            Str, Ste = Ytr / s[gtr], Yte / s[gte]
        for m in names:
            out[m][f] = ((Ste - models[m][f][2] @ Str) ** 2).mean(0)
    return out


def loho_se_dbatch(Xp, y, h, hs, g, ng, alpha=ALPHA_RIDGE):
    """Для D-схем: Xp (P,n,k) — сырые столбцы кандидата с переставленным блоком D, y один. Возвращает SE (n_hist,P)."""
    P, n, k = Xp.shape; SE = np.empty((len(hs), P)); oh = np.broadcast_to(np.eye(ng)[g], (P, n, ng))
    pen = np.diag([0.0] * ng + [alpha] * k) + 1e-9 * np.eye(ng + k)
    for f, hh in enumerate(hs):
        te = h == hh; tr = ~te
        s = np.array([y[tr & (g == kk)].std() + 1e-9 for kk in range(ng)]); ys = y / s[g]
        Z = np.empty_like(Xp)
        for kk in range(ng):
            ma = g == kk; mt = ma & tr
            mu = Xp[:, mt, :].mean(1, keepdims=True); sd = Xp[:, mt, :].std(1, keepdims=True) + 1e-9
            Z[:, ma, :] = (Xp[:, ma, :] - mu) / sd
        A = np.concatenate([oh, Z], -1); At = A[:, tr, :]
        G = np.einsum("pnk,pnl->pkl", At, At) + pen
        b = np.einsum("pnk,n->pk", At, ys[tr])
        w = np.linalg.solve(G, b[..., None])[..., 0]
        SE[f] = ((ys[te][None, :] - np.einsum("pnk,pk->pn", A[:, te, :], w)) ** 2).mean(1)
    return SE


# ---------------------------------------------------------------- нулевые распределения
def nuisance(X, y, g, ng, cols, mode, lev):
    """Редуцированная модель y ~ игра + cols: fit и остатки. mode 'game' = OLS отдельно в каждой игре [1, cols] (свои наклоны);
    'common' = intercept игры + общие наклоны на z-score внутри игры (класс модели LOHO). lev: деление остатков на sqrt(1-h_ii)."""
    n = len(y); fit = np.empty(n); hat = np.zeros(n)
    if mode == "game":
        for kk in range(ng):
            m = np.where(g == kk)[0]; A = np.c_[np.ones(len(m)), X[m][:, cols]]
            fit[m] = A @ np.linalg.lstsq(A, y[m], rcond=None)[0]; hat[m] = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    else:
        Z = np.zeros((n, len(cols)))
        for kk in range(ng):
            m = g == kk; xm = X[m][:, cols]; Z[m] = (xm - xm.mean(0)) / (xm.std(0) + 1e-9)
        A = np.c_[np.eye(ng)[g], Z]
        fit = A @ np.linalg.lstsq(A, y, rcond=None)[0]; hat = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    res = y - fit
    if lev:
        res = res / np.sqrt(np.clip(1 - hat, 0.05, None))
    return fit, res


def donors_within_game(g_hist, n_perm, rng):
    nh = len(g_hist); don = np.empty((n_perm, nh), dtype=int)
    for kk in np.unique(g_hist):
        idx = np.where(g_hist == kk)[0]
        don[:, idx] = idx[np.argsort(rng.random((n_perm, len(idx))), axis=1)]
    return don


def sign_patterns(nh, n_perm, rng):
    """Все 2^nh знаковых векторов, если 2^nh <= n_perm (точный тест), иначе n_perm случайных."""
    if 2 ** nh <= n_perm:
        return np.array(list(itertools.product([-1.0, 1.0], repeat=nh))), True
    return rng.choice([-1.0, 1.0], size=(n_perm, nh)), False


def fl_blocks_D(X, perm, adj, g, rows_of, g_hist, n_perm, rng, flip):
    """Freedman-Lane по блоку D: fit(D | 1, adj; внутри игры) + остатки (переставленные между историями той же игры или со знаком на историю)."""
    n = X.shape[0]; fit = np.empty((n, len(perm))); res = np.empty((n, len(perm)))
    for kk in np.unique(g):
        rows = np.where(g == kk)[0]; A = np.c_[np.ones(len(rows)), X[rows][:, adj]]
        beta = np.linalg.lstsq(A, X[rows][:, perm], rcond=None)[0]; fit[rows] = A @ beta; res[rows] = X[rows][:, perm] - fit[rows]
    out = np.empty((n_perm, n, len(perm))); nh, na = rows_of.shape
    if flip:
        sg = rng.choice([-1.0, 1.0], size=(n_perm, nh))
        for a in range(na):
            out[:, rows_of[:, a], :] = fit[rows_of[:, a]][None] + sg[:, :, None] * res[rows_of[:, a]][None]
    else:
        don = donors_within_game(g_hist, n_perm, rng)
        for a in range(na):
            out[:, rows_of[:, a], :] = fit[rows_of[:, a]][None] + res[rows_of[don, a]]
    return out


# ---------------------------------------------------------------- один набор данных, один уровень
def compare(X, y, h, g, rng, level, cfg, icc=None):
    spec = LEVELS[level]; hs = np.unique(h); nh = len(hs); ng = int(g.max()) + 1; n = len(y)
    g_hist = np.array([g[h == x][0] for x in hs]); rows_of = np.stack([np.where(h == x)[0] for x in hs])
    rank_cols = tuple(spec["perm"]) if cfg["rankD"] else ()
    fb = make_folds(X, g, h, hs, ng, spec["base"], rank_cols); fc = make_folds(X, g, h, hs, ng, spec["cand"], rank_cols)
    se = loho_se_multi({"b": fb, "c": fc}, y[:, None], g, ng, cfg["rankY"]); SEb, SEc = se["b"][:, 0], se["c"][:, 0]
    T = 1.0 - SEc.mean() / SEb.mean()
    scheme, exact, tol = cfg["scheme"], False, 1e-12
    if scheme in ("y_flip", "y_perm"):
        fit, res = nuisance(X, y, g, ng, spec["nuis"], cfg["nuis"], cfg["lev"])
        if scheme == "y_flip":
            sg, exact = sign_patterns(nh, cfg["n_perm"], rng); exact = exact and not cfg["lev"]
            R = np.empty((n, len(sg)))
            for a in range(rows_of.shape[1]):
                R[rows_of[:, a], :] = res[rows_of[:, a]][:, None] * sg.T
            Ystar = fit[:, None] + R
        else:
            don = donors_within_game(g_hist, cfg["n_perm"], rng); Ystar = np.empty((n, len(don)))
            for a in range(rows_of.shape[1]):
                Ystar[rows_of[:, a], :] = fit[rows_of[:, a]][:, None] + res[rows_of[don, a]].T
        sp = loho_se_multi({"b": fb, "c": fc}, Ystar, g, ng, cfg["rankY"])
        Tp = 1.0 - sp["c"].mean(0) / sp["b"].mean(0)
    else:
        assert not cfg["rankY"] and not cfg["rankD"], "ранги в D-схемах не реализованы"
        blocks = fl_blocks_D(X, spec["perm"], spec["nuis"], g, rows_of, g_hist, cfg["n_perm"], rng, scheme == "d_flip")
        Xp = np.repeat(X[None][:, :, spec["cand"]], cfg["n_perm"], 0)
        Xp[:, :, [spec["cand"].index(c) for c in spec["perm"]]] = blocks
        Tp = 1.0 - loho_se_dbatch(Xp, y, h, hs, g, ng).mean(0) / SEb.mean()
    p = float((Tp >= T - tol).mean()) if exact else float((1 + (Tp >= T - tol).sum()) / (len(Tp) + 1))
    idx_by_g = [np.where(g_hist == kk)[0] for kk in np.unique(g_hist)]
    ii = np.concatenate([rng.choice(ix, (N_BOOT, len(ix))) for ix in idx_by_g], axis=1)
    Tb = 1.0 - SEc[ii].mean(1) / SEb[ii].mean(1)
    lb, ub = float(np.quantile(Tb, ALPHA)), float(np.quantile(Tb, 1 - ALPHA))
    enough = nh >= MIN_HIST and min((g_hist == kk).sum() for kk in np.unique(g_hist)) >= MIN_PER_GAME
    inf = bool(enough and T > 0 and p < ALPHA)
    return dict(inf=inf, useful=bool(inf and lb > 0), T=float(T), p=p, lb=lb, ub=ub, enough=bool(enough),
                seb=float(SEb.mean()), sec=float(SEc.mean()), n_perm_used=int(len(Tp)), exact=bool(exact))


# ---------------------------------------------------------------- генератор данных
def simulate(s, n_hist, rng, rep=2, n_age=2):
    """A-DGP v2 + латентное качество Q, j_pre, связанная гетероскедастичность, асимметрия, game x age. Расход rng не зависит от сценария
    (одни и те же данные для разных схем/сценариев при одном seed). Возвращает X, y (среднее по rep повторов), h, g, Yrep_sd (rep x n, в sd игры)."""
    h = np.repeat(np.arange(n_hist), n_age); ai = np.tile(np.arange(n_age), n_hist); age = ai / (n_age - 1.0)
    g = h % 2; a_z = (age - .5) / .5; n = len(h)
    z = rng.normal(size=(8, n)); eta = rng.normal(size=n_hist)[h]; u0 = rng.normal(size=n_hist)[h]; zh = rng.normal(size=n_hist)[h]
    ze = rng.normal(size=(rep, n)); zt = rng.standard_t(3, size=(rep, n))
    Q = 0.6 * a_z + 0.6 * eta + 0.8 * z[0]; Qs = Q / np.sqrt(1.36)
    L = 0.5 * a_z + np.sqrt(0.75) * z[1]
    Rl = 0.5 * a_z + 0.4 * Q + z[2]
    jpre = Q + s.get("j_noise", 0.5) * z[3]
    M = 0.3 * L + z[4]
    pr = s.get("proxy", 0.0)
    if pr > 0:
        base = Qs if s.get("proxy_of", "L") == "Q" else L
        D1 = pr * base + np.sqrt(1 - pr ** 2) * z[5]
    else:
        D1 = M + 0.5 * z[5]
    D1 = D1 + s.get("d_ga", 0.0) * a_z * (2 * g - 1); D2 = z[6]
    sh = np.exp(s["hetero"] * zh) if s.get("hetero", 0) > 0 else np.ones(n)
    if s.get("hetero", 0) > 0 and s.get("hetero_D", True):
        D1, D2 = D1 * sh, D2 * sh
    if s.get("heavy"):
        eps = zt / np.sqrt(3)
    elif s.get("skew"):
        sg_ = 0.6; eps = (np.exp(sg_ * ze) - np.exp(sg_ ** 2 / 2)) / np.sqrt((np.exp(sg_ ** 2) - 1) * np.exp(sg_ ** 2))
    else:
        eps = ze
    sig = (s["b_age"] * a_z + s.get("b_L", 0) * L + s.get("b_M", 0) * M + s.get("b_Q", 0) * Q + s.get("b_ga", 0) * a_z * (2 * g - 1)
           + s["game"] * (g - .5) + s["sigma_u"] * u0 * sh)
    Yrep = sig[None, :] + np.sqrt(2.0) * eps * sh[None, :]                    # шум ОДНОГО повтора: дисперсия 2; среднее по rep: 2/rep
    scale = np.array([1.0, 3.0])[g]
    y = Yrep.mean(0) * scale
    X = np.c_[age, np.array([1.0, 1.5])[g] * L, np.array([0, 1.])[g] + np.array([1, 3.])[g] * Rl,
              np.array([0, 2.])[g] + np.array([1, 2.])[g] * jpre,
              np.array([0, 3.])[g] + np.array([1, 2.])[g] * D1, np.array([0, 3.])[g] + np.array([1, 2.])[g] * D2]
    return X, y, h, g, Yrep


SCN = {
    "null": dict(b_age=0, sigma_u=0.7, game=0),
    "null_high_ICC": dict(b_age=0, sigma_u=1.5, game=0),
    "null_heavy_tail": dict(b_age=0, sigma_u=0.7, game=0, heavy=True),
    "null_hetero_coupled": dict(b_age=0, sigma_u=0.7, game=0, hetero=0.8, hetero_D=True),
    "null_hetero_uncoupled": dict(b_age=0, sigma_u=0.7, game=0, hetero=0.8, hetero_D=False),
    "null_skew": dict(b_age=0, sigma_u=0.7, game=0, skew=True),
    "null_skew_hetero": dict(b_age=0, sigma_u=0.7, game=0, skew=True, hetero=0.8, hetero_D=True),
    "null_age_x_game": dict(b_age=0, sigma_u=0.7, game=0, b_ga=0.8, d_ga=0.8),
    "age_only_signal": dict(b_age=0.8, sigma_u=0.7, game=0),
    "game_offset_only": dict(b_age=0, sigma_u=0.7, game=1.5),
    "loss_signal_D_is_proxy": dict(b_age=0.3, b_L=0.8, sigma_u=0.7, game=0, proxy=0.7, proxy_of="L"),
    "Q_signal_D_proxy": dict(b_age=0.3, b_Q=0.8, sigma_u=0.7, game=0, proxy=0.7, proxy_of="Q"),
    "Q_signal_D_proxy_noisyJ": dict(b_age=0.3, b_Q=0.8, sigma_u=0.7, game=0, proxy=0.7, proxy_of="Q", j_noise=1.0),
    "Q_signal_D_unrelated": dict(b_age=0.3, b_Q=0.8, sigma_u=0.7, game=0),
    "mech_signal_b0.4": dict(b_age=0.3, b_M=0.4, sigma_u=0.7, game=0),
    "mech_signal_b0.8": dict(b_age=0.3, b_M=0.8, sigma_u=0.7, game=0),
    "mech_signal_b1.2": dict(b_age=0.3, b_M=1.2, sigma_u=0.7, game=0),
    "mech_signal_b0.8_hetero": dict(b_age=0.3, b_M=0.8, sigma_u=0.7, game=0, hetero=0.8, hetero_D=True),
}


def signal_share(s, rep, seed=1):
    """Доля дисперсии y (игра 0, масштаб 1, строки) на компоненту, зависящую от D/Q/L-сигнала (b_M M, b_Q Q, b_L L); по большой выборке."""
    rng = np.random.default_rng(seed); n_hist = 4000
    X, y, h, g, Yrep = simulate(s, n_hist, rng, rep); m = g == 0
    # восстановить компоненты: пересчёт через те же формулы, отдельно (дисперсия сигнальной части)
    rng = np.random.default_rng(seed)
    a_z = (np.tile(np.arange(2), n_hist) - .5) / .5; n = len(a_z)
    z = rng.normal(size=(8, n)); eta = rng.normal(size=n_hist)[np.repeat(np.arange(n_hist), 2)]
    Q = 0.6 * a_z + 0.6 * eta + 0.8 * z[0]; L = 0.5 * a_z + np.sqrt(0.75) * z[1]; M = 0.3 * L + z[4]
    comp = s.get("b_M", 0) * M + s.get("b_Q", 0) * Q + s.get("b_L", 0) * L
    return float(comp[m].var() / y[m].var())


# ---------------------------------------------------------------- ICC(1) по повторам
def icc1(Yrep, cell):
    """Одно-измерительный ICC(1) после вычитания средних ячеек (game x age): Yrep (R,n_units). MS между единицами внутри ячеек / MS внутри единицы."""
    R, nu = Yrep.shape; m = Yrep.mean(0)
    msw = ((Yrep - m) ** 2).sum() / (nu * (R - 1)); ssb = 0.0; cells = np.unique(cell)
    for c in cells:
        mc = m[cell == c]; ssb += ((mc - mc.mean()) ** 2).sum()
    msb = R * ssb / (nu - len(cells))
    return float((msb - msw) / (msb + (R - 1) * msw))


def gate_oc(n_hist, rep, rho, n_sims, rng, w_hist=0.5, n_age=2):
    """Рабочие характеристики оценки ICC: единицы = (история, возраст); эффект единицы sqrt(rho)*(sqrt(w) eta_h + sqrt(1-w) zeta_u), шум повтора sqrt(1-rho);
    средние ячеек game x age произвольны (вычитаются). Возвращает среднее/sd оценки и долю >= ICC_THR."""
    h = np.repeat(np.arange(n_hist), n_age); ai = np.tile(np.arange(n_age), n_hist); g = h % 2; cell = g * n_age + ai; n = len(h)
    est = np.empty(n_sims)
    for i in range(n_sims):
        th = np.sqrt(rho) * (np.sqrt(w_hist) * rng.normal(size=n_hist)[h] + np.sqrt(1 - w_hist) * rng.normal(size=n))
        est[i] = icc1(th[None, :] + np.sqrt(1 - rho) * rng.normal(size=(rep, n)), cell)
    return dict(mean=round(float(est.mean()), 3), sd=round(float(est.std()), 3), pass_rate=round(float((est >= ICC_THR).mean()), 3))


# ---------------------------------------------------------------- сводка
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - r), 4), round(min(1.0, c + r), 4)]


def summarise(recs, iccs):
    n = len(recs); out = {}
    inf = np.array([r["inf"] for r in recs]); use = np.array([r["useful"] for r in recs]); en = np.array([r["enough"] for r in recs])
    pas = np.array(iccs) >= ICC_THR
    out["informative_rate"] = round(float(inf.mean()), 4); out["informative_wilson95"] = wilson(int(inf.sum()), n)
    out["useful_rate"] = round(float(use.mean()), 4); out["useful_wilson95"] = wilson(int(use.sum()), n)
    out["mean_T"] = round(float(np.mean([r["T"] for r in recs])), 3); out["mean_p"] = round(float(np.mean([r["p"] for r in recs])), 3)
    Tt = 1.0 - np.mean([r["sec"] for r in recs]) / np.mean([r["seb"] for r in recs]); out["T_true_pop"] = round(float(Tt), 3)
    out["frac_T_pos"] = round(float(np.mean([r["T"] > 0 for r in recs])), 3)
    out["ub_below_Ttrue"] = round(float(np.mean([r["ub"] < Tt for r in recs])), 3); out["lb_above_Ttrue"] = round(float(np.mean([r["lb"] > Tt for r in recs])), 3)
    out["ci_covers_Ttrue"] = round(float(np.mean([r["lb"] <= Tt <= r["ub"] for r in recs])), 3)
    for d in DELTAS:
        ng_ = np.array([r["enough"] and (not r["inf"]) and r["ub"] < d for r in recs])
        out[f"negative_rate_delta{d}"] = round(float(ng_.mean()), 4)
    out["gate_pass_rate"] = round(float(pas.mean()), 3)
    out["informative_given_pass"] = round(float(inf[pas].mean()), 4) if pas.any() else None
    out["informative_given_fail"] = round(float(inf[~pas].mean()), 4) if (~pas).any() else None
    out["useful_given_pass"] = round(float(use[pas].mean()), 4) if pas.any() else None
    out["n_pass"] = int(pas.sum()); out["mean_icc_hat"] = round(float(np.mean(iccs)), 3)
    out["enough_rate"] = round(float(en.mean()), 3); out["n_perm_used"] = recs[0]["n_perm_used"]; out["exact_enumeration"] = bool(recs[0]["exact"])
    return out


def selfcheck():
    sys.path.insert(0, str(ROOT / "scripts")); import debate_prereg_dryrun as A
    rng = np.random.default_rng(11); worst = 0.0
    for sc in ("null", "age_only_signal", "mech_signal_b0.8", "game_offset_only"):
        for _ in range(6):
            X, y, h, g = A.simulate(A.SCN[sc], 12, rng)                       # v2-данные: столбцы age, L, R, D1, D2 (j_pre нет)
            hs = np.unique(h); ng = 2
            for cand, base in (([0, 1, 2, 3, 4], [0, 1, 2]), ([0, 1, 2], [0])):
                P_b = A.loho(X, y, h, g, ng, base); P_c = A.loho(X, y, h, g, ng, cand)
                seb = A.per_hist((P_b[1] - P_b[0]) ** 2, h, hs); sec = A.per_hist((P_c[1] - P_c[0]) ** 2, h, hs)
                T_old = (seb.mean() - sec.mean()) / seb.mean()
                se = loho_se_multi({"b": make_folds(X, g, h, hs, ng, base), "c": make_folds(X, g, h, hs, ng, cand)}, y[:, None], g, ng)
                T_new = 1 - se["c"][:, 0].mean() / se["b"][:, 0].mean()
                worst = max(worst, abs(T_old - T_new), np.abs(seb - se["b"][:, 0]).max(), np.abs(sec - se["c"][:, 0]).max())
    print(f"selfcheck 1 (T и SE по историям, v2 против v3-движка, 48 сравнений): max abs diff = {worst:.2e}")
    # тождественность: fit + res == y, знак +1 воспроизводит наблюдаемую T
    rng = np.random.default_rng(5); X, y, h, g, _ = simulate(SCN["null"], 12, rng); hs = np.unique(h)
    fit, res = nuisance(X, y, g, 2, B_COLS, "game", False); d1 = np.abs(fit + res - y).max()
    fb = make_folds(X, g, h, hs, 2, B_COLS); fc = make_folds(X, g, h, hs, 2, B_COLS + D_COLS)
    s1 = loho_se_multi({"b": fb, "c": fc}, y[:, None], g, 2); s2 = loho_se_multi({"b": fb, "c": fc}, (fit + res)[:, None], g, 2)
    print(f"selfcheck 2: max|fit+res-y| = {d1:.2e}; |T(+1)-T_obs| = {abs((1 - s1['c'].mean() / s1['b'].mean()) - (1 - s2['c'].mean() / s2['b'].mean())):.2e}")
    # D-схема батч против одиночного пересчёта
    rows_of = np.stack([np.where(h == x)[0] for x in hs]); g_hist = np.array([g[h == x][0] for x in hs])
    blocks = fl_blocks_D(X, D_COLS, B_COLS, g, rows_of, g_hist, 5, np.random.default_rng(3), False)
    Xp = np.repeat(X[None][:, :, B_COLS + D_COLS], 5, 0); Xp[:, :, [4, 5]] = blocks
    b1 = loho_se_dbatch(Xp, y, h, hs, g, 2)
    b2 = np.stack([loho_se_multi({"c": make_folds(np.c_[Xp[i][:, :4], Xp[i][:, 4:]], g, h, hs, 2, list(range(6)))}, y[:, None], g, 2)["c"][:, 0] for i in range(5)], 1)
    print(f"selfcheck 3: dbatch против make_folds, max abs diff = {np.abs(b1 - b2).max():.2e}")
    sh = [round(signal_share(SCN[k], 2), 3) for k in ("mech_signal_b0.4", "mech_signal_b0.8", "mech_signal_b1.2")]
    sh4 = [round(signal_share(SCN[k], 4), 3) for k in ("mech_signal_b0.4", "mech_signal_b0.8", "mech_signal_b1.2")]
    print(f"доля дисперсии y на D-сигнал (b=0.4/0.8/1.2): R=2 {sh}, R=4 {sh4}")


def main(args):
    names = list(SCN) if args.scenarios == "all" else args.scenarios.split(",")
    cfg = dict(scheme=args.scheme, n_perm=args.n_perm, rankD=bool(args.rankD), rankY=bool(args.rankY), nuis=args.nuis, lev=bool(args.lev))
    levels = args.levels.split(","); out = {"_config": dict(vars(args), seed=args.seed)}
    for idx, name in enumerate(names):
        t0 = time.time(); s = SCN[name]
        rd = np.random.default_rng([args.seed, list(SCN).index(name)]); rp = np.random.default_rng([args.seed, list(SCN).index(name), 1])
        recs = {lv: [] for lv in levels}; iccs = []
        for _ in range(args.n_sims):
            X, y, h, g, Yrep = simulate(s, args.n_hist, rd, args.rep, args.n_age)
            cell = g * args.n_age + (X[:, 0] * (args.n_age - 1)).round().astype(int)
            iccs.append(icc1(Yrep, cell))                                   # Yrep уже в единицах sd игры (масштаб игры применяется только к y)
            for lv in levels:
                recs[lv].append(compare(X, y, h, g, rp, lv, cfg))
        out[name] = {lv: summarise(recs[lv], iccs) for lv in levels}
        out[name].update(n_hist=args.n_hist, n_sims=args.n_sims, rep=args.rep, scheme=args.scheme, signal_share_D=round(signal_share(s, args.rep), 3),
                         true_icc_single=None, sec=round(time.time() - t0, 1))
        print(name, json.dumps({lv: {k: v for k, v in out[name][lv].items() if k in ("informative_rate", "useful_rate", "mean_T", "negative_rate_delta0.05", "gate_pass_rate", "informative_given_pass", "n_pass")} for lv in levels}), f"{out[name]['sec']}s", flush=True)
    Path(ROOT / f"research/debate/prereg_dryrun_v3_{args.tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def gate_main(args):
    rng = np.random.default_rng([args.seed, 99]); out = {"_config": vars(args)}
    for n_hist in (6, 12, 24):
        for rep in (2, 4):
            for rho in (0.0, 0.15, 0.3, 0.5):
                out[f"n_hist={n_hist}|rep={rep}|icc_true={rho}"] = gate_oc(n_hist, rep, rho, args.n_sims, rng)
    for k, v in out.items():
        if k != "_config":
            print(k, v, flush=True)
    Path(ROOT / f"research/debate/prereg_dryrun_v3_{args.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["selfcheck", "main", "gate"])
    ap.add_argument("--n_hist", type=int, default=12); ap.add_argument("--n_sims", type=int, default=1000); ap.add_argument("--n_perm", type=int, default=999)
    ap.add_argument("--scenarios", default="null"); ap.add_argument("--scheme", default="y_flip", choices=["y_flip", "y_perm", "d_perm", "d_flip"])
    ap.add_argument("--nuis", default="game", choices=["game", "common"]); ap.add_argument("--lev", type=int, default=0)
    ap.add_argument("--rankD", type=int, default=0); ap.add_argument("--rankY", type=int, default=0)
    ap.add_argument("--rep", type=int, default=2); ap.add_argument("--n_age", type=int, default=2); ap.add_argument("--levels", default="L2,L1")
    ap.add_argument("--seed", type=int, default=20261010); ap.add_argument("--tag", default="tmp")
    a = ap.parse_args()
    {"selfcheck": lambda: selfcheck(), "main": lambda: main(a), "gate": lambda: gate_main(a)}[a.mode]()
