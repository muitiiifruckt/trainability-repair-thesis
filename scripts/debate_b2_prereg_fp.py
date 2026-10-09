"""[B2] Независимая (векторизованная) реализация процедуры research/debate/prereg_diag_sign.md и проверка её ложноположительной доли.
Не копирует код A: LOHO-ridge считается батчем по перестановкам. Самопроверка: на одних и тех же данных delta/best совпадают с
scripts/debate_prereg_dryrun.py::procedure (--selfcheck). 1 поток, только numpy, RAM << 1 GB.

Варианты процедуры
  V0  как написано в prereg: z-score по всем строкам обучающей части, перестановка diag по всем историям, baseline = лучший из
      {age_only, diag_only, const, continue} по тем же данным, статистика = mean_h(U_all - U_best), U = реализованный эффект по знаку.
  V1  V0, но z-score признаков внутри игры (по обучающим историям игры) и перестановки внутри игры (страты).
  V2  V1 + индикатор игры во всех моделях (возраст+игра, diag+игра, game_only) — игра известна при развёртывании и не должна
      приписываться диагностике.
  V3  V2 + заранее заданный компаратор (age+game) вместо best-of-k in hindsight и статистика = снижение LOHO-MSE (вложенные модели).
Сценарии: признаки кодируют игру и возраст (как в реальных 12 checkpoint: recent_return разделяет игры полностью), y может
иметь смещение по игре / возрасту / истинный сигнал внутри (игра, возраст).
Использование: python scripts/debate_b2_prereg_fp.py <n_hist> <n_sims> <n_perm> <variants csv> <scenarios csv|all> <tag> [seed]"""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
ALPHA_RIDGE, ALPHA, DELTA, N_BOOT = 1.0, 0.025, 0.2, 1000


# ---------------------------------------------------------------- векторизованный LOHO-ridge
def _standardize(Xf, Xd, tr, game, within):
    P, n, kf = Xf.shape
    parts = []
    if kf:
        if within:
            Zf = np.empty_like(Xf)
            for c in np.unique(game):
                m = game == c; mt = m & tr
                mu = Xf[:, mt, :].mean(1, keepdims=True); sd = Xf[:, mt, :].std(1, keepdims=True) + 1e-9
                Zf[:, m, :] = (Xf[:, m, :] - mu) / sd
        else:
            mu = Xf[:, tr, :].mean(1, keepdims=True); sd = Xf[:, tr, :].std(1, keepdims=True) + 1e-9
            Zf = (Xf - mu) / sd
        parts.append(Zf)
    if Xd.shape[1]:
        mu = Xd[tr].mean(0); sd = Xd[tr].std(0) + 1e-9
        parts.append(np.broadcast_to((Xd - mu) / sd, (P, n, Xd.shape[1])))
    return np.concatenate(parts, -1) if parts else np.zeros((P, n, 0))


def loho_pred(Xf, Xd, y, h, game, within):
    """Xf (P,n,kf) признаки (z-score внутри игры или по всем), Xd (n,kd) индикаторы (возраст, игра; z-score по всем обучающим).
    Возвращает LOHO-предсказания (P,n). intercept не штрафуется, alpha=1 — как в prereg."""
    P, n, _ = Xf.shape
    pred = np.empty((P, n))
    for g in np.unique(h):
        te = h == g; tr = ~te
        Z = _standardize(Xf, Xd, tr, game, within); k = Z.shape[-1]
        if k == 0:
            pred[:, te] = y[tr].mean(); continue
        A = np.concatenate([np.ones((P, n, 1)), Z], -1); At = A[:, tr, :]
        pen = np.eye(k + 1) * ALPHA_RIDGE; pen[0, 0] = 0
        G = np.einsum("pnk,pnl->pkl", At, At) + pen
        b = np.einsum("pnk,n->pk", At, y[tr])
        w = np.linalg.solve(G, b[..., None])[..., 0]
        pred[:, te] = np.einsum("pnk,pk->pn", A[:, te, :], w)
    return pred


def hist_mean(vals, h, hs):
    """(P,n) -> (P,n_hist) среднее по строкам истории."""
    out = np.empty((vals.shape[0], len(hs)))
    for i, g in enumerate(hs):
        out[:, i] = vals[:, h == g].mean(1)
    return out


# ---------------------------------------------------------------- перестановки
def make_perms(hs, game_h, n_perm, rng, stratified):
    """Возвращает (n_perm+1, n_hist) индексов доноров; строка 0 — тождество (наблюдаемые данные)."""
    perms = np.tile(np.arange(len(hs)), (n_perm + 1, 1))
    for i in range(1, n_perm + 1):
        if stratified:
            for c in np.unique(game_h):
                idx = np.where(game_h == c)[0]; perms[i, idx] = rng.permutation(idx)
        else:
            perms[i] = rng.permutation(len(hs))
    return perms


# ---------------------------------------------------------------- сама процедура
def run_procedure(variant, y, h, game, age, diag, rng, n_perm=199, n_boot=N_BOOT):
    """diag: (n,2) [pre_td, recent_return]; строки упорядочены (история, возраст). Возвращает dict с вердиктами."""
    hs = np.unique(h); nh = len(hs); n = len(y)
    game_h = np.array([game[h == g][0] for g in hs])
    rows_of = np.stack([np.where(h == g)[0] for g in hs])                  # (nh, 2) строки истории
    within = variant in ("V1", "V2", "V3"); gd = variant in ("V2", "V3"); stratified = within
    perms = make_perms(hs, game_h, n_perm, rng, stratified)                  # (P,nh)
    P = perms.shape[0]
    # permuted diag: строка k истории g получает diag строки k истории-донора
    Dp = np.empty((P, n, 2))
    for k in range(rows_of.shape[1]):
        Dp[:, rows_of[:, k], :] = diag[rows_of[perms, k]]
    none_f = np.zeros((1, n, 0))
    age_c = age[:, None].astype(float); game_c = game[:, None].astype(float)
    Xd_age = np.concatenate([age_c, game_c], 1) if gd else age_c
    Xd_game = game_c if gd else np.zeros((n, 0))
    Xd_all = np.concatenate([age_c, game_c], 1) if gd else age_c
    def util(pred): return hist_mean(np.where(pred > 0, y, 0.0), h, hs)
    def mse(pred): return hist_mean((y - pred) ** 2, h, hs)
    if variant == "V3":
        score = mse
        base_only = score(loho_pred(none_f, Xd_age, y, h, game, within))               # (1,nh) компаратор age+game
        m_all = score(loho_pred(Dp, Xd_all, y, h, game, within))                       # (P,nh)
        d_all = base_only - m_all                                                     # >0: all лучше (ниже MSE)
        stat = d_all.mean(1); d_obs = d_all[0]
        best = "age+game"
    else:
        score = util
        base = {"age_only": score(loho_pred(none_f, Xd_age, y, h, game, within)),
                "const": score(loho_pred(none_f, np.zeros((n, 0)), y, h, game, within)),
                "continue": np.zeros((1, nh))}
        if gd: base["game_only"] = score(loho_pred(none_f, Xd_game, y, h, game, within))
        d_only = score(loho_pred(Dp, Xd_game, y, h, game, within))                     # diag_only(+game): (P,nh)
        m_all = score(loho_pred(Dp, Xd_all, y, h, game, within))
        # baseline = argmax средней utility; для diag_only он зависит от перестановки
        names = list(base) + ["diag_only"]
        means = np.stack([np.repeat(base[k].mean(1), P) if base[k].shape[0] == 1 else base[k].mean(1) for k in base] + [d_only.mean(1)], 0)  # (kb+1,P)
        pick = means.argmax(0)                                                         # (P,)
        U_best = np.empty((P, nh))
        for i, nm in enumerate(names):
            sel = pick == i
            if sel.any(): U_best[sel] = d_only[sel] if nm == "diag_only" else np.broadcast_to(base[nm], (sel.sum(), nh))
        d_all = m_all - U_best
        stat = d_all.mean(1); d_obs = d_all[0]
        best = names[int(pick[0])]
    delta = float(stat[0])
    p = float((1 + (stat[1:] >= delta).sum()) / (P))
    boots = np.array([d_obs[rng.integers(0, nh, nh)].mean() for _ in range(n_boot)])
    lb, ub = float(np.quantile(boots, ALPHA)), float(np.quantile(boots, 1 - ALPHA))
    positive = bool(nh >= 12 and delta > 0 and p < ALPHA and lb > 0)
    negative = bool(nh >= 12 and ub < DELTA)
    return dict(positive=positive, negative=negative, delta=delta, p=p, lb=lb, ub=ub, best=best)


# ---------------------------------------------------------------- генератор данных
SCN = {   # b_age, b_game, b_diag (в sd шума), sigma_u (историческая случайная составляющая), game_sep (разделение игр признаками)
    "null":               dict(b_age=0.0, b_game=0.0, b_diag=0.0, sigma_u=0.7, game_sep=3.0),
    "null+game_offset":   dict(b_age=0.0, b_game=1.0, b_diag=0.0, sigma_u=0.7, game_sep=3.0),
    "null+age":           dict(b_age=0.8, b_game=0.0, b_diag=0.0, sigma_u=0.7, game_sep=3.0),
    "null+age+game":      dict(b_age=0.8, b_game=1.0, b_diag=0.0, sigma_u=0.7, game_sep=3.0),
    "null+game_opposite": dict(b_age=0.0, b_game=2.0, b_diag=0.0, sigma_u=0.7, game_sep=3.0),
    "diag(b=0.5)+age+game": dict(b_age=0.8, b_game=1.0, b_diag=0.5, sigma_u=0.7, game_sep=3.0),
    "diag(b=1.0)+age+game": dict(b_age=0.8, b_game=1.0, b_diag=1.0, sigma_u=0.7, game_sep=3.0),
    "diag(b=1.0)":        dict(b_age=0.0, b_game=0.0, b_diag=1.0, sigma_u=0.7, game_sep=3.0),
}

def make_data(n_hist, scn, rng):
    half = n_hist // 2
    game_h = np.r_[np.zeros(n_hist - half, int), np.ones(half, int)]
    h = np.repeat(np.arange(n_hist), 2); age = np.tile([0, 1], n_hist); game = game_h[h]
    # признаки как в реальных checkpoint: recent_return сильно кодирует игру и возраст; pre_td слабо; внутри (игра,возраст) — шум
    # + общая историческая компонента (оба возраста одной истории похожи)
    hist_f = rng.normal(size=(n_hist, 2))[h] * 0.6
    recent_idio = rng.normal(size=len(h)) * 0.8 + hist_f[:, 1]
    td_idio = rng.normal(size=len(h)) + hist_f[:, 0]
    recent = scn["game_sep"] * game + 1.2 * age + recent_idio
    pre_td = 0.5 * game + 0.3 * age + td_idio
    u = rng.normal(size=n_hist)[h] * scn["sigma_u"]
    y = (scn["b_age"] * (age - 0.5) * 2 + scn["b_game"] * (game - 0.5) * 2 + scn["b_diag"] * recent_idio
         + u + rng.normal(size=len(h)))
    return y, h, game, age, np.c_[pre_td, recent]


def selfcheck():
    """Совпадение delta/best с реализацией A на тех же данных (V0, одно тождественное «перестановочное» значение)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import debate_prereg_dryrun_v1 as A
    rng = np.random.default_rng(7); worst = 0.0; same_best = 0
    for t in range(30):
        y, h, game, age, diag = make_data(12, SCN["null+age+game"], rng)
        clock = np.c_[age.astype(float), age * 0.0, age * 0.0]
        a = A.procedure(clock, diag, y, h, np.random.default_rng(1), n_perm=1, n_boot=2)
        r = run_procedure("V0", y, h, game, age, diag, np.random.default_rng(1), n_perm=1, n_boot=2)
        worst = max(worst, abs(a["delta"] - r["delta"])); same_best += a["best"] == r["best"]
    print("selfcheck: max|delta_A - delta_B2| =", worst, "; best baseline совпал в", same_best, "из 30")
    return worst


def v1_dgp_run(n_hist, n_sims, n_perm, seed, only=None):
    """V0 (мой код) на DGP из scripts/debate_prereg_dryrun_v1.py: воспроизведение таблицы v1 A (FP и мощность)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import debate_prereg_dryrun_v1 as A1
    out = {}
    for name in (only or list(A1.SCN)):
        rng = np.random.default_rng(seed); res = []; t0 = time.time()
        for _ in range(n_sims):
            clock, diag, y, h = A1.simulate(A1.SCN[name], n_hist, rng)
            age = clock[:, 0].astype(int); game = h % 2
            res.append(run_procedure("V0", y, h, game, age, diag, rng, n_perm=n_perm))
        out[name] = dict(positive_rate=float(np.mean([r["positive"] for r in res])), mean_p=float(np.mean([r["p"] for r in res])),
                         frac_p_lt_alpha=float(np.mean([r["p"] < ALPHA for r in res])), n_sims=n_sims, sec=round(time.time() - t0, 1))
        print("v1-DGP", name, out[name], flush=True)
    Path(ROOT / f"research/debate/b2_v1_reproduction_n{n_hist}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "--selfcheck":
        selfcheck(); sys.exit()
    if sys.argv[1] == "--v1dgp":
        v1_dgp_run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])); sys.exit()
    n_hist, n_sims, n_perm = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    variants = sys.argv[4].split(","); scns = list(SCN) if sys.argv[5] == "all" else sys.argv[5].split(",")
    tag = sys.argv[6]; seed = int(sys.argv[7]) if len(sys.argv) > 7 else 20261009
    out = {}
    for sc in scns:
        for var in variants:
            t0 = time.time(); rng = np.random.default_rng(seed); res = []
            for _ in range(n_sims):
                y, h, game, age, diag = make_data(n_hist, SCN[sc], rng)
                res.append(run_procedure(var, y, h, game, age, diag, rng, n_perm=n_perm))
            pos = np.mean([r["positive"] for r in res]); neg = np.mean([r["negative"] for r in res])
            key = f"{sc}|{var}|n={n_hist}"
            out[key] = dict(positive_rate=float(pos), negative_rate=float(neg), mean_p=float(np.mean([r["p"] for r in res])),
                            frac_p_lt_alpha=float(np.mean([r["p"] < ALPHA for r in res])), mean_delta=float(np.mean([r["delta"] for r in res])),
                            n_sims=n_sims, n_perm=n_perm, sec=round(time.time() - t0, 1))
            print(key, out[key], flush=True)
    Path(ROOT / f"research/debate/b2_prereg_fp_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
