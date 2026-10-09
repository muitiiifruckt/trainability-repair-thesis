"""B3: операционные характеристики primary_gate (rl_analysis._gate) на копии кода pkg0.

Только чтение кода кампании (используется копия research/debate/tmp/pkg0, SHA rl_analysis 76315e19...).
Синтетические исходы, numpy, 1 поток. Ничего не пишет в experiments/, configs/, runs/.

Использование:
  python scripts/debate_b3_gate_sim.py equiv                      # проверка эквивалентности с analyze()
  python scripts/debate_b3_gate_sim.py run <сценарий> <n_sims> <B> [hist_per_game] [repeats] [tag]
Сценарии: null, game_level, age_level, uniform_best, het_tau<значение>, uniform_failed
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "debate" / "tmp" / "pkg0"))
import numpy as np  # noqa: E402

import experiments.rl_analysis as ra  # noqa: E402

assert "pkg0" in ra.__file__, ra.__file__
MENU = list(ra.MENU)
AGES = (50000, 200000)
GAMES = ("breakout", "asterix")
SCALE = {"breakout": 1.0, "asterix": 0.25}


def make_dataset(rng, scenario, hist_per_game=6, repeats=6, sigma=0.7, tau=0.0, fail_one=False):
    """Возвращает (data, truth). Эффекты задаются в единицах возврата; sigma — sd шума независимого j_final на ветку."""
    cps, rows, feats = {}, [], {}
    # детерминированные (игра, возраст)-эффекты по действиям (continue = 0)
    ga = {g: {a: np.zeros(4) for a in AGES} for g in GAMES}
    if scenario == "game_level":      # лучшее действие зависит только от игры
        for a in AGES:
            ga["breakout"][a] = np.array([0.0, -0.5, 0.5, -1.0])
            ga["asterix"][a] = np.array([0.0, 0.8, -0.8, -1.6])       # до масштаба SCALE (0.25) -> +0.2/-0.2/-0.4
    elif scenario == "age_level":     # знак head_reset зависит только от возраста (как в 1-й истории)
        for g in GAMES:
            ga[g][50000] = np.array([0.0, -0.3, 0.6, -0.6])
            ga[g][200000] = np.array([0.0, -0.3, -1.2, -1.5])
    elif scenario in ("uniform_best", "uniform_failed"):  # одно и то же лучшее для всех
        for g in GAMES:
            for a in AGES:
                ga[g][a] = np.array([0.0, 0.0, 0.8, 0.0])
    for g in GAMES:
        for s in range(hist_per_game):
            traj = f"natural_{g}_seed{s}"
            hist_effect = tau * SCALE[g] * rng.standard_normal(4) * np.array([0, 1, 1, 1])  # эффект истории (общий для возрастов)
            for age in AGES:
                cid = f"{traj}_age{age}"
                cps[cid] = {"checkpoint_id": cid, "trajectory_id": traj, "game": g, "nominal_age": age,
                            "environment_steps": age + 499, "source_mode": "natural", "j_pre": 3.0 * SCALE[g],
                            "random_return": 0.5, "baseline_learned": True}
                feats[cid] = {"x": float(rng.standard_normal()), "q_abs_max": float(10.0 * SCALE[g] + rng.standard_normal())}
                cp_effect = tau * SCALE[g] * rng.standard_normal(4) * np.array([0, 1, 1, 1])   # эффект checkpoint
                base = ga[g][age] * SCALE[g] if scenario != "game_level" else ga[g][age] * (1.0 if g == "breakout" else 0.25)
                eff = base + hist_effect + cp_effect
                for r in range(repeats):
                    level = 3.0 * SCALE[g] + SCALE[g] * 0.5 * rng.standard_normal()      # общий шум (сокращается в парах)
                    for ai, action in enumerate(MENU):
                        j = level + eff[ai] + sigma * SCALE[g] * rng.standard_normal()
                        rows.append({"checkpoint_id": cid, "trajectory_id": traj, "game": g, "source_mode": "natural",
                                     "repair_id": action, "repeat": r, "continuation_seed": 1000 * r + 7,
                                     "budget_env_steps": 50000, "completed_env_steps": 50000,
                                     "budget_gradient_updates": 50000, "j_pre": 3.0 * SCALE[g], "j_immediate": j,
                                     "j_final": float(j), "adaptation_auc": float(j), "failure_reason": None})
    if fail_one:
        victim = next(r for r in rows if r["repair_id"] == "head_reset")
        victim["failure_reason"], victim["j_final"] = "nonfinite_update", None
    config = {"kind": "b3_sim", "main_menu": MENU, "screen": {"games": list(GAMES)},
              "adaptive": {"reserved_games": ["freeway", "seaquest"]}}
    data = {"config": config, "checkpoints": cps, "features": feats, "rows": rows, "menu": tuple(MENU),
            "worker": {}, "missing_files": []}
    return data


def gate_for(data, B):
    """Копия строк analyze() 405-447 для основного страта (natural, 50000)."""
    (mode, budget), rows = next(iter(sorted(ra._strata(data).items(), key=lambda p: str(p[0]))))
    panels = ra._panels(data, rows)
    complete = ra._complete(data, panels)
    samples = ra._cp_means(data, complete)
    failures = Counter(str(r.get("failure_reason")) for r in rows if r.get("failure_reason"))
    nonfinite = sum(not ra._finite(r.get("j_final")) for r in rows)
    budget_mismatches = sum(ra._budget_mismatch(r) for r in rows)
    missing_action_rows = sum(len(set(data["menu"]) - set(p)) for p in panels.values())
    crossfit, choices, skipped = ra._crossfit(data, samples)
    cfb = ra._bootstrap_crossfit(data, samples, crossfit, B)
    stats = cfb["advantage_vs_sbs"]
    qualification = ra._qualification(samples)
    minimum = min((len(v) for v in complete.values()), default=0)
    incomplete = bool(data["missing_files"] or missing_action_rows or failures or nonfinite or budget_mismatches or skipped or not samples)
    gate = ra._gate(incomplete, qualification, stats, choices, minimum, float(data["config"].get("heterogeneity_margin", 0.0)))
    ps = ra._parameter_signal(data, panels, B, incomplete)
    return {"gate": gate, "stats": stats, "effect_vs_continue": cfb["effect_vs_continue"], "choices": choices,
            "param": ps["status"], "param_pos": ps["positive_contrasts"], "minimum": minimum}


def equiv_check(seed=1):
    rng = np.random.default_rng(seed)
    data = make_dataset(rng, "game_level", hist_per_game=6, repeats=4)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "config.json").write_text(json.dumps(data["config"]), encoding="utf-8")
        for name, items in (("checkpoints", list(data["checkpoints"].values())),
                            ("diagnostics", [{"checkpoint_id": k, "features": v} for k, v in data["features"].items()]),
                            ("outcomes", data["rows"])):
            (tmp / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")
        t = time.time()
        full = ra.analyze(tmp, bootstrap_repeats=100)
        dt = time.time() - t
    mine = gate_for(make_dataset(np.random.default_rng(seed), "game_level", 6, 4), 100)
    st = full["strata"][0]
    a = st["cross_fitted_estimated_winner"]["advantage_vs_sbs"]
    ok = (st["primary_gate"] == mine["gate"] and all(abs(a[g]["mean"] - mine["stats"][g]["mean"]) < 1e-12 and a[g]["ci95"] == mine["stats"][g]["ci95"] for g in a)
          and st["parameter_repair_signal"]["status"] == mine["param"])
    print(json.dumps({"analyze_seconds": round(dt, 2), "gate_analyze": st["primary_gate"], "gate_mine": mine["gate"], "equal": ok,
                      "adv": {g: [a[g]["mean"], a[g]["ci95"]] for g in a}}, ensure_ascii=False))


def run(scenario, n_sims, B, hist_per_game=6, repeats=6, tag=None, seed=20261009, sigma=0.7):
    tau = 0.0
    if scenario.startswith("het_tau"):
        tau = float(scenario[len("het_tau"):])
    rng = np.random.default_rng(seed)
    out = Counter()
    param = Counter()
    adv_means, lbs = {g: [] for g in GAMES}, {g: [] for g in GAMES}
    uniq_n = []
    t0 = time.time()
    for i in range(n_sims):
        data = make_dataset(rng, scenario if not scenario.startswith("het_tau") else "het", hist_per_game, repeats,
                            sigma=sigma, tau=tau, fail_one=(scenario == "uniform_failed"))
        res = gate_for(data, B)
        out[res["gate"]] += 1
        param[res["param"]] += 1
        for g, s in res["stats"].items():
            adv_means[g].append(s["mean"])
            lbs[g].append(None if s["ci95"] is None else s["ci95"][0])
        uniq_n.append(len({c["chosen_action"] for c in res["choices"]}))
        if time.time() - t0 > 270:
            print("TIME LIMIT: прерываю на", i + 1, "симуляциях")
            n_sims = i + 1
            break
    summary = {"scenario": scenario, "n_sims": n_sims, "B": B, "hist_per_game": hist_per_game, "repeats": repeats, "sigma": sigma, "tau": tau,
               "gate_counts": dict(out), "gate_frac": {k: round(v / n_sims, 3) for k, v in out.items()},
               "param_status": dict(param),
               "mean_adv_vs_sbs": {g: round(float(np.mean(v)), 4) for g, v in adv_means.items() if v},
               "mean_lb": {g: round(float(np.mean([x for x in v if x is not None])), 4) for g, v in lbs.items() if any(x is not None for x in v)},
               "mean_unique_choices": round(float(np.mean(uniq_n)), 2), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, ensure_ascii=False))
    path = ROOT / "research" / "debate" / f"b3_gate_{tag or scenario}_h{hist_per_game}_r{repeats}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__" and sys.argv[1] in ("equiv", "run"):
    if sys.argv[1] == "equiv":
        equiv_check()
    else:
        sc, n, b = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        hpg = int(sys.argv[5]) if len(sys.argv) > 5 else 6
        rep = int(sys.argv[6]) if len(sys.argv) > 6 else 6
        tag = sys.argv[7] if len(sys.argv) > 7 else None
        run(sc, n, b, hpg, rep, tag)


def real_breakout(B=1000):
    """Применяет gate_for() к реальным строкам outcomes.jsonl (только чтение), оставляя игры, по которым есть полные панели."""
    data = ra._load(ROOT / "runs" / "minatar-repair-20261007")
    done = {r["game"] for r in data["rows"]}
    data["rows"] = [r for r in data["rows"] if r["game"] in done]
    data["checkpoints"] = {k: v for k, v in data["checkpoints"].items() if v["game"] in done}
    res = gate_for(data, B)
    out = {"games": sorted(done), "n_rows": len(data["rows"]), "gate_if_complete": res["gate"],
           "adv_vs_sbs": {g: {"mean": s["mean"], "ci95": s["ci95"], "histories": s["source_histories"]} for g, s in res["stats"].items()},
           "effect_vs_continue": {g: {"mean": s["mean"], "ci95": s["ci95"]} for g, s in res["effect_vs_continue"].items()},
           "choices": [(c["checkpoint_id"][8:], c["fold"], c["chosen_action"], c["sbs_action"]) for c in res["choices"]],
           "param_status": res["param"], "param_pos": res["param_pos"]}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "real":
    real_breakout(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)


def selector_check(scenario, n_sims, B, hist_per_game=6, repeats=6, seed=77, sigma=0.7, tau=0.0):
    """Вторая ступень: fit_selectors на тех же синтетических данных (копия кода pkg0). Только selector_gate диагностического Ridge."""
    rng = np.random.default_rng(seed)
    counts = Counter()
    t0 = time.time()
    for i in range(n_sims):
        data = make_dataset(rng, scenario, hist_per_game, repeats, sigma=sigma, tau=tau)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "config.json").write_text(json.dumps(data["config"]), encoding="utf-8")
            for name, items in (("checkpoints", list(data["checkpoints"].values())),
                                ("diagnostics", [{"checkpoint_id": k, "features": v} for k, v in data["features"].items()]),
                                ("outcomes", data["rows"])):
                (tmp / f"{name}.jsonl").write_text("".join(json.dumps(x) + chr(10) for x in items), encoding="utf-8")
            res = ra.fit_selectors(tmp, bootstrap_repeats=B)
        st = res["strata"][0]
        counts[st["selector_gate"]] += 1
        if time.time() - t0 > 250:
            n_sims = i + 1
            break
    print(json.dumps({"scenario": scenario, "n_sims": n_sims, "B": B, "selector_gate_counts": dict(counts), "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))
    (ROOT / "research" / "debate" / f"b3_selector_{scenario}_h{hist_per_game}_r{repeats}.json").write_text(
        json.dumps({"scenario": scenario, "n_sims": n_sims, "B": B, "selector_gate_counts": dict(counts)}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "selector":
    selector_check(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                   int(sys.argv[5]) if len(sys.argv) > 5 else 6, int(sys.argv[6]) if len(sys.argv) > 6 else 6)
