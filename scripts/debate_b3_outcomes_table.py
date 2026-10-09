"""B3: сводка по outcomes.jsonl (только чтение) для таблицы «можно / нельзя утверждать».

Единица независимости — история (trajectory_id). Ничего не пишет в runs/ и experiments/.
Результат: research/debate/b3_outcomes_table.json (+ печать).
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MENU = ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"]
REPAIRS = MENU[1:]


def load():
    path = ROOT / "runs" / "minatar-repair-20261007" / "outcomes.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    import hashlib
    return rows, hashlib.sha256(raw).hexdigest()


def main():
    rows, sha = load()
    panel = defaultdict(dict)
    for r in rows:
        panel[(r["checkpoint_id"], r["repeat"])][r["repair_id"]] = r
    complete = {k: v for k, v in panel.items() if all(a in v and v[a]["j_final"] is not None for a in MENU)}
    games = sorted({v["continue"]["game"] for v in complete.values()})
    out = {"outcomes_rows": len(rows), "outcomes_sha256": sha, "complete_panels": len(complete), "games": games}
    # эффекты по (checkpoint, repeat)
    eff = {a: {} for a in REPAIRS}
    imm = {a: {} for a in REPAIRS}
    for (cp, rep), p in complete.items():
        for a in REPAIRS:
            eff[a][(cp, rep)] = p[a]["j_final"] - p["continue"]["j_final"]
            imm[a][(cp, rep)] = p[a]["j_immediate"] - p["continue"]["j_immediate"]
    cps = sorted({cp for cp, _ in complete})
    hist_of = {cp: complete[(cp, next(r for c, r in complete if c == cp))]["continue"]["trajectory_id"] for cp in cps}
    age_of = {cp: complete[(cp, next(r for c, r in complete if c == cp))]["continue"]["nominal_age"] for cp in cps}
    hists = sorted(set(hist_of.values()))
    out["histories"] = hists
    table = {}
    for a in REPAIRS:
        cp_mean = {cp: float(np.mean([eff[a][(cp, r)] for c, r in complete if c == cp])) for cp in cps}
        h_mean = {h: float(np.mean([cp_mean[cp] for cp in cps if hist_of[cp] == h])) for h in hists}
        hv = np.array(list(h_mean.values()))
        n = len(hv)
        se = float(hv.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        t = float(hv.mean() / se) if se else None
        p = float(2 * stats.t.sf(abs(t), n - 1)) if t is not None else None
        # знаково-перестановочный по историям: минимально достижимое p
        signs = np.sign(hv)
        flips = [np.abs((np.array(s) * hv).mean()) for s in np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T]
        p_flip = float(np.mean([f >= abs(hv.mean()) - 1e-12 for f in flips]))
        rep_vals = np.array([eff[a][k] for k in sorted(eff[a])])
        by_age = {int(age): float(np.mean([cp_mean[cp] for cp in cps if age_of[cp] == age])) for age in sorted(set(age_of.values()))}
        table[a] = {"history_means": {h: round(v, 3) for h, v in h_mean.items()}, "mean_over_histories": round(float(hv.mean()), 3),
                    "se_over_histories": None if se is None else round(se, 3), "t": None if t is None else round(t, 2),
                    "p_t_df": None if p is None else round(p, 3), "p_signflip_over_histories": round(p_flip, 3),
                    "histories_negative": int((hv < 0).sum()), "histories": n,
                    "repeat_level_neg_zero_pos": [int((rep_vals < 0).sum()), int((rep_vals == 0).sum()), int((rep_vals > 0).sum())],
                    "by_age": by_age}
    out["effect_vs_continue"] = table
    # H-age: разность (50k - 200k) по историям для head_reset и остальных
    ages = sorted(set(age_of.values()))
    if len(ages) == 2:
        out["age_difference_low_minus_high"] = {}
        for a in REPAIRS:
            d = []
            for h in hists:
                lo = np.mean([np.mean([eff[a][(cp, r)] for c, r in complete if c == cp]) for cp in cps if hist_of[cp] == h and age_of[cp] == ages[0]])
                hi = np.mean([np.mean([eff[a][(cp, r)] for c, r in complete if c == cp]) for cp in cps if hist_of[cp] == h and age_of[cp] == ages[1]])
                d.append(float(lo - hi))
            dv = np.array(d)
            se = dv.std(ddof=1) / math.sqrt(len(dv))
            out["age_difference_low_minus_high"][a] = {"per_history": [round(x, 3) for x in d], "mean": round(float(dv.mean()), 3),
                                                       "se": round(float(se), 3), "t": round(float(dv.mean() / se), 2),
                                                       "p_t": round(float(2 * stats.t.sf(abs(dv.mean() / se), len(dv) - 1)), 3)}
    # interaction joint - head - opt
    inter = []
    for h in hists:
        v = lambda a: np.mean([np.mean([eff[a][(cp, r)] for c, r in complete if c == cp]) for cp in cps if hist_of[cp] == h])
        inter.append(float(v("head_and_optimizer_reset") - v("head_reset") - v("optimizer_reset")))
    iv = np.array(inter)
    se = iv.std(ddof=1) / math.sqrt(len(iv))
    out["interaction_joint_minus_head_minus_opt"] = {"per_history": [round(x, 3) for x in inter], "mean": round(float(iv.mean()), 3),
                                                     "se": round(float(se), 3), "t": round(float(iv.mean() / se), 2),
                                                     "p_t": round(float(2 * stats.t.sf(abs(iv.mean() / se), len(iv) - 1)), 3)}
    # immediate / recovery для head_reset
    a = "head_reset"
    ks = sorted(eff[a])
    pit = np.array([-imm[a][k] for k in ks])
    fin = np.array([eff[a][k] for k in ks])
    rec = fin + pit
    out["head_reset_pit_recovery"] = {
        "immediate_effect_all_negative": bool((np.array([imm[a][k] for k in ks]) < 0).all()),
        "immediate_mean_by_age": {int(age): round(float(np.mean([imm[a][k] for k in ks if age_of[k[0]] == age])), 3) for age in ages},
        "n_pairs": len(ks),
        "corr_final_effect_vs_pit": round(float(np.corrcoef(fin, pit)[0, 1]), 3),
        "corr_recovery_vs_pit_(math_coupled)": round(float(np.corrcoef(rec, pit)[0, 1]), 3),
        "corr_final_effect_vs_pit_within_age": {int(age): round(float(np.corrcoef(fin[[i for i, k in enumerate(ks) if age_of[k[0]] == age]], pit[[i for i, k in enumerate(ks) if age_of[k[0]] == age]])[0, 1]), 3) for age in ages},
        "recovery_mean_by_age": {int(age): round(float(np.mean([rec[i] for i, k in enumerate(ks) if age_of[k[0]] == age])), 3) for age in ages}}
    # шум одной ветки: continue между повторами
    d_cont = []
    for cp in cps:
        v = [complete[(c, r)]["continue"]["j_final"] for c, r in complete if c == cp]
        if len(v) == 2:
            d_cont.append(v[0] - v[1])
    out["continue_final_between_repeats"] = {"sd_of_difference": round(float(np.std(d_cont, ddof=1)), 3), "implied_sd_single": round(float(np.std(d_cont, ddof=1) / math.sqrt(2)), 3), "n": len(d_cont)}
    # j_pre против j_immediate у continue
    diffs = []
    for (cp, rep), p in complete.items():
        diffs.append(p["continue"]["j_immediate"] - p["continue"]["j_pre"])
    diffs = np.array(diffs)
    out["continue_jimmediate_minus_jpre"] = {"mean": round(float(diffs.mean()), 3), "n": len(diffs), "negative": int((diffs < 0).sum()), "positive": int((diffs > 0).sum())}
    # j_pre по возрастам и истории
    jp = {cp: complete[(cp, next(r for c, r in complete if c == cp))]["continue"]["j_pre"] for cp in cps}
    out["j_pre"] = {cp: jp[cp] for cp in cps}
    # парное попарное сравнение repair-arms между собой: средняя эффективность и порядок
    out["best_action_by_checkpoint_repeat"] = {}
    cnt = defaultdict(int)
    for k, p in complete.items():
        best = max(MENU, key=lambda a: p[a]["j_final"])
        cnt[best] += 1
    out["best_action_by_checkpoint_repeat"] = dict(cnt)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (ROOT / "research" / "debate" / "b3_outcomes_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
