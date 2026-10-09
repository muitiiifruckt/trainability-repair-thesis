"""[B2] Надёжность ИЗМЕРЕНИЯ pre-repair диагностик: шум выборки батча (diagnostics(batch_size=1024, seed)) против разброса между checkpoint.
Только чтение артефактов runs/... (reserved-игр нет в панели; берутся development-игры breakout/asterix), код — копия pkg0, 1 поток.
Для каждого checkpoint считается diagnostics(...) на 30 разных seeds батча (хранимое в diagnostics.jsonl значение — одна реализация с seed_for(cid,'diagnostics')).
Для каждого признака: sd внутри checkpoint (шум измерения), sd между checkpoint внутри (игра, возраст) и надёжность r = var_between/(var_between + var_within) (для одного замера);
плюс где лежит хранимое значение относительно 30 реализаций.
Только .venv/Scripts/python.exe. RAM: по одному checkpoint (<~0.8 GB для 200k)."""
import sys, json, time, gc
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve(); ROOT = HERE.parents[1]; PKG = ROOT / "research/debate/tmp/pkg0"
sys.path.insert(0, str(PKG))
import numpy as np, torch
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
from experiments.rl_core import TrainingState
from experiments.rl_runner import load_state
RUN = ROOT / "runs/minatar-repair-20261007"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
cps = rd(RUN / "checkpoints.jsonl"); stored = {r["checkpoint_id"]: r["features"] for r in rd(RUN / "diagnostics.jsonl")}
KEYS = ["pre_td_huber", "gradient_norm", "gradient_m_cosine", "m_norm", "sqrt_v_norm", "feature_entropy_rank_centered", "dormant_fraction_threshold_0_001", "q_abs_max", "target_abs_max"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
res = {}
for row in sorted(cps, key=lambda r: r["checkpoint_id"]):
    cid = row["checkpoint_id"]
    if row["game"] not in ("breakout", "asterix") or (only and not any(o in cid for o in only)): continue
    state = TrainingState.restore(load_state(Path(row["artifact_path"]))); gc.collect()
    t0 = time.time(); vals = {k: [] for k in KEYS}
    for s in range(1000, 1000 + N):
        d = state.diagnostics(batch_size=1024, seed=s)
        for k in KEYS: vals[k].append(d.get(k))
    out = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), stored=stored[cid].get(k), cv=float(np.std(v, ddof=1) / (abs(np.mean(v)) + 1e-12)),
                   stored_pct=float(np.mean(np.array(v) < stored[cid].get(k)))) for k, v in vals.items()}
    res[cid] = dict(game=row["game"], age=row["nominal_age"], feats=out, sec=round(time.time() - t0, 1))
    print(cid, "sec", res[cid]["sec"], {k: (round(out[k]["mean"], 4), round(out[k]["sd"], 4)) for k in KEYS[:4]}, flush=True)
    del state; gc.collect()
# надёжность одного замера: между checkpoint внутри (игра, возраст) vs шум измерения
summary = {}
for k in KEYS:
    within = [r["feats"][k]["sd"] ** 2 for r in res.values()]
    groups = {}
    for cid, r in res.items(): groups.setdefault((r["game"], r["age"]), []).append(r["feats"][k]["mean"])
    between = [np.var(v, ddof=1) for v in groups.values() if len(v) > 1]
    vb, vw = float(np.mean(between)) if between else float("nan"), float(np.mean(within))
    summary[k] = dict(var_between_within_cells=vb, var_measurement=vw, reliability_single_draw=vb / (vb + vw) if vb == vb else None,
                      n_cells=len(groups), mean_cv=float(np.mean([r["feats"][k]["cv"] for r in res.values()])))
    print("%-34s var_between(в ячейке игра x возраст)=%.4g  var_измерения=%.4g  надёжность=%.2f  (cv измерения %.3f)" % (k, vb, vw, vb / (vb + vw), summary[k]["mean_cv"]), flush=True)
Path(ROOT / "research/debate/b2_diag_noise.json").write_text(json.dumps(dict(per_checkpoint=res, summary=summary), indent=1, ensure_ascii=False), encoding="utf-8")
