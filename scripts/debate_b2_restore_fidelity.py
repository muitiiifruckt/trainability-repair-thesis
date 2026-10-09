"""[B2] Верность восстановления checkpoint и шум оценки политики на РЕАЛЬНЫХ артефактах кампании (только чтение runs/...; код — копия pkg0).
Для каждого breakout-checkpoint:
  (a) sha артефакта совпадает с checkpoints.jsonl;
  (b) повторная оценка на точных 'pre' seeds воспроизводит сохранённый j_pre и список returns (restore/evaluate детерминированы);
  (c) повторная оценка на seeds 'immediate' (repeat 0/1) воспроизводит j_immediate ветки continue (значит, сдвиг j_pre -> j_immediate у continue — чистый шум seeds);
  (d) 200 НОВЫХ seeds: среднее и SE реальной политики, время на эпизод (стоимость увеличения числа eval-эпизодов).
Только .venv/Scripts/python.exe. 1 поток, RAM: по одному checkpoint (<= ~0.5 GB)."""
import sys, json, time, gc, types
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve(); ROOT = HERE.parents[1]; PKG = ROOT / "research/debate/tmp/pkg0"
sys.path.insert(0, str(PKG))
import numpy as np, torch
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
from experiments.rl_core import DQNConfig, QNetwork, TrainingState
from experiments.rl_runner import seed_for, file_hash, load_state
RUN = ROOT / "runs/minatar-repair-20261007"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
cps = [r for r in rd(RUN / "checkpoints.jsonl") if r["game"] == "breakout"]
outs = rd(RUN / "outcomes.jsonl")
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
res = {}
for row in sorted(cps, key=lambda r: r["checkpoint_id"]):
    cid = row["checkpoint_id"]
    if only and not any(o in cid for o in only): continue
    art = Path(row["artifact_path"])
    ok_sha = file_hash(art) == row["sha256"]
    ck = load_state(art)
    cfg = DQNConfig(**ck["config"]); q = QNetwork(cfg, ck.get("q_injected", False)); q.load_state_dict(ck["weights"]); q.eval()
    ns = types.SimpleNamespace(game=ck["game"], config=cfg, q=q)
    del ck; gc.collect()
    ev_pre = TrainingState.evaluate(ns, [seed_for(cid, "pre", i) for i in range(20)])
    out = dict(sha_ok=ok_sha, j_pre_stored=row["j_pre"], j_pre_recomputed=ev_pre["mean_return"],
               pre_returns_identical=ev_pre["returns"] == row["pre_evaluation"]["returns"])
    for rep in (0, 1):
        cont = [o for o in outs if o["checkpoint_id"] == cid and o["repair_id"] == "continue" and o["repeat"] == rep]
        if not cont: continue
        ev = TrainingState.evaluate(ns, cont[0]["evaluation_curve"][0]["evaluation_seeds"])
        out[f"immediate_rep{rep}_stored_vs_recomputed"] = [cont[0]["j_immediate"], ev["mean_return"], ev["returns"] == cont[0]["evaluation_curve"][0]["returns"]]
    t0 = time.time(); fresh = TrainingState.evaluate(ns, [seed_for(cid, "b2_fresh", i) for i in range(200)]); dt = time.time() - t0
    r = np.array(fresh["returns"])
    out.update(fresh200_mean=round(float(r.mean()), 3), fresh200_sd_episode=round(float(r.std(ddof=1)), 3), fresh200_se=round(float(r.std(ddof=1) / np.sqrt(len(r))), 3),
               mean_episode_len=round(float(np.mean(fresh["lengths"])), 1), sec_per_200_episodes=round(dt, 1), sec_per_episode=round(dt / 200, 3))
    # где лежат 20-эпизодные оценки pre / immediate относительно распределения 20-эпизодных средних той же политики (по 200 новым эпизодам)
    boot = np.array([np.random.default_rng(i).choice(r, 20, replace=False).mean() for i in range(4000)])
    out["pct_of_pre_in_fresh20_means"] = round(float((boot < row["j_pre"]).mean() + 0.5 * (boot == row["j_pre"]).mean()), 3)
    for rep in (0, 1):
        k = f"immediate_rep{rep}_stored_vs_recomputed"
        if k in out: out[f"pct_of_immediate_rep{rep}"] = round(float((boot < out[k][0]).mean() + 0.5 * (boot == out[k][0]).mean()), 3)
    out["fresh200_returns"] = fresh["returns"]; out["pre_returns"] = ev_pre["returns"]
    res[cid] = out; print(cid, json.dumps({k: v for k, v in out.items() if not k.endswith("returns")}, ensure_ascii=False), flush=True)
    del ns, q; gc.collect()
Path(ROOT / "research/debate/b2_restore_fidelity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
