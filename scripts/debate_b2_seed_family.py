"""[B2] Есть ли систематическое различие между семействами eval seeds ('pre' vs 'immediate' vs новые)? Только чтение runs/..., код pkg0.
Для каждого breakout-checkpoint политика (веса из артефакта) оценивается на 3 семействах по 300 seeds, ВНЕ исходных первых 20:
  A: seed_for(cid, "pre", i), i=20..319           (семейство j_pre)
  B: seed_for(cid, "natural", 0, "evaluation", 0, i), i=20..319   (семейство j_immediate repeat 0)
  C: seed_for(cid, "b2_fresh2", i), i=0..299      (новое)
Если семейства различаются систематически (хэш-структура seeds), средние A/B/C разойдутся; если нет — расхождение j_pre vs j_immediate на первых 20 seeds — шум.
Только .venv/Scripts/python.exe. 1 поток."""
import sys, json, time, gc, types
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve(); ROOT = HERE.parents[1]; PKG = ROOT / "research/debate/tmp/pkg0"
sys.path.insert(0, str(PKG))
import numpy as np, torch
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
from experiments.rl_core import DQNConfig, QNetwork, TrainingState
from experiments.rl_runner import seed_for, load_state
RUN = ROOT / "runs/minatar-repair-20261007"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
cps = [r for r in rd(RUN / "checkpoints.jsonl") if r["game"] == "breakout"]
res = {}
for row in sorted(cps, key=lambda r: r["checkpoint_id"]):
    cid = row["checkpoint_id"]
    ck = load_state(Path(row["artifact_path"]))
    cfg = DQNConfig(**ck["config"]); q = QNetwork(cfg, ck.get("q_injected", False)); q.load_state_dict(ck["weights"])
    ns = types.SimpleNamespace(game=ck["game"], config=cfg, q=q); del ck; gc.collect()
    fam = {"A_pre_family": [seed_for(cid, "pre", i) for i in range(20, 320)],
           "B_immediate_family": [seed_for(cid, "natural", 0, "evaluation", 0, i) for i in range(20, 320)],
           "C_fresh": [seed_for(cid, "b2_fresh2", i) for i in range(300)]}
    out = {}
    for name, seeds in fam.items():
        r = np.array(TrainingState.evaluate(ns, seeds)["returns"])
        out[name] = [round(float(r.mean()), 3), round(float(r.std(ddof=1) / np.sqrt(len(r))), 3)]
    out["stored_j_pre_first20"] = row["j_pre"]
    res[cid] = out; print(cid, json.dumps(out), flush=True)
    del ns, q; gc.collect()
Path(ROOT / "research/debate/b2_seed_family.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
