"""B-cycle: does stale target / Adam cold start explain head and joint reset damage? Fixed replay, no env stepping.
Read-only on checkpoints. 1 thread. Arms: continue, optimizer_reset, head_reset (campaign: target stays old),
head_reset_sync (target:=online after reset), head_and_optimizer_reset, joint_sync."""
import sys, json, copy, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch, numpy as np
torch.set_num_threads(1)
from experiments.rl_core import TrainingState
cps = sys.argv[1:]
out = {}
for cp in cps:
    path = next((ROOT/"runs/minatar-repair-20261007/workers/source").glob(f"*/checkpoints/{cp}.pt"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for arm in ["continue","optimizer_reset","head_reset","head_reset_sync","head_and_optimizer_reset","joint_sync"]:
        t0=time.time()
        s = TrainingState.restore(ck, continuation_seed=123)
        base = arm.replace("_sync","").replace("joint","head_and_optimizer_reset")
        s.apply_repair(base, 0)
        if arm.endswith("_sync"): s.target = copy.deepcopy(s.q).requires_grad_(False); s.last_target_update = s.gradient_updates
        losses, unorm = [], []
        for i in range(1000):
            idx = s.replay.sample_indices(s.replay_rng, 32)
            r = s.learn_batch(s.replay.batch(idx)); losses.append(r["loss"]); unorm.append(r["update_norm"])
        ev = s.evaluate([900000+i for i in range(20)])["mean_return"]
        out[f"{cp}|{arm}"] = {"loss_1_50": round(float(np.mean(losses[:50])),4), "loss_50_200": round(float(np.mean(losses[50:200])),4),
            "loss_800_1000": round(float(np.mean(losses[800:])),4), "upd_norm_first20": round(float(np.mean(unorm[:20])),4),
            "upd_norm_800_1000": round(float(np.mean(unorm[800:])),4), "eval_after_1000": ev, "sec": round(time.time()-t0,1)}
        print(cp, arm, out[f"{cp}|{arm}"], flush=True)
Path(ROOT/"research/debate/b_stale_target.json").write_text(json.dumps(out, indent=1))
