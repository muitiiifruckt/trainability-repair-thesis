"""A-cycle 2: H-split in the reduced synthetic probe (supervised, NOT RL). Imports experiments/controlled_probe.py read-only.
Adds split Adam-state resets: moments (m,v,t) of head only / body only, with or without head weight reset."""
import sys, json, copy, math, statistics as st, time, collections
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
from experiments import controlled_probe as cp

HEAD = ("head.weight", "head.bias")
def reset_state(model, opt, which):
    names = {id(p): n for n, p in model.named_parameters()}
    for p in list(opt.state.keys()):
        n = names[id(p)]
        if which == "all" or (which == "head" and n in HEAD) or (which == "body" and n not in HEAD):
            del opt.state[p]
def branch(ckpt, action, x, y, tx, ty, batches):
    model, opt = cp.restore(ckpt)
    hr, st_ = action.split("+") if "+" in action else (None, action)
    if action.startswith("headw"):
        with torch.no_grad():
            model.head.weight.copy_(ckpt["initial_weights"]["head.weight"]); model.head.bias.copy_(ckpt["initial_weights"]["head.bias"])
    scope = action.split(":")[1] if ":" in action else None
    if scope: reset_state(model, opt, scope)
    for idx in batches: cp.train_step(model, opt, x[idx], y[idx])
    return cp.evaluate(model, tx, ty)
ACTS = ["continue","state:head","state:body","state:all","headw","headw:head","headw:body","headw:all"]
def main(seeds=8, ages=(100,600), updates=100, lr=0.001):
    t0=time.time(); recs=[]
    for seed in range(seeds):
        for sc in cp.SCENARIOS:
            ox, oy, x, y, tx, ty = cp.data(seed, sc)
            torch.manual_seed(seed); model = cp.Model(); init = copy.deepcopy(model.state_dict()); opt = cp.new_optimizer(model, lr)
            g = torch.Generator().manual_seed(30000+seed)
            for age in range(1, max(ages)+1):
                idx = torch.randint(len(ox), (64,), generator=g); cp.train_step(model, opt, ox[idx], oy[idx])
                if age in ages:
                    ck = cp.snapshot(model, opt, init, g)
                    for rep in range(2):
                        gg = torch.Generator().manual_seed(40000+seed*100+rep)
                        batches = [torch.randint(len(x), (64,), generator=gg) for _ in range(updates)]
                        for a in ACTS:
                            recs.append(dict(seed=seed, sc=sc, age=age, rep=rep, a=a, loss=branch(ck, a, x, y, tx, ty, batches)))
    print("wall", round(time.time()-t0,1))
    return recs
def analyse(recs):
    # log ratio vs continue, mean over reps within seed, then across seeds (unit = seed)
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs: d[(r["sc"], r["age"], r["seed"], r["a"])][r["rep"]] = r["loss"]
    out = {}
    for sc in cp.SCENARIOS:
        for age in (100, 600):
            per = collections.defaultdict(list)
            for seed in sorted({k[2] for k in d}):
                base = st.mean(d[(sc,age,seed,"continue")].values())
                for a in ACTS[1:]:
                    per[a].append(math.log(st.mean(d[(sc,age,seed,a)].values())/base))
            # head state-vs-weights interaction: headw:all - headw - state:all (log scale)
            inter = [per["headw:all"][i]-per["headw"][i]-per["state:all"][i] for i in range(len(per["headw"]))]
            # split decomposition inside joint: which part of state matters when head weights reset
            out[f"{sc}|{age}"] = {a: [round(st.mean(v),3), round(st.stdev(v)/math.sqrt(len(v)),3)] for a, v in per.items()}
            out[f"{sc}|{age}"]["inter_logscale"] = [round(st.mean(inter),3), round(st.stdev(inter)/math.sqrt(len(inter)),3)]
    return out
if __name__ == "__main__":
    recs = main(); res = analyse(recs)
    Path(ROOT/"research/debate/split_probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items(): print(k, {a: x for a, x in v.items()})
