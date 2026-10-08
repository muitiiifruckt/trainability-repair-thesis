"""A-cycle 3: do pre-repair diagnostics predict sign of state-reset / head-reset effect in the reduced synthetic probe?
Unit = seed (leave-one-seed-out). Supervised, NOT RL. Diagnostics here see the future labeled task (known limitation of the probe)."""
import sys, json, copy, math, statistics as st
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import torch; torch.set_num_threads(1)
from experiments import controlled_probe as cp
from debate_split_probe import branch
rows=[]
for seed in range(12):
    for sc in cp.SCENARIOS:
        ox,oy,x,y,tx,ty = cp.data(seed, sc)
        torch.manual_seed(seed); m=cp.Model(); init=copy.deepcopy(m.state_dict()); opt=cp.new_optimizer(m,0.001)
        g=torch.Generator().manual_seed(30000+seed)
        for age in range(1,601):
            i=torch.randint(len(ox),(64,),generator=g); cp.train_step(m,opt,ox[i],oy[i])
            if age in (100,300,600):
                ck=cp.snapshot(m,opt,init,g); diag=cp.diagnostics(m,opt,x,y)
                gg=torch.Generator().manual_seed(40000+seed); b=[torch.randint(len(x),(64,),generator=gg) for _ in range(100)]
                L={a:branch(ck,a,x,y,tx,ty,b) for a in ("continue","state:all","headw")}
                rows.append(dict(seed=seed,sc=sc,age=age,diag=diag,
                    eff_state=math.log(L["state:all"]/L["continue"]), eff_head=math.log(L["headw"]/L["continue"])))
json.dump(rows, open(ROOT/"research/debate/diag_sign_rows.json","w"))
feats=["pre_future_training_loss","weight_norm","gradient_norm","m_norm","sqrt_v_norm","gradient_m_cosine","feature_entropy_rank_centered","dormant_fraction_threshold_0_001","age"]
def val(r,f): return r["age"] if f=="age" else r["diag"][f]
def auc(s,l):
    pos=[a for a,b in zip(s,l) if b]; neg=[a for a,b in zip(s,l) if not b]
    if not pos or not neg: return float("nan")
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
out={}
for tgt in ("eff_state","eff_head"):
    lab=[r[tgt]<0 for r in rows]  # repair helps
    out[tgt]={"frac_helps":round(sum(lab)/len(lab),3),"n":len(rows),"auc_pooled":{}, "auc_within_scenario_mean":{}}
    for f in feats:
        s=[val(r,f) for r in rows]; out[tgt]["auc_pooled"][f]=round(auc(s,lab),3)
        ws=[]
        for sc in cp.SCENARIOS:
            sub=[r for r in rows if r["sc"]==sc]; a=auc([val(r,f) for r in sub],[r[tgt]<0 for r in sub]); 
            if a==a: ws.append(a)
        out[tgt]["auc_within_scenario_mean"][f]=round(st.mean(ws),3) if ws else None
    # scenario-label oracle
    byS={sc:st.mean(r[tgt]<0 for r in rows if r["sc"]==sc) for sc in cp.SCENARIOS}; out[tgt]["helps_by_scenario"]={k:round(v,2) for k,v in byS.items()}
print(json.dumps(out,indent=1)); json.dump(out,open(ROOT/"research/debate/diag_sign.json","w"),indent=1)
