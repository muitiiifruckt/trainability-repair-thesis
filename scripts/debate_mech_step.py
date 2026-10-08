"""A-cycle 6: mechanism check for A4: update norms (body/head) over first 10 steps and heldout loss at 10 steps, sign_flip@600 and scale_drop@600."""
import sys, copy, statistics as st
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import torch; torch.set_num_threads(1)
from experiments import controlled_probe as cp
from debate_split_probe import reset_state
res={}
for sc in ("target_sign_flip","target_scale_drop"):
  for a in ("headw","headw:head","headw:body","headw:all"):
    bn=[];hn=[];l10=[]
    for seed in range(8):
        ox,oy,x,y,tx,ty=cp.data(seed,sc); torch.manual_seed(seed); m=cp.Model(); init=copy.deepcopy(m.state_dict()); o=cp.new_optimizer(m,0.001)
        g=torch.Generator().manual_seed(30000+seed)
        for _ in range(600):
            i=torch.randint(len(ox),(64,),generator=g); cp.train_step(m,o,ox[i],oy[i])
        ck=cp.snapshot(m,o,init,g); m,o=cp.restore(ck)
        with torch.no_grad(): m.head.weight.copy_(init["head.weight"]); m.head.bias.copy_(init["head.bias"])
        if ":" in a: reset_state(m,o,a.split(":")[1])
        gg=torch.Generator().manual_seed(40000+seed*100); b=0;h=0
        for k in range(10):
            idx=torch.randint(len(x),(64,),generator=gg)
            before=[p.detach().clone() for p in m.parameters()]; cp.train_step(m,o,x[idx],y[idx])
            d=[(p-q).norm()**2 for p,q in zip(m.parameters(),before)]
            names=[n for n,_ in m.named_parameters()]
            b+=sum(float(v) for n,v in zip(names,d) if not n.startswith("head")); h+=sum(float(v) for n,v in zip(names,d) if n.startswith("head"))
        bn.append(b**.5);hn.append(h**.5);l10.append(cp.evaluate(m,tx,ty))
    res[(sc,a)]=(round(st.mean(bn),3),round(st.mean(hn),3),round(st.median(l10),4))
    print(sc,a,"body_step_norm_sqsum,head_step_norm_sqsum,median heldout@10:",res[(sc,a)])
