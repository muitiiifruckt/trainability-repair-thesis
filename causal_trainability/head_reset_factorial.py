"""Actor/critic reset factorial, notes/sprint05_heads_prereg.md."""
import argparse
import concurrent.futures
import copy
import json
import math
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, stdev

import torch

from causal_trainability import deficit, hybrids
from causal_trainability.ppo_minatar import PPO, evaluate, reference_batch, seed_of
from causal_trainability.sprint03_live import atomic_json, digest, start_metrics
from causal_trainability.sprint04_pilot import fingerprint
from causal_trainability.telemetry import ROOT, Telemetry, code_revision, record_artifact

ARMS=('neither','actor','critic','both')
STEPS=131072
PROTOCOL=ROOT/'notes/sprint05_heads_prereg.md'
CAL=ROOT/'runs/causal-trainability/sprint05_heads_cal_v1_20261116'
OUT=ROOT/'runs/causal-trainability/sprint05_heads_live_v1_20261116'


def identity():return dict(files=code_revision()['files'],protocol_sha256=digest(PROTOCOL),steps=STEPS)


def reset_heads(model,initial,arm):
    if arm not in ARMS:raise ValueError(arm)
    chosen=('actor.','critic.') if arm=='both' else ((arm+'.',) if arm!='neither' else ())
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.startswith(chosen):p.copy_(initial[name])


@lru_cache(maxsize=2)
def inputs(history):
    source=ROOT/'runs/causal-trainability/ppo_pilot'/history
    fresh=hybrids.load(source,'age0.00')
    state,chain=deficit.chain_end(ROOT/'runs/causal-trainability/deficit_B'/history/'actions_chain')
    return fresh,state,dict(fresh=hybrids.checkpoint_row(source,'age0.00'),chain_sha256=chain['end']['sha256'])


def prepared(history,arm,repeat,out,steps,cal):
    fresh,state,source=inputs(history)
    ppo,cfg,seeds=deficit.prepared(fresh['config'],state,None,'actions',5,steps,(history,repeat),
        's05-heads-cal' if cal else 's05-heads',out,'s05_heads',f'heads_{arm}_{repeat}',
        telemetry_every=64,checkpoint_fractions=[0.,1.],snapshot_fractions=[0.,.5,1.])
    reset_heads(ppo.model,fresh['theta0'],arm)
    assert not ppo.optimizer.state
    return ppo,cfg,seeds,fresh,source


def behavior_hash(batch):
    return fingerprint({k:batch[k] for k in ('obs','actions','raw_rewards','dones','next_obs','shape')})


def factor_effect(r,kind):
    if kind=='actor':return (r['actor']-r['neither']+r['both']-r['critic'])/2
    if kind=='critic':return (r['critic']-r['neither']+r['both']-r['actor'])/2
    if kind=='interaction':return r['both']-r['actor']-r['critic']+r['neither']
    raise ValueError(kind)


def branch(history,arm,repeat,out,steps=STEPS,cal=False,enabled=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ppo,cfg,seeds,fresh,source=prepared(history,arm,repeat,out,steps,cal)
    torch.save(dict(theta0=ppo.theta0),out/'theta0.pt');record_artifact(out,out/'theta0.pt','theta0')
    evaluation=[seed_of('s05-heads-eval',history,i) for i in range(30)]
    evaluation_policy=seed_of('s05-heads-eval-policy',history,repeat)
    policy=lambda:torch.Generator().manual_seed(evaluation_policy)
    before=start_metrics(ppo,fresh['theta0'])
    if not cal:before['return']=evaluate(ppo.model,cfg,evaluation,policy())
    first={};rollout=ppo.rollout
    def capture():
        batch=rollout()
        if not first:
            first.update(behavior_sha256=behavior_hash(batch),targets_sha256=fingerprint([batch['values'],batch['advantages'],batch['returns']]),
                advantage_std=float(batch['advantages'].std()),return_mean=float(batch['returns'].mean()))
        return batch
    ppo.rollout=capture
    global_rng=torch.get_rng_state().clone();started=time.perf_counter()
    initial=Telemetry(out,'initial',ppo.model.layers(),ppo.optimizer,meta=dict(kind='s05_heads_initial'),enabled=enabled)
    initial.step(0,force=True,metrics=before);initial.close()
    ppo.train(telemetry_enabled=enabled)
    seconds=time.perf_counter()-started
    assert torch.equal(global_rng,torch.get_rng_state())
    final=dict(model=copy.deepcopy(ppo.model.state_dict()),optimizer=copy.deepcopy(ppo.optimizer.state_dict()),generator=ppo.generator.get_state().clone())
    torch.save(final,out/'final.pt');record_artifact(out,out/'final.pt','final_state')
    with torch.random.fork_rng(devices=[]):restored=PPO(cfg,out/'resume_check')
    restored.restore(torch.load(out/'checkpoints/age1.00.pt',map_location='cpu',weights_only=False))
    restored.learn(restored.last_rollout)
    assert fingerprint(final)==fingerprint(dict(model=restored.model.state_dict(),optimizer=restored.optimizer.state_dict(),generator=restored.generator.get_state()))
    result=dict(history=history,arm=arm,repeat=repeat,calibration=cal,steps=ppo.global_step,config=cfg,seeds=dict(seeds,evaluation=evaluation,evaluation_policy=evaluation_policy),
        source=source,identity=identity(),revision=code_revision(),start=before,first=first,seconds=seconds,final_sha256=fingerprint(final),
        global_rng_unchanged=True,resume_bit_equal=True,curve=ppo.curve,artifacts_sha256=digest(out/'artifacts.jsonl'))
    if not cal:
        auc,windows=deficit.auc(ppo.curve,STEPS,4)
        if any(v is None for v in windows):raise ValueError('Empty window, preserve attempt')
        result.update(auc=auc,windows=windows,final_return=evaluate(ppo.model,cfg,evaluation,policy()))
    atomic_json(out/'result.json',result)
    return result


def verified(path):
    row=json.loads(path.read_text());folder=path.parent
    assert row['identity']==identity()
    assert digest(folder/'artifacts.jsonl')==row['artifacts_sha256']
    for line in (folder/'artifacts.jsonl').read_text().splitlines():
        a=json.loads(line);assert digest(folder/a['path'])==a['sha256']
    for line in (folder/'checkpoints.jsonl').read_text().splitlines():
        a=json.loads(line);assert digest(folder/a['path'])==a['sha256']
    assert fingerprint(torch.load(folder/'final.pt',weights_only=False,map_location='cpu'))==row['final_sha256']
    if not row['calibration']:
        auc,windows=deficit.auc(row['curve'],STEPS,4)
        assert row['steps']==STEPS and windows==row['windows'] and auc==row['auc'] and all(v is not None for v in windows)
    return row


def calibrate():
    if CAL.exists():raise ValueError('Output already exists')
    CAL.mkdir(parents=True)
    cells={}
    for arm in ARMS:
        for flag in (False,True):branch('breakout_cnn_s0',arm,0,CAL/arm/f'warm_{flag}',4096,True,flag)
        pairs=[]
        for i in range(3):
            rows={flag:branch('breakout_cnn_s0',arm,0,CAL/arm/f'pair{i}_{flag}',STEPS,True,flag) for flag in ((True,False) if i%2==0 else (False,True))}
            assert rows[True]['final_sha256']==rows[False]['final_sha256'] and rows[True]['first']==rows[False]['first']
            pairs.append(dict(on=rows[True]['seconds'],off=rows[False]['seconds'],overhead=rows[True]['seconds']/rows[False]['seconds']-1,bit_equal=True))
            print('CAL',arm,i,flush=True)
        cells[arm]=dict(pairs=pairs,mean_overhead=mean(p['overhead'] for p in pairs),median_overhead=median(p['overhead'] for p in pairs))
    atomic_json(CAL/'summary.json',dict(identity=identity(),cells=cells,passed=all(c['mean_overhead']<=.05 and c['median_overhead']<=.05 for c in cells.values())))


def queue(workers):
    gate=json.loads((CAL/'summary.json').read_text());assert gate['passed'] and gate['identity']==identity()
    current=dict(identity=identity(),gate_sha256=digest(CAL/'summary.json'),histories=list(range(6)),repeats=[0,1],arms=list(ARMS))
    OUT.mkdir(parents=True,exist_ok=True);mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==current
    else:atomic_json(mp,current)
    failed=threading.Event()
    def job(item):
        if failed.is_set() or (OUT/'STOP').exists():return
        h,arm,rep=item;history=f'breakout_cnn_s{h}';folder=OUT/history/f'{arm}_{rep}'
        folder.mkdir(parents=True,exist_ok=True)
        try:
            if (folder/'result.json').exists():return
            attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}'
            print('START',history,arm,rep,flush=True)
            with (folder/(attempt.name+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m','causal_trainability.head_reset_factorial','branch','--history',history,'--arm',arm,'--repeat',str(rep),'--out',str(attempt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            atomic_json(folder/'result.json',dict(attempt=attempt.name,sha256=digest(attempt/'result.json')))
            print('DONE',history,arm,rep,flush=True)
        except BaseException:
            failed.set();raise
    jobs=[(h,arm,rep) for h in range(6) for rep in (0,1) for arm in ARMS]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:list(pool.map(job,jobs))
    if not (OUT/'STOP').exists():summarize()


def summarize():
    rows={};hashes={}
    for h in range(6):
        for rep in (0,1):
            for arm in ARMS:
                folder=OUT/f'breakout_cnn_s{h}'/f'{arm}_{rep}';wrapper=json.loads((folder/'result.json').read_text());path=folder/wrapper['attempt']/'result.json'
                assert digest(path)==wrapper['sha256'];row=verified(path)
                assert (row['history'],row['arm'],row['repeat'])==(f'breakout_cnn_s{h}',arm,rep)
                rows[h,rep,arm]=row;hashes[str(path.relative_to(OUT))]=digest(path)
            assert rows[h,rep,'neither']['first']['behavior_sha256']==rows[h,rep,'critic']['first']['behavior_sha256']
            assert rows[h,rep,'actor']['first']['behavior_sha256']==rows[h,rep,'both']['first']['behavior_sha256']
    effects={}
    for kind in ('actor','critic','interaction'):
        groups=[]
        for h in range(6):
            values=[]
            for rep in (0,1):
                r={arm:rows[h,rep,arm]['auc'] for arm in ARMS}
                values.append(factor_effect(r,kind))
            groups.append(values)
        ds=[mean(v) for v in groups];m=mean(ds);half=3.163381449748609*stdev(ds)/math.sqrt(6);ci=[m-half,m+half];noise=math.sqrt(mean(stdev(v)**2 for v in groups))
        verdict='beneficial' if ci[0]>.1 and sum(d>0 for d in ds)>=5 and m>noise else ('harmful' if ci[1]<-.1 and sum(d<0 for d in ds)>=5 and -m>noise else ('small' if ci[0]>=-.1 and ci[1]<=.1 else 'uncertain'))
        effects[kind]=dict(mean=m,ci975=ci,per_history=ds,within_history_sd=noise,verdict=verdict if kind!='interaction' else 'exploratory')
    atomic_json(OUT/'summary.json',dict(kind='conditional_early_on_policy_factorial_fresh_adam',branches=48,effects=effects,identity=identity(),inputs_sha256=hashes,
        rows=[rows[h,rep,arm] for h in range(6) for rep in (0,1) for arm in ARMS]))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','queue','branch','summarize']);p.add_argument('--workers',type=int,choices=[1,2],default=2);p.add_argument('--history');p.add_argument('--arm',choices=ARMS);p.add_argument('--repeat',type=int,choices=[0,1]);p.add_argument('--out',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    if code_revision()['dirty_causal_trainability']:raise SystemExit('Freeze package before execution')
    if a.mode=='branch':branch(a.history,a.arm,a.repeat,a.out)
    elif a.mode=='calibrate':calibrate()
    elif a.mode=='queue':queue(a.workers)
    else:summarize()


if __name__=='__main__':main()
