"""Initial-policy matching with frozen functional anchors; sprint 06."""
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
from torch import nn

from causal_trainability import deficit, hybrids
from causal_trainability.ppo_minatar import PPO, evaluate, reference_batch, seed_of
from causal_trainability.sprint03_live import atomic_json, digest
from causal_trainability.sprint04_pilot import fingerprint
from causal_trainability.telemetry import ROOT, Telemetry, code_revision, record_artifact

ARMS=('neither','reset','matched')
STEPS=131072
PROTOCOL=ROOT/'notes/sprint06_match_prereg.md'
CAL=ROOT/'runs/causal-trainability/sprint06_match_cal_v1_20261118'
OUT=ROOT/'runs/causal-trainability/sprint06_match_live_v1_20261118'
PILOT=ROOT/'runs/causal-trainability/sprint06_match_pilot_v1_20261118'


def identity():return dict(files=code_revision()['files'],protocol_sha256=digest(PROTOCOL),steps=STEPS)


class MatchedPolicy(nn.Module):
    """Core remains trainable; frozen anchors specify a constant function correction."""
    def __init__(self,core,initial,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.core=core
        self.anchor=copy.deepcopy(core).requires_grad_(False)
        self.reset_actor=copy.deepcopy(core.actor)
        with torch.no_grad():
            self.reset_actor.weight.copy_(initial['actor.weight'])
            self.reset_actor.bias.copy_(initial['actor.bias'])
        self.reset_actor.requires_grad_(False)
        self.arm=arm
        if arm=='reset':
            self.core.actor.load_state_dict(self.reset_actor.state_dict())

    def forward(self,obs):
        logits,value=self.core(obs)
        if self.arm=='matched':
            with torch.no_grad():
                hidden,_=self.anchor.features(obs)
                old_logits=self.anchor.actor(hidden)
                reset_logits=self.reset_actor(hidden)
            logits=reset_logits+(logits-old_logits)
        return logits,value

    def features(self,obs):return self.core.features(obs)

    def layers(self):
        return dict(self.core.layers(),anchors=list(self.anchor.parameters())+list(self.reset_actor.parameters()))


def wrap(ppo,initial,arm):
    ppo.cfg['behavior_arm']=arm
    ppo.model=MatchedPolicy(ppo.model,initial,arm)
    ppo.optimizer=torch.optim.Adam(ppo.model.core.parameters(),lr=ppo.cfg['lr'],eps=ppo.cfg['adam_eps'])


def anchor_hash(model):return fingerprint([model.anchor.state_dict(),model.reset_actor.state_dict()])


def start_metrics(ppo,initial):
    model=ppo.model
    with torch.no_grad():
        reference=reference_batch(ppo.cfg)
        logits,values=model(reference)
        hidden,_=model.anchor.features(reference)
        reset_logits=model.reset_actor(hidden)
        old_values=model.anchor.critic(hidden).squeeze(-1)
        logp=logits.log_softmax(-1)
        result=dict(entropy=float(-(logp.exp()*logp).sum(-1).mean()),value_rms=float(values.square().mean().sqrt()),
            reset_logits_equal=bool(torch.equal(logits,reset_logits)),values_equal_anchor=bool(torch.equal(values,old_values)),
            policy_tv_vs_reset=float((logp.exp()-reset_logits.softmax(-1)).abs().sum(-1).mean()/2),
            raw_actor_old_equal=fingerprint(model.core.actor.state_dict())==fingerprint(model.anchor.actor.state_dict()),
            raw_actor_reset_equal=fingerprint(model.core.actor.state_dict())==fingerprint(model.reset_actor.state_dict()))
    assert result['values_equal_anchor']
    if model.arm in ('matched','reset'):assert result['reset_logits_equal']
    assert result['raw_actor_old_equal'] if model.arm!='reset' else result['raw_actor_reset_equal']
    return result



@lru_cache(maxsize=2)
def inputs(history):
    source=ROOT/'runs/causal-trainability/ppo_pilot'/history
    fresh=hybrids.load(source,'age0.00')
    state,chain=deficit.chain_end(ROOT/'runs/causal-trainability/deficit_B'/history/'actions_chain')
    return fresh,state,dict(fresh=hybrids.checkpoint_row(source,'age0.00'),chain_sha256=chain['end']['sha256'])


def prepared(history,arm,repeat,out,steps,cal):
    fresh,state,source=inputs(history)
    ppo,cfg,seeds=deficit.prepared(fresh['config'],state,None,'actions',5,steps,(history,repeat),
        's06-match-cal' if cal else 's06-match',out,'s06_match',f'heads_{arm}_{repeat}',
        telemetry_every=64,checkpoint_fractions=[0.,1.],snapshot_fractions=[0.,.5,1.])
    wrap(ppo,fresh['theta0'],arm)
    assert not ppo.optimizer.state
    return ppo,cfg,seeds,fresh,source


def behavior_hash(batch):
    return fingerprint({k:batch[k] for k in ('obs','actions','raw_rewards','dones','next_obs','shape')})



def branch(history,arm,repeat,out,steps=STEPS,cal=False,enabled=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ppo,cfg,seeds,fresh,source=prepared(history,arm,repeat,out,steps,cal)
    torch.save(dict(theta0=ppo.theta0),out/'theta0.pt');record_artifact(out,out/'theta0.pt','theta0')
    evaluation=[seed_of('s06-match-eval',history,i) for i in range(30)]
    evaluation_policy=seed_of('s06-match-eval-policy',history,repeat)
    policy=lambda:torch.Generator().manual_seed(evaluation_policy)
    before=start_metrics(ppo,fresh['theta0'])
    if not cal:before['return']=evaluate(ppo.model,cfg,evaluation,policy())
    anchors_before=anchor_hash(ppo.model)
    first={};rollout=ppo.rollout
    def capture():
        batch=rollout()
        if not first:
            first.update(behavior_sha256=behavior_hash(batch),targets_sha256=fingerprint([batch['values'],batch['advantages'],batch['returns']]),
                generator_sha256=fingerprint(ppo.generator.get_state()),
                advantage_std=float(batch['advantages'].std()),return_mean=float(batch['returns'].mean()))
        return batch
    ppo.rollout=capture
    global_rng=torch.get_rng_state().clone();started=time.perf_counter()
    initial=Telemetry(out,'initial',ppo.model.layers(),ppo.optimizer,meta=dict(kind='s06_match_initial'),enabled=enabled)
    initial.step(0,force=True,metrics=before);initial.close()
    ppo.train(telemetry_enabled=enabled)
    seconds=time.perf_counter()-started
    assert torch.equal(global_rng,torch.get_rng_state())
    assert anchor_hash(ppo.model)==anchors_before
    assert all(p.grad is None and p not in ppo.optimizer.state for p in list(ppo.model.anchor.parameters())+list(ppo.model.reset_actor.parameters()))
    final=dict(model=copy.deepcopy(ppo.model.state_dict()),optimizer=copy.deepcopy(ppo.optimizer.state_dict()),generator=ppo.generator.get_state().clone())
    torch.save(final,out/'final.pt');record_artifact(out,out/'final.pt','final_state')
    with torch.random.fork_rng(devices=[]):restored=PPO(cfg,out/'resume_check')
    wrap(restored,fresh['theta0'],arm)
    restored.restore(torch.load(out/'checkpoints/age1.00.pt',map_location='cpu',weights_only=False))
    restored.learn(restored.last_rollout)
    assert fingerprint(final)==fingerprint(dict(model=restored.model.state_dict(),optimizer=restored.optimizer.state_dict(),generator=restored.generator.get_state()))
    result=dict(history=history,arm=arm,repeat=repeat,calibration=cal,steps=ppo.global_step,config=cfg,seeds=dict(seeds,evaluation=evaluation,evaluation_policy=evaluation_policy),
        source=source,identity=identity(),revision=code_revision(),start=before,first=first,seconds=seconds,final_sha256=fingerprint(final),
        anchors_unchanged=True,global_rng_unchanged=True,resume_bit_equal=True,curve=ppo.curve,artifacts_sha256=digest(out/'artifacts.jsonl'))
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
    check_pilot()
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
    check_pilot()
    gate=json.loads((CAL/'summary.json').read_text());assert gate['passed'] and gate['identity']==identity()
    current=dict(identity=identity(),gate_sha256=digest(CAL/'summary.json'),pilot_sha256=digest(PILOT/'summary.json'),histories=list(range(6)),repeats=[0,1],arms=list(ARMS))
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
                subprocess.run([sys.executable,'-m','causal_trainability.behavior_match','branch','--history',history,'--arm',arm,'--repeat',str(rep),'--out',str(attempt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            atomic_json(folder/'result.json',dict(attempt=attempt.name,sha256=digest(attempt/'result.json')))
            print('DONE',history,arm,rep,flush=True)
        except BaseException:
            failed.set();raise
    jobs=[(h,arm,rep) for h in range(6) for rep in (0,1) for arm in ARMS]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:list(pool.map(job,jobs))
    if not (OUT/'STOP').exists():summarize()


def check_pair(rows,with_evaluation):
    a,b=rows['matched'],rows['reset']
    assert a['first']==b['first']
    assert a['start']['reset_logits_equal'] and b['start']['reset_logits_equal']
    assert a['start']['raw_actor_old_equal'] and b['start']['raw_actor_reset_equal']
    if with_evaluation:assert a['start']['return']==b['start']['return']


def pilot():
    root=PILOT
    if root.exists():raise ValueError('Pilot exists')
    root.mkdir(parents=True)
    records=[];gradients=[]
    for h in range(2):
        rows={arm:branch(f'breakout_cnn_s{h}',arm,0,root/f's{h}'/arm,2048,True,True) for arm in ARMS}
        check_pair(rows,False)
        fresh,_,_=inputs(f'breakout_cnn_s{h}')
        grad={arm:first_gradients(root/f's{h}'/arm/'checkpoints/age0.00.pt',fresh['theta0']) for arm in ('reset','matched')}
        assert grad['reset']['order_sha256']==grad['matched']['order_sha256']
        assert all(grad['reset'][name]['sha256']==grad['matched'][name]['sha256'] for name in ('actor','critic'))
        assert any(grad['reset'][name]['sha256']!=grad['matched'][name]['sha256'] for name in ('conv','fc'))
        gradients.append(grad)
        records.append(rows)
    atomic_json(root/'summary.json',dict(passed=True,identity=identity(),kind='technical_only_two_histories',records=records,initial_gradients=gradients))


def first_gradients(path,initial):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);cfg=checkpoint['config']
    with torch.random.fork_rng(devices=[]):ppo=PPO(cfg,path.parent/'unused')
    wrap(ppo,initial,cfg['behavior_arm']);ppo.restore(checkpoint)
    batch=ppo.last_rollout
    idx=torch.randperm(ppo.batch_size,generator=ppo.generator)[:ppo.batch_size//cfg['num_minibatches']]
    logits,value=ppo.model(batch['obs'][idx]);lp=logits.log_softmax(-1)
    selected=lp.gather(1,batch['actions'][idx,None]).squeeze(1)
    ratio=(selected-batch['logprobs'][idx]).exp()
    adv=batch['advantages'][idx];adv=(adv-adv.mean())/(adv.std()+1e-8)
    policy=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-cfg['clip'],1+cfg['clip'])).mean()
    entropy=-(lp.exp()*lp).sum(-1).mean();value_loss=.5*(value-batch['returns'][idx]).square().mean()
    (policy-cfg['ent_coef']*entropy+cfg['vf_coef']*value_loss).backward()
    result=dict(order_sha256=fingerprint(idx))
    for name,params in ppo.model.layers().items():
        grads=[p.grad.detach() for p in params if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        result[name]=dict(sha256=fingerprint(grads),norm=float(torch.sqrt(sum((g.double()**2).sum() for g in grads))) if grads else 0.)
    assert result['anchors']['norm']==0
    return result


def check_pilot():
    data=json.loads((PILOT/'summary.json').read_text())
    assert data['passed'] and data['identity']==identity()
    for h in range(2):
        rows={arm:verified(PILOT/f's{h}'/arm/'result.json') for arm in ARMS}
        assert rows==data['records'][h]
        check_pair(rows,False)


def summarize():
    rows={};hashes={}
    for h in range(6):
        for rep in (0,1):
            group={}
            for arm in ARMS:
                folder=OUT/f'breakout_cnn_s{h}'/f'{arm}_{rep}';wrapper=json.loads((folder/'result.json').read_text());path=folder/wrapper['attempt']/'result.json'
                assert digest(path)==wrapper['sha256'];row=verified(path)
                assert (row['history'],row['arm'],row['repeat'])==(f'breakout_cnn_s{h}',arm,rep)
                assert row['anchors_unchanged']
                rows[h,rep,arm]=row;group[arm]=row;hashes[str(path.relative_to(OUT))]=digest(path)
            check_pair(group,True)
    groups=[[rows[h,r,'matched']['auc']-rows[h,r,'reset']['auc'] for r in (0,1)] for h in range(6)]
    ds=[mean(v) for v in groups];m=mean(ds);se=stdev(ds)/math.sqrt(6)
    ci90=[m-2.0150483733330233*se,m+2.0150483733330233*se]
    ci95=[m-2.570581835636314*se,m+2.570581835636314*se]
    noise=math.sqrt(mean(stdev(v)**2 for v in groups))
    verdict='equivalent' if ci90[0]>-.1 and ci90[1]<.1 else ('matched_better' if ci95[0]>.1 and sum(d>0 for d in ds)>=5 and m>noise else ('matched_worse' if ci95[1]<-.1 and sum(d<0 for d in ds)>=5 and -m>noise else 'uncertain'))
    atomic_json(OUT/'summary.json',dict(kind='conditional_initial_policy_matched_comparison',branches=36,
        primary=dict(mean=m,ci90=ci90,ci95=ci95,within_history_sd=noise,per_history=ds,verdict=verdict),
        identity=identity(),inputs_sha256=hashes,rows=[rows[h,r,arm] for h in range(6) for r in (0,1) for arm in ARMS]))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','calibrate','queue','branch','summarize']);p.add_argument('--workers',type=int,choices=[1,2],default=2);p.add_argument('--history');p.add_argument('--arm',choices=ARMS);p.add_argument('--repeat',type=int,choices=[0,1]);p.add_argument('--out',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    if code_revision()['dirty_causal_trainability']:raise SystemExit('Freeze package before execution')
    if a.mode=='branch':branch(a.history,a.arm,a.repeat,a.out)
    elif a.mode=='pilot':pilot()
    elif a.mode=='calibrate':calibrate()
    elif a.mode=='queue':queue(a.workers)
    else:summarize()


if __name__=='__main__':main()
