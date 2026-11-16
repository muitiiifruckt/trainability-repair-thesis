"""Sprint 05: fixed-batch PPO gains; scientific protocol is frozen separately."""
from __future__ import annotations

import argparse
import queue
import threading
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean, median, stdev

import torch

from causal_trainability import deficit, hybrids
from causal_trainability.ppo_minatar import PPO, VecMinAtar, reference_batch, representation_metrics, seed_of
from causal_trainability.sprint03_live import atomic_json, digest
from causal_trainability.sprint04_pilot import fingerprint
from causal_trainability.telemetry import ROOT, Telemetry, TelemetryStore, code_revision, config_hash, record_artifact

CELLS = ('fresh_full', 'fresh_head_only', 'aged_full', 'aged_head_only')
PROTOCOLS = ('notes/sprint04_common_batch_prereg.md', 'notes/sprint05_execution_v2.md')
CAL = ROOT/'runs/causal-trainability/sprint05_calibration_v2_20261116'
SCI = ROOT/'runs/causal-trainability/sprint05_common_v2_20261116'


class AsyncLogger:
    """Owns independent copies; diagnostics never access the learner tensors."""
    def __init__(self,out,cfg,reference):
        self.out,self.cfg,self.reference=out,cfg,reference
        self.jobs=queue.Queue()
        self.error=None
        self.thread=threading.Thread(target=self.work,daemon=True)
        self.thread.start()

    def push(self,k,model,optimizer):
        copied_model,copied_optimizer=copy.deepcopy((model,optimizer))
        for original,copied in zip(model.parameters(),copied_model.parameters()):
            copied.grad=original.grad.detach().clone() if original.grad is not None else None
        self.jobs.put((k,copied_model,copied_optimizer))

    def work(self):
        store=None
        try:
            store=TelemetryStore(self.out,'cell')
            telemetry=None
            while True:
                job=self.jobs.get()
                if job is None:break
                k,model,optimizer=job
                if telemetry is None:
                    telemetry=Telemetry(self.out,'cell',model.layers(),optimizer,
                        meta=dict(kind='s05_common_batch',config_hash=config_hash(self.cfg)),
                        every=8,store=store)
                else:
                    telemetry.layers=model.layers()
                    telemetry.optimizer=optimizer
                telemetry.step(k,force=True,layer_extra=representation_metrics(model,self.reference))
                telemetry.snapshot(k,model,optimizer)
        except BaseException as error:
            self.error=error
        finally:
            if store is not None:
                try:store.close()
                except BaseException as error:self.error=error

    def close(self):
        self.jobs.put(None)
        self.thread.join()
        if self.error is not None:raise RuntimeError('Background telemetry failed') from self.error


def identity():
    return dict(files=code_revision()['files'], protocols={p:digest(ROOT/p) for p in PROTOCOLS})


def sources(history):
    source = ROOT/'runs/causal-trainability/ppo_pilot'/history
    fresh = hybrids.load(source,'age0.00')
    aged, chain = deficit.chain_end(ROOT/'runs/causal-trainability/deficit_B'/history/'actions_chain')
    return fresh, aged, dict(fresh=hybrids.checkpoint_row(source,'age0.00'),chain_sha256=chain['end']['sha256'])


def prepare(history, repeat, namespace, fresh, out):
    cfg = dict(fresh['config'], total_steps=1024, num_envs=8, num_steps=128, sticky_action_prob=.25,
        shift=deficit.shift_spec(fresh['config'],'actions',4), checkpoint_fractions=[], snapshot_fractions=[])
    seeds = {k:seed_of(namespace,history,repeat,k) for k in ('env','ppo')}
    with torch.random.fork_rng(devices=[]): ppo = PPO(cfg,out)
    ppo.model.load_state_dict(fresh['theta0'])
    zero_heads(ppo.model)
    ppo.envs = VecMinAtar(cfg,seeds['env']); ppo.obs = torch.tensor(ppo.envs.observations())
    ppo.generator.manual_seed(seeds['ppo'])
    batch = ppo.rollout()
    return cfg, batch, ppo.generator.get_state().clone(), seeds, reference_batch(cfg)


def zero_heads(model):
    with torch.no_grad():
        for head in (model.actor,model.critic): head.weight.zero_(); head.bias.zero_()


def objective(model, batch, cfg):
    """Fixed evaluation slices; independent of minibatch permutations used for training."""
    parts = {k:[] for k in ('policy','value','entropy')}
    with torch.no_grad():
        for start in range(0,len(batch['actions']),256):
            sl = slice(start,start+256)
            logits, values = model(batch['obs'][sl]); lp = logits.log_softmax(-1)
            selected = lp.gather(1,batch['actions'][sl,None]).squeeze(1)
            ratio = (selected-batch['logprobs'][sl]).exp()
            adv = batch['advantages'][sl]; adv = (adv-adv.mean())/(adv.std()+1e-8)
            parts['policy'].append(float(torch.maximum(-adv*ratio,-adv*ratio.clamp(1-cfg['clip'],1+cfg['clip'])).mean()))
            parts['value'].append(float(.5*(values-batch['returns'][sl]).square().mean()))
            parts['entropy'].append(float(-(lp.exp()*lp).sum(-1).mean()))
    result={k:mean(v) for k,v in parts.items()}
    result['total']=result['policy']-cfg['ent_coef']*result['entropy']+cfg['vf_coef']*result['value']
    return result


def valid_scale(before, batch):
    values = [before['total'],float(batch['advantages'].var())]
    return all(math.isfinite(v) for v in values) and abs(values[0])>1e-8 and values[1]>1e-8


def order_for(cfg, generator_state):
    g=torch.Generator();g.set_state(generator_state)
    order=[torch.randperm(1024,generator=g) for _ in range(cfg['update_epochs'])]
    return [v[i:i+256] for v in order for i in range(0,1024,256)],g.get_state()


def cell(cfg,batch,generator_state,reference,body,mode,out,enabled=True):
    out.mkdir(parents=True,exist_ok=False)
    batch_hash=fingerprint(batch); global_rng=torch.get_rng_state().clone()
    started=time.perf_counter()
    with torch.random.fork_rng(devices=[]): ppo=PPO(cfg,out)
    ppo.model.load_state_dict(body);zero_heads(ppo.model)
    for name,param in ppo.model.named_parameters():
        if not name.startswith(('actor.','critic.')):param.requires_grad_(mode=='full')
    ppo.generator.set_state(generator_state)
    initial_body={k:v.clone() for k,v in ppo.model.state_dict().items() if not k.startswith(('actor.','critic.'))}
    before=objective(ppo.model,batch,cfg)
    if not all(math.isfinite(v) for v in before.values()):raise ValueError('Nonfinite initial objective')
    order,expected_rng=order_for(cfg,generator_state)
    logger=AsyncLogger(out,cfg,reference) if enabled else None
    def log(k):
        if logger:logger.push(k,ppo.model,ppo.optimizer)
    log(0)
    gradients=[]; update=[0]
    def pre(opt,args,kwargs):
        if len(gradients)<2:
            gradients.append({k:float(torch.sqrt(sum((p.grad.double()**2).sum() for p in ps if p.grad is not None)))
                if any(p.grad is not None for p in ps) else 0. for k,ps in ppo.model.layers().items()})
    def post(opt,args,kwargs):
        update[0]+=1
        if update[0] in (8,16):log(update[0])
    h1=ppo.optimizer.register_step_pre_hook(pre);h2=ppo.optimizer.register_step_post_hook(post)
    ppo.learn(batch);h1.remove();h2.remove()
    assert update[0]==16 and torch.equal(ppo.generator.get_state(),expected_rng)
    after=objective(ppo.model,batch,cfg)
    if logger:logger.close()
    final=dict(model=copy.deepcopy(ppo.model.state_dict()),optimizer=copy.deepcopy(ppo.optimizer.state_dict()),generator=ppo.generator.get_state().clone())
    torch.save(final,out/'final.pt');record_artifact(out,out/'final.pt','final_state')
    elapsed=time.perf_counter()-started
    if mode=='head_only':
        assert all(torch.equal(v,ppo.model.state_dict()[k]) for k,v in initial_body.items())
        assert all(p.grad is None and p not in ppo.optimizer.state for name,p in ppo.model.named_parameters() if not name.startswith(('actor.','critic.')))
    assert torch.equal(global_rng,torch.get_rng_state()) and fingerprint(batch)==batch_hash
    assert gradients[0]['conv']==0 and gradients[0]['fc']==0
    if not all(math.isfinite(v) for v in after.values()):raise ValueError('Nonfinite final objective')
    gain=before['total']-after['total'];valid=valid_scale(before,batch)
    result=dict(before=before,after=after,gain=gain,scale=abs(before['total']),valid=valid,
        normalized_gain=gain/abs(before['total']) if valid else None,
        seconds=elapsed,first_gradients=gradients,order_sha256=fingerprint(order),
        final_sha256=fingerprint(final),batch_sha256=batch_hash,global_rng_unchanged=True,
        artifacts_sha256=digest(out/'artifacts.jsonl'))
    atomic_json(out/'result.json',result)
    return result


def verified(path):
    row=json.loads(path.read_text(encoding='utf-8'));folder=path.parent
    assert digest(folder/'artifacts.jsonl')==row['artifacts_sha256']
    for line in (folder/'artifacts.jsonl').read_text().splitlines():
        a=json.loads(line);assert digest(folder/a['path'])==a['sha256']
    assert row['final_sha256']==fingerprint(torch.load(folder/'final.pt',weights_only=False,map_location='cpu'))
    assert abs(row['gain']-(row['before']['total']-row['after']['total']))<1e-12
    assert row['scale']==abs(row['before']['total'])
    if row['valid']:assert abs(row['normalized_gain']-row['gain']/row['scale'])<1e-12
    return row


def group(history,repeat,namespace,out):
    fresh,aged,source_meta=sources(history)
    cfg,batch,rng,seeds,reference=prepare(history,repeat,namespace,fresh,out)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(batch=batch,generator=rng,reference=reference,theta0=fresh['theta0']),out/'inputs.pt')
    record_artifact(out,out/'inputs.pt','common_batch_theta0_reference')
    atomic_json(out/'manifest.json',dict(history=history,repeat=repeat,config=cfg,config_hash=config_hash(cfg),seeds=seeds,
        sources=source_meta,batch_sha256=fingerprint(batch),targets_sha256=fingerprint([batch['advantages'],batch['returns']]),identity=identity(),revision=code_revision()))
    return cfg,batch,rng,reference,dict(fresh=fresh['theta0'],aged=aged['model'])


def calibration():
    if CAL.exists():raise ValueError('Calibration output exists')
    cfg,batch,rng,ref,bodies=group('breakout_cnn_s0',0,'s05-calibration-v2',CAL)
    results={}
    for name in CELLS:
        body,mode=name.split('_',1)
        for flag in (False,True):cell(cfg,batch,rng,ref,bodies[body],mode,CAL/name/f'warm_{flag}',flag)
        pairs=[]
        for i in range(3):
            rows={flag:cell(cfg,batch,rng,ref,bodies[body],mode,CAL/name/f'pair{i}_{flag}',flag) for flag in ((True,False) if i%2==0 else (False,True))}
            assert rows[True]['final_sha256']==rows[False]['final_sha256']
            pairs.append(dict(on=rows[True]['seconds'],off=rows[False]['seconds'],overhead=rows[True]['seconds']/rows[False]['seconds']-1,bit_equal=True))
        results[name]=dict(pairs=pairs,mean_overhead=mean(r['overhead'] for r in pairs),median_overhead=median(r['overhead'] for r in pairs))
        print(name,results[name]['mean_overhead'],flush=True)
    atomic_json(CAL/'summary.json',dict(kind='technical_only',identity=identity(),revision=code_revision(),cells=results,
        passed=all(r['mean_overhead']<=.05 and r['median_overhead']<=.05 for r in results.values())))


def science():
    gate=json.loads((CAL/'summary.json').read_text());assert gate['passed'] and gate['identity']==identity()
    SCI.mkdir(parents=True,exist_ok=True)
    current=dict(identity=identity(),gate_sha256=digest(CAL/'summary.json'),histories=list(range(6)),repeats=list(range(3)))
    mp=SCI/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==current
    else:atomic_json(mp,current)
    for h in range(6):
        for repeat in range(3):
            if (SCI/'STOP').exists():return
            folder=SCI/f'breakout_cnn_s{h}'/f'repeat{repeat}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():continue
            attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}'
            cfg,batch,rng,ref,bodies=group(f'breakout_cnn_s{h}',repeat,'s04-common',attempt)
            rows={}
            for name in CELLS:
                body,mode=name.split('_',1);rows[name]=cell(cfg,batch,rng,ref,bodies[body],mode,attempt/name)
            assert max(r['before']['total'] for r in rows.values())-min(r['before']['total'] for r in rows.values())<=1e-7
            assert len({r['batch_sha256'] for r in rows.values()})==1 and len({r['order_sha256'] for r in rows.values()})==1
            atomic_json(folder/'result.json',dict(attempt=attempt.name,hashes={n:digest(attempt/n/'result.json') for n in CELLS}))
            print(h,repeat,'done',flush=True)
    summarize()


def summarize():
    records=[];groups=[];extra=[];informative=True;hashes={}
    for h in range(6):
        ds=[];es=[]
        for repeat in range(3):
            folder=SCI/f'breakout_cnn_s{h}'/f'repeat{repeat}';wrapper=json.loads((folder/'result.json').read_text())
            attempt=folder/wrapper['attempt'];rows={}
            manifest=json.loads((attempt/'manifest.json').read_text())
            assert manifest['identity']==identity()
            assert manifest['history']==f'breakout_cnn_s{h}' and manifest['repeat']==repeat
            for line in (attempt/'artifacts.jsonl').read_text().splitlines():
                artifact=json.loads(line);assert digest(attempt/artifact['path'])==artifact['sha256']
            inputs=torch.load(attempt/'inputs.pt',weights_only=False,map_location='cpu')
            assert manifest['batch_sha256']==fingerprint(inputs['batch'])
            assert manifest['targets_sha256']==fingerprint([inputs['batch']['advantages'],inputs['batch']['returns']])
            expected_order,_=order_for(manifest['config'],inputs['generator'])
            for name in CELLS:
                path=attempt/name/'result.json';assert digest(path)==wrapper['hashes'][name]
                rows[name]=verified(path);hashes[str(path.relative_to(SCI))]=digest(path)
                assert rows[name]['batch_sha256']==manifest['batch_sha256']
                assert rows[name]['order_sha256']==fingerprint(expected_order)
                assert rows[name]['valid']==valid_scale(rows[name]['before'],inputs['batch'])
            assert max(r['before']['total'] for r in rows.values())-min(r['before']['total'] for r in rows.values())<=1e-7
            valid=all(r['valid'] for r in rows.values());informative &= valid
            for name,r in rows.items():
                if r['valid']:assert abs(r['normalized_gain']-r['gain']/r['scale'])<1e-12
            records.append(dict(history=h,repeat=repeat,cells=rows))
            if valid:
                g={k:v['normalized_gain'] for k,v in rows.items()}
                ds.append(g['fresh_full']-g['aged_full'])
                es.append((g['fresh_full']-g['fresh_head_only'])-(g['aged_full']-g['aged_head_only']))
        groups.append(ds);extra.append(es)
    result=dict(kind='conditional_on_previously_inspected_histories',records=records,inputs_sha256=hashes,identity=identity(),verdict='uninformative')
    if informative:
        ds=[mean(v) for v in groups];m=mean(ds);half=2.570581835636314*stdev(ds)/math.sqrt(6)
        ci=[m-half,m+half];noise=math.sqrt(mean(stdev(v)**2 for v in groups))
        verdict='aged_smaller_gain' if ci[0]>.01 and sum(v>0 for v in ds)>=5 and m>noise else ('practical_deficit_not_shown' if ci[1]<.01 else 'uncertain')
        result.update(verdict=verdict,primary=dict(mean=m,ci95=ci,per_history=ds,within_history_sd=noise),exploratory_head_adjusted=[mean(v) for v in extra])
    atomic_json(SCI/'summary.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','science','summarize']);args=p.parse_args()
    torch.set_num_threads(1)
    if code_revision()['dirty_causal_trainability']:raise SystemExit('Freeze clean package before execution')
    {'calibrate':calibration,'science':science,'summarize':summarize}[args.mode]()


if __name__=='__main__':main()
