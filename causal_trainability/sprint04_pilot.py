"""Technical matching pilot, notes/sprint04_prereg.md; no scientific return test."""
import copy
import hashlib
import json
import time
from pathlib import Path
from statistics import mean, median

import torch

from causal_trainability import deficit, hybrids
from causal_trainability.ppo_minatar import PPO, reference_batch
from causal_trainability.sprint03_live import atomic_json, digest
from causal_trainability.telemetry import ROOT, Telemetry, code_revision, config_hash, record_artifact
from scripts.bench_sprint03_live import same


def fingerprint(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((str(x.dtype), tuple(x.shape))).encode()); h.update(x.contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def run(history, arm, enabled, out, iterations=64):
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'runs/causal-trainability/ppo_pilot' / history
    fresh = hybrids.load(source, 'age0.00')
    state, chain = deficit.chain_end(ROOT / 'runs/causal-trainability/deficit_B' / history / 'actions_chain')
    ppo, cfg, seeds = deficit.prepared(fresh['config'], state, None, 'actions', 4, iterations*1024,
        (history,), 's04-tech', out, 's04_technical', 's04', sticky_action_prob=.25,
        telemetry_every=16, checkpoint_fractions=[1.0], snapshot_fractions=[0., .5, 1.])
    if arm == 'fresh': ppo.model.load_state_dict(fresh['theta0'])
    with torch.no_grad():
        for layer in (ppo.model.actor, ppo.model.critic):
            layer.weight.zero_(); layer.bias.zero_()
    reference = reference_batch(cfg)
    with torch.no_grad():
        logits, values = ppo.model(reference)
    assert torch.count_nonzero(logits) == 0 and torch.count_nonzero(values) == 0
    torch.save({'theta0': ppo.theta0, 'reference': reference}, out / 'initial.pt')
    record_artifact(out, out / 'initial.pt', 'theta0_reference')
    meta = dict(config=cfg, config_sha256=config_hash(cfg), seeds=seeds,
        chain_sha256=chain['end']['sha256'], source=hybrids.checkpoint_row(source,'age0.00'),
        protocol_sha256=digest(ROOT/'notes/sprint04_prereg.md'), revision=code_revision())
    atomic_json(out/'manifest.json', meta)
    first = {}
    original_rollout = ppo.rollout
    def rollout():
        result = original_rollout()
        if not first: first.update(batch=fingerprint(result), rng=fingerprint(ppo.generator.get_state()))
        return result
    ppo.rollout = rollout
    gradients = []
    def before_step(opt, args, kwargs):
        if len(gradients) < 2:
            gradients.append({k: float(torch.sqrt(sum((p.grad.double()**2).sum() for p in ps if p.grad is not None)))
                              for k, ps in ppo.model.layers().items()})
    hook = ppo.optimizer.register_step_pre_hook(before_step)
    global_rng = torch.get_rng_state().clone()
    started = time.perf_counter()
    initial_log = Telemetry(out, 'initial', ppo.model.layers(), ppo.optimizer, meta={'kind':'s04_technical_initial'}, enabled=enabled)
    initial_log.step(0, force=True); initial_log.close()
    ppo.train(telemetry_enabled=enabled)
    elapsed = time.perf_counter()-started
    hook.remove()
    assert torch.equal(global_rng, torch.get_rng_state())
    assert gradients[0]['conv'] == 0 and gradients[0]['fc'] == 0
    checkpoint = torch.load(out/'checkpoints/age1.00.pt', weights_only=False, map_location='cpu')
    with torch.random.fork_rng(devices=[]): restored = PPO(cfg, out/'resume_check')
    restored.restore(checkpoint); restored.learn(restored.last_rollout)
    equal_resume = same(ppo.model.state_dict(),restored.model.state_dict()) and same(ppo.optimizer.state_dict(), restored.optimizer.state_dict()) and torch.equal(ppo.generator.get_state(),restored.generator.get_state())
    assert equal_resume
    result = dict(seconds=elapsed, first_rollout=first, first_gradients=gradients,
        initial_outputs_zero=True, global_rng_unchanged=True, resume_bit_equal=equal_resume,
        final_sha256=fingerprint([ppo.model.state_dict(),ppo.optimizer.state_dict(),ppo.generator.get_state()]),
        artifacts_sha256=digest(out/'artifacts.jsonl'))
    atomic_json(out/'result.json',result)
    return result


def main():
    torch.set_num_threads(1)
    if code_revision()['dirty_causal_trainability']: raise SystemExit('Commit frozen code before pilot')
    root=ROOT/'runs/causal-trainability/sprint04_pilot_20261113'
    if root.exists(): raise SystemExit('Output exists; preserve attempts, use new version after interruption')
    root.mkdir(parents=True)
    results={}
    for h in ('breakout_cnn_s0','breakout_cnn_s1'):
        for arm in ('fresh','aged'):
            if h.endswith('s0'):
                for flag in (False,True): run(h,arm,flag,root/h/arm/f'warm_{flag}',4)
            pairs=[]
            for i in range(3 if h.endswith('s0') else 1):
                order=(True,False) if i%2==0 else (False,True)
                rows={flag:run(h,arm,flag,root/h/arm/f'pair{i}_{flag}') for flag in order}
                assert rows[True]['final_sha256']==rows[False]['final_sha256']
                pairs.append(dict(on=rows[True],off=rows[False],overhead=rows[True]['seconds']/rows[False]['seconds']-1))
                print(h,arm,i,'done',flush=True)
            results[h+'_'+arm]=dict(pairs=pairs,mean_overhead=mean(p['overhead'] for p in pairs),median_overhead=median(p['overhead'] for p in pairs))
        a=results[h+'_fresh']['pairs'][0]['on']['first_rollout']
        b=results[h+'_aged']['pairs'][0]['on']['first_rollout']
        assert a==b, 'Initial rollouts differ across bodies'
    passed=all(r['mean_overhead']<=.05 and r['median_overhead']<=.05 for k,r in results.items() if 's0_' in k)
    atomic_json(root/'summary.json',dict(kind='technical_only_two_development_histories',passed=passed,cells=results,revision=code_revision(),protocol_sha256=digest(ROOT/'notes/sprint04_prereg.md')))
    print('TECHNICAL_GATE',passed,flush=True)


if __name__=='__main__': main()
