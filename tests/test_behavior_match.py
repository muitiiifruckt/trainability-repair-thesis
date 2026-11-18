import copy

import torch

from causal_trainability.behavior_match import anchor_hash,behavior_hash,first_gradients,wrap
from causal_trainability.ppo_minatar import CONFIG,PPO
from causal_trainability.sprint04_pilot import fingerprint


def learners(tmp_path):
    torch.set_num_threads(1)
    cfg=dict(CONFIG,input_channels=4,conv_channels=2,hidden=8,total_steps=2048,reference_size=16)
    with torch.random.fork_rng(devices=[]):base=PPO(copy.deepcopy(cfg),tmp_path/'base')
    initial=copy.deepcopy(base.model.state_dict())
    with torch.no_grad():
        for p in base.model.parameters():p.add_(.01)
        base.model.actor.weight[0].add_(.3)
    old=copy.deepcopy(base.model.state_dict());rows={}
    for arm in ('neither','reset','matched'):
        with torch.random.fork_rng(devices=[]):ppo=PPO(copy.deepcopy(cfg),tmp_path/arm)
        ppo.model.load_state_dict(old);wrap(ppo,initial,arm);rows[arm]=ppo
    return rows,initial,old


def test_exact_outputs_with_unchanged_raw_parameters(tmp_path):
    rows,initial,old=learners(tmp_path)
    obs=rows['reset'].obs
    with torch.no_grad():
        a,av=rows['reset'].model(obs);b,bv=rows['matched'].model(obs)
    assert torch.equal(a,b) and torch.equal(av,bv)
    assert all(torch.equal(v,rows['matched'].model.core.state_dict()[k]) for k,v in old.items())
    assert torch.equal(rows['reset'].model.core.actor.weight,initial['actor.weight'])
    assert all(not p.requires_grad for p in rows['matched'].model.anchor.parameters())
    assert not rows['matched'].optimizer.state


def test_first_rollout_targets_and_rng_match(tmp_path):
    rows,_,_=learners(tmp_path)
    before=torch.get_rng_state().clone()
    a=rows['reset'].rollout();b=rows['matched'].rollout()
    assert behavior_hash(a)==behavior_hash(b)
    assert fingerprint(a)==fingerprint(b)
    assert torch.equal(rows['reset'].generator.get_state(),rows['matched'].generator.get_state())
    assert torch.equal(before,torch.get_rng_state())


def test_equal_head_gradients_different_body_before_clip(tmp_path):
    rows,initial,_=learners(tmp_path)
    gradients={}
    for arm in ('reset','matched'):
        ppo=rows[arm];ppo.rollout();path=ppo.checkpoint('initial')
        gradients[arm]=first_gradients(path,initial)
    a,b=gradients['reset'],gradients['matched']
    assert a['order_sha256']==b['order_sha256']
    assert a['actor']['sha256']==b['actor']['sha256'] and a['critic']['sha256']==b['critic']['sha256']
    assert any(a[k]['sha256']!=b[k]['sha256'] for k in ('conv','fc'))
    assert a['anchors']['norm']==b['anchors']['norm']==0


def test_logging_resume_and_frozen_anchors(tmp_path):
    rows,initial,_=learners(tmp_path/'on')
    ppo=rows['matched'];anchor=anchor_hash(ppo.model)
    ppo.train()
    final=dict(model=ppo.model.state_dict(),optimizer=ppo.optimizer.state_dict(),generator=ppo.generator.get_state())
    assert anchor_hash(ppo.model)==anchor
    assert all(p.grad is None and p not in ppo.optimizer.state for p in ppo.model.anchor.parameters())
    off,_,_=learners(tmp_path/'off');off=off['matched'];off.train(telemetry_enabled=False)
    assert fingerprint(final)==fingerprint(dict(model=off.model.state_dict(),optimizer=off.optimizer.state_dict(),generator=off.generator.get_state()))
    checkpoint=torch.load(ppo.out/'checkpoints/age1.00.pt',map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=[]):restored=PPO(copy.deepcopy(ppo.cfg),tmp_path/'restored')
    wrap(restored,initial,'matched');restored.restore(checkpoint);restored.learn(restored.last_rollout)
    assert fingerprint(final)==fingerprint(dict(model=restored.model.state_dict(),optimizer=restored.optimizer.state_dict(),generator=restored.generator.get_state()))
