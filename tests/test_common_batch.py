import copy
import math

import pytest
import torch

from causal_trainability.common_batch import cell, objective, valid_scale, verified, zero_heads
from causal_trainability.ppo_minatar import CONFIG, PPO
from causal_trainability.sprint04_pilot import fingerprint


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    cfg=dict(CONFIG,input_channels=4,conv_channels=2,hidden=8,total_steps=1024,reference_size=16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        ppo=PPO(cfg,tmp_path/'base')
        obs=torch.randint(0,2,(1024,4,10,10),dtype=torch.uint8)
        batch=dict(obs=obs,actions=torch.randint(0,6,(1024,)),logprobs=torch.full((1024,),-math.log(6)),
            advantages=torch.randn(1024),returns=torch.randn(1024),values=torch.zeros(1024))
    return cfg,copy.deepcopy(ppo.model.state_dict()),batch,torch.Generator().manual_seed(82).get_state()


def test_original_learn_and_logging_are_identical(fixture,tmp_path):
    cfg,body,batch,rng=fixture
    before_batch=fingerprint(batch);before_rng=torch.get_rng_state().clone()
    off=cell(cfg,batch,rng,batch['obs'][:16],body,'full',tmp_path/'off',False)
    on=cell(cfg,batch,rng,batch['obs'][:16],body,'full',tmp_path/'on',True)
    with torch.random.fork_rng(devices=[]):ppo=PPO(cfg,tmp_path/'original')
    ppo.model.load_state_dict(body);zero_heads(ppo.model);ppo.generator.set_state(rng)
    ppo.learn(batch)
    assert fingerprint(dict(model=ppo.model.state_dict(),optimizer=ppo.optimizer.state_dict(),generator=ppo.generator.get_state()))==on['final_sha256']==off['final_sha256']
    assert before_batch==fingerprint(batch) and torch.equal(before_rng,torch.get_rng_state())
    assert on['first_gradients'][0]['fc']==0
    assert verified(tmp_path/'on/result.json')['final_sha256']==on['final_sha256']


def test_head_only_freezes_body_and_adam(fixture,tmp_path):
    cfg,body,batch,rng=fixture
    result=cell(cfg,batch,rng,batch['obs'][:16],body,'head_only',tmp_path/'frozen',True)
    final=torch.load(tmp_path/'frozen/final.pt',weights_only=False)
    assert all(torch.equal(v,final['model'][k]) for k,v in body.items() if not k.startswith(('actor.','critic.')))
    assert all(row['fc']==0 and row['conv']==0 for row in result['first_gradients'])
    assert len(final['optimizer']['state'])==4


def test_zero_start_and_objective_parts(fixture,tmp_path):
    cfg,body,batch,rng=fixture
    with torch.random.fork_rng(devices=[]):ppo=PPO(cfg,tmp_path/'eval')
    ppo.model.load_state_dict(body);zero_heads(ppo.model)
    parts=objective(ppo.model,batch,cfg)
    assert abs(parts['policy'])<1e-6
    assert parts['entropy']==pytest.approx(math.log(6),abs=1e-6)
    assert parts['value']==pytest.approx(float(.5*batch['returns'].square().mean()),abs=1e-6)
    assert parts['total']==pytest.approx(parts['policy']-.01*parts['entropy']+.5*parts['value'])
    assert valid_scale(parts,batch)
    assert not valid_scale(dict(total=0.),batch)
    assert not valid_scale(dict(total=float('nan')),batch)
    assert not valid_scale(parts,dict(batch,advantages=torch.zeros(1024)))


def test_artifact_corruption_is_detected(fixture,tmp_path):
    cfg,body,batch,rng=fixture
    cell(cfg,batch,rng,batch['obs'][:16],body,'full',tmp_path/'run',False)
    with (tmp_path/'run/final.pt').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(AssertionError):verified(tmp_path/'run/result.json')
