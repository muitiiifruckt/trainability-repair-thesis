import copy

import pytest
import torch

from causal_trainability.head_reset_factorial import ARMS,behavior_hash,factor_effect,reset_heads
from causal_trainability.ppo_minatar import CONFIG,PPO


def learners(tmp_path):
    torch.set_num_threads(1)
    cfg=dict(CONFIG,input_channels=4,conv_channels=2,hidden=8,total_steps=1024)
    with torch.random.fork_rng(devices=[]):base=PPO(cfg,tmp_path/'base')
    initial=copy.deepcopy(base.model.state_dict())
    with torch.no_grad():
        for p in base.model.parameters():p.add_(.1)
    old=copy.deepcopy(base.model.state_dict())
    result={}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):ppo=PPO(cfg,tmp_path/arm)
        ppo.model.load_state_dict(old);reset_heads(ppo.model,initial,arm);result[arm]=ppo
    return result,initial,old


def test_resets_only_chosen_head_parameters(tmp_path):
    rows,initial,old=learners(tmp_path)
    for arm,ppo in rows.items():
        assert not ppo.optimizer.state
        for name,value in ppo.model.state_dict().items():
            reset=(name.startswith('actor.') and arm in ('actor','both')) or (name.startswith('critic.') and arm in ('critic','both'))
            assert torch.equal(value,(initial if reset else old)[name])


def test_critic_does_not_change_first_behavior_but_changes_targets(tmp_path):
    rows,_,_=learners(tmp_path)
    before=torch.get_rng_state().clone()
    batches={k:p.rollout() for k,p in rows.items()}
    assert behavior_hash(batches['neither'])==behavior_hash(batches['critic'])
    assert behavior_hash(batches['actor'])==behavior_hash(batches['both'])
    assert not torch.equal(batches['neither']['values'],batches['critic']['values'])
    assert not torch.equal(batches['neither']['advantages'],batches['critic']['advantages'])
    assert torch.equal(before,torch.get_rng_state())


def test_factorial_contrasts_include_interaction():
    # y=10+2*a-3*c+4*a*c; averaging the other factor gives 4 and -1.
    rows=dict(neither=10.,actor=12.,critic=7.,both=13.)
    assert factor_effect(rows,'actor')==4
    assert factor_effect(rows,'critic')==-1
    assert factor_effect(rows,'interaction')==4
    with pytest.raises(ValueError):factor_effect(rows,'other')
