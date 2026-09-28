"""Synthetic checks only: no original robot, inference rollout or real model fit."""
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import phase2_refine_qualified_grasp as refine


def row(obj, marker, phase='approach', step=0):
    observation={'observation':np.full(54,obj,np.float32),
                 'achieved_goal':np.array([.1,.2,.3],np.float32),
                 'desired_goal':np.array([.4,.5,.6],np.float32)}
    return {'observation':observation,'next_observation':deepcopy(observation),
            'action':np.array([marker,.2,-.3,-1.],np.float32),
            'teacher_phase':phase,'step':step,'reward':-1.,'terminated':False,'truncated':False}


def data():
    old=[];qualified=[]
    for obj in refine.collector.OBJECTS:
        # Different phase/prefix lengths distinguish stratum sampling from row sampling.
        old.append({'summary':{'successful_physical_lift_and_hold':True,
                              'first_goal_decision':None,'object_idx':obj},
                    'transitions':[row(obj,-.1)]+[row(obj,-.2,'close',i+1) for i in range(19)]})
        for marker,length,eligible in [(0.1,1,True),(0.2,19,True),(0.9,100,False)]:
            qualified.append({'case':{'object_idx':obj},'summary':{'bc_eligible':eligible},
                              'transitions':[row(obj,marker,step=i) for i in range(length)]})
    return {'episodes':old},{'episodes':qualified}


def test_exact_half_native_labels_and_original_goals_without_mutation():
    old,new=data();before=deepcopy((old,new));mix=refine.Mixture(old,new,10101)
    obs,actions,provenance=mix.sample(200)
    assert actions.shape==(400,4)
    assert Counter(origin for origin,obj in provenance)=={'old':200,'qualified':200}
    assert np.all(actions[:200,0]<0) and np.all(actions[200:,0]<.3)
    np.testing.assert_array_equal(obs['desired_goal'],np.tile(np.array([.4,.5,.6],np.float32),(400,1)))
    np.testing.assert_array_equal(obs['achieved_goal'],np.tile(np.array([.1,.2,.3],np.float32),(400,1)))
    for i,(_,obj) in enumerate(provenance):assert np.all(obs['observation'][i]==obj)
    assert refine.collector.serial((old,new))==refine.collector.serial(before)
    assert sum(mix.report()['qualified_prefixes_by_object'].values())==10


def test_object_phase_and_prefix_balancing_not_long_trajectory_weighting():
    old,new=data();mix=refine.Mixture(old,new,9401)
    _,actions,provenance=mix.sample(10000)
    for origin in ('old','qualified'):
        counts=Counter(obj for source,obj in provenance if source==origin)
        assert all(1750<count<2250 for count in counts.values())
    # Each source contains a 1-row versus 19-row group: both receive ~half draws.
    assert 4700<np.count_nonzero(actions[:10000,0]==np.float32(-.1))<5300
    assert 4700<np.count_nonzero(actions[10000:,0]==np.float32(.1))<5300
    assert not np.any(actions[:,0]==np.float32(.9))
    replay=refine.Mixture(old,new,9401).sample(10000)
    np.testing.assert_array_equal(replay[1],actions)
    assert replay[2]==provenance


@pytest.mark.parametrize('missing_source',['old','qualified'])
def test_missing_object_cannot_silently_drop_object_or_fall_back(missing_source):
    old,new=data()
    if missing_source=='old':old['episodes']=old['episodes'][1:]
    else:
        for e in new['episodes']:
            if e['case']['object_idx']==14:e['summary']['bc_eligible']=False
    with pytest.raises(ValueError,match='all five objects'):refine.Mixture(old,new,1)


class SmallActor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mu=torch.nn.Linear(54,4);self.log_std=torch.nn.Linear(54,4)
    def forward(self,obs,deterministic):
        assert deterministic
        return torch.tanh(self.mu(obs['observation']))


def test_bc_step_updates_mean_only_and_keeps_rl_ancestry_and_other_modules():
    torch.manual_seed(100)
    actor=SmallActor()
    model=SimpleNamespace(actor=actor,critic=torch.nn.Linear(58,1),critic_target=torch.nn.Linear(58,1),
                          log_ent_coef=torch.tensor(-3.,requires_grad=True),num_timesteps=12000,_n_updates=2000)
    before=refine.fixed_state(model);mu_before=refine.tensor_digest(actor.mu)
    obs,actions,_=refine.Mixture(*data(),10101).sample(4)
    optimizer=torch.optim.Adam(actor.parameters(),lr=.0003)
    loss=refine.bc_update(actor,optimizer,obs,actions,10.)
    assert np.isfinite(loss) and loss>0
    assert refine.tensor_digest(actor.mu)!=mu_before
    assert refine.fixed_state(model)==before
    assert all(p.grad is None for p in actor.log_std.parameters())
    assert all(p.grad is None for p in model.critic.parameters())


def test_nonfinite_fit_does_not_apply_an_update():
    actor=SmallActor();before=refine.tensor_digest(actor)
    obs,actions,_=refine.Mixture(*data(),10101).sample(1);actions[:]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        refine.bc_update(actor,torch.optim.Adam(actor.parameters(),lr=.0003),obs,actions,10.)
    assert refine.tensor_digest(actor)==before


def comparison(wins=3,losses=0):
    cases=[{'case_id':f'train_{i:03d}'} for i in range(20)]
    rows=[]
    for i,case in enumerate(cases):
        for branch in ('control','candidate'):
            success=True
            if i<wins:success=branch=='candidate'
            elif i<wins+losses:success=branch=='control'
            rows.append({'case_id':case['case_id'],'case':dict(case,comparison_branch=branch),
                         'bc_eligible':success,'prepared_handover':True,
                         'cost_accounting_complete':True,'total_physics_ticks_including_initialization':100+i})
    return rows,cases


def test_gate_uses_net_paired_transport_wins_and_reports_all_costs():
    rows,cases=comparison(4,1);result=refine.paired_result(rows,cases)
    assert result['gate_passed'] and result['net_retention_wins']==3
    assert result['branch_successes']=={'control':16,'candidate':19}
    assert result['total_physics_ticks_including_initialization']==sum(2*(100+i) for i in range(20))
    assert not result['full_task_evaluation'] and not result['full60_launched']
    assert not refine.paired_result(*comparison(3,1))['gate_passed']


def test_new_preparation_failure_fails_even_when_retention_net_passes():
    rows,cases=comparison(5,1)
    next(r for r in rows if r['case_id']==cases[5]['case_id'] and r['case']['comparison_branch']=='candidate')['prepared_handover']=False
    result=refine.paired_result(rows,cases)
    assert result['net_retention_wins']==4 and not result['gate_passed']
    assert result['new_preparation_failures']==[cases[5]['case_id']]


@pytest.mark.parametrize('corruption',['missing','unknown','cost','fatal'])
def test_incomplete_or_uncosted_comparison_never_passes(corruption):
    rows,cases=comparison()
    if corruption=='missing':rows.pop()
    elif corruption=='unknown':rows[0]['case_id']='undeclared'
    elif corruption=='cost':rows[0]['cost_accounting_complete']=False
    else:rows[0]['fatal_error']='incomplete native cost'
    assert not refine.paired_result(rows,cases)['gate_passed']


def test_duplicate_branch_is_rejected_and_empty_is_incomplete():
    rows,cases=comparison()
    with pytest.raises(ValueError,match='Duplicate'):refine.paired_result(rows+[deepcopy(rows[0])],cases)
    result=refine.paired_result([],cases)
    assert not result['complete'] and not result['cost_accounting_complete'] and not result['gate_passed']
