"""Collection bookkeeping and physical-preparation boundary invariants."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.phase2_carry_demos import counted_step,recovery_targets,prepare_recovery,COST_FIELDS


def test_targets_keep_both_directions_and_skip_outer_boundaries():
    candidates=recovery_targets({'maze_entrance_cell':[4,6]},6)
    assert [r['requested_offset_y_m'] for r in candidates]==[.75,-.75,1.5,-1.5,2.5,-2.5]
    assert all(r['target_xy'][0]==1.5 for r in candidates)
    boundary=recovery_targets({'maze_entrance_cell':[1,6]},6)
    assert sum(r['geometry_skip'] is None for r in boundary)==3
    assert all(-4.35<=r['target_xy'][1]<=4.35 for r in boundary if r['geometry_skip'] is None)


def test_interrupted_native_step_is_still_counted():
    native=SimpleNamespace(step_count=20,physics_steps=100)
    class Broken:
        def step(self,action):
            native.step_count+=2;native.physics_steps+=10
            raise RuntimeError('physical controller interruption')
    env=Broken();env.native=native;cost={key:0 for key in COST_FIELDS}
    with pytest.raises(RuntimeError,match='interruption'):counted_step(env,[0,0],cost,True)
    assert cost['preparation_source_action_calls']==2 and cost['preparation_physics_ticks']==10
    assert cost['source_action_calls']==0


def test_prepare_moves_physically_without_changing_goal_or_restoring_after_start(tmp_path,monkeypatch):
    import scripts.phase2_carry_demos as collector
    class Fake:
        goal_scale=20.;delta_scale=.075;horizon=900;_retention_failed=False
        def __init__(self):
            self.xy=np.array([1.5,1.5]);self.goal=np.array([17.5,4.5])/20
            self.native=SimpleNamespace(step_count=0,physics_steps=0,cube_positions=[],table_id=1,
                contacts=lambda:{'bilateral_contact':True},p=SimpleNamespace(getAABB=lambda _:None))
            self.controller=SimpleNamespace(base_position=lambda:self.xy.copy())
            self.resets=0;self._episode_steps=0
        def obs(self):return {'achieved_goal':self.xy.astype(np.float32)/20,'desired_goal':self.goal.copy()}
        def reset(self,options):
            self.resets+=1;return self.obs(),{'evaluation_scope':'full_carry'}
        def step(self,action):
            self.xy+=np.asarray(action)*self.delta_scale
            self.native.step_count+=5;self.native.physics_steps+=25;self._episode_steps+=1
            return self.obs(),-1.,False,False,{'bilateral_contact':True}
    monkeypatch.setattr(collector,'maze_astar',lambda walls,start,goal,**kw:[start,np.array(goal)])
    env=Fake();cost={key:0 for key in COST_FIELDS};original=env.goal.copy()
    observation,report=prepare_recovery(env,[1.5,2.25],cost,tmp_path/'STOP')
    assert env.resets==1 and env._episode_steps==0 and env._episode_horizon==900
    assert env.native.step_count==cost['preparation_source_action_calls']>0
    assert report['preparation_decisions']*5==env.native.step_count
    assert report['start_error_m']<.075
    np.testing.assert_array_equal(observation['desired_goal'],original)
