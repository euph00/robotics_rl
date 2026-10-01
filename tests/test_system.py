"""Full-task inference fails honestly when its first learned skill fails."""
import numpy as np
import pytest

from pnr.env import PickNavReachEnv
from pnr.system import LearnedSystem


class ConstantTestPolicy:
    def __init__(self,dimensions):
        self.dimensions=dimensions
        self.calls=0

    def predict(self,observation,deterministic=True):
        self.calls+=1
        return np.zeros(self.dimensions,dtype=np.float32),None


def test_failed_grasp_stops_system_without_resets_or_teacher_descents():
    with PickNavReachEnv(seed=42,object_idx=14) as native:
        # This is a plumbing check with explicit test policies, not a trained
        # policy evaluation or evidence of learned success.
        def forbidden_reset(*args,**kwargs):
            raise AssertionError('No full-system inter-skill resets are allowed')
        native.reset=forbidden_reset
        initial_position=native.observe()['object_pos'].copy()
        original_step=native.step
        policies={name:ConstantTestPolicy(dim) for name,dim in [('grasp',4),('carry',2),('reach',3)]}
        seen=[]
        system=LearnedSystem(native,policies,grasp_horizon=2,
            on_step=lambda system,obs,info:seen.append(native.step_count))
        result=system.run()
        assert not result['success']
        assert result['failed_stage']=='grasp'
        assert result['stages']['grasp']['reason']=='decision_horizon'
        assert policies['grasp'].calls==2
        assert policies['carry'].calls==policies['reach'].calls==0
        assert result['source_calls_by_controller']['learned']==6
        assert len(seen)==native.step_count==sum(result['source_calls_by_controller'].values())
        assert np.linalg.norm(native.observe()['object_pos']-initial_position)<.005
        assert native.step==original_step
        assert native.model.neq==0
        with pytest.raises(ValueError,match='fresh original-start'):
            system.run()


class AnalyticalGraspTestPolicy:
    """Explicit analytical TEST policy; never a learned checkpoint or result."""
    def __init__(self,native):
        self.native=native
        self.controller=None
        self.phase='descend'
        self.close_decisions=0

    def predict(self,observation,deterministic=True):
        ee,_=self.controller.planner.ee_pose()
        if self.phase=='descend':
            target=self.controller.grasp.position
            if np.linalg.norm(target-ee)<.0025:
                self.phase='close'
            else:
                return np.r_[np.clip((target-ee)/.012,-1,1),1.].astype(np.float32),None
        if self.phase=='close':
            self.close_decisions+=1
            if self.close_decisions>=30:
                self.phase='lift'
            return np.array([0,0,0,-1],dtype=np.float32),None
        error=observation['desired_goal']-observation['achieved_goal']
        return np.r_[np.clip(error/.012,-1,1),-1.].astype(np.float32),None


class AnalyticalCarryTestPolicy:
    """A* route TEST oracle, explicitly excluded from any learned-policy claim."""
    def __init__(self,native):
        self.native=native
        self.route=None
        self.index=1

    def predict(self,observation,deterministic=True):
        from pnr.planning import maze_astar
        base=observation['achieved_goal']*20
        if self.route is None:
            self.route=maze_astar(self.native.cube_positions,base,self.native.goal_pos[:2],
                footprint_radius=.36,table_bounds=self.native.table_aabb())
        while self.index<len(self.route)-1 and np.linalg.norm(base-self.route[self.index])<.012:
            self.index+=1
        action=(self.route[self.index]-base)/.045
        action/=max(1.,np.linalg.norm(action))
        return action.astype(np.float32),None


class AnalyticalReachTestPolicy:
    """Direct native XYZ translation TEST controller using bound-frame goals."""
    def predict(self,observation,deterministic=True):
        error=observation['desired_goal']-observation['achieved_goal']
        return np.clip(error/.012,-1,1).astype(np.float32),None


def scripted_handover_diagnostic():
    """Exercise real physical handovers; return unmistakably labeled test data."""
    with PickNavReachEnv(seed=42,object_idx=14) as native:
        policies={'grasp':AnalyticalGraspTestPolicy(native),
                  'carry':AnalyticalCarryTestPolicy(native),'reach':AnalyticalReachTestPolicy()}
        system=LearnedSystem(native,policies)
        policies['grasp'].controller=system.controller
        def forbid_reset(*args,**kwargs):
            raise AssertionError('No handover reset allowed')
        native.reset=forbid_reset
        result=system.run()
        result['kind']='scripted_integration_diagnostic'
        result['scope']='integration_test'
        result['algorithm']='deterministic_analytical_TEST_policies'
        result['not_a_learned_policy_result']=True
        result['assistance']['carry_waypoints_during_policy']=True
        calls=result['source_calls_by_controller']
        calls['analytical_test_policy']=calls.pop('learned')
        assert native.model.neq==0
        return result


def test_successful_physical_handover_with_explicit_analytical_test_policies():
    result=scripted_handover_diagnostic()
    assert result['success'],result
    assert result['metrics']['bilateral_contact']
    assert result['metrics']['ever_stably_held']
    assert result['metrics']['drop_events']==0
    assert result['metrics']['dist_to_goal']<.1
    assert result['wall_contact_calls']==result['robot_table_contact_calls']==0
    assert set(result['stages'])=={'planned_approach','grasp','pick_hold','planned_carry_posture','carry','reach','final_hold'}
    assert all(stage['success'] for stage in result['stages'].values())
    for stage in ('pick_hold','final_hold'):
        assert result['stages'][stage]['height_range_m']<.02
        assert result['stages'][stage]['object_motion_extent_m']<.02
    assert sum(stage['source_action_calls'] for stage in result['stages'].values())==result['source_action_calls']
    assert sum(result['source_calls_by_controller'].values())==result['source_action_calls']
    assert result['first_complete_reach_action']<result['source_action_calls']
    assert result['navigation_reach_source_calls']==result['first_complete_reach_action']-result['navigation_start_action']


def test_evaluation_freezes_the_exact_checkpoint_bytes(tmp_path):
    import hashlib
    import runpy
    evaluate=runpy.run_path('scripts/evaluate_system.py')
    source=tmp_path/'mutable_model.zip'
    source.write_bytes(b'first checkpoint bytes')
    records=evaluate['freeze_checkpoints']({'grasp':source},tmp_path/'frozen')
    source.write_bytes(b'later checkpoint bytes')
    from pathlib import Path
    captured=Path(records['grasp']['frozen_path']).read_bytes()
    assert captured==b'first checkpoint bytes'
    assert records['grasp']['sha256']==hashlib.sha256(captured).hexdigest()
