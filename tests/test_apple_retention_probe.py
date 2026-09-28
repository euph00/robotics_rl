"""The diagnostic pause preserves native targets and refreshes measured state."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.phase2_apple_retention_branches import FirstContactPause


def build_wrapper(first_contact=True, retained=True):
    info = {'bilateral_contact':first_contact,'lifted':False,'object_height':.66,
            'is_success':False,'episode_success':False}
    native = SimpleNamespace(step_count=9,agent_qpos=np.array([1.,2.,.03,.04]),
        finger_slots=np.array([2,3]),drop_events=0,table_height=.626)
    native.evaluate=lambda: {'bilateral_contact':retained,'lifted':False,'object_height':.66}
    native.observe=lambda: {'object_pos':np.array([0.,0.,.66])}
    commands=[]
    def tick(target,stage):
        native.step_count+=1
        commands.append((target.copy(),stage))
    controller=SimpleNamespace(tick=tick,planner=SimpleNamespace(ee_pose=lambda:(np.zeros(3),None)))
    system=SimpleNamespace(native=native,controller=controller,stage='grasp',control_source='learned')
    def observation():
        return {'achieved_goal':np.array([0.,0.,float(native.step_count)]),'desired_goal':np.zeros(3)}
    base=SimpleNamespace(action_space=None,tolerance=.04,_episode_success=False,
                         _observation=observation,measured_grasp_target=lambda:np.ones(3),
                         compute_reward=lambda achieved,desired,info:-float(np.linalg.norm(achieved-desired)))
    base.step=lambda action:(observation(),-9.,False,False,dict(info))
    events=[]
    return FirstContactPause(base,system,events),system,commands,events


def test_contact_pause_holds_current_arm_with_original_closed_fingers_exactly_once():
    wrapper,system,commands,events=build_wrapper()
    observation,reward,terminated,truncated,info=wrapper.step(np.zeros(4))
    assert len(commands)==30 and system.native.step_count==39
    assert all(np.array_equal(command,[1.,2.,0.,0.]) for command,stage in commands)
    assert all(stage=='first_contact_pause' for command,stage in commands)
    assert system.stage=='grasp' and system.control_source=='learned'
    assert observation['achieved_goal'][2]==39 and reward==-39
    assert events[0]['start_call']==9 and events[0]['end_call']==39
    assert not terminated and not truncated
    wrapper.step(np.zeros(4))
    assert len(commands)==30 and len(events)==1


def test_no_pause_without_bilateral_contact():
    wrapper,system,commands,events=build_wrapper(first_contact=False)
    wrapper.step(np.zeros(4))
    assert not commands and not events and not wrapper.invoked


def test_pause_does_not_keep_stale_positive_contact_state():
    wrapper,system,commands,events=build_wrapper(retained=False)
    _,_,_,_,info=wrapper.step(np.zeros(4))
    assert not info['bilateral_contact'] and not info['is_success']
    assert not events[0]['retained']


def test_diagnostic_nested_result_serialization(tmp_path):
    import json
    from pnr.telemetry import atomic_json
    from scripts.evaluate_system import plain
    result={'stages':{'grasp':{'success':np.bool_(True),'position':np.array([1.,2.,3.])}},
            'physics_ticks':np.int64(150)}
    converted=json.loads(json.dumps(result,default=plain,allow_nan=False))
    atomic_json(tmp_path/'result.json',converted)
    assert json.loads((tmp_path/'result.json').read_text())=={
        'stages':{'grasp':{'success':True,'position':[1.,2.,3.]}},'physics_ticks':150}
