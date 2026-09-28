#!/usr/bin/env python3
"""Diagnostic-only shared IK/collision snapshot; canonical source stays unchanged."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation
from pnr.local_control import LocalCartesianIK
from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json


def prototype_class():
    """Derive the prototype from the inspected current solver, with two edits.

    The diagnostic uses local trusted Python source to ensure every numerical
    operation stays identical; only the optional validator executes before
    the existing finally block restores state. No repository module is edited.
    """
    original = textwrap.dedent(inspect.getsource(LocalCartesianIK.ik))
    changed = original.replace('orientation_tolerance=.02):', 'orientation_tolerance=.02, validator=None):')
    needle = '\n                return q\n'
    if changed.count(needle) != 1 or changed == original:
        raise RuntimeError('Canonical IK changed; inspect before deriving prototype')
    changed = changed.replace(needle,
        "\n                if validator is not None and not validator(q):\n"
        "                    raise ValueError('Arm target collision')\n"
        "                return q\n")
    namespace = {'np':np, 'Rotation':Rotation}
    exec(compile(changed, '<diagnostic-shared-transaction-ik>', 'exec'), namespace)
    return type('DiagnosticValidatedIK', (LocalCartesianIK,), {'ik':namespace['ik']}), changed


class DiagnosticBridge:
    """Connect the optional callback without changing canonical skill.step.

    A production integration would simply pass the validator to IK and omit the
    following collision_free call. This bridge verifies that exactly the same
    candidate was checked before satisfying that existing second call.
    """
    def __init__(self, env, solver):
        self.env, self.solver = env, solver
        self.original_collision = env.controller.planner.collision_free
        self.validated = None
        self.fused_checks = 0

    def ik(self, *args, **kwargs):
        self.validated = None
        def validator(q):
            passed = self.env.controller.planner._collision_free(q, include_object=False, margin=-.0005)
            if passed:
                self.validated = q.copy()
            return passed
        return self.solver.ik(*args, **kwargs, validator=validator)

    def collision_free(self, q, include_object=False, margin=.001):
        if self.validated is None or include_object or margin != -.0005 or not np.array_equal(q, self.validated):
            raise AssertionError('Unmatched query in diagnostic fusion bridge')
        self.fused_checks += 1
        self.validated = None
        return True


def state(env):
    native = env.native
    pos, quat = native.p.getBasePositionAndOrientation(native.object_id)
    linear, angular = native.p.getBaseVelocity(native.object_id)
    joint = native.p.getJointStates(native.robot_id, native.joint_indices)
    points = native.p.getContactPoints(native.robot_id, native.object_id)
    # Fixed identity/order fields plus every numerical contact field are checked.
    return {'joint_position':np.array([value[0] for value in joint]),
            'joint_velocity':np.array([value[1] for value in joint]),
            'object_position':np.array(pos), 'object_quaternion':np.array(quat),
            'object_linear_velocity':np.array(linear), 'object_angular_velocity':np.array(angular),
            'contacts':points, 'contact_summary':native.contacts(),
            'last_motor_target':native.last_target.copy(),
            'counters':(native.step_count,native.physics_steps,native.stable_hold_steps,
                        native.ever_lifted,native.ever_stably_held,native.had_grasp,native.drop_events)}


def flatten_contact(points):
    return np.array([[*c[:5], *c[5], *c[6], *c[7], c[8], c[9], c[10], *c[11], c[12], *c[13]]
                     for c in points], dtype=float)


def differences(a, b):
    result = {}
    for key in a:
        if key == 'contacts':
            aa, bb = flatten_contact(a[key]), flatten_contact(b[key])
            result[key] = float(np.max(np.abs(aa-bb))) if aa.shape == bb.shape and aa.size else (0. if aa.shape == bb.shape else 1.)
        elif key in ('counters','contact_summary'):
            result[key] = 0. if a[key] == b[key] else 1.
        else:
            result[key] = float(np.max(np.abs(a[key]-b[key])))
    return result


def to_json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:to_json(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [to_json(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='grasp_query_fusion_prototype')
    parser.add_argument('--actions', default='runs/grasp_simulator_profile/artifacts/decisions.json')
    parser.add_argument('--decisions-per-start', type=int, default=50)
    parser.add_argument('--repeat-canonical', action='store_true')
    parser.add_argument('--fresh-env-per-variant', action='store_true')
    args = parser.parse_args()
    actions_path = ROOT/args.actions
    actions = json.loads(actions_path.read_text())
    if len(actions) != 100 or not 1 <= args.decisions_per_start <= 50:
        raise ValueError('Expected 100 recorded actions and at most 50 decisions/start')
    actions = [row for row in actions if row['group_step'] < args.decisions_per_start]
    modes = ('canonical','canonical_repeat','fused') if args.repeat_canonical else ('canonical','fused')
    logger = RunLogger(args.run_id, kind='validation', skill='grasp', algorithm='Query transaction prototype',
        name='Simulator query equivalence', budget=len(modes)*len(actions),
        summary='Same 100 actual actions and physical starts; canonical code versus diagnostic snapshot fusion.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    env = None
    started = time.perf_counter()
    paths = ('pnr/env.py','pnr/skills.py','pnr/local_control.py','pnr/planning.py')
    source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    cls, prototype_source = prototype_class()
    (artifacts/'prototype_ik.py').write_text(prototype_source)
    atomic_json(logger.directory/'config.json',vars(args)|{'training':False, 'competence_evaluation':False,
        'actions_sha256':hashlib.sha256(actions_path.read_bytes()).hexdigest(),
        'scope':'Only diagnostic object methods substituted; canonical source files unchanged.'})
    try:
        env = ManipulationSkillEnv(skill='grasp', scene_seed=42, object_idx=14, seed=31415,
                                   horizon=100, evaluation=True)
        setup_calls = env.setup_action_calls
        environment_builds = 1
        original_ik = env.local_ik
        original_collision = env.controller.planner.collision_free
        solver = cls(env.native)
        bridge = DiagnosticBridge(env, solver)
        records = {}
        for mode in modes:
            if args.fresh_env_per_variant and mode != modes[0]:
                env.close()
                env = ManipulationSkillEnv(skill='grasp', scene_seed=42, object_idx=14, seed=31415,
                                           horizon=100, evaluation=True)
                setup_calls += env.setup_action_calls
                environment_builds += 1
                original_ik = env.local_ik
                original_collision = env.controller.planner.collision_free
                solver = cls(env.native)
                bridge = DiagnosticBridge(env,solver)
            env.local_ik = original_ik if mode != 'fused' else bridge
            env.controller.planner.collision_free = original_collision if mode != 'fused' else bridge.collision_free
            trace = []
            current_height = None
            for index, item in enumerate(actions):
                height = item['start_height_m']
                if height != current_height:
                    current_height = height
                    env.reset(seed=31415+int(height>.01), options={'full_pregrasp':height>.01})
                begin = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(np.asarray(item['action'],dtype=np.float32))
                elapsed = time.perf_counter()-begin
                trace.append({'state':state(env), 'seconds':elapsed, 'observation':observation,
                    'reward':reward, 'terminated':terminated, 'truncated':truncated,
                    'rejected':info['action_rejected'], 'native_calls':info['action_calls_delta'],
                    'physics_ticks':info['physics_ticks_delta'],
                    'ik_iterations':(original_ik if mode!='fused' else solver).last_iterations})
                if terminated or truncated:
                    env.reset(seed=31415+int(height>.01)+100+item['group_step'], options={'full_pregrasp':height>.01})
                if (index+1)%25 == 0:
                    logger.metric((0 if mode!='fused' else len(actions))+index+1,
                        scope='performance', metric_source='profiling', mode=mode,
                        decisions_per_second=(index+1)/sum(row['seconds'] for row in trace))
            records[mode] = trace
            atomic_json(artifacts/f'{mode}_physical_trace.json', to_json(trace))
        max_diff = {}
        mismatch_steps = []
        for index,(before,after) in enumerate(zip(records['canonical'],records['fused'])):
            diff = differences(before['state'],after['state'])
            for key in ('observation','achieved_goal','desired_goal'):
                diff['obs_'+key] = float(np.max(np.abs(before['observation'][key]-after['observation'][key])))
            for key in ('reward','terminated','truncated','rejected','native_calls','physics_ticks','ik_iterations'):
                diff[key] = 0. if before[key] == after[key] else 1.
            for key,value in diff.items():
                max_diff[key] = max(max_diff.get(key,0.),value)
            if any(value != 0 for value in diff.values()):
                mismatch_steps.append({'decision':index+1, 'differences':{k:v for k,v in diff.items() if v}})
        # Both query exceptions must restore full physical state. These checks
        # take no physics steps and are separate from the 200 action comparison.
        env.local_ik = original_ik
        env.controller.planner.collision_free = original_collision
        env.reset(seed=31415)
        ee,orientation = env.controller.planner.ee_pose()
        failure_checks = []
        for case,position,validator in (
            ('validator_rejection',ee,lambda q:False),
            ('unreachable_target',ee+np.array([5.,0.,0.]),lambda q:True)):
            before = state(env)
            raised = False
            try:
                solver.ik(position,orientation,validator=validator)
            except ValueError:
                raised = True
            diff = differences(before,state(env))
            failure_checks.append({'case':case, 'raised_value_error':raised,
                                   'maximum_state_differences':diff,
                                   'exact_state_restored':all(value==0 for value in diff.values())})
        repeated_baseline = None
        if 'canonical_repeat' in records:
            repeated_baseline = {'maxima':{},'mismatched_steps':[]}
            for index,(first,repeated) in enumerate(zip(records['canonical'],records['canonical_repeat'])):
                diff = differences(first['state'],repeated['state'])
                for key,value in diff.items():
                    repeated_baseline['maxima'][key] = max(repeated_baseline['maxima'].get(key,0.),value)
                if any(value != 0 for value in diff.values()):
                    repeated_baseline['mismatched_steps'].append(index+1)
        elapsed = {mode:sum(row['seconds'] for row in trace) for mode,trace in records.items()}
        source_unchanged = source_hashes == {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
        passed = not mismatch_steps and all(row['raised_value_error'] and row['exact_state_restored'] for row in failure_checks)
        report = {'training':False,'competence_evaluation':False,'exact_equivalence':passed,
            'source_files_unchanged':source_unchanged,'source_hashes':source_hashes,
            'actions_sha256':hashlib.sha256(actions_path.read_bytes()).hexdigest(),
            'comparison_decisions_per_variant':len(actions),'variant_seconds':elapsed,
            'variant_mean_ms':{mode:1000*t/len(actions) for mode,t in elapsed.items()},
            'speedup':elapsed['canonical']/elapsed['fused'],
            'time_reduction_fraction':1-elapsed['fused']/elapsed['canonical'],
            'avoided_snapshot_transactions':bridge.fused_checks,
            'rejections':{mode:sum(row['rejected'] for row in trace) for mode,trace in records.items()},
            'maximum_state_differences':max_diff,'mismatch_steps':mismatch_steps,
            'repeated_baseline':repeated_baseline,
            'failure_cleanup_checks':failure_checks,
            'counts':{'policy_decisions':len(modes)*len(actions),'profile_source_action_calls':sum(row['native_calls'] for trace in records.values() for row in trace),
                'profile_physics_ticks':sum(row['physics_ticks'] for trace in records.values() for row in trace),
                'setup_source_action_calls':setup_calls,'setup_physics_ticks':setup_calls*env.native.substeps,
                'initialization_physics_ticks':environment_builds*env.native.settle_steps,
                'environment_builds':environment_builds,
                'query_only_failure_checks':len(failure_checks)},
            'wall_seconds':time.perf_counter()-started,
            'interpretation_limit':'Sequential variants in one process under concurrent training load; fresh physical builds per variant='+str(args.fresh_env_per_variant)+'. Exact equivalence applies to recorded starts/actions and cleanup cases only.'}
        report['counts']['total_source_action_calls']=report['counts']['setup_source_action_calls']+report['counts']['profile_source_action_calls']
        report['counts']['total_physics_ticks']=report['counts']['setup_physics_ticks']+report['counts']['profile_physics_ticks']+report['counts']['initialization_physics_ticks']
        atomic_json(artifacts/'comparison.json',report)
        # Compact physical trace permits independent inspection without claiming task success.
        physical_trace = {mode:[{key:value for key,value in row.items() if key!='observation'} for row in trace]
                          for mode,trace in records.items()}
        def serializable(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if isinstance(value,np.generic):return value.item()
            if isinstance(value,dict):return {key:serializable(item) for key,item in value.items()}
            if isinstance(value,(tuple,list)):return [serializable(item) for item in value]
            return value
        atomic_json(artifacts/'physical_traces.json',serializable(physical_trace))
        logger.artifact('artifacts/comparison.json','Exact physical state comparison and runtime')
        logger.artifact('artifacts/prototype_ik.py','Diagnostic-only optional-validator prototype')
        logger.artifact('artifacts/physical_traces.json','Both actual physical trajectories')
        logger.finish('completed' if passed and source_unchanged else 'failed',
            f'Exact state match={passed}; measured {report["speedup"]:.2f}× speed. Canonical code unchanged.')
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as error:
        logger.finish('failed',f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
