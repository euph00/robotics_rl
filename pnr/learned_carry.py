"""Opt-in carry preparation from a frozen learned grasp at an original TRAIN start.

Only pre-grasp approach and carry posture use the existing planned controller.
The learned grasp and physical hold are the same prefix as LearnedSystem.run().
No transport teacher, route bank, fabricated state, or scene substitution occurs
inside this preparation. Callers own physical state caching and accounting.
"""
from __future__ import annotations

from functools import partial

from .general_carry import GeneralCarryEnv
from .general_manipulation import GeneralGraspEnv
from .system import LearnedSystem


PREPARATION_PROTOCOL='original-start-frozen-grasp-carry-v1'
PREPARATION_SETTINGS={
    'grasp_proposal_mode':'clearance-v2','grasp_lift_m':.24,
    'grasp_horizon':600,'pick_hold_source_calls':50,
    'controller_speed_scale':1.,'carry_delta_scale':.075,
}


class _UnusedPolicy:
    def predict(self,*args,**kwargs):
        raise AssertionError('Carry/reach policies must not execute during prefix preparation')


class _PreparedHandover(Exception):
    pass


class LearnedCarryPreparation(LearnedSystem):
    """Stop the shared full-system prefix exactly at the carry handover.

    The caller supplies a verified frozen54-state grasp predictor and owns the
    original-start native simulator. The returned snapshot preserves all pickup
    history; restore it with reset_counters=False and use a separate cost ledger.
    Defaults are frozen by PREPARATION_SETTINGS and intentionally match the
    learned-handover collector. No physical transport is performed here.
    """
    def __init__(self,native,grasp,on_step=None):
        super().__init__(native,{'grasp':grasp,'carry':_UnusedPolicy(),'reach':_UnusedPolicy()},
            grasp_horizon=600,carry_horizon=900,reach_horizon=1,on_step=on_step,
            controller_speed_scale=1.,grasp_proposal_mode='clearance-v2',reach_tolerance_m=.08,
            skill_factories={'grasp':partial(GeneralGraspEnv.bind,goal_lift_m=.24)})
        self.calls_by_source['teacher']=0
        self.handover_state=None;self.handover_observation=None;self.prepared_carry=None

    def _skill(self,name):
        if name=='grasp':return super()._skill(name)
        if name!='carry':raise AssertionError('Preparation must stop before reach')
        carry=GeneralCarryEnv.bind(self.native,self.controller,horizon=900,delta_scale=.075)
        self.prepared_carry=carry
        self.handover_observation=carry._observation()
        self.handover_state=self.native.save_state()
        raise _PreparedHandover()

    def prepare(self):
        try:
            result=super().run()
        except _PreparedHandover:
            result={'status':'prepared','success':True,'stages':self.stage_results}
        result.update(scope='original_start_learned_carry_preparation',full_task_evaluation=False,
            prepared_handover=self.handover_state is not None,
            stages=self.stage_results,policy_decisions=dict(self.policy_decisions),
            source_action_calls=self.native.step_count,physics_ticks=self.native.physics_steps,
            first_grasped_lift_action=self.first_grasped_lift_action,
            wall_contact_calls=self.wall_contact_calls,robot_table_contact_calls=self.robot_table_contact_calls)
        return result


# The prefix above is shared with frozen collection runs. Cache bookkeeping below
# never changes its physical actions or supplies transport teacher actions.
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .env import PickNavReachEnv
from .multi_carry import CarrySchema
from .policies import load_policy_checkpoint


CACHE_PROTOCOL = 'fixed-train-learned-handover-cache-v1'
CACHE_RESTORE_SETTINGS = {'source_action_hold_calls': 5,
    'targets': 'restored measured qpos with zero finger targets',
    'history': 'preserved', 'required_final_state': 'lifted bilateral contact and no drops',
    'retry_failed_case': False}


def restore_learned_handover(carry, state):
    """Restore a real snapshot and physically refresh contact forces, with costs.

    Five native hold calls are an explicit part of training reset preparation.
    No contact bit is fabricated. An early collision/drop or exception fails the
    attempt; contact absence alone may settle during the fixed five-call hold.
    """
    native = carry.native
    record = {'success': False, 'source_action_calls': 0, 'physics_ticks': 0,
              'contact_checks': [], 'protocol': 'five-call-measured-hold-v1'}
    before = None
    try:
        native.restore_state(state, reset_counters=False)
        before = (native.step_count, native.physics_steps)
        carry.controller.planner.set_payload()
        target = native.agent_qpos.copy()
        target[native.finger_slots] = 0.
        for _ in range(CACHE_RESTORE_SETTINGS['source_action_hold_calls']):
            _, info = native.step(target)
            collision = carry._collision()
            check = {key: info[key] for key in
                     ('bilateral_contact', 'lifted', 'drop_events', 'ever_lifted', 'ever_stably_held')}
            check['collision'] = bool(collision)
            record['contact_checks'].append(check)
            if collision or info['drop_events']:
                record['error'] = 'collision' if collision else 'dropped_object'
                break
        final = record['contact_checks'][-1]
        record['success'] = bool(len(record['contact_checks']) == 5 and
            final['bilateral_contact'] and final['lifted'] and not final['drop_events'] and
            not final['collision'])
        if not record['success'] and 'error' not in record:
            record['error'] = 'hold_did_not_retain_lifted_bilateral_contact'
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if before is not None:
            record['source_action_calls'] = int(native.step_count - before[0])
            record['physics_ticks'] = int(native.physics_steps - before[1])
    return record


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def learned_carry_identity(case, grasp_sha256, *, horizon=900, delta_scale=.075,
                           gamma=.995, shaping_weight=2.):
    """Content identity, including point seed/object and frozen prefix settings."""
    return _digest({'protocol': CACHE_PROTOCOL, 'case': case,
                    'grasp_sha256': grasp_sha256,
                    'preparation_protocol': PREPARATION_PROTOCOL,
                    'preparation_settings': PREPARATION_SETTINGS,
                    'restore_settings': CACHE_RESTORE_SETTINGS,
                    'carry': {'horizon': horizon, 'delta_scale': delta_scale,
                              'gamma': gamma, 'shaping_weight': shaping_weight}})


class LearnedMultiCaseCarry(CarrySchema):
    """Finite TRAIN-only physical handovers, rotating one episode per case.

    Each declared case is prepared once per process from its original start.
    Failed preparation is retained, never replaced, and yields no carry replay.
    The cache must fit the complete worker slice; there is no silent eviction or
    re-preparation. A resume reconstructs successful physical caches and counts
    the new cost; previously failed cases are retained without automatic retry.
    Full-task evaluation remains unconditional and does not use this cache.
    """
    def __init__(self, cases, grasp_checkpoint, grasp_sha256, *, seed=0,
                 cache_size=None, episodes_per_case=1, horizon=900,
                 delta_scale=.075, gamma=.995, shaping_weight=2.):
        super().__init__(gamma, shaping_weight)
        self.cases = deepcopy(list(cases))
        if not self.cases:
            raise ValueError('At least one declared TRAIN case is required')
        ids = []
        for case in self.cases:
            if case.get('partition') != 'train' or not case.get('layout_id', '').startswith('train_'):
                raise ValueError('Learned handover cache accepts declared TRAIN cases only')
            required = ('case_id', 'maze_hash', 'object_idx', 'env_kwargs')
            if any(key not in case for key in required):
                raise ValueError('Incomplete declared TRAIN case')
            kwargs = case['env_kwargs']
            if any(key not in kwargs for key in ('seed', 'maze_seed', 'object_idx', 'seed_mode')):
                raise ValueError('TRAIN case must freeze point seed, maze seed, object and seed mode')
            if kwargs['object_idx'] != case['object_idx']:
                raise ValueError('Case object does not match native object')
            ids.append(case['case_id'])
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate TRAIN case ID')
        self.cache_size = len(self.cases) if cache_size is None else int(cache_size)
        if self.cache_size < len(self.cases):
            raise ValueError('Finite cache must hold the complete declared worker slice')
        if int(episodes_per_case) != episodes_per_case or episodes_per_case < 1:
            raise ValueError('episodes_per_case must be a positive integer')
        if int(horizon) != horizon or horizon < 1 or not 0. < gamma < 1.:
            raise ValueError('Positive horizon and discount strictly between zero and one required')
        if delta_scale != PREPARATION_SETTINGS['carry_delta_scale']:
            raise ValueError('Learned handover protocol requires carry delta_scale=.075')
        self.grasp_checkpoint = Path(grasp_checkpoint).resolve()
        self.grasp_sha256 = str(grasp_sha256)
        actual = hashlib.sha256(self.grasp_checkpoint.read_bytes()).hexdigest()
        if actual != self.grasp_sha256:
            raise ValueError('Frozen grasp checkpoint digest mismatch')
        self.grasp = load_policy_checkpoint(self.grasp_checkpoint, device='cpu')
        if (self.grasp.observation_space['observation'].shape != (54,) or
                self.grasp.action_space.shape != (4,)):
            raise ValueError('Expected frozen 54-state, four-action grasp policy')
        protocol = getattr(self.grasp, '_pnr_grasp_protocol_signature', None)
        expected = {'observation_version': 'antipodal-relative-grasp-v1',
                    'control_version': 'bounded_local_global_ik-v1',
                    'grasp_proposal_mode': PREPARATION_SETTINGS['grasp_proposal_mode'],
                    'eval_lift_m': PREPARATION_SETTINGS['grasp_lift_m']}
        if not isinstance(protocol, dict) or any(protocol.get(k) != v for k, v in expected.items()):
            raise ValueError('Frozen grasp protocol metadata is absent or incompatible with learned carry preparation')
        if 'sha256' in protocol:
            payload = {k: v for k, v in protocol.items() if k != 'sha256'}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            if protocol['sha256'] != digest:
                raise ValueError('Frozen grasp protocol metadata digest mismatch')
        self.grasp_protocol_signature = deepcopy(protocol)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.episodes_per_case = int(episodes_per_case)
        self.horizon = int(horizon)
        self.delta_scale, self.gamma, self.shaping_weight = delta_scale, gamma, shaping_weight
        self.case_keys = [learned_carry_identity(c, self.grasp_sha256, horizon=self.horizon,
                          delta_scale=delta_scale, gamma=gamma, shaping_weight=shaping_weight)
                          for c in self.cases]
        self.signature = _digest({'case_keys': self.case_keys,
                                  'episodes_per_case': self.episodes_per_case,
                                  'seed': self.seed})
        self.cache = {}
        self.failed_cases = {}
        self.preparation_attempts = []
        self.restore_attempts = []
        self.failed_setup_attempts = []
        self.reset_count = 0
        self.curriculum_level = 0  # Requested schedule only; full exit goal is fixed.
        self.active = self._active_entry = None
        self._closed = False
        # These counters describe new work in this process. The runner separately
        # carries inherited lineage totals, so a resume must not add them twice.
        self.setup_action_calls = self.setup_physics_ticks = self.initialization_physics_ticks = 0
        self.source_action_calls = self.physics_ticks = 0
        self.case_online_counts = {c['case_id']: {'resets': 0, 'decisions': 0,
            'source_action_calls': 0, 'physics_ticks': 0} for c in self.cases}

    @property
    def preparation_source_action_calls(self):
        return self.setup_action_calls

    @property
    def preparation_physics_ticks(self):
        return self.setup_physics_ticks

    def _prepare_case(self, index):
        case, key = self.cases[index], self.case_keys[index]
        if key in self.cache or key in self.failed_cases:
            return
        native = system = None
        record = {'case_id': case['case_id'], 'case': deepcopy(case), 'cache_identity': key,
                  'attempt_index': len(self.preparation_attempts),
                  'prepared_handover': False, 'cost_accounting_complete': True,
                  'preparation_source_action_calls': 0, 'preparation_physics_ticks': 0,
                  'initialization_physics_ticks': 0}
        try:
            native = PickNavReachEnv(**case['env_kwargs'])
            record['initialization_physics_ticks'] = int(native.settle_steps)
            if native.scene_manifest()['maze_hash'] != case['maze_hash']:
                raise ValueError('Native TRAIN maze differs from declared map hash')
            system = LearnedCarryPreparation(native, self.grasp)
            result = system.prepare()
            record['preparation_result'] = result
            if not result.get('prepared_handover', False):
                record['failure_stage'] = result.get('failed_stage')
                record['error'] = result.get('error', 'Learned prefix did not prepare a handover')
                return
            if system.handover_state is None or system.prepared_carry is None:
                raise RuntimeError('Prepared prefix omitted physical state or carry binding')
            carry = system.prepared_carry
            # The shared prefix binds the carry interface without stepping. Only
            # wrapper reward/horizon settings change here, never physical state.
            carry.horizon = self.horizon
            carry.gamma, carry.shaping_weight = self.gamma, self.shaping_weight
            carry.failure_cost = 1. / (1. - self.gamma)
            record['prepared_handover'] = True
            self.cache[key] = {'case': case, 'native': native, 'system': system,
                               'carry': carry, 'state': system.handover_state, 'restore_validated': False}
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            record['failure_stage'] = getattr(system, 'stage', 'environment_setup')
        finally:
            if native is None:
                record['cost_accounting_complete'] = False
                record['cost_accounting_note'] = 'Partial native constructor cost unknown'
            else:
                # Prefix has no restore. Native counters therefore give all new
                # calls, including calls before a preparation exception.
                record['preparation_source_action_calls'] = int(native.step_count)
                record['preparation_physics_ticks'] = int(native.physics_steps)
            self.setup_action_calls += record['preparation_source_action_calls']
            self.setup_physics_ticks += record['preparation_physics_ticks']
            self.initialization_physics_ticks += record['initialization_physics_ticks']
            self.preparation_attempts.append(record)
            if not record['prepared_handover']:
                self.failed_cases[key] = deepcopy(record)
                self.failed_setup_attempts.append(deepcopy(record))
                if native is not None:
                    native.close()

    def prime_cache(self):
        """Prepare each declared case once without changing episode rotation.

        Returns failure/cost records even if no case is ready. The runner should
        persist this report and check global five-object coverage across workers.
        reset() raises if this worker has no valid handover.
        """
        if self._closed:
            raise RuntimeError('Cache is closed')
        for index in range(len(self.cases)):
            self._prepare_case(index)
        for key in list(self.cache):
            if not self.cache[key]['restore_validated']:
                self._restore_entry(key, reason='initial_cache_validation')
        return self.cache_report()

    def _restore_entry(self, key, *, reason):
        entry = self.cache[key]
        result = restore_learned_handover(entry['carry'], entry['state'])
        result.update(case_id=entry['case']['case_id'], case=deepcopy(entry['case']),
                      cache_identity=key, reason=reason,
                      attempt_index=len(self.restore_attempts), cost_accounting_complete=True)
        self.setup_action_calls += result['source_action_calls']
        self.setup_physics_ticks += result['physics_ticks']
        self.restore_attempts.append(result)
        if result['success']:
            entry['restore_validated'] = True
        else:
            self.failed_cases[key] = deepcopy(result)
            self.failed_setup_attempts.append(deepcopy(result))
            entry['native'].close()
            del self.cache[key]
            if self._active_entry is entry:
                self.active = self._active_entry = None
        return result

    def cache_report(self):
        objects = sorted({c['object_idx'] for c in self.cases})
        declared = [c['case_id'] for c in self.cases]
        valid = [c['case_id'] for c, key in zip(self.cases, self.case_keys) if key in self.cache]
        failed = [c['case_id'] for c, key in zip(self.cases, self.case_keys) if key in self.failed_cases]
        by_object = {str(obj): {
            'declared': sum(c['object_idx'] == obj for c in self.cases),
            'valid': sum(c['object_idx'] == obj and key in self.cache
                         for c, key in zip(self.cases, self.case_keys)),
            'failed': sum(c['object_idx'] == obj and key in self.failed_cases
                          for c, key in zip(self.cases, self.case_keys))} for obj in objects}
        return {'protocol': CACHE_PROTOCOL, 'signature': self.signature,
                'declared_case_ids': declared, 'valid_case_ids': valid, 'failed_case_ids': failed,
                'object_counts': by_object,
                'valid_object_counts': {obj: counts['valid'] for obj, counts in by_object.items()},
                'ready_for_training': bool(valid),
                'scenes': len(self.cache), 'cache_capacity': self.cache_size,
                'conditional_on_prepared_handover': True, 'full_task_evaluation': False,
                'preparation_protocol': PREPARATION_PROTOCOL,
                'preparation_settings': deepcopy(PREPARATION_SETTINGS),
                'restore_settings': deepcopy(CACHE_RESTORE_SETTINGS),
                'grasp_sha256': self.grasp_sha256,
                'grasp_protocol_signature': deepcopy(self.grasp_protocol_signature),
                'original_exit_goal': True,
                'curriculum_applied': False, 'requested_curriculum_level': self.curriculum_level,
                'setup_source_action_calls': self.setup_action_calls,
                'setup_physics_ticks': self.setup_physics_ticks,
                'preparation_source_action_calls': self.setup_action_calls,
                'preparation_physics_ticks': self.setup_physics_ticks,
                'preparation_fields_alias_setup_costs': True,
                'initialization_physics_ticks': self.initialization_physics_ticks,
                'incremental_transport_source_action_calls': self.source_action_calls,
                'incremental_transport_physics_ticks': self.physics_ticks,
                'cost_scope': 'incremental_process; attempt history and case counts include lineage',
                'cost_accounting_complete': all(r['cost_accounting_complete'] for r in self.preparation_attempts),
                'preparation_attempts': deepcopy(self.preparation_attempts),
                'restore_attempts': deepcopy(self.restore_attempts),
                'case_online_counts': deepcopy(self.case_online_counts)}

    def reset(self, *, seed=None, options=None):
        if options and any(key != 'full' or not value for key, value in options.items()):
            raise ValueError('Learned handover cache supports full original-exit episodes only')
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.prime_cache()
        while True:
            ready = [key for key in self.case_keys if key in self.cache]
            if not ready:
                raise RuntimeError('No valid learned handover in this worker; inspect preparation coverage and costs')
            key = ready[(self.reset_count // self.episodes_per_case) % len(ready)]
            if self._restore_entry(key, reason='online_episode_reset')['success']:
                break
        entry = self.cache[key]
        native, carry = entry['native'], entry['carry']
        carry._frozen_target = native.agent_qpos.copy()
        carry._frozen_target[native.finger_slots] = 0.
        carry.goal = native.goal_pos[:2].copy() / carry.goal_scale
        carry._episode_steps = carry._goal_hold = 0
        carry._episode_success = carry._retention_failed = False
        carry._missing_contact_calls = 0
        carry._episode_horizon = self.horizon
        carry.episode_scope = 'full_carry'
        carry.reset_count += 1
        observation = carry._observation()
        self.active, self._active_entry = carry, entry
        self.reset_count += 1
        self.case_online_counts[entry['case']['case_id']]['resets'] += 1
        return observation, {'evaluation_scope': 'full_carry',
            'conditional_on_prepared_handover': True, 'case_id': entry['case']['case_id'],
            'object_idx': entry['case']['object_idx'], 'maze_seed': native.maze_seed,
            'point_seed': entry['case']['env_kwargs']['seed'], 'cache_identity': key,
            'original_exit_goal': True, 'curriculum_applied': False,
            'curriculum_level': self.curriculum_level}

    def step(self, action):
        if self.active is None or self._closed:
            raise RuntimeError('Reset an open cache before stepping')
        native = self._active_entry['native']
        counts = self.case_online_counts[self._active_entry['case']['case_id']]
        before = (native.step_count, native.physics_steps)
        try:
            observation, reward, terminated, truncated, info = self.active.step(action)
        finally:
            # Count executed calls even when the wrapper raises after partial work.
            calls, ticks = native.step_count - before[0], native.physics_steps - before[1]
            self.source_action_calls += calls
            self.physics_ticks += ticks
            counts['source_action_calls'] += calls
            counts['physics_ticks'] += ticks
        counts['decisions'] += 1
        info.update(case_id=self._active_entry['case']['case_id'],
                    object_idx=self._active_entry['case']['object_idx'],
                    point_seed=self._active_entry['case']['env_kwargs']['seed'],
                    conditional_on_prepared_handover=True,
                    action_calls_delta=calls, physics_ticks_delta=ticks)
        return observation, reward, terminated, truncated, info

    def set_curriculum_level(self, level):
        # Retain callback compatibility, but never create route-bank/segment goals.
        self.curriculum_level = int(level)

    def get_training_state(self):
        return {'protocol': CACHE_PROTOCOL, 'signature': self.signature,
                'case_keys': list(self.case_keys), 'reset_count': self.reset_count,
                'curriculum_level': self.curriculum_level,
                'rng_state': deepcopy(self.rng.bit_generator.state),
                'failed_cases': deepcopy(self.failed_cases),
                'failed_setup_attempts': deepcopy(self.failed_setup_attempts),
                'preparation_attempts': deepcopy(self.preparation_attempts),
                'restore_attempts': deepcopy(self.restore_attempts),
                'case_online_counts': deepcopy(self.case_online_counts),
                'physical_snapshots_serialized': False,
                'restore_requires_new_physical_preparation': True,
                'accounting': self.cache_report()}

    def set_training_state(self, state):
        if state.get('protocol') != CACHE_PROTOCOL or state.get('signature') != self.signature:
            raise ValueError('Mismatched learned-handover case/checkpoint/preparation signature')
        if self.preparation_attempts or self.cache or self.reset_count:
            raise RuntimeError('Restore rotation metadata before any physical preparation/reset')
        if (state.get('case_keys') != self.case_keys or
                set(state.get('case_online_counts', {})) != set(self.case_online_counts) or
                not set(state.get('failed_cases', {})).issubset(self.case_keys)):
            raise ValueError('Invalid saved learned-handover case records')
        self.reset_count = int(state['reset_count'])
        self.curriculum_level = int(state['curriculum_level'])
        self.rng.bit_generator.state = deepcopy(state['rng_state'])
        self.failed_cases = deepcopy(state['failed_cases'])
        self.failed_setup_attempts = deepcopy(state['failed_setup_attempts'])
        self.preparation_attempts = deepcopy(state['preparation_attempts'])
        self.restore_attempts = deepcopy(state['restore_attempts'])
        self.case_online_counts = deepcopy(state['case_online_counts'])

    def close(self):
        if self._closed:
            return
        for entry in self.cache.values():
            entry['native'].close()
        self.cache.clear()
        self.active = self._active_entry = None
        self._closed = True
