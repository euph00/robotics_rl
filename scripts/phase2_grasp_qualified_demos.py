#!/usr/bin/env python3
"""Conditionally collect real grasp prefixes qualified by subsequent TRAIN transport.

Planning and tests run without physics. Collection requires a reviewed plan and
a completed non-improving frozen-core result. No grasp transition is rewritten
using its later transport outcome; open handover prefixes are BC data, not a
fabricated terminal replay episode. All attempted prefixes remain available.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv
from pnr.general_manipulation import GeneralGraspEnv, GraspSchema
from pnr.grasp_sampling import SAMPLING_PROTOCOL, predictor_for_variant
from pnr.learned_carry import LearnedCarryPreparation, PREPARATION_SETTINGS
from pnr.policies import load_policy_checkpoint
from pnr.scenes import load_scene_bank
from pnr.system import StageFailure
from pnr.telemetry import RunLogger, atomic_json
from scripts.evaluate_system import freeze_checkpoints, plain
from scripts.phase2_evaluate_system import select_cases, sha256_file
from scripts.phase2_learned_carry_demos import atomic_pickle, preparation_schedule
from scripts.phase2_carry_demos import collect_episode
from scripts.phase2_apple_retention_branches import contact_record

PROTOCOL = 'transport-qualified-grasp-prefixes-v1'
OBJECTS = [14, 5, 2, 7, 10]
_GRASP = None


def serial(value):
    return json.loads(json.dumps(value, default=plain, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_observation(observation):
    if set(observation) != {'observation', 'achieved_goal', 'desired_goal'}:
        raise ValueError('Grasp observation keys changed')
    for name, size in [('observation', 54), ('achieved_goal', 3), ('desired_goal', 3)]:
        value = observation[name]
        if (not isinstance(value, np.ndarray) or value.shape != (size,)
                or value.dtype != np.float32 or not np.isfinite(value).all()):
            raise ValueError('Invalid actual grasp observation '+name)


def validate_prefix(rows):
    """Check measured tuples; allow nonterminal caller-handover boundaries."""
    schema = GraspSchema()
    calls = ticks = 0
    for index, row in enumerate(rows):
        for name in ['observation', 'next_observation']:
            validate_observation(row[name])
        action = row['action']
        if action.shape != (4,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.:
            raise ValueError('Invalid actual normalized grasp action')
        if not np.array_equal(row['observation']['desired_goal'], row['next_observation']['desired_goal']):
            raise ValueError('Grasp goal changed within a transition')
        if index:
            previous = rows[index-1]
            if previous['terminated'] or previous['truncated']:
                raise ValueError('Grasp actions after an actual episode boundary')
            if any(not np.array_equal(previous['next_observation'][key], row['observation'][key])
                   for key in row['observation']):
                raise ValueError('Discontinuous grasp observations')
            if previous['native_call_after'] != row['native_call_before']:
                raise ValueError('Unrecorded physical actions inside grasp prefix')
        real_reward = float(schema.compute_reward(row['next_observation']['achieved_goal'],
                           row['next_observation']['desired_goal'], row['info']))
        if real_reward != row['reward']:
            raise ValueError('Stored grasp reward differs from native reward contract')
        delta = row['native_call_after']-row['native_call_before']
        tick_delta = row['physics_tick_after']-row['physics_tick_before']
        if delta != 3 or tick_delta != 5*delta:
            raise ValueError('Original grasp action duration changed')
        if row['info']['action_calls_delta'] != delta or row['info']['physics_ticks_delta'] != tick_delta:
            raise ValueError('Grasp info and measured physical counters disagree')
        calls += delta; ticks += tick_delta
    return {'decisions': len(rows), 'source_action_calls': calls, 'physics_ticks': ticks}


class RecordingGrasp:
    """Transparent non-owning wrapper: one delegated action, one exact saved tuple."""
    def __init__(self, wrapped, persist):
        self.wrapped, self.persist = wrapped, persist
        self.action_space, self.observation_space = wrapped.action_space, wrapped.observation_space
        self.rows = []
        self.current = None
        self.boundary = None

    def _observation(self):
        observed = self.wrapped._observation()
        validate_observation(observed)
        self.current = deepcopy(observed)
        return observed

    def step(self, action):
        if self.boundary is not None:
            raise ValueError('Recorder already sealed')
        if self.current is None:
            raise ValueError('Record the actual policy input before taking an action')
        native = self.wrapped.native
        before = (native.step_count, native.physics_steps)
        prior, executed = deepcopy(self.current), np.asarray(action).copy()
        result = self.wrapped.step(action)
        observed, reward, terminated, truncated, info = result
        row = {'observation': prior, 'action': executed, 'next_observation': deepcopy(observed),
               'reward': float(reward), 'terminated': bool(terminated), 'truncated': bool(truncated),
               'info': deepcopy(info), 'native_call_before': before[0],
               'native_call_after': native.step_count, 'physics_tick_before': before[1],
               'physics_tick_after': native.physics_steps}
        self.rows.append(row)
        self.current = deepcopy(observed)
        # Persist before validation and before the caller can perform hold/posture.
        # A malformed measured transition remains as evidence and halts collection.
        self.persist(self.rows)
        validate_prefix(self.rows)
        if terminated or truncated:
            self.boundary = 'native_termination' if terminated else 'native_truncation'
        return result

    def seal(self, reason):
        if self.boundary is None:
            self.boundary = reason
        return {'kind': self.boundary, 'flags_rewritten': False,
                'directly_replay_seedable': False,
                'note': 'A caller handover may leave both native ending flags false; qualification is BC metadata only.'}


def qualification(prepared, route, held, *, wall_contacts, table_contacts, fatal_error=False):
    return bool(prepared and route and held and not wall_contacts and not table_contacts and not fatal_error)


def checkpoint_contract(model):
    """Inspect saved spaces/feature dimensions and ancestry, without prediction."""
    shapes = {key: list(space.shape) for key, space in model.observation_space.spaces.items()}
    actor = model.actor.features_extractor; critic = model.critic.features_extractor
    result = {'raw_observation_shapes': shapes, 'action_shape': list(model.action_space.shape),
        'actor_extractor': type(actor).__name__, 'actor_engineered_feature_width': int(actor.features_dim),
        'critic_extractor': type(critic).__name__, 'critic_engineered_feature_width': int(critic.features_dim),
        'historical_online_decisions': int(model.num_timesteps),
        'historical_online_sac_updates': int(model._n_updates),
        'model_predictions_during_inspection': 0, 'physical_interactions_during_inspection': 0}
    if (shapes != {'observation': [54], 'achieved_goal': [3], 'desired_goal': [3]}
            or result['action_shape'] != [4]
            or result['actor_extractor'] != 'GeneralGraspFeatures'
            or result['critic_extractor'] != 'GeneralGraspFeatures'
            or result['actor_engineered_feature_width'] != 70
            or result['critic_engineered_feature_width'] != 70):
        raise ValueError('Expected raw54+3+3 spaces and derived70-feature original actor/critic')
    return result


def initialize_worker(checkpoint):
    global _GRASP
    import torch
    torch.set_num_threads(1)
    if sha256_file(checkpoint['frozen_path']) != checkpoint['sha256']:
        raise ValueError('Frozen grasp checkpoint changed')
    _GRASP = load_policy_checkpoint(checkpoint['frozen_path'], device='cpu')
    if checkpoint_contract(_GRASP) != checkpoint['contract']:
        raise ValueError('Frozen grasp dimensions or historical SAC ancestry changed')


def worker(task):
    case, output, settings = task
    out = Path(output); key = case['preparation_id']
    native = system = recorder = predictor = None
    started = time.monotonic(); observed_calls = 0; transport_rows = []
    result = {'preparation_id': key, 'case_id': case['case_id'], 'case': case,
              'prepared_handover': False, 'route_completed': False, 'final_hold_retained': False,
              'bc_eligible': False, 'full_task_evaluation': False, 'partition': 'train',
              'original_start': True, 'retry': False, 'cost_accounting_complete': False}
    prefix_path = out/(key+'_grasp.pkl')
    def persist(rows):
        atomic_pickle(prefix_path, {'protocol': PROTOCOL, 'case': case, 'transitions': rows,
                                   'qualification_not_yet_applied': True})
    with (out/(key+'.log')).open('x') as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            if (out.parent/'STOP').exists():
                result.update(skipped_before_physics=True, fatal_error='Peer collection error')
                return result
            predictor = predictor_for_variant(_GRASP, case['preparation_variant'])
            native = PickNavReachEnv(**case['env_kwargs'])
            if native.scene_manifest()['maze_hash'] != case['maze_hash']:
                raise ValueError('Original TRAIN map differs from frozen plan')
            def observe(current, observation, info):
                nonlocal observed_calls
                observed_calls += 1
            system = LearnedCarryPreparation(native, predictor, on_step=observe)
            def bind(n, controller, horizon):
                nonlocal recorder
                recorder = RecordingGrasp(GeneralGraspEnv.bind(n, controller, horizon=horizon,
                                           goal_lift_m=settings['grasp_lift_m']), persist)
                return recorder
            system.skill_factories['grasp'] = bind
            preparation = system.prepare()
            result['preparation_result'] = preparation
            result['prepared_handover'] = bool(preparation['prepared_handover'])
            if recorder:
                grasp_passed = preparation.get('stages', {}).get('grasp', {}).get('success', False)
                result['grasp_boundary'] = recorder.seal(
                    'caller_handover_prefix_end' if grasp_passed else 'caller_stopped_prefix')
            if not result['prepared_handover']:
                result.update(failed_stage=preparation.get('failed_stage'), error=preparation.get('error'))
            else:
                result['handover_geometry'] = contact_record(native, system.controller,
                                                            native.observe(), native.evaluate(), system)
                carry = GeneralCarryEnv.bind(native, system.controller,
                          horizon=settings['carry_horizon'], delta_scale=settings['delta_scale'])
                costs = {'source_action_calls': 0, 'physics_ticks': 0}
                with system._observe_native_steps():
                    start = system._begin('teacher_carry', 'teacher')
                    transport_rows, success, info, stopped = collect_episode(carry, carry._observation(),
                        np.random.default_rng(case['preparation_variant']['torch_seed']), 0., costs, out.parent/'STOP')
                    result['route_completed'] = bool(success and info.get('is_success') and
                        info.get('bilateral_contact') and not native.drop_events)
                    system._finish('teacher_carry', start, result['route_completed'],
                                   decisions=len(transport_rows), failure_reason=info.get('failure_reason'))
                    result['transport_costs'] = costs
                    if stopped:
                        raise InterruptedError('Collection interrupted; no replacement attempt')
                    if result['route_completed']:
                        system._held_pause('transport_final_hold', settings['final_hold_calls'])
                        result['final_hold_retained'] = True
                        result['final_hold_geometry'] = contact_record(native, system.controller,
                            native.observe(), native.evaluate(), system)
        except StageFailure as error:
            result.update(error=repr(error), failed_stage=system.stage if system else 'environment_setup')
        except Exception as error:
            result.update(error=repr(error), fatal_error=repr(error),
                          failed_stage=system.stage if system else 'environment_setup')
        finally:
            if recorder:
                result['grasp_boundary'] = recorder.seal('caller_stopped_prefix')
                try:
                    result['grasp_costs'] = validate_prefix(recorder.rows)
                except Exception as error:
                    result['fatal_error'] = repr(error)
            else:
                persist([])
            if predictor:
                try:
                    result['sampling_report'] = predictor.sampling_report()
                except Exception as error:
                    result['fatal_error'] = repr(error)
            if system:
                result.update(stages=system.stage_results, source_calls_by_controller=system.calls_by_source,
                    wall_contact_calls=system.wall_contact_calls, robot_table_contact_calls=system.robot_table_contact_calls,
                    proposal_diagnostics=system.controller.stage_results.get('grasp_generation'))
            if native:
                result.update(source_action_calls=native.step_count, physics_ticks=native.physics_steps,
                    initialization_physics_ticks=native.settle_steps,
                    total_physics_ticks_including_initialization=native.physics_steps+native.settle_steps,
                    final_metrics=native.evaluate())
                result['cost_accounting_complete'] = bool(system and observed_calls == native.step_count
                    and sum(system.calls_by_source.values()) == native.step_count
                    and native.physics_steps == native.substeps*native.step_count)
                native.close()
            if not result['cost_accounting_complete'] and not result.get('skipped_before_physics'):
                result['fatal_error'] = result.get('fatal_error', 'Unknown physical accounting; stop collection')
            result['bc_eligible'] = qualification(result['prepared_handover'], result['route_completed'],
                result['final_hold_retained'], wall_contacts=result.get('wall_contact_calls', 0),
                table_contacts=result.get('robot_table_contact_calls', 0), fatal_error=bool(result.get('fatal_error')))
            if result.get('fatal_error'):
                (out.parent/'STOP').touch()
            result['wall_seconds'] = time.monotonic()-started
            # Raw prefix bytes are never rewritten using a transport result.
            result['grasp_prefix_sha256'] = sha256_file(prefix_path)
            atomic_pickle(out/(key+'_transport.pkl'), transport_rows)
            atomic_pickle(out/(key+'_result.pkl'), result)
            result = serial(result)
            atomic_json(out/(key+'.json'), result)
    return result


def summarize(results, schedule):
    expected = {c['preparation_id']: c for c in schedule}
    ids = [r['preparation_id'] for r in results]
    if len(set(ids)) != len(ids) or set(ids)-set(expected):
        raise ValueError('Duplicate or undeclared preparations')
    if any(r['case'] != expected[r['preparation_id']] for r in results):
        raise ValueError('Preparation identity changed')
    return {'protocol': PROTOCOL, 'partition': 'train', 'full_task_evaluation': False,
            'planned_preparations': len(schedule), 'completed_preparations': len(results),
            'prepared_handovers': sum(r['prepared_handover'] for r in results),
            'qualified_prefixes': sum(r['bc_eligible'] for r in results),
            'all_failures_retained': True, 'missing_preparation_ids': sorted(set(expected)-set(ids)),
            'fatal_errors': [r['preparation_id'] for r in results if r.get('fatal_error')],
            'cost_accounting_complete': all(r['cost_accounting_complete'] for r in results),
            **{k: sum(r.get(k, 0) for r in results) for k in ['source_action_calls', 'physics_ticks',
                 'initialization_physics_ticks', 'total_physics_ticks_including_initialization']},
            'results': results}


def make_plan(bank, grasp, old_data, root=ROOT):
    cases = select_cases(bank, 'train', OBJECTS)
    training = [c for c in cases if int(c['layout_id'].split('_')[-1]) < 8]
    comparison = [c for c in cases if 8 <= int(c['layout_id'].split('_')[-1]) < 12]
    if len(training) != 40 or len(comparison) != 20:
        raise ValueError('Expected first8 TRAIN layouts and next4 development layouts')
    return {'protocol': PROTOCOL, 'status': 'frozen_pending_independent_review_and_conditional_launch',
        'collection_run_id': 'phase2_grasp_qualified_demos01', 'source_grasp': grasp,
        'old_demonstrations': old_data, 'sampling_protocol': SAMPLING_PROTOCOL,
        'collection_seed': 9401, 'schedule': preparation_schedule(training, 3, 9401),
        'workers': 8, 'settings': {'grasp_lift_m': .24, 'carry_horizon': 1600,
            'delta_scale': .075, 'final_hold_calls': 50, 'preparation': PREPARATION_SETTINGS},
        'recording': {'raw_observation_shapes': {'observation': [54], 'achieved_goal': [3], 'desired_goal': [3]},
            'action_shape': [4], 'actor_engineered_feature_width': 70, 'critic_engineered_feature_width': 70,
            'actual_native_rewards_and_endings': True,
            'prefix_persisted_before_caller_continues': True, 'qualification_only_changes_bc_eligibility': True,
            'replay_ready': False, 'no_fabricated_terminal_or_delayed_failure_label': True,
            'qualification': 'Full unchanged teacher route plus50 native hold calls, no contact failure/drop/collision.',
            'all_attempts_retained': True, 'retries': 0, 'case_winner_selection': False},
        'bc_refinement': {'updates': 2000, 'batch_size': 400, 'old_examples_per_update': 200,
            'qualified_examples_per_update': 200, 'learning_rate': .0003, 'gradient_clip_norm': 10.,
            'seed': 10101, 'optimizer': 'fresh Adam on actor; critics and temperature unchanged',
            'actor': 'Current clearance01: raw54-state plus two3-value goals; GeneralGraspFeatures derives70 features. No representation/log_std change.',
            'historical_online_decisions': 12000, 'historical_online_sac_updates': 2000,
            'preserve_saved_rl_counters': True, 'new_online_decisions': 0,
            'ancestry_note': '2000 additional BC updates refine an already SAC-trained actor, not a zero-historical-RL initializer.',
            'old_sampling': 'uniform object, then uniform existing nonempty approach/close/lift/settle strata',
            'qualified_sampling': 'uniform object, then uniform eligible prefix, then uniform recorded step',
            'missing_object_rule': 'Stop before fitting if any object lacks an eligible prefix; no replacement collection',
            'selection': 'fixed2000-update endpoint; no intermediate physical checkpoint selection',
            'supervised_examples': 800000, 'online_sac_updates': 0,
            'training_requires_separate_review_of_completed_collection': True},
        'development_comparison': {'cases': comparison, 'branches': ['frozen_clearance01', 'qualified_bc_002000'],
            'independent_original_starts': True, 'deterministic_inference': True, 'teacher_route_only': True,
            'full_task_evaluation': False, 'headline': 'Retained full teacher transport plus hold; not learned navigation',
            'ancestry': 'TRAIN development cases may appear in ancestor training; withheld only from this refinement',
            'report_all_case_outcomes_and_ticks': True, 'candidate_qualification_not_a_carry_network_oracle': True,
            'further_sac_or_full60_requires_review': True},
        'launch_gate': {'requires_independent_review_record': True,
            'frozen_core_summary': str(root/'runs/phase2_system_encoder_gated192_frozen_040k01/artifacts/summary.json'),
            'selected_control_summary': str(root/'runs/phase2_system_learned_balanced_040k01/artifacts/summary.json'),
            'comparison': 'lexicographic full-success count then inclusive15ktick-success count; launch only if core does not beat control'},
        'no_heldout_access': True, 'new_physical_interactions_at_plan_creation': 0,
        'new_optimizer_updates_at_plan_creation': 0}


def verify_gate(plan, review, frozen, control):
    if review.get('approved') is not True or review.get('scope') != 'collection_only' or review.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError('Independent collection review record required for this exact plan')
    for summary in [frozen, control]:
        if (summary.get('partition') != 'validation' or summary.get('planned_cases') != 60
                or summary.get('completed_cases') != 60 or not summary.get('cost_accounting_complete')
                or summary.get('missing_case_ids')):
            raise ValueError('Conditional gate requires complete costed60-case validation')
    score = lambda x: (x['successes'], x['successes_within_deadline_including_initialization'])
    if score(frozen) > score(control):
        raise ValueError('Frozen core beats selected candidate; collection remains disabled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    freeze = sub.add_parser('freeze-plan')
    freeze.add_argument('--directory', type=Path, default=ROOT/'runs/phase2_grasp_qualified_plan01/artifacts')
    freeze.add_argument('--bank', type=Path, default=ROOT/'configs/phase2_scene_bank_expanded.json')
    execute = sub.add_parser('collect')
    execute.add_argument('--plan', type=Path, required=True)
    execute.add_argument('--review-record', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'freeze-plan':
        if args.directory.exists():
            raise ValueError('Refuse replacing frozen plan directory')
        args.directory.mkdir(parents=True)
        bank = load_scene_bank(args.bank)
        source = ROOT/'runs/phase2_sac_grasp_clearance01/candidate/grasp.zip'
        grasp = freeze_checkpoints({'grasp': source}, args.directory/'checkpoints')['grasp']
        model = load_policy_checkpoint(grasp['frozen_path'], device='cpu')
        grasp['contract'] = checkpoint_contract(model)
        if (model.num_timesteps, model._n_updates) != (12000, 2000):
            raise ValueError('Current clearance01 ancestral RL counters changed')
        del model
        old = ROOT/'runs/phase2_grasp_demos06/artifacts/demonstrations.pkl'
        plan = make_plan(bank, grasp, {'path': str(old), 'sha256': sha256_file(old)})
        paths = list((ROOT/'pnr').glob('*.py')) + [Path(__file__),
            ROOT/'scripts/phase2_carry_demos.py', ROOT/'scripts/phase2_learned_carry_demos.py',
            ROOT/'scripts/phase2_apple_retention_branches.py', ROOT/'scripts/evaluate_system.py',
            ROOT/'scripts/phase2_evaluate_system.py']
        plan['source_hashes'] = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}
        plan['scene_bank'] = {'path': str(args.bank.resolve()), 'file_sha256': sha256_file(args.bank)}
        control_path = Path(plan['launch_gate']['selected_control_summary'])
        control = json.loads(control_path.read_text())
        plan['launch_gate']['selected_control_summary_sha256'] = sha256_file(control_path)
        plan['launch_gate']['selected_control_score'] = [control['successes'], control['successes_within_deadline_including_initialization']]
        plan['plan_sha256'] = digest(plan)
        atomic_json(args.directory/'plan.json', plan)
        print(json.dumps({'plan': str(args.directory/'plan.json'), 'sha256': plan['plan_sha256'], 'physics': 0}))
        return
    plan = json.loads(args.plan.read_text()); expected = plan.pop('plan_sha256')
    if digest(plan) != expected:
        raise ValueError('Frozen plan hash mismatch')
    plan['plan_sha256'] = expected
    for path, expected_hash in plan['source_hashes'].items():
        if sha256_file(ROOT/path) != expected_hash:
            raise ValueError('Frozen source changed: '+path)
    for record, field in [(plan['source_grasp'], 'frozen_path'), (plan['old_demonstrations'], 'path')]:
        if sha256_file(record[field]) != record['sha256']:
            raise ValueError('Frozen policy/data changed')
    if sha256_file(plan['scene_bank']['path']) != plan['scene_bank']['file_sha256']:
        raise ValueError('Scene bank changed')
    review = json.loads(args.review_record.read_text())
    gate = plan['launch_gate']
    frozen = json.loads(Path(gate['frozen_core_summary']).read_text())
    control = json.loads(Path(gate['selected_control_summary']).read_text())
    if sha256_file(gate['selected_control_summary']) != gate['selected_control_summary_sha256']:
        raise ValueError('Predeclared selected control changed')
    frozen_plan = json.loads(Path(gate['frozen_core_summary']).with_name('evaluation_plan.json').read_text())
    control_plan = json.loads(Path(gate['selected_control_summary']).with_name('evaluation_plan.json').read_text())
    if frozen_plan['cases'] != control_plan['cases'] or frozen_plan['execution'] != control_plan['execution']:
        raise ValueError('Conditional gate cases/execution differ from selected control')
    verify_gate(plan, review, frozen, control)
    directory = ROOT/'runs'/plan['collection_run_id']
    if directory.exists():
        raise ValueError('No retries, resumed collection or artifact replacement')
    logger = RunLogger(plan['collection_run_id'], kind='data', skill='grasp',
        algorithm='Frozen native-SAC grasp with TRAIN teacher transport qualification',
        budget=len(plan['schedule']), name='Transport-qualified grasp prefixes')
    out = directory/'artifacts'; out.mkdir(exist_ok=True)
    atomic_json(out/'plan.json', plan)
    atomic_json(out/'launch_evidence.json', {'review': review,
        'frozen_core_summary_sha256': sha256_file(gate['frozen_core_summary']),
        'control_summary_sha256': sha256_file(gate['selected_control_summary'])})
    results = []
    with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(plan['source_grasp'],)) as pool:
        futures = [pool.submit(worker, (case, str(out), plan['settings'])) for case in plan['schedule']]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            summary = summarize(results, plan['schedule']); atomic_json(out/'summary.json', summary)
            logger.metric(len(results), scope='training_data', qualified_prefixes=summary['qualified_prefixes'],
                          physics_ticks=summary['total_physics_ticks_including_initialization'])
    summary = summarize(results, plan['schedule'])
    if summary['fatal_errors'] or summary['missing_preparation_ids'] or not summary['cost_accounting_complete']:
        logger.finish(status='failed', summary='Malformed/incomplete collection; raw shards and all costs retained, no qualified dataset published.')
        raise RuntimeError('Collection incomplete or malformed; no training authorized')
    episodes = []
    for result in results:
        path = out/(result['preparation_id']+'_grasp.pkl')
        if sha256_file(path) != result['grasp_prefix_sha256']:
            raise ValueError('Immutable grasp prefix changed after transport')
        raw = pickle.loads(path.read_bytes())
        validate_prefix(raw['transitions'])
        episodes.append({'case': raw['case'], 'transitions': raw['transitions'], 'summary': result})
    atomic_pickle(out/'grasp_prefixes.pkl', {'protocol': PROTOCOL, 'plan_sha256': plan['plan_sha256'],
        'qualification_only_bc': True, 'directly_replay_seedable': False, 'episodes': episodes})
    success = not summary['fatal_errors'] and not summary['missing_preparation_ids'] and summary['cost_accounting_complete']
    logger.finish(status='complete' if success else 'failed',
                  summary=f"{summary['qualified_prefixes']}/{len(plan['schedule'])} transport-qualified prefixes; all attempts retained.")
    if not success:
        raise RuntimeError('Collection incomplete or malformed; no training authorized')


if __name__ == '__main__':
    main()
