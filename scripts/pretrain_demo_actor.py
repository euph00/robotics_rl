#!/usr/bin/env python3
"""Disclosed actor-mean demonstration initialization of an existing SAC model.

Balanced whole-episode supervised split; generic dictionary observations/actions.
The bounded physical evaluator currently supports the four grasp starts. Teacher
labels affect sampling only; observations and learned inference are unchanged.
This is development BC initialization plus SAC+HER, not an independent SAC run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('approach', 'close', 'lift', 'settle')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def dataset_arrays(dataset, seed):
    """Keep two entire episodes per height out of supervised optimization."""
    rng = np.random.default_rng(seed)
    by_height = {}
    for index, episode in enumerate(dataset['episodes']):
        by_height.setdefault(float(episode['summary']['start_height_m']), []).append(index)
    validation = set()
    for indices in by_height.values():
        if len(indices) < 3:
            raise ValueError('Need at least three complete episodes in each start stratum')
        validation.update(int(index) for index in rng.choice(indices, max(1, len(indices)//4), replace=False))
    records, phases, heights, episode_ids = [], [], [], []
    for index, episode in enumerate(dataset['episodes']):
        first_goal = episode['summary']['first_goal_decision']
        for item in episode['transitions']:
            phase = item['teacher_phase']
            if phase == 'held_goal_correction':
                phase = 'settle' if first_goal is not None and item['step'] >= first_goal else 'lift'
            if phase not in PHASES:
                raise ValueError('Unknown demonstration phase; supply a supported dataset')
            records.append(item); phases.append(phase)
            heights.append(float(episode['summary']['start_height_m'])); episode_ids.append(index)
    observations = {key: np.stack([item['observation'][key] for item in records]).astype(np.float32)
                    for key in records[0]['observation']}
    actions = np.stack([item.get('teacher_action',item['action']) for item in records]).astype(np.float32)
    valid_mask = np.isin(episode_ids, list(validation))
    phases, heights = np.asarray(phases), np.asarray(heights)
    train_groups, validation_groups = {}, {}
    for phase in PHASES:
        for height in sorted(by_height):
            mask = (phases == phase) & (heights == height)
            key = f'{phase}:{height:g}'
            train_groups[key] = np.flatnonzero(mask & ~valid_mask)
            validation_groups[key] = np.flatnonzero(mask & valid_mask)
            if not len(train_groups[key]) or not len(validation_groups[key]):
                raise ValueError('Every phase/start stratum must occur on both supervised sides')
    split = {'train_episode_ids': sorted(set(range(len(dataset['episodes']))) - validation),
             'validation_episode_ids': sorted(validation), 'train_transitions': int((~valid_mask).sum()),
             'validation_transitions': int(valid_mask.sum()),
             'stratum_train_counts': {key: len(value) for key, value in train_groups.items()},
             'stratum_validation_counts': {key: len(value) for key, value in validation_groups.items()},
             'limitation': 'Correlated repetitions of one object/maze. Validation episodes are excluded only from BC updates; all were previously exposed to the source SAC replay. This is not held-out generalization.'}
    return observations, actions, train_groups, validation_groups, split


def physical_case(env, model, height, update, horizon=150, hold_calls=50):
    env.reset(seed=41000)
    snapshot = next(state for nominal, state in env.snapshots if abs(nominal-height) < 1e-8)
    env.native.restore_state(snapshot, reset_counters=True)
    env.goal = env.base_goal.copy()
    observation = env._observation()
    initial_object = env.native.observe()['object_pos'].copy()
    initial_action = model.predict(observation, deterministic=True)[0]
    trajectory, held_positions = [], []
    any_hit = False
    failure = 'decision_horizon'
    for decision in range(horizon):
        action = model.predict(observation, deterministic=True)[0]
        observation, _, terminated, truncated, info = env.step(action)
        obj = env.native.observe()['object_pos'].copy()
        trajectory.append({'decision': decision+1, 'action': action.tolist(), 'goal_distance': float(info['goal_distance']),
            'object_z': float(obj[2]), 'bilateral_contact': bool(info['bilateral_contact']),
            'action_rejected': bool(info['action_rejected']),
            'ee_minus_object_xyz_m': (observation['observation'][28:31]-observation['observation'][35:38]).tolist()})
        if info['is_success']:
            any_hit = True
            target = env.native.agent_qpos.copy(); target[env.native.finger_slots] = 0.
            hold_ok = True
            for _ in range(hold_calls):
                _, hold_info = env.native.step(target)
                position = env.native.observe()['object_pos'].copy()
                held_positions.append(position.tolist())
                local = position.copy(); local[:2] -= env.base_xy
                hold_ok &= bool(hold_info['bilateral_contact']) and float(np.linalg.norm(local-env.goal)) < env.tolerance
                hold_ok &= float(position[2]-initial_object[2]) >= .15
            extent = np.ptp(np.asarray(held_positions), axis=0)
            success = bool(hold_ok and extent[2] < .02)
            failure = None if success else 'retention_hold_failed'
            break
        if terminated:
            failure = 'physical_failure'; break
        if truncated:
            break
    else:
        success = False
    success = bool(any_hit and failure is None)
    return {'bc_updates': update, 'scope': 'four_start_grasp_diagnostic', 'nominal_start_height_m': height,
        'requested_lift_m': .20, 'teacher_at_inference': False, 'policy_decisions': len(trajectory),
        'success': success, 'any_goal_hit': any_hit, 'failure': failure, 'initial_action': initial_action.tolist(),
        'max_lift_m': max((item['object_z']-initial_object[2] for item in trajectory), default=0.),
        'final_goal_distance_m': trajectory[-1]['goal_distance'] if trajectory else None,
        'bilateral_contact_decisions': sum(item['bilateral_contact'] for item in trajectory),
        'source_action_calls': env.native.step_count, 'physics_ticks': env.native.step_count*env.native.substeps,
        'hold_calls': len(held_positions), 'hold_z_range_m': float(np.ptp(np.asarray(held_positions)[:,2])) if held_positions else None,
        'trajectory': trajectory, 'held_positions': held_positions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default='sac_grasp_demoseed01')
    parser.add_argument('--demonstrations', type=Path, default=ROOT/'runs/grasp_physical_demos_v1/artifacts/demonstrations.pkl')
    parser.add_argument('--run-id', default='grasp_bc_init01')
    parser.add_argument('--max-updates', type=int, default=5000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=7300)
    parser.add_argument('--geometry-features', action='store_true', help='Isolated measured-geometry representation experiment')
    args = parser.parse_args()
    if not 1 <= args.max_updates <= 5000 or args.eval_every < 100:
        raise ValueError('Bounded protocol:1..5000 updates and evaluation interval>=100')
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source = ROOT/'runs'/args.source_run
    source_config = json.loads((source/'config.json').read_text())
    if source_config['skill'] != 'grasp':
        raise ValueError('Physical evaluator currently supports grasp only')
    generation = json.loads((source/'checkpoints/latest.json').read_text())['generation']
    origin = source/'checkpoints'/generation
    source_counts = json.loads((origin/'progress.json').read_text())
    source_hash, demo_hash = sha256(origin/'model.zip'), sha256(args.demonstrations)
    dataset = pickle.loads(args.demonstrations.read_bytes())
    observations, actions, train_groups, validation_groups, split = dataset_arrays(dataset, args.seed)
    logger = RunLogger(args.run_id, kind='initialization', skill='grasp', algorithm='BC-initialized SAC + HER',
        name='Grasp · demonstration actor initialization', summary='Balanced actor-mean supervision; physical checks pending.')
    directory = logger.directory; artifacts = directory/'artifacts'; inputs = artifacts/'inputs'; inputs.mkdir(parents=True)
    shutil.copy2(origin/'model.zip', inputs/'source_model.zip')
    shutil.copy2(Path(__file__), artifacts/'pretrain_demo_actor.py')
    config = vars(args) | {'demonstrations': str(args.demonstrations), 'source_generation': generation,
        'source_model_sha256': source_hash, 'demonstrations_sha256': demo_hash, 'phase':'development',
        'independent_seed':False, 'new_online_transitions':0, 'source_online_transitions': source_counts['transitions'],
        'protocol':'phase-and-height-balanced tanh-actor-mean MSE; critics and log_std head unchanged',
        'actor_bc_learning_rate': .0003, 'batch_size':256, 'split':split,
        'physical_gate':'20cm object goal within4cm, bilateral contact,50 native closed-joint hold calls,Z-range<2cm',
        'teacher_at_learned_inference':False, 'save_semantics':'Normal SAC model ZIP; actor initializer only, no new online replay. Restore source replay and worker count explicitly for SAC continuation.'}
    atomic_json(directory/'config.json', config)
    critical = ['pnr/skills.py','pnr/env.py','pnr/local_control.py','pnr/reference.py','pnr/planning.py','pnr/features.py']
    code = artifacts/'source'; code.mkdir()
    if args.geometry_features: critical.append('pnr/geometry_features.py')
    for name in critical: shutil.copy2(ROOT/name, code/name.replace('/','__'))
    atomic_json(artifacts/'source_hashes.json', {name:sha256(ROOT/name) for name in critical})
    env = vec = None; evaluations=[]; total_eval_calls=total_eval_ticks=0; best_key=None; stalled=0
    started=time.perf_counter()
    try:
        env = ManipulationSkillEnv(skill='grasp', scene_seed=source_config['scene_seed'], object_idx=source_config['object_idx'],
                                   seed=source_config['seed'], horizon=150, evaluation=True)
        logger.update(setup_action_calls=env.setup_action_calls, setup_physics_ticks=env.setup_action_calls*env.native.substeps, initialization_physics_ticks=env.native.settle_steps)
        vec = DummyVecEnv([lambda:env])
        model = SAC.load(inputs/'source_model.zip', env=vec, device=args.device)
        if model.replay_buffer_class is not HerReplayBuffer:
            raise ValueError('Expected normal SAC/HER model, not inference replay override')
        transfer_report=None
        if args.geometry_features:
            from pnr.geometry_features import MeasuredGeometryExtractor,transfer_expanded_geometry_policy
            original=model
            policy_kwargs=dict(original.policy_kwargs,features_extractor_class=MeasuredGeometryExtractor,
                               features_extractor_kwargs=dict(original.policy_kwargs.get('features_extractor_kwargs',{}),geometry_scale=20.))
            model=SAC('MultiInputPolicy',vec,learning_rate=original.learning_rate,
                buffer_size=original.buffer_size,learning_starts=original.learning_starts,
                batch_size=original.batch_size,tau=original.tau,gamma=original.gamma,
                train_freq=(original.train_freq.frequency,original.train_freq.unit.value),
                gradient_steps=original.gradient_steps,ent_coef='auto',target_entropy=original.target_entropy,
                target_update_interval=original.target_update_interval,policy_kwargs=policy_kwargs,
                replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=original.replay_buffer_kwargs,
                device=args.device,seed=args.seed)
            transfer_report=transfer_expanded_geometry_policy(original.policy,model.policy)
            with torch.no_grad():model.log_ent_coef.copy_(original.log_ent_coef)
            model.num_timesteps=original.num_timesteps;model._n_updates=original._n_updates
            check={key:torch.as_tensor(value[:128],device=model.device) for key,value in observations.items()}
            check_actions=torch.as_tensor(actions[:128],device=model.device)
            errors=[]
            for relabel in (False,True):
                if relabel:check['desired_goal']=check['desired_goal'].roll(19,dims=0)+.03
                with torch.no_grad():
                    pairs=[(original.actor(check,deterministic=True),model.actor(check,deterministic=True))]
                    pairs+=list(zip(original.critic(check,check_actions),model.critic(check,check_actions)))
                    pairs+=list(zip(original.critic_target(check,check_actions),model.critic_target(check,check_actions)))
                for before,after in pairs:
                    torch.testing.assert_close(before,after,atol=1e-5,rtol=1e-5)
                    errors.append(float((before-after).abs().max()))
            transfer_report['stored_and_relabelled_prediction_max_abs_error']=max(errors)
            atomic_json(artifacts/'geometry_transfer.json',transfer_report)
            del original
        actor = model.policy.actor
        critic_before, target_before, std_before = tensor_hash(model.critic), tensor_hash(model.critic_target), tensor_hash(actor.log_std)
        parameters = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
        optimizer = torch.optim.Adam(parameters, lr=.0003)
        tensors = {key:torch.as_tensor(value,device=model.device) for key,value in observations.items()}
        targets = torch.as_tensor(actions,device=model.device)
        def losses(groups):
            with torch.no_grad(): prediction = actor(tensors,deterministic=True)
            squared = ((prediction-targets)**2).mean(dim=1).cpu().numpy()
            values={key:float(squared[indices].mean()) for key,indices in groups.items()}
            return float(np.mean(list(values.values()))),values
        inherited = source_counts.get('accounting',{})
        def accounting():
            setup=env.setup_action_calls
            return {'accounting_version':2,'complete':bool(inherited.get('complete',False)),
                'legacy_missing':inherited.get('legacy_missing',[]),'inherited_source_run':args.source_run,
                'inherited_accounting':inherited,'new_online_transitions':0,
                'demonstration_transitions_imported':len(actions),'supervised_train_transitions':split['train_transitions'],
                'new_setup_source_action_calls':setup,'new_setup_physics_ticks':setup*env.native.substeps,
                'new_evaluation_source_action_calls':total_eval_calls,'new_evaluation_physics_ticks':total_eval_ticks,
                'new_initialization_physics_ticks':env.native.settle_steps,
                'total_source_action_calls_known':int(inherited.get('total_source_action_calls_known',0))+setup+total_eval_calls,
                'total_physics_ticks_known':int(inherited.get('total_physics_ticks_known',0))+setup*env.native.substeps+total_eval_ticks+env.native.settle_steps,
                'demonstration_data_reused_without_new_collection':True}
        val_before,_=losses(validation_groups)
        atomic_json(artifacts/'split.json',split)
        logger.event('Frozen source and episode split; source replay previously exposed all demonstration episodes.',source_generation=generation)
        best_val=val_before
        for update in range(1,args.max_updates+1):
            batch=np.concatenate([rng.choice(indices,16,replace=True) for indices in train_groups.values()])
            obs={key:value[batch] for key,value in tensors.items()}
            prediction=actor(obs,deterministic=True)
            loss=torch.nn.functional.mse_loss(prediction,targets[batch])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,5.); optimizer.step()
            if update==1 or update%100==0:
                validation_loss,phase_losses=losses(validation_groups)
                best_val=min(best_val,validation_loss)
                logger.metric(total_eval_calls,metric_source='supervised_update',scope='actor_initialization',updates=update,
                    bc_train_loss=float(loss.detach()),bc_validation_loss=validation_loss,bc_validation_strata=phase_losses,
                    source_action_calls=total_eval_calls,physics_ticks=total_eval_ticks,accounting=accounting())
                logger.update(summary=f'Actor initialization {update}/{args.max_updates} updates; validation MSE {validation_loss:.4f}.')
                print(json.dumps({'bc_updates':update,'validation_mse':validation_loss}),flush=True)
            evaluate=(update==100 or update%args.eval_every==0 or update==args.max_updates)
            if not evaluate:continue
            heights=[.17] if update==100 and update<args.eval_every else [.17,.008,.04,.09]
            cases=[]
            for height in heights:
                case=physical_case(env,model,height,update)
                total_eval_calls+=case['source_action_calls'];total_eval_ticks+=case['physics_ticks'];cases.append(case)
            score=(sum(case['success'] for case in cases),sum(case['max_lift_m'] for case in cases),
                   -sum(case['final_goal_distance_m'] for case in cases))
            result={'bc_updates':update,'scope':'full_pregrasp_diagnostic' if len(cases)==1 else 'four_start_grasp_diagnostic',
                'episodes':len(cases),'success_rate':sum(case['success'] for case in cases)/len(cases),
                'cases':cases,'accounting':accounting(),'new_online_transitions':0}
            evaluations.append(result);atomic_json(artifacts/f'evaluation_{update:05d}.json',result)
            logger.event(f"BC update{update}: retained physical grasp {sum(case['success'] for case in cases)}/{len(cases)}.")
            logger.metric(total_eval_calls,metric_source='physical_evaluation',scope=result['scope'],evaluation_scope=result['scope'],
                updates=update,success_rate=result['success_rate'],eval_episodes=len(cases),accounting=accounting(),
                source_action_calls=total_eval_calls,physics_ticks=total_eval_ticks,
                mean_goal_distance=float(np.mean([case['final_goal_distance_m'] for case in cases])))
            print(json.dumps({'bc_updates':update,'physical_outcomes':[{key:case[key] for key in ('nominal_start_height_m','success','failure','max_lift_m','bilateral_contact_decisions')} for case in cases]}),flush=True)
            checkpoint_dir=artifacts/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
            model.save(checkpoint_dir/f'bc_update_{update:05d}.zip')
            if len(cases)==4:
                if best_key is None or score>best_key:
                    significant=best_key is None or score[0]>best_key[0] or score[1]>best_key[1]+.02
                    stalled=0 if significant else stalled+1
                    best_key=score;best=directory/'checkpoints/best';best.mkdir(parents=True,exist_ok=True)
                    model.save(best/'model.zip');atomic_json(best/'evaluation.json',result)
                else:stalled+=1
                if score[0]==4:break
                if stalled>=4:
                    logger.event('Stopped: held-out loss alone is insufficient; four full-bank checks lacked meaningful physical improvement.')
                    break
        if tensor_hash(model.critic)!=critic_before or tensor_hash(model.critic_target)!=target_before or tensor_hash(actor.log_std)!=std_before:
            raise RuntimeError('Actor-only update changed critic or log_std head weights')
        if sha256(origin/'model.zip')!=source_hash or sha256(args.demonstrations)!=demo_hash:
            raise RuntimeError('Input artifact changed during initialization')
        final={'kind':'BC_initialized_SAC_HER_development','source_run':args.source_run,'source_generation':generation,
            'source_model_sha256':source_hash,'demonstrations_sha256':demo_hash,'bc_updates':update,
            'source_online_transitions':source_counts['transitions'],'new_online_transitions':0,
            'initial_validation_mse':val_before,'final_validation_mse':losses(validation_groups)[0],
            'split':split,'evaluations':evaluations,'accounting':accounting(),
            'critics_unchanged':True,'log_std_head_weights_unchanged':True,
            'log_std_output_caveat':'Its unchanged head reads the updated shared actor backbone; stochastic outputs can change.',
            'input_artifacts_unchanged':True,'teacher_at_inference':False,'independent_seed':False,'geometry_transfer':transfer_report,
            'best_model':str(directory/'checkpoints/best/model.zip'),'wall_seconds':time.perf_counter()-started}
        atomic_json(artifacts/'result.json',final)
        logger.artifact('artifacts/result.json','BC initialization: split, physics, costs, and limitations','file')
        for result in evaluations:
            logger.artifact(f"artifacts/evaluation_{result['bc_updates']:05d}.json",f"Physical grasp checks after {result['bc_updates']} BC updates",'file')
        logger.finish('complete',f"Actor initialization finished at {update} updates; best all-start retained grasp {best_key[0] if best_key else 0}/4. No new online SAC transitions.")
    except BaseException as error:
        logger.finish('failed',f'{type(error).__name__}: {error}');raise
    finally:
        if vec is not None:vec.close()
        elif env is not None:env.close()


if __name__=='__main__':main()
