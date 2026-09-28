"""Seed HER with explicitly counted physical demonstrations, without actor cloning."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import numpy as np


def _discount_relabel_contract(model, data, target_gamma):
    """Validate source rewards before insertion; schema computation runs no physics."""
    from .multi_carry import CarrySchema
    if (isinstance(target_gamma, bool) or not isinstance(target_gamma, (int, float, np.floating))
            or target_gamma != .999 or getattr(model, 'gamma', None) != target_gamma):
        raise ValueError('Discount relabeling requires explicit target gamma .999 matching the learner')
    metadata = data.get('metadata', {})
    expected = {'partition': 'train', 'gamma': .995, 'shaping_weight': 2.,
                'delta_scale': .075, 'repeat': 5}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Discount relabeling requires the declared original TRAIN carry reward/action contract')
    source = CarrySchema(gamma=.995, shaping_weight=2.)
    target = CarrySchema(gamma=float(target_gamma), shaping_weight=2.)
    if (model.action_space.shape != (2,) or
            {key: space.shape for key, space in model.observation_space.spaces.items()} !=
            {key: space.shape for key, space in source.observation_space.spaces.items()}):
        raise ValueError('Discount relabeling supports only the unchanged 580-state carry contract')
    count = 0
    maximum_error = 0.
    for episode in data['episodes']:
        rows = episode.get('transitions', [])
        if not rows:
            raise ValueError('Empty demonstration episode')
        achieved = np.stack([row['next_observation']['achieved_goal'] for row in rows])
        desired = np.stack([row['observation']['desired_goal'] for row in rows])
        infos = [row['info'] for row in rows]
        # The potential must use the actual previous achieved goal, including HER.
        if any('previous_achieved_goal' not in row['info'] or
               not np.array_equal(row['info']['previous_achieved_goal'], row['observation']['achieved_goal'])
               for row in rows):
            raise ValueError('Carry reward relabeling requires the recorded previous physical achieved goal')
        stored = np.asarray([row['reward'] for row in rows], dtype=np.float32)
        recomputed = source.compute_reward(achieved, desired, infos)
        if not np.allclose(stored, recomputed, atol=1e-5, rtol=0.):
            raise ValueError('Stored source reward differs from the declared original gamma .995 contract')
        maximum_error = max(maximum_error, float(np.max(np.abs(stored.astype(float)-recomputed))))
        count += len(rows)
    # Pure reward queries distinguish gamma, potential weight and absorbing failure
    # cost even for a degenerate dataset; these probes never enter replay.
    achieved = np.array([[.2, 0.], [.2, 0.], [.5, 0.]], dtype=np.float32)
    desired = np.array([[.5, 0.]] * 3, dtype=np.float32)
    infos = [dict(previous_achieved_goal=np.array([.1, 0.], dtype=np.float32),
                  bilateral_contact=True, retention_ok=not failed, physical_failure=failed)
             for failed in (False, True, False)]
    expected_rewards = target.compute_reward(achieved, desired, infos)
    actual = model.env.env_method('compute_reward', achieved, desired, infos, indices=[0])[0]
    if not np.allclose(actual, expected_rewards, atol=1e-5, rtol=0.):
        raise ValueError('Active target reward must use gamma .999, shaping 2 and matching failure cost 1000')
    return target, {'protocol': 'carry-discount-replay-relabel-v1',
        'source_gamma': .995, 'target_gamma': float(target_gamma),
        'source_shaping_weight': 2., 'target_shaping_weight': 2.,
        'source_failure_cost': source.reward_proxy.failure_cost,
        'target_failure_cost': target.reward_proxy.failure_cost,
        'source_transitions_verified': count, 'source_reward_max_abs_error': maximum_error,
        'real_rewards_recomputed': 0, 'changed_real_rewards': 0,
        'reward_delta_sum': 0., 'reward_delta_abs_sum': 0.,
        'reward_delta_min': None, 'reward_delta_max': None,
        'new_physical_interactions': 0, 'source_dataset_mutated': False,
        'relabeling_changes_reward_only': True,
        'existing_seeding_and_timeout_rules_unchanged': True,
        'hindsight_reward_provider': 'active target environment compute_reward',
        'schema_probes_inserted_into_replay': 0}


def seed_replay(model, path, *, object_labels=False, preparation_sources_by_mode=None,
                reward_relabel_gamma=None):
    """Import a trusted local dataset into a fresh vector HER replay buffer.

    Whole demonstration episodes are assigned to worker streams. Only real
    transitions are used. Unequal streams are cropped to the shortest stream;
    the final prefix of each unfinished episode gets a collector timeout, so
    subsequent online resets cannot connect unrelated physical trajectories.

    Opt-in reward_relabel_gamma=.999 verifies the original declared .995 carry
    rewards before importing recomputed real rewards. Native HER keeps using
    the active target environment. The default import path is unchanged.
    """
    if preparation_sources_by_mode is not None:
        if not object_labels:raise ValueError('Preparation labels require explicit original object labels')
        if not preparation_sources_by_mode or any(value not in ('reference','learned') for value in preparation_sources_by_mode.values()):
            raise ValueError('Declare known collection modes and canonical preparation families')
    path=Path(path).resolve()
    data=pickle.loads(path.read_bytes())
    if data.get('schema_version') != 1 or not data.get('episodes'):
        raise ValueError('Expected a schema-1 physical demonstration dataset')
    replay=model.replay_buffer
    if replay.pos or replay.full:
        raise ValueError('Demonstrations can only seed a fresh replay buffer')
    relabel_schema=relabel_report=None
    if reward_relabel_gamma is not None:
        relabel_schema,relabel_report=_discount_relabel_contract(model,data,reward_relabel_gamma)
    streams=[[] for _ in range(model.n_envs)]
    for episode in data['episodes']:
        transitions=episode['transitions']
        if not transitions:
            raise ValueError('Empty demonstration episode')
        if not (transitions[-1]['terminated'] or transitions[-1]['truncated']):
            raise ValueError('Dataset episodes must end explicitly')
        stream=min(streams,key=len)
        if object_labels:
            obj=episode.get('object_idx')
            if isinstance(obj,bool) or not isinstance(obj,(int,np.integer)) or not 0<=obj<15:
                raise ValueError('Object-balanced replay needs each real episode original object_idx')
            source=str(episode.get('collection_mode','demonstration'))
            if preparation_sources_by_mode is not None:
                if source not in preparation_sources_by_mode:raise ValueError('Undeclared preparation collection mode: '+source)
                family=preparation_sources_by_mode[source]
                if episode.get('preparation_source',family)!=family:raise ValueError('Episode preparation source conflicts with declared lineage')
            stream.extend((row,int(obj),source) for row in transitions)
        else:
            stream.extend(transitions)
    length=min(map(len,streams))
    if not length:
        raise ValueError('Need at least one physical episode per vector worker')
    if length>=replay.buffer_size:
        raise ValueError('Seed data must fit the configured replay capacity')
    action_calls=physics_ticks=collector_timeouts=0
    imported_objects={};imported_preparations={}
    for index in range(length):
        entries=[stream[index] for stream in streams]
        records=[entry[0] for entry in entries] if object_labels else entries
        observations={key:np.stack([record['observation'][key] for record in records]).astype(np.float32)
                      for key in model.observation_space.spaces}
        following={key:np.stack([record['next_observation'][key] for record in records]).astype(np.float32)
                   for key in model.observation_space.spaces}
        actions=np.stack([record['action'] for record in records]).astype(np.float32)
        if actions.shape != (model.n_envs,*model.action_space.shape) or not np.isfinite(actions).all() or np.max(np.abs(actions))>1.+1e-6:
            raise ValueError('Demonstration action shape/range mismatch')
        for key,space in model.observation_space.spaces.items():
            if observations[key].shape != (model.n_envs,*space.shape) or following[key].shape != observations[key].shape:
                raise ValueError('Demonstration observation shape mismatch')
            if not np.isfinite(observations[key]).all() or not np.isfinite(following[key]).all():
                raise ValueError('Nonfinite demonstration observation')
        infos=[dict(record['info']) for record in records]
        if object_labels:
            for info,(_,obj,source) in zip(infos,entries):
                if 'object_idx' in info and info['object_idx']!=obj:raise ValueError('Transition and episode object labels disagree')
                info.update(object_idx=obj,replay_source=source)
                if preparation_sources_by_mode is not None:
                    family=preparation_sources_by_mode[source]
                    if info.get('preparation_source',family)!=family:raise ValueError('Transition preparation source conflicts with declared lineage')
                    info['preparation_source']=family
                    key=f'{obj}/{family}';imported_preparations[key]=imported_preparations.get(key,0)+1
                imported_objects[str(obj)]=imported_objects.get(str(obj),0)+1
        dones=np.array([record['terminated'] or record['truncated'] for record in records],dtype=bool)
        for worker,(record,info) in enumerate(zip(records,infos)):
            timeout=bool(record['truncated'] and not record['terminated'])
            if index==length-1 and not dones[worker]:
                dones[worker]=True;timeout=True;collector_timeouts+=1
            info['TimeLimit.truncated']=timeout
            action_calls+=int(info['action_calls_delta'])
            physics_ticks+=int(info['physics_ticks_delta'])
        rewards=np.asarray([record['reward'] for record in records],dtype=np.float32)
        recomputed=model.env.env_method('compute_reward',following['achieved_goal'],observations['desired_goal'],infos,indices=[0])[0]
        if relabel_report is None:
            if not np.allclose(rewards,recomputed,atol=1e-5,rtol=0.):
                raise ValueError('Dataset reward differs from the active environment contract')
        else:
            target_rewards=relabel_schema.compute_reward(following['achieved_goal'],observations['desired_goal'],infos)
            if not np.allclose(recomputed,target_rewards,atol=1e-5,rtol=0.):
                raise ValueError('Active environment differs from declared target carry reward contract')
            target_rewards=np.asarray(recomputed,dtype=np.float32)
            delta=target_rewards.astype(np.float64)-rewards.astype(np.float64)
            relabel_report['real_rewards_recomputed']+=len(rewards)
            relabel_report['changed_real_rewards']+=int(np.count_nonzero(target_rewards!=rewards))
            relabel_report['reward_delta_sum']+=float(delta.sum())
            relabel_report['reward_delta_abs_sum']+=float(np.abs(delta).sum())
            low,high=float(delta.min()),float(delta.max())
            relabel_report['reward_delta_min']=low if relabel_report['reward_delta_min'] is None else min(low,relabel_report['reward_delta_min'])
            relabel_report['reward_delta_max']=high if relabel_report['reward_delta_max'] is None else max(high,relabel_report['reward_delta_max'])
            rewards=target_rewards
        replay.add(observations,following,actions,rewards,dones,infos)
    available=sum(len(stream) for stream in streams)
    report={'method':'physical_demonstration_replay_seed','actor_behavior_cloning':False,
        'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'dataset_metadata':data.get('metadata',{}),'episodes_collected':len(data['episodes']),
        'transitions_collected':available,'transitions_imported':length*model.n_envs,
        'transitions_omitted_from_replay':available-length*model.n_envs,
        'imported_source_action_calls':action_calls,'imported_physics_ticks':physics_ticks,
        'collector_timeouts':collector_timeouts,'object_labels_added':bool(object_labels),
        'imported_object_transition_counts':imported_objects,
        'preparation_sources_by_mode':preparation_sources_by_mode,
        'imported_preparation_transition_counts':imported_preparations,
        'teacher_at_evaluation':False,'hindsight_at_evaluation':False,'reward_roundoff_atol':1e-5}

    if relabel_report is not None:
        relabel_report['source_dataset_sha256']=report['sha256']
        relabel_report['reward_delta_mean']=relabel_report['reward_delta_sum']/relabel_report['real_rewards_recomputed']
        relabel_report['reward_delta_mean_abs']=relabel_report['reward_delta_abs_sum']/relabel_report['real_rewards_recomputed']
        report['reward_relabeling']=relabel_report
    return report


def _carry_supervision_single(dataset):
    """Return rows, disjoint index groups, probabilities and concise BC statistics.

    Failed episodes remain in the dataset/replay but supply no imitation labels.
    A recovery prefix lasts until the FIRST observed base x>3.5 m, using only
    actual stored achieved goals (scale20). This is training-only sampling
    metadata, not an additional actor input or an inference-time waypoint.
    Successful geometric interior recoveries reserve20% for measured westward
    return actions; all other group probabilities retain their prior proportions.
    """
    rows=[];prefix=[];interior_recovery=[];failed=0;successful=0
    for episode in dataset['episodes']:
        if not episode.get('success',False):
            failed+=1;continue
        successful+=1
        before_crossing=(episode.get('collection_mode')=='recovery' or
                         (episode.get('collection_mode')=='geometric-recovery' and episode.get('recovery_kind')=='entry_extreme'))
        interior=(episode.get('collection_mode')=='geometric-recovery' and episode.get('recovery_kind')=='interior_branch')
        for row in episode['transitions']:
            if before_crossing and float(row['observation']['achieved_goal'][0])*20>3.5:
                before_crossing=False
            rows.append(row);prefix.append(before_crossing);interior_recovery.append(interior)
    if not rows:raise ValueError('No successful carry demonstrations for supervision')
    actions=np.stack([row['teacher_action'] for row in rows])
    if actions.shape!=(len(rows),2) or not np.isfinite(actions).all() or np.max(np.abs(actions))>1.:
        raise ValueError('Carry teacher labels must be finite normalized two-dimensional actions')
    prefix=np.asarray(prefix,dtype=bool);norm=np.linalg.norm(actions,axis=1)
    base=[(norm>.5)&(np.abs(actions[:,1])>.15),(norm>.5)&(np.abs(actions[:,1])<=.15),norm<=.5]
    westward=(np.asarray(interior_recovery,dtype=bool)&(actions[:,0]<-.5)&
              (np.asarray([row['observation']['achieved_goal'][0] for row in rows])*20>3.5))
    groups={name:np.flatnonzero(mask&~prefix&~westward) for name,mask in zip(('turn_or_vertical','straight','hold'),base)}
    if any(not len(indices) for indices in groups.values()):
        raise ValueError('Carry supervision needs non-prefix turns/verticals, straights and holds')
    probabilities=np.array([.5,.35,.15],dtype=np.float64)
    if np.any(prefix):
        groups={'recovery_prefix':np.flatnonzero(prefix),**groups}
        probabilities=np.r_[.3,.7*probabilities]
    if np.any(westward):
        groups={'westward_return':np.flatnonzero(westward),**groups}
        probabilities=np.r_[.2,.8*probabilities]
    stats={'successful_episodes_for_bc':successful,'failed_episodes_excluded_from_bc':failed,
           'all_episodes_retained_for_replay':True,'bc_transitions':len(rows),
           'westward_return_rows':int(westward.sum()),
           'westward_return_rule':'successful geometric interior recovery, teacher x<-.5 and observed base x>3.5m',
           'recovery_prefix_rows':int(prefix.sum()),'prefix_end_rule':'first observed base x>3.5m',
           'strata_counts':{name:len(indices) for name,indices in groups.items()},
           'strata_probabilities':dict(zip(groups,probabilities.tolist()))}
    return rows,groups,probabilities,stats


def carry_supervision(dataset, *, balance_objects=False):
    """Optionally balance objects before the established direction strata.

    Object identity is TRAIN sampling metadata only, never a new policy input.
    Failed episodes remain replay data and cannot supply imitation labels. A
    declared object with no successful labels is an explicit error, not silently
    removed from the sampling distribution.
    """
    if not balance_objects:
        return _carry_supervision_single(dataset)
    objects=sorted({int(episode['object_idx']) for episode in dataset['episodes']})
    if not objects:raise ValueError('No object cases for balanced carry supervision')
    rows=[];groups={};probabilities=[];per_object={}
    counters=('successful_episodes_for_bc','failed_episodes_excluded_from_bc',
              'bc_transitions','westward_return_rows','recovery_prefix_rows')
    stats={key:0 for key in counters}
    for object_idx in objects:
        subset={'episodes':[episode for episode in dataset['episodes'] if int(episode['object_idx'])==object_idx]}
        part_rows,part_groups,part_probabilities,part_stats=_carry_supervision_single(subset)
        offset=len(rows);rows.extend(part_rows)
        for name,indices in part_groups.items():
            groups[f'object_{object_idx:02d}/{name}']=indices+offset
        probabilities.extend(part_probabilities/len(objects))
        per_object[str(object_idx)]=part_stats
        for key in counters:stats[key]+=part_stats[key]
    probabilities=np.asarray(probabilities,dtype=np.float64)
    stats.update(object_balance=True,object_indices=objects,
        object_sampling_probabilities={str(obj):1/len(objects) for obj in objects},
        per_object=per_object,all_episodes_retained_for_replay=True,
        object_id_added_to_policy_observation=False,
        strata_counts={name:len(indices) for name,indices in groups.items()},
        strata_probabilities=dict(zip(groups,probabilities.tolist())))
    return rows,groups,probabilities,stats


def merge_carry_datasets(paths,destination,*,bank_path=None,parent_bank_paths=(),object_indices=None):
    """Create an immutable counted carry-data union; return a small manifest.

    Accepts only matching physical TRAIN data under the active580-state carry
    reward/action contract. Sources are kept byte-for-byte, including failed
    episodes and nested failed-attempt metadata. Reusing the same source file
    under another name is rejected by its content hash. No simulator is opened.
    A larger TRAIN bank requires explicit parent manifests whose unchanged
    records, case RNG and validation/test partitions are verified in full.
    """
    import os
    import tempfile
    from .multi_carry import CarrySchema
    from .scenes import load_scene_bank,scene_records,verify_training_extension
    paths=[Path(path).resolve() for path in paths];destination=Path(destination).resolve()
    if len(paths)<2 or len(set(paths))!=len(paths):raise ValueError('Need at least two distinct carry source datasets')
    if destination in paths or destination.exists():raise FileExistsError('Merge destination must be new and separate from inputs')
    bank=load_scene_bank(bank_path or Path(__file__).resolve().parents[1]/'configs/phase2_scene_bank.json')
    compatible_banks={bank['manifest_sha256']:bank}
    for parent_path in parent_bank_paths:
        parent=load_scene_bank(parent_path)
        verify_training_extension(parent,bank)
        compatible_banks[parent['manifest_sha256']]=parent
    declared_objects=None if object_indices is None else sorted(int(i) for i in object_indices)
    if declared_objects is not None and (not declared_objects or len(set(declared_objects))!=len(declared_objects)
            or any(not 0<=i<15 for i in declared_objects)):
        raise ValueError('Declare distinct original object indices for an explicit carry-data union')
    observed_objects=set()
    schema=CarrySchema();merged=[];lineage=[];seen=set();reference=None
    costs={key:0 for key in ('source_action_calls','physics_ticks','preparation_source_action_calls',
        'preparation_physics_ticks','setup_source_action_calls','setup_physics_ticks','initialization_physics_ticks')}
    source_runs=[]
    for path in paths:
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if sha in seen:raise ValueError('Duplicate source content would double-count demonstrations and costs')
        seen.add(sha);data=pickle.loads(raw);metadata=data.get('metadata',{})
        if data.get('schema_version')!=1 or not data.get('episodes'):raise ValueError('Expected nonempty schema-1 carry data')
        if metadata.get('partition')!='train':raise ValueError('Only TRAIN demonstrations may be merged')
        contract={key:metadata.get(key) for key in ('gamma','shaping_weight','delta_scale','repeat')}
        expected={'gamma':.995,'shaping_weight':2.,'delta_scale':.075,'repeat':5}
        if contract!=expected:raise ValueError('Carry action/gamma/reward contract mismatch')
        objects=sorted({int(episode['object_idx']) for episode in data['episodes']})
        if declared_objects is not None and not set(objects)<=set(declared_objects):
            raise ValueError('Carry source contains an object outside the declared object union')
        observed_objects.update(objects)
        contract['object_indices']=objects if declared_objects is None else declared_objects
        if reference is not None and contract!=reference:raise ValueError('Carry source action/reward/object contracts differ')
        reference=contract
        declared_bank=metadata.get('bank_sha256')
        if declared_bank is not None and declared_bank not in compatible_banks:
            raise ValueError('Carry source uses a different frozen scene bank; supply a verified parent bank explicitly')
        source_bank=compatible_banks.get(declared_bank,bank)
        train={layout['layout_id']:layout for layout in scene_records(source_bank,'train')}
        transitions=rollout_calls=rollout_ticks=0
        for episode in data['episodes']:
            layout=train.get(episode.get('layout_id'))
            if layout is None or layout['maze_hash']!=episode.get('maze_hash'):
                raise ValueError('Episode is outside the frozen TRAIN geometry partition')
            rows=episode.get('transitions',[])
            if not rows or not (rows[-1]['terminated'] or rows[-1]['truncated']):
                raise ValueError('Carry episode lacks a closed transition boundary')
            goal=np.asarray(rows[0]['observation']['desired_goal'])
            wall=np.asarray(layout['maze_map'],dtype=np.float32).reshape(-1)
            for index,row in enumerate(rows):
                for label in ('action','teacher_action'):
                    action=np.asarray(row[label])
                    if action.shape!=(2,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.:
                        raise ValueError('Carry action labels must be finite normalized two-dimensional controls')
                for label in ('observation','next_observation'):
                    obs=row[label]
                    if set(obs)!=set(schema.observation_space.spaces):raise ValueError('Carry observation keys differ')
                    for key,space in schema.observation_space.spaces.items():
                        if np.asarray(obs[key]).shape!=space.shape or not np.isfinite(obs[key]).all():
                            raise ValueError('Carry observation shape/values differ')
                    if not np.array_equal(obs['desired_goal'],goal):raise ValueError('Goal changed within carry episode')
                    if not np.array_equal(np.asarray(obs['observation'])[:264],wall):
                        raise ValueError('Stored wall observations differ from declared TRAIN map')
                if index<len(rows)-1:
                    if row['terminated'] or row['truncated']:raise ValueError('Episode contains an internal terminal boundary')
                    if not all(np.array_equal(row['next_observation'][key],rows[index+1]['observation'][key]) for key in row['next_observation']):
                        raise ValueError('Adjacent physical carry transitions are discontinuous')
                transitions+=1;rollout_calls+=int(row['info']['action_calls_delta']);rollout_ticks+=int(row['info']['physics_ticks_delta'])
            achieved=np.stack([row['next_observation']['achieved_goal'] for row in rows])
            reward=schema.compute_reward(achieved,np.broadcast_to(goal,achieved.shape),[row['info'] for row in rows])
            if not np.allclose(reward,[row['reward'] for row in rows],atol=1e-5,rtol=0):
                raise ValueError('Stored carry rewards differ from active HER reward contract')
        if rollout_calls!=metadata.get('source_action_calls') or rollout_ticks!=metadata.get('physics_ticks'):
            raise ValueError('Source rollout physical accounting differs from recorded transitions')
        for key in costs:
            value=metadata.get(key,0)
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<0:
                raise ValueError('Physical cost must be a nonnegative integer: '+key)
            costs[key]+=int(value)
        source_runs.extend(metadata.get('source_runs',[path.parent.parent.name]))
        lineage.append({'path':str(path),'sha256':sha,'episodes':len(data['episodes']),'transitions':transitions,
            'source_bank_sha256':source_bank['manifest_sha256'],
            'verified_parent_extension':source_bank['manifest_sha256']!=bank['manifest_sha256'],
            'bank_validation':'declared hash plus all TRAIN maps' if declared_bank else 'reconstructed from all frozen TRAIN layout IDs/maps',
            'metadata':metadata})
        merged.extend(data['episodes'])
    if declared_objects is not None and sorted(observed_objects)!=declared_objects:
        raise ValueError('Declared carry object union is missing source data for an object')
    metadata={'protocol':'immutable-carry-dataset-merge-v1','partition':'train','bank_sha256':bank['manifest_sha256'],
        **{key:reference[key] for key in ('gamma','shaping_weight','delta_scale','repeat')},
        'object_indices':reference['object_indices'],'source_runs':list(dict.fromkeys(source_runs)),
        'source_datasets':lineage,'verified_parent_bank_sha256s':[sha for sha in compatible_banks if sha!=bank['manifest_sha256']],
        'new_physical_interactions':0,'episodes':len(merged),
        'transitions':sum(len(episode['transitions']) for episode in merged),**costs,
        'all_failed_attempt_metadata_retained':True,'teacher_at_evaluation':False,
        'actual_actions_preserved':True,'teacher_labels_for_BC_only':True}
    metadata['all_source_action_calls']=sum(costs[key] for key in ('source_action_calls','preparation_source_action_calls','setup_source_action_calls'))
    metadata['all_physics_ticks']=sum(costs[key] for key in ('physics_ticks','preparation_physics_ticks','setup_physics_ticks','initialization_physics_ticks'))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',prefix='.carry-merge-',dir=destination.parent,delete=False) as stream:
            temporary=Path(stream.name)
            pickle.dump({'schema_version':1,'metadata':metadata,'episodes':merged},stream,protocol=5)
            stream.flush();os.fsync(stream.fileno())
        os.link(temporary,destination)  # atomic publication, refusing an existing destination
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return {'path':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'episodes':metadata['episodes'],'transitions':metadata['transitions'],'source_datasets':len(lineage),
            'new_physical_interactions':0,'source_runs':metadata['source_runs'],**costs,
            'all_source_action_calls':metadata['all_source_action_calls'],'all_physics_ticks':metadata['all_physics_ticks']}
