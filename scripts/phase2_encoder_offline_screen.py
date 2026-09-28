#!/usr/bin/env python3
"""Prespecified matched offline BC comparison; no simulator or online updates.

Only recorded TRAIN demonstrations are accessed. The last32 TRAIN maps form
an internal development fold; this is not an official held-out evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import SAC
from pnr.demonstrations import carry_supervision
from pnr.gated_spatial_features import GatedCarrySpatialExtractor
from pnr.multi_carry import CarrySchema
from pnr.spatial_features import CarrySpatialExtractor

DATASET_SHA = '42eb7f4673ca3bfc86882d25e868222c78b36e0b158d124f340280c14d2654b3'
SEED = 19041
STEPS, BATCH = 16000, 256
CHECKS = (0, 4000, 8000, 16000)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value, *, exclusive=False):
    if exclusive:
        with Path(path).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        temporary = Path(str(path) + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)


def masks(actions, achieved):
    norm = np.linalg.norm(actions, axis=1)
    return {
        'all': np.ones(len(actions), dtype=bool),
        'turn_or_vertical': (norm > .5) & (np.abs(actions[:, 1]) > .15),
        'straight': (norm > .5) & (np.abs(actions[:, 1]) <= .15),
        'hold': norm <= .5,
        'westward_return': (actions[:, 0] < -.5) & (achieved[:, 0] * 20 > 3.5),
    }


def score(prediction, target):
    if not len(target):
        return {'rows': 0, 'mse': None, 'mae': None,
                'within_0_2_both_axes': None, 'dominant_direction_accuracy': None,
                'strong_axis_sign_accuracy': None, 'hold_accuracy': None}
    moving = np.linalg.norm(target, axis=1) > .5
    axis = np.argmax(np.abs(target), axis=1)
    predicted_axis = np.argmax(np.abs(prediction), axis=1)
    chosen = np.arange(len(target))
    direction = ((axis == predicted_axis)
                 & (np.sign(prediction[chosen, axis]) == np.sign(target[chosen, axis]))
                 & (np.max(np.abs(prediction), axis=1) >= .1))
    strong = np.abs(target) >= .5
    return {
        'rows': len(target),
        'mse': float(np.square(prediction.astype(np.float64) - target).mean()),
        'mae': float(np.abs(prediction.astype(np.float64) - target).mean()),
        'within_0_2_both_axes': float((np.max(np.abs(prediction - target), axis=1) <= .2).mean()),
        'dominant_direction_accuracy': float(direction[moving].mean()) if moving.any() else None,
        'strong_axis_sign_accuracy': float((np.sign(prediction[strong]) == np.sign(target[strong])).mean()) if strong.any() else None,
        'hold_accuracy': float((np.linalg.norm(prediction[~moving], axis=1) <= .5).mean()) if (~moving).any() else None,
    }


def summarize(prediction, target, achieved, layout_ids):
    strata = masks(target, achieved)
    by_map = {}
    for layout in np.unique(layout_ids):
        selected = layout_ids == layout
        by_map[str(layout)] = {name: score(prediction[selected & mask], target[selected & mask])
                               for name, mask in strata.items()}
    macro = {}
    for name in strata:
        macro[name] = {}
        for metric in ('mse', 'mae', 'within_0_2_both_axes', 'dominant_direction_accuracy',
                       'strong_axis_sign_accuracy', 'hold_accuracy'):
            values = [value[name][metric] for value in by_map.values() if value[name][metric] is not None]
            macro[name][metric] = float(np.mean(values)) if values else None
        macro[name]['maps_with_rows'] = sum(value[name]['rows'] > 0 for value in by_map.values())
    return {'row_weighted': {name: score(prediction[mask], target[mask]) for name, mask in strata.items()},
            'map_macro': macro, 'by_map': by_map}


def load_fold(episodes, *, balance_objects):
    rows, groups, probabilities, stats = carry_supervision({'episodes': episodes}, balance_objects=balance_objects)
    identities = {id(row): (episode['layout_id'], episode['object_idx'])
                  for episode in episodes if episode['success'] for row in episode['transitions']}
    observations = {key: np.stack([row['observation'][key] for row in rows]).astype(np.float32)
                    for key in ('observation', 'achieved_goal', 'desired_goal')}
    target = np.stack([row['teacher_action'] for row in rows]).astype(np.float32)
    layouts = np.array([identities[id(row)][0] for row in rows])
    objects = np.array([identities[id(row)][1] for row in rows], dtype=np.int16)
    assert all(np.isfinite(value).all() for value in observations.values())
    return observations, target, layouts, objects, groups, probabilities, stats


def sample_schedule(groups, probabilities, steps=STEPS, batch=BATCH, seed=SEED):
    rng = np.random.default_rng(seed)
    result = np.empty((steps, batch), dtype=np.int32)
    for index in range(steps):
        counts = rng.multinomial(batch, probabilities)
        result[index] = np.concatenate([rng.choice(group, count, replace=True)
                                       for group, count in zip(groups.values(), counts)])
    return result


def paired_map_intervals(first, second):
    """Descriptive paired map bootstrap, conditional on this one trained seed."""
    rng = np.random.default_rng(SEED + 19)
    result = {}
    for stratum in ('all', 'turn_or_vertical', 'straight', 'westward_return'):
        result[stratum] = {}
        for metric in ('mse', 'dominant_direction_accuracy'):
            differences = np.array([second[k][stratum][metric] - first[k][stratum][metric]
                                    for k in sorted(first)
                                    if first[k][stratum][metric] is not None and second[k][stratum][metric] is not None])
            if not len(differences):
                continue
            draws = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
            result[stratum][metric] = {'gated_minus_cnn': float(differences.mean()),
                'map_bootstrap_95_percent_interval': np.quantile(draws, [.025, .975]).tolist(),
                'maps': len(differences)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='phase2_encoder_offline_screen01')
    args = parser.parse_args()
    destination = ROOT / 'runs' / args.run_id
    if destination.exists():
        parser.error('Preserve prior evidence: provide a new run ID')
    destination.mkdir(parents=True)
    artifacts = destination / 'artifacts'; artifacts.mkdir()
    checkpoints = destination / 'checkpoints'; checkpoints.mkdir()
    source_dir = destination / 'source'; source_dir.mkdir()
    started = time.monotonic()
    status = {'status': 'preparing', 'created_at': datetime.now(timezone.utc).isoformat(),
              'pid': os.getpid(), 'physics_ticks': 0, 'source_action_calls': 0, 'online_sac_updates': 0}
    write_json(destination / 'status.json', status)
    try:
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('Declared bounded CUDA comparison requires CUDA')
        dataset_path = ROOT / 'runs/phase2_carry_learned_mix01/artifacts/demonstrations.pkl'
        if digest(dataset_path) != DATASET_SHA:
            raise ValueError('Declared immutable503901-row dataset has changed')
        # Deliberately inspect only TRAIN records; no official validation/test/reserve geometry.
        bank_path = ROOT / 'configs/phase2_scene_bank_expanded.json'
        bank = json.loads(bank_path.read_text())
        train_records = bank['partitions']['train']
        assert [r['layout_id'] for r in train_records] == [f'train_{i:03d}' for i in range(192)]
        train_hashes = {r['layout_id']: r['maze_hash'] for r in train_records}
        fit_ids = [f'train_{i:03d}' for i in range(160)]
        dev_ids = [f'train_{i:03d}' for i in range(160, 192)]
        with dataset_path.open('rb') as stream:
            dataset = pickle.load(stream)
        assert dataset['metadata']['partition'] == 'train'
        assert sum(len(e['transitions']) for e in dataset['episodes']) == 503901
        for episode in dataset['episodes']:
            if episode['layout_id'] not in train_hashes or episode['maze_hash'] != train_hashes[episode['layout_id']]:
                raise ValueError('Dataset episode is not a declared TRAIN map')
        fit_episodes = [e for e in dataset['episodes'] if e['layout_id'] in set(fit_ids)]
        dev_episodes = [e for e in dataset['episodes'] if e['layout_id'] in set(dev_ids)]
        fit = load_fold(fit_episodes, balance_objects=True)
        dev = load_fold(dev_episodes, balance_objects=True)
        schedule = sample_schedule(fit[4], fit[5])
        np.save(artifacts / 'paired_minibatch_indices.npy', schedule, allow_pickle=False)
        rng = np.random.default_rng(SEED + 1)
        fit_eval_indices = np.concatenate([rng.choice(np.flatnonzero(fit[2] == layout), 64, replace=False)
                                           for layout in fit_ids])
        np.savez(artifacts / 'row_identity.npz', fit_layout=fit[2], fit_object=fit[3],
                 dev_layout=dev[2], dev_object=dev[3], fit_eval_indices=fit_eval_indices)
        sources = {}
        for relative in ('scripts/phase2_encoder_offline_screen.py', 'pnr/demonstrations.py',
                         'pnr/spatial_features.py', 'pnr/navigation_features.py',
                         'pnr/gated_spatial_features.py', 'pnr/multi_carry.py'):
            target = source_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
            sources[relative] = digest(target)
        config = {'protocol': 'matched-offline-encoder-development-screen-v1',
            'created_at': datetime.now(timezone.utc).isoformat(), 'dataset_path': str(dataset_path),
            'dataset_sha256': DATASET_SHA, 'source_rows': 503901, 'source_episodes': len(dataset['episodes']),
            'dataset_physics_reused_not_new': {key: dataset['metadata'].get(key) for key in
                ('source_action_calls', 'physics_ticks', 'preparation_source_action_calls', 'preparation_physics_ticks',
                 'setup_source_action_calls', 'setup_physics_ticks', 'initialization_physics_ticks')},
            'new_physics_ticks': 0, 'official_validation_or_heldout_policy_evaluations': 0,
            'bank_path': str(bank_path), 'bank_manifest_sha256': bank['manifest_sha256'],
            'fit_layout_ids': fit_ids, 'dev_layout_ids': dev_ids,
            'train_map_hashes': train_hashes, 'seed': SEED, 'bc_updates_per_model': STEPS,
            'batch_size': BATCH, 'optimizer': 'Adam', 'learning_rate': 3e-4, 'gradient_clip_norm': 10.,
            'loss': 'mean squared error of deterministic tanh continuous action versus real teacher label',
            'check_updates': list(CHECKS), 'selection': 'compare fixed final16000; no best-intermediate selection',
            'sampler': 'object-balanced existing carry_supervision strata; multinomial group counts and uniform rows with replacement',
            'paired_batch_schedule_sha256': digest(artifacts / 'paired_minibatch_indices.npy'),
            'row_identity_sha256': digest(artifacts / 'row_identity.npz'),
            'fit_supervision': fit[6], 'dev_supervision': dev[6],
            'fit_metric_rows_per_map': 64, 'dev_metric_rows': 'all successful recorded rows',
            'observation': 'unchanged580-state measured map and goals; raw48pose masked; compact9grip retained',
            'action_oracle_at_inference': False, 'her_label_changes': False,
            'architectures': {'cnn32': {'channels': 32}, 'gated48_k20_f5': {'channels': 48, 'iterations': 20, 'kernel_size': 5}},
            'features_dim': 128, 'actor_head': [128, 128], 'initialization': 'fresh independent architecture weights with common SAC seed; no pretrained transfer',
            'metrics': {'mse': 'mean squared normalized action error over both axes',
                'mae': 'mean absolute normalized action error over both axes',
                'within_0_2_both_axes': 'fraction with max absolute axis error <=0.2',
                'dominant_direction_accuracy': 'on teacher norm>0.5: same largest-magnitude axis and sign, predicted max magnitude>=0.1; ties choose x',
                'strong_axis_sign_accuracy': 'elementwise sign match on teacher axis magnitude>=0.5',
                'hold_accuracy': 'predicted norm<=0.5 on teacher norm<=0.5',
                'turn_or_vertical': 'teacher norm>0.5 and abs(y)>0.15',
                'straight': 'teacher norm>0.5 and abs(y)<=0.15',
                'westward_return': 'teacher x<-0.5 and measured achieved x*20>3.5m',
                'aggregation': 'all rows and equal-weight macro over maps with stratum rows',
                'uncertainty': 'paired map bootstrap10000 draws, seed+19; conditional on one training seed, no rollout implication'},
            'limitations': ['Internal TRAIN dev fold is not official validation or an unseen-test result.',
                'Dev fold supplies reference foam demonstrations only; no five-object retention claim.',
                'Teacher-state action accuracy does not establish autonomous long-horizon recovery.',
                'Single training seed; GPU float32 non-bitwise numerics; shared machine timings.'],
            'numerics': 'float32; matmul high; cuDNN benchmark true and deterministic false',
            'gpu_name': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda, 'cpu_threads': 4, 'sources': sources}
        write_json(artifacts / 'frozen_config.json', config, exclusive=True)
        frozen_sha = digest(artifacts / 'frozen_config.json')
        print(json.dumps({'event': 'frozen_before_optimizers', 'config_sha256': frozen_sha,
                          'fit_rows': len(fit[1]), 'dev_rows': len(dev[1]), 'fit_objects': np.unique(fit[3]).tolist(),
                          'dev_objects': np.unique(dev[3]).tolist()}), flush=True)
        # Release the1.5GB nested replay objects; compact arrays are the immutable inputs below.
        del dataset, fit_episodes, dev_episodes, bank, train_records
        gc.collect()
        fit_obs = {key: torch.as_tensor(value, device='cuda') for key, value in fit[0].items()}
        fit_target = torch.as_tensor(fit[1], device='cuda')
        dev_obs = {key: torch.as_tensor(value, device='cuda') for key, value in dev[0].items()}
        schedule_tensor = torch.as_tensor(schedule.astype(np.int64), device='cuda')
        results = {'protocol': config['protocol'], 'frozen_config_sha256': frozen_sha, 'models': {},
                   'physics_ticks': 0, 'source_action_calls': 0, 'online_sac_updates': 0}
        for name, extractor, kwargs in (
            ('cnn32', CarrySpatialExtractor, {'channels': 32}),
            ('gated48_k20_f5', GatedCarrySpatialExtractor, {'channels': 48, 'iterations': 20, 'kernel_size': 5})):
            model = SAC('MultiInputPolicy', CarrySchema(), buffer_size=8, batch_size=BATCH,
                        seed=SEED, device='cuda', policy_kwargs={'net_arch': [128, 128],
                        'features_extractor_class': extractor, 'features_extractor_kwargs': {'features_dim': 128, **kwargs}})
            torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            model.actor.features_extractor.set_carry_pose_mask(True)
            optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
            record = {'architecture': kwargs, 'encoder_parameters': sum(p.numel() for p in model.actor.features_extractor.parameters()),
                      'actor_parameters': sum(p.numel() for p in model.actor.parameters()), 'evaluations': [],
                      'training_wall_seconds': 0., 'evaluation_wall_seconds': 0., 'bc_updates': 0,
                      'optimizer_steps': 0, 'model_online_timesteps': 0, 'model_online_updates': 0}
            results['models'][name] = record
            torch.cuda.reset_peak_memory_stats()

            def evaluate(update):
                begin = time.monotonic()
                model.policy.set_training_mode(False)
                scores = {}
                for label, observations, target, layouts, achieved, indices in (
                    ('fit_sample', fit_obs, fit[1][fit_eval_indices], fit[2][fit_eval_indices], fit[0]['achieved_goal'][fit_eval_indices], fit_eval_indices),
                    ('dev_all', dev_obs, dev[1], dev[2], dev[0]['achieved_goal'], np.arange(len(dev[1])))):
                    predictions = []
                    with torch.inference_mode():
                        for offset in range(0, len(indices), BATCH):
                            selected = torch.as_tensor(indices[offset:offset+BATCH], device='cuda')
                            predictions.append(model.actor({key: value[selected] for key, value in observations.items()}, deterministic=True).cpu().numpy())
                    prediction = np.concatenate(predictions)
                    assert np.isfinite(prediction).all()
                    scores[label] = summarize(prediction, target, achieved, layouts)
                    if update == STEPS:
                        np.save(artifacts / f'{name}_{label}_final_actions.npy', prediction, allow_pickle=False)
                model.save(checkpoints / f'{name}_bc_{update:05d}.zip')
                checkpoint = checkpoints / f'{name}_bc_{update:05d}.zip'
                scores.update(bc_updates=update, checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint))
                record['evaluations'].append(scores)
                record['evaluation_wall_seconds'] += time.monotonic() - begin
                write_json(artifacts / 'results.json', results)
                print(json.dumps({'event': 'evaluation', 'model': name, 'bc_updates': update,
                                  'fit': scores['fit_sample']['row_weighted'], 'dev': scores['dev_all']['row_weighted']}), flush=True)
                model.policy.set_training_mode(True)

            evaluate(0)
            block_start = time.monotonic(); block_losses = []
            for update in range(1, STEPS + 1):
                selected = schedule_tensor[update - 1]
                prediction = model.actor({key: value[selected] for key, value in fit_obs.items()}, deterministic=True)
                loss = torch.square(prediction - fit_target[selected]).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.)
                optimizer.step()
                block_losses.append(loss.detach())
                record['bc_updates'] = record['optimizer_steps'] = update
                if update % 1000 == 0:
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - block_start
                    record['training_wall_seconds'] += elapsed
                    mean_loss = float(torch.stack(block_losses).mean())
                    if not np.isfinite(mean_loss):
                        raise FloatingPointError(f'{name} nonfiniteBC loss')
                    status.update(status='running', model=name, bc_updates=update, mean_last1000_loss=mean_loss,
                                  wall_seconds=time.monotonic()-started, frozen_config_sha256=frozen_sha)
                    write_json(destination / 'status.json', status)
                    print(json.dumps({'event': 'training', 'model': name, 'updates': update,
                                      'last1000_seconds': elapsed, 'loss': mean_loss}), flush=True)
                    block_losses = []
                    if update in CHECKS:
                        evaluate(update)
                    block_start = time.monotonic()
            record['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            write_json(artifacts / 'results.json', results)
            model.env.close(); del model, optimizer, prediction, loss
            gc.collect(); torch.cuda.empty_cache()
        first = results['models']['cnn32']['evaluations'][-1]['dev_all']['by_map']
        second = results['models']['gated48_k20_f5']['evaluations'][-1]['dev_all']['by_map']
        results['paired_final_dev_map_difference'] = paired_map_intervals(first, second)
        results['wall_seconds'] = time.monotonic() - started
        results['status'] = 'complete'
        write_json(artifacts / 'results.json', results)
        status.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=results['wall_seconds'])
        write_json(destination / 'status.json', status)
        print(json.dumps({'event': 'complete', 'wall_seconds': results['wall_seconds'],
                          'paired_final_dev_map_difference': results['paired_final_dev_map_difference']}), flush=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), wall_seconds=time.monotonic()-started)
        write_json(destination / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
