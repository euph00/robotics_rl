#!/usr/bin/env python3
"""Collect scripted power-grasp demonstrations on the hand rig (pnr/hand_teacher.py).

Each episode runs the teacher from a normal start (optionally with Gaussian action noise, so
the data covers recoveries) through the lift and, with --carry, the carry phase. Saves
observations, teacher actions (the clean action at noisy steps, as labels) and per-episode
outcomes; imitation uses successful episodes.

  python scripts/hand_teacher_demos.py --objects 2 7 10 --episodes 200 --output runs/hand_demos01
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402


def collect(job):
    idx, seeds, noise_levels, carry, physics = job
    from pnr.hand_env import HandGraspEnv
    from pnr.hand_teacher import PowerGraspTeacher
    env = HandGraspEnv(objects=[idx], seed=0, physics=physics, carry=carry, horizon=160)
    episodes = []
    for seed, noise in zip(seeds, noise_levels):
        obs, _ = env.reset(seed=int(seed), options={'object_idx': idx})
        teacher = PowerGraspTeacher(env, rng=np.random.default_rng(int(seed)))
        rng = np.random.default_rng(int(seed) + 1)
        observations, labels, success, carried = [], [], False, False
        while True:
            label = teacher.act()
            executed = np.clip(label + rng.normal(0, noise, label.shape), -1, 1).astype(np.float32) if noise else label
            observations.append(obs)
            labels.append(label)
            obs, reward, terminated, truncated, info = env.step(executed)
            success |= bool(info.get('success'))
            carried |= bool(info.get('carried'))
            if terminated or truncated:
                break
        episodes.append({'object_idx': idx, 'seed': int(seed), 'noise': float(noise), 'lifted': success,
                         'carried': carried, 'observations': np.asarray(observations, np.float32),
                         'actions': np.asarray(labels, np.float32)})
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--objects', type=int, nargs='+', default=[2, 7, 10])
    parser.add_argument('--episodes', type=int, default=200, help='Per object')
    parser.add_argument('--noise', type=float, nargs='+', default=[0., .1, .2], help='Cycled per episode')
    parser.add_argument('--carry', action='store_true')
    parser.add_argument('--physics', default='realistic')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'{args.output} exists')
    jobs = []
    for idx in args.objects:
        seeds = np.arange(args.episodes) + 20_000 + 1000 * idx
        noise = np.resize(args.noise, args.episodes)
        for chunk_seeds, chunk_noise in zip(np.array_split(seeds, max(1, args.workers // len(args.objects))),
                                            np.array_split(noise, max(1, args.workers // len(args.objects)))):
            jobs.append((idx, chunk_seeds, chunk_noise, args.carry, args.physics))
    with ProcessPoolExecutor(args.workers) as pool:
        episodes = [e for chunk in pool.map(collect, jobs) for e in chunk]
    args.output.mkdir(parents=True)
    keep = [e for e in episodes if e['lifted'] and (e['carried'] or not args.carry)]
    np.savez_compressed(args.output / 'demonstrations.npz',
                        observations=np.concatenate([e['observations'] for e in keep]),
                        actions=np.concatenate([e['actions'] for e in keep]),
                        object_idx=np.concatenate([[e['object_idx']] * len(e['actions']) for e in keep]))
    summary = {'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               'episodes': len(episodes), 'kept': len(keep), 'transitions': int(sum(len(e['actions']) for e in keep)),
               'per_object': {str(i): {'lifted': sum(e['lifted'] for e in episodes if e['object_idx'] == i),
                                       'carried': sum(e['carried'] for e in episodes if e['object_idx'] == i),
                                       'episodes': sum(e['object_idx'] == i for e in episodes)} for i in args.objects}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
