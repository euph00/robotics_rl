#!/usr/bin/env python3
"""Record hand-policy episodes on the rig (grasp, lift, carry) as dashboard videos.

  python scripts/record_hand.py --run hand_r_bc03 --step 6000016 --objects 2 7 10 --output runs/hand_recordings01
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pnr.hand_env import HandGraspEnv  # noqa: E402  (sets the headless rendering defaults first)
import imageio  # noqa: E402
import numpy as np  # noqa: E402

from pnr.telemetry import RunLogger  # noqa: E402
from pnr.video import webm_copy  # noqa: E402

NAMES = {14: 'foam brick', 5: 'banana', 2: 'soup can', 7: 'apple', 10: 'mug'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--step', required=True)
    parser.add_argument('--objects', type=int, nargs='+', default=[2, 7, 10])
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--physics', default='realistic')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    from stable_baselines3 import PPO
    base = ROOT / 'runs' / args.run / 'checkpoints' / f'step_{int(args.step):08d}'
    policy = PPO.load(str(base) + '.zip', device='cpu')
    stats = pickle.loads(Path(str(base) + '_vecnormalize.pkl').read_bytes())
    size = len(stats.obs_rms.mean)
    normalize = lambda o: np.clip((o[:size] - stats.obs_rms.mean) / np.sqrt(stats.obs_rms.var + stats.epsilon),
                                  -stats.clip_obs, stats.clip_obs).astype(np.float32)
    env = HandGraspEnv(objects=args.objects, seed=args.seed, physics=args.physics, carry=True, horizon=args.horizon)
    (args.output / 'artifacts').mkdir(parents=True, exist_ok=False)
    logger = RunLogger(args.output.name, root=args.output.parent, kind='recording',
                       name=args.label or f'Hand policy · {args.run} {args.step}', summary=f'{len(args.objects)} episodes')
    results = []
    for k, idx in enumerate(args.objects):
        obs, _ = env.reset(seed=args.seed + k, options={'object_idx': idx})
        frames, info = [], {}
        while True:
            action, _ = policy.predict(normalize(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            frames.append(env.rig.render(width=640, height=480, distance=.7, azimuth=140, elevation=-20))
            if terminated or truncated:
                break
        name = f'object_{idx:02d}'
        video = args.output / 'artifacts' / f'{name}.mp4'
        imageio.mimwrite(video, frames, fps=30, quality=7, macro_block_size=8)
        webm_copy(video)
        imageio.imwrite(args.output / 'artifacts' / f'{name}_preview.png', frames[min(len(frames) - 1, int(len(frames) * .45))])
        outcome = 'carried' if info.get('carried') else ('lifted' if info.get('ever_success') else 'failed')
        logger.artifact(f'artifacts/{name}.webm', f'{NAMES.get(idx, idx)} · {outcome}', kind='video')
        results.append({'object_idx': idx, 'outcome': outcome, 'steps': len(frames)})
        print(json.dumps(results[-1]), flush=True)
    logger.finish(summary=', '.join(f"{NAMES.get(r['object_idx'], r['object_idx'])} {r['outcome']}" for r in results))


if __name__ == '__main__':
    main()
