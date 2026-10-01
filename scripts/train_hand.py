#!/usr/bin/env python3
"""Train an Allegro grasp-lift-hold policy with SAC on the hand rig (pnr/hand_env.py).

  python scripts/train_hand.py --run-id hand_sac_mu025 --steps 3000000 --envs 20
  python scripts/train_hand.py --run-id hand_sac_mu100 --friction 1.0 ...

Evaluates the deterministic policy every --eval-every transitions on --eval-episodes
episodes per object and logs success, lift and contact statistics to runs/<run-id>/.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np  # noqa: E402

from pnr.hand_env import HandGraspEnv  # noqa: E402
from pnr.telemetry import RunLogger, atomic_json  # noqa: E402


def make_env(rank, args):
    def thunk():
        return HandGraspEnv(objects=args.objects, horizon=args.horizon, seed=args.seed * 1000 + rank,
                            friction=args.friction, finger_kp=args.finger_kp)
    return thunk


def evaluate(model, env, args, rng_seed):
    rows = []
    for idx in args.objects:
        for k in range(args.eval_episodes):
            obs, _ = env.reset(seed=rng_seed + 100 * idx + k, options={'object_idx': idx})
            done, peak, depth = False, 0., 0.
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                peak, depth = max(peak, info['lift']), max(depth, info['penetration_mm'])
                done = terminated or truncated
            rows.append({'object_idx': idx, 'success': bool(info['success']), 'ever_success': bool(info['ever_success']),
                         'final_lift': info['lift'], 'peak_lift': peak, 'contacts': info['contacts'],
                         'palm_contact': info['palm_contact'], 'max_penetration_mm': depth})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--steps', type=int, default=3_000_000)
    parser.add_argument('--envs', type=int, default=20)
    parser.add_argument('--objects', type=int, nargs='+', default=[14, 5, 2, 7, 10])
    parser.add_argument('--horizon', type=int, default=120)
    parser.add_argument('--friction', type=float, help='Hand geom friction (default: course 0.25)')
    parser.add_argument('--finger-kp', type=float, help='Finger position gain (default: Menagerie 1.0)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gradient-steps', type=int, default=8, help='Updates per vectorised step')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--eval-every', type=int, default=200_000)
    parser.add_argument('--eval-episodes', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(4)

    out = ROOT / 'runs' / args.run_id
    if out.exists():
        parser.error(f'{out} exists')
    logger = RunLogger(args.run_id, root=ROOT / 'runs', kind='train', name=f'Allegro grasp SAC · {args.run_id}',
                       summary='Grasp-lift-hold on the hand rig', budget=args.steps, algorithm='SAC', skill='hand_grasp')
    atomic_json(out / 'config.json', dict(vars(args), command=sys.argv))
    venv = VecMonitor(SubprocVecEnv([make_env(i, args) for i in range(args.envs)], start_method='spawn'))
    eval_env = make_env(10_000, args)()
    model = SAC('MlpPolicy', venv, learning_rate=args.learning_rate, buffer_size=2_000_000, batch_size=args.batch_size,
                learning_starts=20_000, gamma=.99, tau=.005, train_freq=1, gradient_steps=args.gradient_steps,
                policy_kwargs=dict(net_arch=[512, 512, 256]), seed=args.seed, device=args.device, verbose=0)

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval, self.started, self.recent = args.eval_every, time.monotonic(), []

        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.recent.append(float(info.get('ever_success', False)))
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                rows = evaluate(self.model, eval_env, args, rng_seed=7_000)
                per_object = {str(i): float(np.mean([r['success'] for r in rows if r['object_idx'] == i])) for i in args.objects}
                metrics = dict(scope='hand_grasp', metric_source='evaluation',
                               success_rate=float(np.mean([r['success'] for r in rows])),
                               ever_success_rate=float(np.mean([r['ever_success'] for r in rows])),
                               peak_lift=float(np.mean([r['peak_lift'] for r in rows])),
                               max_penetration_mm=float(max(r['max_penetration_mm'] for r in rows)),
                               per_object=per_object,
                               train_success_rate=float(np.mean(self.recent[-200:])) if self.recent else 0.,
                               steps_per_second=self.num_timesteps / (time.monotonic() - self.started))
                logger.metric(transitions=self.num_timesteps, **metrics)
                print(json.dumps(dict(steps=self.num_timesteps, **metrics)), flush=True)
                self.model.save(out / 'checkpoints' / f'step_{self.num_timesteps:08d}')
            return True

    try:
        model.learn(total_timesteps=args.steps, callback=Progress())
        model.save(out / 'checkpoints' / 'final')
        logger.finish(summary='Training complete; see evaluation metrics')
    except BaseException as exc:
        logger.finish(status='failed', summary=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        venv.close()


if __name__ == '__main__':
    main()
