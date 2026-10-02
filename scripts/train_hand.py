#!/usr/bin/env python3
"""Train an Allegro grasp-lift-hold policy (PPO or SAC) on the hand rig (pnr/hand_env.py).

  python scripts/train_hand.py --run-id hand_ppo_mu025 --algo ppo --steps 30000000 --envs 20
  python scripts/train_hand.py --run-id hand_sac_mu025 --algo sac --steps 3000000 --gradient-steps 2

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


def make_env(rank, args, evaluation=False):
    def thunk():
        return HandGraspEnv(objects=args.objects, horizon=args.horizon, seed=args.seed * 1000 + rank,
                            friction=args.friction, finger_kp=args.finger_kp, finger_actions=args.finger_actions,
                            near_start=0. if evaluation else args.near_start, wrist_lag=args.wrist_lag,
                            carry=args.carry, carry_calls=args.carry_calls, physics=args.physics)
    return thunk


def evaluate(model, env, args, rng_seed, normalize=lambda o: o):
    rows = []
    for idx in args.objects:
        for k in range(args.eval_episodes):
            obs, _ = env.reset(seed=rng_seed + 100 * idx + k, options={'object_idx': idx})
            done, peak, depth, carried, entered = False, 0., 0., False, False
            while not done:
                action, _ = model.predict(normalize(obs), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                peak, depth = max(peak, info['lift']), max(depth, info['penetration_mm'])
                entered |= info.get('phase') == 'carry'
                carried |= bool(info.get('carried'))
                done = terminated or truncated
            rows.append({'object_idx': idx, 'success': bool(info['success']), 'ever_success': bool(info['ever_success']),
                         'final_lift': info['lift'], 'peak_lift': peak, 'contacts': info['contacts'],
                         'palm_contact': info['palm_contact'], 'max_penetration_mm': depth,
                         'entered_carry': entered, 'carried': carried})
    return rows


# Stages: active training objects, near-grasp start share, and when to advance (evaluation success on the
# stage's objects, or a step budget). Evaluation always covers every object from normal starts.
CURRICULA = {'easy-first': [
    {'objects': [14, 5], 'near_start': .5, 'advance_at': .5, 'max_steps': 12_000_000},
    {'objects': [14, 5, 2, 7, 10], 'near_start': .5, 'advance_at': .4, 'max_steps': 15_000_000},
    {'objects': [14, 5, 2, 7, 10], 'near_start': .2, 'advance_at': None, 'max_steps': None}]}


def warm_start(model, venv, source):
    """Copy a PPO checkpoint's weights and normalisation; new trailing inputs start at zero weight."""
    import pickle
    import torch
    from stable_baselines3 import PPO
    run, step = source.split('/')
    base = ROOT / 'runs' / run / 'checkpoints' / f'step_{int(step):08d}'
    old = PPO.load(str(base) + '.zip', device='cpu').policy.state_dict()
    new = model.policy.state_dict()
    for key, value in old.items():
        target = new[key]
        if value.shape == target.shape:
            new[key] = value.to(target.device)
        elif value.dim() == 2 and value.shape[0] == target.shape[0] and value.shape[1] < target.shape[1]:
            merged = torch.zeros_like(target)
            merged[:, :value.shape[1]] = value
            new[key] = merged
        else:
            raise ValueError(f'Cannot warm-start {key}: {tuple(value.shape)} -> {tuple(target.shape)}')
    model.policy.load_state_dict(new)
    stats = pickle.loads(Path(str(base) + '_vecnormalize.pkl').read_bytes())
    n = len(stats.obs_rms.mean)
    venv.obs_rms.mean[:n], venv.obs_rms.var[:n] = stats.obs_rms.mean, stats.obs_rms.var
    venv.obs_rms.count = stats.obs_rms.count
    venv.ret_rms = stats.ret_rms


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
    parser.add_argument('--algo', choices=('ppo', 'sac'), default='ppo')
    parser.add_argument('--finger-actions', choices=('delta', 'absolute'), default='delta')
    parser.add_argument('--near-start', type=float, default=0., help='Fraction of curriculum starts near a grasp')
    parser.add_argument('--wrist-lag', type=float, default=.02, help='Max wrist target lead (m)')
    parser.add_argument('--physics', choices=('realistic', 'course'), help='Default: PNR_PHYSICS or realistic')
    parser.add_argument('--carry', action='store_true', help='Carry phase after a held lift')
    parser.add_argument('--carry-calls', type=int, default=200)
    parser.add_argument('--init-from', help='run/step of a PPO checkpoint to warm-start from (extra inputs zero-initialised)')
    parser.add_argument('--curriculum', choices=('easy-first',), help='Staged training distribution (see CURRICULA)')
    parser.add_argument('--gradient-steps', type=int, default=8, help='Updates per vectorised step')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--eval-every', type=int, default=200_000)
    parser.add_argument('--eval-episodes', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    torch.set_num_threads(4)

    out = ROOT / 'runs' / args.run_id
    if out.exists():
        parser.error(f'{out} exists')
    logger = RunLogger(args.run_id, root=ROOT / 'runs', kind='train', name=f'Allegro grasp {args.algo.upper()} · {args.run_id}',
                       summary='Grasp-lift-hold on the hand rig', budget=args.steps, algorithm=args.algo.upper(), skill='hand_grasp')
    atomic_json(out / 'config.json', dict(vars(args), command=sys.argv))
    venv = VecMonitor(SubprocVecEnv([make_env(i, args) for i in range(args.envs)], start_method='spawn'))
    eval_env = make_env(10_000, args, evaluation=True)()
    normalize = lambda o: o
    if args.algo == 'ppo':
        # Large-batch on-policy updates: cheap simulation, few gradient steps per sample.
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.99)
        normalize = lambda o: venv.normalize_obs(o)
        model = PPO('MlpPolicy', venv, learning_rate=args.learning_rate, n_steps=256, batch_size=4096, n_epochs=5,
                    gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=1.,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]), log_std_init=-.5),
                    seed=args.seed, device=args.device, verbose=0)
        if args.init_from:
            warm_start(model, venv, args.init_from)
    else:
        model = SAC('MlpPolicy', venv, learning_rate=args.learning_rate, buffer_size=2_000_000, batch_size=args.batch_size,
                    learning_starts=20_000, gamma=.99, tau=.005, train_freq=1, gradient_steps=args.gradient_steps,
                    policy_kwargs=dict(net_arch=[512, 512, 256]), seed=args.seed, device=args.device, verbose=0)

    stages = CURRICULA.get(args.curriculum)

    def apply_stage(index):
        stage = stages[index]
        venv.env_method('set_curriculum', objects=stage['objects'], near_start=stage['near_start'])
        logger.event(f'Curriculum stage {index}: objects {stage["objects"]}, near-grasp starts {stage["near_start"]}')

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval, self.started, self.recent = args.eval_every, time.monotonic(), []
            self.stage, self.stage_start = 0, 0
            if stages:
                apply_stage(0)

        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.recent.append(float(info.get('ever_success', False)))
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                rows = evaluate(self.model, eval_env, args, rng_seed=7_000, normalize=normalize)
                per_object = {str(i): float(np.mean([r['success'] for r in rows if r['object_idx'] == i])) for i in args.objects}
                metrics = dict(scope='hand_grasp', metric_source='evaluation',
                               success_rate=float(np.mean([r['success'] for r in rows])),
                               ever_success_rate=float(np.mean([r['ever_success'] for r in rows])),
                               peak_lift=float(np.mean([r['peak_lift'] for r in rows])),
                               max_penetration_mm=float(max(r['max_penetration_mm'] for r in rows)),
                               per_object=per_object,
                               carry_entered=float(np.mean([r['entered_carry'] for r in rows])),
                               carry_success_rate=float(np.mean([r['carried'] for r in rows])),
                               train_success_rate=float(np.mean(self.recent[-200:])) if self.recent else 0.,
                               steps_per_second=self.num_timesteps / (time.monotonic() - self.started))
                if stages:
                    stage = stages[self.stage]
                    stage_success = float(np.mean([r['success'] for r in rows if r['object_idx'] in stage['objects']]))
                    metrics.update(curriculum_stage=self.stage, stage_success_rate=stage_success)
                    elapsed = self.num_timesteps - self.stage_start
                    if self.stage + 1 < len(stages) and (
                            (stage['advance_at'] is not None and stage_success >= stage['advance_at'])
                            or (stage['max_steps'] is not None and elapsed >= stage['max_steps'])):
                        self.stage, self.stage_start = self.stage + 1, self.num_timesteps
                        apply_stage(self.stage)
                logger.metric(transitions=self.num_timesteps, **metrics)
                print(json.dumps(dict(steps=self.num_timesteps, **metrics)), flush=True)
                self.model.save(out / 'checkpoints' / f'step_{self.num_timesteps:08d}')
                if isinstance(venv, VecNormalize):
                    venv.save(str(out / 'checkpoints' / f'step_{self.num_timesteps:08d}_vecnormalize.pkl'))
            return True

    try:
        model.learn(total_timesteps=args.steps, callback=Progress())
        model.save(out / 'checkpoints' / 'final')
        if isinstance(venv, VecNormalize):
            venv.save(str(out / 'checkpoints' / 'final_vecnormalize.pkl'))
        logger.finish(summary='Training complete; see evaluation metrics')
    except BaseException as exc:
        logger.finish(status='failed', summary=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        venv.close()


if __name__ == '__main__':
    main()
