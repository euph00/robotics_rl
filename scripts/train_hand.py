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
                            carry=args.carry, carry_calls=args.carry_calls, physics=args.physics,
                            robot=args.robot, wrist_box=args.wrist_box, hand_mount=args.hand_mount,
                            task_carry=args.task_carry, start_shift=tuple(args.start_shift),
                            **({} if args.side_grasp_objects is None else {'side_objects': args.side_grasp_objects}))
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


def behaviour_clone(model, venv, demos, updates, log_std, device):
    """Fit the PPO actor's mean to teacher actions; observation statistics come from the demonstrations."""
    import torch
    data = np.load(demos)
    observations, actions = data['observations'], data['actions']
    venv.obs_rms.mean[:], venv.obs_rms.var[:] = observations.mean(0), observations.var(0) + 1e-6
    venv.obs_rms.count = len(observations)
    obs = torch.as_tensor(np.clip((observations - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + venv.epsilon), -10, 10),
                          dtype=torch.float32, device=device)
    target = torch.as_tensor(actions, dtype=torch.float32, device=device)
    policy = model.policy
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    rng = np.random.default_rng(0)
    losses = []
    for update in range(updates):
        batch = torch.as_tensor(rng.integers(0, len(obs), 1024), device=device)
        mean = policy.action_net(policy.mlp_extractor.forward_actor(policy.extract_features(obs[batch], policy.pi_features_extractor)))
        loss = ((mean - target[batch]) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss))
    with torch.no_grad():
        policy.log_std.fill_(log_std)
    return {'examples': len(obs), 'updates': updates, 'final_loss': float(np.mean(losses[-200:]))}


class DemoRegulariser:
    """Interleaved behaviour cloning (DAPG-style): a decaying number of actor updates toward the teacher."""

    def __init__(self, model, venv, demos, updates, decay):
        import torch
        data = np.load(demos)
        self.observations, self.actions = data['observations'], data['actions']
        self.model, self.venv, self.updates, self.decay = model, venv, float(updates), decay
        policy = model.policy
        params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=1e-4)
        self.rng = np.random.default_rng(1)

    def step(self):
        import torch
        count = int(round(self.updates))
        self.updates *= self.decay
        policy, device = self.model.policy, self.model.device
        for _ in range(count):
            batch = self.rng.integers(0, len(self.observations), 512)
            obs = torch.as_tensor(self.venv.normalize_obs(self.observations[batch]), dtype=torch.float32, device=device)
            target = torch.as_tensor(self.actions[batch], dtype=torch.float32, device=device)
            mean = policy.action_net(policy.mlp_extractor.forward_actor(policy.extract_features(obs, policy.pi_features_extractor)))
            loss = ((mean - target) ** 2).mean()
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return count


def set_actor_trainable(model, trainable):
    policy = model.policy
    for module in (policy.mlp_extractor.policy_net, policy.action_net):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)
    policy.log_std.requires_grad_(trainable)


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
    parser.add_argument('--algo', choices=('ppo', 'sac', 'rlpd'), default='ppo')
    parser.add_argument('--resume', type=int, metavar='STEP', help='Continue this PPO run from its checkpoint at STEP (same run directory)')
    parser.add_argument('--rlpd-demos', type=Path, help='RLPD prior data: hand_teacher_demos.py transitions.npz')
    parser.add_argument('--utd', type=int, default=1, help='RLPD gradient steps per environment transition')
    parser.add_argument('--finger-actions', choices=('delta', 'absolute'), default='delta')
    parser.add_argument('--near-start', type=float, default=0., help='Fraction of curriculum starts near a grasp')
    parser.add_argument('--wrist-lag', type=float, default=.02, help='Max wrist target lead (m)')
    parser.add_argument('--physics', choices=('realistic', 'course'), help='Default: PNR_PHYSICS or realistic')
    parser.add_argument('--hand-mount', choices=('wrist', 'dorsal'), default='wrist', help='Allegro mount on the Fetch')
    parser.add_argument('--task-carry', action='store_true', help='Fetch: carry phase = the full task\'s retreat, tuck and base carry')
    parser.add_argument('--start-shift', type=float, nargs=2, default=[.03, .03], metavar=('X', 'Y'), help='Object start offset range (m)')
    parser.add_argument('--side-grasp-objects', type=int, nargs='*',
                        help='Hand: objects whose episodes start in the side pregrasp (default 0 102; none: top-down for all)')
    parser.add_argument('--wrist-box', type=float, nargs=2, metavar=('XY', 'TILT'),
                        help='Keep the wrist within XY m of the pregrasp and its pitch/roll within TILT rad')
    parser.add_argument('--robot', choices=('rig', 'fetch', 'fetch-gripper'), default='rig',
                        help='rig: the hand on an idealised six-axis wrist; fetch: on the Fetch arm and base (pnr/fetch_hand.py)')
    parser.add_argument('--carry', action='store_true', help='Carry phase after a held lift')
    parser.add_argument('--carry-calls', type=int, default=200)
    parser.add_argument('--init-from', help='run/step of a PPO checkpoint to warm-start from (extra inputs zero-initialised)')
    parser.add_argument('--curriculum', choices=('easy-first',), help='Staged training distribution (see CURRICULA)')
    parser.add_argument('--bc-demos', type=Path, help='Teacher demonstrations (scripts/hand_teacher_demos.py) for behaviour cloning')
    parser.add_argument('--bc-updates', type=int, default=20000)
    parser.add_argument('--bc-log-std', type=float, default=-1.2, help='Exploration log-std after cloning')
    parser.add_argument('--critic-warmup', type=int, default=0, help='Environment steps with the actor frozen (value warm-up)')
    parser.add_argument('--bc-interleave', type=int, default=0,
                        help='Behaviour-cloning updates on the demonstrations after every PPO rollout (keeps RL near the teacher)')
    parser.add_argument('--bc-interleave-decay', type=float, default=.9995, help='Per-rollout decay of the interleaved updates')
    parser.add_argument('--target-kl', type=float, help='PPO: stop an update early beyond this KL (guards against collapse)')
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
    if out.exists() and args.resume is None:
        parser.error(f'{out} exists')
    if args.resume is not None and args.algo != 'ppo':
        parser.error('--resume supports PPO runs (an off-policy replay buffer is not saved)')
    effector = 'Gripper' if args.robot == 'fetch-gripper' else 'Allegro'
    where = {'fetch': 'with the hand on the Fetch', 'fetch-gripper': "with the Fetch's gripper"}.get(args.robot, 'on the hand rig')
    logger = RunLogger(args.run_id, root=ROOT / 'runs', kind='train', name=f'{effector} grasp {args.algo.upper()} · {args.run_id}', resume=args.resume is not None,
                       summary=f'Grasp, lift and carry {where}', budget=args.steps, algorithm=args.algo.upper(), skill='hand_grasp')
    config = dict({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, command=sys.argv)
    atomic_json(out / ('config.json' if args.resume is None else f'config_resume_{args.resume}.json'), config)
    venv = VecMonitor(SubprocVecEnv([make_env(i, args) for i in range(args.envs)], start_method='spawn'))
    eval_env = make_env(10_000, args, evaluation=True)()
    normalize = lambda o: o
    if args.algo == 'ppo':
        # Large-batch on-policy updates: cheap simulation, few gradient steps per sample.
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.99)
        normalize = lambda o: venv.normalize_obs(o)
        model = PPO('MlpPolicy', venv, learning_rate=args.learning_rate, n_steps=256, batch_size=4096, n_epochs=5,
                    gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=1., target_kl=args.target_kl,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]), log_std_init=-.5),
                    seed=args.seed, device=args.device, verbose=0)
        if args.resume is not None:
            # Continue after an interruption: weights, optimiser state and normalisation statistics of the
            # run's own checkpoint; the step count carries on (no imitation or value warm-up again).
            base = out / 'checkpoints' / f'step_{args.resume:08d}'
            venv = VecNormalize.load(str(base) + '_vecnormalize.pkl', venv.venv)
            venv.training = True
            normalize = lambda o: venv.normalize_obs(o)
            model = PPO.load(str(base) + '.zip', env=venv, device=args.device)
            model.num_timesteps = args.resume
            args.bc_demos, args.critic_warmup = None, 0
            logger.event(f'Resumed from step {args.resume}')
        if args.init_from:
            warm_start(model, venv, args.init_from)
        if args.bc_demos:
            clone = behaviour_clone(model, venv, args.bc_demos, args.bc_updates, args.bc_log_std, model.device)
            logger.event('Behaviour cloning complete', **clone)
            print(json.dumps({'behaviour_cloning': clone}), flush=True)
        if args.critic_warmup:
            set_actor_trainable(model, False)
    elif args.algo == 'rlpd':
        # RLPD: SAC with layer-normalised critics, prior (demonstration) transitions in half of every batch,
        # and args.utd gradient steps per environment transition. Observation normalisation is frozen at the
        # demonstrations' statistics so prior and online data share it.
        from pnr.rlpd import DemoMixReplayBuffer, LayerNormSACPolicy
        prior = np.load(args.rlpd_demos)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10., gamma=.99, training=False)
        venv.obs_rms.mean = prior['observations'].mean(0).astype(np.float64)
        venv.obs_rms.var = prior['observations'].var(0).astype(np.float64) + 1e-6
        venv.obs_rms.count = float(len(prior['observations']))
        normalize = lambda o: venv.normalize_obs(o)
        model = SAC(LayerNormSACPolicy, venv, learning_rate=args.learning_rate, buffer_size=2_000_000, batch_size=args.batch_size,
                    learning_starts=args.batch_size, gamma=.99, tau=.005, train_freq=1, gradient_steps=args.utd * args.envs,
                    replay_buffer_class=DemoMixReplayBuffer, replay_buffer_kwargs=dict(demos=str(args.rlpd_demos), demo_ratio=.5),
                    policy_kwargs=dict(net_arch=[512, 512, 256]), seed=args.seed, device=args.device, verbose=0)
    else:
        model = SAC('MlpPolicy', venv, learning_rate=args.learning_rate, buffer_size=2_000_000, batch_size=args.batch_size,
                    learning_starts=20_000, gamma=.99, tau=.005, train_freq=1, gradient_steps=args.gradient_steps,
                    policy_kwargs=dict(net_arch=[512, 512, 256]), seed=args.seed, device=args.device, verbose=0)

    regulariser = DemoRegulariser(model, venv, args.bc_demos, args.bc_interleave, args.bc_interleave_decay) \
        if args.bc_interleave and args.bc_demos else None
    stages = CURRICULA.get(args.curriculum)

    def apply_stage(index):
        stage = stages[index]
        venv.env_method('set_curriculum', objects=stage['objects'], near_start=stage['near_start'])
        logger.event(f'Curriculum stage {index}: objects {stage["objects"]}, near-grasp starts {stage["near_start"]}')

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            start = args.resume or 0
            self.next_eval, self.started, self.recent = (start // args.eval_every + 1) * args.eval_every, time.monotonic(), []
            self.start_steps = start
            self.stage, self.stage_start = 0, 0
            if stages:
                apply_stage(0)


        def evaluate_and_save(self):
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
                           steps_per_second=(self.num_timesteps - self.start_steps) / (time.monotonic() - self.started))
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

        def _on_rollout_end(self):
            if regulariser is not None and getattr(self, 'unfrozen', not args.critic_warmup):
                regulariser.step()

        def _on_training_start(self):
            if args.bc_demos:        # the imitation-only policy, before any RL update
                self.evaluate_and_save()

        def _on_step(self):
            if args.critic_warmup and self.num_timesteps >= args.critic_warmup and not getattr(self, 'unfrozen', False):
                set_actor_trainable(self.model, True)
                self.unfrozen = True
                logger.event(f'Actor unfrozen after {self.num_timesteps} steps of value warm-up')
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.recent.append(float(info.get('ever_success', False)))
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                self.evaluate_and_save()
            return True

    try:
        model.learn(total_timesteps=args.steps - (args.resume or 0), callback=Progress(), reset_num_timesteps=args.resume is None)
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
