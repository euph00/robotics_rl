#!/usr/bin/env python3
"""Clips for the project site (site/media): grasp episodes in the grasp-training environment and full-task
episodes of the frozen systems, as H.264 MP4 with a JSON sidecar (outcome, source).

  # one grasp episode on the Fetch: the scripted teacher, or a train_hand.py run at a checkpoint
  python scripts/record_clips.py grasp --policy teacher --robot fetch --object 14 --seed 3 --output site/media/x.mp4
  python scripts/record_clips.py grasp --policy v2_hand_bcppo_std_s0 --step 10000012 --object 14 --seed 3 --output ...
  # outcomes only (no video) over a seed range, to pick an episode
  python scripts/record_clips.py search --policy teacher --robot fetch --object 14 --seeds 0 30
  # a full-task case as scripts/final_v2.py evaluates it (planner, or a learned grasp with the frozen grip)
  python scripts/record_clips.py task --policy planner --set reserve --case <case_id> --objects 0 102 --starts 3 --output ...
  python scripts/record_clips.py task --policy v2_hand_bcppo_wide_s3 --step N --set reserve --case <case_id> ...
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.eval as E  # noqa: E402  (imports pnr first: rendering defaults)
from pnr.env import PickNavReachEnv  # noqa: E402
from pnr.hand_env import HandGraspEnv  # noqa: E402
from pnr.hand_system import HandPlannerSystem, HandPolicy  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402

SIZE = (640, 360)
CHASE = dict(distance=6, pitch=-75)          # near top-down: the default follow camera is occluded by the walls
FAR_STAGES = ('carry', 'reach', 'retreat')


def grasp_env(policy, robot, side_objects, object_idx):
    """The grasp-training environment exactly as the policy's run configured it (teacher: the campaign's settings)."""
    config = {'objects': [object_idx]} if policy == 'teacher' else json.loads((ROOT / 'runs' / policy / 'config.json').read_text())
    robot = config.get('robot', robot)
    side = config.get('side_grasp_objects', side_objects)
    return HandGraspEnv(objects=config['objects'], horizon=config.get('horizon', 300),
                        physics=config.get('physics', 'realistic'), carry=config.get('carry', True),
                        carry_calls=config.get('carry_calls', 200), robot=robot,
                        wrist_box=tuple(config.get('wrist_box') or (.08, .5)), hand_mount=config.get('hand_mount', 'wrist'),
                        start_shift=tuple(config.get('start_shift', (.08, .12))), side_objects=side or [])


def grasp_episode(env, policy, step, object_idx, seed, frames=None):
    obs, _ = env.reset(seed=seed, options={'object_idx': object_idx})
    if policy == 'teacher':
        from pnr.hand_teacher import GripperPlannerTeacher, PowerGraspTeacher, SideGraspTeacher
        Teacher = (GripperPlannerTeacher if env.robot == 'fetch-gripper' else
                   SideGraspTeacher if object_idx in env.side_objects else PowerGraspTeacher)
        teacher = Teacher(env, rng=np.random.default_rng(seed))
        act = lambda o: teacher.act()
    else:
        act = HandPolicy(ROOT / 'runs' / policy, step).act
    info, decisions = {}, 0
    while True:
        obs, _, terminated, truncated, info = env.step(act(obs))
        decisions += 1
        if frames is not None:
            frames.append(env.rig.render(*SIZE, distance=.85, azimuth=150, elevation=-18))
        if terminated or truncated:
            break
    return {'outcome': 'carried' if info.get('carried') else 'lifted' if info.get('ever_success') else 'failed',
            'decisions': decisions}


def task_episode(args):
    cases = {c['case_id']: c for c in E.select_cases(args.set, objects=args.objects, random_object_pose=True, starts=args.starts)}
    case = cases[args.case]
    planner = args.policy == 'planner'
    config = {} if planner else json.loads((ROOT / 'runs' / args.policy / 'config.json').read_text())
    effector = 'gripper' if planner or config.get('robot') == 'fetch-gripper' else 'allegro'
    env = PickNavReachEnv(**case['env_kwargs'], physics='realistic', effector=effector,
                          **({'hand_mount': 'wrist'} if effector == 'allegro' else {}))
    frames, step, holder = [], env.step, {}

    def recording_step(action):
        result = step(action)
        if env.step_count % args.every == 0:
            stage = getattr(holder.get('system'), 'stage', '')
            base = env.root_position[:2] + env.agent_qpos[:2]
            near = np.linalg.norm(env.object_pose()[0][:2] - base) < 1.2
            frames.append(env.render_rgb('grasp', *SIZE, distance=1.1) if near and stage not in FAR_STAGES
                          else env.render_rgb('follow', *SIZE, **CHASE))
        return result
    env.step = recording_step
    if planner:
        execution = dict(E.DEFAULT_EXECUTION, controller='reference', planned_speed_scale=3., holding_speed_scale=2.)
        original = E.ReferenceController.__init__

        def init(self, *a, **k):
            original(self, *a, **k)
            holder['system'] = self
        E.ReferenceController.__init__ = init
        result = E.run_reference(env, execution)
    else:
        side = config.get('side_grasp_objects')
        system = HandPlannerSystem(env, HandPolicy(ROOT / 'runs' / args.policy, args.step), seed=env.seed, speed_scale=3.,
                                   holding_speed_scale=2., fingers_after_grasp='frozen',
                                   **({} if side is None else {'side_grasp_objects': side}))
        holder['system'] = system
        result = system.run()
    frames.append(env.render_rgb('follow', *SIZE, **CHASE))
    env.close()
    return frames, {'outcome': 'delivered' if result.get('success') else f"failed at {result.get('failed_stage')}",
                    'case': args.case, 'set': args.set}


def write(path, frames, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=30, quality=6, macro_block_size=8)
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('grasp', 'search', 'task'))
    parser.add_argument('--policy', required=True, help='teacher, planner (task), or a train_hand.py run id')
    parser.add_argument('--step', type=int)
    parser.add_argument('--robot', choices=('fetch', 'fetch-gripper'), default='fetch', help='For the teacher')
    parser.add_argument('--side-objects', type=int, nargs='*', default=[], help='Teacher: objects grasped from the side')
    parser.add_argument('--object', type=int, default=14)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs=2, default=(0, 20), metavar=('FIRST', 'END'))
    parser.add_argument('--set', default='reserve')
    parser.add_argument('--case')
    parser.add_argument('--objects', type=int, nargs='+', default=[14, 5, 2, 7, 10])
    parser.add_argument('--starts', type=int, default=1)
    parser.add_argument('--every', type=int, default=4, help='Native calls per frame (task)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = {'policy': args.policy, 'step': args.step}
    if args.mode == 'search':
        env = grasp_env(args.policy, args.robot, args.side_objects, args.object)
        for seed in range(*args.seeds):
            print(json.dumps(dict(source, object=args.object, seed=seed, **grasp_episode(env, args.policy, args.step, args.object, seed))),
                  flush=True)
    elif args.mode == 'grasp':
        env, frames = grasp_env(args.policy, args.robot, args.side_objects, args.object), []
        outcome = grasp_episode(env, args.policy, args.step, args.object, args.seed, frames)
        write(args.output, frames, dict(source, object=args.object, seed=args.seed, **outcome))
    else:
        frames, outcome = task_episode(args)
        write(args.output, frames, dict(source, **outcome))


if __name__ == '__main__':
    main()
