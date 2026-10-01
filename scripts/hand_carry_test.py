#!/usr/bin/env python3
"""Does a hand-policy grasp survive real carry motions? Replays recorded maze carries on the rig wrist.

For each object and trial the policy grasps and lifts from a normal pregrasp start until the
object has been held (success) for 10 decisions. The finger targets then freeze and the
wrist replays a recorded base trajectory (x, y, yaw per native call, from scripts/record.py
--no-video), with the hand carried 0.24 m ahead of the base as in the Fetch carry posture.
Retention: after the carry and a 50-decision hold the object is still >=10 cm above its
resting height and touches the hand at >=2 places. Slip: object displacement in the palm frame.

  python scripts/hand_carry_test.py --run hand_ppo_l_mu025 --step 20000000 --friction .25 \
      --motions runs/mj_carry_motions_learned01 --output runs/hand_carry_test01
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from pnr.hand_env import HandGraspEnv  # noqa: E402
from pnr.telemetry import RunLogger, atomic_json  # noqa: E402

LEVER = .24       # hand ahead of the base centre in the carry posture (m)


def load_motions(directory):
    motions = []
    for path in sorted(Path(directory, 'artifacts').glob('*_trajectory.jsonl')):
        rows = [json.loads(line) for line in path.open()]
        carry = np.array([r['qpos'][:3] for r in rows if r['stage'] == 'carry'], dtype=float)
        if len(carry) > 20:
            motions.append((path.name.replace('_trajectory.jsonl', ''), carry))
    return motions


def hand_path(base):
    """World xy of a point LEVER ahead of the base, and base yaw, per call."""
    x, y, yaw = base.T
    return np.stack([x + LEVER * np.cos(yaw), y + LEVER * np.sin(yaw)], axis=1), yaw


def carry_trial(env, policy, normalize, object_idx, seed, motion, carry_friction=None, squeeze=0.):
    obs, _ = env.reset(seed=seed, options={'object_idx': object_idx})
    held = 0
    for _ in range(env.horizon):
        action, _ = policy.predict(normalize(obs), deterministic=True)
        obs, _, terminated, _, info = env.step(action)
        held = held + 1 if info['success'] else 0
        if held >= 10 or terminated:
            break
    if held < 10:
        return {'grasped': False}
    rig = env.rig
    if carry_friction is not None:   # probe: change hand friction for the carry only
        saved = rig.model.geom_friction[:, 0].copy()
        for g in range(rig.model.ngeom):
            if rig.model.geom_bodyid[g] in rig._hand_bodies and rig.model.geom_contype[g]:
                rig.model.geom_friction[g, 0] = carry_friction
    palm, rotation = rig.palm_pose()
    start_local = (rig.object_pose()[0] - palm) @ rotation
    wrist0 = rig.wrist().copy()
    path, yaw = hand_path(motion)
    # Rotate the recorded displacements into the rig so the carry starts along the hand's heading.
    turn = wrist0[3] - yaw[0]
    c, s = np.cos(turn), np.sin(turn)
    fingers = env.finger_target.copy()
    if squeeze:   # probe: push flexion targets further closed (stronger grip from the same pose)
        flex = np.array([j % 4 != 0 for j in range(16)])
        fingers[flex] = np.minimum(fingers[flex] + squeeze, rig.finger_high[flex])
    lost_at = None
    for k in range(1, len(path)):
        delta = path[k] - path[0]
        target = wrist0.copy()
        target[0] += c * delta[0] - s * delta[1]
        target[1] += s * delta[0] + c * delta[1]
        target[3] += yaw[k] - yaw[0]
        rig.step(target, fingers)
        if lost_at is None and rig.object_pose()[0][2] - rig.rest_height < .05:
            lost_at = k
    for _ in range(50):
        rig.step(target, fingers)
    contact, palm_contact, _, _ = rig.contacts()
    palm, rotation = rig.palm_pose()
    slip = np.linalg.norm((rig.object_pose()[0] - palm) @ rotation - start_local)
    lift = rig.object_pose()[0][2] - rig.rest_height
    retained = bool(lift >= .10 and contact.sum() + palm_contact >= 2)
    if carry_friction is not None:
        rig.model.geom_friction[:, 0] = saved
    return {'grasped': True, 'retained': retained, 'final_lift': float(lift), 'slip_mm': float(1000 * slip),
            'lost_at_call': lost_at, 'carry_calls': len(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--step', required=True, help="Checkpoint step (e.g. 20000000) or 'final'")
    parser.add_argument('--friction', type=float, default=None)
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--objects', type=int, nargs='+', default=[14, 5, 2, 7, 10])
    parser.add_argument('--carry-friction', type=float, help='Probe: hand friction during the carry only')
    parser.add_argument('--squeeze', type=float, default=0., help='Probe: extra flexion target (rad) during the carry')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from stable_baselines3 import PPO
    name = 'final' if args.step == 'final' else f'step_{int(args.step):08d}'
    base = ROOT / 'runs' / args.run / 'checkpoints' / name
    policy = PPO.load(str(base) + '.zip', device='cpu')
    stats = pickle.loads(Path(str(base) + '_vecnormalize.pkl').read_bytes())
    normalize = lambda o: np.clip((o - stats.obs_rms.mean) / np.sqrt(stats.obs_rms.var + stats.epsilon),
                                  -stats.clip_obs, stats.clip_obs).astype(np.float32)
    motions = load_motions(args.motions)
    env = HandGraspEnv(objects=args.objects, seed=11, friction=args.friction)
    logger = RunLogger(args.output.name, root=args.output.parent, kind='evaluation_hand_carry',
                       name=f'Hand carry test · {args.run} {name}', summary=f'{len(motions)} motions')
    rows = []
    for idx in args.objects:
        for k, (motion_id, motion) in enumerate(motions):
            row = carry_trial(env, policy, normalize, idx, 9000 + 100 * idx + k, motion, args.carry_friction, args.squeeze)
            rows.append(dict(row, object_idx=idx, motion=motion_id))
            print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for idx in args.objects:
        r = [x for x in rows if x['object_idx'] == idx]
        g = [x for x in r if x['grasped']]
        summary[str(idx)] = {'trials': len(r), 'grasped': len(g), 'retained': sum(x['retained'] for x in g),
                             'median_slip_mm': float(np.median([x['slip_mm'] for x in g])) if g else None}
    atomic_json(args.output / 'summary.json', {'config': vars(args) | {'output': str(args.output), 'motions': str(args.motions)},
                                               'per_object': summary, 'rows': rows})
    print(json.dumps(summary, indent=1))
    logger.finish(summary=' · '.join(f"{k}: {v['retained']}/{v['grasped']} retained" for k, v in summary.items()))


if __name__ == '__main__':
    main()
