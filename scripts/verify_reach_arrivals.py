#!/usr/bin/env python3
"""Bounded learned-reach robustness after physically perturbed reference arrivals.

One reference episode supplies a real contact-held maze endpoint. Snapshots are
restored only between diagnostic cases; each perturbation, learned reach, and
50-source-action hold then runs without resets or teacher corrections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from pnr.env import PickNavReachEnv
from pnr.policies import load_policy_checkpoint
from pnr.reference import ReferenceController
from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json


def video_frame(native, phase):
    base = native.root_position[:2]+native.agent_qpos[:2]
    rgb = native.render_rgb('follow', width=640, height=360, shadow=False,
        target=[base[0]+.12, base[1], .65], distance=2.6, yaw=0, pitch=-78)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    info = native.evaluate()
    draw.rectangle((0, 0, 640, 48), fill=(15, 23, 42))
    draw.text((12, 6), 'Learned reach | reference-generated arrival', font=font, fill='white')
    draw.text((12, 27), f'{phase}  |  original goal {info["dist_to_goal"]*100:.1f} cm  |  held {"yes" if info["bilateral_contact"] else "no"}', font=font, fill=(210, 225, 240))
    return np.asarray(canvas)


def make_figure(cases, path):
    values = np.array([case['final_goal_distance_m']*100 for case in cases]).reshape(3, 3)
    success = np.array([case['success'] for case in cases]).reshape(3, 3)
    fig, axis = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    axis.imshow(values, origin='lower', vmin=0, vmax=10, cmap='Blues')
    for row in range(3):
        for column in range(3):
            label = f'{values[row,column]:.1f} cm' if success[row,column] else f'FAIL\n{values[row,column]:.1f} cm'
            axis.text(column, row, label, ha='center', va='center', fontsize=13,
                color='white' if values[row,column]>5 else '#18334b')
    axis.set_xticks([0, 1, 2], ['−10', '0', '+10'])
    axis.set_yticks([0, 1, 2], ['−10', '0', '+10'])
    axis.set_xlabel('Arrival x offset (cm)'); axis.set_ylabel('Arrival y offset (cm)')
    axis.set_title(f'{int(success.sum())}/9 retained reaches\nGoal error after 50-step hold', fontsize=14)
    axis.set_xticks(np.arange(-.5, 3, 1), minor=True)
    axis.set_yticks(np.arange(-.5, 3, 1), minor=True)
    axis.grid(which='minor', color='white', linewidth=2)
    axis.tick_params(which='minor', bottom=False, left=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='reach_arrival_robustness')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/sac_reach_handoff01/checkpoints/best/model.zip')
    args = parser.parse_args()
    torch.set_num_threads(1)
    logger = RunLogger(args.run_id, kind='validation', skill='reach', algorithm='SAC + HER',
        name='Learned reach arrival robustness', budget=900,
        summary='Nine physical arrival offsets; fixed original goal; reference setup and learned reach.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    frozen = artifacts/'frozen_reach_policy.zip'
    content = args.checkpoint.read_bytes(); frozen.write_bytes(content)
    checkpoint = {'source':str(args.checkpoint), 'frozen_path':str(frozen.resolve()),
        'sha256':hashlib.sha256(content).hexdigest()}
    model = load_policy_checkpoint(frozen, device='cpu')
    checkpoint['trained_transitions'] = int(model.num_timesteps)
    if model.action_space.shape != (3,):
        raise ValueError('Expected the saved three-action reach policy')
    native = None
    writer = None
    cases = []
    started = time.perf_counter()
    try:
        native = PickNavReachEnv(seed=42, object_idx=14)
        controller = ReferenceController(native, seed=42)
        controller.prepare_grasp()
        if not controller.execute_grasp():
            raise RuntimeError('Reference physical grasp failed')
        controller.prepare_carry()
        controller.carry()
        setup_calls = native.step_count
        setup_ticks = native.physics_steps
        initial_base = controller.base_position().copy()
        endpoint = native.save_state()
        logger.update(checkpoint=checkpoint, setup_action_calls=setup_calls,
            setup_physics_ticks=setup_ticks, initialization_physics_ticks=native.settle_steps)
        with (artifacts/'transitions.jsonl').open('w') as trace:
            for dy in (-.10, 0., .10):
                for dx in (-.10, 0., .10):
                    native.restore_state(endpoint)
                    controller.planner.set_payload()
                    before_perturb = native.step_count
                    # Execute the same physical arrival/settling operation at
                    # zero offset too; restored collision caches are not evidence
                    # of current retained contact until physics advances.
                    controller.move_base(initial_base+[dx, dy], stage='carry', step_size=.006)
                    perturb_calls = native.step_count-before_perturb
                    controller.planner.set_payload()
                    initial_arrival_base = controller.base_position().copy()
                    initial_object = native.observe()['object_pos'].copy()
                    initial_error = native.goal_pos-initial_object
                    initial_contact = native.contacts()['bilateral_contact']
                    skill = ManipulationSkillEnv.bind(native, controller, skill='reach', horizon=100)
                    observation = skill._observation()
                    policy_calls_before = native.step_count
                    record_video = dx == 0. and dy == 0.
                    if record_video:
                        writer = imageio.get_writer(artifacts/'learned_reach_diagnostic.mp4', fps=16,
                            codec='libx264', quality=7, macro_block_size=8)
                        writer.append_data(video_frame(native, 'Policy control'))
                    trajectory = []
                    for decision in range(100):
                        action, _ = model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, info = skill.step(action)
                        row = {'arrival_xy_offset':[dx, dy], 'phase':'learned_reach', 'decision':decision+1,
                            'action':action.tolist(), 'reward':reward, 'object_position':native.observe()['object_pos'].tolist(),
                            'goal_position':native.goal_pos.tolist(),
                            **{key:info[key] for key in ('dist_to_goal', 'bilateral_contact', 'is_success',
                                'complete_task_success', 'action_rejected', 'source_action_calls', 'physics_ticks')}}
                        trajectory.append(row); trace.write(json.dumps(row)+'\n')
                        if writer:
                            writer.append_data(video_frame(native, 'Policy control'))
                        if info['is_success'] or terminated or truncated:
                            break
                    policy_calls = native.step_count-policy_calls_before
                    hold_target = native.agent_qpos.copy(); hold_target[native.finger_slots] = 0.
                    hold = []
                    for step in range(50):
                        obs, info = native.step(hold_target)
                        row = {'arrival_xy_offset':[dx, dy], 'phase':'closed_hold', 'hold_action':step+1,
                            'object_position':obs['object_pos'].tolist(), 'goal_position':native.goal_pos.tolist(),
                            **{key:info[key] for key in ('dist_to_goal', 'bilateral_contact', 'complete_task_success',
                                'source_action_calls', 'physics_ticks')}}
                        hold.append(row); trace.write(json.dumps(row)+'\n')
                        if writer and (step+1)%3 == 0:
                            writer.append_data(video_frame(native, 'Closed hold verification'))
                    trace.flush()
                    if writer:
                        writer.close(); writer = None
                    height_range = float(np.ptp([step['object_position'][2] for step in hold]))
                    stable = bool(all(step['bilateral_contact'] for step in hold) and height_range < .02)
                    final = native.evaluate()
                    passed = bool(final['complete_task_success'] and stable)
                    result = {'arrival_xy_offset':[dx, dy], 'physical_initial_base_xy':initial_arrival_base.tolist(), 'final_base_xy':controller.base_position().tolist(),
                        'initial_object_position':initial_object.tolist(), 'initial_bilateral_contact':initial_contact, 'initial_goal_error_xyz':initial_error.tolist(),
                        'initial_goal_distance_m':float(np.linalg.norm(initial_error)),
                        'policy_decisions':len(trajectory), 'policy_source_action_calls':policy_calls,
                        'perturbation_source_action_calls':perturb_calls, 'hold_source_action_calls':50,
                        'minimum_policy_goal_distance_m':min(step['dist_to_goal'] for step in trajectory),
                        'final_goal_distance_m':final['dist_to_goal'], 'success':passed,
                        'reached_strict_4cm':any(step['is_success'] for step in trajectory),
                        'bilateral_contact_fraction_policy':float(np.mean([step['bilateral_contact'] for step in trajectory])),
                        'action_rejection_fraction':float(np.mean([step['action_rejected'] for step in trajectory])),
                        'hold_height_range_m':height_range, 'hold_bilateral_all':all(step['bilateral_contact'] for step in hold), 'stable_hold':stable,
                        'final_metrics':final}
                    cases.append(result)
                    logger.metric(sum(case['policy_decisions'] for case in cases), scope='skill', metric_source='arrival_diagnostic',
                        success_rate=float(np.mean([case['success'] for case in cases])), eval_episodes=len(cases),
                        goal_distance=final['dist_to_goal'])
                    print(json.dumps(result), flush=True)
        native.remove_state(endpoint)
        counted_calls = setup_calls+sum(case['perturbation_source_action_calls']+case['policy_source_action_calls']+case['hold_source_action_calls'] for case in cases)
        report = {'scope':'reference_setup_learned_reach_diagnostic', 'full_learned_system':False,
            'checkpoint':checkpoint, 'scene':native.scene_manifest(), 'reference_endpoint_base_xy':initial_base.tolist(), 'cases':cases,
            'successes':sum(case['success'] for case in cases), 'episodes':len(cases),
            'setup_source_action_calls':setup_calls, 'setup_physics_ticks':setup_ticks,
            'initialization_physics_ticks':native.settle_steps,
            'perturbation_source_action_calls':sum(case['perturbation_source_action_calls'] for case in cases),
            'policy_source_action_calls':sum(case['policy_source_action_calls'] for case in cases),
            'hold_source_action_calls':sum(case['hold_source_action_calls'] for case in cases),
            'total_source_action_calls':counted_calls, 'total_physics_ticks':counted_calls*native.substeps+native.settle_steps,
            'wall_seconds':time.perf_counter()-started,
            'protocol':'One physical reference carry endpoint; restore only between cases, execute actual +/-10cm base perturbation, <=100 learned decisions, then 50 closed hold source calls. Original world goal remains fixed.'}
        atomic_json(artifacts/'results.json', report)
        make_figure(cases, artifacts/'arrival_goal_error.png')
        logger.artifact('artifacts/results.json', 'Nine physical arrival cases and counts')
        logger.artifact('artifacts/arrival_goal_error.png', 'Goal error after held reach', kind='image')
        logger.artifact('artifacts/learned_reach_diagnostic.mp4', 'Reference arrival + learned reach diagnostic', kind='video')
        logger.artifact('artifacts/transitions.jsonl', 'Actual learned and hold transitions')
        logger.finish(summary=f"{report['successes']}/9 learned reaches retained after physical arrival shifts and a 50-action hold. Reference setup; not a full learned-system result.")
    except BaseException as error:
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise
    finally:
        if writer:
            writer.close()
        if native is not None:
            native.close()


if __name__ == '__main__':
    main()
