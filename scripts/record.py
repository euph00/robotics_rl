#!/usr/bin/env python3
"""Record full-task episodes exactly as scripts/eval.py runs them.

Per case: an MP4 (and WebM) from the follow camera, a preview frame, the eval result,
and a per-call trajectory (base pose, object position, contact, stage) used by the
dashboard's maze figure. Example:

  python scripts/record.py --candidate configs/imitation_sac.json --set dev \
      --cases validation_000_object_05 validation_000_object_14 --output runs/recordings_example
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import the evaluator first, exactly as its workers do: library load order changes low-level
# floating-point paths, and borderline episodes (chaotic contact) then end differently.
import scripts.eval as E  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402

from pnr.telemetry import RunLogger, atomic_json  # noqa: E402
from pnr.video import webm_copy  # noqa: E402


VIEW = dict(distance=6, pitch=-75)   # near top-down chase view: the default follow camera is occluded by 2 m walls


def record_case(case, execution, output, every, width, height, video=True):
    current = {}
    for cls in (E.LearnedSystem, E.ReferenceController):   # remember the live controller for stage labels
        original = cls.__init__

        def init(self, *args, _original=original, **kwargs):
            _original(self, *args, **kwargs)
            current.setdefault(type(self).__name__, self)
        cls.__init__ = init
    native = E.PickNavReachEnv(**case['env_kwargs'], physics=execution['physics'])
    step, frames, trajectory = native.step, [], []

    def recording_step(action):
        result = step(action)
        system, controller = current.get('LearnedSystem'), current.get('ReferenceController')
        stage = system.display_stage if system is not None else getattr(controller, 'stage', None)
        trajectory.append({'call': native.step_count, 'stage': stage, 'qpos': native.agent_qpos.round(4).tolist(),
                           'object_position': np.round(native.object_pose()[0], 4).tolist(),
                           'bilateral_contact': bool(native.contacts()['bilateral_contact'])})
        if video and native.step_count % every == 0:
            frames.append(native.render_rgb('follow', width=width, height=height, **VIEW))
        return result
    native.step = recording_step
    try:
        result = E.run_reference(native, execution) if execution['controller'] == 'reference' else E.run_learned(native, execution)
    finally:
        if video:
            frames.append(native.render_rgb('follow', width=width, height=height, **VIEW))
        native.close()
    name = case['case_id']
    if video:
        path = output / 'artifacts' / f'{name}.mp4'
        imageio.mimwrite(path, frames, fps=30, quality=7, macro_block_size=8)
        webm_copy(path)
        carry = [i for i, row in enumerate(trajectory) if row['stage'] == 'carry']
        preview_index = (carry[len(carry) // 2] // every) if carry else len(frames) // 2
        imageio.imwrite(output / 'artifacts' / f'{name}_preview.png', frames[min(preview_index, len(frames) - 1)])
    with (output / 'artifacts' / f'{name}_trajectory.jsonl').open('w') as stream:
        for row in trajectory:
            stream.write(json.dumps(row) + '\n')
    summary = {key: result.get(key) for key in ('success', 'failed_stage', 'error')}
    atomic_json(output / 'artifacts' / f'{name}.json', dict(summary, case_id=name, calls=len(trajectory)))
    return name, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidate', type=Path, help='Candidate JSON; omit with --controller reference')
    parser.add_argument('--controller', choices=('learned', 'reference'), default='learned')
    parser.add_argument('--set', choices=sorted(E.SETS), default='dev')
    parser.add_argument('--cases', nargs='+', required=True, help='case_id values from the set')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--every', type=int, default=3, help='Native calls per video frame')
    parser.add_argument('--size', type=int, nargs=2, default=(960, 540))
    parser.add_argument('--no-video', action='store_true', help='Trajectories and results only')
    parser.add_argument('--random-object-pose', action='store_true', help="The protocol's seeded random object start per case")
    args = parser.parse_args()
    execution, checkpoints = dict(E.DEFAULT_EXECUTION, controller=args.controller), {}
    if args.candidate:
        candidate = json.loads(args.candidate.read_text())
        execution.update(candidate.get('execution', {}))
        checkpoints = {k: v['frozen_path'] for k, v in candidate['checkpoints'].items()}
    if args.controller == 'reference':
        checkpoints = {}
        execution.update(planned_speed_scale=3.0, holding_speed_scale=2.0)
    elif not checkpoints:
        parser.error('Learned recordings need --candidate')
    by_id = {case['case_id']: case for case in E.select_cases(args.set, layout_limit=None, random_object_pose=args.random_object_pose)}
    missing = [case for case in args.cases if case not in by_id]
    if missing:
        parser.error(f'Unknown cases: {missing}')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory already exists')
    (args.output / 'artifacts').mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output.name, root=args.output.parent, kind='recording',
                       name=args.label or f'Recording · {args.output.name}', summary=f'{len(args.cases)} episodes')
    E.initialize_worker(checkpoints)
    for case_id in args.cases:
        name, summary = record_case(by_id[case_id], execution, args.output, args.every, *args.size, video=not args.no_video)
        outcome = 'delivered' if summary['success'] else f"failed in {summary['failed_stage']}"
        if not args.no_video:
            logger.artifact(f'artifacts/{name}.webm', f'{name} · {outcome}', kind='video')
            logger.artifact(f'artifacts/{name}_preview.png', f'{name} preview', kind='image')
        print(name, summary, flush=True)
    logger.finish(summary=f'{len(args.cases)} recorded episodes')


if __name__ == '__main__':
    main()
