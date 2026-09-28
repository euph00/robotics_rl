#!/usr/bin/env python3
"""Freeze a fresh evaluation-only confirmation bank; no simulator or policy runs."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pnr.confirmation_scenes import (
    CONFIRMATION_OBJECTS, CONFIRMATION_SEED, build_confirmation_bank,
    content_digest, evaluation_case_specs,
)
from pnr.scenes import load_scene_bank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data, *, compact=False):
    encoded = json.dumps(data, indent=2, allow_nan=False)
    if compact:
        encoded = re.sub(r'\[([\s\d.,eE+\-]+)\]',
                         lambda match: '[' + ' '.join(match.group(1).split()) + ']', encoded)
    with path.open('x') as stream:
        stream.write(encoded + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'configs/phase2_scene_bank_expanded.json')
    parser.add_argument('--output', type=Path, default=ROOT/'configs/phase2_confirmation_bank_20260930.json')
    parser.add_argument('--retired-evaluation', type=Path,
                        default=ROOT/'runs/phase2_system_milestone_test01/artifacts/summary.json')
    parser.add_argument('--confirmation-seed', type=int, default=CONFIRMATION_SEED)
    args = parser.parse_args()
    parent = load_scene_bank(args.parent)
    before = sha(args.parent)
    paths = [args.output, args.output.with_suffix('.generation.json'),
             args.output.with_suffix('.cases.json'), args.output.with_suffix('.summary.json')]
    if any(path.exists() for path in paths):
        parser.error('Refusing to overwrite confirmation evidence; use a new declared output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'run_id': args.retired_evaluation.parent.parent.name,
                  'summary_sha256': sha(args.retired_evaluation)}
    generation = {'protocol': 'predeclared-confirmation-generation-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'parent_path': str(args.parent.resolve()), 'parent_file_sha256': before,
        'parent_manifest_sha256': parent['manifest_sha256'],
        'confirmation_seed': args.confirmation_seed, 'layout_count': 20,
        'objects': CONFIRMATION_OBJECTS, 'retired_evaluation': provenance,
        'admission': 'exclude every parent and course-legacy content hash; connected gate cells only',
        'route_length_filter': False, 'learned_outcome_filter': False,
        'policy_or_physical_trials': 0,
        'generator_source_sha256': sha(ROOT/'pnr/confirmation_scenes.py'),
        'geometry_source_sha256': sha(ROOT/'pnr/scenes.py')}
    write_new(paths[1], generation)
    bank = build_confirmation_bank(parent, confirmation_seed=args.confirmation_seed,
                                   count=20, retired_evaluation=provenance)
    cases = evaluation_case_specs(bank, 'milestone_test', allow_heldout=True)
    case_manifest = {'version': 'sealed-confirmation-cases-v1',
        'scene_bank_sha256': bank['manifest_sha256'], 'generation_plan_sha256': sha(paths[1]),
        'partition': 'milestone_test', 'status': 'sealed_pending_revised_candidate_selection',
        'object_indices': CONFIRMATION_OBJECTS, 'cases': cases,
        'physical_evaluations': 0, 'training_use_allowed': False}
    case_manifest['case_manifest_sha256'] = content_digest(case_manifest)
    summary = {'version': bank['version'], 'manifest_sha256': bank['manifest_sha256'],
        'parent_manifest_sha256': parent['manifest_sha256'], 'confirmation_seed': args.confirmation_seed,
        'confirmation_layouts': 20, 'confirmation_cases': len(cases),
        'first_layout_id': bank['partitions']['milestone_test'][0]['layout_id'],
        'last_layout_id': bank['partitions']['milestone_test'][-1]['layout_id'],
        'excluded_parent_layouts': len(bank['confirmation']['excluded_parent_layout_hashes']),
        'excluded_course_legacy_layouts': len(bank['excluded_course_legacy_hashes']),
        'geometry_rejections': len(bank['confirmation']['geometry_rejections']),
        'preserved_partition_sha256': bank['confirmation']['preserved_partition_sha256'],
        'retirement_status': bank['confirmation']['retirement_status'],
        'training_loader_rejects_version': True, 'physical_evaluations': 0,
        'case_manifest_sha256': case_manifest['case_manifest_sha256']}
    write_new(paths[0], bank, compact=True)
    write_new(paths[2], case_manifest)
    write_new(paths[3], summary)
    if sha(args.parent) != before:
        raise RuntimeError('Parent file changed during confirmation generation')
    print(json.dumps({'bank': str(args.output), **summary}, indent=2))


if __name__ == '__main__':
    main()
