#!/usr/bin/env python3
"""Freeze content-disjoint layouts without running policies or physics attempts."""
import argparse
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnr.scenes import DEFAULT_COUNTS, PARTITION_SEED, bank_summary, build_scene_bank, expand_training_bank, load_scene_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('configs/phase2_scene_bank.json'))
    parser.add_argument('--extend-from', type=Path, help='Append TRAIN layouts to this frozen bank; preserve all other cases')
    parser.add_argument('--extension-seed', type=int, default=20260929)
    parser.add_argument('--partition-seed', type=int, default=PARTITION_SEED)
    parser.add_argument('--object-indices', type=int, nargs='*', default=[])
    for partition, count in DEFAULT_COUNTS.items():
        parser.add_argument('--' + partition.replace('_', '-') + '-count', type=int, default=count)
    args = parser.parse_args()
    counts = {name: getattr(args, name + '_count') for name in DEFAULT_COUNTS}
    if args.extend_from:
        if (args.object_indices or args.partition_seed != PARTITION_SEED
                or any(counts[name] != DEFAULT_COUNTS[name] for name in DEFAULT_COUNTS if name != 'train')):
            parser.error('Extension accepts only a new TRAIN count and extension seed; parent settings are preserved')
        bank = expand_training_bank(load_scene_bank(args.extend_from), counts['train'], extension_seed=args.extension_seed)
    else:
        bank = build_scene_bank(args.partition_seed, counts, args.object_indices)
    if args.output.exists():
        old = load_scene_bank(args.output)
        if old['manifest_sha256'] != bank['manifest_sha256']:
            parser.error('Refusing to replace an existing frozen bank; use a new versioned output path')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            encoded = json.dumps(bank, indent=2, allow_nan=False)
            # Keep scalar numeric arrays readable without hundreds of lines per map.
            encoded = re.sub(r'\[([\s\d.,eE+\-]+)\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', encoded)
            stream.write(encoded + '\n')
    summary = bank_summary(bank)
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    # Large held-out hash lists are in the sidecar, not learner-facing stdout.
    summary.pop('heldout_hashes')
    print(json.dumps({'manifest': str(args.output), 'summary': str(summary_path), **summary}, indent=2))


if __name__ == '__main__':
    main()
