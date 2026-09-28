#!/usr/bin/env python3
"""Freeze content-disjoint layouts without running policies or physics attempts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnr.scenes import DEFAULT_COUNTS, PARTITION_SEED, bank_summary, build_scene_bank, load_scene_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('configs/phase2_scene_bank.json'))
    parser.add_argument('--partition-seed', type=int, default=PARTITION_SEED)
    parser.add_argument('--object-indices', type=int, nargs='*', default=[])
    for partition, count in DEFAULT_COUNTS.items():
        parser.add_argument('--' + partition.replace('_', '-') + '-count', type=int, default=count)
    args = parser.parse_args()
    counts = {name: getattr(args, name + '_count') for name in DEFAULT_COUNTS}
    bank = build_scene_bank(args.partition_seed, counts, args.object_indices)
    if args.output.exists():
        old = load_scene_bank(args.output)
        if old['manifest_sha256'] != bank['manifest_sha256']:
            parser.error('Refusing to replace an existing frozen bank; use a new versioned output path')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(bank, stream, indent=2, allow_nan=False)
            stream.write('\n')
    summary = bank_summary(bank)
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    # Large held-out hash lists are in the sidecar, not learner-facing stdout.
    summary.pop('heldout_hashes')
    print(json.dumps({'manifest': str(args.output), 'summary': str(summary_path), **summary}, indent=2))


if __name__ == '__main__':
    main()
