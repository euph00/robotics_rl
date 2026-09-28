"""Fresh confirmation layouts with an explicit evaluation-only bank version.

The old training loader rejects this version. Geometry validation uses a private
normalized copy; callers always receive the original version and digest. No
policy, physical trial, or route-length threshold participates in generation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from .scenes import (
    BANK_VERSION, DEFAULT_COUNTS, case_specs, generate_layout, load_scene_bank,
    manifest_digest, validate_scene_bank,
)

CONFIRMATION_VERSION = 'confirmation-scene-bank-v1'
CONFIRMATION_PROTOCOL = 'fresh-confirmation-layouts-v1'
CONFIRMATION_SEED = 20260930
CONFIRMATION_OBJECTS = [14, 5, 2, 7, 10]
PRESERVED_PARTITIONS = ('train', 'validation', 'final_reserve')
RULE_OVERRIDES = {
    'training_usage': 'Evaluation-only bank; use the unchanged parent bank for TRAIN and validation',
    'milestone_test_usage': 'Fresh confirmation only after a revised candidate is frozen using TRAIN/validation alone',
    'final_reserve_usage': 'Original reserve preserved but inaccessible through this evaluation-only clone',
}


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _parent_hashes(parent):
    return sorted(record['maze_hash'] for records in parent['partitions'].values()
                  for record in records)


def build_confirmation_bank(parent, *, confirmation_seed=CONFIRMATION_SEED,
                            count=20, retired_evaluation):
    """Replace only the test partition in a new, non-training bank.

    Candidate seeds are successive draws from one fixed RNG stream. Collisions
    or disconnected gate cells are logged and skipped; there is no seed search
    based on policy behavior and no maximum path-length admission rule.
    """
    validate_scene_bank(parent)
    if (isinstance(count, bool) or not isinstance(count, Integral) or count <= 0
            or isinstance(confirmation_seed, bool) or not isinstance(confirmation_seed, Integral)
            or not 0 <= confirmation_seed < 2**32):
        raise ValueError('Positive integer count and unsigned32 confirmation seed required')
    seed, count = int(confirmation_seed), int(count)
    exclusions = _parent_hashes(parent)
    seen = set(exclusions) | set(parent['excluded_course_legacy_hashes'])
    rng = np.random.default_rng(seed)
    records, rejected = [], []
    prefix = f'confirmation_{seed}'
    for _ in range(count * 1000):
        if len(records) == count:
            break
        candidate = int(rng.integers(0, 2**31))
        record = generate_layout(candidate).as_dict()
        reason = ('duplicate_parent_legacy_or_confirmation_map' if record['maze_hash'] in seen
                  else 'disconnected_gate_cells' if not record['geometry_check']['gate_to_gate_connected']
                  else None)
        if reason:
            rejected.append({'maze_seed': candidate, 'maze_hash': record['maze_hash'],
                             'partition': 'milestone_test', 'reason': reason})
            continue
        record['layout_id'] = f'{prefix}_{len(records):03d}'
        records.append(record)
        seen.add(record['maze_hash'])
    if len(records) != count:
        raise RuntimeError('Bounded confirmation geometry sampling exhausted')
    bank = copy.deepcopy(parent)
    bank.update(version=CONFIRMATION_VERSION, status='frozen_confirmation_evaluation_only',
                object_indices=list(CONFIRMATION_OBJECTS), object_selection_status='declared')
    bank['partitions']['milestone_test'] = records
    bank['partition_counts']['milestone_test'] = count
    bank['geometry_rejections'].extend(rejected)
    bank['rules'].update(RULE_OVERRIDES)
    bank['confirmation'] = {
        'protocol': CONFIRMATION_PROTOCOL,
        'parent_manifest_sha256': parent['manifest_sha256'],
        'parent_partition_seed': parent['partition_seed'],
        'confirmation_seed': seed, 'layout_id_prefix': prefix,
        'candidate_rng': 'numpy.default_rng(confirmation_seed), integers(0, 2**31)',
        'excluded_parent_layout_hashes': exclusions,
        'preserved_partition_sha256': {name: content_digest(parent['partitions'][name])
                                      for name in PRESERVED_PARTITIONS},
        'retired_milestone_test': [{'layout_id': r['layout_id'], 'maze_hash': r['maze_hash']}
                                 for r in parent['partitions']['milestone_test']],
        'retired_evaluation': copy.deepcopy(retired_evaluation),
        'retirement_status': 'retired_from_future_heldout_claims_after_guiding_refinement',
        'retired_trajectories_allowed_in_training': False,
        'confirmation_physical_evaluations': 0,
        'route_length_filter': False, 'learned_outcome_filter': False,
        'required_object_indices': list(CONFIRMATION_OBJECTS),
        'geometry_rejections': rejected,
    }
    bank['manifest_sha256'] = manifest_digest(bank)
    return verify_confirmation_bank(parent, bank)


def validate_confirmation_bank(bank):
    """Verify raw identity, old geometry constraints, and confirmation metadata."""
    if bank.get('version') != CONFIRMATION_VERSION:
        raise ValueError('Unknown confirmation-bank version')
    if bank.get('manifest_sha256') != manifest_digest(bank):
        raise ValueError('Confirmation-bank manifest digest mismatch')
    normalized = copy.deepcopy(bank)
    normalized['version'] = BANK_VERSION
    normalized['manifest_sha256'] = manifest_digest(normalized)
    validate_scene_bank(normalized)
    meta = bank.get('confirmation', {})
    if (bank.get('status') != 'frozen_confirmation_evaluation_only'
            or meta.get('protocol') != CONFIRMATION_PROTOCOL
            or meta.get('parent_partition_seed') != bank['partition_seed']
            or bank['object_indices'] != CONFIRMATION_OBJECTS
            or meta.get('required_object_indices') != CONFIRMATION_OBJECTS):
        raise ValueError('Confirmation scope or original case RNG contract differs')
    seed = meta.get('confirmation_seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Invalid confirmation seed')
    if meta.get('layout_id_prefix') != f'confirmation_{seed}':
        raise ValueError('Confirmation layout prefix differs')
    if (meta.get('route_length_filter') is not False or meta.get('learned_outcome_filter') is not False
            or meta.get('confirmation_physical_evaluations') != 0
            or meta.get('retired_trajectories_allowed_in_training') is not False
            or meta.get('retirement_status') != 'retired_from_future_heldout_claims_after_guiding_refinement'):
        raise ValueError('Confirmation admission or retired-test usage differs')
    if any(bank['rules'].get(key) != value for key, value in RULE_OVERRIDES.items()):
        raise ValueError('Confirmation evaluation-only rules differ')
    excluded = meta.get('excluded_parent_layout_hashes', [])
    if not excluded or excluded != sorted(set(excluded)):
        raise ValueError('Missing or duplicate parent content exclusions')
    retired = meta.get('retired_milestone_test', [])
    if (not retired or len({r['layout_id'] for r in retired}) != len(retired)
            or not {r['maze_hash'] for r in retired}.issubset(excluded)):
        raise ValueError('Retired-test provenance differs')
    provenance = meta.get('retired_evaluation', {})
    if (set(provenance) != {'run_id', 'summary_sha256'} or not provenance['run_id']
            or len(provenance['summary_sha256']) != 64):
        raise ValueError('Retired-test evaluation identity required')
    preserved = meta.get('preserved_partition_sha256', {})
    if set(preserved) != set(PRESERVED_PARTITIONS):
        raise ValueError('Missing preserved partition fingerprints')
    for name in PRESERVED_PARTITIONS:
        if content_digest(bank['partitions'][name]) != preserved[name]:
            raise ValueError('Confirmation clone changed a preserved partition')
        if not {r['maze_hash'] for r in bank['partitions'][name]}.issubset(excluded):
            raise ValueError('Preserved partition missing from content exclusions')
    forbidden = set(excluded) | set(bank['excluded_course_legacy_hashes'])
    records = bank['partitions']['milestone_test']
    if not records:
        raise ValueError('Confirmation partition is empty')
    for index, record in enumerate(records):
        if record['layout_id'] != f'confirmation_{seed}_{index:03d}' or record['maze_hash'] in forbidden:
            raise ValueError('Confirmation layout ID or content exclusion differs')
        expected = generate_layout(record['maze_seed']).as_dict()['geometry_check']
        if record['geometry_check'] != expected or not expected['gate_to_gate_connected']:
            raise ValueError('Confirmation geometry admission differs')
        forbidden.add(record['maze_hash'])
    rejections = meta.get('geometry_rejections')
    if not isinstance(rejections, list) or any(r['reason'] not in {
            'duplicate_parent_legacy_or_confirmation_map', 'disconnected_gate_cells'}
            or r['partition'] != 'milestone_test' for r in rejections):
        raise ValueError('Unknown confirmation rejection rule')
    return bank


def verify_confirmation_bank(parent, bank):
    """Confirm the unchanged parent records and complete content exclusions."""
    validate_scene_bank(parent)
    validate_confirmation_bank(bank)
    meta = bank['confirmation']
    if meta['parent_manifest_sha256'] != parent['manifest_sha256']:
        raise ValueError('Confirmation parent lineage mismatch')
    variable = {'version', 'status', 'manifest_sha256', 'object_indices', 'object_selection_status',
                'confirmation', 'partitions', 'partition_counts', 'geometry_rejections', 'rules'}
    if ({k: v for k, v in parent.items() if k not in variable}
            != {k: v for k, v in bank.items() if k not in variable}):
        raise ValueError('Confirmation changed the parent distribution')
    for name in PRESERVED_PARTITIONS:
        if bank['partitions'][name] != parent['partitions'][name]:
            raise ValueError('Confirmation changed preserved parent records')
    if bank['partition_counts'] != dict(parent['partition_counts'], milestone_test=len(bank['partitions']['milestone_test'])):
        raise ValueError('Confirmation changed preserved partition counts')
    if meta['excluded_parent_layout_hashes'] != _parent_hashes(parent):
        raise ValueError('Confirmation omitted a parent layout exclusion')
    expected_retired = [{'layout_id': r['layout_id'], 'maze_hash': r['maze_hash']}
                       for r in parent['partitions']['milestone_test']]
    if meta['retired_milestone_test'] != expected_retired:
        raise ValueError('Confirmation changed retired-test identity')
    if bank['geometry_rejections'] != parent['geometry_rejections'] + meta['geometry_rejections']:
        raise ValueError('Confirmation changed historical geometry accounting')
    if bank['rules'] != dict(parent['rules'], **RULE_OVERRIDES):
        raise ValueError('Confirmation changed unrelated parent rules')
    return bank


def load_evaluation_bank(path):
    bank = json.loads(Path(path).read_text())
    if bank.get('version') == CONFIRMATION_VERSION:
        return validate_confirmation_bank(bank)
    return load_scene_bank(path)


def require_evaluation_partition(bank, partition):
    if bank.get('version') == CONFIRMATION_VERSION and partition != 'milestone_test':
        raise ValueError('Confirmation bank is evaluation-only: only milestone_test is accessible')


def evaluation_case_specs(bank, partition='train', *, object_indices=None, allow_heldout=False):
    require_evaluation_partition(bank, partition)
    if bank.get('version') == CONFIRMATION_VERSION:
        objects = bank['object_indices'] if object_indices is None else list(object_indices)
        if objects != CONFIRMATION_OBJECTS:
            raise ValueError('Confirmation requires the unchanged declared five-object order')
    return case_specs(bank, partition, object_indices=object_indices, allow_heldout=allow_heldout)
