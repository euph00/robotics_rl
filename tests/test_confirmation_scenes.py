"""Geometry-only confirmation separation; these tests create no simulator clients."""
import copy
import hashlib
import json
import random

import numpy as np
import pytest

from pnr.confirmation_scenes import (
    CONFIRMATION_OBJECTS, CONFIRMATION_VERSION, build_confirmation_bank,
    evaluation_case_specs, load_evaluation_bank, validate_confirmation_bank,
    verify_confirmation_bank,
)
from pnr.scenes import (
    build_scene_bank, case_specs, expand_training_bank, generate_layout,
    load_scene_bank, manifest_digest, validate_scene_bank, verify_training_extension,
)
from scripts.phase2_evaluate_system import (
    PLAN_VERSION, plan_digest, select_cases, verify_frozen_plan,
)


@pytest.fixture(scope='module')
def banks():
    parent = build_scene_bank(counts={'train': 3, 'validation': 2,
                                     'milestone_test': 2, 'final_reserve': 2})
    before = copy.deepcopy(parent)
    confirmation = build_confirmation_bank(parent, count=3,
        retired_evaluation={'run_id': 'completed_first_test', 'summary_sha256': 'a'*64})
    assert parent == before
    return parent, confirmation


def test_deterministic_geometry_exclusions_and_rng_isolation(banks):
    parent, bank = banks
    np.random.seed(71)
    random.seed(29)
    again = build_confirmation_bank(parent, count=3,
        retired_evaluation={'run_id': 'completed_first_test', 'summary_sha256': 'a'*64})
    assert again == bank
    np.testing.assert_array_equal(np.random.random(4), np.random.RandomState(71).random_sample(4))
    assert random.random() == random.Random(29).random()
    expected = np.random.default_rng(20260930).integers(0, 2**31, size=3).tolist()
    assert [r['maze_seed'] for r in bank['partitions']['milestone_test']] == expected
    assert [r['layout_id'] for r in bank['partitions']['milestone_test']] == [
        f'confirmation_20260930_{i:03d}' for i in range(3)]
    excluded = {r['maze_hash'] for records in parent['partitions'].values() for r in records}
    excluded |= set(parent['excluded_course_legacy_hashes'])
    assert not excluded & {r['maze_hash'] for r in bank['partitions']['milestone_test']}
    assert bank['confirmation']['route_length_filter'] is False
    assert bank['confirmation']['learned_outcome_filter'] is False
    assert verify_confirmation_bank(parent, bank) is bank


def test_preserved_records_and_original_point_seed_contract(banks):
    parent, bank = banks
    assert bank['partition_seed'] == parent['partition_seed']
    for partition in ('train', 'validation', 'final_reserve'):
        assert bank['partitions'][partition] == parent['partitions'][partition]
        # Read-only contract comparison, bypassing the public evaluation access
        # wrapper deliberately; standard loading still rejects the new version.
        assert case_specs(bank, partition, object_indices=CONFIRMATION_OBJECTS, allow_heldout=True) == case_specs(
            parent, partition, object_indices=CONFIRMATION_OBJECTS, allow_heldout=True)
    assert bank['confirmation']['retired_trajectories_allowed_in_training'] is False
    assert bank['confirmation']['retirement_status'] == 'retired_from_future_heldout_claims_after_guiding_refinement'


def test_training_loader_and_extension_reject_confirmation_version(banks, tmp_path):
    parent, bank = banks
    path = tmp_path/'confirmation.json'
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match='version'):
        load_scene_bank(path)
    with pytest.raises(ValueError, match='version'):
        validate_scene_bank(bank)
    with pytest.raises(ValueError, match='version'):
        expand_training_bank(bank, 9)
    with pytest.raises(ValueError, match='version'):
        verify_training_extension(parent, bank)
    loaded = load_evaluation_bank(path)
    assert loaded == bank and loaded['version'] == CONFIRMATION_VERSION
    assert loaded['manifest_sha256'] == bank['manifest_sha256']


def test_evaluation_access_is_explicit_complete_and_five_object_only(banks):
    _, bank = banks
    for partition in ('train', 'validation', 'final_reserve', 'course_legacy'):
        with pytest.raises(ValueError, match='evaluation-only'):
            select_cases(bank, partition, CONFIRMATION_OBJECTS, allow_heldout=True)
    with pytest.raises(ValueError, match='Held-out'):
        evaluation_case_specs(bank, 'milestone_test')
    with pytest.raises(ValueError, match='five-object order'):
        evaluation_case_specs(bank, 'milestone_test', object_indices=list(reversed(CONFIRMATION_OBJECTS)), allow_heldout=True)
    with pytest.raises(ValueError, match='complete'):
        select_cases(bank, 'milestone_test', CONFIRMATION_OBJECTS, allow_heldout=True, case_limit=1)
    cases = evaluation_case_specs(bank, 'milestone_test', allow_heldout=True)
    assert len(cases) == len({r['case_id'] for r in cases}) == 15
    assert len({r['env_kwargs']['seed'] for r in cases}) == 15
    assert select_cases(bank, 'milestone_test', CONFIRMATION_OBJECTS, allow_heldout=True) == cases
    assert all(r['partition'] == 'milestone_test' and r['case_id'].startswith('confirmation_20260930_') for r in cases)


def test_normal_bank_evaluation_delegates_unchanged(banks, tmp_path):
    parent, _ = banks
    path = tmp_path/'normal.json';path.write_text(json.dumps(parent))
    assert load_evaluation_bank(path) == load_scene_bank(path) == parent
    for partition in parent['partitions']:
        assert evaluation_case_specs(parent, partition, object_indices=CONFIRMATION_OBJECTS, allow_heldout=True) == case_specs(
            parent, partition, object_indices=CONFIRMATION_OBJECTS, allow_heldout=True)
    assert len(select_cases(parent, 'course_legacy', CONFIRMATION_OBJECTS)) == 5


def test_confirmation_rejects_rehashed_geometry_and_retirement_tampering(banks):
    parent, bank = banks
    for mutation, pattern in [('retired_map', 'exclusion'), ('preserved', 'preserved'),
                              ('point_rng', 'RNG'), ('route_filter', 'admission')]:
        bad = copy.deepcopy(bank)
        if mutation == 'retired_map':
            record = copy.deepcopy(parent['partitions']['milestone_test'][0])
            record['layout_id'] = bad['partitions']['milestone_test'][0]['layout_id']
            bad['partitions']['milestone_test'][0] = record
        elif mutation == 'preserved':
            bad['partitions']['train'].reverse()
        elif mutation == 'point_rng':
            bad['partition_seed'] += 1
        else:
            bad['confirmation']['route_length_filter'] = True
        bad['manifest_sha256'] = manifest_digest(bad)
        with pytest.raises(ValueError, match=pattern):
            validate_confirmation_bank(bad)
    bad = copy.deepcopy(bank);bad['confirmation']['confirmation_seed'] += 1
    with pytest.raises(ValueError, match='digest'):
        validate_confirmation_bank(bad)


def test_parent_verifier_rejects_incomplete_exclusions_after_rehash(banks):
    parent, bank = banks
    bad = copy.deepcopy(bank)
    # Remove a retired hash and its provenance together: the parent comparison
    # still rejects this internally consistent attempt to forget a tested map.
    removed = bad['confirmation']['retired_milestone_test'].pop()['maze_hash']
    bad['confirmation']['excluded_parent_layout_hashes'].remove(removed)
    bad['manifest_sha256'] = manifest_digest(bad)
    with pytest.raises(ValueError, match='parent layout exclusion'):
        verify_confirmation_bank(parent, bad)


def test_raw_confirmation_digest_flows_into_frozen_evaluation_plan(banks):
    _, bank = banks
    plan = dict(version=PLAN_VERSION, status='frozen', partition='milestone_test',
        scene_bank_sha256=bank['manifest_sha256'], object_indices=CONFIRMATION_OBJECTS,
        cases=evaluation_case_specs(bank, 'milestone_test', allow_heldout=True),
        original_start=True, inter_skill_resets=False, teacher_at_inference=False,
        original_gripper=True, retain_all_failures=True,
        execution=dict(delta_scale=.075, planned_speed_scale=1., deadline_physics_ticks=15000,
            grasp_horizon=600, carry_horizon=1200, reach_horizon=150,
            grasp_lift_m=.24, reach_tolerance_m=.08, grasp_proposal_mode='clearance-v2'),
        checkpoints={key:{'sha256':'a'*64,'frozen_path':'unused.zip'} for key in ('grasp','carry','reach')})
    plan['plan_sha256'] = plan_digest(plan)
    verify_frozen_plan(plan, bank, allow_heldout=True, verify_checkpoints=False)
    assert bank['version'] == CONFIRMATION_VERSION
    assert plan['scene_bank_sha256'] == manifest_digest(bank)
    with pytest.raises(ValueError, match='Held-out'):
        verify_frozen_plan(plan, bank, verify_checkpoints=False)
