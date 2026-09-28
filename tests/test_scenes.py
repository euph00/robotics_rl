"""Scene diversity, leakage boundaries, and unchanged legacy-scene regression."""
import copy
import hashlib
import random

import numpy as np
import pytest

from pnr.env import PickNavReachEnv
from pnr.scenes import (
    DEFAULT_COUNTS, SEED_MODE, bank_summary, build_scene_bank, case_specs,
    generate_layout, manifest_digest, map_hash, scene_records, shortest_grid_route,
    validate_scene_bank,
)


@pytest.fixture(scope='module')
def small_bank():
    return build_scene_bank(counts={'train': 3, 'validation': 2, 'milestone_test': 2, 'final_reserve': 2})


def test_layout_determinism_geometry_and_rng_isolation():
    np.random.seed(718)
    random.seed(911)
    numpy_expected = np.random.RandomState(718).random_sample(5)
    python_expected = random.Random(911).random()
    one = generate_layout(11223)
    two = generate_layout(11223)
    np.testing.assert_array_equal(np.random.random_sample(5), numpy_expected)
    assert random.random() == python_expected
    np.testing.assert_array_equal(one.maze_map, two.maze_map)
    assert one.maze_hash == hashlib.sha256(one.maze_map.tobytes()).hexdigest()
    assert one.maze_map.shape == (12, 22) and one.maze_map.dtype == np.int8
    assert one.goal_pos.tolist() == [17.5, 5.5 - one.exit_cell[0], .9]
    route = shortest_grid_route(one.maze_map, one.entrance_cell, one.exit_cell)
    assert route[0] == one.entrance_cell and route[-1] == one.exit_cell
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(route, route[1:]))
    assert all(one.maze_map[p] == 0 for p in route)
    assert all(one.maze_map[0]) and all(one.maze_map[-1]) and all(one.maze_map[:, 0])
    assert generate_layout(11224).maze_hash != one.maze_hash


def test_bank_content_partitions_and_reproducibility(small_bank):
    other = build_scene_bank(counts=small_bank['partition_counts'])
    assert small_bank == other
    hashes = [record['maze_hash'] for partition in DEFAULT_COUNTS
              for record in small_bank['partitions'][partition]]
    assert len(hashes) == len(set(hashes)) == 9
    assert not set(hashes) & set(small_bank['excluded_course_legacy_hashes'])
    assert small_bank['rules']['route_length_filter'] is False
    assert small_bank['rules']['learned_outcome_filter'] is False
    assert small_bank['object_indices'] == []
    summary = bank_summary(small_bank)
    assert set(summary['development_route_lengths_m']) == {'train', 'validation'}
    assert len(summary['heldout_hashes']['milestone_test']) == 2


def test_heldout_cases_require_explicit_access_and_valid_objects(small_bank):
    with pytest.raises(ValueError, match='Declare the object set'):
        case_specs(small_bank)
    with pytest.raises(ValueError, match='Held-out'):
        scene_records(small_bank, 'milestone_test')
    with pytest.raises(ValueError, match='Held-out'):
        case_specs(small_bank, 'final_reserve', object_indices=[14])
    with pytest.raises(ValueError, match='distinct'):
        case_specs(small_bank, object_indices=[14, 14])
    cases = case_specs(small_bank, 'milestone_test', object_indices=[0, 2, 4, 12, 14], allow_heldout=True)
    assert len(cases) == 10
    assert len({c['case_id'] for c in cases}) == 10
    assert len({c['env_kwargs']['seed'] for c in cases}) == 10
    assert all(c['env_kwargs']['seed_mode'] == SEED_MODE for c in cases)
    assert cases[0]['env_kwargs']['maze_seed'] == cases[4]['env_kwargs']['maze_seed']
    assert cases == case_specs(small_bank, 'milestone_test', object_indices=[0, 2, 4, 12, 14], allow_heldout=True)


def test_bank_rejects_manifest_tampering_and_seed_content_mismatch(small_bank):
    changed = copy.deepcopy(small_bank)
    changed['partition_seed'] += 1
    with pytest.raises(ValueError, match='digest'):
        validate_scene_bank(changed)
    changed = copy.deepcopy(small_bank)
    changed['partitions']['train'][0]['maze_seed'] += 1
    changed['manifest_sha256'] = manifest_digest(changed)
    with pytest.raises(ValueError, match='reproduce'):
        validate_scene_bank(changed)
    changed = copy.deepcopy(small_bank)
    changed['partitions']['validation'][0] = copy.deepcopy(changed['partitions']['train'][0])
    changed['manifest_sha256'] = manifest_digest(changed)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_scene_bank(changed)


def test_native_legacy_scene_matches_frozen_milestone():
    with PickNavReachEnv(seed=42, object_idx=14, settle_steps=0) as env:
        assert env.seed_mode == 'course-legacy-v1'
        assert env.maze_seed is None
        assert env.scene_manifest()['maze_hash'] == '8e58110b661405f9069839761a007fe4ec8b599ba4cfdaaf1194aaf45416f90d'
        with pytest.raises(ValueError, match='requires independent'):
            env.reset(maze_seed=19)


def test_native_independent_layout_decouples_points_objects_and_maze():
    with PickNavReachEnv(seed=17, object_idx=14, maze_seed=81, settle_steps=0) as env:
        initial_pc = env.object_canonical_pc.copy()
        initial_map = env.maze_map.copy()
        expected = generate_layout(81)
        np.testing.assert_array_equal(env.maze_map, expected.maze_map)
        np.testing.assert_array_equal(env.cube_positions[:, :2], expected.wall_centers)
        np.testing.assert_array_equal(env.goal_pos, expected.goal_pos)
        assert env.scene_manifest()['maze_seed'] == 81
        assert env.substeps == 5 and env.max_force == 2_000_000
        env.step(env.init_qpos)
        assert env.physics_steps == 5
        env.reset(maze_seed=82)
        np.testing.assert_array_equal(initial_pc, env.object_canonical_pc)
        assert map_hash(env.maze_map) != map_hash(initial_map)
        env.reset(seed=18, maze_seed=81)
        np.testing.assert_array_equal(initial_map, env.maze_map)
        assert not np.array_equal(initial_pc, env.object_canonical_pc)
        assert env.physics_steps == 0
    with PickNavReachEnv(seed=99, object_idx=12, maze_seed=81, settle_steps=0) as env:
        np.testing.assert_array_equal(initial_map, env.maze_map)


@pytest.mark.parametrize('kwargs', [
    {'seed_mode': 'unknown'}, {'seed_mode': 'independent-maze-v1'},
    {'seed_mode': 'course-legacy-v1', 'maze_seed': 3}, {'maze_seed': -1},
])
def test_native_invalid_seed_modes_fail_before_client_creation(kwargs):
    with pytest.raises(ValueError):
        PickNavReachEnv(**kwargs)
