"""Offline comparison measures and paired sampling; never create physics."""
import numpy as np

from scripts.phase2_encoder_offline_screen import masks, sample_schedule, score, summarize


def test_direction_requires_axis_sign_and_nontrivial_magnitude():
    teacher = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    correct = score(teacher, teacher)
    assert correct['dominant_direction_accuracy'] == 1.
    wrong = score(np.array([[0., 1.], [1., 0.], [0., .01], [0., -1.]]), teacher)
    assert wrong['dominant_direction_accuracy'] == .25
    assert wrong['strong_axis_sign_accuracy'] == .5


def test_backtracking_stratum_requires_measured_interior_position():
    teacher = np.array([[-1., 0.], [-1., 0.], [-.5, 0.], [1., 0.]])
    position = np.array([[3.6 / 20, 0.], [3.5 / 20, 0.], [4. / 20, 0.], [4. / 20, 0.]])
    assert masks(teacher, position)['westward_return'].tolist() == [True, False, False, False]


def test_paired_schedule_is_reproducible_and_respects_support():
    groups = {'one': np.array([1, 4]), 'two': np.array([9, 10])}
    first = sample_schedule(groups, np.array([.3, .7]), steps=5, batch=64, seed=2)
    second = sample_schedule(groups, np.array([.3, .7]), steps=5, batch=64, seed=2)
    assert np.array_equal(first, second)
    assert set(first.flatten()) == {1, 4, 9, 10}
    only_two = sample_schedule(groups, np.array([0., 1.]), steps=2, batch=64, seed=2)
    assert set(only_two.flatten()) == {9, 10}


def test_macro_maps_differ_from_imbalanced_row_weighting():
    teacher = np.tile([1., 0.], (4, 1))
    prediction = np.array([[-1., 0.], [1., 0.], [1., 0.], [1., 0.]])
    result = summarize(prediction, teacher, np.zeros((4, 2)), np.array(['a', 'b', 'b', 'b']))
    assert result['row_weighted']['all']['dominant_direction_accuracy'] == .75
    assert result['map_macro']['all']['dominant_direction_accuracy'] == .5
    assert result['row_weighted']['hold']['mse'] is None
    assert result['map_macro']['westward_return']['maps_with_rows'] == 0
