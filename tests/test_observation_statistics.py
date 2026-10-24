import numpy as np
import pytest

from spde_covariance.fragments import FragmentLayout
from spde_covariance.observation_statistics import (
    covariance_quadratic_score,
    kernel_quadratic_score,
    kernel_quadratic_score_from_statistics,
    partial_kernel_from_arrays,
    prediction_quadratic_score,
    risk_proxy_summary,
    summarize_observations,
)
from spde_covariance.observations import Observation
from spde_covariance.partial_benchmark import observed_quadratic_score


def _example():
    rows = (np.array([0, 1]), np.array([1, 2]))
    values = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
    observations = tuple(
        Observation(float(index), y, np.eye(3)[row], 0.5 * np.eye(2))
        for index, (y, row) in enumerate(zip(values, rows, strict=True))
    )
    return rows, values, observations


def test_pair_corrected_means_counts_missing_mask_and_effective_weights():
    rows, values, _ = _example()
    result = partial_kernel_from_arrays(values, rows, 0.5, size=3)
    np.testing.assert_array_equal(result.pair_counts, [[1, 1, 0], [1, 2, 1], [0, 1, 1]])
    np.testing.assert_allclose(result.partial_kernel, [[3.5, 6, 0], [6, 4.5, -4], [0, -4, 15.5]])
    assert not result.observed_mask[0, 2]
    weighted = partial_kernel_from_arrays(values, rows, 0.5, [1.0, 3.0], size=3)
    np.testing.assert_allclose(weighted.partial_kernel[1, 1], 2.5)
    np.testing.assert_allclose(weighted.pair_effective_counts[1, 1], 1.6)
    np.testing.assert_allclose(weighted.weighted_counts, weighted.pair_weight_sums)
    zero_weight = partial_kernel_from_arrays(values, rows, weights=[1.0, 0.0], size=3)
    assert zero_weight.pair_counts[2, 2] == 1 and not zero_weight.observed_mask[2, 2]
    assert zero_weight.partial_kernel[2, 2] == 0


def test_layout_observation_and_raw_statistics_have_the_same_corrected_moments():
    rows, values, observations = _example()
    layout = FragmentLayout(
        [0, 0.5, 1],
        [0.25, 0.5, 0.25],
        ((0, 0.5), (0.5, 1)),
        rows,
        evaluation_matrix=np.eye(3),
        basis_id="grid",
    )
    grouped = summarize_observations(observations, layout=layout)
    raw = partial_kernel_from_arrays(values, rows, 0.5)
    np.testing.assert_array_equal(grouped.partial_kernel, raw.partial_kernel)
    np.testing.assert_array_equal(grouped.pair_counts, raw.pair_counts)
    assert grouped.layout_id == layout.layout_id and len(grouped.groups) == 2
    for array in (
        grouped.weights,
        grouped.groups[0].corrected_sum,
        raw.partial_kernel,
        raw.observed_mask,
        raw.groups[0].row_to_nodes,
    ):
        assert not array.flags.writeable


def test_dense_parent_arrays_are_not_silently_interpreted_as_masked_values():
    rows, _, _ = _example()
    with pytest.raises(ValueError, match="value vector"):
        partial_kernel_from_arrays(np.ones((2, 3)), rows, size=3)


def _legacy_score(matrix, data):
    groups = {}
    for item in data:
        key = (item.operator.shape, item.operator.tobytes())
        if key not in groups:
            groups[key] = [item.operator, 0, np.zeros((item.n_contracts, item.n_contracts))]
        groups[key][1] += 1
        groups[key][2] += item.measurement
    total = 0.0
    for operator, count, corrected_sum in groups.values():
        predicted = operator @ matrix @ operator.T
        total += count * np.sum(predicted**2) - 2 * np.sum(predicted * corrected_sum)
    return float(total / len(data))


def test_legacy_grouped_score_retains_exact_floating_arithmetic_and_psd_guard():
    rng = np.random.default_rng(7021)
    operators = [rng.normal(size=(2, 4)), rng.normal(size=(3, 4))]
    data = tuple(
        Observation(
            float(index),
            rng.normal(size=len(operators[index % 2])),
            operators[index % 2],
            np.eye(len(operators[index % 2])) * 0.001,
        )
        for index in range(37)
    )
    factor = rng.normal(size=(4, 4))
    covariance = factor @ factor.T
    assert observed_quadratic_score(covariance, data) == _legacy_score(covariance, data)
    assert covariance_quadratic_score(covariance, data) == _legacy_score(covariance, data)
    with pytest.raises(ValueError, match="positive semidefinite"):
        observed_quadratic_score(-np.eye(4), data)


def test_kernel_prediction_and_grouped_statistic_scores_agree_for_indefinite_kernel():
    rows, values, observations = _example()
    kernel = np.array([[1.0, 2.0, 0.3], [2.0, 1.0, -0.5], [0.3, -0.5, 0.1]])
    assert np.linalg.eigvalsh(kernel)[0] < 0
    predictions = [kernel[np.ix_(row, row)] for row in rows]
    expected = np.mean(
        [
            np.sum(predicted**2) - 2 * np.sum(predicted * item.measurement)
            for predicted, item in zip(predictions, observations, strict=True)
        ]
    )
    raw = partial_kernel_from_arrays(values, rows, 0.5)
    np.testing.assert_allclose(kernel_quadratic_score(kernel, observations, rows), expected)
    np.testing.assert_allclose(kernel_quadratic_score_from_statistics(kernel, raw), expected)
    np.testing.assert_allclose(prediction_quadratic_score(predictions, observations), expected)


def test_weighted_scores_are_normalized_and_large_weight_safe():
    rows, values, observations = _example()
    matrix = np.eye(3)
    weights = np.array([1.0, 3.0])
    losses = [
        np.sum(matrix[np.ix_(row, row)] ** 2)
        - 2 * np.sum(matrix[np.ix_(row, row)] * item.measurement)
        for row, item in zip(rows, observations, strict=True)
    ]
    expected = np.average(losses, weights=weights)
    np.testing.assert_allclose(
        kernel_quadratic_score(matrix, observations, rows, weights), expected
    )
    np.testing.assert_allclose(covariance_quadratic_score(matrix, observations, weights), expected)
    first = partial_kernel_from_arrays(values, rows, 0.5, weights)
    huge = partial_kernel_from_arrays(values, rows, 0.5, weights * 1e307)
    np.testing.assert_allclose(first.partial_kernel, huge.partial_kernel)
    np.testing.assert_allclose(first.pair_effective_counts, huge.pair_effective_counts)


def test_correlated_known_noise_correction_is_preserved():
    rows = ([0, 1], [0, 1])
    noise = np.array([[1.0, 0.2], [0.2, 0.5]])
    data = tuple(
        Observation(float(i), y, np.eye(2), noise) for i, y in enumerate([[1.0, 2.0], [-1.0, 3.0]])
    )
    layout = FragmentLayout([0.0, 1.0], [0.5, 0.5], ((0.0, 1.0),), rows)
    stats = summarize_observations(data, layout=layout)
    expected = np.mean([item.measurement for item in data], axis=0)
    np.testing.assert_allclose(stats.partial_kernel, expected)


def test_same_date_grid_risk_availability_noise_correction_and_negative_values():
    rows = ([2, 0], [0, 1])
    data = (
        Observation(0.0, [0.0, 0.0], np.eye(3)[[2, 0]], np.eye(2)),
        Observation(1.0, [1.0, 2.0], np.eye(3)[[0, 1]], np.eye(2)),
    )
    result = risk_proxy_summary([1.0, 0.0, -1.0], data, rows)
    np.testing.assert_array_equal(result["available_mask"], [True, False])
    np.testing.assert_array_equal(result["available_indices"], [0])
    np.testing.assert_array_equal(result["values"], [-2.0])
    assert result["mean"] == -2.0 and result["n_available"] == 1
    unavailable = risk_proxy_summary([0.0, 1.0, -1.0], data, rows)
    assert unavailable["mean"] is None and unavailable["n_available"] == 0
    zero = risk_proxy_summary(np.zeros(3), data, rows)
    np.testing.assert_array_equal(zero["values"], [0.0, 0.0])


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [1], [1, np.nan]])
def test_invalid_date_weights_are_rejected(weights):
    rows, values, _ = _example()
    with pytest.raises(ValueError):
        partial_kernel_from_arrays(values, rows, weights=weights)
