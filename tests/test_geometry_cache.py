import numpy as np
import pytest

import spde_covariance.geometry as geometry


def _reference(operators, weights):
    return sum(
        weight * geometry.lifted_matrix(operator).T @ geometry.lifted_matrix(operator)
        for operator, weight in zip(operators, weights, strict=True)
    )


def _count_lifts(monkeypatch):
    original = geometry.lifted_matrix
    calls = []

    def counted(operator):
        calls.append(operator.copy())
        return original(operator)

    monkeypatch.setattr(geometry, "lifted_matrix", counted)
    return calls


def test_repeated_normals_match_individual_weighted_reference(monkeypatch):
    first = np.array([[1.0, 0.2], [-0.4, 2.0]])
    second = np.array([[0.3, -0.7]])
    operators = [first, second, first.copy(), np.array(first, order="F"), second.copy()]
    weights = [0.1, 0.7, 0.2, 1.3, 0.4]
    expected = _reference(operators, weights)
    calls = _count_lifts(monkeypatch)
    actual = geometry.gram_operator(operators, weights)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    assert len(calls) == 2
    # Explicit weights sum to 2.7 and must not be normalized by the cache.
    normalized = geometry.gram_operator(operators, np.asarray(weights) / sum(weights))
    np.testing.assert_allclose(actual, sum(weights) * normalized, atol=2e-14)


def test_zero_budget_disables_cache_and_keeps_default_weights(monkeypatch):
    operator = np.array([[1.0, 0.2]])
    operators = [operator, operator.copy(), operator.copy()]
    expected = _reference(operators, np.full(3, 1 / 3))
    calls = _count_lifts(monkeypatch)
    actual = geometry.gram_operator(operators, cache_max_bytes=0)
    np.testing.assert_allclose(actual, expected)
    assert len(calls) == 3


def test_lru_hit_updates_recency_before_eviction(monkeypatch):
    first = np.array([[1.0, 0.0]])
    second = np.array([[0.0, 1.0]])
    third = np.array([[1.0, -1.0]])
    operators = [first, second, first.copy(), third, second.copy()]
    weights = [0.1, 0.2, 0.3, 0.15, 0.25]
    expected = _reference(operators, weights)
    entry_bytes = geometry.lifted_matrix(first).shape[1] ** 2 * 8 + first.nbytes
    calls = _count_lifts(monkeypatch)
    actual = geometry.gram_operator(operators, weights, cache_max_bytes=2 * entry_bytes)
    np.testing.assert_allclose(actual, expected)
    assert len(calls) == 4
    # Reusing first moves it ahead of second; third therefore evicts second.
    for actual_operator, expected_operator in zip(
        calls, [first, second, third, second], strict=True
    ):
        np.testing.assert_array_equal(actual_operator, expected_operator)


def test_key_bytes_count_toward_budget(monkeypatch):
    operator = np.array([[1.0, 0.5]])
    size = geometry.lifted_matrix(operator).shape[1]
    normal_bytes = size**2 * 8
    calls = _count_lifts(monkeypatch)
    geometry.gram_operator([operator, operator.copy()], cache_max_bytes=normal_bytes)
    assert len(calls) == 2


def test_oversized_entry_does_not_evict_a_cacheable_design(monkeypatch):
    small = np.array([[1.0, -0.2]])
    large = np.arange(10, dtype=float).reshape(5, 2) / 10
    operators = [small, large, small.copy(), large.copy(), small.copy()]
    weights = [0.1, 0.2, 0.3, 0.15, 0.25]
    expected = _reference(operators, weights)
    size = geometry.lifted_matrix(small).shape[1]
    budget = size**2 * 8 + small.nbytes
    calls = _count_lifts(monkeypatch)
    actual = geometry.gram_operator(operators, weights, cache_max_bytes=budget)
    np.testing.assert_allclose(actual, expected)
    assert len(calls) == 3
    assert [operator.shape for operator in calls] == [small.shape, large.shape, large.shape]


def test_zero_weight_designs_are_not_lifted(monkeypatch):
    first = np.array([[1.0, -0.2]])
    second = np.array([[0.3, 1.0]])
    expected = 3 * _reference([first], [1])
    calls = _count_lifts(monkeypatch)
    actual = geometry.gram_operator([first, second, first.copy()], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(actual, expected)
    assert len(calls) == 1


def test_cache_is_local_to_each_call_and_observes_changed_inputs(monkeypatch):
    operator = np.array([[1.0, 0.2]])
    calls = _count_lifts(monkeypatch)
    first = geometry.gram_operator([operator])
    repeated = geometry.gram_operator([operator])
    operator[0, 0] = 2.0
    changed = geometry.gram_operator([operator])
    assert len(calls) == 3
    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(changed, first)


def test_small_weight_can_rescale_an_overflowing_unweighted_normal():
    operator = np.array([[1e80]])
    # The unweighted normal would be 1e320, but the weighted Gram is 1e20.
    actual = geometry.gram_operator([operator, operator.copy()], [1e-300, 1e-300])
    np.testing.assert_allclose(actual, [[2e20]], rtol=1e-14)


@pytest.mark.parametrize("budget", [-1, 0.5, True, None, np.nan, "1024"])
def test_cache_budget_requires_nonnegative_integer(budget):
    with pytest.raises(ValueError, match="cache_max_bytes"):
        geometry.gram_operator([np.eye(2)], cache_max_bytes=budget)


def test_numpy_integer_budget_is_accepted():
    np.testing.assert_allclose(
        geometry.gram_operator([np.eye(2)], cache_max_bytes=np.int64(0)), np.eye(3)
    )
