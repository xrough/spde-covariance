"""Canonical completion, singular controls, truncation, and risk witnesses."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from spde_covariance.completion import (
    complete_kernel,
    observed_mask,
    two_block_risk_range,
    weighted_psd_projection,
)


def partial(matrix, cliques):
    result = np.asarray(matrix, dtype=float).copy()
    result[~observed_mask(cliques, len(result))] = np.nan
    return result


def positive_case():
    nodes = np.linspace(0, 1, 11)
    matrix = np.exp(-3 * np.abs(nodes[:, None] - nodes)) + np.eye(11) * 0.2
    return matrix, (np.arange(7), np.arange(4, 11))


def truncation_case():
    # A and B both equal the second separator coordinate. Their missing
    # covariance is1; dropping that smaller separator mode instead fills0.
    matrix = np.array([[1, 0, 1, 1], [0, 2, 0, 0], [1, 0, 1, 1], [1, 0, 1, 1]], dtype=float)
    cliques = (np.arange(3), np.arange(1, 4))
    return partial(matrix, cliques), cliques


def test_weighted_two_block_formula_and_observed_preservation():
    matrix, cliques = positive_case()
    weights = np.linspace(0.01, 0.3, len(matrix))
    result = complete_kernel(
        partial(matrix, cliques), cliques=cliques, weights=weights, population=True
    )
    assert result.available and result.diagnostics["output_psd"]
    a, s, b = np.arange(4), np.arange(4, 7), np.arange(7, 11)
    expected = matrix[np.ix_(a, s)] @ np.linalg.solve(matrix[np.ix_(s, s)], matrix[np.ix_(s, b)])
    np.testing.assert_allclose(result.kernel[np.ix_(a, b)], expected, atol=1e-13)
    mask = observed_mask(cliques, len(matrix))
    np.testing.assert_array_equal(result.kernel[mask], matrix[mask])
    assert result.diagnostics["observed_preservation_residual"] == 0
    join = result.diagnostics["joins"][0]
    assert join["support_compatible"]
    assert join["selected_rank"] == len(s)
    assert join["inverse_norm"] > 0
    assert join["inverse_perturbation_multiplier"] >= join["cross_norm_bound"]
    assert result.diagnostics["numerical_error_certificate"] is False
    scaled = complete_kernel(
        partial(matrix, cliques), cliques=cliques, weights=weights * 7, population=True
    )
    np.testing.assert_allclose(result.kernel, scaled.kernel, atol=1e-13)


@pytest.mark.parametrize("count", [3, 5])
def test_population_recursive_completion_is_order_invariant(count):
    nodes = np.linspace(0, 1, 41)
    matrix = np.exp(-2 * np.abs(nodes[:, None] - nodes))
    intervals = [(start, start + 0.6) for start in np.linspace(0, 0.4, count)]
    cliques = tuple(
        np.flatnonzero((nodes >= lo - 1e-12) & (nodes <= hi + 1e-12)) for lo, hi in intervals
    )
    kernel = partial(matrix, cliques)
    forward = complete_kernel(
        kernel, cliques=cliques, weights=np.linspace(0.02, 0.03, 41), population=True
    )
    backward = complete_kernel(
        kernel, cliques=cliques[::-1], weights=np.linspace(0.02, 0.03, 41), population=True
    )
    assert forward.available and backward.available
    np.testing.assert_allclose(forward.kernel, backward.kernel, atol=2e-13)
    # The exponential Markov covariance is its own canonical completion.
    np.testing.assert_allclose(forward.kernel, matrix, atol=2e-13)


def test_singular_finite_rank_completion_recovers_kernel():
    nodes = np.linspace(0, 1, 31)
    features = np.column_stack([np.ones(31), nodes, nodes**2, nodes**3])
    matrix = features @ np.diag([1.0, 0.5, 0.25, 0.125]) @ features.T
    cliques = (np.arange(19), np.arange(12, 31))
    result = complete_kernel(partial(matrix, cliques), cliques=cliques, population=True)
    assert result.available
    assert result.diagnostics["joins"][0]["support_rank"] == 4
    np.testing.assert_allclose(result.kernel, matrix, atol=2e-9)


def test_singular_separator_compatibility_and_incompatibility_are_explicit():
    matrix = np.array([[1, 1, 0, 0.5], [1, 2, 0, 1], [0, 0, 0, 0], [0.5, 1, 0, 1]], dtype=float)
    cliques = (np.arange(3), np.arange(1, 4))
    kernel = partial(matrix, cliques)
    result = complete_kernel(kernel, cliques=cliques, population=True)
    assert result.available
    assert result.diagnostics["joins"][0]["support_rank"] == 1
    np.testing.assert_allclose(result.kernel, matrix, atol=1e-14)
    invalid = kernel.copy()
    invalid[0, 2] = invalid[2, 0] = 0.2
    failed = complete_kernel(invalid, cliques=cliques, population=True)
    assert not failed.available and failed.kernel is None
    assert not failed.diagnostics["input_cliques_psd"]
    assert not failed.diagnostics["joins"][0]["support_compatible"]


def test_fve_is_fraction_of_total_trace_and_manual_rank_is_not_capped():
    matrix = np.array(
        [[2, 0.3, 0.1, 0], [0.3, 9, 0, 0.2], [0.1, 0, 1, 0.3], [0, 0.2, 0.3, 2]], dtype=float
    )
    cliques = (np.arange(3), np.arange(1, 4))
    kernel = partial(matrix, cliques)
    result = complete_kernel(kernel, cliques=cliques, fve=0.95)
    assert result.available
    assert result.diagnostics["joins"][0]["selected_rank"] == 2
    assert result.diagnostics["joins"][0]["achieved_fve"] == pytest.approx(1)
    smaller = complete_kernel(kernel, cliques=cliques, fve=0.89)
    assert smaller.diagnostics["joins"][0]["selected_rank"] == 1
    assert smaller.diagnostics["joins"][0]["achieved_fve"] == pytest.approx(0.9)
    # The paper uses strictly greater cumulative trace, including exact ties.
    boundary = complete_kernel(kernel, cliques=cliques, fve=0.9)
    assert boundary.diagnostics["joins"][0]["selected_rank"] == 2
    failed = complete_kernel(kernel, cliques=cliques, rank=3)
    assert not failed.available and failed.kernel is None
    assert "support" in failed.diagnostics["reason"]
    assert failed.diagnostics["joins"][0]["requested_rank"] == 3


def test_fve_indefinite_separator_is_unavailable_without_hidden_clipping():
    matrix = np.eye(4)
    matrix[2, 2] = -0.2
    cliques = (np.arange(3), np.arange(1, 4))
    result = complete_kernel(partial(matrix, cliques), cliques=cliques, fve=0.95)
    assert not result.available and result.kernel is None
    assert "indefinite separator" in result.diagnostics["reason"]
    assert not result.diagnostics["joins"][0]["separator_psd"]


def test_truncation_preserves_valid_partial_data_but_can_be_indefinite():
    kernel, cliques = truncation_case()
    exact = complete_kernel(kernel, cliques=cliques, population=True)
    raw = complete_kernel(kernel, cliques=cliques, rank=1)
    assert exact.available and exact.diagnostics["output_psd"]
    assert raw.available and raw.diagnostics["input_cliques_psd"]
    assert not raw.diagnostics["output_psd"]
    assert raw.kernel[0, 3] == 0
    assert exact.kernel[0, 3] == pytest.approx(1)
    assert raw.diagnostics["output_minimum_eigenvalue"] == pytest.approx(1 - np.sqrt(2))
    mask = observed_mask(cliques, 4)
    np.testing.assert_array_equal(raw.kernel[mask], kernel[mask])
    assert raw.diagnostics["joins"][0]["support_compatible"]
    assert raw.diagnostics["joins"][0]["retained_left_residual"] > 0


def test_observed_validation_scores_cannot_distinguish_completion_ranks():
    kernel, cliques = truncation_case()
    first = complete_kernel(kernel, cliques=cliques, rank=1)
    second = complete_kernel(kernel, cliques=cliques, rank=2)
    rng = np.random.default_rng(20261025)
    scores = []
    for result in (first, second):
        score = 0.0
        for rows in cliques:
            predicted = result.kernel[np.ix_(rows, rows)]
            rng = np.random.default_rng(52)  # Identical observed validation values.
            y = rng.normal(size=len(rows))
            corrected = np.outer(y, y) - np.eye(len(rows)) * 0.1
            score += np.sum(predicted**2) - 2 * np.sum(predicted * corrected)
        scores.append(score)
    assert scores[0] == scores[1]


def test_weighted_projection_is_separate_and_reports_observed_changes():
    kernel, cliques = truncation_case()
    raw = complete_kernel(kernel, cliques=cliques, rank=1)
    weights = np.array([0.1, 0.3, 0.4, 0.2])
    mask = observed_mask(cliques, 4)
    result = weighted_psd_projection(raw.kernel, weights=weights, observed_mask=mask)
    assert np.linalg.eigvalsh(result.kernel)[0] >= -1e-13
    assert result.diagnostics["observed_weighted_hs_change"] > 0
    assert result.diagnostics["observed_maximum_change"] > 0
    assert not result.diagnostics["preserves_observed_entries"]
    root = np.sqrt(weights)
    eigenvalues = np.linalg.eigvalsh(root[:, None] * raw.kernel * root[None, :])
    expected = np.linalg.norm(np.minimum(eigenvalues, 0))
    assert result.diagnostics["weighted_hs_change"] == pytest.approx(expected)
    assert result.diagnostics["weighted_trace_before"] == pytest.approx(
        np.dot(weights, np.diag(raw.kernel))
    )
    assert result.diagnostics["weighted_trace_change"] == pytest.approx(
        -np.minimum(eigenvalues, 0).sum()
    )
    assert result.diagnostics["kernel_diagonal_sum_change"] == pytest.approx(
        np.trace(result.kernel) - np.trace(raw.kernel)
    )
    assert "neither is H-operator trace" in result.diagnostics["trace_scope"]
    np.testing.assert_array_equal(raw.kernel[mask], kernel[mask])


def test_exposure_risk_interval_endpoints_are_attainable_psd_completions():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    kernel = np.array([[1, 0.2, np.nan], [0.2, 1, 0.4], [np.nan, 0.4, 2]])
    coefficients = np.array([1, -0.3, -2])
    result = two_block_risk_range(kernel, coefficients, cliques=cliques, weights=[0.2, 0.5, 0.3])
    assert result.available and not result.identified
    assert result.diagnostics["left_conditional_variance"] == pytest.approx(0.96)
    assert result.diagnostics["right_conditional_variance"] == pytest.approx(4 * 1.84)
    assert result.upper - result.lower == pytest.approx(4 * np.sqrt(0.96 * 4 * 1.84))
    mask = observed_mask(cliques, 3)
    for endpoint, value in (
        (result.lower_kernel, result.lower),
        (result.upper_kernel, result.upper),
    ):
        assert np.linalg.eigvalsh(endpoint)[0] >= -1e-13
        np.testing.assert_array_equal(endpoint[mask], kernel[mask])
        assert coefficients @ endpoint @ coefficients == pytest.approx(value)


def test_zero_conditional_variance_identifies_target_and_zero_separator_is_valid():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    kernel = np.array([[1, 1, np.nan], [1, 1, 0.2], [np.nan, 0.2, 2]])
    result = two_block_risk_range(kernel, [1, 0, -1], cliques=cliques)
    assert result.available and result.identified
    assert result.lower == pytest.approx(result.upper)
    zero = np.diag([1.0, 0.0, 1.0])
    result = complete_kernel(partial(zero, cliques), cliques=cliques, fve=0.95)
    assert result.available
    assert result.diagnostics["joins"][0]["selected_rank"] == 0
    risk = two_block_risk_range(partial(zero, cliques), [1, 0, -1], cliques=cliques)
    assert risk.lower == pytest.approx(0)
    assert risk.upper == pytest.approx(4)


def test_disconnected_and_redundant_cliques_are_handled_explicitly():
    disconnected = (np.array([0, 1]), np.array([2, 3]))
    matrix = partial(np.eye(4), disconnected)
    result = complete_kernel(matrix, cliques=disconnected, population=True)
    assert result.available
    np.testing.assert_array_equal(result.kernel, np.eye(4))
    assert result.diagnostics["joins"][0]["selected_rank"] == 0
    redundant = (np.array([0, 1]), np.array([0, 1]), np.array([1, 2]), np.array([1, 2]))
    matrix = partial(np.eye(3), redundant)
    result = complete_kernel(matrix, cliques=redundant, population=True)
    assert result.available
    assert sum(row.get("skipped", False) for row in result.diagnostics["joins"]) == 2


def test_result_kernel_and_nested_diagnostics_are_frozen_snapshots():
    matrix, cliques = positive_case()
    weights = np.ones(len(matrix))
    result = complete_kernel(partial(matrix, cliques), cliques=cliques, weights=weights)
    weights[0] = 99
    assert result.diagnostics["weights"][0] == 1
    with pytest.raises(ValueError):
        result.kernel[0, 0] = 9
    with pytest.raises(ValueError):
        result.diagnostics["joins"][0]["eigenvalues"][0] = 0
    with pytest.raises(TypeError):
        result.diagnostics["joins"][0]["support_rank"] = 0
    with pytest.raises(FrozenInstanceError):
        result.kernel = np.eye(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": -1},
        {"rank": True},
        {"rank": 1.2},
        {"fve": 0},
        {"fve": 1},
        {"fve": 1.1},
        {"fve": np.nan},
        {"rank": 1, "fve": 0.95},
        {"support_rtol": 0},
        {"support_rtol": True},
        {"population": "yes"},
        {"population": True, "rank": 1},
        {"weights": [1, 0, 1]},
    ],
)
def test_bad_numerical_options_are_rejected(kwargs):
    cliques = (np.array([0, 1]), np.array([1, 2]))
    with pytest.raises(ValueError):
        complete_kernel(partial(np.eye(3), cliques), cliques=cliques, **kwargs)


def test_bad_partial_data_or_non_running_intersection_cover_is_rejected():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    matrix = partial(np.eye(3), cliques)
    asymmetric = matrix.copy()
    asymmetric[0, 1] = 1
    missing_observed = matrix.copy()
    missing_observed[1, 1] = np.nan
    for value in (asymmetric, missing_observed, np.eye(3), matrix.astype(complex)):
        with pytest.raises(ValueError):
            complete_kernel(value, cliques=cliques)
    for invalid in (
        (),
        (np.array([0, 1]),),
        (np.array([0, 1]), np.array([2, 1])),
        (np.array([0, 1]), np.array([1, 2]), np.array([0, 2])),
        (np.array([0.0, 1.0]), np.array([1, 2])),
    ):
        with pytest.raises(ValueError):
            complete_kernel(matrix, cliques=invalid)


def test_small_maximum_entropy_reference_matches_exact_completion():
    cp = pytest.importorskip("cvxpy")
    matrix, cliques = positive_case()
    mask = observed_mask(cliques, len(matrix))
    own = complete_kernel(partial(matrix, cliques), cliques=cliques, population=True)
    variable = cp.Variable(matrix.shape, symmetric=True)
    problem = cp.Problem(
        cp.Maximize(cp.log_det(variable)), [variable >> 0, variable[mask] == matrix[mask]]
    )
    problem.solve(solver="SCS", eps=1e-9, max_iters=100000)
    assert problem.status == "optimal"
    np.testing.assert_allclose(own.kernel, variable.value, atol=3e-7, rtol=0)
    precision = np.linalg.inv(own.kernel)
    assert np.linalg.norm(precision[~mask]) < 1e-12


def test_risk_bounds_match_independent_semidefinite_optimization():
    cp = pytest.importorskip("cvxpy")
    cliques = (np.array([0, 1]), np.array([1, 2]))
    kernel = np.array([[1, 0.2, np.nan], [0.2, 1, 0.4], [np.nan, 0.4, 2]])
    exposure = np.array([1, -0.3, -2])
    own = two_block_risk_range(kernel, exposure, cliques=cliques)
    mask = observed_mask(cliques, 3)
    variable = cp.Variable((3, 3), symmetric=True)
    value = cp.sum(cp.multiply(np.outer(exposure, exposure), variable))
    constraints = [variable >> 0, variable[mask] == kernel[mask]]
    for objective, expected in ((cp.Minimize(value), own.lower), (cp.Maximize(value), own.upper)):
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="SCS", eps=1e-9, max_iters=100000)
        assert problem.status == "optimal"
        assert problem.value == pytest.approx(expected, abs=1e-7)


def test_singular_maximum_entropy_reference_uses_the_known_common_support():
    cp = pytest.importorskip("cvxpy")
    matrix = np.array([[1, 1, 0, 0.5], [1, 2, 0, 1], [0, 0, 0, 0], [0.5, 1, 0, 1]], dtype=float)
    cliques = (np.arange(3), np.arange(1, 4))
    mask = observed_mask(cliques, 4)
    own = complete_kernel(partial(matrix, cliques), cliques=cliques, population=True)
    # The zero variance at node2 forces a common null direction in every PSD
    # completion. Reduce exactly to its orthogonal complement; add no jitter.
    support = np.eye(4)[:, [0, 1, 3]]
    reduced = cp.Variable((3, 3), symmetric=True)
    lifted = support @ reduced @ support.T
    problem = cp.Problem(
        cp.Maximize(cp.log_det(reduced)), [reduced >> 0, lifted[mask] == matrix[mask]]
    )
    problem.solve(solver="SCS", eps=1e-9, max_iters=100000)
    assert problem.status == "optimal"
    np.testing.assert_allclose(own.kernel, lifted.value, atol=3e-7, rtol=0)
    np.testing.assert_array_equal(own.kernel[2], np.zeros(4))
