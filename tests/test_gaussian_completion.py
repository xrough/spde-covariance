"""Independent likelihood, graphical M-step, conditioning and EM controls."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal

from spde_covariance import gaussian_completion as gc
from spde_covariance.gaussian_completion import (
    condition_gaussian_kernel,
    fit_gaussian_completion,
    gaussian_kernel_score,
)


def data(seed=170, count=80):
    rng = np.random.default_rng(seed)
    cliques = (np.arange(3), np.arange(2, 5))
    reference = np.exp(-np.abs(np.arange(5)[:, None] - np.arange(5))) + np.eye(5) * 0.2
    x = rng.multivariate_normal(np.zeros(5), reference, count)
    rows = tuple(cliques[i % 2] for i in range(count))
    y = tuple(x[i, row] + rng.normal(0, np.sqrt(0.1), len(row)) for i, row in enumerate(rows))
    return y, rows, cliques, reference


def fit_case(**options):
    y, rows, cliques, reference = data()
    return fit_gaussian_completion(
        y,
        rows,
        noise_covariances=0.1,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.1,
        **options,
    )


def penalty(precision, reference, regularization):
    return (
        regularization
        / 2
        * (
            np.sum(reference * precision)
            - np.linalg.slogdet(reference)[1]
            - np.linalg.slogdet(precision)[1]
            - len(reference)
        )
    )


def test_fully_observed_noiseless_closed_form_and_independent_convex_reference():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(19)
    y = rng.normal(size=(45, 3)) @ np.array([[1, 0.3, 0], [0, 0.8, 0.4], [0, 0, 1.1]])
    reference, regularization = np.diag([1.0, 0.5, 2.0]), 0.2
    expected = (y.T @ y / len(y) + regularization * reference) / (1 + regularization)
    result = fit_gaussian_completion(
        y,
        (np.arange(3),) * len(y),
        noise_covariances=0,
        cliques=(np.arange(3),),
        reference_kernel=reference,
        regularization=regularization,
        stationarity_tolerance=1e-9,
    )
    assert result.converged
    np.testing.assert_allclose(result.kernel, expected, atol=2e-14)
    theta = cp.Variable((3, 3), symmetric=True)
    problem = cp.Problem(cp.Minimize(cp.trace(expected @ theta) - cp.log_det(theta)), [theta >> 0])
    problem.solve(solver="SCS", eps=1e-10, max_iters=100000)
    assert problem.status == "optimal"
    np.testing.assert_allclose(result.precision, theta.value, rtol=1e-7, atol=1e-8)


def test_exact_graph_mstep_matches_clique_formula_and_independent_sdp():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(22)
    a = rng.normal(size=(7, 7))
    second = a @ a.T + np.eye(7)
    declared = ([2, 1, 0], [1, 2], [2, 4, 3], [2], [6, 5, 2], [2, 3, 4])
    cliques, separators, mask, _ = gc._graph(declared, 7)
    assert len(cliques) == 3
    assert tuple(tuple(s) for s in separators) == ((), (2,), (2,))
    kernel, precision = gc._m_step(second, cliques, separators)
    expected_precision = np.zeros((7, 7))
    for clique in cliques:
        expected_precision[np.ix_(clique, clique)] += np.linalg.inv(second[np.ix_(clique, clique)])
    expected_precision[2, 2] -= 2 / second[2, 2]
    np.testing.assert_allclose(precision, expected_precision, atol=3e-15)
    np.testing.assert_allclose(kernel @ precision, np.eye(7), atol=3e-15)
    for clique in cliques:
        np.testing.assert_allclose(
            kernel[np.ix_(clique, clique)], second[np.ix_(clique, clique)], atol=3e-15
        )
    theta = cp.Variable((7, 7), symmetric=True)
    constraints = [theta >> 0] + [theta[i, j] == 0 for i, j in zip(*np.where(~mask), strict=True)]
    problem = cp.Problem(cp.Minimize(cp.trace(second @ theta) - cp.log_det(theta)), constraints)
    problem.solve(solver="SCS", eps=1e-9, max_iters=100000)
    assert problem.status == "optimal"
    np.testing.assert_allclose(precision, theta.value, rtol=2e-6, atol=2e-7)


def test_full_bridges_are_observations_not_additions_to_model_graph():
    rng = np.random.default_rng(25)
    y = rng.normal(size=(60, 5))
    cliques = (np.arange(3), np.arange(2, 5))
    reference = np.eye(5)
    result = fit_gaussian_completion(
        y,
        (np.arange(5),) * len(y),
        noise_covariances=0,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.2,
    )
    assert result.converged
    assert np.max(np.abs(result.precision[:2, 3:])) == 0
    expected_second = (y.T @ y / len(y) + 0.2 * reference) / 1.2
    for clique in cliques:
        np.testing.assert_allclose(
            result.kernel[np.ix_(clique, clique)],
            expected_second[np.ix_(clique, clique)],
            atol=1e-14,
        )


def test_em_monotonicity_local_stationarity_multistart_and_no_global_claim():
    fit = fit_case(stationarity_tolerance=1e-7)
    assert fit.converged
    assert len(fit.diagnostics["attempts"]) == 3
    assert fit.diagnostics["relative_stationarity"] <= 1e-7
    assert "no global" in fit.diagnostics["certificate_scope"]
    assert not fit.diagnostics["silent_psd_repair"]
    for attempt in fit.diagnostics["attempts"]:
        assert np.max(np.diff(attempt["objective_history"]), initial=0) < 1e-12
    assert fit.diagnostics["objective"] == min(a["objective"] for a in fit.diagnostics["attempts"])
    assert np.linalg.eigvalsh(fit.kernel)[0] > 0
    assert np.linalg.eigvalsh(fit.precision)[0] > 0
    assert fit.diagnostics["outside_graph_precision_max"] == 0


def test_raw_noise_specific_grouped_likelihood_matches_independent_densities():
    kernel = np.array([[1, 0.3], [0.3, 0.8]])
    values = ([1.0, 2.0], [2.0, -1.0], [0.2])
    rows = ([0, 1], [1, 0], [0])
    noises = (np.diag([0.1, 0.2]), np.array([[0.3, 0.05], [0.05, 0.4]]), np.array([[0.5]]))
    weights = np.array([1.0, 2.0, 4.0])
    expected = np.average(
        [
            -multivariate_normal.logpdf(y, cov=kernel[np.ix_(row, row)] + noise)
            for y, row, noise in zip(values, rows, noises, strict=True)
        ],
        weights=weights,
    )
    assert gaussian_kernel_score(
        kernel, values, rows, noise_covariances=noises, weights=weights
    ) == pytest.approx(expected, abs=1e-14)
    wrong = gaussian_kernel_score(kernel, values, rows, noise_covariances=0, weights=weights)
    assert abs(wrong - expected) > 0.01
    grouped = gc._data(values, rows, noises, weights, 2)
    assert grouped.diagnostics["cached_groups"] == 3
    assert "raw moments" in grouped.diagnostics["grouping_mode"]


def test_analytic_precision_gradient_matches_finite_differences_away_from_optimum():
    values, rows, cliques, reference = data(count=16)
    _, _, mask, _ = gc._graph(cliques, 5)
    grouped = gc._data(values, rows, 0.1, None, 5)
    precision = np.diag(np.linspace(0.5, 1.5, 5))
    kernel = np.linalg.inv(precision)
    state = gc._evaluate(kernel, precision, grouped, reference, 0.2, mask)
    assert state["stationarity_norm"] > 0.01

    def objective(theta):
        k = np.linalg.inv(theta)
        return gaussian_kernel_score(k, values, rows, noise_covariances=0.1) + penalty(
            theta, reference, 0.2
        )

    for i, j in zip(*np.triu_indices(5), strict=True):
        direction = np.zeros((5, 5))
        direction[i, j] = direction[j, i] = 1
        step = 1e-5
        fd = (objective(precision + step * direction) - objective(precision - step * direction)) / (
            2 * step
        )
        assert fd == pytest.approx(np.sum(state["gradient"] * direction), abs=1e-9)


@pytest.mark.parametrize(
    "noise", [0.0, np.array([[0.0, 0], [0, 0.2]]), np.array([[0.2, 0.2], [0.2, 0.2]])]
)
def test_joseph_conditioning_handles_zero_and_rank_deficient_noise(noise):
    kernel = np.array([[1, 0.3, 0.4], [0.3, 1.2, 0.2], [0.4, 0.2, 0.9]])
    row = np.array([2, 0])
    y = np.array([1.0, -0.4])
    n = np.eye(2) * noise if np.ndim(noise) == 0 else noise
    sigma = kernel[np.ix_(row, row)] + n
    expected_mean = kernel[:, row] @ np.linalg.solve(sigma, y)
    expected_cov = kernel - kernel[:, row] @ np.linalg.solve(sigma, kernel[row])
    conditional = condition_gaussian_kernel(
        kernel, y, row, noise_covariance=noise, target_nodes=np.arange(3)
    )
    np.testing.assert_allclose(conditional.mean, expected_mean, atol=1e-15)
    np.testing.assert_allclose(conditional.covariance, expected_cov, atol=5e-16)
    assert np.linalg.eigvalsh(conditional.covariance)[0] >= -1e-15
    default = condition_gaussian_kernel(kernel, y, row, noise_covariance=noise)
    np.testing.assert_array_equal(default.target_nodes, [1])
    assert "excludes parameter" in conditional.diagnostics["scope"]


def test_psd_kernel_can_be_scored_and_conditioned_when_observation_covariance_is_spd():
    kernel = np.ones((3, 3))
    score = gaussian_kernel_score(kernel, ([1, 2],), ([0, 1],), noise_covariances=0.1)
    assert np.isfinite(score)
    conditional = condition_gaussian_kernel(kernel, [1, 2], [0, 1], noise_covariance=0.1)
    assert conditional.covariance[0, 0] >= 0
    with pytest.raises(ValueError, match="no jitter"):
        gaussian_kernel_score(kernel, ([1, 2],), ([0, 1],), noise_covariances=0)
    empty = condition_gaussian_kernel(kernel, [], [], noise_covariance=0)
    np.testing.assert_array_equal(empty.covariance, kernel)


def test_scalar_noisy_em_matches_independent_direct_observed_objective_minimum():
    values = np.array([[-1.2], [0.4], [0.8], [1.5]])
    fit = fit_gaussian_completion(
        values,
        ([0],) * 4,
        noise_covariances=0.2,
        cliques=([0],),
        reference_kernel=[[1.0]],
        regularization=0.1,
        stationarity_tolerance=1e-9,
    )

    def objective(variance):
        return gaussian_kernel_score(
            [[variance]], values, ([0],) * 4, noise_covariances=0.2
        ) + 0.05 * (1 / variance + np.log(variance) - 1)

    reference = minimize_scalar(
        objective, bounds=(1e-4, 10), method="bounded", options={"xatol": 1e-12}
    )
    assert fit.converged
    assert fit.kernel[0, 0] == pytest.approx(reference.x, abs=2e-7)
    assert fit.diagnostics["objective"] == pytest.approx(reference.fun, abs=1e-12)
    assert fit.diagnostics["negative_loglikelihood"] == pytest.approx(
        gaussian_kernel_score(fit.kernel, values, ([0],) * 4, noise_covariances=0.2)
    )


def test_fit_supports_per_date_singular_psd_noise_and_ragged_rows():
    rng = np.random.default_rng(36)
    values = (np.array([rng.normal()]), rng.normal(size=2), rng.normal(size=3)) * 8
    rows = ([1], [2, 0], [2, 0, 1]) * 8
    noises = (np.zeros((1, 1)), np.diag([0.1, 0]), np.ones((3, 3)) * 0.05) * 8
    fit = fit_gaussian_completion(
        values,
        rows,
        noise_covariances=noises,
        cliques=([0, 1], [1, 2]),
        reference_kernel=np.eye(3),
        regularization=0.2,
    )
    assert fit.converged
    assert fit.diagnostics["cached_groups"] == 3


def test_unit_changes_transform_prior_noise_kernel_and_score_correctly():
    values, rows, cliques, reference = data(count=60)
    initial = np.diag(np.diag(reference))
    options = dict(
        cliques=cliques, regularization=0.2, stationarity_tolerance=1e-8, relative_tolerance=1e-10
    )
    base = fit_gaussian_completion(
        values,
        rows,
        noise_covariances=0.1,
        reference_kernel=reference,
        initial_kernels=(initial,),
        **options,
    )
    scale = np.array([0.3, 2.0, 0.8, 1.3, 3.0])
    changed_values = tuple(value * scale[row] for value, row in zip(values, rows, strict=True))
    noises = tuple(np.diag(0.1 * scale[row] ** 2) for row in rows)
    changed = fit_gaussian_completion(
        changed_values,
        rows,
        noise_covariances=noises,
        reference_kernel=scale[:, None] * reference * scale[None, :],
        initial_kernels=(scale[:, None] * initial * scale[None, :],),
        **options,
    )
    assert base.converged and changed.converged
    np.testing.assert_allclose(
        changed.kernel, scale[:, None] * base.kernel * scale[None, :], atol=2e-7, rtol=2e-7
    )
    expected_shift = np.mean([np.log(scale[row]).sum() for row in rows])
    assert changed.diagnostics["objective"] - base.diagnostics["objective"] == pytest.approx(
        expected_shift, abs=2e-12
    )
    assert changed.diagnostics["penalty"] == pytest.approx(base.diagnostics["penalty"], abs=2e-12)


def test_node_permutation_and_reordered_observation_rows_preserve_the_model():
    values, rows, cliques, reference = data(count=50)
    base = fit_gaussian_completion(
        values,
        rows,
        noise_covariances=0.1,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.2,
    )
    permutation = np.array([4, 0, 2, 1, 3])
    inverse = np.argsort(permutation)
    changed = fit_gaussian_completion(
        tuple(value[::-1] for value in values),
        tuple(inverse[row[::-1]] for row in rows),
        noise_covariances=0.1,
        cliques=tuple(inverse[group] for group in cliques),
        reference_kernel=reference[np.ix_(permutation, permutation)],
        regularization=0.2,
    )
    assert base.converged and changed.converged
    np.testing.assert_allclose(
        changed.kernel, base.kernel[np.ix_(permutation, permutation)], atol=2e-7
    )


def test_group_cache_has_a_streaming_fallback_with_identical_likelihood(monkeypatch):
    values, rows, cliques, reference = data(count=20)
    expected = gaussian_kernel_score(reference, values, rows, noise_covariances=0.1)
    monkeypatch.setattr(gc, "_MAX_GROUPS", 1)
    assert gaussian_kernel_score(reference, values, rows, noise_covariances=0.1) == pytest.approx(
        expected, abs=2e-15
    )
    fit = fit_gaussian_completion(
        values,
        rows,
        noise_covariances=0.1,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.2,
    )
    assert fit.converged
    assert fit.diagnostics["grouping_mode"] == "streaming observations"
    assert fit.diagnostics["cached_groups"] == 0


def test_iteration_limit_preserves_best_feasible_iterate_and_failed_attempts():
    fit = fit_case(max_iterations=1, stationarity_tolerance=1e-14)
    assert not fit.converged
    assert fit.diagnostics["status"] == "maximum_iterations"
    assert np.linalg.eigvalsh(fit.kernel)[0] > 0
    assert np.isfinite(fit.diagnostics["objective"])
    assert len(fit.diagnostics["attempts"]) == 3
    y, rows, cliques, reference = data()
    invalid = np.ones((5, 5))
    valid = np.eye(5)
    result = fit_gaussian_completion(
        y,
        rows,
        noise_covariances=0.1,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.1,
        initial_kernels={"bad": invalid, "good": valid},
    )
    assert result.converged
    assert result.diagnostics["attempts"][0]["status"] == "initialization_failed"
    assert result.diagnostics["selected_attempt"] == 1


def test_objective_increase_is_rejected_without_repair(monkeypatch):
    values, rows, cliques, reference = data(count=20)
    monkeypatch.setattr(gc, "_m_step", lambda *args: (100 * np.eye(5), np.eye(5) / 100))
    fit = fit_gaussian_completion(
        values,
        rows,
        noise_covariances=0.1,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.1,
        initial_kernels=(np.eye(5),),
    )
    assert not fit.converged
    assert fit.diagnostics["status"] == "objective_increase_rejected"
    np.testing.assert_array_equal(fit.kernel, np.eye(5))
    assert len(fit.diagnostics["objective_history"]) == 1


def test_bad_graph_initialization_and_all_failed_starts_are_explicit():
    values, rows, cliques, reference = data(count=20)
    bad = np.eye(5)
    bad[0, 4] = bad[4, 0] = 0.2
    fit = fit_gaussian_completion(
        values,
        rows,
        noise_covariances=0.1,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.1,
        initial_kernels=(bad,),
    )
    assert not fit.converged and fit.kernel is None and fit.precision is None
    assert "graph" in fit.diagnostics["attempts"][0]["reason"]
    with pytest.raises(ValueError, match="running-intersection"):
        fit_gaussian_completion(
            ([1.0, 2.0],),
            ([0, 1],),
            noise_covariances=0,
            cliques=([0, 1], [1, 2], [2, 3], [3, 0]),
            reference_kernel=np.eye(4),
            regularization=0.1,
        )


def test_zero_weight_dates_and_large_weight_scaling_are_equivalent():
    values, rows, cliques, reference = data(count=20)
    weights = np.arange(20, dtype=float)
    weights[::3] = 0
    a = gaussian_kernel_score(reference, values, rows, noise_covariances=0.1, weights=weights)
    b = gaussian_kernel_score(
        reference, values, rows, noise_covariances=0.1, weights=weights * 1e305
    )
    assert a == pytest.approx(b, abs=1e-14)
    fit = fit_gaussian_completion(
        values,
        rows,
        noise_covariances=0.1,
        cliques=cliques,
        reference_kernel=reference,
        regularization=0.2,
        weights=weights,
    )
    assert fit.converged
    assert fit.diagnostics["positive_weight_observations"] == np.sum(weights > 0)


@pytest.mark.parametrize(
    "change",
    [
        {"regularization": 0},
        {"regularization": np.inf},
        {"reference_kernel": np.ones((5, 5))},
        {"max_iterations": False},
        {"stationarity_tolerance": 0},
        {"relative_tolerance": -1},
        {"weights": [0] * 20},
        {"noise_covariances": -0.1},
    ],
)
def test_invalid_fit_inputs_raise_without_jitter_or_ignored_options(change):
    values, rows, cliques, reference = data(count=20)
    options = dict(
        noise_covariances=0.1, cliques=cliques, reference_kernel=reference, regularization=0.1
    )
    with pytest.raises(ValueError):
        fit_gaussian_completion(values, rows, **{**options, **change})


def test_output_arrays_and_nested_attempts_are_immutable():
    fit = fit_case()
    with pytest.raises(FrozenInstanceError):
        fit.kernel = np.eye(5)
    with pytest.raises(ValueError):
        fit.kernel[0, 0] = 0
    with pytest.raises(TypeError):
        fit.diagnostics["attempts"][0]["status"] = "fake"
    with pytest.raises(ValueError):
        fit.diagnostics["free_precision_mask"][0, 0] = False
    conditional = condition_gaussian_kernel(fit.kernel, [1.0], [0], noise_covariance=0.1)
    with pytest.raises(ValueError):
        conditional.mean[0] = 0
