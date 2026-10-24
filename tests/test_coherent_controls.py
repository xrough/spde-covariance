"""Independent analytic controls for coherent partials and source-cone scope."""

import numpy as np
import pytest

from spde_covariance.admissibility import (
    covariance_integration_matrix,
    diagnose_constant_source_admissibility,
)
from spde_covariance.coherent import fit_coherent_partial
from spde_covariance.completion import complete_kernel, observed_mask
from spde_covariance.fragment_reference import truncation_psd_control
from spde_covariance.geometry import smat, svec
from spde_covariance.laguerre import LaguerreTransport


def _shared_control():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    mask = observed_mask(cliques, 3)
    target = np.array([[1, 2, np.nan], [2, 1, 0], [np.nan, 0, 4]])
    optimum = np.array([[1.5, 1.5, np.nan], [1.5, 1.5, 0], [np.nan, 0, 4]])
    return target, optimum, cliques, mask


def _weighted_difference(left, right, weights, mask):
    root = np.sqrt(weights)
    return np.linalg.norm((root[:, None] * (left - right) * root[None, :])[mask])


def _minimum_clique_eigenvalue(kernel, cliques, weights):
    root = np.sqrt(weights)
    matrix = root[:, None] * kernel * root[None, :]
    return min(np.linalg.eigvalsh(matrix[np.ix_(group, group)])[0] for group in cliques)


def test_consensus_is_needed_independent_psd_projections_then_average_fail():
    target, optimum, cliques, mask = _shared_control()
    projected = []
    for group in cliques:
        values, vectors = np.linalg.eigh(target[np.ix_(group, group)])
        projected.append((vectors * np.maximum(values, 0)) @ vectors.T)
    assert projected[0][1, 1] == pytest.approx(1.5)
    assert projected[1][0, 0] == pytest.approx(1)
    independently_averaged = projected[0].copy()
    independently_averaged[1, 1] = (projected[0][1, 1] + projected[1][0, 0]) / 2
    assert np.linalg.eigvalsh(independently_averaged)[0] < -0.1
    fit = fit_coherent_partial(
        target,
        pair_weights=mask.astype(float),
        cliques=cliques,
        numerical_error_budget=1e-5,
    )
    assert fit.converged
    assert np.all(np.isnan(fit.partial_kernel[~mask]))
    assert not fit.partial_kernel.flags.writeable
    assert (
        _weighted_difference(fit.partial_kernel, optimum, np.ones(3), mask)
        <= fit.partial_error_bound + 1e-10
    )
    assert fit.diagnostics["objective"] == pytest.approx(0.5, abs=2e-8)
    assert fit.diagnostics["objective"] - 0.5 <= fit.objective_gap_bound + 1e-12
    assert fit.partial_error_bound <= 1e-5
    assert _minimum_clique_eigenvalue(fit.partial_kernel, cliques, np.ones(3)) >= -1e-12
    completed = complete_kernel(fit.partial_kernel, cliques=cliques, population=True)
    assert completed.available
    assert completed.kernel[0, 2] == pytest.approx(0, abs=1e-10)
    assert np.linalg.eigvalsh(completed.kernel)[0] >= -1e-10


def test_weighted_ridge_control_matches_closed_form_and_actual_optimizer_bound():
    kernel = np.array([[2, 0.4, 0.1], [0.4, 1, 0.3], [0.1, 0.3, 1.5]])
    _, _, cliques, mask = _shared_control()
    weights = np.array([0.2, 0.3, 0.5])
    ridge = 0.3
    target = np.where(mask, kernel, np.nan)
    fit = fit_coherent_partial(
        target,
        pair_weights=mask.astype(float),
        cliques=cliques,
        weights=weights,
        regularization=ridge,
        numerical_error_budget=1e-5,
    )
    assert fit.converged
    optimum = target / (1 + ridge)
    actual_error = _weighted_difference(fit.partial_kernel, optimum, weights, mask)
    assert actual_error <= fit.partial_error_bound + 1e-12
    root = np.sqrt(weights)
    weighted_target = root[:, None] * kernel * root[None, :]
    exact_objective = ridge / (2 * (1 + ridge)) * np.sum(weighted_target[mask] ** 2)
    assert fit.diagnostics["objective"] - exact_objective <= fit.objective_gap_bound + 1e-12
    assert fit.diagnostics["strong_convexity"] == pytest.approx(1 + ridge)


def test_positive_floor_is_in_weighted_coordinates_not_unchanged_kernel_units():
    _, _, cliques, mask = _shared_control()
    target = np.where(mask, 0.0, np.nan)
    weights = np.array([0.2, 0.3, 0.5])
    floor = 0.01
    fit = fit_coherent_partial(
        target,
        pair_weights=mask.astype(float),
        cliques=cliques,
        weights=weights,
        eigenvalue_floor=floor,
        numerical_error_budget=1e-5,
    )
    assert fit.converged
    np.testing.assert_allclose(np.diag(fit.partial_kernel), floor / weights, atol=1e-8)
    assert _minimum_clique_eigenvalue(fit.partial_kernel, cliques, weights) >= floor - 1e-12
    assert fit.diagnostics["objective"] == pytest.approx(3 * floor**2 / 2, abs=1e-10)


def test_iteration_limit_returns_feasible_but_uncertified_result_explicitly():
    target, optimum, cliques, mask = _shared_control()
    fit = fit_coherent_partial(
        target,
        pair_weights=mask.astype(float),
        cliques=cliques,
        numerical_error_budget=1e-8,
        max_iterations=1,
    )
    assert not fit.converged
    assert fit.partial_error_bound > 1e-8
    assert fit.diagnostics["feasibility_diagonal_shift"] > 0
    assert _minimum_clique_eigenvalue(fit.partial_kernel, cliques, np.ones(3)) >= -1e-12
    actual_error = _weighted_difference(fit.partial_kernel, optimum, np.ones(3), mask)
    assert actual_error <= fit.partial_error_bound + 1e-12


def test_coherent_input_still_has_indefinite_hard_truncated_completion():
    control = truncation_psd_control()
    partial = np.where(control["mask"], control["kernel"], np.nan)
    fit = fit_coherent_partial(
        partial,
        pair_weights=control["mask"].astype(float),
        cliques=control["cliques"],
        numerical_error_budget=1e-5,
    )
    assert fit.converged
    exact = complete_kernel(fit.partial_kernel, cliques=control["cliques"], population=True)
    truncated = complete_kernel(fit.partial_kernel, cliques=control["cliques"], rank=1)
    assert exact.available and truncated.available
    assert np.linalg.eigvalsh(exact.kernel)[0] >= -1e-10
    assert np.linalg.eigvalsh(truncated.kernel)[0] < -0.4
    assert not truncated.diagnostics["output_psd"]


def _inverse_two_mode_source(innovation, delta):
    a = -np.expm1(-delta) / delta
    b = -np.expm1(-2 * delta) / (2 * delta)
    c22 = innovation[1, 1] / b
    c12 = (innovation[0, 1] - (a - b) * c22) / a
    c11 = innovation[0, 0] - 2 * (1 - a) * c12 - (1 - 2 * a + b) * c22
    return np.array([[c11, c12], [c12, c22]])


@pytest.mark.parametrize("interior", [0.0, 0.01])
def test_actual_two_mode_transport_has_psd_innovations_outside_constant_source_cone(interior):
    model = LaguerreTransport(2)
    delta = 1.0
    innovation = np.diag([interior, 1 + interior])
    source = _inverse_two_mode_source(innovation, delta)
    assert np.linalg.eigvalsh(source)[0] < -0.06
    integration = covariance_integration_matrix(model, delta)
    np.testing.assert_allclose(smat(integration @ svec(source)), innovation, atol=1e-13)
    result = diagnose_constant_source_admissibility(
        innovation,
        np.eye(2),
        delta=delta,
        trace_cap=10,
        integration_matrix=integration,
        objective_gap_tolerance=1e-11,
        admissibility_tolerance=1e-3,
    )
    assert result.converged
    assert result.admissibility_status == "separated_from_declared_finite_class"
    assert result.fit_distance_lower_bound > 1e-3
    assert np.linalg.eigvalsh(result.source_candidate)[0] >= -1e-10
    assert np.trace(result.source_candidate) <= 10 + 1e-10


def test_time_varying_psd_sources_remove_the_counterexample_obstruction():
    # This is an explicit algebraic witness, not a claim that the diagnostic
    # fits arbitrary time variation or reconstructs a physical source.
    model = LaguerreTransport(2)
    innovation = np.diag([0.01, 1.01])
    nodes, weights = np.polynomial.legendre.leggauss(16)
    result = np.zeros((2, 2))
    for r, weight in zip((nodes + 1) / 2, weights / 2, strict=True):
        transport = model.matrix(r)
        inverse = np.linalg.inv(transport)
        source_at_lag = inverse @ innovation @ inverse.T
        assert np.linalg.eigvalsh(source_at_lag)[0] > 0
        result += weight * transport @ source_at_lag @ transport.T
    np.testing.assert_allclose(result, innovation, atol=1e-14)
