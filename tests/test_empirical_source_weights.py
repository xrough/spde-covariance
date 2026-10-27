"""Weighted source moments retain physical-time dynamics and certificates."""

import numpy as np
import pytest

from spde_covariance.empirical_calibration import (
    _dynamics,
    extract_coefficient_innovations,
    fit_source_covariance,
    integrated_drift_matrix,
)
from spde_covariance.laguerre import LaguerreTransport
from spde_covariance.linalg import project_psd_trace


def _trajectory(count=15):
    rng = np.random.default_rng(843)
    model = LaguerreTransport(2)
    deltas = np.resize(np.array([1, 4, 2, 3]) / 365.25, count)
    times = np.r_[0, np.cumsum(deltas)]
    values = [np.array([10.0, -0.3])]
    for delta in deltas:
        values.append(
            model.matrix(delta) @ values[-1]
            + integrated_drift_matrix(delta, 2) @ np.array([2.0, -0.4])
            + rng.normal(size=2) * np.sqrt(delta) * np.array([0.6, 0.4])
        )
    return np.array(values), times


def test_interval_dynamics_cache_is_bounded():
    assert _dynamics.cache_info().maxsize == 64


@pytest.mark.parametrize("drift", ["zero", "constant"])
def test_supplied_uniform_weights_preserve_legacy_fit_exactly(drift):
    values, times = _trajectory()
    legacy = fit_source_covariance(values, times, window=12, drift=drift)
    uniform = fit_source_covariance(
        values, times, window=12, drift=drift, interval_weights=np.full(12, 7.0)
    )
    for field in ("source_covariance", "drift", "bridge_covariance", "normalized_residuals"):
        np.testing.assert_array_equal(getattr(legacy, field), getattr(uniform, field))
    assert legacy.diagnostics.keys() == uniform.diagnostics.keys()
    for key in legacy.diagnostics:
        np.testing.assert_equal(legacy.diagnostics[key], uniform.diagnostics[key])
    assert uniform.diagnostics["effective_n"] == pytest.approx(12)
    assert uniform.diagnostics["interval_weight_policy"] == "uniform"


@pytest.mark.parametrize(
    "weights",
    [
        [],
        np.ones(14),
        np.ones((15, 1)),
        np.zeros(15),
        np.r_[1.0, np.zeros(14)],
        np.r_[-1.0, np.ones(14)],
        np.r_[np.nan, np.ones(14)],
        np.r_[np.inf, np.ones(14)],
        np.r_[1e308, 1e-320, np.zeros(13)],
    ],
)
def test_interval_weight_validation(weights):
    values, times = _trajectory()
    with pytest.raises(ValueError, match="interval_weights"):
        fit_source_covariance(values, times, interval_weights=weights)


def test_weights_align_with_selected_window_and_are_immutable():
    values, times = _trajectory()
    weights = np.arange(1.0, 7.0)
    fit = fit_source_covariance(values, times, window=6, interval_weights=weights)
    isolated = fit_source_covariance(values[-7:], times[-7:], interval_weights=weights * 8)
    np.testing.assert_array_equal(fit.covariance, isolated.covariance)
    np.testing.assert_array_equal(fit.drift, isolated.drift)
    np.testing.assert_allclose(fit.diagnostics["interval_weights"], weights / weights.sum())
    np.testing.assert_array_equal(fit.diagnostics["interval_lengths"], np.diff(times[-7:]))
    assert fit.diagnostics["effective_n"] == pytest.approx(weights.sum() ** 2 / (weights @ weights))
    weights[:] = 0
    with pytest.raises(ValueError):
        fit.diagnostics["interval_weights"].flags.writeable = True


def test_weighted_identity_dynamics_objective_scale_and_gap_are_hand_checkable():
    values, times = _trajectory()
    weights = np.array([0, 0, 1, 2, 1, 4, 3, 0, 1, 5, 4, 2, 0, 3, 7], dtype=float)
    weights /= weights.sum()
    cap, ridge = 0.15, 0.03
    fit = fit_source_covariance(
        values,
        times,
        interval_weights=weights,
        drift="zero",
        transport_adjustment=False,
        regularization_relative=ridge,
        trace_cap=cap,
        numerical_error_budget=1e-7,
    )
    residuals = np.diff(values, axis=0) / np.sqrt(np.diff(times))[:, None]
    moments = residuals[:, :, None] * residuals[:, None, :]
    second_moment = np.einsum("i,ijk->jk", weights, moments)
    optimum = project_psd_trace(second_moment / (1 + ridge), cap)
    np.testing.assert_allclose(fit.covariance, optimum, atol=2e-8)
    objective = (
        sum(
            weight * np.linalg.norm(fit.covariance - moment, "fro") ** 2 / 2
            for weight, moment in zip(weights, moments, strict=True)
        )
        + ridge * np.linalg.norm(fit.covariance, "fro") ** 2 / 2
    )
    assert fit.diagnostics["objective"] == pytest.approx(objective, rel=1e-12)
    assert fit.diagnostics["empirical_residual_trace"] == pytest.approx(np.trace(second_moment))
    assert fit.diagnostics["covariance_scale"] == pytest.approx(np.trace(second_moment))
    assert fit.diagnostics["regularization"] == pytest.approx(ridge)
    np.testing.assert_allclose(
        fit.diagnostics["normalized_residual_mean"], weights @ residuals, atol=1e-15
    )
    gradient = (1 + ridge) * fit.covariance - second_moment
    gap = np.sum(gradient * fit.covariance) - cap * min(np.linalg.eigvalsh(gradient)[0], 0)
    assert fit.diagnostics["objective_gap_bound"] == pytest.approx(max(0, gap), abs=1e-13)
    assert np.linalg.norm(fit.covariance - optimum) <= (
        fit.diagnostics["covariance_error_bound"] + 1e-12
    )
    assert fit.converged
    assert fit.diagnostics["positive_weight_interval_count"] == 11
    assert fit.diagnostics["relative_covariance_error_bound"] <= 1e-7
    default_cap = fit_source_covariance(
        values,
        times,
        interval_weights=weights,
        drift="zero",
        transport_adjustment=False,
    )
    assert default_cap.diagnostics["trace_cap"] == pytest.approx(100 * np.trace(second_moment))


def test_weighted_transport_with_drift_matches_independent_convex_references():
    cp = pytest.importorskip("cvxpy")
    values, times = _trajectory(count=12)
    weights = np.geomspace(0.05, 1.0, 12)
    weights /= weights.sum()
    cap, drift_relative = 0.08, 0.02
    fit = fit_source_covariance(
        values,
        times,
        interval_weights=weights,
        trace_cap=cap,
        drift="constant",
        drift_regularization_relative=drift_relative,
        numerical_error_budget=1e-8,
    )
    model = LaguerreTransport(2)
    deltas = np.diff(times)
    nodes, quadrature = np.polynomial.legendre.leggauss(48)
    drift_maps = [
        sum(
            weight * np.sqrt(delta) / 2 * model.matrix(delta * (node + 1) / 2)
            for node, weight in zip(nodes, quadrature, strict=True)
        )
        for delta in deltas
    ]
    raw = extract_coefficient_innovations(values, times)
    drift_gram = sum(
        weight * (matrix.T @ matrix) for weight, matrix in zip(weights, drift_maps, strict=True)
    )
    drift_penalty = drift_relative * np.linalg.eigvalsh(drift_gram)[-1]
    mean = cp.Variable(2)
    drift_objective = (
        sum(
            weight * cp.sum_squares(matrix @ mean - residual) / 2
            for weight, matrix, residual in zip(weights, drift_maps, raw, strict=True)
        )
        + drift_penalty * cp.sum_squares(mean) / 2
    )
    drift_problem = cp.Problem(cp.Minimize(drift_objective))
    drift_problem.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_feas=1e-11, tol_gap_rel=1e-11)
    np.testing.assert_allclose(fit.drift, mean.value, rtol=1e-9, atol=1e-9)
    assert fit.diagnostics["drift_regularization"] == pytest.approx(drift_penalty, rel=1e-12)
    # Build an independent orthonormal symmetric coordinate map by quadrature.
    basis = np.array(
        [np.diag([1.0, 0]), np.diag([0.0, 1]), np.array([[0, 1], [1, 0]]) / np.sqrt(2)]
    )
    gram = np.zeros((3, 3))
    trace_gain = np.zeros((2, 2))
    for probability, delta in zip(weights, deltas, strict=True):
        propagated = sum(
            weight / 2 * np.einsum("ij,bjk,lk->bil", matrix, basis, matrix)
            for node, weight in zip(nodes, quadrature, strict=True)
            for matrix in [model.matrix(delta * (node + 1) / 2)]
        )
        gram += probability * np.einsum("bij,cij->bc", propagated, propagated)
        trace_gain += probability * sum(
            weight / 2 * (matrix.T @ matrix)
            for node, weight in zip(nodes, quadrature, strict=True)
            for matrix in [model.matrix(delta * (node + 1) / 2)]
        )
    regularization = 1e-3 * np.linalg.eigvalsh(gram)[-1]
    assert fit.diagnostics["regularization"] == pytest.approx(regularization, rel=1e-12)
    assert fit.diagnostics["minimum_mean_transport_trace_gain"] == pytest.approx(
        np.linalg.eigvalsh(trace_gain)[0], rel=1e-12
    )
    empirical_trace = np.sum(weights[:, None] * fit.normalized_residuals**2)
    assert fit.diagnostics["covariance_scale"] == pytest.approx(
        empirical_trace / np.linalg.eigvalsh(trace_gain)[0], rel=1e-12
    )
    variable = cp.Variable((2, 2), symmetric=True)
    objective = 0
    for probability, delta, residual in zip(weights, deltas, fit.normalized_residuals, strict=True):
        prediction = sum(
            weight
            / 2
            * model.matrix(delta * (node + 1) / 2)
            @ variable
            @ model.matrix(delta * (node + 1) / 2).T
            for node, weight in zip(nodes, quadrature, strict=True)
        )
        objective += probability * cp.sum_squares(prediction - np.outer(residual, residual)) / 2
    objective += regularization * cp.sum_squares(variable) / 2
    reference = cp.Problem(cp.Minimize(objective), [variable >> 0, cp.trace(variable) <= cap])
    reference.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_feas=1e-11, tol_gap_rel=1e-11)
    assert reference.status == "optimal"
    assert fit.converged
    np.testing.assert_allclose(fit.covariance, variable.value, atol=2e-6)
    assert fit.diagnostics["objective"] == pytest.approx(reference.value, abs=2e-9)
    assert fit.diagnostics["source_trace"] == pytest.approx(cap, abs=2e-8)
    assert np.linalg.norm(fit.covariance - variable.value) <= (
        fit.diagnostics["covariance_error_bound"] + 3e-6
    )
