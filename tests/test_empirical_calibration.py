"""Finite source calibration, actual calendar intervals and anchored forecasts."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.linalg import expm

from spde_covariance.empirical_calibration import (
    extract_coefficient_innovations,
    fit_cross_section,
    fit_source_covariance,
    forecast_contract_changes,
    integrated_drift_matrix,
    source_risk_error_bound,
)
from spde_covariance.laguerre import LaguerreTransport


def trajectory(dimension=2, count=30, seed=83):
    model = LaguerreTransport(dimension)
    rng = np.random.default_rng(seed)
    deltas = np.resize(np.array([1, 3, 2, 5, 1, 4, 7]) / 365.25, count)
    times = np.r_[0, np.cumsum(deltas)]
    source = np.diag(np.linspace(0.2, 0.4, dimension))
    values = [np.r_[20.0, np.ones(dimension - 1)]]
    for delta in deltas:
        covariance = model.cumulative_covariance(source, delta, method="block")
        values.append(
            model.matrix(delta) @ values[-1]
            + rng.multivariate_normal(np.zeros(dimension), covariance)
        )
    return np.asarray(values), times


def test_cross_section_recovers_unpenalized_state_and_preserves_level_translation():
    model = LaguerreTransport(3)
    tenors = np.arange(1, 13) / 12
    truth = np.array([70000, 1000, -500])
    prices = model.evaluate(tenors) @ truth
    exact = fit_cross_section(tenors, prices, smoothing=0)
    assert exact.available
    np.testing.assert_allclose(exact.coefficients, truth, atol=2e-9, rtol=0)
    fitted = fit_cross_section(tenors, prices, smoothing=1e-4)
    translated = fit_cross_section(tenors, prices + 25000, smoothing=1e-4)
    assert fitted.available and translated.available
    np.testing.assert_allclose(
        translated.coefficients - fitted.coefficients, [25000, 0, 0], atol=2e-9, rtol=0
    )
    np.testing.assert_allclose(fitted.residuals, translated.residuals, atol=2e-9, rtol=0)
    assert not fitted.diagnostics["constant_penalized"]
    assert fitted.diagnostics["design_rank"] == 3
    assert fitted.diagnostics["effective_degrees_of_freedom"] < 3
    assert fitted.diagnostics["regularization"] == pytest.approx(
        1e-4 * fitted.diagnostics["evaluation_gram_norm"]
    )


def test_rank_deficient_cross_section_is_not_claimed_identified_after_ridge():
    fit = fit_cross_section([0.2, 0.2, 0.2], [70000, 70010, 69990], dimension=3)
    assert not fit.available
    assert fit.diagnostics["design_rank"] == 1
    assert np.isfinite(fit.coefficients).all()
    assert fit.diagnostics["design_condition_number"] == float("inf")


def test_irregular_exact_transport_and_integrated_drift_remove_deterministic_evolution():
    model = LaguerreTransport(3)
    times = np.array([0, 1, 4, 5, 12, 14]) / 365.25
    drift = np.array([0.4, -0.2, 0.3])
    values = [np.array([100, 2, -1.0])]
    for delta in np.diff(times):
        drift_map = integrated_drift_matrix(delta, 3)
        # Independently integrate exp(r A) by high-order positive quadrature.
        nodes, weights = np.polynomial.legendre.leggauss(48)
        reference = delta * sum(
            weight / 2 * expm(model.generator * delta * (node + 1) / 2)
            for node, weight in zip(nodes, weights, strict=True)
        )
        np.testing.assert_allclose(drift_map, reference, atol=1e-15)
        assert drift_map[0, 0] == pytest.approx(delta)
        values.append(model.matrix(delta) @ values[-1] + drift_map @ drift)
    residuals = extract_coefficient_innovations(values, times, drift=drift)
    np.testing.assert_allclose(residuals, 0, atol=2e-12)
    fit = fit_source_covariance(values, times, drift="constant", drift_regularization_relative=0)
    assert fit.converged
    np.testing.assert_allclose(fit.drift, drift, atol=2e-10)
    assert np.linalg.norm(fit.source_covariance) < 1e-18
    np.testing.assert_allclose(fit.diagnostics["interval_lengths"], np.diff(times))
    assert fit.diagnostics["minimum_interval"] < fit.diagnostics["maximum_interval"]
    delta, tenors = 3 / 365.25, np.array([0.2, 0.6])
    forecast = forecast_contract_changes([70000, 71000], tenors, fit, delta)
    expected_mean = model.evaluate(tenors) @ integrated_drift_matrix(delta, 3) @ drift
    np.testing.assert_allclose(forecast["mean_changes"], expected_mean, atol=2e-12)


def test_source_fit_matches_independent_cvxpy_positive_factor_problem():
    cp = pytest.importorskip("cvxpy")
    values, times = trajectory()
    cap = 0.25
    fit = fit_source_covariance(
        values, times, drift="zero", trace_cap=cap, numerical_error_budget=1e-8
    )
    assert fit.converged
    model = LaguerreTransport(2)
    variable = cp.Variable((2, 2), symmetric=True)
    nodes, weights = np.polynomial.legendre.leggauss(48)
    residuals = extract_coefficient_innovations(values, times)
    objective = 0
    for delta, residual in zip(np.diff(times), residuals, strict=True):
        prediction = sum(
            weight
            / 2
            * model.matrix(delta * (node + 1) / 2)
            @ variable
            @ model.matrix(delta * (node + 1) / 2).T
            for node, weight in zip(nodes, weights, strict=True)
        )
        objective += cp.sum_squares(prediction - np.outer(residual, residual)) / (
            2 * len(residuals)
        )
    objective += fit.diagnostics["regularization"] * cp.sum_squares(variable) / 2
    reference = cp.Problem(cp.Minimize(objective), [variable >> 0, cp.trace(variable) <= cap])
    reference.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    np.testing.assert_allclose(fit.source_covariance, variable.value, atol=2e-6)
    assert fit.diagnostics["objective"] == pytest.approx(reference.value, abs=2e-9)
    assert fit.diagnostics["source_trace"] == pytest.approx(cap, abs=2e-8)
    assert fit.diagnostics["relative_covariance_error_bound"] <= 1e-8
    assert fit.diagnostics["linear_map_rank"] == 3
    assert not fit.diagnostics["source_recovery_claim"]
    assert not fit.diagnostics["latent_fresh_noise_claim"]
    # Portfolio error propagation is a dual-Frobenius bound, in cash units.
    tenors, positions, delta = np.array([0.2, 0.6]), np.array([1, -1]), 4 / 365.25
    exposure = 5 * model.evaluate(tenors).T @ positions
    # Direct linear evaluation permits an indefinite optimizer difference.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    difference = sum(
        delta
        * weight
        / 2
        * model.matrix(delta * (node + 1) / 2)
        @ (fit.covariance - variable.value)
        @ model.matrix(delta * (node + 1) / 2).T
        for node, weight in zip(nodes, weights, strict=True)
    )
    actual_difference = abs(exposure @ difference @ exposure)
    bound = source_risk_error_bound(fit, tenors, positions, delta, multiplier=5)
    assert actual_difference <= bound + 2e-8
    assert bound == pytest.approx(25 * source_risk_error_bound(fit, tenors, positions, delta))


def test_no_transport_ablation_has_identity_covariance_map_and_actual_time_scaling():
    values, times = trajectory()
    fit = fit_source_covariance(values, times, drift="zero", transport_adjustment=False)
    assert fit.converged and not fit.diagnostics["transport_adjustment"]
    raw = np.diff(values, axis=0) / np.sqrt(np.diff(times)[:, None])
    np.testing.assert_allclose(fit.normalized_residuals, raw)
    expected = raw.T @ raw / len(raw) / (1 + fit.diagnostics["regularization"])
    np.testing.assert_allclose(fit.covariance, expected, atol=2e-10)
    np.testing.assert_allclose(fit.bridge_covariance, fit.covariance, atol=1e-12)
    delta, tenors = 3 / 365.25, np.array([0.2, 0.5])
    forecast = forecast_contract_changes([70000, 71000], tenors, fit, delta)
    evaluation = LaguerreTransport(2).evaluate(tenors)
    np.testing.assert_allclose(
        forecast["covariance"], delta * evaluation @ fit.covariance @ evaluation.T
    )


def test_zero_residual_source_accepts_zero_cap_and_does_not_invent_volatility():
    values = np.tile([70000, 0, 0.0], (5, 1))
    times = np.array([0, 1, 4, 5, 8]) / 365.25
    fit = fit_source_covariance(values, times, drift="zero")
    assert fit.converged and fit.diagnostics["trace_cap"] == 0
    assert fit.diagnostics["covariance_error_bound"] == 0
    np.testing.assert_array_equal(fit.source_covariance, np.zeros((3, 3)))
    np.testing.assert_array_equal(fit.bridge_covariance, np.zeros((3, 3)))
    explicit = fit_source_covariance(values, times, drift="zero", trace_cap=0)
    assert explicit.converged
    np.testing.assert_array_equal(explicit.source_covariance, fit.source_covariance)


def test_trailing_training_window_and_price_unit_equivariance():
    values, times = trajectory(count=40)
    fit = fit_source_covariance(values, times, window=12, drift="constant")
    changed = values.copy()
    changed[:-13] = 1e8
    isolated = fit_source_covariance(changed, times, window=12, drift="constant")
    np.testing.assert_array_equal(fit.covariance, isolated.covariance)
    np.testing.assert_array_equal(fit.drift, isolated.drift)
    assert fit.diagnostics["training_start_time"] == times[-13]
    assert fit.diagnostics["training_end_time"] == times[-1]
    scaled = fit_source_covariance(100 * values, times, window=12, drift="constant")
    np.testing.assert_allclose(scaled.covariance, 10000 * fit.covariance, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(scaled.drift, 100 * fit.drift, rtol=2e-9)
    assert scaled.diagnostics["trace_cap"] == pytest.approx(10000 * fit.diagnostics["trace_cap"])
    # Cancellation in a near-zero optimizer gap need not scale bit for bit.
    # Both fits must meet the same dimensionless accuracy contract.
    for candidate in (fit, scaled):
        assert candidate.diagnostics["relative_covariance_error_bound"] <= 1e-5
        assert candidate.diagnostics["covariance_error_bound"] <= (
            candidate.diagnostics["covariance_scale"] * 1e-5
        )


def test_end_tenors_forecast_and_anchor_do_not_create_reconstruction_mean_reversion():
    values, times = trajectory(dimension=3)
    fit = fit_source_covariance(values, times, drift="zero", bridge_delta=2 / 365.25)
    delta, tenors = 3 / 365.25, np.array([0.2, 0.6])
    origin_prices = np.array([70000, 73500])  # Deliberately unrelated to reconstructed level.
    forecast = forecast_contract_changes(origin_prices, tenors, fit, delta)
    np.testing.assert_array_equal(forecast["mean_changes"], [0, 0])
    np.testing.assert_array_equal(forecast["mean_prices"], origin_prices)
    model = LaguerreTransport(3)
    end_evaluation = model.evaluate(tenors)
    np.testing.assert_allclose(
        end_evaluation @ model.matrix(delta), model.evaluate(tenors + delta), atol=1e-15
    )
    expected = (
        end_evaluation
        @ model.cumulative_covariance(fit.covariance, delta, method="block")
        @ end_evaluation.T
    )
    np.testing.assert_allclose(forecast["covariance"], expected, atol=1e-13)
    assert fit.diagnostics["bridge_delta"] != delta
    assert not forecast["reconstruction_error_included"]


def test_unfinished_optimizer_is_feasible_but_not_available_for_forecasting():
    values, times = trajectory()
    fit = fit_source_covariance(values, times, drift="zero", max_iterations=0)
    assert not fit.converged and fit.diagnostics["status"] == "maximum_iterations"
    assert np.linalg.eigvalsh(fit.covariance)[0] >= 0
    assert fit.diagnostics["covariance_error_bound"] > 0
    with pytest.raises(ValueError, match="converged"):
        forecast_contract_changes([1, 2], [0.2, 0.4], fit, 1 / 365.25)


def test_returned_arrays_and_diagnostics_are_immutable_snapshots():
    values, times = trajectory()
    fit = fit_source_covariance(values, times, drift="zero")
    values[:] = 0
    assert np.trace(fit.covariance) > 0
    for array in (fit.covariance, fit.drift, fit.bridge_covariance, fit.normalized_residuals):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        fit.drift = np.zeros(2)
    with pytest.raises(TypeError):
        fit.diagnostics["converged"] = False


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"regularization_relative": 0}, "regularization_relative"),
        ({"trace_cap": -1}, "trace_cap"),
        ({"window": 1}, "window"),
        ({"window": 100}, "window"),
        ({"drift": "future_mean"}, "drift"),
        ({"transport_adjustment": "yes"}, "boolean"),
        ({"numerical_error_budget": 0}, "numerical_error_budget"),
        ({"max_iterations": True}, "max_iterations"),
    ],
)
def test_source_validation(changes, match):
    values, times = trajectory(count=6)
    with pytest.raises(ValueError, match=match):
        fit_source_covariance(values, times, **changes)


def test_reconstruction_and_trajectory_validation():
    with pytest.raises(ValueError, match="finite"):
        fit_cross_section([0.1, 0.2], [1, np.nan])
    with pytest.raises(ValueError, match="nonnegative"):
        fit_cross_section([-0.1, 0.2], [1, 2])
    with pytest.raises(ValueError, match="dimension"):
        fit_cross_section([0.1, 0.2], [1, 2], dimension=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_source_covariance(np.ones((3, 2)), [0, 1, 1])
    with pytest.raises(ValueError, match="state row"):
        fit_source_covariance(np.ones((3, 2)), [0, 1, 2, 3])
