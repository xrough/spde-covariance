"""Additional explicit controls prepared without changing the running source."""

import numpy as np
from numpy.testing import assert_allclose

from spde_covariance.empirical_short_history_models import (
    _dfm_training,
    _ridge_operator,
    interval_weights,
    orthonormal_spline_basis,
    schur_residual_forecast,
)


def test_dfm_reports_degenerate_retained_boundary_eigenvalues():
    tenors = np.linspace(30.0, 365.0, 12)
    phi = orthonormal_spline_basis(tenors)
    reconstruction = _ridge_operator(phi)[0] @ phi
    eigenvalues = np.array([9.0, 4.0, 1.0, 1.0, 0.2, 0.1])
    desired_coefficients = np.vstack(
        [np.diag(np.sqrt(6 * eigenvalues)), -np.diag(np.sqrt(6 * eigenvalues))]
    )
    records = tuple(
        {"tenors_days": tenors, "response": phi @ np.linalg.solve(reconstruction, coefficient)}
        for coefficient in desired_coefficients
    )
    loading, factors, _, _, diagnostics = _dfm_training(records, interval_weights(12))
    assert_allclose(diagnostics["coefficient_second_moment_eigenvalues"], eigenvalues, atol=2e-14)
    assert diagnostics["third_fourth_relative_eigengap"] < 1e-13
    assert np.all(np.isfinite(loading)) and np.all(np.isfinite(factors))
    # No claim that the third loading is uniquely identified at a tied boundary.


def test_noisy_correlated_residual_control_with_changed_factor_forecast():
    h = np.diag([4.0, 1.0])
    a = np.array([[0.4, -0.7], [1.0, 0.2]])
    omega = np.array([[0.3, 0.1], [0.1, 0.5]])
    g = h @ a.T
    moment = np.block([[h, g], [g.T, a @ h @ a.T + omega]])
    b = np.array([[1.0, 0.1], [1.0, 0.4], [1.0, 0.8]])
    psi = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    forecast_h = np.diag([6.0, 2.0])
    diagonal = np.array([0.1, 0.2, 0.3])
    actual, components, diagnostics = schur_residual_forecast(
        b, psi, moment, forecast_h, diagonal=diagonal
    )
    augmented = b + psi @ a
    expected = augmented @ forecast_h @ augmented.T + psi @ omega @ psi.T + np.diag(diagonal)
    assert_allclose(actual, expected, atol=1e-13)
    assert_allclose(components["correlated_residual"], psi @ omega @ psi.T, atol=1e-14)
    assert_allclose(sum(components.values()), expected, atol=1e-13)
    assert diagnostics["relative_support_residual"] < 1e-14
