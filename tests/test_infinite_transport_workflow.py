"""Independent analytical controls and interoperability of the Laguerre backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spde_covariance import (
    CurveTrajectory,
    LaguerreTransport,
    PowerLawSource,
    TransportPropagator,
    extract_innovations,
    fit_innovation_covariance,
    simulate_transport,
)
from spde_covariance.observations import delivery_average_map, point_observation_map


@pytest.mark.parametrize("delta", [0.02, 0.4])
def test_two_mode_closed_form_covariance(delta):
    """Direct elementary integration, independent of either numerical integrator."""
    model = LaguerreTransport(2)
    q0, q1 = PowerLawSource().eigenvalues(2)
    u = -np.expm1(-delta)
    v = -np.expm1(-2 * delta) / 2
    transition = np.array([[1, u], [0, np.exp(-delta)]])
    covariance = np.array(
        [
            [q0 * delta + q1 * (delta - 2 * u + v), q1 * (u - v)],
            [q1 * (u - v), q1 * v],
        ]
    )
    assert_allclose(model.matrix(delta), transition, atol=2e-15)
    for method in ("quadrature", "block"):
        actual = model.cumulative_covariance(np.diag([q0, q1]), delta, method=method)
        assert_allclose(actual, covariance, atol=3e-14, rtol=3e-13)


def test_laguerre_backend_works_with_existing_trajectory_workflow():
    model = LaguerreTransport(4)
    assert isinstance(model, TransportPropagator)
    source = PowerLawSource().finite_matrix(4)
    times = np.arange(65) / 52
    simulation = simulate_transport(model, times, source, seed=20261024)
    batch = extract_innovations(simulation.trajectory, model)
    assert batch.basis_id == model.basis_id
    assert batch.innovation_unit == "arbitrary curve unit/sqrt(year)"
    covariance = model.innovation_covariance(source, 1 / 52)
    assert_allclose(simulation.innovation_covariances, np.broadcast_to(covariance, (64, 4, 4)))
    fit = fit_innovation_covariance(
        batch,
        covariance_assumption="common",
        regularization=1e-3,
        trace_cap=2,
        numerical_error_budget=1e-4,
        expected_basis_id=model.basis_id,
    )
    assert fit.estimate.converged
    assert fit.estimate.covariance_error_bound <= 1e-4
    assert np.linalg.eigvalsh(fit.covariance).min() >= -1e-12
    assert np.trace(fit.covariance) <= 2 + 1e-12


def test_observation_rows_preserve_fixed_infinite_exposures():
    # These coefficients are restrictions of the same functions, without a
    # dimension-dependent normalization of their projected representers.
    low, high = LaguerreTransport(8), LaguerreTransport(32)
    x = np.array([0, 0.5, 1, 1.5, 2])
    assert_allclose(point_observation_map(low, x), point_observation_map(high, x)[:, :8])
    intervals = np.array([[0.5, 1], [1.5, 2]])
    assert_allclose(
        delivery_average_map(low, intervals),
        delivery_average_map(high, intervals)[:, :8],
        atol=2e-14,
    )
    scale = np.sqrt(np.exp(-0.5) - np.exp(-1.5))
    spread = (high.evaluate([0.5])[0] - high.evaluate([1.5])[0]) / scale
    assert spread[0] == 0
    assert np.linalg.norm(spread[:8]) < 1
    assert np.linalg.norm(spread) < 1


def test_projected_level_adjustment_has_hidden_mode_term():
    full, projected = LaguerreTransport(2), LaguerreTransport(1)
    delta = 0.2
    shocks = np.array([[0.2, 0.3], [-0.1, 0.4]])
    levels = np.zeros((3, 2))
    for i in range(2):
        levels[i + 1] = full.matrix(delta) @ levels[i] + np.sqrt(delta) * shocks[i]
    times = np.arange(3) * delta
    trajectory = CurveTrajectory(times, levels[:, :1], projected.basis_id, projected.kind)
    operational = extract_innovations(trajectory, projected).values[:, 0]
    hidden = -np.expm1(-delta) * levels[:-1, 1] / np.sqrt(delta)
    assert_allclose(operational - shocks[:, 0], hidden, atol=1e-16)
    assert hidden[1] > 0
    # The full finite-forcing recursion has no hidden coordinates.
    full_trajectory = CurveTrajectory(times, levels, full.basis_id, full.kind)
    assert_allclose(extract_innovations(full_trajectory, full).values, shocks, atol=1e-16)
