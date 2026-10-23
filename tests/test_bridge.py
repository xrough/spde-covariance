import numpy as np
import pytest

from spde_covariance.bridge import fixed_delivery_increment, integrate_innovation_covariance
from spde_covariance.spaces import SplineSpace


def test_exponential_semigroup_exact_covariance_and_delta_refinement():
    # e^{-kx} is an eigenfunction of the right-shift semigroup. Its H norm
    # for tau=1 is sqrt(1+k^2/(2k-1)); source uses the normalized eigenfunction.
    k, scale = 1.4, 0.8

    def source(time):
        return np.array([[scale * np.exp(0.1 * time)]])

    def shifted(time, lag):
        return np.exp(-k * lag) * source(time)

    discrepancies = []
    for delta in [0.2, 0.1, 0.05]:
        result = integrate_innovation_covariance(
            source, start=0, delta=delta, shifted_source=shifted
        )
        exact = scale**2 * (np.exp(0.2 * delta) - np.exp(-2 * k * delta)) / ((0.2 + 2 * k) * delta)
        assert result.innovation_covariance[0, 0] == pytest.approx(exact, rel=1e-12)
        assert result.instantaneous_covariance[0, 0] == pytest.approx(scale**2)
        discrepancies.append(result.discrepancy)
    assert discrepancies[0] > discrepancies[1] > discrepancies[2]


def test_time_discretization_and_projected_shift_are_reported_separately():
    def source(time):
        return np.array([[np.exp(0.2 * time)], [0.4 * np.cos(time)]])

    rates = np.array([1.0, 2.0])

    def shifted(time, lag):
        return np.exp(-rates[:, None] * lag) * source(time)

    reference = integrate_innovation_covariance(
        source, start=0.3, delta=0.5, shifted_source=shifted, quadrature_order=40
    )
    errors = []
    for steps in [4, 8, 16]:
        approximation = integrate_innovation_covariance(
            source, start=0.3, delta=0.5, shifted_source=shifted, time_steps=steps
        )
        errors.append(
            np.linalg.norm(approximation.innovation_covariance - reference.innovation_covariance)
        )
        assert approximation.time_steps == steps
        assert approximation.quadrature_order is None
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8
    projected = integrate_innovation_covariance(
        source,
        start=0.3,
        delta=0.5,
        shift_matrix=lambda lag: np.diag(np.exp(-rates * lag)),
    )
    assert projected.method == "projected_shift"
    np.testing.assert_allclose(
        projected.innovation_covariance, reference.innovation_covariance, atol=1e-13
    )


def test_fixed_deliveries_age_and_transport_cancels():
    # f_t(x)=g(t+x): pure shift evolution has zero fixed-contract increment.
    def curve(time, maturity):
        return np.exp(-0.3 * (maturity + time))

    increments = fixed_delivery_increment(curve, start=0.2, end=0.3, deliveries=[1.2, 2.0, 3.0])
    np.testing.assert_allclose(increments, 0, atol=1e-14)

    # A common linear stochastic source leaves just its realized time increment.
    def curve_with_innovation(time, maturity):
        return curve(time, maturity) + 2 * time

    increments = fixed_delivery_increment(
        curve_with_innovation, start=0.2, end=0.3, deliveries=[1.2, 2.0, 3.0]
    )
    np.testing.assert_allclose(increments, 0.2, atol=1e-14)
    with pytest.raises(ValueError, match="no earlier"):
        fixed_delivery_increment(curve, start=0.2, end=0.3, deliveries=[0.25])


def test_bridge_rejects_ambiguous_or_invalid_interfaces():
    def source(time):
        return np.eye(2)

    with pytest.raises(ValueError, match="exactly one"):
        integrate_innovation_covariance(source, start=0, delta=0.1)
    with pytest.raises(ValueError, match="strictly positive"):
        integrate_innovation_covariance(
            source, start=0, delta=0, shift_matrix=lambda lag: np.eye(2)
        )


def test_spline_projection_error_decreases_for_a_noninvariant_source():
    errors = []
    for dimension in [5, 8, 14]:
        space = SplineSpace(n_basis=dimension, length=6)

        def function(x):
            return np.exp(-np.minimum(x, 6))

        def derivative(x):
            return -np.exp(-x) * (x < 6)

        coefficients = space.project(function, derivative)

        def source(time):
            return coefficients

        def shifted(time, lag):
            return space.project(
                lambda x: function(x + lag), lambda x: derivative(x + lag), breakpoints=[6 - lag]
            )

        direct = integrate_innovation_covariance(source, start=0, delta=0.2, shifted_source=shifted)
        projected = integrate_innovation_covariance(
            source, start=0, delta=0.2, shift_matrix=space.shift_matrix
        )
        errors.append(
            np.linalg.norm(direct.innovation_covariance - projected.innovation_covariance)
        )
    assert errors[0] > 5 * errors[1]
    assert errors[1] > 5 * errors[2]
