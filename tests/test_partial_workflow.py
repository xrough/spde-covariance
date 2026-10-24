"""Independent controls for full observations, sieve bias and risk geometry."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from spde_covariance.laguerre import LaguerreTransport
from spde_covariance.partial_design import (
    Contract,
    LinearFunctional,
    contract_pool,
    fixed_exposures,
    functional_coefficients,
    functional_gram,
    observation_omission,
    prepare_parent_model,
    simulate_parent_pool,
    span_diagnostics,
)


def test_omitted_coordinates_cannot_be_treated_as_independent_noise():
    # A valid Gaussian covariance whose retained/omitted coordinates correlate.
    # Omission lowers observed variance here; no PSD noise correction can equal it.
    covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
    result = observation_omission(covariance, [[1.0, -1.0]], 1)
    assert_allclose(result["full"], [[0.4]], atol=1e-15)
    assert_allclose(result["sieve"], [[1.0]])
    assert_allclose(result["tail_tail"], [[1.0]])
    assert_allclose(result["cross"] + result["cross_transpose"], [[-1.6]])
    assert_allclose(result["omission"], [[-0.6]], atol=1e-15)
    assert_allclose(
        result["omission"],
        result["tail_tail"] + result["cross"] + result["cross_transpose"],
    )


def test_full_h_gram_agrees_with_independent_kernel_and_delivery_integrals():
    points = [Contract("point", 0.5), Contract("point", 1.5)]
    expected = np.array(
        [[2 - np.exp(-0.5), 2 - np.exp(-0.5)], [2 - np.exp(-0.5), 2 - np.exp(-1.5)]]
    )
    assert_allclose(functional_gram(points), expected, atol=3e-15)
    delivery = LinearFunctional(
        (Contract("delivery", 0.5, 1), Contract("delivery", 1.5, 2)),
        np.array([1.0, -1.0]),
    )
    independent = (
        quad(lambda u: np.exp(-u) * (2 * (u - 0.5)) ** 2, 0.5, 1)[0]
        + np.exp(-1)
        - np.exp(-1.5)
        + quad(lambda u: np.exp(-u) * (2 * (2 - u)) ** 2, 1.5, 2)[0]
    )
    assert functional_gram([delivery])[0, 0] == pytest.approx(independent, abs=2e-15)
    exposures = fixed_exposures()
    for exposure in exposures.values():
        assert functional_gram([exposure])[0, 0] == pytest.approx(1, abs=1e-14)
    full_span = span_diagnostics(exposures["offgrid_spread"], contract_pool())
    assert not full_span["available"]
    assert full_span["relative_residual"] > 0.2


def test_full_observed_spread_survives_when_a_sieve_representation_vanishes():
    prepared = prepare_parent_model(dimension=32)
    parent = simulate_parent_pool(prepared, 32, seed=873, noise_std=0)
    coarse = parent.observations("fixed_points", 1)
    finer = parent.observations("fixed_points", 8)
    for low, high in zip(coarse, finer, strict=True):
        assert_array_equal(low.y, high.y)
        assert_array_equal(low.noise_covariance, high.noise_covariance)
    exposure = fixed_exposures()["point_spread"]
    assert_array_equal(functional_coefficients([exposure], LaguerreTransport(1)), [[0]])
    assert span_diagnostics(exposure, contract_pool()[:2])["available"]
    measured = np.array([item.y[0] - item.y[1] for item in coarse])
    assert np.mean(measured**2) > 1e-3
    # Simulating the data after J=1 projection would incorrectly erase this spread.
    assert_array_equal(coarse[0].operator[0], coarse[0].operator[1])


def test_rotated_source_zero_angle_is_the_same_model_without_renormalizing():
    original = prepare_parent_model(dimension=32, source_exponent=4, source_trace=0.7)
    rotated = prepare_parent_model(
        source="rotated",
        dimension=32,
        source_exponent=4,
        source_trace=0.7,
        rotation_angle=0,
    )
    assert_array_equal(original.source_covariance, rotated.source_covariance)
    assert_array_equal(original.innovation_covariance, rotated.innovation_covariance)
    assert np.trace(original.source_covariance) < 0.7
    assert original.tail_bounds["full_source_trace_tail"] + np.trace(
        original.source_covariance
    ) == pytest.approx(0.7)
