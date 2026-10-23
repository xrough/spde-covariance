"""Independent metric, transport, covariance and infinite-tail checks."""

import numpy as np
import pytest
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm
from scipy.special import eval_laguerre, roots_laguerre, zeta

from spde_covariance.laguerre import LaguerreTransport, PowerLawSource
from spde_covariance.transport import TransportPropagator


def test_basis_is_orthonormal_in_the_fixed_half_line_metric():
    model = LaguerreTransport(8)
    nodes, weights = roots_laguerre(32)
    # The Gaussian weight already supplies exp(-x); undo the derivative's
    # exp(-x) factor to independently integrate products of Laguerre polynomials.
    polynomials = model.derivative(nodes) * np.exp(nodes[:, None])
    origin = model.evaluate(0)[0]
    gram = np.outer(origin, origin) + polynomials.T @ (weights[:, None] * polynomials)
    np.testing.assert_allclose(gram, np.eye(model.dimension), atol=4e-14)
    for x in (0.0, 0.13, 1.0, 5.0):
        expected = [1.0] + [
            quad(lambda u: np.exp(-u) * eval_laguerre(j - 1, u), 0, x)[0]
            for j in range(1, model.dimension)
        ]
        np.testing.assert_allclose(model.evaluate(x)[0], expected, atol=2e-15)
    assert isinstance(model, TransportPropagator)


def test_stable_tiny_and_large_maturity_evaluation_at_high_dimension():
    model = LaguerreTransport(256)
    points = np.array([0.0, 1e-15, 0.1, 1.0, 10.0, 1000.0, 1e300])
    values, slopes = model.evaluate(points), model.derivative(points)
    assert np.isfinite(values).all() and np.isfinite(slopes).all()
    np.testing.assert_array_equal(values[0], np.r_[1.0, np.zeros(255)])
    np.testing.assert_array_equal(slopes[0], np.r_[0.0, np.ones(255)])
    np.testing.assert_allclose(values[:, 1], -np.expm1(-points), atol=0)
    np.testing.assert_allclose(values[:, 2], points * np.exp(-points), atol=0)
    assert values[1, 2] == pytest.approx(1e-15, rel=2e-15, abs=0)
    np.testing.assert_array_equal(values[-1], np.r_[1.0, 1.0, np.zeros(254)])


def test_generator_shift_semigroup_and_nested_prefixes_are_exact():
    model = LaguerreTransport(20)
    generator = model.generator
    np.testing.assert_array_equal(generator[:, 0], 0)
    for j in range(1, model.dimension):
        np.testing.assert_array_equal(generator[:, j], np.r_[1.0, -np.ones(j), np.zeros(19 - j)])
    for time in (0.0, 1e-12, 0.2, 2.0):
        shift = model.matrix(time)
        np.testing.assert_allclose(shift, expm(time * generator), atol=2e-14)
        points = np.array([0, 0.1, 0.7, 3.0])
        np.testing.assert_allclose(
            model.evaluate(points) @ shift, model.evaluate(points + time), atol=1e-14
        )
        np.testing.assert_allclose(
            model.derivative(points) @ shift, model.derivative(points + time), atol=1e-14
        )
        assert np.linalg.norm(shift, 2) ** 2 <= 2 - np.exp(-time) + 1e-13
    np.testing.assert_allclose(
        model.matrix(0.17) @ model.matrix(0.53), model.matrix(0.7), atol=2e-15
    )
    smaller = LaguerreTransport(7)
    np.testing.assert_allclose(model.matrix(0.4)[:7, :7], smaller.matrix(0.4), atol=0)
    np.testing.assert_array_equal(model.matrix(0.4)[7:, :7], 0)
    assert np.linalg.norm(model.matrix(0.4)[:7, 7:]) > 0.1  # Projection does not commute.
    assert model.family_id == smaller.family_id and model.basis_id != smaller.basis_id
    generator[:] = 0
    assert np.any(model.generator)


def test_fixed_infinite_source_prefixes_tail_and_generator_regularity():
    source = PowerLawSource()
    np.testing.assert_array_equal(source.eigenvalues(6), source.eigenvalues(12)[:6])
    assert source.tail_trace(0) == 1.0
    for cutoff in (1, 8, 64, 256):
        assert source.eigenvalues(cutoff).sum() + source.tail_trace(cutoff) == pytest.approx(
            1, abs=2e-15
        )
        tail = source.tail_trace(cutoff)
        assert 1 / (2 * zeta(3, 1) * (cutoff + 1) ** 2) <= tail <= 1 / (2 * zeta(3, 1) * cutoff**2)
        assert np.trace(source.finite_matrix(cutoff)) < 1
    assert source.tail_trace(256) == pytest.approx(6.322205210295422e-6, rel=2e-15)
    model = LaguerreTransport(32)
    covariance = source.finite_matrix(32)
    assert np.linalg.norm(model.generator @ covariance - covariance @ model.generator) > 0.1
    np.testing.assert_array_equal(np.sum(model.generator[:, 1:] ** 2, axis=0), np.arange(2, 33))
    assert np.sum(source.eigenvalues(32)[1:] * np.arange(2, 33)) < (zeta(2, 1) - 1) / zeta(3, 1)
    assert source.basis_family_id == model.family_id
    assert source.family_id == PowerLawSource().spectrum_id
    assert source.family_id != PowerLawSource(exponent=4).family_id


def test_independent_covariance_integrators_match_direct_adaptive_quadrature():
    model = LaguerreTransport(6)
    factor = np.random.default_rng(31).normal(size=(6, 4))
    covariance = factor @ factor.T
    duration = 0.43
    # expm(generator*r) and adaptive integration are independent of the
    # closed translation and small-step Gaussian integration implementations.
    generator = model.generator
    expected, _ = quad_vec(
        lambda r: expm(r * generator) @ covariance @ expm(r * generator).T,
        0,
        duration,
        epsabs=1e-12,
    )
    for method in ("quadrature", "block"):
        actual = model.innovation_covariance(covariance, duration, method=method)
        np.testing.assert_allclose(actual, expected / duration, atol=2e-13, rtol=2e-14)
        np.testing.assert_allclose(
            model.cumulative_covariance(covariance, duration, method=method), expected, atol=1e-13
        )
        assert np.linalg.eigvalsh(actual)[0] >= -1e-13
        assert not np.allclose(actual, covariance)
    diagnostics = model.covariance_diagnostics(covariance, duration)
    assert diagnostics["absolute_hs_discrepancy"] < 1e-12
    assert diagnostics["numerical_error_certificate"] is False
    assert diagnostics["small_step"] * np.linalg.norm(generator, np.inf) <= 1


def test_covariance_composition_long_horizon_scalar_and_tiny_time_limits():
    model = LaguerreTransport(12)
    covariance = PowerLawSource().finite_matrix(12)
    for method in ("quadrature", "block"):
        left = model.cumulative_covariance(covariance, 4, method=method)
        second = model.matrix(3.7)
        right = second @ model.cumulative_covariance(covariance, 0.3, method=method) @ second.T
        right += model.cumulative_covariance(covariance, 3.7, method=method)
        np.testing.assert_allclose(left, right, atol=2e-13)
        np.testing.assert_allclose(
            model.innovation_covariance(covariance, 1e-300, method=method), covariance, atol=1e-15
        )
        np.testing.assert_array_equal(model.cumulative_covariance(covariance, 0, method=method), 0)
        scalar = LaguerreTransport(1)
        np.testing.assert_allclose(
            scalar.innovation_covariance([[0.7]], 3, method=method), [[0.7]], atol=2e-16
        )


def test_covariance_cache_is_bounded_and_has_no_mutable_input_or_output_aliases():
    model = LaguerreTransport(4, cache_max_bytes=260)  # One 128-byte key and value.
    covariance = np.eye(4)
    first = model.innovation_covariance(covariance, 0.1)
    expected = first.copy()
    first[:] = 100
    np.testing.assert_array_equal(model.innovation_covariance(covariance, 0.1), expected)
    assert model.cache_info()["hits"] == 1
    covariance[0, 0] = 2
    changed = model.innovation_covariance(covariance, 0.1)
    assert not np.array_equal(changed, expected)
    assert model.cache_info()["entries"] == 1
    assert model.cache_info()["bytes"] <= 260
    disabled = LaguerreTransport(4, cache_max_bytes=0)
    disabled.innovation_covariance(covariance, 0.1)
    assert disabled.cache_info()["entries"] == 0


def test_certificates_dominate_an_independent_finite_omitted_source_band():
    source = PowerLawSource()
    fine, observed, cutoff, duration, time = 64, 12, 24, 1 / 52, 1.0
    model = LaguerreTransport(fine)
    band = source.finite_matrix(fine)
    band[:cutoff, :cutoff] = 0
    innovation = model.innovation_covariance(band, duration)
    state = model.cumulative_covariance(band, time)
    assert np.trace(innovation) <= source.innovation_tail_bound(cutoff, duration)
    assert np.trace(state) <= source.state_mse_tail_bound(cutoff, time)
    projected = source.projected_tail_bound(observed, cutoff, duration)
    assert np.linalg.eigvalsh(projected["matrix"] - innovation[:observed, :observed])[0] >= -1e-15
    leakage = model.matrix(duration)[:observed].copy()
    leakage[:, :observed] = 0
    actual = innovation[:observed, :observed] + leakage @ state @ leakage.T / duration
    operational = source.operational_tail_bound(observed, cutoff, duration, time)
    assert np.linalg.eigvalsh(operational["matrix"] - actual)[0] >= -1e-15
    assert np.linalg.norm(actual, "fro") <= operational["hs_bound"]


def test_sharp_bound_matches_audited_numbers_and_is_stable_at_tiny_dt():
    source = PowerLawSource()
    assert source.innovation_tail_bound(256, 1 / 52) == pytest.approx(
        6.38260782902583e-6, rel=2e-15
    )
    assert source.state_mse_tail_bound(256, 1) == pytest.approx(8.648014530030083e-6, rel=2e-15)
    assert source.operational_tail_bound(64, 256, 1 / 208, 1)["hs_bound"] == pytest.approx(
        1.7891871034556024e-6, rel=2e-14
    )
    model = LaguerreTransport(9)
    duration = 0.3
    shift = model.matrix(duration)
    direct = np.diag(np.r_[2 - np.exp(-duration), np.full(8, np.exp(-duration))]) - shift @ shift.T
    np.testing.assert_allclose(model._leakage_rate(duration) * duration, direct, atol=1e-15)
    vector = np.r_[1.0, -np.ones(8)]
    tiny = source.operational_tail_bound(9, 256, 1e-300, 1)
    expected = tiny["q_M"] * (np.eye(9) + (1 - np.exp(-1)) * np.outer(vector, vector))
    np.testing.assert_allclose(tiny["matrix"], expected, atol=2e-22, rtol=2e-15)
    np.testing.assert_array_equal(
        source.operational_tail_bound(9, 256, duration, 0)["matrix"],
        source.projected_tail_bound(9, 256, duration)["matrix"],
    )


@pytest.mark.parametrize("dimension", (0, -1, True, 4.0, 1j))
def test_invalid_prefix_dimensions_are_rejected(dimension):
    with pytest.raises(ValueError):
        LaguerreTransport(dimension)
    with pytest.raises(ValueError):
        PowerLawSource().finite_matrix(dimension)


@pytest.mark.parametrize("value", (-1, np.nan, np.inf, True, 1j, [1]))
def test_invalid_time_and_maturities_are_rejected(value):
    model = LaguerreTransport(3)
    for method in (model.matrix, lambda t: model.innovation_covariance(np.eye(3), t)):
        with pytest.raises(ValueError):
            method(value)
    if not isinstance(value, list):
        with pytest.raises(ValueError):
            model.evaluate(value)


def test_source_and_covariance_validation():
    for options in (
        {"exponent": 1},
        {"exponent": np.inf},
        {"exponent": True},
        {"trace": 0},
        {"trace": 1j},
    ):
        with pytest.raises(ValueError):
            PowerLawSource(**options)
    with pytest.raises(ValueError):
        LaguerreTransport(3, cache_max_bytes=True)
    model = LaguerreTransport(3)
    for covariance in (
        np.eye(2),
        -np.eye(3),
        np.eye(3) * np.nan,
        np.eye(3) * 1j,
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
    ):
        with pytest.raises(ValueError):
            model.innovation_covariance(covariance, 0.1)
    for options in ({"quadrature_order": 0}, {"method": "unknown"}):
        with pytest.raises(ValueError):
            model.innovation_covariance(np.eye(3), 0.1, **options)
    with pytest.raises(ValueError):
        model.innovation_covariance(np.eye(3), 0)
    with pytest.raises(ValueError):
        PowerLawSource().tail_trace(-1)
