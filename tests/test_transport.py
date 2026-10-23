import numpy as np
import pytest
from scipy.integrate import quad_vec

from spde_covariance.observations import delivery_average_map
from spde_covariance.spaces import SplineSpace
from spde_covariance.transport import (
    ExponentialTransport,
    ProjectedSplineTransport,
    TransportPropagator,
)


def test_exact_shift_agrees_with_curve_values_and_has_semigroup_identity():
    model = ExponentialTransport()
    assert isinstance(model, TransportPropagator)
    assert model.kind == "exact_transport"
    np.testing.assert_allclose(
        model.transform.T @ model.raw_gram @ model.transform, np.eye(model.dimension), atol=2e-13
    )
    points = np.array([0.0, 0.2, 1.0, 4.0])
    np.testing.assert_allclose(
        model.evaluate(points) @ model.matrix(0.3), model.evaluate(points + 0.3), atol=2e-14
    )
    np.testing.assert_allclose(
        model.derivative(points) @ model.matrix(0.3), model.derivative(points + 0.3), atol=2e-14
    )
    np.testing.assert_allclose(model.matrix(0.2) @ model.matrix(0.7), model.matrix(0.9), atol=3e-14)
    np.testing.assert_array_equal(model.matrix(0), np.eye(model.dimension))
    assert not np.allclose(model.matrix(0.3), model.matrix(0.3).T)


def test_exact_innovation_covariance_matches_independent_integration():
    model = ExponentialTransport()
    factor = np.random.default_rng(19).normal(size=(model.dimension, model.dimension))
    covariance = factor @ factor.T
    duration = 0.4
    reference, _ = quad_vec(
        lambda time: model.matrix(time) @ covariance @ model.matrix(time).T,
        0,
        duration,
        epsabs=1e-12,
    )
    actual = model.innovation_covariance(covariance, duration)
    np.testing.assert_allclose(actual, reference / duration, rtol=2e-12, atol=2e-12)
    assert np.linalg.eigvalsh(actual)[0] >= -1e-12
    assert not np.allclose(actual, covariance)
    np.testing.assert_allclose(
        model.innovation_covariance(covariance, 1e-11), covariance, atol=2e-9
    )


def test_scalar_innovation_covariance_and_tiny_interval_limit():
    model = ExponentialTransport(rates=[2.0], tau=0.5)
    duration = 0.3
    expected = 3 * -np.expm1(-4 * duration) / (4 * duration)
    assert model.innovation_covariance([[3.0]], duration)[0, 0] == pytest.approx(expected)
    np.testing.assert_allclose(model.innovation_covariance([[3.0]], 1e-300), [[3.0]], atol=1e-14)


def test_raw_basis_reordering_preserves_operator_and_covariance():
    first = ExponentialTransport()
    permutation = [2, 0, 3, 1]
    second = ExponentialTransport(rates=first.rates[permutation])
    raw_permutation = np.eye(first.dimension)[:, permutation]
    rotation = np.linalg.solve(first.transform, raw_permutation @ second.transform)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(first.dimension), atol=2e-13)
    np.testing.assert_allclose(
        second.matrix(0.25), rotation.T @ first.matrix(0.25) @ rotation, atol=2e-13
    )
    covariance = np.diag([0.5, 0.3, 0.2, 0.1])
    rotated = rotation.T @ covariance @ rotation
    np.testing.assert_allclose(
        second.innovation_covariance(rotated, 0.25),
        rotation.T @ first.innovation_covariance(covariance, 0.25) @ rotation,
        atol=2e-13,
    )
    assert first.basis_id != second.basis_id


def test_exponential_space_adapter_delivery_averages_and_basis_identity():
    model = ExponentialTransport()
    rows = delivery_average_map(model, [[0.0, 1.0], [1.0, 2.0]])
    expected_raw = np.asarray(
        [
            (np.exp(-start * model.rates) - np.exp(-end * model.rates)) / model.rates
            for start, end in [[0, 1], [1, 2]]
        ]
    )
    np.testing.assert_allclose(rows, expected_raw @ model.transform, atol=1e-13)
    assert ExponentialTransport(rates=list(model.rates)).basis_id == model.basis_id
    assert ExponentialTransport(tau=2).basis_id != model.basis_id
    exposed = model.transform
    exposed[:] = 0
    assert np.any(model.transform)


def test_projected_spline_covariance_is_the_declared_recursion_target():
    space = SplineSpace(n_basis=5, length=2)
    model = ProjectedSplineTransport(space)
    assert isinstance(model, TransportPropagator)
    assert model.kind == "projected_mild_recursion"
    np.testing.assert_allclose(model.matrix(0.3), space.shift_matrix(0.3), atol=0)
    np.testing.assert_array_equal(model.matrix(0), np.eye(space.dimension))
    assert np.linalg.norm(model.matrix(0.3) @ model.matrix(0.3) - model.matrix(0.6)) > 1e-3
    covariance = np.diag(np.arange(1, space.dimension + 1) / space.dimension)
    duration = 1.2
    reference, _ = quad_vec(
        lambda time: model.matrix(time) @ covariance @ model.matrix(time).T,
        0,
        duration,
        points=[1.0],
        epsabs=1e-11,
    )
    actual = model.innovation_covariance(covariance, duration)
    np.testing.assert_allclose(actual, reference / duration, atol=2e-11)
    assert np.linalg.eigvalsh(actual)[0] >= -1e-12
    assert ProjectedSplineTransport(SplineSpace(n_basis=5, length=2)).basis_id == model.basis_id
    assert ProjectedSplineTransport(SplineSpace(n_basis=5, length=3)).basis_id != model.basis_id


@pytest.mark.parametrize(
    "options, message",
    [
        ({"rates": []}, "nonempty"),
        ({"rates": [1, 1]}, "distinct"),
        ({"rates": [1.0, 1.0 + 1e-10, 2.0]}, "ill-conditioned"),
        ({"rates": [0.5]}, "strictly greater"),
        ({"rates": [1j]}, "real"),
        ({"rates": [np.inf]}, "finite"),
        ({"tau": 0}, "strictly positive"),
    ],
)
def test_exponential_configuration_validation(options, message):
    with pytest.raises(ValueError, match=message):
        ExponentialTransport(**options)


@pytest.mark.parametrize("dt", [-0.1, np.inf, np.nan, 1j, [0.1], True])
def test_transport_rejects_invalid_durations(dt):
    for model in (ExponentialTransport(), ProjectedSplineTransport(SplineSpace())):
        with pytest.raises(ValueError):
            model.matrix(dt)
        with pytest.raises(ValueError):
            model.innovation_covariance(np.eye(model.dimension), dt)


def test_transport_covariance_dimension_psd_and_time_validation():
    for model in (ExponentialTransport(), ProjectedSplineTransport(SplineSpace())):
        with pytest.raises(ValueError, match="dimension"):
            model.innovation_covariance(np.eye(model.dimension + 1), 0.1)
        with pytest.raises(ValueError, match="positive semidefinite"):
            model.innovation_covariance(-np.eye(model.dimension), 0.1)
        with pytest.raises(ValueError, match="strictly positive"):
            model.innovation_covariance(np.eye(model.dimension), 0)
        with pytest.raises(ValueError, match="quadrature_order"):
            model.innovation_covariance(np.eye(model.dimension), 0.1, quadrature_order=0)
