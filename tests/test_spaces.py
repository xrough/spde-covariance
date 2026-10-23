import numpy as np
import pytest

from spde_covariance.spaces import SplineSpace


def test_orthonormal_sobolev_metric_and_coordinate_covariance():
    space = SplineSpace(n_basis=7, length=3, tau=2)
    transform = space.transform
    np.testing.assert_allclose(
        transform.T @ space.raw_gram @ transform, np.eye(space.dimension), atol=2e-13
    )
    x = np.array([0.1, 0.9, 2.8])
    raw_covariance = np.diag(np.arange(1, space.dimension + 1))
    inverse = np.linalg.inv(transform)
    covariance = inverse @ raw_covariance @ inverse.T
    expected = space.raw_evaluate(x) @ raw_covariance @ space.raw_evaluate(x).T
    np.testing.assert_allclose(
        space.evaluate(x) @ covariance @ space.evaluate(x).T, expected, atol=1e-12
    )


def test_project_recovers_sieve_and_constant_tail():
    space = SplineSpace(n_basis=6, length=2)
    coefficients = np.arange(space.dimension) / space.dimension
    projected = space.project(
        lambda x: space.evaluate(x) @ coefficients, lambda x: space.derivative(x) @ coefficients
    )
    np.testing.assert_allclose(projected, coefficients, atol=1e-12)
    np.testing.assert_allclose(
        space.evaluate([2, 3, 20]), np.repeat(space.evaluate([2]), 3, axis=0)
    )
    np.testing.assert_array_equal(space.derivative([2, 3]), np.zeros((2, space.dimension)))
    constant = space.project(lambda x: np.ones_like(x), lambda x: np.zeros_like(x))
    np.testing.assert_allclose(space.evaluate([0, 0.7, 2, 5]) @ constant, 1, atol=1e-12)
    np.testing.assert_allclose(space.shift_matrix(0.5) @ constant, constant, atol=1e-12)


def test_shift_matches_direct_hilbert_projection():
    space = SplineSpace(n_basis=5, length=2)
    shifted = space.project(
        lambda x: space.evaluate(x + 0.25),
        lambda x: space.derivative(x + 0.25),
        quadrature_order=24,
        breakpoints=space.breakpoints - 0.25,
    )
    np.testing.assert_allclose(space.shift_matrix(0.25), shifted, atol=1e-12)
    # At a shift beyond L the exact constant-tail function lies in the sieve.
    coefficients = np.arange(space.dimension, dtype=float)
    output = space.shift_matrix(3) @ coefficients
    np.testing.assert_allclose(
        space.evaluate([0, 0.5, 2]) @ output, (space.evaluate([3]) @ coefficients)[0], atol=1e-12
    )
    np.testing.assert_array_equal(space.shift_matrix(0), np.eye(space.dimension))


def test_project_numerical_derivative_for_smooth_function():
    space = SplineSpace(n_basis=5, length=2)
    analytic = space.project(lambda x: np.exp(-x), lambda x: -np.exp(-x))
    approximate = space.project(lambda x: np.exp(-x))
    np.testing.assert_allclose(approximate, analytic, atol=1e-10)


def test_space_validation():
    with pytest.raises(ValueError):
        SplineSpace(n_basis=3)
    with pytest.raises(ValueError):
        SplineSpace(tau=0)
    with pytest.raises(ValueError):
        SplineSpace().evaluate([-0.1])
