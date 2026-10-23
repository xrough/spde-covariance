import numpy as np
import pytest

from spde_covariance.linalg import project_psd_trace, validate_psd, validate_symmetric


def test_trace_projection_and_variational_inequality():
    original = np.diag([3.0, 1.0, -2.0])
    projected = project_psd_trace(original, 3.0)
    np.testing.assert_allclose(projected, np.diag([2.5, 0.5, 0.0]))
    rng = np.random.default_rng(937)
    for _ in range(20):
        factor = rng.normal(size=(3, 3))
        feasible = factor @ factor.T
        feasible *= 3 / np.trace(feasible)
        assert np.sum((original - projected) * (feasible - projected)) <= 1e-12
    np.testing.assert_array_equal(project_psd_trace(original, 0), np.zeros((3, 3)))


def test_psd_projection_is_orthogonally_equivariant():
    rotation, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [2.0, -1.0, 1.0], [1.0, 1.0, -2.0]]))
    original = np.diag([3.0, 1.0, -2.0])
    expected = rotation @ project_psd_trace(original, 2.0) @ rotation.T
    np.testing.assert_allclose(
        project_psd_trace(rotation @ original @ rotation.T, 2.0), expected, atol=1e-12
    )


@pytest.mark.parametrize(
    "matrix", [np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([[np.nan]]), np.array([[1j]])]
)
def test_invalid_symmetric_inputs(matrix):
    with pytest.raises(ValueError):
        validate_symmetric(matrix)


def test_invalid_psd_and_trace_cap():
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_psd(np.diag([1.0, -0.01]))
    with pytest.raises(ValueError, match="trace_cap"):
        project_psd_trace(np.eye(2), -1)
