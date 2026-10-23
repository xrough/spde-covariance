import numpy as np
import pytest

import spde_covariance.geometry as geometry
from spde_covariance.geometry import (
    diagnose_design,
    gram_operator,
    lifted_matrix,
    smat,
    svec,
    symmetric_basis,
)


def test_symmetric_coordinates_preserve_frobenius_geometry():
    first = np.array([[1.0, 2.0], [2.0, 4.0]])
    second = np.array([[3.0, -1.0], [-1.0, 2.0]])
    np.testing.assert_allclose(smat(svec(first)), first)
    assert svec(first) @ svec(second) == pytest.approx(np.sum(first * second))
    basis = symmetric_basis(3)
    np.testing.assert_allclose(np.einsum("aij,bij->ab", basis, basis), np.eye(6), atol=1e-15)
    with pytest.raises(ValueError, match="triangular"):
        smat(np.ones(4))


def test_lifted_map_and_adjoint():
    operator = np.array([[1.0, 2.0], [-2.0, 0.5], [0.0, 1.0]])
    covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
    measurement = np.array([[1.0, 0.2, 0.3], [0.2, -1.0, 0.5], [0.3, 0.5, 2.0]])
    lifted = lifted_matrix(operator)
    np.testing.assert_allclose(lifted @ svec(covariance), svec(operator @ covariance @ operator.T))
    np.testing.assert_allclose(
        lifted.T @ svec(measurement), svec(operator.T @ measurement @ operator)
    )


def test_separate_coordinate_days_do_not_identify_cross_covariance():
    operators = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    cross = np.array([[0.0, 1.0], [1.0, 0.0]])
    variance = np.diag([1.0, 0.0])
    report = diagnose_design(operators, targets={"cross": cross, "variance": variance})
    assert report["rank"] == 2
    assert not report["targets"]["cross"]["identified"]
    assert np.isinf(report["targets"]["cross"]["variance_factor"])
    assert report["targets"]["variance"]["identified"]
    assert report["targets"]["variance"]["variance_factor"] == pytest.approx(2.0)
    np.testing.assert_allclose(gram_operator([np.eye(2)]), np.eye(3))


def test_gram_respects_supplied_weights_and_ragged_designs():
    first = np.eye(2)
    second = np.array([[1.0, -1.0]])
    weights = [2.0, 3.0]
    expected = sum(
        weight * lifted_matrix(op).T @ lifted_matrix(op)
        for op, weight in zip([first, second], weights, strict=True)
    )
    np.testing.assert_allclose(gram_operator([first, second], weights), expected)
    with pytest.raises(ValueError, match="weights"):
        gram_operator([first], [-1.0])
    with pytest.raises(ValueError, match="common curve dimension"):
        gram_operator([first, np.eye(3)])


def test_repeated_designs_reuse_lifts_without_changing_weighted_gram(monkeypatch):
    first = np.array([[1.0, 0.2], [-0.4, 2.0]])
    second = np.array([[1.0, -1.0]])
    operators = [first, first.copy(), second, np.array(first, order="F"), second.copy()]
    weights = [0.3, 0.1, 0.2, 0.05, 0.35]
    original = geometry.lifted_matrix
    expected = sum(
        weight * original(operator).T @ original(operator)
        for operator, weight in zip(operators, weights, strict=True)
    )
    calls = []

    def counted(operator):
        calls.append(operator.shape)
        return original(operator)

    monkeypatch.setattr(geometry, "lifted_matrix", counted)
    np.testing.assert_allclose(gram_operator(operators, weights), expected)
    assert len(calls) == 2


def test_target_risk_diagnostics_are_rotation_invariant():
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    operator = np.array([[1.0, 0.3], [0.2, 2.0]])
    target = np.array([[1.0, -1.0], [-1.0, 1.0]])
    first = diagnose_design([operator], targets={"risk": target})
    second = diagnose_design(
        [operator @ rotation], targets={"risk": rotation.T @ target @ rotation}
    )
    np.testing.assert_allclose(first["eigenvalues"], second["eigenvalues"], atol=1e-12)
    assert first["targets"]["risk"]["observability_factor"] == pytest.approx(
        second["targets"]["risk"]["observability_factor"]
    )
