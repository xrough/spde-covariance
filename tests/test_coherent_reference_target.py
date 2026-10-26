"""Target-centred moment shrinkage keeps the original convex certificate."""

import numpy as np
import pytest

from spde_covariance.coherent import fit_coherent_partial, fit_coherent_partial_reference


def problem():
    target = np.array([[2, 1.7, np.nan], [1.7, 1, -0.3], [np.nan, -0.3, 0.5]])
    coverage = np.array([[0.3, 0.3, 0], [0.3, 1, 0.7], [0, 0.7, 0.7]])
    return target, {
        "pair_weights": coverage,
        "cliques": ((0, 1), (1, 2)),
        "weights": [0.2, 0.6, 0.2],
        "regularization": 0.2,
        "numerical_error_budget": 1e-6,
    }


def test_prior_target_matches_independent_convex_objective_and_distance_bound():
    pytest.importorskip("cvxpy")
    target, options = problem()
    reference = np.diag([1.0, 2.0, 3.0])
    own = fit_coherent_partial(target, reference_kernel=reference, **options)
    independent = fit_coherent_partial_reference(target, reference_kernel=reference, **options)
    assert own.converged and independent.converged
    mask = np.isfinite(target)
    root = np.sqrt(options["weights"])
    difference = root[:, None] * (own.partial_kernel - independent.partial_kernel) * root[None, :]
    assert (
        np.linalg.norm(difference[mask])
        <= own.partial_error_bound + independent.partial_error_bound
    )
    weighted = root[:, None] * own.partial_kernel * root[None, :]
    data = root[:, None] * target * root[None, :]
    prior = root[:, None] * reference * root[None, :]
    objective = 0.5 * np.sum(
        options["pair_weights"][mask] * (weighted - data)[mask] ** 2
    ) + 0.1 * np.sum((weighted - prior)[mask] ** 2)
    assert own.diagnostics["objective"] == pytest.approx(objective)
    assert own.diagnostics["dual_objective"] <= independent.diagnostics["objective"] + 1e-10


def test_default_zero_reference_retains_legacy_solution():
    target, options = problem()
    implicit = fit_coherent_partial(target, **options)
    explicit = fit_coherent_partial(target, reference_kernel=np.zeros((3, 3)), **options)
    np.testing.assert_array_equal(implicit.partial_kernel, explicit.partial_kernel)
    assert implicit.objective_gap_bound == explicit.objective_gap_bound
    with pytest.raises(ValueError, match="match"):
        fit_coherent_partial(target, reference_kernel=np.eye(2), **options)
