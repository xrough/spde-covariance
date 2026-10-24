"""Observed-clique consensus, feasibility and independent primal-dual checks."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from spde_covariance.coherent import (
    CoherentConvergenceError,
    fit_coherent_partial,
    fit_coherent_partial_reference,
)
from spde_covariance.completion import complete_kernel, observed_mask
from spde_covariance.observation_statistics import partial_kernel_from_arrays


def consensus_case():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    kernel = np.array([[1, 2, np.nan], [2, 1, 0], [np.nan, 0, 4]], dtype=float)
    return kernel, cliques, observed_mask(cliques, 3).astype(float)


def weighted_distance(a, b, mask, weights):
    root = np.sqrt(weights)
    return np.linalg.norm((root[:, None] * (a - b) * root[None, :])[mask])


def test_shared_overlap_optimum_is_not_independent_psd_projection_and_averaging():
    kernel, cliques, coverage = consensus_case()
    result = fit_coherent_partial(
        kernel, pair_weights=coverage, cliques=cliques, numerical_error_budget=1e-6
    )
    assert result.converged
    expected = np.array([[1.5, 1.5, np.nan], [1.5, 1.5, 0], [np.nan, 0, 4]])
    mask = coverage > 0
    assert np.linalg.norm((result.partial_kernel - expected)[mask]) <= result.partial_error_bound
    assert result.diagnostics["objective"] == pytest.approx(0.5, abs=1e-9)
    assert result.diagnostics["dual_objective"] <= 0.5 + 1e-12
    assert result.objective_gap_bound >= result.diagnostics["objective"] - 0.5 - 1e-12
    assert np.isnan(result.partial_kernel[~mask]).all()
    for clique in cliques:
        assert np.linalg.eigvalsh(result.partial_kernel[np.ix_(clique, clique)])[0] >= -1e-14
    completion = complete_kernel(result.partial_kernel, cliques=cliques, population=True)
    assert completion.available and completion.diagnostics["output_psd"]
    # Separate projections produce shared diagonal1.5 and1.0; averaging1.25
    # makes the first block indefinite instead of solving the joint problem.
    assert np.linalg.eigvalsh([[1.5, 1.5], [1.5, 1.25]])[0] < 0


def test_population_identity_and_known_uniform_ridge_shrinkage_in_weighted_coordinates():
    kernel = np.array([[2.0, 0.2, np.nan], [0.2, 1.0, 0.3], [np.nan, 0.3, 1.5]])
    cliques = (np.array([0, 1]), np.array([1, 2]))
    mask = observed_mask(cliques, 3)
    weights = np.array([0.2, 0.3, 0.5])
    for regularization in (0.0, 0.1):
        result = fit_coherent_partial(
            kernel,
            pair_weights=mask.astype(float) * 0.4,
            cliques=cliques,
            weights=weights,
            regularization=regularization,
            numerical_error_budget=1e-6,
        )
        assert result.converged
        expected = kernel * 0.4 / (0.4 + regularization)
        assert (
            weighted_distance(result.partial_kernel, expected, mask, weights)
            <= result.partial_error_bound
        )
        assert result.diagnostics["feasibility_diagonal_shift"] == 0
        assert result.diagnostics["strong_convexity"] == pytest.approx(0.4 + regularization)
        assert result.diagnostics["admm_residuals_within_tolerance"] == (
            result.diagnostics["primal_residual"] <= result.diagnostics["primal_tolerance"]
            and result.diagnostics["dual_residual"] <= result.diagnostics["dual_tolerance"]
        )


def test_eigenvalue_floor_is_in_weighted_grid_coordinates_and_is_an_explicit_constraint():
    weights = np.array([0.2, 0.8])
    kernel = np.array([[0.1, 0.3], [0.3, -0.2]])
    root = np.sqrt(weights)
    target = root[:, None] * kernel * root[None, :]
    values, vectors = np.linalg.eigh(target)
    expected_weighted = (vectors * np.maximum(values, 0.1)) @ vectors.T
    result = fit_coherent_partial(
        kernel,
        pair_weights=np.ones((2, 2)),
        cliques=(np.arange(2),),
        weights=weights,
        eigenvalue_floor=0.1,
        numerical_error_budget=1e-6,
    )
    assert result.converged
    actual = root[:, None] * result.partial_kernel * root[None, :]
    assert np.linalg.eigvalsh(actual)[0] >= 0.1 - 1e-13
    assert np.linalg.norm(actual - expected_weighted) <= result.partial_error_bound
    assert result.diagnostics["eigenvalue_floor"] == 0.1
    assert "not H coordinates" in result.diagnostics["coordinate_convention"]


def test_negative_noise_corrected_input_is_used_without_a_preprojection():
    result = fit_coherent_partial(
        [[-1.0]],
        pair_weights=[[1.0]],
        cliques=([0],),
        regularization=0.2,
        numerical_error_budget=1e-6,
    )
    assert result.converged
    assert result.partial_kernel[0, 0] == pytest.approx(0, abs=1e-12)
    # Projecting the corrected target before fitting would wrongly report0.
    assert result.diagnostics["objective"] == pytest.approx(0.5, abs=1e-12)


def test_maxiter_failure_returns_feasible_audit_with_explicit_restoration():
    kernel, cliques, coverage = consensus_case()
    result = fit_coherent_partial(
        kernel,
        pair_weights=coverage,
        cliques=cliques,
        max_iterations=1,
        numerical_error_budget=1e-10,
    )
    assert not result.converged
    assert "maximum iterations" in result.diagnostics["status"]
    assert result.partial_error_bound > 1e-10
    shift = result.diagnostics["feasibility_diagonal_shift"]
    assert shift > 0
    assert result.diagnostics["restoration_weighted_hs_change"] == pytest.approx(shift * np.sqrt(3))
    # Restoration from an infeasible point can change objective in either
    # direction; its signed effect must be retained rather than hidden.
    assert result.diagnostics["restoration_objective_change"] != 0
    for clique in cliques:
        assert np.linalg.eigvalsh(result.partial_kernel[np.ix_(clique, clique)])[0] >= -1e-14
    assert complete_kernel(result.partial_kernel, cliques=cliques, population=True).available
    error = CoherentConvergenceError(result)
    assert error.fit is result
    assert "observed-partial error bound" in str(error)


def test_independent_cvxpy_reference_matches_weighted_coverage_objective_and_certificate():
    pytest.importorskip("cvxpy")
    cliques = (np.arange(3), np.arange(1, 4))
    mask = observed_mask(cliques, 4)
    kernel = np.array(
        [[1, 0.8, 0.4, np.nan], [0.8, 0.2, 0.4, 0.5], [0.4, 0.4, 0.1, 0.8], [np.nan, 0.5, 0.8, 1.0]]
    )
    weights = np.array([0.1, 0.3, 0.2, 0.4])
    coverage = np.where(
        mask,
        np.array([[0.3, 0.3, 0.3, 0], [0.3, 1, 1, 0.7], [0.3, 1, 1, 0.7], [0, 0.7, 0.7, 0.7]]),
        0.0,
    )
    settings = dict(
        pair_weights=coverage,
        cliques=cliques,
        weights=weights,
        regularization=0.03,
        eigenvalue_floor=0.01,
        numerical_error_budget=1e-5,
    )
    own = fit_coherent_partial(kernel, **settings)
    reference = fit_coherent_partial_reference(kernel, **settings, solver_tolerance=1e-11)
    assert own.converged and reference.converged
    assert weighted_distance(own.partial_kernel, reference.partial_kernel, mask, weights) <= (
        own.partial_error_bound + reference.partial_error_bound
    )
    assert own.diagnostics["dual_objective"] <= reference.diagnostics["objective"] + 1e-12
    assert reference.diagnostics["dual_objective"] <= own.diagnostics["objective"] + 1e-12
    gap = max(own.diagnostics["objective"], reference.diagnostics["objective"]) - min(
        own.diagnostics["dual_objective"], reference.diagnostics["dual_objective"]
    )
    assert gap <= own.objective_gap_bound + reference.objective_gap_bound + 1e-12


def test_already_consistent_singular_population_support_is_preserved_by_fit_then_completion():
    nodes = np.linspace(0, 1, 7)
    features = np.column_stack([np.ones(7), nodes])
    truth = features @ features.T
    cliques = (np.arange(5), np.arange(2, 7))
    mask = observed_mask(cliques, 7)
    kernel = np.where(mask, truth, np.nan)
    result = fit_coherent_partial(
        kernel, pair_weights=mask.astype(float), cliques=cliques, numerical_error_budget=1e-5
    )
    assert result.converged
    exact = complete_kernel(result.partial_kernel, cliques=cliques, population=True)
    assert exact.available
    assert np.linalg.norm(result.partial_kernel[mask] - truth[mask]) <= result.partial_error_bound
    np.testing.assert_allclose(exact.kernel, truth, atol=2e-5)


def test_solver_only_sees_observed_corrected_statistics_and_known_noise():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    parent = np.array([[1.2, -0.3, 1e8], [-1e8, 0.7, -1.1], [-0.4, 0.2, 1e8]])
    rows = (cliques[0], cliques[1], cliques[0])

    def fit(values):
        stats = partial_kernel_from_arrays(
            tuple(value[ids] for value, ids in zip(values, rows)), rows, noise_variance=0.1, size=3
        )
        return fit_coherent_partial(
            np.where(stats.observed_mask, stats.partial_kernel, np.nan),
            pair_weights=stats.pair_weight_sums,
            cliques=cliques,
        )

    first = fit(parent)
    changed = parent.copy()
    changed[0, 2], changed[1, 0], changed[2, 2] = -9e10, 7e10, 5e10
    second = fit(changed)
    np.testing.assert_array_equal(first.partial_kernel, second.partial_kernel)
    assert first.objective_gap_bound == second.objective_gap_bound
    assert first.converged and second.converged


def test_psd_consistent_input_still_does_not_make_later_hard_truncation_psd():
    kernel = np.array(
        [[1, 0, 1, np.nan], [0, 2, 0, 0], [1, 0, 1, 1], [np.nan, 0, 1, 1]], dtype=float
    )
    cliques = (np.arange(3), np.arange(1, 4))
    mask = observed_mask(cliques, 4)
    result = fit_coherent_partial(
        kernel, pair_weights=mask.astype(float), cliques=cliques, numerical_error_budget=1e-5
    )
    assert result.converged
    raw = complete_kernel(result.partial_kernel, cliques=cliques, rank=1)
    assert raw.available
    assert raw.diagnostics["output_minimum_eigenvalue"] < -0.4


def test_output_and_nested_diagnostics_are_immutable_snapshots():
    kernel, cliques, coverage = consensus_case()
    result = fit_coherent_partial(kernel, pair_weights=coverage, cliques=cliques)
    kernel[0, 0] = 99
    assert result.partial_kernel[0, 0] < 2
    with pytest.raises(ValueError):
        result.partial_kernel[0, 0] = 7
    with pytest.raises(ValueError):
        result.diagnostics["weights"][0] = 7
    with pytest.raises(TypeError):
        result.diagnostics["objective"] = 7
    with pytest.raises(FrozenInstanceError):
        result.partial_kernel = np.eye(3)


def test_penalty_balancing_changes_only_numerical_path_not_the_optimizer():
    kernel, cliques, coverage = consensus_case()
    adaptive = fit_coherent_partial(
        kernel, pair_weights=coverage, cliques=cliques, rho=0.01, adaptive_rho=True
    )
    fixed = fit_coherent_partial(
        kernel, pair_weights=coverage, cliques=cliques, rho=1.0, adaptive_rho=False
    )
    assert adaptive.converged and fixed.converged
    assert adaptive.diagnostics["rho_adjustments"] > 0
    assert fixed.diagnostics["rho_adjustments"] == 0
    assert weighted_distance(
        adaptive.partial_kernel, fixed.partial_kernel, coverage > 0, np.ones(3)
    ) <= (adaptive.partial_error_bound + fixed.partial_error_bound)


@pytest.mark.parametrize(
    "changes",
    [
        {"regularization": -0.1},
        {"regularization": True},
        {"eigenvalue_floor": -1},
        {"numerical_error_budget": 0},
        {"rho": 0},
        {"rho": np.nan},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"check_every": 0},
        {"absolute_tolerance": -1},
        {"relative_tolerance": -1},
        {"adaptive_rho": "yes"},
        {"weights": [1, 0, 1]},
    ],
)
def test_invalid_numerical_options_fail_explicitly(changes):
    kernel, cliques, coverage = consensus_case()
    with pytest.raises(ValueError):
        fit_coherent_partial(kernel, pair_weights=coverage, cliques=cliques, **changes)


def test_missing_or_zero_pair_coverage_is_never_imputed_and_hidden_values_are_rejected():
    kernel, cliques, coverage = consensus_case()
    for invalid in (np.zeros((3, 3)), np.ones((3, 3)), np.ones((2, 2))):
        with pytest.raises(ValueError):
            fit_coherent_partial(kernel, pair_weights=invalid, cliques=cliques)
    full = np.nan_to_num(kernel)
    with pytest.raises(ValueError, match="NaN"):
        fit_coherent_partial(full, pair_weights=coverage, cliques=cliques)


def test_reference_failure_is_not_mislabeled_success():
    pytest.importorskip("cvxpy")
    kernel, cliques, coverage = consensus_case()
    with pytest.warns(UserWarning):
        result = fit_coherent_partial_reference(
            kernel,
            pair_weights=coverage,
            cliques=cliques,
            max_iterations=1,
            numerical_error_budget=1e-9,
        )
    assert not result.converged
    assert result.partial_error_bound > 1e-9
