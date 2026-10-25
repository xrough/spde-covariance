"""Identification ranges, primal witnesses and finite robust hedge controls."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from spde_covariance.completion import complete_kernel
from spde_covariance.hedging import (
    CovarianceAmbiguity,
    covariance_functional_interval,
    solve_covariance_functional,
    solve_plugin_hedge,
    solve_robust_hedge,
    solve_target_block_hedge,
    transport_covariance_factors,
)
from spde_covariance.transport import ExponentialTransport


def three_node_case():
    kernel = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 2] = mask[2, 0] = False
    model = CovarianceAmbiguity.from_covariance(kernel, mask, 3)
    return kernel, mask, model, np.eye(3)[0], np.eye(3)[:, [2]]


def require_cvxpy():
    return pytest.importorskip("cvxpy")


def test_analytic_interval_witnesses_and_existing_canonical_completion():
    require_cvxpy()
    kernel, mask, model, target, instruments = three_node_case()
    result = covariance_functional_interval(model, target, instruments[:, 0])
    assert result.available
    assert result.lower == pytest.approx(-0.5, abs=2e-6)
    assert result.upper == pytest.approx(1, abs=2e-6)
    assert not result.diagnostics["numerical_certificate"]
    assert not result.diagnostics["statistical_coverage_verified"]
    for witness in (result.minimizer, result.maximizer):
        np.testing.assert_allclose(witness.covariance[mask], kernel[mask], atol=2e-6)
        np.testing.assert_allclose(witness.source_covariance, witness.covariance, atol=1e-12)
        assert np.linalg.eigvalsh(witness.source_covariance)[0] >= -2e-6
        assert witness.value == pytest.approx(target @ witness.covariance @ instruments[:, 0])
        assert witness.diagnostics["feasibility"]["scaled_max_violation"] < 1e-6
        assert witness.diagnostics["objective_recomputation_error"] < 1e-8
    partial = kernel.copy()
    partial[~mask] = np.nan
    canonical = complete_kernel(partial, cliques=([0, 1], [1, 2]), population=True)
    assert canonical.available
    assert canonical.kernel[0, 2] == pytest.approx(0.25)
    point = solve_plugin_hedge(canonical.kernel, target, instruments)
    assert point.available
    assert point.positions[0] == pytest.approx(0.25)
    assert point.risk == pytest.approx(0.9375)


def test_robust_hedge_matches_independent_endpoint_minimax():
    cp = require_cvxpy()
    _, _, model, target, instruments = three_node_case()
    robust = solve_robust_hedge(model, target, instruments)
    # Independent scalar epigraph, using analytic PSD completion endpoints.
    h, risk = cp.Variable(), cp.Variable()
    problem = cp.Problem(
        cp.Minimize(risk),
        [
            cp.square(h) + h + 1 <= risk,
            cp.square(h) - 2 * h + 1 <= risk,
        ],
    )
    problem.solve(solver="CLARABEL")
    assert robust.available
    assert robust.positions[0] == pytest.approx(h.value, abs=2e-5)
    assert robust.positions[0] == pytest.approx(0, abs=2e-6)
    assert robust.risk == pytest.approx(problem.value, abs=2e-6)
    assert robust.risk == pytest.approx(1, abs=2e-6)
    assert robust.diagnostics["support_available"]
    assert abs(robust.diagnostics["support_crosscheck_gap"]) < 1e-5
    assert not robust.diagnostics["numerical_certificate"]


def test_scaled_factor_observations_agree_with_independent_primal_sdp():
    cp = require_cvxpy()
    factors = (
        np.array([[1, 0], [0.3, 0.4], [0, 0.7]]) / np.sqrt(2),
        np.array([[0.6, 0.2], [0.1, 1], [0.2, -0.3]]) / np.sqrt(2),
    )
    source_truth = np.array([[0.7, 0.1], [0.1, 0.5]])
    kernel = sum(factor @ source_truth @ factor.T for factor in factors)
    mask = np.eye(3, dtype=bool)
    model = CovarianceAmbiguity.from_covariance(
        kernel, mask, 2, factors=factors, entry_tolerance=0.03
    )
    left, right = np.array([1, -0.4, 0.3]), np.array([-0.2, 0.7, 0.1])
    result = covariance_functional_interval(model, left, right)
    source = cp.Variable((2, 2), symmetric=True)
    image = sum(factor @ source @ factor.T for factor in factors)
    constraints = [source >> 0, cp.trace(source) <= 2]
    constraints.extend([cp.diag(image) >= np.diag(kernel) - 0.03])
    constraints.extend([cp.diag(image) <= np.diag(kernel) + 0.03])
    objective = left @ image @ right
    lower = cp.Problem(cp.Minimize(objective), constraints)
    upper = cp.Problem(cp.Maximize(objective), constraints)
    lower.solve(solver="CLARABEL")
    upper.solve(solver="CLARABEL")
    assert result.available
    assert result.lower == pytest.approx(lower.value, abs=2e-6)
    assert result.upper == pytest.approx(upper.value, abs=2e-6)
    for matrix, (row, col) in zip(model.entry_matrices, model.entry_pairs, strict=True):
        assert np.trace(matrix @ source_truth) == pytest.approx(kernel[row, col])


def test_no_observations_robust_hedge_matches_spectral_norm_formulation():
    cp = require_cvxpy()
    factors = (
        np.array([[1, 0], [0.3, 0.4], [0, 0.7]]),
        np.array([[0.6, 0.2], [0.1, 1], [0.2, -0.3]]),
    )
    model = CovarianceAmbiguity(factors, [], [], [], 1.5)
    target, instruments = np.array([1, -0.4, 0.3]), np.eye(3)[:, [1, 2]]
    result = solve_robust_hedge(model, target, instruments, solver="CLARABEL")
    positions = cp.Variable(2)
    residual = target - instruments @ positions
    image = cp.hstack([cp.reshape(f.T @ residual, (2, 1), order="C") for f in factors])
    reference = cp.Problem(cp.Minimize(1.5 * cp.square(cp.norm(image, 2))))
    reference.solve(solver="CLARABEL")
    assert result.available
    np.testing.assert_allclose(result.positions, positions.value, atol=2e-4)
    assert result.risk == pytest.approx(reference.value, abs=2e-6)


def test_infeasible_ambiguity_is_explicit_and_never_a_robust_solution():
    require_cvxpy()
    model = CovarianceAmbiguity((np.eye(2),), [[0, 0]], [-1], [-1], 2)
    result = covariance_functional_interval(model, [1, 0])
    assert not result.available
    assert result.lower is result.upper is None
    assert result.minimizer.diagnostics["status"] == "infeasible"
    robust = solve_robust_hedge(model, [1, 0], [[0], [1]])
    assert not robust.available and robust.positions is None
    assert robust.diagnostics["ambiguity_feasibility_status"] == "infeasible"


def test_inaccurate_status_retains_candidate_but_is_not_accepted(monkeypatch):
    cp = require_cvxpy()
    original = cp.Problem.solve

    def inaccurate(problem, **options):
        result = original(problem, **options)
        problem._status = cp.OPTIMAL_INACCURATE
        return result

    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    _, _, model, target, _ = three_node_case()
    result = solve_covariance_functional(model, target)
    assert result.value is not None and result.source_covariance is not None
    assert result.diagnostics["status"] == "optimal_inaccurate"
    assert not result.available
    interval = covariance_functional_interval(model, target)
    assert not interval.available and interval.lower is interval.upper is None


def test_solver_errors_are_explicit_unavailable_results(monkeypatch):
    cp = require_cvxpy()

    def fail(problem, **options):
        raise cp.error.SolverError("deliberate solver failure")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    _, _, model, target, instruments = three_node_case()
    result = solve_covariance_functional(model, target)
    assert not result.available and result.value is None
    assert result.diagnostics["status"] == "solver_error"
    assert "deliberate solver failure" in result.diagnostics["error"]
    robust = solve_robust_hedge(model, target, instruments)
    assert not robust.available
    assert robust.diagnostics["ambiguity_feasibility_status"] == "solver_error"


def test_unobserved_truth_isolation_and_immutable_snapshots():
    kernel, mask, model, _, _ = three_node_case()
    changed = kernel.copy()
    changed[0, 2], changed[2, 0] = np.nan, 1e50
    isolated = CovarianceAmbiguity.from_covariance(changed, mask, 3)
    np.testing.assert_array_equal(model.lower, isolated.lower)
    np.testing.assert_array_equal(model.upper, isolated.upper)
    kernel[0, 0] = 99
    mask[0, 0] = False
    assert model.lower[0] == 1
    with pytest.raises(FrozenInstanceError):
        model.trace_cap = 100
    for array in (model.lower, model.entry_pairs, *model.factors):
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_source_orthogonal_coordinate_change_preserves_ranges_and_hedges():
    require_cvxpy()
    _, _, model, target, instruments = three_node_case()
    angle = 0.63
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    transformed = CovarianceAmbiguity(
        (rotation,), model.entry_pairs, model.lower, model.upper, model.trace_cap
    )
    original_range = covariance_functional_interval(model, target, instruments[:, 0])
    rotated_range = covariance_functional_interval(transformed, target, instruments[:, 0])
    assert original_range.available and rotated_range.available
    assert rotated_range.lower == pytest.approx(original_range.lower, abs=2e-6)
    assert rotated_range.upper == pytest.approx(original_range.upper, abs=2e-6)
    original = solve_robust_hedge(model, target, instruments)
    rotated = solve_robust_hedge(transformed, target, instruments)
    assert original.available and rotated.available
    np.testing.assert_allclose(original.positions, rotated.positions, atol=2e-6)
    assert original.risk == pytest.approx(rotated.risk, abs=2e-6)


def test_output_orthogonal_coordinates_preserve_a_jointly_observed_problem():
    require_cvxpy()
    kernel, _, _, target, instruments = three_node_case()
    rotation, _ = np.linalg.qr(np.array([[1, 2, 0.5], [-0.3, 1, 1], [2, -1, 0.5]]))
    transformed_kernel = rotation @ kernel @ rotation.T
    transformed_target, transformed_instruments = rotation @ target, rotation @ instruments
    full_mask = np.ones((3, 3), dtype=bool)
    model = CovarianceAmbiguity.from_covariance(kernel, full_mask, 3)
    transformed = CovarianceAmbiguity.from_covariance(transformed_kernel, full_mask, 3)
    original = solve_robust_hedge(model, target, instruments, tolerance=1e-9)
    rotated = solve_robust_hedge(
        transformed, transformed_target, transformed_instruments, tolerance=1e-9
    )
    assert original.available and rotated.available
    np.testing.assert_allclose(original.positions, rotated.positions, atol=2e-5)
    assert original.risk == pytest.approx(rotated.risk, abs=2e-6)
    point = solve_plugin_hedge(kernel, target, instruments)
    rotated_point = solve_plugin_hedge(
        transformed_kernel, transformed_target, transformed_instruments
    )
    np.testing.assert_allclose(point.positions, rotated_point.positions, atol=1e-12)
    assert point.risk == pytest.approx(rotated_point.risk, abs=1e-12)


def test_normalized_transport_factors_match_analytic_innovation_and_evaluation():
    propagator = ExponentialTransport([0.6, 1.2, 2.5])
    source = np.array([[0.7, 0.1, 0], [0.1, 0.4, 0.03], [0, 0.03, 0.3]])
    evaluation = propagator.evaluate([0, 0.7, 2, 3])
    delta = 0.12
    factors = transport_covariance_factors(propagator, delta, evaluation=evaluation)
    image = sum(factor @ source @ factor.T for factor in factors)
    expected = evaluation @ propagator.innovation_covariance(source, delta) @ evaluation.T
    np.testing.assert_allclose(image, expected, rtol=2e-13, atol=2e-13)
    # A second division by delta would fail this normalization check.
    near_zero = transport_covariance_factors(propagator, 1e-10)
    np.testing.assert_allclose(sum(f @ source @ f.T for f in near_zero), source, atol=1e-8)


def test_known_psd_background_is_present_in_constraints_risk_and_robust_objective():
    require_cvxpy()
    background = np.array([[0.4, 0.1], [0.1, 0.2]])
    finite = np.array([[0.6, 0.2], [0.2, 0.8]])
    total = background + finite
    model = CovarianceAmbiguity.from_covariance(
        total, np.ones((2, 2), dtype=bool), 2, offset_covariance=background
    )
    np.testing.assert_allclose(model.covariance(finite), total)
    assert model.residuals(finite)["scaled_max_violation"] < 1e-12
    target, instruments = np.eye(2)[:, 0], np.eye(2)[:, [1]]
    point = solve_plugin_hedge(total, target, instruments)
    robust = solve_robust_hedge(model, target, instruments, solver="CLARABEL", tolerance=1e-9)
    interval = covariance_functional_interval(model, target)
    assert interval.available and robust.available
    assert interval.lower == pytest.approx(1, abs=2e-6)
    assert interval.upper == pytest.approx(1, abs=2e-6)
    np.testing.assert_allclose(robust.positions, point.positions, atol=2e-5)
    assert robust.risk == pytest.approx(point.risk, abs=2e-6)


def test_duplicate_and_ill_conditioned_equality_rows_are_whitened():
    require_cvxpy()
    # Near-coincident evaluation rows make raw entry constraints ill conditioned.
    factor = np.array([[1, 0], [1, 1e-5], [0, 1]])
    source = np.array([[0.6, 0.2], [0.2, 0.9]])
    kernel = factor @ source @ factor.T
    model = CovarianceAmbiguity.from_covariance(
        kernel, np.ones((3, 3), dtype=bool), 2, factors=(factor,)
    )
    result = covariance_functional_interval(model, [1, -0.5, 0.3], solver="CLARABEL")
    assert result.available
    expected = np.array([1, -0.5, 0.3]) @ kernel @ np.array([1, -0.5, 0.3])
    assert result.lower == pytest.approx(expected, abs=2e-6)
    assert result.upper == pytest.approx(expected, abs=2e-6)
    assert result.minimizer.diagnostics["constraint_conditioning"]["equality_rank"] == 3
    assert result.minimizer.diagnostics["constraint_conditioning"]["equality_row_count"] == 6
    # Contradictory dependent observations are still infeasible after whitening.
    impossible = kernel.copy()
    impossible[0, 1] += 0.1
    impossible[1, 0] += 0.1
    invalid = CovarianceAmbiguity.from_covariance(
        impossible, np.ones((3, 3), dtype=bool), 2, factors=(factor,)
    )
    infeasible = covariance_functional_interval(invalid, [1, 0, 0])
    assert not infeasible.available
    assert infeasible.minimizer.diagnostics["status"] == "infeasible"


def test_plugin_bounds_singular_covariance_and_direct_joint_block():
    covariance = np.array([[1, 0.5], [0.5, 2]])
    target, instruments = np.eye(2)[:, 0], np.eye(2)[:, [1]]
    point = solve_plugin_hedge(covariance, target, instruments, risk_weight=3)
    direct = solve_target_block_hedge(1, [0.5], [[2]], risk_weight=3)
    assert point.available and direct.available
    np.testing.assert_allclose(point.positions, [0.25])
    np.testing.assert_allclose(point.positions, direct.positions)
    assert point.risk == pytest.approx(0.875)
    assert point.objective == pytest.approx(3 * 0.875)
    bounded = solve_plugin_hedge(covariance, target, instruments, position_bounds=([-1], [0.1]))
    assert bounded.available and bounded.positions[0] == pytest.approx(0.1, abs=2e-6)
    fixed = solve_plugin_hedge(covariance, target, instruments, position_bounds=([0.1], [0.1]))
    assert fixed.available and fixed.positions[0] == 0.1
    singular = solve_plugin_hedge(np.ones((2, 2)), target, np.ones((2, 2)))
    assert singular.available and singular.risk == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(singular.positions, [0.25, 0.25])


def test_weighted_risk_with_convex_holding_and_turnover_penalties():
    require_cvxpy()
    model = CovarianceAmbiguity((np.eye(1),), [[0, 0]], [1], [1], 1)
    options = {
        "risk_weight": 2,
        "holding_costs": [0.2],
        "turnover_costs": [0.2],
        "initial_positions": [0.5],
        "position_bounds": ([0], [2]),
    }
    point = solve_plugin_hedge([[1]], [1], [[1]], **options)
    robust = solve_robust_hedge(model, [1], [[1]], **options)
    assert point.available and robust.available
    assert point.positions[0] == pytest.approx(0.9, abs=2e-5)
    assert robust.positions[0] == pytest.approx(0.9, abs=2e-5)
    assert point.risk == pytest.approx(0.01, abs=2e-6)
    assert point.objective == pytest.approx(0.28, abs=2e-6)
    assert robust.objective == pytest.approx(point.objective, abs=2e-6)


def test_trace_lower_and_known_trace_are_explicit_source_assumptions():
    require_cvxpy()
    bounded = CovarianceAmbiguity((np.eye(1),), [], [], [], 1.5, trace_lower=0.8)
    known = CovarianceAmbiguity.from_covariance([[np.nan]], [[False]], 1, trace_lower=1)
    interval = covariance_functional_interval(bounded, [1])
    exact = covariance_functional_interval(known, [1])
    assert interval.available and exact.available
    assert interval.lower == pytest.approx(0.8, abs=2e-6)
    assert interval.upper == pytest.approx(1.5, abs=2e-6)
    assert exact.lower == pytest.approx(1, abs=2e-6)
    assert exact.upper == pytest.approx(1, abs=2e-6)
    assert bounded.residuals([[0.3]])["trace_lower_violation"] == pytest.approx(0.5)
    options = {"position_bounds": ([0], [0.5])}
    robust_bound = solve_robust_hedge(bounded, [1], [[1]], **options)
    robust_known = solve_robust_hedge(known, [1], [[1]], **options)
    assert robust_bound.available and robust_known.available
    assert robust_bound.positions[0] == pytest.approx(0.5, abs=2e-6)
    assert robust_bound.risk == pytest.approx(0.375, abs=2e-6)
    assert robust_known.risk == pytest.approx(0.25, abs=2e-6)
    with pytest.raises(ValueError, match="trace_lower"):
        CovarianceAmbiguity((np.eye(1),), [], [], [], 1, trace_lower=1.1)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"trace_cap": -1}, "trace_cap"),
        ({"trace_cap": True}, "trace_cap"),
        ({"entry_pairs": [[0, 0], [0, 0]], "lower": [0, 0], "upper": [1, 1]}, "repeat"),
        ({"entry_pairs": [[0, 2]]}, "index"),
        ({"entry_pairs": [[0.0, 0.0]]}, "integer"),
        ({"lower": [2]}, "lower <= upper"),
        ({"upper": [np.inf]}, "finite"),
        ({"factors": [np.eye(2), np.eye(3)]}, "share"),
        ({"factors": []}, "nonempty"),
        ({"calibrated_bands": "yes"}, "boolean"),
    ],
)
def test_ambiguity_input_validation(kwargs, match):
    inputs = {
        "factors": (np.eye(2),),
        "entry_pairs": [[0, 0]],
        "lower": [0],
        "upper": [1],
        "trace_cap": 2,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CovarianceAmbiguity(**inputs)


def test_observations_and_hedge_input_validation():
    kernel, mask, model, target, instruments = three_node_case()
    with pytest.raises(ValueError, match="boolean"):
        CovarianceAmbiguity.from_covariance(kernel, mask.astype(int), 3)
    with pytest.raises(ValueError, match="symmetric"):
        CovarianceAmbiguity.from_covariance(kernel, np.triu(mask), 3)
    with pytest.raises(ValueError, match="nonnegative"):
        CovarianceAmbiguity.from_covariance(kernel, mask, 3, entry_tolerance=-1)
    kernel[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CovarianceAmbiguity.from_covariance(kernel, mask, 3)
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_plugin_hedge([[1, 2], [2, 1]], [1, 0], [[0], [1]])
    with pytest.raises(ValueError, match="joint target"):
        solve_target_block_hedge(1, [2], [[1]])
    with pytest.raises(ValueError, match="instruments"):
        solve_robust_hedge(model, target, instruments[:, 0])
    with pytest.raises(ValueError, match="ordered"):
        solve_robust_hedge(model, target, instruments, position_bounds=([1], [0]))
    with pytest.raises(ValueError, match="holding_costs"):
        solve_robust_hedge(model, target, instruments, holding_costs=[-1])
    with pytest.raises(ValueError, match="positive"):
        solve_robust_hedge(model, target, instruments, risk_weight=0)
    with pytest.raises(ValueError, match="CLARABEL"):
        solve_covariance_functional(model, target, solver="OSQP")
    with pytest.raises(ValueError, match="source dimension"):
        model.covariance(np.eye(2))
    with pytest.raises(ValueError, match="one entry"):
        solve_covariance_functional(model, [1, 0])
