"""Finite-source cone controls independent of kernel-completion estimators."""

import numpy as np
import pytest

from spde_covariance.admissibility import (
    _compressed_problem,
    _problem,
    covariance_integration_matrix,
    diagnose_constant_source_admissibility,
)
from spde_covariance.geometry import lifted_matrix, smat, svec
from spde_covariance.laguerre import LaguerreTransport
from spde_covariance.source_classes import (
    embed_source_covariance,
    evaluate_source_witness,
    projected_covariance_integration_matrix,
)


class DiagonalTransport:
    dimension = 2
    basis_id = "analytic-diagonal-two-mode"
    kind = "exact_transport"

    def innovation_covariance(self, source, dt, quadrature_order=24):
        rates = np.array([1.0, 2.0])
        sums = (rates[:, None] + rates[None, :]) * dt
        return source * (-np.expm1(-sums) / sums)


def test_psd_polarization_matches_independent_diagonal_transport_formula():
    delta = 0.7
    model = DiagonalTransport()
    integration = covariance_integration_matrix(model, delta)
    sums = np.array([2.0, 3.0, 4.0]) * delta
    expected = np.diag(-np.expm1(-sums) / sums)
    np.testing.assert_allclose(integration, expected, atol=2e-16)
    assert not integration.flags.writeable
    source = np.array([[1.0, 0.3], [0.3, 0.4]])
    np.testing.assert_allclose(
        smat(integration @ svec(source)), model.innovation_covariance(source, delta), atol=2e-16
    )


def test_feasible_control_and_source_witness_are_not_claimed_source_recovery():
    model, delta = DiagonalTransport(), 0.7
    source = np.array([[1.0, 0.3], [0.3, 0.4]])
    kernel = model.innovation_covariance(source, delta)
    result = diagnose_constant_source_admissibility(
        kernel,
        np.eye(2),
        delta=delta,
        trace_cap=2,
        propagator=model,
        objective_gap_tolerance=1e-12,
        admissibility_tolerance=1e-5,
    )
    assert result.converged and result.source_feasibility["feasible"]
    assert result.fit_distance_lower_bound == 0
    assert result.fit_distance_upper_bound < 1e-6
    assert result.admissibility_status == "compatible_with_declared_tolerance"
    assert result.metadata["infinite_dimensional_exclusion"] is False
    assert result.metadata["recovered_physical_source"] is False
    np.testing.assert_allclose(result.source_candidate, source, atol=2e-6)


def test_psd_rank_one_innovation_can_be_separated_from_diagonal_source_cone():
    model, delta = DiagonalTransport(), 1.0
    kernel = np.ones((2, 2))
    integration = covariance_integration_matrix(model, delta)
    inverse = smat(np.linalg.solve(integration, svec(kernel)))
    assert np.linalg.eigvalsh(inverse)[0] < -0.05
    result = diagnose_constant_source_admissibility(
        kernel,
        np.eye(2),
        delta=delta,
        trace_cap=10,
        integration_matrix=integration,
        objective_gap_tolerance=1e-12,
        admissibility_tolerance=1e-3,
    )
    assert result.converged
    assert result.fit_distance_lower_bound > 0.01
    assert result.fit_distance_upper_bound - result.fit_distance_lower_bound < 1e-8
    assert result.admissibility_status == "separated_from_declared_finite_class"


def test_actual_laguerre_two_mode_cone_obstruction_and_integration_error_scope():
    model = LaguerreTransport(2)
    kernel = np.diag([0.0, 1.0])
    integration = covariance_integration_matrix(model, 1.0)
    inverse = smat(np.linalg.solve(integration, svec(kernel)))
    assert np.linalg.eigvalsh(inverse)[0] < -0.07
    result = diagnose_constant_source_admissibility(
        kernel,
        np.eye(2),
        delta=1,
        trace_cap=10,
        propagator=model,
        objective_gap_tolerance=1e-11,
        admissibility_tolerance=0.01,
    )
    assert result.converged
    assert result.fit_distance_lower_bound > 0.08
    assert result.fit_distance_upper_bound < 0.082
    assert (
        "not interval arithmetic or an integration-error certificate"
        in result.metadata["certificate_scope"]
    )


@pytest.mark.parametrize("dimension", [4, 8, 16])
def test_laguerre_lift_reuses_existing_normalized_covariance_map(dimension):
    model, delta = LaguerreTransport(dimension), 1 / 52
    integration = covariance_integration_matrix(model, delta)
    rng = np.random.default_rng(617 + dimension)
    factor = rng.normal(size=(dimension, 3))
    source = factor @ factor.T / dimension
    expected = model.innovation_covariance(source, delta)
    np.testing.assert_allclose(smat(integration @ svec(source)), expected, atol=2e-15)
    evaluation = model.evaluate(np.linspace(0, 3, 13))
    result = diagnose_constant_source_admissibility(
        evaluation @ expected @ evaluation.T,
        evaluation,
        delta=delta,
        trace_cap=np.trace(source) + 1,
        integration_matrix=integration,
        initial_source=source,
        objective_gap_tolerance=1e-9,
    )
    assert result.converged and result.iterations == 0
    assert result.fit_distance_upper_bound < 2e-14


def test_non_self_adjoint_map_has_correct_weighted_adjoint_and_lipschitz():
    shift = np.array([[1.0, 0.8], [0.0, 0.4]])
    integration = lifted_matrix(shift)
    assert not np.allclose(integration, integration.T)
    evaluation = np.array([[1.0, 2.0], [0.1, 1.0], [-1.0, 0.3]])
    weights = np.array([0.1, 0.3, 0.6])
    kernel = np.array([[1.0, 0.3, -0.1], [0.3, 0.4, 0.2], [-0.1, 0.2, 0.5]])
    normal, rhs, weighted_evaluation, weighted_kernel = _problem(
        kernel,
        evaluation,
        weights,
        integration,
    )
    source = np.array([[0.7, 0.1], [0.1, 0.5]])
    direction = np.array([[0.2, -0.3], [-0.3, 0.1]])
    residual = (
        weighted_evaluation @ shift @ source @ shift.T @ weighted_evaluation.T - weighted_kernel
    )
    image = weighted_evaluation @ shift @ direction @ shift.T @ weighted_evaluation.T
    gradient = normal @ svec(source) - rhs
    assert gradient @ svec(direction) == pytest.approx(np.sum(residual * image), abs=2e-15)
    assert svec(direction) @ normal @ svec(direction) == pytest.approx(np.sum(image**2), abs=2e-15)
    assert np.linalg.eigvalsh(normal)[-1] * np.sum(direction**2) >= np.sum(image**2) - 1e-14


def test_orthogonal_source_coordinates_preserve_the_weighted_grid_problem():
    model, delta = DiagonalTransport(), 0.6
    integration = covariance_integration_matrix(model, delta)
    evaluation = np.array([[1.0, 0.3], [0.7, -1.0], [0.2, 1.0]])
    kernel = evaluation @ np.ones((2, 2)) @ evaluation.T
    weights = np.array([0.2, 0.5, 0.3])
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    coordinate_map = lifted_matrix(rotation)
    transformed_map = coordinate_map.T @ integration @ coordinate_map
    options = dict(delta=delta, trace_cap=10, weights=weights, objective_gap_tolerance=1e-11)
    result = diagnose_constant_source_admissibility(
        kernel,
        evaluation,
        integration_matrix=integration,
        **options,
    )
    rotated = diagnose_constant_source_admissibility(
        kernel,
        evaluation @ rotation,
        integration_matrix=transformed_map,
        **options,
    )
    assert result.converged and rotated.converged
    assert result.objective == pytest.approx(rotated.objective, abs=2e-13)
    np.testing.assert_allclose(
        result.projected_innovation_kernel, rotated.projected_innovation_kernel, atol=2e-7
    )


def test_caller_error_bounds_widen_only_the_reference_interval():
    options = dict(delta=1, trace_cap=0, integration_matrix=np.eye(3))
    result = diagnose_constant_source_admissibility(
        np.eye(2),
        np.eye(2),
        grid_error_bound=0.2,
        source_tail_bound=0.3,
        admissibility_tolerance=1.0,
        **options,
    )
    np.testing.assert_allclose(result.fit_distance_bounds, [np.sqrt(2)] * 2, atol=1e-15)
    np.testing.assert_allclose(
        result.reference_distance_bounds, [np.sqrt(2) - 0.5, np.sqrt(2) + 0.5]
    )
    assert result.admissibility_status == "unresolved_at_declared_tolerance"
    assert result.converged and result.status == "optimal_degenerate_class"


def test_unconverged_exit_still_bounds_independent_reference_distance():
    integration = covariance_integration_matrix(DiagonalTransport(), 1.0)
    result = diagnose_constant_source_admissibility(
        np.ones((2, 2)),
        np.eye(2),
        delta=1,
        trace_cap=10,
        integration_matrix=integration,
        max_iterations=0,
    )
    optimum = diagnose_constant_source_admissibility(
        np.ones((2, 2)),
        np.eye(2),
        delta=1,
        trace_cap=10,
        integration_matrix=integration,
        objective_gap_tolerance=1e-12,
    )
    assert not result.converged and result.status == "maximum_iterations"
    assert result.fit_distance_lower_bound <= optimum.fit_distance_upper_bound
    assert result.fit_distance_upper_bound >= optimum.fit_distance_lower_bound
    assert result.source_feasibility["feasible"]


def test_zero_evaluation_map_is_a_singleton_without_dividing_by_zero():
    result = diagnose_constant_source_admissibility(
        np.eye(2),
        np.zeros((2, 3)),
        delta=1,
        trace_cap=10,
        integration_matrix=np.eye(6),
    )
    assert result.converged and result.status == "optimal_degenerate_class"
    assert result.objective_gap_bound == 0
    np.testing.assert_allclose(result.fit_distance_bounds, [np.sqrt(2)] * 2)
    np.testing.assert_array_equal(result.source_candidate, np.zeros((3, 3)))


def test_indefinite_kernel_is_diagnosed_and_snapshots_do_not_change_inputs():
    kernel, evaluation, weights = np.diag([1.0, -0.2]), np.eye(2), np.array([0.3, 0.7])
    result = diagnose_constant_source_admissibility(
        kernel,
        evaluation,
        weights=weights,
        delta=1,
        trace_cap=2,
        integration_matrix=np.eye(3),
        objective_gap_tolerance=1e-12,
    )
    assert result.converged and result.fit_distance_lower_bound > 0.13
    assert np.linalg.eigvalsh(result.projected_innovation_kernel)[0] >= -1e-14
    kernel[:], evaluation[:], weights[:] = 10, 10, 10
    np.testing.assert_allclose(result.metadata["weights"], [0.3, 0.7])
    for array in (
        result.source_candidate,
        result.projected_innovation,
        result.projected_innovation_kernel,
        result.metadata["weights"],
    ):
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(TypeError):
        result.metadata["class"] = "infinite"


def test_weights_are_not_renormalized():
    options = dict(delta=1, trace_cap=0, integration_matrix=np.eye(3))
    first = diagnose_constant_source_admissibility(
        np.eye(2), np.eye(2), weights=[0.2, 0.8], **options
    )
    second = diagnose_constant_source_admissibility(
        np.eye(2), np.eye(2), weights=[0.6, 2.4], **options
    )
    assert second.fit_distance_upper_bound == pytest.approx(3 * first.fit_distance_upper_bound)
    assert second.objective == pytest.approx(9 * first.objective)


@pytest.mark.parametrize(
    "field,value",
    [
        ("delta", 0),
        ("delta", True),
        ("trace_cap", -1),
        ("trace_cap", np.inf),
        ("objective_gap_tolerance", -1),
        ("max_iterations", 1.5),
        ("quadrature_order", 0),
        ("grid_error_bound", -1),
        ("source_tail_bound", np.nan),
        ("admissibility_tolerance", -1),
        ("weights", [1.0, 0.0]),
        ("evaluation_matrix", np.ones((3, 2))),
        ("integration_matrix", np.eye(2)),
        ("initial_source", np.eye(3)),
        ("kernel", [[1.0, 1.0], [0.0, 1.0]]),
    ],
)
def test_invalid_inputs_raise_clear_errors(field, value):
    arguments = dict(
        kernel=np.eye(2),
        evaluation_matrix=np.eye(2),
        delta=1,
        trace_cap=1,
        integration_matrix=np.eye(3),
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        diagnose_constant_source_admissibility(**arguments)


def test_exactly_one_matching_integration_specification_is_required():
    options = dict(kernel=np.eye(2), evaluation_matrix=np.eye(2), delta=1, trace_cap=1)
    with pytest.raises(ValueError, match="exactly one"):
        diagnose_constant_source_admissibility(**options)
    with pytest.raises(ValueError, match="exactly one"):
        diagnose_constant_source_admissibility(
            **options,
            propagator=DiagonalTransport(),
            integration_matrix=np.eye(3),
        )
    with pytest.raises(ValueError, match="dimension"):
        diagnose_constant_source_admissibility(**options, propagator=LaguerreTransport(3))


def test_small_cvxpy_reference_is_inside_the_certified_distance_interval():
    cp = pytest.importorskip("cvxpy")
    model, delta = LaguerreTransport(2), 1.0
    integration = covariance_integration_matrix(model, delta)
    evaluation = np.array([[1.0, 0.2], [0.5, 1.0], [0.2, -0.3]])
    weights = np.array([0.2, 0.5, 0.3])
    kernel = evaluation @ np.diag([0.0, 1.0]) @ evaluation.T
    source = cp.Variable((2, 2), symmetric=True)
    coordinates = cp.hstack([source[0, 0], np.sqrt(2) * source[0, 1], source[1, 1]])
    innovation_coordinates = integration @ coordinates
    innovation = cp.bmat(
        [
            [innovation_coordinates[0], innovation_coordinates[1] / np.sqrt(2)],
            [innovation_coordinates[1] / np.sqrt(2), innovation_coordinates[2]],
        ]
    )
    root = np.sqrt(weights)
    residual = cp.multiply(
        root[:, None] * root[None, :], evaluation @ innovation @ evaluation.T - kernel
    )
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(residual)), [source >> 0, cp.trace(source) <= 10]
    )
    problem.solve(solver="SCS", eps=1e-10, max_iters=100_000)
    assert problem.status in ("optimal", "optimal_inaccurate")
    result = diagnose_constant_source_admissibility(
        kernel,
        evaluation,
        delta=delta,
        trace_cap=10,
        integration_matrix=integration,
        weights=weights,
        objective_gap_tolerance=1e-12,
    )
    assert result.converged
    assert result.objective_lower_bound - 2e-10 <= problem.value <= result.objective + 2e-10
    assert result.fit_distance_lower_bound - 2e-8 <= np.sqrt(2 * problem.value)
    assert np.sqrt(2 * problem.value) <= result.fit_distance_upper_bound + 2e-8


def test_larger_forcing_support_strictly_enlarges_a_matched_transport_source_class():
    vector = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)
    source = np.outer(vector, vector)
    larger_map = projected_covariance_integration_matrix(3, 2, 1.0)
    kernel = smat(larger_map @ svec(source))
    matched_map = projected_covariance_integration_matrix(2, 2, 1.0)
    inverse = smat(np.linalg.solve(matched_map, svec(kernel)))
    assert np.linalg.eigvalsh(inverse)[0] < -0.078
    matched = diagnose_constant_source_admissibility(
        kernel,
        np.eye(2),
        delta=1,
        trace_cap=2,
        integration_matrix=matched_map,
        objective_gap_tolerance=1e-11,
    )
    larger = diagnose_constant_source_admissibility(
        kernel,
        np.eye(2),
        delta=1,
        trace_cap=2,
        integration_matrix=larger_map,
        initial_source=source,
        objective_gap_tolerance=1e-11,
    )
    largest = diagnose_constant_source_admissibility(
        kernel,
        np.eye(2),
        delta=1,
        trace_cap=2,
        integration_matrix=projected_covariance_integration_matrix(4, 2, 1.0),
        initial_source=embed_source_covariance(source, 4),
        objective_gap_tolerance=1e-11,
    )
    assert matched.converged and matched.fit_distance_lower_bound > 0.039
    assert larger.converged and largest.converged
    assert larger.fit_distance_upper_bound < 1e-14
    assert largest.fit_distance_upper_bound < 1e-14
    assert largest.fit_distance_lower_bound <= larger.fit_distance_lower_bound
    assert larger.fit_distance_upper_bound <= matched.fit_distance_lower_bound
    assert larger.source_candidate.shape == (3, 3)
    assert larger.projected_innovation.shape == (2, 2)
    assert larger.source_dimension == 3 and larger.output_dimension == 2
    assert larger.metadata["source_cutoff_dimension"] == 3
    assert larger.metadata["output_dimension"] == 2


def test_rectangular_normal_factor_has_the_exact_dense_adjoint_and_lipschitz():
    rng = np.random.default_rng(741)
    integration = projected_covariance_integration_matrix(6, 3, 0.2)
    evaluation = rng.normal(size=(5, 3))
    kernel = rng.normal(size=(5, 5))
    kernel = (kernel + kernel.T) / 2
    weights = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
    hessian, rhs, _, _ = _problem(kernel, evaluation, weights, integration)
    factor, reduced_rhs, _, _, lipschitz = _compressed_problem(
        kernel,
        evaluation,
        weights,
        integration,
    )
    np.testing.assert_allclose(factor.T @ factor, hessian, atol=2e-14)
    np.testing.assert_allclose(reduced_rhs, rhs, atol=0)
    assert lipschitz == pytest.approx(np.linalg.eigvalsh(hessian)[-1], rel=2e-14)


def test_independent_source_and_output_rotations_preserve_the_rectangular_problem():
    rng = np.random.default_rng(962)
    source_rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    output_rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    integration = projected_covariance_integration_matrix(3, 2, 1.0)
    rotated = lifted_matrix(output_rotation.T) @ integration @ lifted_matrix(source_rotation)
    evaluation = np.array([[1.0, 0.3], [0.4, 1.0], [1.0, -1.0]])
    kernel = evaluation @ np.diag([1.0, -0.2]) @ evaluation.T
    options = dict(delta=1, trace_cap=1.5, weights=[0.2, 0.3, 0.5], objective_gap_tolerance=1e-11)
    first = diagnose_constant_source_admissibility(
        kernel, evaluation, integration_matrix=integration, **options
    )
    second = diagnose_constant_source_admissibility(
        kernel, evaluation @ output_rotation, integration_matrix=rotated, **options
    )
    assert first.converged and second.converged
    assert first.objective == pytest.approx(second.objective, abs=2e-12)
    np.testing.assert_allclose(
        first.projected_innovation_kernel, second.projected_innovation_kernel, atol=2e-6
    )


def test_embedded_witness_keeps_a_certified_nonincreasing_feasible_upper_envelope():
    target = np.diag([0.01, 1.01])
    current_source = np.zeros((2, 2))
    envelope = np.inf
    previous_source = None
    for dimension in (2, 3, 4):
        integration = projected_covariance_integration_matrix(dimension, 2, 1.0)
        if previous_source is not None:
            current_source = embed_source_covariance(previous_source, dimension)
        embedded = evaluate_source_witness(
            current_source, integration, np.eye(2), target_kernel=target, trace_cap=4
        )
        # Deliberately short solves retain useful intervals but need not be
        # optimized. The embedded source itself certifies the upper envelope.
        fit = diagnose_constant_source_admissibility(
            target,
            np.eye(2),
            delta=1,
            trace_cap=4,
            integration_matrix=integration,
            initial_source=current_source,
            max_iterations=10,
        )
        next_envelope = min(envelope, embedded["weighted_distance"], fit.fit_distance_upper_bound)
        assert next_envelope <= envelope + 1e-14
        assert fit.fit_distance_lower_bound <= next_envelope + 1e-14
        previous_source = (
            fit.source_candidate
            if fit.fit_distance_upper_bound <= embedded["weighted_distance"]
            else current_source
        )
        envelope = next_envelope


def test_large_source_small_output_uses_output_space_normal_factorization():
    source_dimension, output_dimension = 64, 16
    integration = projected_covariance_integration_matrix(
        source_dimension, output_dimension, 1 / 52
    )
    source = np.diag(1 / (np.arange(source_dimension) + 1) ** 3)
    kernel = smat(integration @ svec(source))
    fit = diagnose_constant_source_admissibility(
        kernel,
        np.eye(output_dimension),
        delta=1 / 52,
        trace_cap=2,
        integration_matrix=integration,
        initial_source=source,
    )
    assert fit.converged and fit.iterations == 0
    assert fit.metadata["normal_equations"] == "output-space factorization"
    assert fit.fit_distance_upper_bound < 1e-14


def test_rectangular_source_problem_matches_an_independent_cvxpy_solve():
    cp = pytest.importorskip("cvxpy")
    integration = projected_covariance_integration_matrix(3, 2, 1.0)
    kernel = np.diag([0.01, 1.01])
    source = cp.Variable((3, 3), symmetric=True)
    rows, columns = np.triu_indices(3)
    coordinates = cp.hstack(
        [source[a, b] * (1 if a == b else np.sqrt(2)) for a, b in zip(rows, columns, strict=True)]
    )
    innovation = integration @ coordinates
    target = svec(kernel)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(innovation - target)), [source >> 0, cp.trace(source) <= 4]
    )
    problem.solve(solver="SCS", eps=1e-10, max_iters=100_000)
    fit = diagnose_constant_source_admissibility(
        kernel,
        np.eye(2),
        delta=1,
        trace_cap=4,
        integration_matrix=integration,
        objective_gap_tolerance=1e-11,
    )
    assert problem.status in ("optimal", "optimal_inaccurate")
    assert fit.converged
    assert fit.objective_lower_bound - 2e-9 <= problem.value <= fit.objective + 2e-9


def test_independent_cvxpy_confirms_strict_projected_forcing_control():
    cp = pytest.importorskip("cvxpy")
    vector = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)
    kernel = smat(
        projected_covariance_integration_matrix(3, 2, 1.0) @ svec(np.outer(vector, vector))
    )
    values = []
    for dimension in (2, 3):
        source = cp.Variable((dimension, dimension), symmetric=True)
        rows, columns = np.triu_indices(dimension)
        coordinates = cp.hstack(
            [
                source[a, b] * (1 if a == b else np.sqrt(2))
                for a, b in zip(rows, columns, strict=True)
            ]
        )
        matrix = projected_covariance_integration_matrix(dimension, 2, 1.0)
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(matrix @ coordinates - svec(kernel))),
            [source >> 0, cp.trace(source) <= 2],
        )
        problem.solve(solver="SCS", eps=1e-10, max_iters=100_000)
        assert problem.status in ("optimal", "optimal_inaccurate")
        values.append(problem.value)
    assert values[0] > 0.0007
    assert values[1] < 1e-14
