import numpy as np
import pytest

from spde_covariance.estimation import (
    estimate_risk,
    fit_covariance,
    fit_covariance_reference,
    uniform_trailing_weights,
)
from spde_covariance.geometry import gram_operator, svec
from spde_covariance.observations import Observation
from spde_covariance.simulation import simulate_innovations


def example_data():
    covariance = np.array([[0.8, 0.2, -0.1], [0.2, 0.6, 0.15], [-0.1, 0.15, 0.4]])
    rng = np.random.default_rng(37)
    operators = [rng.normal(size=(2 + i % 2, 3)) for i in range(12)]
    noises = [0.03 * np.eye(len(operator)) for operator in operators]
    return covariance, simulate_innovations(
        covariance, operators, noise_covariances=noises, seed=47
    ).observations


def test_constraints_certificate_and_reference_agreement():
    pytest.importorskip("cvxpy")
    _, data = example_data()
    fit = fit_covariance(
        data, regularization=0.15, trace_cap=0.7, absolute_tolerance=1e-11, relative_tolerance=0
    )
    reference = fit_covariance_reference(
        data, regularization=0.15, trace_cap=0.7, solver_tolerance=1e-10
    )
    assert fit.converged
    assert fit.trace <= 0.7 + 1e-12
    assert fit.min_eigenvalue >= -1e-12
    assert fit.objective_gap_bound <= 1e-11
    assert abs(fit.objective - reference.objective) <= 2e-8
    assert np.linalg.norm(fit.covariance - reference.covariance) < 2e-5
    # Each feasible solution's first-order bound controls suboptimality.
    assert fit.objective - reference.objective <= fit.objective_gap_bound + 1e-8
    assert reference.objective - fit.objective <= reference.objective_gap_bound + 1e-8


def test_indefinite_corrected_measurements_are_not_projected():
    data = [Observation(0.0, np.zeros(2), np.eye(2), np.diag([1.0, 2.0]))]
    assert np.linalg.eigvalsh(data[0].measurement)[0] < 0
    fit = fit_covariance(data, regularization=0.1, trace_cap=3)
    np.testing.assert_allclose(fit.covariance, 0, atol=1e-14)
    assert fit.objective == pytest.approx(2.5)
    assert fit.converged


def test_basic_inequality_and_numerical_remainder():
    truth, data = example_data()
    regularization = 0.05
    fit = fit_covariance(
        data,
        regularization=regularization,
        trace_cap=3,
        absolute_tolerance=1e-12,
        relative_tolerance=0,
    )
    weights = np.full(len(data), 1 / len(data))
    q = gram_operator([item.operator for item in data], weights)
    hessian = q + regularization * np.eye(len(q))
    noise = sum(
        weight
        * item.operator.T
        @ (item.measurement - item.operator @ truth @ item.operator.T)
        @ item.operator
        for weight, item in zip(weights, data, strict=True)
    )
    perturbation = svec(noise - regularization * truth)
    difference = svec(fit.covariance - truth)
    left = np.sqrt(difference @ hessian @ difference)
    right = np.sqrt(perturbation @ np.linalg.solve(hessian, perturbation))
    numerical_h_error = np.sqrt(fit.lipschitz) * fit.covariance_error_bound
    assert left <= right + numerical_h_error + 1e-10


def test_covariance_fit_is_orthogonal_coordinate_invariant():
    _, data = example_data()
    rotation, _ = np.linalg.qr(np.random.default_rng(98).normal(size=(3, 3)))
    rotated_data = [
        Observation(item.time, item.y, item.operator @ rotation, item.noise_covariance)
        for item in data
    ]
    parameters = dict(
        regularization=0.1, trace_cap=2, absolute_tolerance=1e-10, relative_tolerance=0
    )
    first = fit_covariance(data, **parameters)
    second = fit_covariance(rotated_data, **parameters)
    np.testing.assert_allclose(
        rotation @ second.covariance @ rotation.T, first.covariance, atol=1e-8
    )


def test_maximum_iterations_returns_explicit_failure_and_a_valid_bound():
    _, data = example_data()
    fit = fit_covariance(data, regularization=0.1, trace_cap=2, max_iterations=0)
    assert not fit.converged
    assert fit.status == "maximum_iterations"
    assert fit.iterations == 0
    assert fit.objective_gap_bound > 0
    assert fit.covariance_error_bound > 0


def test_acceleration_reaches_numerical_budget_on_an_ill_conditioned_problem():
    pytest.importorskip("cvxpy")
    operator = np.diag([1.0, 0.1, 0.01])
    truth = np.diag([0.4, 0.3, 0.2])
    # Three deterministic measurements have exactly the desired average outer product.
    samples = np.sqrt(3) * operator @ np.linalg.cholesky(truth)
    data = [
        Observation(float(index), samples[:, index], operator, np.zeros((3, 3)))
        for index in range(3)
    ]
    regularization, budget = 1e-4, 1e-3
    fit = fit_covariance(
        data,
        regularization=regularization,
        trace_cap=1,
        absolute_tolerance=1e-5,
        relative_tolerance=0,
        numerical_error_budget=budget,
        max_iterations=3000,
    )
    exact = np.diag(
        np.diag(operator) ** 4 / (np.diag(operator) ** 4 + regularization) * np.diag(truth)
    )
    reference = fit_covariance_reference(
        data, regularization=regularization, trace_cap=1, solver_tolerance=1e-10
    )
    assert fit.lipschitz / (fit.q_min + regularization) > 9000
    assert fit.converged
    assert fit.solver == "accelerated_projected_gradient"
    assert fit.covariance_error_bound <= budget
    assert np.linalg.norm(fit.covariance - exact) <= fit.covariance_error_bound + 1e-12
    assert np.linalg.norm(fit.covariance - reference.covariance) <= (
        fit.covariance_error_bound + reference.covariance_error_bound + 1e-9
    )
    # The numerical budget remains required even when objective tolerance is loose.
    unfinished = fit_covariance(
        data,
        regularization=regularization,
        trace_cap=1,
        absolute_tolerance=1e6,
        relative_tolerance=0,
        numerical_error_budget=budget,
        max_iterations=0,
    )
    assert not unfinished.converged
    assert unfinished.covariance_error_bound > budget


def test_direct_risk_rejects_unidentified_contemporaneous_covariance():
    data = [
        Observation(0.0, [1.0], [[1.0, 0.0]], [[0.0]]),
        Observation(1.0, [1.0], [[0.0, 1.0]], [[0.0]]),
    ]
    with pytest.raises(ValueError, match="not observable"):
        estimate_risk(data, [[0.0, 0.5], [0.5, 0.0]])
    # Observing both coordinates on different dates identifies their variances.
    assert estimate_risk(data, np.eye(2)).estimate == pytest.approx(2)


def test_observable_risk_unbiased_within_declared_monte_carlo_uncertainty():
    truth = np.array([[0.8, 0.25], [0.25, 0.5]])
    exposure = np.array([1.0, -1.0])
    target = np.outer(exposure, exposure)
    sample_size = 5000
    noise_variance = 0.12
    simulation = simulate_innovations(
        truth,
        [exposure[None, :]] * sample_size,
        noise_covariances=np.array([[noise_variance]]),
        seed=2204,
    )
    fit = estimate_risk(simulation.observations, target)
    expected = float(exposure @ truth @ exposure)
    mc_se = np.sqrt(2 / sample_size) * (expected + noise_variance)
    assert abs(fit.estimate - expected) < 4 * mc_se
    assert fit.observability_residual < 1e-12


def test_zero_weight_dates_and_trailing_window():
    weights = uniform_trailing_weights([0, 0.2, 0.4, 0.7], 0.31)
    np.testing.assert_array_equal(weights, [0, 0, 0.5, 0.5])
    data = [
        Observation(0.0, [1.0], [[1.0, 0.0]], [[0.0]]),
        Observation(1.0, [2.0], [[0.0, 1.0]], [[0.0]]),
    ]
    with pytest.raises(ValueError, match="not observable"):
        estimate_risk(data, np.diag([1.0, 0]), weights=[0, 1])
    assert estimate_risk(data, np.diag([0.0, 1]), weights=[0, 1]).estimate == pytest.approx(4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization": 0, "trace_cap": 1},
        {"regularization": 0.1, "trace_cap": -1},
        {"regularization": 0.1, "trace_cap": 1, "weights": [-1] * 12},
        {"regularization": 0.1, "trace_cap": 1, "numerical_error_budget": 0},
        {"regularization": 0.1, "trace_cap": 1, "numerical_error_budget": np.nan},
    ],
)
def test_invalid_parameters_fail(kwargs):
    _, data = example_data()
    with pytest.raises(ValueError):
        fit_covariance(data, **kwargs)
