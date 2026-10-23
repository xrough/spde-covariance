import numpy as np
import pytest

from spde_covariance.estimation import fit_covariance
from spde_covariance.observations import Observation
from spde_covariance.trajectories import InnovationBatch, extract_innovations, simulate_transport
from spde_covariance.transport import ExponentialTransport
from spde_covariance.workflow import (
    ConvergenceError,
    ObservedInnovationBatch,
    fit_innovation_covariance,
    interval_diagnostics,
    observe_innovations,
)


def _batch():
    model = ExponentialTransport(rates=[0.8, 1.8])
    simulation = simulate_transport(
        model,
        np.arange(49) / 52,
        np.array([[0.5, 0.1], [0.1, 0.3]]),
        curve_unit="USD/MWh",
        seed=273,
    )
    return extract_innovations(simulation.trajectory, model)


def _parameters():
    return dict(
        regularization=0.05,
        trace_cap=2,
        covariance_assumption="common",
        numerical_error_budget=1e-4,
    )


def test_end_to_end_fit_retains_target_units_basis_and_solver_certificate():
    batch = _batch()
    observed = observe_innovations(batch)
    fit = fit_innovation_covariance(
        observed,
        targets={"first_variance": np.diag([1.0, 0.0])},
        expected_basis_id=batch.basis_id,
        **_parameters(),
    )
    direct = fit_innovation_covariance(batch, **_parameters())
    np.testing.assert_allclose(fit.covariance, direct.covariance, atol=1e-12)
    np.testing.assert_array_equal(fit.times, batch.times)
    assert fit.basis_id == batch.basis_id
    assert fit.propagation_method == batch.propagation_method
    assert fit.interval_length == pytest.approx(1 / 52)
    assert fit.estimate.converged
    assert fit.estimate.covariance_error_bound <= 1e-4
    assert fit.diagnostics["targets"]["first_variance"]["identified"]
    assert "V_delta" in fit.target
    assert fit.covariance_assumption == "common"
    assert fit.metadata["instantaneous_covariance_recovery"] is False
    assert fit.curve_unit == "USD/MWh"
    assert fit.innovation_unit == "USD/MWh/sqrt(year)"
    assert fit.metadata["covariance_unit"] == "(USD/MWh)^2/year"
    assert not hasattr(observed, "values")
    assert not hasattr(observed, "source_covariances")
    assert not fit.covariance.flags.writeable
    assert not fit.estimate.weights.flags.writeable


def test_irregular_intervals_are_diagnosed_and_rejected_even_with_explicit_assumption():
    batch = InnovationBatch([0, 0.01, 0.05, 0.2], np.ones((3, 2)), "basis", "method")
    diagnostics = interval_diagnostics(batch)
    assert not diagnostics["equal_intervals"]
    assert diagnostics["delta_min"] == 0.01
    assert diagnostics["delta_max"] == pytest.approx(0.15)
    assert diagnostics["relative_interval_variation"] == pytest.approx(14)
    assert "interval-specific" in diagnostics["target"]
    for data in (batch, observe_innovations(batch)):
        for assumption in ("common", "local"):
            with pytest.raises(ValueError, match="irregular intervals"):
                fit_innovation_covariance(
                    data, **(_parameters() | {"covariance_assumption": assumption})
                )
    with pytest.raises(ValueError, match="irregular intervals"):
        fit_innovation_covariance(batch, weights=[0, 0, 1], **_parameters())


def test_tiny_irregular_intervals_do_not_pass_an_absolute_tolerance_gate():
    batch = InnovationBatch([0, 1e-14, 3e-14], [[0], [0]], "basis", "method")
    assert not interval_diagnostics(batch)["equal_intervals"]
    with pytest.raises(ValueError, match="irregular intervals"):
        fit_innovation_covariance(batch, **_parameters())


def test_indefinite_noise_corrected_measurements_remain_valid():
    observation = Observation(0.1, np.zeros(2), np.eye(2), np.eye(2))
    data = ObservedInnovationBatch([0, 0.1], (observation,), "declared-basis", "known extraction")
    assert np.linalg.eigvalsh(observation.measurement)[0] < 0
    fit = fit_innovation_covariance(data, **_parameters())
    np.testing.assert_array_equal(fit.covariance, np.zeros((2, 2)))
    assert fit.estimate.converged
    assert fit.estimate.objective == 1


def test_ragged_partial_observations_and_fresh_noise_preserve_provenance():
    batch = _batch()
    maps = [
        np.array([[1.0, -1.0]]) if index % 2 else np.eye(2) for index in range(batch.n_intervals)
    ]
    noises = [0.03 * np.eye(len(operator)) for operator in maps]
    observed = observe_innovations(batch, maps, noise_covariances=noises, seed=61)
    repeated = observe_innovations(batch, maps, noise_covariances=noises, seed=61)
    assert observed.dimension == 2
    assert observed.n_intervals == batch.n_intervals
    assert observed.curve_unit == batch.curve_unit
    assert observed.innovation_unit == batch.innovation_unit
    assert observed.metadata["drift_convention"] == batch.drift_convention
    for one, two in zip(observed.observations, repeated.observations, strict=True):
        np.testing.assert_array_equal(one.y, two.y)
    fit = fit_innovation_covariance(observed, **_parameters())
    assert fit.estimate.converged
    assert fit.diagnostics["rank"] == 3


def test_trailing_window_is_explicit_and_matches_the_low_level_fit():
    batch = InnovationBatch(
        np.arange(5) / 10, [[0.1, 0.2], [0.2, -0.3], [0.8, 0.4], [-0.5, 0.2]], "basis", "method"
    )
    observed = observe_innovations(batch)
    parameters = _parameters() | {"covariance_assumption": "local"}
    fit = fit_innovation_covariance(observed, trailing_window=0.11, **parameters)
    expected_weights = np.array([0, 0, 0.5, 0.5])
    np.testing.assert_array_equal(fit.estimate.weights, expected_weights)
    reference = fit_covariance(
        observed.observations,
        weights=expected_weights,
        regularization=parameters["regularization"],
        trace_cap=parameters["trace_cap"],
        numerical_error_budget=parameters["numerical_error_budget"],
    )
    np.testing.assert_allclose(fit.covariance, reference.covariance, atol=1e-12)
    assert "local approximation" in fit.target
    assert "not statistically verified" in fit.metadata["assumption_status"]
    assert fit.metadata["trailing_window_years"] == 0.11
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_innovation_covariance(observed, trailing_window=0.1, weights=np.ones(4), **parameters)


def test_convergence_failure_exposes_feasible_estimate_without_returning_a_success():
    batch = InnovationBatch([0, 0.1], [[2.0, 1.0]], "basis", "method")
    with pytest.raises(ConvergenceError, match="did not converge") as caught:
        fit_innovation_covariance(batch, max_iterations=0, **_parameters())
    estimate = caught.value.estimate
    assert not estimate.converged
    assert estimate.iterations == 0
    assert estimate.covariance_error_bound > 1e-4
    assert estimate.min_eigenvalue >= -1e-12
    assert estimate.trace <= 2


def test_observed_batch_rejects_time_dimension_unit_and_basis_mismatches():
    data = (Observation(0.1, [0], [[1, 0]], [[0]]), Observation(0.2, [0], [[0, 1]], [[0]]))
    with pytest.raises(ValueError, match="right endpoints"):
        ObservedInnovationBatch([0, 0.1, 0.3], data, "basis", "method")
    different_dimension = (data[0], Observation(0.2, [0], [[1]], [[0]]))
    with pytest.raises(ValueError, match="basis dimension"):
        ObservedInnovationBatch([0, 0.1, 0.2], different_dimension, "basis", "method")
    with pytest.raises(ValueError, match="one Observation"):
        ObservedInnovationBatch([0, 0.1, 0.2], data[:1], "basis", "method")
    with pytest.raises(ValueError, match="year"):
        ObservedInnovationBatch([0, 0.1, 0.2], data, "basis", "method", time_unit="day")
    observed = ObservedInnovationBatch([0, 0.1, 0.2], data, "basis", "method")
    with pytest.raises(ValueError, match="expected_basis_id"):
        fit_innovation_covariance(observed, expected_basis_id="different", **_parameters())


def test_metadata_and_interval_snapshots_are_detached_and_readonly():
    times = np.array([0, 0.1, 0.2])
    metadata = {"source": {"tags": ["curve extraction"]}}
    observations = (Observation(0.1, [1], [[1]], [[0]]), Observation(0.2, [-1], [[1]], [[0]]))
    observed = ObservedInnovationBatch(times, observations, "basis", "method", metadata=metadata)
    times[:] = 9
    metadata["source"]["tags"].append("mutated")
    assert observed.metadata["source"]["tags"] == ("curve extraction",)
    np.testing.assert_array_equal(observed.times, [0, 0.1, 0.2])
    for array in (
        observed.times,
        observed.interval_lengths,
        observed.start_times,
        observed.end_times,
    ):
        assert not array.flags.writeable
    with pytest.raises(TypeError):
        observed.metadata["other"] = 3


@pytest.mark.parametrize(
    "metadata",
    [
        {"basis_id": "other basis"},
        {"propagation_method": "other method"},
        {"time_unit": "day"},
        {"curve_unit": "other unit"},
        {"innovation_unit": "other unit/sqrt(year)"},
    ],
)
def test_reserved_metadata_cannot_contradict_batch_provenance(metadata):
    observations = (Observation(0.1, [1], [[1]], [[0]]),)
    with pytest.raises(ValueError, match="contradicts"):
        ObservedInnovationBatch([0, 0.1], observations, "basis", "method", metadata=metadata)


@pytest.mark.parametrize("budget", [None, 0, -1, np.nan, np.inf, True, 1j, [0.1]])
def test_numerical_budget_is_required_and_strictly_positive(budget):
    parameters = _parameters() | {"numerical_error_budget": budget}
    with pytest.raises(ValueError, match="numerical_error_budget"):
        fit_innovation_covariance(_batch(), **parameters)


def test_assumption_and_budget_are_required_keywords():
    with pytest.raises(TypeError, match="covariance_assumption"):
        fit_innovation_covariance(
            _batch(), regularization=0.1, trace_cap=2, numerical_error_budget=1e-3
        )
    with pytest.raises(TypeError, match="numerical_error_budget"):
        fit_innovation_covariance(
            _batch(), regularization=0.1, trace_cap=2, covariance_assumption="common"
        )
    with pytest.raises(ValueError, match="covariance_assumption"):
        fit_innovation_covariance(_batch(), **(_parameters() | {"covariance_assumption": "auto"}))
