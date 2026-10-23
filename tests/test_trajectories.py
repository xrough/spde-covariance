import numpy as np
import pytest

from spde_covariance.spaces import SplineSpace
from spde_covariance.trajectories import (
    CurveTrajectory,
    InnovationBatch,
    extract_innovations,
    simulate_transport,
)
from spde_covariance.transport import ExponentialTransport, ProjectedSplineTransport


def _known_path(propagator, times, normalized_values, drift=None):
    coefficients = [np.linspace(0.2, 0.8, propagator.dimension)]
    drift = np.zeros_like(normalized_values) if drift is None else drift
    for index, dt in enumerate(np.diff(times)):
        coefficients.append(
            propagator.matrix(dt) @ coefficients[-1]
            + drift[index]
            + np.sqrt(dt) * normalized_values[index]
        )
    return CurveTrajectory(times, coefficients, propagator.basis_id, propagator.kind)


def test_exact_transport_and_known_integrated_drift_cancel_on_irregular_dates():
    propagator = ExponentialTransport(rates=[0.8, 1.6])
    times = np.array([0.1, 0.12, 0.25, 0.7])
    zeros = np.zeros((3, 2))
    drift = np.array([[0.1, 0.2], [-0.03, 0.1], [0.5, -0.4]])
    trajectory = _known_path(propagator, times, zeros, drift)
    batch = extract_innovations(trajectory, propagator, integrated_drift=drift)
    np.testing.assert_allclose(batch.values, 0, atol=1e-14)
    np.testing.assert_allclose(batch.interval_lengths, [0.02, 0.13, 0.45])
    assert batch.drift_convention == "supplied integrated mild drift"
    uncorrected = extract_innovations(trajectory, propagator)
    np.testing.assert_allclose(uncorrected.values, drift / np.sqrt(np.diff(times))[:, None])
    assert uncorrected.drift_convention == "zero integrated mild drift"


def test_irregular_increments_use_each_interval_normalization():
    propagator = ExponentialTransport(rates=[0.9, 1.8])
    times = np.array([0, 0.01, 0.09, 0.6])
    expected = np.array([[1.0, -0.4], [0.2, 0.8], [-0.5, 0.1]])
    trajectory = _known_path(propagator, times, expected)
    batch = extract_innovations(trajectory, propagator)
    np.testing.assert_allclose(batch.values, expected, atol=1e-14)
    np.testing.assert_array_equal(batch.start_times, times[:-1])
    np.testing.assert_array_equal(batch.end_times, times[1:])
    assert batch.n_intervals == 3
    assert batch.dimension == 2
    assert batch.metadata["pooling"] == "none; interval endpoints retained"


def test_extraction_and_observation_are_orthogonal_coordinate_invariant():
    base = ExponentialTransport(rates=[0.8, 1.3, 2.2])
    rotation, _ = np.linalg.qr(np.random.default_rng(22).normal(size=(3, 3)))

    class RotatedPropagator:
        dimension = base.dimension
        basis_id = base.basis_id + ":rotated"
        kind = base.kind

        def matrix(self, dt):
            return rotation.T @ base.matrix(dt) @ rotation

        def innovation_covariance(self, covariance, dt, quadrature_order=24):
            return (
                rotation.T
                @ base.innovation_covariance(
                    rotation @ covariance @ rotation.T, dt, quadrature_order=quadrature_order
                )
                @ rotation
            )

    rotated = RotatedPropagator()
    times = [0, 0.03, 0.08, 0.2]
    known = np.random.default_rng(33).normal(size=(3, 3))
    drift = np.random.default_rng(44).normal(size=(3, 3)) * 0.01
    trajectory = _known_path(base, times, known, drift)
    transformed = CurveTrajectory(
        times, trajectory.coefficients @ rotation, rotated.basis_id, rotated.kind
    )
    original = extract_innovations(trajectory, base, integrated_drift=drift)
    other = extract_innovations(transformed, rotated, integrated_drift=drift @ rotation)
    np.testing.assert_allclose(other.values, original.values @ rotation, atol=1e-14)
    operator = np.array([[1.0, -0.5, 0.3], [0.2, 0.7, -0.1]])
    first = original.observe(operator)
    second = other.observe(operator @ rotation)
    for one, two in zip(first, second, strict=True):
        np.testing.assert_allclose(one.y, two.y, atol=1e-14)


def test_simulation_seed_reproducibility_truth_separation_and_covariance_target():
    propagator = ExponentialTransport(rates=[0.8])
    times = np.array([0, 0.01, 0.1, 0.5])
    sources = np.array([[[0.3]], [[0.5]], [[0.7]]])
    initial = np.array([0.4])
    drift = np.array([[0.1], [0.2], [-0.1]])
    first = simulate_transport(
        propagator,
        times,
        sources,
        initial=initial,
        integrated_drift=drift,
        seed=151,
        curve_unit="EUR/MWh",
    )
    second = simulate_transport(
        propagator, times, sources, initial=initial, integrated_drift=drift, seed=151
    )
    np.testing.assert_array_equal(first.trajectory.coefficients, second.trajectory.coefficients)
    expected = sources[:, 0, 0] * (-np.expm1(-1.6 * np.diff(times))) / (1.6 * np.diff(times))
    np.testing.assert_allclose(first.innovation_covariances[:, 0, 0], expected, atol=1e-14)
    assert not hasattr(first.trajectory, "source_covariances")
    assert first.trajectory.curve_unit == "EUR/MWh"
    assert extract_innovations(first.trajectory, propagator).innovation_unit == "EUR/MWh/sqrt(year)"
    assert not hasattr(extract_innovations(first.trajectory, propagator), "source_covariances")
    initial[:] = -99
    times[:] = -99
    sources[:] = -99
    assert first.trajectory.coefficients[0, 0] == 0.4
    assert first.trajectory.times[0] == 0
    assert first.source_covariances[0, 0, 0] == 0.3
    for array in (
        first.trajectory.times,
        first.trajectory.coefficients,
        first.source_covariances,
        first.innovation_covariances,
    ):
        assert not array.flags.writeable


def test_monte_carlo_extracted_covariance_matches_exact_gaussian_oracle():
    propagator = ExponentialTransport(rates=[0.8, 1.5])
    source = np.array([[0.6, 0.15], [0.15, 0.4]])
    count, dt = 5000, 0.04
    result = simulate_transport(propagator, np.arange(count + 1) * dt, source, seed=723)
    batch = extract_innovations(result.trajectory, propagator)
    expected = propagator.innovation_covariance(source, dt)
    empirical = batch.values.T @ batch.values / count
    # Gaussian covariance-measurement variance: E||XX'-V||_F^2=tr(V)^2+||V||_F^2.
    root_mse = np.sqrt((np.trace(expected) ** 2 + np.sum(expected**2)) / count)
    assert np.linalg.norm(empirical - expected) < 4 * root_mse
    assert np.linalg.norm(np.mean(batch.values, axis=0)) < 4 * np.sqrt(np.trace(expected) / count)


def test_batch_observe_handles_ragged_maps_fresh_noise_and_seed():
    values = np.array([[0.2, -0.3], [0.8, 0.4], [-0.1, 0.6]])
    batch = InnovationBatch([0, 0.1, 0.3, 0.8], values, "test-basis", "known recursion")
    maps = [np.array([[1.0, -1.0]]), np.eye(2), np.array([[0.0, 1.0]])]
    noises = [np.array([[0.2]]), 0.1 * np.eye(2), np.array([[0.3]])]
    first = batch.observe(maps, noise_covariances=noises, seed=81)
    second = batch.observe(maps, noise_covariances=noises, seed=81)
    for index, (one, two) in enumerate(zip(first, second, strict=True)):
        np.testing.assert_array_equal(one.y, two.y)
        np.testing.assert_array_equal(one.noise_covariance, noises[index])
        assert one.time == batch.end_times[index]
    for index, observation in enumerate(batch.observe()):
        np.testing.assert_array_equal(observation.y, values[index])
        np.testing.assert_array_equal(observation.operator, np.eye(2))
        np.testing.assert_array_equal(observation.noise_covariance, np.zeros((2, 2)))


def test_fresh_observation_noise_correction_has_zero_monte_carlo_mean():
    count, variance = 4000, 0.3
    batch = InnovationBatch(
        np.arange(count + 1) / 52, np.zeros((count, 1)), "scalar-basis", "zero latent innovations"
    )
    observations = batch.observe(noise_covariances=[[variance]], seed=51)
    measurements = np.array([item.measurement[0, 0] for item in observations])
    standard_error = variance * np.sqrt(2 / count)
    assert abs(np.mean(measurements)) < 4 * standard_error


def test_snapshots_are_readonly_and_metadata_is_detached():
    times = np.array([0, 0.1, 0.2])
    values = np.ones((2, 2))
    metadata = {"nested": {"values": [1, 2]}}
    batch = InnovationBatch(times, values, "basis", "method", metadata=metadata)
    values[:] = 9
    times[:] = 9
    metadata["nested"]["values"].append(3)
    np.testing.assert_array_equal(batch.values, np.ones((2, 2)))
    assert batch.metadata["nested"]["values"] == (1, 2)
    for array in (
        batch.times,
        batch.values,
        batch.interval_lengths,
        batch.start_times,
        batch.end_times,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 2
    with pytest.raises(TypeError):
        batch.metadata["new"] = 3


def test_projected_spline_simulation_is_labelled_and_transport_cancels():
    propagator = ProjectedSplineTransport(SplineSpace(n_basis=4, length=5))
    result = simulate_transport(
        propagator, [0, 0.1, 0.3], np.zeros((4, 4)), initial=[0.1, -0.2, 0.3, 0.4], seed=1
    )
    np.testing.assert_allclose(
        extract_innovations(result.trajectory, propagator).values, 0, atol=1e-14
    )
    assert result.trajectory.propagation_method == "projected_mild_recursion"
    assert "not an exact infinite-dimensional" in result.metadata["simulation_model"]


@pytest.mark.parametrize("times", [[0], [0, 0], [1, 0], [0, np.nan], [-1e308, 1e308]])
def test_invalid_times_rejected(times):
    with pytest.raises(ValueError):
        CurveTrajectory(times, np.zeros((len(times), 1)), "basis", "method")


def test_extraction_rejects_wrong_basis_dimension_drift_and_nonfinite_values():
    propagator = ExponentialTransport(rates=[0.8, 1.6])
    trajectory = CurveTrajectory([0, 0.1], np.zeros((2, 2)), "other-basis", propagator.kind)
    with pytest.raises(ValueError, match="same basis"):
        extract_innovations(trajectory, propagator)
    trajectory = CurveTrajectory([0, 0.1], np.zeros((2, 1)), propagator.basis_id, propagator.kind)
    with pytest.raises(ValueError, match="same basis"):
        extract_innovations(trajectory, propagator)
    trajectory = CurveTrajectory([0, 0.1], np.zeros((2, 2)), propagator.basis_id, propagator.kind)
    with pytest.raises(ValueError, match="integrated_drift"):
        extract_innovations(trajectory, propagator, integrated_drift=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        InnovationBatch([0, 0.1], [[np.inf]], "basis", "method")
    with pytest.raises(ValueError, match="year"):
        InnovationBatch([0, 1], [[0]], "basis", "method", time_unit="day")
    with pytest.raises(ValueError, match="source_covariance"):
        simulate_transport(propagator, [0, 0.1], np.diag([1.0, -1.0]))
    with pytest.raises(ValueError, match="source_covariance"):
        simulate_transport(propagator, [0, 0.1], np.eye(3))


def test_observation_validation_rejects_wrong_rows_noise_and_dimension():
    batch = InnovationBatch([0, 0.1, 0.2], np.ones((2, 2)), "basis", "method")
    with pytest.raises(ValueError, match="operators"):
        batch.observe(np.ones((2, 3)))
    with pytest.raises(ValueError, match="noise covariance"):
        batch.observe(noise_covariances=np.eye(1))
    with pytest.raises(ValueError, match="one matrix per interval"):
        batch.observe([np.eye(2)])
