import numpy as np

from spde_covariance.simulation import simulate_innovations


def test_ragged_simulation_separates_latent_truth_and_is_reproducible():
    covariance = np.array([[0.8, 0.3], [0.3, 0.5]])
    operators = [np.eye(2), np.array([[1, -1]])]
    noise = [0.1 * np.eye(2), np.array([[0.2]])]
    first = simulate_innovations(covariance, operators, noise_covariances=noise, seed=34)
    second = simulate_innovations(covariance, operators, noise_covariances=noise, seed=34)
    assert first.latent_innovations.shape == (2, 2)
    assert len(first.observations[0].y) == 2
    assert len(first.observations[1].y) == 1
    np.testing.assert_array_equal(first.latent_innovations, second.latent_innovations)
    np.testing.assert_array_equal(first.observations[0].y, second.observations[0].y)
    assert not hasattr(first.observations[0], "latent_covariances")
    np.testing.assert_array_equal(first.observations[1].noise_covariance, noise[1])


def test_varying_latent_covariance_and_noiseless_measurements():
    covariances = np.array([[[1.0, 0.0], [0.0, 0.0]], np.zeros((2, 2))])
    simulation = simulate_innovations(covariances, [np.eye(2)] * 2, seed=5)
    np.testing.assert_array_equal(simulation.observations[1].y, [0, 0])
    np.testing.assert_array_equal(simulation.observations[0].y, simulation.latent_innovations[0])
