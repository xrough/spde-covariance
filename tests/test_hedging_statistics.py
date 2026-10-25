import numpy as np
import pytest
from scipy.stats import chi2

from spde_covariance.hedging_statistics import gaussian_pairwise_bands


def test_exact_diagonal_noise_correction_and_chi_square_quantiles():
    y = np.array([[1.0], [2.0], [-1.0], [0.0]])
    fit = gaussian_pairwise_bands(y, noise_covariance=[[0.2]], confidence_level=0.9)
    np.testing.assert_allclose(fit.lower, [6 / chi2.ppf(0.95, 4) - 0.2])
    np.testing.assert_allclose(fit.upper, [6 / chi2.ppf(0.05, 4) - 0.2])
    assert fit.corrected_covariance[0, 0] == 1.3


def test_polarization_and_missing_pairs_use_only_joint_observations():
    y = np.array([[1.0, 2.0, np.nan], [-1.0, 1.0, np.nan], [np.nan, 3.0, -2.0]])
    noise = np.eye(3) * 0.1
    fit = gaussian_pairwise_bands(y, noise_covariance=noise)
    assert (0, 2) not in map(tuple, fit.entry_pairs)
    assert fit.pair_counts[0, 2] == 0
    assert fit.corrected_covariance[0, 1] == 0.5
    index = list(map(tuple, fit.entry_pairs)).index((0, 1))
    tail = fit.diagnostics["one_sided_tail_probability"]
    plus = np.array([9 / chi2.ppf(1 - tail, 2), 9 / chi2.ppf(tail, 2)]) - 0.2
    minus = np.array([5 / chi2.ppf(1 - tail, 2), 5 / chi2.ppf(tail, 2)]) - 0.2
    np.testing.assert_allclose(fit.lower[index], (plus[0] - minus[1]) / 4)
    np.testing.assert_allclose(fit.upper[index], (plus[1] - minus[0]) / 4)


def test_indefinite_corrected_products_and_bounds_are_not_projected():
    fit = gaussian_pairwise_bands(np.zeros((5, 2)), noise_covariance=np.eye(2))
    assert fit.corrected_covariance[0, 0] == -1
    assert fit.upper[0] == -1


@pytest.mark.parametrize("data", [[[np.inf]], [[np.nan]], [[1 + 1j]], []])
def test_invalid_samples(data):
    with pytest.raises(ValueError):
        gaussian_pairwise_bands(data, noise_covariance=[[0.0]])


def test_no_write_or_mutable_input_alias():
    y = np.arange(6.0).reshape(3, 2)
    fit = gaussian_pairwise_bands(y, noise_covariance=np.eye(2))
    previous = fit.lower.copy()
    y[:] = 100
    np.testing.assert_array_equal(fit.lower, previous)
    with pytest.raises(ValueError):
        fit.lower.setflags(write=True)
