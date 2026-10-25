"""Simultaneous observed-entry bands under a declared Gaussian experiment.

These bands require independent, centered Gaussian rows with one common
covariance and a fixed, known additive noise covariance. Deterministic missing
entries are represented by NaN. They do not cover dependent futures returns,
unknown means/noise, data-dependent missingness or a changing covariance.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .hedging import _snapshot
from .linalg import validate_psd


@dataclass(frozen=True)
class GaussianCovarianceBands:
    entry_pairs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    corrected_covariance: np.ndarray
    pair_counts: np.ndarray
    confidence_level: float
    diagnostics: dict

    def __post_init__(self):
        for name in (
            "entry_pairs",
            "lower",
            "upper",
            "corrected_covariance",
            "pair_counts",
            "diagnostics",
        ):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))


def gaussian_pairwise_bands(values, *, noise_covariance, confidence_level=0.95):
    """Bonferroni chi-square bands, using polarization for cross covariances.

    Means are specified as zero, not estimated. For each observed pair use
    Var(Y_j +/- Y_k), subtract the corresponding known noise variance and
    polarize. Union bounding all scalar variance intervals supplies coverage
    at least confidence_level under the stated experiment. Finite-sample
    corrected moments, and lower bounds, may be negative. No PSD repair or
    latent-truth input is used. Missing pairs receive no finite constraints.
    """
    raw = np.asarray(values)
    if raw.ndim != 2 or not all(raw.shape) or raw.dtype.kind not in "iuf":
        raise ValueError("values must be a nonempty real sample-by-coordinate matrix")
    data = np.asarray(raw, dtype=float)
    if np.any(np.isinf(data)):
        raise ValueError("only NaN may mark missing values")
    if (
        isinstance(confidence_level, (bool, np.bool_))
        or not np.isscalar(confidence_level)
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must lie strictly between zero and one")
    noise = validate_psd(noise_covariance, name="noise_covariance")
    if noise.shape != (data.shape[1], data.shape[1]):
        raise ValueError("noise_covariance must match the observed coordinates")
    valid = np.isfinite(data)
    counts = valid.astype(np.int64).T @ valid.astype(np.int64)
    pairs = np.column_stack(np.where(np.triu(counts > 0)))
    if not len(pairs):
        raise ValueError("at least one observed covariance entry is required")
    scalar_count = sum(1 if j == k else 2 for j, k in pairs)
    tail = (1 - confidence_level) / (2 * scalar_count)

    def variance_interval(sample, noise_variance):
        count = len(sample)
        total = float(sample @ sample)
        return (
            np.array([total / chi2.ppf(1 - tail, count), total / chi2.ppf(tail, count)])
            - noise_variance
        )

    lower, upper = [], []
    moments = np.full(noise.shape, np.nan)
    for j, k in pairs:
        selected = valid[:, j] & valid[:, k]
        left, right = data[selected, j], data[selected, k]
        moment = float(left @ right / len(left) - noise[j, k])
        moments[j, k] = moments[k, j] = moment
        if j == k:
            interval = variance_interval(left, noise[j, j])
        else:
            plus = variance_interval(left + right, noise[j, j] + noise[k, k] + 2 * noise[j, k])
            minus = variance_interval(left - right, noise[j, j] + noise[k, k] - 2 * noise[j, k])
            interval = np.array([plus[0] - minus[1], plus[1] - minus[0]]) / 4
        lower.append(interval[0])
        upper.append(interval[1])
    return GaussianCovarianceBands(
        pairs,
        np.array(lower),
        np.array(upper),
        moments,
        counts,
        float(confidence_level),
        {
            "method": "chi-square scalar variance intervals + polarization + Bonferroni",
            "scalar_intervals": int(scalar_count),
            "one_sided_tail_probability": float(tail),
            "means": "specified zero; not estimated",
            "scope": "independent common centered Gaussian covariance; deterministic masks; known fixed fresh noise",
            "dependent_market_return_coverage": False,
        },
    )
