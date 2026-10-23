"""Independent Gaussian innovation observations with separate latent truth."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import validate_psd
from .observations import Observation


@dataclass(frozen=True)
class SimulationResult:
    observations: tuple[Observation, ...]
    latent_innovations: NDArray[np.float64]
    latent_covariances: NDArray[np.float64]
    metadata: dict


def _gaussian_sample(rng, covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0)) * rng.standard_normal(len(eigenvalues))
    )


def simulate_innovations(
    covariance: ArrayLike,
    operators: Sequence[ArrayLike],
    *,
    times: ArrayLike | None = None,
    noise_covariances: Sequence[ArrayLike] | ArrayLike | None = None,
    seed: int = 0,
) -> SimulationResult:
    """Sample independent ``y_i=O_i X_i+epsilon_i`` with declared covariance.

    ``covariance`` is a common J by J matrix or an N by J by J array. Operators
    may have different row counts. Noise covariance is either common (equal row
    counts), one matrix per date, or zero. Time is measured in years by default.
    """
    maps = tuple(np.asarray(operator, dtype=float) for operator in operators)
    if not maps:
        raise ValueError("at least one operator is required")
    if any(
        operator.ndim != 2 or min(operator.shape) == 0 or not np.all(np.isfinite(operator))
        for operator in maps
    ):
        raise ValueError("operators must be nonempty finite matrices")
    dimension = maps[0].shape[1]
    if any(operator.shape[1] != dimension for operator in maps):
        raise ValueError("all operators must use the same latent basis")
    count = len(maps)
    dates = np.arange(count, dtype=float) if times is None else np.asarray(times, dtype=float)
    if dates.shape != (count,) or not np.all(np.isfinite(dates)):
        raise ValueError("times must be finite, with one entry per observation")
    covariances = np.asarray(covariance, dtype=float)
    if covariances.ndim == 2:
        covariances = np.broadcast_to(covariances, (count, dimension, dimension)).copy()
    if covariances.shape != (count, dimension, dimension):
        raise ValueError("covariance must have shape (J,J) or (N,J,J)")
    covariances = np.stack([validate_psd(matrix) for matrix in covariances])
    if noise_covariances is None:
        noises = tuple(np.zeros((len(operator), len(operator))) for operator in maps)
    elif isinstance(noise_covariances, np.ndarray) and noise_covariances.ndim == 2:
        noises = tuple(noise_covariances for _ in maps)
    else:
        # Permit an ordinary nested list for a common covariance too.
        try:
            candidate = np.asarray(noise_covariances, dtype=float)
        except ValueError:
            candidate = None
        noises = (
            tuple(candidate for _ in maps)
            if candidate is not None and candidate.ndim == 2
            else tuple(noise_covariances)
        )
    if len(noises) != count:
        raise ValueError("provide one noise covariance per observation")
    noises = tuple(validate_psd(matrix) for matrix in noises)
    if any(
        noise.shape != (len(operator), len(operator))
        for noise, operator in zip(noises, maps, strict=True)
    ):
        raise ValueError("noise covariance must match the corresponding observation rows")
    rng = np.random.default_rng(seed)
    latent = np.stack([_gaussian_sample(rng, matrix) for matrix in covariances])
    observations = tuple(
        Observation(
            float(time), operator @ innovation + _gaussian_sample(rng, noise), operator, noise
        )
        for time, operator, innovation, noise in zip(dates, maps, latent, noises, strict=True)
    )
    return SimulationResult(
        observations,
        latent,
        covariances,
        {
            "seed": seed,
            "distribution": "independent centered Gaussian innovations",
            "noise": "independent fresh Gaussian measurements with known covariance",
            "target": "innovation covariance V",
            "basis": "orthonormal H coordinates",
        },
    )
