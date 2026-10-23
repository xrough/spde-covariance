"""Small symmetric-matrix primitives in orthonormal Hilbert coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _real_array(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_symmetric(
    matrix: ArrayLike,
    *,
    name: str = "matrix",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> FloatArray:
    """Validate a nonempty real symmetric matrix and remove rounding asymmetry.

    The tolerance is ``atol + rtol * max(abs(matrix))``. Substantive
    asymmetry raises an error rather than silently changing the input.
    """
    array = _real_array(matrix, name=name)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not array.size:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if rtol < 0 or atol < 0 or not np.isfinite(rtol + atol):
        raise ValueError("symmetry tolerances must be finite and nonnegative")
    tolerance = atol + rtol * float(np.max(np.abs(array)))
    if float(np.max(np.abs(array - array.T))) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    return (array + array.T) / 2


def validate_psd(
    matrix: ArrayLike,
    *,
    name: str = "matrix",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> FloatArray:
    """Validate positive semidefiniteness up to the stated spectral tolerance."""
    array = validate_symmetric(matrix, name=name, rtol=rtol, atol=atol)
    eigenvalues = np.linalg.eigvalsh(array)
    tolerance = atol + rtol * float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return array


def project_psd_trace(matrix: ArrayLike, trace_cap: float) -> FloatArray:
    """Frobenius projection onto ``B >= 0, trace(B) <= trace_cap``.

    Eigenvalues are projected onto the nonnegative simplex with an inequality
    constraint. This is a full covariance projection; no rank is prescribed.
    """
    array = validate_symmetric(matrix)
    if not np.isscalar(trace_cap) or not np.isfinite(trace_cap) or trace_cap < 0:
        raise ValueError("trace_cap must be finite and nonnegative")
    if trace_cap == 0:
        return np.zeros_like(array)
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    projected = np.maximum(eigenvalues, 0)
    if float(np.sum(projected)) > trace_cap:
        descending = eigenvalues[::-1]
        partial_sums = np.cumsum(descending) - trace_cap
        indices = np.arange(1, len(descending) + 1)
        active = np.flatnonzero(descending - partial_sums / indices > 0)
        last = int(active[-1])
        threshold = partial_sums[last] / (last + 1)
        projected = np.maximum(eigenvalues - threshold, 0)
    result = (eigenvectors * projected) @ eigenvectors.T
    return (result + result.T) / 2
