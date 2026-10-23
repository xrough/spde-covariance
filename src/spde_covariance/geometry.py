"""Lifted covariance observation geometry with a Frobenius-isometric metric."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from math import isqrt
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import _real_array, validate_symmetric


def svec(matrix: ArrayLike) -> NDArray[np.float64]:
    """Upper-triangular vectorization, with sqrt(2) on off-diagonal entries."""
    array = validate_symmetric(matrix)
    rows, columns = np.triu_indices(len(array))
    result = array[rows, columns].copy()
    result[rows != columns] *= np.sqrt(2)
    return result


def smat(vector: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`svec`; reject lengths that are not triangular."""
    values = _real_array(vector, name="vector")
    if values.ndim != 1 or not len(values):
        raise ValueError("vector must be nonempty and one-dimensional")
    root = isqrt(8 * len(values) + 1)
    dimension = (root - 1) // 2
    if dimension * (dimension + 1) // 2 != len(values):
        raise ValueError("vector length must be a triangular number")
    rows, columns = np.triu_indices(dimension)
    entries = values.copy()
    entries[rows != columns] /= np.sqrt(2)
    result = np.zeros((dimension, dimension))
    result[rows, columns] = entries
    result[columns, rows] = entries
    return result


def symmetric_basis(dimension: int) -> NDArray[np.float64]:
    """Orthonormal symmetric basis with shape ``(J*(J+1)/2, J, J)``."""
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    size = dimension * (dimension + 1) // 2
    return np.asarray([smat(row) for row in np.eye(size)])


def lifted_matrix(operator: ArrayLike) -> NDArray[np.float64]:
    """Matrix representing ``B -> O B O.T`` in symmetric coordinates."""
    op = _real_array(operator, name="operator")
    if op.ndim != 2 or not op.shape[0] or not op.shape[1]:
        raise ValueError("operator must be a nonempty matrix")
    return np.column_stack([svec(op @ basis @ op.T) for basis in symmetric_basis(op.shape[1])])


def _design_inputs(
    operators: Sequence[ArrayLike], weights: ArrayLike | None
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    ops = [_real_array(op, name="operator") for op in operators]
    if not ops:
        raise ValueError("at least one operator is required")
    dimension = ops[0].shape[1] if ops[0].ndim == 2 else 0
    if dimension < 1 or any(
        op.ndim != 2 or not op.shape[0] or op.shape[1] != dimension for op in ops
    ):
        raise ValueError("operators must be nonempty matrices with a common curve dimension")
    if weights is None:
        local_weights = np.full(len(ops), 1 / len(ops))
    else:
        local_weights = _real_array(weights, name="weights")
        if (
            local_weights.shape != (len(ops),)
            or np.any(local_weights < 0)
            or not np.any(local_weights > 0)
        ):
            raise ValueError("weights must be nonnegative, nonzero, and match the operators")
    return ops, local_weights


def gram_operator(
    operators: Sequence[ArrayLike],
    weights: ArrayLike | None = None,
    *,
    cache_max_bytes: int = 32 * 1024 * 1024,
) -> NDArray[np.float64]:
    """Return ``sum_i w_i A_i.T A_i`` for lifted observation maps.

    Default weights are equal and sum to one. Supplied weights are preserved;
    callers using a local covariance average should normalize them explicitly.

    A local least-recently-used cache stores repeated designs' normal matrices
    ``A_i.T @ A_i``. Its payload budget counts normal-array and serialized-key
    bytes, excluding Python container overhead and temporary working arrays.
    Entries larger than the budget are never cached; zero disables caching.
    Nothing is retained between calls, and zero-weight designs are not lifted.
    """
    if (
        isinstance(cache_max_bytes, bool)
        or not isinstance(cache_max_bytes, (int, np.integer))
        or cache_max_bytes < 0
    ):
        raise ValueError("cache_max_bytes must be a nonnegative integer")
    cache_max_bytes = int(cache_max_bytes)
    ops, local_weights = _design_inputs(operators, weights)
    size = ops[0].shape[1] * (ops[0].shape[1] + 1) // 2
    result = np.zeros((size, size))
    cache: OrderedDict[tuple[tuple[int, ...], bytes], tuple[NDArray[np.float64], int]] = (
        OrderedDict()
    )
    cached_bytes = 0
    for op, weight in zip(ops, local_weights, strict=True):
        if weight == 0:
            continue
        # Every normal matrix has result's shape and float64 payload. Check the
        # size before serializing an operator that could never fit the budget.
        entry_bytes = result.nbytes + op.nbytes
        key = (op.shape, op.tobytes()) if entry_bytes <= cache_max_bytes else None
        cached = cache.get(key) if key is not None else None
        if cached is not None:
            normal, _ = cached
            cache.move_to_end(key)
        else:
            lifted = lifted_matrix(op)
            with np.errstate(over="ignore", invalid="ignore"):
                normal = lifted.T @ lifted
            if not np.all(np.isfinite(normal)):
                # Preserve the prior weighted-product arithmetic when an
                # unweighted normal overflows but a small weight rescales it.
                with np.errstate(over="ignore", invalid="ignore"):
                    result += (weight * lifted.T) @ lifted
                continue
            if key is not None:
                while cached_bytes + entry_bytes > cache_max_bytes:
                    _, (_, evicted_bytes) = cache.popitem(last=False)
                    cached_bytes -= evicted_bytes
                cache[key] = (normal, entry_bytes)
                cached_bytes += entry_bytes
        result += weight * normal
    if not np.all(np.isfinite(result)):
        raise ValueError("observation Gram matrix overflowed; rescale the design")
    return (result + result.T) / 2


def diagnose_design(
    operators: Sequence[ArrayLike],
    weights: ArrayLike | None = None,
    targets: Mapping[str, ArrayLike] | Sequence[ArrayLike] | ArrayLike | None = None,
    regularization: float = 1e-3,
    rank_rtol: float = 1e-10,
) -> dict[str, Any]:
    """Numerical observability diagnostics for a finite covariance sieve.

    Target identification is relative to the explicitly reported spectral
    threshold; it is not a proof of infinite-dimensional identifiability.
    """
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    if not np.isfinite(rank_rtol) or rank_rtol <= 0 or rank_rtol >= 1:
        raise ValueError("rank_rtol must lie strictly between zero and one")
    gram = gram_operator(operators, weights)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0)
    threshold = rank_rtol * float(eigenvalues[-1])
    observed = eigenvalues > threshold
    rank = int(np.count_nonzero(observed))
    result: dict[str, Any] = {
        "gram": gram,
        "eigenvalues": eigenvalues,
        "dimension": len(gram),
        "rank": rank,
        "rank_threshold": threshold,
        "rank_rtol": rank_rtol,
        "regularization": regularization,
        "smallest_eigenvalue": float(eigenvalues[0]),
        "smallest_positive_eigenvalue": float(eigenvalues[observed][0]) if rank else None,
        "condition_number": float(eigenvalues[-1] / eigenvalues[0])
        if rank == len(gram)
        else float("inf"),
        "targets": {},
    }
    if targets is None:
        return result
    if isinstance(targets, Mapping):
        named_targets = targets.items()
    else:
        array = np.asarray(targets)
        target_list = [array] if array.ndim == 2 else targets
        named_targets = ((f"target_{index}", target) for index, target in enumerate(target_list))
    for name, target in named_targets:
        vector = svec(target)
        if vector.shape != (len(gram),):
            raise ValueError("target dimension must match the design")
        coordinates = eigenvectors.T @ vector
        norm = float(np.linalg.norm(vector))
        residual = float(np.linalg.norm(coordinates[~observed]))
        relative_residual = residual / norm if norm else 0.0
        identified = relative_residual <= rank_rtol
        quadratic_form = float(np.sum(coordinates[observed] ** 2 / eigenvalues[observed]))
        result["targets"][str(name)] = {
            "identified": identified,
            "nullspace_residual": residual,
            "nullspace_relative_residual": relative_residual,
            "observability_factor": float(
                np.sqrt(np.sum(coordinates**2 / (eigenvalues + regularization)))
            ),
            "pseudoinverse_quadratic_form": quadratic_form,
            "variance_factor": quadratic_form if identified else float("inf"),
        }
    return result
