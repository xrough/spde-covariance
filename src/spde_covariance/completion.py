"""Canonical finite-grid completion and explicit, separate PSD repair.

The empirical spectral-truncation rule preserves observed entries. It need not
produce a PSD matrix, even when its input cliques are PSD and consistent. No
repair is performed silently. Quadrature weights define the separator integral
operator; these are not the weighted Sobolev coefficient coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np

from .linalg import _real_array


def _snapshot(value):
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, Mapping):
        return MappingProxyType({key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class KernelCompletion:
    """A grid-kernel estimate, with no implicit conversion to an H matrix.

    An unavailable numerical rank or FVE rule returns ``kernel=None`` and an
    explicit reason. A returned empirical kernel may be indefinite; consult
    ``diagnostics['output_psd']`` before using it as a covariance.
    """

    kernel: np.ndarray | None
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "kernel", _snapshot(self.kernel))
        object.__setattr__(self, "diagnostics", _snapshot(self.diagnostics))

    @property
    def available(self):
        return self.diagnostics["available"]


@dataclass(frozen=True, slots=True)
class ExposureRiskRange:
    """Attainable variance range over all PSD two-block matrix completions.

    ``exposure`` means coefficients of a linear combination of grid values,
    not H-coordinate coefficients. The range is an identification range, not a
    sampling confidence interval or an SPDE-source admissibility statement.
    """

    lower: float | None
    upper: float | None
    canonical: float | None
    identified: bool | None
    lower_kernel: np.ndarray | None
    upper_kernel: np.ndarray | None
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("lower_kernel", "upper_kernel", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))

    @property
    def available(self):
        return self.diagnostics["available"]


def _relative_tolerance(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError("support_rtol must lie strictly between zero and one")
    return float(value)


def _weights(value, size):
    result = np.ones(size) if value is None else _real_array(value, name="weights")
    if result.shape != (size,) or np.any(result <= 0):
        raise ValueError("weights must be finite, strictly positive and match the grid")
    return result


def _symmetric_matrix(value, *, missing=False):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise ValueError("kernel must be a real numeric matrix")
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("kernel must be a nonempty square matrix")
    if np.any(np.isinf(matrix)) or (not missing and np.any(np.isnan(matrix))):
        raise ValueError("kernel entries must be finite, with NaN only for missing pairs")
    finite = np.isfinite(matrix)
    if not np.array_equal(finite, finite.T):
        raise ValueError("kernel missingness must be symmetric")
    if np.any(finite):
        scale = float(np.max(np.abs(matrix[finite])))
        if np.max(np.abs(matrix[finite] - matrix.T[finite])) > 1e-12 * scale:
            raise ValueError("kernel must be symmetric on its finite entries")
    return (matrix + matrix.T) / 2


def _cover(cliques, size):
    try:
        raw = tuple(cliques)
    except TypeError as error:
        raise ValueError("cliques must be a nonempty ordered cover") from error
    if not raw:
        raise ValueError("cliques must be a nonempty ordered cover")
    groups, covered, mask = [], set(), np.zeros((size, size), dtype=bool)
    for item in raw:
        indices = np.asarray(item)
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= size)
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("each clique must contain strictly increasing valid integer indices")
        indices = indices.astype(int)
        separator = covered.intersection(indices)
        if separator and not any(separator.issubset(previous) for previous in groups):
            raise ValueError("ordered cliques do not satisfy the running-intersection property")
        groups.append(tuple(int(index) for index in indices))
        covered.update(indices)
        mask[np.ix_(indices, indices)] = True
    if covered != set(range(size)):
        raise ValueError("cliques must cover every grid node")
    return tuple(groups), mask


def _inputs(partial_kernel, cliques, weights, support_rtol):
    matrix = _symmetric_matrix(partial_kernel, missing=True)
    groups, mask = _cover(cliques, len(matrix))
    if not np.all(np.isfinite(matrix[mask])):
        raise ValueError("every observed clique entry must be finite")
    if not np.all(np.isnan(matrix[~mask])):
        raise ValueError("entries outside the observed clique mask must be NaN")
    return matrix, groups, mask, _weights(weights, len(matrix)), _relative_tolerance(support_rtol)


def observed_mask(cliques, size):
    """Validate an ordered running-intersection cover and return its pair mask."""
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    return _snapshot(_cover(cliques, size)[1])


def _psd_diagnostic(matrix, rtol):
    values = np.linalg.eigvalsh(matrix)
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    minimum = float(values[0]) if len(values) else 0.0
    return {
        "minimum_eigenvalue": minimum,
        "scale": scale,
        "tolerance": rtol * scale,
        "psd": minimum >= -rtol * scale,
    }


def _norm(matrix):
    return float(np.linalg.norm(matrix, ord=2)) if min(matrix.shape, default=0) else 0.0


def _separator(matrix, a, s, b, weights, rank, fve, rtol):
    root = np.sqrt(weights[s])
    separator = root[:, None] * matrix[np.ix_(s, s)] * root[None, :]
    values, vectors = np.linalg.eigh(separator)
    values, vectors = values[::-1], vectors[:, ::-1]
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    threshold = rtol * scale
    supported = values > threshold
    support_values, support_vectors = values[supported], vectors[:, supported]
    left = matrix[np.ix_(a, s)] * root[None, :]
    right = root[:, None] * matrix[np.ix_(s, b)]
    projector = support_vectors @ support_vectors.T
    residual_left = float(np.linalg.norm(left - left @ projector))
    residual_right = float(np.linalg.norm(right - projector @ right))
    # Modes below the numerical support threshold need not be mathematically
    # zero. PSD block Cauchy--Schwarz scales their cross terms as sqrt(threshold).
    cross_scale = max(
        np.linalg.norm(left),
        np.linalg.norm(right),
        np.sqrt(max(_norm(matrix[np.ix_(a, a)]), _norm(matrix[np.ix_(b, b)])) * scale),
    )
    compatibility_tolerance = float(np.sqrt(rtol) * cross_scale * np.sqrt(max(len(s), 1)))
    support_compatible = max(residual_left, residual_right) <= compatibility_tolerance
    left_factor = (left @ support_vectors) / np.sqrt(support_values)
    right_factor = (support_vectors.T @ right) / np.sqrt(support_values)[:, None]
    ra = matrix[np.ix_(a, a)] - left_factor @ left_factor.T
    rb = matrix[np.ix_(b, b)] - right_factor.T @ right_factor
    ra, rb = (ra + ra.T) / 2, (rb + rb.T) / 2
    diagnostic = {
        "separator_indices": s,
        "left_indices": a,
        "right_indices": b,
        "eigenvalues": values,
        "support_threshold": threshold,
        "support_rank": int(np.sum(supported)),
        "separator_psd": bool(np.all(values >= -threshold)),
        "support_left_residual": residual_left,
        "support_right_residual": residual_right,
        "support_compatibility_tolerance": compatibility_tolerance,
        "support_compatible": support_compatible,
        "left_schur": _psd_diagnostic(
            np.sqrt(weights[a])[:, None] * ra * np.sqrt(weights[a])[None, :], rtol
        ),
        "right_schur": _psd_diagnostic(
            np.sqrt(weights[b])[:, None] * rb * np.sqrt(weights[b])[None, :], rtol
        ),
    }
    if fve is not None and not diagnostic["separator_psd"]:
        return (
            None,
            {
                **diagnostic,
                "available": False,
                "reason": "FVE is undefined for an indefinite separator",
            },
            ra,
            rb,
        )
    selected = len(support_values) if rank is None else rank
    if fve is not None:
        total = float(np.sum(np.maximum(values, 0.0)))
        selected = (
            int(np.searchsorted(np.cumsum(support_values), fve * total, side="right") + 1)
            if total > 0
            else 0
        )
    if selected > len(support_values):
        return (
            None,
            {
                **diagnostic,
                "available": False,
                "reason": "requested rank/FVE exceeds the positive numerical support",
                "requested_rank": selected,
            },
            ra,
            rb,
        )
    retained_values, retained_vectors = support_values[:selected], support_vectors[:, :selected]
    # Contract with cross covariances before dividing by small eigenvalues;
    # forming a dense pseudoinverse first magnifies cancellation unnecessarily.
    prediction = ((left @ retained_vectors) / np.sqrt(retained_values)) @ (
        (retained_vectors.T @ right) / np.sqrt(retained_values)[:, None]
    )
    inverse_norm = float(1 / retained_values[-1]) if selected else 0.0
    retained_projector = retained_vectors @ retained_vectors.T
    diagnostic.update(
        available=True,
        reason=None,
        selected_rank=selected,
        retained_eigenvalues=retained_values,
        achieved_fve=float(retained_values.sum() / np.maximum(values, 0.0).sum())
        if np.any(values > 0)
        else None,
        retained_condition_number=float(retained_values[0] / retained_values[-1])
        if selected
        else None,
        inverse_norm=inverse_norm,
        retained_left_residual=float(np.linalg.norm(left - left @ retained_projector)),
        retained_right_residual=float(np.linalg.norm(right - retained_projector @ right)),
        cross_norm_bound=_norm(left) * inverse_norm * _norm(right),
        inverse_perturbation_multiplier=_norm(left) * inverse_norm**2 * _norm(right),
    )
    return prediction, diagnostic, ra, rb


def complete_kernel(
    partial_kernel,
    *,
    cliques,
    weights=None,
    rank=None,
    fve=None,
    support_rtol=1e-12,
    population=False,
):
    """Complete a consistent grid partial kernel on an ordered clique cover.

    Missing pairs must be NaN, observed pairs finite and symmetric. ``rank``
    applies the same explicit rank at every nontrivial separator; ``fve`` uses
    a strictly greater fraction of the *total positive trace*, not the mean eigenvalue. They
    are mutually exclusive. Neither rule is selected by unseen values.

    ``population=True`` uses numerical support only and requires PSD input
    cliques, compatible separator supports and a PSD completed output, within
    the declared floating-point tolerance. These are numerical checks, not an
    interval-arithmetic certificate. Rank/FVE unavailability is returned as a
    result with ``kernel=None``; malformed inputs raise ValueError.
    """
    matrix, groups, mask, weights, rtol = _inputs(partial_kernel, cliques, weights, support_rtol)
    if not isinstance(population, (bool, np.bool_)):
        raise ValueError("population must be a boolean")
    if rank is not None and (
        isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or rank < 0
    ):
        raise ValueError("rank must be a nonnegative integer or None")
    if fve is not None and (
        isinstance(fve, (bool, np.bool_))
        or not isinstance(fve, Real)
        or not np.isfinite(fve)
        or not 0 < fve < 1
    ):
        raise ValueError("fve must lie strictly between zero and one")
    if rank is not None and fve is not None:
        raise ValueError("rank and fve are mutually exclusive")
    if population and (rank is not None or fve is not None):
        raise ValueError("population completion uses numerical support, not empirical rank/FVE")
    clique_checks = []
    for group in groups:
        index = np.asarray(group)
        root = np.sqrt(weights[index])
        clique_checks.append(
            _psd_diagnostic(root[:, None] * matrix[np.ix_(index, index)] * root[None, :], rtol)
        )
    diagnostics = {
        "available": True,
        "reason": None,
        "population": bool(population),
        "target": "canonical grid kernel"
        if rank is None and fve is None
        else "spectrally regularized canonical grid-kernel estimator",
        "observed_mask": mask,
        "cliques": groups,
        "weights": weights,
        "support_rtol": rtol,
        "rank_rule": rank,
        "fve_rule": fve,
        "clique_checks": clique_checks,
        "input_cliques_psd": all(item["psd"] for item in clique_checks),
        "joins": [],
        "numerical_error_certificate": False,
        "coordinate_convention": "kernel values on grid; not orthonormal H coefficients",
    }
    completed, covered = matrix.copy(), set(groups[0])
    for number, group in enumerate(groups[1:], start=1):
        current = set(group)
        a = np.array(sorted(covered - current), dtype=int)
        s = np.array(sorted(covered.intersection(current)), dtype=int)
        b = np.array(sorted(current - covered), dtype=int)
        if not len(a) or not len(b):
            diagnostics["joins"].append(
                {
                    "clique_number": number,
                    "available": True,
                    "skipped": True,
                    "reason": "no missing cross block",
                }
            )
        else:
            prediction, diagnostic, _, _ = _separator(completed, a, s, b, weights, rank, fve, rtol)
            diagnostic["clique_number"] = number
            diagnostics["joins"].append(diagnostic)
            if prediction is None:
                return KernelCompletion(
                    None, {**diagnostics, "available": False, "reason": diagnostic["reason"]}
                )
            completed[np.ix_(a, b)] = prediction
            completed[np.ix_(b, a)] = prediction.T
        covered.update(current)
    # Restore the input bit pattern on the observed region after the symmetric
    # algebra above; in particular a regularizer never silently changes it.
    completed[mask] = matrix[mask]
    if not np.all(np.isfinite(completed)):
        return KernelCompletion(
            None, {**diagnostics, "available": False, "reason": "completion is not finite"}
        )
    root = np.sqrt(weights)
    output_check = _psd_diagnostic(root[:, None] * completed * root[None, :], rtol)
    diagnostics.update(
        output_psd=output_check["psd"],
        output_psd_check=output_check,
        output_minimum_eigenvalue=float(np.linalg.eigvalsh(completed)[0]),
        observed_preservation_residual=float(np.max(np.abs(completed[mask] - matrix[mask]))),
        maximum_inverse_norm=max(
            (item.get("inverse_norm", 0.0) for item in diagnostics["joins"]), default=0.0
        ),
        support_compatible=all(
            item.get("support_compatible", True) for item in diagnostics["joins"]
        ),
    )
    if population and not (
        diagnostics["input_cliques_psd"]
        and diagnostics["support_compatible"]
        and diagnostics["output_psd"]
    ):
        return KernelCompletion(
            None,
            {
                **diagnostics,
                "available": False,
                "reason": "population input/support/output failed PSD compatibility checks",
            },
        )
    return KernelCompletion(completed, diagnostics)


def weighted_psd_projection(kernel, *, weights=None, observed_mask=None):
    """Project a complete kernel in weighted L2/Frobenius coordinates.

    This is an explicitly different estimate. Both total and observed-region
    changes are reported; the result is not called a canonical completion.
    """
    matrix = _symmetric_matrix(kernel)
    weights = _weights(weights, len(matrix))
    root = np.sqrt(weights)
    weighted = root[:, None] * matrix * root[None, :]
    values, vectors = np.linalg.eigh(weighted)
    repaired = (vectors * np.maximum(values, 0.0)) @ vectors.T
    result = repaired / root[:, None] / root[None, :]
    difference = result - matrix
    if observed_mask is None:
        mask = np.ones(matrix.shape, dtype=bool)
    else:
        mask = np.asarray(observed_mask)
        if mask.dtype.kind != "b" or mask.shape != matrix.shape or not np.array_equal(mask, mask.T):
            raise ValueError("observed_mask must be a symmetric boolean kernel-sized matrix")
    weighted_difference = root[:, None] * difference * root[None, :]
    return KernelCompletion(
        result,
        {
            "available": True,
            "reason": None,
            "target": "weighted PSD projection of estimated grid kernel",
            "negative_eigenvalues": int(np.sum(values < 0)),
            "input_weighted_minimum_eigenvalue": float(values[0]),
            "output_weighted_minimum_eigenvalue": float(np.linalg.eigvalsh(repaired)[0]),
            "weighted_hs_change": float(np.linalg.norm(weighted_difference)),
            "observed_weighted_hs_change": float(np.linalg.norm(weighted_difference[mask])),
            "observed_maximum_change": float(np.max(np.abs(difference[mask])))
            if np.any(mask)
            else 0.0,
            "weighted_trace_before": float(np.trace(weighted)),
            "weighted_trace_after": float(np.trace(repaired)),
            "weighted_trace_change": float(np.trace(repaired) - np.trace(weighted)),
            "kernel_diagonal_sum_before": float(np.trace(matrix)),
            "kernel_diagonal_sum_after": float(np.trace(result)),
            "kernel_diagonal_sum_change": float(np.trace(result) - np.trace(matrix)),
            "trace_scope": "weighted grid L2 trace and unweighted node-diagonal sum; neither is H-operator trace",
            "weights": weights,
            "observed_mask": mask,
            "preserves_observed_entries": bool(np.array_equal(result[mask], matrix[mask])),
            "numerical_error_certificate": False,
        },
    )


def two_block_risk_range(partial_kernel, exposure, *, cliques, weights=None, support_rtol=1e-12):
    """Return the attainable scalar-risk interval for all PSD two-block completions.

    The interval has center c.T K_star c and half-width
    ``2*sqrt(c_A.T R_A c_A)*sqrt(c_B.T R_B c_B)``. Endpoint kernels are returned
    as witnesses. No Gaussian assumption is needed for the covariance algebra.
    """
    matrix, groups, _, weights, rtol = _inputs(partial_kernel, cliques, weights, support_rtol)
    if len(groups) != 2:
        raise ValueError("risk ranges require exactly two cliques")
    coefficients = _real_array(exposure, name="exposure")
    if coefficients.shape != (len(matrix),):
        raise ValueError("exposure must contain one coefficient per grid node")
    result = complete_kernel(
        matrix, cliques=groups, weights=weights, support_rtol=rtol, population=True
    )
    if not result.available:
        return ExposureRiskRange(None, None, None, None, None, None, result.diagnostics)
    first, second = set(groups[0]), set(groups[1])
    a = np.array(sorted(first - second), dtype=int)
    s = np.array(sorted(first.intersection(second)), dtype=int)
    b = np.array(sorted(second - first), dtype=int)
    _, diagnostic, ra, rb = _separator(matrix, a, s, b, weights, None, None, rtol)
    va, vb = (
        float(coefficients[a] @ ra @ coefficients[a]),
        float(coefficients[b] @ rb @ coefficients[b]),
    )
    va, vb = max(0.0, va), max(0.0, vb)
    perturbation = (
        np.outer(ra @ coefficients[a], rb @ coefficients[b]) / np.sqrt(va * vb)
        if va > 0 and vb > 0
        else np.zeros((len(a), len(b)))
    )
    lower_kernel, upper_kernel = result.kernel.copy(), result.kernel.copy()
    for output, sign in ((lower_kernel, -1), (upper_kernel, 1)):
        output[np.ix_(a, b)] += sign * perturbation
        output[np.ix_(b, a)] += sign * perturbation.T
    center = float(coefficients @ result.kernel @ coefficients)
    half_width = float(2 * np.sqrt(va * vb))
    scale = float(np.abs(coefficients) @ np.abs(result.kernel) @ np.abs(coefficients))
    return ExposureRiskRange(
        center - half_width,
        center + half_width,
        center,
        half_width <= rtol * scale,
        lower_kernel,
        upper_kernel,
        {
            "available": True,
            "reason": None,
            "half_width": half_width,
            "left_conditional_variance": va,
            "right_conditional_variance": vb,
            "left_schur_residual": ra,
            "right_schur_residual": rb,
            "separator": diagnostic,
            "scope": "all PSD grid-matrix completions; not a confidence interval or SPDE-source class",
        },
    )
