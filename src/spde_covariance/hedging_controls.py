"""Deterministic observational-equivalence controls for a hedging pilot.

The two sources share the fixed infinite diagonal background
q_k=(k+1)**-3/zeta(3). Only a declared finite source block is perturbed;
changing the reference cutoff never renormalizes or changes either source.
All kernels concern the retained J coordinates evaluated on the stated grid.
Floating-point refinement and tail allowances are not interval certificates
or uniform identification statements about an infinite source class.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .admissibility import _integer, _scalar
from .geometry import lifted_matrix, smat, svec
from .hedging import transport_covariance_factors
from .laguerre import LaguerreTransport, PowerLawSource
from .linalg import _real_array, validate_psd
from .source_classes import projected_covariance_integration_matrix
from .trajectories import _metadata_snapshot


def _clique_mask(cliques, dimension):
    normalized = []
    for clique in cliques:
        raw = np.asarray(clique)
        if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iu":
            raise ValueError("cliques must contain nonempty integer index sequences")
        if np.any(raw < 0) or np.any(raw >= dimension) or len(np.unique(raw)) != len(raw):
            raise ValueError("cliques must contain distinct valid grid indices")
        normalized.append(tuple(sorted(map(int, raw))))
    if not normalized or set().union(*map(set, normalized)) != set(range(dimension)):
        raise ValueError("cliques must cover the evaluation grid")
    mask = np.zeros((dimension, dimension), dtype=bool)
    for clique in normalized:
        mask[np.ix_(clique, clique)] = True
    return tuple(normalized), mask


def _common_tail_kernel(source, source_dimension, reference_dimension, evaluation, delta, order):
    """Integrate only the common diagonal modes M,...,R-1, with positive weights."""
    dimension = evaluation.shape[1]
    result = np.zeros((len(evaluation), len(evaluation)))
    if reference_dimension == source_dimension:
        return result
    propagator = LaguerreTransport(reference_dimension, cache_max_bytes=0)
    eigenvalues = source.eigenvalues(reference_dimension)[source_dimension:]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for node, weight in zip(nodes, weights, strict=True):
        operator = (
            evaluation @ propagator.matrix(delta * (node + 1) / 2)[:dimension, source_dimension:]
        )
        result += (weight / 2) * (operator * eigenvalues) @ operator.T
    return (result + result.T) / 2


def _paired_decisions(kernels, target, hedge):
    target_variances = np.array([target @ kernel @ target for kernel in kernels])
    hedge_variances = np.array([hedge @ kernel @ hedge for kernel in kernels])
    cross = np.array([target @ kernel @ hedge for kernel in kernels])
    if np.any(hedge_variances <= 0):
        raise ValueError("the declared hedge must have positive variance in both worlds")
    if not np.allclose(target_variances, target_variances[0], rtol=1e-10, atol=1e-13):
        raise ValueError("paired target variances must be observed and equal")
    if not np.allclose(hedge_variances, hedge_variances[0], rtol=1e-10, atol=1e-13):
        raise ValueError("paired hedge variances must be observed and equal")
    positions = cross / hedge_variances
    risk_matrix = np.array(
        [
            [
                (target - position * hedge) @ kernel @ (target - position * hedge)
                for position in positions
            ]
            for kernel in kernels
        ]
    )
    oracle = np.diag(risk_matrix).copy()
    regret_position = float(np.mean(positions))
    variance_position = float(np.clip(0.0, np.min(positions), np.max(positions)))

    def risks(position):
        residual = target - position * hedge
        return np.array([residual @ kernel @ residual for kernel in kernels])

    regret_risks = risks(regret_position)
    variance_risks = risks(variance_position)
    return {
        "target_variances": target_variances,
        "hedge_variances": hedge_variances,
        "cross_covariances": cross,
        "optimal_hedges": positions,
        "optimal_risks": oracle,
        "risk_matrix": risk_matrix,
        "risk_matrix_axes": "rows are true worlds (-,+); columns are chosen oracle hedges (-,+)",
        "wrong_hedge_excess": np.array(
            [
                risk_matrix[0, 1] - oracle[0],
                risk_matrix[1, 0] - oracle[1],
            ]
        ),
        "hedge_gap": float(positions[1] - positions[0]),
        "minimax_regret_hedge": regret_position,
        "minimax_regret": float(np.max(regret_risks - oracle)),
        "minimax_regret_risks": regret_risks,
        "minimax_variance_hedge": variance_position,
        "minimax_variance": float(np.max(variance_risks)),
        "minimax_variance_risks": variance_risks,
        "scope": "the two displayed worlds and one unconstrained scalar hedge only",
    }


def fixed_power_hedge_control(
    *,
    source_dimension: int = 16,
    output_dimension: int = 8,
    reference_dimension: int = 128,
    delta: float = 1 / 52,
    grid: ArrayLike | None = None,
    quadrature_order: int = 24,
    safety: float = 0.95,
    exposure: str = "spread",
    cliques: Sequence[Sequence[int]] = ((0, 1, 2, 3), (3, 4, 5, 6)),
    rank_rtol: float = 1e-12,
):
    """Construct two PSD transport sources agreeing on every observed pair.

    Let Q=diag(q_0,...,q_{M-1}), D=Q**.5 Z Q**.5. In whitened svec
    coordinates, project the target/hedge cross-covariance functional onto
    the nullspace of all observed entries and the source trace. Normalize Z
    to operator norm one, then use Q +/- safety*D and the same unmodified
    infinite tail. A numerically identified target raises instead of silently
    normalizing a zero null projection.

    ``source_minus`` and ``source_plus`` are the finite M blocks. ``factors``
    map those blocks to grid covariance and ``offset_covariance`` adds common
    modes M,...,R-1. The remaining fixed power-law tail is bounded separately.
    Source dimensions are model choices; reference dimensions refine the
    same sources. ``perturbation`` is D before multiplication by safety.
    """
    source_dimension = _integer(source_dimension, "source_dimension")
    output_dimension = _integer(output_dimension, "output_dimension")
    reference_dimension = _integer(reference_dimension, "reference_dimension")
    order = _integer(quadrature_order, "quadrature_order")
    duration = _scalar(delta, "delta", positive=True)
    safety = _scalar(safety, "safety", positive=True)
    rank_rtol = _scalar(rank_rtol, "rank_rtol", positive=True)
    if not output_dimension <= source_dimension <= reference_dimension:
        raise ValueError("dimensions must satisfy output <= source <= reference")
    if safety >= 1 or rank_rtol >= 1:
        raise ValueError("safety and rank_rtol must be strictly below one")
    points = np.linspace(0, 3, 7) if grid is None else _real_array(grid, name="grid")
    if points.shape != (7,) or np.any(points < 0) or np.any(np.diff(points) <= 0):
        raise ValueError("grid must contain seven strictly increasing nonnegative maturities")
    if exposure not in ("spread", "point"):
        raise ValueError("exposure must be 'spread' or 'point'")
    cliques, mask = _clique_mask(cliques, 7)
    target, hedge = np.eye(7)[0].copy(), np.eye(7)[5] - np.eye(7)[6]
    if exposure == "spread":
        target -= np.eye(7)[1]
    for vector in (target, hedge):
        support = set(np.flatnonzero(vector))
        if not any(support <= set(clique) for clique in cliques):
            raise ValueError("target and hedge variances must each lie in an observed clique")
    source = PowerLawSource(exponent=3, trace=1)
    eigenvalues = source.eigenvalues(source_dimension)
    finite_base = np.diag(eigenvalues)
    evaluation = LaguerreTransport(output_dimension).evaluate(points)
    integration = projected_covariance_integration_matrix(
        source_dimension,
        output_dimension,
        duration,
        quadrature_order=order,
    )
    kernel_map = lifted_matrix(evaluation) @ integration
    upper = np.triu_indices(7)
    svec_mask = mask[upper]
    rows, columns = np.triu_indices(source_dimension)
    whitening = np.sqrt(eigenvalues[rows] * eigenvalues[columns])
    weighted_map = kernel_map * whitening[None, :]
    constraints = np.vstack(
        [
            weighted_map[svec_mask],
            svec(np.eye(source_dimension)) * whitening,
        ]
    )
    _, singular_values, right = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > rank_rtol * singular_values[0]))
    nullspace = right[rank:].T
    cross_matrix = (np.outer(target, hedge) + np.outer(hedge, target)) / 2
    functional = weighted_map.T @ svec(cross_matrix)
    projected = nullspace @ (nullspace.T @ functional)
    target_norm, null_norm = np.linalg.norm(functional), np.linalg.norm(projected)
    if target_norm == 0 or null_norm <= rank_rtol * target_norm:
        raise ValueError("target cross covariance is identified at the declared rank threshold")
    whitened = smat(projected / null_norm)
    whitened /= np.max(np.abs(np.linalg.eigvalsh(whitened)))
    perturbation = np.sqrt(eigenvalues)[:, None] * whitened * np.sqrt(eigenvalues)[None, :]
    finite_sources = (finite_base - safety * perturbation, finite_base + safety * perturbation)
    finite_kernels = tuple(smat(kernel_map @ svec(item)) for item in finite_sources)
    offset = _common_tail_kernel(
        source,
        source_dimension,
        reference_dimension,
        evaluation,
        duration,
        order,
    )
    baseline = offset + smat(kernel_map @ svec(finite_base))
    kernels = tuple(offset + item for item in finite_kernels)
    padded_evaluation = np.zeros((7, source_dimension))
    padded_evaluation[:, :output_dimension] = evaluation
    factors = transport_covariance_factors(
        LaguerreTransport(source_dimension),
        duration,
        evaluation=padded_evaluation,
        quadrature_order=order,
    )
    decisions = _paired_decisions(kernels, target, hedge)
    tail_output_bound = source.projected_tail_bound(
        output_dimension,
        reference_dimension,
        duration,
    )["matrix"]
    tail_kernel_bound = evaluation @ tail_output_bound @ evaluation.T
    target_tail, hedge_tail = target @ tail_kernel_bound @ target, hedge @ tail_kernel_bound @ hedge
    cross_tail = float(np.sqrt(max(target_tail, 0) * max(hedge_tail, 0)))
    hedge_errors, risk_allowances = [], []
    for position, variance in zip(
        decisions["optimal_hedges"],
        decisions["hedge_variances"],
        strict=True,
    ):
        hedge_errors.append((cross_tail + abs(position) * hedge_tail) / variance)
        residual = target - position * hedge
        risk_allowances.append(residual @ tail_kernel_bound @ residual)
    refined = projected_covariance_integration_matrix(
        source_dimension,
        output_dimension,
        duration,
        quadrature_order=2 * order,
    )
    refined_map = lifted_matrix(evaluation) @ refined
    refined_offset = _common_tail_kernel(
        source,
        source_dimension,
        reference_dimension,
        evaluation,
        duration,
        2 * order,
    )
    block_errors = []
    model = LaguerreTransport(source_dimension, cache_max_bytes=0)
    for covariance, kernel in zip(finite_sources, finite_kernels, strict=True):
        block = model.innovation_covariance(covariance, duration, method="block")
        block_errors.append(
            np.linalg.norm(
                evaluation @ block[:output_dimension, :output_dimension] @ evaluation.T - kernel,
                "fro",
            )
        )
    entry_rows, entry_columns = np.where(np.triu(mask))
    return _metadata_snapshot(
        {
            "config": {
                "source_dimension": source_dimension,
                "output_dimension": output_dimension,
                "reference_dimension": reference_dimension,
                "delta": duration,
                "quadrature_order": order,
                "safety": safety,
                "exposure": exposure,
                "rank_rtol": rank_rtol,
                "source_exponent": 3.0,
                "source_trace": 1.0,
            },
            "grid": points,
            "evaluation": evaluation,
            "integration_matrix": integration,
            "kernel_map": kernel_map,
            "factors": factors,
            "finite_source_base": finite_base,
            "perturbation": perturbation,
            "whitened_perturbation": whitened,
            "source_minus": finite_sources[0],
            "source_plus": finite_sources[1],
            "offset_covariance": offset,
            "baseline_covariance": baseline,
            "kernel_minus": kernels[0],
            "kernel_plus": kernels[1],
            "kernels": kernels,
            "target": target,
            "hedge": hedge,
            "target_vector": target,
            "hedge_vector": hedge,
            "cliques": cliques,
            "observed_mask": mask,
            "observed_svec_mask": svec_mask,
            "observed_pairs": np.column_stack([entry_rows, entry_columns]),
            "trace_value": float(np.sum(eigenvalues)),
            "paired_decisions": decisions,
            "construction": {
                "constraint_rank": rank,
                "source_coordinate_dimension": len(whitening),
                "rank_threshold": float(rank_rtol * singular_values[0]),
                "constraint_singular_values": singular_values,
                "target_null_relative_residual": float(null_norm / target_norm),
                "perturbation_trace": float(np.trace(perturbation)),
                "observed_image_difference": float(
                    np.linalg.norm(
                        (kernel_map @ svec(finite_sources[1] - finite_sources[0]))[svec_mask],
                    )
                ),
                "minimum_finite_source_eigenvalues": np.array(
                    [np.linalg.eigvalsh(item)[0] for item in finite_sources]
                ),
                "source_formula": "diag(q_k) +/- safety * embed_M(perturbation), q_k=(k+1)^-3/zeta(3)",
                "scope": "two fixed infinite-rank sources; finite J projected observations",
            },
            "tail_diagnostics": {
                "finite_source_trace": float(np.sum(eigenvalues)),
                "common_tail_trace_beyond_source": source.tail_trace(source_dimension),
                "omitted_trace_beyond_reference": source.tail_trace(reference_dimension),
                "kernel_psd_upper_bound": tail_kernel_bound,
                "kernel_operator_upper_bound": float(np.linalg.eigvalsh(tail_kernel_bound)[-1]),
                "optimal_hedge_absolute_error_bounds": np.asarray(hedge_errors),
                "optimal_risk_upper_allowances": np.asarray(risk_allowances),
                "scope": "omitted common fixed power-law tail only; bounds evaluated in floating point",
                "uniform_source_class_certificate": False,
            },
            "integration_diagnostics": {
                "refined_quadrature_order": 2 * order,
                "map_frobenius_refinement_discrepancy": float(
                    np.linalg.norm(refined - integration)
                ),
                "fixed_tail_kernel_refinement_discrepancy": float(
                    np.linalg.norm(refined_offset - offset)
                ),
                "fixed_source_kernel_refinement_discrepancies": np.array(
                    [
                        np.linalg.norm(smat((refined_map - kernel_map) @ svec(item)))
                        for item in finite_sources
                    ]
                ),
                "fixed_source_kernel_block_discrepancies": np.asarray(block_errors),
                "refined_observed_image_difference": float(
                    np.linalg.norm(
                        (refined_map @ svec(finite_sources[1] - finite_sources[0]))[svec_mask],
                    )
                ),
                "numerical_error_certificate": False,
            },
            "infinite_dimensional_identification_claim": False,
        }
    )


def free_psd_clique_hedge_interval(
    covariance: ArrayLike,
    target: ArrayLike,
    hedge: ArrayLike,
    cliques: Sequence[Sequence[int]],
    *,
    offset_covariance: ArrayLike | None = None,
):
    """Exact two-clique PSD completion interval, evaluated in floating point.

    The target is supported in the first clique and the hedge in the second.
    All clique entries are observed. Conditional on their overlap O, the
    unknown residual cross covariance ranges over +/-sqrt(v_a|O * v_b|O).
    An optional known PSD offset is subtracted before completion and restored
    to the cross covariance. This interval applies to ordinary PSD completion;
    it imposes no source transport or source trace restrictions.
    """
    covariance = validate_psd(covariance, name="covariance")
    dimension = len(covariance)
    target = _real_array(target, name="target")
    hedge = _real_array(hedge, name="hedge")
    if target.shape != (dimension,) or hedge.shape != (dimension,):
        raise ValueError("target and hedge must match covariance")
    cliques, _ = _clique_mask(cliques, dimension)
    if len(cliques) != 2:
        raise ValueError("the analytic interval requires exactly two covering cliques")
    if not set(np.flatnonzero(target)) <= set(cliques[0]):
        raise ValueError("target must be supported in the first clique")
    if not set(np.flatnonzero(hedge)) <= set(cliques[1]):
        raise ValueError("hedge must be supported in the second clique")
    offset = np.zeros_like(covariance)
    if offset_covariance is not None:
        offset = validate_psd(offset_covariance, name="offset_covariance")
        if offset.shape != covariance.shape:
            raise ValueError("offset_covariance must match covariance")
    completion = validate_psd(covariance - offset, name="covariance minus offset")
    overlap = sorted(set(cliques[0]) & set(cliques[1]))
    center = 0.0
    target_conditional = float(target @ completion @ target)
    hedge_conditional = float(hedge @ completion @ hedge)
    if overlap:
        overlap_covariance = completion[np.ix_(overlap, overlap)]
        inverse = np.linalg.pinv(overlap_covariance, rcond=1e-12, hermitian=True)
        target_cross, hedge_cross = (target @ completion)[overlap], (hedge @ completion)[overlap]
        center = float(target_cross @ inverse @ hedge_cross)
        target_conditional -= float(target_cross @ inverse @ target_cross)
        hedge_conditional -= float(hedge_cross @ inverse @ hedge_cross)
    # A payoff entirely in the shared coordinates has no unobserved residual.
    # Preserve this structural identity without a cancellation-sensitive square root.
    target_shared = set(np.flatnonzero(target)) <= set(overlap)
    hedge_shared = set(np.flatnonzero(hedge)) <= set(overlap)
    if target_shared or hedge_shared:
        center = float(target @ completion @ hedge)
        if target_shared:
            target_conditional = 0.0
        if hedge_shared:
            hedge_conditional = 0.0
    radius = float(np.sqrt(max(target_conditional, 0) * max(hedge_conditional, 0)))
    center += float(target @ offset @ hedge)
    variance = float(hedge @ covariance @ hedge)
    if variance <= 0:
        raise ValueError("hedge variance must be positive")
    interval = np.array([center - radius, center + radius]) / variance
    return _metadata_snapshot(
        {
            "hedge_interval": interval,
            "cross_covariance_interval": np.array([center - radius, center + radius]),
            "overlap": np.asarray(overlap, dtype=int),
            "target_conditional_variance": max(target_conditional, 0.0),
            "hedge_conditional_variance": max(hedge_conditional, 0.0),
            "minimax_regret_hedge": float(np.mean(interval)),
            "minimax_regret": float(variance * (interval[1] - interval[0]) ** 2 / 4),
            "minimax_variance_hedge": float(np.clip(0, interval[0], interval[1])),
            "scope": "ordinary two-clique PSD completion; optional known offset; no source trace cap",
            "numerical_error_certificate": False,
        }
    )
