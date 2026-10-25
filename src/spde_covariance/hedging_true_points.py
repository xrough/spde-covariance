"""A separate deterministic hedge-identification control for true point values.

The observation factor at integration time r is evaluation at grid+r in the
full Laguerre family. There is no retained-output projection before evaluation.
Two infinite-rank sources share the fixed p=3 diagonal tail; only their finite
M block differs. This annex constructs a new witness and does not reinterpret
the projected J=8 pilot or repeat its Monte Carlo experiment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .admissibility import _integer, _scalar
from .geometry import lifted_matrix, smat, svec
from .hedging_controls import _clique_mask, _paired_decisions, fixed_power_hedge_control
from .laguerre import LaguerreTransport, PowerLawSource
from .linalg import _real_array
from .provenance import _jsonable


def _point_map(source_dimension, reference_dimension, grid, delta, order, source):
    """Positive integration of true point evaluations, with a common tail."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    model = LaguerreTransport(reference_dimension, cache_max_bytes=0)
    eigenvalues = source.eigenvalues(reference_dimension)
    kernel_map = np.zeros((28, source_dimension * (source_dimension + 1) // 2))
    offset = np.zeros((7, 7))
    factors = []
    for node, weight in zip(nodes, weights, strict=True):
        evaluation = model.evaluate(grid + delta * (node + 1) / 2)
        finite = evaluation[:, :source_dimension]
        kernel_map += (weight / 2) * lifted_matrix(finite)
        tail = evaluation[:, source_dimension:]
        offset += (weight / 2) * (tail * eigenvalues[source_dimension:]) @ tail.T
        factors.append(np.sqrt(weight / 2) * finite)
    return kernel_map, (offset + offset.T) / 2, tuple(factors)


def _worlds(source_base, perturbation, safety, kernel_map, offset, source, target, hedge, mask):
    finite_sources = (source_base - safety * perturbation, source_base + safety * perturbation)
    kernels = tuple(offset + smat(kernel_map @ svec(item)) for item in finite_sources)
    target_variances = np.array([target @ kernel @ target for kernel in kernels])
    hedge_variances = np.array([hedge @ kernel @ hedge for kernel in kernels])
    cross = np.array([target @ kernel @ hedge for kernel in kernels])
    if np.any(hedge_variances <= 0):
        raise ValueError("the true-point hedge must have positive variance in both worlds")
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
    excess = np.array(
        [
            risk_matrix[0, 1] - risk_matrix[0, 0],
            risk_matrix[1, 0] - risk_matrix[1, 1],
        ]
    )
    upper_mask = mask[np.triu_indices(7)]
    observed_difference = (kernel_map @ svec(finite_sources[1] - finite_sources[0]))[upper_mask]
    return {
        "source_minus": finite_sources[0],
        "source_plus": finite_sources[1],
        "kernels": kernels,
        "target_variances": target_variances,
        "hedge_variances": hedge_variances,
        "cross_covariances": cross,
        "optimal_hedges": positions,
        "hedge_gap": float(positions[1] - positions[0]),
        "risk_matrix": risk_matrix,
        "wrong_hedge_excess": excess,
        "wrong_hedge_excess_fraction_unhedged": excess / target_variances,
        "full_source_traces": np.array(
            [np.trace(item) + source.tail_trace(len(source_base)) for item in finite_sources]
        ),
        "minimum_finite_source_eigenvalues": np.array(
            [np.linalg.eigvalsh(item)[0] for item in finite_sources]
        ),
        "observed_svec_difference": float(np.linalg.norm(observed_difference)),
        "observed_entry_max_difference": float(np.max(np.abs((kernels[1] - kernels[0])[mask]))),
        "observed_relative_difference": float(
            np.linalg.norm(observed_difference)
            / np.linalg.norm(
                svec((kernels[0] + kernels[1]) / 2)[upper_mask],
            )
        ),
    }


def true_point_hedge_control(
    *,
    source_dimension: int = 16,
    reference_dimension: int = 512,
    quadrature_order: int = 24,
    delta: float = 1 / 52,
    grid: ArrayLike | None = None,
    safety: float = 0.95,
    rank_rtol: float = 1e-12,
):
    """Return a JSON-safe control with a separately built true-point witness.

    In whitened coordinates D=diag(sqrt(q)) Z diag(sqrt(q)), project the
    target/hedge cross functional onto the nullspace of all 19 observed pairs
    and source trace, then set ||Z||op=1. The sources are diag(q_k) +/- safety
    * embed_M(D), q_k=(k+1)^-3/zeta(3), without truncation renormalization.

    The common tail M,...,R-1 is integrated using true point values. Modes
    beyond R have a separate continuous-time allowance from C_tail<=q_R I_H.
    Quadrature refinement and an independent finite-source block check are
    diagnostic discrepancies, not exact numerical-error certificates.
    """
    source_dimension = _integer(source_dimension, "source_dimension")
    reference_dimension = _integer(reference_dimension, "reference_dimension")
    order = _integer(quadrature_order, "quadrature_order")
    duration = _scalar(delta, "delta", positive=True)
    safety = _scalar(safety, "safety", positive=True)
    rank_rtol = _scalar(rank_rtol, "rank_rtol", positive=True)
    if not 8 <= source_dimension <= reference_dimension:
        raise ValueError("require 8 <= source_dimension <= reference_dimension")
    if safety >= 1 or rank_rtol >= 1:
        raise ValueError("safety and rank_rtol must be strictly below one")
    points = np.linspace(0, 3, 7) if grid is None else _real_array(grid, name="grid")
    if points.shape != (7,) or np.any(points < 0) or np.any(np.diff(points) <= 0):
        raise ValueError("grid must contain seven strictly increasing nonnegative maturities")
    cliques, mask = _clique_mask(((0, 1, 2, 3), (3, 4, 5, 6)), 7)
    target, hedge = np.eye(7)[0] - np.eye(7)[1], np.eye(7)[5] - np.eye(7)[6]
    source = PowerLawSource(exponent=3, trace=1)
    eigenvalues = source.eigenvalues(source_dimension)
    source_base = np.diag(eigenvalues)
    kernel_map, offset, factors = _point_map(
        source_dimension,
        reference_dimension,
        points,
        duration,
        order,
        source,
    )
    rows, columns = np.triu_indices(source_dimension)
    whitening = np.sqrt(eigenvalues[rows] * eigenvalues[columns])
    weighted_map = kernel_map * whitening[None, :]
    upper_mask = mask[np.triu_indices(7)]
    constraints = np.vstack(
        [
            weighted_map[upper_mask],
            svec(np.eye(source_dimension)) * whitening,
        ]
    )
    _, singular_values, right = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > rank_rtol * singular_values[0]))
    nullspace = right[rank:].T
    cross_matrix = (np.outer(target, hedge) + np.outer(hedge, target)) / 2
    functional = weighted_map.T @ svec(cross_matrix)
    projected = nullspace @ (nullspace.T @ functional)
    null_norm, target_norm = np.linalg.norm(projected), np.linalg.norm(functional)
    if target_norm == 0 or null_norm <= rank_rtol * target_norm:
        raise ValueError("target cross covariance is identified at the declared rank threshold")
    whitened = smat(projected / null_norm)
    whitened /= np.max(np.abs(np.linalg.eigvalsh(whitened)))
    perturbation = np.sqrt(eigenvalues)[:, None] * whitened * np.sqrt(eigenvalues)[None, :]
    worlds = _worlds(
        source_base,
        perturbation,
        safety,
        kernel_map,
        offset,
        source,
        target,
        hedge,
        mask,
    )
    decisions = _paired_decisions(tuple(worlds["kernels"]), target, hedge)
    refined_map, refined_offset, _ = _point_map(
        source_dimension,
        reference_dimension,
        points,
        duration,
        2 * order,
        source,
    )
    block_model = LaguerreTransport(source_dimension, cache_max_bytes=0)
    unshifted_evaluation = block_model.evaluate(points)
    block_errors, factor_errors = [], []
    for covariance, kernel in zip(
        (worlds["source_minus"], worlds["source_plus"]),
        worlds["kernels"],
        strict=True,
    ):
        block = block_model.innovation_covariance(covariance, duration, method="block")
        block_errors.append(
            np.linalg.norm(
                unshifted_evaluation @ block @ unshifted_evaluation.T - (kernel - offset),
                "fro",
            )
        )
        factor_errors.append(
            np.linalg.norm(
                offset + sum(factor @ covariance @ factor.T for factor in factors) - kernel,
                "fro",
            )
        )
    # Evaluation representers satisfy <ev_x,ev_y>_H = 2-exp(-min(x,y)).
    # The omitted source is bounded by q_R I_H. Integrate the shifted full
    # representer Gram exactly: 2-gamma*exp(-min(x,y)), gamma=mean(exp(-r)).
    omitted_eigenvalue = source.eigenvalues(reference_dimension + 1)[-1]
    gamma = -np.expm1(-duration) / duration
    tail_bound = omitted_eigenvalue * (2 - gamma * np.exp(-np.minimum.outer(points, points)))
    target_tail, hedge_tail = target @ tail_bound @ target, hedge @ tail_bound @ hedge
    cross_tail = np.sqrt(max(target_tail, 0) * max(hedge_tail, 0))
    hedge_allowances = np.array(
        [
            (cross_tail + abs(position) * hedge_tail) / variance
            for position, variance in zip(
                worlds["optimal_hedges"],
                worlds["hedge_variances"],
                strict=True,
            )
        ]
    )
    original = fixed_power_hedge_control(
        source_dimension=source_dimension,
        output_dimension=8,
        reference_dimension=reference_dimension,
        delta=duration,
        grid=points,
        quadrature_order=order,
        safety=safety,
        rank_rtol=rank_rtol,
    )
    original_true_points = _worlds(
        source_base,
        original["perturbation"],
        safety,
        kernel_map,
        offset,
        source,
        target,
        hedge,
        mask,
    )
    return _jsonable(
        {
            "observation_model": "true point evaluation at grid+r; no output projection",
            "config": {
                "source_dimension": source_dimension,
                "reference_dimension": reference_dimension,
                "quadrature_order": order,
                "delta": duration,
                "safety": safety,
                "rank_rtol": rank_rtol,
                "source_exponent": 3.0,
                "source_trace": 1.0,
            },
            "grid": points,
            "cliques": cliques,
            "observed_mask": mask,
            "target": target,
            "hedge": hedge,
            "finite_source_base": source_base,
            "perturbation": perturbation,
            "whitened_perturbation": whitened,
            "kernel_map": kernel_map,
            "factors": factors,
            "offset_covariance": offset,
            "true_point_pair": worlds,
            "paired_decisions": decisions,
            "construction": {
                "constraint_rank": rank,
                "source_coordinate_dimension": len(whitening),
                "rank_threshold": float(rank_rtol * singular_values[0]),
                "constraint_singular_values": singular_values,
                "target_null_relative_residual": float(null_norm / target_norm),
                "perturbation_trace": float(np.trace(perturbation)),
                "source_formula": "diag(q_k) +/- safety*embed_M(perturbation), q_k=(k+1)^-3/zeta(3)",
            },
            "integration_diagnostics": {
                "refined_quadrature_order": 2 * order,
                "map_frobenius_refinement_discrepancy": float(
                    np.linalg.norm(refined_map - kernel_map)
                ),
                "fixed_tail_kernel_refinement_discrepancy": float(
                    np.linalg.norm(refined_offset - offset)
                ),
                "same_witness_refined_observed_difference": float(
                    np.linalg.norm(
                        (refined_map @ svec(2 * safety * perturbation))[upper_mask],
                    )
                ),
                "independent_block_finite_kernel_discrepancies": np.asarray(block_errors),
                "positive_factor_kernel_discrepancies": np.asarray(factor_errors),
                "numerical_error_certificate": False,
            },
            "tail_diagnostics": {
                "common_tail_trace_beyond_source": source.tail_trace(source_dimension),
                "omitted_trace_beyond_reference": source.tail_trace(reference_dimension),
                "omitted_eigenvalue_upper_bound": omitted_eigenvalue,
                "kernel_psd_upper_bound": tail_bound,
                "target_variance_upper_bound": float(target_tail),
                "hedge_variance_upper_bound": float(hedge_tail),
                "cross_covariance_absolute_bound": float(cross_tail),
                "optimal_hedge_absolute_error_bounds": hedge_allowances,
                "optimal_risk_upper_allowances": np.array(
                    [
                        (target - position * hedge) @ tail_bound @ (target - position * hedge)
                        for position in worlds["optimal_hedges"]
                    ]
                ),
                "argument": "C_tail<=q_R I_H; integrate translated RKHS evaluation Gram exactly",
                "scope": "omitted common fixed p=3 tail only; continuous-time formula evaluated in float",
                "uniform_source_class_certificate": False,
            },
            "original_projected_witness": {
                "observation_model": "J=8 projection before point evaluation",
                "perturbation": original["perturbation"],
                "projected_observed_difference": original["construction"][
                    "observed_image_difference"
                ],
                "projected_optimal_hedges": original["paired_decisions"]["optimal_hedges"],
                "evaluated_as_true_points": original_true_points,
                "is_true_point_alias_at_1e_minus_10": original_true_points[
                    "observed_svec_difference"
                ]
                < 1e-10,
            },
            "scope": "deterministic point-observation annex; finite unknown source block with fixed common infinite tail",
            "main_projected_benchmark_reinterpreted": False,
            "monte_carlo_performance_claim": False,
            "uniform_infinite_source_identification_claim": False,
        }
    )
