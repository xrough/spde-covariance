"""Fit finite hedge rules using only supplied masked training innovations.

This pilot accepts no latent covariance, future outcomes, or simulation
metadata. Missing values are NaN. Gaussian bands require centered independent
rows, deterministic masks, and the declared known fresh observation noise.
They are conditional on those assumptions, including a fixed source tail.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .coherent import fit_coherent_partial
from .completion import complete_kernel, observed_mask
from .estimation import fit_covariance
from .geometry import gram_operator
from .hedging import (
    CovarianceAmbiguity,
    _scalar,
    solve_plugin_hedge,
    solve_robust_hedge,
    solve_target_block_hedge,
)
from .hedging_statistics import gaussian_pairwise_bands
from .linalg import _real_array
from .observation_statistics import partial_kernel_from_arrays
from .observations import Observation


def _plain(value):
    """Keep arrays for the caller's serializer, but remove mapping proxies."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _record(
    result=None,
    *,
    reason=None,
    fit_diagnostics=None,
    bounds=None,
    risk_type=None,
    fitted_kernel=None,
    fitted_target_block=None,
):
    available = result is not None and bool(result.available)
    if reason is None and not available:
        reason = "hedge optimization unavailable"
        if result is not None:
            reason += f": {result.diagnostics.get('status', 'unverified numerical result')}"
    return {
        "positions": None if result is None else result.positions,
        "available": available,
        "reason": None if available else reason,
        "risk": None if result is None else result.risk,
        "objective": None if result is None else result.objective,
        "risk_type": risk_type,
        "fitted_kernel": fitted_kernel,
        "fitted_kernel_scope": "grid_covariance" if fitted_kernel is not None else None,
        "fitted_target_block": fitted_target_block,
        "fit_diagnostics": _plain(fit_diagnostics or {}),
        "hedge_diagnostics": {} if result is None else _plain(result.diagnostics),
        "training_observation_bounds": bounds,
    }


def _data_and_exposures(masked_values, evaluation_matrix, target, instrument):
    raw = np.asarray(masked_values)
    if raw.ndim != 2 or not all(raw.shape) or raw.dtype.kind not in "iuf":
        raise ValueError("masked_values must be a nonempty real sample-by-grid matrix")
    values = np.asarray(raw, dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError("only NaN may mark missing observations")
    if not np.any(np.isfinite(values)):
        raise ValueError("at least one training observation is required")
    evaluation = _real_array(evaluation_matrix, name="evaluation_matrix")
    size = values.shape[1]
    if evaluation.ndim != 2 or evaluation.shape[0] != size or not evaluation.shape[1]:
        raise ValueError("evaluation_matrix must have one row per observed grid coordinate")
    if target is None or instrument is None:
        if size != 7:
            raise ValueError("target and instrument must be supplied outside the seven-node pilot")
    default_target, default_instrument = np.zeros(size), np.zeros(size)
    if size == 7:
        default_target[[0, 1]], default_instrument[[5, 6]] = [1, -1], [1, -1]
    target = default_target if target is None else _real_array(target, name="target")
    instrument = (
        default_instrument if instrument is None else _real_array(instrument, name="instrument")
    )
    if target.shape != (size,) or instrument.shape != (size,):
        raise ValueError("target and instrument must be grid exposure vectors")
    return values, evaluation, target, instrument, default_target, default_instrument


def _contrast_data(values, contrast):
    """A transformed coordinate is observed only when its full support is."""
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    for index, row in enumerate(contrast):
        support = np.flatnonzero(row != 0)
        selected = np.all(finite[:, support], axis=1)
        result[selected, index] = values[np.ix_(selected, support)] @ row[support]
    return result


def fit_hedging_rules(
    masked_values: ArrayLike,
    *,
    evaluation_matrix: ArrayLike,
    source_factors,
    source_offset: ArrayLike | None,
    source_trace_cap: float = 2.0,
    cliques,
    noise_variance: float = 0.001,
    confidence_level: float = 0.95,
    relative_regularization: float = 0.001,
    numerical_error_budget: float = 1e-4,
    position_limit: float = 8.0,
    solver_tolerance: float = 1e-7,
    target: ArrayLike | None = None,
    instrument: ArrayLike | None = None,
    contrast_matrix: ArrayLike | None = None,
    ridge_trace_cap: float = 2.0,
    shrinkage_alpha: float | None = 0.1,
):
    """Fit canonical, ridge, robust and jointly observed direct hedge rules.

    The default target is e0-e1 and the sole instrument is e5-e6 on seven
    grid nodes. The default invertible contrast places those spreads in rows
    0 and 5 before constructing simultaneous Gaussian observed-entry bands.
    Custom exposures use identity contrasts unless contrast_matrix is given.

    Both robust models use exactly the same transformed observations/bands.
    The free PSD model bounds total grid covariance trace by a derived outer
    cap, while the source model bounds only Q trace and fixes source_offset.
    Ridge instead fits total innovation covariance V in H coordinates, with
    its separately declared ridge_trace_cap. These trace caps have different
    meanings. Point estimates are numerical selection assumptions; no source
    admissibility or finite-sample risk guarantee is inferred for them.

    Each method reports its own failures without replacing its estimator.
    The direct and optional fixed-alpha shrinkage methods require dates on
    which the entire target/instrument support was jointly measured.
    """
    values, evaluation, target, instrument, default_target, default_instrument = (
        _data_and_exposures(masked_values, evaluation_matrix, target, instrument)
    )
    size = values.shape[1]
    noise = _scalar(noise_variance, "noise_variance")
    relative = _scalar(relative_regularization, "relative_regularization", positive=True)
    budget = _scalar(numerical_error_budget, "numerical_error_budget", positive=True)
    limit = _scalar(position_limit, "position_limit")
    tolerance = _scalar(solver_tolerance, "solver_tolerance", positive=True)
    ridge_cap = _scalar(ridge_trace_cap, "ridge_trace_cap")
    if shrinkage_alpha is not None:
        shrinkage_alpha = _scalar(shrinkage_alpha, "shrinkage_alpha")
        if shrinkage_alpha > 1:
            raise ValueError("shrinkage_alpha must lie between zero and one")
    groups = tuple(tuple(group) for group in cliques)
    clique_mask = observed_mask(groups, size)
    base_source_model = CovarianceAmbiguity(
        source_factors, [], [], [], source_trace_cap, offset_covariance=source_offset
    )
    if base_source_model.output_dimension != size:
        raise ValueError("source_factors must output the observed grid coordinates")
    if contrast_matrix is None:
        contrast = np.eye(size)
        if (
            size == 7
            and np.array_equal(target, default_target)
            and np.array_equal(instrument, default_instrument)
        ):
            contrast[0], contrast[5] = target, instrument
    else:
        contrast = _real_array(contrast_matrix, name="contrast_matrix")
    if contrast.shape != (size, size) or np.linalg.matrix_rank(contrast) != size:
        raise ValueError("contrast_matrix must be an invertible square grid-coordinate map")
    transformed = _contrast_data(values, contrast)
    transformed_noise = noise * contrast @ contrast.T
    bands = gaussian_pairwise_bands(
        transformed, noise_covariance=transformed_noise, confidence_level=confidence_level
    )
    bounds = {
        "entry_pairs": bands.entry_pairs,
        "lower": bands.lower,
        "upper": bands.upper,
        "pair_counts": bands.pair_counts,
        "corrected_covariance": bands.corrected_covariance,
        "confidence_level": bands.confidence_level,
        "contrast_matrix": contrast,
        "noise_covariance": transformed_noise,
        "diagnostics": _plain(bands.diagnostics),
        "scope": "simultaneous transformed-entry bands under the declared Gaussian experiment",
    }
    finite = np.isfinite(values)
    nonempty = np.any(finite, axis=1)
    nodes = tuple(np.flatnonzero(row) for row in finite[nonempty])
    samples = tuple(row[index] for row, index in zip(values[nonempty], nodes, strict=True))
    statistics = partial_kernel_from_arrays(samples, nodes, noise_variance=noise, size=size)
    metadata = {
        "training_rows": len(values),
        "nonempty_training_rows": int(np.count_nonzero(nonempty)),
        "target": target,
        "instrument": instrument,
        "position_limit": limit,
        "noise_variance": noise,
        "pair_counts": statistics.pair_counts,
        "means": "specified zero; not estimated",
        "uses_latent_truth": False,
        "uses_future_outcomes": False,
    }
    bounds_positions = ([-limit], [limit])
    methods = {}

    try:
        if not np.all(statistics.observed_mask[clique_mask]):
            raise ValueError("declared clique entries lack observed training pairs")
        partial = np.where(clique_mask, statistics.partial_kernel, np.nan)
        pair_weights = np.where(clique_mask, statistics.pair_weight_sums, 0.0)
        regularization = relative * float(np.max(pair_weights))
        coherent = fit_coherent_partial(
            partial,
            pair_weights=pair_weights,
            cliques=groups,
            regularization=regularization,
            numerical_error_budget=budget,
        )
        diagnostics = {
            **metadata,
            "coherent": _plain(coherent.diagnostics),
            "relative_regularization": relative,
            "regularization": regularization,
            "full_numerical_support": True,
            "hard_rank_constraint": None,
        }
        if not coherent.converged:
            methods["canonical_coherent"] = _record(
                reason="coherent partial fit did not meet its numerical error budget",
                fit_diagnostics=diagnostics,
                bounds=bounds,
            )
        else:
            completion = complete_kernel(coherent.partial_kernel, cliques=groups, population=True)
            diagnostics["completion"] = _plain(completion.diagnostics)
            if not completion.available or not completion.diagnostics["output_psd"]:
                methods["canonical_coherent"] = _record(
                    reason="full-support canonical completion unavailable or numerically indefinite",
                    fit_diagnostics=diagnostics,
                    bounds=bounds,
                )
            else:
                point = solve_plugin_hedge(
                    completion.kernel, target, instrument[:, None], position_bounds=bounds_positions
                )
                methods["canonical_coherent"] = _record(
                    point,
                    fit_diagnostics=diagnostics,
                    bounds=bounds,
                    risk_type="point_covariance",
                    fitted_kernel=completion.kernel,
                )
    except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as error:
        methods["canonical_coherent"] = _record(
            reason=f"canonical coherent fit failed: {error}",
            fit_diagnostics=metadata,
            bounds=bounds,
        )

    try:
        observations = tuple(
            Observation(float(i), sample, evaluation[index], noise * np.eye(len(index)))
            for i, (sample, index) in enumerate(zip(samples, nodes, strict=True))
        )
        date_weights = np.full(len(observations), 1 / len(observations))
        gram = gram_operator([item.operator for item in observations], date_weights)
        gram_norm = float(np.linalg.eigvalsh(gram)[-1])
        if gram_norm <= 0:
            raise ValueError("observation geometry has zero covariance Gram norm")
        regularization = relative * gram_norm
        fit = fit_covariance(
            observations,
            regularization=regularization,
            trace_cap=ridge_cap,
            weights=date_weights,
            numerical_error_budget=budget,
            relative_tolerance=0,
            max_iterations=30_000,
        )
        diagnostics = {
            **metadata,
            "status": fit.status,
            "converged": fit.converged,
            "iterations": fit.iterations,
            "covariance_error_bound": fit.covariance_error_bound,
            "objective_gap_bound": fit.objective_gap_bound,
            "gram_operator_norm": gram_norm,
            "relative_regularization": relative,
            "regularization": regularization,
            "trace_cap": ridge_cap,
            "trace_cap_scope": "total innovation V in orthonormal H coordinates",
            "source_trace_cap_equivalent": False,
        }
        if not fit.converged:
            methods["ridge"] = _record(
                reason="ridge H fit did not meet its numerical error budget",
                fit_diagnostics=diagnostics,
                bounds=bounds,
            )
        else:
            kernel = evaluation @ fit.covariance @ evaluation.T
            point = solve_plugin_hedge(
                kernel, target, instrument[:, None], position_bounds=bounds_positions
            )
            methods["ridge"] = _record(
                point,
                fit_diagnostics=diagnostics,
                bounds=bounds,
                risk_type="point_covariance",
                fitted_kernel=kernel,
            )
    except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as error:
        methods["ridge"] = _record(
            reason=f"ridge fit failed: {error}", fit_diagnostics=metadata, bounds=bounds
        )

    transformed_factors = tuple(contrast @ factor for factor in base_source_model.factors)
    transformed_offset = contrast @ base_source_model.offset_covariance @ contrast.T
    source_gram = sum(factor.T @ factor for factor in transformed_factors)
    source_map_trace_norm = float(np.linalg.eigvalsh(source_gram)[-1])
    outer_trace_cap = source_trace_cap * source_map_trace_norm + float(np.trace(transformed_offset))
    transformed_target = np.linalg.solve(contrast.T, target)
    transformed_instrument = np.linalg.solve(contrast.T, instrument)
    for method in ("robust_psd", "robust_source"):
        diagnostics = {
            **metadata,
            "source_trace_cap": source_trace_cap,
            "source_map_trace_norm": source_map_trace_norm,
            "outer_grid_trace_cap": outer_trace_cap,
            "shared_observation_bands": True,
            "calibrated_bands": True,
            "source_class_is_assumed": method == "robust_source",
            "fixed_known_background": method == "robust_source",
        }
        try:
            if method == "robust_source":
                model = CovarianceAmbiguity(
                    transformed_factors,
                    bands.entry_pairs,
                    bands.lower,
                    bands.upper,
                    source_trace_cap,
                    calibrated_bands=True,
                    offset_covariance=transformed_offset,
                )
                diagnostics["trace_cap_scope"] = (
                    "finite uncertain source Q in orthonormal coordinates"
                )
                diagnostics["fixed_background_covariance"] = transformed_offset
                diagnostics["source_dimension"] = model.source_dimension
            else:
                model = CovarianceAmbiguity(
                    (np.eye(size),),
                    bands.entry_pairs,
                    bands.lower,
                    bands.upper,
                    outer_trace_cap,
                    calibrated_bands=True,
                )
                diagnostics["trace_cap_scope"] = (
                    "total contrast-grid covariance K; derived outer cap"
                )
            result = solve_robust_hedge(
                model,
                transformed_target,
                transformed_instrument[:, None],
                position_bounds=bounds_positions,
                tolerance=tolerance,
            )
            attempts = [dict(result.diagnostics)]
            if not result.available and result.diagnostics.get("status") in (
                "optimal",
                "optimal_inaccurate",
                "solver_error",
                "ambiguity_feasibility_unverified",
            ):
                result = solve_robust_hedge(
                    model,
                    transformed_target,
                    transformed_instrument[:, None],
                    position_bounds=bounds_positions,
                    tolerance=tolerance / 10,
                    max_iterations=300_000,
                )
                attempts.append(dict(result.diagnostics))
            if not result.available:
                result = solve_robust_hedge(
                    model,
                    transformed_target,
                    transformed_instrument[:, None],
                    position_bounds=bounds_positions,
                    tolerance=tolerance / 10,
                    solver="CLARABEL",
                    max_iterations=1000,
                )
                attempts.append(dict(result.diagnostics))
            diagnostics["solver_attempts"] = attempts
            methods[method] = _record(
                result,
                fit_diagnostics=diagnostics,
                bounds=bounds,
                risk_type="numerical_worst_case_candidate",
            )
        except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as error:
            methods[method] = _record(
                reason=f"{method} fit failed: {error}", fit_diagnostics=diagnostics, bounds=bounds
            )

    support = np.flatnonzero((target != 0) | (instrument != 0))
    jointly_observed = np.all(finite[:, support], axis=1)
    joint_count = int(np.count_nonzero(jointly_observed))
    direct_diagnostics = {
        **metadata,
        "joint_training_rows": joint_count,
        "joint_support": support,
        "joint_covariance_required": True,
    }
    direct_methods = ["direct_target_block"]
    if shrinkage_alpha is not None:
        direct_methods.append("empirical_shrinkage")
    if not joint_count:
        for method in direct_methods:
            methods[method] = _record(
                reason="no training dates jointly observe the full target and instrument support",
                fit_diagnostics=direct_diagnostics,
                bounds=bounds,
            )
    else:
        exposure_block = np.column_stack([target[support], instrument[support]])
        joint_samples = values[np.ix_(jointly_observed, support)] @ exposure_block
        corrected = (
            joint_samples.T @ joint_samples / joint_count
            - noise * exposure_block.T @ exposure_block
        )
        for method in direct_methods:
            diagnostics = {**direct_diagnostics, "corrected_joint_covariance": corrected}
            joint_covariance = corrected
            if method == "empirical_shrinkage":
                joint_covariance = (1 - shrinkage_alpha) * corrected + shrinkage_alpha * np.diag(
                    np.diag(corrected)
                )
                diagnostics["shrinkage_alpha"] = shrinkage_alpha
                diagnostics["selection"] = (
                    "fixed declared shrinkage toward the joint block diagonal"
                )
            try:
                result = solve_target_block_hedge(
                    joint_covariance[0, 0],
                    [joint_covariance[0, 1]],
                    joint_covariance[1:, 1:],
                    position_bounds=bounds_positions,
                )
                methods[method] = _record(
                    result,
                    fit_diagnostics=diagnostics,
                    bounds=bounds,
                    risk_type="jointly_observed_point_covariance",
                    fitted_target_block=joint_covariance,
                )
            except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as error:
                methods[method] = _record(
                    reason=f"joint target block unavailable: {error}",
                    fit_diagnostics=diagnostics,
                    bounds=bounds,
                )
    return methods
