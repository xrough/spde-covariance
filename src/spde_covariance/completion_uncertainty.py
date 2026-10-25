"""Gaussian confidence regions without a canonical-completion assumption.

Independent zero-mean latent vectors share one covariance. Deterministic point
masks and independent fresh Gaussian noise, with known per-date covariance,
define the experiment. Exact scalar variance intervals and a union bound supply
calibration; numerical SDP endpoints are not exact coverage certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.stats import chi2

from .fragments import _row_indices
from .hedging import (
    CovarianceAmbiguity,
    CovarianceFunctionalResult,
    _cvxpy,
    _snapshot,
    _solver_options,
)
from .linalg import validate_psd


@dataclass(frozen=True, slots=True, eq=False)
class GaussianCompletionRegion:
    """Simultaneous observed-entry bands and an ordinary PSD ambiguity model.

    Arrays use raw grid-value covariance coordinates. NaN denotes an unobserved
    entry. ``available`` means finite consistent bands and a data-derived finite
    trace cap were constructed; global PSD feasibility is checked by a requested
    risk SDP, not asserted by this class. No band is clipped or widened.
    """

    lower: np.ndarray
    upper: np.ndarray
    observed_mask: np.ndarray
    pair_counts: np.ndarray
    corrected_covariance: np.ndarray
    confidence_level: float
    ambiguity: CovarianceAmbiguity | None
    diagnostics: dict

    def __post_init__(self):
        for field in (
            "lower",
            "upper",
            "observed_mask",
            "pair_counts",
            "corrected_covariance",
            "diagnostics",
        ):
            object.__setattr__(self, field, _snapshot(getattr(self, field)))

    @property
    def available(self):
        return self.diagnostics["available"]

    @property
    def reason(self):
        return self.diagnostics["reason"]

    @property
    def entry_pairs(self):
        return _snapshot(np.column_stack(np.where(np.triu(self.observed_mask))))

    @property
    def size(self):
        return len(self.lower)


def _input_data(values, row_to_nodes, noise_covariances, size):
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    rows = _row_indices(row_to_nodes, size=size)
    # A full-grid ndarray is explicitly indexed. Unobserved fillers are never read.
    dense = isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] == size
    if dense:
        if len(values) != len(rows):
            raise ValueError("values must match the observation count")
        samples = tuple(values[i, row] for i, row in enumerate(rows))
    else:
        try:
            samples = tuple(values)
        except TypeError as error:
            raise ValueError("values must contain one observed vector per row map") from error
    if len(samples) != len(rows):
        raise ValueError("values must match the observation count")
    data = []
    for value, row in zip(samples, rows, strict=True):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.shape != row.shape or not np.all(np.isfinite(raw)):
            raise ValueError("each observed value must be finite, real and match its row map")
        data.append(np.asarray(raw, dtype=float))
    if isinstance(noise_covariances, Real) and not isinstance(noise_covariances, (bool, np.bool_)):
        raw_noise = (noise_covariances,) * len(rows)
    elif isinstance(noise_covariances, np.ndarray) and noise_covariances.ndim == 2:
        if noise_covariances.shape != (size, size):
            raise ValueError("a common noise matrix must have the full grid shape")
        raw_noise = tuple(noise_covariances[np.ix_(row, row)] for row in rows)
    else:
        try:
            raw_noise = tuple(noise_covariances)
        except TypeError as error:
            raise ValueError(
                "noise_covariances must give one covariance per observed vector"
            ) from error
    if len(raw_noise) != len(rows):
        raise ValueError("noise_covariances must match the observation count")
    noises = []
    for noise, row in zip(raw_noise, rows, strict=True):
        if isinstance(noise, Real) and not isinstance(noise, (bool, np.bool_)):
            if not np.isfinite(noise) or noise < 0:
                raise ValueError("noise variances must be finite and nonnegative")
            noise = float(noise) * np.eye(len(row))
        matrix = validate_psd(noise, name="known Gaussian noise covariance")
        if matrix.shape != (len(row), len(row)):
            raise ValueError("noise covariance must match its observed row order")
        noises.append(matrix)
    return tuple(data), rows, tuple(noises)


def gaussian_completion_region(
    values, row_to_nodes, *, noise_covariances, size, confidence_level=0.95
):
    """Construct identification-inclusive simultaneous Gaussian covariance bands.

    Supply row-order observed vectors and row-order noise matrices. Alternatively,
    a 2D ndarray with ``size`` columns contains full-grid values; only declared
    rows are read. A common full-grid noise ndarray or common scalar variance is
    also accepted. Each scalar node/sum/difference probe is partitioned by exact
    equality of its supplied known projected-noise variance. Every resulting
    group consumes one interval in a single global Bonferroni allocation.

    Within each group, sum(y**2)/total_variance has a chi-square distribution
    with the actual sample count (mean specified zero). Subtracting known noise
    and intersecting group intervals bounds the common signal-probe variance.
    Polarization then bounds jointly observed covariance entries. No canonical
    graph, latent truth, estimated mean or estimated noise enters calibration.
    """
    if (
        isinstance(confidence_level, (bool, np.bool_))
        or not isinstance(confidence_level, Real)
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must lie strictly between zero and one")
    samples, rows, noises = _input_data(values, row_to_nodes, noise_covariances, size)
    counts = np.zeros((size, size), dtype=np.int64)
    corrected_sum = np.zeros((size, size))
    probes = {}
    for date, (sample, row, noise) in enumerate(zip(samples, rows, noises, strict=True)):
        counts[np.ix_(row, row)] += 1
        corrected_sum[np.ix_(row, row)] += np.outer(sample, sample) - noise
        positions = {int(node): position for position, node in enumerate(row)}
        for j in sorted(positions):
            a = positions[j]
            probes.setdefault(("node", j, j), []).append(
                (date, float(sample[a]), float(noise[a, a]))
            )
            for k in sorted(node for node in positions if node > j):
                b = positions[k]
                for kind, sign in (("sum", 1), ("difference", -1)):
                    variance = float(noise[a, a] + noise[b, b] + 2 * sign * noise[a, b])
                    if variance < 0:
                        raise ValueError("known noise must have nonnegative scalar-probe variances")
                    probes.setdefault((kind, j, k), []).append(
                        (date, float(sample[a] + sign * sample[b]), variance)
                    )
    grouped = {}
    for key, observations in probes.items():
        groups = {}
        for date, value, variance in observations:
            if variance < 0:
                raise ValueError("known noise must have nonnegative scalar-probe variances")
            groups.setdefault(variance, []).append((date, value))
        grouped[key] = groups
    interval_count = sum(len(groups) for groups in grouped.values())
    tail = (1 - float(confidence_level)) / (2 * interval_count)
    scalar_intervals, probe_intervals, failures = [], {}, []
    for (kind, j, k), groups in sorted(grouped.items()):
        intervals = []
        for noise_variance, observations in sorted(groups.items()):
            values = np.asarray([value for _, value in observations])
            total = float(values @ values)
            count = len(values)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                lower = float(total / chi2.isf(tail, count) - noise_variance)
                upper = float(total / chi2.ppf(tail, count) - noise_variance)
            scalar_intervals.append(
                {
                    "probe": kind,
                    "nodes": (j, k),
                    "noise_variance": noise_variance,
                    "count": count,
                    "sum_squares": total,
                    "lower": lower,
                    "upper": upper,
                    "date_indices": tuple(date for date, _ in observations),
                }
            )
            intervals.append((lower, upper))
        lower = max(interval[0] for interval in intervals)
        upper = min(interval[1] for interval in intervals)
        if not np.isfinite(lower) or not np.isfinite(upper):
            failures.append(f"nonfinite scalar interval for {kind}({j},{k})")
        elif lower > upper:
            failures.append(f"empty noise-group intersection for {kind}({j},{k})")
        elif upper < 0:
            failures.append(f"negative upper signal variance for {kind}({j},{k})")
        probe_intervals[kind, j, k] = (lower, upper)
    observed = counts > 0
    lower, upper, corrected = (np.full((size, size), np.nan) for _ in range(3))
    corrected[observed] = corrected_sum[observed] / counts[observed]
    for j, k in zip(*np.where(np.triu(observed)), strict=True):
        if j == k:
            bounds = probe_intervals["node", j, j]
        else:
            plus, minus = probe_intervals["sum", j, k], probe_intervals["difference", j, k]
            bounds = ((plus[0] - minus[1]) / 4, (plus[1] - minus[0]) / 4)
        lower[j, k] = lower[k, j] = bounds[0]
        upper[j, k] = upper[k, j] = bounds[1]
    missing_nodes = tuple(np.flatnonzero(np.diag(counts) == 0))
    if missing_nodes:
        failures.append(
            f"unobserved nodes prevent a finite data-derived trace cap: {missing_nodes}"
        )
    trace_cap = None
    if not failures:
        diagonal = np.diag(upper)
        trace_cap = float(np.sum(diagonal))
        if not np.isfinite(trace_cap):
            failures.append("diagonal upper bounds give a nonfinite trace cap")
        # Necessary PSD checks diagnose obvious empty sets without altering bands.
        for j, k in zip(*np.where(np.triu(observed, k=1)), strict=True):
            maximum = np.sqrt(diagonal[j]) * np.sqrt(diagonal[k])
            if lower[j, k] > maximum or upper[j, k] < -maximum:
                failures.append(f"entry ({j},{k}) contradicts PSD diagonal upper bounds")
    ambiguity = None
    if not failures:
        pairs = np.column_stack(np.where(np.triu(observed)))
        ambiguity = CovarianceAmbiguity(
            (np.eye(size),),
            pairs,
            lower[tuple(pairs.T)],
            upper[tuple(pairs.T)],
            trace_cap,
            calibrated_bands=True,
        )
    return GaussianCompletionRegion(
        lower,
        upper,
        observed,
        counts,
        corrected,
        float(confidence_level),
        ambiguity,
        {
            "available": ambiguity is not None,
            "reason": "; ".join(failures) if failures else None,
            "method": "chi-square scalar variance groups + polarization + global Bonferroni",
            "scalar_intervals": interval_count,
            "scalar_probes": len(grouped),
            "one_sided_tail_probability": tail,
            "per_group_miscoverage": 2 * tail,
            "allocated_miscoverage": 2 * tail * interval_count,
            "scalar_group_intervals": scalar_intervals,
            "probe_intervals": tuple(
                {"probe": key[0], "nodes": key[1:], "lower": value[0], "upper": value[1]}
                for key, value in probe_intervals.items()
            ),
            "noise_grouping": "exact supplied scalar projected-noise variance equality; no approximate pooling",
            "means": "specified zero; not estimated",
            "noise": "known fresh independent Gaussian noise, possibly heterogeneous/correlated within date",
            "assumptions": "independent common centered Gaussian latent covariance; deterministic masks; noise independent across dates and of latent vectors",
            "canonical_assumption": False,
            "analytic_region_calibration": True,
            "empirical_coverage_verified": False,
            "psd_feasibility_status": "not solved; necessary conditions only"
            if ambiguity is not None
            else "unavailable",
            "trace_cap": trace_cap,
            "trace_cap_scope": "sum of finite nonnegative raw-grid diagonal upper bands; not an arbitrary cap or H/source trace",
            "interval_scope": "finite-sample identification-inclusive confidence region under declared Gaussian assumptions",
            "numerical_quantile_arithmetic": "floating point, not interval arithmetic",
        },
    )


@dataclass(frozen=True, slots=True, eq=False)
class CompletionRiskInterval:
    """Outer dual bounds, separately labelled inner primal risk witnesses.

    Dual feasibility is repaired and checked in floating point. This avoids
    mistaking attainable primal endpoints for outer confidence bounds, while
    retaining the limitation that no interval-arithmetic proof is computed.
    """

    lower: float | None
    upper: float | None
    inner_lower: float | None
    inner_upper: float | None
    minimizer: CovarianceFunctionalResult
    maximizer: CovarianceFunctionalResult
    diagnostics: dict

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", _snapshot(self.diagnostics))

    @property
    def available(self):
        return self.diagnostics["available"]


def _repair_identity_dual(objective, model, trace_dual, lower_dual, upper_dual):
    """Repair a dual upper bound for max <objective,K> over ordinary PSD K.

    Off-diagonal entry constraints have symmetric adjoint coefficients gamma/2.
    Increasing the trace multiplier by the negative slack eigenvalue restores
    the PSD dual constraint without altering the statistical confidence set.
    """
    alpha = np.asarray(lower_dual, dtype=float)
    beta = np.asarray(upper_dual, dtype=float)
    if alpha.shape != model.lower.shape or beta.shape != model.upper.shape:
        raise ValueError("dual multipliers must match observed entries")
    if (
        not np.all(np.isfinite(alpha))
        or not np.all(np.isfinite(beta))
        or not np.isfinite(trace_dual)
    ):
        raise ValueError("dual multipliers must be finite")
    correction = max(
        float(np.max(-alpha, initial=0)),
        float(np.max(-beta, initial=0)),
        max(0.0, -float(trace_dual)),
    )
    alpha, beta, trace_dual = np.maximum(alpha, 0), np.maximum(beta, 0), max(0.0, float(trace_dual))
    adjoint = np.zeros_like(objective)
    for (j, k), coefficient in zip(model.entry_pairs, beta - alpha, strict=True):
        if j == k:
            adjoint[j, j] += coefficient
        else:
            adjoint[j, k] += coefficient / 2
            adjoint[k, j] += coefficient / 2
    slack = trace_dual * np.eye(len(objective)) + adjoint - objective
    minimum = float(np.linalg.eigvalsh(slack)[0])
    scale = max(
        1.0, abs(trace_dual), float(np.linalg.norm(adjoint, 2)), float(np.linalg.norm(objective, 2))
    )
    margin = 64 * np.finfo(float).eps * len(objective) * scale
    shift = max(0.0, -minimum) + margin
    repaired_trace = trace_dual + shift
    repaired_slack = repaired_trace * np.eye(len(objective)) + adjoint - objective
    repaired_minimum = float(np.linalg.eigvalsh(repaired_slack)[0])
    raw_bound = float(model.trace_cap * repaired_trace + model.upper @ beta - model.lower @ alpha)
    summation_scale = (
        abs(model.trace_cap * repaired_trace)
        + float(np.abs(model.upper) @ beta)
        + float(np.abs(model.lower) @ alpha)
    )
    summation_margin = 64 * np.finfo(float).eps * max(summation_scale, np.finfo(float).tiny)
    bound = raw_bound + summation_margin
    return {
        "available": bool(np.isfinite(bound) and repaired_minimum >= 0),
        "upper_bound": bound,
        "raw_dual_objective": raw_bound,
        "objective_summation_roundoff_margin": summation_margin,
        "trace_multiplier": repaired_trace,
        "lower_multipliers": alpha,
        "upper_multipliers": beta,
        "nonnegative_multiplier_correction": correction,
        "trace_multiplier_psd_repair": shift,
        "slack_roundoff_margin": margin,
        "minimum_slack_eigenvalue_before_repair": minimum,
        "minimum_slack_eigenvalue": repaired_minimum,
        "dual_slack": repaired_slack,
        "weak_duality_scope": "ordinary PSD K, observed entry bands and data-derived trace cap",
        "certificate_arithmetic": "floating-point feasibility checks and PSD repair; not interval arithmetic",
    }


def _maximize_region(model, objective, *, solver, tolerance, max_iterations, feasibility_tolerance):
    cp = _cvxpy()
    options = _solver_options(solver, tolerance, max_iterations)
    covariance = cp.Variable(objective.shape, symmetric=True)
    entries = cp.hstack([covariance[j, k] for j, k in model.entry_pairs])
    trace_constraint = cp.trace(covariance) <= model.trace_cap
    lower_constraint, upper_constraint = entries >= model.lower, entries <= model.upper
    problem = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(objective, covariance))),
        [covariance >> 0, trace_constraint, lower_constraint, upper_constraint],
    )
    try:
        problem.solve(**options)
    except cp.error.SolverError as error:
        return (
            None,
            None,
            {
                "available": False,
                "status": "solver_error",
                "error": str(error),
                "solver": solver,
                "numerical_certificate": False,
            },
        )
    candidate = None if covariance.value is None else (covariance.value + covariance.value.T) / 2
    residuals = None if candidate is None else model.residuals(candidate)
    primal_value = None if candidate is None else float(np.sum(objective * candidate))
    dual = None
    if all(
        constraint.dual_value is not None
        for constraint in (trace_constraint, lower_constraint, upper_constraint)
    ):
        try:
            dual = _repair_identity_dual(
                objective,
                model,
                trace_constraint.dual_value,
                lower_constraint.dual_value,
                upper_constraint.dual_value,
            )
        except (ValueError, np.linalg.LinAlgError):
            dual = None
    gap = None if primal_value is None or dual is None else dual["upper_bound"] - primal_value
    available = bool(
        problem.status == cp.OPTIMAL
        and residuals is not None
        and residuals["scaled_max_violation"] <= feasibility_tolerance
        and primal_value is not None
        and np.isfinite(primal_value)
        and dual is not None
        and dual["available"]
        and gap >= -feasibility_tolerance * (1 + abs(primal_value))
    )
    return (
        candidate,
        primal_value,
        {
            "available": available,
            "status": problem.status,
            "solver": solver,
            "feasibility": residuals,
            "feasibility_tolerance": feasibility_tolerance,
            "dual": dual,
            "primal_dual_gap": gap,
            "primal_objective": primal_value,
            "solver_objective": problem.value,
            "numerical_certificate": False,
            "interpretation": "numerically checked primal witness and repaired weak-duality outer bound",
        },
    )


def completion_risk_interval(
    region,
    exposure,
    *,
    solver="CLARABEL",
    tolerance=1e-8,
    max_iterations=100000,
    feasibility_tolerance=1e-6,
):
    """Bound risk using repaired dual outer endpoints, not primal extrema alone.

    Two ordinary PSD SDPs retain primal witnesses and their gaps. Nonnegative
    entry/trace dual multipliers are checked and their PSD slack is repaired by
    a disclosed trace-multiplier increase. No statistical band is widened or
    clipped to accommodate an estimate. Empty/inaccurate solves are unavailable.
    """
    if not isinstance(region, GaussianCompletionRegion):
        raise TypeError("region must be GaussianCompletionRegion")
    vector = np.asarray(exposure)
    if (
        vector.dtype.kind not in "iuf"
        or vector.shape != (region.size,)
        or not np.all(np.isfinite(vector))
    ):
        raise ValueError("exposure must be a finite real vector matching the region")
    scope = {
        "input_confidence_level": region.confidence_level,
        "canonical_assumption": False,
        "analytic_region_calibration": True,
        "numerical_endpoint_coverage_certificate": False,
        "outer_bounds": True,
        "interpretation": "finite-sample identification-inclusive variance region; repaired numerical dual outer bounds under declared Gaussian assumptions",
    }
    if not region.available:
        result = CovarianceFunctionalResult(
            None,
            None,
            None,
            {
                "available": False,
                "status": "region_unavailable",
                "reason": region.reason,
                "numerical_certificate": False,
            },
        )
        return CompletionRiskInterval(
            None,
            None,
            None,
            None,
            result,
            result,
            {
                "available": False,
                "status": "region_unavailable",
                "reason": region.reason,
                "numerical_certificate": False,
                **scope,
            },
        )
    if (
        not isinstance(feasibility_tolerance, Real)
        or isinstance(feasibility_tolerance, (bool, np.bool_))
        or not np.isfinite(feasibility_tolerance)
        or feasibility_tolerance <= 0
    ):
        raise ValueError("feasibility_tolerance must be finite and positive")
    model = region.ambiguity
    if (
        len(model.factors) != 1
        or not np.array_equal(model.factors[0], np.eye(region.size))
        or np.any(model.offset_covariance)
        or model.trace_lower != 0
    ):
        raise ValueError("risk bounds require an ordinary identity-factor PSD ambiguity region")
    objective = np.outer(np.asarray(vector, dtype=float), np.asarray(vector, dtype=float))
    options = dict(
        solver=solver,
        tolerance=tolerance,
        max_iterations=max_iterations,
        feasibility_tolerance=feasibility_tolerance,
    )
    small, small_value, small_diagnostic = _maximize_region(model, -objective, **options)
    large, large_value, large_diagnostic = _maximize_region(model, objective, **options)
    small_diagnostic["risk_objective_sign"] = -1
    large_diagnostic["risk_objective_sign"] = 1
    minimum = CovarianceFunctionalResult(
        None if small_value is None else -small_value, small, small, small_diagnostic
    )
    maximum = CovarianceFunctionalResult(large_value, large, large, large_diagnostic)
    raw_lower = -small_diagnostic["dual"]["upper_bound"] if minimum.available else None
    # This is the analytic PSD variance bound, not a change to empirical bands.
    lower = max(0.0, raw_lower) if raw_lower is not None else None
    upper = large_diagnostic["dual"]["upper_bound"] if maximum.available else None
    available = minimum.available and maximum.available and lower <= upper
    return CompletionRiskInterval(
        lower,
        upper,
        minimum.value if minimum.available else None,
        maximum.value if maximum.available else None,
        minimum,
        maximum,
        {
            "available": bool(available),
            "raw_dual_lower": raw_lower,
            "analytic_psd_variance_lower": 0.0,
            "status": "available" if available else "numerical_region_bounds_unavailable",
            "reason": None
            if available
            else "a primal/dual check failed or the outer endpoints are reversed",
            "numerical_certificate": False,
            **scope,
            "region_psd_feasibility_status": tuple(
                x.diagnostics["status"] for x in (minimum, maximum)
            ),
        },
    )
