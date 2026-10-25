"""Population hedge ranges for nested observations of one fixed source family.

This study uses exact synthetic covariance entries, not sampling bands. The
finite unknown PSD source block has known trace and a common fixed tail offset.
All observation designs use the same population covariance and source family.
Numerical SDP ranges are conditional on those explicit restrictions.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .admissibility import _scalar
from .completion import complete_kernel
from .hedging import CovarianceAmbiguity, covariance_functional_interval
from .hedging_controls import free_psd_clique_hedge_interval
from .linalg import validate_psd
from .trajectories import _metadata_snapshot


def _mask(cliques, dimension):
    mask = np.zeros((dimension, dimension), dtype=bool)
    for clique in cliques:
        mask[np.ix_(clique, clique)] = True
    return mask


def _scalar_decisions(interval, target_variance, hedge_variance):
    lower, upper = interval
    midpoint = float((lower + upper) / 2)
    variance_position = float(np.clip(0.0, lower, upper))
    cross = hedge_variance * np.asarray(interval)
    endpoint_risks = (
        target_variance + hedge_variance * variance_position**2 - 2 * variance_position * cross
    )
    return {
        "minimax_regret_hedge": midpoint,
        "minimax_regret": float(hedge_variance * (upper - lower) ** 2 / 4),
        "minimax_variance_hedge": variance_position,
        "minimax_variance": float(np.max(endpoint_risks)),
        "hedge_interval_width": float(upper - lower),
        "scope": "one unconstrained scalar hedge; fixed observed target and hedge variances",
    }


def _support_snapshot(result):
    return {
        "available": result.available,
        "candidate_value": result.value,
        "source_covariance": result.source_covariance,
        "covariance": result.covariance,
        "diagnostics": result.diagnostics,
    }


def _source_interval(model, target, hedge, target_variance, hedge_variance, tolerance):
    try:
        interval = covariance_functional_interval(
            model,
            target,
            hedge,
            solver="CLARABEL",
            tolerance=tolerance,
            feasibility_tolerance=max(10 * tolerance, 1e-7),
            max_iterations=500,
        )
    except ImportError as error:
        return {"available": False, "reason": str(error), "solver": "CLARABEL"}
    ordered = bool(
        interval.available
        and interval.lower is not None
        and interval.upper is not None
        and interval.lower <= interval.upper
    )
    functional = (np.outer(target, hedge) + np.outer(hedge, target)) / 2
    observed = np.zeros_like(functional, dtype=bool)
    observed_value = 0.0
    for (left, right), lower, upper in zip(
        model.entry_pairs,
        model.lower,
        model.upper,
        strict=True,
    ):
        if lower == upper:
            observed[left, right] = observed[right, left] = True
            observed_value += functional[left, right] * lower * (1 if left == right else 2)
    identified = bool(np.all(observed[functional != 0]))
    usable = bool(interval.available and (ordered or identified))
    result = {
        "available": usable,
        "diagnostics": interval.diagnostics,
        "interval_order_valid": ordered,
        "observed_functional_identified": identified,
        "endpoint_policy": "observed functional value with successful source witnesses"
        if identified
        else "numerical support endpoints",
        "minimizer": _support_snapshot(interval.minimizer),
        "maximizer": _support_snapshot(interval.maximizer),
        "cross_covariance_interval": None,
        "hedge_interval": None,
        "scalar_decisions": None,
        "scope": "PSD finite source block, known source trace, same fixed tail offset",
        "numerical_error_certificate": False,
    }
    if usable:
        # Complete observed functionals are fixed algebraically; support-solver
        # roundoff must not create a tiny reversed interval or artificial regret.
        cross = (
            np.repeat(observed_value, 2)
            if identified
            else np.array([interval.lower, interval.upper])
        )
        hedge_interval = cross / hedge_variance
        result.update(
            {
                "cross_covariance_interval": cross,
                "hedge_interval": hedge_interval,
                "scalar_decisions": _scalar_decisions(
                    hedge_interval,
                    target_variance,
                    hedge_variance,
                ),
                "endpoint_known_variance_mismatches": np.array(
                    [
                        [
                            target @ witness.covariance @ target - target_variance,
                            hedge @ witness.covariance @ hedge - hedge_variance,
                        ]
                        for witness in (interval.minimizer, interval.maximizer)
                    ]
                ),
            }
        )
    return result


def _canonical_estimate(covariance, target, hedge, cliques, mask):
    partial = np.full_like(covariance, np.nan)
    partial[mask] = covariance[mask]
    completed = complete_kernel(partial, cliques=cliques, population=True)
    result = {
        "available": completed.available,
        "kernel": completed.kernel,
        "diagnostics": completed.diagnostics,
        "hedge": None,
        "population_residual_variance": None,
        "source_compatibility": "unverified; canonical PSD completion is not a source witness",
    }
    if completed.available:
        variance = float(hedge @ completed.kernel @ hedge)
        if variance <= 0:
            result.update({"available": False, "reason": "nonpositive canonical hedge variance"})
        else:
            position = float(target @ completed.kernel @ hedge / variance)
            residual = target - position * hedge
            result.update(
                {
                    "hedge": position,
                    "population_residual_variance": float(residual @ covariance @ residual),
                }
            )
    return result


def run_population_hedge_study(control, *, solver_tolerance: float = 1e-8):
    """Compare source and free PSD ranges after adding observable maturities.

    ``control`` is the result of ``fixed_power_hedge_control``. The baseline,
    add-grid-index-6-to-left, add-grid-index-0-to-right and complete designs
    share its baseline covariance, transport factors, source trace and offset.
    Every source endpoint retains its numerical witness and solver diagnostics.
    Failed or inaccurate solves have no usable interval or hedge decision.

    Minimax variance chooses the position nearest zero in the hedge interval.
    Minimax regret chooses its midpoint; regret equals V_b * width**2 / 4.
    These formulas assume an unconstrained scalar hedge with no trading costs.
    """
    if not isinstance(control, Mapping):
        raise ValueError("control must be a fixed-power control mapping")
    tolerance = _scalar(solver_tolerance, "solver_tolerance", positive=True)
    covariance = validate_psd(control["baseline_covariance"], name="baseline_covariance")
    if covariance.shape != (7, 7):
        raise ValueError("the population study requires the seven-point control")
    target, hedge = np.asarray(control["target"]), np.asarray(control["hedge"])
    if target.shape != (7,) or hedge.shape != (7,):
        raise ValueError("control target and hedge must match the seven-point grid")
    original = tuple(tuple(map(int, clique)) for clique in control["cliques"])
    if len(original) != 2:
        raise ValueError("the population study requires two baseline cliques")
    designs = {
        "baseline": original,
        "add_maturity_3_to_left": (tuple(sorted(set(original[0]) | {6})), original[1]),
        "add_maturity_0_to_right": (original[0], tuple(sorted(set(original[1]) | {0}))),
        "complete": (tuple(range(7)), tuple(range(7))),
    }
    target_variance, hedge_variance = (
        float(target @ covariance @ target),
        float(hedge @ covariance @ hedge),
    )
    if hedge_variance <= 0:
        raise ValueError("the control hedge must have positive population variance")
    trace_value = _scalar(control["trace_value"], "trace_value", positive=True)
    results = {}
    for name, cliques in designs.items():
        mask = _mask(cliques, 7)
        for vector in (target, hedge):
            support = np.flatnonzero(vector)
            if not np.all(mask[np.ix_(support, support)]):
                raise ValueError(
                    "each design must observe the full target and hedge variance blocks"
                )
        model = CovarianceAmbiguity.from_covariance(
            covariance,
            mask,
            trace_value,
            factors=control["factors"],
            offset_covariance=control["offset_covariance"],
            trace_lower=trace_value,
        )
        source = _source_interval(
            model,
            target,
            hedge,
            target_variance,
            hedge_variance,
            tolerance,
        )
        free = free_psd_clique_hedge_interval(covariance, target, hedge, cliques)
        fixed_tail_free = free_psd_clique_hedge_interval(
            covariance,
            target,
            hedge,
            cliques,
            offset_covariance=control["offset_covariance"],
        )
        results[name] = {
            "cliques": cliques,
            "observed_mask": mask,
            "observed_pair_count": int(np.count_nonzero(np.triu(mask))),
            "known_variances": {"target": target_variance, "hedge": hedge_variance},
            "source_trace_lower": trace_value,
            "source_trace_cap": trace_value,
            "source_interval": source,
            "free_psd_interval": free,
            "free_psd_fixed_tail_interval": fixed_tail_free,
            "canonical_estimate": _canonical_estimate(covariance, target, hedge, cliques, mask),
            "baseline_source_feasibility": model.residuals(control["finite_source_base"]),
        }
    nesting = {}
    baseline_interval = results["baseline"]["source_interval"]
    baseline_mask = results["baseline"]["observed_mask"]
    for name, row in results.items():
        if name == "baseline":
            continue
        current = row["source_interval"]
        comparable = bool(baseline_interval["available"] and current["available"])
        diagnostics = {
            "observed_mask_contains_baseline": bool(np.all(row["observed_mask"][baseline_mask])),
            "available": comparable,
            "tolerance": 1e-5,
            "source_interval_inside_baseline": None,
            "maximum_interval_nesting_violation": None,
        }
        if comparable:
            outer, inner = baseline_interval["hedge_interval"], current["hedge_interval"]
            violation = float(max(0.0, outer[0] - inner[0], inner[1] - outer[1]))
            diagnostics.update(
                {
                    "source_interval_inside_baseline": violation <= 1e-5,
                    "maximum_interval_nesting_violation": violation,
                }
            )
        nesting[name] = diagnostics
    true_cross = float(target @ covariance @ hedge)
    true_hedge = true_cross / hedge_variance
    return _metadata_snapshot(
        {
            "config": {
                "solver": "CLARABEL",
                "solver_tolerance": tolerance,
                "reference_dimension": control["config"]["reference_dimension"],
                "source_dimension": control["config"]["source_dimension"],
                "output_dimension": control["config"]["output_dimension"],
                "source_trace_lower": trace_value,
                "source_trace_cap": trace_value,
                "fixed_tail_scope": "same reference-truncated fixed p=3 tail for every observation design",
            },
            "baseline_covariance": covariance,
            "target": target,
            "hedge": hedge,
            "population_oracle": {
                "hedge": true_hedge,
                "residual_variance": target_variance - true_cross**2 / hedge_variance,
            },
            "designs": results,
            "nesting_diagnostics": nesting,
            "all_source_intervals_available": all(
                row["source_interval"]["available"] for row in results.values()
            ),
            "statistical_confidence_interval": False,
            "infinite_dimensional_identification_claim": False,
            "scope": "population identification ranges in one fixed finite-block/common-tail source family",
        }
    )
