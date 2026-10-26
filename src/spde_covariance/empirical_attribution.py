"""Matched, exploratory attribution of futures risk forecast differences.

This module does not retune the published historical study. It changes one
declared mechanism at a time on its already inspected evaluation period.
Individual predictions remain internal; the runner returns aggregate evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from . import empirical_study as study
from .empirical_calibration import fit_cross_section
from .empirical_diagnostics import reconstruction_risk_decomposition

MASKS = ("all", "alternate", "sparse4")
METHODS = ("factor_adjusted", "factor_unadjusted", "direct", "shrinkage")
SHORT_WINDOWS = (16, 32)


def maturity_mask(node_count, mask):
    """Deterministic nested maturity subsets, independent of prices/outcomes.

    ``alternate`` keeps indices 0, 2, ... after maturity sorting. ``sparse4``
    keeps at most four evenly spaced indices within that alternate set. Thus
    sparsity never changes the original exposure-selection information.
    """
    if isinstance(node_count, bool) or int(node_count) != node_count or node_count < 0:
        raise ValueError("node_count must be a nonnegative integer")
    if mask not in MASKS:
        raise ValueError(f"mask must be one of {MASKS}")
    full = np.arange(node_count, dtype=int)
    if mask == "all":
        return full
    alternate = full[::2]
    if mask == "alternate" or len(alternate) <= 4:
        return alternate
    return alternate[np.rint(np.linspace(0, len(alternate) - 1, 4)).astype(int)]


def prepare_masked_curves(daily, config, mask):
    """Reconstruct only retained nodes; retain the same raw evaluation universe.

    Fits are cross-sectional and use only that session. The ``sections`` and
    ``settlement_sections`` are the full original information: masks affect
    factor reconstruction, not exposure selection or the direct-risk baseline.
    Comparisons against direct under masking are therefore information stress
    tests, not matched-information tests of transport.
    """
    if mask not in MASKS:
        raise ValueError(f"mask must be one of {MASKS}")
    if mask == "all":
        return daily
    coefficients = np.full_like(daily["coefficients"], np.nan)
    diagnostics = []
    for i, section in enumerate(daily["sections"]):
        selected = maturity_mask(len(section["ids"]), mask)
        if len(selected) < config.dimension:
            diagnostics.append({"available": False, "nodes": len(selected)})
            continue
        tenors = (section["maturity"][selected] - daily["dates"][i]).astype(int) / 365.25
        fit = fit_cross_section(
            tenors,
            section["prices"][selected],
            dimension=config.dimension,
            smoothing=config.smoothing,
        )
        if fit.available:
            coefficients[i] = fit.coefficients
        diagnostics.append(
            {
                "available": bool(fit.available),
                "nodes": len(selected),
                **fit.diagnostics,
                "rmse_cny_per_tonne": float(np.sqrt(np.mean(fit.residuals**2))),
            }
        )
    return {**daily, "coefficients": coefficients, "diagnostics": diagnostics}


def issue_attribution_forecasts(daily, origin, window, config, *, next_date):
    """Issue matched-window forecasts without receiving future prices.

    All methods use the same trailing calendar interval count. Direct/shrinkage
    covariance is estimated separately on each held exposure's own complete
    raw-contract support. Missing historical contract legs stay unavailable.
    Factor methods share dimension, smoothing, reconstructed states and window;
    transport adjustment/integration is their sole intended difference.
    """
    date = daily["dates"][origin]
    next_date = np.datetime64(next_date, "D")
    if next_date <= date or window < 2 or origin < window:
        raise ValueError("a full historical window and a later next_date are required")
    chosen = study.select_contracts(daily["sections"][origin], date, config.target_tenors_days)
    if chosen is None:
        return None
    delta = float((next_date - date).astype(int)) / 365.25
    methods, unavailable, timings, diagnostics = {}, {}, {}, {}
    for method, adjustment in (("factor_adjusted", True), ("factor_unadjusted", False)):
        started = time.perf_counter()
        try:
            forecast = study.issue_risk_forecast(
                daily, origin, window, config, next_date=next_date, adjustment=adjustment
            )
        except ValueError:
            unavailable[method] = "training_state_unavailable"
            forecast = None
        except RuntimeError:
            unavailable[method] = "numerical_budget_not_met"
            forecast = None
        timings[method] = time.perf_counter() - started
        if forecast is not None:
            methods[method] = forecast["variances"]
            diagnostics[method] = {
                **forecast["diagnostics"],
                "risk_numerical_error_bounds": forecast["risk_numerical_error_bounds"],
                "risk_relative_numerical_error_bounds": {
                    name: value / max(forecast["variances"][name], config.pnl_variance_floor)
                    for name, value in forecast["risk_numerical_error_bounds"].items()
                },
            }
        elif method not in unavailable:
            unavailable[method] = "forecast_geometry_unavailable"
    started = time.perf_counter()
    baselines = {method: {} for method in ("direct", "shrinkage")}
    direct_counts = {}
    for name, exposure in study.EXPOSURES.items():
        support = exposure != 0
        covariance, count = study._historical_contract_covariance(
            daily, origin, chosen["ids"][support], window, config.min_direct_count
        )
        direct_counts[name] = count
        if covariance is None:
            continue
        shrunk = (1 - config.shrinkage) * covariance + config.shrinkage * np.diag(
            np.diag(covariance)
        )
        for method, value in (("direct", covariance), ("shrinkage", shrunk)):
            baselines[method][name] = max(
                0.0,
                float(config.multiplier**2 * delta * exposure[support] @ value @ exposure[support]),
            )
    baseline_seconds = time.perf_counter() - started
    for method, values in baselines.items():
        methods[method] = values
        timings[method] = baseline_seconds  # joint baseline calculation, not separate timers
        if len(values) < len(study.EXPOSURES):
            unavailable[method] = "insufficient_same_contract_history_for_some_exposures"
    return {
        "origin": str(date),
        "next_date": str(next_date),
        "training_start": str(daily["dates"][origin - window]),
        "training_end": str(date),
        "training_intervals": window,
        "contract_ids": chosen["ids"].tolist(),
        "origin_prices": chosen["prices"],
        "methods": methods,
        "unavailable": unavailable,
        "diagnostics": diagnostics,
        "runtime_seconds": timings,
        "direct_joint_counts": direct_counts,
    }


def score_attribution_forecast(issued, future_section, config):
    """Attach realized proxies only after issuance; absent held legs stay missing."""
    return study.score_issued_forecast(issued, future_section, config)


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.quantile(values, [0, 0.5, 0.95, 1]).tolist() if len(values) else None


def _fit_summary(daily):
    diagnostics = daily["diagnostics"]
    return {
        "sessions": len(diagnostics),
        "available_sessions": int(np.sum(np.all(np.isfinite(daily["coefficients"]), axis=1))),
        "retained_nodes_min_median_p95_max": _quantiles([r["nodes"] for r in diagnostics]),
        "design_condition_min_median_p95_max": _quantiles(
            [r.get("design_condition_number", np.nan) for r in diagnostics]
        ),
        "retained_node_rmse_min_median_p95_max": _quantiles(
            [r.get("rmse_cny_per_tonne", np.nan) for r in diagnostics]
        ),
    }


def _score_rows(records, *, window, mask, config, bootstrap_replications):
    """Keep common-four, marginal and paired-common samples explicitly distinct."""
    score_rows, available_rows, paired, availability = [], [], [], []
    for exposure in study.EXPOSURES:
        rows = [r for r in records if r["exposure"] == exposure]
        common = [r for r in rows if set(r["variances"]) == set(METHODS)]
        availability.extend(
            {
                "mask": mask,
                "window": window,
                "exposure": exposure,
                "method": method,
                "candidate_origins": len(rows),
                "issued_count": sum(method in r["variances"] for r in rows),
                "scored_count": sum(
                    method in r["variances"] and r["proxy"] is not None for r in rows
                ),
            }
            for method in METHODS
        )
        valid = [r for r in rows if r["proxy"] is not None]
        for year in ("all", *sorted({r["year"] for r in valid})):
            for sample, source, destination in (
                ("common_four_methods", common, score_rows),
                ("method_available", rows, available_rows),
            ):
                group = [
                    r
                    for r in source
                    if r["proxy"] is not None and (year == "all" or r["year"] == year)
                ]
                for method in METHODS:
                    selected = [r for r in group if method in r["variances"]]
                    if not selected:
                        continue
                    proxy = np.array([r["proxy"] for r in selected])
                    variance = np.array([r["variances"][method] for r in selected])
                    destination.append(
                        {
                            "mask": mask,
                            "window": window,
                            "year": year,
                            "exposure": exposure,
                            "method": method,
                            "sample": sample,
                            "count": len(selected),
                            "qlike": float(
                                np.mean(
                                    study.qlike(variance, proxy, floor=config.pnl_variance_floor)
                                )
                            ),
                            "quadratic": float(
                                np.mean(study.quadratic_risk_score(variance, proxy))
                            ),
                            "risk_ratio": float(np.sum(proxy) / np.sum(variance))
                            if np.sum(variance) > 0
                            else None,
                            "floor_hits": int(np.sum(variance < config.pnl_variance_floor)),
                        }
                    )
            for method, reference in (
                ("factor_adjusted", "factor_unadjusted"),
                ("factor_adjusted", "direct"),
                ("factor_unadjusted", "direct"),
                ("shrinkage", "direct"),
            ):
                pair = [
                    r
                    for r in valid
                    if method in r["variances"]
                    and reference in r["variances"]
                    and (year == "all" or r["year"] == year)
                ]
                if not pair:
                    continue
                proxy = np.array([r["proxy"] for r in pair])
                first = np.array([r["variances"][method] for r in pair])
                second = np.array([r["variances"][reference] for r in pair])
                for score in ("qlike", "quadratic"):
                    difference = (
                        study.qlike(first, proxy, floor=config.pnl_variance_floor)
                        - study.qlike(second, proxy, floor=config.pnl_variance_floor)
                        if score == "qlike"
                        else (first - second) * (first + second - 2 * proxy)
                    )
                    paired.append(
                        {
                            "mask": mask,
                            "window": window,
                            "year": year,
                            "exposure": exposure,
                            "method": method,
                            "reference": reference,
                            "score": score,
                            "sample": "pairwise_common_observations",
                            **study.paired_block_summary(
                                difference,
                                block=config.bootstrap_block,
                                replications=bootstrap_replications,
                                seed=config.seed,
                            ),
                        }
                    )
    return score_rows, available_rows, paired, availability


def run_attribution_study(
    panel, config, *, bootstrap_replications=1000, short_windows=SHORT_WINDOWS
):
    """Run fixed matched mechanisms and return only aggregate public evidence.

    No model or window is selected using these results. This is exploratory
    development on the already inspected historical test period. A future
    claim for a revised policy needs a genuinely uninspected evaluation period.
    Runtime summaries include cross-sectional reconstruction and repeated
    baseline work and are implementation measurements, not complexity claims.
    """
    if bootstrap_replications < 2:
        raise ValueError("bootstrap_replications must be at least two")
    if any(isinstance(w, bool) or int(w) != w or w < 2 for w in short_windows):
        raise ValueError("short_windows must contain interval counts >=2")
    windows = tuple(sorted(set(config.windows) | set(short_windows)))
    started = time.perf_counter()
    daily = study.prepare_daily_curves(panel, config)
    original = {
        "dimension": config.dimension,
        "smoothing": config.smoothing,
        "windows": list(config.windows),
        "short_windows": list(short_windows),
        "all_study_windows": list(windows),
        "relative_regularization": config.relative_regularization,
        "shrinkage": config.shrinkage,
        "min_direct_count": config.min_direct_count,
        "pnl_variance_floor": config.pnl_variance_floor,
        "multiplier": config.multiplier,
        "origin_stride": config.origin_stride,
        "bootstrap_block": config.bootstrap_block,
        "zero_drift": True,
        "trace_cap_multiplier": 100.0,
        "numerical_error_budget": 1e-5,
    }
    config_digest = hashlib.sha256(
        json.dumps(
            {"empirical_config": asdict(config), "short_windows": list(short_windows)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    scores, comparisons, availability, reconstruction, numerical = [], [], [], [], []
    available_scores = []
    candidate_origins = [
        i
        for i in range(max(windows), len(daily["dates"]) - 1, config.origin_stride)
        if daily["dates"][i] > np.datetime64(config.validation_end)
        and daily["dates"][i + 1] <= np.datetime64(config.test_end)
    ]
    for mask in MASKS:
        masked = prepare_masked_curves(daily, config, mask)
        reconstruction.append(
            {
                "mask": mask,
                "reconstruction": _fit_summary(masked),
                "risk_decomposition": reconstruction_risk_decomposition(masked, config),
            }
        )
        for window in windows:
            records, fit_diagnostics, timings = [], [], {method: [] for method in METHODS}
            failures = {}
            for origin in candidate_origins:
                issued = issue_attribution_forecasts(
                    masked,
                    origin,
                    window,
                    config,
                    next_date=daily["dates"][origin + 1],
                )
                if issued is None:
                    failures["origin_selection_unavailable"] = (
                        failures.get("origin_selection_unavailable", 0) + 1
                    )
                    for name in study.EXPOSURES:
                        records.append(
                            {
                                "year": str(daily["dates"][origin + 1])[:4],
                                "exposure": name,
                                "proxy": None,
                                "variances": {},
                            }
                        )
                    continue
                for method, reason in issued["unavailable"].items():
                    key = f"{method}:{reason}"
                    failures[key] = failures.get(key, 0) + 1
                for method, elapsed in issued["runtime_seconds"].items():
                    timings[method].append(elapsed)
                fit_diagnostics.extend(issued["diagnostics"].values())
                proxy = score_attribution_forecast(
                    issued, daily["settlement_sections"][origin + 1], config
                )
                for name in study.EXPOSURES:
                    records.append(
                        {
                            "year": str(daily["dates"][origin + 1])[:4],
                            "exposure": name,
                            "proxy": None if proxy is None else proxy[name],
                            "variances": {
                                method: values[name]
                                for method, values in issued["methods"].items()
                                if name in values
                            },
                        }
                    )
            local_scores, method_scores, paired, available = _score_rows(
                records,
                window=window,
                mask=mask,
                config=config,
                bootstrap_replications=bootstrap_replications,
            )
            scores.extend(local_scores)
            available_scores.extend(method_scores)
            comparisons.extend(paired)
            availability.extend(available)
            numerical.append(
                {
                    "mask": mask,
                    "window": window,
                    "candidate_origins": len(candidate_origins),
                    "successful_factor_fits": len(fit_diagnostics),
                    "unavailable_reasons": failures,
                    "relative_covariance_error_bound_max": max(
                        (r.get("relative_covariance_error_bound", 0.0) for r in fit_diagnostics),
                        default=None,
                    ),
                    "objective_gap_bound_scaled_max": max(
                        (r.get("objective_gap_bound_scaled", 0.0) for r in fit_diagnostics),
                        default=None,
                    ),
                    "portfolio_relative_numerical_error_bound_max": max(
                        (
                            bound
                            for r in fit_diagnostics
                            for bound in r["risk_relative_numerical_error_bounds"].values()
                        ),
                        default=None,
                    ),
                    "source_minimum_eigenvalue_min": min(
                        (r.get("minimum_source_eigenvalue", 0.0) for r in fit_diagnostics),
                        default=None,
                    ),
                    "trace_cap_violation_max": max(
                        (
                            max(0.0, r["source_trace"] - r["trace_cap"])
                            for r in fit_diagnostics
                            if "source_trace" in r and "trace_cap" in r
                        ),
                        default=None,
                    ),
                    "runtime_seconds_min_median_p95_max": {
                        method: _quantiles(values) for method, values in timings.items()
                    },
                }
            )
    return {
        "schema": "empirical_attribution_v1",
        "scope": {
            "research_status": "exploratory_development_on_previously_inspected_period",
            "evaluation_start_after": config.validation_end,
            "evaluation_end": config.test_end,
            "no_test_selection": True,
            "unchanged_original_benchmark": True,
            "target": "next_session_same_contract_cash_second_moment",
            "primary_score": "qlike",
            "secondary_score": "quadratic",
            "year_grouping": "outcome_session_calendar_year",
            "score_sample": "scores_common_all_four_methods_available_scores_method_specific",
            "paired_sample": "each_comparison_common_observations_by_exposure_mask_window_year",
            "short_history_baseline": "min_direct_count_unchanged_unavailable_if_window_too_short",
            "transport_ablation": "same_states_dimension_smoothing_window_and_raw_outcomes",
            "masking": "nested_price_independent_maturity_nodes_for_factor_reconstruction_only",
            "masked_baseline_information": "direct_baselines_retain_full_raw_contract_history",
            "matched_information_transport_claims": "use_within_mask_adjusted_vs_unadjusted",
            "predefined_exposures": list(study.EXPOSURES),
            "interval_convention": "ACT/365.25",
            "bootstrap": "paired_circular_moving_blocks_of_scored_sessions_not_IID_MCSE",
            "bootstrap_replications": int(bootstrap_replications),
            "seed": config.seed,
            "masks": list(MASKS),
            "settings": original,
            "configuration_sha256": config_digest,
            "raw_data_published": False,
            "runtime_seconds": time.perf_counter() - started,
        },
        "scores": scores,
        "available_scores": available_scores,
        "paired_comparisons": comparisons,
        "availability": availability,
        "representation": reconstruction,
        "numerical_diagnostics": numerical,
    }
