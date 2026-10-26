"""Aggregate-only forecast-score and model-risk analysis.

This module consumes local scored forecasts; it neither fits models nor chooses
one after inspecting test losses. QLIKE remains the primary criterion and the
quadratic proxy score is a secondary, differently weighted robustness check.
Both target the conditional second moment of P&L. Their expected regrets at a
positive target m are log(v/m) + m/v - 1 and (v-m)**2, respectively. No Gaussian
assumption is needed. QLIKE evaluates max(v, floor); the quadratic score retains
the original nonnegative forecast, as in the frozen first empirical study.

All dates, individual predictions, outcomes, contract identifiers and arbitrary
input metadata stay private. Small cells report counts but suppress statistics.
Realized-outcome strata and influence deletions are explicitly ex-post
diagnostics, never alternative filtered headline results or forecast regimes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .empirical_study import EXPOSURES, paired_block_summary, qlike, quadratic_risk_score

_METHODS = frozenset(
    {
        "direct",
        "shrinkage",
        "spde_dim2",
        "spde_dim4",
        "spde_fixed126",
        "spde_local",
        "spde_smoother",
        "spde_static",
        "unadjusted",
    }
)
_MINIMUM_CELL = 5
_LOSSES = ("qlike", "quadratic")


@dataclass(frozen=True)
class _Record:
    origin: int
    year: str
    method: str
    exposure: str
    variance: float
    outcome: float
    calendar_days: int


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _canonical_records(records):
    """Validate exact pairing privately, then impose chronological ordering."""
    result, seen, outcomes = [], set(), {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("each forecast record must be a mapping")
        method, exposure = row.get("method"), row.get("exposure")
        if not isinstance(method, str) or method not in _METHODS:
            raise ValueError("unrecognized forecast method; public labels are allowlisted")
        if not isinstance(exposure, str) or exposure not in EXPOSURES:
            raise ValueError("unrecognized exposure; public labels are allowlisted")
        origin = row.get("origin")
        if not isinstance(origin, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", origin):
            raise ValueError("origin must be an ISO calendar date")
        try:
            day = np.datetime64(origin, "D")
        except ValueError as exc:
            raise ValueError("origin must be a valid calendar date") from exc
        if np.isnat(day) or str(day) != origin:
            raise ValueError("origin must be a valid calendar date")
        interval = _integer(row.get("calendar_days"), "calendar_days", minimum=1)
        year = row.get("year")
        if year != str(day + np.timedelta64(interval, "D"))[:4]:
            raise ValueError("record year must be the outcome calendar year")
        ordinal = int(day.astype(int))
        key = (exposure, method, ordinal)
        if key in seen:
            raise ValueError("duplicate exposure/method/origin forecast record")
        seen.add(key)
        variance = _number(row.get("variance"), "variance")
        outcome = _number(row.get("squared_pnl"), "squared_pnl")
        target_key = (exposure, ordinal)
        signature = (outcome, interval, year)
        if target_key in outcomes and outcomes[target_key] != signature:
            raise ValueError(
                "methods at one origin must share exactly the same outcome and interval"
            )
        outcomes[target_key] = signature
        result.append(_Record(ordinal, year, method, exposure, variance, outcome, interval))
    return sorted(result, key=lambda row: (row.exposure, row.method, row.origin))


def _calendar_group(days):
    return "1_day" if days == 1 else "2_3_days" if days <= 3 else "4_plus_days"


def _score_arrays(rows, floor):
    variance = np.array([row.variance for row in rows], dtype=float)
    outcome = np.array([row.outcome for row in rows], dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = {
            "qlike": qlike(variance, outcome, floor=floor),
            "quadratic": quadratic_risk_score(variance, outcome),
        }
    with np.errstate(over="ignore", invalid="ignore"):
        finite_sums = all(
            np.isfinite(np.sum(np.abs(values))) for values in (variance, outcome, *result.values())
        )
    if any(np.any(~np.isfinite(values)) for values in result.values()) or not finite_sums:
        raise ValueError("forecast scores overflow; rescale risk units")
    return variance, outcome, result


def _aggregate(rows, floor):
    count = len(rows)
    result = {"count": count, "available": count >= _MINIMUM_CELL}
    names = (
        "mean_predicted_variance_cny2",
        "mean_squared_pnl_cny2",
        "predicted_rms_cny",
        "realized_rms_cny",
        "risk_ratio",
        "mean_qlike",
        "mean_quadratic_score_cny4",
    )
    result.update(dict.fromkeys(names))
    result["floor_hits"] = sum(row.variance < floor for row in rows)
    if not result["available"]:
        return result
    variance, outcome, scores = _score_arrays(rows, floor)
    predicted_mean, outcome_mean = float(np.mean(variance)), float(np.mean(outcome))
    result.update(
        {
            "mean_predicted_variance_cny2": predicted_mean,
            "mean_squared_pnl_cny2": outcome_mean,
            "predicted_rms_cny": float(np.sqrt(predicted_mean)),
            "realized_rms_cny": float(np.sqrt(outcome_mean)),
            "risk_ratio": outcome_mean / predicted_mean if predicted_mean > 0 else None,
            "mean_qlike": float(np.mean(scores["qlike"])),
            "mean_quadratic_score_cny4": float(np.mean(scores["quadratic"])),
        }
    )
    return result


def _bootstrap(values, *, block, replications, seed):
    """Suppress small samples while retaining their availability counts."""
    if len(values) >= _MINIMUM_CELL:
        return {
            "available": True,
            **paired_block_summary(values, block=block, replications=replications, seed=seed),
        }
    return {
        "available": False,
        "count": len(values),
        "mean": None,
        "block_length": min(block, len(values)),
        "bootstrap_standard_error": None,
        "bootstrap_percentile_95": None,
    }


def _alignment(method_rows, reference_rows):
    methods = {row.origin: row for row in method_rows}
    reference = {row.origin: row for row in reference_rows}
    common = sorted(methods.keys() & reference.keys())
    return (
        [methods[day] for day in common],
        [reference[day] for day in common],
        {
            "method_count": len(methods),
            "reference_count": len(reference),
            "union_count": len(methods.keys() | reference.keys()),
            "common_count": len(common),
            "method_only_count": len(methods.keys() - reference.keys()),
            "reference_only_count": len(reference.keys() - methods.keys()),
        },
    )


def _strata(rows):
    yield "pooled", "all", rows
    for year in sorted({row.year for row in rows}):
        yield "year", year, [row for row in rows if row.year == year]
    for label in ("1_day", "2_3_days", "4_plus_days"):
        yield (
            "calendar_gap",
            label,
            [row for row in rows if _calendar_group(row.calendar_days) == label],
        )


def _influence(values, fraction):
    """Leave-out diagnostics retain the original result as the headline."""
    count = len(values)
    removed = int(np.ceil(fraction * count))
    result = {
        "fraction": fraction,
        "count": count,
        "removed_count": removed,
        "retained_count": count - removed,
        "available": removed >= _MINIMUM_CELL and count - removed >= _MINIMUM_CELL,
        "full_mean_difference": None,
        "removed_mean_difference": None,
        "remaining_mean_difference": None,
        "removed_absolute_difference_share": None,
    }
    if not result["available"]:
        return result
    indices = np.argsort(-np.abs(values), kind="stable")
    total_absolute = float(np.sum(np.abs(values)))
    result.update(
        {
            "full_mean_difference": float(np.mean(values)),
            "removed_mean_difference": float(np.mean(values[indices[:removed]])),
            "remaining_mean_difference": float(np.mean(values[indices[removed:]])),
            "removed_absolute_difference_share": (
                float(np.sum(np.abs(values[indices[:removed]]))) / total_absolute
                if total_absolute > 0
                else None
            ),
        }
    )
    return result


def analyze_forecast_scores(records, config, *, blocks=(5, 10, 20, 40), replications=None):
    """Analyze local forecasts without exporting individual market observations.

    ``records`` follows ``empirical_study``'s local prediction-record schema.
    Methods are paired exactly by exposure and origin; conflicting outcomes or
    duplicate records fail. Every comparison discloses unmatched counts. The
    bootstrap resamples the chronologically ordered *scored* origins, so a
    block is a run of scored sessions rather than a fixed calendar duration.
    It is conditional descriptive uncertainty, not a model confidence interval.

    Calibration bins are forecast-variance quintiles; numeric cutoffs are not
    exported. Outcome thirds use common realized squared P&L, so are ex post.
    Neither diagnostic selects or refits a forecasting method. Cells below
    five observations disclose counts only; influence diagnostics require at
    least five omitted and five retained observations.
    """
    read = config.get if isinstance(config, Mapping) else lambda name: getattr(config, name)
    floor = _number(read("pnl_variance_floor"), "pnl_variance_floor")
    if floor <= 0:
        raise ValueError("pnl_variance_floor must be positive")
    seed = _integer(read("seed"), "seed")
    primary_block = _integer(read("bootstrap_block"), "bootstrap_block", minimum=1)
    replications = _integer(
        read("bootstrap_replications") if replications is None else replications,
        "replications",
        minimum=2,
    )
    blocks = tuple(_integer(block, "block length", minimum=1) for block in blocks)
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("blocks must be nonempty distinct positive integers")
    rows = _canonical_records(records)
    # Validate even suppressed/small cells and unpaired methods for overflow.
    _score_arrays(rows, floor)
    groups = defaultdict(list)
    for row in rows:
        groups[row.exposure].append(row)
    summaries, paired, sensitivity, calibration, influence, outcomes, availability = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for exposure, exposure_rows in sorted(groups.items()):
        methods = sorted({row.method for row in exposure_rows})
        reference_all = [row for row in exposure_rows if row.method == "direct"]
        for method in methods:
            method_all = [row for row in exposure_rows if row.method == method]
            _, _, counts = _alignment(method_all, reference_all)
            availability.append({"exposure": exposure, "method": method, **counts})
            variance = np.array([row.variance for row in method_all])
            cuts = np.quantile(variance, [0.2, 0.4, 0.6, 0.8])
            bins = np.searchsorted(cuts, variance, side="left")
            for index in range(5):
                calibration.append(
                    {
                        "exposure": exposure,
                        "method": method,
                        "forecast_variance_quintile": index + 1,
                        **_aggregate(
                            [
                                row
                                for row, value in zip(method_all, bins, strict=True)
                                if value == index
                            ],
                            floor,
                        ),
                    }
                )
        for stratum, label, stratum_rows in _strata(exposure_rows):
            reference = [row for row in stratum_rows if row.method == "direct"]
            for method in methods:
                method_rows = [row for row in stratum_rows if row.method == method]
                identity = {
                    "exposure": exposure,
                    "method": method,
                    "stratum": stratum,
                    "group": label,
                }
                summaries.append({**identity, **_aggregate(method_rows, floor)})
                if method == "direct":
                    continue
                left, right, counts = _alignment(method_rows, reference)
                _, _, left_scores = _score_arrays(left, floor)
                _, _, right_scores = _score_arrays(right, floor)
                for loss in _LOSSES:
                    differences = left_scores[loss] - right_scores[loss]
                    if np.any(~np.isfinite(differences)):
                        raise ValueError("paired score differences overflow; rescale risk units")
                    pair_identity = {**identity, "reference": "direct", "loss": loss}
                    paired.append(
                        {
                            **pair_identity,
                            **counts,
                            **_bootstrap(
                                differences,
                                block=primary_block,
                                replications=replications,
                                seed=seed,
                            ),
                        }
                    )
                    if stratum != "pooled":
                        continue
                    for block in sorted(blocks):
                        sensitivity.append(
                            {
                                **pair_identity,
                                **counts,
                                "requested_block_length": block,
                                **_bootstrap(
                                    differences,
                                    block=block,
                                    replications=replications,
                                    seed=seed,
                                ),
                            }
                        )
                    for fraction in (0.01, 0.05):
                        influence.append({**pair_identity, **_influence(differences, fraction)})
        # One common ex-post classification, independent of a method's coverage.
        unique_outcomes = {row.origin: row.outcome for row in exposure_rows}
        cuts = np.quantile(list(unique_outcomes.values()), [1 / 3, 2 / 3])
        labels = {
            day: int(np.searchsorted(cuts, value, side="left"))
            for day, value in unique_outcomes.items()
        }
        for index, label in enumerate(("lower", "middle", "upper")):
            for method in methods:
                selected = [
                    row
                    for row in exposure_rows
                    if row.method == method and labels[row.origin] == index
                ]
                outcomes.append(
                    {
                        "exposure": exposure,
                        "method": method,
                        "realized_squared_pnl_third": label,
                        **_aggregate(selected, floor),
                    }
                )
    return {
        "schema": "empirical-forecast-score-analysis-v1",
        "protocol": {
            "primary_loss": "qlike",
            "secondary_loss": "quadratic",
            "reference": "direct",
            "difference_sign": "method minus direct; negative favours method",
            "pnl_variance_floor_cny2": floor,
            "primary_bootstrap_block": primary_block,
            "bootstrap_replications": replications,
            "bootstrap_seed": seed,
            "bootstrap_unit": "chronologically ordered scored sessions, circular moving blocks",
            "minimum_public_cell_count": _MINIMUM_CELL,
            "uncertainty_scope": (
                "conditional descriptive bootstrap with nominal 95% percentile intervals; "
                "not a model confidence interval; no model-selection or multiple-comparison correction"
            ),
            "coverage_scope": "scored-record union; attempted and issued counts require the separate forecast audit",
            "outcome_strata_scope": "ex-post realized outcomes, not past-only regime forecasts",
            "influence_scope": "post-hoc deletions; full-sample primary results remain unchanged",
            "loss_interpretation": {
                "target": "conditional second moment E[P&L squared | past], not automatically conditional variance",
                "qlike_expected_regret": "log(v/m) + m/v - 1 for v,m>0; v is floored before QLIKE scoring",
                "quadratic_expected_regret": "(v-m)^2; original nonnegative v is used without flooring",
                "proxy_requirement": "paired methods share exactly the same outcome; conditional proxy mean must equal the target",
                "rank_reversal": "QLIKE weights relative underprediction more strongly; quadratic weights absolute squared-risk errors",
                "distribution": "no Gaussian likelihood or trading profitability interpretation",
                "score_units": "quadratic in CNY^4; QLIKE uses the fixed CNY^2 convention and paired differences are unit invariant",
            },
        },
        "availability": availability,
        "method_summaries": summaries,
        "paired_comparisons": paired,
        "block_sensitivity": sensitivity,
        "calibration_bins": calibration,
        "realized_outcome_strata": outcomes,
        "score_influence": influence,
    }
