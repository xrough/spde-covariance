"""Past-only factor/residual covariance and bounded hedge development study.

The target is an empirical second moment of joint reconstructed-factor and
same-contract residual innovations. It is not a physical SPDE forcing estimate.
Calendar maturity transport stays physical under either diffusion clock.
2024--2025 is a development sample: previous results already informed these
models. Public study outputs contain aggregates only; issued objects are local.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from .empirical_calibration import fit_source_covariance
from .empirical_study import (
    EXPOSURES,
    paired_block_summary,
    prepare_daily_curves,
    qlike,
    quadratic_risk_score,
    select_contracts,
)
from .hedging import _snapshot
from .laguerre import LaguerreTransport


@dataclass(frozen=True)
class JointRiskConfig:
    """Small, prespecified candidate set; choose using observed validation only."""

    clocks: tuple[str, ...] = ("calendar", "session")
    shrinkages: tuple[float, ...] = (0.0, 0.1)
    session_year: float = 252.0
    hedge_bounds: tuple[float, float] = (-2.0, 2.0)
    rank_rtol: float = 1e-10
    include_source: bool = True

    def __post_init__(self):
        if (
            not self.clocks
            or len(set(self.clocks)) != len(self.clocks)
            or any(clock not in ("calendar", "session") for clock in self.clocks)
        ):
            raise ValueError("clocks must be distinct calendar/session choices")
        if (
            not self.shrinkages
            or len(set(self.shrinkages)) != len(self.shrinkages)
            or any(not np.isfinite(a) or not 0 <= a <= 1 for a in self.shrinkages)
        ):
            raise ValueError("shrinkages must be distinct values in [0,1]")
        if not np.isfinite(self.session_year) or self.session_year <= 0:
            raise ValueError("session_year must be positive")
        if (
            len(self.hedge_bounds) != 2
            or not all(np.isfinite(self.hedge_bounds))
            or self.hedge_bounds[0] >= self.hedge_bounds[1]
            or not self.hedge_bounds[0] <= 0 <= self.hedge_bounds[1]
        ):
            raise ValueError("hedge_bounds must be increasing, finite and contain zero")
        if not np.isfinite(self.rank_rtol) or not 0 < self.rank_rtol < 1:
            raise ValueError("rank_rtol must be between zero and one")
        if self.include_source and "calendar" not in self.clocks:
            raise ValueError("include_source requires the calendar clock")


def diffusion_duration(first, last, clock, *, session_year=252.0):
    """One adjacent session in session time; ACT/365.25 in calendar time."""
    days = int((np.datetime64(last, "D") - np.datetime64(first, "D")).astype(int))
    if days <= 0:
        raise ValueError("dates must strictly increase")
    if clock == "calendar":
        return days / 365.25
    if clock == "session" and np.isfinite(session_year) and session_year > 0:
        return 1.0 / session_year
    raise ValueError("clock must be calendar/session with positive session_year")


def daily_training_prefix(daily, origin):
    """Copy only data through origin; keep the issuer's API future-free."""
    if not isinstance(origin, (int, np.integer)) or not 0 <= origin < len(daily["dates"]):
        raise ValueError("origin must index a daily date")
    keys = ("dates", "times", "coefficients", "sections", "settlement_sections", "diagnostics")
    return {key: daily[key][: origin + 1] for key in keys if key in daily}


def _validate_training(training):
    dates = np.asarray(training["dates"], dtype="datetime64[D]")
    states = np.asarray(training["coefficients"], dtype=float)
    if (
        dates.ndim != 1
        or len(dates) < 2
        or np.any(np.isnat(dates))
        or np.any(np.diff(dates).astype(int) <= 0)
        or states.ndim != 2
        or states.shape[0] != len(dates)
        or states.shape[1] < 1
        or len(training["sections"]) != len(dates)
        or len(training["settlement_sections"]) != len(dates)
    ):
        raise ValueError("training must be an increasing daily prefix with aligned states/sections")
    return dates, states


def joint_training_statistics(training, chosen, window, *, clock="calendar", session_year=252.0):
    """Joint complete cases with exact same-contract factor/residual identities.

    For each adjacent historical interval, z=(r,u)/sqrt(diffusion duration),
    r=a_i-M(calendar delta)a_{i-1}, u=Delta p-E(end tenors)r.
    The current origin's three actual contracts are used at every historical
    endpoint. No absent quote or curve is filled and no gap is bridged.
    """
    dates, states = _validate_training(training)
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must count at least two adjacent intervals")
    ids = np.asarray(chosen["ids"])
    maturity = np.asarray(chosen["maturity"], dtype="datetime64[D]")
    if ids.shape != (3,) or len(set(ids.tolist())) != 3 or maturity.shape != (3,):
        raise ValueError("chosen must describe three distinct actual contracts")
    model = LaguerreTransport(states.shape[1])
    start = max(1, len(dates) - window)
    joint, direct, factors, residuals, physical, diffusion = [], [], [], [], [], []
    unavailable_quotes = unavailable_curves = 0
    identity_error = 0.0
    for i in range(start, len(dates)):
        if not np.all(np.isfinite(states[i - 1 : i + 1])):
            unavailable_curves += 1
            continue
        first = training["settlement_sections"][i - 1]
        last = training["settlement_sections"][i]
        left = dict(zip(first["ids"], first["prices"], strict=True))
        right = dict(zip(last["ids"], last["prices"], strict=True))
        if not all(key in left and key in right for key in ids):
            unavailable_quotes += 1
            continue
        change = np.array([right[key] - left[key] for key in ids], dtype=float)
        end_tenors = (maturity - dates[i]).astype(int) / 365.25
        if not np.all(np.isfinite(change)) or np.any(end_tenors < 0):
            unavailable_quotes += 1
            continue
        delta = diffusion_duration(dates[i - 1], dates[i], "calendar")
        duration = diffusion_duration(dates[i - 1], dates[i], clock, session_year=session_year)
        factor = states[i] - model.matrix(delta) @ states[i - 1]
        fitted_change = model.evaluate(end_tenors) @ factor
        residual = change - fitted_change
        # Check both the algebraic decomposition and the exact-shift identity.
        endpoint = (
            model.evaluate(end_tenors) @ states[i]
            - model.evaluate(end_tenors + delta) @ states[i - 1]
        )
        identity_error = max(identity_error, float(np.max(np.abs(endpoint - fitted_change))))
        joint.append(np.r_[factor, residual] / np.sqrt(duration))
        direct.append(change / np.sqrt(duration))
        factors.append(factor)
        residuals.append(residual)
        physical.append(delta)
        diffusion.append(duration)
    dimension = states.shape[1]
    return {
        "joint_values": np.asarray(joint).reshape(-1, dimension + 3),
        "direct_values": np.asarray(direct).reshape(-1, 3),
        "factor_changes": np.asarray(factors).reshape(-1, dimension),
        "residual_changes": np.asarray(residuals).reshape(-1, 3),
        "physical_durations": np.asarray(physical),
        "diffusion_durations": np.asarray(diffusion),
        "count": len(joint),
        "nominal_intervals": len(dates) - start,
        "unavailable_quotes": unavailable_quotes,
        "unavailable_curves": unavailable_curves,
        "exact_transport_identity_max_abs": identity_error,
        "training_start": str(dates[start - 1]),
        "training_end": str(dates[-1]),
    }


def _second_moment(values):
    return values.T @ values / len(values)


def _psd_diagnostics(matrix, rank_rtol):
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    support = eigenvalues[eigenvalues > rank_rtol * scale]
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "relative_minimum_eigenvalue": float(eigenvalues[0] / scale),
        "numerical_rank": int(np.sum(eigenvalues > rank_rtol * scale)),
        "rank_relative_threshold": rank_rtol,
        "resolved_condition_number": float(support[-1] / support[0]) if len(support) else None,
        "trace": float(np.trace(matrix)),
        "psd_within_roundoff": bool(eigenvalues[0] >= -1e-10 * scale),
    }


def issue_joint_risk_forecast(
    training_daily,
    *,
    next_date,
    window,
    config,
    method="joint",
    clock="calendar",
    shrinkage=0.0,
    joint_config=None,
):
    """Issue local forecasts using a training prefix and known next date only.

    Methods factor/joint/direct share exactly the same jointly available
    historical intervals. source is the original source-C pipeline, uses its
    own full coefficient window and calendar time, and is labelled separately.
    Pooling r/sqrt(q) is an empirical local-moment approximation: exact SPDE
    normalized innovation covariances can still vary with physical intervals.
    Only source uses the interval-specific source-to-innovation bridge.
    Returned contract IDs, quotes, covariances and hedges are local artifacts.
    """
    options = JointRiskConfig() if joint_config is None else joint_config
    dates, states = _validate_training(training_daily)
    next_date = np.datetime64(next_date, "D")
    if next_date <= dates[-1]:
        raise ValueError("next_date must follow the final training date")
    if method not in ("factor", "joint", "direct", "source"):
        raise ValueError("unknown joint risk method")
    if clock not in options.clocks or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("invalid clock or shrinkage")
    if method != "joint" and shrinkage != 0:
        raise ValueError("shrinkage is an explicit joint-model option only")
    if method == "source" and clock != "calendar":
        raise ValueError("original source model uses physical calendar time")
    chosen = select_contracts(training_daily["sections"][-1], dates[-1], config.target_tenors_days)
    if chosen is None:
        return None
    end_tenors = (chosen["maturity"] - next_date).astype(int) / 365.25
    if np.any(end_tenors < 0):
        return None
    model = LaguerreTransport(states.shape[1])
    evaluation = model.evaluate(end_tenors)
    duration = diffusion_duration(dates[-1], next_date, clock, session_year=options.session_year)
    physical_delta = diffusion_duration(dates[-1], next_date, "calendar")
    if method == "source":
        values = states[max(0, len(dates) - window - 1) :]
        times = (dates[max(0, len(dates) - window - 1) :] - dates[0]).astype(int) / 365.25
        if len(values) < 3 or not np.all(np.isfinite(values)):
            return None
        fit = fit_source_covariance(
            values,
            times,
            regularization_relative=config.relative_regularization,
            drift="zero",
        )
        if not fit.converged:
            raise RuntimeError("original source fit did not converge")
        kernel = (
            evaluation
            @ model.cumulative_covariance(fit.source_covariance, physical_delta)
            @ evaluation.T
        )
        diagnostics = {
            "training_count": len(values) - 1,
            "training_start": str(dates[max(0, len(dates) - window - 1)]),
            "source_relative_covariance_error_bound": fit.diagnostics[
                "relative_covariance_error_bound"
            ],
            "sample_scope": "full coefficient window; not matched complete-contract samples",
        }
        components = None
        joint_covariance = None
    else:
        statistics = joint_training_statistics(
            training_daily,
            chosen,
            window,
            clock=clock,
            session_year=options.session_year,
        )
        if statistics["count"] < config.min_direct_count:
            return None
        raw = _second_moment(statistics["joint_values"])
        covariance = (1 - shrinkage) * raw + shrinkage * np.diag(np.diag(raw))
        factor_covariance = covariance[: states.shape[1], : states.shape[1]]
        cross = covariance[: states.shape[1], states.shape[1] :]
        residual_covariance = covariance[states.shape[1] :, states.shape[1] :]
        factor_kernel = duration * evaluation @ factor_covariance @ evaluation.T
        residual_kernel = duration * residual_covariance
        cross_kernel = duration * (evaluation @ cross + cross.T @ evaluation.T)
        if method == "factor":
            kernel = factor_kernel
        elif method == "direct":
            kernel = duration * _second_moment(statistics["direct_values"])
        else:
            kernel = factor_kernel + residual_kernel + cross_kernel
        diagnostics = {
            "training_count": statistics["count"],
            "nominal_intervals": statistics["nominal_intervals"],
            "training_start": statistics["training_start"],
            "unavailable_quotes": statistics["unavailable_quotes"],
            "unavailable_curves": statistics["unavailable_curves"],
            "exact_transport_identity_max_abs": statistics["exact_transport_identity_max_abs"],
            "joint_psd": _psd_diagnostics(covariance, options.rank_rtol),
            "cross_block_relative_frobenius_norm": float(
                np.linalg.norm(cross) / max(np.linalg.norm(covariance), np.finfo(float).tiny)
            ),
            "shrinkage_relative_matrix_change": float(
                np.linalg.norm(covariance - raw) / max(np.linalg.norm(raw), np.finfo(float).tiny)
            ),
            "sample_scope": "same complete-contract and reconstructed-curve intervals for factor/joint/direct",
            "minimum_physical_days": float(np.min(statistics["physical_durations"]) * 365.25),
            "maximum_physical_days": float(np.max(statistics["physical_durations"]) * 365.25),
        }
        components = {
            "factor": factor_kernel,
            "residual": residual_kernel,
            "cross": cross_kernel,
        }
        joint_covariance = covariance
    kernel = (kernel + kernel.T) / 2
    diagnostics["forecast_psd"] = _psd_diagnostics(kernel, options.rank_rtol)
    if not diagnostics["forecast_psd"]["psd_within_roundoff"]:
        raise RuntimeError("forecast covariance is materially indefinite")
    hedge = bounded_minimum_variance_hedge(kernel, bounds=options.hedge_bounds)
    weights = np.r_[1.0, hedge["hedge_positions"]]
    variances = {
        name: max(0.0, float(config.multiplier**2 * a @ kernel @ a))
        for name, a in EXPOSURES.items()
    }
    component_risks = (
        {
            exposure: {
                part: float(config.multiplier**2 * a @ value @ a)
                for part, value in components.items()
            }
            for exposure, a in EXPOSURES.items()
        }
        if components is not None
        else None
    )
    return _snapshot(
        {
            "origin": str(dates[-1]),
            "next_date": str(next_date),
            "training_end": str(dates[-1]),
            "training_start": diagnostics["training_start"],
            "window": window,
            "method": method,
            "clock": clock,
            "shrinkage": shrinkage,
            "physical_duration": physical_delta,
            "diffusion_duration": duration,
            "contract_ids": chosen["ids"],
            "origin_prices": chosen["prices"],
            "covariance": kernel,
            "joint_covariance": joint_covariance,
            "risk_components": component_risks,
            "variances": variances,
            "hedge_weights": weights,
            "hedge_predicted_second_moment": max(
                0.0, float(config.multiplier**2 * weights @ kernel @ weights)
            ),
            "hedge_diagnostics": hedge,
            "diagnostics": diagnostics,
            "estimand": "empirical same-contract innovation second moment conditional on reconstructed curves",
        }
    )


def score_joint_risk_forecast(issued, future_section, config):
    """Score separately from issuance; unavailable required legs stay unavailable."""
    lookup = dict(zip(future_section["ids"], future_section["prices"], strict=True))
    changes = (
        np.array([lookup.get(key, np.nan) for key in issued["contract_ids"]])
        - issued["origin_prices"]
    )
    risks = {}
    for name, a in EXPOSURES.items():
        support = a != 0
        risks[name] = (
            float((config.multiplier * a[support] @ changes[support]) ** 2)
            if np.all(np.isfinite(changes[support]))
            else None
        )
    support = issued["hedge_weights"] != 0
    pnl = (
        float(config.multiplier * issued["hedge_weights"][support] @ changes[support])
        if np.all(np.isfinite(changes[support]))
        else None
    )
    return {"squared_pnl": risks, "hedge_pnl": pnl}


def bounded_minimum_variance_hedge(covariance, *, bounds=(-2.0, 2.0)):
    """Globally minimize (1,h)' K (1,h) for two bounded hedge positions.

    Enumerate the unconstrained minimum, the four one-dimensional edges and
    corners. The problem is a convex two-dimensional quadratic. Equal minima
    select the smaller Euclidean hedge norm, then lexicographic coordinates.
    No mean-return forecast, cost model or tradable execution claim is made.
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 3 by 3 matrix")
    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("bounds must be increasing finite values")
    scale = max(float(np.linalg.norm(matrix, ord=2)), np.finfo(float).tiny)
    if np.max(np.abs(matrix - matrix.T)) > 1e-10 * scale:
        raise ValueError("covariance must be symmetric")
    matrix = (matrix + matrix.T) / 2
    if np.linalg.eigvalsh(matrix)[0] < -1e-10 * scale:
        raise ValueError("covariance must be PSD")
    lower, upper = map(float, bounds)
    block, linear = matrix[1:, 1:], matrix[1:, 0]
    candidates = [np.array(point) for point in product((lower, upper), repeat=2)]
    interior = -np.linalg.pinv(block, rcond=1e-12) @ linear
    if np.all(interior >= lower) and np.all(interior <= upper):
        candidates.append(interior)
    if lower <= 0 <= upper:
        candidates.append(np.zeros(2))
    for fixed_axis in (0, 1):
        free_axis = 1 - fixed_axis
        for fixed in (lower, upper):
            value = np.zeros(2)
            value[fixed_axis] = fixed
            numerator = linear[free_axis] + block[free_axis, fixed_axis] * fixed
            value[free_axis] = (
                np.clip(-numerator / block[free_axis, free_axis], lower, upper)
                if block[free_axis, free_axis] > 1e-14 * scale
                else np.clip(0.0, lower, upper)
            )
            candidates.append(value)
    values = np.array([np.r_[1, h] @ matrix @ np.r_[1, h] for h in candidates])
    minimum = np.min(values)
    tolerance = 1e-12 * scale
    eligible = [
        h for h, value in zip(candidates, values, strict=True) if value <= minimum + tolerance
    ]
    solution = min(eligible, key=lambda h: (float(h @ h), float(h[0]), float(h[1])))
    value = float(np.r_[1, solution] @ matrix @ np.r_[1, solution])
    return {
        "hedge_positions": solution,
        "predicted_second_moment_price_units": max(0.0, value),
        "bound_hits": int(np.sum(np.isclose(solution, lower) | np.isclose(solution, upper))),
        "bounds": [lower, upper],
        "objective_enumeration_gap": max(0.0, value - float(minimum)),
        "scope": "one near contract plus two maturity-target hedge positions; settlement P&L research",
    }


def _candidate_policies(config, options):
    result = {}
    for method in ("factor", "joint", "direct"):
        alphas = options.shrinkages if method == "joint" else (0.0,)
        result[method] = [
            {"window": window, "clock": clock, "shrinkage": alpha, "method": method}
            for window, clock, alpha in product(config.windows, options.clocks, alphas)
        ]
    if options.include_source:
        result["source"] = [
            {"window": window, "clock": "calendar", "shrinkage": 0.0, "method": "source"}
            for window in config.windows
        ]
    return result


def select_joint_policies(daily, config, *, joint_config=None):
    """Freeze each rule using mean observed validation QLIKE across three risks.

    All candidates within a rule are scored on identical future observations.
    Exact ties prefer a longer nominal window, calendar clock, lower shrinkage.
    Later development outcomes never enter selection.
    """
    options = JointRiskConfig() if joint_config is None else joint_config
    candidates = _candidate_policies(config, options)
    losses = {method: [[] for _ in policies] for method, policies in candidates.items()}
    boundaries = {method: [] for method in candidates}
    unavailable = {method: 0 for method in candidates}
    for origin in range(max(config.windows), len(daily["dates"]) - 1, config.origin_stride):
        day, next_day = daily["dates"][origin : origin + 2]
        if day <= np.datetime64(config.training_end) or next_day > np.datetime64(
            config.validation_end
        ):
            continue
        prefix = daily_training_prefix(daily, origin)
        for method, policies in candidates.items():
            values = []
            for policy in policies:
                try:
                    forecast = issue_joint_risk_forecast(
                        prefix,
                        next_date=next_day,
                        config=config,
                        joint_config=options,
                        **policy,
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    forecast = None
                if forecast is None:
                    values.append(None)
                    unavailable[method] += 1
                    continue
                scored = score_joint_risk_forecast(
                    forecast, daily["settlement_sections"][origin + 1], config
                )["squared_pnl"]
                values.append(
                    float(
                        np.mean(
                            [
                                qlike(
                                    forecast["variances"][name],
                                    scored[name],
                                    floor=config.pnl_variance_floor,
                                )
                                for name in EXPOSURES
                            ]
                        )
                    )
                    if all(value is not None for value in scored.values())
                    else None
                )
            if all(value is not None for value in values):
                for scores, value in zip(losses[method], values, strict=True):
                    scores.append(value)
                boundaries[method].append(str(next_day))
    selected, audit = {}, {}
    for method, policies in candidates.items():
        if not boundaries[method]:
            raise ValueError(f"no common validation origins for joint rule {method}")
        means = [float(np.mean(values)) for values in losses[method]]
        index = min(
            range(len(policies)),
            key=lambda i: (
                means[i],
                -policies[i]["window"],
                policies[i]["clock"] != "calendar",
                policies[i]["shrinkage"],
            ),
        )
        selected[method] = policies[index]
        audit[method] = {
            "selected": policies[index],
            "candidates": [
                dict(policy, mean_validation_qlike=score)
                for policy, score in zip(policies, means, strict=True)
            ],
            "common_validation_origins": len(boundaries[method]),
            "selection_last_outcome": max(boundaries[method]),
            "unavailable_candidate_fits": unavailable[method],
        }
    return selected, audit


def _paired(values, config):
    return paired_block_summary(
        values,
        block=config.bootstrap_block,
        replications=config.bootstrap_replications,
        seed=config.seed,
    )


def run_joint_risk_study(panel, config, *, joint_config=None, daily=None):
    """Return public aggregates for notebooks 05/06; no per-date market output.

    Selection uses 2023 (configured validation) only. Test-labelled 2024--2025
    observations are explicitly a development sample because earlier findings
    motivated this increment. Same-contract future outcomes enter only scoring.
    """
    options = JointRiskConfig() if joint_config is None else joint_config
    daily = prepare_daily_curves(panel, config) if daily is None else daily
    policies, selection = select_joint_policies(daily, config, joint_config=options)
    records, coverage, diagnostics = [], {}, {}
    for method in policies:
        coverage[method] = {"attempted": 0, "issued": 0, "fully_scored": 0, "hedge_scored": 0}
        diagnostics[method] = []
    # A raw joint comparator uses exactly the selected joint window and clock;
    # this does not create an additional tuned policy.
    evaluation_policies = dict(policies)
    evaluation_policies["joint_raw"] = dict(policies["joint"], shrinkage=0.0)
    for method in ("factor", "joint", "direct"):
        for clock in options.clocks:
            evaluation_policies[f"{method}_{clock}"] = dict(policies[method], clock=clock)
    for label in evaluation_policies:
        coverage.setdefault(
            label, {"attempted": 0, "issued": 0, "fully_scored": 0, "hedge_scored": 0}
        )
        diagnostics.setdefault(label, [])
    for origin in range(max(config.windows), len(daily["dates"]) - 1, config.origin_stride):
        day, next_day = daily["dates"][origin : origin + 2]
        if day <= np.datetime64(config.validation_end) or next_day > np.datetime64(config.test_end):
            continue
        prefix = daily_training_prefix(daily, origin)
        for label, policy in evaluation_policies.items():
            coverage[label]["attempted"] += 1
            try:
                forecast = issue_joint_risk_forecast(
                    prefix,
                    next_date=next_day,
                    config=config,
                    joint_config=options,
                    **policy,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                forecast = None
            if forecast is None:
                continue
            coverage[label]["issued"] += 1
            diagnostics[label].append(forecast["diagnostics"])
            outcomes = score_joint_risk_forecast(
                forecast, daily["settlement_sections"][origin + 1], config
            )
            complete = all(value is not None for value in outcomes["squared_pnl"].values())
            coverage[label]["fully_scored"] += int(complete)
            coverage[label]["hedge_scored"] += int(outcomes["hedge_pnl"] is not None)
            records.append(
                {
                    "origin_index": origin,
                    "method": label,
                    "year": str(next_day)[:4],
                    "physical_gap_days": int((next_day - day).astype(int)),
                    "variance": dict(forecast["variances"]),
                    "squared_pnl": outcomes["squared_pnl"],
                    "hedge_pnl": outcomes["hedge_pnl"],
                    "hedge_weights": forecast["hedge_weights"],
                    "hedge_predicted": forecast["hedge_predicted_second_moment"],
                    "hedge_bound_hits": forecast["hedge_diagnostics"]["bound_hits"],
                    "components": forecast["risk_components"],
                }
            )
    risk_summary, paired_scores, components = [], [], []
    methods = sorted(evaluation_policies)
    for exposure in EXPOSURES:
        by_method = {
            method: {
                r["origin_index"]: r
                for r in records
                if r["method"] == method and r["squared_pnl"][exposure] is not None
            }
            for method in methods
        }
        available = {method: rows for method, rows in by_method.items() if rows}
        common = (
            sorted(set.intersection(*(set(rows) for rows in available.values())))
            if available
            else []
        )
        for method, lookup in by_method.items():
            if not lookup:
                continue
            rows = [lookup[key] for key in common]
            if not rows:
                continue
            predicted = np.array([r["variance"][exposure] for r in rows])
            realized = np.array([r["squared_pnl"][exposure] for r in rows])
            risk_summary.append(
                {
                    "method": method,
                    "exposure": exposure,
                    "common_count": len(rows),
                    "qlike": float(
                        np.mean(qlike(predicted, realized, floor=config.pnl_variance_floor))
                    ),
                    "quadratic_score": float(np.mean(quadratic_risk_score(predicted, realized))),
                    "realized_over_predicted_second_moment": float(
                        np.sum(realized) / np.sum(predicted)
                    )
                    if np.sum(predicted) > 0
                    else None,
                    "predicted_rms_cny": float(np.sqrt(np.mean(predicted))),
                    "realized_rms_cny": float(np.sqrt(np.mean(realized))),
                    "floor_hits": int(np.sum(predicted < config.pnl_variance_floor)),
                }
            )
            reference = by_method["direct"]
            if method != "direct" and reference:
                reference_pred = np.array([reference[key]["variance"][exposure] for key in common])
                for score, difference in (
                    (
                        "qlike",
                        qlike(predicted, realized, floor=config.pnl_variance_floor)
                        - qlike(reference_pred, realized, floor=config.pnl_variance_floor),
                    ),
                    (
                        "quadratic",
                        (predicted - reference_pred) * (predicted + reference_pred - 2 * realized),
                    ),
                ):
                    paired_scores.append(
                        {
                            "method": method,
                            "reference": "direct",
                            "exposure": exposure,
                            "score": score,
                            **_paired(difference, config),
                        }
                    )
            decomposable = [r for r in rows if r["components"] is not None]
            if decomposable:
                values = {
                    part: float(np.mean([r["components"][exposure][part] for r in decomposable]))
                    for part in ("factor", "residual", "cross")
                }
                total = sum(values.values())
                components.append(
                    {
                        "method": method,
                        "exposure": exposure,
                        "count": len(decomposable),
                        "mean_factor_plus_residual_second_moment_cny2": total,
                        "components_cny2": values,
                        "cross_over_total": values["cross"] / total if total > 0 else None,
                        "scope": "joint model decomposition; factor/direct forecasts may use another kernel",
                    }
                )
    hedge_summary, hedge_paired = _hedge_summaries(records, methods, config)
    gap_summary = []
    for method in methods:
        for label, low, high in (
            ("one_day", 1, 1),
            ("two_or_three_days", 2, 3),
            ("four_or_more_days", 4, 10000),
        ):
            for exposure in EXPOSURES:
                rows = [
                    r
                    for r in records
                    if r["method"] == method
                    and low <= r["physical_gap_days"] <= high
                    and r["squared_pnl"][exposure] is not None
                ]
                if not rows:
                    continue
                predicted = np.array([r["variance"][exposure] for r in rows])
                realized = np.array([r["squared_pnl"][exposure] for r in rows])
                gap_summary.append(
                    {
                        "method": method,
                        "gap_group": label,
                        "exposure": exposure,
                        "count": len(rows),
                        "qlike": float(
                            np.mean(qlike(predicted, realized, floor=config.pnl_variance_floor))
                        ),
                        "realized_over_predicted_second_moment": float(
                            np.sum(realized) / np.sum(predicted)
                        )
                        if np.sum(predicted) > 0
                        else None,
                    }
                )
    numerical = {}
    for method, rows in diagnostics.items():
        if not rows:
            numerical[method] = {"available": False}
            continue
        counts = [r["training_count"] for r in rows]
        numerical[method] = {
            "available": True,
            "training_count_min_median_max": np.quantile(counts, [0, 0.5, 1]).tolist(),
            "minimum_relative_forecast_eigenvalue": min(
                r["forecast_psd"]["relative_minimum_eigenvalue"] for r in rows
            ),
            "all_forecast_psd_within_roundoff": all(
                r["forecast_psd"]["psd_within_roundoff"] for r in rows
            ),
            "maximum_exact_transport_identity_absolute_error": max(
                (r.get("exact_transport_identity_max_abs", 0.0) for r in rows)
            ),
        }
        if method != "source":
            numerical[method]["joint_rank_min_median_max"] = np.quantile(
                [r["joint_psd"]["numerical_rank"] for r in rows], [0, 0.5, 1]
            ).tolist()
            resolved = [
                r["joint_psd"]["resolved_condition_number"]
                for r in rows
                if r["joint_psd"]["resolved_condition_number"] is not None
            ]
            numerical[method]["joint_resolved_condition_min_median_max"] = (
                np.quantile(resolved, [0, 0.5, 1]).tolist() if resolved else None
            )
            numerical[method]["cross_block_relative_norm_min_median_max"] = np.quantile(
                [r["cross_block_relative_frobenius_norm"] for r in rows], [0, 0.5, 1]
            ).tolist()
            numerical[method]["maximum_shrinkage_relative_matrix_change"] = max(
                r["shrinkage_relative_matrix_change"] for r in rows
            )
    return {
        "schema": "spde-covariance-joint-risk-development-v1",
        "research_status": "exploratory development evidence; 2024--2025 informed model construction",
        "estimand": "next-session conditional squared P&L proxy; empirical joint innovation second moment, not physical source C",
        "config": asdict(config),
        "joint_config": asdict(options),
        "selection": selection,
        "evaluation_policies": evaluation_policies,
        "selection_criterion": "mean validation QLIKE across outright, spread and butterfly on identical validation origins within each rule; exact ties longer window then calendar clock then smaller shrinkage",
        "chronology": {
            "initial_training_end": config.training_end,
            "validation_end": config.validation_end,
            "development_end": config.test_end,
            "earliest_candidate_origin_index": max(config.windows),
            "no_untouched_test_claim": True,
        },
        "risk_summary": risk_summary,
        "paired_scores": paired_scores,
        "risk_components": components,
        "hedge_summary": hedge_summary,
        "hedge_paired": hedge_paired,
        "clock_gap_summary": gap_summary,
        "availability": coverage,
        "numerical_diagnostics": numerical,
        "clock_convention": "physical ACT/365.25 maturity transport unchanged; diffusion normalization calendar ACT/365.25 or one session/252",
        "sampling": "factor, joint and direct share complete historical contract/curve intervals at each identical nominal window; independently selected policies may use different windows",
        "interval_heterogeneity_caveat": "pooled factor/residual clock-normalized moments are a local empirical approximation; under a transport SPDE, normalized V_delta can vary with physical delta. Only original source-C control fits the interval-specific semigroup covariance bridge",
        "hedge_scope": "one near contract plus two bounded maturity-target positions, settlement P&L; turnover is a position-change proxy and excludes execution/roll costs",
        "data_boundary": "aggregate-only; no quotes, contract IDs, daily outcomes, fitted curves or issued hedges",
        "common_cohort_policy": "each risk/hedge table uses the intersection of scored origins across methods with at least one outcome; wholly unavailable methods are omitted and recorded in availability",
    }


def _hedge_summaries(records, methods, config):
    by_method = {
        method: {
            r["origin_index"]: r
            for r in records
            if r["method"] == method
            and r["hedge_pnl"] is not None
            and r["squared_pnl"]["outright_3m"] is not None
        }
        for method in methods
    }
    available = {method: rows for method, rows in by_method.items() if rows}
    common = (
        sorted(set.intersection(*(set(rows) for rows in available.values()))) if available else []
    )
    if not common:
        return [], []
    reference_method = "direct" if by_method["direct"] else next(iter(available))
    reference_rows = [by_method[reference_method][i] for i in common]
    unhedged = np.array([r["squared_pnl"]["outright_3m"] for r in reference_rows])
    direct_squared = np.array([r["hedge_pnl"] ** 2 for r in reference_rows])
    summary = [
        {
            "method": "unhedged",
            "common_count": len(common),
            "realized_rms_cny": float(np.sqrt(np.mean(unhedged))),
            "realized_second_moment_cny2": float(np.mean(unhedged)),
            "reduction_vs_unhedged": 0.0,
            "scope": "actual near-contract squared settlement P&L, no hedge",
        }
    ]
    paired = []
    for method, lookup in by_method.items():
        if not lookup:
            continue
        rows = [lookup[i] for i in common]
        pnl = np.array([r["hedge_pnl"] for r in rows])
        realized = pnl**2
        predicted = np.array([r["hedge_predicted"] for r in rows])
        positions = np.array([r["hedge_weights"][1:] for r in rows])
        # Compare adjacent issued maturity-target positions; contract identity
        # changes make this a notional-position proxy, not executable turnover.
        turnover = np.sum(np.abs(np.diff(positions, axis=0)), axis=1)
        summary.append(
            {
                "method": method,
                "common_count": len(rows),
                "realized_rms_cny": float(np.sqrt(np.mean(realized))),
                "realized_second_moment_cny2": float(np.mean(realized)),
                "realized_centered_variance_cny2": float(np.var(pnl, ddof=1))
                if len(pnl) > 1
                else None,
                "mean_pnl_cny": float(np.mean(pnl)),
                "predicted_rms_cny": float(np.sqrt(np.mean(predicted))),
                "realized_over_predicted_second_moment": float(np.sum(realized) / np.sum(predicted))
                if np.sum(predicted) > 0
                else None,
                "reduction_vs_unhedged": float(1 - np.sum(realized) / np.sum(unhedged))
                if np.sum(unhedged) > 0
                else None,
                "mean_hedge_positions": np.mean(positions, axis=0).tolist(),
                "mean_absolute_hedge_positions": np.mean(np.abs(positions), axis=0).tolist(),
                "bound_hit_fraction": float(np.mean([r["hedge_bound_hits"] > 0 for r in rows])),
                "mean_position_change_proxy": float(np.mean(turnover)) if len(turnover) else None,
                "median_position_change_proxy": float(np.median(turnover))
                if len(turnover)
                else None,
                "turnover_scope": "target-position changes on common scored origins; ignores contract switches and execution/roll costs",
            }
        )
        for reference, reference_squared in (
            ("unhedged", unhedged),
            (reference_method, direct_squared),
        ):
            if reference == method:
                continue
            paired.append(
                {
                    "method": method,
                    "reference": reference,
                    "metric": "squared_hedge_pnl_difference_cny2",
                    **_paired(realized - reference_squared, config),
                }
            )
    return summary, paired
