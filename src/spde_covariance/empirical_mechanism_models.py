"""Controlled reconstruction/source/residual contrasts for copper risk transfer.

Every estimator receives one already-masked training view.  The two covariance
rows share reconstructed innovations and residual statistics; only the forecast
of their main covariance changes.  Real-data contrasts concern modeling
policies, not error attribution against unobserved curve or forcing truth.
"""

from __future__ import annotations

import numpy as np

from .empirical_calibration import fit_cross_section, fit_source_covariance
from .empirical_short_history_data import daily_sections, interval_records
from .empirical_short_history_models import (
    _payload,
    _residual_statistics,
    _ridge_operator,
    _symmetric,
    _tenors,
    _unavailable,
    residual_maturity_features,
    schur_residual_forecast,
)
from .hedging import _snapshot
from .laguerre import LaguerreTransport

_MECHANISM_NAMES = tuple(
    f"{row}_{column}"
    for row in ("source", "empirical")
    for column in ("factor", "correlated", "diagonal")
) + ("reconstructed_pool",)


def _settings(config):
    names = (
        ("smoothing", 1e-6),
        ("reconstruction_ridge", 1e-6),
        ("half_life", 63.0),
        ("relative_regularization", 1e-3),
        ("trace_cap_multiplier", 100.0),
        ("numerical_error_budget", 1e-5),
        ("rank_rtol", 1e-10),
    )
    return tuple((name, float(getattr(config, name, default))) for name, default in names) + (
        ("target_tenors_days", tuple(config.target_tenors_days)),
        ("maturity_order_scope", "episode evaluation IDs and masked-view fingerprint"),
    )


def _source_donor_exclusions(section, date, config, dimension):
    """Batched leave-one-out shape ridge, equivalent to fit_cross_section.

    The centered sufficient statistics avoid cancellation at settlement-price
    levels.  Each deleted contract is absent from the level fit at this date.
    """
    prices = np.asarray(section["prices"], dtype=float)
    count = len(prices)
    if count <= dimension:
        raise ValueError("insufficient source donors after exclusion")
    model = LaguerreTransport(dimension, cache_max_bytes=0)
    evaluation = model.evaluate(_tenors(section, date) / 365.25)
    keep = ~np.eye(count, dtype=bool)
    excluded_evaluation = np.broadcast_to(evaluation, (count, count, dimension))[keep].reshape(
        count, count - 1, dimension
    )
    singular = np.linalg.svd(excluded_evaluation, compute_uv=False)
    rtol = float(getattr(config, "rank_rtol", 1e-10))
    if np.any(singular[:, -1] <= rtol * singular[:, 0]):
        raise ValueError("rank-deficient leave-one-donor-out source reconstruction")
    normal = (evaluation.T @ evaluation - evaluation[:, :, None] * evaluation[:, None, :]) / (
        count - 1
    )
    regularization = float(getattr(config, "smoothing", 1e-6)) * np.linalg.eigvalsh(normal)[:, -1]
    level = float(prices.mean())
    centered = prices - level
    excluded_mean = -centered / (count - 1)
    sums = evaluation.sum(axis=0) - evaluation
    rhs = (
        evaluation.T @ centered - evaluation * centered[:, None] - sums * excluded_mean[:, None]
    ) / (count - 1)
    penalty = np.eye(dimension)
    penalty[0, 0] = 0.0
    hessian = normal + regularization[:, None, None] * penalty
    coefficients = np.linalg.solve(hessian, rhs[:, :, None])[:, :, 0]
    coefficients[:, 0] += level + excluded_mean
    rhs_level = (evaluation.T @ prices - evaluation * prices[:, None]) / (count - 1)
    gradient = np.einsum("nij,nj->ni", hessian, coefficients) - rhs_level
    relative_residual = np.linalg.norm(gradient, axis=1) / (1 + np.linalg.norm(rhs_level, axis=1))
    if np.any(relative_residual > 1e-8) or not np.all(np.isfinite(coefficients)):
        raise ValueError("invalid leave-one-donor-out source normal equations")
    return dict(zip(section["ids"], coefficients, strict=True)), {
        "condition_max": float(np.max(np.linalg.cond(hessian))),
        "normal_equation_residual_max": float(np.max(relative_residual)),
    }


def _reconstruct(view, config, dimension):
    """Frequency-independent masked level fits and excluded-donor innovations."""
    dates = np.asarray(view.panel.session_dates, dtype="datetime64[D]")
    model = LaguerreTransport(dimension, cache_max_bytes=0)
    sections = tuple(daily_sections(view, eligible=True))
    records = tuple(interval_records(view, config))
    raw_records = tuple(interval_records(view, config, eligible=False))
    if len(records) < 2 or len(dates) != len(records) + 1:
        raise ValueError("fewer than two adjacent training intervals")
    fits = [
        fit_cross_section(
            _tenors(section, date) / 365.25,
            section["prices"],
            dimension=dimension,
            smoothing=getattr(config, "smoothing", 1e-6),
            rank_rtol=getattr(config, "rank_rtol", 1e-10),
        )
        for date, section in zip(dates, sections, strict=True)
    ]
    if not all(fit.available for fit in fits):
        raise ValueError("unavailable source curve reconstruction")
    states = np.asarray([fit.coefficients for fit in fits])
    deltas = np.diff(dates).astype(int) / 365.25
    innovations = np.asarray(
        [
            (right - model.matrix(float(delta)) @ left) / np.sqrt(delta)
            for left, right, delta in zip(states[:-1], states[1:], deltas, strict=True)
        ]
    )
    main = tuple(model.evaluate(record["tenors_days"] / 365.25) for record in records)
    hats = tuple(residual_maturity_features(record["tenors_days"]) for record in records)
    geometry = {
        "residual_operators": tuple(_ridge_operator(item) for item in hats),
        "residual_loo_operators": tuple(
            tuple(_ridge_operator(item[np.arange(len(item)) != k])[0] for k in range(len(item)))
            for item in hats
        ),
    }
    leverage = []
    for section, date in zip(sections, dates, strict=True):
        evaluation = model.evaluate(_tenors(section, date) / 365.25)
        normal = evaluation.T @ evaluation / len(evaluation)
        penalty = np.eye(dimension)
        penalty[0, 0] = 0.0
        hessian = (
            normal + getattr(config, "smoothing", 1e-6) * np.linalg.eigvalsh(normal)[-1] * penalty
        )
        smooth = evaluation @ np.linalg.solve(hessian, evaluation.T) / len(evaluation)
        leverage.extend(np.diag(smooth).tolist())
    reconstruction = {
        "dimension": dimension,
        "basis_id": model.basis_id,
        "curve_condition_max": float(max(f.diagnostics["design_condition_number"] for f in fits)),
        "curve_condition_median": float(
            np.median([f.diagnostics["design_condition_number"] for f in fits])
        ),
        "curve_rmse_mean": float(np.mean([f.diagnostics["rmse"] for f in fits])),
        "curve_maximum_absolute_error": float(
            max(f.diagnostics["maximum_absolute_residual"] for f in fits)
        ),
        "level_fit_leverage_max": float(max(leverage)),
        "level_fit_leverage_p95": float(np.quantile(leverage, 0.95)),
        "continuum_tail_error": None,
        "continuum_tail_error_status": "unavailable: no latent infinite-dimensional curve truth in settlements",
    }
    prepared = {
        "dates": dates,
        "states": states,
        "deltas": deltas,
        "innovations": innovations,
        "records": records,
        "raw_records": raw_records,
        "main": main,
        "geometry": geometry,
        "reconstruction": reconstruction,
    }
    try:
        excluded, excluded_diagnostics = [], []
        for date, section in zip(dates, sections, strict=True):
            values, diagnostics = _source_donor_exclusions(section, date, config, dimension)
            excluded.append(values)
            excluded_diagnostics.append(diagnostics)
        loo = []
        for index, record in enumerate(records):
            shift = model.matrix(float(record["delta"]))
            loo.append(
                np.asarray(
                    [
                        (excluded[index + 1][donor] - shift @ excluded[index][donor])
                        / np.sqrt(record["delta"])
                        for donor in record["ids"]
                    ]
                )
            )
        prepared["source_loo"] = tuple(loo)
        reconstruction.update(
            loo_condition_max=max(d["condition_max"] for d in excluded_diagnostics),
            loo_normal_equation_residual_max=max(
                d["normal_equation_residual_max"] for d in excluded_diagnostics
            ),
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        prepared["source_loo_error"] = str(error)
    return prepared


def _weighted_moment(values, ages, half_life):
    if not len(values):
        return None, 0.0
    values = np.asarray(values)
    weights = np.exp2(-np.asarray(ages) / half_life)
    weights /= weights.sum()
    return _symmetric(values.T @ (weights[:, None] * values)), float(1 / (weights @ weights))


def _exposure_retention(raw, fitted, residual, cross):
    exposures = {"outright": np.array([1.0, 0.0, 0.0]), "spread": np.array([1.0, -1.0, 0.0])}
    # Caller supplies maturity-order butterfly separately below.
    return {
        name: {
            "raw_second_moment": float(a @ raw @ a),
            "fitted_second_moment": float(a @ fitted @ a),
            "residual_second_moment": float(a @ residual @ a),
            "signed_cross_second_moment": float(a @ cross @ a),
            "fitted_to_raw": float((a @ fitted @ a) / (a @ raw @ a)) if a @ raw @ a > 0 else None,
        }
        for name, a in exposures.items()
    }


def _reconstructed_pool(view, config, prepared, dimension):
    """Matched raw-pooling cells with only quotation reconstruction changed."""
    model = LaguerreTransport(dimension, cache_max_bytes=0)
    records, raw_records, innovations = (
        prepared["records"],
        prepared["raw_records"],
        prepared["innovations"],
    )
    inverse_order = np.argsort(view.episode.maturity_order)
    groups = {
        name: {kind: [] for kind in ("raw", "fitted", "residual", "ages")}
        for name in ("own", "pool")
    }
    for index, (record, raw_record, z) in enumerate(
        zip(records, raw_records, innovations, strict=True)
    ):
        ids, raw_ids = record["ids"].tolist(), raw_record["ids"].tolist()
        selections = {}
        if all(identifier in raw_ids for identifier in view.evaluation_ids):
            selections["own"] = (
                raw_record,
                [raw_ids.index(identifier) for identifier in view.evaluation_ids],
            )
        if len(ids) >= 3:
            slots = [
                int(np.argmin(np.abs(record["tenors_days"] - t))) for t in config.target_tenors_days
            ]
            if len(set(slots)) == 3:
                selections["pool"] = (record, np.asarray(slots)[inverse_order])
        for group, (selected_record, indexes) in selections.items():
            indexes = np.asarray(indexes)
            raw_values = selected_record["response"][indexes]
            fitted_values = model.evaluate(selected_record["tenors_days"][indexes] / 365.25) @ z
            groups[group]["raw"].append(raw_values)
            groups[group]["fitted"].append(fitted_values)
            groups[group]["residual"].append(raw_values - fitted_values)
            groups[group]["ages"].append(len(records) - 1 - index)
    half_life = float(getattr(config, "half_life", 63.0))
    moments = {}
    for group, values in groups.items():
        moments[group] = {
            kind: _weighted_moment(values[kind], values["ages"], half_life)[0]
            for kind in ("raw", "fitted", "residual")
        }
    if moments["pool"]["fitted"] is None:
        raise ValueError("no complete maturity slots for reconstructed pooling")
    count = len(groups["own"]["raw"])
    alpha = count / (count + 20.0)
    combined = {
        kind: moments["pool"][kind]
        if moments["own"][kind] is None
        else (1 - alpha) * moments["pool"][kind] + alpha * moments["own"][kind]
        for kind in ("raw", "fitted", "residual")
    }
    raw, fitted, residual = (combined[kind] for kind in ("raw", "fitted", "residual"))
    cross = _symmetric(raw - fitted - residual)
    retention = _exposure_retention(raw, fitted, residual, cross)
    butterfly = np.zeros(3)
    butterfly[np.asarray(view.episode.maturity_order)] = [1.0, -2.0, 1.0]
    retention["butterfly"] = {
        "raw_second_moment": float(butterfly @ raw @ butterfly),
        "fitted_second_moment": float(butterfly @ fitted @ butterfly),
        "residual_second_moment": float(butterfly @ residual @ butterfly),
        "signed_cross_second_moment": float(butterfly @ cross @ butterfly),
        "fitted_to_raw": float((butterfly @ fitted @ butterfly) / (butterfly @ raw @ butterfly))
        if butterfly @ raw @ butterfly > 0
        else None,
    }
    delta = float((view.next_date - view.origin).astype(int)) / 365.25
    return _payload(
        delta * fitted,
        {"reconstructed_price_change": delta * fitted},
        {
            "complete_changes": count,
            "pooled_changes": len(groups["pool"]["raw"]),
            "short_blend_weight": alpha,
            "half_life_sessions": half_life,
            "interval_policy": "all",
            "matched_raw_pool_covariance": delta * raw,
            "reconstruction_retention": retention,
            "reconstruction_identity_relative_residual": float(
                np.linalg.norm(raw - fitted - residual - cross) / max(np.linalg.norm(raw), 1.0)
            ),
            "scope": "same raw pooling IDs, intervals, weights and own-history blend counts; only levels reconstructed",
        },
    )


def known_source_kernel(source_covariance, tenors_days, delta):
    """Analytical finite-model oracle for synthetic controls, never fit to data."""
    source = np.asarray(source_covariance, dtype=float)
    if source.ndim != 2 or source.shape[0] != source.shape[1] or not source.shape[0]:
        raise ValueError("source covariance must be a nonempty square matrix")
    model = LaguerreTransport(len(source), cache_max_bytes=0)
    evaluation = model.evaluate(np.asarray(tenors_days, dtype=float) / 365.25)
    return _symmetric(evaluation @ model.cumulative_covariance(source, delta) @ evaluation.T)


def mechanism_covariances(view, config, *, dimension=3, interval_policy="all", cache=None):
    """Issue one six-cell mechanism comparison and matched pooling control.

    ``all`` uses all original adjacent intervals. ``one_day`` gives zero
    training weight to longer gaps without compressing their calendar ages.
    Direct empirical one-day forecasts are explicitly unavailable for a longer
    future interval. Source forecasts can integrate into such an interval;
    compare rows only on the common one-day future stratum.
    """
    if (
        not isinstance(dimension, (int, np.integer))
        or isinstance(dimension, bool)
        or dimension not in (2, 3, 4)
    ):
        raise ValueError("mechanism study supports nested Laguerre dimensions 2, 3 and 4")
    if interval_policy not in ("all", "one_day"):
        raise ValueError("interval_policy must be all or one_day")
    if not view.available:
        return {name: _unavailable(view.reason) for name in _MECHANISM_NAMES}
    settings = _settings(config)
    key = (
        "mechanism-preparation-v1",
        view.fingerprint,
        tuple(view.episode.maturity_order),
        dimension,
        settings,
    )
    prepared = None if cache is None else cache.get(key)
    if prepared is None:
        try:
            prepared = _reconstruct(view, config, dimension)
        except (ValueError, np.linalg.LinAlgError) as error:
            return {name: _unavailable(error) for name in _MECHANISM_NAMES}
        if cache is not None:
            cache.put(key, prepared)
    result_key = ("mechanism-result-v1", key, interval_policy)
    cached = None if cache is None else cache.get(result_key)
    if cached is not None:
        return cached
    output = {name: _unavailable("not fitted") for name in _MECHANISM_NAMES}
    dates, deltas, factors, records, main = (
        prepared["dates"],
        prepared["deltas"],
        prepared["innovations"],
        prepared["records"],
        prepared["main"],
    )
    count = len(records)
    ages = np.arange(count - 1, -1, -1, dtype=float)
    weights = np.exp2(-ages / float(getattr(config, "half_life", 63.0)))
    if interval_policy == "one_day":
        weights[~np.isclose(deltas, 1 / 365.25, rtol=0, atol=1e-14)] = 0.0
    if np.count_nonzero(weights) < 2:
        return {
            name: _unavailable("fewer than two selected training intervals")
            for name in _MECHANISM_NAMES
        }
    weights /= weights.sum()
    future_delta = float((view.next_date - view.origin).astype(int)) / 365.25
    future_tenors = (view.maturities - view.next_date).astype(int).astype(float)
    model = LaguerreTransport(dimension, cache_max_bytes=0)
    evaluation = model.evaluate(future_tenors / 365.25)
    psi = residual_maturity_features(future_tenors)
    empirical = _symmetric(factors.T @ (weights[:, None] * factors))
    common = {
        "dimension": dimension,
        "basis_id": model.basis_id,
        "training_interval_count": count,
        "selected_training_interval_count": int(np.count_nonzero(weights)),
        "effective_interval_count": float(1 / (weights @ weights)),
        "interval_policy": interval_policy,
        "interval_age_policy": "original session index; excluded intervals retain their ages",
        "future_interval_years": future_delta,
        "half_life_sessions": float(getattr(config, "half_life", 63.0)),
        "information_fingerprint": view.fingerprint,
        "target": "next-session price-change second moment",
        "clock": "ACT/365.25",
        "conditional_mean_assumption": "zero",
        "reconstruction": prepared["reconstruction"],
        "moment_centered": False,
        "empirical_factor_eigenvalues": np.linalg.eigvalsh(empirical)[::-1],
        "empirical_factor_rank": int(
            np.linalg.matrix_rank(
                empirical,
                tol=getattr(config, "rank_rtol", 1e-10)
                * max(np.linalg.norm(empirical, 2), np.finfo(float).tiny),
            )
        ),
    }
    forecasts = {"empirical": empirical}
    source_diagnostics = {}
    try:
        source = fit_source_covariance(
            prepared["states"],
            (dates - dates[0]).astype(int) / 365.25,
            interval_weights=weights,
            drift="zero",
            regularization_relative=getattr(config, "relative_regularization", 1e-3),
            trace_cap_multiplier=getattr(config, "trace_cap_multiplier", 100.0),
            numerical_error_budget=getattr(config, "numerical_error_budget", 1e-5),
        )
        if not source.converged:
            raise ValueError("source optimizer did not meet declared covariance-error budget")
        forecasts["source"] = (
            model.cumulative_covariance(source.source_covariance, future_delta) / future_delta
        )
        source_diagnostics = dict(source.diagnostics)
    except (ValueError, np.linalg.LinAlgError) as error:
        for column in ("factor", "correlated", "diagonal"):
            output[f"source_{column}"] = _unavailable(error)
    residual = None
    diagonal_error = prepared.get("source_loo_error")
    try:
        residual = _residual_statistics(
            records,
            factors,
            main,
            weights,
            main_loo=None
            if diagonal_error is not None
            else lambda interval, donor: prepared["source_loo"][interval][donor],
            prepared=prepared["geometry"],
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        # An unavailable excluded-donor error must not suppress the correlated
        # correction, which needs no leave-one-out reconstruction.
        diagonal_error = str(error)
        try:
            residual = _residual_statistics(
                records, factors, main, weights, prepared=prepared["geometry"]
            )
        except (ValueError, np.linalg.LinAlgError) as residual_error:
            for row in ("source", "empirical"):
                for column in ("correlated", "diagonal"):
                    output[f"{row}_{column}"] = _unavailable(residual_error)
    if residual is not None:
        residual[2]["positive_weight_leave_one_donor_out_count"] = (
            int(
                sum(
                    len(record["response"])
                    for record, weight in zip(records, weights, strict=True)
                    if weight > 0
                )
            )
            if diagonal_error is None
            else 0
        )
        residual[2]["zero_weight_training_interval_count"] = int(np.count_nonzero(weights == 0))
    for row, forecast in forecasts.items():
        if (
            row == "empirical"
            and interval_policy == "one_day"
            and not np.isclose(future_delta, 1 / 365.25, rtol=0, atol=1e-14)
        ):
            for column in ("factor", "correlated", "diagonal"):
                output[f"{row}_{column}"] = _unavailable(
                    "one-day empirical covariance cannot forecast a multi-day interval"
                )
            continue
        diagnostics = {**common, "main_covariance_policy": row}
        if row == "source":
            diagnostics["source_fit"] = source_diagnostics
        output[f"{row}_factor"] = _payload(
            future_delta * _symmetric(evaluation @ forecast @ evaluation.T),
            {"factor": future_delta * _symmetric(evaluation @ forecast @ evaluation.T)},
            diagnostics,
        )
        if residual is None:
            continue
        moment, diagonal, residual_diagnostics = residual
        for column, correction in (("correlated", np.zeros(3)), ("diagonal", psi @ diagonal)):
            if column == "diagonal" and diagonal_error is not None:
                output[f"{row}_{column}"] = _unavailable(diagonal_error)
                continue
            try:
                annual, components, geometry = schur_residual_forecast(
                    evaluation,
                    psi,
                    moment,
                    forecast,
                    diagonal=correction,
                    rank_rtol=getattr(config, "rank_rtol", 1e-10),
                )
                output[f"{row}_{column}"] = _payload(
                    future_delta * annual,
                    {name: future_delta * value for name, value in components.items()},
                    {
                        **diagnostics,
                        **residual_diagnostics,
                        **geometry,
                        "residual_statistics_policy": "identical frozen training joint moment, Schur blocks and donor-excluded diagonal across covariance rows",
                    },
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                output[f"{row}_{column}"] = _unavailable(error)
    try:
        output["reconstructed_pool"] = _reconstructed_pool(view, config, prepared, dimension)
    except (ValueError, np.linalg.LinAlgError) as error:
        output["reconstructed_pool"] = _unavailable(error)
    output = _snapshot(output)
    if cache is not None:
        cache.put(result_key, output)
    return output
