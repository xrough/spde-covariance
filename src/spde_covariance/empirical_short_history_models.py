"""Past-only maturity-factor and transport risk transfer for masked contracts.

All empirical targets are conditional second moments under the maintained zero
mean approximation. The residual correction is an empirical prediction-error
model; it is not an identified forcing or fresh observation-noise covariance.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import nnls

from .empirical_calibration import fit_cross_section, fit_source_covariance
from .laguerre import LaguerreTransport

_MODEL_NAMES = (
    "spde_uniform",
    "spde_ewma",
    "spde_joint",
    "dfm_joint",
    "dfm_uniform_joint",
    "dfm_factor",
    "spde_joint_no_nugget",
    "dfm_joint_no_nugget",
)


class BoundedModelCache:
    """Per-run LRU preparation cache, bounded by both entries and bytes.

    Cache keys contain the full masked-view fingerprint. There is no global
    data cache, and the cache never has access to an unmasked panel.
    """

    def __init__(self, max_entries=64, max_bytes=64 * 1024**2):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("cache bounds must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._values = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _size(value, seen=None):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            return size + (value.nbytes if value.base is not None else 0)
        if isinstance(value, Mapping):
            return size + sum(
                BoundedModelCache._size(k, seen) + BoundedModelCache._size(v, seen)
                for k, v in value.items()
            )
        if isinstance(value, (tuple, list)):
            return size + sum(BoundedModelCache._size(v, seen) for v in value)
        return size

    def get(self, key):
        if key not in self._values:
            self.misses += 1
            return None
        self.hits += 1
        self._values.move_to_end(key)
        return self._values[key][0]

    def put(self, key, value):
        size = self._size(value)
        if key in self._values:
            self.bytes -= self._values.pop(key)[1]
        if size > self.max_bytes:
            return
        while self._values and (
            len(self._values) >= self.max_entries or self.bytes + size > self.max_bytes
        ):
            self.bytes -= self._values.popitem(last=False)[1][1]
        self._values[key] = (value, size)
        self.bytes += size

    @property
    def entries(self):
        return len(self._values)

    @property
    def diagnostics(self):
        return {
            "entries": self.entries,
            "bytes": self.bytes,
            "hits": self.hits,
            "misses": self.misses,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
        }


def _symmetric(value):
    return (np.asarray(value, dtype=float) + np.asarray(value, dtype=float).T) / 2


@lru_cache(maxsize=1)
def _spline_transform():
    # Four-point Gauss quadrature is exact for products of cubic pieces.
    knots = np.array([30.0] * 4 + [120.0, 240.0] + [365.0] * 4)
    basis = BSpline(knots, np.eye(6), 3, extrapolate=False)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    gram = np.zeros((6, 6))
    for left, right in zip((30.0, 120.0, 240.0), (120.0, 240.0, 365.0), strict=True):
        points = left + (nodes + 1) * (right - left) / 2
        evaluation = basis(points)
        gram += evaluation.T @ (weights[:, None] * evaluation) * (right - left) / (2 * 335)
    values, vectors = np.linalg.eigh(gram)
    transform = (vectors / np.sqrt(values)) @ vectors.T
    transform.flags.writeable = False
    return basis, transform


def orthonormal_spline_basis(tenors_days):
    """Six cubic splines, orthonormal for uniform probability on [30,365]."""
    points = np.asarray(tenors_days, dtype=float)
    if points.ndim != 1 or np.any(~np.isfinite(points)) or np.any((points < 30) | (points > 365)):
        raise ValueError("spline maturities must be a finite vector in [30,365] days")
    basis, transform = _spline_transform()
    return basis(points) @ transform


def residual_maturity_features(tenors_days):
    """Four nonnegative piecewise-linear hats; flat endpoint extrapolation.

    Flat endpoint extension is only used for the known next-session maturity,
    which can fall slightly below the 30-day selection threshold.
    """
    points = np.asarray(tenors_days, dtype=float)
    if points.ndim != 1 or np.any(~np.isfinite(points)):
        raise ValueError("residual maturities must be a finite vector")
    knots = np.array([30.0, 140.0, 250.0, 365.0])
    return np.column_stack([np.interp(points, knots, row) for row in np.eye(4)])


def _ridge_operator(evaluation, relative=1e-6):
    evaluation = np.asarray(evaluation, dtype=float)
    if evaluation.ndim != 2 or not len(evaluation):
        raise ValueError("empty reconstruction cross-section")
    normal = evaluation.T @ evaluation / len(evaluation)
    eigenvalues = np.linalg.eigvalsh(normal)
    regularization = relative * max(float(eigenvalues[-1]), np.finfo(float).tiny)
    operator = np.linalg.solve(normal + regularization * np.eye(normal.shape[0]), evaluation.T)
    operator /= len(evaluation)
    singular = np.linalg.svd(evaluation, compute_uv=False)
    return operator, {
        "regularization": regularization,
        "gram_norm": float(eigenvalues[-1]),
        "condition_number": float(
            (eigenvalues[-1] + regularization) / (eigenvalues[0] + regularization)
        ),
        "rank": int(np.sum(singular > 1e-10 * singular[0])),
    }


def interval_weights(count, half_life=None):
    """Normalized trading-session EWMA weights, newest interval last."""
    if count < 2 or (half_life is not None and (not np.isfinite(half_life) or half_life <= 0)):
        raise ValueError("weights need at least two intervals and a positive half-life")
    weights = (
        np.ones(count) if half_life is None else np.exp2(-np.arange(count - 1, -1, -1) / half_life)
    )
    return weights / weights.sum()


def _psd_roundoff(matrix, name, tolerance=1e-8):
    matrix = _symmetric(matrix)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if values[0] < -tolerance * scale:
        raise ValueError(f"materially indefinite {name}")
    corrected = (vectors * np.maximum(values, 0)) @ vectors.T if values[0] < 0 else matrix
    return _symmetric(corrected), float(np.linalg.norm(corrected - matrix)), float(values[0])


def schur_residual_forecast(
    main_evaluation,
    residual_evaluation,
    joint_moment,
    main_forecast,
    *,
    diagonal=None,
    rank_rtol=1e-10,
):
    """Retain all main/residual cross terms with explicit singular support checks.

    Returns an annualized kernel and additive components. The matrix supplied
    as ``main_forecast`` can differ from the training main block (SPDE bridge).
    """
    b = np.asarray(main_evaluation, dtype=float)
    psi = np.asarray(residual_evaluation, dtype=float)
    moment, joint_correction, _ = _psd_roundoff(joint_moment, "joint second moment")
    p = b.shape[1]
    if moment.shape != (p + psi.shape[1],) * 2 or len(b) != len(psi):
        raise ValueError("joint moment and evaluation dimensions differ")
    h, g, r = moment[:p, :p], moment[:p, p:], moment[p:, p:]
    h_forecast, forecast_correction, _ = _psd_roundoff(main_forecast, "factor forecast")
    if h_forecast.shape != h.shape:
        raise ValueError("factor forecast has the wrong shape")
    inverse = np.linalg.pinv(h, rcond=rank_rtol, hermitian=True)
    support = float(np.linalg.norm(g - h @ inverse @ g))
    support_scale = max(float(np.linalg.norm(moment)), np.finfo(float).tiny)
    if support > 1e-8 * support_scale:
        raise ValueError("main/residual cross block is incompatible with numerical factor support")
    a = g.T @ inverse
    omega, omega_correction, omega_minimum = _psd_roundoff(r - g.T @ inverse @ g, "Schur residual")
    augmented = b + psi @ a
    factor = b @ h_forecast @ b.T
    with_cross = augmented @ h_forecast @ augmented.T
    correlated = psi @ omega @ psi.T
    diagonal_values = np.zeros(len(b)) if diagonal is None else np.asarray(diagonal, dtype=float)
    if diagonal_values.shape != (len(b),) or np.any(diagonal_values < 0):
        raise ValueError("diagonal correction must be a nonnegative evaluation vector")
    nugget = np.diag(diagonal_values)
    components = {
        "factor": _symmetric(factor),
        "cross_correction": _symmetric(with_cross - factor),
        "correlated_residual": _symmetric(correlated),
        "diagonal_prediction_error": nugget,
    }
    covariance, covariance_correction, minimum = _psd_roundoff(
        with_cross + correlated + nugget, "risk forecast"
    )
    diagnostics = {
        "support_residual": support,
        "relative_support_residual": support / support_scale,
        "pseudoinverse_relative_cutoff": rank_rtol,
        "factor_numerical_rank": int(
            np.linalg.matrix_rank(
                h, tol=rank_rtol * max(np.linalg.norm(h, 2), np.finfo(float).tiny)
            )
        ),
        "joint_psd_roundoff_correction": joint_correction,
        "main_forecast_psd_roundoff_correction": forecast_correction,
        "schur_psd_roundoff_correction": omega_correction,
        "schur_minimum_eigenvalue": omega_minimum,
        "forecast_psd_roundoff_correction": covariance_correction,
        "minimum_eigenvalue": minimum,
        "component_sum_relative_residual": float(
            np.linalg.norm(covariance - sum(components.values()))
            / max(np.linalg.norm(covariance), 1.0)
        ),
    }
    return covariance, components, diagnostics


def _tenors(section, date):
    if "tenors_days" in section:
        return np.asarray(section["tenors_days"], dtype=float)
    if "maturity" in section:
        maturity = np.asarray(section["maturity"])
        if maturity.dtype.kind == "M":
            return (maturity - np.datetime64(date, "D")).astype(int).astype(float)
        return maturity.astype(float) * 365.25
    raise ValueError("section needs maturity metadata")


def _source_training(view, config, sections):
    dates = np.asarray(view.panel.session_dates, dtype="datetime64[D]")
    coefficients = []
    diagnostics = []
    for date, section in zip(dates, sections, strict=True):
        fit = fit_cross_section(
            _tenors(section, date) / 365.25,
            section["prices"],
            dimension=3,
            smoothing=getattr(config, "smoothing", 1e-6),
        )
        if not fit.available:
            raise ValueError("unavailable source curve reconstruction")
        coefficients.append(fit.coefficients)
        diagnostics.append(fit.diagnostics)
    values = np.asarray(coefficients)
    deltas = np.diff(dates).astype(int) / 365.25
    model = LaguerreTransport(3, cache_max_bytes=0)
    innovations = np.array(
        [
            (right - model.matrix(float(delta)) @ left) / np.sqrt(delta)
            for left, right, delta in zip(values[:-1], values[1:], deltas, strict=True)
        ]
    )
    return dates, values, innovations, diagnostics


def _source_without_donor(section, date, donor_id, config):
    mask = np.asarray(section["ids"]) != donor_id
    if np.count_nonzero(mask) < 3:
        raise ValueError("fewer than three source donors after exclusion")
    fit = fit_cross_section(
        _tenors(section, date)[mask] / 365.25,
        np.asarray(section["prices"])[mask],
        dimension=3,
        smoothing=getattr(config, "smoothing", 1e-6),
    )
    if not fit.available:
        raise ValueError("unavailable leave-one-donor-out source reconstruction")
    return fit.coefficients


def _source_donor_exclusions(section, date, config):
    """Batched exact shape-ridge normal equations for every donor deletion.

    Centering before forming sufficient statistics avoids subtracting large
    settlement-price levels. This is independently checked against the
    augmented least-squares ``fit_cross_section`` implementation.
    """
    prices = np.asarray(section["prices"], dtype=float)
    count = len(prices)
    if count < 4:
        raise ValueError("fewer than three source donors after exclusion")
    evaluation = LaguerreTransport(3, cache_max_bytes=0).evaluate(_tenors(section, date) / 365.25)
    keep = ~np.eye(count, dtype=bool)
    excluded_evaluation = np.broadcast_to(evaluation, (count, count, 3))[keep].reshape(
        count, count - 1, 3
    )
    singular = np.linalg.svd(excluded_evaluation, compute_uv=False)
    if np.any(singular[:, -1] <= 1e-10 * singular[:, 0]):
        raise ValueError("unavailable leave-one-donor-out source reconstruction")
    normal = (evaluation.T @ evaluation - evaluation[:, :, None] * evaluation[:, None, :]) / (
        count - 1
    )
    regularization = getattr(config, "smoothing", 1e-6) * np.linalg.eigvalsh(normal)[:, -1]
    level = float(prices.mean())
    centered = prices - level
    excluded_mean = -centered / (count - 1)
    sums = evaluation.sum(axis=0) - evaluation
    rhs = (
        evaluation.T @ centered - evaluation * centered[:, None] - sums * excluded_mean[:, None]
    ) / (count - 1)
    penalty = np.diag([0.0, 1.0, 1.0])
    hessian = normal + regularization[:, None, None] * penalty
    coefficients = np.linalg.solve(hessian, rhs[:, :, None])[:, :, 0]
    coefficients[:, 0] += level + excluded_mean
    gradient = np.einsum("nij,nj->ni", hessian, coefficients) - (
        evaluation.T @ prices - evaluation * prices[:, None]
    ) / (count - 1)
    relative_residual = np.linalg.norm(gradient, axis=1) / (
        1
        + np.linalg.norm(
            (evaluation.T @ prices - evaluation * prices[:, None]) / (count - 1), axis=1
        )
    )
    if np.any(relative_residual > 1e-8) or not np.all(np.isfinite(coefficients)):
        raise ValueError("unavailable leave-one-donor-out source normal equations")
    return dict(zip(section["ids"], coefficients, strict=True))


def _residual_statistics(
    records, factors, main_evaluations, weights, *, main_loo=None, prepared=None
):
    """Fit pooled hats, joint moment and genuinely excluded-donor errors."""
    residual_coefficients, remaining, hat_rows, loo_errors, loo_weights = [], [], [], [], []
    max_condition = 0.0
    for j, (record, factor, main) in enumerate(
        zip(records, factors, main_evaluations, strict=True)
    ):
        response = np.asarray(record["response"], dtype=float)
        hats = residual_maturity_features(record["tenors_days"])
        operator, geometry = (
            _ridge_operator(hats) if prepared is None else prepared["residual_operators"][j]
        )
        max_condition = max(max_condition, geometry["condition_number"])
        residual = response - main @ factor
        coefficient = operator @ residual
        residual_coefficients.append(coefficient)
        remaining.append(residual - hats @ coefficient)
        if main_loo is not None:
            for k in range(len(response)):
                keep = np.arange(len(response)) != k
                if np.count_nonzero(keep) < 3:
                    continue
                loo_factor = main_loo(j, k)
                loo_operator = (
                    _ridge_operator(hats[keep])[0]
                    if prepared is None
                    else prepared["residual_loo_operators"][j][k]
                )
                loo_coefficient = loo_operator @ (response[keep] - main[keep] @ loo_factor)
                error = response[k] - main[k] @ loo_factor - hats[k] @ loo_coefficient
                hat_rows.append(hats[k])
                loo_errors.append(error)
                loo_weights.append(weights[j] / len(response))
    residual_coefficients = np.asarray(residual_coefficients)
    joint = np.column_stack([factors, residual_coefficients])
    moment = joint.T @ (weights[:, None] * joint)
    diagonal_coefficients = np.zeros(4)
    if main_loo is not None and not loo_errors:
        raise ValueError("no available leave-one-donor-out prediction errors")
    if loo_errors:
        rows = np.asarray(hat_rows)
        errors = np.asarray(loo_errors)
        square_weights = np.sqrt(np.asarray(loo_weights))
        diagonal_coefficients = nnls(rows * square_weights[:, None], errors**2 * square_weights)[0]
    # Signed pooled remaining cross products disclose unmodeled dependence.
    pair_product = pair_scale = pair_weight = 0.0
    for residual, weight in zip(remaining, weights, strict=True):
        if len(residual) > 1:
            pair_product += weight * float(
                (residual.sum() ** 2 - residual @ residual) / (len(residual) * (len(residual) - 1))
            )
            pair_scale += weight * float(np.mean(residual**2))
            pair_weight += weight
    diagnostics = {
        "residual_feature_condition_max": max_condition,
        "leave_one_donor_out_count": len(loo_errors),
        "leave_one_donor_out_unavailable_count": int(
            sum(len(record["response"]) for record in records) - len(loo_errors)
        )
        if main_loo is not None
        else None,
        "leave_one_donor_out_annual_mse": float(
            np.average(np.asarray(loo_errors) ** 2, weights=loo_weights)
        )
        if loo_errors
        else None,
        "remaining_residual_pair_product": pair_product / pair_weight if pair_weight else None,
        "remaining_residual_signed_pair_product_ratio": pair_product / pair_scale
        if pair_scale
        else 0.0,
        "diagonal_scope": "leave-one-donor-out prediction error; includes estimation error, not identified latent noise",
        "residual_restriction": "four correlated maturity features plus nonnegative diagonal prediction-error correction",
    }
    return moment, diagonal_coefficients, diagnostics


def _dfm_training(records, weights, prepared=None):
    evaluations = (
        [orthonormal_spline_basis(record["tenors_days"]) for record in records]
        if prepared is None
        else prepared["dfm_evaluations"]
    )
    reconstructions = (
        [_ridge_operator(evaluation) for evaluation in evaluations]
        if prepared is None
        else prepared["dfm_operators"]
    )
    coefficients = (
        np.array(
            [
                operator[0] @ record["response"]
                for operator, record in zip(reconstructions, records, strict=True)
            ]
        )
        if prepared is None
        else prepared["dfm_coefficients"]
    )
    moment = coefficients.T @ (weights[:, None] * coefficients)
    eigenvalues, vectors = np.linalg.eigh(_symmetric(moment))
    loading = vectors[:, -3:][:, ::-1]
    factors = coefficients @ loading
    main = [evaluation @ loading for evaluation in evaluations]

    def exclude(interval, donor):
        if prepared is not None:
            return prepared["dfm_loo_coefficients"][interval][donor] @ loading
        keep = np.arange(len(records[interval]["response"])) != donor
        return (
            _ridge_operator(evaluations[interval][keep])[0] @ records[interval]["response"][keep]
        ) @ loading

    diagnostics = {
        "spline_basis_size": 6,
        "learned_factor_count": 3,
        "loading_metric": "uniform probability L2 on [30,365] days",
        "coefficient_second_moment_eigenvalues": eigenvalues[::-1].tolist(),
        "third_fourth_relative_eigengap": float(
            (eigenvalues[-3] - eigenvalues[-4]) / max(eigenvalues[-1], np.finfo(float).tiny)
        ),
        "reconstruction_condition_max": max(
            item[1]["condition_number"] for item in reconstructions
        ),
        "minimum_return_design_rank": min(item[1]["rank"] for item in reconstructions),
        "moment_centered": False,
    }
    return loading, factors, main, exclude, diagnostics


def _payload(covariance, components, diagnostics):
    minimum = float(np.linalg.eigvalsh(covariance)[0])
    scale = max(float(np.linalg.norm(covariance, 2)), np.finfo(float).tiny)
    if not np.all(np.isfinite(covariance)) or minimum < -1e-8 * scale:
        raise ValueError("forecast violates numerical PSD feasibility")
    return {
        "available": True,
        "covariance": covariance,
        "components": components,
        "diagnostics": {
            **diagnostics,
            "minimum_eigenvalue": minimum,
            "relative_minimum_eigenvalue": minimum / scale,
        },
        "reason": None,
    }


def _unavailable(reason):
    return {
        "available": False,
        "covariance": None,
        "components": {},
        "diagnostics": {},
        "reason": str(reason),
    }


def _prepare(view, config, records):
    """Only frequency-independent geometry and excluded-donor predictions."""
    from .empirical_short_history_data import daily_sections

    prepared = {}
    try:
        phi = tuple(orthonormal_spline_basis(record["tenors_days"]) for record in records)
        operators = tuple(_ridge_operator(item) for item in phi)
        prepared.update(
            dfm_evaluations=phi,
            dfm_operators=operators,
            dfm_coefficients=np.array(
                [
                    operator[0] @ record["response"]
                    for operator, record in zip(operators, records, strict=True)
                ]
            ),
            dfm_loo_coefficients=tuple(
                np.array(
                    [
                        _ridge_operator(item[np.arange(len(item)) != k])[0]
                        @ record["response"][np.arange(len(item)) != k]
                        for k in range(len(item))
                    ]
                )
                for item, record in zip(phi, records, strict=True)
            ),
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        prepared["dfm_error"] = str(error)
    try:
        hats = tuple(residual_maturity_features(record["tenors_days"]) for record in records)
        prepared["residual_operators"] = tuple(_ridge_operator(item) for item in hats)
        prepared["residual_loo_operators"] = tuple(
            tuple(_ridge_operator(item[np.arange(len(item)) != k])[0] for k in range(len(item)))
            for item in hats
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        prepared["residual_error"] = str(error)
    sections = tuple(daily_sections(view, eligible=True))
    try:
        prepared["source_training"] = _source_training(view, config, sections)
    except (ValueError, np.linalg.LinAlgError) as error:
        prepared["source_error"] = str(error)
        return prepared
    try:
        dates = prepared["source_training"][0]
        section_lookup = {date: section for date, section in zip(dates, sections, strict=True)}
        excluded = {}
        model = LaguerreTransport(3, cache_max_bytes=0)
        loo = []
        for record in records:
            values = []
            for donor in record["ids"]:
                endpoints = []
                for date in (record["left_date"], record["right_date"]):
                    if date not in excluded:
                        excluded[date] = _source_donor_exclusions(
                            section_lookup[date], date, config
                        )
                    endpoints.append(excluded[date][donor])
                values.append(
                    (endpoints[1] - model.matrix(float(record["delta"])) @ endpoints[0])
                    / np.sqrt(record["delta"])
                )
            loo.append(np.asarray(values))
        prepared["source_loo"] = tuple(loo)
    except (ValueError, np.linalg.LinAlgError) as error:
        prepared["source_loo_error"] = str(error)
    return prepared


def model_covariances(view, config, half_life=63, *, cache=None):
    """Issue all model policies from one immutable masked training view only.

    Covariances use CNY/tonne price-change units. No multiplier is applied here.
    The next-session calendar date is known; its prices are never accepted.
    """
    from .empirical_short_history_data import interval_records

    output = {name: _unavailable("not fitted") for name in _MODEL_NAMES}
    if not view.available:
        return {name: _unavailable(view.reason) for name in _MODEL_NAMES}
    records = tuple(interval_records(view, config))
    if len(records) < 2:
        return {
            name: _unavailable("fewer than two valid adjacent training intervals")
            for name in _MODEL_NAMES
        }
    key = (
        view.fingerprint,
        getattr(config, "smoothing", 1e-6),
        getattr(config, "reconstruction_ridge", 1e-6),
    )
    prepared = None if cache is None else cache.get(key)
    if prepared is None:
        prepared = _prepare(view, config, records)
        if cache is not None:
            cache.put(key, prepared)
    future_delta = float((view.next_date - view.origin).astype(int) / 365.25)
    maturity = np.asarray(view.maturities)
    future_tenors = (
        (maturity - view.next_date).astype(int).astype(float)
        if maturity.dtype.kind == "M"
        else np.asarray(maturity, dtype=float) * 365.25 - future_delta * 365.25
    )
    # Origin selection is in [30,365]; next-session extrapolation is flat for
    # DFM loadings and residual hats, and exact half-line for Laguerre loadings.
    dfm_tenors = np.clip(future_tenors, 30, 365)
    psi = residual_maturity_features(future_tenors)
    rank_rtol = getattr(config, "rank_rtol", 1e-10)
    weights = interval_weights(len(records), half_life)
    uniform = interval_weights(len(records))
    common = {
        "training_interval_count": len(records),
        "training_start": str(records[0]["left_date"]),
        "training_end": str(records[-1]["right_date"]),
        "future_interval_years": future_delta,
        "half_life_sessions": half_life,
        "effective_interval_count": float(1 / (weights @ weights)),
        "information_fingerprint": view.fingerprint,
        "target": "next-session price-change second moment",
        "clock": "ACT/365.25",
        "conditional_mean_assumption": "zero",
    }
    for mode, mode_weights in (("ewma", weights), ("uniform", uniform)):
        names = (
            ("dfm_joint", "dfm_factor", "dfm_joint_no_nugget")
            if mode == "ewma"
            else ("dfm_uniform_joint",)
        )
        try:
            if "dfm_error" in prepared:
                raise ValueError(prepared["dfm_error"])
            loading, factors, main, exclude, diagnostics = _dfm_training(
                records, mode_weights, prepared
            )
            b = orthonormal_spline_basis(dfm_tenors) @ loading
            if mode == "ewma":
                factor_moment = factors.T @ (mode_weights[:, None] * factors)
                factor_kernel = future_delta * _symmetric(b @ factor_moment @ b.T)
                output["dfm_factor"] = _payload(
                    factor_kernel,
                    {"factor": factor_kernel},
                    {**common, **diagnostics, "weighting": mode},
                )
            if "residual_error" in prepared:
                raise ValueError(prepared["residual_error"])
            moment, diagonal, residual_diagnostics = _residual_statistics(
                records, factors, main, mode_weights, main_loo=exclude, prepared=prepared
            )
            h_forecast = moment[:3, :3]
            for name in names:
                if name == "dfm_factor":
                    annual = _symmetric(b @ h_forecast @ b.T)
                    components, geometry = {"factor": annual}, {}
                else:
                    correction = psi @ diagonal if name != "dfm_joint_no_nugget" else np.zeros(3)
                    annual, components, geometry = schur_residual_forecast(
                        b, psi, moment, h_forecast, diagonal=correction, rank_rtol=rank_rtol
                    )
                output[name] = _payload(
                    future_delta * annual,
                    {key: future_delta * value for key, value in components.items()},
                    {
                        **common,
                        **diagnostics,
                        **residual_diagnostics,
                        **geometry,
                        "weighting": mode,
                        "half_life_sessions": half_life if mode == "ewma" else None,
                        "effective_interval_count": float(1 / (mode_weights @ mode_weights)),
                    },
                )
        except (ValueError, np.linalg.LinAlgError) as error:
            for name in names:
                if not output[name]["available"]:
                    output[name] = _unavailable(error)
    source_names = ("spde_uniform", "spde_ewma", "spde_joint", "spde_joint_no_nugget")
    try:
        if "source_error" in prepared:
            raise ValueError(prepared["source_error"])
        dates, states, innovations, curve_diagnostics = prepared["source_training"]
        # Never bridge absent intervals: align by their right endpoint.
        lookup = {date: i for i, date in enumerate(dates[1:])}
        indexes = np.array([lookup[record["right_date"]] for record in records])
        factors = innovations[indexes]
        model = LaguerreTransport(3, cache_max_bytes=0)
        main = [model.evaluate(np.asarray(record["tenors_days"]) / 365.25) for record in records]
        evaluation = model.evaluate(future_tenors / 365.25)
        times = (dates - dates[0]).astype(int) / 365.25
        source_weights = interval_weights(len(innovations), half_life)
        fits = {}
        for mode, supplied_weights in (("uniform", None), ("ewma", source_weights)):
            fit = fit_source_covariance(
                states,
                times,
                interval_weights=supplied_weights,
                drift="zero",
                regularization_relative=getattr(config, "relative_regularization", 1e-3),
                trace_cap_multiplier=getattr(config, "trace_cap_multiplier", 100.0),
                numerical_error_budget=getattr(config, "numerical_error_budget", 1e-5),
            )
            if not fit.converged:
                raise ValueError(f"{mode} source optimizer did not meet its declared error budget")
            fits[mode] = fit
            kernel = (
                evaluation
                @ model.cumulative_covariance(fit.source_covariance, future_delta)
                @ evaluation.T
            )
            output[f"spde_{mode}"] = _payload(
                kernel,
                {"factor": kernel},
                {
                    **common,
                    "weighting": mode,
                    "source_fit": dict(fit.diagnostics),
                    "source_basis_id": model.basis_id,
                    "curve_condition_max": max(
                        item["design_condition_number"] for item in curve_diagnostics
                    ),
                },
            )

        def exclude(interval, donor):
            if "source_loo_error" in prepared:
                raise ValueError(prepared["source_loo_error"])
            return prepared["source_loo"][interval][donor]

        if "residual_error" in prepared:
            raise ValueError(prepared["residual_error"])
        moment, diagonal, residual_diagnostics = _residual_statistics(
            records, factors, main, weights, main_loo=exclude, prepared=prepared
        )
        forecast_h = (
            model.cumulative_covariance(fits["ewma"].source_covariance, future_delta) / future_delta
        )
        for name, correction in (
            ("spde_joint", psi @ diagonal),
            ("spde_joint_no_nugget", np.zeros(3)),
        ):
            annual, components, geometry = schur_residual_forecast(
                evaluation, psi, moment, forecast_h, diagonal=correction, rank_rtol=rank_rtol
            )
            output[name] = _payload(
                future_delta * annual,
                {key: future_delta * value for key, value in components.items()},
                {
                    **common,
                    **residual_diagnostics,
                    **geometry,
                    "source_fit": dict(fits["ewma"].diagnostics),
                    "scope": "empirical factor/residual risk correction, not physical forcing recovery",
                },
            )
    except (ValueError, np.linalg.LinAlgError) as error:
        for name in source_names:
            if not output[name]["available"]:
                output[name] = _unavailable(error)
    return output
