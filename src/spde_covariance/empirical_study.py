"""Past-only SHFE copper calibration and next-session risk experiment.

Raw archives, fitted daily curves and prediction-level outcomes are local.
Only allowlisted aggregate evidence is exported. Settlement levels are used
as reduced-form point observations at contract-month midpoint coordinates;
these are neither actual expiry dates nor physical delivery averages.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .empirical_calibration import fit_cross_section, fit_source_covariance, source_risk_error_bound
from .empirical_data import public_data_provenance, public_data_validation
from .laguerre import LaguerreTransport
from .provenance import _git, _jsonable, _source_root


@dataclass(frozen=True)
class EmpiricalConfig:
    """Protocol fixed before reading test outcomes; no automatic test tuning."""

    name: str = "shfe-cu-first-study"
    training_end: str = "2022-12-31"
    validation_end: str = "2023-12-31"
    test_end: str = "2025-12-31"
    dimension: int = 3
    smoothing: float = 1e-6
    relative_regularization: float = 1e-3
    windows: tuple[int, ...] = (63, 126, 252)
    min_tenor_days: int = 30
    max_tenor_days: int = 365
    target_tenors_days: tuple[int, ...] = (90, 180, 270)
    min_direct_count: int = 20
    shrinkage: float = 0.1
    pnl_variance_floor: float = 25.0
    multiplier: float = 5.0
    origin_stride: int = 1
    bootstrap_block: int = 20
    bootstrap_replications: int = 1000
    seed: int = 20261026
    sensitivities: bool = True

    def __post_init__(self):
        dates = [
            np.datetime64(s, "D") for s in (self.training_end, self.validation_end, self.test_end)
        ]
        if not dates[0] < dates[1] < dates[2]:
            raise ValueError("training, validation and test boundaries must increase")
        if self.dimension < 1 or self.origin_stride < 1:
            raise ValueError("dimension and origin_stride must be positive")
        if (
            not self.windows
            or any(w < 2 for w in self.windows)
            or len(set(self.windows)) != len(self.windows)
        ):
            raise ValueError("windows must be distinct interval counts >=2")
        if (
            len(self.target_tenors_days) != 3
            or tuple(sorted(self.target_tenors_days)) != self.target_tenors_days
        ):
            raise ValueError("three increasing target tenors required")
        if not 0 <= self.shrinkage <= 1 or self.min_direct_count < 2:
            raise ValueError("invalid shrinkage or min_direct_count")
        if self.min_tenor_days < 1 or self.max_tenor_days <= self.min_tenor_days:
            raise ValueError("invalid tenor range")
        if not all(
            np.isfinite(v) and v > 0
            for v in (
                self.smoothing,
                self.relative_regularization,
                self.pnl_variance_floor,
                self.multiplier,
            )
        ):
            raise ValueError("scales and regularization must be finite and positive")
        if self.bootstrap_block < 1 or self.bootstrap_replications < 2:
            raise ValueError("invalid bootstrap settings")


def load_empirical_config(path):
    import tomllib

    values = tomllib.loads(Path(path).read_text())
    for key in ("windows", "target_tenors_days"):
        if key in values:
            values[key] = tuple(values[key])
    return EmpiricalConfig(**values)


def quadratic_risk_score(variance, squared_pnl):
    """Proper squared-risk proxy score; no Gaussian-density interpretation."""
    return np.asarray(variance) ** 2 - 2 * np.asarray(variance) * np.asarray(squared_pnl)


def qlike(variance, squared_pnl, *, floor):
    """log(v)+z/v, proper for E[z|past] with v>0, including z=0."""
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be positive")
    value, proxy = np.asarray(variance), np.asarray(squared_pnl)
    if np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ValueError("variance must be finite and nonnegative")
    if np.any(~np.isfinite(proxy)) or np.any(proxy < 0):
        raise ValueError("squared_pnl must be finite and nonnegative")
    positive = np.maximum(value, floor)
    return np.log(positive) + proxy / positive


def paired_block_summary(differences, *, block=20, replications=1000, seed=20261026):
    """Circular moving-block bootstrap of paired daily averages, not IID MCSE."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a nonempty finite vector")
    if block < 1 or replications < 2:
        raise ValueError("invalid bootstrap settings")
    generator = np.random.default_rng(seed)
    block = min(block, len(values))
    starts = generator.integers(0, len(values), (replications, int(np.ceil(len(values) / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % len(values)).reshape(replications, -1)
    means = np.mean(values[indices[:, : len(values)]], axis=1)
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "block_length": block,
        "bootstrap_standard_error": float(np.std(means, ddof=1)),
        "bootstrap_percentile_95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def prepare_daily_curves(panel, config, *, dimension=None, smoothing=None):
    """Reconstruct each session independently; never fill missing sessions."""
    dimension = config.dimension if dimension is None else dimension
    smoothing = config.smoothing if smoothing is None else smoothing
    dates = np.asarray(panel.session_dates, dtype="datetime64[D]")
    dates = dates[dates <= np.datetime64(config.test_end)]
    coefficients = np.full((len(dates), dimension), np.nan)
    sections, settlement_sections, diagnostics = [], [], []
    for i, day in enumerate(dates):
        rows = np.flatnonzero(panel.dates == day)
        quoted = rows[np.isfinite(panel.settlement[rows]) & (panel.settlement[rows] > 0)]
        settlement_sections.append(
            {
                "ids": panel.contract_ids[quoted],
                "maturity": panel.contract_month_dates[quoted],
                "prices": panel.settlement[quoted],
            }
        )
        days = (panel.contract_month_dates[rows] - day).astype(int)
        valid = (
            (days >= config.min_tenor_days)
            & (days <= config.max_tenor_days)
            & np.isfinite(panel.settlement[rows])
            & (panel.settlement[rows] > 0)
            & (panel.volume[rows] > 0)
            & (panel.open_interest[rows] > 0)
        )
        selected = rows[valid]
        order = np.argsort(panel.contract_month_dates[selected])
        selected = selected[order]
        section = {
            "ids": panel.contract_ids[selected],
            "maturity": panel.contract_month_dates[selected],
            "prices": panel.settlement[selected],
        }
        sections.append(section)
        if len(selected) < dimension:
            diagnostics.append({"available": False, "nodes": len(selected)})
            continue
        fit = fit_cross_section(
            (section["maturity"] - day).astype(int) / 365.25,
            section["prices"],
            dimension=dimension,
            smoothing=smoothing,
        )
        if fit.available:
            coefficients[i] = fit.coefficients
        diagnostics.append(
            {
                "available": True,
                "nodes": len(selected),
                **fit.diagnostics,
                "rmse_cny_per_tonne": float(np.sqrt(np.mean(fit.residuals**2))),
                "max_abs_error": float(np.max(np.abs(fit.residuals))),
            }
        )
    return {
        "dates": dates,
        "times": (dates - dates[0]).astype(int) / 365.25,
        "coefficients": coefficients,
        "sections": sections,
        "settlement_sections": settlement_sections,
        "diagnostics": diagnostics,
    }


def select_contracts(section, origin, targets=(90, 180, 270)):
    """Contracts chosen using origin maturity and origin liquidity only."""
    if len(section["ids"]) < 3:
        return None
    days = (section["maturity"] - origin).astype(int)
    selected = [int(np.argmin(np.abs(days - target))) for target in targets]
    if len(set(selected)) != 3:
        return None
    return {key: value[selected] for key, value in section.items()}


EXPOSURES = {
    "outright_3m": np.array([1.0, 0, 0]),
    "spread_3m_6m": np.array([1.0, -1.0, 0]),
    "butterfly_3m_6m_9m": np.array([1.0, -2.0, 1.0]),
}


def _historical_contract_covariance(daily, origin, chosen_ids, window, min_count):
    """Only adjacent same-contract changes; joint complete cases, no rolling."""
    changes = []
    start = max(1, origin - window + 1)
    for i in range(start, origin + 1):
        before = daily["settlement_sections"][i - 1]
        after = daily["settlement_sections"][i]
        lookup0 = dict(zip(before["ids"], before["prices"], strict=True))
        lookup1 = dict(zip(after["ids"], after["prices"], strict=True))
        if all(key in lookup0 and key in lookup1 for key in chosen_ids):
            dt = daily["times"][i] - daily["times"][i - 1]
            changes.append([(lookup1[key] - lookup0[key]) / np.sqrt(dt) for key in chosen_ids])
    if len(changes) < min_count:
        return None, len(changes)
    values = np.asarray(changes)
    return values.T @ values / len(values), len(values)


def _source_fit(daily, origin, window, config, *, adjustment=True):
    start = max(0, origin - window)
    values = daily["coefficients"][start : origin + 1]
    if len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("training window includes an unavailable curve; no gap bridging")
    return fit_source_covariance(
        values,
        daily["times"][start : origin + 1],
        regularization_relative=config.relative_regularization,
        drift="zero",
        transport_adjustment=adjustment,
    )


def issue_risk_forecast(
    daily, origin, window, config, *, next_date, source_fit=None, adjustment=True
):
    """Training-only issuance: receives next DATE, never next prices or truth.

    The known session calendar supplies the physical forecast duration. The
    returned portfolios hold the actual IDs selected at origin to next date.
    """
    date = daily["dates"][origin]
    if np.datetime64(next_date) <= date:
        raise ValueError("next_date must follow origin")
    chosen = select_contracts(daily["sections"][origin], date, config.target_tenors_days)
    if chosen is None:
        return None
    delta = float((np.datetime64(next_date) - date).astype(int)) / 365.25
    fit = (
        _source_fit(daily, origin, window, config, adjustment=adjustment)
        if source_fit is None
        else source_fit
    )
    if not fit.converged:
        raise RuntimeError("source covariance solver did not satisfy numerical budget")
    model = LaguerreTransport(len(fit.source_covariance))
    tenors = (chosen["maturity"] - np.datetime64(next_date)).astype(int) / 365.25
    if np.any(tenors < 0):
        return None
    evaluation = model.evaluate(tenors)
    innovation = (
        model.innovation_covariance(fit.source_covariance, delta)
        if adjustment
        else fit.source_covariance
    )
    kernel = evaluation @ (delta * innovation) @ evaluation.T
    variances = {
        name: max(0.0, float(config.multiplier**2 * a @ kernel @ a))
        for name, a in EXPOSURES.items()
    }
    numerical_bounds = {
        name: source_risk_error_bound(fit, tenors, a, delta, config.multiplier)
        for name, a in EXPOSURES.items()
    }
    return {
        "origin": str(date),
        "next_date": str(np.datetime64(next_date, "D")),
        "training_start": str(daily["dates"][max(0, origin - window)]),
        "training_end": str(date),
        "training_intervals": min(window, origin),
        "contract_ids": chosen["ids"].tolist(),
        "origin_prices": chosen["prices"],
        "delta": delta,
        "variances": variances,
        "diagnostics": fit.diagnostics,
        "source_covariance": fit.source_covariance,
        "risk_numerical_error_bounds": numerical_bounds,
    }


def score_issued_forecast(forecast, future_section, config):
    """Outcome lookup is separate from issuance; missing outcomes stay missing."""
    lookup = dict(zip(future_section["ids"], future_section["prices"], strict=True))
    changes = (
        np.array([lookup.get(key, np.nan) for key in forecast["contract_ids"]])
        - forecast["origin_prices"]
    )
    result = {
        name: (
            float((config.multiplier * a[a != 0] @ changes[a != 0]) ** 2)
            if np.all(np.isfinite(changes[a != 0]))
            else None
        )
        for name, a in EXPOSURES.items()
    }
    return result if any(v is not None for v in result.values()) else None


def select_window(daily, config):
    """Observed validation selection; exact ties prefer the longer window."""
    scores = {window: [] for window in config.windows}
    boundaries = []
    unavailable = 0
    for i in range(max(config.windows), len(daily["dates"]) - 1, config.origin_stride):
        day, next_day = daily["dates"][i : i + 2]
        if day <= np.datetime64(config.training_end) or next_day > np.datetime64(
            config.validation_end
        ):
            continue
        candidates = {}
        for window in config.windows:
            try:
                forecast = issue_risk_forecast(daily, i, window, config, next_date=next_day)
            except (ValueError, RuntimeError):
                unavailable += 1
                continue
            if forecast is None:
                continue
            proxy = score_issued_forecast(forecast, daily["settlement_sections"][i + 1], config)
            if proxy is not None and all(v is not None for v in proxy.values()):
                candidates[window] = float(
                    np.mean(
                        [
                            qlike(
                                forecast["variances"][name],
                                proxy[name],
                                floor=config.pnl_variance_floor,
                            )
                            for name in EXPOSURES
                        ]
                    )
                )
        if len(candidates) == len(config.windows):
            for window, score in candidates.items():
                scores[window].append(score)
            boundaries.append({"origin": str(day), "scored": str(next_day)})
    if not boundaries:
        raise ValueError("no common validation origins; check calendar and protocol")
    means = {w: float(np.mean(values)) for w, values in scores.items()}
    selected = min(config.windows, key=lambda w: (means[w], -w))
    return selected, {
        "selected_window": selected,
        "mean_qlike": means,
        "origins": len(boundaries),
        "boundaries": boundaries,
        "selection_last_outcome": max(item["scored"] for item in boundaries),
        "unavailable_candidate_fits": unavailable,
    }


def select_baseline_windows(daily, config):
    """Direct and shrinkage choose their own windows on validation only."""
    scores = {m: {w: [] for w in config.windows} for m in ("direct", "shrinkage")}
    dates_used = []
    for i in range(max(config.windows), len(daily["dates"]) - 1, config.origin_stride):
        day, next_day = daily["dates"][i : i + 2]
        if day <= np.datetime64(config.training_end) or next_day > np.datetime64(
            config.validation_end
        ):
            continue
        chosen = select_contracts(daily["sections"][i], day, config.target_tenors_days)
        if chosen is None:
            continue
        outcomes = score_issued_forecast(
            {"contract_ids": chosen["ids"], "origin_prices": chosen["prices"]},
            daily["settlement_sections"][i + 1],
            config,
        )
        if outcomes is None or any(value is None for value in outcomes.values()):
            continue
        candidates = {m: {} for m in scores}
        for window in config.windows:
            losses = {m: [] for m in scores}
            for exposure, a in EXPOSURES.items():
                support = a != 0
                cov, _ = _historical_contract_covariance(
                    daily, i, chosen["ids"][support], window, config.min_direct_count
                )
                if cov is None:
                    break
                shrunk = (1 - config.shrinkage) * cov + config.shrinkage * np.diag(np.diag(cov))
                for method, covariance in (("direct", cov), ("shrinkage", shrunk)):
                    variance = float(
                        config.multiplier**2
                        * (next_day - day).astype(int)
                        / 365.25
                        * a[support]
                        @ covariance
                        @ a[support]
                    )
                    losses[method].append(
                        float(
                            qlike(
                                max(0.0, variance),
                                outcomes[exposure],
                                floor=config.pnl_variance_floor,
                            )
                        )
                    )
            for method in scores:
                if len(losses[method]) == len(EXPOSURES):
                    candidates[method][window] = float(np.mean(losses[method]))
        if all(len(values) == len(config.windows) for values in candidates.values()):
            for method in scores:
                for window, value in candidates[method].items():
                    scores[method][window].append(value)
            dates_used.append(str(next_day))
    if not dates_used:
        raise ValueError("no common validation origins for direct baselines")
    means = {
        method: {window: float(np.mean(values)) for window, values in bywindow.items()}
        for method, bywindow in scores.items()
    }
    selected = {
        method: min(config.windows, key=lambda w: (bywindow[w], -w))
        for method, bywindow in means.items()
    }
    return selected, {
        "selected_windows": selected,
        "mean_qlike": means,
        "origins": len(dates_used),
        "selection_last_outcome": max(dates_used),
    }


def summarize_predictions(records, config):
    """Aggregate-only public view; no dates, IDs, individual outcomes or quotes."""
    summary, paired = [], []
    for exposure in EXPOSURES:
        selected = [r for r in records if r["exposure"] == exposure]
        methods = sorted({r["method"] for r in selected})
        for year in ("all", *sorted({r["year"] for r in selected})):
            rows = [r for r in selected if year == "all" or r["year"] == year]
            for method in methods:
                group = [r for r in rows if r["method"] == method]
                if not group:
                    continue
                pred = np.array([r["variance"] for r in group])
                proxy = np.array([r["squared_pnl"] for r in group])
                summary.append(
                    {
                        "exposure": exposure,
                        "year": year,
                        "method": method,
                        "count": len(group),
                        "qlike": float(
                            np.mean(qlike(pred, proxy, floor=config.pnl_variance_floor))
                        ),
                        "mean_quadratic_score": float(np.mean(quadratic_risk_score(pred, proxy))),
                        "risk_ratio": float(np.sum(proxy) / np.sum(pred))
                        if np.sum(pred) > 0
                        else None,
                        "predicted_rms_cny": float(np.sqrt(np.mean(pred))),
                        "realized_rms_cny": float(np.sqrt(np.mean(proxy))),
                        "floor_hits": int(np.sum(pred < config.pnl_variance_floor)),
                    }
                )
            reference = {r["origin"]: r for r in rows if r["method"] == "direct"}
            for method in methods:
                if method == "direct":
                    continue
                common = [r for r in rows if r["method"] == method and r["origin"] in reference]
                if not common:
                    continue
                differences = [
                    float(
                        qlike(r["variance"], r["squared_pnl"], floor=config.pnl_variance_floor)
                        - qlike(
                            reference[r["origin"]]["variance"],
                            r["squared_pnl"],
                            floor=config.pnl_variance_floor,
                        )
                    )
                    for r in common
                ]
                paired.append(
                    {
                        "exposure": exposure,
                        "year": year,
                        "method": method,
                        "reference": "direct",
                        **paired_block_summary(
                            differences,
                            block=config.bootstrap_block,
                            replications=config.bootstrap_replications,
                            seed=config.seed,
                        ),
                    }
                )
    histograms = []
    bins = np.linspace(-6, 6, 49)
    for exposure in EXPOSURES:
        rows = [r for r in records if r["exposure"] == exposure and r["method"] == "spde_local"]
        z = np.array(
            [r["signed_pnl"] / np.sqrt(max(r["variance"], config.pnl_variance_floor)) for r in rows]
        )
        histograms.append(
            {
                "exposure": exposure,
                "bins": bins.tolist(),
                "counts": np.histogram(z, bins)[0].tolist(),
                "outside_count": int(np.sum(np.abs(z) > 6)),
                "count": len(z),
                "mean": float(np.mean(z)) if len(z) else None,
                "second_moment": float(np.mean(z**2)) if len(z) else None,
                "fourth_moment": float(np.mean(z**4)) if len(z) else None,
            }
        )
    return {
        "risk_summary": summary,
        "paired_qlike": paired,
        "standardized_pnl_histograms": histograms,
    }


def run_empirical_study(panel, config, output, *, data_validation=None):
    """Run in a local output directory; public export is an explicit allowlist."""
    output = Path(output)
    repo = _source_root()
    git_root = str(repo) if repo is not None and output.resolve().is_relative_to(repo) else None
    if git_root is not None:
        probe = subprocess.run(
            [
                "git",
                "-C",
                git_root,
                "check-ignore",
                "-q",
                "--no-index",
                str(output.resolve() / "private_predictions.json"),
            ],
            check=False,
        )
        if probe.returncode != 0:
            raise ValueError(
                "empirical output inside a repository must be Git-ignored; use data/local/"
            )
    output.mkdir(parents=True, exist_ok=False)
    start_time = time.perf_counter()
    daily = prepare_daily_curves(panel, config)
    selected, selection = select_window(daily, config)
    baseline_windows, baseline_selection = select_baseline_windows(daily, config)
    training_indices = np.flatnonzero(daily["dates"] <= np.datetime64(config.training_end))
    if len(training_indices) < 3:
        raise ValueError("insufficient initial training data")
    static_end = int(training_indices[-1])
    try:
        static_fit = _source_fit(daily, static_end, static_end, config)
    except (ValueError, RuntimeError):
        static_fit = None
    alternatives = {}
    if config.sensitivities:
        alternatives = {
            "spde_dim2": prepare_daily_curves(panel, config, dimension=2),
            "spde_dim4": prepare_daily_curves(panel, config, dimension=4),
            "spde_smoother": prepare_daily_curves(panel, config, smoothing=1e-4),
        }
    records, audit, failures, fit_times, sources = [], [], [], [], []
    test_origins = []
    for i in range(max(config.windows), len(daily["dates"]) - 1, config.origin_stride):
        day, next_day = daily["dates"][i : i + 2]
        if day <= np.datetime64(config.validation_end) or next_day > np.datetime64(config.test_end):
            continue
        test_origins.append(str(day))
        # Baselines are issued independently of SPDE fit/convergence. Each
        # exposure requires only its own observed contract support.
        chosen = select_contracts(daily["sections"][i], day, config.target_tenors_days)
        if chosen is not None:
            base = {"contract_ids": chosen["ids"], "origin_prices": chosen["prices"]}
            outcomes = score_issued_forecast(base, daily["settlement_sections"][i + 1], config)
            future = dict(
                zip(
                    daily["settlement_sections"][i + 1]["ids"],
                    daily["settlement_sections"][i + 1]["prices"],
                    strict=True,
                )
            )
            if outcomes is not None:
                for exposure, a in EXPOSURES.items():
                    support = a != 0
                    if outcomes[exposure] is None:
                        continue
                    changes = (
                        np.array([future[key] for key in chosen["ids"][support]])
                        - chosen["prices"][support]
                    )
                    for name in ("direct", "shrinkage"):
                        cov, count = _historical_contract_covariance(
                            daily,
                            i,
                            chosen["ids"][support],
                            baseline_windows[name],
                            config.min_direct_count,
                        )
                        if cov is None:
                            continue
                        if name == "shrinkage":
                            cov = (1 - config.shrinkage) * cov + config.shrinkage * np.diag(
                                np.diag(cov)
                            )
                        variance = float(
                            config.multiplier**2
                            * (next_day - day).astype(int)
                            / 365.25
                            * a[support]
                            @ cov
                            @ a[support]
                        )
                        records.append(
                            {
                                "origin": str(day),
                                "year": str(next_day)[:4],
                                "method": name,
                                "exposure": exposure,
                                "variance": max(0.0, variance),
                                "squared_pnl": outcomes[exposure],
                                "signed_pnl": float(config.multiplier * a[support] @ changes),
                                "direct_joint_count": count,
                                "calendar_days": int((next_day - day).astype(int)),
                            }
                        )
        for method, view, window, fixed, adjustment in [
            ("spde_local", daily, selected, None, True),
            ("spde_fixed126", daily, 126, None, True),
            ("spde_static", daily, static_end, static_fit, True),
            ("unadjusted", daily, selected, None, False),
            *[(name, view, selected, None, True) for name, view in alternatives.items()],
        ]:
            if method == "spde_static" and static_fit is None:
                failures.append(
                    {
                        "origin": str(day),
                        "method": method,
                        "reason": "initial static fit unavailable",
                    }
                )
                continue
            tick = time.perf_counter()
            try:
                forecast = issue_risk_forecast(
                    view,
                    i,
                    window,
                    config,
                    next_date=next_day,
                    source_fit=fixed,
                    adjustment=adjustment,
                )
            except (ValueError, RuntimeError) as error:
                failures.append({"origin": str(day), "method": method, "reason": str(error)})
                continue
            fit_times.append(time.perf_counter() - tick)
            if forecast is None:
                failures.append(
                    {
                        "origin": str(day),
                        "method": method,
                        "reason": "contract selection unavailable",
                    }
                )
                continue
            if fixed is not None:
                forecast["training_end"] = str(daily["dates"][static_end])
                forecast["training_start"] = str(daily["dates"][0])
            outcomes = score_issued_forecast(forecast, daily["settlement_sections"][i + 1], config)
            audit.append(
                {key: value for key, value in forecast.items() if key != "origin_prices"}
                | {
                    "method": method,
                    "scoring_available": {
                        e: outcomes is not None and outcomes[e] is not None for e in EXPOSURES
                    },
                }
            )
            if outcomes is None:
                failures.append(
                    {
                        "origin": str(day),
                        "method": method,
                        "reason": "future same-contract outcome unavailable",
                    }
                )
                continue
            future_quotes = daily["settlement_sections"][i + 1]
            next_lookup = dict(zip(future_quotes["ids"], future_quotes["prices"], strict=True))
            changes = (
                np.array([next_lookup.get(key, np.nan) for key in forecast["contract_ids"]])
                - forecast["origin_prices"]
            )
            if method == "spde_local":
                sources.append(forecast["source_covariance"])
            for name, values in {method: forecast["variances"]}.items():
                for exposure, variance in values.items():
                    if outcomes[exposure] is None:
                        continue
                    support = EXPOSURES[exposure] != 0
                    records.append(
                        {
                            "origin": str(day),
                            "year": str(next_day)[:4],
                            "method": name,
                            "exposure": exposure,
                            "variance": variance,
                            "squared_pnl": outcomes[exposure],
                            "signed_pnl": float(
                                config.multiplier * EXPOSURES[exposure][support] @ changes[support]
                            ),
                            "direct_joint_count": None,
                            "calendar_days": int((next_day - day).astype(int)),
                        }
                    )
    if not records:
        raise ValueError("no test forecasts could be scored")
    available = [d for d in daily["diagnostics"] if d["available"]]
    reconstruction = {
        "sessions": len(daily["dates"]),
        "available_sessions": len(available),
        "node_count_range": [
            min(d["nodes"] for d in available),
            max(d["nodes"] for d in available),
        ],
        "rmse_quantiles_cny_per_tonne": np.quantile(
            [d["rmse_cny_per_tonne"] for d in available], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "design_condition_quantiles": np.quantile(
            [d["design_condition_number"] for d in available], [0, 0.5, 1]
        ).tolist(),
    }
    numerical = (
        {
            key: max(float(r["diagnostics"][key]) for r in audit)
            for key in ("relative_covariance_error_bound", "objective_gap_bound_scaled")
        }
        if audit
        else {}
    )
    numerical["maximum_relative_risk_bound"] = max(
        (
            r["risk_numerical_error_bounds"][e] / max(r["variances"][e], config.pnl_variance_floor)
            for r in audit
            for e in EXPOSURES
        ),
        default=None,
    )
    numerical["available"] = bool(audit)
    numerical["all_converged"] = bool(audit) and all(r["diagnostics"]["converged"] for r in audit)
    numerical["minimum_source_eigenvalue"] = min(
        (float(r["diagnostics"]["minimum_source_eigenvalue"]) for r in audit), default=None
    )
    numerical["maximum_relative_trace_violation"] = max(
        (
            max(0.0, float(r["diagnostics"]["source_trace"]) - float(r["diagnostics"]["trace_cap"]))
            / max(1.0, float(r["diagnostics"]["trace_cap"]))
            for r in audit
        ),
        default=None,
    )
    interval_groups = []
    for exposure in EXPOSURES:
        for label, lower, upper in (("1 day", 1, 1), ("2–3 days", 2, 3), ("4+ days", 4, 100)):
            rows = [
                r
                for r in records
                if r["method"] == "spde_local"
                and r["exposure"] == exposure
                and lower <= r["calendar_days"] <= upper
            ]
            if rows and sum(r["variance"] for r in rows) > 0:
                interval_groups.append(
                    {
                        "exposure": exposure,
                        "calendar_interval": label,
                        "count": len(rows),
                        "risk_ratio": sum(r["squared_pnl"] for r in rows)
                        / sum(r["variance"] for r in rows),
                    }
                )
    summary = {
        "schema": "shfe-empirical-aggregate-v1",
        "config": asdict(config),
        "data_audit": public_data_provenance(panel.provenance),
        "data_overview": {
            "contract_rows": len(panel.dates),
            "distinct_contracts": len(np.unique(panel.contract_ids)),
            "sessions": len(panel.session_dates),
            "settlement_missing": int(np.sum(~np.isfinite(panel.settlement))),
            "settlement_nonpositive": int(np.sum(panel.settlement <= 0)),
            "zero_volume": int(np.sum(panel.volume == 0)),
            "zero_open_interest": int(np.sum(panel.open_interest == 0)),
        },
        "data_validation": public_data_validation(data_validation),
        "reconstruction": reconstruction,
        "numerical_diagnostics": numerical,
        "calendar_interval_diagnostics": interval_groups,
        "representation_sensitivity": {
            name: {
                "rmse_quantiles_cny_per_tonne": np.quantile(
                    [d["rmse_cny_per_tonne"] for d in view["diagnostics"] if d["available"]],
                    [0, 0.5, 1],
                ).tolist(),
                "design_condition_quantiles": np.quantile(
                    [d["design_condition_number"] for d in view["diagnostics"] if d["available"]],
                    [0, 0.5, 1],
                ).tolist(),
            }
            for name, view in alternatives.items()
        },
        "selection": {k: v for k, v in selection.items() if k != "boundaries"},
        "baseline_selection": baseline_selection,
        "test_origin_count": len({r["origin"] for r in records}),
        "attempted_test_origin_count": len(test_origins),
        "availability": [
            {
                "method": method,
                "exposure": exposure,
                "attempted": len(test_origins),
                "issued": (
                    sum(r["method"] == method for r in audit)
                    if method not in ("direct", "shrinkage")
                    else None
                ),
                "scored": sum(r["method"] == method and r["exposure"] == exposure for r in records),
            }
            for method in (
                "direct",
                "shrinkage",
                "spde_local",
                "spde_fixed126",
                "spde_static",
                "unadjusted",
                *alternatives,
            )
            for exposure in EXPOSURES
        ],
        "test_period": [min(r["origin"] for r in records), max(r["origin"] for r in records)],
        "failure_count": len(failures),
        "failure_reasons": sorted({r["reason"] for r in failures}),
        "mean_source_covariance": np.mean(sources, axis=0).tolist() if sources else None,
        "source_eigenvalue_quantiles": np.quantile(
            np.linalg.eigvalsh(sources), [0, 0.5, 1], axis=0
        ).tolist()
        if sources
        else None,
        "fit_runtime_seconds_quantiles": np.quantile(fit_times, [0, 0.5, 0.95, 1]).tolist()
        if fit_times
        else None,
        "wall_seconds": time.perf_counter() - start_time,
        **summarize_predictions(records, config),
    }
    root = _source_root()
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "python-calamine"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        "git_revision": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "data_provenance": public_data_provenance(panel.provenance),
        "data_validation": public_data_validation(data_validation),
        "experiment": "historical SHFE settlement risk; not independent Gaussian simulation",
    }
    for name, value in (
        ("summary.json", summary),
        ("manifest.json", manifest),
        ("private_predictions.json", records),
        ("private_forecast_audit.json", audit),
        ("private_selection.json", selection),
        ("private_failures.json", failures),
    ):
        (output / name).write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    return summary


def export_empirical_evidence(local_run, destination):
    """Only aggregate summaries and source hashes leave the private local run."""
    import shutil

    import matplotlib

    matplotlib.use("Agg", force=True)

    local_run, destination = Path(local_run), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "manifest.json"):
        shutil.copyfile(local_run / name, destination / name)
    summary = json.loads((destination / "summary.json").read_text())
    render_empirical_figures(summary, destination)
    write_empirical_report(summary, destination / "report.md")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.iterdir())
        if p.is_file()
    }
    (destination / "evidence-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return destination


def render_empirical_figures(summary, destination=None):
    """Plain Matplotlib aggregate diagnostics, with no price histories."""
    import matplotlib.pyplot as plt

    figures = []
    methods = ["spde_local", "direct", "shrinkage", "spde_static", "unadjusted"]
    labels = ["SPDE local", "Direct", "Shrinkage", "SPDE static", "Unadjusted"]
    exposures = list(EXPOSURES)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, exposure in zip(axes, exposures, strict=True):
        rows = {
            r["method"]: r
            for r in summary["risk_summary"]
            if r["year"] == "all" and r["exposure"] == exposure
        }
        ax.bar(
            np.arange(len(methods)),
            [rows[m]["risk_ratio"] if m in rows else np.nan for m in methods],
            color="0.45",
        )
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.set_xticks(range(len(methods)), labels, rotation=60, ha="right")
        ax.set_title(exposure.replace("_", " "), fontsize=10)
        ax.set_ylabel("Realized squared P&L / predicted variance")
    figures.append(("risk-calibration.png", fig))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, exposure in zip(axes, exposures, strict=True):
        rows = [
            r for r in summary["paired_qlike"] if r["year"] == "all" and r["exposure"] == exposure
        ]
        means = np.array([r["mean"] for r in rows])
        # Show bootstrap SE, not a coverage guarantee for a fitted latent operator.
        errors = [r["bootstrap_standard_error"] for r in rows]
        ax.errorbar(means, np.arange(len(rows)), xerr=errors, fmt="o", color="black", capsize=3)
        ax.axvline(0, color="0.5", ls="--", lw=1)
        ax.set_yticks(range(len(rows)), [r["method"].replace("spde_", "") for r in rows])
        ax.set_title(exposure.replace("_", " "), fontsize=10)
        ax.set_xlabel("Paired QLIKE minus direct (± block bootstrap SE)")
    figures.append(("paired-risk-scores.png", fig))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for ax, item in zip(axes, summary["standardized_pnl_histograms"], strict=True):
        bins, counts = np.array(item["bins"]), np.array(item["counts"])
        centers, widths = (bins[:-1] + bins[1:]) / 2, np.diff(bins)
        total = counts.sum() + item["outside_count"]
        if total:
            ax.bar(centers, counts / (total * widths), width=widths, color="0.6", label="Empirical")
        else:
            ax.text(0.5, 0.5, "SPDE forecast unavailable", transform=ax.transAxes, ha="center")
        x = np.linspace(-6, 6, 300)
        ax.plot(x, np.exp(-x * x / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1) reference")
        ax.set_xlabel("Next-session P&L / predicted standard deviation")
        ax.set_ylabel("Density")
        ax.set_title(item["exposure"].replace("_", " "), fontsize=10)
        ax.legend(fontsize=8)
    figures.append(("standardized-pnl.png", fig))
    if destination is not None:
        for name, fig in figures:
            fig.savefig(Path(destination) / name, dpi=160)
            plt.close(fig)
    return figures


def write_empirical_report(summary, path):
    lines = [
        "# First SHFE copper calibration and risk study",
        "",
        "This is a historical settlement-risk experiment, not a trading strategy or a proof of the latent SPDE model.",
        "",
        f"Test origins: {summary['test_period'][0]}–{summary['test_period'][1]} ({summary['test_origin_count']} origins). "
        f"Validation selected a trailing window of {summary['selection']['selected_window']} intervals.",
        f"Direct and shrinkage baselines independently selected {summary['baseline_selection']['selected_windows']}.",
        "",
        f"Data: {summary['data_overview']['contract_rows']} copper rows, {summary['data_overview']['sessions']} sessions "
        f"and {summary['data_overview']['distinct_contracts']} contract IDs. "
        f"Cross-sectional RMSE [min, Q1, median, Q3, max] in CNY/tonne: "
        f"{np.round(summary['reconstruction']['rmse_quantiles_cny_per_tonne'], 2).tolist()}.",
        "",
        "The finite exact Laguerre model reconstructs daily curves at contract-month midpoint tenors. "
        "Source covariance C is fitted through the actual calendar-interval integration maps; "
        "it is a model-dependent calibration, not identified physical forcing. Drift is set to zero. "
        "The scored squared P&L is a conditional second-moment proxy, approximating variance under that drift assumption.",
        "",
        "## Out-of-sample risk calibration",
        "",
        "| Exposure | Method | N | Realized / predicted variance | QLIKE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["risk_summary"]:
        if row["year"] == "all":
            ratio = "unavailable" if row["risk_ratio"] is None else f"{row['risk_ratio']:.3f}"
            lines.append(
                f"| {row['exposure']} | {row['method']} | {row['count']} | {ratio} | {row['qlike']:.3f} |"
            )
    lines += [
        "",
        "QLIKE levels include log(CNY²), and comparisons are valid within each exposure on common dates. "
        "Availability differs by method; paired differences below use the intersection of each method's and the direct baseline's available dates.",
        "",
        "## Paired score comparisons",
        "",
        "| Exposure | Method minus direct | N | Mean QLIKE difference | Block bootstrap SE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["paired_qlike"]:
        if row["year"] == "all":
            lines.append(
                f"| {row['exposure']} | {row['method']} | {row['count']} | {row['mean']:.4f} | {row['bootstrap_standard_error']:.4f} |"
            )
    lines += [
        "",
        "Negative differences favor the listed method. Blocks contain 20 consecutive scored origins; "
        "these are time-series bootstrap summaries, not independent Monte Carlo replications or operator confidence intervals.",
        "",
        "![Risk calibration](risk-calibration.png)",
        "![Paired scores](paired-risk-scores.png)",
        "![Standardized P&L](standardized-pnl.png)",
        "",
        "## Calendar-clock check",
        "",
        "| Exposure | Calendar interval | N | Realized / predicted variance |",
        "| --- | --- | ---: | ---: |",
        *[
            f"| {r['exposure']} | {r['calendar_interval']} | {r['count']} | {r['risk_ratio']:.3f} |"
            for r in summary["calendar_interval_diagnostics"]
        ],
        "",
        "## Numerical verification",
        "",
        f"Convergence and feasibility summary: `{json.dumps(summary['numerical_diagnostics'], sort_keys=True)}`. "
        "The risk bound propagates numerical distance to the constrained optimizer; it excludes sampling, reconstruction and model error. "
        "Detailed attempted/issued/scored availability is retained by method and exposure in `summary.json`.",
        "",
        "## Interpretation and limits",
        "",
        "- One next-session holding period, actual contract IDs, 5-tonne multipliers, no continuous-contract roll returns.",
        "- Window selection uses 2023 outcomes only; 2024–2025 are subsequent tests. All refits use observations through origin.",
        "- Direct and fixed 10% diagonal shrinkage baselines use jointly observed changes of the same current contracts, with at least 20 historical intervals.",
        "- Curve reconstruction, short maturity range, quote microstructure, calendar-time volatility and the constant-local-C assumption affect fitted source covariance.",
        "- No known fresh noise correction is claimed for settlement-level reconstruction; no Gaussian tail guarantee, VaR validation, execution costs or profitable-trading claim.",
        "- The session calendar is treated as known. Annual archive versions are retrospective; first-publication-vintage data have not been reconstructed.",
        "- Raw quotes, daily curves and forecast-level outcomes stay local. This report contains aggregate numerical findings only.",
        "",
        "See [the empirical protocol](../../../docs/shfe-empirical.md) and [the executed notebook](../../../notebooks/shfe_empirical.ipynb). "
        "`summary.json`, `manifest.json` and `evidence-sha256.json` retain configuration, archive hashes, numerical source and environment provenance.",
        "",
    ]
    Path(path).write_text("\n".join(lines))
