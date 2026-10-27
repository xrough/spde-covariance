"""Masked-history copper forecasts, future-only scoring and aggregate evidence.

No issuer accepts the original panel. Contract-level forecasts and P&Ls are
written only to a local ignored run; the evidence exporter has a fixed whitelist.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .empirical_data import ShfeCuPanel, public_data_provenance
from .empirical_joint import bounded_minimum_variance_hedge
from .empirical_research import _local_output, _package_versions
from .empirical_short_history_data import (
    ShortHistoryConfig,
    ShortHistoryEpisode,
    ShortHistoryView,
    baseline_covariances,
    build_short_history_view,
    load_short_history_config,
    make_short_history_episodes,
)
from .empirical_short_history_models import BoundedModelCache, model_covariances
from .empirical_study import qlike, quadratic_risk_score
from .hedging import _snapshot
from .provenance import _git, _jsonable, _source_root

PRIMARY_METHODS = (
    "direct",
    "pooled_shrink",
    "spde_uniform",
    "spde_ewma",
    "spde_joint",
    "dfm_joint",
    "dfm_uniform_joint",
    "dfm_factor",
)
SENSITIVITY_METHODS = (
    "direct",
    "pooled_shrink",
    "spde_ewma",
    "spde_joint",
    "dfm_joint",
    "dfm_factor",
)
NO_DIAGONAL_METHODS = ("spde_joint_no_nugget", "dfm_joint_no_nugget")
EVIDENCE_FILES = (
    "summary.json",
    "paired.json",
    "diagnostics.json",
    "protocol.json",
    "manifest.json",
)


def _write(path, value):
    path.write_text(json.dumps(_jsonable(value), allow_nan=False, indent=2) + "\n")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _budget_label(value):
    return "all" if value is None else str(value)


def _compact(value):
    """Keep numerical certificates, summarize long local diagnostic vectors."""
    if isinstance(value, np.ndarray) and value.size > 25:
        finite = value[np.isfinite(value)]
        return {
            "count": value.size,
            "finite_count": finite.size,
            "min_median_max": np.quantile(finite, [0, 0.5, 1]).tolist() if finite.size else [],
        }
    if hasattr(value, "items"):
        return {str(k): _compact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_compact(v) for v in value]
    return _jsonable(value)


@dataclass(frozen=True, slots=True, eq=False)
class IssuedShortHistoryRisk:
    """An immutable forecast issued before the future outcome is accepted."""

    method: str
    origin: str
    next_date: str
    phase: str
    history_budget: str
    target_tenor_days: int
    evaluation_ids: tuple[str, str, str]
    origin_prices: np.ndarray
    exposures: dict
    covariance: np.ndarray | None
    components: dict
    available: bool
    reason: str | None
    diagnostics: dict
    half_life: float
    multiplier: float
    pnl_variance_floor: float
    hedge_bounds: tuple[float, float]

    def __post_init__(self):
        object.__setattr__(self, "origin_prices", np.asarray(self.origin_prices, dtype=float))
        if self.covariance is not None:
            object.__setattr__(self, "covariance", np.asarray(self.covariance, dtype=float))
        for name in ("exposures", "components"):
            object.__setattr__(
                self, name, {k: np.asarray(v, dtype=float) for k, v in getattr(self, name).items()}
            )
        for name in ("origin_prices", "exposures", "covariance", "components", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))


def issue_short_history_risks(view, config, *, half_life=None, cache=None):
    """Issue all policies from a *masked view*, without any future price input."""
    half_life = config.half_life if half_life is None else half_life
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive")
    key = (
        "issued_payloads",
        view.fingerprint,
        float(half_life),
        json.dumps(asdict(config), sort_keys=True),
    )
    payloads = cache.get(key) if cache is not None else None
    if payloads is None:
        payloads = {
            **baseline_covariances(view, config, half_life),
            **model_covariances(view, config, half_life, cache=cache),
        }
        payloads = _snapshot(payloads)
        if cache is not None:
            cache.put(key, payloads)
    butterfly = np.zeros(3)
    butterfly[np.asarray(view.episode.maturity_order)] = [1.0, -2.0, 1.0]
    exposures = {
        "outright": np.array([1.0, 0.0, 0.0]),
        "spread": np.array([1.0, -1.0, 0.0]),
        "butterfly": butterfly,
    }
    issued = {}
    for method, payload in payloads.items():
        covariance = payload["covariance"]
        if payload["available"]:
            if covariance is None or np.shape(covariance) != (3, 3):
                raise ValueError(f"{method}: expected a three-contract covariance")
            if not np.all(np.isfinite(covariance)):
                raise ValueError(f"{method}: nonfinite covariance")
            eigenvalues = np.linalg.eigvalsh(covariance)
            scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
            if eigenvalues[0] < -1e-10 * scale:
                raise ValueError(f"{method}: materially indefinite forecast")
        issued[method] = IssuedShortHistoryRisk(
            method,
            str(view.origin),
            str(view.next_date),
            view.episode.phase,
            _budget_label(view.history_budget),
            view.target_tenor_days,
            view.evaluation_ids,
            view.origin_prices,
            exposures,
            covariance,
            payload.get("components", {}),
            payload["available"],
            payload["reason"],
            {
                **payload["diagnostics"],
                "information_fingerprint": view.fingerprint,
                "training_session_start": str(view.dates[0]),
                "training_session_end": str(view.dates[-1]),
                "training_calendar_intervals": len(view.dates) - 1,
            },
            float(half_life),
            config.multiplier,
            config.pnl_variance_floor,
            config.hedge_bounds,
        )
    return issued


def issue_short_history_risk(view, config, *, method="dfm_joint", half_life=None, cache=None):
    """Singular convenience interface; cache makes repeated policy requests cheap."""
    results = issue_short_history_risks(view, config, half_life=half_life, cache=cache)
    if method not in results:
        raise ValueError(f"unknown short-history method {method!r}")
    return results[method]


def score_short_history_risk(issued, *, future_date, future_ids, future_prices):
    """Score an already-issued object, using only its same-ID next-session prices."""
    if str(np.datetime64(future_date, "D")) != issued.next_date:
        raise ValueError("outcome must be the forecast's known next session")
    ids, prices = np.asarray(future_ids), np.asarray(future_prices, dtype=float)
    if ids.ndim != 1 or prices.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("future IDs and prices must be aligned and unique")
    result = {
        "origin": issued.origin,
        "next_date": issued.next_date,
        "phase": issued.phase,
        "budget": issued.history_budget,
        "tenor": issued.target_tenor_days,
        "method": issued.method,
        "half_life": issued.half_life,
        "available": issued.available,
        "reason": issued.reason,
        "scored": False,
        "diagnostics": _compact(issued.diagnostics),
        "metrics": {},
    }
    if not issued.available:
        return result
    lookup = dict(zip(ids.tolist(), prices.tolist(), strict=True))
    if any(
        key not in lookup or not np.isfinite(lookup[key]) or lookup[key] <= 0
        for key in issued.evaluation_ids
    ):
        result["reason"] = "missing_positive_same_contract_future_settlement"
        return result
    pnl = issued.multiplier * (
        np.array([lookup[key] for key in issued.evaluation_ids]) - issued.origin_prices
    )
    covariance = issued.multiplier**2 * issued.covariance
    metrics = {}
    for name, exposure in issued.exposures.items():
        variance = max(float(exposure @ covariance @ exposure), 0.0)
        squared = float(exposure @ pnl) ** 2
        metrics[name] = {
            "forecast": variance,
            "realized": squared,
            "qlike": float(qlike(variance, squared, floor=issued.pnl_variance_floor)),
            "quadratic": float(quadratic_risk_score(variance, squared)),
            "floor_active": int(variance < issued.pnl_variance_floor),
            "components": {
                key: float(exposure @ (issued.multiplier**2 * value) @ exposure)
                for key, value in issued.components.items()
            },
        }
    metrics["joint"] = {"quadratic": float(np.sum(covariance**2) - 2 * pnl @ covariance @ pnl)}
    hedge = bounded_minimum_variance_hedge(covariance, bounds=issued.hedge_bounds)
    positions = np.r_[1.0, hedge["hedge_positions"]]
    metrics["hedge"] = {
        "forecast": max(float(positions @ covariance @ positions), 0.0),
        "realized": float(positions @ pnl) ** 2,
        "bound_active": int(
            np.any(
                np.isclose(positions[1:], issued.hedge_bounds[0])
                | np.isclose(positions[1:], issued.hedge_bounds[1])
            )
        ),
        "position_norm": float(np.linalg.norm(positions[1:])),
    }
    result.update(scored=True, metrics=metrics)
    # These fields remain strictly local; the public exporter cannot copy them.
    result["private"] = {
        "ids": issued.evaluation_ids,
        "pnl": pnl,
        "hedge_positions": positions,
        "forecast": covariance,
        "information_fingerprint": issued.diagnostics["information_fingerprint"],
    }
    return result


def _origin_mean(values):
    grouped = defaultdict(list)
    for date, value in values:
        if np.isfinite(value):
            grouped[date].append(float(value))
    return {date: float(np.mean(rows)) for date, rows in grouped.items()}


def aggregate_short_history(records, config):
    """Aggregate targets within each calendar origin before any pooled mean."""
    grouped = defaultdict(list)
    for row in records:
        for tenor in ("all", str(row["tenor"])):
            key = (row["phase"], row["budget"], tenor, row["method"], row["half_life"])
            grouped[key].append(row)
    summaries = []
    for key, rows in sorted(grouped.items()):
        phase, budget, tenor, method, half_life = key
        scored = [r for r in rows if r["scored"]]
        metrics = {}
        for exposure in ("outright", "spread", "butterfly", "joint", "hedge"):
            names = sorted({k for r in scored for k in r["metrics"][exposure] if k != "components"})
            values = {}
            for metric in names:
                by_origin = _origin_mean(
                    [(r["origin"], r["metrics"][exposure][metric]) for r in scored]
                )
                values[metric] = float(np.mean(list(by_origin.values()))) if by_origin else None
            if "forecast" in values and values.get("realized", 0) > 0:
                values["calibration_ratio"] = values["forecast"] / values["realized"]
                values["realized_rms_cny"] = float(np.sqrt(values["realized"]))
            components = sorted(
                {c for r in scored for c in r["metrics"][exposure].get("components", {})}
            )
            values["components"] = {}
            for component in components:
                by_origin = _origin_mean(
                    [
                        (
                            r["origin"],
                            r["metrics"][exposure].get("components", {}).get(component, 0.0),
                        )
                        for r in scored
                    ]
                )
                values["components"][component] = float(np.mean(list(by_origin.values())))
            metrics[exposure] = values
        summaries.append(
            {
                "phase": phase,
                "budget": budget,
                "tenor": tenor,
                "method": method,
                "half_life": half_life,
                "episodes": len(rows),
                "available": sum(r["available"] for r in rows),
                "scored": len(scored),
                "origins": len({r["origin"] for r in scored}),
                "failures": dict(Counter(str(r["reason"]) for r in rows if not r["scored"])),
                "metrics": metrics,
            }
        )
    return {
        "aggregation": "equal targets within origin, then equal calendar origins",
        "units": "five-tonne CNY P&L second moments",
        "rows": summaries,
    }


def paired_short_history(records, config):
    """Joint calendar-block resampling; neither targets nor budgets are IID draws."""
    output = []
    blocks = (config.bootstrap_block, *config.bootstrap_sensitivity_blocks)
    for phase in sorted({r["phase"] for r in records}):
        phase_rows = [r for r in records if r["phase"] == phase]
        origins = sorted({r["origin"] for r in phase_rows})
        n = len(origins)
        if not n:
            continue
        position = {date: index for index, date in enumerate(origins)}
        # One index draw per phase/block is reused for EVERY model/target/budget.
        draws = {}
        for block in dict.fromkeys(blocks):
            rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, block, int(phase == "evaluation")])
            )
            starts = rng.integers(
                0, n, size=(config.bootstrap_replications, int(np.ceil(n / block)))
            )
            draws[block] = ((starts[..., None] + np.arange(block)) % n).reshape(
                config.bootstrap_replications, -1
            )[:, :n]
        cells = defaultdict(dict)
        for row in phase_rows:
            cells[(row["budget"], row["half_life"])][
                (row["origin"], row["tenor"], row["method"])
            ] = row
        for (budget, half_life), lookup in sorted(cells.items()):
            methods = sorted({key[2] for key in lookup})
            references = [
                m for m in ("spde_joint", "pooled_shrink", "direct", "spde_uniform") if m in methods
            ]
            for method in methods:
                for reference in references:
                    if method == reference:
                        continue
                    for tenor in ("all", *(str(t) for t in config.target_tenors_days)):
                        for exposure, metric in (
                            ("outright", "qlike"),
                            ("spread", "qlike"),
                            ("butterfly", "qlike"),
                            ("joint", "quadratic"),
                            ("hedge", "realized"),
                        ):
                            differences = []
                            for (date, maturity, name), row in lookup.items():
                                if name != method or (tenor != "all" and str(maturity) != tenor):
                                    continue
                                other = lookup.get((date, maturity, reference))
                                if row["scored"] and other is not None and other["scored"]:
                                    differences.append(
                                        (
                                            date,
                                            row["metrics"][exposure][metric]
                                            - other["metrics"][exposure][metric],
                                        )
                                    )
                            origin_values = _origin_mean(differences)
                            if not origin_values:
                                continue
                            vector = np.full(n, np.nan)
                            for date, value in origin_values.items():
                                vector[position[date]] = value
                            intervals = []
                            for block, indices in draws.items():
                                sampled = vector[indices]
                                valid = np.sum(np.isfinite(sampled), axis=1)
                                bootstrap = np.nansum(sampled, axis=1)[valid > 0] / valid[valid > 0]
                                intervals.append(
                                    {
                                        "block_origins": block,
                                        "bootstrap_se": float(np.std(bootstrap, ddof=1)),
                                        "percentile_95": np.quantile(
                                            bootstrap, [0.025, 0.975]
                                        ).tolist(),
                                    }
                                )
                            output.append(
                                {
                                    "phase": phase,
                                    "budget": budget,
                                    "tenor": tenor,
                                    "half_life": half_life,
                                    "method": method,
                                    "reference": reference,
                                    "exposure": exposure,
                                    "metric": metric,
                                    "paired_episodes": len(differences),
                                    "paired_origins": len(origin_values),
                                    "difference": float(np.mean(list(origin_values.values()))),
                                    "bootstrap": intervals,
                                }
                            )
    return {
        "sign": "method minus reference; lower is better",
        "scope": "descriptive paired calendar-origin bootstrap on inspected years",
        "joint_resampling": True,
        "replications": config.bootstrap_replications,
        "rows": output,
    }


def _diagnostic_summary(records, timings, cache):
    by_method = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)
    output = []
    wanted = (
        "effective_interval_count",
        "curve_condition_max",
        "relative_support_residual",
        "schur_psd_roundoff_correction",
        "third_fourth_relative_eigengap",
        "forecast_psd_roundoff_correction",
        "component_sum_relative_residual",
        "remaining_residual_signed_pair_product_ratio",
        "residual_feature_condition_max",
        "leave_one_donor_out_unavailable_count",
    )
    for method, rows in sorted(by_method.items()):
        numeric = defaultdict(list)
        for row in rows:
            diagnostics = row["diagnostics"]
            for key in wanted:
                value = diagnostics.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    numeric[key].append(value)
            source = diagnostics.get("source_fit", {})
            for key in (
                "relative_covariance_error_bound",
                "relative_objective_gap_bound",
                "objective_gap_bound",
                "iterations",
                "source_trace",
                "trace_cap",
            ):
                value = source.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    numeric["source_" + key].append(value)
        output.append(
            {
                "method": method,
                "issued": len(rows),
                "available": sum(r["available"] for r in rows),
                "scored": sum(r["scored"] for r in rows),
                "failures": dict(Counter(str(r["reason"]) for r in rows if not r["scored"])),
                "numeric_min_median_max": {
                    k: np.quantile(v, [0, 0.5, 1]).tolist() for k, v in numeric.items()
                },
            }
        )
    return {
        "methods": output,
        "cache": cache.diagnostics,
        "episode_runtime_seconds_min_median_max": np.quantile(timings, [0, 0.5, 1]).tolist()
        if timings
        else [],
        "timing_scope": "per target episode, all budgets/methods/half-lives; includes masked preparation",
    }


_PROCESS_CACHE = None


def _masked_wire(view):
    """Serialize retained data only; process workers never receive the full panel."""
    panel_fields = (
        "dates",
        "contract_ids",
        "contract_month_dates",
        "settlement",
        "volume",
        "open_interest",
        "session_dates",
    )
    return {
        "panel": {name: getattr(view.panel, name) for name in panel_fields},
        "episode": asdict(view.episode),
        **{
            f.name: getattr(view, f.name)
            for f in fields(view)
            if f.name not in ("panel", "episode")
        },
    }


def _process_issue(job):
    global _PROCESS_CACHE
    wire, config, half_lives = job
    if _PROCESS_CACHE is None:
        _PROCESS_CACHE = BoundedModelCache(config.cache_entries, config.cache_bytes)
    view = ShortHistoryView(
        panel=ShfeCuPanel(**wire["panel"]),
        episode=ShortHistoryEpisode(**wire["episode"]),
        **{k: v for k, v in wire.items() if k not in ("panel", "episode")},
    )
    tick = time.perf_counter()
    results = []
    for half_life in half_lives:
        issued = issue_short_history_risks(view, config, half_life=half_life, cache=_PROCESS_CACHE)
        methods = (
            (*PRIMARY_METHODS, *NO_DIAGONAL_METHODS)
            if half_life == config.half_life
            else SENSITIVITY_METHODS
        )
        for method in methods:
            forecast = issued[method]
            results.append(_jsonable({f.name: getattr(forecast, f.name) for f in fields(forecast)}))
    return results, time.perf_counter() - tick, _PROCESS_CACHE.diagnostics


def _bounded_process_results(jobs, workers):
    """Bound queued masked views; preserve episode/budget order across workers."""
    import multiprocessing

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        queue, iterator = deque(), iter(jobs)
        for _ in range(2 * workers):
            job = next(iterator, None)
            if job is not None:
                queue.append(executor.submit(_process_issue, job))
        while queue:
            yield queue.popleft().result()
            job = next(iterator, None)
            if job is not None:
                queue.append(executor.submit(_process_issue, job))


def run_short_history_benchmark(panel, config, output, *, workers=1):
    """Run local issued forecasts, then publishable aggregate products.

    Output must be new, and Git-ignored when inside the source repository.
    A caller may supply a synthetic panel without empirical-data dependencies.
    """
    if isinstance(config, (str, Path)):
        config = load_short_history_config(config)
    if not isinstance(config, ShortHistoryConfig):
        raise TypeError("config must be ShortHistoryConfig or a TOML path")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from one to eight")
    output = Path(output)
    _local_output(output)
    start = time.perf_counter()
    cache = BoundedModelCache(config.cache_entries, config.cache_bytes)
    records, timings = [], []
    half_lives = tuple(dict.fromkeys((config.half_life, *config.sensitivity_half_lives)))
    episodes = [
        episode
        for phase in ("validation", "evaluation")
        for episode in make_short_history_episodes(panel, config, phase)
    ]
    worker_caches = []
    with (output / "private_predictions.jsonl").open("w") as local:
        if workers > 1:
            jobs = (
                (
                    _masked_wire(build_short_history_view(panel, episode, budget, config)),
                    config,
                    half_lives,
                )
                for episode in episodes
                for budget in config.history_budgets
            )
            total = len(episodes) * len(config.history_budgets)
            for index, (forecasts, duration, cache_info) in enumerate(
                _bounded_process_results(jobs, workers)
            ):
                timings.append(duration)
                worker_caches.append(cache_info)
                for wire in forecasts:
                    issued = IssuedShortHistoryRisk(**wire)
                    future_mask = panel.dates == np.datetime64(issued.next_date)
                    row = score_short_history_risk(
                        issued,
                        future_date=issued.next_date,
                        future_ids=panel.contract_ids[future_mask],
                        future_prices=panel.settlement[future_mask],
                    )
                    local.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
                    records.append(row)
                if (index + 1) % 48 == 0 or index + 1 == total:
                    print(
                        f"short-history: {index + 1}/{total} masked views, "
                        f"{time.perf_counter() - start:.1f}s",
                        flush=True,
                    )
        else:
            _run_serial(panel, config, episodes, half_lives, cache, records, timings, local, start)
    summary = aggregate_short_history(records, config)
    paired = paired_short_history(records, config)
    diagnostics = _diagnostic_summary(records, timings, cache)
    if worker_caches:
        diagnostics["cache"] = {
            "scope": "per worker; bounded independently",
            "workers": workers,
            "max_entries_observed": max(x["entries"] for x in worker_caches),
            "max_bytes_observed": max(x["bytes"] for x in worker_caches),
            "max_entries": config.cache_entries,
            "max_bytes": config.cache_bytes,
        }
        diagnostics["timing_scope"] = (
            "per masked view, all methods and half-lives; worker wall time"
        )
    return _finish_run(
        panel, config, output, records, summary, paired, diagnostics, episodes, start, workers
    )


def _run_serial(panel, config, episodes, half_lives, cache, records, timings, local, start):
    for index, episode in enumerate(episodes):
        tick = time.perf_counter()
        future_mask = panel.dates == episode.next_date
        future_ids, future_prices = panel.contract_ids[future_mask], panel.settlement[future_mask]
        for budget in config.history_budgets:
            view = build_short_history_view(panel, episode, budget, config)
            for half_life in half_lives:
                issued = issue_short_history_risks(view, config, half_life=half_life, cache=cache)
                methods = (
                    (*PRIMARY_METHODS, *NO_DIAGONAL_METHODS)
                    if half_life == config.half_life
                    else SENSITIVITY_METHODS
                )
                for method in methods:
                    row = score_short_history_risk(
                        issued[method],
                        future_date=episode.next_date,
                        future_ids=future_ids,
                        future_prices=future_prices,
                    )
                    local.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
                    records.append(row)
        timings.append(time.perf_counter() - tick)
        if (index + 1) % 12 == 0 or index + 1 == len(episodes):
            print(
                f"short-history: {index + 1}/{len(episodes)} episodes, "
                f"{time.perf_counter() - start:.1f}s",
                flush=True,
            )


def _finish_run(
    panel, config, output, records, summary, paired, diagnostics, episodes, start, workers
):
    protocol = {
        "question": "next-session risk transfer to same-ID copper contracts with masked own history",
        "config": asdict(config),
        "chronology": "2022 initial; 2023 development validation; 2024–2025 inspected evaluation",
        "information": "mask target price rows before every transformation; current settlement allowed",
        "forecast_target": "uncentered same-contract price-change second moments",
        "variance_assumption": "zero conditional mean approximation",
        "diagonal_correction": "leave-one-donor-out predictive error, not identified latent noise",
        "residual_restriction": "four correlated maturity features plus diagonal remainder",
        "primary_comparison": "dfm_joint versus spde_joint: complete policies, not causal transport effect",
        "unavailable": "no fallback; direct requires at least 20 complete same-contract changes",
        "selection": "all settings fixed; sensitivities are not selected by outcomes",
        "limitations": [
            "inspected retrospective daily data",
            "maturity coordinate is contract-month 15th",
            "masked history is not a new-listing experiment",
            "repeated contracts and origins dependent",
            "not price forecasting, pricing or an execution backtest",
        ],
    }
    root = _source_root()
    sources = sorted((root / "src/spde_covariance").glob("*.py")) if root else []
    manifest = {
        "schema": "short-history-v1",
        "code_revision": _git(root, "rev-parse", "HEAD") if root else None,
        "source_dirty": bool(_git(root, "status", "--porcelain")) if root else None,
        "source_sha256": {p.name: _hash(p) for p in sources},
        "uv_lock_sha256": _hash(root / "uv.lock") if root else None,
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "data_provenance": public_data_provenance(panel.provenance),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "threads": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "runtime_seconds": time.perf_counter() - start,
        "workers": workers,
        "episode_count": len(episodes),
        "issued_count": len(records),
        "local_records_sha256": _hash(output / "private_predictions.jsonl"),
        "local_record_policy": "contract-level prices, forecasts and P&Ls are not exported",
    }
    products = dict(
        summary=summary,
        paired=paired,
        diagnostics=diagnostics,
        protocol=protocol,
        manifest=manifest,
    )
    for name, value in products.items():
        _write(output / f"{name}.json", value)
    _write(output / "checksums.json", {name: _hash(output / name) for name in EVIDENCE_FILES})
    return products


def load_short_history_evidence(path):
    """Verify the complete, fixed aggregate manifest before reading evidence."""
    path = Path(path)
    checksums = json.loads((path / "checksums.json").read_text())
    if set(checksums) != set(EVIDENCE_FILES):
        raise ValueError("short-history evidence whitelist mismatch")
    for name, expected in checksums.items():
        if _hash(path / name) != expected:
            raise ValueError(f"short-history checksum mismatch: {name}")
    return {Path(name).stem: json.loads((path / name).read_text()) for name in EVIDENCE_FILES}


def export_short_history_evidence(local_run, destination):
    """Export aggregates only; private JSONL is deliberately outside the whitelist."""
    local_run, destination = Path(local_run), Path(destination)
    load_short_history_evidence(local_run)
    destination.mkdir(parents=True, exist_ok=False)
    for name in (*EVIDENCE_FILES, "checksums.json"):
        (destination / name).write_bytes((local_run / name).read_bytes())
    return destination
