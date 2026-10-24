"""Observed-only, frozen-block forecasts of transported covariance kernels.

The deterministic source covariance is constant within each sampling interval.
Simulation truth is held outside the history/settings accepted by the fitter.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import time
import tomllib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .completion import weighted_psd_projection
from .completion_forecast_config import CompletionForecastConfig
from .fragment_benchmark import (
    _compact,
    _csv,
    _exposures,
    _json,
    _summary,
    grid_nodes,
    quadrature_weights,
)
from .fragment_reference import paper_cover
from .fragments import FragmentLayout
from .laguerre import LaguerreTransport, PowerLawSource
from .observation_statistics import partial_kernel_from_arrays
from .provenance import _manifest, _source_root
from .rq3 import select_window

METHODS = ("rank2", "rank2_psd", "coherent")
RULES = ("static", "fixed", "selected")
_WORKER_CONTEXT = None


def source_path_parameters(config, scenario):
    """Deterministic interval-source scales and first-two-mode rotation angles."""
    if scenario not in config.scenarios:
        raise ValueError("scenario is not configured")
    scale, angle = np.ones(config.intervals), np.zeros(config.intervals)
    if scenario == "scale":
        scale = np.linspace(0.75, 1.25, config.intervals)
    elif scenario == "rotating":
        angle = np.linspace(0, np.pi / 2, config.intervals)
    elif scenario == "abrupt":
        angle[config.intervals // 2 :] = np.pi / 2
    return scale, angle


def prepare_forecast_model(config):
    """Evaluate V_i=T_delta(C_i) by linearity, retaining one fixed source tail.

    Only first-two-mode orientation changes. The remaining infinite spectrum
    stays fixed; the M-prefix is never renormalized. Four PSD integrations
    suffice, avoiding storage of intervals-by-M-by-M covariance arrays.
    """
    model = LaguerreTransport(config.reference_dimension)
    source = PowerLawSource(config.source_exponent, config.source_trace)
    nodes = grid_nodes(config.grid, "spde")
    evaluation = model.evaluate(nodes)
    diagonal = source.finite_matrix(model.dimension)
    components = []
    for vector in (np.eye(model.dimension)[0], np.eye(model.dimension)[1]):
        components.append(np.outer(vector, vector))
    vector = np.eye(model.dimension)[0] + np.eye(model.dimension)[1]
    components.append(np.outer(vector, vector))
    matrices = [
        model.innovation_covariance(c, config.delta, config.quadrature_order)
        for c in (diagonal, *components)
    ]
    base, b00, b11, bsum = [evaluation @ v @ evaluation.T for v in matrices]
    cross = bsum - b00 - b11
    q0, q1 = source.eigenvalues(2)
    remainder = base - q0 * b00 - q1 * b11
    paths, roots = {}, {}
    for scenario in config.scenarios:
        scales, angles = source_path_parameters(config, scenario)
        c, s = np.cos(angles), np.sin(angles)
        kernels = scales[:, None, None] * (
            remainder[None]
            + (q0 * c**2 + q1 * s**2)[:, None, None] * b00
            + (q0 * s**2 + q1 * c**2)[:, None, None] * b11
            + ((q0 - q1) * c * s)[:, None, None] * cross
        )
        kernels = (kernels + kernels.transpose(0, 2, 1)) / 2
        eigen, vectors = np.linalg.eigh(kernels)
        if eigen.min() < -1e-10 * max(1.0, float(eigen.max())):
            raise RuntimeError("transported generating covariance is not numerically PSD")
        paths[scenario] = kernels
        roots[scenario] = vectors * np.sqrt(np.maximum(eigen, 0))[:, None, :]
    geometry = {}
    weights = quadrature_weights(nodes)
    for cover in config.covers:
        intervals = paper_cover(cover, domain=(0, 3))
        # Use the exact same membership convention as the public fragment API.
        first_start, first_end = intervals[0]
        first_row = np.flatnonzero(
            (nodes >= np.nextafter(first_start, -np.inf))
            & (nodes <= np.nextafter(first_end, np.inf))
        )
        catalogue = FragmentLayout(nodes, weights, intervals, (first_row,))
        geometry[cover] = {
            "cliques": catalogue.cliques,
            "intervals": intervals,
            "mask": catalogue.declared_mask,
        }
    diagnostic = model.covariance_diagnostics(diagonal, config.delta, config.quadrature_order)
    return {
        "nodes": nodes,
        "weights": weights,
        "paths": paths,
        "roots": roots,
        "geometry": geometry,
        "exposures": {
            k: v
            for k, v in _exposures(nodes, "spde").items()
            if k in ("point_spread", "offgrid_spread", "long_spread")
        },
        "metadata": {
            "source": "fixed infinite p-power spectrum; first-two-mode deterministic rotations",
            "source_exponent": source.exponent,
            "source_infinite_trace": source.trace,
            "source_prefix_trace": float(np.trace(diagonal)),
            "forcing_cutoff": model.dimension,
            "basis_id": model.basis_id,
            "rotation_pairs": "(e0,e1) only; all remaining source modes unchanged",
            "rotation_convention": "R=[[cos,-sin],[sin,cos]], C_i=s_i R_i diag(q) R_i.T",
            "target": "V_i=delta^-1 integral_0^delta S_r C_i S_r.T dr",
            "driver": "deterministic within-interval source; independent Gaussian increments",
            "source_tail_trace_max": 1.25 * source.tail_trace(model.dimension),
            "innovation_trace_tail_bound_max": 1.25
            * source.innovation_tail_bound(model.dimension, config.delta),
            "tail_scope": "analytic full-H trace bounds; finite grid reference is not exact infinite truth",
            "integration_absolute_hs_crosscheck": diagnostic["absolute_hs_discrepancy"],
            "integration_numerical_error_certificate": False,
        },
    }


@dataclass(frozen=True)
class FragmentHistory:
    """Past observed fragments only, with no latent values or source metadata."""

    values: tuple[np.ndarray, ...]
    row_to_nodes: tuple[np.ndarray, ...]
    size: int
    noise_variance: float

    def __post_init__(self):
        if len(self.values) != len(self.row_to_nodes) or not self.values:
            raise ValueError("history must contain matching nonempty observed values and rows")
        # Shared statistics validation also rejects invalid indices/noise/shape.
        partial_kernel_from_arrays(
            self.values, self.row_to_nodes, self.noise_variance, size=self.size
        )
        for field in ("values", "row_to_nodes"):
            copies = []
            for value in getattr(self, field):
                copy = np.array(value, copy=True)
                copy.flags.writeable = False
                copies.append(copy)
            object.__setattr__(self, field, tuple(copies))

    def prefix(self, endpoint):
        return FragmentHistory(
            self.values[:endpoint], self.row_to_nodes[:endpoint], self.size, self.noise_variance
        )


@dataclass(frozen=True)
class IssuedCompletionForecast:
    kernel: np.ndarray | None
    available: bool
    reason: str | None
    origin: int
    window: int
    training_first_interval: int
    training_last_interval: int
    training_sha256: str
    diagnostics: dict
    runtime_seconds: float


def fit_completion_forecast(
    history, *, origin, window, cliques, weights, settings, project_psd=False
):
    """Fit a persistent kernel from an exact past prefix; no future/truth input."""
    from .completion_strategy import CompletionStrategySettings, fit_completion_strategy

    if not isinstance(history, FragmentHistory) or not isinstance(
        settings, CompletionStrategySettings
    ):
        raise TypeError("fit requires FragmentHistory and scenario-free CompletionStrategySettings")
    if (
        isinstance(origin, bool)
        or isinstance(window, bool)
        or not isinstance(origin, int)
        or not isinstance(window, int)
    ):
        raise ValueError("origin and window must be integers")
    if len(history.values) != origin or not 0 < window <= origin:
        raise ValueError("history must end exactly at origin and contain the requested window")
    values, rows = history.values[-window:], history.row_to_nodes[-window:]
    fingerprint = hashlib.sha256()
    for value, row in zip(values, rows, strict=True):
        fingerprint.update(np.asarray(value, dtype=float).tobytes())
        fingerprint.update(np.asarray(row, dtype=np.int64).tobytes())
    statistics = partial_kernel_from_arrays(values, rows, history.noise_variance, size=history.size)
    started = time.perf_counter()
    result = fit_completion_strategy(
        statistics, cliques=cliques, weights=weights, settings=settings
    )
    kernel = result.kernel
    diagnostics = dict(result.diagnostics)
    if result.available and project_psd:
        projection = weighted_psd_projection(
            kernel, weights=weights, observed_mask=statistics.observed_mask
        )
        kernel = projection.kernel
        diagnostics["global_psd_projection"] = projection.diagnostics
    if kernel is not None:
        kernel = np.array(kernel, copy=True)
        kernel.flags.writeable = False
    return IssuedCompletionForecast(
        kernel,
        result.available,
        result.reason,
        origin,
        window,
        origin - window + 1,
        origin,
        fingerprint.hexdigest(),
        _compact(diagnostics),
        time.perf_counter() - started,
    )


def observed_kernel_score(kernel, value, row, noise_variance):
    """Reduced Frobenius score, allowing explicitly labelled indefinite baselines."""
    prediction = np.asarray(kernel)[np.ix_(row, row)]
    corrected = np.outer(value, value) - noise_variance * np.eye(len(row))
    return float(np.sum(prediction**2) - 2 * np.sum(prediction * corrected))


def exposure_proxy(value, row, exposure, noise_variance):
    """Exact same-fragment point-functional proxy; unavailable stays explicit."""
    support = np.flatnonzero(exposure)
    if not np.all(np.isin(support, row)):
        return None
    coefficients = exposure[np.asarray(row)]
    return float((coefficients @ value) ** 2 - noise_variance * np.sum(coefficients**2))


def select_forecast_window(candidate_scores):
    """No fallback or silent exclusion if any predeclared window is unavailable."""
    if not candidate_scores or any(
        not values or any(v is None or not np.isfinite(v) for v in values)
        for values in candidate_scores.values()
    ):
        return None, "a predeclared candidate has unavailable validation predictions"
    return select_window(candidate_scores)[0], None


def _settings(config, method):
    from .completion_strategy import CompletionStrategySettings

    return CompletionStrategySettings(
        partial_method="coherent" if method == "coherent" else "pairwise",
        relative_regularization=config.relative_regularization if method == "coherent" else 0,
        completion_rule="support" if method == "coherent" else "rank2",
        floor_policy="resolution" if method == "coherent" else "zero",
        floor_gamma=config.floor_gamma,
        numerical_error_budget=config.numerical_error_budget,
        rho=config.admm_rho,
        max_iterations=config.admm_max_iterations,
        support_rtol=config.support_rtol,
    )


def _initialize_worker(config):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (config, prepare_forecast_model(config))


def _worker_replication(replication):
    return _fit_replication(replication, *_WORKER_CONTEXT)


def _fit_replication(replication, config, prepared):
    fits, validations, selections, evaluation, risks = [], [], [], [], []
    nodes, weights = prepared["nodes"], prepared["weights"]
    rng = np.random.default_rng(config.seed + 1009 * replication)
    # Shared exogenous random numbers across scenarios, before any method/cover.
    normals = rng.standard_normal((config.intervals, len(nodes)))
    noise_bank = rng.standard_normal(normals.shape)
    parent_hashes = {}
    for scenario in config.scenarios:
        parent_values = np.einsum("tij,tj->ti", prepared["roots"][scenario], normals)
        parent_values += np.sqrt(config.noise_variance) * noise_bank
        parent_hashes[scenario] = hashlib.sha256(parent_values.tobytes()).hexdigest()
        for cover in config.covers:
            geo = prepared["geometry"][cover]
            cliques = geo["cliques"]
            catalogue = np.random.default_rng(config.seed + 7919 * replication + cover).permutation(
                cover
            )
            assignments = np.resize(catalogue, config.intervals)
            rows = tuple(cliques[group] for group in assignments)
            values = tuple(parent_values[i, row] for i, row in enumerate(rows))
            history = FragmentHistory(values, rows, len(nodes), config.noise_variance)
            cache = {}
            base = {"replication": replication, "scenario": scenario, "cover": cover}

            def issue(method, origin, window):
                key = (method, origin, window)
                if key not in cache:
                    forecast = fit_completion_forecast(
                        history.prefix(origin),
                        origin=origin,
                        window=window,
                        cliques=cliques,
                        weights=weights,
                        settings=_settings(config, method),
                        project_psd=method == "rank2_psd",
                    )
                    cache[key] = forecast
                    fits.append(
                        {
                            **base,
                            "method": method,
                            **{
                                field: getattr(forecast, field)
                                for field in (
                                    "available",
                                    "reason",
                                    "origin",
                                    "window",
                                    "training_first_interval",
                                    "training_last_interval",
                                    "training_sha256",
                                    "diagnostics",
                                    "runtime_seconds",
                                )
                            },
                        }
                    )
                return cache[key]

            selected = {}
            for method in METHODS:
                scores = {window: [] for window in config.windows}
                for origin in config.validation_origins:
                    for window in config.windows:
                        forecast = issue(method, origin, window)
                        for horizon in range(1, config.horizon + 1):
                            index = origin + horizon - 1
                            score = (
                                observed_kernel_score(
                                    forecast.kernel,
                                    values[index],
                                    rows[index],
                                    config.noise_variance,
                                )
                                if forecast.available
                                else None
                            )
                            scores[window].append(score)
                            validations.append(
                                {
                                    **base,
                                    "method": method,
                                    "origin": origin,
                                    "window": window,
                                    "horizon": horizon,
                                    "target_interval": index + 1,
                                    "observed_score": score,
                                    "available": forecast.available,
                                }
                            )
                selected[method], reason = select_forecast_window(scores)
                selections.append(
                    {
                        **base,
                        "method": method,
                        "selected_window": selected[method],
                        "reason": reason,
                        "validation_scores": scores,
                        "selection_endpoint": config.validation_origins[-1] + config.horizon,
                    }
                )
            for method in METHODS:
                static = issue(method, config.static_window, config.static_window)
                for origin in config.test_origins:
                    for rule in RULES:
                        window = (
                            config.static_window
                            if rule == "static"
                            else config.fixed_window
                            if rule == "fixed"
                            else selected[method]
                        )
                        forecast = (
                            static
                            if rule == "static"
                            else issue(method, origin, window)
                            if window is not None
                            else None
                        )
                        available = forecast is not None and forecast.available
                        for horizon in range(1, config.horizon + 1):
                            index = origin + horizon - 1
                            record = {
                                **base,
                                "method": method,
                                "rule": rule,
                                "origin": origin,
                                "horizon": horizon,
                                "target_interval": index + 1,
                                "window": window,
                                "available": available,
                                "reason": None
                                if available
                                else forecast.reason
                                if forecast is not None
                                else "window selection unavailable",
                                "observed_score": None,
                                "kernel_error": None,
                                "missing_kernel_error": None,
                                "minimum_weighted_eigenvalue": None,
                            }
                            kernel = forecast.kernel if available else None
                            truth = prepared["paths"][scenario][index]
                            if available:
                                difference = kernel - truth
                                measure = np.outer(weights, weights)
                                record.update(
                                    observed_score=observed_kernel_score(
                                        kernel, values[index], rows[index], config.noise_variance
                                    ),
                                    kernel_error=float(np.sqrt(np.sum(measure * difference**2))),
                                    missing_kernel_error=float(
                                        np.sqrt(
                                            np.sum(
                                                measure[~geo["mask"]]
                                                * difference[~geo["mask"]] ** 2
                                            )
                                        )
                                    ),
                                    minimum_weighted_eigenvalue=float(
                                        np.linalg.eigvalsh(
                                            np.sqrt(weights)[:, None]
                                            * kernel
                                            * np.sqrt(weights)[None, :]
                                        )[0]
                                    ),
                                )
                            evaluation.append(record)
                            for target, exposure in prepared["exposures"].items():
                                proxy = exposure_proxy(
                                    values[index], rows[index], exposure, config.noise_variance
                                )
                                theta = float(exposure @ kernel @ exposure) if available else None
                                truth_risk = float(exposure @ truth @ exposure)
                                risks.append(
                                    {
                                        **base,
                                        "method": method,
                                        "rule": rule,
                                        "origin": origin,
                                        "horizon": horizon,
                                        "target_interval": index + 1,
                                        "target": target,
                                        "available": available,
                                        "forecast_risk": theta,
                                        "truth_risk": truth_risk,
                                        "risk_squared_error": (theta - truth_risk) ** 2
                                        if available
                                        else None,
                                        "scoring_available": proxy is not None,
                                        "risk_score": theta**2 - 2 * theta * proxy
                                        if available and proxy is not None
                                        else None,
                                    }
                                )
    return {
        "replication": replication,
        "fits": fits,
        "validations": validations,
        "selections": selections,
        "evaluation": evaluation,
        "risks": risks,
        "parent_values_sha256": parent_hashes,
    }


def _replication_averages(rows, fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    output = []
    for key, values in groups.items():
        record = dict(zip(fields, key, strict=True))
        record["available"] = all(v["available"] for v in values)
        record["dates"] = len(values)
        for metric in metrics:
            observed = [v[metric] for v in values if v[metric] is not None]
            # An unavailable forecast cannot improve an average by omission.
            record[metric] = float(np.mean(observed)) if record["available"] and observed else None
            record[metric + "_available_dates"] = len(observed)
        output.append(record)
    return output


def _aggregate(records):
    evaluation = [x for r in records for x in r["evaluation"]]
    risks = [x for r in records for x in r["risks"]]
    base = ("replication", "scenario", "cover", "method", "rule")
    metrics = ("observed_score", "kernel_error", "missing_kernel_error")
    averages = _replication_averages(evaluation, base, metrics)
    summary = _summary(averages, metrics, base[1:])
    horizons = _replication_averages(evaluation, (*base, "horizon"), metrics)
    horizon_summary = _summary(horizons, metrics, (*base[1:], "horizon"))
    risk_metrics = ("risk_score", "risk_squared_error")
    risk_averages = _replication_averages(risks, (*base, "target"), risk_metrics)
    risk_summary = _summary(risk_averages, risk_metrics, (*base[1:], "target"))
    # Compare rules within method and methods within rule on identical observations.
    lookup = {tuple(row[f] for f in base): row for row in averages}
    paired = []
    for row in averages:
        comparisons = [
            ("static", row["method"], "static"),
            ("fixed", row["method"], "fixed"),
            ("rank2", "rank2", row["rule"]),
        ]
        for label, method, rule in comparisons:
            reference = lookup[(row["replication"], row["scenario"], row["cover"], method, rule)]
            paired.append(
                {
                    **{f: row[f] for f in base},
                    "comparison": label,
                    "available": row["available"] and reference["available"],
                    **{
                        metric: row[metric] - reference[metric]
                        if row[metric] is not None and reference[metric] is not None
                        else None
                        for metric in metrics
                    },
                }
            )
    paired_summary = _summary(paired, metrics, (*base[1:], "comparison"))
    risk_lookup = {tuple(row[f] for f in (*base, "target")): row for row in risk_averages}
    risk_paired = []
    for row in risk_averages:
        reference = risk_lookup[
            (
                row["replication"],
                row["scenario"],
                row["cover"],
                row["method"],
                "static",
                row["target"],
            )
        ]
        risk_paired.append(
            {
                **{f: row[f] for f in (*base, "target")},
                "available": row["available"] and reference["available"],
                **{
                    metric: row[metric] - reference[metric]
                    if row[metric] is not None and reference[metric] is not None
                    else None
                    for metric in risk_metrics
                },
            }
        )
    risk_paired_summary = _summary(risk_paired, risk_metrics, (*base[1:], "target"))
    return (
        {
            "summary": summary,
            "horizon_summary": horizon_summary,
            "risk_summary": risk_summary,
            "paired_differences": paired_summary,
            "risk_paired_differences": risk_paired_summary,
        },
        averages,
        risk_averages,
    )


def run_completion_forecast(config, output=None):
    """Run all declared cases; failed/unavailable fits remain in the evidence."""
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as handle:
            config = CompletionForecastConfig(**tomllib.load(handle))
    if not isinstance(config, CompletionForecastConfig):
        raise TypeError("completion forecast requires CompletionForecastConfig")
    output = Path(output or Path("experiments/runs") / config.name)
    if output.exists():
        raise FileExistsError("output must be a new directory")
    output.mkdir(parents=True)
    try:
        _run(config, output)
        return output
    except (Exception, KeyboardInterrupt) as error:
        _json(output / "status.json", {"status": "failed", "error": str(error)})
        raise


def _run(config, output):
    from .completion_forecast_reporting import write_report

    manifest = _manifest(config, _source_root())
    manifest.update(
        experiment="fixed-grid chronological forecasts of transported interval source covariances",
        solver={
            "name": "consensus_ADMM",
            "numerical_error_budget": config.numerical_error_budget,
            "max_iterations": config.admm_max_iterations,
            "rho": config.admm_rho,
            "relative_regularization": config.relative_regularization,
            "floor_gamma": config.floor_gamma,
            "floor_policy": "gamma / actual_grid_size**2",
        },
        execution={
            "workers": config.workers,
            "process_start_method": "spawn" if config.workers > 1 else None,
            "aggregation_order": "replication index",
            "logical_cpu_count": os.cpu_count(),
        },
        selection_information="each method selects its own window on identical past validation outcomes; exact ties longest; unavailable candidates never excluded",
    )
    _json(output / "manifest.json", manifest)
    _json(output / "config.json", asdict(config))
    _json(output / "status.json", {"status": "running"})
    prepared = prepare_forecast_model(config)
    _json(
        output / "model.json",
        {
            "metadata": prepared["metadata"],
            "nodes": prepared["nodes"],
            "weights": prepared["weights"],
            "exposures": prepared["exposures"],
            "geometry": prepared["geometry"],
            "source_paths": {
                s: dict(zip(("scales", "angles"), source_path_parameters(config, s), strict=True))
                for s in config.scenarios
            },
            "truth_risk_paths": {
                s: {
                    k: np.einsum("i,tij,j->t", a, prepared["paths"][s], a)
                    for k, a in prepared["exposures"].items()
                }
                for s in config.scenarios
            },
        },
    )
    manifest_sha = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    retained = {}

    def retain(record):
        replication = record["replication"]
        record["parent_manifest_sha256"] = manifest_sha
        _json(output / f"replication-{replication:04d}.json", record)
        retained[replication] = record
        _json(
            output / "status.json",
            {
                "status": "running",
                "completed_replications": len(retained),
                "completed_replication_indices": sorted(retained),
            },
        )

    if config.workers == 1:
        for replication in range(config.replications):
            retain(_fit_replication(replication, config, prepared))
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            futures = [
                executor.submit(_worker_replication, rep) for rep in range(config.replications)
            ]
            try:
                for future in as_completed(futures):
                    retain(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    records = [retained[i] for i in range(config.replications)]
    results, averages, risk_averages = _aggregate(records)
    fits = [x for r in records for x in r["fits"]]
    expected_evaluation = (
        config.replications
        * len(config.scenarios)
        * len(config.covers)
        * len(METHODS)
        * len(RULES)
        * len(config.test_origins)
        * config.horizon
    )
    evaluation = [x for r in records for x in r["evaluation"]]
    acceptance = {
        "declared_coverage_complete": len(evaluation) == expected_evaluation,
        "coherent_issued_fits_available": all(
            x["available"] for x in fits if x["method"] == "coherent"
        ),
        "coherent_test_forecasts_available": all(
            x["available"] for x in evaluation if x["method"] == "coherent"
        ),
        "coherent_outputs_psd": all(
            x["minimum_weighted_eigenvalue"] is not None
            and x["minimum_weighted_eigenvalue"] >= -1e-10
            for x in evaluation
            if x["method"] == "coherent"
        ),
        "selection_uses_past_only": config.validation_origins[-1] + config.horizon
        <= min(config.test_origins),
        "coherent_certificates_within_budget": bool(fits)
        and all(
            x["diagnostics"]["converged"]
            and x["diagnostics"]["partial_error_bound"] <= config.numerical_error_budget
            for x in fits
            if x["method"] == "coherent"
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    results.update(
        acceptance=acceptance,
        availability={
            "all_issued_fits_available": all(x["available"] for x in fits),
            "all_test_forecasts_available": all(x["available"] for x in evaluation),
            "unavailable_fits": sum(not x["available"] for x in fits),
            "unavailable_test_rows": sum(not x["available"] for x in evaluation),
        },
        model=prepared["metadata"],
        fit_count=len(fits),
        evaluation_count=len(evaluation),
        selections=[x for r in records for x in r["selections"]],
    )
    _json(output / "results.json", results)
    _json(output / "fits.json", fits)
    _json(output / "selections.json", results["selections"])
    _csv(output / "validation.csv", [x for r in records for x in r["validations"]])
    _csv(output / "forecasts.csv", evaluation)
    _csv(output / "risks.csv", [x for r in records for x in r["risks"]])
    _csv(output / "replication_averages.csv", averages)
    _csv(output / "risk_replication_averages.csv", risk_averages)
    _csv(output / "summary.csv", results["summary"])
    _csv(output / "paired.csv", results["paired_differences"])
    write_report(output, config, results)
    _json(
        output / "status.json",
        {"status": "complete" if acceptance["passed"] else "failed", "acceptance": acceptance},
    )
    return results
