"""Chronological, frozen-block evaluation of innovation-covariance forecasts.

Only observations enter fitting and window selection. Latent covariance paths
are read afterwards for separately labelled simulation diagnostics. The primary
score omits the forecast-independent squared corrected-measurement constant.
"""

from __future__ import annotations

import csv
import json
import tomllib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .estimation import CovarianceEstimate, fit_covariance, fit_covariance_reference
from .geometry import diagnose_design, gram_operator, svec
from .linalg import validate_psd
from .observations import Observation
from .provenance import _jsonable, _manifest, _source_root
from .rq2_design import _integer, build_geometry, generate_data
from .rq3_config import ForecastSettings, RQ3Config

RULES = ("static", "fixed", "selected")
SOLVER_OPTIONS = {
    "absolute_tolerance": 1e-9,
    "relative_tolerance": 0.0,
    "max_iterations": 30_000,
}
TRUTH_METRICS = (
    "operator_hs_error",
    "operator_hs_relative_error",
    "observed_covariance_error",
    "kernel_grid_rmse",
    "exposure_risk_rmse",
    "nullspace_hs_error",
    "weak_direction_hs_error",
)


def _snapshot(array):
    result = np.asarray(array, dtype=float).copy()
    result.flags.writeable = False
    return result


def _history(observations, config, expected_count):
    data = tuple(observations)
    if len(data) != expected_count or not data:
        raise ValueError("history must contain exactly the observations available at its endpoint")
    if any(not isinstance(item, Observation) for item in data):
        raise TypeError("history must contain Observation records")
    if any(item.dimension != data[0].dimension for item in data):
        raise ValueError("history must use a common coefficient dimension")
    times = np.array([item.time for item in data])
    expected = np.arange(1, expected_count + 1) * config.delta
    if not np.allclose(times, expected, rtol=1e-12, atol=0):
        raise ValueError("history times must match the declared equally spaced interval endpoints")
    return data


@dataclass(frozen=True)
class IssuedForecast:
    """A copied, read-only covariance issued from a strictly past-only history."""

    covariance: np.ndarray
    basis_id: str
    origin: int
    origin_time: float
    window: int
    training_first_interval: int
    training_last_interval: int
    training_times: tuple[float, ...]
    estimate: CovarianceEstimate

    def __post_init__(self):
        if not isinstance(self.basis_id, str) or not self.basis_id.strip():
            raise ValueError("basis_id must be a nonempty string")
        matrix = _snapshot(validate_psd(self.covariance, name="forecast covariance"))
        weights = _snapshot(self.estimate.weights)
        object.__setattr__(self, "covariance", matrix)
        object.__setattr__(self, "training_times", tuple(self.training_times))
        object.__setattr__(
            self, "estimate", replace(self.estimate, covariance=matrix, weights=weights)
        )


def fit_forecast(
    history, settings: ForecastSettings, *, origin, window, basis_id
) -> IssuedForecast:
    """Issue a persistent covariance using observations 1..origin only.

    No scenario, latent truth, future observation or covariance path is an input.
    A failed solver iterate is retained with its failure diagnostics; the run
    acceptance gate rejects any such fit, rather than changing the forecast rule.
    """
    if not isinstance(settings, ForecastSettings):
        raise TypeError("fit_forecast requires the scenario-free ForecastSettings subset")
    origin = _integer(origin, "origin")
    window = _integer(window, "window")
    if window > origin:
        raise ValueError("window must not exceed the issue origin")
    data = _history(history, settings, origin)
    selected = data[-window:]
    gram = gram_operator([item.operator for item in selected])
    penalty = float(settings.relative_regularization * np.linalg.eigvalsh(gram)[-1])
    if penalty <= 0:
        raise ValueError("the training design must contain a nonzero observation map")
    estimate = fit_covariance(
        selected,
        regularization=penalty,
        trace_cap=settings.trace_cap,
        numerical_error_budget=settings.numerical_error_budget,
        **SOLVER_OPTIONS,
    )
    return IssuedForecast(
        estimate.covariance,
        basis_id,
        origin,
        data[-1].time,
        window,
        origin - window + 1,
        origin,
        tuple(item.time for item in selected),
        estimate,
    )


def observed_score(covariance, observation: Observation) -> float:
    """Reduced corrected Frobenius score; a lower value is better.

    Add ||yy.T-N||_F^2 to obtain the full squared loss. That constant is common
    to all candidates at the same future observation. Scores may be negative.
    """
    matrix = validate_psd(covariance, name="forecast covariance")
    if matrix.shape != (observation.dimension, observation.dimension):
        raise ValueError("forecast covariance must match the observation basis dimension")
    visible = observation.operator @ matrix @ observation.operator.T
    return float(np.sum(visible**2) - 2 * np.sum(visible * observation.measurement))


def score_exposure(covariance, exposure, observation: Observation, *, rank_rtol=1e-10):
    """Score plug-in risk only when the future exposure is observed jointly.

    If a=O.T c, the proxy (c.T y)^2-c.T N c has expectation a.T V a.
    The reduced score theta^2-2 theta*proxy omits its candidate-independent
    squared proxy. A historical pooled target-range test cannot replace this
    next-observation availability check.
    """
    matrix = validate_psd(covariance, name="forecast covariance")
    vector = np.asarray(exposure, dtype=float)
    if matrix.shape != (observation.dimension, observation.dimension):
        raise ValueError("forecast covariance must match the observation basis dimension")
    if vector.shape != (observation.dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError("exposure must be a finite coefficient vector")
    if np.linalg.norm(vector) == 0:
        raise ValueError("exposure must be nonzero")
    if not np.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie strictly between zero and one")
    coefficients = np.linalg.lstsq(observation.operator.T, vector, rcond=rank_rtol)[0]
    residual = float(
        np.linalg.norm(observation.operator.T @ coefficients - vector) / np.linalg.norm(vector)
    )
    forecast_risk = float(vector @ matrix @ vector)
    result = {
        "available": residual <= rank_rtol,
        "availability_residual": residual,
        "unavailable_reason": None,
        "forecast_risk": forecast_risk,
        "realized_risk_proxy": None,
        "risk_proxy_noise_correction": None,
        "risk_score": None,
    }
    if residual > rank_rtol:
        result["unavailable_reason"] = "exposure is not in the future observation row space"
        return result
    correction = float(coefficients @ observation.noise_covariance @ coefficients)
    proxy = float((coefficients @ observation.y) ** 2 - correction)
    result.update(
        realized_risk_proxy=proxy,
        risk_proxy_noise_correction=correction,
        risk_score=forecast_risk**2 - 2 * forecast_risk * proxy,
    )
    return result


def select_window(candidate_scores: Mapping[int, list[float]]) -> tuple[int, list[dict]]:
    """Select by observed validation mean; exact ties choose the longest window."""
    if not isinstance(candidate_scores, Mapping) or not candidate_scores:
        raise ValueError("candidate_scores must be a nonempty mapping")
    candidates, count = [], None
    for window, scores in candidate_scores.items():
        window = _integer(window, "candidate window")
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("every candidate must have finite nonempty validation scores")
        count = len(values) if count is None else count
        if len(values) != count:
            raise ValueError("all candidates must be scored on the same number of observations")
        candidates.append(
            {
                "window": window,
                "validation_score": float(values.mean()),
                "validation_scores": values.tolist(),
            }
        )
    candidates.sort(key=lambda item: item["window"])
    winner = min(candidates, key=lambda item: (item["validation_score"], -item["window"]))
    return winner["window"], candidates


def _validate_forecasts(history, config, issue):
    """The selector receives only observations through the last validation outcome."""
    endpoint = config.validation_origins[-1] + config.horizon
    data = _history(history, config, endpoint)
    scores = {window: [] for window in config.windows}
    rows = []
    for origin in config.validation_origins:
        for window in config.windows:
            forecast = issue(origin, window)
            for horizon in range(1, config.horizon + 1):
                score = observed_score(forecast.covariance, data[origin + horizon - 1])
                scores[window].append(score)
                rows.append(
                    {
                        "origin": origin,
                        "horizon": horizon,
                        "window": window,
                        "target_interval": origin + horizon,
                        "observed_score": score,
                    }
                )
    selected, candidates = select_window(scores)
    return {
        "selected_window": selected,
        "candidates": candidates,
        "validation_origins": config.validation_origins,
        "selection_complete_interval": endpoint,
        "tie_rule": "exact ties choose the longest window",
    }, rows


def select_forecast_window(history, config: RQ3Config, *, basis_id):
    """Fit and select on the declared validation prefix, with no test data input."""
    endpoint = config.validation_origins[-1] + config.horizon
    data = _history(history, config, endpoint)
    return _validate_forecasts(
        data,
        config,
        lambda origin, window: fit_forecast(
            data[:origin],
            config.fit_settings(),
            origin=origin,
            window=window,
            basis_id=basis_id,
        ),
    )[0]


def _groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def _mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    )


def _cluster_summary(rows, keys, metrics):
    """Average dates within each independent replication before calculating MCSE."""
    summaries = []
    for key, records in _groups(rows, keys).items():
        result = dict(zip(keys, key, strict=True))
        clusters = _groups(records, ("replication",))
        result["replications"] = len(clusters)
        for metric in metrics:
            values = [np.mean([row[metric] for row in group]) for group in clusters.values()]
            result[metric + "_mean"], result[metric + "_mcse"] = _mean_mcse(values)
        summaries.append(result)
    return summaries


def summarize_scores(scores, risks, truths):
    """Return replication-clustered means and paired differences from raw rows."""
    keys = ("scenario", "design", "rule")
    summary = _cluster_summary(scores, keys, ("observed_score",))
    origin_summary = _cluster_summary(scores, (*keys, "origin"), ("observed_score",))
    horizon_summary = _cluster_summary(scores, (*keys, "horizon"), ("observed_score",))
    truth_summary = _cluster_summary(truths, keys, TRUTH_METRICS)
    risk_summary = []
    for key, records in _groups(risks, (*keys, "target")).items():
        item = dict(zip((*keys, "target"), key, strict=True))
        available = [row for row in records if row["available"]]
        clusters = _groups(available, ("replication",))
        item.update(
            replications=len({row["replication"] for row in records}),
            available_replications=len(clusters),
            available_scores=len(available),
            risk_score_mean=None,
            risk_score_mcse=None,
        )
        if clusters:
            values = [np.mean([row["risk_score"] for row in group]) for group in clusters.values()]
            item["risk_score_mean"], item["risk_score_mcse"] = _mean_mcse(values)
        risk_summary.append(item)
    paired = []
    for metric, raw, additional in (
        ("observed_score", scores, ()),
        ("risk_score", risks, ("target",)),
    ):
        for key, records in _groups(raw, ("scenario", "design", *additional)).items():
            baseline = {
                (row["replication"], row["origin"], row["horizon"]): row
                for row in records
                if row["rule"] == "fixed"
            }
            for rule in ("static", "selected"):
                differences = defaultdict(list)
                for row in records:
                    if row["rule"] != rule or row[metric] is None:
                        continue
                    base = baseline[(row["replication"], row["origin"], row["horizon"])]
                    if base[metric] is not None:
                        differences[row["replication"]].append(row[metric] - base[metric])
                clusters = [np.mean(values) for values in differences.values()]
                item = dict(zip(("scenario", "design", *additional), key, strict=True))
                item.setdefault("target", None)
                item.update(
                    comparison_rule=rule,
                    baseline_rule="fixed",
                    metric=metric,
                    replications=len(clusters),
                    mean_difference=None,
                    mcse=None,
                )
                if clusters:
                    item["mean_difference"], item["mcse"] = _mean_mcse(clusters)
                paired.append(item)
    return {
        "summary": summary,
        "origin_summary": origin_summary,
        "horizon_summary": horizon_summary,
        "truth_summary": truth_summary,
        "risk_summary": risk_summary,
        "paired_differences": paired,
    }


def _geometry(data, forecast, targets, config, design):
    operators = [item.operator for item in data]
    geometry = diagnose_design(
        operators,
        targets=targets,
        regularization=forecast.estimate.regularization,
        rank_rtol=config.rank_rtol,
    )
    return {
        "geometry_id": f"{design}:{forecast.origin}:{forecast.window}",
        "design": design,
        "origin": forecast.origin,
        "window": forecast.window,
        "training_first_interval": forecast.training_first_interval,
        "training_last_interval": forecast.training_last_interval,
        "regularization": forecast.estimate.regularization,
        "rank": geometry["rank"],
        "eigenvalues": geometry["eigenvalues"],
        "rank_threshold": geometry["rank_threshold"],
        "rank_rtol": config.rank_rtol,
        "targets": geometry["targets"],
        "gram": geometry["gram"],
    }


def _forecast_record(forecast, common, geometry_id):
    estimate = forecast.estimate
    return {
        **common,
        "issue_origin": forecast.origin,
        "issue_time": forecast.origin_time,
        "window": forecast.window,
        "basis_id": forecast.basis_id,
        "training_first_interval": forecast.training_first_interval,
        "training_last_interval": forecast.training_last_interval,
        "training_times": forecast.training_times,
        "covariance": forecast.covariance,
        "regularization": estimate.regularization,
        "geometry_id": geometry_id,
        "solver_converged": estimate.converged,
        "solver_status": estimate.status,
        "iterations": estimate.iterations,
        "objective_gap_bound": estimate.objective_gap_bound,
        "covariance_error_bound": estimate.covariance_error_bound,
        "min_eigenvalue": estimate.min_eigenvalue,
        "trace": estimate.trace,
    }


def _truth_record(covariance, truth, observation, grid, exposures, geometry):
    error = covariance - truth
    eigenvalues, eigenvectors = np.linalg.eigh(geometry["gram"])
    coordinates = eigenvectors.T @ svec(error)
    null = eigenvalues <= geometry["rank_threshold"]
    weak = (~null) & (eigenvalues <= geometry["regularization"])
    result = {
        "operator_hs_error": float(np.linalg.norm(error)),
        "operator_hs_relative_error": float(np.linalg.norm(error) / np.linalg.norm(truth)),
        "observed_covariance_error": float(
            np.linalg.norm(observation.operator @ error @ observation.operator.T)
        ),
        "kernel_grid_rmse": float(np.sqrt(np.mean((grid @ error @ grid.T) ** 2))),
        "exposure_risk_rmse": float(
            np.sqrt(np.mean([(a @ error @ a) ** 2 for a in exposures.values()]))
        ),
        "nullspace_hs_error": float(np.linalg.norm(coordinates[null])),
        "weak_direction_hs_error": float(np.linalg.norm(coordinates[weak])),
    }
    for name, exposure in exposures.items():
        result[name + "_risk_error"] = float(exposure @ error @ exposure)
    return result


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_rq3_benchmark(config, output=None):
    """Run into a fresh output directory; preserve provenance on every failure."""
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as stream:
            config = RQ3Config(**tomllib.load(stream))
    if not isinstance(config, RQ3Config):
        raise TypeError("config must be RQ3Config or a TOML path")
    root = _source_root()
    output = (
        Path(output)
        if output is not None
        else ((root or Path.cwd()) / "experiments" / "runs" / config.name)
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _run(config, output, root)
    except (Exception, KeyboardInterrupt) as error:
        _write_json(
            output / "status.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _run(config, output, root):
    manifest = _manifest(config, root)
    manifest["solver"].update(SOLVER_OPTIONS)
    manifest.update(
        experiment="RQ3: frozen-block innovation-covariance forecast evaluation",
        target="weekly innovation covariance at each future date; not accumulated block covariance",
        fitting_information="past observations, known maps/noise and fixed ridge settings only",
        selection_information="declared observed validation blocks only; no latent truth or test data",
        scoring="reduced corrected Frobenius score; candidate-independent constant omitted",
        pairing="shared latent innovations and exact dense-noise subsets across designs",
    )
    _write_json(output / "config.json", asdict(config))
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "status.json", {"status": "running"})
    model, point_map, exposures = build_geometry()
    targets = {name: np.outer(value, value) for name, value in exposures.items()}
    grid_points = np.linspace(0, 3, 101)
    grid = model.evaluate(grid_points)
    scores, risks, truths, forecasts, selections, validation_rows = [], [], [], [], [], []
    geometry, fits, failures, references, observation_schedule = {}, [], [], [], {}
    selection_endpoint = config.validation_origins[-1] + config.horizon
    fit_settings = config.fit_settings()
    reference_scenarios = tuple(s for s in ("constant", "rotating") if s in config.scenarios)
    reference_scenarios = reference_scenarios or config.scenarios[:1]
    reference_designs = tuple(d for d in ("dense", "split") if d in config.designs)
    reference_designs = reference_designs or config.designs[:1]
    expected_references = len(reference_scenarios) * len(reference_designs)
    for replication in range(config.replications):
        generated = generate_data(config.simulation_config(), replication)
        for scenario in config.scenarios:
            for design in config.designs:
                observations = generated.designs[scenario][design]
                common = {"scenario": scenario, "design": design, "replication": replication}
                cache = {}

                def issue(origin, window):
                    key = (origin, window)
                    if key not in cache:
                        forecast = fit_forecast(
                            observations[:origin],
                            fit_settings,
                            origin=origin,
                            window=window,
                            basis_id=generated.basis_id,
                        )
                        cache[key] = forecast
                        geometry_id = f"{design}:{origin}:{window}"
                        if geometry_id not in geometry:
                            geometry[geometry_id] = _geometry(
                                observations[origin - window : origin],
                                forecast,
                                targets,
                                config,
                                design,
                            )
                        record = _forecast_record(
                            forecast, {**common, "origin": origin}, geometry_id
                        )
                        fits.append(record)
                        if (
                            not forecast.estimate.converged
                            or forecast.estimate.covariance_error_bound
                            > config.numerical_error_budget
                        ):
                            failures.append(
                                {
                                    **common,
                                    "origin": origin,
                                    "window": window,
                                    "reason": forecast.estimate.status,
                                    "covariance_error_bound": forecast.estimate.covariance_error_bound,
                                }
                            )
                    return cache[key]

                selection, validation = _validate_forecasts(
                    observations[:selection_endpoint],
                    config,
                    issue,
                )
                selections.append({**common, **selection})
                validation_rows.extend({**common, **row} for row in validation)
                if replication == 0 and scenario == config.scenarios[0]:
                    for origin in (*config.validation_origins, *config.test_origins):
                        for horizon in range(1, config.horizon + 1):
                            interval = origin + horizon
                            item = observations[interval - 1]
                            observation_schedule[(design, interval)] = {
                                "design": design,
                                "interval": interval,
                                "time": item.time,
                                "observed_dimension": item.n_contracts,
                                "operator": item.operator,
                                "noise_covariance": item.noise_covariance,
                            }
                static = issue(config.static_window, config.static_window)
                for origin in config.test_origins:
                    issued = {
                        "static": static,
                        "fixed": issue(origin, config.fixed_window),
                        "selected": issue(origin, selection["selected_window"]),
                    }
                    for rule, forecast in issued.items():
                        geometry_id = f"{design}:{forecast.origin}:{forecast.window}"
                        forecast_record = _forecast_record(
                            forecast,
                            {**common, "rule": rule, "origin": origin},
                            geometry_id,
                        )
                        forecasts.append(forecast_record)
                        for horizon in range(1, config.horizon + 1):
                            index = origin + horizon - 1
                            observation = observations[index]
                            row = {
                                **common,
                                "rule": rule,
                                "origin": origin,
                                "horizon": horizon,
                                "target_interval": index + 1,
                                "target_time": observation.time,
                                "observed_dimension": observation.n_contracts,
                                "window": forecast.window,
                                "issue_origin": forecast.origin,
                                "training_first_interval": forecast.training_first_interval,
                                "training_last_interval": forecast.training_last_interval,
                            }
                            scores.append(
                                {
                                    **row,
                                    "observed_score": observed_score(
                                        forecast.covariance, observation
                                    ),
                                    "observation_constant": float(
                                        np.sum(observation.measurement**2)
                                    ),
                                }
                            )
                            for name, exposure in exposures.items():
                                risks.append(
                                    {
                                        **row,
                                        "target": name,
                                        **score_exposure(
                                            forecast.covariance,
                                            exposure,
                                            observation,
                                            rank_rtol=config.rank_rtol,
                                        ),
                                    }
                                )
                            # Latent truth enters only this separate diagnostic path.
                            truths.append(
                                {
                                    **row,
                                    **_truth_record(
                                        forecast.covariance,
                                        generated.truths[scenario][index],
                                        observation,
                                        grid,
                                        exposures,
                                        geometry[geometry_id],
                                    ),
                                }
                            )
                    if (
                        config.reference_checks
                        and replication == 0
                        and scenario in reference_scenarios
                        and design in reference_designs
                        and origin == config.test_origins[0]
                    ):
                        own = issued["fixed"].estimate
                        reference = fit_covariance_reference(
                            observations[origin - config.fixed_window : origin],
                            regularization=own.regularization,
                            trace_cap=config.trace_cap,
                            solver_tolerance=1e-10,
                            max_iterations=100_000,
                        )
                        references.append(
                            {
                                **common,
                                "origin": origin,
                                "window": config.fixed_window,
                                "matrix_discrepancy": float(
                                    np.linalg.norm(reference.covariance - own.covariance)
                                ),
                                "objective_discrepancy": abs(reference.objective - own.objective),
                                "reference_converged": reference.converged,
                                "reference_covariance_error_bound": reference.covariance_error_bound,
                            }
                        )
        print(f"RQ3 replication {replication + 1}/{config.replications} complete", flush=True)
    certificates = [row["covariance_error_bound"] for row in fits]
    feasibility = all(
        row["min_eigenvalue"] >= -1e-10 and row["trace"] <= config.trace_cap + 1e-10 for row in fits
    )
    reference_passed = (
        not config.reference_checks or len(references) == expected_references
    ) and all(
        row["reference_converged"]
        and row["reference_covariance_error_bound"] <= config.numerical_error_budget
        and row["matrix_discrepancy"] <= 2 * config.numerical_error_budget
        for row in references
    )
    results = {
        **summarize_scores(scores, risks, truths),
        "selections": selections,
        "geometry": list(geometry.values()),
        "reference_checks": references,
        "acceptance": {
            "passed": not failures and feasibility and reference_passed,
            "failed_fits": len(failures),
            "failures": failures,
            "feasibility_passed": feasibility,
            "reference_passed": reference_passed,
            "reference_checks_requested": config.reference_checks,
            "reference_cases": len(references),
            "reference_cases_expected": expected_references if config.reference_checks else 0,
            "maximum_covariance_error_bound": max(certificates),
            "numerical_error_budget": config.numerical_error_budget,
            "statistical_performance_gate": None,
        },
        "model": {
            "basis_id": model.basis_id,
            "maturities": [1 / 12, 0.5, 1.5, 3.0],
            "point_map": point_map,
            "exposures": exposures,
            "evaluation_grid": grid_points,
            "exposure_normalization": "unit H norm; standardized risk, not currency P&L",
        },
        "protocol": {
            "rules": {
                "static": f"fit first {config.static_window} observations once",
                "fixed": f"refit the latest {config.fixed_window} observations at each origin",
                "selected": "validation-selected window, refit at each test origin",
            },
            "forecast": "covariance held fixed for each subsequent weekly innovation in the block",
            "selection": "observed validation mean only; exact ties choose longest window",
            "selection_complete_interval": selection_endpoint,
            "scoring": "||O B O.T||_F^2 - 2 <y y.T-N, O B O.T>; constant omitted",
            "risk_scoring": "theta^2 - 2 theta*((c.T y)^2-c.T N c), only if a=O.T c",
            "mcse": "average dates/origins within replication, then use independent replications",
            "weak_direction_definition": "training Gram eigenvalue above rank threshold and <= ridge",
            "truth_diagnostics": "latent truth is used only after fitting and observed selection",
            "comparisons": "within design; noiseless identity is a separate information control",
            "no_nll_or_direct_risk": True,
        },
    }
    _write_json(output / "results.json", results)
    _write_json(output / "selection_audit.json", selections)
    _write_json(output / "forecasts.json", forecasts)
    _write_json(output / "fit_diagnostics.json", fits)
    _write_json(output / "geometry.json", list(geometry.values()))
    _write_json(output / "observation_schedule.json", list(observation_schedule.values()))
    _write_csv(output / "scores.csv", scores)
    _write_csv(output / "validation_scores.csv", validation_rows)
    _write_csv(output / "risks.csv", risks)
    _write_csv(output / "truth_metrics.csv", truths)
    from .rq3_reporting import write_report

    write_report(output, config, results)
    _write_json(
        output / "status.json",
        {
            "status": "complete" if results["acceptance"]["passed"] else "failed",
        },
    )
    if not results["acceptance"]["passed"]:
        raise RuntimeError(f"RQ3 numerical checks failed; inspect {output / 'report.md'}")
    return output
