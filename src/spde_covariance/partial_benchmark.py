"""Observation-only sieve/penalty selection for one parent innovation experiment.

All candidate dimensions consume exactly the same observed values and known
noise. Population truth is used only for separate approximation/evaluation
diagnostics. Validation selects a training fit, with no validation refit.
"""

from __future__ import annotations

import csv
import json
import tomllib
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .estimation import CovarianceEstimate, fit_covariance, fit_covariance_reference
from .geometry import diagnose_design, gram_operator
from .laguerre import LaguerreTransport
from .linalg import validate_psd
from .observations import Observation
from .partial_config import PartialBenchmarkConfig
from .partial_design import (
    design_indices,
    fixed_grid_witness,
    observation_omission,
    prepare_parent_model,
    simulate_parent_pool,
    span_diagnostics,
)
from .provenance import _jsonable, _manifest, _source_root

SOLVER_OPTIONS = {"absolute_tolerance": 1e-9, "relative_tolerance": 0.0, "max_iterations": 30_000}
EVALUATION_METRICS = (
    "test_observed_score",
    "operator_hs_error",
    "operator_hs_relative_error",
    "operator_norm_error",
    "observed_covariance_rmse",
    "kernel_grid_rmse",
    "exposure_risk_rmse",
    "covariance_error_bound",
)


def _readonly(values):
    result = np.asarray(values, dtype=float).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FitSettings:
    """Only numerical constraints; no scenario, truth or validation information."""

    trace_cap: float
    numerical_error_budget: float


@dataclass(frozen=True)
class CandidateFit:
    dimension: int
    relative_regularization: float
    gram_operator_norm: float
    basis_id: str
    estimate: CovarianceEstimate

    def __post_init__(self):
        matrix, weights = _readonly(self.estimate.covariance), _readonly(self.estimate.weights)
        object.__setattr__(
            self, "estimate", replace(self.estimate, covariance=matrix, weights=weights)
        )

    @property
    def covariance(self):
        return self.estimate.covariance


def fit_candidate(
    observations,
    settings: FitSettings,
    *,
    dimension,
    relative_regularization,
    gram_operator_norm,
    basis_id,
):
    """Fit a candidate from a supplied training block only, never latent truth."""
    if not isinstance(settings, FitSettings):
        raise TypeError("fit_candidate requires the truth-free FitSettings subset")
    data = tuple(observations)
    if not data or any(
        not isinstance(item, Observation) or item.dimension != dimension for item in data
    ):
        raise ValueError("training observations must have the declared candidate dimension")
    if any(right.time <= left.time for left, right in zip(data, data[1:])):
        raise ValueError("training times must be strictly increasing")
    if not isinstance(basis_id, str) or not basis_id:
        raise ValueError("basis_id must be explicit")
    if not np.isfinite(gram_operator_norm) or gram_operator_norm <= 0:
        raise ValueError("gram_operator_norm must be finite and positive")
    if not np.isfinite(relative_regularization) or relative_regularization <= 0:
        raise ValueError("relative_regularization must be finite and positive")
    fit = fit_covariance(
        data,
        regularization=relative_regularization * gram_operator_norm,
        trace_cap=settings.trace_cap,
        numerical_error_budget=settings.numerical_error_budget,
        **SOLVER_OPTIONS,
    )
    return CandidateFit(
        dimension, float(relative_regularization), float(gram_operator_norm), basis_id, fit
    )


def observed_quadratic_score(covariance, observations):
    """Mean corrected Frobenius score, omitting the shared ||yy.T-N||² constant."""
    matrix = validate_psd(covariance, name="candidate covariance")
    data = tuple(observations)
    if not data or any(item.dimension != len(matrix) for item in data):
        raise ValueError("score observations must be nonempty and match the candidate basis")
    # Grouping identical maps avoids repeating a candidate's predicted covariance.
    groups = {}
    for item in data:
        key = (item.operator.shape, item.operator.tobytes())
        if key not in groups:
            groups[key] = [item.operator, 0, np.zeros((item.n_contracts, item.n_contracts))]
        groups[key][1] += 1
        groups[key][2] += item.measurement
    total = 0.0
    for operator, count, corrected_sum in groups.values():
        predicted = operator @ matrix @ operator.T
        total += count * np.sum(predicted**2) - 2 * np.sum(predicted * corrected_sum)
    value = float(total / len(data))
    if not np.isfinite(value):
        raise ValueError("observed score exceeds floating-point range")
    return value


def choose_candidate(rows):
    """Observed validation score only; exact ties choose smaller J then larger eta."""
    data = tuple(rows)
    if not data or any(not np.isfinite(row["validation_score"]) for row in data):
        raise ValueError("all declared candidates need a finite validation score")
    keys = [(row["J"], row["relative_regularization"]) for row in data]
    if len(set(keys)) != len(keys):
        raise ValueError("candidate grid must not contain duplicate choices")
    return min(
        data, key=lambda row: (row["validation_score"], row["J"], -row["relative_regularization"])
    )


def _proxy_values(observations, representations):
    data, spans = tuple(observations), tuple(representations)
    if len(data) != len(spans):
        raise ValueError("one full-H representation is required per observation date")
    values = []
    for item, span in zip(data, spans, strict=True):
        if span["available"]:
            coefficient = np.asarray(span["coefficients"])
            if coefficient.shape != item.y.shape:
                raise ValueError("full-H proxy coefficients must match the actual observation rows")
            values.append(
                float(
                    (coefficient @ item.y) ** 2 - coefficient @ item.noise_covariance @ coefficient
                )
            )
    return np.asarray(values)


def direct_risk_estimate(observations, representations):
    """Average available full-H squared-exposure proxies; negative values remain valid."""
    proxies = _proxy_values(observations, representations)
    return {
        "available": bool(len(proxies)),
        "available_train_dates": len(proxies),
        "estimate": float(proxies.mean()) if len(proxies) else None,
        "unavailable_reason": None if len(proxies) else "no same-date full-H training proxy",
    }


def score_risk(estimate, observations, representations):
    """Reduced scalar score on actually available future full-H exposure proxies."""
    proxies = _proxy_values(observations, representations)
    return {
        "available_test_dates": len(proxies),
        "risk_score": (
            float(estimate**2 - 2 * estimate * proxies.mean())
            if estimate is not None and len(proxies)
            else None
        ),
    }


def _groups(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    )


def _summaries(candidates, selected, risks):
    summaries = {}
    for name, rows, fields, metrics in (
        (
            "candidate_summary",
            candidates,
            ("scenario", "design", "N", "J", "relative_regularization"),
            ("validation_score", "covariance_error_bound", "iterations"),
        ),
        ("selected_summary", selected, ("scenario", "design", "N", "rule"), EVALUATION_METRICS),
    ):
        output = []
        for key, records in _groups(rows, fields).items():
            item = dict(zip(fields, key, strict=True))
            item["replications"] = len(records)
            for metric in metrics:
                item[metric + "_mean"], item[metric + "_mcse"] = _mean_mcse(
                    [row[metric] for row in records]
                )
            output.append(item)
        summaries[name] = output
    risk_summary = []
    fields = ("scenario", "design", "N", "rule", "target")
    for key, records in _groups(risks, fields).items():
        item = dict(zip(fields, key, strict=True))
        item.update(
            replications=len(records),
            available_test_dates=records[0]["available_test_dates"],
            available_train_dates=records[0]["available_train_dates"],
            available_replications=sum(row["estimate"] is not None for row in records),
            risk_reference_tail_bound=records[0]["risk_reference_tail_bound"],
        )
        for metric in ("estimate", "risk_error", "risk_mse", "risk_score"):
            values = [row[metric] for row in records if row[metric] is not None]
            item[metric + "_mean"], item[metric + "_mcse"] = (
                _mean_mcse(values) if values else (None, None)
            )
        risk_summary.append(item)
    paired = []
    for metric, raw, extra in (
        ("observed_score", selected, ()),
        ("risk_score", risks, ("target",)),
    ):
        raw_metric = "test_observed_score" if metric == "observed_score" else metric
        for key, records in _groups(raw, ("scenario", "design", "N", *extra)).items():
            fixed = {row["replication"]: row for row in records if row["rule"] == "fixed"}
            differences = [
                row[raw_metric] - fixed[row["replication"]][raw_metric]
                for row in records
                if row["rule"] == "selected"
                and row[raw_metric] is not None
                and fixed[row["replication"]][raw_metric] is not None
            ]
            item = dict(zip(("scenario", "design", "N", *extra), key, strict=True))
            item.setdefault("target", None)
            item.update(
                metric=metric,
                comparison_rule="selected",
                baseline_rule="fixed",
                replications=len(differences),
                available_replications=len(differences),
                mean_difference=None,
                mcse=None,
            )
            if differences:
                item["mean_difference"], item["mcse"] = _mean_mcse(differences)
            paired.append(item)
    return {**summaries, "risk_summary": risk_summary, "paired_differences": paired}


def _prepare(config):
    parents = {
        scenario: prepare_parent_model(
            scenario,
            config.reference_dimension,
            config.delta,
            config.quadrature_order,
            source_exponent=config.source_exponent,
            source_trace=config.source_trace,
            rotation_angle=config.rotation_angle,
        )
        for scenario in config.scenarios
    }
    base = parents[config.scenarios[0]]
    spans, geometry, geometry_cache, full_h_geometry = {}, [], {}, []
    for design in config.designs:
        span_cache = {}
        spans[design] = {name: [] for name in base.exposures}
        for index in range(config.total_count):
            group = design_indices(design, index)
            if group not in span_cache:
                span_cache[group] = {
                    name: span_diagnostics(
                        exposure,
                        [base.contracts[row] for row in group],
                        rtol=config.rank_rtol,
                    )
                    for name, exposure in base.exposures.items()
                }
                full_h_geometry.extend(
                    {
                        "design": design,
                        "kind": "same_date",
                        "contract_indices": group,
                        "target": name,
                        **diagnostic,
                    }
                    for name, diagnostic in span_cache[group].items()
                )
            for name in base.exposures:
                spans[design][name].append(span_cache[group][name])
        for n in config.training_counts:
            union = sorted({row for index in range(n) for row in design_indices(design, index)})
            full_h_geometry.extend(
                {
                    "design": design,
                    "N": n,
                    "kind": "training_union_functional_span",
                    "contract_indices": union,
                    "target": name,
                    **span_diagnostics(
                        exposure, [base.contracts[row] for row in union], rtol=config.rank_rtol
                    ),
                    "scope": "union of linear functionals only; not pooled covariance identification",
                }
                for name, exposure in base.exposures.items()
            )
            for j in config.geometry_dimensions:
                operators = [
                    base.operator[list(design_indices(design, index)), :j] for index in range(n)
                ]
                gram = gram_operator(operators)
                norm = float(np.linalg.eigvalsh(gram)[-1])
                targets = {
                    name: np.outer(a[:j], a[:j]) for name, a in base.exposure_coefficients.items()
                }
                diagnostic = diagnose_design(
                    operators,
                    targets=targets,
                    regularization=min(config.relative_regularizations) * norm,
                    rank_rtol=config.rank_rtol,
                )
                geometry_cache[(design, n, j)] = {"gram_operator_norm": norm, "gram": gram}
                record = {key: value for key, value in diagnostic.items() if key != "gram"}
                geometry.append(
                    _jsonable(
                        {**record, "design": design, "N": n, "J": j, "gram_operator_norm": norm}
                    )
                )
    approximation = []
    for scenario, parent in parents.items():
        for j in config.geometry_dimensions:
            # Compute the parent sixteen-contract decomposition once; all
            # designs are exact row/column subsets, preserving cross terms.
            omission = observation_omission(parent.innovation_covariance, parent.operator, j)
            for design in config.designs:
                for n in config.training_counts:
                    sums = dict.fromkeys(("omission", "full", "tail_tail", "cross_sum"), 0.0)
                    normal_bias = np.zeros((j, j))
                    minimum, observed_tail_squared = np.inf, 0.0
                    for index in range(n):
                        rows = list(design_indices(design, index))
                        indices = np.ix_(rows, rows)
                        difference = omission["omission"][indices]
                        sums["omission"] += np.sum(difference**2)
                        sums["full"] += np.sum(omission["full"][indices] ** 2)
                        sums["tail_tail"] += np.sum(omission["tail_tail"][indices] ** 2)
                        sums["cross_sum"] += np.sum(
                            (omission["cross"][indices] + omission["cross_transpose"][indices]) ** 2
                        )
                        operator = parent.operator[rows, :j]
                        normal_bias += operator.T @ difference @ operator / n
                        minimum = min(minimum, float(np.linalg.eigvalsh(difference)[0]))
                        observed_tail_squared += np.sum(
                            parent.tail_bounds["observed_matrix"][indices] ** 2
                        )
                    hs = float(np.sqrt(sums["omission"] / n))
                    approximation.append(
                        {
                            "scenario": scenario,
                            "design": design,
                            "N": n,
                            "J": j,
                            "hs_error": hs,
                            "relative_hs_error": hs
                            / max(np.sqrt(sums["full"] / n), np.finfo(float).tiny),
                            "tail_tail_hs_error": float(np.sqrt(sums["tail_tail"] / n)),
                            "cross_sum_hs_error": float(np.sqrt(sums["cross_sum"] / n)),
                            "normal_equation_bias_hs": float(np.linalg.norm(normal_bias)),
                            "minimum_omission_eigenvalue": minimum,
                            "observed_reference_tail_bound": float(
                                np.sqrt(observed_tail_squared / n)
                            ),
                            "full_reference_tail_bound": parent.tail_bounds[
                                "full_innovation_trace_bound"
                            ],
                            "interpretation": "full-minus-sieve covariance including cross terms; not independent noise",
                        }
                    )
    try:
        witness = fixed_grid_witness(base.contracts, base.model, base.exposures["offgrid_spread"])
        witness = {
            "available": True,
            **{key: value for key, value in witness.items() if key != "rank_one_covariance"},
        }
    except ValueError as error:
        witness = {
            "available": False,
            "reason": str(error),
            "scope": "finite-prefix generic witness unavailable; full-H spans remain separate",
        }
    return {
        "parents": parents,
        "spans": spans,
        "geometry": geometry,
        "geometry_cache": geometry_cache,
        "observation_approximation": approximation,
        "fixed_grid_witness": witness,
        "full_h_geometry": full_h_geometry,
    }


def _evaluate(covariance, parent, test, grid, cache):
    j = len(covariance)
    error = -parent.innovation_covariance.copy()
    error[:j, :j] += covariance
    grouped = defaultdict(int)
    for index, item in test:
        grouped[tuple(index)] += 1
    observed_squared, count = 0.0, 0
    for index, frequency in grouped.items():
        rows = list(index)
        operator = parent.operator[rows, :j]
        difference = operator @ covariance @ operator.T - cache["observed"][np.ix_(rows, rows)]
        observed_squared += frequency * np.sum(difference**2)
        count += frequency
    return {
        "operator_hs_error": float(np.linalg.norm(error)),
        "operator_hs_relative_error": float(
            np.linalg.norm(error) / np.linalg.norm(parent.innovation_covariance)
        ),
        "operator_norm_error": float(np.max(np.abs(np.linalg.eigvalsh(error)))),
        "observed_covariance_rmse": float(np.sqrt(observed_squared / count)),
        "kernel_grid_rmse": float(
            np.sqrt(np.mean((grid[:, :j] @ covariance @ grid[:, :j].T - cache["kernel"]) ** 2))
        ),
        "exposure_risk_rmse": float(
            np.sqrt(np.mean([(a @ error @ a) ** 2 for a in parent.exposure_coefficients.values()]))
        ),
    }


def _candidate_record(candidate, common):
    fit = candidate.estimate
    return {
        **common,
        "J": candidate.dimension,
        "relative_regularization": candidate.relative_regularization,
        "regularization": fit.regularization,
        "gram_operator_norm": candidate.gram_operator_norm,
        "basis_id": candidate.basis_id,
        "covariance": candidate.covariance,
        "covariance_error_bound": fit.covariance_error_bound,
        "objective_gap_bound": fit.objective_gap_bound,
        "objective": fit.objective,
        "converged": fit.converged,
        "status": fit.status,
        "iterations": fit.iterations,
        "minimum_eigenvalue": fit.min_eigenvalue,
        "trace": fit.trace,
        "training_first_interval": 1,
        "training_last_interval": common["N"],
    }


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        fields = [key for key in rows[0] if key != "covariance"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_partial_benchmark(config, output=None):
    """Run into a fresh directory, retaining numerical failures and every candidate."""
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as stream:
            config = PartialBenchmarkConfig(**tomllib.load(stream))
    if not isinstance(config, PartialBenchmarkConfig):
        raise TypeError("config must be PartialBenchmarkConfig or a TOML path")
    root = _source_root()
    output = (
        Path(output)
        if output is not None
        else (root or Path.cwd()) / "experiments" / "runs" / config.name
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
        experiment="partial linear observations of fixed-interval infinite-model Gaussian innovations",
        fitting_information="training observations only; fixed known noise; finite-J PSD ridge",
        selection_information="same later observed validation block for every candidate; no latent truth",
        evaluation_information="untouched later test block; no validation refit",
        space={
            "metric": "f(0)g(0)+integral_0^infinity exp(x) f'g' dx",
            "maturity_unit": "year",
            "basis": "nested integrated Laguerre",
        },
    )
    _write_json(output / "config.json", config.__dict__)
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "status.json", {"status": "running"})
    prepared = _prepare(config)
    _write_json(output / "geometry.json", prepared["geometry"])
    _write_json(output / "observation_approximation.json", prepared["observation_approximation"])
    settings = FitSettings(config.trace_cap, config.numerical_error_budget)
    validation_start = max(config.training_counts)
    test_start = validation_start + config.validation_count
    grid_points = np.linspace(0.0, 3.0, 61)
    grid = LaguerreTransport(config.reference_dimension).evaluate(grid_points)
    evaluation_cache = {
        name: {
            "observed": parent.operator @ parent.innovation_covariance @ parent.operator.T,
            "kernel": grid @ parent.innovation_covariance @ grid.T,
        }
        for name, parent in prepared["parents"].items()
    }
    candidates, selections, selected_rows, risk_rows, references = [], [], [], [], []
    for replication in range(config.replications):
        for scenario, parent in prepared["parents"].items():
            pool = simulate_parent_pool(
                parent,
                config.total_count,
                seed=config.seed + 1009 * replication,
                noise_std=np.sqrt(config.noise_variance),
            )
            for design in config.designs:
                # Views share the already generated y and N exactly. Only O_J changes.
                views = {j: pool.observations(design, j) for j in config.candidate_dimensions}
                span_by_target = prepared["spans"][design]
                for n in config.training_counts:
                    common = {
                        "scenario": scenario,
                        "design": design,
                        "N": n,
                        "replication": replication,
                    }
                    fitted, records = {}, []
                    for j in config.candidate_dimensions:
                        for relative in config.relative_regularizations:
                            candidate = fit_candidate(
                                views[j][:n],
                                settings,
                                dimension=j,
                                relative_regularization=relative,
                                gram_operator_norm=prepared["geometry_cache"][(design, n, j)][
                                    "gram_operator_norm"
                                ],
                                basis_id=LaguerreTransport(j).basis_id,
                            )
                            record = _candidate_record(candidate, common)
                            record["validation_score"] = observed_quadratic_score(
                                candidate.covariance,
                                views[j][validation_start:test_start],
                            )
                            records.append(record)
                            fitted[(j, relative)] = candidate
                    winner = choose_candidate(records)
                    key = (winner["J"], winner["relative_regularization"])
                    for record in records:
                        record["selected"] = (record["J"], record["relative_regularization"]) == key
                    candidates.extend(records)
                    selections.append(
                        {
                            **common,
                            "selected_J": key[0],
                            "selected_relative_regularization": key[1],
                            "validation_score": winner["validation_score"],
                            "selection_valid": all(
                                record["converged"]
                                and record["covariance_error_bound"]
                                <= config.numerical_error_budget
                                for record in records
                            ),
                            "validation_first_interval": validation_start + 1,
                            "validation_last_interval": test_start,
                            "test_first_interval": test_start + 1,
                            "test_last_interval": config.total_count,
                            "refit_on_validation": False,
                        }
                    )
                    rules = {
                        "selected": fitted[key],
                        "fixed": fitted[
                            (config.baseline_dimension, config.baseline_relative_regularization)
                        ],
                    }
                    for rule, candidate in rules.items():
                        j = candidate.dimension
                        test = views[j][test_start:]
                        selected_rows.append(
                            {
                                **common,
                                "rule": rule,
                                "J": j,
                                "relative_regularization": candidate.relative_regularization,
                                "test_observed_score": observed_quadratic_score(
                                    candidate.covariance, test
                                ),
                                **_evaluate(
                                    candidate.covariance,
                                    parent,
                                    [
                                        (design_indices(design, index), item)
                                        for index, item in enumerate(test, start=test_start)
                                    ],
                                    grid,
                                    evaluation_cache[scenario],
                                ),
                                "covariance_error_bound": candidate.estimate.covariance_error_bound,
                                "basis_id": candidate.basis_id,
                                "full_reference_tail_bound": parent.tail_bounds[
                                    "full_innovation_trace_bound"
                                ],
                            }
                        )
                        for target, a in parent.exposure_coefficients.items():
                            estimate = float(a[:j] @ candidate.covariance @ a[:j])
                            truth = float(a @ parent.innovation_covariance @ a)
                            spans = span_by_target[target]
                            risk_rows.append(
                                {
                                    **common,
                                    "rule": rule,
                                    "target": target,
                                    "estimate": estimate,
                                    "truth": truth,
                                    "risk_error": estimate - truth,
                                    "risk_mse": (estimate - truth) ** 2,
                                    "available_train_dates": sum(
                                        span["available"] for span in spans[:n]
                                    ),
                                    **score_risk(estimate, test, spans[test_start:]),
                                    "risk_reference_tail_bound": parent.tail_bounds["risk_bounds"][
                                        target
                                    ],
                                    "unavailable_reason": (
                                        None
                                        if any(span["available"] for span in spans[test_start:])
                                        else "no same-date full-H future proxy"
                                    ),
                                }
                            )
                    # Direct risk never uses a finite-J functional approximation.
                    # Any view has identical y/N and row order, so use the first one.
                    observations = views[config.candidate_dimensions[0]]
                    for target, a in parent.exposure_coefficients.items():
                        spans = span_by_target[target]
                        direct = direct_risk_estimate(observations[:n], spans[:n])
                        truth = float(a @ parent.innovation_covariance @ a)
                        estimate = direct["estimate"]
                        risk_rows.append(
                            {
                                **common,
                                "rule": "direct",
                                "target": target,
                                "estimate": estimate,
                                "truth": truth,
                                "risk_error": estimate - truth if estimate is not None else None,
                                "risk_mse": (estimate - truth) ** 2
                                if estimate is not None
                                else None,
                                "available_train_dates": direct["available_train_dates"],
                                **score_risk(
                                    estimate, observations[test_start:], spans[test_start:]
                                ),
                                "risk_reference_tail_bound": parent.tail_bounds["risk_bounds"][
                                    target
                                ],
                                "unavailable_reason": direct["unavailable_reason"],
                            }
                        )
                    if (
                        config.reference_checks
                        and replication == 0
                        and design == config.designs[0]
                        and n == min(config.training_counts)
                    ):
                        j, relative = (
                            min(config.candidate_dimensions),
                            min(config.relative_regularizations),
                        )
                        own = fitted[(j, relative)].estimate
                        reference = fit_covariance_reference(
                            views[j][:n],
                            regularization=own.regularization,
                            trace_cap=config.trace_cap,
                            numerical_error_budget=config.numerical_error_budget,
                            solver_tolerance=1e-10,
                        )
                        references.append(
                            {
                                **common,
                                "J": j,
                                "relative_regularization": relative,
                                "matrix_discrepancy": float(
                                    np.linalg.norm(own.covariance - reference.covariance)
                                ),
                                "converged": reference.converged,
                                "covariance_error_bound": reference.covariance_error_bound,
                                "reference_refinements": reference.reference_refinements,
                                "reference_solver_tolerance": reference.reference_solver_tolerance,
                            }
                        )
        print(
            f"Partial observations replication {replication + 1}/{config.replications} complete",
            flush=True,
        )
    failures = [
        {
            key: row[key]
            for key in (
                "scenario",
                "design",
                "N",
                "replication",
                "J",
                "relative_regularization",
                "status",
                "covariance_error_bound",
            )
        }
        for row in candidates
        if not row["converged"]
        or row["covariance_error_bound"] > config.numerical_error_budget
        or row["minimum_eigenvalue"] < -config.numerical_tolerance
        or row["trace"] > config.trace_cap + config.numerical_tolerance
    ]
    reference_passed = len(references) == (
        len(config.scenarios) if config.reference_checks else 0
    ) and all(
        row["converged"]
        and row["covariance_error_bound"] <= config.numerical_error_budget
        and row["matrix_discrepancy"] <= 2 * config.numerical_error_budget
        for row in references
    )
    population_passed = all(
        parent.numerical_checks["absolute_hs_discrepancy"] <= config.numerical_tolerance
        and parent.numerical_checks["minimum_eigenvalue"] >= -config.numerical_tolerance
        for parent in prepared["parents"].values()
    )
    results = {
        **_summaries(candidates, selected_rows, risk_rows),
        "geometry": prepared["geometry"],
        "full_h_geometry": prepared["full_h_geometry"],
        "observation_approximation": prepared["observation_approximation"],
        "selections": selections,
        "reference_checks": references,
        "numerical_checks": {
            name: parent.numerical_checks for name, parent in prepared["parents"].items()
        },
        "acceptance": {
            "passed": not failures and reference_passed and population_passed,
            "failed_fits": len(failures),
            "failures": failures,
            "candidate_fits": len(candidates),
            "reference_passed": reference_passed,
            "reference_cases": len(references),
            "population_passed": population_passed,
            "maximum_covariance_error_bound": max(
                row["covariance_error_bound"] for row in candidates
            ),
            "numerical_error_budget": config.numerical_error_budget,
            "reference_resolution_passed": all(
                parent.tail_bounds["full_innovation_trace_bound"] <= config.tail_budget
                and parent.tail_bounds["observed_hs_bound"] <= config.tail_budget
                and max(parent.tail_bounds["risk_bounds"].values()) <= config.tail_budget
                for parent in prepared["parents"].values()
            ),
            "statistical_performance_gate": None,
        },
        "model": {
            "basis_family_id": next(iter(prepared["parents"].values())).model.family_id,
            "sources": {
                name: parent.source_metadata for name, parent in prepared["parents"].items()
            },
            "tail_bounds": {
                name: parent.tail_bounds for name, parent in prepared["parents"].items()
            },
            "contracts": [
                dict(kind=value.kind, start=value.start, end=value.end)
                for value in next(iter(prepared["parents"].values())).contracts
            ],
            "exposure_coefficients": next(iter(prepared["parents"].values())).exposure_coefficients,
            "fixed_grid_witness": prepared["fixed_grid_witness"],
        },
        "protocol": {
            "selection": "observed validation score only; exact ties smaller J then larger relative ridge",
            "data": "one noisy sixteen-contract parent observation before choosing J or design subsets",
            "training": "nested first-N prefixes; common validation begins after maximum training count",
            "test": "common disjoint block after validation; fitted matrices frozen, no validation refit",
            "baseline": f"J={config.baseline_dimension}, relative ridge={config.baseline_relative_regularization}",
            "score": "||O_J B O_J.T||² - 2 <yy.T-N,O_J B O_J.T>; common constant omitted",
            "omission": "full-minus-sieve covariance includes cross terms; not known independent noise",
            "direct_risk": "mean of same-date analytic full-H proxies on available training dates",
            "availability": "same-date proxy failure is not a proof of pooled covariance nonidentification",
            "finite_geometry": "Gram/range statements concern the unrestricted finite-J covariance class",
            "witness_scope": "generic unrestricted covariance experiment; no SPDE-source admissibility claim",
            "mcse": "independent-replication MCSE; rule comparisons paired on identical observations",
            "latent_information": "used only for population approximation and post-selection diagnostics",
        },
    }
    _write_json(output / "results.json", results)
    _write_json(output / "candidate_fits.json", candidates)
    _write_json(output / "selection_audit.json", selections)
    _write_json(output / "full_h_geometry.json", prepared["full_h_geometry"])
    _write_csv(output / "candidates.csv", candidates)
    _write_csv(output / "selected.csv", selected_rows)
    _write_csv(output / "risks.csv", risk_rows)
    from .partial_reporting import write_report

    write_report(output, config, _jsonable(results))
    _write_json(
        output / "status.json",
        {"status": "complete" if results["acceptance"]["passed"] else "failed"},
    )
    if not results["acceptance"]["passed"]:
        raise RuntimeError(
            f"Partial-observation numerical checks failed; inspect {output / 'report.md'}"
        )
    return output
