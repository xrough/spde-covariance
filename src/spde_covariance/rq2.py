"""Local covariance measurement under time variation and partial observation.

Fitting consumes observations only. Known simulation covariances are used in
evaluation, moment-bias calculations and exact Gaussian direct-risk diagnostics.
All primary windows and penalties are predeclared; this suite performs no tuning.
"""

from __future__ import annotations

import csv
import json
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baselines import empirical_covariance, projected_ridge_covariance
from .estimation import CovarianceEstimate, estimate_risk, fit_covariance, fit_covariance_reference
from .geometry import diagnose_design, gram_operator, svec
from .provenance import _jsonable, _manifest, _source_root
from .rq2_design import RQ2Config, build_geometry, generate_data

SOLVER_OPTIONS = {
    "absolute_tolerance": 1e-9,
    "relative_tolerance": 0.0,
    "max_iterations": 30_000,
}
GROUP_KEYS = (
    "scenario",
    "design",
    "endpoint",
    "window",
    "relative_regularization",
    "primary",
)


def fit_window(observations, config, relative_regularization, *, timings=None):
    """Fit the declared methods to one past-only window, without latent truth."""
    data = tuple(observations)
    gram = gram_operator([item.operator for item in data])
    penalty = float(relative_regularization * np.linalg.eigvalsh(gram)[-1])

    def timed(method, function, **kwargs):
        started = time.perf_counter()
        value = function(data, **kwargs)
        if timings is not None:
            timings[method] = time.perf_counter() - started
        return value

    fitted = {
        "constrained_ridge": timed(
            "constrained_ridge",
            fit_covariance,
            regularization=penalty,
            trace_cap=config.trace_cap,
            numerical_error_budget=config.numerical_error_budget,
            **SOLVER_OPTIONS,
        ),
        "projected_ridge": timed(
            "projected_ridge",
            projected_ridge_covariance,
            regularization=penalty,
            trace_cap=config.trace_cap,
        ),
    }
    identity = np.eye(data[0].dimension)
    if all(np.array_equal(item.operator, identity) for item in data):
        fitted["empirical"] = timed("empirical", empirical_covariance)
    return fitted


def _aligned_truth(observations, truths):
    data = tuple(observations)
    matrices = np.asarray(truths, dtype=float)
    if not data:
        raise ValueError("at least one observation is required")
    dimension = data[0].dimension
    if matrices.shape != (len(data), dimension, dimension) or not np.all(np.isfinite(matrices)):
        raise ValueError("truths must be finite, with one matching covariance per observation")
    return data, matrices


def moment_bias(observations, truths):
    """Return sum w_i A_i* A_i(V_i - V_terminal), for uniform local weights."""
    data, matrices = _aligned_truth(observations, truths)
    target = matrices[-1]
    return sum(
        item.operator.T @ (item.operator @ (matrix - target) @ item.operator.T) @ item.operator
        for item, matrix in zip(data, matrices, strict=True)
    ) / len(data)


def direct_risk_diagnostics(observations, truths, target, *, rank_rtol=1e-10):
    """Evaluate a linear risk estimate and its exact independent-Gaussian MSE.

    Variance is 2 sum w_i^2 tr(M_i U_i M_i U_i), with
    U_i=O_i V_i O_i.T+N_i. The estimated risk is left unclipped. A range failure
    reports unavailability rather than a fabricated zero estimate or variance.
    """
    data, matrices = _aligned_truth(observations, truths)
    truth = float(np.sum(target * matrices[-1]))
    result = {
        "identified": False,
        "unavailable_reason": None,
        "estimate": None,
        "truth": truth,
        "error": None,
        "squared_error": None,
        "expected_direct": None,
        "analytic_bias": None,
        "analytic_variance": None,
        "analytic_mse": None,
    }
    try:
        risk = estimate_risk(data, target, relative_threshold=rank_rtol)
    except ValueError as error:
        if "target is not observable" not in str(error):
            raise
        result["unavailable_reason"] = str(error)
        return result
    expected, variance = 0.0, 0.0
    for item, matrix, weight, influence in zip(
        data, matrices, risk.weights, risk.influence_matrices, strict=True
    ):
        visible = item.operator @ matrix @ item.operator.T
        observed_covariance = visible + item.noise_covariance
        expected += weight * np.sum(influence * visible)
        # The symmetric sandwich avoids a negative round-off trace from M U M U.
        eigenvalues, eigenvectors = np.linalg.eigh(observed_covariance)
        root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
        variance += 2 * weight**2 * np.sum((root @ influence @ root) ** 2)
    bias = float(expected - truth)
    result.update(
        identified=True,
        estimate=risk.estimate,
        error=risk.estimate - truth,
        squared_error=(risk.estimate - truth) ** 2,
        expected_direct=float(expected),
        analytic_bias=bias,
        analytic_variance=float(variance),
        analytic_mse=float(variance + bias**2),
    )
    return result


def _mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    )


def _groups(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups


def _summaries(rows, risks):
    summary = []
    metrics = (
        "operator_hs_error",
        "operator_hs_relative_error",
        "observed_covariance_rmse",
        "kernel_grid_rmse",
        "exposure_risk_rmse",
        "runtime_seconds",
        "covariance_error_bound",
    )
    for key, records in _groups(rows, (*GROUP_KEYS, "method")).items():
        item = dict(zip((*GROUP_KEYS, "method"), key, strict=True))
        item["replications"] = len(records)
        for metric in metrics:
            values = [r[metric] for r in records if r[metric] is not None]
            mean, mcse = _mean_mcse(values) if values else (None, None)
            item.update({metric + "_mean": mean, metric + "_mcse": mcse})
        summary.append(item)
    risk_summary = []
    for key, records in _groups(risks, (*GROUP_KEYS, "target", "method")).items():
        item = dict(zip((*GROUP_KEYS, "target", "method"), key, strict=True))
        available = [r for r in records if r["estimate"] is not None]
        item.update(
            identified=records[0]["identified"],
            replications=len(records),
            available_replications=len(available),
            bias=None,
            bias_mcse=None,
            mse=None,
            mse_mcse=None,
            analytic_bias=records[0]["analytic_bias"],
            analytic_variance=records[0]["analytic_variance"],
            analytic_mse=records[0]["analytic_mse"],
        )
        if available:
            item["bias"], item["bias_mcse"] = _mean_mcse([r["error"] for r in available])
            item["mse"], item["mse_mcse"] = _mean_mcse([r["squared_error"] for r in available])
        risk_summary.append(item)
    return summary, risk_summary


def _paired_differences(risks):
    paired = []
    for key, records in _groups(risks, (*GROUP_KEYS, "target")).items():
        baseline = {r["replication"]: r for r in records if r["method"] == "constrained_ridge"}
        for method in dict.fromkeys(r["method"] for r in records):
            if method == "constrained_ridge":
                continue
            differences = [
                r["squared_error"] - baseline[r["replication"]]["squared_error"]
                for r in records
                if r["method"] == method and r["squared_error"] is not None
            ]
            mean, mcse = _mean_mcse(differences) if differences else (None, None)
            item = dict(zip((*GROUP_KEYS, "target"), key, strict=True))
            item.update(
                method_vs="constrained_ridge",
                comparison_method=method,
                replications=len(differences),
                mse_difference=mean,
                mcse=mcse,
            )
            paired.append(item)
    return paired


def _settings(config):
    settings = [
        (endpoint, window, config.relative_regularization, True)
        for endpoint in config.endpoints
        for window in config.windows
    ]
    settings.extend(
        (config.intervals, config.sensitivity_window, relative, False)
        for relative in config.sensitivity_relatives
    )
    return settings


def _geometry_record(data, truths, targets, config, relative, common):
    operators = [item.operator for item in data]
    gram = gram_operator(operators)
    penalty = float(relative * np.linalg.eigvalsh(gram)[-1])
    geometry = diagnose_design(
        operators,
        weights=np.full(len(data), 1 / len(data)),
        targets=targets,
        regularization=penalty,
        rank_rtol=config.rank_rtol,
    )
    bias = moment_bias(data, truths)
    bias_coordinates = svec(bias)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    regularized_bias = np.linalg.norm(
        (eigenvectors.T @ bias_coordinates) / np.sqrt(np.maximum(eigenvalues, 0) + penalty)
    )
    return {
        **common,
        "rank": geometry["rank"],
        "eigenvalues": geometry["eigenvalues"],
        "rank_rtol": config.rank_rtol,
        "rank_threshold": geometry["rank_threshold"],
        "rank_at_1e-8": int(np.count_nonzero(geometry["eigenvalues"] > 1e-8 * eigenvalues[-1])),
        "rank_at_1e-12": int(np.count_nonzero(geometry["eigenvalues"] > 1e-12 * eigenvalues[-1])),
        "targets": geometry["targets"],
        "effective_sample_size": len(data),
        "trailing_window_years": len(data) * config.delta,
        "first_interval": common["endpoint"] - len(data) + 1,
        "last_interval": common["endpoint"],
        "regularization": penalty,
        "moment_bias_matrix": bias,
        "moment_bias_hs": float(np.linalg.norm(bias)),
        "regularized_moment_bias_norm": float(regularized_bias),
    }


def _evaluate(covariance, terminal, operators, grid, exposures):
    error = covariance - terminal
    return {
        "operator_hs_error": float(np.linalg.norm(error)),
        "operator_hs_relative_error": float(np.linalg.norm(error) / np.linalg.norm(terminal)),
        "observed_covariance_rmse": float(
            np.sqrt(np.mean([np.sum((op @ error @ op.T) ** 2) for op in operators]))
        ),
        "kernel_grid_rmse": float(np.sqrt(np.mean((grid @ error @ grid.T) ** 2))),
        "exposure_risk_rmse": float(
            np.sqrt(np.mean([(a @ error @ a) ** 2 for a in exposures.values()]))
        ),
    }


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_rq2_benchmark(config, output=None):
    """Run the predeclared local-estimation experiment into a fresh directory."""
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as stream:
            config = RQ2Config(**tomllib.load(stream))
    if not isinstance(config, RQ2Config):
        raise TypeError("config must be RQ2Config or a TOML path")
    root = _source_root()
    if output is None:
        output = (root or Path.cwd()) / "experiments" / "runs" / config.name
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _run_rq2(config, output, root)
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


def _run_rq2(config, output, root):
    manifest = _manifest(config, root)
    manifest["solver"].update(SOLVER_OPTIONS)
    manifest.update(
        experiment="RQ2: deterministic time-varying Gaussian innovation covariance",
        target="terminal finite-interval innovation covariance V; no C recovery or forecasting",
        fitting_information="observations, known maps/noise, predeclared windows and ridge only",
        pairing="shared latent innovations and exact dense-noise subsets across designs",
    )
    _write_json(output / "config.json", asdict(config))
    _write_json(output / "manifest.json", manifest)
    (output / "status.json").write_text('{"status": "running"}\n')
    model, point_map, exposures = build_geometry()
    targets = {name: np.outer(a, a) for name, a in exposures.items()}
    grid = model.evaluate(np.linspace(0, 3, 101))
    rows, risks, estimates, geometry, references, failures = [], [], [], [], [], []
    reference_endpoint = config.endpoints[min(1, len(config.endpoints) - 1)]
    reference_window = (
        config.sensitivity_window
        if config.sensitivity_window in config.windows
        else max(config.windows)
    )
    for replication in range(config.replications):
        generated = generate_data(config, replication)
        for scenario, designs in generated.designs.items():
            for design, observations in designs.items():
                for endpoint, window, relative, primary in _settings(config):
                    common = dict(
                        scenario=scenario,
                        design=design,
                        endpoint=endpoint,
                        window=window,
                        relative_regularization=relative,
                        primary=primary,
                    )
                    selected = observations[endpoint - window : endpoint]
                    truths = generated.truths[scenario][endpoint - window : endpoint]
                    operators = [item.operator for item in selected]
                    local_geometry = _geometry_record(
                        selected, truths, targets, config, relative, common
                    )
                    if replication == 0:
                        geometry.append(local_geometry)
                    timings = {}
                    fitted = fit_window(selected, config, relative, timings=timings)
                    for method, fit in fitted.items():
                        covariance = fit.covariance if isinstance(fit, CovarianceEstimate) else fit
                        certificate = fit if isinstance(fit, CovarianceEstimate) else None
                        failed = certificate is not None and (
                            not certificate.converged
                            or certificate.covariance_error_bound > config.numerical_error_budget
                        )
                        if failed:
                            failures.append(
                                {
                                    **common,
                                    "replication": replication,
                                    "method": method,
                                    "reason": certificate.status,
                                    "covariance_error_bound": certificate.covariance_error_bound,
                                }
                            )
                        eigenvalues = np.linalg.eigvalsh(covariance)
                        row = {
                            **common,
                            "replication": replication,
                            "method": method,
                            **_evaluate(covariance, truths[-1], operators, grid, exposures),
                            "runtime_seconds": timings[method],
                            "covariance_error_bound": (
                                certificate.covariance_error_bound if certificate else None
                            ),
                            "objective_gap_bound": (
                                certificate.objective_gap_bound if certificate else None
                            ),
                            "solver_converged": certificate.converged if certificate else None,
                            "iterations": certificate.iterations if certificate else None,
                            "min_eigenvalue": float(eigenvalues[0]),
                            "trace": float(np.trace(covariance)),
                            "regularization": local_geometry["regularization"],
                            "rank": local_geometry["rank"],
                        }
                        rows.append(row)
                        estimates.append(
                            {
                                **common,
                                "replication": replication,
                                "method": method,
                                "covariance": covariance,
                            }
                        )
                        for target_name, target in targets.items():
                            estimate = float(np.sum(target * covariance))
                            truth = float(np.sum(target * truths[-1]))
                            risks.append(
                                {
                                    **common,
                                    "replication": replication,
                                    "method": method,
                                    "target": target_name,
                                    "identified": local_geometry["targets"][target_name][
                                        "identified"
                                    ],
                                    "unavailable_reason": None,
                                    "estimate": estimate,
                                    "truth": truth,
                                    "error": estimate - truth,
                                    "squared_error": (estimate - truth) ** 2,
                                    "expected_direct": None,
                                    "analytic_bias": None,
                                    "analytic_variance": None,
                                    "analytic_mse": None,
                                }
                            )
                    for target_name, target in targets.items():
                        risks.append(
                            {
                                **common,
                                "replication": replication,
                                "method": "direct",
                                "target": target_name,
                                **direct_risk_diagnostics(
                                    selected, truths, target, rank_rtol=config.rank_rtol
                                ),
                            }
                        )
                    if (
                        config.reference_checks
                        and replication == 0
                        and scenario in ("constant", "rotating")
                        and design in ("dense", "split")
                        and endpoint == reference_endpoint
                        and window == reference_window
                        and primary
                    ):
                        own = fitted["constrained_ridge"]
                        reference = fit_covariance_reference(
                            selected,
                            regularization=own.regularization,
                            trace_cap=config.trace_cap,
                            solver_tolerance=1e-10,
                            max_iterations=100_000,
                        )
                        references.append(
                            {
                                **common,
                                "matrix_discrepancy": float(
                                    np.linalg.norm(reference.covariance - own.covariance)
                                ),
                                "objective_discrepancy": abs(reference.objective - own.objective),
                                "reference_converged": reference.converged,
                                "reference_status": reference.status,
                                "reference_covariance_error_bound": reference.covariance_error_bound,
                            }
                        )
        print(f"RQ2 replication {replication + 1}/{config.replications} complete", flush=True)
    summary, risk_summary = _summaries(rows, risks)
    certificates = [r["covariance_error_bound"] for r in rows if r["method"] == "constrained_ridge"]
    feasibility = all(
        r["min_eigenvalue"] >= -1e-10 and r["trace"] <= config.trace_cap + 1e-10
        for r in rows
        if r["method"] != "empirical"
    )
    reference_passed = all(
        r["reference_converged"]
        and r["reference_covariance_error_bound"] <= config.numerical_error_budget
        and r["matrix_discrepancy"] <= 2 * config.numerical_error_budget
        for r in references
    )
    results = {
        "summary": summary,
        "risk_summary": risk_summary,
        "paired_differences": _paired_differences(risks),
        "geometry": geometry,
        "reference_checks": references,
        "acceptance": {
            "passed": not failures and feasibility and reference_passed,
            "failed_fits": len(failures),
            "failures": failures,
            "feasibility_passed": feasibility,
            "reference_passed": reference_passed,
            "reference_checks_requested": config.reference_checks,
            "reference_cases": len(references),
            "maximum_covariance_error_bound": max(certificates),
            "numerical_error_budget": config.numerical_error_budget,
            "statistical_performance_gate": None,
        },
        "model": {
            "basis_id": model.basis_id,
            "maturities": [1 / 12, 0.5, 1.5, 3.0],
            "exposures": exposures,
            "exposure_normalization": "unit H norm; standardized risk, not currency P&L",
            "point_map": point_map,
            "evaluation_grid": np.linspace(0, 3, 101),
            "covariance_target": "terminal V at last included interval",
            "design_interpretation": "synthetic fixed-maturity innovation maps; not a delivery calendar",
        },
    }
    _write_json(output / "results.json", results)
    _write_json(output / "geometry.json", geometry)
    _write_json(output / "estimates.json", estimates)
    _write_csv(output / "replications.csv", rows)
    _write_csv(output / "risks.csv", risks)
    from .rq2_reporting import write_report

    write_report(output, config, results)
    _write_json(
        output / "status.json",
        {"status": "complete" if results["acceptance"]["passed"] else "failed"},
    )
    if not results["acceptance"]["passed"]:
        raise RuntimeError(f"RQ2 numerical checks failed; inspect {output / 'report.md'}")
    return output
