"""Paired, observed-data-tuned benchmarks for the transport-SPDE workflow.

Truth is passed only to evaluation. Complete and partial information sets are
compared separately; unadjusted increments are a labelled transport ablation.
"""

from __future__ import annotations

import csv
import json
import time
import tomllib
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .baselines import refit_selection, select_covariance
from .benchmark_stress import run_stress_suite
from .estimation import estimate_risk, fit_covariance_reference
from .geometry import diagnose_design, gram_operator, svec
from .observations import Observation, delivery_average_map, point_observation_map
from .provenance import _jsonable, _manifest, _source_root
from .trajectories import InnovationBatch, extract_innovations, simulate_transport
from .transport import ExponentialTransport


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "stage1"
    seed: int = 20261023
    replications: int = 20
    train: int = 256
    validation: int = 128
    evaluation: int = 128
    delta: float = 1 / 52
    trace_cap: float = 2.0
    noise_variance: float = 0.001
    numerical_error_budget: float = 1e-4
    relative_candidates: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    shrinkage_candidates: tuple[float, ...] = (0.0, 0.1, 0.5)
    window_candidates: tuple[int, ...] = (128, 256)
    rank_rtol: float = 1e-10
    reference_checks: bool = True
    stress_smoke: bool = False

    def __post_init__(self):
        for name, minimum in (
            ("seed", 0),
            ("replications", 2),
            ("train", 8),
            ("validation", 4),
            ("evaluation", 4),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        for name in ("delta", "trace_cap", "numerical_error_budget"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.noise_variance) or self.noise_variance < 0:
            raise ValueError("noise_variance must be finite and nonnegative")
        if not 0 < self.rank_rtol < 1:
            raise ValueError("rank_rtol must be between zero and one")
        for name in ("relative_candidates", "shrinkage_candidates", "window_candidates"):
            values = tuple(getattr(self, name))
            if not values:
                raise ValueError(f"{name} must be nonempty")
            object.__setattr__(self, name, values)
        if any(not np.isfinite(x) or x <= 0 for x in self.relative_candidates):
            raise ValueError("relative_candidates must be finite and positive")
        if any(not np.isfinite(x) or not 0 <= x <= 1 for x in self.shrinkage_candidates):
            raise ValueError("shrinkage_candidates must lie in [0,1]")
        if any(
            isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= self.train
            for x in self.window_candidates
        ):
            raise ValueError("window_candidates must be observation counts <= train")


SOLVER_OPTIONS = {
    "absolute_tolerance": 1e-13,
    "relative_tolerance": 1e-13,
    "max_iterations": 30_000,
}

SOURCE_COVARIANCE = np.array(
    [[0.4, 0.04, 0.03, 0], [0.04, 0.3, 0, 0.02], [0.03, 0, 0.2, 0.02], [0, 0.02, 0.02, 0.1]]
)
SOURCE_COVARIANCE.flags.writeable = False


def _designs(model, count):
    # Sparse rows are exact subsets of the dense rows, including their noise.
    dense = np.vstack(
        (
            point_observation_map(model, [0, 0.15, 0.5, 1, 2, 3]),
            delivery_average_map(model, [[0.25, 0.75], [1.5, 2.5]]),
        )
    )
    # Explicit unit-H exposure scaling improves numerical comparability.
    dense /= np.linalg.norm(dense, axis=1)[:, None]
    subsets = [np.array([i % 8, (i + 3) % 8]) for i in range(count)]
    missing = [np.eye(4)[[0, 1] if i % 2 == 0 else [2, 3]] for i in range(count)]
    return dense, subsets, missing


def benchmark_data(config: BenchmarkConfig, replication: int):
    """Generate paired information sets; never called by the tuning functions."""
    model = ExponentialTransport()
    count = config.train + config.validation + config.evaluation
    source = SOURCE_COVARIANCE
    seed = config.seed + 1009 * replication
    simulation = simulate_transport(model, np.arange(count + 1) * config.delta, source, seed=seed)
    batch = extract_innovations(simulation.trajectory, model)
    raw = InnovationBatch(
        batch.times,
        np.diff(simulation.trajectory.coefficients, axis=0)
        / np.sqrt(batch.interval_lengths)[:, None],
        batch.basis_id,
        "ordinary_unadjusted_increment",
    )
    dense, subsets, missing = _designs(model, count)
    rng = np.random.default_rng(seed + 17)
    dense_noise = rng.normal(scale=np.sqrt(config.noise_variance), size=(count, len(dense)))
    missing_noise = rng.normal(scale=np.sqrt(config.noise_variance), size=(count, 2))
    designs, ablations = {}, {}
    for label, maps in (
        ("full", [np.eye(4)] * count),
        ("dense", [dense] * count),
        ("sparse", [dense[subset] for subset in subsets]),
        ("missing_pair", missing),
    ):
        noise = (
            np.zeros((count, 4))
            if label == "full"
            else dense_noise
            if label == "dense"
            else np.stack([dense_noise[i, subset] for i, subset in enumerate(subsets)])
            if label == "sparse"
            else missing_noise
        )
        variance = 0 if label == "full" else config.noise_variance
        designs[label] = tuple(
            Observation(float(t), op @ x + e, op, variance * np.eye(len(op)))
            for t, x, op, e in zip(batch.end_times, batch.values, maps, noise, strict=True)
        )
        ablations[label] = tuple(
            Observation(float(t), op @ x + e, op, variance * np.eye(len(op)))
            for t, x, op, e in zip(batch.end_times, raw.values, maps, noise, strict=True)
        )
    return model, model.innovation_covariance(source, config.delta), designs, ablations


def evaluation_metrics(covariance, truth, operators, grid, exposures, *, rank_rtol=1e-10):
    """Errors in explicit orthonormal-H, grid, observation and exposure metrics."""
    difference = covariance - truth
    gram = gram_operator(operators)
    spectrum, vectors = np.linalg.eigh(gram)
    spectrum = np.maximum(spectrum, 0)
    active = spectrum > rank_rtol * spectrum[-1]
    coordinates = vectors.T @ svec(difference)
    risks = np.array([v @ difference @ v for v in exposures])
    true_risks = np.array([v @ truth @ v for v in exposures])
    kernel = grid @ difference @ grid.T
    return {
        "operator_hs_error": float(np.linalg.norm(difference)),
        "operator_hs_relative_error": float(np.linalg.norm(difference) / np.linalg.norm(truth)),
        "operator_spectral_error": float(np.linalg.norm(difference, 2)),
        "operator_spectral_relative_error": float(
            np.linalg.norm(difference, 2) / np.linalg.norm(truth, 2)
        ),
        "observed_covariance_rmse": float(
            np.sqrt(np.mean([np.sum((op @ difference @ op.T) ** 2) for op in operators]))
        ),
        "kernel_grid_rmse": float(np.sqrt(np.mean(kernel**2))),
        "kernel_grid_relative_rmse": float(
            np.linalg.norm(kernel) / np.linalg.norm(grid @ truth @ grid.T)
        ),
        "exposure_risk_rmse": float(np.sqrt(np.mean(risks**2))),
        "exposure_risk_relative_rmse": float(np.linalg.norm(risks) / np.linalg.norm(true_risks)),
        "resolved_direction_error": float(np.linalg.norm(coordinates[active])),
        "weak_direction_error": float(np.linalg.norm(coordinates[~active])),
        "numerical_rank": int(active.sum()),
        "rank_threshold": float(rank_rtol * spectrum[-1]),
        "rank_rtol": rank_rtol,
        "rank_at_1e-8": int(np.count_nonzero(spectrum > 1e-8 * spectrum[-1])),
        "rank_at_1e-12": int(np.count_nonzero(spectrum > 1e-12 * spectrum[-1])),
    }


def _summary(rows):
    results = []
    keys = (
        "operator_hs_relative_error",
        "operator_spectral_relative_error",
        "observed_covariance_rmse",
        "kernel_grid_rmse",
        "exposure_risk_rmse",
        "resolved_direction_error",
        "weak_direction_error",
        "runtime_seconds",
        "peak_traced_bytes",
        "covariance_error_bound",
    )
    for case, method in dict.fromkeys((r["case"], r["method"]) for r in rows):
        matching = [r for r in rows if r["case"] == case and r["method"] == method]
        result = {
            "case": case,
            "method": method,
            "replications": len(matching),
            "numerical_rank": matching[0]["numerical_rank"],
        }
        for key in keys:
            values = np.array([r[key] for r in matching if r.get(key) is not None], dtype=float)
            result[key + "_mean"] = float(np.mean(values)) if len(values) else None
            result[key + "_mcse"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
            )
        results.append(result)
    return results


def _paired_differences(rows):
    results = []
    for case in dict.fromkeys(r["case"] for r in rows):
        base = {
            r["replication"]: r
            for r in rows
            if r["case"] == case and r["method"] == "constrained_ridge"
        }
        for method in dict.fromkeys(r["method"] for r in rows if r["case"] == case):
            if method == "constrained_ridge":
                continue
            values = np.array(
                [
                    r["operator_hs_relative_error"]
                    - base[r["replication"]]["operator_hs_relative_error"]
                    for r in rows
                    if r["case"] == case and r["method"] == method
                ]
            )
            results.append(
                {
                    "case": case,
                    "method_minus_constrained": method,
                    "paired_relative_error_difference": float(np.mean(values)),
                    "mcse": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                }
            )
    return results


def _direct_summary(records):
    summary = []
    for case, target in dict.fromkeys((r["case"], r["target"]) for r in records):
        matching = [r for r in records if r["case"] == case and r["target"] == target]
        errors = np.array([r["error"] for r in matching if r["identified"]])
        row = {"case": case, "target": target, "identified": len(errors) == len(matching)}
        if len(errors):
            row.update(
                bias=float(errors.mean()),
                bias_mcse=float(errors.std(ddof=1) / np.sqrt(len(errors))),
                mse=float(np.mean(errors**2)),
                mse_mcse=float(np.std(errors**2, ddof=1) / np.sqrt(len(errors))),
            )
        summary.append(row)
    return summary


def _fit_case(config, data, *, ablation=False):
    train, validation = data[: config.train], data[config.train : config.train + config.validation]
    prefix = data[: config.train + config.validation]
    methods = ["constrained_ridge", "projected_ridge"]
    if all(np.array_equal(item.operator, np.eye(4)) for item in data):
        methods += ["empirical", "linear_shrinkage"]
    if ablation:
        methods = ["constrained_ridge"]
    for method in methods:
        start = time.perf_counter()
        tracemalloc.start()
        try:
            selected = select_covariance(
                train,
                validation,
                method=method,
                relative_candidates=(
                    config.shrinkage_candidates
                    if method == "linear_shrinkage"
                    else config.relative_candidates
                ),
                window_candidates=config.window_candidates,
                trace_cap=config.trace_cap,
                numerical_error_budget=config.numerical_error_budget,
                solver_options=SOLVER_OPTIONS,
            )
            fitted = refit_selection(
                prefix,
                selected,
                trace_cap=config.trace_cap,
                numerical_error_budget=config.numerical_error_budget,
                solver_options=SOLVER_OPTIONS,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        elapsed = time.perf_counter() - start
        yield method, selected, fitted, elapsed, peak


def run_benchmark(config: BenchmarkConfig | str | Path, output: str | Path | None = None):
    if not isinstance(config, BenchmarkConfig):
        with Path(config).open("rb") as stream:
            config = BenchmarkConfig(**tomllib.load(stream))
    output = Path(output or f"experiments/runs/{config.name}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(config, _source_root())
    manifest["solver"].update(SOLVER_OPTIONS)
    manifest["experiment"] = "paired clean transport trajectories; observed-only validation tuning"
    manifest["measurement"] = "dense moderate dimensions; peak traced allocations, not process RSS"
    (output / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, allow_nan=False)
    )
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    rows, audits, direct, references = [], [], [], []
    geometry = {}
    for replication in range(config.replications):
        model, truth, designs, ablations = benchmark_data(config, replication)
        grid = model.evaluate(np.linspace(0, 3, 101))
        spread = model.evaluate([0.5])[0] - model.evaluate([2.0])[0]
        exposures = [np.eye(4)[0], spread / np.linalg.norm(spread)]
        targets = {
            "coordinate_0": np.outer(exposures[0], exposures[0]),
            "unit_H_spread_0.5_2": np.outer(exposures[1], exposures[1]),
        }
        for case, data in designs.items():
            evaluation = data[config.train + config.validation :]
            operators = [item.operator for item in evaluation]
            if replication == 0:
                geometry[case] = diagnose_design(
                    operators, targets=targets, rank_rtol=config.rank_rtol
                )
            for is_ablation, fitting_data in ((False, data), (True, ablations[case])):
                for method, selected, fitted, elapsed, peak in _fit_case(
                    config, fitting_data, ablation=is_ablation
                ):
                    row = {
                        "case": case,
                        "method": "unadjusted_increment_ablation" if is_ablation else method,
                        "replication": replication,
                        "seed": config.seed + 1009 * replication,
                        **evaluation_metrics(
                            fitted.covariance,
                            truth,
                            operators,
                            grid,
                            exposures,
                            rank_rtol=config.rank_rtol,
                        ),
                        "runtime_seconds": elapsed,
                        "peak_traced_bytes": peak,
                        "regularization": fitted.selected_parameters.get("regularization"),
                        "relative_regularization": fitted.selected_parameters.get(
                            "relative_regularization"
                        ),
                        "window": fitted.selected_parameters["window"],
                        "min_eigenvalue": float(np.linalg.eigvalsh(fitted.covariance)[0]),
                        "trace": float(np.trace(fitted.covariance)),
                        "objective_gap_bound": None,
                        "covariance_error_bound": None,
                        "iterations": None,
                        "solver_converged": None,
                    }
                    if fitted.estimate is not None:
                        estimate = fitted.estimate
                        row.update(
                            objective_gap_bound=estimate.objective_gap_bound,
                            covariance_error_bound=estimate.covariance_error_bound,
                            iterations=estimate.iterations,
                            solver_converged=estimate.converged,
                        )
                    rows.append(row)
                    audits.append(
                        {
                            "case": case,
                            "method": row["method"],
                            "replication": replication,
                            "selected_parameters": selected.selected_parameters,
                            "refit_parameters": fitted.selected_parameters,
                            "validation_score": selected.validation_score,
                            "candidates": selected.candidates,
                        }
                    )
                    if (
                        not is_ablation
                        and method == "constrained_ridge"
                        and replication == 0
                        and config.reference_checks
                    ):
                        window = fitted.selected_parameters["window"]
                        prefix = data[: config.train + config.validation][-window:]
                        reference = fit_covariance_reference(
                            prefix,
                            regularization=fitted.selected_parameters["regularization"],
                            trace_cap=config.trace_cap,
                            solver_tolerance=1e-8,
                            max_iterations=100_000,
                        )
                        references.append(
                            {
                                "case": case,
                                "solver": reference.solver,
                                "matrix_discrepancy": float(
                                    np.linalg.norm(reference.covariance - fitted.covariance)
                                ),
                                "reference_objective": reference.objective,
                                "reference_converged": reference.converged,
                                "reference_objective_gap_bound": reference.objective_gap_bound,
                                "reference_covariance_error_bound": reference.covariance_error_bound,
                                "objective_discrepancy": abs(
                                    reference.objective - estimate.objective
                                ),
                            }
                        )
            prefix = data[: config.train + config.validation][-max(config.window_candidates) :]
            design = diagnose_design(
                [item.operator for item in prefix], targets=targets, rank_rtol=config.rank_rtol
            )
            for target_name, target in targets.items():
                identified = design["targets"][target_name]["identified"]
                record = {
                    "case": case,
                    "replication": replication,
                    "target": target_name,
                    "identified": identified,
                    "window": len(prefix),
                    "nullspace_relative_residual": design["targets"][target_name][
                        "nullspace_relative_residual"
                    ],
                }
                if identified:
                    risk = estimate_risk(prefix, target, relative_threshold=config.rank_rtol)
                    record.update(
                        estimate=risk.estimate,
                        truth=float(np.sum(target * truth)),
                        error=risk.estimate - float(np.sum(target * truth)),
                    )
                direct.append(record)
        print(f"replication {replication + 1}/{config.replications} complete", flush=True)
    stress = run_stress_suite(
        seed=config.seed,
        smoke=config.stress_smoke,
        numerical_error_budget=config.numerical_error_budget,
    )
    summary = _summary(rows)
    paired = _paired_differences(rows)
    control = next(
        row for row in summary if row["case"] == "full" and row["method"] == "constrained_ridge"
    )
    certificates = [r for r in rows if r["covariance_error_bound"] is not None]
    accepted = (
        control["operator_hs_relative_error_mean"] < 0.3
        and all(
            r["solver_converged"] and r["covariance_error_bound"] <= config.numerical_error_budget
            for r in certificates
        )
        and geometry["missing_pair"]["rank"] < 10
        and all(r["matrix_discrepancy"] < 2 * config.numerical_error_budget for r in references)
        and all(
            r["converged"]
            and r["numerical_covariance_error_bound"] <= config.numerical_error_budget
            for section in ("performance", "temporal_variation")
            for r in stress.get(section, {}).get("rows", [])
        )
    )
    results = {
        "summary": summary,
        "paired_differences": paired,
        "geometry": geometry,
        "direct_risks": direct,
        "direct_risk_summary": _direct_summary(direct),
        "reference_checks": references,
        "stress": stress,
        "analytical_model": {
            "rates_per_year": model.rates,
            "basis_id": model.basis_id,
            "propagation_method": model.kind,
            "source_covariance_C": SOURCE_COVARIANCE,
            "normalized_innovation_covariance_V": truth,
            "exposures": exposures,
            "evaluation_maturities": np.linspace(0, 3, 101),
            "covariance_unit": "arbitrary curve unit squared/year",
        },
        "acceptance": {
            "passed": accepted,
            "full_control_relative_error_limit": 0.3,
            "numerical_error_budget": config.numerical_error_budget,
        },
    }
    for name, value in (("results.json", results), ("tuning.json", audits)):
        (output / name).write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False))
    with (output / "replications.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _figure(summary, output)
    (output / "report.md").write_text(_report(config, results, manifest))
    if not accepted:
        raise RuntimeError(f"benchmark acceptance failed; inspect {output / 'report.md'}")
    return output


def _figure(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["operator_hs_relative_error", "exposure_risk_rmse"],
        ["Relative operator error (Hilbert–Schmidt)", "Exposure-risk RMSE"],
        strict=True,
    ):
        methods = list(dict.fromkeys(r["method"] for r in summary))
        cases = list(dict.fromkeys(r["case"] for r in summary))
        for index, method in enumerate(methods):
            matching = [r for r in summary if r["method"] == method]
            ax.errorbar(
                [cases.index(r["case"]) + (index - 2) * 0.11 for r in matching],
                [r[metric + "_mean"] for r in matching],
                yerr=[r[metric + "_mcse"] for r in matching],
                fmt="o",
                markersize=4,
                capsize=2,
                label=method.replace("_", " "),
            )
        ax.set_xticks(range(len(cases)), [x.replace("_", " ") for x in cases])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def _report(config, results, manifest):
    lines = [
        "# Stage 1 synthetic benchmark report",
        "",
        f"Configuration `{config.name}`; {config.replications} paired replications, "
        f"{config.train}/{config.validation}/{config.evaluation} training/validation/evaluation "
        f"intervals; Δ={config.delta:.8g} years; dimension 4.",
        "",
        f"Source revision: `{manifest['git_revision']}`; dirty={manifest['git_dirty']}. "
        "`manifest.json` records source hashes, lockfile, environment and solver settings.",
        "",
        "The exact exponential model supplies analytic V independently of covariance fitting. "
        "Candidates use corrected observed validation outer products; latent truth is used "
        "only for evaluation. Ridge candidates are relative to the training Gram norm and "
        "are rescaled on refit. Refit uses the selected trailing count through validation. "
        "The evaluation block is untouched by tuning.",
        "",
        "## Paired comparisons",
        "",
        "Entries are Monte Carlo mean ± standard error, not confidence intervals. "
        "Full and partial settings have different information and are compared separately.",
        "",
        "| Setting | Method | Relative HS error | Kernel grid RMSE | Exposure-risk RMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in results["summary"]:
        values = [
            f"{row[key + '_mean']:.4g} ± {row[key + '_mcse']:.2g}"
            for key in ("operator_hs_relative_error", "kernel_grid_rmse", "exposure_risk_rmse")
        ]
        lines.append(f"| {row['case']} | {row['method']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Paired benchmark comparison](comparison.png)",
        "",
        "## Interpretation and diagnostics",
        "",
        "Dense rows are six point evaluations and two delivery averages, each scaled to "
        "unit H norm. Sparse observations are nested rotating two-row subsets, including "
        "the exact same noise draws. The full case observes coefficients with O=I and "
        "zero noise; it is a recovery control, not a fair competitor to partial data. "
        "The missing-pair design alternates coordinate blocks (0,1) and (2,3), so "
        "cross-block covariance is unidentified. Designs are deterministic.",
        "",
        "The transport ablation uses ordinary increments of the same trajectory with the "
        "same observation maps and noise. It tunes against its own unadjusted validation "
        "increments and is evaluated against V to expose transport bias. It is not an "
        "additional valid V estimator under the stated model.",
        "",
        "Kernel error is RMSE on a fixed 101-point maturity grid [0,3] years. Exposures "
        "are coordinate 0 and the 0.5-versus-2-year point spread normalized to unit H "
        "norm. Both are declared before simulation. Direct risk estimates use the "
        "largest declared trailing count and are reported only when numerically identified; "
        "they need not produce a PSD full covariance.",
        "",
        "`replications.csv` includes absolute/relative HS and spectral operator errors, "
        "observed covariance, kernel and risk errors, resolved/weak-direction errors, "
        "rank at 1e-8/1e-10/1e-12, feasibility, certificates and timings. `tuning.json` "
        "preserves all candidate scores and selected relative/actual penalties. "
        "`results.json` includes target nullspace residuals, paired method differences "
        "with Monte Carlo standard errors, direct risks, independent convex references "
        "and all stress/performance results.",
        "",
        "Optimization certificates bound error to the constrained optimum, not truth. "
        "Runtime includes tuning and refitting; peak traced allocation excludes process "
        "RSS and native allocations not registered with Python tracing. The separate "
        "dimension ladder measures one fit at a common certificate budget. This is a "
        "dense moderate-dimensional implementation.",
        "",
        "## Stress cases",
        "",
        "The stress suite separates analytic V−C bridge error, quadrature discrepancy, "
        "spline interior projection and omitted-tail error, projected semigroup defect, "
        "padding, conditioning, temporal-window bias, and J=4/8/16/24 fit performance. "
        "Temporal stress fits are labelled local approximations; irregular intervals "
        "retain interval-specific targets and common-V fitting is rejected.",
        "",
        f"Acceptance: **{'PASS' if results['acceptance']['passed'] else 'FAIL'}**. "
        "The predeclared O=I mean relative-HS limit is 0.30, selected/refitted constrained fits must "
        "meet their numerical budget, reference discrepancies must be within twice that "
        "budget, and missing-pair rank must be deficient. This does not require ridge "
        "to outperform every baseline. Failed tuning candidates are recorded and skipped.",
        "",
        "These synthetic experiments do not establish empirical market performance, "
        "robustness to outliers or model misspecification, or futures-curve prediction.",
        "",
    ]
    if results.get("direct_risk_summary"):
        lines += [
            "## Direct exposure-risk estimates",
            "",
            "Bias and MSE have Monte Carlo standard errors. Unidentified targets are "
            "reported as unavailable rather than estimated by regularization.",
            "",
            "| Setting | Target | Identified | Bias ± MCSE | MSE ± MCSE |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for row in results["direct_risk_summary"]:
            values = (
                f"{row['bias']:.4g} ± {row['bias_mcse']:.2g} | "
                f"{row['mse']:.4g} ± {row['mse_mcse']:.2g}"
                if row["identified"]
                else "unavailable | unavailable"
            )
            lines.append(f"| {row['case']} | {row['target']} | {row['identified']} | {values} |")
        lines.append("")
    stress = results["stress"]
    if stress:
        lines += [
            "## Bridge and numerical stress measurements",
            "",
            "| Δ (years) | ‖V−C‖ HS | Analytic versus quadrature |",
            "| ---: | ---: | ---: |",
        ]
        for row in stress["dt_bridge"]:
            lines.append(
                f"| {row['dt']:.6g} | {row['v_minus_c_hs']:.5g} | "
                f"{row['analytic_vs_gauss_hs']:.3g} |"
            )
        lines += [
            "",
            "Spline errors below concern the four smooth reference basis functions, "
            "summed in squared H norm. A finite constant-tail domain leaves a tail "
            "error floor. More padding at fixed dimension can reduce tail error while "
            "worsening interior resolution; whole-sieve semigroup defects need not "
            "decrease with dimension.",
            "",
            "| J | Padding (years) | H projection error | Omitted-tail H error | Shift grid RMSE |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in stress["spline_projection"]["rows"]:
            lines.append(
                f"| {row['basis_size']} | {row['padding_length']:g} | "
                f"{row['projection_error_h']:.5g} | {row['omitted_tail_error_h']:.5g} | "
                f"{row['shift_function_rmse']:.5g} |"
            )
        lines += [
            "",
            "| Regime checkpoint | Window | Window-target bias HS | Actual HS error |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in stress["temporal_variation"]["rows"]:
            lines.append(
                f"| {row['endpoint_label']} | {row['window_intervals']} | "
                f"{row['window_covariance_bias_hs']:.5g} | "
                f"{row['actual_covariance_error_hs']:.5g} |"
            )
        lines += [
            "",
            "| J | Fit time (s) | Peak traced MiB | Iterations | Certified error bound |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in stress["performance"]["rows"]:
            lines.append(
                f"| {row['basis_size']} | {row['runtime_seconds']:.4g} | "
                f"{row['python_tracemalloc_peak_bytes'] / 2**20:.4g} | "
                f"{row['solver_iterations']} | "
                f"{row['numerical_covariance_error_bound']:.3g} |"
            )
        lines += [
            "",
            "Performance entries are single diagnostic runs on the environment in "
            "the manifest, with 64 observations and a common singular-value range "
            "[0.5,1]. They do not provide Monte Carlo uncertainty for timing. "
            "Statistical comparisons above use paired replications.",
            "",
        ]
    return "\n".join(lines)
