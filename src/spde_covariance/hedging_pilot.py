"""Identification-to-hedging pilot with paired, past-only synthetic evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .hedging_controls import fixed_power_hedge_control
from .provenance import _jsonable, _manifest, _source_root

DESIGNS = {
    "fragments": ((0, 1, 2, 3), (3, 4, 5, 6)),
    "bridge": ((0, 1, 2, 3, 6), (3, 4, 5, 6)),
    "complete": (tuple(range(7)),),
}


@dataclass(frozen=True)
class HedgingPilotConfig:
    name: str = "identified-hedging"
    seed: int = 20261029
    replications: int = 20
    training_counts: tuple[int, ...] = (128, 512)
    heldout_count: int = 2048
    source_dimension: int = 16
    output_dimension: int = 8
    reference_dimension: int = 512
    quadrature_order: int = 24
    delta: float = 1 / 52
    noise_variance: float = 0.001
    confidence_level: float = 0.95
    source_trace_cap: float = 2.0
    ridge_trace_cap: float = 2.0
    relative_regularization: float = 0.001
    numerical_error_budget: float = 1e-4
    solver_tolerance: float = 1e-7
    position_limit: float = 8.0

    def __post_init__(self):
        for name in (
            "seed",
            "replications",
            "heldout_count",
            "source_dimension",
            "output_dimension",
            "reference_dimension",
            "quadrature_order",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (name != "seed"):
                raise ValueError(f"invalid {name}")
        counts = tuple(self.training_counts)
        if (
            not counts
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 8 for n in counts)
            or tuple(sorted(set(counts))) != counts
            or any(n % 2 for n in counts)
        ):
            raise ValueError("training_counts must be increasing distinct positive even counts >=8")
        object.__setattr__(self, "training_counts", counts)
        if not 8 <= self.output_dimension <= self.source_dimension <= self.reference_dimension:
            raise ValueError(
                "require 8 <= output_dimension <= source_dimension <= reference_dimension"
            )
        for name in (
            "delta",
            "source_trace_cap",
            "ridge_trace_cap",
            "relative_regularization",
            "numerical_error_budget",
            "solver_tolerance",
            "position_limit",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.noise_variance) or self.noise_variance < 0:
            raise ValueError("noise_variance must be finite and nonnegative")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie in (0,1)")
        if self.source_trace_cap < 1:
            raise ValueError("source_trace_cap must include the declared trace-one sources")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")


def _root(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if values[0] < -1e-10 * max(1, values[-1]):
        raise ValueError("conditional simulation covariance is materially indefinite")
    return vectors * np.sqrt(np.maximum(values, 0))


def paired_observation_parents(control, *, count, noise_variance, seed):
    """Correct Gaussian parents with exactly shared native observed fragments.

    Condition each world's unobserved coordinates on a common observed-clique
    draw. The native observed laws coincide, so training data can be shared
    exactly without supplying truth to fitting. Bridge/full views are nested
    subsets of these same parents. Conditional simulation uses truth only here.
    """
    rng = np.random.default_rng(seed)
    groups = DESIGNS["fragments"]
    noise = noise_variance * np.eye(7)
    base = np.asarray(control["baseline_covariance"]) + noise
    parents = [np.empty((count, 7)), np.empty((count, 7))]
    for parity, group in enumerate(groups):
        rows = np.arange(parity, count, 2)
        observed = np.array(group)
        missing = np.array(sorted(set(range(7)) - set(group)))
        block = base[np.ix_(observed, observed)]
        visible = rng.normal(size=(len(rows), len(observed))) @ _root(block).T
        remaining = rng.normal(size=(len(rows), len(missing)))
        for parent, kernel in zip(parents, control["kernels"], strict=True):
            covariance = np.asarray(kernel) + noise
            cross = covariance[np.ix_(missing, observed)]
            regression = np.linalg.solve(block, cross.T).T
            conditional = covariance[np.ix_(missing, missing)] - regression @ cross.T
            parent[np.ix_(rows, observed)] = visible
            parent[np.ix_(rows, missing)] = (
                visible @ regression.T + remaining @ _root(conditional).T
            )
    return tuple(parents)


def mask_training_prefix(parent, count, design):
    """Issue a fitting input from a past prefix; never read its future rows."""
    if design not in DESIGNS:
        raise ValueError("unknown observation design")
    if not isinstance(count, int) or count < 1 or count > len(parent):
        raise ValueError("count must select an available past prefix")
    masked = np.full((count, 7), np.nan)
    groups = DESIGNS[design]
    for index in range(count):
        rows = groups[index % len(groups)]
        masked[index, list(rows)] = parent[index, list(rows)]
    return masked


def evaluate_issued_hedge(
    positions,
    audit_values,
    kernel,
    *,
    target,
    instrument,
    noise_variance,
    native_design="fragments",
):
    """Score an already-issued hedge; truth is only an evaluation diagnostic.

    audit_values is a separate complete future panel. It is not a native
    fragment score and must not be described as one. A known-noise correction
    leaves an unbiased quadratic risk proxy in this synthetic experiment.
    """
    h = float(np.asarray(positions).reshape(-1)[0])
    residual = target - h * instrument
    proxy = (audit_values @ residual) ** 2 - noise_variance * (residual @ residual)
    true_risk = float(residual @ kernel @ residual)
    variance = float(instrument @ kernel @ instrument)
    optimum = float(target @ kernel @ instrument / variance)
    oracle = float(target @ kernel @ target - (target @ kernel @ instrument) ** 2 / variance)
    groups = DESIGNS[native_design]
    support = set(np.flatnonzero(residual != 0))
    native_rows = np.array([support.issubset(groups[i % len(groups)]) for i in range(len(proxy))])
    return {
        "position": h,
        "true_residual_variance": true_risk,
        "true_oracle_hedge": optimum,
        "true_oracle_variance": oracle,
        "true_regret": true_risk - oracle,
        "complete_audit_score": float(np.mean(proxy)),
        "complete_audit_count": len(proxy),
        "native_scoring_available": bool(np.any(native_rows)),
        "native_score_count": int(np.count_nonzero(native_rows)),
        "native_score": float(np.mean(proxy[native_rows])) if np.any(native_rows) else None,
        "scope": "future complete audit observations; controlled masking, not native-fragment evidence",
    }


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _summaries(rows):
    result = []
    keys = sorted({(r["world"], r["design"], r["training_count"], r["method"]) for r in rows})
    for world, design, count, method in keys:
        selected = [
            r
            for r in rows
            if (r["world"], r["design"], r["training_count"], r["method"])
            == (world, design, count, method)
        ]
        available = [r for r in selected if r["available"]]
        summary = {
            "world": world,
            "design": design,
            "training_count": count,
            "method": method,
            "replications": len(selected),
            "available": len(available),
        }
        for metric in ("position", "true_residual_variance", "true_regret", "complete_audit_score"):
            values = np.array([r[metric] for r in available])
            summary[metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "mcse": float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else None,
            }
        result.append(summary)
    return result


def _paired(rows):
    result = []
    index = {
        (r["replication"], r["world"], r["design"], r["training_count"], r["method"]): r
        for r in rows
    }
    settings = sorted({(r["world"], r["design"], r["training_count"]) for r in rows})
    for world, design, count in settings:
        methods = sorted({r["method"] for r in rows})
        for method in methods:
            for metric in ("complete_audit_score", "true_residual_variance"):
                differences = []
                for rep in sorted({r["replication"] for r in rows}):
                    first = index.get((rep, world, design, count, method))
                    base = index.get((rep, world, design, count, "canonical_coherent"))
                    if first and base and first["available"] and base["available"]:
                        differences.append(first[metric] - base[metric])
                result.append(
                    {
                        "world": world,
                        "design": design,
                        "training_count": count,
                        "metric": metric,
                        "contrast": f"{method} minus canonical_coherent",
                        "paired_replications": len(differences),
                        "mean": float(np.mean(differences)) if differences else None,
                        "mcse": float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
                        if len(differences) > 1
                        else None,
                    }
                )
    return result


def run_hedging_pilot(config_path, output=None):
    """Run a declared pilot into a new directory, retaining every unavailable fit."""
    from .hedging_population import run_population_hedge_study
    from .hedging_reporting import write_hedging_report
    from .hedging_workflow import fit_hedging_rules

    path = Path(config_path)
    config = HedgingPilotConfig(**tomllib.loads(path.read_text()))
    destination = Path(output) if output is not None else Path("experiments/runs") / config.name
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "config.json", asdict(config))
    manifest = _manifest(config, _source_root())
    manifest["solver"] = {
        "name": "SCS robust support/dual; CLARABEL population endpoint; existing ridge/coherent",
        "tolerance": config.solver_tolerance,
        "max_iterations": 100_000,
        "numerical_error_budget": config.numerical_error_budget,
        "numerical_identification_ranges_are_not_exact_certificates": True,
    }
    manifest["config_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["experiment"] = (
        "fixed infinite-rank sources; shared masked Gaussian training; complete future audit"
    )
    _write_json(destination / "manifest.json", manifest)
    _write_json(destination / "status.json", {"status": "running", "completed_replications": 0})
    control = fixed_power_hedge_control(
        source_dimension=config.source_dimension,
        output_dimension=config.output_dimension,
        reference_dimension=config.reference_dimension,
        delta=config.delta,
        quadrature_order=config.quadrature_order,
    )
    _write_json(destination / "control.json", control)
    population = run_population_hedge_study(
        control, solver_tolerance=min(config.solver_tolerance, 1e-8)
    )
    _write_json(destination / "population.json", population)
    rows, fit_audits = [], []
    maximum = max(config.training_counts)
    for replication in range(config.replications):
        parents = paired_observation_parents(
            control,
            count=maximum + config.heldout_count,
            noise_variance=config.noise_variance,
            seed=np.random.SeedSequence([config.seed, replication]),
        )
        cached = {}
        for world, (parent, truth) in enumerate(zip(parents, control["kernels"], strict=True)):
            for design, cliques in DESIGNS.items():
                for count in config.training_counts:
                    cache_key = (design, count) if design == "fragments" else (world, design, count)
                    if cache_key not in cached:
                        start = time.perf_counter()
                        cached[cache_key] = fit_hedging_rules(
                            mask_training_prefix(parent, count, design),
                            evaluation_matrix=control["evaluation"],
                            source_factors=control["factors"],
                            source_offset=control["offset_covariance"],
                            source_trace_cap=config.source_trace_cap,
                            ridge_trace_cap=config.ridge_trace_cap,
                            cliques=cliques,
                            noise_variance=config.noise_variance,
                            confidence_level=config.confidence_level,
                            relative_regularization=config.relative_regularization,
                            numerical_error_budget=config.numerical_error_budget,
                            position_limit=config.position_limit,
                            solver_tolerance=config.solver_tolerance,
                        )
                        elapsed = time.perf_counter() - start
                    else:
                        elapsed = 0.0
                    for method, fit in cached[cache_key].items():
                        row = {
                            "replication": replication,
                            "world": world,
                            "design": design,
                            "training_count": count,
                            "forecast_origin": count,
                            "training_start": 1,
                            "training_end": count,
                            "audit_start": maximum + 1,
                            "audit_end": len(parent),
                            "method": method,
                            "available": bool(fit["available"]),
                            "reason": fit.get("reason"),
                            "fit_group_runtime_seconds": elapsed,
                        }
                        if fit["available"]:
                            row.update(
                                evaluate_issued_hedge(
                                    fit["positions"],
                                    parent[maximum:],
                                    np.asarray(truth),
                                    target=np.asarray(control["target"]),
                                    instrument=np.asarray(control["hedge"]),
                                    noise_variance=config.noise_variance,
                                    native_design=design,
                                )
                            )
                            kernel = fit.get("fitted_kernel")
                            row["kernel_errors"] = None
                            if kernel is not None:
                                difference = np.asarray(kernel) - truth
                                mask = np.zeros((7, 7), dtype=bool)
                                for clique in cliques:
                                    mask[np.ix_(clique, clique)] = True
                                weights = np.array([0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
                                weighted = np.sqrt(weights[:, None] * weights[None, :]) * difference
                                row["kernel_errors"] = {
                                    "complete": float(np.linalg.norm(weighted, "fro")),
                                    "observed": float(np.linalg.norm(weighted[mask])),
                                    "missing": float(np.linalg.norm(weighted[~mask])),
                                    "target_variance_error": float(
                                        control["target"] @ difference @ control["target"]
                                    ),
                                    "off_grid": None,
                                    "off_grid_reason": "kernel-only completion has no declared off-grid extension",
                                }
                        bounds = fit["training_observation_bounds"]
                        transformed_truth = (
                            bounds["contrast_matrix"] @ truth @ bounds["contrast_matrix"].T
                        )
                        entries = transformed_truth[tuple(bounds["entry_pairs"].T)]
                        row["simulated_truth_in_training_bands"] = bool(
                            np.all((entries >= bounds["lower"]) & (entries <= bounds["upper"]))
                        )
                        rows.append(row)
                        fit_audits.append(
                            {
                                "replication": replication,
                                "world": world,
                                "design": design,
                                "training_count": count,
                                "method": method,
                                "fit": fit,
                            }
                        )
        _write_json(destination / "estimates.json", rows)
        _write_json(destination / "fit_audit.json", fit_audits)
        _write_json(
            destination / "status.json",
            {"status": "running", "completed_replications": replication + 1},
        )
    summary, contrasts = _summaries(rows), _paired(rows)
    population_passed = bool(
        population["all_source_intervals_available"]
        and all(
            row["source_interval_inside_baseline"]
            for row in population["nesting_diagnostics"].values()
        )
        and control["paired_decisions"]["hedge_gap"] > 0.1
        and control["construction"]["observed_image_difference"] < 1e-10
        and np.min(control["construction"]["minimum_finite_source_eigenvalues"]) >= -1e-12
        and max(control["integration_diagnostics"]["fixed_source_kernel_block_discrepancies"])
        < 1e-8
    )
    acceptance = {
        "passed": population_passed,
        "population_identification_gate": population_passed,
        "all_scheduled_replications_completed": len(rows)
        == config.replications * 2 * len(DESIGNS) * len(config.training_counts) * 6,
        "unavailable_results_retained": sum(not r["available"] for r in rows),
        "no_estimator_superiority_requirement": True,
    }
    acceptance["all_prespecified_point_and_robust_fits_available"] = all(
        r["available"]
        for r in rows
        if r["method"] in ("canonical_coherent", "ridge", "robust_psd", "robust_source")
    )
    acceptance["complete_direct_control_available"] = any(
        r["available"]
        for r in rows
        if r["design"] == "complete" and r["method"] == "direct_target_block"
    )
    acceptance["direct_noise_correction_may_be_indefinite"] = True
    acceptance["passed"] &= acceptance["all_scheduled_replications_completed"]
    acceptance["passed"] &= acceptance["all_prespecified_point_and_robust_fits_available"]
    acceptance["passed"] &= acceptance["complete_direct_control_available"]
    results = {
        "acceptance": acceptance,
        "summary": summary,
        "paired_contrasts": contrasts,
        "monte_carlo_unit": "independent paired replication; not future rows or worlds",
        "summary_scope": "conditional on available fits; paired contrasts use common available replications",
        "conditional_band_scope": "independent centered common Gaussian rows; fixed noise, masks and shared source tail",
        "future_market_evidence": False,
    }
    _write_json(destination / "results.json", results)
    for filename, data in (("summary.csv", summary), ("paired.csv", contrasts)):
        with (destination / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()}
                for row in data
            )
    write_hedging_report(destination, config, control, population, results)
    _write_json(
        destination / "status.json",
        {"status": "complete", "completed_replications": config.replications},
    )
    if not acceptance["passed"]:
        raise RuntimeError(
            f"hedging pilot acceptance failed; diagnostics retained in {destination}"
        )
    return destination
