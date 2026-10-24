"""Compatible PSD fragment estimation in the existing partial-observation workflow."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .admissibility import covariance_integration_matrix, diagnose_constant_source_admissibility
from .coherent import fit_coherent_partial
from .coherent_config import CoherentBenchmarkConfig
from .completion import complete_kernel, weighted_psd_projection
from .fragment_benchmark import (
    _array_statistics,
    _compact,
    _csv,
    _json,
    _metrics,
    _prepare,
    _summary,
    balanced_assignments,
    exposure_score,
    fit_fragment_kernel,
)
from .laguerre import LaguerreTransport
from .observation_statistics import kernel_quadratic_score_from_statistics
from .provenance import _manifest, _source_root

_WORKER_CONTEXT = None


def _initialize_worker(config, settings):
    """Bounded per-process model cache, released when this executor closes."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (config, _prepare(config), settings)


def _worker_replication(replication):
    return _fit_replication(replication, *_WORKER_CONTEXT)


def _run_replications(config, output, prepared, settings):
    """Checkpoint each completed replication and aggregate in seed order."""
    records = {}
    manifest_sha = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()

    def retain(record):
        replication = record["replication"]
        record["parent_manifest_sha256"] = manifest_sha
        _json(output / f"replication-{replication:04d}.json", record)
        if record["error"] is not None:
            _json(
                output / "candidates.json",
                [c for r in records.values() for c in r["candidates"]] + record["candidates"],
            )
            raise RuntimeError(f"replication {replication}: {record['error']}; checkpoint retained")
        records[replication] = record
        _json(
            output / "status.json",
            {
                "status": "running",
                "completed_replications": len(records),
                "completed_replication_indices": sorted(records),
            },
        )

    if config.workers == 1:
        for replication in range(config.replications):
            retain(_fit_replication(replication, config, prepared, settings))
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config, settings),
        ) as executor:
            pending = [
                executor.submit(_worker_replication, rep) for rep in range(config.replications)
            ]
            try:
                for future in as_completed(pending):
                    retain(future.result())
            except BaseException:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
    return [records[rep] for rep in range(config.replications)]


def fit_coherent_candidates(training, validation, *, cliques, weights, settings):
    """Fit/choose from observed statistics only; no simulation geometry or truth."""
    coverage = training.pair_weight_sums
    scale = float(np.max(coverage))
    fits, audit = [], []
    for relative in settings["relative_regularizations"]:
        started = time.perf_counter()
        fit = fit_coherent_partial(
            np.where(training.observed_mask, training.partial_kernel, np.nan),
            pair_weights=coverage,
            cliques=cliques,
            weights=weights,
            regularization=relative * scale,
            eigenvalue_floor=settings["eigenvalue_floor"],
            numerical_error_budget=settings["numerical_error_budget"],
            rho=settings["rho"],
            max_iterations=settings["max_iterations"],
        )
        completion = (
            complete_kernel(
                fit.partial_kernel,
                cliques=cliques,
                weights=weights,
                support_rtol=settings["support_rtol"],
                population=True,
            )
            if fit.converged
            else None
        )
        available = completion is not None and completion.kernel is not None
        record = {
            "relative_regularization": relative,
            "regularization": relative * scale,
            "coverage_norm": scale,
            "available": available,
            "converged": fit.converged,
            "runtime_seconds": time.perf_counter() - started,
            "validation_score": kernel_quadratic_score_from_statistics(
                completion.kernel, validation
            )
            if available
            else None,
            "partial_error_bound": fit.partial_error_bound,
            "objective_gap_bound": fit.objective_gap_bound,
            "iterations": fit.diagnostics["iterations"],
            "partial": _compact(fit.diagnostics),
            "completion": _compact(completion.diagnostics) if completion is not None else None,
        }
        fits.append((completion.kernel if available else None, fit))
        audit.append(record)
    # Do not silently discard a numerically failed candidate from the declared grid.
    if not all(row["available"] for row in audit):
        return fits, audit, None
    selected = min(
        range(len(audit)),
        key=lambda i: (audit[i]["validation_score"], -audit[i]["relative_regularization"]),
    )
    return fits, audit, selected


def _cone_record(kernel, nodes, weights, dimension, config, cache, label):
    key = (
        dimension,
        hashlib.sha256(
            np.asarray(kernel).tobytes()
            + np.asarray(nodes).tobytes()
            + np.asarray(weights).tobytes()
        ).hexdigest(),
    )
    if key in cache:
        return {**cache[key], **label, "reused_numerical_fit": True}
    if dimension not in cache:
        propagator = LaguerreTransport(dimension)
        cache[dimension] = covariance_integration_matrix(
            propagator, config.delta, quadrature_order=config.quadrature_order
        )
    started = time.perf_counter()
    evaluation = LaguerreTransport(dimension).evaluate(nodes)
    root = np.sqrt(weights)
    vectors, singular, _ = np.linalg.svd(root[:, None] * evaluation, full_matrices=False)
    retained = singular > 1e-12 * singular[0]
    basis = vectors[:, retained]
    weighted_kernel = root[:, None] * kernel * root[None, :]
    representation_residual = np.linalg.norm(
        weighted_kernel - basis @ (basis.T @ weighted_kernel @ basis) @ basis.T
    )
    fit = diagnose_constant_source_admissibility(
        kernel,
        evaluation,
        delta=config.delta,
        trace_cap=config.trace_cap,
        integration_matrix=cache[dimension],
        weights=weights,
        objective_gap_tolerance=config.admissibility_gap,
        max_iterations=config.admissibility_max_iterations,
    )
    record = {
        "J": dimension,
        "source_basis_id": LaguerreTransport(dimension).basis_id,
        "propagation_method": LaguerreTransport(dimension).kind,
        "integration_matrix_sha256": hashlib.sha256(cache[dimension].tobytes()).hexdigest(),
        "source_trace_cap": config.trace_cap,
        "distance_lower": fit.fit_distance_lower_bound,
        "distance_upper": fit.fit_distance_upper_bound,
        "representation_residual": float(representation_residual),
        "evaluation_rank": int(retained.sum()),
        "evaluation_rank_rtol": 1e-12,
        "objective_gap_bound": fit.objective_gap_bound,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "status": fit.admissibility_status,
        "runtime_seconds": time.perf_counter() - started,
        "source_feasibility": _compact(fit.source_feasibility),
        "metadata": _compact(fit.metadata),
    }
    cache[key] = record
    return {**record, **label, "reused_numerical_fit": False}


def _source_controls(config):
    """A realizable control and a strictly PSD kernel outside a finite source cone."""
    model = LaguerreTransport(2)
    integration = covariance_integration_matrix(model, 1.0)
    kernels = {
        "generated": model.innovation_covariance(np.diag([0.4, 0.3]), 1.0),
        "positive_definite_obstruction": np.diag([0.01, 1.01]),
    }
    records = []
    for name, kernel in kernels.items():
        fit = diagnose_constant_source_admissibility(
            kernel,
            np.eye(2),
            delta=1.0,
            trace_cap=4.0,
            integration_matrix=integration,
            objective_gap_tolerance=1e-12,
        )
        records.append(
            {
                "control": name,
                "distance_lower": fit.fit_distance_lower_bound,
                "distance_upper": fit.fit_distance_upper_bound,
                "objective_gap_bound": fit.objective_gap_bound,
                "converged": fit.converged,
                "status": fit.admissibility_status,
            }
        )
    return records


def run_coherent_benchmark(config, output=None):
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as handle:
            config = CoherentBenchmarkConfig(**tomllib.load(handle))
    if not isinstance(config, CoherentBenchmarkConfig):
        raise TypeError("coherent benchmark requires CoherentBenchmarkConfig")
    output = Path(output or Path("experiments/runs") / config.name)
    if output.exists():
        raise FileExistsError("output must be a new directory")
    output.mkdir(parents=True)
    try:
        return _run(config, output)
    except (Exception, KeyboardInterrupt) as error:
        _json(output / "status.json", {"status": "failed", "error": str(error)})
        raise


def _run(config, output):
    from .coherent_reporting import write_coherent_report

    settings = {
        "relative_regularizations": config.partial_regularizations,
        "eigenvalue_floor": config.eigenvalue_floor,
        "numerical_error_budget": config.numerical_error_budget,
        "rho": config.admm_rho,
        "max_iterations": config.admm_max_iterations,
        "support_rtol": config.support_rtol,
    }
    manifest = _manifest(config, _source_root())
    manifest.update(
        solver={
            "coherent_partial": {"name": "consensus_ADMM", **settings},
            "completion": "full numerical support, population compatibility checks",
            "source_cone": {
                "name": "projected_gradient",
                "gap": config.admissibility_gap,
                "max_iterations": config.admissibility_max_iterations,
            },
        },
        selection_information="observed validation only; larger penalty on exact ties; no refit",
        experiment="paired follow-up to historical canonical SPDE fragments",
        execution={
            "workers": config.workers,
            "process_start_method": "spawn" if config.workers > 1 else None,
            "logical_cpu_count": os.cpu_count(),
            "aggregation_order": "replication index",
            "worker_cache": "one prepared configuration per process; released with executor",
        },
        numerical_environment={
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    )
    _json(output / "manifest.json", manifest)
    _json(output / "config.json", asdict(config))
    _json(output / "status.json", {"status": "running"})
    parent_nodes, models, geometry = _prepare(config)
    _json(
        output / "geometry.json",
        [
            {
                "model": name,
                "base_grid": q,
                "cover": m,
                "nodes": g["nodes"],
                "weights": g["weights"],
                "cliques": g["cliques"],
                "population_completion": _compact(g["population_diagnostics"]),
            }
            for (name, q, m), g in geometry.items()
        ],
    )
    controls = _source_controls(config)
    cone_rows, integration_cache = [], {}
    for (name, q, m), geo in geometry.items():
        for target, kernel in (("generating", geo["truth"]), ("canonical", geo["canonical"])):
            if kernel is not None:
                for dimension in config.admissibility_dimensions:
                    cone_rows.append(
                        _cone_record(
                            kernel,
                            geo["nodes"],
                            geo["weights"],
                            dimension,
                            config,
                            integration_cache,
                            {
                                "model": name,
                                "base_grid": q,
                                "cover": m,
                                "target": target,
                                "scope": "population finite grid",
                            },
                        )
                    )
    rows, risks, candidates, selections, examples = [], [], [], [], []
    replication_records = _run_replications(
        config, output, (parent_nodes, models, geometry), settings
    )
    for record in replication_records:
        rows.extend(record["rows"])
        risks.extend(record["risks"])
        candidates.extend(record["candidates"])
        selections.extend(record["selections"])
        examples.extend(record["examples"])
        cone_rows.extend(record["cone_rows"])
    fields = ("model", "noise_variance", "base_grid", "cover", "N", "policy")
    metrics = (
        "test_observed_score",
        "observed_weighted_error",
        "missing_weighted_error",
        "complete_weighted_error",
        "canonical_weighted_error",
        "canonical_generating_discrepancy",
        "observed_entry_change",
        "minimum_weighted_eigenvalue",
        "runtime_seconds",
    )
    summary = _summary(rows, metrics, fields)
    risk_summary = _summary(
        risks, ("risk_score", "risk_squared_error", "risk_error"), (*fields, "target")
    )
    baseline = {
        tuple(r[f] for f in fields[:-1]) + (r["replication"],): r
        for r in rows
        if r["policy"] == "pairwise_rank2"
    }
    differences = []
    for row in rows:
        base = baseline[tuple(row[f] for f in fields[:-1]) + (row["replication"],)]
        differences.append(
            {
                **{k: row[k] for k in (*fields, "replication")},
                **{
                    metric: row[metric] - base[metric]
                    for metric in (
                        "test_observed_score",
                        "complete_weighted_error",
                        "missing_weighted_error",
                    )
                },
            }
        )
    paired = _summary(
        differences,
        ("test_observed_score", "complete_weighted_error", "missing_weighted_error"),
        fields,
    )
    acceptance = {
        "partial_certificates_within_budget": all(
            c["converged"] and c["partial_error_bound"] <= config.numerical_error_budget
            for c in candidates
        ),
        "coherent_outputs_psd": all(
            r["minimum_weighted_eigenvalue"] >= -1e-8
            for r in rows
            if r["policy"].startswith("coherent")
        ),
        "source_controls": controls[0]["distance_upper"] < 1e-5
        and controls[1]["distance_lower"] > 0.001,
        "population_reference_available": all(
            g["canonical"] is not None for g in geometry.values()
        ),
        "source_witnesses_feasible": all(r["source_feasibility"]["feasible"] for r in cone_rows),
    }
    acceptance["passed"] = all(acceptance.values())
    results = {
        "acceptance": acceptance,
        "summary": summary,
        "risk_summary": risk_summary,
        "paired_differences": paired,
        "source_controls": controls,
        "candidate_count": len(candidates),
        "cone_checks": len(cone_rows),
        "cone_unconverged": sum(not r["converged"] for r in cone_rows),
        "models": {name: data["metadata"] for name, data in models.items()},
    }
    for filename, data in (
        ("results.json", results),
        ("candidates.json", candidates),
        ("selections.json", selections),
        ("source_cone.json", cone_rows),
        ("representative_kernels.json", examples),
    ):
        _json(output / filename, data)
    for filename, data in (
        ("estimates.csv", rows),
        ("risks.csv", risks),
        ("summary.csv", summary),
        ("risk_summary.csv", risk_summary),
        ("paired.csv", paired),
    ):
        _csv(output / filename, data)
    write_coherent_report(config, output, results, examples, cone_rows)
    _json(output / "status.json", {"status": "complete" if acceptance["passed"] else "failed"})
    if not acceptance["passed"]:
        raise RuntimeError("coherent benchmark acceptance failed; inspect results.json")
    return output


def _fit_replication(replication, config, prepared, settings):
    """One deterministic paired replication; workers never select on evaluation truth."""
    _, models, geometry = prepared
    rows, risks, candidates, selections, examples, cone_rows = [], [], [], [], [], []
    integration_cache = {}
    validation_start = max(config.training_counts)
    test_start = validation_start + config.validation_count
    assignments = {
        m: balanced_assignments(config.total_count, m, config.seed + 7919 * replication + m)
        for m in config.covers
    }
    for name, model in models.items():
        rng = np.random.default_rng(config.seed + 1009 * replication)
        parent_values = (
            rng.standard_normal((config.total_count, model["sample_map"].shape[1]))
            @ model["sample_map"].T
        )
        parent_noise = rng.standard_normal(parent_values.shape)
        for variance in config.noise_variances:
            noisy_parent = parent_values + np.sqrt(variance) * parent_noise
            for q in config.grids:
                for m in config.covers:
                    geo = geometry[(name, q, m)]
                    values = noisy_parent[:, geo["indices"]]
                    assigned, cliques = assignments[m], geo["cliques"]
                    validation = _array_statistics(
                        values[validation_start:test_start],
                        assigned[validation_start:test_start],
                        cliques,
                        variance,
                    )
                    test = _array_statistics(
                        values[test_start:], assigned[test_start:], cliques, variance
                    )
                    for n in config.training_counts:
                        common = {
                            "replication": replication,
                            "model": name,
                            "noise_variance": variance,
                            "base_grid": q,
                            "actual_grid": len(geo["nodes"]),
                            "cover": m,
                            "N": n,
                        }
                        train = _array_statistics(values[:n], assigned[:n], cliques, variance)
                        fits, audit, selected = fit_coherent_candidates(
                            train,
                            validation,
                            cliques=cliques,
                            weights=geo["weights"],
                            settings=settings,
                        )
                        digest = hashlib.sha256(values[:n].tobytes()).hexdigest()
                        candidates.extend(
                            {
                                **common,
                                **a,
                                "training_values_sha256": digest,
                                "train_end": n,
                                "validation_start": validation_start,
                                "validation_end": test_start,
                            }
                            for a in audit
                        )
                        if selected is None:
                            return {
                                "replication": replication,
                                "error": "declared coherent candidate failed",
                                "candidates": candidates,
                            }
                        selections.append(
                            {
                                **common,
                                "selected_index": selected,
                                "relative_regularization": audit[selected][
                                    "relative_regularization"
                                ],
                                "validation_score": audit[selected]["validation_score"],
                            }
                        )
                        started = time.perf_counter()
                        raw = fit_fragment_kernel(
                            train,
                            cliques=cliques,
                            weights=geo["weights"],
                            rank=2,
                            support_rtol=config.support_rtol,
                        )
                        if raw.kernel is None:
                            raise RuntimeError("historical rank-two baseline unavailable")
                        raw_runtime = time.perf_counter() - started
                        projection_started = time.perf_counter()
                        projected = weighted_psd_projection(
                            raw.kernel, weights=geo["weights"], observed_mask=geo["mask"]
                        )
                        policies = [
                            ("pairwise_rank2", raw.kernel, raw_runtime, {}),
                            (
                                "pairwise_rank2_psd",
                                projected.kernel,
                                raw_runtime + time.perf_counter() - projection_started,
                                {"projection": _compact(projected.diagnostics)},
                            ),
                        ]
                        middle = min(
                            range(len(audit)),
                            key=lambda i: abs(np.log(audit[i]["relative_regularization"] / 0.001)),
                        )
                        for policy, i in (
                            ("coherent_fixed", middle),
                            ("coherent_selected", selected),
                        ):
                            policies.append(
                                (
                                    policy,
                                    fits[i][0],
                                    sum(a["runtime_seconds"] for a in audit)
                                    if policy == "coherent_selected"
                                    else audit[i]["runtime_seconds"],
                                    {
                                        "partial_error_bound": audit[i]["partial_error_bound"],
                                        "objective_gap_bound": audit[i]["objective_gap_bound"],
                                        "relative_regularization": audit[i][
                                            "relative_regularization"
                                        ],
                                        "completion": audit[i]["completion"],
                                    },
                                )
                            )
                        for policy, kernel, runtime, diagnostic in policies:
                            root = np.sqrt(geo["weights"])
                            drift = root[:, None] * (kernel - train.partial_kernel) * root[None, :]
                            rows.append(
                                {
                                    **common,
                                    "policy": policy,
                                    "available": True,
                                    "runtime_seconds": runtime,
                                    "test_observed_score": kernel_quadratic_score_from_statistics(
                                        kernel, test
                                    ),
                                    "observed_entry_change": float(
                                        np.linalg.norm(drift[geo["mask"]])
                                    ),
                                    **{
                                        k: v
                                        for k, v in diagnostic.items()
                                        if k not in ("completion", "projection")
                                    },
                                    **_metrics(
                                        kernel,
                                        geo["truth"],
                                        geo["canonical"],
                                        geo["weights"],
                                        geo["mask"],
                                    ),
                                }
                            )
                            for target, exposure in geo["exposures"].items():
                                estimate = float(exposure @ kernel @ exposure)
                                truth = float(exposure @ geo["truth"] @ exposure)
                                risks.append(
                                    {
                                        **common,
                                        "policy": policy,
                                        "target": target,
                                        "estimate": estimate,
                                        "truth": truth,
                                        "risk_error": estimate - truth,
                                        "risk_squared_error": (estimate - truth) ** 2,
                                        **exposure_score(
                                            estimate,
                                            exposure,
                                            values[test_start:],
                                            assigned[test_start:],
                                            cliques,
                                            variance,
                                        ),
                                    }
                                )
                            if (
                                replication == 0
                                and n == config.training_counts[len(config.training_counts) // 2]
                                and variance == config.noise_variances[-1]
                            ):
                                examples.append(
                                    {
                                        **common,
                                        "policy": policy,
                                        "nodes": geo["nodes"],
                                        "kernel": kernel,
                                        "diagnostics": diagnostic,
                                    }
                                )
                                for dimension in config.admissibility_dimensions:
                                    cone_rows.append(
                                        _cone_record(
                                            kernel,
                                            geo["nodes"],
                                            geo["weights"],
                                            dimension,
                                            config,
                                            integration_cache,
                                            {
                                                **common,
                                                "target": policy,
                                                "scope": "illustrative replication zero; not Monte Carlo summary",
                                            },
                                        )
                                    )
    return {
        "replication": replication,
        "rows": rows,
        "risks": risks,
        "candidates": candidates,
        "selections": selections,
        "examples": examples,
        "cone_rows": cone_rows,
        "error": None,
    }
