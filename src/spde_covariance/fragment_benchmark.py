"""Dense fragments integrated with the existing partial-innovation experiment.

Simulation/evaluation truth lives outside fitting and validation functions.
Parent measurements precede every grid, method, and covariance-basis view.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .completion import complete_kernel, weighted_psd_projection
from .fragment_config import FragmentBenchmarkConfig
from .fragment_reference import paper_cover, paper_kernel, population_controls
from .geometry import diagnose_design, gram_operator
from .laguerre import LaguerreTransport
from .observation_statistics import (
    kernel_quadratic_score_from_statistics,
    partial_kernel_from_arrays,
)
from .observations import Observation
from .partial_benchmark import SOLVER_OPTIONS, FitSettings, choose_candidate, fit_candidate
from .partial_design import fixed_exposures, functional_coefficients, prepare_parent_model
from .provenance import _jsonable, _manifest, _source_root


def quadrature_weights(nodes):
    """Physical trapezoidal measure on a possibly augmented grid."""
    x = np.asarray(nodes)
    result = np.empty(len(x))
    result[0], result[-1] = (x[1] - x[0]) / 2, (x[-1] - x[-2]) / 2
    result[1:-1] = (x[2:] - x[:-2]) / 2
    return result


def balanced_assignments(count, fragments, seed):
    """Shuffled balanced cycles; every smaller N uses the same prefix."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.permutation(fragments) for _ in range((count + fragments - 1) // fragments)]
    )[:count]


def grid_nodes(size, profile):
    if profile == "reference":
        return np.linspace(0, 1, size)
    return np.unique(np.r_[np.linspace(0, 3, size), 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.75])


def clique_indices(nodes, cover):
    return tuple(np.flatnonzero((nodes >= a - 1e-12) & (nodes <= b + 1e-12)) for a, b in cover)


def observed_mask(size, cliques):
    mask = np.zeros((size, size), dtype=bool)
    for indices in cliques:
        mask[np.ix_(indices, indices)] = True
    return mask


def fit_fragment_kernel(statistics, *, cliques, weights, rank=None, fve=None, support_rtol=1e-12):
    """Truth-free canonical strategy: corrected empirical pairs, then completion."""
    return complete_kernel(
        np.where(statistics.observed_mask, statistics.partial_kernel, np.nan),
        cliques=cliques,
        weights=weights,
        rank=rank,
        fve=fve,
        support_rtol=support_rtol,
    )


def array_score(kernel, values, assignments, cliques, noise_variance):
    """Original fragment Frobenius score, including known noise correction."""
    return kernel_quadratic_score_from_statistics(
        kernel, _array_statistics(values, assignments, cliques, noise_variance)
    )


def _array_statistics(values, assignments, cliques, noise_variance):
    row_nodes = tuple(cliques[g] for g in assignments)
    return partial_kernel_from_arrays(
        tuple(values[i, ids] for i, ids in enumerate(row_nodes)),
        row_nodes,
        noise_variance=noise_variance,
        size=values.shape[1],
    )


def _compact(diagnostic):
    """Store spectra/checks without repeating grid masks on every fit."""
    if hasattr(diagnostic, "items"):
        return {
            k: (
                len(v)
                if k in ("separator_indices", "left_indices", "right_indices")
                else _compact(v)
            )
            for k, v in diagnostic.items()
            if k not in ("observed_mask", "weights", "cliques")
        }
    if isinstance(diagnostic, (tuple, list)):
        return [_compact(v) for v in diagnostic]
    return diagnostic


def exposure_score(estimate, exposure, values, assignments, cliques, noise_variance):
    support = np.flatnonzero(exposure)
    proxies = []
    for group, indices in enumerate(cliques):
        if np.all(np.isin(support, indices)):
            data = values[assignments == group]
            proxies.extend((data @ exposure) ** 2 - noise_variance * np.sum(exposure**2))
    return {
        "scoring_available": bool(len(proxies)),
        "available_test_dates": len(proxies),
        "risk_score": float(estimate**2 - 2 * estimate * np.mean(proxies))
        if len(proxies)
        else None,
    }


def _exposures(nodes, profile):
    result = {}
    if profile == "reference":
        # Fixed grid indices across all reference replications (not rounded latent targets).
        definitions = {
            "point_spread": (len(nodes) // 5, 4 * len(nodes) // 5),
            "long_spread": (0, len(nodes) - 1),
        }
        for name, (a, b) in definitions.items():
            u = np.zeros(len(nodes))
            u[a], u[b] = 1, -1
            result[name] = u
        return result
    for name, exposure in fixed_exposures().items():
        if name == "delivery_spread":
            continue
        u = np.zeros(len(nodes))
        for contract, value in zip(exposure.contracts, exposure.weights, strict=True):
            matches = np.flatnonzero(np.isclose(nodes, contract.start, rtol=0, atol=1e-12))
            if len(matches) != 1:
                raise ValueError("fixed exposure maturity must occur exactly once")
            u[matches[0]] = value
        result[name] = u
    u = np.zeros(len(nodes))
    u[np.flatnonzero(np.isclose(nodes, 0.25))[0]] = 1
    u[np.flatnonzero(np.isclose(nodes, 2.75))[0]] = -1
    result["long_spread"] = u
    delivery = fixed_exposures()["delivery_spread"]
    u = np.zeros(len(nodes))
    for contract, scale in zip(delivery.contracts, delivery.weights, strict=True):
        indices = np.flatnonzero(
            (nodes >= contract.start - 1e-12) & (nodes <= contract.end + 1e-12)
        )
        u[indices] += scale * quadrature_weights(nodes[indices]) / (contract.end - contract.start)
    result["grid_delivery_spread"] = u
    return result


def _metrics(kernel, truth, canonical, weights, mask):
    measure = np.outer(weights, weights)
    delta = kernel - truth
    result = {}
    for name, region in (("observed", mask), ("missing", ~mask), ("complete", np.ones_like(mask))):
        squared = float(np.sum(measure[region] * delta[region] ** 2))
        scale = float(np.sum(measure[region] * truth[region] ** 2))
        result[name + "_weighted_error"] = np.sqrt(squared)
        result[name + "_integrated_squared_error"] = squared
        result[name + "_relative_error"] = np.sqrt(squared / scale) if scale > 0 else None
    result["integrated_squared_error"] = result["complete_weighted_error"] ** 2
    result["canonical_weighted_error"] = (
        float(np.sqrt(np.sum(measure * (kernel - canonical) ** 2)))
        if canonical is not None
        else None
    )
    result["canonical_generating_discrepancy"] = (
        float(np.sqrt(np.sum(measure * (canonical - truth) ** 2)))
        if canonical is not None
        else None
    )
    eigen = np.linalg.eigvalsh(np.sqrt(weights)[:, None] * kernel * np.sqrt(weights)[None, :])
    result.update(
        minimum_weighted_eigenvalue=float(eigen[0]),
        weighted_trace=float(eigen.sum()),
        negative_eigenvalues=int(np.sum(eigen < -1e-10 * max(1.0, abs(eigen[-1])))),
    )
    return result


def _json(path, data):
    path.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False) + "\n")


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_jsonable(rows))


def _summary(rows, metrics, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    output = []
    for key, records in groups.items():
        record = dict(zip(fields, key, strict=True))
        record["replications"] = len(records)
        record["available_replications"] = sum(r.get("available", True) for r in records)
        record["unavailable_replications"] = sum(not r.get("available", True) for r in records)
        for metric in metrics:
            v = np.array([row[metric] for row in records if row.get(metric) is not None], float)
            record[metric + "_available"] = len(v)
            record[metric + "_mean"] = float(v.mean()) if len(v) else None
            record[metric + "_mcse"] = (
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
            )
            record[metric + "_median"] = float(np.median(v)) if len(v) else None
            record[metric + "_mad"] = float(np.median(np.abs(v - np.median(v)))) if len(v) else None
            record[metric + "_mean_absolute_deviation_from_median"] = (
                float(np.mean(np.abs(v - np.median(v)))) if len(v) else None
            )
            record[metric + "_mean_absolute_deviation_from_mean"] = (
                float(np.mean(np.abs(v - np.mean(v)))) if len(v) else None
            )
        output.append(record)
    return output


def _prepare(config):
    parent_nodes = np.unique(np.concatenate([grid_nodes(g, config.profile) for g in config.grids]))
    models = {}
    for name in config.models:
        if config.profile == "reference":
            truth = paper_kernel(name, parent_nodes)
            values, vectors = np.linalg.eigh(truth)
            root = vectors * np.sqrt(np.maximum(values, 0))
            models[name] = {
                "truth": truth,
                "sample_map": root,
                "parent": None,
                "metadata": {
                    "process": "generic centered Gaussian paper control",
                    "domain": [0.0, 1.0],
                },
            }
        else:
            parent = prepare_parent_model(
                name,
                config.reference_dimension,
                config.delta,
                config.quadrature_order,
                source_exponent=config.source_exponent,
                source_trace=config.source_trace,
                rotation_angle=config.rotation_angle,
            )
            values, vectors = np.linalg.eigh(parent.innovation_covariance)
            evaluation = parent.model.evaluate(parent_nodes)
            models[name] = {
                "truth": evaluation @ parent.innovation_covariance @ evaluation.T,
                "sample_map": evaluation @ (vectors * np.sqrt(np.maximum(values, 0))),
                "parent": parent,
                "metadata": {
                    **dict(parent.source_metadata),
                    "integration": dict(parent.numerical_checks),
                },
            }
    geometry = {}
    for name, model in models.items():
        for resolution in config.grids:
            nodes = grid_nodes(resolution, config.profile)
            ids = np.searchsorted(parent_nodes, nodes)
            weights = (
                np.ones(len(nodes)) / len(nodes)
                if config.profile == "reference"
                else quadrature_weights(nodes)
            )
            truth = model["truth"][np.ix_(ids, ids)]
            for m in config.covers:
                cover = paper_cover(m, domain=(0.0, 1.0 if config.profile == "reference" else 3.0))
                cliques = clique_indices(nodes, cover)
                mask = observed_mask(len(nodes), cliques)
                partial = np.where(mask, truth, np.nan)
                completed = complete_kernel(
                    partial,
                    cliques=cliques,
                    weights=weights,
                    support_rtol=config.support_rtol,
                    population=True,
                )
                geometry[(name, resolution, m)] = {
                    "nodes": nodes,
                    "indices": ids,
                    "weights": weights,
                    "truth": truth,
                    "cliques": cliques,
                    "mask": mask,
                    "canonical": completed.kernel,
                    "population_diagnostics": completed.diagnostics,
                    "exposures": _exposures(nodes, config.profile),
                }
    return parent_nodes, models, geometry


def _population(config, models):
    controls = population_controls()
    sensitivity = []
    if config.profile != "spde" or not config.population_checks:
        return {"controls": controls, "sensitivity": sensitivity}
    for name, model in models.items():
        parent = model["parent"]
        previous = prepare_parent_model(
            name,
            config.reference_dimension // 2,
            config.delta,
            config.quadrature_order,
            source_exponent=config.source_exponent,
            source_trace=config.source_trace,
            rotation_angle=config.rotation_angle,
        )
        resolutions = tuple(sorted(set((*config.grids, config.refinement_grid))))
        # A common set of physical evaluation targets permits actual grid refinement comparisons.
        anchor = grid_nodes(min(config.grids), "spde")
        for width in (0.05, 0.2, 0.4):
            cover = ((0.0, 1.5 + 1.5 * width), (1.5 - 1.5 * width, 3.0))
            baselines = {}
            for q in resolutions:
                nodes = np.unique(np.r_[grid_nodes(q, "spde"), anchor])
                weights = quadrature_weights(nodes)
                cliques = clique_indices(nodes, cover)
                mask = observed_mask(len(nodes), cliques)
                for forcing in (previous, parent):
                    E = forcing.model.evaluate(nodes)
                    truth = E @ forcing.innovation_covariance @ E.T
                    for cutoff in (1e-10, 1e-12, 1e-14):
                        result = complete_kernel(
                            np.where(mask, truth, np.nan),
                            cliques=cliques,
                            weights=weights,
                            support_rtol=cutoff,
                            population=True,
                        )
                        entry = {
                            "model": name,
                            "overlap_fraction": width,
                            "base_grid": q,
                            "actual_grid": len(nodes),
                            "forcing_modes": forcing.model.dimension,
                            "support_rtol": cutoff,
                            "available": result.kernel is not None,
                            "diagnostics": _compact(result.diagnostics),
                        }
                        if result.kernel is not None:
                            selected = np.searchsorted(nodes, anchor)
                            view = result.kernel[np.ix_(selected, selected)]
                            baselines[(q, forcing.model.dimension, cutoff)] = view
                        sensitivity.append(entry)
            reference = baselines.get((max(resolutions), parent.model.dimension, 1e-12))
            anchor_w = np.outer(quadrature_weights(anchor), quadrature_weights(anchor))
            for entry in sensitivity:
                if entry["model"] != name or entry["overlap_fraction"] != width:
                    continue
                view = baselines.get(
                    (entry["base_grid"], entry["forcing_modes"], entry["support_rtol"])
                )
                error = (
                    np.sqrt(
                        np.sum(anchor_w * (view - reference) ** 2)
                        / max(np.sum(anchor_w * reference**2), 1e-300)
                    )
                    if view is not None and reference is not None
                    else None
                )
                entry.update(
                    relative_weighted_reference_difference=error,
                    numerical_reference_unstable=error is None or error > 1e-4,
                    interpretation="sensitivity on common physical targets; agreement is not continuum convergence",
                )
    return {"controls": controls, "sensitivity": sensitivity}


def _ridge(values, assignments, geo, config, n, basis_cache, geometry_records=None):
    """Observed-validation selection; neither truth nor latent source is an input."""
    nodes, cliques = geo["nodes"], geo["cliques"]
    validation_start = max(config.training_counts)
    test_start = validation_start + config.validation_count
    records, fitted = [], {}
    validation_statistics = _array_statistics(
        values[validation_start:test_start],
        assignments[validation_start:test_start],
        cliques,
        geo["noise_variance"],
    )
    for j in config.candidate_dimensions:
        model = LaguerreTransport(j)
        E = model.evaluate(nodes)
        observations = tuple(
            Observation(
                (i + 1) * config.delta,
                values[i, cliques[group]],
                E[cliques[group]],
                geo["noise_variance"] * np.eye(len(cliques[group])),
            )
            for i, group in enumerate(assignments)
        )
        group_counts = np.bincount(assignments[:n], minlength=len(cliques))
        operators = [E[indices] for indices in cliques]
        key = (geo["base_grid"], len(cliques), j, tuple(group_counts))
        if key not in basis_cache:
            if len(basis_cache) >= 32:
                basis_cache.pop(next(iter(basis_cache)))
            gram = gram_operator(operators, weights=group_counts / n)
            basis_cache[key] = (
                float(np.linalg.eigvalsh(gram)[-1]),
                diagnose_design(operators, weights=group_counts / n),
            )
        norm, diagnostic = basis_cache[key]
        if geometry_records is not None:
            geometry_records[key] = diagnostic
        for eta in config.relative_regularizations:
            started = time.perf_counter()
            candidate = fit_candidate(
                observations[:n],
                FitSettings(config.trace_cap, config.numerical_error_budget),
                dimension=j,
                relative_regularization=eta,
                gram_operator_norm=norm,
                basis_id=model.basis_id,
            )
            kernel = E @ candidate.covariance @ E.T
            validation_score = kernel_quadratic_score_from_statistics(kernel, validation_statistics)
            estimate = candidate.estimate
            record = {
                "J": j,
                "basis_id": model.basis_id,
                "relative_regularization": eta,
                "regularization": eta * norm,
                "gram_operator_norm": norm,
                "validation_score": validation_score,
                "covariance_error_bound": estimate.covariance_error_bound,
                "objective_gap_bound": estimate.objective_gap_bound,
                "converged": estimate.converged,
                "iterations": estimate.iterations,
                "trace": estimate.trace,
                "runtime_seconds": time.perf_counter() - started,
                "numerical_gram_rank": diagnostic["rank"],
                "training_last_interval": n,
                "validation_first_interval": validation_start + 1,
                "validation_last_interval": test_start,
                "refit_on_validation": False,
            }
            records.append(record)
            fitted[(j, eta)] = (kernel, record, candidate.covariance)
    chosen = choose_candidate(records)
    return fitted[(chosen["J"], chosen["relative_regularization"])], records


def run_fragment_benchmark(config, output=None):
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as handle:
            config = FragmentBenchmarkConfig(**tomllib.load(handle))
    if not isinstance(config, FragmentBenchmarkConfig):
        raise TypeError("fragment benchmark requires FragmentBenchmarkConfig")
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
    from .fragment_reporting import write_fragment_report

    manifest = _manifest(config, _source_root())
    manifest["solver"] = {
        "ridge": {
            "name": "accelerated_projected_gradient",
            **SOLVER_OPTIONS,
            "numerical_error_budget": config.numerical_error_budget,
            "trace_cap": config.trace_cap,
        },
        "canonical": {
            "name": "weighted_separator_spectral_inverse",
            "support_rtol": config.support_rtol,
            "rank_candidates": config.ranks
            if config.profile == "spde"
            else "4 for finite_rank; 2 otherwise",
            "fve": 0.95 if config.include_fve else None,
            "rank_selection": "predeclared; never latent or observed-score tuned",
        },
        "projection": {"name": "weighted_full_kernel_PSD_projection", "H_trace_cap": False},
    }
    manifest.update(
        numerical_environment={
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        experiment="dense point fragments of centered common-covariance Gaussian curves",
        fitting_information="corrected observed products only; canonical ranks predeclared",
        selection_information="ridge only: later observed validation; no rank tuning",
        space={
            "measure": "equal 1/Q weights on [0,1]"
            if config.profile == "reference"
            else "physical trapezoidal weights on [0,3] years",
            "H_operator_diagnostics": "ridge only",
        },
        methods={
            "canonical": "weighted separator spectral inverse; raw and separate PSD projection",
            "ridge": "existing certified H-coordinate constrained solver",
        },
    )
    _json(output / "manifest.json", manifest)
    _json(output / "config.json", asdict(config))
    _json(output / "status.json", {"status": "running"})
    parent_nodes, models, geometry = _prepare(config)
    population = _population(config, models)
    _json(output / "population.json", population)
    _json(
        output / "geometry.json",
        [
            {
                "model": name,
                "base_grid": q,
                "cover": m,
                "actual_grid": len(g["nodes"]),
                "observed_fraction": float(g["mask"].mean()),
                "population_diagnostics": _compact(g["population_diagnostics"]),
                "nodes": g["nodes"],
                "weights": g["weights"],
                "cliques": g["cliques"],
            }
            for (name, q, m), g in geometry.items()
        ],
    )
    rows, risks, audits, candidates, selections, examples = [], [], [], [], [], []
    basis_cache, geometry_records = {}, {}
    test_start = max(config.training_counts) + config.validation_count
    for replication in range(config.replications):
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
                        test_statistics = _array_statistics(
                            values[test_start:], assigned[test_start:], cliques, variance
                        )
                        policy_ranks = (
                            (4 if name == "finite_rank" else 2,)
                            if config.profile == "reference"
                            else config.ranks
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
                            row_nodes = tuple(cliques[g] for g in assigned[:n])
                            statistics = partial_kernel_from_arrays(
                                tuple(values[i, ids] for i, ids in enumerate(row_nodes)),
                                row_nodes,
                                noise_variance=variance,
                                size=len(geo["nodes"]),
                            )
                            policies = [(f"rank{r}", {"rank": r}) for r in policy_ranks]
                            if config.include_fve:
                                policies.append(("fve95", {"fve": 0.95}))
                            fits = []
                            for policy, options in policies:
                                started = time.perf_counter()
                                fit = fit_fragment_kernel(
                                    statistics,
                                    cliques=cliques,
                                    weights=geo["weights"],
                                    support_rtol=config.support_rtol,
                                    **options,
                                )
                                runtime = time.perf_counter() - started
                                audits.append(
                                    {
                                        **common,
                                        "policy": policy,
                                        "diagnostics": _compact(fit.diagnostics),
                                        "pair_count_min": int(
                                            statistics.pair_counts[geo["mask"]].min()
                                        ),
                                        "parent_values_sha256": hashlib.sha256(
                                            values[:n].tobytes()
                                        ).hexdigest(),
                                        "assignment": assigned[:n] if replication == 0 else None,
                                    }
                                )
                                if fit.kernel is None:
                                    rows.extend(
                                        {
                                            **common,
                                            "method": "canonical",
                                            "policy": policy,
                                            "variant": variant,
                                            "available": False,
                                            "unavailable_reason": fit.diagnostics["reason"],
                                        }
                                        for variant in ("raw", "psd")
                                    )
                                    continue
                                fits.append(("canonical", policy, "raw", fit.kernel, runtime, {}))
                                started = time.perf_counter()
                                projected = weighted_psd_projection(
                                    fit.kernel, weights=geo["weights"], observed_mask=geo["mask"]
                                )
                                fits.append(
                                    (
                                        "canonical",
                                        policy,
                                        "psd",
                                        projected.kernel,
                                        runtime + time.perf_counter() - started,
                                        {"projection": _compact(projected.diagnostics)},
                                    )
                                )
                            if config.ridge_comparison and q == 81 and n == 512:
                                # Only observed data and numerical settings cross the selection boundary.
                                safe_geo = {k: geo[k] for k in ("nodes", "cliques")}
                                safe_geo.update(noise_variance=variance, base_grid=q)
                                (kernel, record, operator), grid_records = _ridge(
                                    values[:test_start],
                                    assigned[:test_start],
                                    safe_geo,
                                    config,
                                    n,
                                    basis_cache,
                                    geometry_records,
                                )
                                candidates.extend({**common, **r} for r in grid_records)
                                selections.append({**common, **record})
                                parent = model["parent"]
                                operator_error = -parent.innovation_covariance.copy()
                                operator_error[: len(operator), : len(operator)] += operator
                                fits.append(
                                    (
                                        "ridge",
                                        "validation_selected",
                                        "psd",
                                        kernel,
                                        record["runtime_seconds"],
                                        {
                                            "ridge": record,
                                            "operator_hs_error": float(
                                                np.linalg.norm(operator_error)
                                            ),
                                            "operator_hs_relative_error": float(
                                                np.linalg.norm(operator_error)
                                                / np.linalg.norm(parent.innovation_covariance)
                                            ),
                                            "operator_norm_error": float(
                                                np.max(np.abs(np.linalg.eigvalsh(operator_error)))
                                            ),
                                            "operator": operator,
                                        },
                                    )
                                )
                            for method, policy, variant, kernel, runtime, extra in fits:
                                record = {
                                    **common,
                                    "method": method,
                                    "policy": policy,
                                    "variant": variant,
                                    "available": True,
                                    "runtime_seconds": runtime,
                                    "test_observed_score": kernel_quadratic_score_from_statistics(
                                        kernel, test_statistics
                                    ),
                                    **_metrics(
                                        kernel,
                                        geo["truth"],
                                        geo["canonical"],
                                        geo["weights"],
                                        geo["mask"],
                                    ),
                                }
                                if "operator_hs_error" in extra:
                                    record.update(
                                        operator_hs_error=extra["operator_hs_error"],
                                        operator_hs_relative_error=extra[
                                            "operator_hs_relative_error"
                                        ],
                                        operator_norm_error=extra["operator_norm_error"],
                                        **extra["ridge"],
                                    )
                                if "projection" in extra:
                                    audits.append(
                                        {
                                            **common,
                                            "method": method,
                                            "policy": policy,
                                            "variant": variant,
                                            "projection": extra["projection"],
                                        }
                                    )
                                rows.append(record)
                                for target, exposure in geo["exposures"].items():
                                    estimate = float(exposure @ kernel @ exposure)
                                    truth = float(exposure @ geo["truth"] @ exposure)
                                    risk_record = {
                                        **common,
                                        "method": method,
                                        "policy": policy,
                                        "variant": variant,
                                        "target": target,
                                        "estimate": estimate,
                                        "truth": truth,
                                        "risk_error": estimate - truth,
                                        "risk_squared_error": (estimate - truth) ** 2,
                                        "pair_coverage_identified": bool(
                                            np.all(
                                                geo["mask"][np.outer(exposure != 0, exposure != 0)]
                                            )
                                        ),
                                        "identification_range_available": False,
                                        "identification_range_scope": "exact ranges supplied for two-block population controls only",
                                        **exposure_score(
                                            estimate,
                                            exposure,
                                            values[test_start:],
                                            assigned[test_start:],
                                            cliques,
                                            variance,
                                        ),
                                    }
                                    if variant == "psd" and method == "canonical":
                                        raw_kernel = next(
                                            f[3] for f in fits if f[:3] == (method, policy, "raw")
                                        )
                                        risk_record["psd_projection_risk_change"] = (
                                            estimate - float(exposure @ raw_kernel @ exposure)
                                        )
                                    risks.append(risk_record)
                                    if (
                                        target == "grid_delivery_spread"
                                        and model["parent"] is not None
                                    ):
                                        parent = model["parent"]
                                        a = functional_coefficients(
                                            [fixed_exposures()["delivery_spread"]], parent.model
                                        )[0]
                                        continuous_truth = float(
                                            a @ parent.innovation_covariance @ a
                                        )
                                        continuous_estimate = estimate
                                        if "operator" in extra:
                                            j = len(extra["operator"])
                                            continuous_estimate = float(
                                                a[:j] @ extra["operator"] @ a[:j]
                                            )
                                        risks.append(
                                            {
                                                **risk_record,
                                                "target": "continuous_delivery_spread",
                                                "grid_pair_coverage_identified": risk_record[
                                                    "pair_coverage_identified"
                                                ],
                                                "pair_coverage_identified": False,
                                                "estimate": continuous_estimate,
                                                "truth": continuous_truth,
                                                "risk_error": continuous_estimate
                                                - continuous_truth,
                                                "risk_squared_error": (
                                                    continuous_estimate - continuous_truth
                                                )
                                                ** 2,
                                                "grid_truth_minus_continuous_truth": truth
                                                - continuous_truth,
                                                "risk_score": None,
                                                "scoring_available": False,
                                                "available_test_dates": 0,
                                                "unavailable_reason": "finite point fragments do not measure a continuous delivery integral",
                                            }
                                        )
                                if (
                                    replication == 0
                                    and q == config.grids[len(config.grids) // 2]
                                    and n
                                    == config.training_counts[len(config.training_counts) // 2]
                                    and policy in ("rank2", "rank4", "validation_selected")
                                ):
                                    examples.append(
                                        {
                                            **common,
                                            "method": method,
                                            "policy": policy,
                                            "variant": variant,
                                            "nodes": geo["nodes"],
                                            "kernel": kernel,
                                            "truth": geo["truth"],
                                            "canonical": geo["canonical"],
                                            "mask": geo["mask"],
                                        }
                                    )
        _json(
            output / "status.json", {"status": "running", "completed_replications": replication + 1}
        )
    fields = ("model", "noise_variance", "base_grid", "cover", "N", "method", "policy", "variant")
    metrics = (
        "test_observed_score",
        "observed_weighted_error",
        "missing_weighted_error",
        "complete_weighted_error",
        "integrated_squared_error",
        "observed_integrated_squared_error",
        "missing_integrated_squared_error",
        "canonical_weighted_error",
        "canonical_generating_discrepancy",
        "runtime_seconds",
        "minimum_weighted_eigenvalue",
        "operator_hs_error",
        "operator_hs_relative_error",
        "operator_norm_error",
    )
    summary = _summary(rows, metrics, fields)
    risk_summary = _summary(
        risks,
        (
            "risk_score",
            "risk_error",
            "risk_squared_error",
            "available_test_dates",
            "psd_projection_risk_change",
        ),
        (*fields, "target"),
    )
    paired = []
    groups = defaultdict(dict)
    for row in rows:
        if row["available"]:
            groups[tuple(row[f] for f in fields[:5])][
                (row["replication"], row["method"], row["policy"], row["variant"])
            ] = row
    for key, group in groups.items():
        model = key[0]
        primary = "rank4" if model == "finite_rank" else "rank2"
        for choice in sorted(set((k[1], k[2], k[3]) for k in group)):
            diffs = []
            for rep in range(config.replications):
                current, base = (
                    group.get((rep, *choice)),
                    group.get((rep, "canonical", primary, "raw")),
                )
                if current is not None and base is not None:
                    diffs.append(current["test_observed_score"] - base["test_observed_score"])
            paired.append(
                {
                    **dict(zip(fields[:5], key, strict=True)),
                    "method": choice[0],
                    "policy": choice[1],
                    "variant": choice[2],
                    "baseline": primary + "_raw",
                    "paired_replications": len(diffs),
                    "score_difference_mean": float(np.mean(diffs)) if diffs else None,
                    "score_difference_mcse": float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
                    if len(diffs) > 1
                    else None,
                }
            )
    feasible = all(
        r["minimum_weighted_eigenvalue"] >= -1e-8
        for r in rows
        if r["available"] and r["variant"] == "psd"
    )
    converged = all(
        r["converged"] and r["covariance_error_bound"] <= config.numerical_error_budget
        for r in candidates
    )
    primary_present = all(
        any(
            r["available"]
            and r["model"] == name
            and r["method"] == "canonical"
            and r["policy"] == ("rank4" if name == "finite_rank" else "rank2")
            for r in rows
        )
        for name in config.models
    )
    controls = population["controls"]
    controlled_recovery = all(
        r["observed_max_abs_error"] < 1e-10
        and (r["model"] != "finite_rank" or r["missing_ise"] < 1e-16)
        for r in controls["paper"]
    )
    controlled_recovery = (
        controlled_recovery and controls["noncanonical"]["observed_difference_max"] < 1e-12
    )
    controlled_recovery = (
        controlled_recovery and controls["truncation"]["rank1_min_eigenvalue"] < -0.4
    )
    results = {
        "acceptance": {
            "passed": bool(feasible and converged and primary_present and controlled_recovery),
            "psd_outputs_feasible": feasible,
            "ridge_certificates_within_budget": converged,
            "raw_indefiniteness_is_allowed": True,
            "primary_manual_policy_present_per_model": primary_present,
            "controlled_population_recovery": controlled_recovery,
        },
        "summary": summary,
        "risk_summary": risk_summary,
        "paired_differences": paired,
        "available_fits": sum(r["available"] for r in rows),
        "unavailable_fits": sum(not r["available"] for r in rows),
        "raw_indefinite_fits": sum(
            r["available"] and r["variant"] == "raw" and r["negative_eigenvalues"] > 0 for r in rows
        ),
        "models": {name: data["metadata"] for name, data in models.items()},
    }
    for filename, data in (
        ("results.json", results),
        ("completion_audit.json", audits),
        ("representative_kernels.json", examples),
        ("ridge_candidates.json", candidates),
        ("ridge_selection.json", selections),
        (
            "ridge_geometry.json",
            [
                {
                    "base_grid": key[0],
                    "cover": key[1],
                    "J": key[2],
                    "fragment_counts": key[3],
                    "diagnostics": {k: v for k, v in diagnostic.items() if k != "gram"},
                }
                for key, diagnostic in geometry_records.items()
            ],
        ),
    ):
        _json(output / filename, data)
    _csv(output / "estimates.csv", rows)
    _csv(output / "risks.csv", risks)
    _csv(output / "summary.csv", summary)
    _csv(output / "risk_summary.csv", risk_summary)
    _csv(output / "paired.csv", paired)
    write_fragment_report(config, output, results, examples, population)
    _json(
        output / "status.json",
        {"status": "complete" if results["acceptance"]["passed"] else "failed"},
    )
    if not results["acceptance"]["passed"]:
        raise RuntimeError("fragment benchmark numerical acceptance failed; inspect results.json")
    return output
