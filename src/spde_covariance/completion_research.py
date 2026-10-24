"""Controlled regularization ablations and joint fixed-model refinement.

All fitting/selection receives observed statistics only. Latent references are
confined to evaluation. A parent sample is generated before grid/method views.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .completion import complete_kernel, weighted_psd_projection
from .completion_research_config import CompletionResearchConfig
from .completion_strategy import (
    CompletionStabilityGuard,
    CompletionStrategySettings,
    fit_completion_strategy,
    select_completion_strategy,
    split_completion_stability,
)
from .fragment_benchmark import (
    _array_statistics,
    _compact,
    _csv,
    _json,
    _metrics,
    _prepare,
    _summary,
    balanced_assignments,
    clique_indices,
    exposure_score,
    grid_nodes,
    observed_mask,
    quadrature_weights,
)
from .fragment_reference import paper_cover
from .laguerre import PowerLawSource
from .observation_statistics import kernel_quadratic_score_from_statistics
from .partial_design import fixed_exposures, functional_coefficients, prepare_parent_model
from .provenance import _manifest, _source_root

_WORKER = None


def strategy_grid(config):
    """Declared factorial, with equal partial fits reused across completion rules."""
    base = CompletionStrategySettings(
        numerical_error_budget=config.numerical_error_budget,
        rho=config.rho,
        max_iterations=config.max_iterations,
        floor_gamma=config.floor_gamma,
        constant_floor=config.constant_floor,
        support_rtol=config.support_rtol,
    )
    choices = {"pairwise_rank2": replace(base, completion_rule="rank2")}
    if config.experiment == "refinement":
        for floor in ("constant", "resolution"):
            choices[f"coherent_{floor}_support"] = replace(
                base,
                partial_method="coherent",
                floor_policy=floor,
                relative_regularization=config.relative_regularization,
            )
        return choices
    choices["pairwise_support"] = base
    for rule in ("rank2", "support"):
        choices[f"ridge_{rule}"] = replace(
            base,
            partial_method="ridge",
            completion_rule=rule,
            relative_regularization=config.relative_regularization,
        )
    for eta in (0.0, config.relative_regularization):
        for floor in ("zero", "constant", "resolution"):
            for rule in ("rank2", "support"):
                choices[f"coherent_eta{eta:g}_{floor}_{rule}"] = replace(
                    base,
                    partial_method="coherent",
                    floor_policy=floor,
                    relative_regularization=eta,
                    completion_rule=rule,
                )
    return choices


def fit_strategy_grid(statistics, *, cliques, weights, strategies):
    """Reuse exactly identical upstream objectives; rank is invisible to this fit."""
    from .completion_strategy import CompletionStrategyResult

    cache, results = {}, {}
    for name, settings in strategies.items():
        key = (
            settings.partial_method,
            settings.relative_regularization,
            settings.eigenvalue_floor(len(weights)),
        )
        start = time.perf_counter()
        if key not in cache:
            result = fit_completion_strategy(
                statistics, cliques=cliques, weights=weights, settings=settings
            )
            cache[key] = result
            reused = False
        else:
            upstream = cache[key]
            if not upstream.converged:
                result = upstream
            else:
                completion = complete_kernel(
                    upstream.partial_kernel,
                    cliques=cliques,
                    weights=weights,
                    rank=2 if settings.completion_rule == "rank2" else None,
                    support_rtol=settings.support_rtol,
                    population=settings.partial_method == "coherent"
                    and settings.completion_rule == "support",
                )
                from .completion_strategy import _conditioning

                diagnostic = {
                    **dict(upstream.diagnostics),
                    "completion": completion.diagnostics,
                    **_conditioning(completion.diagnostics),
                    "settings": asdict(settings),
                }
                result = CompletionStrategyResult(
                    completion.available,
                    completion.diagnostics["reason"],
                    upstream.partial_kernel,
                    completion.kernel,
                    diagnostic,
                )
            reused = True
        results[name] = (result, time.perf_counter() - start, reused)
    return results


def separator_perturbation_bound(a, s, b, a_prime, s_prime, b_prime):
    """Finite SPD inverse perturbation bound for D=A S^-1 B, Frobenius norm.

    Inputs already use sqrt(quadrature) block coordinates. A bound is unavailable
    if the separator perturbation reaches its positive spectral gap. It is not
    a bound for hard rank changes or a statistical/continuum theorem.
    """
    a, s, b, ap, sp, bp = map(np.asarray, (a, s, b, a_prime, s_prime, b_prime))
    gap = float(np.linalg.eigvalsh(s)[0])
    eps_s = float(np.linalg.norm(sp - s, 2))
    if gap <= eps_s or gap <= 0:
        return {
            "available": False,
            "gap": gap,
            "separator_perturbation": eps_s,
            "bound": None,
            "actual": None,
        }
    eps_a = float(np.linalg.norm(ap - a))
    eps_b = float(np.linalg.norm(bp - b))
    bound = (
        eps_a / (gap - eps_s) * (np.linalg.norm(b, 2) + np.linalg.norm(bp - b, 2))
        + np.linalg.norm(a, 2) * eps_s / (gap * (gap - eps_s)) * (np.linalg.norm(b) + eps_b)
        + np.linalg.norm(a, 2) / gap * eps_b
    )
    actual = np.linalg.norm(ap @ np.linalg.solve(sp, bp) - a @ np.linalg.solve(s, b))
    return {
        "available": True,
        "gap": gap,
        "separator_perturbation": eps_s,
        "bound": float(bound),
        "actual": float(actual),
    }


def _population(config, prepared):
    """Separate forcing, output projection and physical-grid reference changes."""
    _, models, _ = prepared
    rows, perturbations = [], []
    source = PowerLawSource(config.source_exponent, config.source_trace)
    anchor = grid_nodes(min(config.population_grids), "spde")
    for name, data in models.items():
        parent = data["parent"]
        evaluation = parent.model.evaluate(anchor)
        truth = evaluation @ parent.innovation_covariance @ evaluation.T
        weights = quadrature_weights(anchor)

        def norm(x, w=weights):
            return float(np.linalg.norm(np.sqrt(w)[:, None] * x * np.sqrt(w)[None, :]))

        forcing_parents = {}
        for m in config.forcing_dimensions:
            finite = prepare_parent_model(
                name,
                m,
                config.delta,
                config.quadrature_order,
                source_exponent=config.source_exponent,
                source_trace=config.source_trace,
                rotation_angle=config.rotation_angle,
            )
            ef = finite.model.evaluate(anchor)
            forcing_parents[m] = finite
            difference = ef @ finite.innovation_covariance @ ef.T - truth
            rows.append(
                {
                    "record_type": "forcing",
                    "model": name,
                    "M": m,
                    "forcing_error": norm(difference),
                    "tail_bound": source.innovation_tail_bound(m, config.delta)
                    * np.sum(weights * (2 - np.exp(-anchor))),
                    "reference_available": True,
                }
            )
        for j in config.output_dimensions:
            ej = evaluation[:, :j]
            projected = ej @ parent.innovation_covariance[:j, :j] @ ej.T
            rows.append(
                {
                    "record_type": "representation",
                    "model": name,
                    "J": j,
                    "representation_error": norm(projected - truth),
                    "reference_available": True,
                }
            )
        references = {}
        # Descending order establishes the finest available target first.
        for q in reversed(config.population_grids):
            nodes = np.unique(np.r_[grid_nodes(q, "spde"), anchor])
            w = quadrature_weights(nodes)
            e = parent.model.evaluate(nodes)
            kernel = e @ parent.innovation_covariance @ e.T
            previous = forcing_parents[min(config.forcing_dimensions)]
            ep = previous.model.evaluate(nodes)
            previous_kernel = ep @ previous.innovation_covariance @ ep.T
            ids = np.searchsorted(nodes, anchor)
            for cover in config.covers:
                cliques = clique_indices(nodes, paper_cover(cover, domain=(0, 3)))
                mask = observed_mask(len(nodes), cliques)
                left = np.setdiff1d(cliques[0], cliques[1])
                separator = np.intersect1d(cliques[0], cliques[1])
                right = np.setdiff1d(cliques[1], cliques[0])
                weighted = np.sqrt(w)[:, None] * kernel * np.sqrt(w)[None, :]
                weighted_previous = np.sqrt(w)[:, None] * previous_kernel * np.sqrt(w)[None, :]
                for floor_policy, floor in (
                    ("constant", config.constant_floor),
                    ("resolution", config.floor_gamma / len(w) ** 2),
                ):

                    def blocks(matrix):
                        return (
                            matrix[np.ix_(left, separator)],
                            matrix[np.ix_(separator, separator)] + floor * np.eye(len(separator)),
                            matrix[np.ix_(separator, right)],
                        )

                    audit = separator_perturbation_bound(
                        *blocks(weighted), *blocks(weighted_previous)
                    )
                    perturbations.append(
                        {
                            "record_type": "population_first_join",
                            "model": name,
                            "base_grid": q,
                            "cover": cover,
                            "M": previous.model.dimension,
                            "reference_M": config.reference_dimension,
                            "floor_policy": floor_policy,
                            "weighted_floor": floor,
                            **audit,
                        }
                    )
                result = complete_kernel(
                    np.where(mask, kernel, np.nan),
                    cliques=cliques,
                    weights=w,
                    population=True,
                    support_rtol=config.support_rtol,
                )
                available = result.available
                if available and cover not in references:
                    references[cover] = (q, result.kernel[np.ix_(ids, ids)])
                joins = result.diagnostics.get("joins", ())
                rows.append(
                    {
                        "record_type": "grid",
                        "model": name,
                        "base_grid": q,
                        "cover": cover,
                        "reference_available": available,
                        "canonical_generating_discrepancy": norm(
                            result.kernel[np.ix_(ids, ids)] - truth
                        )
                        if available
                        else None,
                        "canonical_reference_error": norm(
                            result.kernel[np.ix_(ids, ids)] - references[cover][1]
                        )
                        if available
                        else None,
                        "canonical_reference_grid": references[cover][0]
                        if cover in references
                        else None,
                        "canonical_reference_relative_error": norm(
                            result.kernel[np.ix_(ids, ids)] - references[cover][1]
                        )
                        / max(norm(references[cover][1]), np.finfo(float).tiny)
                        if available
                        else None,
                        "inverse_norm": max((j.get("inverse_norm", 0) for j in joins), default=0),
                        "weighted_floor_trace_constant": len(nodes) * config.constant_floor,
                        "weighted_floor_trace_resolution": config.floor_gamma / len(nodes),
                        "diagnostics": _compact(result.diagnostics),
                    }
                )
        # A controlled SPD separator audit quantifies the inverse amplification.
        for gap in (1.0, 1e-2, 1e-4):
            a, b, s = np.ones((1, 2)) / 3, np.ones((2, 1)) / 4, np.diag([1, gap])
            eps = gap / 10
            audit = separator_perturbation_bound(a, s, b, a + eps, s + eps * np.eye(2), b - eps)
            perturbations.append({"record_type": "control", "model": name, **audit})
    return rows, perturbations


def _initialize(config):
    global _WORKER
    _WORKER = (config, _prepare(config))


def _worker(rep):
    return _replication(rep, *_WORKER)


def _replication(rep, config, prepared):
    _, models, geometry = prepared
    rows, risks, audit, stability_rows = [], [], [], []
    validation_start = max(config.training_counts)
    test_start = validation_start + config.validation_count
    assignments = {
        m: balanced_assignments(config.total_count, m, config.seed + 7919 * rep + m)
        for m in config.covers
    }
    strategies = strategy_grid(config)
    for name, model in models.items():
        rng = np.random.default_rng(config.seed + 1009 * rep)
        values = rng.standard_normal((config.total_count, model["sample_map"].shape[1]))
        values = values @ model["sample_map"].T
        values += np.sqrt(config.noise_variance) * rng.standard_normal(values.shape)
        for q in config.grids:
            for m in config.covers:
                geo = geometry[name, q, m]
                x, groups = values[:, geo["indices"]], assignments[m]
                cliques, w = geo["cliques"], geo["weights"]
                validation = _array_statistics(
                    x[validation_start:test_start],
                    groups[validation_start:test_start],
                    cliques,
                    config.noise_variance,
                )
                test = _array_statistics(
                    x[test_start:], groups[test_start:], cliques, config.noise_variance
                )
                for n in config.training_counts:
                    common = {
                        "replication": rep,
                        "model": name,
                        "base_grid": q,
                        "actual_grid": len(w),
                        "cover": m,
                        "N": n,
                    }
                    train = _array_statistics(x[:n], groups[:n], cliques, config.noise_variance)
                    fitted = fit_strategy_grid(
                        train, cliques=cliques, weights=w, strategies=strategies
                    )
                    digest = hashlib.sha256(x[:n].tobytes()).hexdigest()
                    kernels = {}
                    for policy, (fit, runtime, reused) in fitted.items():
                        s = strategies[policy]
                        kernels[policy] = (
                            fit.kernel,
                            runtime,
                            fit.available,
                            fit.reason,
                            fit.diagnostics,
                        )
                        audit.append(
                            {
                                **common,
                                "policy": policy,
                                "available": fit.available,
                                "reason": fit.reason,
                                "training_values_sha256": digest,
                                "coherent": s.partial_method == "coherent",
                                "reused_partial": reused,
                                "settings": asdict(s),
                                "diagnostics": _compact(fit.diagnostics),
                            }
                        )
                    raw, runtime, available, reason, diag = kernels["pairwise_rank2"]
                    projection_start = time.perf_counter()
                    projected = (
                        weighted_psd_projection(raw, weights=w).kernel if available else None
                    )
                    projection_seconds = time.perf_counter() - projection_start
                    kernels["pairwise_rank2_psd"] = (
                        projected,
                        runtime + projection_seconds,
                        available,
                        reason,
                        diag,
                    )
                    if (
                        config.stability
                        and q == min(config.grids)
                        and n == max(config.training_counts)
                    ):
                        options, split_results = {}, {}
                        for eta in (0.0001, 0.001, 0.01):
                            s = replace(
                                next(iter(strategies.values())),
                                partial_method="coherent",
                                completion_rule="support",
                                floor_policy="resolution",
                                relative_regularization=eta,
                            )
                            label = f"eta{eta:g}"
                            options[label] = fit_completion_strategy(
                                train, cliques=cliques, weights=w, settings=s
                            )
                            split_results[label] = split_completion_stability(
                                tuple(x[i, cliques[g]] for i, g in enumerate(groups[:n])),
                                tuple(cliques[g] for g in groups[:n]),
                                size=len(w),
                                cliques=cliques,
                                weights=w,
                                settings=s,
                                noise_variance=config.noise_variance,
                                seed=config.seed + rep,
                            )
                        declared_complete = all(f.available for f in options.values())
                        for guarded in (False, True):
                            guard = (
                                CompletionStabilityGuard(
                                    max_relative_difference=config.stability_relative_limit,
                                    max_condition_number=config.stability_condition_limit,
                                )
                                if guarded
                                else None
                            )
                            selected = select_completion_strategy(
                                options, validation, stability=split_results, guard=guard
                            )
                            good = declared_complete and selected.available
                            label = "stability_guarded" if guarded else "validation_only"
                            k = selected.selected.kernel if good else None
                            kernels[label] = (
                                k,
                                None,
                                good,
                                selected.reason
                                if declared_complete
                                else "declared candidate unavailable",
                                selected.selected.diagnostics if good else {},
                            )
                            stability_rows.append(
                                {
                                    **common,
                                    "policy": label,
                                    "available": good,
                                    "selected": selected.selected_name if good else None,
                                    "selection": _compact(selected.diagnostics),
                                    "candidates": {
                                        key: {
                                            "available": f.available,
                                            "reason": f.reason,
                                            "diagnostics": _compact(f.diagnostics),
                                        }
                                        for key, f in options.items()
                                    },
                                    "split_diagnostics": {
                                        key: {
                                            "available": f.available,
                                            "reason": f.reason,
                                            "diagnostics": _compact(f.diagnostics),
                                        }
                                        for key, f in split_results.items()
                                    },
                                }
                            )
                    for policy, (kernel, runtime, available, reason, diag) in kernels.items():
                        row = {
                            **common,
                            "policy": policy,
                            "available": available,
                            "reason": reason,
                            "runtime_seconds": runtime,
                            "inverse_norm": diag.get("maximum_inverse_norm"),
                            "floor_trace": len(w) * diag.get("eigenvalue_floor", 0),
                            "observed_entry_change": None,
                            "test_observed_score": None,
                        }
                        if available:
                            row.update(
                                _metrics(kernel, geo["truth"], geo["canonical"], w, geo["mask"])
                            )
                            row["test_observed_score"] = kernel_quadratic_score_from_statistics(
                                kernel, test
                            )
                            row["observed_entry_change"] = float(
                                np.linalg.norm(
                                    np.where(geo["mask"], kernel - train.partial_kernel, 0)
                                )
                            )
                        rows.append(row)
                        for target, exposure in geo["exposures"].items():
                            estimate = float(exposure @ kernel @ exposure) if available else None
                            true_risk = float(exposure @ geo["truth"] @ exposure)
                            risk = {
                                **common,
                                "policy": policy,
                                "target": target,
                                "available": available,
                                "estimate": estimate,
                                "truth": true_risk,
                                "risk_error": abs(estimate - true_risk) if available else None,
                            }
                            if target == "grid_delivery_spread":
                                functional = functional_coefficients(
                                    (fixed_exposures()["delivery_spread"],), model["parent"].model
                                )[0]
                                continuous = float(
                                    functional @ model["parent"].innovation_covariance @ functional
                                )
                                risk.update(
                                    continuous_delivery_truth=continuous,
                                    grid_quadrature_truth=true_risk,
                                    delivery_quadrature_error=abs(true_risk - continuous),
                                    continuous_delivery_error=abs(estimate - continuous)
                                    if available
                                    else None,
                                    continuous_delivery_scoring_available=False,
                                    estimand="grid-quadrature approximation to continuous delivery risk",
                                )
                            if available:
                                risk.update(
                                    exposure_score(
                                        estimate,
                                        exposure,
                                        x[test_start:],
                                        groups[test_start:],
                                        cliques,
                                        config.noise_variance,
                                    )
                                )
                            risks.append(risk)
    return {
        "replication": rep,
        "rows": rows,
        "risks": risks,
        "audit": audit,
        "stability": stability_rows,
    }


def run_completion_research(config, output=None):
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as handle:
            config = CompletionResearchConfig(**tomllib.load(handle))
    if not isinstance(config, CompletionResearchConfig):
        raise TypeError("requires CompletionResearchConfig")
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
    from .completion_research_reporting import write_completion_research_report
    from .completion_source_study import run_source_class_study

    manifest = _manifest(config, _source_root())
    manifest.update(
        experiment="fixed-model completion ablations/refinement",
        solver={
            "strategies": {name: asdict(s) for name, s in strategy_grid(config).items()},
            "selection": "observed validation; declared candidate failures retained",
            "stability_guard": {
                "relative_limit": config.stability_relative_limit,
                "condition_limit": config.stability_condition_limit,
            },
        },
        execution={
            "workers": config.workers,
            "start_method": "spawn",
            "aggregation": "replication",
        },
    )
    _json(output / "manifest.json", manifest)
    _json(output / "config.json", asdict(config))
    _json(output / "status.json", {"status": "running"})
    prepared = _prepare(config)
    population, perturbations = _population(config, prepared)
    _json(output / "population.json", population)
    _json(output / "separator_perturbations.json", perturbations)
    source_rows, controls = run_source_class_study(config)
    _json(output / "source_classes.json", source_rows)
    _json(output / "source_controls.json", controls)
    records = {}
    manifest_hash = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()

    def retain(record):
        record["parent_manifest_sha256"] = manifest_hash
        rep = record["replication"]
        records[rep] = record
        _json(output / f"replication-{rep:04d}.json", record)
        _json(output / "status.json", {"status": "running", "completed_replications": len(records)})

    if config.workers == 1:
        for rep in range(config.replications):
            retain(_replication(rep, config, prepared))
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(config,),
        ) as executor:
            futures = [executor.submit(_worker, rep) for rep in range(config.replications)]
            for future in as_completed(futures):
                retain(future.result())
    rows, risks, audits, stability = [], [], [], []
    for rep in sorted(records):
        record = records[rep]
        rows.extend(record["rows"])
        risks.extend(record["risks"])
        audits.extend(record["audit"])
        stability.extend(record["stability"])
    fields = ("model", "base_grid", "cover", "N", "policy")
    metrics = (
        "test_observed_score",
        "complete_weighted_error",
        "missing_weighted_error",
        "canonical_weighted_error",
        "canonical_generating_discrepancy",
        "observed_entry_change",
        "minimum_weighted_eigenvalue",
        "runtime_seconds",
        "inverse_norm",
        "floor_trace",
    )
    summary = _summary(rows, metrics, fields)
    lookup = {
        (r["replication"], r["model"], r["base_grid"], r["cover"], r["N"]): r
        for r in rows
        if r["policy"] == "pairwise_rank2"
    }
    differences = []
    for row in rows:
        baseline = lookup[
            row["replication"], row["model"], row["base_grid"], row["cover"], row["N"]
        ]
        differences.append(
            {
                **{f: row[f] for f in (*fields, "replication")},
                "available": row["available"] and baseline["available"],
                **{
                    m: row.get(m) - baseline.get(m)
                    if row.get(m) is not None and baseline.get(m) is not None
                    else None
                    for m in metrics[:3]
                },
            }
        )
    coherent = [a for a in audits if a["coherent"] and not a["reused_partial"]]
    # Zero-floor ablations may expose numerical incompatibility. Every failure
    # is retained; successful fits alone receive a certificate claim.
    failures = [a for a in coherent if not a["diagnostics"]["converged"]]
    primary = [r for r in rows if "resolution_support" in r["policy"] and r["available"]]
    declared_primary = [
        r
        for r in rows
        if r["policy"]
        in (
            "coherent_resolution_support",
            f"coherent_eta{config.relative_regularization:g}_resolution_support",
        )
    ]
    expected_rows = (
        config.replications
        * len(config.models)
        * len(config.covers)
        * (
            len(config.grids) * len(config.training_counts) * (len(strategy_grid(config)) + 1)
            + (2 if config.stability else 0)
        )
    )
    acceptance = {
        "all_declared_cases_recorded": len(records) == config.replications,
        "all_declared_estimates_recorded": len(rows) == expected_rows,
        "declared_primary_available": bool(declared_primary)
        and all(r["available"] for r in declared_primary),
        "available_primary_outputs_psd": bool(primary)
        and all(r["minimum_weighted_eigenvalue"] >= -1e-8 for r in primary),
        "available_certificates_within_budget": all(
            a["diagnostics"]["partial_error_bound"] <= config.numerical_error_budget
            for a in coherent
            if a["diagnostics"]["converged"]
        ),
        "finite_separator_bound_controls": all(
            p["available"] and p["actual"] <= p["bound"]
            for p in perturbations
            if p["record_type"] == "control"
        ),
        "available_population_separator_bounds": all(
            p["actual"] <= p["bound"] for p in perturbations if p["available"]
        ),
        "source_controls": controls["passed"],
        "finite_numeric_outputs": all(
            np.isfinite(r["complete_weighted_error"]) for r in rows if r["available"]
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    results = {
        "acceptance": acceptance,
        "summary": summary,
        "paired_differences": _summary(differences, metrics[:3], fields),
        "stability_summary": stability,
        "separator_perturbations": perturbations,
        "source_controls": controls,
        "unavailable_count": sum(not r["available"] for r in rows),
        "coherent_fit_count": len(coherent),
        "failed_certificate_count": len(failures),
        "risk_summary": _summary(risks, ("risk_error", "risk_score"), (*fields, "target")),
        "scope": "finite numerical evidence for one fixed infinite-rank source family; no continuum theorem",
    }
    _json(output / "results.json", results)
    _json(output / "fit_diagnostics.json", audits)
    _json(output / "stability.json", stability)
    for filename, data in (
        ("estimates.csv", rows),
        ("risks.csv", risks),
        ("summary.csv", summary),
        ("paired.csv", results["paired_differences"]),
    ):
        _csv(output / filename, data)
    write_completion_research_report(config, output, results, population, source_rows)
    _json(output / "status.json", {"status": "complete" if acceptance["passed"] else "failed"})
    if not acceptance["passed"]:
        raise RuntimeError("research acceptance failed; inspect results.json")
    return output
