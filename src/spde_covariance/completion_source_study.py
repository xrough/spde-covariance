"""Population-only source-support comparisons at fixed output coordinates.

The source optimizer sees a target grid kernel, never a latent source as an
initialization. Increasing forcing support uses only embedded fitted witnesses.
Fixed generating-source tails and numerical-map comparisons remain separate.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from .admissibility import diagnose_constant_source_admissibility
from .completion import complete_kernel
from .fragment_benchmark import clique_indices, grid_nodes, observed_mask, quadrature_weights
from .fragment_reference import paper_cover
from .geometry import smat, svec
from .laguerre import LaguerreTransport, PowerLawSource
from .partial_design import prepare_parent_model
from .source_classes import (
    embed_source_covariance,
    evaluate_source_witness,
    fixed_source_tail_allowance,
    integration_map_comparison_allowance,
    projected_covariance_integration_matrix,
    projected_integration_diagnostics,
)


def _source_support_controls():
    """A declared three-mode source excluded by the matched two-mode cone."""
    vector = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)
    source = np.outer(vector, vector)
    target = smat(projected_covariance_integration_matrix(3, 2, 1.0) @ svec(source))
    records = []
    for dimension in (2, 3, 4):
        matrix = projected_covariance_integration_matrix(dimension, 2, 1.0)
        # The control has an explicitly constructed feasible source. This is
        # separate from population-study fits, which never use generating C.
        initial = None if dimension == 2 else embed_source_covariance(source, dimension)
        fit = diagnose_constant_source_admissibility(
            target,
            np.eye(2),
            delta=1.0,
            trace_cap=2.0,
            integration_matrix=matrix,
            initial_source=initial,
            objective_gap_tolerance=1e-11,
        )
        valid = (
            fit.fit_distance_lower_bound > 0.03
            if dimension == 2
            else fit.fit_distance_upper_bound < 1e-12
        )
        records.append(
            {
                "control": "projected_forcing_support",
                "J": 2,
                "M": dimension,
                "delta": 1.0,
                "trace_cap": 2.0,
                "kernel": target,
                "distance_lower": fit.fit_distance_lower_bound,
                "distance_upper": fit.fit_distance_upper_bound,
                "objective_gap_bound": fit.objective_gap_bound,
                "converged": fit.converged,
                "valid": bool(valid and fit.converged and fit.source_feasibility["feasible"]),
                "source_feasibility": fit.source_feasibility,
                "source_candidate": fit.source_candidate,
                "scope": "deterministic finite-source-support control; not a statistical test",
            }
        )
    return {"passed": all(row["valid"] for row in records), "rows": records}


def _representation(kernel, evaluation, weights, rtol):
    root = np.sqrt(weights)
    vectors, singular, _ = np.linalg.svd(root[:, None] * evaluation, full_matrices=False)
    retained = singular > rtol * singular[0]
    span = vectors[:, retained]
    weighted = root[:, None] * kernel * root[None, :]
    residual = weighted - span @ (span.T @ weighted @ span) @ span.T
    return float(np.linalg.norm(residual, "fro")), int(retained.sum())


def run_source_class_study(config):
    """Return (rows, controls) without sampling, writing files, or editing fits.

    Each generating target is E_J P_J V_reference P_J* E_J*. Population
    canonical targets complete that same projected generating partial on each
    configured cover. The grid is predeclared as grid_nodes(41, 'spde').
    A nonempty study retains all declared numerical solves, including
    unconverged intervals. Empty source_dimensions skips large population
    preparation but still evaluates the cheap deterministic controls.
    """
    if not config.source_dimensions:
        return [], _source_support_controls()
    if max(config.source_dimensions) > config.reference_dimension:
        raise ValueError("source-study output dimensions must lie within the fixed reference")
    if max(config.source_dimensions) * max(config.source_multipliers) > 64:
        raise ValueError("the dense source-support study is limited to 64 forcing coordinates")
    if "rotated" in config.models and any(
        (j * multiple) % 2
        for j in config.source_dimensions
        for multiple in config.source_multipliers
    ):
        raise ValueError(
            "rotated fixed-source tail allowances require pair-complete forcing prefixes"
        )
    nodes = grid_nodes(41, "spde")
    weights = quadrature_weights(nodes)
    spectrum = PowerLawSource(config.source_exponent, config.source_trace)
    parents = {
        name: prepare_parent_model(
            name,
            config.reference_dimension,
            config.delta,
            config.quadrature_order,
            source_exponent=config.source_exponent,
            source_trace=config.source_trace,
            rotation_angle=config.rotation_angle,
        )
        for name in config.models
    }
    rows, maps, fits = [], {}, {}
    for output_dimension in config.source_dimensions:
        evaluation = LaguerreTransport(output_dimension).evaluate(nodes)
        for source_dimension in sorted({output_dimension * m for m in config.source_multipliers}):
            key = output_dimension, source_dimension
            matrix = projected_covariance_integration_matrix(
                source_dimension,
                output_dimension,
                config.delta,
                quadrature_order=config.quadrature_order,
            )
            refined = projected_covariance_integration_matrix(
                source_dimension,
                output_dimension,
                config.delta,
                quadrature_order=2 * config.quadrature_order,
            )
            maps[key] = {
                "matrix": matrix,
                "sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                "comparison": integration_map_comparison_allowance(
                    matrix,
                    refined,
                    trace_cap=config.trace_cap,
                    evaluation_matrix=evaluation,
                    weights=weights,
                ),
                "diagnostics": projected_integration_diagnostics(
                    source_dimension,
                    output_dimension,
                    config.delta,
                    quadrature_order=config.quadrature_order,
                    evaluation_matrix=evaluation,
                    weights=weights,
                ),
            }
        for name, parent in parents.items():
            angle = config.rotation_angle if name == "rotated" else 0.0
            truth = (
                evaluation
                @ parent.innovation_covariance[:output_dimension, :output_dimension]
                @ evaluation.T
            )
            reference_tail = fixed_source_tail_allowance(
                spectrum,
                config.reference_dimension,
                output_dimension,
                config.delta,
                evaluation_matrix=evaluation,
                weights=weights,
                rotation_angle=angle,
            )
            for cover in config.covers:
                cliques = clique_indices(nodes, paper_cover(cover, (0.0, 3.0)))
                mask = observed_mask(len(nodes), cliques)
                completion = complete_kernel(
                    np.where(mask, truth, np.nan),
                    cliques=cliques,
                    weights=weights,
                    support_rtol=config.support_rtol,
                    population=True,
                )
                targets = {"generating": truth, "canonical": completion.kernel}
                for target_name, target in targets.items():
                    if target is None:
                        for source_dimension in sorted(
                            {output_dimension * m for m in config.source_multipliers}
                        ):
                            rows.append(
                                {
                                    "model": name,
                                    "cover": cover,
                                    "J": output_dimension,
                                    "M": source_dimension,
                                    "source_multiplier": source_dimension // output_dimension,
                                    "target": target_name,
                                    "available": False,
                                    "unavailable_reason": completion.diagnostics.get(
                                        "reason", "population target unavailable"
                                    ),
                                    "distance_lower": None,
                                    "distance_upper": None,
                                    "nested_witness_upper": None,
                                    "map_comparison_allowance": None,
                                    "fixed_generating_tail_bound": None,
                                    "converged": False,
                                    "canonical_target_diagnostics": completion.diagnostics,
                                    "scope": "population target unavailable; no source-class fit claimed",
                                }
                            )
                        continue
                    target_hash = hashlib.sha256(target.tobytes()).hexdigest()
                    representation, evaluation_rank = _representation(
                        target,
                        evaluation,
                        weights,
                        config.support_rtol,
                    )
                    previous_source, previous_upper = None, None
                    for source_dimension in sorted(
                        {output_dimension * m for m in config.source_multipliers}
                    ):
                        prepared = maps[output_dimension, source_dimension]
                        matrix = prepared["matrix"]
                        initial = (
                            np.zeros((source_dimension, source_dimension))
                            if previous_source is None
                            else embed_source_covariance(previous_source, source_dimension)
                        )
                        embedded = evaluate_source_witness(
                            initial,
                            matrix,
                            evaluation,
                            target_kernel=target,
                            weights=weights,
                            trace_cap=config.trace_cap,
                        )
                        fit_key = output_dimension, source_dimension, target_hash
                        reused = fit_key in fits
                        started = time.perf_counter()
                        if not reused:
                            fits[fit_key] = diagnose_constant_source_admissibility(
                                target,
                                evaluation,
                                delta=config.delta,
                                trace_cap=config.trace_cap,
                                integration_matrix=matrix,
                                weights=weights,
                                initial_source=initial,
                                objective_gap_tolerance=config.source_gap,
                                max_iterations=config.source_max_iterations,
                            )
                        fit = fits[fit_key]
                        current_better = (
                            fit.fit_distance_upper_bound <= embedded["weighted_distance"]
                        )
                        chosen_source = fit.source_candidate if current_better else initial
                        chosen_upper = min(
                            fit.fit_distance_upper_bound, embedded["weighted_distance"]
                        )
                        embedding_difference = (
                            None
                            if previous_upper is None
                            else embedded["weighted_distance"] - previous_upper
                        )
                        tail = fixed_source_tail_allowance(
                            spectrum,
                            source_dimension,
                            output_dimension,
                            config.delta,
                            evaluation_matrix=evaluation,
                            weights=weights,
                            rotation_angle=angle,
                        )
                        rows.append(
                            {
                                "model": name,
                                "cover": cover,
                                "J": output_dimension,
                                "M": source_dimension,
                                "source_multiplier": source_dimension // output_dimension,
                                "target": target_name,
                                "available": True,
                                "unavailable_reason": None,
                                "target_kernel": target,
                                "target_sha256": target_hash,
                                "reference_dimension": config.reference_dimension,
                                "grid_size": len(nodes),
                                "base_grid": 41,
                                "nodes": nodes,
                                "weights": weights,
                                "delta": config.delta,
                                "trace_cap": config.trace_cap,
                                "distance_lower": fit.fit_distance_lower_bound,
                                "distance_upper": fit.fit_distance_upper_bound,
                                "nested_witness_upper": float(chosen_upper),
                                "embedded_previous_witness_distance": embedded["weighted_distance"],
                                "embedding_distance_change": embedding_difference,
                                "nested_witness_source": chosen_source,
                                "map_comparison_allowance": prepared["comparison"][
                                    "weighted_kernel_allowance"
                                ],
                                "fixed_generating_tail_bound": tail["weighted_kernel_tail_bound"],
                                "reference_generating_tail_bound": reference_tail[
                                    "weighted_kernel_tail_bound"
                                ],
                                "fixed_generating_tail_scope": "fixed source only; no allowance automatically applied to the canonical target or infinite source class",
                                "converged": fit.converged,
                                "status": fit.status,
                                "admissibility_status": fit.admissibility_status,
                                "objective": fit.objective,
                                "objective_gap_bound": fit.objective_gap_bound,
                                "iterations": fit.iterations,
                                "source_candidate": fit.source_candidate,
                                "projected_innovation": fit.projected_innovation,
                                "projected_innovation_kernel": fit.projected_innovation_kernel,
                                "source_feasibility": fit.source_feasibility,
                                "metadata": fit.metadata,
                                "source_basis_id": LaguerreTransport(source_dimension).basis_id,
                                "output_basis_id": LaguerreTransport(output_dimension).basis_id,
                                "integration_matrix_sha256": prepared["sha256"],
                                "integration_diagnostics": prepared["diagnostics"],
                                "map_comparison": prepared["comparison"],
                                "fixed_generating_tail": tail,
                                "reference_generating_tail": reference_tail,
                                "reference_integration": parent.numerical_checks,
                                "canonical_target_diagnostics": completion.diagnostics,
                                "representation_residual": representation,
                                "evaluation_rank": evaluation_rank,
                                "evaluation_rank_rtol": config.support_rtol,
                                "reused_numerical_fit": reused,
                                "runtime_seconds": 0.0 if reused else time.perf_counter() - started,
                                "scope": "population finite source-support comparison; no infinite-class exclusion or statistical test",
                            }
                        )
                        previous_source, previous_upper = chosen_source, chosen_upper
    return rows, _source_support_controls()
