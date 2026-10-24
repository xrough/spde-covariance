"""Finite forcing support, output projection, and fixed-source error allowances.

Output dimension J and forcing dimension M are distinct. The map below is
P_J T_delta,M(C_M) P_J*, not T_delta,J(P_J C_M P_J*). No finite cone or fixed
generating-source tail certificate excludes all infinite-dimensional sources.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from .admissibility import _integer, _scalar
from .geometry import lifted_matrix, smat, svec
from .laguerre import LaguerreTransport, PowerLawSource
from .linalg import _real_array, validate_psd, validate_symmetric
from .trajectories import _metadata_snapshot, _readonly


def _dimensions(source_dimension, output_dimension):
    source_dimension = _integer(source_dimension, "source_dimension")
    output_dimension = _integer(output_dimension, "output_dimension")
    if output_dimension > source_dimension:
        raise ValueError("output_dimension must not exceed source_dimension")
    return source_dimension, output_dimension


def _rectangular_lift(operator):
    """Frobenius-isometric covariance lift without allocating M**4 bases."""
    output_rows, output_columns = np.triu_indices(operator.shape[0])
    source_rows, source_columns = np.triu_indices(operator.shape[1])
    left = operator[output_rows[:, None], source_rows[None, :]]
    right = operator[output_columns[:, None], source_columns[None, :]]
    result = left * right
    result += (
        operator[output_rows[:, None], source_columns[None, :]]
        * operator[output_columns[:, None], source_rows[None, :]]
    )
    result *= np.where(source_rows == source_columns, 0.5, 1 / np.sqrt(2))[None, :]
    result *= np.where(output_rows == output_columns, 1.0, np.sqrt(2))[:, None]
    return result


def projected_covariance_integration_matrix(
    source_dimension,
    output_dimension,
    delta,
    *,
    quadrature_order=24,
):
    """Positive Gauss integration of C_M -> P_J T_delta,M(C_M) P_J*.

    The returned immutable array has shape (J(J+1)/2, M(M+1)/2) and acts on
    svec coordinates in the fixed orthonormal Laguerre family. Quadrature
    weights sum to one, so this is normalized innovation covariance V_delta.
    Transport is evaluated in dimension M before taking its first J rows.
    Use refinement and an independent block check below at declared settings;
    a positive quadrature rule is not an automatic integration-error bound.
    """
    source_dimension, output_dimension = _dimensions(source_dimension, output_dimension)
    delta = _scalar(delta, "delta", positive=True)
    order = _integer(quadrature_order, "quadrature_order")
    model = LaguerreTransport(source_dimension, cache_max_bytes=0)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros(
        (
            output_dimension * (output_dimension + 1) // 2,
            source_dimension * (source_dimension + 1) // 2,
        )
    )
    for node, weight in zip(nodes, weights, strict=True):
        operator = model.matrix((node + 1) * (delta / 2))[:output_dimension]
        result += (weight / 2) * _rectangular_lift(operator)
    return _readonly(result, name="projected normalized covariance integration matrix")


def embed_source_covariance(source_covariance, dimension):
    """Embed a feasible source witness in a larger nested Laguerre prefix."""
    source = validate_psd(source_covariance, name="source_covariance")
    dimension = _integer(dimension, "dimension")
    if dimension < len(source):
        raise ValueError("embedding dimension must not be smaller than source covariance")
    result = np.zeros((dimension, dimension))
    result[: len(source), : len(source)] = source
    return _readonly(result, name="embedded source covariance")


def _weighted_evaluation(output_dimension, evaluation_matrix, weights):
    evaluation = (
        np.eye(output_dimension)
        if evaluation_matrix is None
        else _real_array(evaluation_matrix, name="evaluation_matrix")
    )
    if evaluation.ndim != 2 or not len(evaluation) or evaluation.shape[1] != output_dimension:
        raise ValueError("evaluation_matrix must have one column per output coordinate")
    measure = np.ones(len(evaluation)) if weights is None else _real_array(weights, name="weights")
    if measure.shape != (len(evaluation),) or np.any(measure <= 0):
        raise ValueError("weights must be positive and match the evaluation grid")
    return np.sqrt(measure)[:, None] * evaluation, measure


def evaluate_source_witness(
    source_covariance,
    integration_matrix,
    evaluation_matrix,
    *,
    target_kernel=None,
    weights=None,
    trace_cap=None,
):
    """Evaluate a declared feasible witness under this particular numerical map.

    Re-evaluating an embedded lower-support witness supplies a valid candidate
    upper distance in a larger finite cone. Taking the smaller of such upper
    distances is distinct from claiming that independently stopped solver
    iterates must be monotone. A cap violation raises rather than supplying an
    invalid trace-capped upper bound.
    """
    source = validate_psd(source_covariance, name="source_covariance")
    integration = _real_array(integration_matrix, name="integration_matrix")
    size = len(source) * (len(source) + 1) // 2
    if integration.ndim != 2 or integration.shape[1] != size:
        raise ValueError("integration_matrix columns must match source covariance svec coordinates")
    output_dimension = len(smat(np.zeros(integration.shape[0])))
    _, measure = _weighted_evaluation(output_dimension, evaluation_matrix, weights)
    evaluation = _real_array(evaluation_matrix, name="evaluation_matrix")
    trace = float(np.trace(source))
    if not np.isfinite(trace):
        raise ValueError("source witness trace exceeds floating-point range")
    minimum = float(np.linalg.eigvalsh(source)[0])
    tolerance = 100 * np.finfo(float).eps * len(source) * max(1.0, trace)
    if minimum < -tolerance:
        raise ValueError(
            "source witness is not PSD within its floating-point feasibility tolerance"
        )
    if trace_cap is not None:
        trace_cap = _scalar(trace_cap, "trace_cap")
        if trace > trace_cap + tolerance:
            raise ValueError("source witness exceeds trace_cap")
    innovation = smat(integration @ svec(source))
    prediction = evaluation @ innovation @ evaluation.T
    distance = None
    if target_kernel is not None:
        target = validate_symmetric(target_kernel, name="target_kernel")
        if target.shape != prediction.shape:
            raise ValueError("target_kernel must match the evaluation grid")
        root = np.sqrt(measure)
        distance = float(
            np.linalg.norm(root[:, None] * (prediction - target) * root[None, :], "fro")
        )
    if not np.all(np.isfinite(prediction)) or (distance is not None and not np.isfinite(distance)):
        raise ValueError("source witness prediction exceeds floating-point range")
    return _metadata_snapshot(
        {
            "source_dimension": len(source),
            "output_dimension": output_dimension,
            "projected_innovation": innovation,
            "projected_innovation_kernel": prediction,
            "weighted_distance": distance,
            "weights": measure,
            "source_feasibility": {
                "minimum_eigenvalue": minimum,
                "trace": trace,
                "trace_cap": trace_cap,
                "feasible": True,
                "floating_point_tolerance": tolerance,
            },
            "scope": "one feasible source witness under the supplied finite numerical map",
            "infinite_dimensional_exclusion": False,
        }
    )


def integration_map_comparison_allowance(
    first_map,
    second_map,
    *,
    trace_cap,
    evaluation_matrix=None,
    weights=None,
):
    """Uniform trace-capped image discrepancy between two *numerical* maps.

    For PSD C with trace(C)<=R, ||C||_F<=R. Consequently R times the spectral
    norm of their weighted kernel-map difference bounds the image difference
    at the same C, and bounds the distance between their finite image sets.
    This compares the supplied maps, not either map with an exact integral.
    """
    first = _real_array(first_map, name="first_map")
    second = _real_array(second_map, name="second_map")
    if first.ndim != 2 or first.shape != second.shape or not first.size:
        raise ValueError("integration maps must be nonempty matrices of the same shape")
    output_dimension = len(smat(np.zeros(first.shape[0])))
    source_dimension = len(smat(np.zeros(first.shape[1])))
    cap = _scalar(trace_cap, "trace_cap")
    evaluation, _ = _weighted_evaluation(output_dimension, evaluation_matrix, weights)
    # Factor through output coordinates, avoiding a grid-squared lifted matrix.
    _, singular, right = np.linalg.svd(evaluation, full_matrices=False)
    gram_root = (right.T * singular) @ right
    difference = lifted_matrix(gram_root) @ (first - second)
    if not np.all(np.isfinite(difference)):
        raise ValueError("weighted integration-map comparison exceeds floating-point range")
    squared = difference @ difference.T
    if not np.all(np.isfinite(squared)):
        raise ValueError("weighted integration-map comparison exceeds floating-point range")
    norm = np.sqrt(max(0.0, float(np.linalg.eigvalsh((squared + squared.T) / 2)[-1])))
    return _metadata_snapshot(
        {
            "source_dimension": source_dimension,
            "output_dimension": output_dimension,
            "trace_cap": cap,
            "weighted_map_operator_discrepancy": float(norm),
            "weighted_kernel_allowance": float(cap * norm),
            "scope": "uniform comparison of the two supplied finite numerical maps at the same PSD trace-capped source",
            "exact_integration_error_certificate": False,
            "infinite_dimensional_exclusion": False,
        }
    )


def projected_integration_diagnostics(
    source_dimension,
    output_dimension,
    delta,
    *,
    source_covariance=None,
    quadrature_order=24,
    evaluation_matrix=None,
    weights=None,
):
    """Refine the map and block-check one declared PSD source covariance.

    Full-map refinement is recorded separately from the independent block
    comparison at this source. Neither discrepancy certifies error to exact
    integration. Providing a fixed source makes the validation scope explicit.
    """
    source_dimension, output_dimension = _dimensions(source_dimension, output_dimension)
    delta = _scalar(delta, "delta", positive=True)
    order = _integer(quadrature_order, "quadrature_order")
    source = (
        np.eye(source_dimension) / source_dimension
        if source_covariance is None
        else validate_psd(source_covariance, name="source_covariance")
    )
    if source.shape != (source_dimension, source_dimension):
        raise ValueError("source_covariance must match source_dimension")
    coarse = projected_covariance_integration_matrix(
        source_dimension,
        output_dimension,
        delta,
        quadrature_order=order,
    )
    refined = projected_covariance_integration_matrix(
        source_dimension,
        output_dimension,
        delta,
        quadrature_order=2 * order,
    )
    quadrature = smat(coarse @ svec(source))
    fine_covariance = smat(refined @ svec(source))
    block = LaguerreTransport(source_dimension, cache_max_bytes=0).innovation_covariance(
        source,
        delta,
        order,
        method="block",
    )[:output_dimension, :output_dimension]
    evaluation, measure = _weighted_evaluation(output_dimension, evaluation_matrix, weights)

    def kernel_norm(matrix):
        return float(np.linalg.norm(evaluation @ matrix @ evaluation.T, "fro"))

    delta_map = refined - coarse
    return _metadata_snapshot(
        {
            "source_dimension": source_dimension,
            "output_dimension": output_dimension,
            "source_basis_id": LaguerreTransport(source_dimension).basis_id,
            "output_basis_id": LaguerreTransport(output_dimension).basis_id,
            "delta": delta,
            "quadrature_order": order,
            "refined_quadrature_order": 2 * order,
            "map_frobenius_refinement_discrepancy": float(np.linalg.norm(delta_map, "fro")),
            "quadrature_output_covariance": quadrature,
            "refined_output_covariance": fine_covariance,
            "block_output_covariance": block,
            "output_hs_refinement_discrepancy": float(
                np.linalg.norm(quadrature - fine_covariance, "fro")
            ),
            "output_hs_block_discrepancy": float(np.linalg.norm(quadrature - block, "fro")),
            "weighted_kernel_refinement_discrepancy": kernel_norm(quadrature - fine_covariance),
            "weighted_kernel_block_discrepancy": kernel_norm(quadrature - block),
            "minimum_output_eigenvalue": float(np.linalg.eigvalsh(quadrature)[0]),
            "source_trace": float(np.trace(source)),
            "weights": measure,
            "source_scope": "the supplied fixed source covariance only; identity/trace-one if omitted",
            "numerical_error_certificate": False,
            "uniform_source_class_certificate": False,
            "infinite_dimensional_exclusion": False,
        }
    )


def fixed_source_tail_allowance(
    source,
    source_dimension,
    output_dimension,
    delta,
    *,
    evaluation_matrix=None,
    weights=None,
    rotation_angle=0.0,
):
    """Analytic omitted-forcing allowance for one declared PowerLawSource.

    The default source is diagonal in the fixed Laguerre family. A declared
    adjacent-pair rotation preserves this bound at even forcing cutoffs.
    It bounds a fixed generating kernel's target perturbation, not a uniform
    approximation of all PSD trace-capped sources or inverse source recovery.
    """
    if not isinstance(source, PowerLawSource):
        raise TypeError("source must be a PowerLawSource in the fixed Laguerre family")
    source_dimension, output_dimension = _dimensions(source_dimension, output_dimension)
    delta = _scalar(delta, "delta", positive=True)
    if (
        isinstance(rotation_angle, (bool, np.bool_))
        or not isinstance(rotation_angle, Real)
        or not np.isfinite(rotation_angle)
    ):
        raise ValueError("rotation_angle must be finite and real")
    if rotation_angle != 0 and source_dimension % 2:
        raise ValueError("adjacent-pair rotated source tail requires an even source cutoff")
    bound = source.projected_tail_bound(output_dimension, source_dimension, delta)
    evaluation, measure = _weighted_evaluation(output_dimension, evaluation_matrix, weights)
    kernel_bound = evaluation @ bound["matrix"] @ evaluation.T
    kernel_norm = float(np.linalg.norm(kernel_bound, "fro"))
    if not np.all(np.isfinite(kernel_bound)) or not np.isfinite(kernel_norm):
        raise ValueError("weighted fixed-source tail allowance exceeds floating-point range")
    return _metadata_snapshot(
        {
            "source_family_id": source.family_id,
            "basis_family_id": source.basis_family_id,
            "source_exponent": source.exponent,
            "source_trace": source.trace,
            "source_dimension": source_dimension,
            "output_dimension": output_dimension,
            "rotation_angle": float(rotation_angle),
            "delta": delta,
            "source_tail_trace": source.tail_trace(source_dimension),
            "full_innovation_trace_bound": source.innovation_tail_bound(source_dimension, delta),
            "output_covariance_upper_matrix": bound["matrix"],
            "output_covariance_hs_bound": bound["hs_bound"],
            "weighted_kernel_upper_matrix": kernel_bound,
            "weighted_kernel_tail_bound": kernel_norm,
            "q_M": bound["q_M"],
            "weights": measure,
            "scope": "fixed declared generating-source truncation; same-target perturbation allowance only",
            "uniform_source_class_certificate": False,
            "infinite_dimensional_exclusion": False,
            "arithmetic": "analytic PSD-order bound evaluated in floating point; not interval arithmetic",
        }
    )
