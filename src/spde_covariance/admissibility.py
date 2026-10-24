"""Distance to a declared finite constant-source transport covariance class.

This is a diagnostic on an already fitted innovation kernel. It does not
change that kernel's fitting target and does not establish exclusion from an
infinite-dimensional or time-varying source class. Candidate sources are
optimization witnesses, not claimed recoveries of the physical noise source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isqrt
from numbers import Real

import numpy as np

from .geometry import lifted_matrix, smat, svec
from .linalg import _real_array, project_psd_trace, validate_psd, validate_symmetric
from .trajectories import _metadata_snapshot, _readonly


def _scalar(value, name, *, positive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _integer(value, name, *, zero=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return int(value)


def covariance_integration_matrix(propagator, delta, *, quadrature_order=24):
    """Lift normalized constant-source integration to Frobenius-isometric svec.

    Existing propagators accept PSD source covariances rather than indefinite
    symmetric basis vectors. Polarization of rank-one PSD inputs constructs
    the same linear map with J(J+1)/2 integrations. The returned matrix is
    reusable across kernels and evaluation grids; there is no persistent cache.
    """
    delta = _scalar(delta, "delta", positive=True)
    quadrature_order = _integer(quadrature_order, "quadrature_order")
    dimension = _integer(getattr(propagator, "dimension", None), "propagator dimension")
    integrate = getattr(propagator, "innovation_covariance", None)
    if not callable(integrate):
        raise ValueError("propagator must provide innovation_covariance")
    identity = np.eye(dimension)

    def image(source):
        value = validate_psd(
            integrate(source, delta, quadrature_order=quadrature_order),
            name="integrated source covariance",
        )
        if value.shape != (dimension, dimension):
            raise ValueError("propagator covariance output has the wrong dimension")
        return value

    diagonal = [image(np.outer(row, row)) for row in identity]
    columns = []
    for left, right in zip(*np.triu_indices(dimension), strict=True):
        if left == right:
            value = diagonal[left]
        else:
            vector = (identity[left] + identity[right]) / np.sqrt(2)
            value = np.sqrt(2) * image(np.outer(vector, vector))
            value -= (diagonal[left] + diagonal[right]) / np.sqrt(2)
        columns.append(svec(value))
    return _readonly(np.column_stack(columns), name="covariance integration matrix")


@dataclass(frozen=True, slots=True)
class AdmissibilityDiagnostic:
    source_candidate: np.ndarray
    projected_innovation: np.ndarray
    projected_innovation_kernel: np.ndarray
    objective: float
    objective_lower_bound: float
    objective_gap_bound: float
    fit_distance_lower_bound: float
    fit_distance_upper_bound: float
    reference_distance_lower_bound: float
    reference_distance_upper_bound: float
    iterations: int
    converged: bool
    status: str
    admissibility_status: str
    source_feasibility: Mapping
    metadata: Mapping

    def __post_init__(self):
        for name in ("source_candidate", "projected_innovation", "projected_innovation_kernel"):
            object.__setattr__(self, name, _readonly(getattr(self, name), name=name))
        object.__setattr__(self, "source_feasibility", _metadata_snapshot(self.source_feasibility))
        object.__setattr__(self, "metadata", _metadata_snapshot(self.metadata))

    @property
    def fit_distance_bounds(self):
        return self.fit_distance_lower_bound, self.fit_distance_upper_bound

    @property
    def reference_distance_bounds(self):
        return self.reference_distance_lower_bound, self.reference_distance_upper_bound

    @property
    def source_dimension(self):
        return len(self.source_candidate)

    @property
    def output_dimension(self):
        return len(self.projected_innovation)


def _problem(kernel, evaluation, weights, integration):
    """Form a small source-space Hessian without a grid-squared design matrix."""
    root = np.sqrt(weights)
    weighted_evaluation = root[:, None] * evaluation
    weighted_kernel = root[:, None] * kernel * root[None, :]
    gram = weighted_evaluation.T @ weighted_evaluation
    # A(V)=G V G*, so A*A(V)=(G*G)V(G*G). The covariance
    # integration map need not be self-adjoint: its adjoint is integration.T.
    normal = lifted_matrix(gram)
    hessian = integration.T @ normal @ integration
    hessian = (hessian + hessian.T) / 2
    rhs = integration.T @ svec(weighted_evaluation.T @ weighted_kernel @ weighted_evaluation)
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(rhs)):
        raise ValueError("admissibility normal equations overflowed; rescale the kernel or grid")
    return hessian, rhs, weighted_evaluation, weighted_kernel


def _compressed_problem(kernel, evaluation, weights, integration):
    """A source-space normal map factored through the smaller output space.

    For M>J the Hessian has a nullspace and need not be formed as a dense
    dM-by-dM array. L(sqrt(E* W E)) B has exactly the same normal map. Its
    nonzero squared singular values give the Lipschitz constant in dJ space.
    """
    root = np.sqrt(weights)
    weighted_evaluation = root[:, None] * evaluation
    weighted_kernel = root[:, None] * kernel * root[None, :]
    # SVD avoids squaring the evaluation condition number before taking its
    # square root, which matters for long Laguerre prefixes on short grids.
    _, singular, right = np.linalg.svd(weighted_evaluation, full_matrices=False)
    gram_root = (right.T * singular) @ right
    factor = lifted_matrix(gram_root) @ integration
    rhs = integration.T @ svec(weighted_evaluation.T @ weighted_kernel @ weighted_evaluation)
    if not np.all(np.isfinite(factor)) or not np.all(np.isfinite(rhs)):
        raise ValueError("admissibility normal equations overflowed; rescale the kernel or grid")
    lipschitz = max(0.0, float(np.linalg.eigvalsh(factor @ factor.T)[-1]))
    return factor, rhs, weighted_evaluation, weighted_kernel, lipschitz


def _gap(source, gradient, cap):
    minimum = float(np.linalg.eigvalsh(gradient)[0])
    raw = float(np.sum(source * gradient) - cap * min(minimum, 0.0))
    return max(raw, 0.0), raw


def diagnose_constant_source_admissibility(
    kernel,
    evaluation_matrix,
    *,
    delta,
    trace_cap,
    propagator=None,
    integration_matrix=None,
    weights=None,
    quadrature_order=24,
    objective_gap_tolerance=1e-10,
    max_iterations=20_000,
    initial_source=None,
    admissibility_tolerance=0.0,
    grid_error_bound=0.0,
    source_tail_bound=0.0,
):
    """Bound distance to E T_delta(C) E* with C>=0 and trace(C)<=trace_cap.

    T_delta(C) is the normalized innovation covariance under one constant
    finite-dimensional source. Exactly one propagator or precomputed linear
    svec integration matrix is required. A rectangular matrix with shape
    (J(J+1)/2, M(M+1)/2) maps an M-dimensional source to a J-dimensional
    output covariance; evaluation_matrix has J columns. The propagator-only
    interface keeps its existing matched source/output dimensions. The
    kernel may be indefinite.

    The global convex gap is <gradient,C>-min_D<gradient,D>, with D in the
    source constraint set. It supplies an interval for the *minimum* weighted
    kernel distance even when iteration limits are reached. Supplied grid and
    tail bounds must already be expressed in this weighted kernel norm; their
    sum widens the reference-distance interval for a bounded perturbation of
    the input kernel against the *same finite source class*. A tail bound for
    one generating source is not a uniform approximation of all infinite
    sources. Integration and floating-point errors are not automatically
    certified. Invalid inputs or nonfinite calculations raise ValueError;
    an iteration-limit exit remains available with converged=False.
    """
    matrix = validate_symmetric(kernel, name="input kernel")
    evaluation = _real_array(evaluation_matrix, name="evaluation_matrix")
    if evaluation.ndim != 2 or evaluation.shape[0] != len(matrix) or not evaluation.shape[1]:
        raise ValueError("evaluation_matrix must have shape (kernel grid size, output dimension)")
    delta, cap = _scalar(delta, "delta", positive=True), _scalar(trace_cap, "trace_cap")
    tolerance = _scalar(objective_gap_tolerance, "objective_gap_tolerance")
    admissibility_tolerance = _scalar(admissibility_tolerance, "admissibility_tolerance")
    grid_error_bound = _scalar(grid_error_bound, "grid_error_bound")
    source_tail_bound = _scalar(source_tail_bound, "source_tail_bound")
    max_iterations = _integer(max_iterations, "max_iterations", zero=True)
    quadrature_order = _integer(quadrature_order, "quadrature_order")
    output_dimension = evaluation.shape[1]
    output_size = output_dimension * (output_dimension + 1) // 2
    if (propagator is None) == (integration_matrix is None):
        raise ValueError("provide exactly one propagator or integration_matrix")
    if propagator is not None:
        if getattr(propagator, "dimension", None) != output_dimension:
            raise ValueError("propagator dimension does not match evaluation_matrix")
        integration = covariance_integration_matrix(
            propagator, delta, quadrature_order=quadrature_order
        )
        integration_origin = "normalized covariance integration of the supplied propagator"
        dimension = output_dimension
    else:
        integration = _real_array(integration_matrix, name="integration_matrix")
        if integration.ndim != 2 or integration.shape[0] != output_size:
            raise ValueError("integration_matrix rows must match symmetric svec output coordinates")
        size = integration.shape[1]
        dimension = (isqrt(8 * size + 1) - 1) // 2
        if dimension < 1 or dimension * (dimension + 1) // 2 != size:
            raise ValueError(
                "integration_matrix columns must be nonempty symmetric svec source coordinates"
            )
        integration = integration.copy()
        integration_origin = "caller-supplied normalized linear covariance map in svec coordinates"
    size = dimension * (dimension + 1) // 2
    if weights is None:
        measure = np.ones(len(matrix))
    else:
        measure = _real_array(weights, name="weights")
        if measure.shape != (len(matrix),) or np.any(measure <= 0):
            raise ValueError("quadrature weights must be positive and match the kernel grid")
    compressed = integration.shape[1] > integration.shape[0]
    if compressed:
        factor, rhs, weighted_evaluation, weighted_kernel, lipschitz = _compressed_problem(
            matrix,
            evaluation,
            measure,
            integration,
        )
        strong_convexity = 0.0

        def normal(vector):
            return factor.T @ (factor @ vector)
    else:
        hessian, rhs, weighted_evaluation, weighted_kernel = _problem(
            matrix,
            evaluation,
            measure,
            integration,
        )
        hessian_eigenvalues = np.linalg.eigvalsh(hessian)
        lipschitz = max(0.0, float(hessian_eigenvalues[-1]))
        strong_convexity = max(0.0, float(hessian_eigenvalues[0]))

        def normal(vector):
            return hessian @ vector

    initial = (
        np.zeros((dimension, dimension))
        if initial_source is None
        else validate_symmetric(
            initial_source,
            name="initial_source",
        )
    )
    if initial.shape != (dimension, dimension):
        raise ValueError("initial_source must match the source dimension")
    source = project_psd_trace(initial, cap)
    vector = svec(source)
    extrapolated, acceleration = vector.copy(), 1.0
    iterations, converged = 0, False
    best, best_gap = vector.copy(), np.inf
    value = 0.5 * vector @ normal(vector) - rhs @ vector

    def explicit_certificate(coordinates):
        candidate = smat(coordinates)
        innovation = smat(integration @ coordinates)
        prediction = evaluation @ innovation @ evaluation.T
        residual = weighted_evaluation @ innovation @ weighted_evaluation.T - weighted_kernel
        objective = 0.5 * float(np.sum(residual**2))
        if not np.isfinite(objective) or not np.all(np.isfinite(prediction)):
            raise ValueError("admissibility residual overflowed; rescale the kernel or grid")
        gradient = smat(
            integration.T @ svec(weighted_evaluation.T @ residual @ weighted_evaluation)
        )
        gap, raw_gap = _gap(candidate, gradient, cap)
        return candidate, innovation, prediction, objective, gap, raw_gap

    if lipschitz == 0.0 or cap == 0.0:
        vector = np.zeros(size)
        converged = True
        status = "optimal_degenerate_class"
    else:
        status = "maximum_iterations"
        for iteration in range(max_iterations + 1):
            gradient = smat(normal(vector) - rhs)
            approximate_gap, _ = _gap(smat(vector), gradient, cap)
            if approximate_gap < best_gap:
                best, best_gap = vector.copy(), approximate_gap
            if approximate_gap <= tolerance:
                certificate = explicit_certificate(vector)
                if certificate[4] <= tolerance:
                    converged, status = True, "objective_gap_tolerance"
                    break
            if iteration == max_iterations:
                break
            proposed = svec(
                project_psd_trace(
                    smat(extrapolated - (normal(extrapolated) - rhs) / lipschitz),
                    cap,
                )
            )
            proposed_value = 0.5 * proposed @ normal(proposed) - rhs @ proposed
            if proposed_value > value + 1e-14 * max(1.0, abs(value)):
                # Monotone restart: objective comparisons omit only a common
                # constant, while the final certificate uses explicit residuals.
                extrapolated, acceleration = vector.copy(), 1.0
                proposed = svec(
                    project_psd_trace(
                        smat(extrapolated - (normal(extrapolated) - rhs) / lipschitz),
                        cap,
                    )
                )
                proposed_value = 0.5 * proposed @ normal(proposed) - rhs @ proposed
            previous = vector
            vector = proposed
            value = proposed_value
            if strong_convexity > 1e-14 * lipschitz:
                momentum = (np.sqrt(lipschitz) - np.sqrt(strong_convexity)) / (
                    np.sqrt(lipschitz) + np.sqrt(strong_convexity)
                )
            else:
                following = (1 + np.sqrt(1 + 4 * acceleration**2)) / 2
                momentum = (acceleration - 1) / following
                acceleration = following
            extrapolated = vector + momentum * (vector - previous)
            iterations = iteration + 1
        if not converged:
            vector = best
    source, innovation, prediction, objective, gap, raw_gap = explicit_certificate(vector)
    if gap <= tolerance:
        converged = True
        if status == "maximum_iterations":
            status = "objective_gap_tolerance"
    lower_objective = max(0.0, objective - gap)
    fit_lower, fit_upper = np.sqrt(2 * lower_objective), np.sqrt(2 * objective)
    uncertainty = grid_error_bound + source_tail_bound
    reference_lower, reference_upper = max(0.0, fit_lower - uncertainty), fit_upper + uncertainty
    if reference_upper <= admissibility_tolerance:
        admissibility = "compatible_with_declared_tolerance"
    elif reference_lower > admissibility_tolerance:
        admissibility = "separated_from_declared_finite_class"
    else:
        admissibility = "unresolved_at_declared_tolerance"
    minimum, trace = float(np.linalg.eigvalsh(source)[0]), float(np.trace(source))
    feasibility_tolerance = (
        100 * np.finfo(float).eps * dimension * max(1.0, cap, np.linalg.norm(source))
    )
    feasibility = {
        "minimum_eigenvalue": minimum,
        "trace": trace,
        "trace_cap": cap,
        "psd_violation": max(0.0, -minimum),
        "trace_violation": max(0.0, trace - cap),
        "feasible": bool(
            minimum >= -feasibility_tolerance and trace <= cap + feasibility_tolerance
        ),
        "floating_point_tolerance": feasibility_tolerance,
    }
    metadata = {
        "target": "admissibility diagnostic of an input innovation kernel; no change of fitting target",
        "class": "finite-dimensional constant-source transport covariance with a source trace cap",
        "infinite_dimensional_exclusion": False,
        "recovered_physical_source": False,
        "dimension": dimension,
        "source_dimension": dimension,
        "output_dimension": output_dimension,
        "source_cutoff_dimension": dimension,
        "grid_size": len(matrix),
        "delta": delta,
        "basis_id": getattr(propagator, "basis_id", None),
        "propagation_method": getattr(propagator, "kind", None),
        "integration_origin": integration_origin,
        "quadrature_order": quadrature_order,
        "integration_map_assumption": "normalized linear covariance map; positivity is caller-declared for supplied matrices",
        "objective": "0.5 * ||sqrt(W) (E T_delta(C) E.T - K) sqrt(W)||_F^2",
        "metric": "weighted grid-kernel Frobenius; not source-Hilbert covariance distance",
        "weights": measure,
        "lipschitz_constant": lipschitz,
        "normal_equations": "output-space factorization" if compressed else "dense source Hessian",
        "objective_gap_tolerance": tolerance,
        "raw_objective_gap": raw_gap,
        "admissibility_tolerance": admissibility_tolerance,
        "grid_error_bound": grid_error_bound,
        "source_tail_bound": source_tail_bound,
        "reference_error_bound_units": "the same weighted grid-kernel norm as fit_distance_bounds",
        "reference_distance_scope": "bounded perturbation of the input kernel against the same finite source class; no larger source-class approximation inferred",
        "certificate_scope": "convex finite numerical-map problem in floating point; not interval arithmetic or an integration-error certificate",
        "source_interpretation": "feasible optimization witness; nonuniqueness and model restriction remain",
    }
    return AdmissibilityDiagnostic(
        source,
        innovation,
        prediction,
        objective,
        lower_objective,
        gap,
        float(fit_lower),
        float(fit_upper),
        float(reference_lower),
        float(reference_upper),
        iterations,
        converged,
        status,
        admissibility,
        feasibility,
        metadata,
    )
