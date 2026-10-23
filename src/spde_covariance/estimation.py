"""Full PSD ridge estimation and identifiable linear covariance functionals.

All matrices use an orthonormal Hilbert-space basis.  The objective is
``sum(w_i * ||O_i B O_i.T - Z_i||_F**2) / 2 + lambda * ||B||_F**2 / 2``.
Negative eigenvalues of the noise-corrected measurements are retained.
"""

from dataclasses import dataclass, replace
from numbers import Real
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import gram_operator, smat, svec
from .linalg import project_psd_trace, validate_symmetric
from .observations import Observation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CovarianceEstimate:
    """A feasible iterate with a first-order global objective-gap certificate.

    The certificate is an exact-arithmetic convexity bound evaluated in floating
    point, not a computer-assisted interval-arithmetic proof.  ``converged`` is
    true only when that certificate meets the requested tolerance.
    Reference diagnostics record additional SCS solves and the tolerance of the
    returned iterate; ``iterations`` includes all attempted reference solves.
    """

    covariance: FloatArray
    objective: float
    objective_gap_bound: float
    covariance_error_bound: float
    converged: bool
    status: str
    iterations: int
    q_min: float
    lipschitz: float
    min_eigenvalue: float
    trace: float
    trace_cap: float
    regularization: float
    weights: FloatArray
    times: tuple
    solver: str = "accelerated_projected_gradient"
    target: str = "innovation covariance V"
    basis: str = "orthonormal H coordinates"
    reference_refinements: int = 0
    reference_solver_tolerance: float | None = None


@dataclass(frozen=True)
class RiskEstimate:
    """Unclipped linear estimate; it can be negative for a variance target."""

    estimate: float
    target_matrix: FloatArray
    observability_residual: float
    observability_factor_squared: float
    influence_matrices: tuple[FloatArray, ...]
    weights: FloatArray
    relative_threshold: float
    target: str = "linear functional of innovation covariance V"
    basis: str = "orthonormal H coordinates"


def _normalized_weights(count: int, weights: ArrayLike | None) -> FloatArray:
    if count == 0:
        raise ValueError("at least one observation is required")
    values = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a finite vector, one per date")
    if np.any(values < 0) or np.sum(values) <= 0:
        raise ValueError("weights must be nonnegative with a positive sum")
    return values / np.sum(values)


def uniform_trailing_weights(times: ArrayLike, width: float) -> FloatArray:
    """Uniform weights on ``[max(times)-width, max(times)]`` (same time units)."""
    dates = np.asarray(times, dtype=float)
    if dates.ndim != 1 or dates.size == 0 or not np.all(np.isfinite(dates)):
        raise ValueError("times must be a nonempty finite vector")
    if not np.isfinite(width) or width < 0:
        raise ValueError("width must be finite and nonnegative")
    return _normalized_weights(len(dates), dates >= np.max(dates) - width)


def _problem(observations: Sequence[Observation], regularization, trace_cap, weights):
    data = tuple(observations)
    values = _normalized_weights(len(data), weights)
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and strictly positive")
    if not np.isfinite(trace_cap) or trace_cap < 0:
        raise ValueError("trace_cap must be finite and nonnegative")
    dimension = data[0].operator.shape[1]
    if any(item.operator.shape[1] != dimension for item in data):
        raise ValueError("all observation operators must use the same basis")
    operators = [item.operator for item in data]
    q = gram_operator(operators, values)
    hessian = q + regularization * np.eye(q.shape[0])
    rhs = sum(
        weight * item.operator.T @ item.measurement @ item.operator
        for weight, item in zip(values, data, strict=True)
    )
    constant = sum(
        weight * np.sum(item.measurement**2) / 2 for weight, item in zip(values, data, strict=True)
    )
    return data, values, q, hessian, svec(rhs), constant


def _gap_bound(matrix, hessian, rhs, trace_cap):
    vector = svec(matrix)
    gradient = smat(hessian @ vector - rhs)
    gradient = (gradient + gradient.T) / 2
    # min_{D PSD, tr D <= R} <G,D> = R min(lambda_min(G), 0).
    raw_gap = float(
        np.sum(gradient * matrix) - trace_cap * min(np.linalg.eigvalsh(gradient)[0], 0.0)
    )
    # Numerical round-off can make a zero mathematical certificate slightly negative.
    return max(0.0, raw_gap)


def _result(
    matrix,
    data,
    weights,
    q,
    hessian,
    rhs,
    constant,
    regularization,
    trace_cap,
    iterations,
    converged,
    status,
    solver="accelerated_projected_gradient",
):
    gap = _gap_bound(matrix, hessian, rhs, trace_cap)
    q_min = max(0.0, float(np.linalg.eigvalsh(q)[0]))
    eigenvalues = np.linalg.eigvalsh(matrix)
    # Evaluate residuals directly to avoid cancellation of large sufficient statistics.
    objective = float(
        sum(
            weight * np.sum((item.operator @ matrix @ item.operator.T - item.measurement) ** 2) / 2
            for weight, item in zip(weights, data, strict=True)
        )
        + regularization * np.sum(matrix**2) / 2
    )
    return CovarianceEstimate(
        covariance=matrix,
        objective=objective,
        objective_gap_bound=gap,
        covariance_error_bound=float(np.sqrt(2 * gap / (q_min + regularization))),
        converged=converged,
        status=status,
        iterations=iterations,
        q_min=q_min,
        lipschitz=float(np.linalg.eigvalsh(hessian)[-1]),
        min_eigenvalue=float(eigenvalues[0]),
        trace=float(np.trace(matrix)),
        trace_cap=float(trace_cap),
        regularization=float(regularization),
        weights=weights,
        times=tuple(item.time for item in data),
        solver=solver,
    )


def fit_covariance(
    observations: Sequence[Observation],
    *,
    regularization: float,
    trace_cap: float,
    weights: ArrayLike | None = None,
    max_iterations: int = 20_000,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-8,
    numerical_error_budget: float | None = None,
    initial: ArrayLike | None = None,
) -> CovarianceEstimate:
    """Minimize the full PSD ridge objective by accelerated projected gradient.

    Stop when the global objective-gap bound is at most
    ``absolute_tolerance + relative_tolerance * max(1, abs(objective))``.
    If ``numerical_error_budget`` is given, also require the certified Frobenius
    covariance-error bound to be at most that budget. Strongly convex Nesterov
    momentum uses the exact finite-dimensional Hessian eigenvalue bounds, with
    an objective-increase restart. Returned iterates are always projected.
    A maximum-iteration exit returns a feasible estimate with ``converged=False``.
    No rank constraint or PSD truncation of measurements is used.
    """
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if (
        not np.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
        or not np.isfinite(relative_tolerance)
        or relative_tolerance < 0
    ):
        raise ValueError("tolerances must be finite and nonnegative")
    if numerical_error_budget is not None and (
        not np.isfinite(numerical_error_budget) or numerical_error_budget <= 0
    ):
        raise ValueError("numerical_error_budget must be finite and strictly positive")
    data, values, q, hessian, rhs, constant = _problem(
        observations, regularization, trace_cap, weights
    )
    dimension = data[0].operator.shape[1]
    matrix = np.zeros((dimension, dimension)) if initial is None else validate_symmetric(initial)
    if matrix.shape != (dimension, dimension):
        raise ValueError("initial covariance must match the observation basis")
    matrix = project_psd_trace(matrix, trace_cap)
    lipschitz = float(np.linalg.eigvalsh(hessian)[-1])
    strong_convexity = max(0.0, float(np.linalg.eigvalsh(q)[0])) + regularization
    momentum = (np.sqrt(lipschitz) - np.sqrt(strong_convexity)) / (
        np.sqrt(lipschitz) + np.sqrt(strong_convexity)
    )
    extrapolated = matrix.copy()
    for iteration in range(max_iterations + 1):
        gap = _gap_bound(matrix, hessian, rhs, trace_cap)
        vector = svec(matrix)
        objective = float(0.5 * vector @ hessian @ vector - rhs @ vector + constant)
        tolerance = absolute_tolerance + relative_tolerance * max(1, abs(objective))
        if numerical_error_budget is not None:
            tolerance = min(tolerance, 0.5 * strong_convexity * numerical_error_budget**2)
        if gap <= tolerance:
            return _result(
                matrix,
                data,
                values,
                q,
                hessian,
                rhs,
                constant,
                regularization,
                trace_cap,
                iteration,
                True,
                "converged",
            )
        if iteration < max_iterations:
            extrapolated_vector = svec(extrapolated)
            gradient = smat(hessian @ extrapolated_vector - rhs)
            candidate = project_psd_trace(extrapolated - gradient / lipschitz, trace_cap)
            candidate_vector = svec(candidate)
            reduced_objective = float(0.5 * vector @ hessian @ vector - rhs @ vector)
            candidate_objective = float(
                0.5 * candidate_vector @ hessian @ candidate_vector - rhs @ candidate_vector
            )
            # Ignore round-off-sized changes when deciding whether to restart.
            restart = candidate_objective > reduced_objective + 1e-14 * max(
                1.0, abs(reduced_objective)
            )
            if restart:
                gradient = smat(hessian @ vector - rhs)
                candidate = project_psd_trace(matrix - gradient / lipschitz, trace_cap)
                extrapolated = candidate.copy()
            else:
                extrapolated = candidate + momentum * (candidate - matrix)
            matrix = candidate
    return _result(
        matrix,
        data,
        values,
        q,
        hessian,
        rhs,
        constant,
        regularization,
        trace_cap,
        max_iterations,
        False,
        "maximum_iterations",
    )


def fit_covariance_reference(
    observations: Sequence[Observation],
    *,
    regularization: float,
    trace_cap: float,
    weights: ArrayLike | None = None,
    solver_tolerance: float = 1e-8,
    max_iterations: int = 100_000,
    numerical_error_budget: float | None = None,
) -> CovarianceEstimate:
    """Independent small-problem CVXPY/SCS formulation (optional dependency).

    Project the returned solver iterate to feasibility and evaluate the same gap
    certificate. Solver status alone is never treated as a numerical certificate.
    With ``numerical_error_budget``, require the same certified Frobenius error
    bound as the primary solver. If needed, independently refine the same SCS
    problem up to twice, reducing its tolerance by a factor of 100 each time.
    Return a certified feasible iterate when available, otherwise the iterate
    with the smallest gap and ``converged=False`` after bounded refinements.
    Without a budget, retain the original single-solve objective-gap criterion.
    """
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ImportError("Install the 'reference' extra to use CVXPY/SCS") from error
    if not np.isfinite(solver_tolerance) or solver_tolerance <= 0:
        raise ValueError("solver_tolerance must be positive and finite")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if numerical_error_budget is not None and (
        isinstance(numerical_error_budget, (bool, np.bool_))
        or not isinstance(numerical_error_budget, Real)
        or not np.isfinite(numerical_error_budget)
        or numerical_error_budget <= 0
    ):
        raise ValueError("numerical_error_budget must be finite and strictly positive")
    data, values, q, hessian, rhs, constant = _problem(
        observations, regularization, trace_cap, weights
    )
    dimension = data[0].operator.shape[1]
    variable = cp.Variable((dimension, dimension), symmetric=True)
    loss = (
        sum(
            weight
            * cp.sum_squares(item.operator @ variable @ item.operator.T - item.measurement)
            / 2
            for weight, item in zip(values, data, strict=True)
        )
        + regularization * cp.sum_squares(variable) / 2
    )
    problem = cp.Problem(cp.Minimize(loss), [variable >> 0, cp.trace(variable) <= trace_cap])
    best = None
    total_iterations = 0
    refinements_performed = 0
    attempts = 1 if numerical_error_budget is None else 3
    for refinement in range(attempts):
        tolerance = float(solver_tolerance * 0.01**refinement)
        if tolerance == 0:
            break
        problem.solve(
            solver="SCS", eps=tolerance, max_iters=int(max_iterations), warm_start=refinement > 0
        )
        refinements_performed = refinement
        if variable.value is None:
            raise RuntimeError(f"SCS returned no estimate: {problem.status}")
        total_iterations += problem.solver_stats.num_iters or 0
        matrix = project_psd_trace((variable.value + variable.value.T) / 2, trace_cap)
        result = _result(
            matrix,
            data,
            values,
            q,
            hessian,
            rhs,
            constant,
            regularization,
            trace_cap,
            total_iterations,
            False,
            str(problem.status),
            "cvxpy_scs",
        )
        certified = result.objective_gap_bound <= solver_tolerance * max(1, abs(result.objective))
        if numerical_error_budget is not None:
            certified = certified and result.covariance_error_bound <= numerical_error_budget
        result = replace(
            result,
            converged=certified,
            reference_solver_tolerance=tolerance,
        )
        if (
            best is None
            or (result.converged and not best.converged)
            or result.objective_gap_bound < best.objective_gap_bound
        ):
            best = result
        if best.converged:
            break
    return replace(best, iterations=total_iterations, reference_refinements=refinements_performed)


def estimate_risk(
    observations: Sequence[Observation],
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    relative_threshold: float = 1e-10,
) -> RiskEstimate:
    """Estimate ``<target,V>`` only if the target lies in the numerical range of Q.

    Designs and weights are assumed deterministic.  The estimate is unbiased
    under a common innovation covariance and correct noise correction. A target
    eigen-direction is treated as observable when its eigenvalue exceeds
    ``relative_threshold * lambda_max(Q)``; the same relative tolerance is used
    for the target's nullspace residual. No confidence interval is implied.
    """
    if not np.isfinite(relative_threshold) or not 0 < relative_threshold < 1:
        raise ValueError("relative_threshold must lie strictly between zero and one")
    data = tuple(observations)
    values = _normalized_weights(len(data), weights)
    matrix = validate_symmetric(target)
    dimension = data[0].operator.shape[1]
    if matrix.shape != (dimension, dimension):
        raise ValueError("target must use the observation basis")
    if any(item.operator.shape[1] != dimension for item in data):
        raise ValueError("all observation operators must use the same basis")
    q = gram_operator([item.operator for item in data], values)
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    active = eigenvalues > relative_threshold * max(0.0, eigenvalues[-1])
    vector = svec(matrix)
    coefficients = eigenvectors.T @ vector
    norm = float(np.linalg.norm(vector))
    residual = float(np.linalg.norm(coefficients[~active]))
    relative_residual = residual / norm if norm > 0 else 0.0
    if relative_residual > relative_threshold:
        raise ValueError(
            f"target is not observable at threshold {relative_threshold:g}; "
            f"relative nullspace residual={relative_residual:.3g}"
        )
    inverse = np.zeros_like(coefficients)
    inverse[active] = coefficients[active] / eigenvalues[active]
    dual = smat(eigenvectors @ inverse)
    influences = tuple(item.operator @ dual @ item.operator.T for item in data)
    value = sum(
        weight * np.sum(influence * item.measurement)
        for weight, influence, item in zip(values, influences, data, strict=True)
    )
    return RiskEstimate(
        estimate=float(value),
        target_matrix=matrix,
        observability_residual=relative_residual,
        observability_factor_squared=float(vector @ (eigenvectors @ inverse)),
        influence_matrices=influences,
        weights=values,
        relative_threshold=relative_threshold,
    )
