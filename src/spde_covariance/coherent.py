"""PSD-consistent observed kernels, before any covariance completion.

The shared variable is A=sqrt(W) K sqrt(W) on the observed pair mask. Clique
copies are constrained to A_j >= delta I. The Frobenius objective sums both
orientations of off-diagonal entries, consistently with the consensus penalty.
An explicit shared diagonal loading restores numerical primal feasibility;
it is recorded and is never a global projection of a completed kernel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .completion import _inputs, _snapshot, _symmetric_matrix


@dataclass(frozen=True, slots=True)
class CoherentPartialFit:
    """A feasible shared partial kernel and its optimization diagnostics.

    Failure to reach the requested budget leaves ``converged=False`` while
    retaining a feasible iterate for audit. The bound concerns distance to the
    observed-partial optimizer in weighted Frobenius coordinates, not a
    completed kernel, sampling error, H-operator error or source covariance.
    """

    partial_kernel: np.ndarray
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "partial_kernel", _snapshot(self.partial_kernel))
        object.__setattr__(self, "diagnostics", _snapshot(self.diagnostics))

    @property
    def converged(self):
        return self.diagnostics["converged"]

    @property
    def objective_gap_bound(self):
        return self.diagnostics["objective_gap_bound"]

    @property
    def partial_error_bound(self):
        return self.diagnostics["partial_error_bound"]


class CoherentConvergenceError(RuntimeError):
    """High-level failure retaining the feasible low-level partial fit."""

    def __init__(self, fit: CoherentPartialFit):
        self.fit = fit
        super().__init__(
            f"coherent partial fit did not converge: {fit.diagnostics['status']} "
            f"after {fit.diagnostics['iterations']} iterations; "
            f"weighted observed-partial error bound={fit.partial_error_bound:.6g}"
        )


def _scalar(value, name, *, positive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _problem(partial_kernel, pair_weights, cliques, weights, regularization, floor, budget):
    partial, groups, mask, quadrature, _ = _inputs(partial_kernel, cliques, weights, 1e-12)
    coverage = _symmetric_matrix(pair_weights)
    if coverage.shape != partial.shape or np.any(coverage < 0) or np.any(coverage[mask] <= 0):
        raise ValueError("pair_weights must be nonnegative and strictly positive on observed pairs")
    if np.any(coverage[~mask] != 0):
        raise ValueError("pair_weights outside the observed mask must be zero")
    regularization = _scalar(regularization, "regularization")
    floor = _scalar(floor, "eigenvalue_floor")
    budget = _scalar(budget, "numerical_error_budget", positive=True)
    root = np.sqrt(quadrature)
    target = np.zeros_like(partial)
    with np.errstate(over="ignore", invalid="ignore"):
        target[mask] = (root[:, None] * partial * root[None, :])[mask]
    if not np.all(np.isfinite(target)):
        raise ValueError("weighted observed kernel exceeds floating-point range")
    hessian = np.where(mask, coverage + regularization, 0.0)
    rhs = coverage * target
    indices = tuple(np.ix_(group, group) for group in groups)
    multiplicity = np.zeros_like(target)
    for index in indices:
        multiplicity[index] += 1
    return {
        "mask": mask,
        "cliques": groups,
        "indices": indices,
        "weights": quadrature,
        "target": target,
        "coverage": coverage,
        "hessian": hessian,
        "rhs": rhs,
        "multiplicity": multiplicity,
        "regularization": regularization,
        "floor": floor,
        "budget": budget,
        "mu": float(np.min(hessian[mask])),
    }


def _objective(matrix, problem):
    mask = problem["mask"]
    return float(
        np.sum(problem["coverage"][mask] * (matrix[mask] - problem["target"][mask]) ** 2) / 2
        + problem["regularization"] * np.sum(matrix[mask] ** 2) / 2
    )


def _feasible(matrix, problem):
    """Restore all clique constraints using one disclosed shared diagonal shift."""
    minimum, scale, size = np.inf, problem["floor"], 0
    for index in problem["indices"]:
        values = np.linalg.eigvalsh(matrix[index])
        minimum = min(minimum, float(values[0]))
        scale = max(scale, float(np.max(np.abs(values))))
        size = max(size, len(values))
    margin = 16 * np.finfo(float).eps * size * scale
    shift = max(0.0, problem["floor"] - minimum + margin)
    result = matrix.copy()
    result.flat[:: len(result) + 1] += shift
    return result, {
        "feasibility_diagonal_shift": shift,
        "feasibility_roundoff_margin": margin,
        "minimum_clique_eigenvalue_before_restoration": minimum,
        "minimum_clique_eigenvalue_after_restoration": minimum + shift,
        "restoration_objective_change": _objective(result, problem) - _objective(matrix, problem),
        "restoration_weighted_hs_change": float(shift * np.sqrt(len(result))),
        "restoration_scope": "uniform diagonal loading in weighted grid coordinates only; not global kernel PSD projection",
    }


def _certificate(matrix, duals, problem):
    """Evaluate a feasible primal upper and an NSD-dual lower bound stably."""
    feasible, restoration = _feasible(matrix, problem)
    total_dual = np.zeros_like(matrix)
    complementarity, product_scale, maximum_dual_shift = 0.0, 0.0, 0.0
    dual_trace, dual_cone_residual = 0.0, 0.0
    for index, candidate in zip(problem["indices"], duals, strict=True):
        # Y <=0 is required by L=f(Z)+<Y,PZ-X>, X>=floor I. Certificate
        # correction changes only the dual witness, never a fitted observation.
        values, vectors = np.linalg.eigh((candidate + candidate.T) / 2)
        cone_margin = 16 * np.finfo(float).eps * len(values) * float(np.max(np.abs(values)))
        dual = (vectors * np.minimum(values, 0.0)) @ vectors.T
        dual.flat[:: len(dual) + 1] -= cone_margin
        maximum_dual_shift = max(maximum_dual_shift, cone_margin)
        dual_cone_residual = max(dual_cone_residual, float(max(values[-1], 0.0)))
        total_dual[index] += dual
        local = feasible[index].copy()
        local.flat[:: len(local) + 1] -= problem["floor"]
        complementarity -= float(np.sum(dual * local))
        product_scale += float(np.sum(np.abs(dual * local)))
        dual_trace += float(np.trace(dual))
    mask, hessian, rhs = problem["mask"], problem["hessian"], problem["rhs"]
    stationary = np.zeros_like(matrix)
    stationary[mask] = (rhs[mask] - total_dual[mask]) / hessian[mask]
    square = float(np.sum(hessian[mask] * (feasible[mask] - stationary[mask]) ** 2) / 2)
    objective = _objective(feasible, problem)
    constant = float(np.sum(problem["coverage"][mask] * problem["target"][mask] ** 2) / 2)
    dual_objective = float(
        constant
        - np.sum((rhs[mask] - total_dual[mask]) ** 2 / hessian[mask]) / 2
        - problem["floor"] * dual_trace
    )
    stable_gap = square + complementarity
    roundoff = (
        64
        * np.finfo(float).eps
        * (product_scale + abs(square) + abs(objective) + abs(dual_objective) + abs(constant))
    )
    gap = max(0.0, stable_gap, objective - dual_objective) + roundoff
    return feasible, {
        **restoration,
        "objective": objective,
        "dual_objective": dual_objective,
        "objective_gap_bound": float(gap),
        "partial_error_bound": float(np.sqrt(2 * gap / problem["mu"])),
        "strong_convexity": problem["mu"],
        "stationarity_gap_component": square,
        "complementarity_gap_component": complementarity,
        "gap_roundoff_margin": roundoff,
        "dual_cone_correction": dual_cone_residual,
        "dual_roundoff_diagonal_shift": maximum_dual_shift,
        "certificate_scope": "weighted Frobenius distance on observed pairs to the coherent partial optimizer only",
        "certificate_arithmetic": "primal-dual weak duality; floating-point margins recorded, not interval arithmetic",
    }


def _result(matrix, certificate, problem, *, converged, iterations, status, **extra):
    root = np.sqrt(problem["weights"])
    kernel = np.full_like(matrix, np.nan)
    mask = problem["mask"]
    kernel[mask] = (matrix / root[:, None] / root[None, :])[mask]
    return CoherentPartialFit(
        kernel,
        {
            **certificate,
            **extra,
            "converged": bool(converged),
            "iterations": int(iterations),
            "status": status,
            "regularization": problem["regularization"],
            "eigenvalue_floor": problem["floor"],
            "numerical_error_budget": problem["budget"],
            "weights": problem["weights"],
            "observed_mask": mask,
            "cliques": problem["cliques"],
            "objective_convention": "one half sum over all ordered observed pairs, including both off-diagonal orientations",
            "coordinate_convention": "A=sqrt(W) K sqrt(W); delta I floor in weighted grid L2 coordinates, not H coordinates",
            "population_or_completion_error_certificate": False,
        },
    )


def fit_coherent_partial(
    partial_kernel,
    *,
    pair_weights,
    cliques,
    weights=None,
    regularization=0.0,
    eigenvalue_floor=0.0,
    numerical_error_budget=1e-5,
    rho=1.0,
    max_iterations=20_000,
    absolute_tolerance=1e-8,
    relative_tolerance=1e-7,
    check_every=10,
    adaptive_rho=True,
):
    """Fit jointly consistent PSD cliques by NumPy consensus ADMM.

    The objective is .5 sum_Omega c*(A-T)^2 + .5 lambda sum_Omega A^2,
    where c is supplied pair coverage, T is the corrected weighted kernel and
    every clique of A is constrained to >= eigenvalue_floor I. Positive date
    coverage must exist on each declared observed pair. No unobserved value,
    latent covariance or validation data is an input.

    Convergence requires the feasible primal-dual distance bound. ADMM
    residual tolerances are reported as independent algorithm diagnostics.
    A maximum-iteration result remains feasible, but has converged=False; the
    caller must not substitute it as a successful fit. Hard-rank completion
    afterward is a different estimator and need not be PSD.
    """
    problem = _problem(
        partial_kernel,
        pair_weights,
        cliques,
        weights,
        regularization,
        eigenvalue_floor,
        numerical_error_budget,
    )
    rho = _scalar(rho, "rho", positive=True)
    count = _integer(max_iterations, "max_iterations")
    interval = _integer(check_every, "check_every")
    absolute = _scalar(absolute_tolerance, "absolute_tolerance")
    relative = _scalar(relative_tolerance, "relative_tolerance")
    if not isinstance(adaptive_rho, (bool, np.bool_)):
        raise ValueError("adaptive_rho must be a boolean")
    initial_rho, adjustments = rho, 0
    matrix = np.zeros_like(problem["target"])
    local = [np.eye(len(group)) * problem["floor"] for group in problem["cliques"]]
    scaled_duals = [np.zeros_like(item) for item in local]
    denominator = problem["hessian"] + rho * problem["multiplicity"]
    mask = problem["mask"]
    primal, dual_residual = np.inf, np.inf
    primal_tolerance, dual_tolerance = 0.0, 0.0
    feasible, certificate = None, None
    for iteration in range(1, count + 1):
        numerator = problem["rhs"].copy()
        for index, item, dual in zip(problem["indices"], local, scaled_duals, strict=True):
            numerator[index] += rho * (item - dual)
        matrix[mask] = numerator[mask] / denominator[mask]
        changes = np.zeros_like(matrix)
        primal_squared, local_squared, shared_squared = 0.0, 0.0, 0.0
        for j, index in enumerate(problem["indices"]):
            candidate = matrix[index] + scaled_duals[j]
            values, vectors = np.linalg.eigh((candidate + candidate.T) / 2)
            projected = (vectors * np.maximum(values, problem["floor"])) @ vectors.T
            changes[index] += projected - local[j]
            # The negative spectral part produces the correct NSD dual sign.
            scaled_duals[j] = (vectors * np.minimum(values - problem["floor"], 0.0)) @ vectors.T
            local[j] = projected
            primal_squared += np.sum((matrix[index] - projected) ** 2)
            local_squared += np.sum(projected**2)
            shared_squared += np.sum(matrix[index] ** 2)
        primal = float(np.sqrt(primal_squared))
        dual_residual = float(rho * np.linalg.norm(changes))
        if iteration % interval == 0 or iteration == count:
            total_dual = np.zeros_like(matrix)
            for index, dual in zip(problem["indices"], scaled_duals, strict=True):
                total_dual[index] += rho * dual
            primal_tolerance = float(
                np.sqrt(sum(item.size for item in local)) * absolute
                + relative * max(np.sqrt(local_squared), np.sqrt(shared_squared))
            )
            dual_tolerance = float(
                np.sqrt(np.sum(mask)) * absolute + relative * np.linalg.norm(total_dual)
            )
            feasible, certificate = _certificate(
                matrix, [rho * value for value in scaled_duals], problem
            )
            if certificate["partial_error_bound"] <= problem["budget"]:
                break
            if adaptive_rho and adjustments < 30:
                new_rho = rho
                if primal > 10 * dual_residual:
                    new_rho = min(rho * 2, 1e6)
                elif dual_residual > 10 * primal:
                    new_rho = max(rho / 2, 1e-6)
                if new_rho != rho:
                    scaled_duals = [value * rho / new_rho for value in scaled_duals]
                    rho, adjustments = new_rho, adjustments + 1
                    denominator = problem["hessian"] + rho * problem["multiplicity"]
    converged = certificate["partial_error_bound"] <= problem["budget"]
    return _result(
        feasible,
        certificate,
        problem,
        converged=converged,
        iterations=iteration,
        status="converged"
        if converged
        else "maximum iterations reached without requested certificate",
        solver="consensus_admm",
        rho=rho,
        initial_rho=initial_rho,
        adaptive_rho=bool(adaptive_rho),
        rho_adjustments=adjustments,
        admm_residuals_within_tolerance=primal <= primal_tolerance
        and dual_residual <= dual_tolerance,
        convergence_rule="feasible primal and dual objective-gap distance certificate; ADMM residuals reported separately",
        primal_residual=primal,
        dual_residual=dual_residual,
        primal_tolerance=primal_tolerance,
        dual_tolerance=dual_tolerance,
        max_iterations=count,
        check_every=interval,
    )


def fit_coherent_partial_reference(
    partial_kernel,
    *,
    pair_weights,
    cliques,
    weights=None,
    regularization=0.0,
    eigenvalue_floor=0.0,
    numerical_error_budget=1e-5,
    solver_tolerance=1e-10,
    max_iterations=100_000,
):
    """Independent optional CVXPY/SCS formulation of the same clique objective."""
    import cvxpy as cp

    problem = _problem(
        partial_kernel,
        pair_weights,
        cliques,
        weights,
        regularization,
        eigenvalue_floor,
        numerical_error_budget,
    )
    tolerance = _scalar(solver_tolerance, "solver_tolerance", positive=True)
    count = _integer(max_iterations, "max_iterations")
    variable = cp.Variable(problem["target"].shape, symmetric=True)
    mask = problem["mask"]
    objective = (
        cp.sum(
            cp.multiply(
                problem["coverage"][mask], cp.square(variable[mask] - problem["target"][mask])
            )
        )
        / 2
        + problem["regularization"] * cp.sum_squares(variable[mask]) / 2
    )
    constraints = [
        variable[np.ix_(group, group)] >> problem["floor"] * np.eye(len(group))
        for group in problem["cliques"]
    ]
    optimization = cp.Problem(cp.Minimize(objective), constraints)
    optimization.solve(solver="SCS", eps=tolerance, max_iters=count)
    if variable.value is None or any(item.dual_value is None for item in constraints):
        raise RuntimeError(f"coherent reference solver failed: {optimization.status}")
    matrix = np.where(mask, (variable.value + variable.value.T) / 2, 0.0)
    feasible, certificate = _certificate(
        matrix, [-item.dual_value for item in constraints], problem
    )
    converged = (
        optimization.status in ("optimal", "optimal_inaccurate")
        and certificate["partial_error_bound"] <= problem["budget"]
    )
    return _result(
        feasible,
        certificate,
        problem,
        converged=converged,
        iterations=optimization.solver_stats.num_iters,
        status=optimization.status,
        solver="cvxpy_scs",
        solver_tolerance=tolerance,
        max_iterations=count,
    )
