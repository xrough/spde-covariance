"""Finite covariance ambiguity and minimum-variance hedging research primitives.

An ambiguity model declares ``Q >= 0, trace_lower <= trace(Q) <= trace_cap`` and the image
``K = offset_covariance + sum(F @ Q @ F.T for F in factors)``. Q uses orthonormal source
coordinates; K uses the explicitly chosen output coordinates, often grid
values. Identity factors give ordinary PSD covariance completion. None of the
solvers infer a source class, recover a physical source, or calibrate bands.

SDPs import CVXPY lazily (the ``reference`` extra). Returned bounds are
numerical identification ranges. A solver status and small residuals are not
an exact certificate, and an inaccurate solver status is never accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import lsq_linear

from .linalg import _real_array, validate_psd, validate_symmetric


def _snapshot(value):
    if isinstance(value, np.ndarray):
        # An immutable bytes backing also prevents callers re-enabling writes.
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(item) for item in value)
    return value


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


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True, eq=False)
class CovarianceAmbiguity:
    """Immutable finite source class and observed output-entry constraints.

    ``entry_pairs`` has shape (m, 2); each unordered pair may occur once.
    Equal lower/upper values impose equality. Bounds must be finite; omit
    unconstrained entries entirely. Trace bounds are on Q, not generally K.
    ``trace_lower=trace_cap`` explicitly declares known source trace; the
    default imposes no source trace lower bound.
    ``calibrated_bands`` records caller provenance only: this class performs
    no statistical calibration or coverage verification. An optional known
    PSD ``offset_covariance`` represents a fixed background, such as a shared
    infinite source tail; uncertainty then concerns only the finite Q block.
    """

    factors: Sequence[ArrayLike]
    entry_pairs: ArrayLike
    lower: ArrayLike
    upper: ArrayLike
    trace_cap: float
    calibrated_bands: bool = False
    offset_covariance: ArrayLike | None = None
    trace_lower: float = 0.0

    def __post_init__(self):
        factors = tuple(_real_array(factor, name="factor") for factor in self.factors)
        if not factors or any(factor.ndim != 2 or not all(factor.shape) for factor in factors):
            raise ValueError("factors must be a nonempty sequence of nonempty matrices")
        if any(factor.shape != factors[0].shape for factor in factors):
            raise ValueError("factors must share their output and source dimensions")
        raw_pairs = np.asarray(self.entry_pairs)
        if raw_pairs.size == 0:
            pairs = np.empty((0, 2), dtype=int)
        elif raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2 or raw_pairs.dtype.kind not in "iu":
            raise ValueError("entry_pairs must be an integer array with shape (m, 2)")
        else:
            pairs = np.sort(raw_pairs.astype(int), axis=1)
        if np.any(pairs < 0) or np.any(pairs >= factors[0].shape[0]):
            raise ValueError("entry_pairs must index the output covariance")
        if len(set(map(tuple, pairs))) != len(pairs):
            raise ValueError("entry_pairs must not repeat an unordered pair")
        lower = _real_array(self.lower, name="lower")
        upper = _real_array(self.upper, name="upper")
        if lower.shape != (len(pairs),) or upper.shape != lower.shape or np.any(lower > upper):
            raise ValueError("lower and upper must match entry_pairs and satisfy lower <= upper")
        if not isinstance(self.calibrated_bands, (bool, np.bool_)):
            raise ValueError("calibrated_bands must be boolean")
        offset = np.zeros((factors[0].shape[0], factors[0].shape[0]))
        if self.offset_covariance is not None:
            offset = validate_psd(self.offset_covariance, name="offset_covariance")
            if offset.shape != (factors[0].shape[0], factors[0].shape[0]):
                raise ValueError("offset_covariance must match the output dimension")
        for name, value in (
            ("factors", factors),
            ("entry_pairs", pairs),
            ("lower", lower),
            ("upper", upper),
            ("offset_covariance", offset),
        ):
            object.__setattr__(self, name, _snapshot(value))
        object.__setattr__(self, "trace_cap", _scalar(self.trace_cap, "trace_cap"))
        trace_lower = _scalar(self.trace_lower, "trace_lower")
        if trace_lower > self.trace_cap:
            raise ValueError("trace_lower must not exceed trace_cap")
        object.__setattr__(self, "trace_lower", trace_lower)
        object.__setattr__(self, "calibrated_bands", bool(self.calibrated_bands))

    @classmethod
    def from_covariance(
        cls,
        covariance: ArrayLike,
        observed_mask: ArrayLike,
        trace_cap: float,
        *,
        factors: Sequence[ArrayLike] | None = None,
        entry_tolerance: ArrayLike = 0.0,
        calibrated_bands: bool = False,
        offset_covariance: ArrayLike | None = None,
        trace_lower: float = 0.0,
    ) -> CovarianceAmbiguity:
        """Read only observed entries, even when other entries contain truth.

        Unobserved values, including NaNs and asymmetric fillers, are ignored.
        ``entry_tolerance`` is a scalar or a symmetric matrix of nonnegative
        half-widths on the observed entries. A zero width imposes equality.
        """
        raw = np.asarray(covariance)
        if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
            raise ValueError("covariance must be real numeric")
        if raw.ndim != 2 or not raw.shape[0] or raw.shape[0] != raw.shape[1]:
            raise ValueError("covariance must be a nonempty square matrix")
        mask = np.asarray(observed_mask)
        if mask.dtype.kind != "b" or mask.shape != raw.shape or not np.array_equal(mask, mask.T):
            raise ValueError("observed_mask must be boolean, symmetric and match covariance")
        if not np.all(np.isfinite(raw[mask])):
            raise ValueError("observed covariance entries must be finite")
        if not np.allclose(raw[mask], raw.T[mask], rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric on observed entries")
        widths = np.asarray(entry_tolerance)
        if widths.ndim == 0:
            widths = np.full(raw.shape, _scalar(widths.item(), "entry_tolerance"))
        elif widths.shape != raw.shape or np.iscomplexobj(widths):
            raise ValueError("entry_tolerance must be a scalar or match covariance")
        if (
            not np.all(np.isfinite(widths[mask]))
            or np.any(widths[mask] < 0)
            or not np.allclose(widths[mask], widths.T[mask], rtol=1e-10, atol=1e-12)
        ):
            raise ValueError(
                "observed entry_tolerance values must be finite, symmetric, nonnegative"
            )
        rows, columns = np.where(np.triu(mask))
        values = (raw[rows, columns] + raw[columns, rows]) / 2
        local_widths = widths[rows, columns]
        selected_factors = (np.eye(len(raw)),) if factors is None else factors
        model = cls(
            selected_factors,
            np.column_stack([rows, columns]),
            values - local_widths,
            values + local_widths,
            trace_cap,
            calibrated_bands,
            offset_covariance,
            trace_lower,
        )
        if model.output_dimension != len(raw):
            raise ValueError("factor output dimension must match covariance")
        return model

    @property
    def source_dimension(self):
        return self.factors[0].shape[1]

    @property
    def output_dimension(self):
        return self.factors[0].shape[0]

    @property
    def entry_matrices(self):
        """Symmetric A_i such that trace(A_i Q) is the observed K entry."""
        result = np.zeros((len(self.entry_pairs), self.source_dimension, self.source_dimension))
        for index, (left, right) in enumerate(self.entry_pairs):
            for factor in self.factors:
                product = np.outer(factor[left], factor[right])
                result[index] += (product + product.T) / 2
        return _snapshot(result)

    def covariance(self, source_covariance: ArrayLike):
        """Apply the declared linear image; the input need not be feasible."""
        source = validate_symmetric(source_covariance, name="source_covariance")
        if source.shape != (self.source_dimension, self.source_dimension):
            raise ValueError("source_covariance must match the source dimension")
        result = self.offset_covariance + sum(factor @ source @ factor.T for factor in self.factors)
        return _snapshot((result + result.T) / 2)

    def residuals(self, source_covariance: ArrayLike):
        """Absolute constraint residuals and a dimensionless maximum violation."""
        source = validate_symmetric(source_covariance, name="source_covariance")
        covariance = self.covariance(source)
        entries = covariance[tuple(self.entry_pairs.T)]
        equal = self.lower == self.upper
        equality = float(np.max(np.abs(entries[equal] - self.lower[equal]), initial=0))
        band = float(np.max(np.maximum(self.lower - entries, entries - self.upper), initial=0))
        psd = max(0.0, -float(np.linalg.eigvalsh(source)[0]))
        trace = max(0.0, float(np.trace(source)) - self.trace_cap)
        trace_lower = max(0.0, self.trace_lower - float(np.trace(source)))
        entry_scale = 1 + max(
            float(np.max(np.abs(self.lower), initial=0)),
            float(np.max(np.abs(self.upper), initial=0)),
        )
        source_scale = 1 + self.trace_cap
        return _snapshot(
            {
                "equality_residual": equality,
                "entry_bound_violation": band,
                "psd_violation": psd,
                "trace_violation": trace,
                "trace_lower_violation": trace_lower,
                "minimum_source_eigenvalue": float(np.linalg.eigvalsh(source)[0]),
                "source_trace": float(np.trace(source)),
                "scaled_max_violation": max(
                    band / entry_scale,
                    psd / source_scale,
                    trace / source_scale,
                    trace_lower / source_scale,
                ),
            }
        )


def transport_covariance_factors(
    propagator, delta: float, *, evaluation: ArrayLike | None = None, quadrature_order: int = 24
):
    """Positive Gauss factors for normalized innovation, including evaluation.

    Returned F_l approximate ``delta^-1 integral E S(t) Q S(t).T E.T dt``.
    The normalization is inside each factor as sqrt(weight/2); do not apply
    delta a second time. This explicit quadrature is its own declared finite
    model, which may differ from a propagator's analytic or split quadrature.
    """
    duration = _scalar(delta, "delta", positive=True)
    order = _integer(quadrature_order, "quadrature_order")
    dimension = _integer(getattr(propagator, "dimension", None), "propagator dimension")
    if not callable(getattr(propagator, "matrix", None)):
        raise ValueError("propagator must provide matrix")
    evaluate = (
        np.eye(dimension) if evaluation is None else _real_array(evaluation, name="evaluation")
    )
    if evaluate.ndim != 2 or not evaluate.shape[0] or evaluate.shape[1] != dimension:
        raise ValueError("evaluation must have one column per source coordinate")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    factors = []
    for node, weight in zip(nodes, weights, strict=True):
        matrix = _real_array(propagator.matrix(duration * (node + 1) / 2), name="transport matrix")
        if matrix.shape != (dimension, dimension):
            raise ValueError("transport matrix must match propagator dimension")
        factors.append(np.sqrt(weight / 2) * evaluate @ matrix)
    return _snapshot(factors)


@dataclass(frozen=True, slots=True, eq=False)
class CovarianceFunctionalResult:
    """Numerical support witness; consult available before using its value."""

    value: float | None
    source_covariance: np.ndarray | None
    covariance: np.ndarray | None
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("source_covariance", "covariance", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))

    @property
    def available(self):
        return self.diagnostics["available"]


@dataclass(frozen=True, slots=True, eq=False)
class CovarianceFunctionalInterval:
    """Attainable numerical endpoints, conditional on the declared model."""

    lower: float | None
    upper: float | None
    minimizer: CovarianceFunctionalResult
    maximizer: CovarianceFunctionalResult
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", _snapshot(self.diagnostics))

    @property
    def available(self):
        return self.diagnostics["available"]


@dataclass(frozen=True, slots=True, eq=False)
class HedgeResult:
    """Finite positions minimizing residual variance plus declared penalties."""

    positions: np.ndarray | None
    risk: float | None
    objective: float | None
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("positions", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))

    @property
    def available(self):
        return self.diagnostics["available"]


def _cvxpy():
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ImportError(
            "Install the 'reference' extra to use covariance ambiguity SDPs"
        ) from error
    return cp


def _solver_options(solver, tolerance, max_iterations):
    tolerance = _scalar(tolerance, "tolerance", positive=True)
    max_iterations = _integer(max_iterations, "max_iterations")
    if not isinstance(solver, str) or not solver:
        raise ValueError("solver must be a nonempty solver name")
    if solver.upper() == "SCS":
        return {"solver": solver, "eps": tolerance, "max_iters": max_iterations}
    if solver.upper() == "CLARABEL":
        return {
            "solver": solver,
            "tol_gap_abs": tolerance,
            "tol_feas": tolerance,
            "tol_gap_rel": tolerance,
            "max_iter": max_iterations,
        }
    raise ValueError("solver must be SCS or CLARABEL, which support the required PSD cones")


def _exposure(value, dimension, name):
    array = _real_array(value, name=name)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have one entry per output coordinate")
    return array


def _constraint_system(model):
    """Whiten equality rows without changing their finite linear constraint set.

    SVD removes numerically duplicate rows at a machine precision threshold.
    The original data constraints remain the basis for all residual checks.
    This avoids presenting ill-conditioned evaluation-entry rows directly to
    an SDP solver; inequality rows are only individually rescaled.
    """
    matrices = model.entry_matrices
    offset_entries = model.offset_covariance[tuple(model.entry_pairs.T)]
    lower, upper = model.lower - offset_entries, model.upper - offset_entries
    equal = lower == upper
    rows = []
    if np.any(equal):
        flat = matrices[equal].reshape(np.count_nonzero(equal), -1)
        left, singular, right = np.linalg.svd(flat, full_matrices=False)
        threshold = np.finfo(float).eps * max(flat.shape) * float(singular[0])
        rank = int(np.count_nonzero(singular > threshold))
        values = (left[:, :rank].T @ lower[equal]) / singular[:rank]
        for matrix, value in zip(right[:rank], values, strict=True):
            rows.append(
                (matrix.reshape(model.source_dimension, model.source_dimension), value, value)
            )
        remainder = lower[equal] - left[:, :rank] @ (left[:, :rank].T @ lower[equal])
        consistency_tolerance = 100 * np.finfo(float).eps * (1 + np.linalg.norm(lower[equal]))
        if np.max(np.abs(remainder), initial=0) > consistency_tolerance:
            # Explicitly preserve infeasibility from contradictory duplicate rows.
            rows.append((np.zeros((model.source_dimension, model.source_dimension)), 1.0, 1.0))
    else:
        threshold, rank = 0.0, 0
    for matrix, lo, hi in zip(matrices[~equal], lower[~equal], upper[~equal], strict=True):
        scale = float(np.linalg.norm(matrix)) or 1.0
        rows.append((matrix / scale, lo / scale, hi / scale))
    return rows, {
        "equality_rank": rank,
        "equality_rank_threshold": threshold,
        "equality_row_count": int(np.count_nonzero(equal)),
    }


def _source_constraints(cp, model, source, rows):
    constraints = [source >> 0]
    if model.trace_lower == model.trace_cap:
        constraints.append(cp.trace(source) == model.trace_cap)
    else:
        constraints.append(cp.trace(source) <= model.trace_cap)
        if model.trace_lower > 0:
            constraints.append(cp.trace(source) >= model.trace_lower)
    for matrix, lower, upper in rows:
        expression = cp.sum(cp.multiply(matrix, source))
        if lower == upper:
            constraints.append(expression == lower)
        else:
            constraints.extend([expression >= lower, expression <= upper])
    return constraints


def solve_covariance_functional(
    model: CovarianceAmbiguity,
    left: ArrayLike,
    right: ArrayLike | None = None,
    *,
    maximize: bool = True,
    solver: str = "SCS",
    tolerance: float = 1e-7,
    feasibility_tolerance: float = 1e-6,
    max_iterations: int = 100_000,
) -> CovarianceFunctionalResult:
    """Optimize left.T K right; omitted right gives a scalar variance.

    Candidates from inaccurate or failed statuses remain diagnostic witnesses
    only. The returned value is recomputed from Q and the declared image.
    """
    if not isinstance(model, CovarianceAmbiguity):
        raise ValueError("model must be CovarianceAmbiguity")
    if not isinstance(maximize, (bool, np.bool_)):
        raise ValueError("maximize must be boolean")
    left = _exposure(left, model.output_dimension, "left")
    right = left if right is None else _exposure(right, model.output_dimension, "right")
    options = _solver_options(solver, tolerance, max_iterations)
    feasibility_tolerance = _scalar(feasibility_tolerance, "feasibility_tolerance", positive=True)
    matrix = np.zeros((model.source_dimension, model.source_dimension))
    for factor in model.factors:
        product = np.outer(factor.T @ left, factor.T @ right)
        matrix += (product + product.T) / 2
    cp = _cvxpy()
    source = cp.Variable((model.source_dimension, model.source_dimension), symmetric=True)
    expression = cp.sum(cp.multiply(matrix, source)) + float(left @ model.offset_covariance @ right)
    objective = cp.Maximize(expression) if maximize else cp.Minimize(expression)
    rows, conditioning = _constraint_system(model)
    problem = cp.Problem(objective, _source_constraints(cp, model, source, rows))
    try:
        problem.solve(**options)
    except cp.error.SolverError as error:
        return CovarianceFunctionalResult(
            None,
            None,
            None,
            {
                "available": False,
                "status": "solver_error",
                "solver": solver,
                "error": str(error),
                "numerical_certificate": False,
                "feasibility_tolerance": feasibility_tolerance,
                "constraint_conditioning": conditioning,
            },
        )
    candidate = None if source.value is None else (source.value + source.value.T) / 2
    covariance = None if candidate is None else model.covariance(candidate)
    value = None if covariance is None else float(left @ covariance @ right)
    residuals = None if candidate is None else model.residuals(candidate)
    available = bool(
        problem.status == cp.OPTIMAL
        and residuals is not None
        and residuals["scaled_max_violation"] <= feasibility_tolerance
        and value is not None
        and np.isfinite(value)
    )
    diagnostics = {
        "available": available,
        "status": problem.status,
        "solver": solver,
        "maximize": bool(maximize),
        "solver_objective": problem.value,
        "objective_recomputation_error": None if value is None else abs(value - problem.value),
        "feasibility_tolerance": feasibility_tolerance,
        "feasibility": residuals,
        "numerical_certificate": False,
        "calibrated_bands": model.calibrated_bands,
        "constraint_conditioning": conditioning,
        "interpretation": "numerical identification bound conditional on supplied constraints",
    }
    return CovarianceFunctionalResult(value, candidate, covariance, diagnostics)


def covariance_functional_interval(model, left, right=None, **solver_options):
    """Numerical attainable variance/cross-covariance interval, with witnesses."""
    minimum = solve_covariance_functional(model, left, right, maximize=False, **solver_options)
    maximum = solve_covariance_functional(model, left, right, maximize=True, **solver_options)
    available = minimum.available and maximum.available
    lower = minimum.value if minimum.available else None
    upper = maximum.value if maximum.available else None
    return CovarianceFunctionalInterval(
        lower,
        upper,
        minimum,
        maximum,
        {
            "available": available,
            "numerical_certificate": False,
            "calibrated_bands": model.calibrated_bands,
            "statistical_coverage_verified": False,
            "interpretation": "numerical identification interval; input calibration is caller supplied",
        },
    )


def _hedge_inputs(target, instruments, dimension, position_bounds):
    target = _exposure(target, dimension, "target")
    instruments = _real_array(instruments, name="instruments")
    if instruments.ndim != 2 or instruments.shape[0] != dimension or not instruments.shape[1]:
        raise ValueError(
            "instruments must have one row per output coordinate and at least one column"
        )
    count = instruments.shape[1]
    if position_bounds is None:
        lower, upper = np.full(count, -np.inf), np.full(count, np.inf)
    else:
        try:
            bounds = tuple(position_bounds)
        except TypeError as error:
            raise ValueError("position_bounds must contain (lower, upper)") from error
        if len(bounds) != 2:
            raise ValueError("position_bounds must contain (lower, upper)")
        if any(np.iscomplexobj(value) for value in bounds):
            raise ValueError("position bounds must be real")
        lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (count,)).copy()
        upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (count,)).copy()
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("position bounds must be ordered and cannot contain NaN")
        if np.any(np.isposinf(lower)) or np.any(np.isneginf(upper)):
            raise ValueError("position bounds must permit finite positions")
    return target, instruments, lower, upper


def _cost_inputs(count, holding_costs, turnover_costs, initial_positions):
    def costs(value, name):
        array = np.zeros(count) if value is None else _real_array(value, name=name)
        if array.shape != (count,) or np.any(array < 0):
            raise ValueError(f"{name} must be a nonnegative vector matching the instruments")
        return array

    holding = costs(holding_costs, "holding_costs")
    turnover = costs(turnover_costs, "turnover_costs")
    initial = (
        np.zeros(count)
        if initial_positions is None
        else _real_array(initial_positions, name="initial_positions")
    )
    if initial.shape != (count,):
        raise ValueError("initial_positions must match the instruments")
    return holding, turnover, initial


def _position_constraints(positions, lower, upper):
    result = []
    if np.any(np.isfinite(lower)):
        finite = np.isfinite(lower)
        result.append(positions[finite] >= lower[finite])
    if np.any(np.isfinite(upper)):
        finite = np.isfinite(upper)
        result.append(positions[finite] <= upper[finite])
    return result


def solve_robust_hedge(
    model: CovarianceAmbiguity,
    target: ArrayLike,
    instruments: ArrayLike,
    *,
    position_bounds=None,
    risk_weight: float = 1.0,
    holding_costs: ArrayLike | None = None,
    turnover_costs: ArrayLike | None = None,
    initial_positions: ArrayLike | None = None,
    solver: str = "SCS",
    tolerance: float = 1e-7,
    feasibility_tolerance: float = 1e-6,
    max_iterations: int = 100_000,
) -> HedgeResult:
    """Minimize worst-case residual variance using a dual SDP Schur block.

    Columns of B are finite tradable payoff exposures. Residual exposure is
    ``target - B @ positions``. Optional penalties are linear absolute holding
    and turnover costs in the caller's chosen objective units. A separate
    primal support solve checks the candidate's worst-case risk and dual gap.
    """
    if not isinstance(model, CovarianceAmbiguity):
        raise ValueError("model must be CovarianceAmbiguity")
    target, instruments, lower, upper = _hedge_inputs(
        target, instruments, model.output_dimension, position_bounds
    )
    weight = _scalar(risk_weight, "risk_weight", positive=True)
    options = _solver_options(solver, tolerance, max_iterations)
    feasible_tol = _scalar(feasibility_tolerance, "feasibility_tolerance", positive=True)
    holding, turnover, initial = _cost_inputs(
        instruments.shape[1], holding_costs, turnover_costs, initial_positions
    )
    # An empty primal ambiguity set makes its formal support dual misleading.
    feasibility = solve_covariance_functional(
        model,
        np.zeros(model.output_dimension),
        solver=solver,
        tolerance=tolerance,
        feasibility_tolerance=feasible_tol,
        max_iterations=max_iterations,
    )
    if not feasibility.available:
        return HedgeResult(
            None,
            None,
            None,
            {
                "available": False,
                "status": "ambiguity_feasibility_unverified",
                "ambiguity_feasibility_status": feasibility.diagnostics["status"],
                "ambiguity_feasibility": feasibility.diagnostics,
                "numerical_certificate": False,
            },
        )
    cp = _cvxpy()
    positions = cp.Variable(instruments.shape[1])
    z = cp.Variable(nonneg=model.trace_lower != model.trace_cap)
    top = z * np.eye(model.source_dimension)
    bound = model.trace_cap * z
    if 0 < model.trace_lower < model.trace_cap:
        lower_trace_multiplier = cp.Variable(nonneg=True)
        top = top - lower_trace_multiplier * np.eye(model.source_dimension)
        bound = bound - model.trace_lower * lower_trace_multiplier
    rows, conditioning = _constraint_system(model)
    for matrix, lo, hi in rows:
        if lo == hi:
            multiplier = cp.Variable()
            top = top + multiplier * matrix
            bound = bound + hi * multiplier
        else:
            plus, minus = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
            top = top + (plus - minus) * matrix
            bound = bound + hi * plus - lo * minus
    residual = target - instruments @ positions
    columns = cp.hstack(
        [
            cp.reshape(factor.T @ residual, (model.source_dimension, 1), order="C")
            for factor in model.factors
        ]
    )
    block = cp.bmat([[top, columns], [columns.T, np.eye(len(model.factors))]])
    offset_eigenvalues, offset_eigenvectors = np.linalg.eigh(model.offset_covariance)
    offset_root = np.sqrt(np.maximum(offset_eigenvalues, 0))[:, None] * offset_eigenvectors.T
    bound = bound + cp.sum_squares(offset_root @ residual)
    penalty = holding @ cp.abs(positions) + turnover @ cp.abs(positions - initial)
    constraints = [block >> 0, *_position_constraints(positions, lower, upper)]
    problem = cp.Problem(cp.Minimize(weight * bound + penalty), constraints)
    try:
        problem.solve(**options)
    except cp.error.SolverError as error:
        return HedgeResult(
            None,
            None,
            None,
            {
                "available": False,
                "status": "solver_error",
                "solver": solver,
                "error": str(error),
                "numerical_certificate": False,
            },
        )
    if positions.value is None:
        return HedgeResult(
            None,
            None,
            None,
            {
                "available": False,
                "status": problem.status,
                "solver": solver,
                "numerical_certificate": False,
            },
        )
    values = np.asarray(positions.value).reshape(-1)
    support = solve_covariance_functional(
        model,
        target - instruments @ values,
        solver=solver,
        tolerance=tolerance,
        feasibility_tolerance=feasible_tol,
        max_iterations=max_iterations,
    )
    risk = support.value
    cost = float(holding @ np.abs(values) + turnover @ np.abs(values - initial))
    objective = None if risk is None else weight * risk + cost
    dual_risk = float(bound.value)
    gap = None if risk is None else dual_risk - risk
    block_values = np.asarray(block.value)
    block_violation = max(0.0, -float(np.linalg.eigvalsh((block_values + block_values.T) / 2)[0]))
    position_violation = float(np.max(np.maximum(lower - values, values - upper), initial=0))
    scaled_block = block_violation / (1 + float(np.linalg.norm(block_values, ord=2)))
    scaled_gap = None if gap is None else abs(gap) / (1 + abs(risk) + abs(dual_risk))
    available = bool(
        problem.status == cp.OPTIMAL
        and support.available
        and scaled_block <= feasible_tol
        and position_violation <= feasible_tol
        and scaled_gap is not None
        and scaled_gap <= 10 * feasible_tol
    )
    return HedgeResult(
        values,
        risk,
        objective,
        {
            "available": available,
            "status": problem.status,
            "solver": solver,
            "solver_objective": problem.value,
            "risk_weight": weight,
            "penalty": cost,
            "dual_risk_bound_candidate": dual_risk,
            "support_risk_candidate": risk,
            "support_status": support.diagnostics["status"],
            "support_available": support.available,
            "support_feasibility": support.diagnostics["feasibility"],
            "support_crosscheck_gap": gap,
            "scaled_support_crosscheck_gap": scaled_gap,
            "schur_psd_violation": block_violation,
            "scaled_schur_psd_violation": scaled_block,
            "position_bound_violation": position_violation,
            "feasibility_tolerance": feasible_tol,
            "objective_recomputation_error": None
            if objective is None
            else abs(problem.value - objective),
            "numerical_certificate": False,
            "calibrated_bands": model.calibrated_bands,
            "constraint_conditioning": conditioning,
        },
    )


def solve_plugin_hedge(
    covariance: ArrayLike,
    target: ArrayLike,
    instruments: ArrayLike,
    *,
    position_bounds=None,
    risk_weight: float = 1.0,
    holding_costs: ArrayLike | None = None,
    turnover_costs: ArrayLike | None = None,
    initial_positions: ArrayLike | None = None,
    solver: str = "SCS",
    tolerance: float = 1e-7,
    max_iterations: int = 100_000,
) -> HedgeResult:
    """Point covariance minimum variance hedge, including singular PSD inputs.

    Plain and box constrained least squares need no CVXPY. Nonzero convex
    absolute costs use CVXPY. risk_weight multiplies variance in the objective;
    the reported risk always remains the unweighted residual variance.
    """
    covariance = validate_psd(covariance, name="covariance")
    target, instruments, lower, upper = _hedge_inputs(
        target, instruments, len(covariance), position_bounds
    )
    weight = _scalar(risk_weight, "risk_weight", positive=True)
    tolerance = _scalar(tolerance, "tolerance", positive=True)
    max_iterations = _integer(max_iterations, "max_iterations")
    holding, turnover, initial = _cost_inputs(
        instruments.shape[1], holding_costs, turnover_costs, initial_positions
    )
    eig, vectors = np.linalg.eigh(covariance)
    square_root = np.sqrt(np.maximum(eig, 0))[:, None] * vectors.T
    design, response = square_root @ instruments, square_root @ target
    if np.any(holding) or np.any(turnover):
        cp = _cvxpy()
        positions = cp.Variable(instruments.shape[1])
        expression = weight * cp.sum_squares(design @ positions - response)
        expression += holding @ cp.abs(positions) + turnover @ cp.abs(positions - initial)
        problem = cp.Problem(
            cp.Minimize(expression), _position_constraints(positions, lower, upper)
        )
        try:
            problem.solve(**_solver_options(solver, tolerance, max_iterations))
        except cp.error.SolverError as error:
            return HedgeResult(
                None,
                None,
                None,
                {
                    "available": False,
                    "status": "solver_error",
                    "solver": solver,
                    "error": str(error),
                    "numerical_certificate": False,
                },
            )
        if positions.value is None:
            return HedgeResult(
                None,
                None,
                None,
                {
                    "available": False,
                    "status": problem.status,
                    "numerical_certificate": False,
                },
            )
        values, status = positions.value, problem.status
        available = status == cp.OPTIMAL
    elif np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
        values = np.linalg.lstsq(design, response, rcond=None)[0]
        status, available = "least_squares", True
    else:
        # scipy requires strict bounds; eliminate fixed positions first.
        fixed = lower == upper
        values = np.zeros(instruments.shape[1])
        values[fixed] = lower[fixed]
        if np.any(~fixed):
            fit = lsq_linear(
                design[:, ~fixed],
                response - design[:, fixed] @ values[fixed],
                bounds=(lower[~fixed], upper[~fixed]),
                tol=tolerance,
                max_iter=max_iterations,
            )
            values[~fixed] = fit.x
            status, available = f"bounded_least_squares:{fit.status}", bool(fit.success)
        else:
            status, available = "fixed_positions", True
    residual = target - instruments @ values
    risk = float(residual @ covariance @ residual)
    cost = float(holding @ np.abs(values) + turnover @ np.abs(values - initial))
    position_violation = float(np.max(np.maximum(lower - values, values - upper), initial=0))
    return HedgeResult(
        values,
        risk,
        weight * risk + cost,
        {
            "available": bool(available and position_violation <= 10 * tolerance),
            "status": status,
            "risk_weight": weight,
            "penalty": cost,
            "position_bound_violation": position_violation,
            "numerical_certificate": False,
        },
    )


def solve_target_block_hedge(
    target_variance: float,
    target_cross_covariance: ArrayLike,
    hedge_covariance: ArrayLike,
    **hedge_options,
) -> HedgeResult:
    """Direct baseline using a fully jointly measured target/hedge block.

    The caller must supply the target-instrument cross covariances from joint
    measurements. This baseline does not complete unobserved cross entries.
    """
    variance = _scalar(target_variance, "target_variance")
    hedge = validate_psd(hedge_covariance, name="hedge_covariance")
    cross = _exposure(target_cross_covariance, len(hedge), "target_cross_covariance")
    joint = np.block([[np.array([[variance]]), cross[None, :]], [cross[:, None], hedge]])
    validate_psd(joint, name="joint target and hedge covariance")
    target = np.eye(len(joint))[:, 0]
    instruments = np.eye(len(joint))[:, 1:]
    return solve_plugin_hedge(joint, target, instruments, **hedge_options)
