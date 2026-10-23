"""Covariance baselines and observed-data-only trailing-window selection.

Empirical and linear-shrinkage estimators require identity observation maps.
Their native noise-corrected matrices are not silently projected or trace
capped. Ridge methods share a declared cap; relative ridge strength is scaled
by the training window's lifted Gram operator norm. No API accepts latent truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimation import CovarianceEstimate, fit_covariance
from .geometry import gram_operator, smat, svec
from .linalg import _real_array, project_psd_trace, validate_symmetric
from .observations import Observation

FloatArray = NDArray[np.float64]
_METHODS = {"constrained_ridge", "projected_ridge", "empirical", "linear_shrinkage"}


@dataclass(frozen=True, slots=True)
class Selection:
    """Selected covariance, fitted parameters, and complete candidate audit.

    ``estimate`` is the constrained solver's certificate-bearing result, or
    None for closed-form baselines. After ``refit_selection``, validation_score
    and candidates retain the original selection audit: no new validation data
    have been supplied, so they do not score the refitted covariance.
    """

    covariance: FloatArray
    method: str
    selected_parameters: dict[str, Any]
    validation_score: float
    candidates: tuple[dict[str, Any], ...]
    estimate: CovarianceEstimate | None

    def __post_init__(self) -> None:
        covariance = validate_symmetric(self.covariance, name="covariance")
        covariance.flags.writeable = False
        if self.method not in _METHODS:
            raise ValueError(f"unknown covariance method: {self.method}")
        if not np.isfinite(self.validation_score):
            raise ValueError("validation_score must be finite")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "selected_parameters", deepcopy(dict(self.selected_parameters)))
        object.__setattr__(self, "candidates", tuple(deepcopy(self.candidates)))


def _data(observations: Sequence[Observation]) -> tuple[Observation, ...]:
    data = tuple(observations)
    if not data:
        raise ValueError("at least one observation is required")
    if any(not isinstance(item, Observation) for item in data):
        raise ValueError("observations must contain Observation objects")
    if any(item.dimension != data[0].dimension for item in data):
        raise ValueError("all observation operators must use the same curve dimension")
    return data


def _weights(count: int, weights: ArrayLike | None) -> FloatArray:
    if weights is None:
        return np.full(count, 1 / count)
    values = _real_array(weights, name="weights")
    if values.shape != (count,) or np.any(values < 0) or not np.any(values > 0):
        raise ValueError("weights must be nonnegative, nonzero, and match the observations")
    # Scaling first avoids overflow when otherwise finite weights have a large sum.
    scaled = values / np.max(values)
    return scaled / np.sum(scaled)


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        boundary = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {boundary}")
    return result


def _require_identity(data: tuple[Observation, ...]) -> None:
    identity = np.eye(data[0].dimension)
    if any(not np.array_equal(item.operator, identity) for item in data):
        raise ValueError("this baseline requires identity observation maps (O=I) at every date")


def empirical_covariance(
    observations: Sequence[Observation], weights: ArrayLike | None = None
) -> FloatArray:
    """Weighted mean of corrected innovation outer products for full O=I data.

    This is PSD for noiseless data. Finite-sample noise correction can make the
    native matrix indefinite; no projection or trace restriction is applied.
    """
    data = _data(observations)
    _require_identity(data)
    values = _weights(len(data), weights)
    result = sum(weight * item.measurement for item, weight in zip(data, values, strict=True))
    return (result + result.T) / 2


def linear_shrinkage_covariance(
    observations: Sequence[Observation], shrinkage: float, weights: ArrayLike | None = None
) -> FloatArray:
    """Shrink the full-observation empirical matrix toward its scaled identity."""
    alpha = _scalar(shrinkage, "shrinkage")
    if alpha > 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    empirical = empirical_covariance(observations, weights)
    identity_target = np.trace(empirical) / len(empirical) * np.eye(len(empirical))
    return (1 - alpha) * empirical + alpha * identity_target


def projected_ridge_covariance(
    observations: Sequence[Observation],
    *,
    regularization: float,
    trace_cap: float,
    weights: ArrayLike | None = None,
) -> FloatArray:
    """Fit unconstrained symmetric ridge, then Euclidean-project to PSD/trace.

    This postprojection generally differs from optimizing over the constrained
    set in the observation metric. It is a baseline, not the constrained solver.
    """
    regularization = _scalar(regularization, "regularization", positive=True)
    trace_cap = _scalar(trace_cap, "trace_cap")
    data = _data(observations)
    values = _weights(len(data), weights)
    gram = gram_operator([item.operator for item in data], values)
    rhs = sum(
        weight * item.operator.T @ item.measurement @ item.operator
        for item, weight in zip(data, values, strict=True)
    )
    hessian = gram + regularization * np.eye(len(gram))
    unconstrained = smat(np.linalg.solve(hessian, svec(rhs)))
    return project_psd_trace(unconstrained, trace_cap)


def validation_score(covariance: ArrayLike, observations: Sequence[Observation]) -> float:
    """Observed covariance score with the candidate-independent constant omitted.

    Mean score is ``||OBO.T||_F^2 - 2 <Z,OBO.T>``. Corrected outer products Z
    may be indefinite; there is no positivity projection of validation labels.
    """
    matrix = validate_symmetric(covariance, name="covariance")
    data = _data(observations)
    if matrix.shape != (data[0].dimension, data[0].dimension):
        raise ValueError("covariance dimension must match validation observations")
    scores = []
    for item in data:
        predicted = item.operator @ matrix @ item.operator.T
        scores.append(float(np.sum(predicted**2) - 2 * np.sum(item.measurement * predicted)))
    result = float(np.mean(scores))
    if not np.isfinite(result):
        raise ValueError("validation score is not finite; rescale the data")
    return result


def _solver_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("solver_options must be a mapping")
    supplied = {} if options is None else dict(options)
    allowed = {"max_iterations", "absolute_tolerance", "relative_tolerance", "initial"}
    unexpected = supplied.keys() - allowed
    if unexpected:
        raise ValueError(f"unsupported solver_options: {sorted(unexpected)}")
    result = {"absolute_tolerance": 1e-12, "relative_tolerance": 0.0}
    result.update(supplied)
    for name in ("absolute_tolerance", "relative_tolerance"):
        result[name] = _scalar(result[name], name)
    if "max_iterations" in result:
        value = result["max_iterations"]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        result["max_iterations"] = int(value)
    if "initial" in result and result["initial"] is not None:
        result["initial"] = validate_symmetric(result["initial"], name="initial")
    return result


def _windows(values: Sequence[int], count: int) -> tuple[int, ...]:
    try:
        windows = tuple(values)
    except TypeError as error:
        raise ValueError("window_candidates must be a sequence of integer counts") from error
    if not windows or any(
        isinstance(window, bool)
        or not isinstance(window, (int, np.integer))
        or window < 1
        or window > count
        for window in windows
    ):
        raise ValueError("window_candidates must be integer counts between 1 and training length")
    return tuple(int(window) for window in windows)


def _relatives(method: str, values: ArrayLike | None) -> tuple[float | None, ...]:
    if method == "empirical":
        return (None,)
    grid = _real_array(values, name="relative_candidates")
    if grid.ndim != 1 or not len(grid):
        raise ValueError("relative_candidates must be a nonempty vector")
    if method == "linear_shrinkage":
        if np.any(grid < 0) or np.any(grid > 1):
            raise ValueError("linear-shrinkage candidates must lie in [0, 1]")
    elif np.any(grid <= 0):
        raise ValueError("relative ridge candidates must be strictly positive")
    return tuple(float(value) for value in grid)


def _fit_candidate(
    data: tuple[Observation, ...],
    method: str,
    relative: float | None,
    trace_cap: float,
    numerical_error_budget: float,
    solver_options: dict[str, Any],
    gram_norm: float | None = None,
) -> tuple[FloatArray, dict[str, Any], CovarianceEstimate | None]:
    parameters: dict[str, Any] = {"window": len(data)}
    estimate = None
    if method == "empirical":
        covariance = empirical_covariance(data)
    elif method == "linear_shrinkage":
        parameters["shrinkage"] = relative
        covariance = linear_shrinkage_covariance(data, relative)
    else:
        if gram_norm is None:
            gram = gram_operator([item.operator for item in data])
            gram_norm = float(np.linalg.eigvalsh(gram)[-1])
        if gram_norm <= 0 or not np.isfinite(gram_norm):
            raise ValueError("a positive finite Gram operator norm is required for relative ridge")
        regularization = float(relative * gram_norm)
        parameters.update(
            relative_regularization=relative,
            gram_operator_norm=gram_norm,
            regularization=regularization,
        )
        if method == "projected_ridge":
            covariance = projected_ridge_covariance(
                data, regularization=regularization, trace_cap=trace_cap
            )
        else:
            estimate = fit_covariance(
                data,
                regularization=regularization,
                trace_cap=trace_cap,
                numerical_error_budget=numerical_error_budget,
                **solver_options,
            )
            if not estimate.converged:
                raise RuntimeError(f"constrained solver did not converge: {estimate.status}")
            covariance = estimate.covariance
    return covariance, parameters, estimate


def select_covariance(
    observations: Sequence[Observation],
    validation: Sequence[Observation],
    *,
    method: str,
    relative_candidates: ArrayLike | None,
    window_candidates: Sequence[int],
    trace_cap: float,
    numerical_error_budget: float = 1e-4,
    solver_options: Mapping[str, Any] | None = None,
) -> Selection:
    """Select a trailing-window estimate using only held-out observed outcomes.

    Windows are observation counts, taken from the end of the supplied order.
    The caller supplies independent validation data and the intended temporal
    order; no latent truth, model, or instantaneous-covariance target is used.
    Relative ridge is lambda/||Q||op, with Q recomputed on each training window.
    For shrinkage, relative_candidates instead supplies the shrinkage alphas;
    empirical estimation has only window candidates. Ties select the first
    successful candidate in input order. Failed numerical candidates are
    recorded and skipped; if every candidate fails, RuntimeError is raised.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown covariance method: {method}")
    trace_cap = _scalar(trace_cap, "trace_cap")
    numerical_error_budget = _scalar(
        numerical_error_budget, "numerical_error_budget", positive=True
    )
    options = _solver_options(solver_options)
    data = _data(observations)
    holdout = _data(validation)
    if data[0].dimension != holdout[0].dimension:
        raise ValueError("training and validation must use the same curve dimension")
    windows = _windows(window_candidates, len(data))
    relatives = _relatives(method, relative_candidates)
    candidates = []
    best = None
    for window in windows:
        local = data[-window:]
        gram_norm = None
        if method.endswith("ridge"):
            gram = gram_operator([item.operator for item in local])
            gram_norm = float(np.linalg.eigvalsh(gram)[-1])
        for relative in relatives:
            parameters: dict[str, Any] = {"window": window}
            if method == "linear_shrinkage":
                parameters["shrinkage"] = relative
            elif method.endswith("ridge"):
                parameters.update(
                    relative_regularization=relative,
                    gram_operator_norm=gram_norm,
                    regularization=float(relative * gram_norm),
                )
            try:
                covariance, parameters, estimate = _fit_candidate(
                    local, method, relative, trace_cap, numerical_error_budget, options, gram_norm
                )
                score = validation_score(covariance, holdout)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                candidates.append(
                    {
                        "parameters": parameters,
                        "status": "failed",
                        "failure": str(error),
                        "validation_score": None,
                    }
                )
                continue
            record: dict[str, Any] = {
                "parameters": parameters,
                "status": "converged" if estimate is not None else "closed_form",
                "validation_score": score,
            }
            if estimate is not None:
                record.update(
                    objective_gap_bound=estimate.objective_gap_bound,
                    numerical_covariance_error_bound=estimate.covariance_error_bound,
                    iterations=estimate.iterations,
                )
            candidates.append(record)
            if best is None or score < best[0]:
                best = (score, covariance, parameters, estimate)
    if best is None:
        reasons = "; ".join(candidate["failure"] for candidate in candidates)
        raise RuntimeError(f"all covariance candidates failed: {reasons}")
    score, covariance, parameters, estimate = best
    return Selection(covariance, method, parameters, score, tuple(candidates), estimate)


def refit_selection(
    observations: Sequence[Observation],
    selection: Selection,
    *,
    trace_cap: float,
    numerical_error_budget: float = 1e-4,
    solver_options: Mapping[str, Any] | None = None,
) -> Selection:
    """Refit chosen window/relative strength, recalibrating lambda on new data.

    Original validation score and candidate audit are retained for provenance;
    they must not be interpreted as a score of the newly refitted covariance.
    """
    if not isinstance(selection, Selection):
        raise ValueError("selection must be a Selection")
    trace_cap = _scalar(trace_cap, "trace_cap")
    numerical_error_budget = _scalar(
        numerical_error_budget, "numerical_error_budget", positive=True
    )
    options = _solver_options(solver_options)
    data = _data(observations)
    if data[0].dimension != len(selection.covariance):
        raise ValueError("refit observations must use the selected covariance dimension")
    window = _windows([selection.selected_parameters.get("window")], len(data))[0]
    if selection.method == "empirical":
        relative = None
    else:
        name = "shrinkage" if selection.method == "linear_shrinkage" else "relative_regularization"
        relative = _relatives(selection.method, [selection.selected_parameters.get(name)])[0]
    covariance, parameters, estimate = _fit_candidate(
        data[-window:], selection.method, relative, trace_cap, numerical_error_budget, options
    )
    return Selection(
        covariance,
        selection.method,
        parameters,
        selection.validation_score,
        selection.candidates,
        estimate,
    )
