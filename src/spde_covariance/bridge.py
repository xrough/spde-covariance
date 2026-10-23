"""Deterministic-volatility SPDE bridge benchmarks in orthonormal coordinates.

Use an analytic shifted-source callback for the direct reference; replacing it
with a projected shift matrix is a distinct approximation. Strong continuity
alone is not asserted to give any uniform power rate.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BridgeResult:
    innovation_covariance: NDArray[np.float64]
    instantaneous_covariance: NDArray[np.float64]
    discrepancy: float
    start: float
    delta: float
    method: str
    quadrature_order: int | None
    time_steps: int | None
    target: str = "V = integral S C S* / delta; C = source(start) source(start)*"
    basis: str = "orthonormal H coordinates"


def _source_matrix(source, time):
    result = np.asarray(source(time), dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or min(result.shape) == 0 or not np.all(np.isfinite(result)):
        raise ValueError("source must return a finite J by q coefficient matrix")
    return result


def integrate_innovation_covariance(
    source: Callable,
    *,
    start: float,
    delta: float,
    shifted_source: Callable | None = None,
    shift_matrix: Callable | None = None,
    quadrature_order: int = 24,
    time_steps: int | None = None,
) -> BridgeResult:
    """Integrate deterministic shifted source outer products over one increment.

    ``source(s)`` returns J by q coefficients. Supply exactly one of
    ``shifted_source(s,lag)`` for direct Hilbert projections of shifted functions,
    or ``shift_matrix(lag)`` for projected-semigroup matrices. Gauss--Legendre
    quadrature is the default; ``time_steps`` selects composite midpoint time
    discretization. All callbacks must use the same orthonormal coordinates.
    """
    if not np.all(np.isfinite([start, delta])) or delta <= 0:
        raise ValueError("start must be finite and delta strictly positive")
    if (shifted_source is None) == (shift_matrix is None):
        raise ValueError("provide exactly one of shifted_source or shift_matrix")
    if not isinstance(quadrature_order, int) or quadrature_order < 1:
        raise ValueError("quadrature_order must be a positive integer")
    if time_steps is not None and (not isinstance(time_steps, int) or time_steps < 1):
        raise ValueError("time_steps must be a positive integer")
    at_start = _source_matrix(source, start)
    if time_steps is None:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        fractions, weights = (nodes + 1) / 2, weights / 2
    else:
        fractions = (np.arange(time_steps) + 0.5) / time_steps
        weights = np.full(time_steps, 1 / time_steps)
    covariance = np.zeros((len(at_start), len(at_start)))
    for fraction, weight in zip(fractions, weights, strict=True):
        time = start + delta * fraction
        lag = delta * (1 - fraction)
        if shifted_source is not None:
            shifted = np.asarray(shifted_source(time, lag), dtype=float)
            if shifted.ndim == 1:
                shifted = shifted[:, None]
        else:
            shift = np.asarray(shift_matrix(lag), dtype=float)
            if shift.shape != (len(at_start), len(at_start)) or not np.all(np.isfinite(shift)):
                raise ValueError("shift matrices must be finite J by J arrays")
            shifted = shift @ _source_matrix(source, time)
        if shifted.shape != at_start.shape or not np.all(np.isfinite(shifted)):
            raise ValueError("shifted source must have constant finite J by q shape")
        covariance += weight * shifted @ shifted.T
    instantaneous = at_start @ at_start.T
    return BridgeResult(
        innovation_covariance=covariance,
        instantaneous_covariance=instantaneous,
        discrepancy=float(np.linalg.norm(covariance - instantaneous, ord="fro")),
        start=float(start),
        delta=float(delta),
        method="direct_shifted_source" if shifted_source is not None else "projected_shift",
        quadrature_order=quadrature_order if time_steps is None else None,
        time_steps=time_steps,
    )


def fixed_delivery_increment(curve: Callable, *, start: float, end: float, deliveries):
    """Compute f_end(T-end)-f_start(T-start) for the same delivery dates T.

    This algebraic helper subtracts neither an integrated mild drift nor noisy
    price-level errors. Such corrections require their own observation model.
    """
    dates = np.atleast_1d(np.asarray(deliveries, dtype=float))
    if not np.all(np.isfinite([start, end])) or end <= start:
        raise ValueError("end must be finite and strictly later than start")
    if dates.ndim != 1 or not np.all(np.isfinite(dates)) or np.any(dates < end):
        raise ValueError("delivery dates must be finite and no earlier than end")
    return np.asarray(curve(end, dates - end)) - np.asarray(curve(start, dates - start))
