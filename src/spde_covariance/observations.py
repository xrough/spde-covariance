"""Contract observation maps and clean innovation measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import _real_array, validate_psd

if TYPE_CHECKING:
    from .spaces import SplineSpace


@dataclass(frozen=True, slots=True)
class Observation:
    """One predictable observation map applied to a normalized innovation.

    ``noise_covariance`` is the covariance of fresh innovation noise. It must
    not be interpreted automatically as a correction for differenced noisy
    price levels. A corrected measurement can be indefinite.
    """

    time: float
    y: NDArray[np.float64]
    operator: NDArray[np.float64]
    noise_covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not np.isscalar(self.time) or not np.isfinite(self.time):
            raise ValueError("time must be finite")
        y = _real_array(self.y, name="y")
        operator = _real_array(self.operator, name="operator")
        if y.ndim != 1 or not y.size:
            raise ValueError("y must be a nonempty vector")
        if operator.ndim != 2 or operator.shape[0] != len(y) or not operator.shape[1]:
            raise ValueError("operator must have shape (len(y), dimension)")
        noise = validate_psd(self.noise_covariance, name="noise_covariance")
        if noise.shape != (len(y), len(y)):
            raise ValueError("noise_covariance must have shape (len(y), len(y))")
        for name, value in (("y", y), ("operator", operator), ("noise_covariance", noise)):
            copy = value.copy()
            copy.flags.writeable = False
            object.__setattr__(self, name, copy)
        object.__setattr__(self, "time", float(self.time))

    @property
    def dimension(self) -> int:
        return self.operator.shape[1]

    @property
    def n_contracts(self) -> int:
        return len(self.y)

    @property
    def measurement(self) -> NDArray[np.float64]:
        """Return ``y y.T - N`` without a positivity projection."""
        return np.outer(self.y, self.y) - self.noise_covariance


def point_observation_map(space: SplineSpace, maturities: ArrayLike) -> NDArray[np.float64]:
    """Rows of point-evaluation functionals in orthonormal coordinates."""
    return space.evaluate(maturities)


def delivery_average_map(
    space: SplineSpace,
    intervals: ArrayLike,
    *,
    quadrature_order: int = 16,
) -> NDArray[np.float64]:
    """Uniform delivery averages over rows ``[start, end]``.

    Integration is split at spline knots and at the constant-tail boundary.
    Intervals beyond that boundary therefore use the documented constant tail.
    """
    bounds = _real_array(intervals, name="intervals")
    if bounds.shape == (2,):
        bounds = bounds.reshape(1, 2)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds):
        raise ValueError("intervals must have shape (n, 2)")
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("delivery intervals require 0 <= start < end")
    if not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 1:
        raise ValueError("quadrature_order must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    rows = []
    for start, end in bounds:
        interior = space.breakpoints[(space.breakpoints > start) & (space.breakpoints < end)]
        cuts = np.concatenate(([start], interior, [end]))
        row = np.zeros(space.dimension)
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            x = (left + right) / 2 + (right - left) * nodes / 2
            row += weights @ space.evaluate(x) * (right - left) / 2
        rows.append(row / (end - start))
    return np.asarray(rows)
