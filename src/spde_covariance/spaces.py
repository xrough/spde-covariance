"""Weighted-Sobolev spline sieves with explicit constant-tail boundary.

The curve metric is ``f(0)g(0) + integral_0^L exp(x/tau) f'(x)g'(x) dx``.
Every spline is extended constantly beyond ``L``. This is a finite sieve in
the corresponding half-line Sobolev space, not a fixed covariance-rank model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular

from .linalg import _real_array


class SplineSpace:
    """Cubic B-splines orthonormalized in the stated curve metric.

    ``shift_matrix(h)`` is the Hilbert projection ``P S_h P``. Projecting
    after each shift generally breaks the exact semigroup identity. Interior
    SPDE experiments must report shift and right-boundary approximation error.
    """

    def __init__(self, n_basis: int = 8, length: float = 6.0, tau: float = 1.0) -> None:
        if isinstance(n_basis, bool) or not isinstance(n_basis, (int, np.integer)) or n_basis < 4:
            raise ValueError("n_basis must be an integer of at least four for cubic splines")
        if not np.isfinite(length) or length <= 0 or not np.isfinite(tau) or tau <= 0:
            raise ValueError("length and tau must be finite and positive")
        self.dimension = int(n_basis)
        self.length = float(length)
        self.tau = float(tau)
        self.breakpoints = np.linspace(0, self.length, self.dimension - 2)
        knots = np.concatenate((np.zeros(4), self.breakpoints[1:-1], np.full(4, self.length)))
        self._basis = BSpline(knots, np.eye(self.dimension), 3, extrapolate=False)
        self._basis_derivative = self._basis.derivative()
        nodes, weights = self.quadrature(weighted=True)
        derivatives = self.raw_derivative(nodes)
        origin = self.raw_evaluate([0])[0]
        gram = np.outer(origin, origin) + derivatives.T @ (weights[:, None] * derivatives)
        self._raw_gram = (gram + gram.T) / 2
        factor = np.linalg.cholesky(self._raw_gram)
        self._transform = solve_triangular(factor.T, np.eye(self.dimension), lower=False)

    @property
    def raw_gram(self) -> NDArray[np.float64]:
        """Gram matrix of the raw clamped B-spline coordinates."""
        return self._raw_gram.copy()

    @property
    def transform(self) -> NDArray[np.float64]:
        """Columns express orthonormal basis functions in raw coordinates."""
        return self._transform.copy()

    def _points(self, x: ArrayLike) -> NDArray[np.float64]:
        points = np.atleast_1d(_real_array(x, name="maturities"))
        if points.ndim != 1 or np.any(points < 0):
            raise ValueError("maturities must be a nonnegative scalar or vector")
        return points

    def raw_evaluate(self, x: ArrayLike) -> NDArray[np.float64]:
        points = self._points(x)
        return np.asarray(self._basis(np.minimum(points, self.length)))

    def raw_derivative(self, x: ArrayLike) -> NDArray[np.float64]:
        points = self._points(x)
        values = np.asarray(self._basis_derivative(np.minimum(points, self.length)))
        values[points >= self.length] = 0
        return values

    def evaluate(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return one orthonormal evaluation row per maturity."""
        return self.raw_evaluate(x) @ self._transform

    def derivative(self, x: ArrayLike) -> NDArray[np.float64]:
        """Derivative rows; derivatives are zero on the constant tail."""
        return self.raw_derivative(x) @ self._transform

    def quadrature(
        self,
        *,
        order: int = 16,
        breakpoints: ArrayLike | None = None,
        weighted: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Piecewise Gauss-Legendre quadrature on ``[0, length]``.

        Extra breakpoints are merged with the spline knots. With ``weighted``
        true, weights include the Hilbert derivative weight ``exp(x/tau)``.
        """
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("order must be a positive integer")
        cuts = self.breakpoints
        if breakpoints is not None:
            extra = _real_array(breakpoints, name="breakpoints")
            if extra.ndim != 1:
                raise ValueError("breakpoints must be a vector")
            cuts = np.unique(np.concatenate((cuts, extra[(extra > 0) & (extra < self.length)])))
        base_nodes, base_weights = np.polynomial.legendre.leggauss(order)
        widths = np.diff(cuts)
        midpoints = (cuts[:-1] + cuts[1:]) / 2
        nodes = (midpoints[:, None] + widths[:, None] * base_nodes / 2).ravel()
        weights = (widths[:, None] * base_weights / 2).ravel()
        if weighted:
            with np.errstate(over="ignore"):
                weights = weights * np.exp(nodes / self.tau)
            if not np.all(np.isfinite(weights)):
                raise ValueError("length/tau makes the Hilbert weights overflow")
        return nodes, weights

    def shift_matrix(self, dt: float, *, quadrature_order: int = 16) -> NDArray[np.float64]:
        """Projected right shift, with the constant-tail extension explicit."""
        if not np.isscalar(dt) or not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if dt == 0:
            return np.eye(self.dimension)
        shifted_knots = self.breakpoints - dt
        nodes, weights = self.quadrature(
            order=quadrature_order, breakpoints=shifted_knots, weighted=True
        )
        origin = self.evaluate([0])[0]
        shifted_origin = self.evaluate([dt])[0]
        return np.outer(origin, shifted_origin) + self.derivative(nodes).T @ (
            weights[:, None] * self.derivative(nodes + dt)
        )

    def project(
        self,
        function: Callable[[NDArray[np.float64]], ArrayLike],
        derivative: Callable[[NDArray[np.float64]], ArrayLike] | None = None,
        *,
        quadrature_order: int = 16,
        breakpoints: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Hilbert-project a scalar or column-vector-valued function.

        Callables accept an array of nodes and return shape ``(n,)`` or
        ``(n, k)``. The function is restricted to ``[0, length]`` and extended
        constantly. If no derivative is provided, fourth-order centered
        differences approximate it at positive quadrature nodes; analytical
        derivatives should be supplied for numerical-convergence benchmarks.
        Supply extra ``breakpoints`` for a function with derivative knots or
        a shifted constant-tail boundary inside the integration interval.
        """
        nodes, weights = self.quadrature(
            order=quadrature_order, breakpoints=breakpoints, weighted=True
        )

        def values(
            callable_: Callable[[NDArray[np.float64]], ArrayLike], points: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            array = _real_array(callable_(points), name="function values")
            if array.ndim == 0:
                return np.full((len(points), 1), float(array))
            if array.shape == (len(points),):
                return array[:, None]
            if array.ndim == 2 and array.shape[0] == len(points) and array.shape[1]:
                return array
            raise ValueError("function must return shape (n,) or (n, k)")

        at_origin = values(function, np.array([0.0]))
        if derivative is None:
            step = np.minimum(np.finfo(float).eps ** 0.2 * max(1.0, self.length), nodes / 3)
            slopes = (
                -values(function, nodes + 2 * step)
                + 8 * values(function, nodes + step)
                - 8 * values(function, nodes - step)
                + values(function, nodes - 2 * step)
            ) / (12 * step[:, None])
        else:
            slopes = values(derivative, nodes)
        if slopes.shape[1] != at_origin.shape[1]:
            raise ValueError("function and derivative output dimensions differ")
        coefficients = np.outer(self.evaluate([0])[0], at_origin[0]) + self.derivative(nodes).T @ (
            weights[:, None] * slopes
        )
        sample = np.asarray(function(np.array([0.0])))
        return coefficients[:, 0] if sample.ndim < 2 else coefficients
