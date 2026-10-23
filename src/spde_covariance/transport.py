"""Transport propagators in declared orthonormal Hilbert coordinates.

``ExponentialTransport`` is an exact finite, shift-invariant submodel of the
half-line transport SPDE. ``ProjectedSplineTransport`` defines a projected mild
recursion, whose one-step matrices generally do not form an exact semigroup.
Both return normalized innovation covariance V, not diffusion covariance C.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_triangular

from .linalg import _real_array, validate_psd
from .spaces import SplineSpace

FloatArray = NDArray[np.float64]


@runtime_checkable
class TransportPropagator(Protocol):
    """Finite-coordinate transport contract consumed by trajectory routines.

    ``basis_id`` identifies the coordinate convention and is an opaque string.
    Input source covariance C and output innovation covariance V use those same
    coordinates. Only ``kind='exact_transport'`` asserts an exact semigroup.
    """

    dimension: int
    basis_id: str
    kind: str

    def matrix(self, dt: float) -> FloatArray:
        """Return a transport matrix for finite ``dt >= 0``."""
        ...

    def innovation_covariance(
        self, source_covariance: ArrayLike, dt: float, quadrature_order: int = 24
    ) -> FloatArray:
        """Return ``integral_0^dt S(r) C S(r).T dr / dt`` for ``dt > 0``."""
        ...


def _positive_scalar(value: float, *, name: str, allow_zero: bool = False) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        boundary = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be finite and {boundary}")
    return result


def _quadrature_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("quadrature_order must be a positive integer")
    return int(order)


def _source_covariance(matrix: ArrayLike, dimension: int) -> FloatArray:
    covariance = validate_psd(matrix, name="source_covariance")
    if covariance.shape != (dimension, dimension):
        raise ValueError("source_covariance must match the propagator dimension")
    return covariance


def _basis_digest(prefix: str, configuration: dict) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


class ExponentialTransport:
    """Exact right shift on an orthonormalized exponential span.

    Raw functions are ``g_j(x)=exp(-rates[j]*x)`` on the nonnegative half-line.
    The metric is ``f(0)g(0)+integral exp(x/tau) f'(x)g'(x) dx``. Distinct
    rates strictly above ``1/(2*tau)`` guarantee membership and independence.
    An excessively ill-conditioned raw Gram matrix is rejected instead of
    presenting an unreliable whitening transform as an orthonormal basis.

    If columns of T describe the orthonormal basis in raw coordinates, the
    exact propagator is ``T^-1 diag(exp(-rates*dt)) T``. It is generally
    nonnormal in the stated metric, so it must not be replaced by that diagonal.
    """

    kind = "exact_transport"
    _MAX_GRAM_CONDITION = 1e10

    def __init__(self, rates: ArrayLike = (0.6, 1.0, 2.0, 4.0), tau: float = 1.0):
        self._tau = _positive_scalar(tau, name="tau")
        values = _real_array(rates, name="rates")
        if values.ndim != 1 or not len(values):
            raise ValueError("rates must be a nonempty vector")
        if len(np.unique(values)) != len(values):
            raise ValueError("rates must be distinct")
        with np.errstate(over="ignore", divide="ignore"):
            inverse_tau = np.divide(1.0, self._tau)
        if not np.isfinite(inverse_tau):
            raise ValueError("tau is too small to form a finite Hilbert metric")
        if np.any(values <= inverse_tau / 2):
            raise ValueError("each rate must be strictly greater than 1/(2*tau)")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denominator = values[:, None] + values[None, :] - inverse_tau
            gram = 1 + np.outer(values, values) / denominator
        if not np.all(np.isfinite(gram)):
            raise ValueError("rates and tau produce a nonfinite Hilbert Gram matrix")
        condition = float(np.linalg.cond(gram))
        if not np.isfinite(condition) or condition > self._MAX_GRAM_CONDITION:
            raise ValueError(
                f"exponential Gram matrix is ill-conditioned (condition={condition:.3g}); "
                "use fewer or more widely separated rates"
            )
        try:
            factor = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "exponential Gram matrix is not numerically positive definite"
            ) from error
        self._rates = values.copy()
        self._rates.flags.writeable = False
        self._raw_gram = gram
        self._transform = solve_triangular(factor.T, np.eye(len(values)), lower=False)
        self._inverse_transform = factor.T
        self._gram_condition_number = condition
        self._basis_id = _basis_digest(
            "exponential-transport",
            {
                "metric": "weighted_sobolev_anchor_v1",
                "orthonormalization": "inverse_transpose_cholesky_v1",
                "rates": values.tolist(),
                "tau": self._tau,
            },
        )

    @property
    def dimension(self) -> int:
        return len(self._rates)

    @property
    def basis_id(self) -> str:
        return self._basis_id

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def rates(self) -> FloatArray:
        return self._rates.copy()

    @property
    def raw_gram(self) -> FloatArray:
        return self._raw_gram.copy()

    @property
    def transform(self) -> FloatArray:
        return self._transform.copy()

    @property
    def gram_condition_number(self) -> float:
        return self._gram_condition_number

    @property
    def breakpoints(self) -> FloatArray:
        """No interior derivative knots or finite truncation boundary."""
        return np.empty(0)

    def _points(self, x: ArrayLike) -> FloatArray:
        points = np.atleast_1d(_real_array(x, name="maturities"))
        if points.ndim != 1 or np.any(points < 0):
            raise ValueError("maturities must be a nonnegative scalar or vector")
        return points

    def evaluate(self, x: ArrayLike) -> FloatArray:
        """One orthonormal curve-evaluation row per maturity."""
        points = self._points(x)
        with np.errstate(over="ignore"):
            raw = np.exp(-points[:, None] * self._rates)
        return raw @ self._transform

    def derivative(self, x: ArrayLike) -> FloatArray:
        points = self._points(x)
        with np.errstate(over="ignore"):
            raw = -self._rates * np.exp(-points[:, None] * self._rates)
        return raw @ self._transform

    def matrix(self, dt: float) -> FloatArray:
        """Exact coordinate matrix of ``f(x) -> f(x+dt)``."""
        duration = _positive_scalar(dt, name="dt", allow_zero=True)
        if duration == 0:
            return np.eye(self.dimension)
        with np.errstate(over="ignore"):
            decay = np.exp(-self._rates * duration)
        return self._inverse_transform @ (decay[:, None] * self._transform)

    def innovation_covariance(
        self, source_covariance: ArrayLike, dt: float, quadrature_order: int = 24
    ) -> FloatArray:
        """Analytic normalized innovation V for constant diffusion covariance C.

        ``quadrature_order`` is validated for protocol compatibility; this
        backend integrates analytically. V generally differs from C at finite
        dt even though both matrices use the same orthonormal coordinates.
        """
        duration = _positive_scalar(dt, name="dt")
        _quadrature_order(quadrature_order)
        covariance = _source_covariance(source_covariance, self.dimension)
        raw_covariance = self._transform @ covariance @ self._transform.T
        with np.errstate(over="ignore"):
            argument = (self._rates[:, None] + self._rates[None, :]) * duration
        # lim_{z->0} -expm1(-z)/z = 1; division is safe for positive rates/dt,
        # apart from representable underflow handled explicitly below.
        attenuation = np.ones_like(argument)
        np.divide(-np.expm1(-argument), argument, out=attenuation, where=argument != 0)
        raw_innovation = raw_covariance * attenuation
        result = self._inverse_transform @ raw_innovation @ self._inverse_transform.T
        return (result + result.T) / 2


class ProjectedSplineTransport:
    """Adapter for the constant-tail spline projected mild recursion.

    ``matrix(h)=P S_h P`` generally fails the semigroup identity. Innovation
    covariance integrates these projected matrices, so it is the declared
    numerical recursion's target, not an exact continuous transport law.
    Rebuild the adapter if the underlying spline-space configuration changes.
    """

    kind = "projected_mild_recursion"

    def __init__(self, space: SplineSpace):
        if not isinstance(space, SplineSpace):
            raise ValueError("space must be a SplineSpace")
        self._space = space
        self._dimension = space.dimension
        self._basis_id = _basis_digest(
            "projected-spline",
            {
                "metric": "weighted_sobolev_anchor_v1",
                "boundary": "constant_tail_v1",
                "dimension": space.dimension,
                "length": space.length,
                "tau": space.tau,
                "transform_sha256": hashlib.sha256(
                    np.asarray(space.transform, dtype="<f8").tobytes()
                ).hexdigest(),
            },
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def basis_id(self) -> str:
        return self._basis_id

    def matrix(self, dt: float) -> FloatArray:
        duration = _positive_scalar(dt, name="dt", allow_zero=True)
        return self._space.shift_matrix(duration)

    def innovation_covariance(
        self, source_covariance: ArrayLike, dt: float, quadrature_order: int = 24
    ) -> FloatArray:
        """Quadrature of projected shifts; result is normalized by dt.

        Time integration splits at differences between spline knots, including
        the constant-tail boundary. The wrapped space controls spatial
        quadrature, while ``quadrature_order`` controls this time integration.
        """
        duration = _positive_scalar(dt, name="dt")
        order = _quadrature_order(quadrature_order)
        covariance = _source_covariance(source_covariance, self.dimension)
        knots = self._space.breakpoints
        differences = (knots[:, None] - knots[None, :]).ravel()
        cuts = np.unique(
            np.concatenate(
                ([0.0], differences[(differences > 0) & (differences < duration)], [duration])
            )
        )
        nodes, weights = np.polynomial.legendre.leggauss(order)
        result = np.zeros_like(covariance)
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            times = (left + right) / 2 + (right - left) * nodes / 2
            local_weights = (right - left) / (2 * duration) * weights
            for time, weight in zip(times, local_weights, strict=True):
                matrix = self.matrix(float(time))
                result += weight * matrix @ covariance @ matrix.T
        return (result + result.T) / 2
