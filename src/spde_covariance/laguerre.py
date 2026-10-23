"""Nested Laguerre transport levels of a fixed half-line Hilbert-space model.

The metric is f(0)g(0) + integral exp(x) f'(x)g'(x) dx. The basis is
e_0=1 and e_j(x)=integral_0^x exp(-u)L_{j-1}(u) du. Finite prefixes are
shift invariant, but their orthogonal projections do not commute with the
full shift. A truncated source model is therefore not the projection of the
full-source process. Covariance integration and analytic tail bounds below
are evaluated in floating point, not interval-arithmetic proofs.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm, toeplitz
from scipy.special import zeta

from .linalg import _real_array
from .transport import _basis_digest, _positive_scalar, _quadrature_order, _source_covariance

FloatArray = NDArray[np.float64]
_FAMILY_ID = _basis_digest(
    "laguerre-halfline",
    {"metric": "anchor_plus_exp_x_derivative_v1", "basis": "integrated_laguerre_v1"},
)


def _integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return int(value)


def _finite(matrix: FloatArray, name: str) -> FloatArray:
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} exceeds floating-point range")
    return matrix


def _symmetric(matrix: FloatArray) -> FloatArray:
    return _finite((matrix + matrix.T) / 2, "covariance")


def _weighted_laguerre(points: FloatArray, count: int, parameter: int) -> FloatArray:
    """Evaluate exp(-x)L_n^(parameter)(x), avoiding unweighted polynomial overflow."""
    values = np.empty((len(points), count))
    if not count:
        return values
    values[:, 0] = np.exp(-points)
    if count > 1:
        values[:, 1] = (1 + parameter - points) * values[:, 0]
    for degree in range(1, count - 1):
        values[:, degree + 1] = (
            (2 * degree + 1 + parameter - points) * values[:, degree]
            - (degree + parameter) * values[:, degree - 1]
        ) / (degree + 1)
    return _finite(values, "weighted Laguerre evaluation")


class LaguerreTransport:
    """Exact restricted transport on an orthonormal nested Laguerre prefix.

    No maturity cutoff or constant-tail extension is introduced. ``family_id``
    is common across levels; ``basis_id`` additionally identifies the prefix
    dimension. Both covariance methods integrate a bounded small step then
    compose it by doubling. ``method='block'`` independently uses a scaled
    Van Loan exponential; it never exponentiates inverse transport over a
    long physical horizon. A bounded per-instance cache stores immutable
    covariance keys and returns copies, so mutating inputs cannot stale it.
    """

    kind = "exact_transport"
    tau = 1.0
    family_id = _FAMILY_ID
    basis_family_id = _FAMILY_ID

    def __init__(self, dimension: int, *, cache_max_bytes: int = 16 * 1024**2):
        self._dimension = _integer(dimension, "dimension")
        self._cache_max_bytes = _integer(cache_max_bytes, "cache_max_bytes", allow_zero=True)
        self._basis_id = _basis_digest(
            "laguerre-prefix", {"family_id": self.family_id, "dimension": self.dimension}
        )
        generator = -np.triu(np.ones((self.dimension, self.dimension)))
        generator[0] = 1
        generator[0, 0] = 0
        self._generator = generator
        self._generator.flags.writeable = False
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def basis_id(self) -> str:
        return self._basis_id

    @property
    def generator(self) -> FloatArray:
        return self._generator.copy()

    @property
    def raw_gram(self) -> FloatArray:
        return np.eye(self.dimension)

    @property
    def transform(self) -> FloatArray:
        return np.eye(self.dimension)

    @property
    def breakpoints(self) -> FloatArray:
        return np.empty(0)

    def _points(self, x: ArrayLike) -> FloatArray:
        if np.asarray(x).dtype.kind not in "iuf":
            raise ValueError("maturities must be real numeric values")
        points = np.atleast_1d(_real_array(x, name="maturities"))
        if points.ndim != 1 or np.any(points < 0):
            raise ValueError("maturities must be a nonnegative scalar or vector")
        return points

    def evaluate(self, x: ArrayLike) -> FloatArray:
        points = self._points(x)
        values = np.ones((len(points), self.dimension))
        if self.dimension > 1:
            values[:, 1] = -np.expm1(-points)
        if self.dimension > 2:
            # x L_(j-2)^1(x)/(j-1) avoids cancellation of nearby Laguerre values.
            weighted = _weighted_laguerre(points, self.dimension - 2, 1)
            values[:, 2:] = (points[:, None] / np.arange(1, self.dimension - 1)) * weighted
        return _finite(values, "basis evaluation")

    def derivative(self, x: ArrayLike) -> FloatArray:
        points = self._points(x)
        return np.column_stack(
            (np.zeros(len(points)), _weighted_laguerre(points, self.dimension - 1, 0))
        )

    def matrix(self, dt: float) -> FloatArray:
        """Exact right-shift matrix, using the Laguerre addition identity."""
        duration = _positive_scalar(dt, name="dt", allow_zero=True)
        if duration == 0:
            return np.eye(self.dimension)
        values = self.evaluate([duration])[0]
        result = np.zeros((self.dimension, self.dimension))
        result[0] = values
        if self.dimension > 1:
            decay = np.exp(-duration)
            result[1:, 1:] = toeplitz(
                np.r_[decay, np.zeros(self.dimension - 2)], np.r_[decay, -values[2:]]
            )
        return result

    def _small_step(self, duration: float) -> tuple[float, int]:
        # ||h A||_infinity <= 1 controls inverse-transport growth in the block check.
        norm = max(self.dimension - 1, 1)
        doublings = max(0, int(np.ceil(np.log2(duration) + np.log2(norm))))
        return float(np.ldexp(duration, -doublings)), doublings

    @staticmethod
    def _compose(normalized: FloatArray, shift: FloatArray, doublings: int) -> FloatArray:
        for _ in range(doublings):
            normalized = _symmetric((normalized + shift @ normalized @ shift.T) / 2)
            shift = shift @ shift
        return normalized

    def _quadrature(self, covariance: FloatArray, duration: float, order: int) -> FloatArray:
        step, doublings = self._small_step(duration)
        nodes, weights = np.polynomial.legendre.leggauss(order)
        result = np.zeros_like(covariance)
        for node, weight in zip(nodes, weights, strict=True):
            shift = self.matrix((node + 1) * (step / 2))
            result += (weight / 2) * (shift @ covariance @ shift.T)
        return self._compose(_symmetric(result), self.matrix(step), doublings)

    def _block(self, covariance: FloatArray, duration: float) -> FloatArray:
        step, doublings = self._small_step(duration)
        scale = float(np.max(np.abs(covariance)))
        if scale == 0:
            return covariance.copy()
        # Off-diagonal C (rather than h*C) directly yields the normalized
        # small-step covariance; this also avoids division by a tiny duration.
        block = np.zeros((2 * self.dimension, 2 * self.dimension))
        block[: self.dimension, : self.dimension] = step * self._generator
        block[self.dimension :, self.dimension :] = -step * self._generator.T
        block[: self.dimension, self.dimension :] = covariance / scale
        exponential = expm(block)
        shift = exponential[: self.dimension, : self.dimension]
        normalized = _symmetric((exponential[: self.dimension, self.dimension :] @ shift.T) * scale)
        return self._compose(normalized, shift, doublings)

    def _normalized(
        self, covariance: FloatArray, duration: float, order: int, method: str
    ) -> FloatArray:
        if method not in ("quadrature", "block"):
            raise ValueError("method must be 'quadrature' or 'block'")
        key = (duration, order, method, covariance.tobytes())
        if key in self._cache:
            self._cache_hits += 1
            value = self._cache.pop(key)
            self._cache[key] = value
            return value[0].copy()
        self._cache_misses += 1
        value = (
            self._quadrature(covariance, duration, order)
            if method == "quadrature"
            else self._block(covariance, duration)
        )
        size = len(key[-1]) + value.nbytes
        if size <= self._cache_max_bytes:
            while self._cache and self._cache_bytes + size > self._cache_max_bytes:
                _, (_, removed) = self._cache.popitem(last=False)
                self._cache_bytes -= removed
            stored = value.copy()
            stored.flags.writeable = False
            self._cache[key] = (stored, size)
            self._cache_bytes += size
        return value

    def cache_info(self) -> dict:
        return {
            "entries": len(self._cache),
            "bytes": self._cache_bytes,
            "max_bytes": self._cache_max_bytes,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def _leakage_rate(self, duration: float) -> FloatArray:
        """Evaluate (D_J(t)-S_J(t)S_J(t)*)/t without small-t cancellation.

        The Gram solves G'=A G+G A*+exp(-t)vv*, v=(1,-1,...,-1).
        Positive small-step quadrature and G(2h)=S(h)G(h)S(h)*+exp(-h)G(h)
        retain its PSD structure and its finite limit at t=0.
        """
        step, doublings = self._small_step(duration)
        vector = np.r_[1.0, -np.ones(self.dimension - 1)]
        nodes, weights = np.polynomial.legendre.leggauss(24)
        rate = np.zeros((self.dimension, self.dimension))
        for node, weight in zip(nodes, weights, strict=True):
            time = (node + 1) * (step / 2)
            shifted = self.matrix(time) @ vector
            rate += (weight / 2) * np.exp(-(step - time)) * np.outer(shifted, shifted)
        shift, decay = self.matrix(step), np.exp(-step)
        for _ in range(doublings):
            rate = (shift @ rate @ shift.T + decay * rate) / 2
            shift = shift @ shift
            decay *= decay
        return _symmetric(rate)

    def innovation_covariance(
        self,
        source_covariance: ArrayLike,
        dt: float,
        quadrature_order: int = 24,
        *,
        method: str = "quadrature",
    ) -> FloatArray:
        """Normalized V = integral S(r) C S(r)* dr / dt; not source covariance C."""
        duration = _positive_scalar(dt, name="dt")
        order = _quadrature_order(quadrature_order)
        covariance = _source_covariance(source_covariance, self.dimension)
        return self._normalized(covariance, duration, order, method)

    def cumulative_covariance(
        self,
        source_covariance: ArrayLike,
        dt: float,
        quadrature_order: int = 24,
        *,
        method: str = "quadrature",
    ) -> FloatArray:
        """Unnormalized stochastic-convolution covariance over dt, including dt=0."""
        duration = _positive_scalar(dt, name="dt", allow_zero=True)
        order = _quadrature_order(quadrature_order)
        covariance = _source_covariance(source_covariance, self.dimension)
        if method not in ("quadrature", "block"):
            raise ValueError("method must be 'quadrature' or 'block'")
        if duration == 0:
            return np.zeros_like(covariance)
        return _finite(
            duration * self._normalized(covariance, duration, order, method),
            "cumulative covariance",
        )

    def covariance_diagnostics(
        self, source_covariance: ArrayLike, dt: float, quadrature_order: int = 24
    ) -> dict:
        """Independent-method crosscheck, not a rigorous numerical-error certificate."""
        duration = _positive_scalar(dt, name="dt")
        quadrature = self.innovation_covariance(source_covariance, duration, quadrature_order)
        block = self.innovation_covariance(
            source_covariance, duration, quadrature_order, method="block"
        )
        step, doublings = self._small_step(duration)
        difference = float(np.linalg.norm(quadrature - block, "fro"))
        norm = float(np.linalg.norm(block, "fro"))
        return {
            "quadrature_innovation_covariance": quadrature,
            "block_innovation_covariance": block,
            "absolute_hs_discrepancy": difference,
            "relative_hs_discrepancy": difference / norm if norm else 0.0,
            "quadrature_order": _quadrature_order(quadrature_order),
            "small_step": step,
            "doublings": doublings,
            "minimum_eigenvalue": float(np.linalg.eigvalsh(quadrature)[0]),
            "numerical_error_certificate": False,
        }


class PowerLawSource:
    """One infinite-rank diagonal source; prefixes are never renormalized.

    q_k = trace * (k+1)^(-exponent) / zeta(exponent), k>=0. Exponent>1
    gives trace class; first-generator HS regularity additionally requires
    exponent>2 in this basis. The benchmark defaults are exponent=3, trace=1.
    """

    basis_family_id = _FAMILY_ID

    def __init__(self, exponent: float = 3.0, trace: float = 1.0):
        self._exponent = _positive_scalar(exponent, name="exponent")
        if self._exponent <= 1:
            raise ValueError("exponent must be strictly greater than one for a trace-class source")
        self._trace = _positive_scalar(trace, name="trace")
        self._normalizer = float(zeta(self.exponent, 1))
        self._family_id = _basis_digest(
            "power-law-source",
            {
                "basis_family_id": self.basis_family_id,
                "exponent": self.exponent,
                "trace": self.trace,
            },
        )

    @property
    def exponent(self) -> float:
        return self._exponent

    @property
    def trace(self) -> float:
        return self._trace

    @property
    def family_id(self) -> str:
        return self._family_id

    @property
    def spectrum_id(self) -> str:
        return self.family_id

    def eigenvalues(self, dimension: int) -> FloatArray:
        count = _integer(dimension, "dimension")
        values = (self.trace / self._normalizer) * np.power(
            np.arange(1, count + 1, dtype=float), -self.exponent
        )
        if np.any(values == 0):
            raise ValueError("requested source eigenvalues underflow floating point")
        return _finite(values, "source eigenvalues")

    def finite_matrix(self, dimension: int) -> FloatArray:
        return np.diag(self.eigenvalues(dimension))

    def tail_trace(self, cutoff: int) -> float:
        count = _integer(cutoff, "cutoff", allow_zero=True)
        result = self.trace * (float(zeta(self.exponent, count + 1)) / self._normalizer)
        if result == 0 or not np.isfinite(result):
            raise ValueError("source trace tail cannot be represented reliably")
        return result

    def _omitted_eigenvalue(self, cutoff: int) -> float:
        count = _integer(cutoff, "cutoff", allow_zero=True)
        result = (self.trace / self._normalizer) * float(count + 1) ** -self.exponent
        if result == 0 or not np.isfinite(result):
            raise ValueError("omitted eigenvalue cannot be represented reliably")
        return result

    def innovation_tail_bound(self, cutoff: int, dt: float) -> float:
        duration = _positive_scalar(dt, name="dt")
        average = -np.expm1(-duration) / duration
        return float(
            _finite(np.asarray(self.tail_trace(cutoff) * (2 - average)), "innovation tail bound")
        )

    def state_mse_tail_bound(self, cutoff: int, time: float) -> float:
        duration = _positive_scalar(time, name="time", allow_zero=True)
        if duration == 0:
            _integer(cutoff, "cutoff", allow_zero=True)
            return 0.0
        factor = 2 + np.expm1(-duration) / duration
        return float(
            _finite(np.asarray((self.tail_trace(cutoff) * duration) * factor), "state tail bound")
        )

    @staticmethod
    def _bound(matrix: FloatArray, q: float) -> dict:
        matrix = _symmetric(matrix)
        return {
            "matrix": matrix,
            "hs_bound": float(np.linalg.norm(matrix, "fro")),
            "operator_bound": float(np.linalg.eigvalsh(matrix)[-1]),
            "trace_bound": float(np.trace(matrix)),
            "q_M": q,
            "interpretation": "analytic finite-observation tail bound evaluated in floating point",
        }

    def projected_tail_bound(self, dimension: int, cutoff: int, dt: float) -> dict:
        """Finite-J projected V tail, not a bound on the full infinite-state trace."""
        count = _integer(dimension, "dimension")
        duration = _positive_scalar(dt, name="dt")
        q = self._omitted_eigenvalue(cutoff)
        average = -np.expm1(-duration) / duration
        return self._bound(q * np.diag(np.r_[2 - average, np.full(count - 1, average)]), q)

    def operational_tail_bound(self, dimension: int, cutoff: int, dt: float, time: float) -> dict:
        """Covariance tail of the projected operational residual / sqrt(dt).

        ``time`` is the preceding state time. Source truncation uses independent
        omitted forcing and the same deterministic initial curve. The sharper
        bound uses L e_0=0; it does not assert vanishing leakage as dt->0.
        """
        count = _integer(dimension, "dimension")
        duration = _positive_scalar(dt, name="dt")
        state_time = _positive_scalar(time, name="time", allow_zero=True)
        projected = self.projected_tail_bound(count, cutoff, duration)
        rate = LaguerreTransport(count, cache_max_bytes=0)._leakage_rate(duration)
        coefficient = -np.expm1(-state_time)
        result = self._bound(
            projected["matrix"] + (projected["q_M"] * coefficient) * rate, projected["q_M"]
        )
        result["state_time"] = state_time
        result["interval"] = duration
        return result
