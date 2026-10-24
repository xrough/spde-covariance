"""Full-H contract geometry and one parent observation experiment.

The parent noisy observations are generated before a covariance-sieve dimension
is chosen. Finite-J views change only the model operator, never the data or its
known noise covariance. Full-H functional spans and finite-J covariance
observability are different questions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
from scipy.special import gammainc

from .laguerre import LaguerreTransport, PowerLawSource
from .linalg import _real_array, validate_psd
from .observations import Observation

DESIGN_NAMES = ("fixed_points", "fixed_deliveries", "nested8", "nested16", "rotating4")


def _readonly(value):
    result = np.asarray(value, dtype=float).copy()
    result.flags.writeable = False
    return result


def _positive(value, name, *, zero=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
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


@dataclass(frozen=True, slots=True)
class Contract:
    """A point evaluation or a uniform delivery average, in maturity years."""

    kind: str
    start: float
    end: float | None = None

    def __post_init__(self):
        if self.kind not in ("point", "delivery"):
            raise ValueError("contract kind must be point or delivery")
        object.__setattr__(self, "start", _positive(self.start, "start", zero=True))
        if self.kind == "point":
            if self.end is not None:
                raise ValueError("a point contract has no delivery end")
        else:
            end = _positive(self.end, "end")
            if end <= self.start:
                raise ValueError("delivery end must exceed start")
            object.__setattr__(self, "end", end)

    @property
    def label(self):
        return (
            f"point({self.start:g})"
            if self.kind == "point"
            else f"delivery({self.start:g},{self.end:g})"
        )


@dataclass(frozen=True, slots=True)
class LinearFunctional:
    """A signed linear combination of contracts; weights include normalization."""

    contracts: tuple[Contract, ...]
    weights: np.ndarray
    name: str = ""

    def __post_init__(self):
        contracts = tuple(self.contracts)
        weights = _real_array(self.weights, name="functional weights")
        if (
            not contracts
            or any(not isinstance(value, Contract) for value in contracts)
            or weights.shape != (len(contracts),)
        ):
            raise ValueError("functional contracts and weights must be nonempty and match")
        if not isinstance(self.name, str):
            raise ValueError("functional name must be a string")
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "weights", _readonly(weights))

    @property
    def anchor(self):
        """Value of this functional on the constant basis vector e0."""
        return float(np.sum(self.weights))


def contract_pool():
    """The predeclared nested pool, excluding both off-grid exposure maturities."""
    initial = (
        Contract("point", 0.5),
        Contract("point", 1.5),
        Contract("delivery", 0.5, 1.0),
        Contract("delivery", 1.5, 2.0),
    )
    extra = (0.0, 1.0, 2.0, 3.0, 0.25, 1.25, 2.5, 3.5, 4.0, 4.5, 5.0, 6.0)
    return initial + tuple(Contract("point", point) for point in extra)


def _functional(value):
    if isinstance(value, Contract):
        return LinearFunctional((value,), np.ones(1), value.label)
    if not isinstance(value, LinearFunctional):
        raise TypeError("expected a Contract or LinearFunctional")
    return value


def _affine(functional, left, right):
    """Derivative representer on a knot-free interval: c+s*(u-left)."""
    constant, slope, middle = 0.0, 0.0, (left + right) / 2
    for contract, weight in zip(functional.contracts, functional.weights, strict=True):
        if contract.kind == "point":
            constant += weight * (middle < contract.start)
        elif middle < contract.start:
            constant += weight
        elif middle < contract.end:
            width = contract.end - contract.start
            constant += weight * (contract.end - left) / width
            slope -= weight / width
    return constant, slope


def functional_gram(functionals: Sequence[Contract | LinearFunctional]):
    """Exact elementary full-H Gram, with stable exponential-moment evaluation.

    The RKHS kernel is ``2-exp(-min(x,y))``. Under the H isometry a contract
    representer is its constant anchor plus ``exp(-u/2)`` times a step or ramp.
    Products are integrated analytically on every interval using incomplete
    gamma moments of degrees zero, one and two. There is no basis truncation.
    """
    values = tuple(_functional(value) for value in functionals)
    if not values:
        raise ValueError("at least one functional is required")
    anchors = np.array([value.anchor for value in values])
    result = np.outer(anchors, anchors)
    knots = {0.0}
    for functional in values:
        for contract in functional.contracts:
            knots.add(contract.start)
            if contract.end is not None:
                knots.add(contract.end)
    cuts = sorted(knots)
    for left, right in zip(cuts[:-1], cuts[1:], strict=True):
        length = right - left
        moments = (-np.expm1(-length), gammainc(2, length), 2 * gammainc(3, length))
        constant, slope = np.array([_affine(value, left, right) for value in values]).T
        result += np.exp(-left) * (
            moments[0] * np.outer(constant, constant)
            + moments[1] * (np.outer(constant, slope) + np.outer(slope, constant))
            + moments[2] * np.outer(slope, slope)
        )
    return _readonly((result + result.T) / 2)


def fixed_exposures():
    """Three unit-full-H spreads, with no finite-J renormalization."""
    contracts = contract_pool()
    definitions = {
        "point_spread": contracts[:2],
        "delivery_spread": contracts[2:4],
        "offgrid_spread": (Contract("point", 0.75), Contract("point", 2.25)),
    }
    result = {}
    for name, pair in definitions.items():
        raw = LinearFunctional(pair, np.array([1.0, -1.0]), name)
        norm = np.sqrt(functional_gram([raw])[0, 0])
        result[name] = LinearFunctional(pair, raw.weights / norm, name)
    return result


def _contract_coefficients(contract, model):
    if contract.kind == "point":
        return model.evaluate([contract.start])[0]
    width = contract.end - contract.start
    if width < 1e-4:
        # Avoid endpoint subtraction for an interval approaching a point.
        nodes, weights = np.polynomial.legendre.leggauss(8)
        points = contract.start + width * (nodes + 1) / 2
        return weights @ model.evaluate(points) / 2
    endpoints = model.evaluate([contract.start, contract.end])
    result = np.ones(model.dimension)
    if model.dimension > 1:
        result[1] = 1 + np.exp(-contract.start) * np.expm1(-width) / width
    if model.dimension > 2:
        result[2:] = (
            (endpoints[1, 1:-1] - endpoints[1, 2:]) - (endpoints[0, 1:-1] - endpoints[0, 2:])
        ) / width
    return result


def functional_coefficients(functionals, model: LaguerreTransport):
    """Rows of full functionals restricted to the model's orthonormal prefix."""
    values = tuple(_functional(value) for value in functionals)
    if not values:
        raise ValueError("at least one functional is required")
    return _readonly(
        np.array(
            [
                sum(
                    (
                        weight * _contract_coefficients(contract, model)
                        for contract, weight in zip(value.contracts, value.weights, strict=True)
                    ),
                    np.zeros(model.dimension),
                )
                for value in values
            ]
        )
    )


def design_indices(name, date_index):
    """Rotating4 slides a consecutive four-contract window, with period sixteen."""
    date_index = _integer(date_index, "date index", zero=True)
    fixed = {
        "fixed_points": (0, 1),
        "fixed_deliveries": (2, 3),
        "nested8": tuple(range(8)),
        "nested16": tuple(range(16)),
    }
    if name in fixed:
        return fixed[name]
    if name == "rotating4":
        return tuple((date_index + offset) % 16 for offset in range(4))
    raise ValueError(f"unknown design {name!r}")


def design_groups(name):
    """All predeclared same-date groups (one group or sixteen rotating groups)."""
    return tuple(design_indices(name, index) for index in range(16 if name == "rotating4" else 1))


def span_diagnostics(exposure, contracts, *, rtol=1e-10):
    """Full-H same-date proxy availability, independent of any covariance sieve.

    Failure means a direct squared-exposure proxy cannot be made from this
    date's contracts. It does not establish nonidentification by pooled
    covariance observations with different maps.
    """
    rtol = _positive(rtol, "rtol")
    if rtol >= 1:
        raise ValueError("rtol must be less than one")
    target = _functional(exposure)
    observed = tuple(_functional(value) for value in contracts)
    if not observed:
        raise ValueError("at least one observed functional is required")
    gram = functional_gram((*observed, target))
    values, vectors = np.linalg.eigh(gram[:-1, :-1])
    threshold = rtol * max(float(values[-1]), 0.0)
    retained = values > threshold
    cross = gram[:-1, -1]
    coefficients = (
        vectors[:, retained] @ ((vectors[:, retained].T @ cross) / values[retained])
        if np.any(retained)
        else np.zeros(len(observed))
    )
    residual_contracts = list(target.contracts)
    residual_weights = list(target.weights)
    for coefficient, value in zip(coefficients, observed, strict=True):
        residual_contracts.extend(value.contracts)
        residual_weights.extend(-coefficient * value.weights)
    residual = LinearFunctional(tuple(residual_contracts), np.array(residual_weights))
    # Re-evaluate the residual representer itself; norm(target)^2-c.Gc would
    # lose all significant digits for an exposure exactly in the span.
    residual_squared = max(0.0, float(functional_gram([residual])[0, 0]))
    target_squared = max(0.0, float(gram[-1, -1]))
    relative = np.sqrt(residual_squared / target_squared) if target_squared else 0.0
    return {
        "available": bool(relative <= rtol),
        "coefficients": _readonly(coefficients),
        "residual_squared": residual_squared,
        "relative_residual": float(relative),
        "target_norm_squared": target_squared,
        "rank": int(np.sum(retained)),
        "rank_threshold": threshold,
        "rtol": rtol,
        "scope": "same-date linear risk proxy in full H; not pooled covariance identification",
    }


def fixed_grid_witness(contracts, model, exposure=None):
    """Finite-prefix invisible rank-one covariance for an unrestricted experiment.

    No claim is made that this rank-one covariance is an SPDE-admissible
    innovation covariance under the fixed generator or either source family.
    """
    operator = functional_coefficients(contracts, model)
    _, singular, right = np.linalg.svd(operator, full_matrices=False)
    threshold = 1e-12 * (singular[0] if len(singular) else 0.0)
    rows = right[singular > threshold]
    if len(rows) == model.dimension:
        raise ValueError("contracts span this entire finite prefix")
    if exposure is None:
        coordinate = int(np.argmin(np.sum(rows**2, axis=0)))
        vector = np.eye(1, model.dimension, coordinate)[0]
        target = None
    else:
        target = functional_coefficients([exposure], model)[0]
        vector = target.copy()
    vector -= rows.T @ (rows @ vector)
    # A small target-null component can amplify the first projection's
    # roundoff on normalization. Reorthogonalize before constructing hh*.
    vector -= rows.T @ (rows @ vector)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("the requested exposure has no finite-prefix invisible component")
    vector /= norm
    return {
        "vector": _readonly(vector),
        "rank_one_covariance": _readonly(np.outer(vector, vector)),
        "observation_residual": float(np.linalg.norm(operator @ vector)),
        "exposure_risk_difference": None if target is None else float((target @ vector) ** 2),
        "scope": "unrestricted covariance experiment; no SPDE source-family admissibility claim",
    }


def observation_omission(innovation_covariance, operator, dimension):
    """Full observed covariance minus the sieve prediction, retaining cross terms."""
    covariance = validate_psd(innovation_covariance, name="innovation covariance")
    operator = _real_array(operator, name="operator")
    dimension = _integer(dimension, "dimension")
    if operator.ndim != 2 or operator.shape[1] != len(covariance) or dimension > len(covariance):
        raise ValueError("operator and covariance dimensions must match the sieve")
    low, high = operator[:, :dimension], operator[:, dimension:]
    full = operator @ covariance @ operator.T
    sieve = low @ covariance[:dimension, :dimension] @ low.T
    tail_tail = high @ covariance[dimension:, dimension:] @ high.T
    cross = low @ covariance[:dimension, dimension:] @ high.T
    omission = full - sieve
    return {
        **{
            name: _readonly(value)
            for name, value in {
                "full": full,
                "sieve": sieve,
                "omission": omission,
                "tail_tail": tail_tail,
                "cross": cross,
                "cross_transpose": cross.T,
            }.items()
        },
        "omission_hs": float(np.linalg.norm(omission)),
        "tail_tail_hs": float(np.linalg.norm(tail_tail)),
        "cross_hs": float(np.linalg.norm(cross)),
        "cross_sum_hs": float(np.linalg.norm(cross + cross.T)),
        "full_hs": float(np.linalg.norm(full)),
        "relative_hs_error": float(
            np.linalg.norm(omission) / max(np.linalg.norm(full), np.finfo(float).tiny)
        ),
        "scope": "full observed covariance minus zero-extended finite-J prediction; cross terms included",
    }


@dataclass(frozen=True, slots=True)
class PreparedParent:
    """Population truth and geometry, kept separate from the observed data."""

    source_name: str
    model: LaguerreTransport
    source_covariance: np.ndarray
    innovation_covariance: np.ndarray
    operator: np.ndarray
    contract_gram: np.ndarray
    contracts: tuple[Contract, ...]
    exposures: MappingProxyType
    exposure_coefficients: MappingProxyType
    source_metadata: MappingProxyType
    tail_bounds: MappingProxyType
    numerical_checks: MappingProxyType
    delta: float


def prepare_parent_model(
    source="powerlaw",
    dimension=256,
    delta=1 / 52,
    quadrature_order=24,
    *,
    source_exponent=3.0,
    source_trace=1.0,
    rotation_angle=np.pi / 6,
):
    """Prepare one finite reference to a fixed infinite-source weekly experiment."""
    dimension = _integer(dimension, "dimension")
    delta = _positive(delta, "delta")
    quadrature_order = _integer(quadrature_order, "quadrature order")
    if source not in ("powerlaw", "rotated"):
        raise ValueError("source must be powerlaw or rotated")
    if source == "rotated" and dimension % 2:
        raise ValueError("rotated source requires even pair-complete prefixes")
    if (
        isinstance(rotation_angle, (bool, np.bool_))
        or not isinstance(rotation_angle, Real)
        or not np.isfinite(rotation_angle)
    ):
        raise ValueError("rotation angle must be finite")
    model, spectrum = LaguerreTransport(dimension), PowerLawSource(source_exponent, source_trace)
    covariance = spectrum.finite_matrix(dimension)
    angle = float(rotation_angle) if source == "rotated" else 0.0
    if angle:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for start in range(0, dimension, 2):
            block = covariance[start : start + 2, start : start + 2]
            covariance[start : start + 2, start : start + 2] = rotation @ block @ rotation.T
    diagnostic = model.covariance_diagnostics(covariance, delta, quadrature_order)
    innovation = diagnostic["quadrature_innovation_covariance"]
    contracts, exposures = contract_pool(), fixed_exposures()
    operator, gram = functional_coefficients(contracts, model), functional_gram(contracts)
    exposure_coefficients = {
        name: functional_coefficients([exposure], model)[0] for name, exposure in exposures.items()
    }
    average = -np.expm1(-delta) / delta
    q_m = float(spectrum.eigenvalues(dimension + 1)[-1])
    observed_bound = q_m * (average * gram + 2 * (1 - average) * np.ones_like(gram))
    risk_bounds = {
        name: q_m
        * (average * functional_gram([exposure])[0, 0] + 2 * (1 - average) * exposure.anchor**2)
        for name, exposure in exposures.items()
    }
    metadata = {
        "name": source,
        "exponent": spectrum.exponent,
        "infinite_trace": spectrum.trace,
        "finite_trace": float(np.trace(covariance)),
        "rotation_angle": angle,
        "rotation_pairs": "(e0,e1),(e2,e3),...",
        "rotation_convention": "R=[[cos,-sin],[sin,cos]]; C=R diag(q) R.T",
        "spectrum_family_id": spectrum.family_id,
    }
    metadata["source_family_id"] = (
        "partial-source:"
        + hashlib.sha256(
            json.dumps(
                {key: value for key, value in metadata.items() if key != "finite_trace"},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:20]
    )
    tail_bounds = {
        "full_source_trace_tail": spectrum.tail_trace(dimension),
        "full_innovation_trace_bound": spectrum.innovation_tail_bound(dimension, delta),
        "observed_matrix": _readonly(observed_bound),
        "observed_hs_bound": float(np.linalg.norm(observed_bound)),
        "observed_operator_bound": float(np.linalg.eigvalsh(observed_bound)[-1]),
        "risk_bounds": MappingProxyType(risk_bounds),
        "q_M": q_m,
        "scope": "analytic infinite-source omission bounds; finite reference is not infinite truth",
    }
    checks = {
        key: diagnostic[key]
        for key in (
            "absolute_hs_discrepancy",
            "relative_hs_discrepancy",
            "minimum_eigenvalue",
            "quadrature_order",
            "small_step",
            "doublings",
            "numerical_error_certificate",
        )
    }
    return PreparedParent(
        source,
        model,
        _readonly(covariance),
        _readonly(innovation),
        operator,
        gram,
        contracts,
        MappingProxyType(exposures),
        MappingProxyType(exposure_coefficients),
        MappingProxyType(metadata),
        MappingProxyType(tail_bounds),
        MappingProxyType(checks),
        delta,
    )


@dataclass(frozen=True, slots=True)
class ParentPool:
    """One immutable full-contract noisy data set; no latent reference draws."""

    times: np.ndarray
    y: np.ndarray
    operator: np.ndarray
    noise_covariance: np.ndarray
    basis_id: str
    delta: float

    def __post_init__(self):
        times = _real_array(self.times, name="times")
        values = _real_array(self.y, name="parent measurements")
        operator = _real_array(self.operator, name="parent operator")
        noise = validate_psd(self.noise_covariance, name="parent noise covariance")
        if (
            times.ndim != 1
            or not len(times)
            or values.shape != (len(times), 16)
            or operator.ndim != 2
            or operator.shape[0] != 16
            or not operator.shape[1]
            or noise.shape != (16, 16)
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "parent times, sixteen-contract measurements, operator and noise must match"
            )
        if not isinstance(self.basis_id, str) or not self.basis_id:
            raise ValueError("basis_id must be nonempty")
        object.__setattr__(self, "delta", _positive(self.delta, "delta"))
        for name, value in (
            ("times", times),
            ("y", values),
            ("operator", operator),
            ("noise_covariance", noise),
        ):
            object.__setattr__(self, name, _readonly(value))

    def observations(self, design, dimension):
        """Sieve views preserve the same observed y and noise N on every date."""
        dimension = _integer(dimension, "dimension")
        if dimension > self.operator.shape[1]:
            raise ValueError("sieve dimension exceeds the parent reference")
        result = []
        for index, (time, value) in enumerate(zip(self.times, self.y, strict=True)):
            indices = list(design_indices(design, index))
            result.append(
                Observation(
                    time,
                    value[indices],
                    self.operator[indices, :dimension],
                    self.noise_covariance[np.ix_(indices, indices)],
                )
            )
        return tuple(result)


def simulate_parent_pool(prepared, n_dates, *, seed, noise_std):
    """Map full-reference innovations to all contracts, then add fresh noise.

    The identical parent arrays supply every design and every J. Noise is
    independent on dates and contracts and is measured on normalized
    innovations, not differenced noisy curve levels.
    """
    if not isinstance(prepared, PreparedParent):
        raise TypeError("prepared must be a PreparedParent")
    n_dates, seed = _integer(n_dates, "n_dates"), _integer(seed, "seed", zero=True)
    noise_std = _positive(noise_std, "noise standard deviation", zero=True)
    eigenvalues, vectors = np.linalg.eigh(prepared.innovation_covariance)
    root = (vectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ vectors.T
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal((n_dates, prepared.model.dimension)) @ root.T
    noise = noise_std * rng.standard_normal((n_dates, 16))
    values = innovations @ prepared.operator.T + noise
    return ParentPool(
        np.arange(1, n_dates + 1) * prepared.delta,
        values,
        prepared.operator,
        noise_std**2 * np.eye(16),
        prepared.model.basis_id,
        prepared.delta,
    )
