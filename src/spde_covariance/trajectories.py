"""Transport trajectories and normalized semigroup-adjusted innovations.

Time is measured in years. Extraction acts on curve coefficients before any
fresh observation noise is added. It does not difference noisy price levels or
pool innovations across unequal interval lengths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import _real_array, validate_psd
from .observations import Observation

if TYPE_CHECKING:
    from .transport import TransportPropagator

FloatArray = NDArray[np.float64]


def _readonly(array: ArrayLike, *, name: str) -> FloatArray:
    result = _real_array(array, name=name).copy()
    result.flags.writeable = False
    return result


def _metadata_snapshot(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        return MappingProxyType({key: _metadata_snapshot(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_snapshot(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return value
    raise ValueError("metadata must contain mappings, arrays, sequences, or scalar values")


def _times(times: ArrayLike) -> FloatArray:
    result = _readonly(times, name="times")
    if result.ndim != 1 or len(result) < 2:
        raise ValueError("times must contain at least two interval endpoints")
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.diff(result)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("times must be strictly increasing with finite interval lengths")
    return result


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _provenance(instance):
    object.__setattr__(instance, "basis_id", _label(instance.basis_id, "basis_id"))
    object.__setattr__(
        instance, "propagation_method", _label(instance.propagation_method, "propagation_method")
    )
    if instance.time_unit != "year":
        raise ValueError("time_unit must be 'year'; convert timestamps before extraction")
    object.__setattr__(instance, "curve_unit", _label(instance.curve_unit, "curve_unit"))
    if not isinstance(instance.metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    object.__setattr__(instance, "metadata", _metadata_snapshot(instance.metadata))


@dataclass(frozen=True, slots=True)
class CurveTrajectory:
    """A snapshot of coefficients at all N+1 observation endpoints."""

    times: FloatArray
    coefficients: FloatArray
    basis_id: str
    propagation_method: str
    time_unit: str = "year"
    curve_unit: str = "arbitrary curve unit"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        times = _times(self.times)
        coefficients = _readonly(self.coefficients, name="coefficients")
        if (
            coefficients.ndim != 2
            or coefficients.shape[0] != len(times)
            or coefficients.shape[1] < 1
        ):
            raise ValueError("coefficients must have shape (len(times), J) with J positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coefficients", coefficients)
        _provenance(self)

    @property
    def dimension(self) -> int:
        return self.coefficients.shape[1]

    @property
    def n_intervals(self) -> int:
        return len(self.times) - 1

    @property
    def interval_lengths(self) -> FloatArray:
        return _readonly(np.diff(self.times), name="interval_lengths")


@dataclass(frozen=True, slots=True)
class InnovationBatch:
    """Normalized increments with their original interval endpoints preserved.

    ``values[i] = (f[i+1] - S_dt f[i] - integrated_drift[i]) / sqrt(dt)``.
    Covariance need not be common across irregular intervals. Estimation code
    must choose an explicit common-target or pooling convention separately.
    """

    times: FloatArray
    values: FloatArray
    basis_id: str
    propagation_method: str
    drift_convention: str = "zero integrated mild drift"
    time_unit: str = "year"
    curve_unit: str = "arbitrary curve unit"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        times = _times(self.times)
        values = _readonly(self.values, name="values")
        if values.ndim != 2 or values.shape[0] != len(times) - 1 or values.shape[1] < 1:
            raise ValueError("values must have shape (len(times)-1, J) with J positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "drift_convention", _label(self.drift_convention, "drift_convention")
        )
        _provenance(self)

    @property
    def dimension(self) -> int:
        return self.values.shape[1]

    @property
    def n_intervals(self) -> int:
        return len(self.values)

    @property
    def interval_lengths(self) -> FloatArray:
        return _readonly(np.diff(self.times), name="interval_lengths")

    @property
    def start_times(self) -> FloatArray:
        return self.times[:-1]

    @property
    def end_times(self) -> FloatArray:
        return self.times[1:]

    @property
    def innovation_unit(self) -> str:
        return f"{self.curve_unit}/sqrt(year)"

    def observe(
        self,
        operators: Sequence[ArrayLike] | ArrayLike | None = None,
        *,
        noise_covariances: Sequence[ArrayLike] | ArrayLike | None = None,
        seed: int = 0,
    ) -> tuple[Observation, ...]:
        """Apply contract maps and optionally add fresh independent Gaussian noise.

        Identity maps are the default. Supply a common matrix or one matrix per
        interval; row counts may differ. Noise is added after extraction, and
        the returned Observation contains its known covariance correction.
        The original interval lengths remain available on this batch.
        """
        maps = (
            [np.eye(self.dimension)] * self.n_intervals
            if operators is None
            else _matrix_sequence(operators, self.n_intervals, "operators")
        )
        maps = tuple(_real_array(operator, name="operator") for operator in maps)
        if any(
            operator.ndim != 2 or operator.shape[0] < 1 or operator.shape[1] != self.dimension
            for operator in maps
        ):
            raise ValueError("operators must have shape (m_i, dimension), with m_i positive")
        noises = (
            [np.zeros((len(operator), len(operator))) for operator in maps]
            if noise_covariances is None
            else _matrix_sequence(noise_covariances, self.n_intervals, "noise_covariances")
        )
        noises = tuple(validate_psd(noise, name="noise_covariance") for noise in noises)
        if any(
            noise.shape != (len(operator), len(operator))
            for noise, operator in zip(noises, maps, strict=True)
        ):
            raise ValueError("noise covariance must match the corresponding observation rows")
        rng = np.random.default_rng(seed)
        return tuple(
            Observation(
                float(time), operator @ value + _gaussian_sample(rng, noise), operator, noise
            )
            for time, value, operator, noise in zip(
                self.end_times, self.values, maps, noises, strict=True
            )
        )


@dataclass(frozen=True, slots=True)
class TransportSimulation:
    """Trajectory plus separate latent source and normalized-innovation truths."""

    trajectory: CurveTrajectory
    innovation_covariances: FloatArray
    source_covariances: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.trajectory, CurveTrajectory):
            raise ValueError("trajectory must be a CurveTrajectory")
        count, dimension = self.trajectory.n_intervals, self.trajectory.dimension
        for name in ("innovation_covariances", "source_covariances"):
            raw = _real_array(getattr(self, name), name=name)
            if raw.shape != (count, dimension, dimension):
                raise ValueError(f"{name} must have shape (N, J, J)")
            matrices = np.stack([validate_psd(matrix, name=name) for matrix in raw])
            object.__setattr__(self, name, _readonly(matrices, name=name))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _metadata_snapshot(self.metadata))


def _matrix_sequence(value, count, name):
    try:
        candidate = np.asarray(value)
    except ValueError:
        candidate = None
    if candidate is not None and candidate.ndim == 2 and candidate.dtype != object:
        return tuple(candidate for _ in range(count))
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a common matrix or one matrix per interval") from error
    if len(result) != count:
        raise ValueError(f"{name} must contain one matrix per interval")
    return result


def _propagator_details(propagator):
    dimension = propagator.dimension
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("propagator dimension must be a positive integer")
    return int(dimension), _label(propagator.basis_id, "basis_id"), _label(propagator.kind, "kind")


def _transition(propagator, dt, dimension):
    matrix = _real_array(propagator.matrix(float(dt)), name="transport matrix")
    if matrix.shape != (dimension, dimension):
        raise ValueError("transport matrix must have shape (dimension, dimension)")
    return matrix


def _drift(integrated_drift, count, dimension):
    if integrated_drift is None:
        return np.zeros((count, dimension)), "zero integrated mild drift"
    drift = _real_array(integrated_drift, name="integrated_drift")
    if drift.shape != (count, dimension):
        raise ValueError("integrated_drift must have shape (N, J)")
    return drift, "supplied integrated mild drift"


def _gaussian_sample(rng, covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0)) * rng.standard_normal(len(eigenvalues))
    )


def extract_innovations(
    trajectory: CurveTrajectory,
    propagator: TransportPropagator,
    *,
    integrated_drift=None,
) -> InnovationBatch:
    """Extract normalized mild increments using actual, possibly irregular dt.

    ``integrated_drift[i]`` is the already integrated and transported mild drift
    on the interval, in the same coefficient basis. It is not a drift rate.
    Basis identity is checked in addition to dimension, preventing accidental
    subtraction in different orthonormal coordinates.
    """
    if not isinstance(trajectory, CurveTrajectory):
        raise ValueError("trajectory must be a CurveTrajectory")
    dimension, basis_id, method = _propagator_details(propagator)
    if trajectory.dimension != dimension or trajectory.basis_id != basis_id:
        raise ValueError("trajectory and propagator must use the same basis and dimension")
    drift, convention = _drift(integrated_drift, trajectory.n_intervals, dimension)
    values = np.stack(
        [
            (
                trajectory.coefficients[index + 1]
                - _transition(propagator, dt, dimension) @ trajectory.coefficients[index]
                - drift[index]
            )
            / np.sqrt(dt)
            for index, dt in enumerate(trajectory.interval_lengths)
        ]
    )
    return InnovationBatch(
        trajectory.times,
        values,
        basis_id,
        method,
        convention,
        curve_unit=trajectory.curve_unit,
        metadata={
            "normalization": "divide mild increment by sqrt(interval length)",
            "trajectory_propagation_method": trajectory.propagation_method,
            "pooling": "none; interval endpoints retained",
        },
    )


def simulate_transport(
    propagator: TransportPropagator,
    times: ArrayLike,
    source_covariance: ArrayLike,
    *,
    initial: ArrayLike | None = None,
    integrated_drift: ArrayLike | None = None,
    seed: int = 0,
    curve_unit: str = "arbitrary curve unit",
) -> TransportSimulation:
    """Simulate a deterministic-source Gaussian mild recursion on supplied dates.

    Source covariance is common J by J or N by J by J, held constant on each
    interval. The exact exponential backend supplies the analytic transported
    source integral. A spline backend defines an explicitly projected mild
    Gaussian recursion rather than an exact sample of the infinite-dimensional
    SPDE. No observation noise is added to the trajectory.
    """
    dimension, basis_id, method = _propagator_details(propagator)
    dates = _times(times)
    lengths = np.diff(dates)
    count = len(lengths)
    covariances = _real_array(source_covariance, name="source_covariance")
    if covariances.shape == (dimension, dimension):
        covariances = np.broadcast_to(covariances, (count, dimension, dimension))
    if covariances.shape != (count, dimension, dimension):
        raise ValueError("source_covariance must have shape (J, J) or (N, J, J)")
    covariances = np.stack(
        [validate_psd(matrix, name="source_covariance") for matrix in covariances]
    )
    start = np.zeros(dimension) if initial is None else _real_array(initial, name="initial")
    if start.shape != (dimension,):
        raise ValueError("initial must have shape (J,)")
    drift, convention = _drift(integrated_drift, count, dimension)
    rng = np.random.default_rng(seed)
    coefficients = np.empty((count + 1, dimension))
    coefficients[0] = start
    innovations = []
    for index, dt in enumerate(lengths):
        covariance = validate_psd(
            propagator.innovation_covariance(covariances[index], float(dt)),
            name="innovation_covariance",
        )
        if covariance.shape != (dimension, dimension):
            raise ValueError("propagator innovation covariance must have shape (J, J)")
        innovations.append(covariance)
        coefficients[index + 1] = (
            _transition(propagator, dt, dimension) @ coefficients[index]
            + drift[index]
            + np.sqrt(dt) * _gaussian_sample(rng, covariance)
        )
    interpretation = (
        "projected mild Gaussian discrete recursion; not an exact infinite-dimensional SPDE sample"
        if method == "projected_mild_recursion"
        else "deterministic-source Gaussian recursion using the propagator covariance integral"
    )
    metadata = {
        "seed": seed,
        "propagation_method": method,
        "simulation_model": interpretation,
        "source_convention": "deterministic covariance, constant on each interval",
        "drift_convention": convention,
        "time_unit": "year",
        "curve_unit": curve_unit,
        "innovation_unit": f"{curve_unit}/sqrt(year)",
        "noise_convention": "latent curve coefficients; no observation noise",
        "innovation_target": "V_i = integral S_r C_i S_r* dr / dt_i",
    }
    trajectory = CurveTrajectory(
        dates, coefficients, basis_id, method, curve_unit=curve_unit, metadata=metadata
    )
    return TransportSimulation(trajectory, np.stack(innovations), covariances, metadata)
