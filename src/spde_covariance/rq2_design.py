"""Predeclared innovation-level synthetic design for research question 2.

The target is finite-interval innovation covariance V in exact orthonormal
exponential coordinates. Fixed maturity evaluations isolate local estimation
and observation coverage; they do not represent an exchange contract calendar
or estimate the instantaneous diffusion covariance C. Latent truth is kept
separately from the estimator-facing :class:`Observation` records.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .observations import Observation
from .simulation import simulate_innovations
from .transport import ExponentialTransport

FloatArray = NDArray[np.float64]
SCENARIOS = ("constant", "scalar", "rotating", "abrupt")


def _integer(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        boundary = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be {boundary}")
    return result


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite strictly positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite strictly positive real number")
    return result


def _sequence(value, name, *, allow_empty=False):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    return result


@dataclass(frozen=True)
class RQ2Config:
    """Reproducible settings; window and endpoint values count intervals.

    Sequence settings accept ordinary TOML lists and are normalized to tuples.
    Sensitivity penalties may be omitted by supplying an empty sequence.
    """

    name: str = "rq2"
    seed: int = 20261023
    replications: int = 20
    intervals: int = 512
    delta: float = 1 / 52
    windows: tuple[int, ...] = (16, 32, 64, 128)
    endpoints: tuple[int, ...] = (256, 272, 304, 512)
    trace_cap: float = 2.0
    noise_variance: float = 0.001
    relative_regularization: float = 0.001
    sensitivity_relatives: tuple[float, ...] = (0.0001, 0.01)
    sensitivity_window: int = 64
    numerical_error_budget: float = 1e-4
    rank_rtol: float = 1e-10
    reference_checks: bool = True
    scenarios: tuple[str, ...] = SCENARIOS

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        for name in ("seed", "replications", "intervals", "sensitivity_window"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, allow_zero=name == "seed")
            )
        for name in (
            "delta",
            "trace_cap",
            "noise_variance",
            "relative_regularization",
            "numerical_error_budget",
            "rank_rtol",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("windows", "endpoints"):
            values = tuple(_integer(value, name) for value in _sequence(getattr(self, name), name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, values)
        sensitivities = tuple(
            _positive(value, "sensitivity_relatives")
            for value in _sequence(
                self.sensitivity_relatives, "sensitivity_relatives", allow_empty=True
            )
        )
        if len(set(sensitivities)) != len(sensitivities):
            raise ValueError("sensitivity_relatives must contain unique values")
        object.__setattr__(self, "sensitivity_relatives", sensitivities)
        scenarios = _sequence(self.scenarios, "scenarios")
        if any(not isinstance(value, str) or value not in SCENARIOS for value in scenarios):
            raise ValueError(f"scenarios must contain only {SCENARIOS}")
        if len(set(scenarios)) != len(scenarios):
            raise ValueError("scenarios must contain unique values")
        object.__setattr__(self, "scenarios", scenarios)
        if not isinstance(self.reference_checks, bool):
            raise ValueError("reference_checks must be a boolean")
        if max(self.windows) > min(self.endpoints):
            raise ValueError("all windows must be no larger than the earliest endpoint")
        if max(self.endpoints) > self.intervals:
            raise ValueError("endpoints must be no larger than intervals")
        if self.sensitivity_window > self.intervals:
            raise ValueError("sensitivity_window must be no larger than intervals")
        if self.trace_cap < 1.25:
            raise ValueError("trace_cap must accommodate the maximum truth trace of 1.25")
        if self.rank_rtol >= 1:
            raise ValueError("rank_rtol must be smaller than one")


def covariance_paths(config: RQ2Config) -> dict[str, FloatArray]:
    """Return the declared V paths, indexed by intervals 1 through N.

    Scalar and rotation paths include both endpoints when N > 1. The abrupt
    path changes immediately after interval floor(N/2); it is a stress case
    outside the smooth covariance-path model.
    """
    count = config.intervals
    base = np.diag([0.4, 0.3, 0.2, 0.1])
    constant = np.repeat(base[None], count, axis=0)
    scalar = np.linspace(0.75, 1.25, count)[:, None, None] * base
    angles = np.linspace(0, np.pi / 2, count)
    rotations = np.repeat(np.eye(4)[None], count, axis=0)
    rotations[:, 0, 0] = rotations[:, 3, 3] = np.cos(angles)
    rotations[:, 0, 3] = -np.sin(angles)
    rotations[:, 3, 0] = np.sin(angles)
    rotating = rotations @ base @ np.swapaxes(rotations, 1, 2)
    abrupt = constant.copy()
    abrupt[count // 2 :] = np.diag([0.1, 0.2, 0.3, 0.4])
    paths = dict(zip(SCENARIOS, (constant, scalar, rotating, abrupt), strict=True))
    return {scenario: paths[scenario] for scenario in config.scenarios}


def build_geometry() -> tuple[ExponentialTransport, FloatArray, dict[str, FloatArray]]:
    """Exact four-dimensional basis, raw point rows and unit-H exposures.

    The butterfly weights sum to zero and have zero maturity-weighted sum.
    Euclidean normalization in orthonormal H coordinates gives unit H norm
    to each exposure's Riesz representer; observation rows are not normalized.
    """
    model = ExponentialTransport(rates=(0.6, 1.0, 2.0, 4.0))
    point_map = model.evaluate([1 / 12, 0.5, 1.5, 3.0])
    exposures = {
        "near_spread": point_map[0] - point_map[1],
        "cross_spread": point_map[1] - point_map[2],
        "butterfly": 12 * point_map[0] - 17 * point_map[1] + 5 * point_map[2],
    }
    exposures = {name: value / np.linalg.norm(value) for name, value in exposures.items()}
    return model, point_map, exposures


@dataclass(frozen=True)
class RQ2Data:
    """Paired observations and separate latent simulation diagnostics."""

    truths: dict[str, FloatArray]
    designs: dict[str, dict[str, tuple[Observation, ...]]]
    latent: dict[str, FloatArray]
    exposures: dict[str, FloatArray]
    point_map: FloatArray
    basis_id: str
    times: FloatArray


def _coverage_observations(dense, joint_every):
    selected = []
    for index, observation in enumerate(dense, start=1):
        if joint_every is not None and index % joint_every == 0:
            selected.append(observation)
            continue
        rows = np.array([0, 1] if index % 2 else [2, 3])
        selected.append(
            Observation(
                observation.time,
                observation.y[rows],
                observation.operator[rows],
                observation.noise_covariance[np.ix_(rows, rows)],
            )
        )
    return tuple(selected)


def generate_data(config: RQ2Config, replication: int) -> RQ2Data:
    """Generate each scenario once and slice coverage designs without resampling.

    Replications are zero-based. A shared seed ``seed + 1009 * replication``
    supplies the same base Gaussian draws to every scenario. All partial-map
    designs share their dense observations' exact noise samples. ``full`` is a
    separately labelled noiseless identity observation of the same latent X,
    serving as a recovery control rather than a maturity-observation design.
    """
    replication = _integer(replication, "replication", allow_zero=True)
    if replication >= config.replications:
        raise ValueError("replication must be smaller than config.replications")
    model, point_map, exposures = build_geometry()
    truths = covariance_paths(config)
    times = np.arange(1, config.intervals + 1, dtype=float) * config.delta
    operators = (point_map,) * config.intervals
    noise = config.noise_variance * np.eye(model.dimension)
    identity, zero_noise = np.eye(model.dimension), np.zeros((model.dimension, model.dimension))
    designs, latent = {}, {}
    for scenario, truth in truths.items():
        simulation = simulate_innovations(
            truth,
            operators,
            times=times,
            noise_covariances=noise,
            seed=config.seed + 1009 * replication,
        )
        dense = simulation.observations
        latent[scenario] = simulation.latent_innovations
        designs[scenario] = {
            "dense": dense,
            "split": _coverage_observations(dense, None),
            "joint8": _coverage_observations(dense, 8),
            "joint64": _coverage_observations(dense, 64),
            "full": tuple(
                Observation(float(time), value, identity, zero_noise)
                for time, value in zip(times, simulation.latent_innovations, strict=True)
            ),
        }
    return RQ2Data(truths, designs, latent, exposures, point_map, model.basis_id, times)
