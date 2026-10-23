"""Configuration for a predeclared frozen-block covariance forecast study."""

from __future__ import annotations

from dataclasses import dataclass

from .rq2_design import SCENARIOS, RQ2Config, _integer, _positive, _sequence

DESIGNS = ("full", "dense", "split", "joint8", "joint64")


@dataclass(frozen=True)
class ForecastSettings:
    """Estimator settings with no scenario, covariance-path or future metadata."""

    delta: float
    relative_regularization: float
    trace_cap: float
    numerical_error_budget: float

    def __post_init__(self):
        for name in ("delta", "relative_regularization", "trace_cap", "numerical_error_budget"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


@dataclass(frozen=True)
class RQ3Config:
    """Counts refer to independent weekly innovations, numbered from one.

    Forecasts are frozen for ``horizon`` subsequent weekly innovations. They
    are not forecasts of an accumulated multiweek innovation. The static rule
    fits the first ``static_window`` observations once. Validation outcomes must
    all be available before the first test forecast is issued.
    """

    name: str = "rq3"
    seed: int = 20261023
    replications: int = 20
    intervals: int = 512
    delta: float = 1 / 52
    windows: tuple[int, ...] = (16, 32, 64, 128)
    validation_origins: tuple[int, ...] = (128, 192)
    test_origins: tuple[int, ...] = (256, 272, 304, 448)
    horizon: int = 16
    static_window: int = 128
    fixed_window: int = 128
    trace_cap: float = 2.0
    noise_variance: float = 0.001
    relative_regularization: float = 0.001
    numerical_error_budget: float = 1e-4
    rank_rtol: float = 1e-10
    reference_checks: bool = True
    scenarios: tuple[str, ...] = SCENARIOS
    designs: tuple[str, ...] = DESIGNS

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        for name in (
            "seed",
            "replications",
            "intervals",
            "horizon",
            "static_window",
            "fixed_window",
        ):
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
        for name in ("windows", "validation_origins", "test_origins"):
            values = tuple(_integer(value, name) for value in _sequence(getattr(self, name), name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
            if name != "windows" and tuple(sorted(values)) != values:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)
        for name, allowed in (("scenarios", SCENARIOS), ("designs", DESIGNS)):
            values = tuple(_sequence(getattr(self, name), name))
            if any(not isinstance(value, str) or value not in allowed for value in values):
                raise ValueError(f"{name} must contain only {allowed}")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, values)
        if not isinstance(self.reference_checks, bool):
            raise ValueError("reference_checks must be a boolean")
        if self.rank_rtol >= 1:
            raise ValueError("rank_rtol must be smaller than one")
        if self.trace_cap < 1.25:
            raise ValueError("trace_cap must accommodate the maximum truth trace of 1.25")
        earliest = min(self.validation_origins[0], self.test_origins[0])
        if max((*self.windows, self.static_window, self.fixed_window)) > earliest:
            raise ValueError("all training windows must fit before the earliest forecast origin")
        if self.validation_origins[-1] + self.horizon > self.test_origins[0]:
            raise ValueError("all validation outcomes must precede the first test origin")
        for name in ("validation_origins", "test_origins"):
            origins = getattr(self, name)
            if any(right - left < self.horizon for left, right in zip(origins, origins[1:])):
                raise ValueError(f"{name} forecast blocks must not overlap")
            if origins[-1] + self.horizon > self.intervals:
                raise ValueError("forecast outcomes must fit inside the simulated intervals")

    def fit_settings(self) -> ForecastSettings:
        """Expose only the settings the observation-only covariance fit needs."""
        return ForecastSettings(
            self.delta,
            self.relative_regularization,
            self.trace_cap,
            self.numerical_error_budget,
        )

    def simulation_config(self) -> RQ2Config:
        """Adapt only simulation settings; preserve the existing RQ2 generator."""
        return RQ2Config(
            name=self.name,
            seed=self.seed,
            replications=self.replications,
            intervals=self.intervals,
            delta=self.delta,
            windows=self.windows,
            endpoints=self.validation_origins,
            trace_cap=self.trace_cap,
            noise_variance=self.noise_variance,
            relative_regularization=self.relative_regularization,
            sensitivity_relatives=(),
            sensitivity_window=min(self.windows),
            numerical_error_budget=self.numerical_error_budget,
            rank_rtol=self.rank_rtol,
            reference_checks=False,
            scenarios=self.scenarios,
        )
