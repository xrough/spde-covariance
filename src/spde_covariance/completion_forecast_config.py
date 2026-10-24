"""Predeclared chronological forecasts from observed covariance fragments."""

from dataclasses import dataclass

from .infinite_config import _integer, _positive, _sequence

SCENARIOS = ("constant", "scale", "rotating", "abrupt")


@dataclass(frozen=True)
class CompletionForecastConfig:
    """A fixed-grid prediction study, separate from joint resolution limits."""

    study: str = "completion_forecast"
    name: str = "completion-forecast"
    seed: int = 20261028
    replications: int = 20
    intervals: int = 512
    delta: float = 1 / 52
    reference_dimension: int = 256
    grid: int = 21
    covers: tuple[int, ...] = (3, 9)
    scenarios: tuple[str, ...] = SCENARIOS
    windows: tuple[int, ...] = (16, 32, 64, 128)
    validation_origins: tuple[int, ...] = (128, 192)
    test_origins: tuple[int, ...] = (256, 272, 304, 448)
    horizon: int = 16
    static_window: int = 128
    fixed_window: int = 128
    source_exponent: float = 3.0
    source_trace: float = 1.0
    noise_variance: float = 0.001
    relative_regularization: float = 0.001
    floor_gamma: float = 1e-5 * 46**2
    numerical_error_budget: float = 1e-4
    support_rtol: float = 1e-12
    quadrature_order: int = 24
    admm_rho: float = 0.2
    admm_max_iterations: int = 100_000
    workers: int = 1

    def __post_init__(self):
        if self.study != "completion_forecast":
            raise ValueError("completion_forecast study marker is required")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        for field in (
            "seed",
            "replications",
            "intervals",
            "reference_dimension",
            "grid",
            "horizon",
            "static_window",
            "fixed_window",
            "quadrature_order",
            "admm_max_iterations",
            "workers",
        ):
            object.__setattr__(
                self, field, _integer(getattr(self, field), field, allow_zero=field == "seed")
            )
        for field in (
            "delta",
            "source_exponent",
            "source_trace",
            "relative_regularization",
            "floor_gamma",
            "numerical_error_budget",
            "support_rtol",
            "admm_rho",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if isinstance(self.noise_variance, bool) or not isinstance(
            self.noise_variance, (int, float)
        ):
            raise ValueError("noise_variance must be a finite nonnegative number")
        import math

        if not math.isfinite(self.noise_variance) or self.noise_variance < 0:
            raise ValueError("noise_variance must be a finite nonnegative number")
        if self.source_exponent <= 2 or self.reference_dimension < 2:
            raise ValueError("source_exponent must exceed two and reference_dimension must be >=2")
        if self.grid < 5 or self.support_rtol >= 1 or self.workers > 4:
            raise ValueError("grid >=5, support_rtol <1 and at most four workers are required")
        for field in ("windows", "validation_origins", "test_origins", "covers"):
            values = tuple(_integer(v, field) for v in _sequence(getattr(self, field), field))
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{field} must be strictly increasing")
            object.__setattr__(self, field, values)
        if any(v < 2 for v in self.covers):
            raise ValueError("at least two overlapping fragments are required")
        scenarios = tuple(_sequence(self.scenarios, "scenarios"))
        if len(set(scenarios)) != len(scenarios) or any(v not in SCENARIOS for v in scenarios):
            raise ValueError(f"scenarios must be distinct members of {SCENARIOS}")
        object.__setattr__(self, "scenarios", scenarios)
        earliest = min(self.validation_origins[0], self.test_origins[0])
        if max((*self.windows, self.static_window, self.fixed_window)) > earliest:
            raise ValueError("all training windows must precede the earliest forecast origin")
        if min((*self.windows, self.static_window, self.fixed_window)) < max(self.covers):
            raise ValueError("every training window must contain a complete fragment cycle")
        if self.validation_origins[-1] + self.horizon > self.test_origins[0]:
            raise ValueError("all validation outcomes must precede test forecasts")
        for field in ("validation_origins", "test_origins"):
            origins = getattr(self, field)
            if any(b - a < self.horizon for a, b in zip(origins, origins[1:])):
                raise ValueError("forecast blocks must not overlap")
            if origins[-1] + self.horizon > self.intervals:
                raise ValueError("forecast outcomes must fit within intervals")
