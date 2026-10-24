"""Predeclared partial-innovation observations and covariance selection protocol."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .infinite_config import _integer, _positive, _sequence


@dataclass(frozen=True)
class PartialBenchmarkConfig:
    """Fit every covariance sieve to the same parent observations at fixed delta.

    Smaller training counts use nested prefixes of the training block. All
    candidates share a subsequent validation block and a disjoint test block.
    Selection never refits on validation data. A fixed comparator uses the
    largest candidate dimension and smallest relative ridge strength.
    """

    name: str = "partial-observations"
    seed: int = 20261025
    replications: int = 20
    reference_dimension: int = 256
    delta: float = 1 / 52
    training_counts: tuple[int, ...] = (128, 512)
    validation_count: int = 128
    test_count: int = 128
    candidate_dimensions: tuple[int, ...] = (4, 8, 16)
    geometry_dimensions: tuple[int, ...] = (4, 8, 16, 32)
    relative_regularizations: tuple[float, ...] = (0.001, 0.01, 0.1)
    trace_cap: float = 2.0
    numerical_error_budget: float = 1e-4
    rank_rtol: float = 1e-10
    noise_variance: float = 0.001
    source_exponent: float = 3.0
    source_trace: float = 1.0
    rotation_angle: float = np.pi / 6
    scenarios: tuple[str, ...] = ("powerlaw", "rotated")
    designs: tuple[str, ...] = (
        "fixed_points",
        "fixed_deliveries",
        "nested8",
        "nested16",
        "rotating4",
    )
    quadrature_order: int = 24
    reference_checks: bool = True
    tail_budget: float = 1e-5
    numerical_tolerance: float = 1e-9

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        for name in (
            "seed",
            "replications",
            "reference_dimension",
            "validation_count",
            "test_count",
            "quadrature_order",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, allow_zero=name == "seed")
            )
        for name in (
            "delta",
            "trace_cap",
            "numerical_error_budget",
            "rank_rtol",
            "source_exponent",
            "source_trace",
            "tail_budget",
            "numerical_tolerance",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.source_exponent <= 2:
            raise ValueError("source_exponent must exceed two for the declared regularity")
        if self.rank_rtol >= 1:
            raise ValueError("rank_rtol must be less than one")
        for name in ("noise_variance", "rotation_angle"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real scalar")
            if name == "noise_variance" and value < 0:
                raise ValueError("noise_variance must be nonnegative")
            object.__setattr__(self, name, float(value))
        for name in ("training_counts", "candidate_dimensions", "geometry_dimensions"):
            values = tuple(_integer(value, name) for value in _sequence(getattr(self, name), name))
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)
        if max(self.geometry_dimensions) > self.reference_dimension:
            raise ValueError("geometry dimensions cannot exceed the forcing reference")
        if any(value not in self.geometry_dimensions for value in self.candidate_dimensions):
            raise ValueError("candidate dimensions must be included in geometry_dimensions")
        if max(self.candidate_dimensions) > 16:
            raise ValueError("the dense benchmark fits candidate dimensions no larger than sixteen")
        strengths = tuple(
            _positive(value, "relative_regularizations")
            for value in _sequence(self.relative_regularizations, "relative_regularizations")
        )
        if tuple(sorted(set(strengths))) != strengths:
            raise ValueError("relative_regularizations must be strictly increasing")
        object.__setattr__(self, "relative_regularizations", strengths)
        for name, supported in (
            ("scenarios", {"powerlaw", "rotated"}),
            ("designs", {"fixed_points", "fixed_deliveries", "nested8", "nested16", "rotating4"}),
        ):
            values = _sequence(getattr(self, name), name)
            if any(not isinstance(value, str) or value not in supported for value in values):
                raise ValueError(f"{name} contains an unsupported value")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, values)
        if "rotated" in self.scenarios and self.reference_dimension % 2:
            raise ValueError("rotated source prefixes must end on a complete adjacent pair")
        if not isinstance(self.reference_checks, bool):
            raise ValueError("reference_checks must be a boolean")
        if self.reference_checks and min(self.candidate_dimensions) > 8:
            raise ValueError("reference checks require a candidate dimension no larger than eight")

    @property
    def total_count(self) -> int:
        return max(self.training_counts) + self.validation_count + self.test_count

    @property
    def baseline_dimension(self) -> int:
        return max(self.candidate_dimensions)

    @property
    def baseline_relative_regularization(self) -> float:
        return min(self.relative_regularizations)
