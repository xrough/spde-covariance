"""Predeclared resolutions of one infinite-rank Laguerre transport model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


def _integer(value, name, *, allow_zero=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return int(value)


def _positive(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and strictly positive")
    return float(value)


def _sequence(value, name, *, allow_empty=False):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error
    if not values and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    return values


@dataclass(frozen=True)
class InfiniteTransportConfig:
    """Refine forcing M, observation projection J and time step separately.

    The source spectrum is fixed across all levels; finite prefixes are never
    renormalized. Every time grid must be an integer coarsening of the finest
    configured grid. Large J population diagnostics do not invoke a lifted
    covariance fit. Unresolved finite-source tails remain reported flags.
    """

    name: str = "infinite-transport"
    seed: int = 20261024
    replications: int = 20
    source_exponent: float = 3.0
    source_trace: float = 1.0
    horizon: float = 1.0
    forcing_dimensions: tuple[int, ...] = (64, 128, 256)
    projection_dimensions: tuple[int, ...] = (8, 16, 32, 64)
    deltas: tuple[float, ...] = (1 / 52, 1 / 104, 1 / 208)
    quadrature_order: int = 24
    tail_budget: float = 1e-5
    numerical_tolerance: float = 1e-9
    transport_tolerance: float = 1e-10
    fit_dimensions: tuple[int, ...] = (8, 16)
    relative_regularization: float = 0.001
    trace_cap: float = 2.0
    numerical_error_budget: float = 1e-4
    reference_checks: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        for name in ("seed", "replications", "quadrature_order"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, allow_zero=name == "seed")
            )
        for name in (
            "source_exponent",
            "source_trace",
            "horizon",
            "tail_budget",
            "numerical_tolerance",
            "transport_tolerance",
            "relative_regularization",
            "trace_cap",
            "numerical_error_budget",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.source_exponent <= 2:
            raise ValueError(
                "source_exponent must exceed two for the declared generator regularity"
            )
        for name in ("forcing_dimensions", "projection_dimensions", "fit_dimensions"):
            values = tuple(
                _integer(value, name)
                for value in _sequence(
                    getattr(self, name), name, allow_empty=name == "fit_dimensions"
                )
            )
            if len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)
        if max(self.projection_dimensions) > max(self.forcing_dimensions):
            raise ValueError("projection dimensions must fit inside the largest forcing level")
        if any(
            value not in self.projection_dimensions or value > 16 for value in self.fit_dimensions
        ):
            raise ValueError("fit_dimensions must be configured projections no larger than sixteen")
        values = tuple(_positive(value, "deltas") for value in _sequence(self.deltas, "deltas"))
        if len(set(values)) != len(values):
            raise ValueError("deltas must be unique")
        object.__setattr__(self, "deltas", values)
        fine = min(values)
        for value in values:
            if value > self.horizon or not np.isclose(
                value / fine, round(value / fine), rtol=1e-12, atol=0
            ):
                raise ValueError("every delta must be an integer coarsening of the finest grid")
            if not np.isclose(
                self.horizon / value, round(self.horizon / value), rtol=1e-12, atol=0
            ):
                raise ValueError("the horizon must contain an integer number of every interval")
        if not isinstance(self.reference_checks, bool):
            raise ValueError("reference_checks must be a boolean")
        if self.reference_checks and self.fit_dimensions and min(self.fit_dimensions) > 8:
            raise ValueError(
                "reference checks require a configured fit dimension no larger than eight"
            )

    @property
    def fine_delta(self) -> float:
        return min(self.deltas)

    @property
    def fine_intervals(self) -> int:
        return round(self.horizon / self.fine_delta)
