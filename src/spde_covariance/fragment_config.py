"""Predeclared dense-fragment controls within the partial-observation study."""

from dataclasses import dataclass

import numpy as np

from .infinite_config import _integer, _positive, _sequence


@dataclass(frozen=True)
class FragmentBenchmarkConfig:
    study: str = "fragments"
    name: str = "canonical-spde"
    profile: str = "spde"
    seed: int = 20261026
    replications: int = 20
    training_counts: tuple[int, ...] = (128, 512, 2048)
    covers: tuple[int, ...] = (3, 9)
    grids: tuple[int, ...] = (41, 81, 161)
    models: tuple[str, ...] = ("powerlaw", "rotated")
    noise_variances: tuple[float, ...] = (0.0, 0.001)
    reference_dimension: int = 512
    delta: float = 1 / 52
    validation_count: int = 128
    test_count: int = 512
    ranks: tuple[int, ...] = (1, 2, 4, 8)
    include_fve: bool = True
    support_rtol: float = 1e-12
    numerical_error_budget: float = 1e-4
    trace_cap: float = 2.0
    candidate_dimensions: tuple[int, ...] = (4, 8, 16)
    relative_regularizations: tuple[float, ...] = (0.001, 0.01, 0.1)
    ridge_comparison: bool = True
    population_checks: bool = True
    refinement_grid: int = 321
    source_exponent: float = 3.0
    source_trace: float = 1.0
    rotation_angle: float = np.pi / 6
    quadrature_order: int = 24

    def __post_init__(self):
        if self.study != "fragments" or self.profile not in ("spde", "reference"):
            raise ValueError("fragment study requires profile spde or reference")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        for name in (
            "seed",
            "replications",
            "reference_dimension",
            "validation_count",
            "test_count",
            "refinement_grid",
            "quadrature_order",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, allow_zero=name == "seed")
            )
        for name in ("training_counts", "covers", "grids", "ranks", "candidate_dimensions"):
            values = tuple(_integer(v, name) for v in _sequence(getattr(self, name), name))
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)
        if any(m not in (2, 3, 5, 9) for m in self.covers):
            raise ValueError("covers must be paper covers 2, 3, 5, or 9")
        if min(self.training_counts) < max(self.covers):
            raise ValueError("every training prefix must include every declared fragment")
        if min(self.grids) < 11 or max(self.candidate_dimensions) > 16:
            raise ValueError("grids require >=11 points and dense ridge dimensions <=16")
        if max(self.candidate_dimensions) > self.reference_dimension:
            raise ValueError("candidate dimensions exceed forcing dimension")
        if self.profile == "spde" and self.reference_dimension % 2:
            raise ValueError("SPDE forcing prefixes must contain complete rotation pairs")
        if self.profile == "spde" and self.population_checks and self.reference_dimension % 4:
            raise ValueError("half-forcing population checks require a multiple of four modes")
        permitted = {
            "spde": {"powerlaw", "rotated"},
            "reference": {"finite_rank", "brownian", "analytic"},
        }[self.profile]
        models = tuple(self.models)
        if not models or len(set(models)) != len(models) or not set(models) <= permitted:
            raise ValueError("models must be unique and match the selected profile")
        object.__setattr__(self, "models", models)
        for name in (
            "delta",
            "support_rtol",
            "numerical_error_budget",
            "trace_cap",
            "source_exponent",
            "source_trace",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.source_exponent <= 2 or self.support_rtol >= 1:
            raise ValueError("source exponent must exceed two and support_rtol must be below one")
        for name in ("noise_variances", "relative_regularizations"):
            if any(isinstance(v, (bool, np.bool_)) for v in _sequence(getattr(self, name), name)):
                raise ValueError(f"{name} must contain real numbers, not booleans")
            values = tuple(float(v) for v in _sequence(getattr(self, name), name))
            if any(
                not np.isfinite(v) or v < 0 or (name == "relative_regularizations" and v == 0)
                for v in values
            ):
                raise ValueError(f"{name} contains an invalid value")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)
        if not np.isfinite(self.rotation_angle):
            raise ValueError("rotation_angle must be finite")
        for name in ("include_fve", "ridge_comparison", "population_checks"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.profile == "reference" and self.ridge_comparison:
            raise ValueError("generic paper controls have no declared H-coordinate ridge basis")
        if self.ridge_comparison and (81 not in self.grids or 512 not in self.training_counts):
            raise ValueError("ridge comparison requires base grid 81 and training count 512")

    @property
    def total_count(self):
        return max(self.training_counts) + self.validation_count + self.test_count
