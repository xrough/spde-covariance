"""Predeclared ablations and fixed-model refinement, separate from forecasting."""

from dataclasses import dataclass

import numpy as np

from .infinite_config import _integer, _positive


@dataclass(frozen=True)
class CompletionResearchConfig:
    study: str = "completion_research"
    name: str = "completion-refinement"
    experiment: str = "refinement"
    profile: str = "spde"
    seed: int = 20261028
    replications: int = 20
    workers: int = 4
    models: tuple[str, ...] = ("powerlaw", "rotated")
    grids: tuple[int, ...] = (21, 41, 81)
    covers: tuple[int, ...] = (3, 9)
    training_counts: tuple[int, ...] = (128, 512, 2048)
    reference_dimension: int = 512
    validation_count: int = 128
    test_count: int = 512
    delta: float = 1 / 52
    noise_variance: float = 0.001
    source_exponent: float = 3.0
    source_trace: float = 1.0
    rotation_angle: float = np.pi / 6
    quadrature_order: int = 24
    support_rtol: float = 1e-12
    numerical_error_budget: float = 1e-4
    max_iterations: int = 100000
    rho: float = 0.2
    floor_gamma: float = 0.02116
    constant_floor: float = 1e-5
    relative_regularization: float = 0.001
    population_grids: tuple[int, ...] = (21, 41, 81, 161, 321)
    forcing_dimensions: tuple[int, ...] = (64, 128, 256, 512)
    output_dimensions: tuple[int, ...] = (4, 8, 16, 32)
    source_dimensions: tuple[int, ...] = (4, 8, 16)
    source_multipliers: tuple[int, ...] = (1, 2, 4)
    source_max_iterations: int = 20000
    source_gap: float = 1e-8
    trace_cap: float = 2.0
    stability: bool = True
    stability_relative_limit: float = 1.0
    stability_condition_limit: float = 1e8

    def __post_init__(self):
        if self.study != "completion_research" or self.profile != "spde":
            raise ValueError("completion_research is a fixed SPDE-model study")
        if self.experiment not in {"ablation", "refinement"}:
            raise ValueError("experiment must be ablation or refinement")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        for name in (
            "seed",
            "replications",
            "workers",
            "reference_dimension",
            "validation_count",
            "test_count",
            "quadrature_order",
            "max_iterations",
            "source_max_iterations",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, allow_zero=name == "seed")
            )
        if self.workers > 4 or self.reference_dimension % 2:
            raise ValueError("use at most four workers and complete source rotation pairs")
        for name in (
            "grids",
            "covers",
            "training_counts",
            "population_grids",
            "forcing_dimensions",
            "output_dimensions",
            "source_dimensions",
            "source_multipliers",
        ):
            values = tuple(_integer(v, name) for v in getattr(self, name))
            if tuple(sorted(set(values))) != values or (not values and name != "source_dimensions"):
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)
        if not set(self.covers) <= {3, 9} or min(self.grids) < 21:
            raise ValueError("use three/nine fragments and grids of at least 21 points")
        if min(self.training_counts) < 2 * max(self.covers):
            raise ValueError("training prefixes must permit two observations per fragment")
        if (
            max(self.forcing_dimensions) > self.reference_dimension
            or any(m % 2 for m in self.forcing_dimensions)
            or max(self.output_dimensions) > self.reference_dimension
        ):
            raise ValueError("refinement dimensions must be nested within the source reference")
        if (
            self.source_dimensions
            and max(self.source_dimensions) * max(self.source_multipliers) > 64
        ):
            raise ValueError("dense source-class studies support at most 64 forcing modes")
        models = tuple(self.models)
        if (
            not models
            or len(set(models)) != len(models)
            or not set(models) <= {"powerlaw", "rotated"}
        ):
            raise ValueError("models must be unique fixed powerlaw/rotated sources")
        object.__setattr__(self, "models", models)
        for name in (
            "delta",
            "source_exponent",
            "source_trace",
            "support_rtol",
            "numerical_error_budget",
            "rho",
            "floor_gamma",
            "constant_floor",
            "relative_regularization",
            "source_gap",
            "trace_cap",
            "stability_relative_limit",
            "stability_condition_limit",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.source_exponent <= 2 or self.support_rtol >= 1:
            raise ValueError("source exponent must exceed two; support cutoff must be below one")
        if not np.isfinite(self.noise_variance) or self.noise_variance < 0:
            raise ValueError("known noise variance must be finite and nonnegative")
        if not np.isfinite(self.rotation_angle) or not isinstance(self.stability, bool):
            raise ValueError("rotation angle must be finite and stability must be boolean")

    @property
    def total_count(self):
        return max(self.training_counts) + self.validation_count + self.test_count
