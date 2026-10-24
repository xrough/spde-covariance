"""A focused paired follow-up to the historical dense-fragment study."""

from dataclasses import asdict, dataclass, fields

import numpy as np

from .fragment_config import FragmentBenchmarkConfig
from .infinite_config import _integer, _positive


@dataclass(frozen=True)
class CoherentBenchmarkConfig(FragmentBenchmarkConfig):
    study: str = "coherent"
    name: str = "coherent-completion"
    grids: tuple[int, ...] = (41,)
    ranks: tuple[int, ...] = (2,)
    include_fve: bool = False
    ridge_comparison: bool = False
    population_checks: bool = False
    partial_regularizations: tuple[float, ...] = (0.0001, 0.001, 0.01)
    eigenvalue_floor: float = 1e-5
    admm_rho: float = 0.2
    admm_max_iterations: int = 20_000
    admissibility_dimensions: tuple[int, ...] = (4, 8, 16)
    admissibility_gap: float = 1e-8
    admissibility_max_iterations: int = 20_000

    def __post_init__(self):
        if self.study != "coherent":
            raise ValueError("coherent study marker is required")
        values = asdict(self)
        inherited = {f.name: values[f.name] for f in fields(FragmentBenchmarkConfig)}
        inherited["study"] = "fragments"
        validated = FragmentBenchmarkConfig(**inherited)
        for field in fields(FragmentBenchmarkConfig):
            if field.name != "study":
                object.__setattr__(self, field.name, getattr(validated, field.name))
        if self.profile != "spde" or self.ridge_comparison or self.population_checks:
            raise ValueError(
                "this follow-up uses SPDE fragments; historical ridge/controls stay separate"
            )
        if self.ranks != (2,) or self.include_fve:
            raise ValueError("the historical baseline is fixed rank two")
        candidates = tuple(self.partial_regularizations)
        if not candidates or tuple(sorted(set(candidates))) != candidates:
            raise ValueError("partial_regularizations must be strictly increasing")
        object.__setattr__(
            self,
            "partial_regularizations",
            tuple(_positive(v, "partial_regularizations") for v in candidates),
        )
        for name in ("eigenvalue_floor", "admm_rho", "admissibility_gap"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("admm_max_iterations", "admissibility_max_iterations"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        dimensions = tuple(
            _integer(v, "admissibility_dimensions") for v in self.admissibility_dimensions
        )
        if not dimensions or tuple(sorted(set(dimensions))) != dimensions:
            raise ValueError("admissibility_dimensions must be strictly increasing")
        if max(dimensions) > min(24, self.reference_dimension):
            raise ValueError("source-cone diagnostics require dimensions <=24 and <=forcing cutoff")
        object.__setattr__(self, "admissibility_dimensions", dimensions)
        if not np.isfinite(self.eigenvalue_floor):
            raise ValueError("eigenvalue_floor must be finite")
