"""Configuration boundaries of the fixed-interval partial-observation study."""

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spde_covariance.partial_config import PartialBenchmarkConfig


def test_defaults_preserve_static_target_and_disjoint_selection_blocks():
    config = PartialBenchmarkConfig()
    assert config.delta == 1 / 52
    assert config.total_count == 768
    assert config.baseline_dimension == 16
    assert config.baseline_relative_regularization == 0.001
    assert config.training_counts == (128, 512)
    assert config.noise_variance == 0.001
    assert config.replications == 20


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " "},
        {"seed": True},
        {"replications": 0},
        {"reference_dimension": 255},
        {"delta": 0},
        {"training_counts": ()},
        {"training_counts": (512, 128)},
        {"training_counts": (128, 128)},
        {"candidate_dimensions": (4, 32)},
        {"candidate_dimensions": (3, 8)},
        {"geometry_dimensions": (4, 8, 16, 512)},
        {"relative_regularizations": (0, 0.1)},
        {"relative_regularizations": (0.01, 0.001)},
        {"trace_cap": 0},
        {"numerical_error_budget": np.inf},
        {"rank_rtol": 1},
        {"noise_variance": -1},
        {"noise_variance": True},
        {"rotation_angle": np.nan},
        {"source_exponent": 2},
        {"scenarios": ("unknown",)},
        {"designs": ("nested8", "nested8")},
        {"designs": ("unknown",)},
        {"reference_checks": 1},
        {"candidate_dimensions": (16,)},
    ],
)
def test_invalid_configurations_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(PartialBenchmarkConfig(), **changes)


def test_zero_fresh_noise_and_complete_pair_rotation_limit_are_supported():
    config = PartialBenchmarkConfig(noise_variance=0, rotation_angle=0)
    assert config.noise_variance == 0
    assert config.rotation_angle == 0
    config = PartialBenchmarkConfig(
        reference_dimension=33,
        scenarios=("powerlaw",),
    )
    assert config.reference_dimension == 33


def test_committed_full_and_smoke_configs_match_declared_scope():
    root = Path(__file__).resolve().parents[1]
    configs = []
    for name in ("partial-observations.toml", "partial-observations-smoke.toml"):
        with (root / "experiments" / "configs" / name).open("rb") as stream:
            configs.append(PartialBenchmarkConfig(**tomllib.load(stream)))
    full, smoke = configs
    assert full == PartialBenchmarkConfig()
    assert smoke.total_count == 128
    assert smoke.baseline_dimension == 8
    assert full.scenarios == smoke.scenarios
    assert full.designs == smoke.designs
